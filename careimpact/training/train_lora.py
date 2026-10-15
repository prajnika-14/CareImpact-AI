"""LoRA supervised fine-tuning for CareImpact AI (CPU or GPU).

    python -m careimpact.training.train_lora --config configs/training/cpu_smoke.yaml

What it does
* Loads chat-format JSONL produced by ``cli build-dataset``.
* Applies the model's own chat template; loss is computed ONLY on the
  assistant JSON (prompt tokens are masked with -100).
* Examples longer than ``max_seq_len`` are skipped, never truncated, so the
  model never learns from cut-off JSON.
* Logs loss, seconds/step and peak RSS to ``<output_dir>/metrics.jsonl`` and a
  ``summary.json`` the dashboard reads. Saves the LoRA adapter.

Hardware notes
* CPU (e.g. i5-1235U, 16 GB): fp32 LoRA works for 0.5B and, slowly, 1.5B.
  bitsandbytes 4-bit (QLoRA) requires CUDA and is NOT available on CPU.
* GPU (Colab/Kaggle T4): set device: cuda, dtype: float16, optionally load_in_4bit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import psutil
import torch
import yaml

from careimpact.config import ROOT


def _load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[:limit] if limit else rows


def _encode(tok, messages: list[dict], max_len: int):
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(f_ids) > max_len or f_ids[: len(p_ids)] != p_ids:
        return None, len(f_ids)
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": torch.tensor([f_ids]), "labels": torch.tensor([labels])}, len(f_ids)


def _rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def _write_summary(out: Path, summary: dict) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


@torch.no_grad()
def _val_loss(model, batches) -> float | None:
    if not batches:
        return None
    model.eval()
    losses = [model(**{k: v.to(model.device) for k, v in b.items()}).loss.item() for b in batches]
    model.train()
    return sum(losses) / len(losses)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = cfg.get("seed", 13)
    random.seed(seed)
    torch.manual_seed(seed)
    if cfg.get("num_threads"):
        torch.set_num_threads(int(cfg["num_threads"]))
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.get("dtype", "float32")]

    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    summary = {"status": "loading", "base_model": cfg["base_model"], "method": "QLoRA" if cfg.get("load_in_4bit") else "LoRA",
               "device": device, "dtype": cfg.get("dtype", "float32"), "config": cfg,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write_summary(out, summary)

    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model_kwargs = {"dtype": dtype}
    if cfg.get("load_in_4bit"):
        if device != "cuda":
            raise SystemExit("load_in_4bit (QLoRA via bitsandbytes) requires a CUDA GPU; use plain LoRA on CPU.")
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                 bnb_4bit_compute_dtype=torch.float16)
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **model_kwargs)
    if not cfg.get("load_in_4bit"):
        model.to(device)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    lcfg = LoraConfig(task_type="CAUSAL_LM", r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
                      lora_dropout=cfg.get("lora_dropout", 0.05),
                      target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    load_s = time.perf_counter() - t_load

    max_len = cfg.get("max_seq_len", 2048)
    train_rows = _load_jsonl(ROOT / cfg["train_file"], cfg.get("limit_train"))
    val_rows = _load_jsonl(ROOT / cfg["val_file"], cfg.get("limit_val", 16)) if cfg.get("val_file") else []
    train, lengths, skipped = [], [], 0
    for r in train_rows:
        b, n = _encode(tok, r["messages"], max_len)
        lengths.append(n)
        if b is None:
            skipped += 1
        else:
            train.append(b)
    val = [b for b in (_encode(tok, r["messages"], max_len)[0] for r in val_rows) if b is not None]
    if not train:
        raise SystemExit(f"No training examples fit in max_seq_len={max_len}.")

    epochs = cfg.get("epochs", 1)
    accum = cfg.get("grad_accum", 8)
    steps_per_epoch = math.ceil(len(train) / accum)
    max_steps = cfg.get("max_steps") or steps_per_epoch * epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.get("learning_rate", 2e-4),
                            weight_decay=0.0)
    warmup = max(1, int(cfg.get("warmup_ratio", 0.05) * max_steps))
    # Linear warmup, then linear decay to zero.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (max_steps - s) / max(1, max_steps - warmup)))

    summary.update(status="training", trainable_params=trainable, total_params=total, load_seconds=round(load_s, 1),
                   train_examples=len(train), skipped_too_long=skipped, val_examples=len(val),
                   token_length_mean=round(sum(lengths) / len(lengths)), token_length_max=max(lengths),
                   planned_steps=max_steps, grad_accum=accum, max_seq_len=max_len)
    _write_summary(out, summary)
    print(json.dumps({k: summary[k] for k in ("base_model", "device", "trainable_params", "train_examples",
                                              "skipped_too_long", "token_length_mean", "token_length_max", "planned_steps")}))

    metrics_f = (out / "metrics.jsonl").open("w", encoding="utf-8")
    model.train()
    step, micro, peak = 0, 0, _rss_gb()
    t_step = time.perf_counter()
    order = list(range(len(train)))
    running: list[float] = []
    bad_losses = 0  # fp16 on T4 can overflow; skip those micro-batches instead of corrupting the adapter
    t_train = time.perf_counter()
    try:
        while step < max_steps:
            random.shuffle(order)
            for idx in order:
                b = {k: v.to(model.device) for k, v in train[idx].items()}
                loss = model(**b).loss / accum
                if not torch.isfinite(loss):
                    bad_losses += 1
                    summary["non_finite_losses"] = bad_losses
                    if bad_losses > max(10, len(train) // 50):
                        raise RuntimeError(f"{bad_losses} non-finite losses; lower learning_rate or use dtype float32/bfloat16.")
                    continue
                loss.backward()
                running.append(loss.item() * accum)
                micro += 1
                peak = max(peak, _rss_gb())
                if micro % accum:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                dt_s = time.perf_counter() - t_step
                t_step = time.perf_counter()
                rec = {"step": step, "loss": round(sum(running) / len(running), 4), "lr": sched.get_last_lr()[0],
                       "seconds": round(dt_s, 1), "rss_gb": round(peak, 2)}
                running = []
                if cfg.get("eval_every") and step % cfg["eval_every"] == 0:
                    rec["val_loss"] = _val_loss(model, val)
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
                print(json.dumps(rec), flush=True)
                summary.update(steps=step, final_train_loss=rec["loss"], peak_rss_gb=round(peak, 2),
                               seconds_per_step=round((time.perf_counter() - t_train) / step, 1))
                _write_summary(out, summary)
                if cfg.get("save_every") and step % cfg["save_every"] == 0:
                    model.save_pretrained(out / f"checkpoint-{step}")
                if step >= max_steps:
                    break
        summary["val_loss"] = _val_loss(model, val)
        model.save_pretrained(out / "adapter")
        tok.save_pretrained(out / "adapter")
        summary.update(status="completed", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       train_seconds=round(time.perf_counter() - t_train, 1), adapter_dir=str(out / "adapter"))
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        model.save_pretrained(out / "adapter-interrupted")
    except Exception as exc:
        summary.update(status="failed", error=repr(exc))
        raise
    finally:
        metrics_f.close()
        _write_summary(out, summary)
    print(json.dumps({k: summary.get(k) for k in ("status", "steps", "final_train_loss", "val_loss", "seconds_per_step", "peak_rss_gb")}))


if __name__ == "__main__":
    main()
