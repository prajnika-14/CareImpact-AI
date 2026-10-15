"""Combine approved real + synthetic examples into SFT chat-format files.

Only records that are ``training_eligible`` AND either ``approved`` (human
review) or ``auto_validated`` synthetic records (labels generated from the
scenario spec) are included. Real LLM/baseline drafts must be approved first.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml

from careimpact.analysis.prompts import PROMPT_VERSION, chat_messages
from careimpact.config import CONFIG_DIR, Paths
from careimpact.diffing import diff_documents
from careimpact.evaluation.runner import load_records
from careimpact.schemas import ExampleRecord


def _eligible(rec: ExampleRecord) -> bool:
    if not rec.training_eligible:
        return False
    if rec.data_origin == "synthetic":
        return rec.validation_status in {"auto_validated", "approved"} and rec.target_source == "scenario_spec"
    return rec.validation_status == "approved"


def _real_split(doc_id: str, test_fraction: float, val_fraction: float) -> str:
    """Deterministic split by policy ID so versions of one policy never straddle splits."""
    h = int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % 1000 / 1000
    if h < test_fraction:
        return "test"
    if h < test_fraction + val_fraction:
        return "val"
    return "train"


def _to_chat(rec: ExampleRecord) -> dict:
    diff = diff_documents(rec.old, rec.new)
    return {
        "messages": chat_messages(rec.old, rec.new, diff, rec.target),
        "meta": {"example_id": rec.example_id, "data_origin": rec.data_origin, "family": rec.family,
                 "scenario": rec.scenario, "classification": rec.target.change_classification.value,
                 "prompt_version": PROMPT_VERSION},
    }


def build(config_path: Path | None = None) -> dict:
    cfg = yaml.safe_load((config_path or CONFIG_DIR / "dataset.yaml").read_text(encoding="utf-8"))
    rng = random.Random(cfg.get("seed", 13))
    Paths.ensure()

    synth = {s: [r for r in load_records(Paths.training_synthetic / f"{s}.jsonl") if _eligible(r)] for s in ("train", "val")}
    real_all = []
    for p in sorted(Paths.training_real.glob("*.jsonl")):
        real_all += [r for r in load_records(p) if _eligible(r)]
    real = {"train": [], "val": [], "test": []}
    for r in real_all:
        real[_real_split(r.new.doc_id, cfg["real_test_fraction"], cfg["real_val_fraction"])].append(r)

    # Cap examples per scenario to keep classes balanced.
    cap = cfg.get("max_per_scenario")
    if cap:
        for split in synth:
            counts: Counter = Counter()
            kept = []
            for r in synth[split]:
                if counts[r.scenario] < cap:
                    counts[r.scenario] += 1
                    kept.append(r)
            synth[split] = kept

    # Limit synthetic dominance when real data exists.
    ratio = cfg.get("max_synthetic_per_real")
    if ratio and real["train"]:
        limit = int(ratio * len(real["train"]))
        rng.shuffle(synth["train"])
        synth["train"] = synth["train"][:limit]

    train = [_to_chat(r) for r in synth["train"] + real["train"]]
    val = [_to_chat(r) for r in synth["val"] + real["val"]]
    rng.shuffle(train)

    # Leakage checks against held-out evaluation sets.
    test_synth = load_records(Paths.evaluation / "synthetic_test.jsonl")
    test_families = {r.family for r in test_synth}
    train_families = {m["meta"]["family"] for m in train if m["meta"]["data_origin"] == "synthetic"}
    leak_families = sorted(test_families & train_families)
    real_test_ids = {r.new.doc_id for r in real["test"]}
    leak_ids = sorted(real_test_ids & {r.new.doc_id for r in real["train"] + real["val"]})
    if leak_families or leak_ids:
        raise RuntimeError(f"Leakage detected: families={leak_families} doc_ids={leak_ids}")

    Paths.training_final.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (Paths.training_final / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (Paths.evaluation / "real_test.jsonl").open("w", encoding="utf-8") as fh:
        for r in real["test"]:
            fh.write(r.model_dump_json() + "\n")

    manifest = {
        "prompt_version": PROMPT_VERSION,
        "config": cfg,
        "train": len(train), "val": len(val),
        "train_synthetic": len(synth["train"]), "train_real": len(real["train"]),
        "val_synthetic": len(synth["val"]), "val_real": len(real["val"]),
        "real_test": len(real["test"]), "synthetic_test": len(test_synth),
        "train_by_classification": dict(Counter(m["meta"]["classification"] for m in train)),
        "train_by_scenario": dict(Counter(m["meta"]["scenario"] or "real" for m in train)),
        "leakage_check": "passed",
        "train_families": sorted(train_families), "test_families": sorted(test_families),
    }
    (Paths.training_final / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (Paths.reports / "dataset_report.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
