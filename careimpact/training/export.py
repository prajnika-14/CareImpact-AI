"""Merge a LoRA adapter into its base model and prepare it for Ollama.

    python -m careimpact.training.export --run models/runs/cpu_smoke --name careimpact-ft

Produces ``<run>/merged`` (Hugging Face safetensors) and ``<run>/Modelfile``.
With ``--llama-cpp <checkout>`` it also converts to ``<run>/<name>.gguf`` (q8_0 by
default) and points the Modelfile at it. Use this route: recent Ollama versions
cannot import Qwen2 safetensors directly ("unsupported MLX architecture").

       python -m careimpact.training.export --run models/runs/<run> --llama-cpp ../llama.cpp
       cd models/runs/<run>; ollama create careimpact-ft -f Modelfile

The Modelfile reuses Qwen's ChatML template so prompts match training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from careimpact.config import ROOT

QWEN_TEMPLATE = '''{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
'''


def _to_gguf(merged: Path, out: Path, llama_cpp: Path, outtype: str) -> None:
    """Convert with llama.cpp's converter, using its bundled gguf-py so nothing extra is installed."""
    script = llama_cpp / "convert_hf_to_gguf.py"
    if not script.exists():
        raise SystemExit(f"{script} not found. Clone it first: git clone --depth 1 https://github.com/ggml-org/llama.cpp")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(llama_cpp / "gguf-py"), os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, str(script), str(merged), "--outfile", str(out), "--outtype", outtype], env=env, check=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory, e.g. models/runs/cpu_smoke")
    ap.add_argument("--name", default="careimpact-ft")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                    help="precision of the merged weights; float16 halves the download vs float32")
    ap.add_argument("--llama-cpp", help="path to a llama.cpp checkout; when given, also writes <name>.gguf "
                                        "(recommended: recent Ollama cannot import Qwen2 safetensors directly)")
    ap.add_argument("--gguf-type", default="q8_0", choices=["q8_0", "f16", "bf16"])
    args = ap.parse_args(argv)
    run = (ROOT / args.run) if not Path(args.run).is_absolute() else Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.float32)
    # Merge in float32 for accuracy, then store in the requested precision.
    model = PeftModel.from_pretrained(base, run / "adapter").merge_and_unload().to(getattr(torch, args.dtype))
    merged = run / "merged"
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(run / "adapter").save_pretrained(merged)

    source = "./merged"
    if args.llama_cpp:
        gguf = run / f"{args.name}.gguf"
        _to_gguf(merged, gguf, Path(args.llama_cpp), args.gguf_type)
        source = f"./{gguf.name}"
    modelfile = (f"FROM {source}\nTEMPLATE \"\"\"{QWEN_TEMPLATE}\"\"\"\n"
                 "PARAMETER temperature 0\nPARAMETER num_ctx 4096\n"
                 'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|im_start|>"\n')
    (run / "Modelfile").write_text(modelfile, encoding="utf-8")
    summary_p = run / "summary.json"
    s = json.loads(summary_p.read_text(encoding="utf-8"))
    s.update(merged_dir=str(merged), ollama_name=args.name)
    summary_p.write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"Model for Ollama: {run / source}\nNext: cd {run}; ollama create {args.name} -f Modelfile")


if __name__ == "__main__":
    main()
