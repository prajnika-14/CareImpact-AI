"""Run a predictor over a held-out evaluation set and save results."""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

from careimpact.config import Paths
from careimpact.diffing import diff_documents
from careimpact.evaluation.metrics import aggregate, score_example
from careimpact.inference.ollama_client import OllamaClient, OllamaError
from careimpact.inference.service import BASELINE, predict
from careimpact.schemas import ExampleRecord

EVAL_SETS = {
    "synthetic_test": Paths.evaluation / "synthetic_test.jsonl",
    "real_test": Paths.evaluation / "real_test.jsonl",
}


def load_records(path: Path, limit: int | None = None) -> list[ExampleRecord]:
    if not path.exists():
        return []
    recs = [ExampleRecord.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return recs[:limit] if limit else recs


def _memory_snapshot(client: OllamaClient, model: str) -> int | None:
    for m in client.loaded():
        if m.get("name") == model or m.get("model") == model:
            return m.get("size")
    return None


def run_eval(eval_set: str, model: str, limit: int | None = None, progress: bool = True) -> dict:
    path = EVAL_SETS.get(eval_set, Path(eval_set))
    recs = load_records(path, limit)
    if not recs:
        raise FileNotFoundError(f"No evaluation records in {path}. Generate or approve them first.")
    client = OllamaClient()
    if model != BASELINE and not client.available():
        raise OllamaError(f"Ollama is not reachable at {client.base_url}. Start it with 'ollama serve'.")
    rows, latencies, mem = [], [], None
    failures = 0
    t0 = time.perf_counter()
    for i, rec in enumerate(recs, 1):
        diff = diff_documents(rec.old, rec.new)
        try:
            res = predict(rec.old, rec.new, model, diff, client)
        except OllamaError as exc:
            failures += 1
            res = {"analysis": None, "raw": None, "errors": [str(exc)], "latency_s": None}
        row = score_example(rec, res["analysis"], diff, res["errors"])
        row["latency_s"] = res.get("latency_s")
        row["raw_output"] = (res.get("raw") or "")[:4000] if res["analysis"] is None else None
        row["prediction"] = res["analysis"].model_dump(mode="json") if res["analysis"] else None
        rows.append(row)
        if res.get("latency_s") is not None and model != BASELINE:
            latencies.append(res["latency_s"])
        if model != BASELINE and (i == 1 or i % 10 == 0):
            mem = max(filter(None, [mem, _memory_snapshot(client, model)]), default=None)
        if progress and (i % 10 == 0 or i == len(recs)):
            print(f"  [{i}/{len(recs)}] {model} acc so far: {sum(r['class_correct'] for r in rows) / len(rows):.2f}", flush=True)
    metrics = aggregate(rows, latencies)
    metrics["model_memory_bytes"] = mem
    metrics["request_failures"] = failures
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    result = {
        "model": model,
        "eval_set": eval_set,
        "eval_file": str(path),
        "data_origin": "synthetic" if "synthetic" in eval_set else "real",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": round(time.perf_counter() - t0, 1),
        "metrics": metrics,
        "rows": rows,
    }
    Paths.eval_results.mkdir(parents=True, exist_ok=True)
    out = Paths.eval_results / f"{eval_set}__{safe}__{stamp}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    result["output"] = str(out)
    return result


def latest_results() -> list[dict]:
    """Most recent result per (eval_set, model), without per-row detail."""
    best: dict[tuple, dict] = {}
    for p in sorted(Paths.eval_results.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data.pop("rows", None)
        data["file"] = p.name
        best[(data["eval_set"], data["model"])] = data
    return sorted(best.values(), key=lambda d: (d["eval_set"], d["model"]))


def write_report() -> Path:
    """Markdown benchmark report comparing all latest results."""
    res = latest_results()
    keys = ["n", "schema_valid_rate", "classification_accuracy", "substantive_f1", "finding_recall",
            "missed_important_change_rate", "unsupported_claim_rate", "evidence_exact_match_rate",
            "section_coverage", "no_change_accuracy", "abstention_recall", "false_abstention_rate", "latency_mean_s"]
    lines = ["# CareImpact AI benchmark report", "",
             "Generated from saved results in `data/evaluation/results/`. Synthetic and real results are reported",
             "separately; synthetic scores alone are not evidence of real-world policy accuracy.", ""]
    for eval_set in sorted({r["eval_set"] for r in res}):
        lines += [f"## {eval_set}", "", "| metric | " + " | ".join(r["model"] for r in res if r["eval_set"] == eval_set) + " |",
                  "|---|" + "---|" * sum(r["eval_set"] == eval_set for r in res)]
        for k in keys:
            vals = [r["metrics"].get(k) for r in res if r["eval_set"] == eval_set]
            lines.append(f"| {k} | " + " | ".join("—" if v is None else str(v) for v in vals) + " |")
        lines.append("")
    out = Paths.reports / "benchmark_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
