"""Measured dataset / pipeline statistics for the dashboard (no invented numbers)."""
from __future__ import annotations

import json
from pathlib import Path

from careimpact.config import Paths


def _lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dataset_stats() -> dict:
    pairs = []
    if Paths.pairs.exists():
        pairs = [json.loads(x) for x in Paths.pairs.read_text(encoding="utf-8").splitlines() if x.strip()]
    manifest = _json(Paths.training_final / "manifest.json") or {}
    synth_report = _json(Paths.reports / "synthetic_train_report.json") or {}
    collection = _json(Paths.reports / "collection_report.json") or {}
    return {
        "documents_retrieved": len(list(Paths.documents.glob("L*_v*[0-9].json"))) if Paths.documents.exists() else 0,
        "pairs_total": len(pairs),
        "pairs_verified": sum(p["status"] == "verified" for p in pairs),
        "pairs_unverified": sum(p["status"] == "unverified" for p in pairs),
        "pairs_unavailable": sum(p["status"] == "unavailable" for p in pairs),
        "pairs_identical_content": sum(bool(p.get("identical_content")) for p in pairs),
        "synthetic_train": _lines(Paths.training_synthetic / "train.jsonl"),
        "synthetic_val": _lines(Paths.training_synthetic / "val.jsonl"),
        "synthetic_test": _lines(Paths.evaluation / "synthetic_test.jsonl"),
        "synthetic_failed_checks": sum(_lines(p) for p in Paths.training_review.glob("synthetic_*_failed.jsonl")),
        "real_pending_review": _lines(Paths.training_review / "real_candidates.jsonl"),
        "real_approved": _lines(Paths.training_real / "approved.jsonl"),
        "real_rejected": _lines(Paths.training_review / "real_rejected.jsonl"),
        "real_test": _lines(Paths.evaluation / "real_test.jsonl"),
        "training_train": manifest.get("train", 0),
        "training_val": manifest.get("val", 0),
        "training_manifest": manifest or None,
        "synthetic_by_scenario": synth_report.get("by_scenario", {}),
        "synthetic_by_classification": synth_report.get("by_classification", {}),
        "synthetic_generator_version": synth_report.get("generator_version"),
        "collection_last_run": collection.get("run_at"),
        "collection_failed": len(collection.get("failed_retrievals", [])),
    }


def training_runs() -> list[dict]:
    runs = []
    for p in sorted(Paths.runs.glob("*/summary.json")) if Paths.runs.exists() else []:
        data = _json(p)
        if data:
            data["run_dir"] = p.parent.name
            runs.append(data)
    return runs
