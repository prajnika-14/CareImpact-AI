"""Batch generation with balancing, deduplication and reporting."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from careimpact.config import Paths
from careimpact.diffing import diff_documents, normalize_ws
from careimpact.schemas import ExampleRecord
from careimpact.synthetic.generator import GENERATOR_VERSION, SCENARIOS, generate_example
from careimpact.synthetic.topics import SPLIT_FAMILIES

NEAR_DUP_JACCARD = 0.9
# Over-sample negatives and hard cases so the model learns to report "no
# substantive change" and to abstain, not only to find changes.
SCENARIO_WEIGHTS = {"no_substantive_change": 2, "ambiguous_evidence": 2}


def _changed_text(rec: ExampleRecord) -> str:
    d = diff_documents(rec.old, rec.new)
    return normalize_ws(" ".join(s.old_text + " || " + s.new_text for s in d.changed)).lower()


def _shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9.]+", text)
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def output_path(split: str) -> Path:
    if split == "test":
        return Paths.evaluation / "synthetic_test.jsonl"
    return Paths.training_synthetic / f"{split}.jsonl"


def generate_split(split: str, n: int, seed: int, scenarios: list[str] | None = None) -> dict:
    """Generate ``n`` validated, deduplicated examples for ``split``."""
    Paths.ensure()
    families = SPLIT_FAMILIES[split]
    scenarios = scenarios or list(SCENARIOS)
    rng = random.Random(seed)
    accepted: list[ExampleRecord] = []
    failed: list[ExampleRecord] = []
    exact: set[str] = set()
    near: dict[tuple, list[set]] = defaultdict(list)
    dup_exact = dup_near = errors = 0
    attempt = 0
    queue: list[str] = []
    while len(accepted) < n and attempt < n * 6:
        if not queue:  # balanced round-robin over scenarios, shuffled per round
            queue = [sc for sc in scenarios for _ in range(SCENARIO_WEIGHTS.get(sc, 1))]
            rng.shuffle(queue)
        scenario = queue.pop()
        family = rng.choice(families)
        ex_seed = seed * 1_000_003 + attempt
        attempt += 1
        try:
            rec = generate_example(scenario, family, ex_seed, f"syn-{split}-{seed}-{attempt:05d}")
        except Exception as exc:  # generator bug: record and continue
            errors += 1
            failed.append({"scenario": scenario, "family": family, "seed": ex_seed, "error": repr(exc)})  # type: ignore[arg-type]
            continue
        if rec.validation_status != "auto_validated":
            failed.append(rec)
            continue
        text = _changed_text(rec)
        h = hashlib.sha1(text.encode()).hexdigest()
        if h in exact:
            dup_exact += 1
            continue
        sh = _shingles(text)
        group = near[(family, scenario)]
        if any(len(sh & o) / max(1, len(sh | o)) >= NEAR_DUP_JACCARD for o in group):
            dup_near += 1
            continue
        exact.add(h)
        group.append(sh)
        accepted.append(rec)

    out = output_path(split)
    with out.open("w", encoding="utf-8") as fh:
        for rec in accepted:
            fh.write(rec.model_dump_json() + "\n")
    review = Paths.training_review / f"synthetic_{split}_failed.jsonl"
    with review.open("w", encoding="utf-8") as fh:
        for rec in failed:
            fh.write((rec.model_dump_json() if isinstance(rec, ExampleRecord) else json.dumps(rec)) + "\n")

    report = {
        "split": split,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "requested": n,
        "accepted": len(accepted),
        "attempts": attempt,
        "failed_checks": sum(isinstance(r, ExampleRecord) for r in failed),
        "generator_errors": errors,
        "duplicates_exact": dup_exact,
        "duplicates_near": dup_near,
        "families": sorted(families),
        "by_scenario": dict(sorted(Counter(r.scenario for r in accepted).items())),
        "by_family": dict(sorted(Counter(r.family for r in accepted).items())),
        "by_classification": dict(sorted(Counter(r.target.change_classification.value for r in accepted).items())),
        "substantive": sum(r.target.is_substantive for r in accepted),
        "abstain": sum(r.target.abstain for r in accepted),
        "multi_finding": sum(len(r.target.findings) > 1 for r in accepted),
        "output": str(out),
        "failed_output": str(review),
    }
    (Paths.reports / f"synthetic_{split}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
