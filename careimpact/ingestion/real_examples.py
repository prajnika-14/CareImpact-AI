"""Turn verified real pairs into review candidates, and record human review decisions.

Real examples never enter training automatically. A draft (baseline or LLM)
is created with status ``pending_review``; a person must approve it via
``python -m careimpact.cli review``. Drafts with unsupported quotes cannot be
approved.
"""
from __future__ import annotations

import datetime as dt
import json

from careimpact.analysis.validation import check_grounding
from careimpact.config import Paths
from careimpact.diffing import diff_documents
from careimpact.inference.service import BASELINE, predict
from careimpact.ingestion.pipeline import load_document, verified_pairs
from careimpact.schemas import ExampleRecord

CANDIDATES = Paths.training_review / "real_candidates.jsonl"
APPROVED = Paths.training_real / "approved.jsonl"
REJECTED = Paths.training_review / "real_rejected.jsonl"


def _read(path) -> list[ExampleRecord]:
    if not path.exists():
        return []
    return [ExampleRecord.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(path, recs: list[ExampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(r.model_dump_json() + "\n")


def draft_candidates(model: str = BASELINE) -> dict:
    existing = {r.example_id for r in _read(CANDIDATES) + _read(APPROVED) + _read(REJECTED)}
    out = _read(CANDIDATES)
    skipped_identical = made = 0
    for p in verified_pairs():
        ex_id = f"real-{p['pair_id'].replace(':', '-').replace('>', '')}"
        if ex_id in existing:
            continue
        old = load_document(p["doc_id"], p["old_version"])
        new = load_document(p["doc_id"], p["new_version"])
        if old is None or new is None:
            continue
        diff = diff_documents(old, new)
        if not diff.changed:
            skipped_identical += 1
            continue
        res = predict(old, new, model, diff)
        if res["analysis"] is None:
            continue
        g = check_grounding(res["analysis"], old, new, diff)
        out.append(ExampleRecord(
            example_id=ex_id, data_origin="real", family=new.doc_id, old=old, new=new, target=res["analysis"],
            target_source="baseline_draft" if model == BASELINE else "llm_draft",
            validation_status="pending_review",
            validation_errors=[f"{i['type']}: {i.get('quote', i.get('section', ''))}" for i in g["issues"]],
            training_eligible=False, created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        ))
        made += 1
    _write(CANDIDATES, out)
    return {"new_candidates": made, "pending_total": len(out), "skipped_identical": skipped_identical, "file": str(CANDIDATES)}


def review_interactive() -> dict:
    pending = _read(CANDIDATES)
    approved, rejected = _read(APPROVED), _read(REJECTED)
    keep = []
    for i, rec in enumerate(pending):
        t = rec.target
        print("=" * 80)
        print(f"[{i + 1}/{len(pending)}] {rec.example_id}  {rec.new.title}  v{rec.old.version} -> v{rec.new.version}")
        print(f"old: {rec.old.source_url}\nnew: {rec.new.source_url}")
        print(f"classification: {t.change_classification.value}  substantive={t.is_substantive}  abstain={t.abstain}")
        print(f"summary: {t.change_summary}")
        for f in t.findings:
            print(f"  - [{f.category.value}/{f.change_type.value}] {f.section}: {f.what_changed}")
            print(f"      old: {(f.evidence_old or '-')[:200]}\n      new: {(f.evidence_new or '-')[:200]}")
        if rec.validation_errors:
            print("grounding issues:", "; ".join(rec.validation_errors[:5]))
        ans = input("[a]pprove  [r]eject  [s]kip  [q]uit > ").strip().lower()
        if ans == "q":
            keep += pending[i:]
            break
        if ans == "a":
            if any(e.startswith(("unsupported_quote", "no_evidence")) for e in rec.validation_errors):
                print("Cannot approve: draft contains unsupported or missing evidence. Edit the JSONL record or reject it.")
                keep.append(rec)
                continue
            rec.validation_status, rec.training_eligible = "approved", True
            approved.append(rec)
        elif ans == "r":
            rec.validation_status = "rejected"
            rejected.append(rec)
        else:
            keep.append(rec)
    _write(CANDIDATES, keep)
    _write(APPROVED, approved)
    _write(REJECTED, rejected)
    return {"pending": len(keep), "approved": len(approved), "rejected": len(rejected)}


def review_counts() -> dict:
    return {"pending": len(_read(CANDIDATES)), "approved": len(_read(APPROVED)), "rejected": len(_read(REJECTED))}


if __name__ == "__main__":
    print(json.dumps(review_counts()))
