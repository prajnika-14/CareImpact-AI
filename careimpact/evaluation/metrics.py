"""Per-example scoring and aggregation.

Matching rules (kept deliberately simple and documented in the report):
* Sections are compared by canonical key (see diffing.sections).
* A predicted finding matches an expected one when their sections match;
  category agreement is scored separately.
* Evidence exact-match uses whitespace-normalised substring search.
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from careimpact.analysis.validation import check_grounding
from careimpact.diffing.sections import canonical_key
from careimpact.schemas import SUBSTANTIVE_CLASSES, ChangeClass, DocumentDiff, ExampleRecord, PolicyAnalysis


def _keys(headings) -> set[str]:
    return {canonical_key(h) for h in headings}


def score_example(rec: ExampleRecord, pred: PolicyAnalysis | None, diff: DocumentDiff, parse_errors: list[str]) -> dict:
    gold = rec.target
    row: dict = {
        "example_id": rec.example_id, "scenario": rec.scenario, "gold_class": gold.change_classification.value,
        "json_valid": pred is not None or not any("not JSON" in e or "invalid JSON" in e for e in parse_errors),
        "schema_valid": pred is not None, "parse_errors": parse_errors,
    }
    n_gold_sub = sum(f.category in SUBSTANTIVE_CLASSES for f in gold.findings)
    if pred is None:
        # Invalid output: every expected substantive finding counts as missed.
        row.update(pred_class=None, class_correct=False, substantive_gold=gold.is_substantive,
                   substantive_pred=None, gold_substantive_findings=n_gold_sub, matched_substantive_findings=0,
                   gold_abstain=gold.abstain, pred_abstain=False,
                   no_change_case=not gold.is_substantive and gold.change_classification != ChangeClass.INSUFFICIENT_EVIDENCE)
        return row
    gold_sub = gold.is_substantive
    pred_sub = pred.change_classification in SUBSTANTIVE_CLASSES
    grounding = check_grounding(pred, rec.old, rec.new, diff)

    gold_sub_findings = [f for f in gold.findings if f.category in SUBSTANTIVE_CLASSES]
    pred_keys = [canonical_key(f.section) for f in pred.findings]
    matched_gold = [f for f in gold_sub_findings if canonical_key(f.section) in pred_keys]
    cat_ok = sum(
        any(canonical_key(p.section) == canonical_key(g.section) and p.category == g.category for p in pred.findings)
        for g in gold_sub_findings
    )
    gold_all_keys = {canonical_key(f.section) for f in gold.findings}
    pred_matched = sum(k in gold_all_keys for k in pred_keys)
    unsupported = {i["finding"] for i in grounding["issues"] if i["type"] in {"unsupported_quote", "no_evidence", "section_not_changed"}}
    gold_sections = _keys(gold.changed_sections)
    pred_sections = _keys(pred.changed_sections) | set(pred_keys)

    row.update(
        pred_class=pred.change_classification.value,
        class_correct=pred.change_classification == gold.change_classification,
        substantive_gold=gold_sub, substantive_pred=pred_sub,
        gold_substantive_findings=len(gold_sub_findings), matched_substantive_findings=len(matched_gold),
        category_correct_findings=cat_ok,
        pred_findings=len(pred.findings), pred_findings_matched=pred_matched,
        unsupported_findings=len(unsupported),
        quotes=grounding["quotes"], quotes_matched=grounding["quotes_matched"],
        section_recall=(len(gold_sections & pred_sections) / len(gold_sections)) if gold_sections else None,
        gold_abstain=gold.abstain, pred_abstain=pred.abstain,
        no_change_case=not gold_sub and gold.change_classification != ChangeClass.INSUFFICIENT_EVIDENCE,
    )
    return row


def _rate(num: float, den: float):
    return round(num / den, 4) if den else None


def aggregate(rows: list[dict], latencies: list[float]) -> dict:
    n = len(rows)
    valid = [r for r in rows if r["schema_valid"]]
    s = lambda key, rs=valid: sum(r.get(key) or 0 for r in rs)  # noqa: E731
    # Invalid outputs are counted as predicting "not substantive" (they report nothing).
    tp = sum(bool(r["substantive_pred"]) and r["substantive_gold"] for r in rows)
    fp = sum(bool(r["substantive_pred"]) and not r["substantive_gold"] for r in rows)
    fn = sum((not r["substantive_pred"]) and r["substantive_gold"] for r in rows)
    prec, rec = _rate(tp, tp + fp), _rate(tp, tp + fn)
    gold_f = s("gold_substantive_findings", rows)
    matched = s("matched_substantive_findings", rows)
    pred_f = s("pred_findings")
    no_change = [r for r in rows if r["no_change_case"]]
    abst = [r for r in rows if r["gold_abstain"]]
    non_abst = [r for r in valid if not r["gold_abstain"]]
    sec = [r["section_recall"] for r in valid if r.get("section_recall") is not None]
    by_class: dict[str, list] = defaultdict(list)
    by_scenario: dict[str, list] = defaultdict(list)
    for r in rows:
        by_class[r["gold_class"]].append(r["class_correct"])
        by_scenario[r["scenario"] or "real"].append(r["class_correct"])
    lat = sorted(latencies)
    return {
        "n": n,
        "json_valid_rate": _rate(sum(r["json_valid"] for r in rows), n),
        "schema_valid_rate": _rate(len(valid), n),
        "classification_accuracy": _rate(sum(r["class_correct"] for r in rows), n),
        "substantive_precision": prec,
        "substantive_recall": rec,
        "substantive_f1": round(2 * prec * rec / (prec + rec), 4) if prec and rec else None,
        "finding_recall": _rate(matched, gold_f),
        "missed_important_change_rate": round(1 - matched / gold_f, 4) if gold_f else None,
        "finding_category_accuracy": _rate(s("category_correct_findings"), gold_f),
        "finding_precision": _rate(s("pred_findings_matched"), pred_f),
        "unsupported_claim_rate": _rate(s("unsupported_findings"), pred_f),
        "evidence_exact_match_rate": _rate(s("quotes_matched"), s("quotes")),
        "section_coverage": round(statistics.mean(sec), 4) if sec else None,
        "no_change_accuracy": _rate(sum(r["substantive_pred"] is False for r in no_change), len(no_change)),
        "abstention_recall": _rate(sum(bool(r["pred_abstain"]) for r in abst), len(abst)),
        "false_abstention_rate": _rate(sum(bool(r["pred_abstain"]) for r in non_abst), len(non_abst)),
        "latency_mean_s": round(statistics.mean(lat), 3) if lat else None,
        "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 3) if lat else None,
        "accuracy_by_class": {k: _rate(sum(v), len(v)) for k, v in sorted(by_class.items())},
        "accuracy_by_scenario": {k: _rate(sum(v), len(v)) for k, v in sorted(by_scenario.items())},
    }
