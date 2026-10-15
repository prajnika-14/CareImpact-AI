"""Grounding checks: does an analysis only quote text that exists in the source?"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from careimpact.diffing.sections import canonical_key
from careimpact.diffing.text import quote_in_source
from careimpact.schemas import (
    SUBSTANTIVE_CLASSES,
    ChangeClass,
    DocumentDiff,
    PolicyAnalysis,
    PolicyDocument,
)


def _doc_text(doc: PolicyDocument) -> str:
    return "\n".join(f"{s.heading}\n{s.text}" for s in doc.sections)


def parse_analysis(raw: str) -> tuple[PolicyAnalysis | None, list[str]]:
    """Parse model output into a PolicyAnalysis; returns (analysis, errors)."""
    errors: list[str] = []
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None, ["output is not JSON"]
        try:
            data = json.loads(m.group(0))
            errors.append("JSON had surrounding text")
        except json.JSONDecodeError as exc:
            return None, [f"invalid JSON: {exc.msg}"]
    try:
        return PolicyAnalysis.model_validate(data), errors
    except ValidationError as exc:
        return None, errors + [f"schema: {e['loc']} {e['msg']}" for e in exc.errors()[:8]]


def check_grounding(analysis: PolicyAnalysis, old: PolicyDocument, new: PolicyDocument, diff: DocumentDiff) -> dict:
    """Evidence exact-match and consistency report for one analysis."""
    old_text, new_text = _doc_text(old), _doc_text(new)
    changed_keys = {s.key for s in diff.changed} | {canonical_key(s.heading) for s in diff.changed}
    quotes = matched = 0
    issues: list[dict] = []
    for i, f in enumerate(analysis.findings):
        for side, quote, source in (("old", f.evidence_old, old_text), ("new", f.evidence_new, new_text)):
            if quote is None:
                continue
            quotes += 1
            if quote_in_source(quote, source):
                matched += 1
            else:
                issues.append({"finding": i, "type": "unsupported_quote", "side": side, "quote": quote[:200]})
        if f.evidence_old is None and f.evidence_new is None:
            issues.append({"finding": i, "type": "no_evidence"})
        if canonical_key(f.section) not in changed_keys:
            issues.append({"finding": i, "type": "section_not_changed", "section": f.section})
    if analysis.is_substantive != (analysis.change_classification in SUBSTANTIVE_CLASSES) and \
            analysis.change_classification != ChangeClass.INSUFFICIENT_EVIDENCE:
        issues.append({"type": "substantive_flag_inconsistent"})
    return {
        "quotes": quotes,
        "quotes_matched": matched,
        "evidence_exact_match_rate": (matched / quotes) if quotes else None,
        "issues": issues,
        "grounded": not any(x["type"] in {"unsupported_quote", "no_evidence"} for x in issues),
    }
