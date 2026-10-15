"""Deterministic diff-only analyzer.

Serves two purposes:
1. The benchmark baseline every model must beat.
2. The fallback the app uses when no language model is reachable.

It never invents content: every quote is a diff unit from the source text, and
the operational language is generic and explicitly marked as unverified.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from careimpact.diffing.text import normalize_for_compare
from careimpact.schemas import (
    CLASS_PRIORITY,
    SUBSTANTIVE_CLASSES,
    ChangeClass,
    ChangeType,
    DocumentDiff,
    Finding,
    ImpactArea,
    PolicyAnalysis,
    PolicyDocument,
    SectionDiff,
)

SECTION_CATEGORY = {
    "retirement_notice": ChangeClass.RETIREMENT_REPLACEMENT,
    "coverage_indications": ChangeClass.COVERAGE,
    "limitations": ChangeClass.COVERAGE,
    "utilization_guidelines": ChangeClass.COVERAGE,
    "eligibility": ChangeClass.CLINICAL_CRITERIA,
    "evidence_requirements": ChangeClass.CLINICAL_CRITERIA,
    "documentation_requirements": ChangeClass.DOCUMENTATION,
    "coding": ChangeClass.CODING_BILLING,
    "billing": ChangeClass.CODING_BILLING,
    "provider_qualifications": ChangeClass.PROVIDER_REQUIREMENTS,
    "summary_of_evidence": ChangeClass.ADMINISTRATIVE,
    "effective_dates": ChangeClass.ADMINISTRATIVE,
    "sources": ChangeClass.ADMINISTRATIVE,
    "contacts": ChangeClass.ADMINISTRATIVE,
    "associated_information": ChangeClass.ADMINISTRATIVE,
    "issue": ChangeClass.ADMINISTRATIVE,
    "national_policy": ChangeClass.ADMINISTRATIVE,
}

IMPACT = {
    ChangeClass.COVERAGE: [ImpactArea.PATIENT_ELIGIBILITY, ImpactArea.BILLING],
    ChangeClass.CLINICAL_CRITERIA: [ImpactArea.PATIENT_ELIGIBILITY, ImpactArea.PROVIDER_WORKFLOW],
    ChangeClass.DOCUMENTATION: [ImpactArea.DOCUMENTATION, ImpactArea.COMPLIANCE],
    ChangeClass.CODING_BILLING: [ImpactArea.BILLING],
    ChangeClass.PROVIDER_REQUIREMENTS: [ImpactArea.PROVIDER_WORKFLOW, ImpactArea.COMPLIANCE],
    ChangeClass.RETIREMENT_REPLACEMENT: [ImpactArea.COMPLIANCE, ImpactArea.BILLING],
    ChangeClass.ADMINISTRATIVE: [],
}

UNAVAILABLE = re.compile(r"could not be retrieved|text unavailable|\[truncated\]|content not available", re.I)
REWORD_THRESHOLD = 0.82


def _category(sd: SectionDiff) -> ChangeClass:
    if sd.key.split("_")[0] == "retirement" or re.search(r"\bretired\b", sd.new_text, re.I) and sd.status == "added":
        return ChangeClass.RETIREMENT_REPLACEMENT
    base = re.sub(r"_\d+$", "", sd.key)
    return SECTION_CATEGORY.get(base, ChangeClass.ADMINISTRATIVE)


def _is_reword(sd: SectionDiff) -> bool:
    """All replaced units are close paraphrases and nothing purely added/removed."""
    if sd.status == "formatting_only":
        return True
    for op in sd.ops:
        if op.op in {"insert", "delete"}:
            return False
        if op.op == "replace":
            if len(op.old) != len(op.new):
                return False
            for a, b in zip(op.old, op.new):
                na, nb = normalize_for_compare(a), normalize_for_compare(b)
                if re.findall(r"\d+(?:\.\d+)?", na) != re.findall(r"\d+(?:\.\d+)?", nb):
                    return False
                if SequenceMatcher(None, na, nb).ratio() < REWORD_THRESHOLD:
                    return False
    return True


def _section_finding(sd: SectionDiff) -> Finding:
    cat = _category(sd)
    first = next((o for o in sd.ops if o.op != "equal"), None)
    ev_old = first.old[0] if first and first.old else None
    ev_new = first.new[0] if first and first.new else None
    if sd.status == "added" or (first and first.op == "insert"):
        ctype, verb = ChangeType.ADDED, "added"
    elif sd.status == "removed" or (first and first.op == "delete"):
        ctype, verb = ChangeType.REMOVED, "removed"
    else:
        ctype, verb = ChangeType.MODIFIED, "modified"
    n_changed = sum(max(len(o.old), len(o.new)) for o in sd.ops if o.op != "equal")
    return Finding(
        section=sd.heading,
        change_type=ctype,
        category=cat,
        what_changed=f"Text was {verb} in the {sd.heading} section ({n_changed} changed sentence(s)).",
        evidence_old=ev_old,
        evidence_new=ev_new,
        impact_areas=IMPACT.get(cat, []),
        potential_operational_impact=f"Teams that rely on the {sd.heading} section may need to review the change.",
        recommended_action="Review the highlighted text against the official policy before changing workflows.",
        confidence="low",
        requires_verification=True,
    )


def analyze_diff_only(old: PolicyDocument, new: PolicyDocument, diff: DocumentDiff) -> PolicyAnalysis:
    changed = diff.changed
    limitations = [
        "Deterministic diff baseline: categories are inferred from section headings, not from the meaning of the text.",
        "Operational impact has not been independently verified.",
    ]
    if any(UNAVAILABLE.search(s.new_text) or UNAVAILABLE.search(s.old_text) for s in changed):
        return PolicyAnalysis(
            document_id=new.doc_id,
            change_classification=ChangeClass.INSUFFICIENT_EVIDENCE,
            is_substantive=False,
            change_summary="Some section text is missing or could not be retrieved, so the comparison is incomplete.",
            changed_sections=[s.heading for s in changed],
            findings=[],
            abstain=True,
            limitations=limitations + ["Missing source text prevents a reliable comparison."],
        )
    findings = [_section_finding(sd) for sd in changed if not _is_reword(sd)]
    reworded = [sd for sd in changed if _is_reword(sd)]
    if not findings:
        cls = ChangeClass.NO_SUBSTANTIVE_CHANGE
        summary = ("Only wording or formatting differences were detected." if reworded
                   else "No text differences were detected between the supplied versions.")
    else:
        cats = {f.category for f in findings}
        cls = next(c for c in CLASS_PRIORITY if c in cats)
        summary = "Changes detected in: " + ", ".join(dict.fromkeys(f.section for f in findings)) + "."
    return PolicyAnalysis(
        document_id=new.doc_id,
        change_classification=cls,
        is_substantive=cls in SUBSTANTIVE_CLASSES,
        change_summary=summary,
        changed_sections=[s.heading for s in changed],
        findings=findings,
        abstain=False,
        limitations=limitations,
    )
