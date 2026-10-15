"""Prompt construction shared by training-data export and inference.

Training and inference MUST use the same builder so the fine-tuned model sees
inputs in exactly the format it was trained on.
"""
from __future__ import annotations

import json

from careimpact.config import settings
from careimpact.schemas import DocumentDiff, PolicyAnalysis, PolicyDocument, SectionDiff

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are CareImpact AI, an assistant that compares two versions of a Medicare coverage policy.
Rules:
- Use ONLY the supplied policy text. Do not use outside knowledge about CMS policies.
- evidence_old and evidence_new must be exact quotes copied from the OLD and NEW text. Use null when a side has no relevant text (e.g. an addition has no old quote).
- Distinguish substantive changes (coverage, clinical criteria, documentation, coding/billing, provider requirements, retirement) from administrative or wording-only changes.
- If wording differs but the meaning is the same, report change_classification "no_substantive_change" with no findings.
- If the text is missing, truncated or contradictory, set abstain to true, use "insufficient_evidence", and explain in limitations.
- Operational impacts are inferences, not official CMS guidance. Confidence is a qualitative label, not a probability.
Return a single JSON object with keys: document_id, change_classification, is_substantive, change_summary, changed_sections, findings, abstain, limitations.
Each finding has: section, change_type (added|removed|modified|reworded), category, what_changed, evidence_old, evidence_new, impact_areas, potential_operational_impact, recommended_action, confidence (low|medium|high), requires_verification.
Allowed change_classification/category values: coverage, clinical_criteria, documentation, coding_billing, provider_requirements, retirement_replacement, administrative, no_substantive_change, insufficient_evidence.
Allowed impact_areas: patient_eligibility, provider_workflow, documentation, billing, compliance, implementation_timeline."""


def _excerpt(sd: SectionDiff, text: str, side: str, max_chars: int) -> str:
    """Full section text if short; otherwise only changed units with context."""
    if len(text) <= max_chars:
        return text
    lines: list[str] = []
    ops = sd.ops
    for i, op in enumerate(ops):
        units = op.old if side == "old" else op.new
        if op.op == "equal":
            near = (i > 0 and ops[i - 1].op != "equal") or (i + 1 < len(ops) and ops[i + 1].op != "equal")
            if near and units:
                lines.append("[...]")
                lines.extend(units[:1] if i > 0 and ops[i - 1].op != "equal" else units[-1:])
        else:
            lines.extend(units)
    out = "\n".join(lines) + "\n[...]"
    return out[: max_chars * 2]


def build_user_message(old: PolicyDocument, new: PolicyDocument, diff: DocumentDiff) -> str:
    max_chars = int(settings()["prompt_max_section_chars"])
    origin = {
        "synthetic": "synthetic (fictional policy for training; not CMS guidance)",
        "real": "real CMS document",
        "user": "user-supplied text (not verified against CMS)",
    }[new.data_origin]
    verification = "verified" if old.verification_status == new.verification_status == "verified" else "NOT verified"
    unchanged = [s.heading for s in diff.sections if s.status == "unchanged"]
    parts = [
        "Compare the two policy versions below.",
        "",
        "POLICY METADATA",
        f"- Document ID: {new.doc_id}",
        f"- Title: {new.title}",
        f"- Data origin: {origin}",
        f"- Old version: {old.version or 'unknown'} (revision effective {old.revision_effective_date or 'unknown'}, status {old.status or 'unknown'})",
        f"- New version: {new.version or 'unknown'} (revision effective {new.revision_effective_date or 'unknown'}, status {new.status or 'unknown'})",
        f"- Source version verification: {verification}",
        "",
        "UNCHANGED SECTIONS: " + ("; ".join(unchanged) if unchanged else "none"),
        "",
        "CHANGED SECTIONS",
    ]
    changed = diff.changed
    if not changed:
        parts.append("(none: the deterministic diff found no text differences)")
    for sd in changed:
        parts.append(f"### Section: {sd.heading}")
        parts.append("[OLD]")
        parts.append(_excerpt(sd, sd.old_text, "old", max_chars) if sd.old_text else "(section not present)")
        parts.append("[NEW]")
        parts.append(_excerpt(sd, sd.new_text, "new", max_chars) if sd.new_text else "(section not present)")
        parts.append("")
    parts.append("Return only the JSON object.")
    return "\n".join(parts)


def target_json(analysis: PolicyAnalysis) -> str:
    return json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False)


def chat_messages(old: PolicyDocument, new: PolicyDocument, diff: DocumentDiff, target: PolicyAnalysis | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(old, new, diff)},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": target_json(target)})
    return msgs
