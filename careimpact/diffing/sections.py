"""Section canonicalisation and plain-text section parsing."""
from __future__ import annotations

import re

from careimpact.schemas import Section

# canonical key -> heading fragments that map to it (lower-case, substring match)
CANONICAL_SECTIONS: dict[str, list[str]] = {
    "retirement_notice": ["retirement notice", "retirement and replacement"],
    "coverage_indications": [
        "coverage indications", "indications and limitations", "covered indications",
        "indications for coverage", "coverage criteria", "covered services",
    ],
    "limitations": ["limitations", "non-covered", "noncovered", "contraindications", "exclusions"],
    "eligibility": ["patient selection", "eligibility", "beneficiary criteria", "medical necessity criteria"],
    "utilization_guidelines": ["utilization", "frequency", "duration of therapy"],
    "documentation_requirements": ["documentation"],
    "provider_qualifications": ["provider qualification", "practitioner qualification", "qualified provider", "supervision"],
    "coding": ["icd-10", "diagnosis code", "coding information", "codes that support", "code list"],
    "billing": ["billing", "coding guidelines", "claim submission"],
    "evidence_requirements": ["required testing", "supporting evidence", "evidence requirement", "clinical evidence required"],
    "summary_of_evidence": ["summary of evidence", "analysis of evidence", "rationale"],
    "effective_dates": ["applicability", "implementation", "effective date"],
    "national_policy": ["national coverage policy"],
    "issue": ["issue"],
    "associated_information": ["associated information"],
    "sources": ["sources of information", "bibliography", "references"],
    "contacts": ["contact", "contractor information"],
}


def canonical_key(heading: str) -> str:
    h = heading.lower().strip()
    for key, frags in CANONICAL_SECTIONS.items():
        if any(f in h for f in frags):
            return key
    return re.sub(r"[^a-z0-9]+", "_", h).strip("_") or "section"


_HEADING_MD = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$")


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90 or s.endswith((".", ";", ",")):
        return False
    if s.endswith(":") and len(s) < 70:
        return True
    words = re.findall(r"[A-Za-z][A-Za-z/&-]*", s)
    if not words or len(words) > 10:
        return False
    if s.isupper() and len(words) >= 1:
        return True
    small = {"and", "or", "of", "the", "for", "to", "in", "a", "an", "on", "with", "and/or"}
    return all(w[0].isupper() or w.lower() in small for w in words) and len(words) >= 2


def parse_sections_from_text(text: str) -> list[Section]:
    """Split pasted plain text into sections using markdown or title-case headings."""
    sections: list[Section] = []
    heading = "Policy Text"
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(heading=heading, text=body))

    for raw in (text or "").splitlines():
        m = _HEADING_MD.match(raw)
        if m or _looks_like_heading(raw):
            flush()
            heading = (m.group(1) if m else raw.strip()).rstrip(":").strip()
            buf = []
        else:
            buf.append(raw.rstrip())
    flush()
    return sections
