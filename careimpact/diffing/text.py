"""Text normalisation and splitting into quotable units."""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_BULLET = re.compile(r"^\s*(?:[-*•●]|\(?[0-9]{1,2}[.)]|\(?[a-zA-Z][.)]|[ivx]{1,4}[.)])\s+")
# Split after sentence-ending punctuation followed by a capital/opening char,
# but not after common abbreviations.
_SENT = re.compile(
    r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\betc\.)(?<!\bvs\.)(?<!\bNo\.)(?<!\bU\.S\.)(?<!\bDr\.)(?<=[.!?])\s+(?=[A-Z(\"])"
)


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def normalize_for_compare(text: str) -> str:
    """Whitespace/case/bullet-insensitive form used to align units."""
    t = _BULLET.sub("", text or "")
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    return normalize_ws(t).lower().rstrip(" .;")


def strip_bullet(line: str) -> str:
    return _BULLET.sub("", line).strip()


def split_units(text: str) -> list[str]:
    """Split section text into sentence-like units.

    Every unit is a verbatim (whitespace-trimmed) substring of the input so
    that it can be used as an exact evidence quote.
    """
    units: list[str] = []
    for line in (text or "").splitlines():
        line = strip_bullet(line)
        if not line:
            continue
        for part in _SENT.split(line):
            part = part.strip()
            if part:
                units.append(part)
    return units


def quote_in_source(quote: str | None, source: str) -> bool:
    """Exact-match test used for evidence validation (whitespace-normalised)."""
    if not quote:
        return False
    return normalize_ws(quote) in normalize_ws(source)
