"""Deterministic section-level and sentence-level diffing."""
from __future__ import annotations

from difflib import SequenceMatcher

from careimpact.diffing.sections import canonical_key
from careimpact.diffing.text import normalize_for_compare, normalize_ws, split_units
from careimpact.schemas import DiffOp, DocumentDiff, PolicyDocument, Section, SectionDiff


def _match_sections(old: list[Section], new: list[Section]) -> list[tuple[Section | None, Section | None, str]]:
    """Pair sections by canonical key, then by fuzzy heading similarity."""
    pairs: list[tuple[Section | None, Section | None, str]] = []
    new_left = list(new)
    used_keys: dict[str, int] = {}

    def keyed(sec: Section) -> str:
        return canonical_key(sec.heading)

    unmatched_old: list[Section] = []
    for o in old:
        k = keyed(o)
        hit = next((n for n in new_left if keyed(n) == k), None)
        if hit is not None:
            new_left.remove(hit)
            pairs.append((o, hit, k))
        else:
            unmatched_old.append(o)

    for o in unmatched_old:
        best, best_score = None, 0.0
        for n in new_left:
            score = SequenceMatcher(None, o.heading.lower(), n.heading.lower()).ratio()
            if score > best_score:
                best, best_score = n, score
        if best is not None and best_score >= 0.6:
            new_left.remove(best)
            pairs.append((o, best, keyed(o)))
        else:
            pairs.append((o, None, keyed(o)))
    for n in new_left:
        pairs.append((None, n, keyed(n)))

    # Disambiguate duplicate keys (e.g. two "limitations" sections).
    out = []
    for o, n, k in pairs:
        used_keys[k] = used_keys.get(k, 0) + 1
        out.append((o, n, k if used_keys[k] == 1 else f"{k}_{used_keys[k]}"))
    # Keep the order of the new document where possible.
    order = {id(s): i for i, s in enumerate(new)}
    old_order = {id(s): i for i, s in enumerate(old)}
    out.sort(key=lambda p: (order.get(id(p[1]), old_order.get(id(p[0]), 0) + 0.5)))
    return out


def diff_units(old_text: str, new_text: str) -> tuple[list[DiffOp], float]:
    a, b = split_units(old_text), split_units(new_text)
    na, nb = [normalize_for_compare(x) for x in a], [normalize_for_compare(x) for x in b]
    sm = SequenceMatcher(None, na, nb, autojunk=False)
    ops: list[DiffOp] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ops.append(DiffOp(op=tag, old=a[i1:i2], new=b[j1:j2]))
    char_sim = SequenceMatcher(None, " ".join(na), " ".join(nb), autojunk=False).ratio() if (na or nb) else 1.0
    return ops, round(char_sim, 4)


def diff_sections(old: Section | None, new: Section | None, key: str) -> SectionDiff:
    old_text = old.text if old else ""
    new_text = new.text if new else ""
    if old is None:
        ops, sim = diff_units("", new_text)
        status = "added"
    elif new is None:
        ops, sim = diff_units(old_text, "")
        status = "removed"
    elif normalize_ws(old_text) == normalize_ws(new_text):
        ops, sim = [DiffOp(op="equal", old=split_units(old_text), new=split_units(new_text))], 1.0
        # A renamed heading over identical text is a formatting change, not "unchanged".
        status = "unchanged" if normalize_ws(old.heading) == normalize_ws(new.heading) else "formatting_only"
    else:
        ops, sim = diff_units(old_text, new_text)
        status = "formatting_only" if all(o.op == "equal" for o in ops) else "modified"
    return SectionDiff(
        key=key,
        heading_old=old.heading if old else None,
        heading_new=new.heading if new else None,
        status=status,
        similarity=sim,
        old_text=old_text,
        new_text=new_text,
        ops=ops,
    )


def diff_documents(old: PolicyDocument, new: PolicyDocument) -> DocumentDiff:
    return DocumentDiff(sections=[diff_sections(o, n, k) for o, n, k in _match_sections(old.sections, new.sections)])
