"""Automated collection: discover LCDs -> pair versions -> retrieve -> validate -> report.

Pairs are (previous public version, current version) taken from the MCD
"Public Versions" table on the current page. Nothing is fabricated: if an
older version cannot be retrieved or validated, the pair is recorded as
``unavailable`` or ``unverified`` with reasons.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

from careimpact.config import Paths, settings
from careimpact.ingestion.cms_api import CoverageAPI
from careimpact.ingestion.http import PoliteClient
from careimpact.ingestion.mcd_pages import lcd_url, parse_lcd_page
from careimpact.schemas import PolicyDocument

log = logging.getLogger(__name__)


def doc_path(doc_id: str, version: str) -> Path:
    return Paths.documents / f"{doc_id}_v{version}.json"


def load_document(doc_id: str, version: str) -> Optional[PolicyDocument]:
    p = doc_path(doc_id, version)
    return PolicyDocument.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None


def fetch_document(http: PoliteClient, lcd_id: int, ver: int, refresh: bool = False) -> tuple[Optional[PolicyDocument], dict]:
    """Retrieve + parse + validate one LCD version, saving raw HTML and the parsed document."""
    existing = None if refresh else load_document(f"L{lcd_id}", str(ver))
    if existing is not None:
        return existing, {"from_store": True, "versions": json.loads((Paths.documents / f"L{lcd_id}_v{ver}.versions.json").read_text())
                          if (Paths.documents / f"L{lcd_id}_v{ver}.versions.json").exists() else []}
    url = lcd_url(lcd_id, ver)
    res = http.get(url, use_cache=not refresh)
    if not res.ok:
        return None, {"error": res.error, "url": url}
    raw_dir = Paths.raw / "mcd_pages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"L{lcd_id}_v{ver}.html").write_text(res.text, encoding="utf-8")
    try:
        doc, extras = parse_lcd_page(res.text, url, res.fetched_at)
    except Exception as exc:  # malformed HTML
        return None, {"error": f"parse failed: {exc}", "url": url}
    if res.final_url != url:
        doc.verification_notes.append(f"redirected to {res.final_url}")
    Paths.documents.mkdir(parents=True, exist_ok=True)
    doc_path(doc.doc_id, doc.version or str(ver)).write_text(doc.model_dump_json(indent=1), encoding="utf-8")
    (Paths.documents / f"L{lcd_id}_v{ver}.versions.json").write_text(json.dumps(extras["versions"]), encoding="utf-8")
    return doc, extras


def collect(limit: int = 10, lcd_ids: Optional[list[int]] = None, refresh: bool = False,
            include_retired: bool = False) -> dict:
    Paths.ensure()
    http = PoliteClient()
    api = CoverageAPI(http)
    listing = api.final_lcds(use_cache=not refresh)
    (Paths.raw / "cms_api").mkdir(parents=True, exist_ok=True)
    (Paths.raw / "cms_api" / "final_lcds.json").write_text(json.dumps(listing, indent=1), encoding="utf-8")

    if lcd_ids:
        wanted = {int(x) for x in lcd_ids}
        targets = [r for r in listing if r["document_id"] in wanted]
    else:
        targets = [r for r in listing if include_retired or r.get("retirement_date") in (None, "", "N/A")]
        targets.sort(key=lambda r: r.get("updated_on_sort") or "", reverse=True)
        targets = targets[:limit]

    existing_pairs = {}
    if Paths.pairs.exists():
        for line in Paths.pairs.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                existing_pairs[p["pair_id"]] = p

    pairs, failures = [], []
    for row in targets:
        lcd_id, cur_ver = int(row["document_id"]), int(row["document_version"])
        new_doc, extras = fetch_document(http, lcd_id, cur_ver, refresh)
        if new_doc is None:
            failures.append({"lcd_id": lcd_id, "version": cur_ver, **extras})
            continue
        older = sorted({v["version"] for v in extras.get("versions", []) if v.get("version") and v["version"] < cur_ver}, reverse=True)
        pair_base = {"doc_id": new_doc.doc_id, "title": new_doc.title, "new_version": str(cur_ver),
                     "new_url": new_doc.source_url, "contractor": new_doc.contractor,
                     "related_documents": new_doc.related_documents}
        if not older:
            pairs.append({**pair_base, "pair_id": f"{new_doc.doc_id}:none->{cur_ver}", "old_version": None, "old_url": None,
                          "status": "unavailable", "reasons": ["no earlier public version listed on the MCD page"]})
            continue
        old_ver = older[0]
        old_doc, old_extras = fetch_document(http, lcd_id, old_ver, refresh)
        reasons: list[str] = []
        if old_doc is None:
            status = "unavailable"
            reasons.append(old_extras.get("error", "retrieval failed"))
        else:
            reasons += [f"old: {n}" for n in old_doc.verification_notes] + [f"new: {n}" for n in new_doc.verification_notes]
            if old_doc.version != str(old_ver):
                reasons.append(f"requested version {old_ver} but page reports {old_doc.version}")
            status = "verified" if (old_doc.verification_status == new_doc.verification_status == "verified" and not reasons) else "unverified"
            if old_doc.content_hash == new_doc.content_hash:
                reasons.append("policy content identical between versions")
        pairs.append({**pair_base, "pair_id": f"{new_doc.doc_id}:{old_ver}->{cur_ver}", "old_version": str(old_ver),
                      "old_url": lcd_url(lcd_id, old_ver), "status": status, "reasons": reasons,
                      "identical_content": bool(old_doc and old_doc.content_hash == new_doc.content_hash),
                      "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})

    if settings()["accept_cms_api_license"]:
        for p in pairs:
            if p["status"] == "verified":
                try:
                    p["revision_history"] = api.revision_history(int(p["doc_id"].lstrip("L")), int(p["new_version"]))[:5]
                except Exception as exc:
                    p["revision_history_error"] = str(exc)

    for p in pairs:
        existing_pairs[p["pair_id"]] = p
    with Paths.pairs.open("w", encoding="utf-8") as fh:
        for p in existing_pairs.values():
            fh.write(json.dumps(p) + "\n")

    all_pairs = list(existing_pairs.values())
    report = {
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "lcds_listed": len(listing),
        "lcds_targeted": len(targets),
        "documents_stored": len(list(Paths.documents.glob("L*_v*[0-9].json"))),
        "pairs_this_run": len(pairs),
        "pairs_total": len(all_pairs),
        "pairs_by_status": {s: sum(p["status"] == s for p in all_pairs) for s in ("verified", "unverified", "unavailable")},
        "identical_content_pairs": sum(bool(p.get("identical_content")) for p in all_pairs),
        "failed_retrievals": failures,
        "unresolved_pairs": [{"pair_id": p["pair_id"], "status": p["status"], "reasons": p["reasons"]}
                             for p in all_pairs if p["status"] != "verified"],
    }
    (Paths.reports / "collection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def verified_pairs() -> list[dict]:
    if not Paths.pairs.exists():
        return []
    return [p for p in map(json.loads, Paths.pairs.read_text(encoding="utf-8").splitlines()) if p["status"] == "verified"]
