"""Parse Medicare Coverage Database (MCD) LCD pages, including superseded versions.

Version-specific URL: https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=<id>&ver=<ver>

An HTTP 200 is NOT treated as success. A page is "verified" only if the
embedded documentId/documentVersion, the LCD ID field and the title are
present and match the request, and real policy content was extracted.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup, Tag

from careimpact.schemas import PolicyDocument, Section

MCD_BASE = "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx"

# Section fields that are page furniture or metadata rather than policy content.
SKIP_FIELDS = {
    "divCopyrightStatement", "divStickyNote", "divRevisionHistory", "divPublicVersions", "divKeywords",
    "divOpenMeetings", "divCACMeetings", "divMACMeetingURLs", "divMCDPublishDate", "divCommentStartDate",
    "divCommentEndDate", "divProposedLCDContact", "divProposedSourceID", "divLcdId", "divOriginalIcd9LcdIds",
    "divLcdTitle", "divProposedLcdComment", "divProposedLCD", "divOriginalEffectiveDate",
    "divRevisionEffectiveDate", "divRevisionEndingDate", "divRetirementDate", "divNoticePeriodStartDate",
    "divNoticePeriodEndDate", "divSynopsisOfChanges", "divLcdAttachmentText", "divRelatedLocal",
    "divRelatedNational", "divRequestorInformation",
}
LICENSE_PLACEHOLDER = re.compile(r"please accept the license to see the codes", re.I)
DATE = re.compile(r"\d{2}/\d{2}/\d{4}")


def lcd_url(lcd_id: int | str, ver: int | str) -> str:
    return f"{MCD_BASE}?lcdid={int(str(lcd_id).lstrip('L'))}&ver={int(ver)}"


def is_mcd_url(url: str) -> bool:
    u = urlparse(url)
    return u.scheme == "https" and u.netloc in {"www.cms.gov", "cms.gov"} and u.path.lower().startswith("/medicare-coverage-database/view/")


def parse_query(url: str) -> tuple[Optional[int], Optional[int]]:
    q = {k.lower(): v for k, v in parse_qs(urlparse(url).query).items()}
    lcd = q.get("lcdid", [None])[0]
    ver = q.get("ver", [None])[0]
    return (int(lcd) if lcd and lcd.isdigit() else None, int(ver) if ver and ver.isdigit() else None)


def html_to_text(el: Tag) -> str:
    """Block-aware text extraction that keeps list items and paragraphs on their own lines."""
    el = BeautifulSoup(str(el), "lxml")
    for s in el.find_all(["sup", "script", "style"]):
        s.decompose()
    for br in el.find_all("br"):
        br.replace_with("\n")
    for li in el.find_all("li"):
        li.insert_before("\n- ")
        li.insert_after("\n")
    for blk in el.find_all(["p", "div", "tr", "h2", "h3", "h4", "h5", "ol", "ul", "table"]):
        blk.insert_before("\n")
        blk.insert_after("\n")
    for td in el.find_all(["td", "th"]):
        td.insert_after(" | ")
    text = el.get_text()
    # Some MCD pages contain cp1252 bullets that decode to U+FFFD; treat them as bullets.
    text = re.sub(r"(^|\n)\s*�\s*", r"\1- ", text).replace("�", "")
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln or ln == "-":
            continue
        out.append(ln.rstrip(" |"))
    return "\n".join(out)


def _field_text(soup: BeautifulSoup, div_id: str, label: str = "") -> Optional[str]:
    el = soup.find(id=div_id)
    if not el:
        return None
    t = re.sub(r"\s+", " ", el.get_text(" ")).strip()
    if label and t.lower().startswith(label.lower()):
        t = t[len(label):].strip()
    return t or None


def _date(text: Optional[str]) -> Optional[str]:
    m = DATE.search(text or "")
    return m.group(0) if m else None


def parse_versions_table(soup: BeautifulSoup, requested_ver: Optional[int]) -> list[dict]:
    box = soup.find(id="divPublicVersions")
    rows = []
    if not box:
        return rows
    for tr in box.find_all("tr"):
        cells = [re.sub(r"\s+", " ", td.get_text(" ")).strip() for td in tr.find_all("td")]
        if len(cells) < 3:
            continue
        a = tr.find("a", href=True)
        ver = parse_query("https://x/?" + a["href"].split("?", 1)[1])[1] if a and "?" in a["href"] else None
        here = "You are here" in " ".join(cells)
        if ver is None and here:
            ver = requested_ver
        dates = DATE.findall(cells[1]) if len(cells) > 1 else []
        status = next((s for s in ("Currently in Effect", "Superseded", "Retired", "Future Effective") if s in " ".join(cells)), cells[2])
        rows.append({"version": ver, "updated_on": _date(cells[0]), "effective_from": dates[0] if dates else None,
                     "effective_to": dates[1] if len(dates) > 1 else None, "status": status, "is_requested": here})
    return rows


def parse_lcd_page(page_html: str, url: str, retrieved_at: str) -> tuple[PolicyDocument, dict]:
    """Return (document, extras). extras has versions table, archive link and checks."""
    soup = BeautifulSoup(page_html, "lxml")
    req_id, req_ver = parse_query(url)
    notes: list[str] = []

    m_id = re.search(r"let documentId = (\d+);", page_html)
    m_ver = re.search(r"let documentVersion = (\d+);", page_html)
    page_id = int(m_id.group(1)) if m_id else None
    page_ver = int(m_ver.group(1)) if m_ver else None
    lcd_label = _field_text(soup, "divLcdId", "LCD ID")
    title = _field_text(soup, "divLcdTitle", "LCD Title") or ""

    checks = {
        "embedded_id_matches": page_id is not None and page_id == req_id,
        "embedded_version_matches": page_ver is not None and (req_ver is None or page_ver == req_ver),
        "lcd_id_field_matches": bool(lcd_label) and req_id is not None and lcd_label.strip() == f"L{req_id}",
        "title_present": bool(title),
    }

    sections: list[Section] = []
    for field in soup.select("div.document-view-section-field[id]"):
        fid = field.get("id")
        if fid in SKIP_FIELDS:
            continue
        sub = field.find(class_="document-view-section-subheader")
        body = field.find(class_="document-view-section-text")
        if not sub or not body:
            continue
        heading = re.sub(r"\s+", " ", sub.get_text(" ")).strip()
        text = html_to_text(body)
        if not text or text.upper() == "N/A" or LICENSE_PLACEHOLDER.search(text):
            continue
        sections.append(Section(heading=heading, text=text))
    content_chars = sum(len(s.text) for s in sections)
    checks["has_policy_content"] = content_chars >= 300 and any("coverage" in s.heading.lower() for s in sections)

    versions = parse_versions_table(soup, page_ver)
    status = next((v["status"] for v in versions if v["is_requested"]), None)
    banner = _field_text(soup, "divStatusBanner") or ""
    if not status:
        status = "Superseded" if "Superseded" in banner else ("Retired" if "Retired" in banner else None)

    contractor = None
    ctab = soup.find(id="divContractorInformationFields")
    if ctab:
        td = ctab.find("td")
        contractor = re.sub(r"\s+", " ", td.get_text(" ")).strip() if td else None

    related = []
    rel = soup.find(id="divRelatedLocal")
    if rel:
        related = [re.sub(r"\s+", " ", a.get_text(" ")).strip() for a in rel.find_all("a") if a.get_text(strip=True)]
    archive = soup.find(id="hplPublicVersionMcdArchiveLink")

    for k, ok in checks.items():
        if not ok:
            notes.append(f"check failed: {k}")
    verified = all(checks.values())
    doc = PolicyDocument(
        doc_id=lcd_label.strip() if lcd_label else (f"L{req_id}" if req_id else "unknown"),
        title=title,
        version=str(page_ver) if page_ver is not None else (str(req_ver) if req_ver else None),
        status=status,
        contractor=contractor,
        original_effective_date=_date(_field_text(soup, "divOriginalEffectiveDate")),
        revision_effective_date=_date(_field_text(soup, "divRevisionEffectiveDate")),
        revision_ending_date=_date(_field_text(soup, "divRevisionEndingDate")),
        retirement_date=_date(_field_text(soup, "divRetirementDate")),
        source_url=url,
        retrieved_at=retrieved_at,
        data_origin="real",
        verification_status="verified" if verified else "unverified",
        verification_notes=notes,
        related_documents=related,
        sections=sections,
        content_hash=hashlib.sha256("\n".join(s.heading + s.text for s in sections).encode()).hexdigest()[:16],
    )
    extras = {"versions": versions, "checks": checks, "archive_url": archive.get("href") if archive else None,
              "content_chars": content_chars}
    return doc, extras
