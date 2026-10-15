from pathlib import Path

from careimpact.ingestion.mcd_pages import is_mcd_url, lcd_url, parse_lcd_page

FIXTURE = (Path(__file__).parent / "fixtures" / "lcd_page.html").read_text(encoding="utf-8")
URL = "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=99999&ver=4"


def test_parse_verified_page():
    doc, extras = parse_lcd_page(FIXTURE, URL, "2026-01-01T00:00:00Z")
    assert doc.doc_id == "L99999"
    assert doc.version == "4"
    assert doc.verification_status == "verified", doc.verification_notes
    assert doc.status == "Superseded"
    assert doc.revision_effective_date == "01/01/2024"
    assert doc.contractor == "Example MAC, Inc."
    headings = [s.heading for s in doc.sections]
    assert "Coverage Indications, Limitations, and/or Medical Necessity" in headings
    # Furniture, copyright and license placeholders are excluded.
    assert "Copyright" not in headings and "CPT/HCPCS Codes" not in headings
    cov = doc.sections[headings.index("Coverage Indications, Limitations, and/or Medical Necessity")].text
    assert "- The patient has a documented diagnosis of the condition; AND" in cov  # footnote <sup> removed
    assert "navigation" not in cov
    assert extras["versions"][0]["version"] == 5 and extras["versions"][1]["is_requested"]
    assert doc.related_documents == ["A59999 - Billing and Coding: Example Test Policy"]


def test_version_mismatch_is_not_verified():
    # Requesting ver=3 but the page reports version 4: HTTP 200 is not success.
    doc, extras = parse_lcd_page(FIXTURE, URL.replace("ver=4", "ver=3"), "t")
    assert doc.verification_status == "unverified"
    assert not extras["checks"]["embedded_version_matches"]


def test_wrong_document_is_not_verified():
    doc, _ = parse_lcd_page(FIXTURE, URL.replace("99999", "12345"), "t")
    assert doc.verification_status == "unverified"


def test_url_helpers():
    assert lcd_url("L33912", 16).endswith("lcdid=33912&ver=16")
    assert is_mcd_url(URL)
    assert not is_mcd_url("http://www.cms.gov/medicare-coverage-database/view/lcd.aspx")
    assert not is_mcd_url("https://evil.example.com/medicare-coverage-database/view/lcd.aspx")
