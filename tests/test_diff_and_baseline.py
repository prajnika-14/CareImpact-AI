from careimpact.analysis.baseline import analyze_diff_only
from careimpact.analysis.validation import check_grounding, parse_analysis
from careimpact.diffing import diff_documents, parse_sections_from_text, split_units
from careimpact.schemas import PolicyDocument, Section


def doc(sections, version="1"):
    return PolicyDocument(doc_id="T1", title="Test", version=version, data_origin="user",
                          sections=[Section(heading=h, text=t) for h, t in sections])


def test_split_units_are_verbatim_substrings():
    text = "1. First rule applies, e.g. Rule A. Second rule.\n- Third item"
    units = split_units(text)
    assert units == ["First rule applies, e.g. Rule A.", "Second rule.", "Third item"]
    assert all(u in text for u in units)


def test_section_diff_statuses():
    old = doc([("Coverage Indications", "A is covered."), ("Documentation Requirements", "Keep the order."),
               ("References", "Ref 1.")])
    new = doc([("Indications for Coverage", "A is covered. B is covered."), ("Documentation Requirements", "Keep the order."),
               ("Billing Instructions", "Report one unit.")], "2")
    d = diff_documents(old, new)
    by = {s.key: s for s in d.sections}
    assert by["coverage_indications"].status == "modified"
    assert by["documentation_requirements"].status == "unchanged"
    assert by["sources"].status == "removed"
    assert by["billing"].status == "added"
    ins = [o for o in by["coverage_indications"].ops if o.op == "insert"]
    assert ins and ins[0].new == ["B is covered."]


def test_formatting_only_and_rename():
    old = doc([("Limitations", "1. X is not covered.\n2. Y is not covered.")])
    new = doc([("Limitations", "- X is not covered.\n- Y is not covered.")], "2")
    assert diff_documents(old, new).sections[0].status == "formatting_only"
    renamed = doc([("Limitations and Exclusions", "1. X is not covered.\n2. Y is not covered.")], "2")
    assert diff_documents(old, renamed).sections[0].status == "formatting_only"


def test_baseline_quotes_are_grounded():
    old = doc([("Documentation Requirements", "The record must contain the order.")])
    new = doc([("Documentation Requirements", "The record must contain the order. Progress notes are required.")], "2")
    d = diff_documents(old, new)
    a = analyze_diff_only(old, new, d)
    assert a.change_classification.value == "documentation"
    g = check_grounding(a, old, new, d)
    assert g["grounded"] and g["evidence_exact_match_rate"] == 1.0


def test_grounding_flags_fabricated_quote():
    old = doc([("Documentation Requirements", "Old text.")])
    new = doc([("Documentation Requirements", "New text.")], "2")
    d = diff_documents(old, new)
    a = analyze_diff_only(old, new, d)
    a.findings[0].evidence_new = "A quote that does not exist."
    g = check_grounding(a, old, new, d)
    assert not g["grounded"]
    assert any(i["type"] == "unsupported_quote" for i in g["issues"])


def test_parse_analysis_handles_bad_output():
    assert parse_analysis("not json")[0] is None
    a, errs = parse_analysis('Sure! {"document_id": "x", "change_classification": "coverage", "is_substantive": true, '
                             '"change_summary": "s", "changed_sections": [], "findings": [], "abstain": false, "limitations": []}')
    assert a is not None and errs == ["JSON had surrounding text"]
    a, errs = parse_analysis('{"document_id": "x", "change_classification": "made_up"}')
    assert a is None and errs


def test_parse_sections_from_text():
    secs = parse_sections_from_text("## Coverage Indications\nA is covered.\n\nDocumentation Requirements\nKeep records.")
    assert [s.heading for s in secs] == ["Coverage Indications", "Documentation Requirements"]
