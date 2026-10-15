import pytest

from careimpact.diffing import diff_documents, quote_in_source
from careimpact.schemas import SUBSTANTIVE_CLASSES, ChangeClass
from careimpact.synthetic.generator import SCENARIOS, generate_example
from careimpact.synthetic.topics import SPLIT_FAMILIES, TOPICS

EXPECTED_CLASS = {
    "new_coverage_indication": {"coverage"}, "removed_coverage_indication": {"coverage"},
    "expanded_eligibility": {"clinical_criteria"}, "narrowed_eligibility": {"clinical_criteria"},
    "threshold_change": {"clinical_criteria"}, "treatment_duration": {"coverage"},
    "added_documentation": {"documentation"}, "removed_documentation": {"documentation"},
    "diagnosis_codes": {"coding_billing"}, "billing_instructions": {"coding_billing"},
    "billing_article_reference": {"coding_billing"}, "provider_qualifications": {"provider_requirements"},
    "supervision_requirements": {"provider_requirements"}, "new_limitation": {"coverage"},
    "effective_date_revision": {"administrative"}, "retirement_replacement": {"retirement_replacement"},
    "administrative_edit": {"administrative"}, "no_substantive_change": {"no_substantive_change"},
    "ambiguous_evidence": {"insufficient_evidence", "documentation"},
    "evidence_requirements": {"clinical_criteria", "administrative"},
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenario_labels_are_consistent(scenario):
    for i, topic in enumerate(TOPICS):
        rec = generate_example(scenario, topic.key, 1000 + i, "t")
        assert rec.validation_status == "auto_validated", rec.validation_errors
        t = rec.target
        # The scenario's own finding is first among substantive ones unless a compound
        # edit with higher priority was added; check the scenario class is present.
        classes = {f.category.value for f in t.findings} | {t.change_classification.value}
        assert classes & EXPECTED_CLASS[scenario], (scenario, classes)
        assert t.is_substantive == (t.change_classification in SUBSTANTIVE_CLASSES)
        old_text = "\n".join(s.text for s in rec.old.sections)
        new_text = "\n".join(s.text for s in rec.new.sections)
        for f in t.findings:
            assert f.evidence_old is None or quote_in_source(f.evidence_old, old_text)
            assert f.evidence_new is None or quote_in_source(f.evidence_new, new_text)
        assert set(t.changed_sections) == {s.heading for s in diff_documents(rec.old, rec.new).changed}
        assert rec.old.doc_id.startswith("SYN-") and "(Synthetic)" in rec.new.title
        if scenario == "no_substantive_change":
            assert t.findings == [] and not t.is_substantive
        if t.change_classification == ChangeClass.INSUFFICIENT_EVIDENCE:
            assert t.abstain


def test_generation_is_deterministic():
    a = generate_example("threshold_change", "pulmonary_rehab", 42, "a")
    b = generate_example("threshold_change", "pulmonary_rehab", 42, "a")
    assert a.target == b.target and a.old.sections == b.old.sections


def test_split_families_disjoint():
    s = {k: set(v) for k, v in SPLIT_FAMILIES.items()}
    assert not (s["train"] & s["test"]) and not (s["train"] & s["val"]) and not (s["val"] & s["test"])
    assert s["train"] | s["val"] | s["test"] == {t.key for t in TOPICS}
