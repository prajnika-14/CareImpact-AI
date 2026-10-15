from fastapi.testclient import TestClient

from careimpact.analysis.baseline import analyze_diff_only
from careimpact.api.app import app
from careimpact.diffing import diff_documents
from careimpact.evaluation.metrics import aggregate, score_example
from careimpact.synthetic.generator import generate_example


def test_gold_scores_perfectly():
    rows = []
    for sc in ["threshold_change", "no_substantive_change", "ambiguous_evidence", "added_documentation"]:
        rec = generate_example(sc, "tms", 7, sc)
        d = diff_documents(rec.old, rec.new)
        rows.append(score_example(rec, rec.target, d, []))
    m = aggregate(rows, [])
    assert m["classification_accuracy"] == 1.0
    assert m["evidence_exact_match_rate"] == 1.0
    assert m["unsupported_claim_rate"] in (0.0, None)
    assert m["finding_recall"] == 1.0


def test_invalid_output_counts_as_missed():
    rec = generate_example("added_documentation", "tms", 3, "x")
    d = diff_documents(rec.old, rec.new)
    m = aggregate([score_example(rec, None, d, ["output is not JSON"])], [])
    assert m["schema_valid_rate"] == 0.0 and m["finding_recall"] == 0.0 and m["classification_accuracy"] == 0.0


def test_baseline_scoring_runs():
    rec = generate_example("diagnosis_codes", "bone_density", 5, "x")
    d = diff_documents(rec.old, rec.new)
    row = score_example(rec, analyze_diff_only(rec.old, rec.new, d), d, [])
    assert row["schema_valid"] and row["quotes"] == row["quotes_matched"]


client = TestClient(app)


def test_api_compare_text_falls_back_to_baseline():
    body = {"old": {"text": "## Documentation Requirements\nKeep the order."},
            "new": {"text": "## Documentation Requirements\nKeep the order. Keep progress notes."},
            "model": "diff-baseline"}
    r = client.post("/api/compare", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["predictor"] == "diff-baseline"
    assert data["analysis"]["change_classification"] == "documentation"
    assert any("not been verified" in w for w in data["warnings"])


def test_api_rejects_non_cms_url():
    r = client.post("/api/compare", json={"old": {"url": "https://example.com/x"}, "new": {"text": "a"}})
    assert r.status_code == 400


def test_api_status_and_pages():
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/dataset").status_code == 200
    assert client.get("/api/library").status_code == 200
    assert client.get("/").status_code == 200


def test_api_health_and_security_headers():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]


def test_unknown_page_returns_friendly_404_but_api_stays_json():
    r = client.get("/no-such-page", headers={"accept": "text/html"})
    assert r.status_code == 404 and "couldn't find that page" in r.text
    r = client.get("/api/no-such-endpoint", headers={"accept": "text/html"})
    assert r.status_code == 404 and r.json()["detail"] == "Not Found"


class _OfflineOllama:
    base_url = "http://offline.test"

    def available(self):
        return False


def test_chat_faq_fallback_answers_and_uses_context():
    from careimpact.api.chat import ChatIn, answer
    res = answer(ChatIn(messages=[{"role": "user", "content": "What does verified mean?"}]), client=_OfflineOllama())
    assert res["source"] == "faq" and "Verified" in res["reply"]
    res = answer(ChatIn(messages=[{"role": "user", "content": "Summarize my current result"}],
                        context="Classification: Documentation"), client=_OfflineOllama())
    assert "Classification: Documentation" in res["reply"]


def test_api_chat_validates_input():
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
    assert client.post("/api/chat", json={"messages": [{"role": "system", "content": "x"}]}).status_code == 422
