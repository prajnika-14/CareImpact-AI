"""Comparison service used by the API and the evaluator."""
from __future__ import annotations

from typing import Optional

from careimpact.analysis.baseline import analyze_diff_only
from careimpact.analysis.prompts import chat_messages
from careimpact.analysis.validation import check_grounding, parse_analysis
from careimpact.config import settings
from careimpact.diffing import diff_documents
from careimpact.inference.ollama_client import OllamaClient, OllamaError
from careimpact.schemas import DocumentDiff, PolicyAnalysis, PolicyDocument

BASELINE = "diff-baseline"


def predict(old: PolicyDocument, new: PolicyDocument, model: str, diff: Optional[DocumentDiff] = None,
            client: Optional[OllamaClient] = None) -> dict:
    """Run one predictor. Returns analysis (or None), raw output, errors and timing."""
    diff = diff or diff_documents(old, new)
    if model == BASELINE:
        return {"analysis": analyze_diff_only(old, new, diff), "raw": None, "errors": [], "latency_s": 0.0,
                "predictor": BASELINE}
    client = client or OllamaClient()
    res = client.chat(model, chat_messages(old, new, diff))
    analysis, errors = parse_analysis(res["content"])
    return {"analysis": analysis, "raw": res["content"], "errors": errors, "latency_s": res["latency_s"],
            "prompt_tokens": res["prompt_tokens"], "output_tokens": res["output_tokens"], "predictor": model}


def compare(old: PolicyDocument, new: PolicyDocument, model: Optional[str] = None, use_model: bool = True) -> dict:
    """Full comparison for the UI: diff + analysis + grounding report + warnings.

    Falls back to the deterministic baseline when the model is unreachable or
    returns invalid output, and says so explicitly.
    """
    diff = diff_documents(old, new)
    model = model or settings()["ollama_model"]
    warnings: list[str] = []
    result = None
    if use_model and model != BASELINE:
        try:
            result = predict(old, new, model, diff)
            if result["analysis"] is None:
                warnings.append("The model returned output that did not match the schema; showing the deterministic diff baseline instead.")
                result = None
        except OllamaError as exc:
            warnings.append(f"Model unavailable ({exc}); showing the deterministic diff baseline instead.")
    if result is None:
        result = predict(old, new, BASELINE, diff)

    analysis: PolicyAnalysis = result["analysis"]
    grounding = check_grounding(analysis, old, new, diff)
    if grounding["issues"]:
        bad = [i for i in grounding["issues"] if i["type"] == "unsupported_quote"]
        if bad:
            warnings.append(f"{len(bad)} evidence quote(s) could not be found in the source text and are flagged as unsupported.")
    for doc, label in ((old, "Old"), (new, "New")):
        if doc.data_origin == "synthetic":
            warnings.append(f"{label} version is a fictional synthetic policy, not CMS guidance.")
        elif doc.verification_status != "verified":
            warnings.append(f"{label} version has not been verified against an official CMS source ({doc.verification_status}).")
    return {
        "predictor": result["predictor"],
        "latency_s": result["latency_s"],
        "model_errors": result.get("errors", []),
        "analysis": analysis.model_dump(mode="json"),
        "grounding": grounding,
        "diff": diff.model_dump(mode="json"),
        "diff_counts": diff.counts(),
        "warnings": warnings,
        "documents": {"old": old.model_dump(mode="json", exclude={"sections"}),
                      "new": new.model_dump(mode="json", exclude={"sections"})},
        "disclaimer": "Generated analysis is an operational aid, not official CMS guidance or legal/compliance advice. Verify against the official policy.",
    }
