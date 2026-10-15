"""FastAPI application: JSON API under /api and the dashboard at /."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from careimpact import __version__
from careimpact.api.chat import ChatIn, answer as chat_answer
from careimpact.api.stats import dataset_stats, training_runs
from careimpact.config import Paths, settings
from careimpact.diffing import parse_sections_from_text
from careimpact.evaluation.runner import latest_results
from careimpact.inference.ollama_client import OllamaClient
from careimpact.inference.service import BASELINE, compare
from careimpact.schemas import ExampleRecord, PolicyDocument

app = FastAPI(title="CareImpact AI", version=__version__)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
                               "connect-src 'self'; frame-ancestors 'none'",
}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    return response


@app.exception_handler(StarletteHTTPException)
async def not_found(request: Request, exc: StarletteHTTPException):
    # Browsers get the friendly page; API clients keep the JSON error.
    if exc.status_code == 404 and not request.url.path.startswith("/api") and "text/html" in request.headers.get("accept", ""):
        return FileResponse(Paths.frontend / "404.html", status_code=404)
    return await http_exception_handler(request, exc)


# ---------------------------------------------------------------- library
def _samples() -> dict[str, ExampleRecord]:
    out = {}
    for p in sorted(Paths.samples.glob("demo_*.json")) if Paths.samples.exists() else []:
        out[p.stem] = ExampleRecord.model_validate_json(p.read_text(encoding="utf-8"))
    return out


def _real_docs() -> dict[str, PolicyDocument]:
    out = {}
    for p in sorted(Paths.documents.glob("L*_v*[0-9].json")) if Paths.documents.exists() else []:
        d = PolicyDocument.model_validate_json(p.read_text(encoding="utf-8"))
        out[f"real:{d.doc_id}:{d.version}"] = d
    return out


def _resolve(key: str) -> PolicyDocument:
    if key.startswith("sample:"):
        _, name, side = key.split(":")
        rec = _samples().get(name)
        if rec is None or side not in {"old", "new"}:
            raise HTTPException(404, f"Unknown sample {key}")
        return getattr(rec, side)
    doc = _real_docs().get(key)
    if doc is None:
        raise HTTPException(404, f"Unknown document {key}")
    return doc


def _summary(doc: PolicyDocument, key: str) -> dict:
    return {"key": key, "doc_id": doc.doc_id, "title": doc.title, "version": doc.version, "status": doc.status,
            "revision_effective_date": doc.revision_effective_date, "data_origin": doc.data_origin,
            "verification_status": doc.verification_status, "source_url": doc.source_url,
            "retrieved_at": doc.retrieved_at, "sections": len(doc.sections), "contractor": doc.contractor}


@app.get("/api/library")
def library():
    pairs = []
    for name, rec in _samples().items():
        pairs.append({"pair_id": name, "kind": "sample", "title": rec.new.title, "doc_id": rec.new.doc_id,
                      "scenario": rec.scenario, "old_key": f"sample:{name}:old", "new_key": f"sample:{name}:new",
                      "old_version": rec.old.version, "new_version": rec.new.version, "status": "synthetic"})
    docs = _real_docs()
    if Paths.pairs.exists():
        for line in Paths.pairs.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            ok, nk = f"real:{p['doc_id']}:{p.get('old_version')}", f"real:{p['doc_id']}:{p['new_version']}"
            pairs.append({"pair_id": p["pair_id"], "kind": "real", "title": p["title"], "doc_id": p["doc_id"],
                          "old_key": ok if ok in docs else None, "new_key": nk if nk in docs else None,
                          "old_version": p.get("old_version"), "new_version": p["new_version"], "status": p["status"],
                          "reasons": p.get("reasons", []), "old_url": p.get("old_url"), "new_url": p.get("new_url"),
                          "contractor": p.get("contractor"), "identical_content": p.get("identical_content", False)})
    documents = [_summary(rec.old, f"sample:{n}:old") for n, rec in _samples().items()] + \
                [_summary(rec.new, f"sample:{n}:new") for n, rec in _samples().items()] + \
                [_summary(d, k) for k, d in docs.items()]
    return {"pairs": pairs, "documents": documents}


@app.get("/api/documents/{key}")
def document(key: str):
    return _resolve(key).model_dump(mode="json")


# ---------------------------------------------------------------- compare
class SourceIn(BaseModel):
    document_key: Optional[str] = None
    text: Optional[str] = Field(default=None, max_length=400_000)
    url: Optional[str] = None
    title: Optional[str] = None
    version: Optional[str] = None


class CompareIn(BaseModel):
    old: SourceIn
    new: SourceIn
    model: Optional[str] = None
    use_model: bool = True


def _load_source(src: SourceIn, side: str) -> PolicyDocument:
    if src.document_key:
        return _resolve(src.document_key)
    if src.url:
        from careimpact.ingestion.http import PoliteClient
        from careimpact.ingestion.mcd_pages import is_mcd_url, parse_lcd_page
        if not is_mcd_url(src.url):
            raise HTTPException(400, "Only official Medicare Coverage Database URLs (https://www.cms.gov/medicare-coverage-database/view/...) are accepted.")
        res = PoliteClient().get(src.url)
        if not res.ok:
            raise HTTPException(502, f"Could not retrieve the {side} version from CMS ({res.error}). The historical version may be unavailable.")
        doc, extras = parse_lcd_page(res.text, src.url, res.fetched_at)
        if not doc.sections:
            raise HTTPException(422, f"The {side} page was retrieved but no policy sections could be extracted.")
        return doc
    if src.text and src.text.strip():
        sections = parse_sections_from_text(src.text)
        return PolicyDocument(doc_id="user-input", title=src.title or "User-supplied policy text", version=src.version or side,
                              data_origin="user", verification_status="unverified",
                              verification_notes=["Pasted text; not retrieved from an official source."], sections=sections)
    raise HTTPException(400, f"Provide a saved document, an MCD URL or text for the {side} version.")


@app.post("/api/compare")
def compare_route(body: CompareIn):
    old, new = _load_source(body.old, "old"), _load_source(body.new, "new")
    if old.doc_id != new.doc_id and "user-input" not in (old.doc_id, new.doc_id):
        warn = f"The two versions have different document IDs ({old.doc_id} vs {new.doc_id})."
    else:
        warn = None
    result = compare(old, new, model=body.model, use_model=body.use_model)
    if warn:
        result["warnings"].insert(0, warn)
    return result


# ------------------------------------------------------------------- chat
@app.post("/api/chat")
def chat(body: ChatIn):
    return chat_answer(body)


# ------------------------------------------------------- status / dashboards
@app.get("/api/health")
def health():
    """Liveness probe: no network calls, safe to poll."""
    return {"status": "ok", "app_version": __version__}


@app.get("/api/status")
def status():
    c = OllamaClient()
    reachable = c.available()
    return {"app_version": __version__, "configured_model": settings()["ollama_model"], "baseline": BASELINE,
            "ollama_url": c.base_url, "ollama_reachable": reachable, "ollama_models": c.models() if reachable else [],
            "dataset": dataset_stats()}


@app.get("/api/dataset")
def dataset():
    return dataset_stats()


@app.get("/api/benchmarks")
def benchmarks():
    return {"results": latest_results()}


@app.get("/api/benchmarks/examples")
def benchmark_examples(eval_set: str = "synthetic_test", model: str = BASELINE, kind: str = "failures", limit: int = 8):
    files = sorted(Paths.eval_results.glob(f"{eval_set}__*.json")) if Paths.eval_results.exists() else []
    for p in reversed(files):
        data = json.loads(p.read_text(encoding="utf-8"))
        if data["model"] != model:
            continue
        rows = [r for r in data["rows"] if (not r["class_correct"]) == (kind == "failures")]
        return {"model": model, "eval_set": eval_set, "file": p.name,
                "rows": [{k: r.get(k) for k in ("example_id", "scenario", "gold_class", "pred_class", "parse_errors",
                                                "unsupported_findings", "pred_abstain", "gold_abstain")} for r in rows[:limit]]}
    return {"model": model, "eval_set": eval_set, "rows": []}


@app.get("/api/training")
def training():
    return {"runs": training_runs()}


# ------------------------------------------------------------------ static
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Paths.frontend / "index.html")


app.mount("/static", StaticFiles(directory=Paths.frontend), name="static")
