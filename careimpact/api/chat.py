"""Help assistant for the dashboard.

Answers questions about CareImpact AI and, when the user has a comparison on
screen, about that result. Uses the configured Ollama model when it is
installed; otherwise answers from a small built-in knowledge base so the
assistant still works offline.
"""
from __future__ import annotations

import re
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

from careimpact.config import settings
from careimpact.inference.ollama_client import OllamaClient, OllamaError

MAX_TURNS = 10
MODEL_TIMEOUT_S = 90  # chat must stay interactive; fall back to the guide if the model is slower
AVAILABILITY_TTL_S = 30  # probing an offline Ollama takes seconds, so don't do it on every message
_availability: dict[str, tuple[float, bool]] = {}

# (topic, keywords, answer). Keywords are matched against the question's words.
KNOWLEDGE: list[tuple[str, set[str], str]] = [
    ("overview", {"careimpact", "project", "about", "purpose", "app", "tool", "website", "overview"},
     "CareImpact AI compares two versions of a Medicare coverage policy (LCD). It finds the sections that changed "
     "with a deterministic diff, then a small language model classifies each change, quotes the evidence and drafts "
     "the possible operational impact. Every quote is checked against the source text. It is an operational aid, "
     "not official CMS guidance or legal advice."),
    ("compare", {"compare", "comparison", "start", "how", "use", "run", "begin", "analyse", "analyze", "versions"},
     "To compare two versions: open **Compare Policies**, choose the old and new version (a saved document, an "
     "official MCD URL, or pasted text), pick an analyzer and press **Compare Policies**. The quickest start is the "
     "**Policy Library**: press **Compare** on any row and both sides are filled in for you."),
    ("library", {"library", "pair", "pairs", "sample", "samples", "demo", "saved", "documents"},
     "The **Policy Library** lists version pairs. *Synthetic samples* are fictional demo policies (IDs start with "
     "SYN-). *Real CMS pairs* were downloaded from the official Medicare Coverage Database and validated. Use the "
     "filter to switch between them and press **Compare** to open a pair."),
    ("url", {"url", "link", "mcd", "cms", "fetch", "download", "website", "gov"},
     "In the **MCD URL** tab you can paste a version-specific Medicare Coverage Database link, e.g. "
     "`https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=33912&ver=26`. Only official cms.gov MCD "
     "links are accepted. The page is fetched and validated when you press Compare."),
    ("paste", {"paste", "text", "own", "upload", "custom", "heading", "headings"},
     "In the **Paste text** tab you can paste any policy text. Put each heading on its own line (or start it with "
     "`## `) so the text is split into sections. Pasted text is marked *unverified* because it did not come from an "
     "official source."),
    ("status", {"verified", "unverified", "unavailable", "status", "badge", "validated", "validation"},
     "**Verified** means the CMS page's document ID, version and title matched the request and real policy content "
     "was extracted. **Unverified** means one of those checks failed, so it is never shown as a confirmed change. "
     "**Unavailable** means the old version could not be retrieved. **Synthetic** marks fictional demo data."),
    ("synthetic", {"synthetic", "fictional", "fake", "generated", "syn"},
     "Synthetic policies are fictional examples generated from 12 made-up policy families and 20 change scenarios. "
     "They have SYN- IDs and '(Synthetic)' titles, and their labels come from the scenario definition, not from an "
     "AI. They are used for training and for demos, and never describe a real LCD."),
    ("classification", {"classification", "category", "categories", "coverage", "clinical", "criteria", "documentation",
                        "coding", "billing", "provider", "administrative", "retirement", "substantive", "type"},
     "Each change gets one category: **Coverage**, **Clinical criteria**, **Documentation**, **Coding & billing**, "
     "**Provider requirements**, **Retirement / replacement**, **Administrative**, **No substantive change**, or "
     "**Insufficient evidence** (the analyzer abstains when the text does not support a conclusion). Administrative "
     "and no-change findings are not counted as substantive."),
    ("evidence", {"evidence", "quote", "quotes", "found", "source", "grounding", "unsupported", "hallucination"},
     "Every finding quotes the old and new text. Each quote is exact-match checked (ignoring whitespace) against the "
     "source document. **✓ found in source** means it matched; **⚠ not found in source** means the model may have "
     "paraphrased or invented it, so verify that finding by hand."),
    ("confidence", {"confidence", "high", "medium", "low", "probability", "sure", "accurate", "trust", "reliable"},
     "Confidence labels (high / medium / low) are qualitative and are not calibrated probabilities. Always confirm a "
     "finding in the side-by-side diff and the official policy before acting on it."),
    ("diff", {"diff", "side", "sentence", "sentences", "added", "removed", "modified", "red", "green", "colors"},
     "The **side-by-side diff** shows each section's old and new text. Lines marked **−** were removed, **+** added "
     "and **~** modified. Unchanged runs are collapsed. Sections with changes open automatically; use *Expand all* "
     "to see everything."),
    ("baseline", {"baseline", "deterministic", "fallback", "without", "offline", "no"},
     "The **deterministic diff baseline** needs no AI model: it classifies changes with rules over the section diff. "
     "The app falls back to it automatically when Ollama is offline or the model returns invalid output, and it "
     "tells you when that happened."),
    ("ollama", {"ollama", "model", "models", "llm", "ai", "qwen", "install", "offline", "connected", "pull"},
     "Model analysis runs locally through Ollama. Install it from https://ollama.com/download, run "
     "`ollama pull qwen2.5:1.5b`, and restart the app. The status chips at the top show whether Ollama is connected "
     "and which model is used. CPU inference can take a minute per comparison."),
    ("training", {"train", "training", "fine", "tune", "finetune", "lora", "dataset", "data"},
     "The model is fine-tuned with LoRA on synthetic examples plus real CMS examples that a person approved. The "
     "**Dataset & Training** page shows the measured counts, scenario balance and training runs. Real drafts never "
     "enter training without human review."),
    ("benchmarks", {"benchmark", "benchmarks", "evaluation", "metrics", "score", "accuracy", "f1", "recall", "results"},
     "**Model Benchmarking** compares the diff baseline, the base model and the fine-tuned model on held-out "
     "synthetic policy families and on human-approved real pairs, reported separately. Synthetic scores are not "
     "evidence of real-world accuracy."),
    ("safety", {"legal", "advice", "official", "guidance", "compliance", "patient", "privacy", "phi", "safe", "license", "cpt"},
     "CareImpact AI is not official CMS guidance or legal/compliance advice. No patient data is used or stored. CPT "
     "codes are subject to AMA license terms, which the app never accepts on your behalf. Always verify findings "
     "against the official policy."),
]

SUGGESTIONS = ["How do I compare two policies?", "What does 'verified' mean?",
               "What do the change categories mean?", "How do I connect Ollama?"]

SYSTEM_PROMPT = (
    "You are the help assistant inside CareImpact AI, a tool that compares versions of Medicare coverage policies "
    "(LCDs). Answer briefly (under 120 words) in plain language using Markdown bold for UI names. Only use the facts "
    "below and the comparison context if given. If you do not know, say so and suggest where in the app to look. "
    "Never present output as official CMS guidance or legal advice, and never invent policy text.\n\nFACTS:\n"
    + "\n".join(f"- {a}" for _, _, a in KNOWLEDGE)
)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    messages: list[ChatTurn] = Field(min_length=1, max_length=40)
    context: Optional[str] = Field(default=None, max_length=6000)


STOPWORDS = {"a", "an", "the", "is", "are", "do", "does", "did", "what", "why", "how", "i", "can", "to", "of", "in",
             "it", "me", "on", "for", "and", "or", "you", "be", "with", "about", "please", "tell"}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


RESULT_WORDS = {"result", "results", "summary", "summarize", "summarise", "changed", "change", "changes", "finding",
                "findings", "this", "my", "explain", "mean", "impact"}


def faq_answer(question: str, context: Optional[str] = None) -> str:
    words = _words(question)
    if not words:
        return "Ask me anything about using CareImpact AI."
    if words & {"hi", "hello", "hey", "thanks", "thank"} and len(words) <= 3:
        return "Hi! I can explain how to compare policies, what the badges and categories mean, or summarise your current result."
    if context and len(words & RESULT_WORDS) >= 2:
        return "Here is the comparison currently on screen:\n\n" + context[:1500] + \
               "\n\nCheck each finding against the side-by-side diff before acting on it."
    words = (words - STOPWORDS) or words
    if words <= {"careimpact", "ai"} or words & {"overview", "purpose"}:
        return KNOWLEDGE[0][2]
    scored = sorted(((len(words & kw), topic, ans) for topic, kw, ans in KNOWLEDGE), reverse=True)
    best, second = scored[0], scored[1]
    if best[0] == 0:
        return ("I'm not sure about that one. I can help with comparing policies, the Policy Library, badges and "
                "categories, evidence checks, Ollama setup, training and benchmarks.")
    if second[0] == best[0] and second[0] >= 2:
        return f"{best[2]}\n\n{second[2]}"
    return best[2]


def _model_ready(client: OllamaClient, model: str) -> bool:
    key = f"{client.base_url}|{model}"
    cached = _availability.get(key)
    if cached and time.monotonic() - cached[0] < AVAILABILITY_TTL_S:
        return cached[1]
    ready = client.available() and model in client.models()
    _availability[key] = (time.monotonic(), ready)
    return ready


def answer(body: ChatIn, client: Optional[OllamaClient] = None) -> dict:
    turns = body.messages[-MAX_TURNS:]
    question = turns[-1].content
    model = settings()["ollama_model"]
    client = client or OllamaClient(timeout=MODEL_TIMEOUT_S)
    if _model_ready(client, model):
        system = SYSTEM_PROMPT + (f"\n\nCURRENT COMPARISON ON SCREEN:\n{body.context}" if body.context else "")
        messages = [{"role": "system", "content": system}] + [t.model_dump() for t in turns]
        try:
            res = client.chat(model, messages, json_schema=False)
            if res["content"].strip():
                return {"reply": res["content"].strip(), "source": "model", "model": model, "suggestions": []}
        except OllamaError:
            _availability.clear()
    return {"reply": faq_answer(question, body.context), "source": "faq", "model": None, "suggestions": SUGGESTIONS}
