"""CareImpact AI command line.

Usage (PowerShell, from the project root):
    python -m careimpact.cli synth --split train --n 100 --seed 7
    python -m careimpact.cli build-dataset
    python -m careimpact.cli evaluate --model diff-baseline
    python -m careimpact.cli collect --limit 10
    python -m careimpact.cli serve
Run ``python -m careimpact.cli <command> -h`` for options.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_synth(a):
    from careimpact.synthetic.runner import generate_split
    _print(generate_split(a.split, a.n, a.seed))


def cmd_synth_all(a):
    from careimpact.synthetic.runner import generate_split
    out = {}
    for split, n, seed in (("train", a.train, a.seed), ("val", a.val, a.seed + 1), ("test", a.test, a.seed + 2)):
        r = generate_split(split, n, seed)
        out[split] = {k: r[k] for k in ("accepted", "failed_checks", "duplicates_exact", "duplicates_near", "output")}
    _print(out)


def cmd_samples(a):
    from careimpact.config import Paths
    from careimpact.synthetic.generator import generate_example
    Paths.samples.mkdir(parents=True, exist_ok=True)
    demos = [("threshold_change", "tms", 101), ("added_documentation", "bone_density", 202),
             ("billing_article_reference", "hyperbaric_oxygen", 303), ("no_substantive_change", "tms", 404),
             ("retirement_replacement", "bone_density", 505), ("ambiguous_evidence", "hyperbaric_oxygen", 606)]
    for scenario, fam, seed in demos:
        rec = generate_example(scenario, fam, seed, f"demo-{scenario}")
        (Paths.samples / f"demo_{scenario}.json").write_text(rec.model_dump_json(indent=1), encoding="utf-8")
    _print({"samples": len(demos), "dir": str(Paths.samples)})


def cmd_build(a):
    from careimpact.dataset.builder import build
    _print(build())


def cmd_evaluate(a):
    from careimpact.evaluation.runner import run_eval, write_report
    res = run_eval(a.set, a.model, a.limit)
    m = res["metrics"]
    _print({k: v for k, v in m.items() if not isinstance(v, dict)} | {"output": res["output"]})
    print("report:", write_report())


def cmd_report(a):
    from careimpact.evaluation.runner import write_report
    print(write_report())


def cmd_collect(a):
    from careimpact.ingestion.pipeline import collect
    ids = [int(x.strip().lstrip("Ll")) for x in a.ids.split(",")] if a.ids else None
    r = collect(limit=a.limit, lcd_ids=ids, refresh=a.refresh)
    _print({k: v for k, v in r.items() if k not in {"unresolved_pairs"}} | {"unresolved_pairs": len(r["unresolved_pairs"])})


def cmd_draft_real(a):
    from careimpact.ingestion.real_examples import draft_candidates
    _print(draft_candidates(a.model))


def cmd_review(a):
    from careimpact.ingestion.real_examples import review_interactive
    _print(review_interactive())


def cmd_status(a):
    from careimpact.api.stats import dataset_stats
    from careimpact.inference.ollama_client import OllamaClient
    c = OllamaClient()
    _print({"ollama_reachable": c.available(), "ollama_models": c.models(), "dataset": dataset_stats()})


def cmd_serve(a):
    import uvicorn
    uvicorn.run("careimpact.api.app:app", host=a.host, port=a.port, reload=False)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="careimpact")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate one synthetic split")
    s.add_argument("--split", choices=["train", "val", "test"], default="train")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("synth-all", help="generate train/val/test synthetic splits")
    s.add_argument("--train", type=int, default=800)
    s.add_argument("--val", type=int, default=100)
    s.add_argument("--test", type=int, default=200)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_synth_all)

    sub.add_parser("samples", help="write clearly labelled demo samples for the UI").set_defaults(fn=cmd_samples)
    sub.add_parser("build-dataset", help="combine approved examples into SFT files").set_defaults(fn=cmd_build)

    s = sub.add_parser("evaluate", help="benchmark a predictor on a held-out set")
    s.add_argument("--model", default="diff-baseline", help="'diff-baseline' or an Ollama model tag, e.g. qwen2.5:1.5b")
    s.add_argument("--set", default="synthetic_test", help="synthetic_test | real_test | path to JSONL")
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(fn=cmd_evaluate)

    sub.add_parser("report", help="rebuild the markdown benchmark report").set_defaults(fn=cmd_report)

    s = sub.add_parser("collect", help="collect LCD version pairs from official CMS sources")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--ids", default=None, help="comma-separated LCD IDs, e.g. 33912,38672")
    s.add_argument("--refresh", action="store_true", help="ignore the HTTP cache and stored documents")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("draft-real", help="draft review candidates from verified real pairs")
    s.add_argument("--model", default="diff-baseline")
    s.set_defaults(fn=cmd_draft_real)

    sub.add_parser("review", help="interactively approve/reject real candidates").set_defaults(fn=cmd_review)
    sub.add_parser("status", help="show model and dataset status").set_defaults(fn=cmd_status)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
