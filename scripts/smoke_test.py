"""Smoke test: one hard-coded question through every model x retriever, once.

Checks that keys, model ids, provider slugs and response parsers work before the pilot.
Does not touch data/questions.jsonl or data/raw/. Full records (including raw api
responses) go to logs/smoke_<timestamp>.jsonl so parser problems can be debugged.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --model deepseek qwen --retriever native
    python scripts/smoke_test.py --text "..." --lang zh
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_queries as rq  # noqa: E402

# recent, neutral, and not one of the study questions
DEFAULT_TEXT = "What were the main outcomes of the most recent UN climate conference (COP)?"


def host(url):
    try:
        return urlparse(url).hostname or url
    except ValueError:
        return url


def main():
    p = argparse.ArgumentParser(description="one question through every model x retriever")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--lang", default="en")
    p.add_argument("--model", nargs="+")
    p.add_argument("--retriever", choices=["common", "native"])
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(rq.ROOT / ".env")
    except ImportError:
        pass

    config = rq.load_config()
    q = {"question_id": "SMOKE", "lang": args.lang, "text": args.text,
         "topic_country": None, "controversy": None}
    jobs = []
    for model in config["models"]:
        if args.model and model["name"] not in args.model:
            continue
        for retriever in ("common", "native"):
            if args.retriever and retriever != args.retriever:
                continue
            jobs.append({"job_id": rq.job_id(q, model, retriever, 0), "question": q,
                         "model": model, "retriever": retriever, "rep": 0})

    rq.MAX_ATTEMPTS = 1  # fail fast: we want to see errors, not retry them
    print(f"{len(jobs)} calls: {args.text!r}\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda j: rq.run_job(j, config, "smoke"), jobs))

    out = rq.ROOT / "logs" / f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fmt = "{:<9} {:<7} {:<6} {:>4} {:>5} {:>5}  {}"
    print(fmt.format("model", "retr", "status", "srch", "retr", "cited", "first cited / error"))
    print("-" * 90)
    for r in sorted(records, key=lambda r: (r["model"], r["retriever"])):
        if r["status"] == "ok":
            n_ret = len(r.get("retrieved_urls") or []) if r.get("retrieved_available", True) else "n/a"
            cited = r.get("cited_urls") or []
            note = host(cited[0]) if cited else "(no citations)"
            if r.get("model_returned") and r["model_returned"] != r["model_requested"]:
                note += f"  [model_returned={r['model_returned']}]"
            print(fmt.format(r["model"], r["retriever"], "ok", r.get("n_searches") or 0,
                             n_ret, len(cited), note))
        else:
            print(fmt.format(r["model"], r["retriever"], "ERROR", "", "", "", r["error"][:80]))

    n_ok = sum(r["status"] == "ok" for r in records)
    print(f"\n{n_ok}/{len(records)} ok. full records: {out}")
    print("check: srch > 0 everywhere, retr > 0 where available, cited > 0, model_returned matches.")


if __name__ == "__main__":
    main()
