"""Send every question x model x retriever x rep to the APIs and log raw responses.

Input:  data/questions.jsonl, config.yaml, .env
Output: data/raw/responses_<YYYYMMDD>.jsonl (append-only, one line per call)
        logs/run_<timestamp>.log

Usage:
    python scripts/run_queries.py --dry-run              # list jobs, call nothing
    python scripts/run_queries.py --limit 5              # pilot
    python scripts/run_queries.py --retriever native --model deepseek
    python scripts/run_queries.py                        # full run, resumes automatically

Resume: every job has a deterministic job_id. Jobs that already have a status=ok line
in data/raw/ are skipped, so rerunning after a crash only runs what's missing or failed.
Failed jobs are logged with status=error and retried on the next run.
Never edit raw files; reruns only append.
"""

import argparse
import importlib
import json
import logging
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# native api name in config -> client module in scripts/clients/
NATIVE_CLIENTS = {
    "openai": "clients.openai_native",
    "anthropic": "clients.anthropic_native",
    "google": "clients.google_native",
    "mistral": "clients.mistral_native",
    "deepseek": "clients.deepseek_native",
    "dashscope": "clients.dashscope_native",
}
COMMON_CLIENT = "clients.openrouter"

MAX_ATTEMPTS = 4
BACKOFF_BASE = 5  # seconds; waits 5, 10, 20

log = logging.getLogger("run_queries")
_write_lock = threading.Lock()


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_questions(path=ROOT / "data" / "questions.jsonl"):
    questions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def job_id(q, model, retriever, rep):
    return f"{q['question_id']}|{q['lang']}|{model['name']}|{retriever}|{rep}"


def build_jobs(questions, config, args):
    run = config["run"]
    langs = set(run["languages"])
    jobs = []
    for q in questions:
        if q["lang"] not in langs:
            continue
        if args.lang and q["lang"] not in args.lang:
            continue
        if args.question and q["question_id"] not in args.question:
            continue
        for model in config["models"]:
            if args.model and model["name"] not in args.model:
                continue
            for retriever in ("common", "native"):
                if args.retriever and retriever != args.retriever:
                    continue
                for rep in range(run["reps"]):
                    jobs.append({
                        "job_id": job_id(q, model, retriever, rep),
                        "question": q,
                        "model": model,
                        "retriever": retriever,
                        "rep": rep,
                    })
    if run.get("shuffle_order", True):
        random.Random(run.get("seed", 0)).shuffle(jobs)
    return jobs


def completed_job_ids(raw_dir):
    done = set()
    for path in sorted(Path(raw_dir).glob("responses_*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # partial line from a crash
                if rec.get("status") == "ok":
                    done.add(rec["job_id"])
    return done


def client_for(job):
    if job["retriever"] == "common":
        name = COMMON_CLIENT
    else:
        name = NATIVE_CLIENTS[job["model"]["native"]["api"]]
    return importlib.import_module(name)  # lazy: unused providers need no SDK


def model_requested(job):
    if job["retriever"] == "common":
        return job["model"]["openrouter_id"]
    return job["model"]["native"]["model"]


def run_job(job, config, run_id):
    q, model = job["question"], job["model"]
    base = {
        "run_id": run_id,
        "job_id": job["job_id"],
        "question_id": q["question_id"],
        "lang": q["lang"],
        "topic_country": q.get("topic_country"),
        "controversy": q.get("controversy"),
        "model": model["name"],
        "origin": model["origin"],
        "retriever": job["retriever"],
        "rep": job["rep"],
        "model_requested": model_requested(job),
        "provider_requested": model.get("openrouter_provider") if job["retriever"] == "common" else model["native"]["api"],
        "question_text": q["text"],
    }
    try:
        client = client_for(job)
    except Exception as e:  # noqa: BLE001 - missing/broken sdk: record, don't crash the run
        return {
            **base,
            "status": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attempts": 0,
            "error": f"client import failed: {type(e).__name__}: {e}",
        }
    error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = datetime.now(timezone.utc)
        t0 = time.monotonic()
        try:
            out = client.query(job, config)
            out.setdefault("retrieved_available", True)
            return {
                **base,
                "status": "ok",
                "timestamp": started.isoformat(),
                "latency_s": round(time.monotonic() - t0, 2),
                "attempts": attempt,
                **out,
            }
        except Exception as e:  # noqa: BLE001 - any api failure is retried then logged
            error = f"{type(e).__name__}: {e}"
            log.warning("%s attempt %d failed: %s", job["job_id"], attempt, error)
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_BASE * 2 ** (attempt - 1))
    return {
        **base,
        "status": "error",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attempts": MAX_ATTEMPTS,
        "error": error,
    }


def write_record(record, out_path):
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def setup_logging():
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
    )
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--retriever", choices=["common", "native"])
    p.add_argument("--model", nargs="+", help="model names from config, e.g. claude deepseek")
    p.add_argument("--lang", nargs="+", help="e.g. en zh")
    p.add_argument("--question", nargs="+", help="question_ids")
    p.add_argument("--limit", type=int, help="run at most N jobs (pilot)")
    p.add_argument("--workers", type=int, default=4, help="parallel calls")
    p.add_argument("--dry-run", action="store_true", help="list jobs without calling apis")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    config = load_config()
    raw_dir = ROOT / config["run"].get("output_dir", "data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(load_questions(), config, args)
    done = completed_job_ids(raw_dir)
    todo = [j for j in jobs if j["job_id"] not in done]
    n_done = len(jobs) - len(todo)
    if args.limit:
        todo = todo[: args.limit]

    if args.dry_run:
        print(f"{len(jobs)} jobs matched, {n_done} already done, {len(todo)} to run")
        for j in todo[:50]:
            print(j["job_id"], "->", model_requested(j))
        if len(todo) > 50:
            print(f"... and {len(todo) - 50} more")
        return

    log_path = setup_logging()
    run_id = uuid.uuid4().hex[:12]
    out_path = raw_dir / f"responses_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    log.info("run %s: %d jobs to run (%d skipped as done) -> %s", run_id, len(todo), n_done, out_path)

    ok = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, j, config, run_id): j for j in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            write_record(rec, out_path)
            if rec["status"] == "ok":
                ok += 1
            else:
                err += 1
                log.error("%s failed: %s", rec["job_id"], rec.get("error"))
            if i % 25 == 0 or i == len(todo):
                log.info("%d/%d done (%d ok, %d error)", i, len(todo), ok, err)

    log.info("finished run %s: %d ok, %d error. log: %s", run_id, ok, err, log_path)


if __name__ == "__main__":
    main()
