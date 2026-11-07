"""Turn raw responses into analysis-ready tables, separating retrieved from cited urls.

Input:  data/raw/responses_*.jsonl
Output: data/processed/calls.csv           one row per successful job
        data/processed/citations.csv       one row per (job, url)
        data/processed/domains_to_label.csv  domains not yet in data/labels/domain_labels.csv,
                                              with how often each was cited / retrieved

If a job_id appears more than once with status=ok (e.g. a manual rerun), the earliest
ok record is kept so results don't depend on rerun order.

Domains: registrable domain via tldextract using its bundled suffix list (no network,
reproducible). Google grounding redirect urls carry no real domain, so for those the
citation title (which Gemini sets to the source domain) is used and domain_source="title".
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
LABELS = ROOT / "data" / "labels" / "domain_labels.csv"

REDIRECT_HOSTS = {"vertexaisearch.cloud.google.com"}

CALL_FIELDS = [
    "job_id", "run_id", "timestamp", "question_id", "lang", "topic_country",
    "controversy", "model", "origin", "retriever", "rep", "model_requested",
    "model_returned", "provider_requested", "provider_returned", "retrieved_available",
    "cited_available",
    "n_searches", "latency_s", "attempts", "stop_reason",
]

_extract = None


def _tld():
    global _extract
    if _extract is None:
        try:
            import tldextract
        except ImportError:
            sys.exit("pip install tldextract")
        _extract = tldextract.TLDExtract(suffix_list_urls=())  # bundled snapshot, offline
    return _extract


def to_domain(url):
    """Registrable domain, e.g. https://www.bbc.co.uk/x -> bbc.co.uk. None if unparsable."""
    try:
        host = urlparse(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    ext = _tld()(host)
    return (ext.top_domain_under_public_suffix if hasattr(ext, "top_domain_under_public_suffix")
            else ext.registered_domain) or host


def is_redirect(url):
    try:
        return urlparse(url).hostname in REDIRECT_HOSTS
    except ValueError:
        return False


def load_raw(raw_dir=RAW_DIR):
    """Earliest status=ok record per job_id, plus a count of error lines."""
    records, n_err, n_bad = {}, 0, 0
    for path in sorted(Path(raw_dir).glob("responses_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    n_bad += 1
                    continue
                if rec.get("status") != "ok":
                    n_err += 1
                    continue
                prev = records.get(rec["job_id"])
                if prev is None or rec["timestamp"] < prev["timestamp"]:
                    records[rec["job_id"]] = rec
    return list(records.values()), n_err, n_bad


def explode_record(rec):
    """One row per unique url in the record, flagging retrieved and cited with ranks."""
    retrieved = rec.get("retrieved_urls") or []
    cited = rec.get("cited_urls") or []
    titles = {r.get("url"): r.get("title") for r in rec.get("references") or [] if r.get("url")}
    r_rank = {u: i + 1 for i, u in enumerate(retrieved)}
    c_rank = {u: i + 1 for i, u in enumerate(cited)}
    rows = []
    for url in list(dict.fromkeys(retrieved + cited)):
        if is_redirect(url):
            title = (titles.get(url) or "").strip().lower()
            domain = to_domain("https://" + title) if title else None
            source = "title"
        else:
            domain, source = to_domain(url), "url"
        rows.append({
            "job_id": rec["job_id"],
            "url": url,
            "domain": domain,
            "domain_source": source,
            "retrieved": url in r_rank,
            "cited": url in c_rank,
            "retrieve_rank": r_rank.get(url),
            "cite_rank": c_rank.get(url),
        })
    return rows


def call_row(rec):
    row = {k: rec.get(k) for k in CALL_FIELDS}
    row["n_retrieved"] = len(rec.get("retrieved_urls") or [])
    row["n_cited"] = len(rec.get("cited_urls") or [])
    row["search_queries"] = json.dumps(rec.get("search_queries") or [], ensure_ascii=False)
    row["text_len"] = len(rec.get("text") or "")
    return row


def unlabeled_domains(cit, labels_path=LABELS):
    labeled = set()
    if Path(labels_path).exists():
        lab = pd.read_csv(labels_path)
        if "domain" in lab:
            labeled = set(lab["domain"].dropna().str.lower())
    d = cit.dropna(subset=["domain"])
    agg = d.groupby("domain").agg(n_cited=("cited", "sum"), n_retrieved=("retrieved", "sum"),
                                  n_jobs=("job_id", "nunique")).reset_index()
    agg = agg[~agg["domain"].str.lower().isin(labeled)]
    return agg.sort_values(["n_cited", "n_retrieved"], ascending=False)


def main():
    records, n_err, n_bad = load_raw()
    if not records:
        sys.exit(f"no ok records in {RAW_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    calls = pd.DataFrame([call_row(r) for r in records])
    cit = pd.DataFrame([row for r in records for row in explode_record(r)])
    if cit.empty:
        cit = pd.DataFrame(columns=["job_id", "url", "domain", "domain_source", "retrieved",
                                    "cited", "retrieve_rank", "cite_rank"])
    todo = unlabeled_domains(cit)

    calls.to_csv(OUT_DIR / "calls.csv", index=False)
    cit.to_csv(OUT_DIR / "citations.csv", index=False)
    todo.to_csv(OUT_DIR / "domains_to_label.csv", index=False)

    print(f"{len(calls)} ok calls ({n_err} error lines, {n_bad} unreadable lines skipped)")
    print(f"{len(cit)} url rows, {int(cit['cited'].sum())} cited, "
          f"{cit['domain'].isna().sum()} without a domain")
    print(f"{len(todo)} domains still to label -> {OUT_DIR / 'domains_to_label.csv'}")
    no_cite = calls[(calls["n_cited"] == 0) & (calls["cited_available"] != False)]  # noqa: E712
    if len(no_cite):
        print(f"warning: {len(no_cite)} calls cited nothing (check refusals / no-search answers)")


if __name__ == "__main__":
    main()
