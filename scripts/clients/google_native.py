"""Native retriever for Gemini: Interactions API + the google_search tool.

Settings come from config.yaml -> models[name=gemini].native:
    model  gemini-3.8-flash
    tool   google_search

Response steps: thought -> google_search_call (arguments.queries) ->
google_search_result (only a search-suggestions html snippet) -> model_output
(text with url_citation annotations).

Limitations:
- No list of retrieved results: retrieved_urls is empty, retrieved_available=False.
- Citation urls come back as Google grounding redirect links
  (vertexaisearch.cloud.google.com/grounding-api-redirect/...) that expire. At query time
  each one is resolved to its target (reading the Location header, not fetching the page);
  cited_urls holds the resolved urls and references keeps redirect_url / resolved_url.
  If resolution fails the redirect stays, and extract_citations falls back to the
  annotation title as the domain.
"""

import os
from urllib.parse import urlparse

import requests
from google import genai

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _client


def _dedupe(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def _to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"can't serialize {type(obj)}")


def _build_request(job, config):
    native = job["model"]["native"]
    gen = config.get("generation", {})
    req = {
        "model": native["model"],
        "input": job["question"]["text"],
        "tools": [{"type": native.get("tool", "google_search")}],
    }
    if gen.get("temperature") is not None:
        req["generation_config"] = {"temperature": gen["temperature"]}
    if gen.get("system_prompt"):
        req["system_instruction"] = gen["system_prompt"]
    return req


def parse(resp):
    """Normalize one interaction dict."""
    steps = resp.get("steps") or resp.get("outputs") or []
    text_parts, queries, cited, references = [], [], [], []
    n_searches = 0
    for step in steps:
        stype = step.get("type")
        if stype == "google_search_call":
            n_searches += 1
            queries += (step.get("arguments") or {}).get("queries") or []
        elif stype == "model_output":
            for c in step.get("content") or []:
                if c.get("type") != "text":
                    continue
                text_parts.append(c.get("text", ""))
                for a in c.get("annotations") or []:
                    if a.get("type") == "url_citation" and a.get("url"):
                        cited.append(a["url"])
                        references.append({"url": a["url"], "title": a.get("title")})
    return {
        "text": "".join(text_parts),
        "search_queries": _dedupe(queries),
        "retrieved_urls": [],
        "retrieved_available": False,
        "cited_urls": _dedupe(cited),
        "references": references,
        "n_searches": n_searches,
        "usage": resp.get("usage"),
        "model_returned": resp.get("model"),
        "stop_reason": resp.get("status"),
        "raw": resp,
    }


REDIRECT_HOST = "vertexaisearch.cloud.google.com"


def resolve_redirect(url, timeout=10):
    """Read where a grounding redirect points without following it or downloading the page.
    Redirect links expire, so this runs at query time. Returns None on failure."""
    try:
        r = requests.head(url, allow_redirects=False, timeout=timeout)
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            return r.headers["Location"]
        r = requests.get(url, allow_redirects=False, timeout=timeout, stream=True)
        r.close()
        return r.headers.get("Location")
    except requests.RequestException:
        return None


def resolve_record(rec):
    """Replace redirect links in cited_urls with their targets; keep originals in references."""
    resolved = {}
    for ref in rec.get("references", []):
        url = ref.get("url") or ""
        if urlparse(url).hostname == REDIRECT_HOST:
            if url not in resolved:
                resolved[url] = resolve_redirect(url)
            ref["redirect_url"] = url
            ref["resolved_url"] = resolved[url]
    rec["cited_urls"] = _dedupe([resolved.get(u) or u for u in rec["cited_urls"]])
    rec["n_unresolved_redirects"] = sum(1 for v in resolved.values() if not v)
    return rec


def query(job, config):
    """Run one question through Gemini with google_search. Raises on API errors."""
    resp = _get_client().interactions.create(**_build_request(job, config))
    return resolve_record(parse(_to_dict(resp)))
