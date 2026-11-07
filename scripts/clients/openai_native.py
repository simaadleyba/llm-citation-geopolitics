"""Native retriever for GPT: Responses API + OpenAI's hosted web_search tool.

Settings come from config.yaml -> models[name=gpt].native:
    model        e.g. gpt-6.1-sol
    tool         web_search (not web_search_preview)
    tool_choice  auto (default; same as other providers, which can't force search)
    reasoning    optional effort override; omit to use the model default

Request includes web_search_call.action.sources so we get every url the model
consulted, not only the cited ones. No filters, no user_location; search_context_size
and return_token_budget left at defaults.
"""

import os

from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _dedupe(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def _build_request(job, config):
    native = job["model"]["native"]
    gen = config.get("generation", {})
    req = {
        "model": native["model"],
        "input": job["question"]["text"],
        "tools": [{"type": native.get("tool", "web_search")}],
        "tool_choice": native.get("tool_choice", "auto"),
        "include": ["web_search_call.action.sources"],
    }
    if native.get("reasoning"):
        req["reasoning"] = {"effort": native["reasoning"]}
    if gen.get("temperature") is not None:
        req["temperature"] = gen["temperature"]
    if gen.get("system_prompt"):
        req["instructions"] = gen["system_prompt"]
    return req


def parse(resp):
    """Normalize one Responses API response dict."""
    text_parts, queries, retrieved, opened, cited, feeds = [], [], [], [], [], []
    n_searches = 0
    for item in resp.get("output", []):
        itype = item.get("type")
        if itype == "web_search_call":
            action = item.get("action") or {}
            atype = action.get("type")
            if atype == "search":
                n_searches += 1
                if action.get("query"):
                    queries.append(action["query"])
                queries += action.get("queries") or []
                for s in action.get("sources") or []:
                    if s.get("url"):
                        retrieved.append(s["url"])
                    elif s.get("name"):  # oai-sports / oai-weather / oai-finance feeds
                        feeds.append(s["name"])
            elif atype in ("open_page", "find_in_page") and action.get("url"):
                opened.append(action["url"])
        elif itype == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text_parts.append(c.get("text", ""))
                    for a in c.get("annotations") or []:
                        if a.get("type") == "url_citation" and a.get("url"):
                            cited.append(a["url"])
    return {
        "text": "".join(text_parts),
        "search_queries": _dedupe(queries),
        # pages the model opened count as retrieved too
        "retrieved_urls": _dedupe(retrieved + opened),
        "opened_urls": _dedupe(opened),
        "cited_urls": _dedupe(cited),
        "n_searches": n_searches,
        "feeds": feeds,
        "usage": resp.get("usage"),
        "model_returned": resp.get("model"),
        "stop_reason": resp.get("status"),
        "raw": resp,
    }


def _scan_urls(obj, out):
    """Collect every 'url' value nested anywhere in obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "url" and isinstance(v, str):
                out.append(v)
            else:
                _scan_urls(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _scan_urls(v, out)


def parse_lenient(resp):
    """parse() for OpenAI-compatible Responses APIs whose search items aren't shaped
    exactly like OpenAI's (Qwen, OpenRouter): any output item whose type contains
    'search' counts as a search call, and urls/queries are scanned out of it."""
    rec = parse(resp)
    search_items = [i for i in resp.get("output", []) if "search" in (i.get("type") or "")]
    if not rec["retrieved_urls"]:
        found = []
        for item in search_items:
            _scan_urls(item, found)
        rec["retrieved_urls"] = _dedupe(found)
    if not rec["n_searches"]:
        rec["n_searches"] = len(search_items)
    if not rec["search_queries"]:
        qs = []
        for item in search_items:
            for src in (item.get("action") or {}, item.get("arguments") or {}, item):
                if isinstance(src, str):
                    continue
                if isinstance(src.get("query"), str):
                    qs.append(src["query"])
                qs += [q for q in src.get("queries") or [] if isinstance(q, str)]
        rec["search_queries"] = _dedupe(qs)
    return rec


def query(job, config):
    """Run one question through GPT with native web search. Raises on API errors."""
    resp = _get_client().responses.create(**_build_request(job, config))
    return parse(resp.model_dump(mode="json"))
