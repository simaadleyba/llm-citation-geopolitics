"""Native retriever for Mistral: Conversations API (beta) + Mistral's web_search tool.

Settings come from config.yaml -> models[name=mistral].native:
    model  pinned Large 4 id from client.models.list() (not a -latest alias)
    tool   web_search (not web_search_premium, which adds a curated news feed)

No agent object and no instructions: conversations.start takes model + tools directly,
so Mistral gets no system prompt the other providers don't.

Limitation: Mistral returns only cited sources (tool_reference chunks), not the full
list of search results. retrieved_urls is therefore empty and retrieved_available=False;
selection bias can't be computed for Mistral native.
Check that the org has no admin domain filter: filtered results are dropped silently.
"""

import json
import os

from mistralai import Mistral

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    return _client


def _dedupe(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def _build_request(job, config):
    native = job["model"]["native"]
    gen = config.get("generation", {})
    req = {
        "model": native["model"],
        "inputs": job["question"]["text"],
        "tools": [{"type": native.get("tool", "web_search")}],
        "store": False,
    }
    if gen.get("temperature") is not None:
        req["completion_args"] = {"temperature": gen["temperature"]}
    if gen.get("system_prompt"):
        req["instructions"] = gen["system_prompt"]
    return req


def _query_from_args(args):
    """tool.execution arguments may be a json string or dict; pull out the query if present."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if isinstance(args, dict):
        return args.get("query")
    return None


def parse(resp):
    """Normalize one conversations.start response dict."""
    text_parts, queries, cited, references = [], [], [], []
    n_searches = 0
    model_returned = None
    for entry in resp.get("outputs", []):
        etype = entry.get("type")
        if etype == "tool.execution" and entry.get("name", "").startswith("web_search"):
            n_searches += 1
            q = _query_from_args(entry.get("arguments"))
            if q:
                queries.append(q)
        elif etype == "message.output":
            model_returned = entry.get("model") or model_returned
            content = entry.get("content")
            if isinstance(content, str):
                text_parts.append(content)
                continue
            for chunk in content or []:
                ctype = chunk.get("type")
                if ctype == "text":
                    text_parts.append(chunk.get("text", ""))
                elif ctype == "tool_reference" and chunk.get("url"):
                    cited.append(chunk["url"])
                    references.append(
                        {k: chunk.get(k) for k in ("url", "title", "source", "tool")}
                    )
    return {
        "text": "".join(text_parts),
        "search_queries": _dedupe(queries),
        "retrieved_urls": [],
        "retrieved_available": False,
        "cited_urls": _dedupe(cited),
        "references": references,
        "n_searches": n_searches,
        "usage": resp.get("usage"),
        "model_returned": model_returned,
        "stop_reason": None,
        "raw": resp,
    }


def query(job, config):
    """Run one question through Mistral with native web search. Raises on API errors."""
    resp = _get_client().beta.conversations.start(**_build_request(job, config))
    return parse(resp.model_dump(mode="json"))
