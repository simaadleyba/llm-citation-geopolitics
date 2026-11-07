"""Native retriever for Mistral: Conversations API (beta) + Mistral's web_search tool.

Settings come from config.yaml -> models[name=mistral].native:
    model  pinned Large 4 id from client.models.list() (not a -latest alias)
    tool   web_search (not web_search_premium, which adds a curated news feed)

No agent object and no instructions: conversations.start takes model + tools directly,
so Mistral gets no system prompt the other providers don't.

The conversation is stored (the default), because responses can end right after the
search calls and the answer is fetched by continuing the same conversation (see query()).

Retrieved = the urls in each tool.execution's info.result (the search results Mistral got).
Cited = tool_reference chunks in the answer.
Check that the org has no admin domain filter: filtered results are dropped silently.
"""

import json
import os

try:
    from mistralai.client import Mistral  # sdk v2+
except ImportError:
    from mistralai import Mistral  # sdk v1

_client = None
MAX_CONTINUATIONS = 3
# the api rejects an empty append, so the continuation is one neutral word in the question's language
CONTINUE = {"en": "Continue.", "fr": "Continuez.", "zh": "请继续。"}


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


def _result_urls(info):
    """tool.execution info.result is a json string mapping search-result ids to
    {url, title, source, ...}; return the urls in order."""
    result = (info or {}).get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return []
    if not isinstance(result, dict):
        return []
    return [v["url"] for v in result.values() if isinstance(v, dict) and v.get("url")]


def parse(responses):
    """Normalize the response dicts of one conversation (start + any continuations)."""
    text_parts, queries, retrieved, cited, references = [], [], [], [], []
    n_searches = 0
    model_returned = None
    for resp in responses:
        for entry in resp.get("outputs", []):
            etype = entry.get("type")
            if etype == "tool.execution" and entry.get("name", "").startswith("web_search"):
                n_searches += 1
                q = _query_from_args(entry.get("arguments"))
                if q:
                    queries.append(q)
                retrieved += _result_urls(entry.get("info"))
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
        "retrieved_urls": _dedupe(retrieved),
        "retrieved_available": True,
        "cited_urls": _dedupe(cited),
        "references": references,
        "n_searches": n_searches,
        "n_continuations": len(responses) - 1,
        "usage": [r.get("usage") for r in responses],
        "model_returned": model_returned,
        "stop_reason": None,
        "raw": responses,
    }


def _ends_after_tools(resp):
    """True when the last output entry is a tool execution: the answer hasn't been written yet."""
    outputs = resp.get("outputs") or []
    return bool(outputs) and outputs[-1].get("type") == "tool.execution"


def query(job, config):
    """Run one question through Mistral with native web search. Raises on API errors.

    Mistral Large 4 often ends a response right after its search calls (preamble line,
    tool executions, no answer). When that happens the conversation is continued with an
    empty append until the answer arrives (at most MAX_CONTINUATIONS times).
    """
    client = _get_client()
    resp = client.beta.conversations.start(**_build_request(job, config)).model_dump(mode="json")
    responses = [resp]
    while _ends_after_tools(resp) and len(responses) <= MAX_CONTINUATIONS:
        resp = client.beta.conversations.append(
            conversation_id=resp["conversation_id"],
            inputs=CONTINUE.get(job["question"].get("lang"), CONTINUE["en"]),
        ).model_dump(mode="json")
        responses.append(resp)
    if _ends_after_tools(resp):
        raise RuntimeError(
            "mistral still had no answer after its search calls "
            f"(conversation_id={resp.get('conversation_id')})"
        )
    return parse(responses)
