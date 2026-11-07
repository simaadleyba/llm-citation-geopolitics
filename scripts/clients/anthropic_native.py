"""Native retriever for Claude: Messages API + Anthropic's web search tool.

Settings come from config.yaml -> models[name=claude].native:
    model     e.g. claude-sonnet-5-5
    tool      web_search_20250305 (basic search, no dynamic filtering;
              newer versions let Claude prune results with code, an extra
              selection layer other providers don't have)
    max_uses  searches allowed per turn

No domain filters and no user_location: either would bias or localize results.
"""

import os

from anthropic import Anthropic

MAX_CONTINUATIONS = 5  # pause_turn resends before giving up

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _dedupe(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def _build_request(job, config):
    native = job["model"]["native"]
    gen = config.get("generation", {})
    tool = {
        "type": native.get("tool", "web_search_20250305"),
        "name": "web_search",
        "max_uses": native.get("max_uses", 5),
    }
    req = {
        "model": native["model"],
        "max_tokens": native.get("max_tokens", 4096),
        "messages": [{"role": "user", "content": job["question"]["text"]}],
        "tools": [tool],
    }
    if gen.get("temperature") is not None:
        req["temperature"] = gen["temperature"]
    if gen.get("system_prompt"):
        req["system"] = gen["system_prompt"]
    return req


def parse(responses):
    """Turn a list of response dicts (one per pause_turn continuation) into the normalized record."""
    text_parts, queries, retrieved, cited, errors = [], [], [], [], []
    n_searches = 0
    for resp in responses:
        for block in resp.get("content", []):
            btype = block.get("type")
            if btype == "server_tool_use" and block.get("name") == "web_search":
                q = (block.get("input") or {}).get("query")
                if q:
                    queries.append(q)
            elif btype == "web_search_tool_result":
                content = block.get("content")
                if isinstance(content, list):
                    retrieved += [r["url"] for r in content if r.get("url")]
                elif isinstance(content, dict):
                    errors.append(content.get("error_code"))
            elif btype == "text":
                text_parts.append(block.get("text", ""))
                for c in block.get("citations") or []:
                    if c.get("url"):
                        cited.append(c["url"])
        n_searches += ((resp.get("usage") or {}).get("server_tool_use") or {}).get("web_search_requests", 0) or 0

    last = responses[-1]
    return {
        "text": "".join(text_parts),
        "search_queries": queries,
        "retrieved_urls": _dedupe(retrieved),
        "cited_urls": _dedupe(cited),
        "n_searches": n_searches,
        "usage": [r.get("usage") for r in responses],
        "model_returned": last.get("model"),
        "stop_reason": last.get("stop_reason"),
        "search_errors": errors,
        "raw": responses,
    }


def query(job, config):
    """Run one question through Claude with native web search.

    Handles pause_turn by resending the paused assistant message unchanged.
    Raises on API errors; the orchestrator handles retries.
    """
    client = _get_client()
    req = _build_request(job, config)
    responses = []
    for _ in range(MAX_CONTINUATIONS + 1):
        resp = client.messages.create(**req).model_dump(mode="json")
        responses.append(resp)
        if resp.get("stop_reason") != "pause_turn":
            break
        req["messages"] = req["messages"] + [
            {"role": "assistant", "content": resp["content"]}
        ]
    return parse(responses)
