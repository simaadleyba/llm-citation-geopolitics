"""Native retriever for Qwen: Model Studio (DashScope) OpenAI-compatible Responses API + web_search.

The Chat Completions endpoint doesn't return search sources, so this uses Responses.
Request and response follow OpenAI's Responses format, so this reuses openai_native's parser.

Settings come from config.yaml -> models[name=qwen].native:
    model     pinned qwen3.8-max snapshot (0902), not the moving alias
    tool      web_search only (no web_extractor / code_interpreter: other providers get search only)
    thinking  optional bool; omit to keep the model default
    base_url  international (Singapore) endpoint. The mainland endpoint may use a
              different search backend; report which one ran.
"""

import os

from openai import OpenAI

from . import openai_native

DEFAULT_BASE_URL = "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1"

_clients = {}


def _get_client(base_url):
    if base_url not in _clients:
        _clients[base_url] = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=base_url)
    return _clients[base_url]


def _build_request(job, config):
    native = job["model"]["native"]
    gen = config.get("generation", {})
    req = {
        "model": native["model"],
        "input": job["question"]["text"],
        "tools": [{"type": native.get("tool", "web_search")}],
    }
    if native.get("thinking") is not None:
        req["extra_body"] = {"enable_thinking": native["thinking"]}
    if gen.get("temperature") is not None:
        req["temperature"] = gen["temperature"]
    if gen.get("system_prompt"):
        req["instructions"] = gen["system_prompt"]
    return req


def parse(resp):
    """Qwen's search items hold several queries each (fan-out), so search_queries can be
    much longer than n_searches. The API returned no url_citation annotations in any smoke
    run; Qwen only cites as links in its text, and only when asked. cited_available=False
    marks that the structured channel is absent (a format limit, not behavior)."""
    rec = openai_native.parse_lenient(resp)
    if not rec["cited_urls"]:
        rec["cited_available"] = False
    return rec


def query(job, config):
    """Run one question through Qwen with native web search. Raises on API errors."""
    native = job["model"]["native"]
    client = _get_client(native.get("base_url", DEFAULT_BASE_URL))
    resp = client.responses.create(**_build_request(job, config))
    return parse(resp.model_dump(mode="json"))
