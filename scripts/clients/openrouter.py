"""Common retriever: OpenRouter Responses API + the openrouter:web_search server tool, engine forced to Exa.

Every model gets the same search engine and settings. The model writes its own queries
and may search up to max_uses times; we log every query and every retrieved url, so
selection (cited vs retrieved) and query formulation are both measurable.

Settings come from config.yaml:
    models[*].openrouter_id        model slug
    models[*].openrouter_provider  provider slug pinned with fallbacks disabled
    retrievers.common              engine, mode, max_results, max_uses

No reasoning field (model default, as in the native clients), no domain filters,
no user_location (Exa ignores it anyway).
"""

import os

import requests

from . import openai_native

URL = "https://openrouter.ai/api/v1/responses"
TIMEOUT = 300


def _build_request(job, config):
    model = job["model"]
    common = config["retrievers"]["common"]
    gen = config.get("generation", {})
    params = {
        "engine": common.get("engine", "exa"),
        "max_results": common.get("max_results", 10),
        "max_uses": common.get("max_uses", 5),
    }
    if common.get("mode"):
        params["mode"] = common["mode"]
    req = {
        "model": model["openrouter_id"],
        "input": job["question"]["text"],
        "tools": [{"type": "openrouter:web_search", "parameters": params}],
    }
    if model.get("openrouter_provider"):
        req["provider"] = {"order": [model["openrouter_provider"]], "allow_fallbacks": False}
    if gen.get("temperature") is not None:
        req["temperature"] = gen["temperature"]
    if gen.get("system_prompt"):
        req["instructions"] = gen["system_prompt"]
    return req


def parse(resp):
    rec = openai_native.parse_lenient(resp)
    rec["provider_returned"] = resp.get("provider")
    return rec


def query(job, config):
    """Run one question through OpenRouter with Exa search. Raises on HTTP errors."""
    r = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=_build_request(job, config),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return parse(r.json())
