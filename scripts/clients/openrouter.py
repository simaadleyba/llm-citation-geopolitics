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
import re

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


URL_RE = re.compile(r"https?://[^\s)\]>\"'`]+")


def parse(resp):
    """OpenRouter attaches every search result to the message as a url_citation annotation
    (start_index = end_index = 0, i.e. not anchored to any text), so annotations are not
    citations. Cited = urls the model actually wrote into its answer; the annotation list is
    kept as annotated_urls. Most models write no urls without being asked, so cited is
    usually empty here, which is why the common retriever needs a citation instruction."""
    rec = openai_native.parse_lenient(resp)
    rec["annotated_urls"] = rec["cited_urls"]
    rec["cited_urls"] = openai_native._dedupe(u.rstrip(".,;:") for u in URL_RE.findall(rec["text"]))
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
    if not r.ok:
        # openrouter explains 4xx in the body (no credit, no endpoint for provider, bad param)
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return parse(r.json())
