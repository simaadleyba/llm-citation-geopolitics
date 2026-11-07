"""Native retriever for DeepSeek: Anthropic-compatible endpoint + DeepSeek's server-side web search.

DeepSeek exposes search two ways:
  1. a server-side web_search tool on its Anthropic-format endpoint
     (https://api.deepseek.com/anthropic), where the model decides when and what to search
     and DeepSeek's in-house engine runs it. Used here: it's the closest thing to the
     native product, and it matches how the Claude client works.
  2. a standalone Search API (/v1/tools/search) that we would have to wire into our own
     tool loop. Not used: the harness would be ours, not DeepSeek's.

The request and response follow the Anthropic Messages format, so this reuses
anthropic_native's request builder, pause_turn loop and parser.

Settings come from config.yaml -> models[name=deepseek].native:
    model     deepseek-v4-pro  (= DeepSeek-V4-Pro-0813, same model as the OpenRouter slug)
    tool      web_search_20250305 to match Claude; if DeepSeek rejects it, try
              web_search_20260209 and record which one ran
    max_uses  searches allowed per turn

Thinking mode is DeepSeek's default and left on. No localization.
"""

import os

from anthropic import Anthropic

from . import anthropic_native

BASE_URL = "https://api.deepseek.com/anthropic"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)
    return _client


def query(job, config):
    """Run one question through DeepSeek with its server-side web search. Raises on API errors."""
    req = anthropic_native._build_request(job, config)
    return anthropic_native.run(_get_client(), req)
