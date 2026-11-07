"""API clients. Every client exposes query(job, config) -> dict with the same keys:

text            final answer text
search_queries  list[str], queries the search tool ran (empty if not exposed)
retrieved_urls  list[str], every url the search returned, in order, deduped
retrieved_available  bool, False when the api exposes only cited urls (mistral native)
cited_urls      list[str], urls cited in the answer, in order, deduped
n_searches      int, number of searches run (None if not exposed)
usage           dict, token and tool usage as reported by the api
model_returned  str, model id the api says it used
stop_reason     str
raw             dict, full api response(s), json-serializable
"""
