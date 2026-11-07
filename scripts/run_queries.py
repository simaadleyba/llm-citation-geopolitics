"""Send every question x model x retriever x rep to the APIs and log raw responses.

Input:  data/questions.jsonl, config.yaml, .env
Output: data/raw/responses_<YYYYMMDD>.jsonl (append-only, one line per call)

Each output line follows RECORD_FIELDS. Never edit raw files after writing;
reruns append new lines with a new run_id.
"""

RECORD_FIELDS = [
    "run_id", "timestamp", "model", "version", "provider", "retriever",
    "lang", "question_id", "rep", "raw_response", "search_queries",
    "retrieved_urls", "cited_urls", "refusal_flag", "latency", "cost", "error",
]


def load_config(path="config.yaml"):
    """Read config.yaml into a dict."""
    raise NotImplementedError


def load_questions(path="data/questions.jsonl"):
    """Return a list of question dicts (one per question x language)."""
    raise NotImplementedError


def build_jobs(questions, config):
    """Cross questions with models, retrievers and reps; shuffle if configured.

    Returns a list of dicts: {question, model, retriever, rep}.
    """
    raise NotImplementedError


def call_common(job, config):
    """OpenRouter request with the web plugin forced to Exa.

    Must return retrieved urls (the search results) and cited urls separately.
    """
    raise NotImplementedError


def call_native(job, config):
    """Call the provider's own API with its own search tool.

    Dispatch on job['model']['native']['api']: openai, anthropic, mistral,
    deepseek, dashscope. Log search queries and retrieved urls where exposed.
    """
    raise NotImplementedError


def detect_refusal(text):
    """Flag responses that decline to answer. Keep the rule simple and documented."""
    raise NotImplementedError


def write_record(record, out_path):
    """Append one record as a json line."""
    raise NotImplementedError


def main():
    """Load config and questions, run all jobs with retries, log errors to logs/."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
