# llm-citation-geopolitics

Auditing which news sources LLM search cites, and how that shifts with model origin (US/FR/CN), query language, and topic controversy.

## Design

| Factor | Levels |
| --- | --- |
| Model origin | US (GPT-6.1 Sol, Claude Sonnet 5.5, Gemini 3.8 Flash), FR (Mistral Large 4), CN (Qwen3.8 Max, DeepSeek V4 Pro) |
| Query language | EN, FR, ZH (Simplified) |
| Topic country | US, FR, CN |
| Controversy | factual, disputed |
| Prompt condition | default (no instruction), cite ("Cite your sources inline as links.") |

Main run: 36 questions (6 per topic country × controversy cell) × 3 languages × 6 models × 3 reps = 1,944 calls, default condition.
Robustness arm: the same questions in the cite condition, 1 rep = 648 calls.

### Retrieval: native only (what users get)

Each system is queried through its vendor's own search-enabled API, so we audit the system as shipped:
model plus the vendor's search engine. Claims are about systems ("DeepSeek's search-enabled API cites ..."),
not about models in isolation, and about the API, not the consumer apps (which add their own prompts,
memory and personalization).

The earlier common-retriever condition (every model through OpenRouter with Exa) is off: OpenRouter
annotates every search result as a citation, so it gave no citation measure, and it was the most expensive
part. `clients/openrouter.py` is kept; add `common` to `run.retrievers` in config, or pass
`--retriever common`, to run it.

### Prompt conditions

- **default** (primary): the bare question, no system prompt. Five of six APIs return structured citations
  without being asked.
- **cite** (robustness, 1 rep): the same question with one uniform instruction. Asking for citations can change
  what gets cited, so this arm measures that effect rather than assuming it away. It also recovers citations
  for DeepSeek, whose API returns none by default.

### What is measured per call

| | GPT | Claude | Gemini | Mistral | Qwen | DeepSeek |
| --- | --- | --- | --- | --- | --- | --- |
| sites checked (retrieved) | yes | yes | no | no | check in pilot | yes |
| structured citations | yes | yes | yes | yes | yes | no |

Cited = structured citations ∪ urls written in the answer text. `calls.csv` also records why a call has
no citations (`citation_status`): `no_search` (answered from memory), `searched_no_cite` (a behavioral
signal), `api_no_citations` (DeepSeek default: a format limit, not behavior). Citation rate is itself
an outcome.

## Layout

```
config.yaml              models, versions, providers, retrievers, prompt conditions
data/
  questions.jsonl        one row per question × language (schema below)
  raw/                   API outputs, append-only, gitignored
  processed/             citations.csv, analysis.csv
  labels/                domain_labels.csv
scripts/
  smoke_test.py          one hard-coded question × all models → logs/ (checks keys, ids, parsers)
  run_queries.py         questions × config → data/raw/ (shuffled, retried, resumable)
  clients/               one module per api, same query(job, config) -> record interface
    *_native.py          openai, anthropic, google, mistral, deepseek, dashscope (main design)
    openrouter.py        common retriever via OpenRouter + Exa (kept, off by default)
  extract_citations.py   raw → processed/calls.csv, citations.csv, domains_to_label.csv
  merge_labels.py        citations + labels + questions → processed/analysis.csv
notebooks/               analysis only, never calls APIs
logs/                    runtime logs (errors, retries, cost)
```

## Pipeline

```
cp .env.example .env     # add keys
pip install -r requirements.txt
python scripts/smoke_test.py                # one fixed question through all 6 systems
python scripts/run_queries.py --dry-run     # list jobs, no calls
python scripts/run_queries.py --limit 12    # pilot
python scripts/run_queries.py               # main run (default condition); rerun to resume after a crash
python scripts/run_queries.py --condition cite   # robustness arm
python scripts/extract_citations.py
# label new domains in data/labels/domain_labels.csv
python scripts/merge_labels.py
```

## questions.jsonl schema

One json object per line, one line per question × language. Translations share `question_id`.

| Field | Description |
| --- | --- |
| question_id | e.g. `CN-D-01` (topic country, controversy, number) |
| pair_id | links a disputed question to its factual counterpart |
| topic_country | US, FR, CN |
| controversy | factual, disputed |
| domain | e.g. military, elections, economy |
| template | what_happened, why, purpose |
| event | short event description |
| event_date | YYYY-MM-DD |
| lang | en, fr, zh |
| text | the question as sent |
| translator | who translated |
| back_translation_ok | bool |
| verification_urls | sources justifying the factual/disputed label |

Model, retriever, condition and rep are run configuration, not question fields.

## domain_labels.csv

`domain, country, language, type, state_affiliated, us_lean, annotator, notes`

- `type`: news, encyclopedic, government, social, other
- `us_lean`: US outlets only (MBFC or DomainDemo), blank otherwise
- labeled by two annotators independently; report Cohen's kappa
