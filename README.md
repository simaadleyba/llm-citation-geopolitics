# llm-citation-geopolitics

Auditing which news sources LLM search cites, and how that shifts with model origin (US/FR/CN), query language, and topic controversy.

## Design

| Factor | Levels |
| --- | --- |
| Model origin | US (GPT-6.1 Sol, Claude Sonnet 5.5, Gemini 3.8 Flash), FR (Mistral Large 4), CN (Qwen3.8 Max, DeepSeek V4 Pro) |
| Query language | EN, FR, ZH (Simplified) |
| Topic country | US, FR, CN |
| Controversy | factual, disputed |
| Prompt condition | default (bare question), cite (question + "Cite your sources inline as links.", in the question's language) |

36 questions (6 per topic country × controversy cell) × 3 languages × 6 systems × 2 conditions × 1 call = 1,296 calls (~$65).
Every question × language × system runs once in each condition, so the condition effect is a paired comparison.
Single calls mean run-to-run noise is averaged across the 108 question × language items rather than estimated per item.

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

- **default**: the bare question, no system prompt. GPT, Claude and Gemini return structured citations
  without being asked; Mistral, Qwen and DeepSeek return none.
- **cite**: the same question with one instruction appended to the question text, translated into the
  question's language (config `conditions.cite.suffix`). It goes in the question, not a system prompt,
  because DeepSeek ignores system prompts when search is on. Asking for citations can change what gets
  cited, so this condition measures that effect rather than assuming it away. It also recovers citations
  for Mistral, Qwen and DeepSeek, whose APIs return none by default. Records keep both `question_text` and the
  `prompt_text` actually sent.

### What is measured per call

Observed in the smoke tests (re-check in the pilot):

| | GPT | Claude | Gemini | Mistral | Qwen | DeepSeek |
| --- | --- | --- | --- | --- | --- | --- |
| sites checked (retrieved) | yes | yes | no | yes | yes | yes |
| structured citations | yes | yes | yes | no | no | no |
| links in text when asked (cite) | yes | yes | – | check | yes | yes |

Qwen fans out (several queries per search call, often 5+ calls), so it retrieves far more pages than the others.
Mistral Large 4 often stops after its searches without answering; the client continues the conversation
with one word ("Continue." in the question's language) and logs how often (`n_continuations`).

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
python scripts/run_queries.py               # full run, both conditions; rerun to resume after a crash
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
