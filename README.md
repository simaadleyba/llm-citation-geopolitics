# llm-citation-geopolitics

Auditing which news sources LLM search cites, and how that shifts with model origin (US/FR/CN), query language, and topic controversy.

## Design

| Factor | Levels |
| --- | --- |
| Model origin | US (GPT-6.1 Sol, Claude Sonnet 5.5, Gemini 3.8 Flash), FR (Mistral Large 4), CN (Qwen3.8 Max, DeepSeek V4 Pro) |
| Query language | EN, FR, ZH (Simplified) |
| Topic country | US, FR, CN |
| Controversy | factual, disputed |
| Retriever | common (OpenRouter + Exa), native (provider's own search) |

36 questions (6 per topic country × controversy cell) × 3 languages × 6 models × 2 retrievers × 5 reps = 6,480 calls.

## Layout

```
config.yaml              models, versions, providers, retriever settings
data/
  questions.jsonl        one row per question × language (schema below)
  raw/                   API outputs, append-only, gitignored
  processed/             citations.csv, analysis.csv
  labels/                domain_labels.csv
scripts/
  run_queries.py         questions × config → data/raw/
  extract_citations.py   raw → processed/citations.csv
  merge_labels.py        citations + labels + questions → processed/analysis.csv
notebooks/               analysis only, never calls APIs
logs/                    runtime logs (errors, retries, cost)
```

## Pipeline

```
cp .env.example .env     # add keys
pip install -r requirements.txt
python scripts/run_queries.py
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

Model, retriever and rep are run configuration, not question fields.

## domain_labels.csv

`domain, country, language, type, state_affiliated, us_lean, annotator, notes`

- `type`: news, encyclopedic, government, social, other
- `us_lean`: US outlets only (MBFC or DomainDemo), blank otherwise
- labeled by two annotators independently; report Cohen's kappa
