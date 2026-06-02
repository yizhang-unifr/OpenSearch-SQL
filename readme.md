# OpenSearch-SQL (with plugins)
A comprehensive Text-to-SQL framework that achieved first place on [BIRD](https://bird-bench.github.io/) in August 2024. Below is the complete flowchart.
<p align="center">
  <img src="./image/overview3.jpg" alt="image" />
</p>
<div align="center">
  

</div>

## Pipeline

### Overall flowchart with domain extensions highlighted:
```mermaid
flowchart TD
    A([Input question])

    subgraph PRE [Preprocessing]
        direction LR
        p1[generate_db_schema] --> p2[extract_col_value] --> p3[extract_query_noun] --> p4[column_retrieve_and_other_info]
    end

    subgraph POST [Postprocessing]
        direction LR
        q1[align_correct] --> q2[vote] --> q3[evaluation]
    end

    subgraph LEGEND [Legend]
        direction LR
        L1[new node]:::new
        L2[extended node]:::mod
        L3[original node]
    end

    A --> p1
    p4 --> F[implicit_context_enhance]:::new
    F --> G[candidate_generate]:::mod
    G --> H[query_optimizer]:::new
    H --> q1
    q3 --> Z([Result])

    classDef new fill:#d4edda,stroke:#28a745,stroke-width:3px,color:#000
    classDef mod fill:#fff3cd,stroke:#ffa500,stroke-width:3px,color:#000
```

### Detailed chart of enhanced `candidate_generate` node:

```mermaid
flowchart TD
    subgraph LEGEND [Legend]
        direction LR
        L1[new injection]:::new
        L2[original content]
    end

    A1[schema · columns · FK] --> P
    A2[geo_context block]:::new --> P
    A3[OGF block]:::new --> P
    A4[entity hint]:::new --> P
    A5[semantic hint]:::new --> P

    P([Prompt]) --> L[LLM]
    L --> C[SQL candidates]
    C --> V[constraint_validator]:::new
    V --> O([corrected SQL])

    classDef new fill:#d4edda,stroke:#28a745,stroke-width:3px,color:#000
```

| Node | Role |
| --- | --- |
| `generate_db_schema` | Embed schema columns with sentence-transformer |
| `extract_col_value` | Identify relevant columns and values from the question |
| `extract_query_noun` | Parse `#columns:` / `#values:` markers |
| `column_retrieve_and_other_info` | Top-k schema linking via BERT similarity |
| **`implicit_context_enhance`** | **Inject geo context + ontology grounding** _(domain extension)_ |
| **`candidate_generate`** | **Beam-search SQL candidates + entity/semantic hints** _(domain extension)_ |
| **`sql_audit`** | **Constraint validation of generated SQL candidates** _(domain extension)_ |
| **`query_optimizer`** | **Rewrite large lat/lon IN-lists to VALUES JOIN** _(domain extension, `full` only)_ |
| `align_correct` | Style & function alignment across candidates |
| `vote` | Self-consistency selection |
| `evaluation` | Execution match (EX) scoring |

**Ablation modes** — each step adds one complete contribution. OGF and validator are always paired: OGF discovers unit semantics, validator enforces them in SQL.

| Mode | Geo context | OGF + Validator | Entity + Semantic hints | Query optimizer |
| --- | :---: | :---: | :---: | :---: |
| `baseline` | | | | |
| `geo` | ✓ | | | |
| `ogf` | ✓ | ✓ | | |
| `hints` | ✓ | ✓ | ✓ | |
| `full` | ✓ | ✓ | ✓ | ✓ |

Legacy fine-grained modes (`a1`–`a5`) are still available for detailed analysis.

---

## Overview

OpenSearch-SQL consists of modules for Preprocessing, Extraction, Generation, Refinement, and Alignment. The entire framework operates without additional training; GPT, DeepSeek, Gemini, Qwen, and other LLMs are supported.

## Installation

```shell
uv sync
```

> This project uses [uv](https://docs.astral.sh/uv/) for dependency management. All commands below use `uv run`.

---

## Setup: Generate Embeddings

Column-value embeddings must be generated once before running evaluations. They are read by the `column_retrieve_and_other_info` node for schema linking.

**Requirements:** a running PostgreSQL instance with env vars set (see `.env`):

```
DB_HOST=...  DB_PORT=5432  DB_NAME=meteo  DB_USER=...  DB_PASS=...
DB_SCHEMA=public          # optional, default: public
```

```shell
uv run src/database_process/make_emb.py \
    --emb_dir data/emb \
    --db_name meteo \
    --bert_model all-mpnet-base-v2 \
    --env_file .env
```

Output: `data/emb/meteo.pkl.gz` and `data/emb/meteo_value.pkl.gz`

| Flag | Default | Description |
|---|---|---|
| `--emb_dir` | required | Output directory for `.pkl.gz` files |
| `--db_name` | `meteo` | Logical database name |
| `--bert_model` | `all-mpnet-base-v2` | SentenceTransformer model |
| `--env_file` | `.env` | Path to `.env` file with DB credentials |

---

## Setup: Fewshot Index

Vector-based fewshot retrieval must be built offline before running evaluations with `--fewshot`. It embeds all training questions into ChromaDB and retrieves the top-K most similar examples for each test question.

**Step 1 — Preprocess test data to eval JSON:**

```shell
# No flags needed — defaults to split_config_III_tiered
uv run src/database_process/preprocess_test_data.py

# Explicit split
uv run src/database_process/preprocess_test_data.py \
    --split split_config_III_tiered
```

Outputs `data/data_preprocess/test_data_point.json` and `test_data_bbox.json` (752 rows each).

| Flag | Default | Description |
|---|---|---|
| `--split` | `split_config_III_tiered` | Split config name under `../../data/` |
| `--test_xlsx` | auto-detected | Explicit path to `test_data.xlsx` (overrides `--split`) |
| `--out_dir` | `data/data_preprocess` | Output directory |
| `--limit` | — | Restrict to first N rows per variant (for quick tests) |

**Step 2 — Build ChromaDB index and generate `fewshot/questions.json`:**

```shell
# No flags needed — train_xlsx defaults to split_config_III_tiered, eval_files auto-detected
uv run src/database_process/build_fewshot_index.py --rebuild
```

Outputs `data/chroma_fewshot/` (persistent embedding index, 6016 documents — each training row indexed twice as point and bbox variants) and `data/fewshot/questions.json` (lookup used by the pipeline at inference time). At retrieval time, only the matching geo_mode variant is queried so the returned SQL always matches the target style.

| Flag | Default | Description |
|---|---|---|
| `--train_xlsx` | auto-detected | Path to `train_data.xlsx` |
| `--eval_files` | all `*.json` in `data/data_preprocess/` | Explicit list of eval JSON files to generate fewshot for |
| `--top_k` | `3` | Number of similar training examples per test question |
| `--model` | `all-mpnet-base-v2` | SentenceTransformer model |
| `--rebuild` | off | Drop and recreate the ChromaDB collection (needed when training data changes) |

> **Note:** `--rebuild` is only needed when `train_data.xlsx` changes. Subsequent runs reuse the existing ChromaDB index automatically.

---

## Running Evaluations

All commands run from `src/OpenSearch-SQL/`.

### Single configuration — `run_eval.py`

Runs one ablation mode on one dataset and optionally exports results to XLSX.

```shell
# Quick smoke-test: first question only, full pipeline
uv run run/run_eval.py \
    --dataset data/data_preprocess/test_data_point.json \
    --ablation full --fewshot --geo-anchor points --end 1

# Full test set, few-shot, export XLSX when done
uv run run/run_eval.py \
    --dataset data/data_preprocess/test_data_point.json \
    --ablation full --fewshot --geo-anchor points \
    --end -1 --export-xlsx

# Zero-shot baseline, Swiss AI provider
uv run run/run_eval.py \
    --dataset data/data_preprocess/test_data_point.json \
    --ablation baseline --provider swiss_ai \
    --geo-anchor points --end -1 --export-xlsx

# Bbox variant with few-shot
uv run run/run_eval.py \
    --dataset data/data_preprocess/test_data_bbox.json \
    --ablation full --fewshot --geo-anchor bbox \
    --end -1 --export-xlsx

# Dry-run: print resolved config without executing
uv run run/run_eval.py --dry-run
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `landcover10` | Dataset name or path. Name only → `src/database_process/<name>.json`; relative path → relative to `src/OpenSearch-SQL/`; absolute path → used as-is |
| `--ablation` | `full` | Ablation mode. Canonical: `baseline`, `geo`, `ogf`, `hints`, `full`. Legacy: `a1`–`a5` |
| `--start` / `--end` | `0` / `1` | Question index range (0-based, exclusive end). `-1` runs all questions |
| `--fewshot` | off | Enable few-shot retrieval. Requires `data/fewshot/questions.json` (see Setup: Fewshot Index) |
| `--geo-anchor` | `points` | Geo anchor style. `points` = point IN-list SQL; `bbox` = bounding-box lat/lon range SQL. Must match the dataset variant |
| `--provider` | — | LLM provider shorthand: `openai`, `swiss_ai`, `bedrock`, `ollama`, `anthropic`. Overrides `models.yaml` |
| `--model` | — | Model name, used together with `--provider` |
| `--llm-config` | `config/models.yaml` | Path to LLM YAML config file |
| `--n-candidates` | `5` | Number of SQL candidates generated per question (beam width) |
| `--temperature` | `0.7` | Sampling temperature for candidate generation |
| `--export-xlsx` | off | Export a multi-sheet XLSX report to the run directory after completion |
| `--skip-existing` | off | Skip a run if its result directory already exists (safe for resuming interrupted jobs) |
| `--bert-model` | `all-mpnet-base-v2` | SentenceTransformer model used for schema linking |
| `--dry-run` | off | Print the fully resolved config and pipeline graph without executing |

### Multiple configurations — `run_ablation.py`

Runs multiple ablation modes sequentially on the same dataset. All `run_eval.py` flags are accepted and forwarded to each mode.

```shell
# Canonical 5-level ablation, point dataset, few-shot, export XLSX per mode
uv run run/run_ablation.py \
    --dataset data/data_preprocess/test_data_point.json \
    --fewshot --geo-anchor points \
    --end -1 --export-xlsx --skip-existing

# Bbox variant
uv run run/run_ablation.py \
    --dataset data/data_preprocess/test_data_bbox.json \
    --fewshot --geo-anchor bbox \
    --end -1 --export-xlsx --skip-existing

# Run only specific modes
uv run run/run_ablation.py \
    --modes baseline,ogf,full \
    --dataset data/data_preprocess/test_data_point.json \
    --fewshot --geo-anchor points --end -1 --export-xlsx

# Zero-shot run (no --fewshot)
uv run run/run_ablation.py \
    --dataset data/data_preprocess/test_data_point.json \
    --geo-anchor points --end -1 --export-xlsx --skip-existing

# Dry-run to preview all modes
uv run run/run_ablation.py --dry-run
```

**Additional flag:**

| Flag | Default | Description |
| --- | --- | --- |
| `--modes` | `baseline,geo,ogf,hints,full` | Comma-separated list of ablation modes to run in order. Use legacy `a1`–`a5` for fine-grained analysis |

**XLSX output location** — each mode writes its report to:

```text
results/<dataset_name>/<no_few_shot|with_few_shot>/<mode>/<pipe_hash>/<timestamp>/results_<ts>.xlsx
```

---

## LLM Provider Configuration

### `config/models.yaml` (default)

```yaml
provider: openai          # openai | bedrock | swiss_ai | ollama | anthropic
model: gpt-4o
temperature: 0.0
```

### Supported providers

| Provider | `--provider` value | Required env vars |
|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| AWS Bedrock | `bedrock` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| Swiss AI (CSCS) | `swiss_ai` | `SWISS_AI_BASE_URL`, `SWISS_AI_API_KEY`, `SWISS_AI_MODEL` |
| Ollama (local) | `ollama` | — |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |

### Swiss AI (Qwen3 on CSCS vLLM)

Copy `config/models_swiss_qwen3_5_27B.yaml` or pass `--provider swiss_ai`:

```yaml
# config/models_swiss_qwen3_5_27B.yaml
provider: swiss_ai
model: "Qwen/Qwen3.5-27B"
temperature: 0.0
enable_thinking: false       # suppress Qwen3 chain-of-thought output
use_structured_output: true  # use JSON format for extract_col_value
```

```shell
uv run run/run_eval.py --llm-config config/models_swiss_qwen3_5_27B.yaml --dataset landcover10 --end -1
# or shorthand:
uv run run/run_eval.py --provider swiss_ai --end -1
```

### Quick provider override

Use `--provider` (and optionally `--model`) without creating a YAML file:

```shell
uv run run/run_eval.py --provider openai --model gpt-4o --end -1
uv run run/run_eval.py --provider swiss_ai --model Qwen/Qwen3.5-27B --end -1
```

