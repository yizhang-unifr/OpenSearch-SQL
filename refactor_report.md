# OpenSearch-SQL Refactoring Report

**Objective:** Adapt the [OpenSearch-SQL](https://github.com/xiangyue9607/OpenSearch-SQL) Text-to-SQL framework — originally designed for the BIRD benchmark with SQLite — to work with a meteorological (Meteo) dataset on PostgreSQL.

---

## 1. Overview

The OpenSearch-SQL framework implements an 8-node LangGraph pipeline for Text-to-SQL:

```
generate_db_schema → extract_col_value → extract_query_noun →
column_retrieve_and_other_info → candidate_generate → align_correct → vote → evaluation
```

The refactoring touched **~20 source files** across the following areas:

| Area                     | Changes                                              |
|--------------------------|------------------------------------------------------|
| Database layer           | SQLite → PostgreSQL (psycopg2)                       |
| Schema introspection     | PRAGMA/sqlite_master → information_schema            |
| SQL syntax               | strftime → EXTRACT, backticks → double quotes        |
| Data model               | BIRD benchmark → Meteo dataset                       |
| LLM integration          | Hardcoded OpenAI/DashScope → LLMFactory (multi-provider) |
| Few-shot selection       | DAIL-SQL external tool → ChromaDB vector similarity  |
| Evaluation wrapper       | New `eval/OpenSearch-SQL-eval.py` end-to-end script  |
| Timeout/reliability      | Added multi-layer timeouts to prevent pipeline hangs |
| Dependencies             | Removed `dashscope`, `chardet`; added `psycopg2-binary`, `langgraph`, `chromadb` |

---

## 2. Database Layer

### 2.1 Connection Management

**File:** `src/runner/execution.py` (rewritten)

| Original (SQLite)                      | Refactored (PostgreSQL)                                    |
|----------------------------------------|------------------------------------------------------------|
| `sqlite3.connect(db_path)`             | `psycopg2.connect(host, dbname, user, port, password)`     |
| File-path-based DB selection           | Environment variables (`DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PORT`, `DB_PASS`) |
| No query timeout                       | `statement_timeout` set on every connection (default 30s)  |

Key functions introduced:
- `_get_pg_connection(statement_timeout_ms=30000)` — creates a psycopg2 connection with a per-statement timeout baked in via the `options` parameter.
- `_normalize_sql(sql)` — strips schema qualifications (e.g. `era5_land2.meteo_tmax` → `meteo_tmax`) so queries work against the default `search_path`.
- `execute_sql(sql, fetch, timeout_s)` — executes SQL with explicit `SET statement_timeout`.
- `sql_exec(sql)` — lightweight execution wrapper returning `(result_set, time_cost)`.
- `compare_sqls(predicted_sql, ground_truth_sql)` — compares execution outputs of two queries, used in the evaluation node.

### 2.2 DatabaseManager Singleton

**File:** `src/runner/database_manager.py` (rewritten)

Replaced BIRD's path-based filesystem management with a singleton that manages:
- PostgreSQL connection lifecycle
- Paths to pipeline artifacts (`data_preprocess/dev.json`, `fewshot/questions.json`, `emb/`, `db_schema.json`)
- Comparison delegation to `execution.compare_sqls()`

```python
class DatabaseManager:
    """Singleton managing PostgreSQL connection and data paths for the Meteo dataset."""
    db_id = "meteo"  # always meteo (BIRD had hundreds of DBs)
```

### 2.3 Schema Introspection

**File:** `src/llm/db_conclusion.py` (rewritten)

| Original (SQLite)                                  | Refactored (PostgreSQL)                                      |
|----------------------------------------------------|--------------------------------------------------------------|
| `PRAGMA table_info(table_name)`                    | `SELECT * FROM information_schema.columns WHERE table_schema = %s` |
| `SELECT * FROM sqlite_master WHERE type='table'`   | `SELECT table_name FROM information_schema.tables WHERE table_schema = %s` |
| Foreign keys via BIRD's `tables.json`              | `information_schema.table_constraints` + `key_column_usage` + `constraint_column_usage` |

Key functions:
- `find_foreign_keys_pg()` — discovers foreign keys from the PostgreSQL system catalog.
- `get_complete_table_info()` — returns column metadata for all tables in the configured schema.
- `db_agent_string()` — generates a human-readable schema description for the LLM prompt (uses `get_db_des()` internally).

All functions respect the `DB_SCHEMA` environment variable (defaults to `public`).

---

## 3. SQL Syntax Adaptations

### 3.1 Date/Time Functions

**File:** `src/runner/check_and_correct.py`

SQLite's `strftime()` is not available in PostgreSQL. All date extraction was converted:

| SQLite                          | PostgreSQL                          |
|---------------------------------|-------------------------------------|
| `strftime('%Y', column)`        | `EXTRACT(YEAR FROM column)`         |
| `strftime('%m', column)`        | `EXTRACT(MONTH FROM column)`        |
| `strftime('%d', column)`        | `EXTRACT(DAY FROM column)`          |

### 3.2 Identifier Quoting

**File:** `src/runner/extract.py`

| SQLite          | PostgreSQL          |
|-----------------|---------------------|
| `` `column` ``  | `"column"`          |

The `quote_field()` utility now uses double quotes for identifiers containing special characters.

### 3.3 Type Casting & Division

**File:** `src/pipeline/candidate_generate.py`

Added a PostgreSQL division precision hint to the `rewrite_question()` function:

```
"Note: when dividing, use CAST(xxx AS NUMERIC) or xxx::numeric to avoid integer division."
```

### 3.4 Removed SQLite-Specific Filters

**File:** `src/runner/extract.py`

Removed `sqlite_sequence` table filtering (not applicable to PostgreSQL).

---

## 4. Data Model Adaptation

### 4.1 Task Dataclass

**File:** `src/runner/task.py` (rewritten)

Adapted for Meteo dataset fields while retaining BIRD fields for pipeline compatibility:

| BIRD field                | Meteo mapping                                    |
|---------------------------|--------------------------------------------------|
| `db_id` (per-question)    | Always `"meteo"` (single database)               |
| `question`                | `"Natural language question"` from dataset JSON  |
| `SQL`                     | `"SQL query"` from dataset JSON                  |
| `evidence`                | Optionally populated from `knowledge/meteo.md`   |
| `difficulty`              | Mapped from `"Category"` field                   |

New field: `category` — preserves the Meteo-specific category label (e.g. "Simple aggregation", "Temporal analysis").

### 4.2 Data Preprocessing

**File:** `src/database_process/data_preprocess.py` (rewritten)

Converts the Meteo dataset format into the OpenSearch-SQL pipeline format:
- Generates `data/data_preprocess/dev.json` (test questions in pipeline format)
- Generates a minimal `tables.json` from PostgreSQL introspection (BIRD-compatible format)

### 4.3 Column-Value Embeddings

**File:** `src/database_process/make_emb.py` (rewritten)

| Original (SQLite)                           | Refactored (PostgreSQL)                                 |
|---------------------------------------------|---------------------------------------------------------|
| Read from SQLite `.db` files                | Query PostgreSQL via `information_schema.tables`        |
| Process all Bird benchmark databases        | Process single `meteo` database                         |
| Store as plain pickle                       | Store as gzip-compressed pickle (`meteo.pkl.gz`, `meteo_value.pkl.gz`) |

The function `make_emb_pg()`:
1. Queries each table's string columns via `information_schema`
2. Samples up to 10,000 rows per table
3. Filters out UUIDs and numeric-only values
4. Generates SentenceTransformer embeddings (`all-mpnet-base-v2`)
5. Saves to `data/emb/`

---

## 5. LLM Integration

### 5.1 Model Adapter

**File:** `src/llm/model.py` (rewritten)

Replaced the original hardcoded OpenAI/DashScope integration with an adapter to the project's `LLMFactory`:

| Original                                     | Refactored                                             |
|----------------------------------------------|--------------------------------------------------------|
| Hardcoded `openai.ChatCompletion.create()`   | `LLMFactory.create()` via `config/models.yaml`        |
| DashScope API support                        | Bedrock, OpenAI, Scayle, Ollama support                |
| Model selected by engine string              | Model configured in YAML; engine string ignored        |

The `LLMFactoryAdapter` class preserves the same interface expected by all pipeline nodes:

```python
chat_model = model_chose(step, engine)      # returns LLMFactoryAdapter
response = chat_model.get_ans(prompt, temperature, n=1)
```

Internally, `get_ans()`:
1. Builds LangChain messages from the prompt
2. Invokes the LLM with retry logic (max 2 retries, 5s max backoff)
3. Logs prompt/response pairs via the Logger singleton

For `n > 1` (candidate generation), it uses `ThreadPoolExecutor` to invoke multiple completions concurrently.

### 5.2 Removed Dependencies

| Dependency    | Reason for removal                                                                |
|---------------|-----------------------------------------------------------------------------------|
| `dashscope`   | Was the DashScope (Alibaba Cloud) LLM SDK. Replaced by LLMFactory multi-provider support. |
| `chardet`     | Was used for character encoding detection when reading BIRD dataset files. Not needed for Meteo (UTF-8 only). |

---

## 6. Few-Shot Example Selection

### 6.1 Original Approach

The original OpenSearch-SQL relied on an external [DAIL-SQL](https://github.com/BeachWang/DAIL-SQL) tool to pre-compute similarity scores between questions and select few-shot examples. This required:
1. Running the DAIL-SQL pipeline separately
2. Storing pre-computed similarity files
3. Reading those files at runtime

### 6.2 Refactored Approach — ChromaDB

**File:** `eval/OpenSearch-SQL-eval.py` → `prepare_opensearch_data()`

Replaced with an in-process embedding-based similarity retrieval using **ChromaDB** as a persistent vector database:

1. **Embedding**: Training questions are embedded using ChromaDB's default model (`all-MiniLM-L6-v2`) and stored persistently in `data/fewshot/chromadb/`.
2. **Retrieval**: For each test question, the top-N most similar training questions are retrieved via cosine similarity.
3. **Prompt construction**: Retrieved examples are formatted into few-shot prompts matching the pipeline's expected format.

```python
TOP_N_FEWSHOT = 1  # number of similar training examples per test question

collection = chroma_client.get_or_create_collection(
    name="train_fewshot",
    metadata={"hnsw:space": "cosine"},
)
# Embeddings are computed only from the question text (not from SQL)
collection.upsert(
    ids=train_ids,
    documents=[r.get(QUESTION_COL, "") for r in train_data],
    ...
)
results = collection.query(query_texts=test_questions, n_results=TOP_N_FEWSHOT)
```

This removes the DAIL-SQL dependency entirely and makes few-shot selection self-contained.

---

## 7. Pipeline Node Modifications

All pipeline nodes were updated to reference PostgreSQL instead of SQLite in their DB info strings and prompts. Specific changes per node:

| Node                             | Key Changes                                                                                     |
|----------------------------------|-------------------------------------------------------------------------------------------------|
| `generate_db_schema`             | Introspects PostgreSQL via `db_agent_string()` instead of reading SQLite files / `tables.json`  |
| `extract_col_value`              | No structural changes; consumes fewshot from `questions.json` as before                         |
| `extract_query_noun`             | No structural changes                                                                            |
| `column_retrieve_and_other_info` | Uses `find_foreign_keys_pg()` instead of BIRD's `tables.json` for FK discovery                  |
| `candidate_generate`             | DB info string: `"SQLite"` → `"PostgreSQL"`; added numeric division hint                        |
| `align_correct`                  | DB info string: `"SQLite"` → `"PostgreSQL"`; removed `db_sqlite_path` parameter                 |
| `vote`                           | Uses PostgreSQL execution for SQL result comparison                                              |
| `evaluation`                     | Delegates to `execution.compare_sqls()` (PostgreSQL-based)                                      |

---

## 8. Evaluation Wrapper

**File:** `eval/OpenSearch-SQL-eval.py` (new)

A comprehensive end-to-end evaluation script that orchestrates the full pipeline:

### 8.1 Steps

1. **Split data** — Stratified train/test split by category (1 train example per category, rest for test).
2. **Check PostgreSQL** — Validates database connectivity.
3. **Prepare data directory** — Converts Meteo format → OpenSearch-SQL format; builds ChromaDB fewshot index.
4. **Generate embeddings** — Runs `make_emb_pg()` if `data/emb/meteo.pkl.gz` doesn't exist.
5. **Build pipeline** — Constructs the LangGraph pipeline with configurable nodes.
6. **Run per question** — Iterates over test questions with per-question timeout protection.
7. **Save results** — Outputs JSON, CSV, and log files with per-question predictions.
8. **Summary** — Prints accuracy metrics (or SQL generation counts if `--skip-exec`).

### 8.2 Arguments

| Flag                | Description                                      | Default       |
|---------------------|--------------------------------------------------|---------------|
| `--n`               | Number of SQL candidates                         | 5             |
| `--temperature`     | LLM sampling temperature                         | 0.7           |
| `--timeout`         | Per-question timeout (seconds)                   | 120           |
| `--with-knowledge`  | Inject `knowledge/meteo.md` as evidence          | False         |
| `--skip-exec`       | Skip execution-based evaluation (no `evaluation` + omit `align_correct`)  | False |

### 8.3 Output Structure

```
datasets/eval_result/
├── json/   opensearch_pipeline_<provider>_<model>_<timestamp>.json
├── csv/    opensearch_pipeline_<provider>_<model>_<timestamp>.csv
├── logs/   opensearch_pipeline_<provider>_<model>_<timestamp>.log
└── opensearch_pipeline_results/
    └── <timestamp>/
        ├── 0_meteo.json   (per-question pipeline state)
        ├── 1_meteo.json
        └── logs/
            ├── 0_meteo.log
            └── 1_meteo.log
```

---

## 9. Timeout and Reliability Fixes

The original pipeline had no timeout protection. With Bedrock API calls taking up to 120s and PostgreSQL queries potentially running indefinitely, the pipeline would frequently freeze.

### 9.1 Multi-Layer Timeout Architecture

| Layer                     | Mechanism                                             | Timeout  |
|---------------------------|-------------------------------------------------------|----------|
| PostgreSQL statements     | `statement_timeout` on every connection               | 30s      |
| SQL execution wrappers    | `func_timeout()` around `sql_exec()`                  | 30s      |
| Per-candidate correction  | `future.result(timeout=...)` in `muti_process_sql`    | 60s      |
| All candidates combined   | `as_completed(timeout=...)` in `muti_process_sql`     | 90s      |
| Per-question pipeline     | `ThreadPoolExecutor` with `shutdown(wait=False)`      | 120s     |
| Bedrock API               | `botocore.config.Config(read_timeout=120, connect_timeout=30)` | 120s/30s |

### 9.2 Key Fix: Non-Blocking Executor Shutdown

The critical freezing bug: `ThreadPoolExecutor` used as a context manager (`with _TPE(...)`) calls `shutdown(wait=True)` on exit, which blocks forever when a worker thread is stuck on I/O.

**Fix:** Manual pool management with `shutdown(wait=False, cancel_futures=True)`:

```python
_pool = _TPE(max_workers=1)
_fut = _pool.submit(_run_stream)
try:
    final_state = _fut.result(timeout=per_question_timeout)
except TimeoutError:
    print(f"TIMEOUT after {per_question_timeout}s – skipping question")
finally:
    _pool.shutdown(wait=False, cancel_futures=True)
```

This pattern was applied both in `eval/OpenSearch-SQL-eval.py` (outer per-question timeout) and `src/runner/check_and_correct.py` (inner per-candidate timeout).

### 9.3 LLM Retry Reduction

| Parameter        | Original | Refactored |
|------------------|----------|------------|
| Max retries      | 3        | 2          |
| Max backoff wait | 10s      | 5s         |

---

## 10. Dependency Changes

### `requirements.txt`

```
sentence_transformers
func_timeout
torch
pandas
numpy
psycopg2-binary        # NEW – PostgreSQL driver
python-dotenv           # NEW – .env file loading
scikit-learn
langgraph               # NEW – LangGraph pipeline framework
langchain-core          # NEW – LangChain core (LLM orchestration)
```

### Parent project `pyproject.toml`

Added: `chromadb` — persistent vector database for fewshot similarity retrieval.

### Removed

| Package     | Reason                                                                        |
|-------------|-------------------------------------------------------------------------------|
| `dashscope` | Alibaba Cloud LLM SDK — replaced by LLMFactory (Bedrock/OpenAI/Scayle/Ollama) |
| `chardet`   | Character encoding detection — not needed (Meteo data is UTF-8)               |

---

## 11. Deleted / Unused Components

| Component              | Status                                                   |
|------------------------|----------------------------------------------------------|
| `Bird/` directory      | Removed (BIRD benchmark data, not needed for Meteo)      |
| DAIL-SQL integration   | Removed (replaced by ChromaDB-based fewshot selection)   |
| SQLite `.db` files     | Not applicable (all data lives in PostgreSQL)            |
| `run/run_main.sh`      | Updated comments; kept for reference                     |
| `run/run_preprocess.sh`| Updated for PostgreSQL embedding generation              |

---

## 12. File-by-File Change Summary

| File                                                    | Change Type   | Description                                                    |
|---------------------------------------------------------|---------------|----------------------------------------------------------------|
| `src/runner/execution.py`                               | Rewritten     | SQLite → psycopg2; statement timeouts; schema normalization    |
| `src/runner/database_manager.py`                        | Rewritten     | Path-based → connection-based; Meteo directory structure       |
| `src/runner/task.py`                                    | Rewritten     | BIRD fields → Meteo fields with backward compatibility         |
| `src/runner/check_and_correct.py`                       | Modified      | strftime→EXTRACT; psycopg2 connections; non-blocking executor  |
| `src/runner/extract.py`                                 | Modified      | Backtick→double-quote; removed sqlite_sequence filter          |
| `src/llm/model.py`                                      | Rewritten     | OpenAI/DashScope → LLMFactory adapter                         |
| `src/llm/db_conclusion.py`                              | Rewritten     | PRAGMA → information_schema; FK discovery via system catalog   |
| `src/database_process/make_emb.py`                      | Rewritten     | SQLite value extraction → PostgreSQL queries; gzip compression |
| `src/database_process/data_preprocess.py`               | Rewritten     | BIRD format → Meteo format; PostgreSQL introspection           |
| `src/pipeline/generate_db_schema.py`                    | Modified      | Uses `db_agent_string()` for PostgreSQL introspection          |
| `src/pipeline/candidate_generate.py`                    | Modified      | "SQLite"→"PostgreSQL"; division hint                           |
| `src/pipeline/align_correct.py`                         | Modified      | "SQLite"→"PostgreSQL"; removed `db_sqlite_path`               |
| `src/pipeline/column_retrieve_and_other_info.py`        | Modified      | Uses `find_foreign_keys_pg()` for FK discovery                 |
| `src/pipeline/vote.py`                                  | Minor         | Uses PostgreSQL execution for result comparison                |
| `src/pipeline/evaluation.py`                            | Minor         | Delegates to `execution.compare_sqls()`                        |
| `requirements.txt`                                      | Updated       | +psycopg2-binary, +langgraph, +langchain-core; −dashscope, −chardet |
| `eval/OpenSearch-SQL-eval.py`                           | New           | End-to-end evaluation wrapper with ChromaDB fewshot            |
