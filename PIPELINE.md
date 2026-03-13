# RAG Evaluation Pipeline

This document explains the current end-to-end evaluation flow in this repository so another AI agent or engineer can continue the work without reverse-engineering the scripts.

## Goal

Given a spreadsheet of evaluation questions and reference answers, the pipeline:

1. normalizes the source data,
2. retrieves context from Qdrant,
3. generates answers with an LLM,
4. evaluates the answers with RAGAS metrics,
5. exports a clean Excel report.

The current reporting standard excludes `noise_sensitivity` because it repeatedly timed out in real runs.

## Expected Input

Primary source file:

- `data_awal.xlsx`

Assumption:

- the column structure stays the same as the current workbook,
- new runs may add more rows, but should not rename the required columns.

Relevant fields through the pipeline:

- `question`
- `ground_truth_answer`
- `context`
- `generated_answer`

## Environment

This repo expects a `.env` with working credentials for:

- generator LLM
- evaluator LLM
- embeddings
- Qdrant
- Ollama embeddings endpoint

Important variables used by the scripts:

- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `QDRANT_API_KEY` (optional)
- `OLLAMA_BASE_URL`
- `EMBEDDING_MODEL`
- `QDRANT_TEXT_KEYS`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL_NAME`
- `EVALUATOR_LLM_BASE_URL`
- `EVALUATOR_LLM_API_KEY`
- `EVALUATOR_LLM_MODEL_NAME`
- `OPENAI_API_KEY`
- `EVAL_METRIC_TIMEOUT`

## Script Order

### 1. Normalize Source Data

Script:

- `02_rename_columns.py`

Purpose:

- standardize source columns into the evaluation CSV structure.

Typical output:

- `02_data_evaluation.csv`

Run:

```bash
uv run python 02_rename_columns.py
```

## 2. Fill Retrieved Context from Qdrant

Script:

- `01b_fill_context_qdrant.py`

Purpose:

- embed each question,
- search Qdrant,
- rerank candidates using:
  - filename match,
  - title hint match,
  - keyword overlap,
- write the top contexts back into the dataset.

Important behavior:

- supports `Question`, `Pertanyaan`, or `question`,
- supports `Context` or `context`,
- joins selected chunks with `\n\n---\n\n`,
- skips rows where context is already present,
- can test retrieval with `--test-query`.

Useful arguments:

- `--input`
- `--output`
- `--topk`
- `--prefilter-k`
- `--min-chars`
- `--test-query`

Recommended run:

```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=my_documents \
uv run python 01b_fill_context_qdrant.py \
  --input 02_data_evaluation.csv \
  --output 02_data_evaluation_with_context.csv \
  --prefilter-k 1500 \
  --topk 4
```

Quick retrieval smoke test:

```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=my_documents \
uv run python 01b_fill_context_qdrant.py \
  --prefilter-k 1500 \
  --topk 4 \
  --test-query "Provide an academic summary of the abstract from the article A Cloud-Assisted Anonymous and Privacy-Preserving Authentication Scheme for Internet of Medical Things!"
```

## 3. Generate Answers

Script:

- `03_generate_answers.py`

Purpose:

- generate `generated_answer` from `question + context`.

Current generator policy:

- prefer retrieved context,
- if the context is weak or irrelevant, fall back to general knowledge,
- do not refuse to answer,
- do not emit `The answer is not available in the provided context.`

Useful arguments:

- `--input`
- `--output`
- `--start`
- `--limit`
- `--concurrency`

Recommended full run:

```bash
uv run python 03_generate_answers.py \
  --input 02_data_evaluation_with_context.csv \
  --output 03_with_answers.csv \
  --concurrency 5
```

Notes:

- concurrency improves throughput significantly,
- this step is usually much faster than evaluation,
- if needed, it can be resumed on subsets with `--start` and `--limit`.

## 4. Evaluate with RAGAS

Script:

- `04_eval.py`

Purpose:

- compute evaluation metrics for each row.

Primary metrics currently used:

- `context_precision`
- `context_recall`
- `context_entity_recall`
- `faithfulness`
- `answer_relevancy`

Optional but unstable:

- `noise_sensitivity`

Current behavior:

- supports resume by reusing an existing output CSV,
- skips metrics already present,
- writes progress row-by-row,
- supports subset reruns,
- supports per-metric timeout via `EVAL_METRIC_TIMEOUT`,
- supports metric selection via `--metrics`,
- supports row-level concurrency via `--concurrency`.

Important implementation details:

- evaluator LLM is configured from `EVALUATOR_*` env vars,
- `answer_relevancy` uses OpenAI embeddings (`text-embedding-3-small`),
- context strings are split on `\n\n---\n\n`.

Recommended full run:

```bash
EVAL_METRIC_TIMEOUT=120 \
uv run python 04_eval.py \
  --input 03_with_answers.csv \
  --output 04_evaluated.csv \
  --concurrency 2 \
  --metrics context_precision,context_recall,context_entity_recall,faithfulness,answer_relevancy
```

Why `noise_sensitivity` is excluded:

- it was the most frequent timeout source,
- it blocked long runs without adding enough practical value,
- final reporting currently excludes it.

Resume example:

```bash
EVAL_METRIC_TIMEOUT=120 \
uv run python 04_eval.py \
  --input 03_with_answers.csv \
  --output 04_evaluated.csv \
  --concurrency 1 \
  --metrics context_precision,context_recall,context_entity_recall,faithfulness,answer_relevancy
```

Subset rerun example:

```bash
EVAL_METRIC_TIMEOUT=120 \
uv run python 04_eval.py \
  --input 03_with_answers.csv \
  --output 04_evaluated_subset.csv \
  --start 0 \
  --limit 10 \
  --concurrency 1 \
  --metrics context_precision,context_recall,context_entity_recall,faithfulness,answer_relevancy
```

## 5. Export Final Clean Excel

Script:

- `05_export_clean_eval.py`

Purpose:

- export a clean Excel workbook for reporting.

Current exported columns:

- `question`
- `context`
- `ground_truth_answer`
- `generated_answer`
- `context_precision`
- `context_recall`
- `context_entity_recall`
- `faithfulness`
- `answer_relevancy`
- `status`

Workbook sheets:

- `evaluation`
- `summary`

Behavior:

- normalizes legacy column names,
- converts metric strings into numeric values,
- marks each row as:
  - `complete`
  - `partial`
- excludes `noise_sensitivity` from the final clean report.

Run:

```bash
uv run python 05_export_clean_eval.py \
  --input 04_evaluated.csv \
  --output 05_evaluated_clean.xlsx
```

## Recommended Production Flow

For a fresh real run with many rows:

1. Make sure all PDFs are already ingested into the correct Qdrant collection.
2. Replace or update `data_awal.xlsx`.
3. Run normalization.
4. Run context filling.
5. Inspect 5 to 10 sampled rows manually for retrieval quality.
6. Run answer generation.
7. Inspect a few generated answers manually.
8. Run evaluation without `noise_sensitivity`.
9. Export clean Excel.

## Recommended Validation Checkpoints

Before running the expensive evaluation stage, verify:

1. the target Qdrant collection is correct,
2. the retrieved `context` actually belongs to the paper named in the question,
3. `ground_truth_answer` matches the same paper,
4. the generator is not outputting refusal-style answers,
5. the sampled rows look semantically aligned.

## Known Failure Modes

### 1. Wrong Qdrant Collection

Symptom:

- retrieval returns chunks from unrelated papers.

Fix:

- verify `QDRANT_COLLECTION`,
- run retrieval with `--test-query`.

### 2. Data Exists but Retrieval Is Still Off

Symptom:

- top chunks come from unrelated documents even though ingestion completed.

Fix:

- increase `--prefilter-k`,
- verify `file_name` payload quality in Qdrant,
- check whether the target paper is actually in the collection,
- inspect question phrasing and title hint extraction.

### 3. Ground Truth Mismatch

Symptom:

- generated answer and context look reasonable,
- metrics are still near zero or unstable.

Fix:

- audit `ground_truth_answer`,
- confirm it belongs to the same paper as the question.

### 4. Evaluator Timeouts

Symptom:

- metrics remain blank,
- progress slows dramatically,
- long-running rows hang.

Fix:

- set `EVAL_METRIC_TIMEOUT`,
- rerun only the missing metrics,
- use low evaluator concurrency,
- exclude `noise_sensitivity`.

### 5. Fallback Answers Improve Usability but Hurt Strict RAG Metrics

Symptom:

- generated answers read better,
- `faithfulness` or `context_precision` may become harder to satisfy.

Reason:

- the generator can now use general knowledge when retrieval is weak.

Interpretation:

- this is good for answer usability,
- but it is no longer a pure context-only RAG evaluation.

## Files Worth Keeping After a Run

Suggested minimum retained artifacts:

- final evaluated CSV
- final ready CSV
- final ready XLSX

In this repo, the most presentation-ready output is:

- `05_evaluated_real_ready.xlsx`

## Suggested Commands for a New Large Run

```bash
uv run python 02_rename_columns.py
```

```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_COLLECTION=my_documents \
uv run python 01b_fill_context_qdrant.py \
  --input 02_data_evaluation.csv \
  --output 02_data_evaluation_with_context.csv \
  --prefilter-k 1500 \
  --topk 4
```

```bash
uv run python 03_generate_answers.py \
  --input 02_data_evaluation_with_context.csv \
  --output 03_with_answers.csv \
  --concurrency 5
```

```bash
EVAL_METRIC_TIMEOUT=120 \
uv run python 04_eval.py \
  --input 03_with_answers.csv \
  --output 04_evaluated.csv \
  --concurrency 2 \
  --metrics context_precision,context_recall,context_entity_recall,faithfulness,answer_relevancy
```

```bash
uv run python 05_export_clean_eval.py \
  --input 04_evaluated.csv \
  --output 05_evaluated_clean.xlsx
```

## Final Recommendation

For future runs with 100 or more rows:

- keep the same pipeline,
- sample-check retrieval before evaluation,
- exclude `noise_sensitivity`,
- treat evaluation as reliable only when question, context, and ground truth all point to the same paper.
