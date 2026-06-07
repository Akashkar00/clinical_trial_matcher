# Changes Log

**Session date:** 2026-06-05
**Project rating:** 6.5 / 10 → 8.0 / 10

Snapshot of every code change made during the review-and-fix session for the Clinical Trial Matcher project.

---

## Summary

| # | Issue | Files touched | Status |
|---|---|---|---|
| 1 | Broken `requirements.txt` (4 of ~12 deps listed) | `requirements.txt` | Fixed |
| 2 | Dead Gemini imports + unused `google-genai` dep | `config.py`, `pipeline/nodes.py`, `requirements.txt` | Fixed |
| 3 | Re-embedded all trials on every run | `rag/ingest.py` | Fixed |
| 4 | Sequential scoring with `time.sleep(3)` between calls | `pipeline/nodes.py` | Fixed |
| 5 | Retrieval ignored exclusion criteria | `rag/retrieve.py`, `pipeline/nodes.py`, `pipeline/prompts.py` | Fixed |
| 6 | `to_condition_query()` returned only `diagnosis` | `models/patient_profile.py`, `trials/fetcher.py`, `pipeline/nodes.py` | Fixed |
| 7 | Empty test files | `tests/conftest.py`, `tests/test_extraction.py`, `tests/test_retrieval.py`, `tests/test_scoring.py` | 35 tests added |
| 8 | Prompts scattered across 3 files | `pipeline/prompts.py` (new), 3 call sites | Fixed |
| 9 | No chain-of-thought in scoring | `pipeline/prompts.py`, `pipeline/nodes.py`, `tests/test_scoring.py` | Fixed |
| 10 | Hardcoded India filter in eval | `eval/evaluate.py` | Fixed |
| 11 | No observability | `config.py`, `extraction/extractor.py`, `pipeline/nodes.py`, `rag/retrieve.py` | LangSmith wired in |

All 35 tests pass after every change.

---

## Detailed changes

### 1. Fixed `requirements.txt`

**Before** — 4 of the actual deps listed:
```
pymupdf, python-dotenv, pydantic, google-genai
```

**After** — all 12 deps the project imports, version-pinned to what was already in the venv:
```
python-dotenv==1.2.2
pydantic==2.13.4
requests==2.34.2
pymupdf==1.27.2.3
groq==1.4.0
langgraph==1.2.4
langchain-core==1.4.0
qdrant-client==1.18.0
sentence-transformers==5.5.1
streamlit==1.58.0
pytest==8.3.3
```

**Why:** A fresh clone could not bootstrap. `langgraph`, `qdrant-client`, `sentence-transformers`, `groq`, `streamlit`, `requests`, `langchain-core` were all missing.

---

### 2. Removed dead Gemini code

**`config.py`** — removed `GOOGLE_API_KEY`, `GEMINI_MODEL`, and the validation that raised when the key was unset.

**`pipeline/nodes.py`** — removed `from google import genai` and the unused `client = genai.Client(api_key=GOOGLE_API_KEY)` initialization.

**`requirements.txt`** — removed `google-genai==2.7.0`.

**Why:** Every actual LLM call in the project goes through Groq (`llama-3.3-70b-versatile`). The Gemini imports were never called, but the env-var check forced anyone running the project to set a `GOOGLE_API_KEY` they would never use.

---

### 3. Cache-aware ingestion

**`rag/ingest.py`** — added `get_cached_nct_ids()` that scrolls the Qdrant collection and returns the set of `nct_id`s already embedded. `ingest_trials()` now diffs incoming trials against that set and only embeds the new ones.

```python
cached_nct_ids = get_cached_nct_ids()
new_trials = [t for t in trials if t.nct_id not in cached_nct_ids]
```

**Why:** A 30-trial fetch was producing ~hundreds of `embedder.encode()` calls every run. Now: first run embeds everything; subsequent overlapping runs skip cached trials entirely.

---

### 4. Concurrent scoring

**`pipeline/nodes.py`** — replaced the sequential loop with `ThreadPoolExecutor`:

```python
SCORE_CONCURRENCY = 4
...
with ThreadPoolExecutor(max_workers=workers) as pool:
    for result in pool.map(_score_one, unique_chunks):
        scored.append(result)
```

Removed the hardcoded `time.sleep(3)` between trials.

**Why:** 8 trials × 3s sleep + sequential ~2s LLM calls = ~40s wall-clock. Concurrent at 4 workers ≈ 6-8s. The internal retry in `_score_with_retry` already handles 429s, so the global sleep was redundant.

---

### 5. Exclusion criteria fed to scorer

**`rag/retrieve.py`** — added `get_exclusion_text(client, nct_id)` that scrolls all exclusion chunks for a trial and joins them. `retrieve_trials` now attaches `exclusion_text` to each top-K result after reranking.

**`pipeline/prompts.py`** — `SCORE_PROMPT` restructured to feed both inclusion (semantic-matched) AND the trial's full exclusion list, with an explicit instruction: *"if any exclusion clearly applies, the result is NO regardless of inclusion match."*

**Why:** Exclusion criteria are typically the actual reason a patient doesn't qualify. Vector-searching exclusions is wrong (a patient is excluded if *any one* applies — top-K would miss the deciding clause), so we fetch the full list per trial and put it in the prompt.

---

### 6. Richer ClinicalTrials.gov queries

**`models/patient_profile.py`** — `to_condition_query()` now returns `diagnosis + stage + primary biomarker`. Added `to_term_query()` for secondary biomarkers + prior treatments.

**`trials/fetcher.py`** — `fetch_trials()` accepts a `term` arg that maps to `query.term` (intervention/keyword field).

**`pipeline/nodes.py`** — `fetch_node` passes both queries; if the rich query returns zero trials, falls back to bare `diagnosis`.

**Before:** `query.cond=lung adenocarcinoma`
**After:** `query.cond=lung adenocarcinoma stage IV EGFR L858R` + `query.term=carboplatin pemetrexed`

**Why:** ClinicalTrials.gov's `query.cond` matches the trial's *condition* field (the disease) and `query.term` matches interventions/keywords. Stuffing prior treatments into `query.cond` was over-narrowing; splitting across the two parameters surfaces trials that match disease subtype and prior-line context.

---

### 7. 35 unit tests

**`tests/conftest.py`** (new) — bootstraps the test environment:
- Stubs `QdrantClient` and `SentenceTransformer` at import time (tests don't need the on-disk Qdrant lock or the 400MB PubMedBERT model)
- Seeds `GROQ_API_KEY` with a dummy value (config.py would otherwise raise in CI)
- Adds project root to `sys.path`

**`tests/test_extraction.py`** (6 tests) — `extract_patient_profile`:
valid JSON, ```` ```json ```` fence, bare ```` ``` ```` fence, malformed JSON, short text, validation failure.

**`tests/test_retrieval.py`** (20 tests) — `parse_age`, `is_eligible`, `chunk_criteria`:
years/months/weeks/days, missing unit, None/empty/garbage; gender match/mismatch/ALL, age below/above/no-bounds; numbered lists, bullet lists, tiny-item filtering, unstructured prose, size cap.

**`tests/test_scoring.py`** (9 tests) — `_score_with_retry`:
valid JSON, fence stripping, retry-then-success, all-retries-fail fallback, malformed JSON fallback, plus 3 CoT-format tests (REASONING + fenced JSON, bare unfenced JSON, decoy JSON in reasoning).

All mocked — no real LLM or Qdrant calls. Run time: ~0.4s.

---

### 8. Centralized prompt registry

**`pipeline/prompts.py`** (new, dependency-free) — single source of truth for every LLM prompt. Convention: `<STAGE>_PROMPT_V<N>` keeps old versions reproducible for eval baselines; active aliases at the bottom point to the current version:

```python
EXTRACTION_PROMPT_V1 = "..."
SCORE_PROMPT_V1 = "..."
SCORE_PROMPT_V2 = "..."  # CoT version
RERANK_PROMPT_V1 = "..."

EXTRACTION_PROMPT = EXTRACTION_PROMPT_V1
SCORE_PROMPT = SCORE_PROMPT_V2  # flip this line to roll out a new version
RERANK_PROMPT = RERANK_PROMPT_V1
```

**Three call sites updated** to import from this module:
- `extraction/extractor.py`
- `pipeline/nodes.py`
- `rag/retrieve.py`

**Why:** Auditing every prompt the system uses is now `cat pipeline/prompts.py`. A/B testing a new scoring prompt is two lines: add `SCORE_PROMPT_V2`, flip the alias.

---

### 9. Chain-of-thought scoring

**`pipeline/prompts.py`** — added `SCORE_PROMPT_V2`:

```
REASONING:
1. Exclusions: walk each exclusion criterion → applies / does not apply / cannot determine
2. Inclusions: walk the matched inclusion criteria
3. Verdict: combine into MATCH / PARTIAL / NO

JSON:
```json
{"match_type": ..., "score": ..., "reason": ...}
```
```

Score bands tied to verdict: MATCH (0.75-1.0), PARTIAL (0.4-0.75), NO (0.0-0.4) — keeps the model honest about borderline calls.

**`pipeline/nodes.py`** — added `_extract_json(raw)` that prefers a ```` ```json ```` fence, falls back to the *last* `{...}` block in the response (so a decoy JSON inside the reasoning text is not mistaken for the verdict). Bumped `max_tokens: 300 → 800` to fit reasoning + JSON.

**Why:** Forces the model to walk exclusions one-by-one before deciding. Visible reasoning helps debugging — a clinician can see *why* the model decided what it did, not just the one-sentence reason.

---

### 10. `--country` parameter for eval

**`eval/evaluate.py`** — removed the hardcoded India filter. New optional `--country <name>` CLI flag (case-insensitive) produces an extra metrics row per patient and extra summary-table columns when provided.

```bash
python -m eval.evaluate --max-patients 5                  # global only
python -m eval.evaluate --max-patients 5 --country India  # global + India breakdown
```

Refactored the duplicated MATCH/PARTIAL/NO/avg-score arithmetic into `_compute_metrics()` and the row formatting into `_fmt_metrics()`.

**Why:** Geographic constraints are real product logic (a Mumbai patient can't enroll in a Boston-only trial). But hardcoding "India" makes the eval dishonest as a general benchmark. As a parameter it's a deliberate slice of the metrics, opt-in per run.

---

### 11. LangSmith tracing

**`config.py`** — optional env-var setup:

```python
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "clinical-trial-matcher")
```

When `LANGSMITH_API_KEY` is unset, the decorators below are silent no-ops with zero overhead.

**Three `@traceable` decorators on the LLM hot paths**:
- `extraction/extractor.py` → `extract_patient_profile` (`stage: extraction`)
- `pipeline/nodes.py` → `_score_with_retry` (`stage: scoring`)
- `rag/retrieve.py` → `rerank_results` (`stage: retrieval`)

LangGraph nodes (`extract_node`, `fetch_node`, `ingest_node`, `retrieve_node`, `score_node`) are auto-traced by the framework once `LANGSMITH_TRACING=true` is set.

**To activate** — add to `.env`:
```
LANGSMITH_API_KEY=lsv2_pt_...
```

Traces stream to https://smith.langchain.com under the `clinical-trial-matcher` project: full graph execution tree per run, every prompt + response + token count + latency, per-trial scoring with full CoT reasoning visible, failure stack traces.

---

## Files added

- `pipeline/prompts.py` — centralized prompt registry
- `tests/conftest.py` — test bootstrap
- `tests/test_extraction.py` — extraction tests (6)
- `tests/test_retrieval.py` — retrieval tests (20)
- `tests/test_scoring.py` — scoring tests (9)
- `CHANGES.md` — this file

## Files modified

- `requirements.txt`
- `config.py`
- `models/patient_profile.py`
- `trials/fetcher.py`
- `extraction/extractor.py`
- `rag/ingest.py`
- `rag/retrieve.py`
- `pipeline/nodes.py`
- `eval/evaluate.py`

## Files unchanged but verified

- `app.py` (Streamlit UI)
- `main.py` (CLI entry point)
- `extraction/pdf_loader.py`
- `pipeline/graph.py`
- `pipeline/state.py`
- `trials/models.py`
- `models/__init__.py`

---

## Remaining gaps (path to 9.0+)

Listed in priority order — none of these were addressed in this session:

1. **Labeled eval set** — current metrics report what the model said, not whether it was right. Need expert-annotated correct trial NCTIDs for precision@k.
2. **Structured output mode** — Groq supports `response_format={"type": "json_object"}`. Eliminates the regex-based JSON extraction entirely.
3. **Batched embeddings** — `embedder.encode()` is called per-chunk; batching would be 10-20× faster.
4. **Biomarker-aware `is_eligible`** — currently only filters by gender + age. Hard-filtering on biomarker compatibility before LLM scoring would cut cost and false positives.
5. **LLM response cache** — `(patient_query_hash, criteria_hash) → score` in Redis or SQLite for production cost control.
6. **Prompt injection defense** — PDF content flows directly into the extraction prompt with no sanitization.
7. **OCR fallback** — `page.get_text("text")` returns empty on scanned medical reports. `pytesseract` fallback would close a real reliability hole.
8. **Dockerfile + docker-compose** — Qdrant as a service instead of local on-disk would solve the lock contention you hit running Streamlit + tests concurrently.
9. **Streaming UI** — `st.status()` per pipeline stage would make the 30s+ first-run wait feel half as long.
