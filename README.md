# Clinical Trial Matcher

An AI-powered pipeline that matches patients to relevant clinical trials using LLM clinical reasoning, semantic retrieval, and live data from ClinicalTrials.gov. Built with LangGraph, Groq (Llama 3.3 70B), PubMedBERT embeddings, and Qdrant.

---

## Why this exists

Matching a patient to clinical trials is a manual chore: oncologists comb through ClinicalTrials.gov, read pages of inclusion/exclusion criteria per trial, and reject most based on a single line buried in the exclusions. This project automates the eligibility-filtering pass — extracting a structured patient profile from a medical PDF, fetching candidate trials, and scoring each with explicit reasoning over inclusion AND exclusion criteria. The output is a ranked, human-auditable list with the deciding criterion cited per trial.

---

## Architecture

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Patient  │──▶│ Extract  │──▶│  Fetch   │──▶│ Ingest   │──▶│ Retrieve │──▶ ┐
│  PDF     │   │  (Groq)  │   │ (CT.gov) │   │ (Qdrant) │   │ (cosine+ │   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   │  rerank) │   │
                                                            └──────────┘   │
                                                                           ▼
                                                                  ┌──────────────┐
                                                                  │    Score     │
                                                                  │ (LLM CoT,    │
                                                                  │  excl-aware) │
                                                                  └──────┬───────┘
                                                                         ▼
                                                                  ┌──────────────┐
                                                                  │   Ranked     │
                                                                  │   matches    │
                                                                  └──────────────┘
```

Each box is a LangGraph node with explicit error edges and (for `extract`) a retry loop. State is a single `PipelineState` `TypedDict`.

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Groq — Llama 3.3 70B (`llama-3.3-70b-versatile`) | Fast, cheap, JSON-mode capable, good clinical reasoning at this size |
| Embeddings | `pritamdeka/S-PubMedBert-MS-MARCO` (768-dim) | Domain-tuned medical embeddings beat generic ones for clinical vocab |
| Vector DB | Qdrant (local disk OR remote via `QDRANT_URL`) | Fast cosine search, payload filtering by criteria_type/nct_id |
| Orchestration | LangGraph (5-node state graph) | Deterministic flow, conditional retry edges, framework-traced |
| Tracing | LangSmith (`@traceable` decorators) | Zero-overhead when unset; full trace tree when configured |
| Validation | Pydantic v2 | Schema enforcement at every system boundary |
| Trial source | ClinicalTrials.gov API v2 | Authoritative, free, two-parameter query (`cond` + `term`) |

---

## Project structure

```
clinical_trial_matcher/
├── main.py                    # CLI entry point + cost report
├── app.py                     # Streamlit UI
├── config.py                  # env loading, logging, Qdrant URL
├── Dockerfile                 # Streamlit container
├── docker-compose.yml         # app + Qdrant service
├── requirements.txt
├── .env.example               # template — copy to .env and fill in
│
├── pipeline/
│   ├── state.py               # PipelineState TypedDict
│   ├── graph.py               # LangGraph wiring (5 nodes + error edges)
│   ├── nodes.py               # extract / fetch / ingest / retrieve / score
│   └── prompts.py             # versioned prompt registry (V1/V2/V3 + alias)
│
├── extraction/
│   ├── pdf_loader.py          # PyMuPDF → cleaned text
│   ├── sanitizer.py           # prompt-injection defense
│   └── extractor.py           # Groq → PatientProfile (sanitized + traced)
│
├── models/
│   └── patient_profile.py     # Pydantic schema + query builders
│
├── trials/
│   ├── fetcher.py             # ClinicalTrials.gov API client
│   └── models.py              # ClinicalTrial / TrialLocation
│
├── rag/
│   ├── ingest.py              # batch embed → Qdrant (cache-aware, lazy-load)
│   └── retrieve.py            # cosine search + biomarker filter + concurrent rerank
│
├── observability/
│   └── __init__.py            # CostTracker (tokens, latency, USD per stage)
│
├── eval/
│   ├── synthetic_patients/    # 20 JSON profiles (no PII, generated)
│   ├── ground_truth.json      # labeled (patient, trial, verdict) pairs
│   ├── metrics.py             # recall@k, MRR, precision/recall/F1, confusion
│   ├── run_eval.py            # CLI: full eval / dump-trials / compare-prompts
│   └── evaluate.py            # legacy distribution-only metrics
│
└── tests/
    ├── conftest.py            # stubs Qdrant + SentenceTransformer at import
    ├── test_extraction.py     # 6 tests — JSON parsing, validation
    ├── test_retrieval.py      # 20 tests — age/gender filtering, chunking
    ├── test_scoring.py        # 9 tests — fence stripping, CoT extraction, retry
    ├── test_sanitizer.py      # 9 tests — injection patterns, truncation
    ├── test_biomarker_filter.py  # 6 tests — HER2/EGFR conflict detection
    ├── test_observability.py  # 6 tests — cost tracker, aggregation
    └── test_metrics.py        # 12 tests — eval metrics
```

---

## Setup

### Local (venv)

```bash
git clone <repo>
cd clinical_trial_matcher

python -m venv clinical_trial
source clinical_trial/bin/activate     # Windows: clinical_trial\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: GROQ_API_KEY is required, others optional
```

### Docker

```bash
cp .env.example .env
docker compose up --build
# Streamlit at http://localhost:8501
# Qdrant at http://localhost:6333
```

The compose file runs Qdrant as a service with a healthcheck — no more local file-lock contention between Streamlit and tests.

### Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | yes | — | LLM extraction + scoring + reranking |
| `LANGSMITH_API_KEY` | no | — | Enables tracing; `@traceable` becomes no-op when unset |
| `LANGSMITH_PROJECT` | no | `clinical-trial-matcher` | LangSmith project name |
| `QDRANT_URL` | no | — | Remote Qdrant URL; falls back to on-disk when unset |
| `QDRANT_PATH` | no | `./qdrant_storage` | On-disk Qdrant location |
| `EMBEDDING_MODEL` | no | `pritamdeka/S-PubMedBert-MS-MARCO` | Override for HF embedding model |
| `LOG_LEVEL` | no | `INFO` | Root logger level |

---

## Usage

### CLI

```bash
python main.py
# Runs the pipeline on tests/patient_3.pdf and prints a ranked verdict
# table plus a token / latency / USD report per stage.
```

### Streamlit

```bash
source clinical_trial/bin/activate
streamlit run app.py
# Or: docker compose up
```

Then open `http://localhost:8501` in a browser, upload a patient PDF, and click **Find Matching Trials**. The UI renders the extracted profile in the sidebar and MATCH / PARTIAL / NO trial cards with reasoning and a "view on ClinicalTrials.gov" link.

#### Upload format — what to give the app

- **File type:** `.pdf` only (single file, ≤30k characters of text after extraction; longer reports are truncated).
- **Source:** text-based PDF. Scanned/image-only PDFs return empty text from PyMuPDF and will fail extraction — OCR fallback is on the roadmap.
- **No PII required.** The app does not need a real name, MRN, or DOB. Strip them before upload if you can.

The extractor pulls these fields (everything optional except age, gender, diagnosis):

| Field | Type | Notes |
|---|---|---|
| `age` | integer 0–120 | Required. Drives the demographic eligibility filter. |
| `gender` | male / female / other / unknown | Required. Drives the demographic eligibility filter. |
| `diagnosis` | string | Required. Primary disease — e.g. `"Stage IV non-small cell lung cancer"`. |
| `stage` | string or null | e.g. `"IV"`, `"IIIB"`, `"metastatic"`. |
| `biomarkers` | list of strings | e.g. `["EGFR L858R", "PD-L1 70%", "HER2 negative"]`. Drives the hard biomarker filter — be explicit about positive / negative / mutation state. |
| `prior_treatments` | list of strings | e.g. `["carboplatin/pemetrexed", "osimertinib"]`. |
| `current_status` | string or null | e.g. `"progressing on first-line therapy"`. |
| `location` | string or null | City or country. |
| `ecog_status` | integer 0–4 | Performance score. |
| `comorbidities` | list of strings | e.g. `["controlled hypertension", "type 2 diabetes"]`. |
| `key_labs` | dict or null | e.g. `{"hemoglobin": "11.2 g/dL", "ANC": 1800}`. |

**Minimum useful report.** A one-page PDF with this much will already produce ranked results:

```
Patient: 62-year-old female
Diagnosis: Stage IV non-small cell lung cancer (adenocarcinoma)
Biomarkers: EGFR exon 19 deletion, PD-L1 30%, HER2 negative
Prior treatments: osimertinib (progressed after 14 months), carboplatin/pemetrexed
ECOG: 1
Comorbidities: controlled hypertension
Current status: progression on first-line targeted therapy, seeking second-line trials
Location: Boston, MA
```

The richer the biomarker and prior-treatment lines, the better the match — those are what the retriever and scorer key off. Vague reports ("lung cancer, on chemo") will still run but produce mostly PARTIAL verdicts.

Sample PDFs that work end-to-end live in `tests/patient_1.pdf` … `patient_9.pdf`.

### Evaluation

```bash
# Isolated scoring eval — production scorer over labeled pairs, retrieval-
# independent, rate-limit-safe (paced). This is the source of the numbers below.
python -m eval.score_eval --patients 1,2,3,5,8,11,13,15 --pace 15

# Full end-to-end eval (retrieval recall/MRR + scoring) against ground_truth.json
python -m eval.run_eval
python -m eval.run_eval --patients 1,2,3,5,8   # restrict to specific patients
python -m eval.run_eval --max-patients 5        # first N patients (smoke test)

# Annotation helper — list trials currently retrieved for a patient
python -m eval.run_eval --dump-trials 3

# A/B test SCORE_PROMPT_V2 vs V3 against ground truth
python -m eval.run_eval --compare-prompts
```

> **Rate limits.** On the Groq free tier (~30 req/min, ~12k tokens/min, daily
> cap) the concurrent rerank + score pools blow the budget and the scorer falls
> back to `PARTIAL@0.0`. Set `CT_SCORE_CONCURRENCY=1 CT_RERANK_CONCURRENCY=1` and
> use `eval.score_eval --pace 15` for clean, reproducible eval runs; the eval now
> **excludes** these fallbacks (`scoring_ok=False`) from metrics rather than
> counting them as verdicts.

### Tests

```bash
python -m pytest tests/ -q          # 68 tests, ~0.5s
```

---

## Pipeline details

### 1. Extract — `extraction/extractor.py`

Sanitize the PDF text, then prompt Groq for a structured `PatientProfile` (Pydantic, age 0–120, ECOG 0–4, etc).

- **PDF text → sanitizer.py** — length cap (30k chars), injection-pattern scan, delimited-block wrapping.
- **Groq call** wrapped with `track_call("extraction", ...)` for cost telemetry.
- **JSON-mode + tolerant `_extract_json`** — fenced/bare/last-balanced-block fallback chain.
- **Pydantic validation** rejects malformed extractions.

### 2. Fetch — `trials/fetcher.py`

Two-parameter ClinicalTrials.gov query:
- `query.cond` = diagnosis + stage + primary biomarker
- `query.term` = secondary biomarkers + prior treatments

Falls back to bare diagnosis if the rich query returns zero (over-narrowing protection).

### 3. Ingest — `rag/ingest.py`

- **Cache-aware**: `get_cached_nct_ids()` scrolls existing chunks, skips re-embedding trials we already have.
- **Batched**: `embedder.encode(texts, batch_size=32)` rather than per-chunk calls.
- **Lazy-loaded** PubMedBERT model — paid only on first encode, not at import.
- Stores inclusion AND exclusion chunks separately (criteria_type payload).

### 4. Retrieve — `rag/retrieve.py`

1. Cosine search over **inclusion-only** chunks (top_k * 4 candidates).
2. **Demographic filter** — age + gender hard-filter.
3. **Biomarker filter** — `_trial_biomarker_conflict` rejects HER2+/HER2− style mismatches before LLM scoring (cost saver + correctness — LLMs are unreliable on exact-token logic).
4. **Concurrent rerank** — `ThreadPoolExecutor(8)` for the LLM relevance pass; combined score = 0.4·cosine + 0.6·rerank.
5. **Attach exclusion text** — for each surviving trial, scroll its full exclusion block from Qdrant. Vector-searching exclusions is wrong (a patient is excluded if *any one* applies, not the top-k); fetching the whole block and feeding it to the scorer is the right shape.

### 5. Score — `pipeline/nodes.py`

Concurrent (`ThreadPoolExecutor(4)`), CoT-prompted Groq calls per unique trial. The active prompt is `SCORE_PROMPT_V3`:

```
Reasoning (in JSON):
  exclusions:        walk each — applies / does not apply / cannot determine
  inclusions:        walk matched — satisfied / failed / missing info
  verdict_rationale: combine into final call

Verdict:
  match_type: MATCH | PARTIAL | NO
  score:      MATCH 0.75-1.0, PARTIAL 0.4-0.75, NO 0.0-0.4
  reason:     one sentence citing the deciding criterion
```

Retries on 429/timeout; falls back to `PARTIAL @ 0.0` if all retries fail rather than raising.

---

## Security

| Concern | Defense | File |
|---|---|---|
| Prompt injection in PDFs | Length cap + keyword scan + delimited-block wrapping | `extraction/sanitizer.py` |
| Secrets in repo | `.env` gitignored, `.env.example` template | `.gitignore`, `.env.example` |
| Output validation | Pydantic schemas at every boundary | `models/`, `trials/models.py` |
| Hostile model output | Tolerant JSON extraction with fallbacks | `_extract_json` in `extractor.py`, `nodes.py` |

The sanitizer **flags but does not reject** injection signals — a real medical report can legitimately contain words like "system" or "instructions". Hits are logged at WARN level for audit, with the original length and truncation status in the same log line.

---

## Observability

```
─────────────────────────────────────────────────
Stage           Calls      In     Out      ms        USD
─────────────────────────────────────────────────
extraction          1    1842     287    1240   $  0.0014
rerank             32    9600    1600    4100   $  0.0070
scoring             8   12480    4800    8200   $  0.0111
─────────────────────────────────────────────────
TOTAL              41                            $  0.0195
```

Every Groq call is wrapped in `track_call(stage, model)`. The `CostTracker` is thread-safe (matters because both rerank and scoring use thread pools), aggregates by stage, and reports tokens / latency / estimated USD at end of run.

LangSmith tracing is zero-overhead when `LANGSMITH_API_KEY` is unset. With it set, every node + every `@traceable` LLM call streams to `https://smith.langchain.com` under the configured project.

Logs use stdlib `logging` with a structured-ish format (`node.score.start`, `retrieve.filter_summary candidates=32 kept=18 ...`) for grep-ability.

---

## Evaluation framework

Two surfaces, reported separately so a regression in one isn't masked by stability in the other.

### Retrieval

For each patient, `eval/ground_truth.json` lists known-relevant NCT ids. We measure:

- **recall@K** — fraction of relevant trials that survived to the top-K post-filter, post-rerank.
- **MRR** — mean reciprocal rank of the first relevant hit.

A retrieval miss is the retriever's fault; we don't punish the scorer for it.

### Scoring

For each (patient, trial) pair the labelers reviewed AND that survived retrieval, compare the LLM verdict (MATCH/PARTIAL/NO) against the human verdict. Reports:

- **Per-class precision / recall / F1**
- **3×3 confusion matrix**
- **Overall accuracy**

### Ground-truth status

`eval/ground_truth.json` ships with **60 criteria-derived labels** across 8 patients (diverse tumor types: HER2+ breast, EGFR+ NSCLC, FLT3-ITD AML, KRAS/MSS CRC, BRAF+ melanoma, BRCA1+ ovarian, HR+/HER2- breast, driver-negative NSCLC). Each `(patient, trial)` pair was labeled by reading the trial's **actual ClinicalTrials.gov eligibility criteria** and applying deterministic rules — tumor type, biomarker/receptor state, disease stage, prior-therapy exclusions, demographic bounds. This is **criteria-derived, not clinician-reviewed** (`annotator: criteria-derived-v1`); judgment calls needing true oncology expertise were labeled PARTIAL rather than guessed. Verdict distribution: **1 MATCH / 20 PARTIAL / 39 NO**. See the `_meta` block for the full protocol and the honesty caveats.

To extend or upgrade the labels:

1. `python -m eval.run_eval --dump-trials <N>` lists what's currently retrieved.
2. Read each trial's criteria against the patient profile; assign MATCH / PARTIAL / NO with a one-sentence rationale citing the deciding criterion.
3. For a real recall study, add independently-sourced known-eligible NCTIDs per patient (may NOT appear in current retrieval — that's how recall gets measured).
4. For clinical use, have an oncologist review and overwrite `annotator` with their id.

### Measured results (2026-07-10)

Run with the **isolated scorer** (`python -m eval.score_eval`), which drives the production scoring call (`SCORE_PROMPT_V3`, `llama-3.3-70b-versatile`, `temperature=0.1`) directly over the labeled pairs — decoupled from retrieval so the metric reflects the scorer alone. Rate-limit fallbacks (`scoring_ok=False`, i.e. `PARTIAL@0.0`) are **excluded**, not counted as verdicts. **54 of 60** pairs produced a real verdict (6 excluded as rate-limited on the Groq free tier). Full per-pair detail in [eval/results_2026-07-10.json](eval/results_2026-07-10.json).

```
accuracy = 0.833   (n = 54)

class        P       R      F1    support
MATCH      0.500   1.000   0.667      1
PARTIAL    0.909   0.556   0.690     18
NO         0.829   0.971   0.895     35

Confusion (rows = truth, cols = predicted):
             MATCH  PARTIAL   NO
MATCH            1        0    0
PARTIAL          1       10    7
NO               0        1   34

Binary — surface [MATCH|PARTIAL] vs reject [NO]:
P = 0.923   R = 0.632   F1 = 0.750   (tp=12 fp=1 fn=7 tn=34)
```

**What this says.** The scorer is **precise and conservative**: it rejects clear mismatches almost perfectly (NO recall 0.97) and rarely surfaces a trial it shouldn't (binary precision 0.92, one false surface in 35 true-NO). Its weakness is **over-rejection** — 7 of 18 borderline-eligible (PARTIAL) trials were scored NO, dragging PARTIAL recall to 0.56. For an eligibility pre-screen where a missed eligible trial is the costlier error, that recall gap is the number to improve next (candidate levers: soften the "any ambiguous exclusion ⇒ NO" instruction in `SCORE_PROMPT`, or add a distinct `UNCERTAIN` verdict so missing-data cases don't collapse into NO).

**Caveats, stated plainly.** Labels are criteria-derived, not clinician-reviewed. n=54 is small and MATCH support is 1, so the MATCH row is illustrative only. Retrieval recall is **not** reported as a clean number here: the labeled-relevant trials were drawn from what the retriever surfaced, so recall@k is bounded near 1.0 by construction — a real recall study needs independently-sourced eligible trials (noted in `_meta` as the next step). A separate observation from the run: re-running retrieval on the same patients surfaced a materially different trial set, so retrieval is **not run-to-run deterministic** — worth pinning before quoting retrieval numbers.

### Prompt A/B testing

```bash
python -m eval.run_eval --compare-prompts
```

Runs the eval twice — once with `SCORE_PROMPT_V2`, once with `V3` — and prints both metric blocks. Versioning convention: `<STAGE>_PROMPT_V<N>` constants stay forever (so old eval baselines remain reproducible); the alias at the bottom of `pipeline/prompts.py` selects the active version.

---

## Testing

68 tests, ~0.5s wall clock, all mocked — no real LLM or Qdrant calls.

```
test_extraction.py        6 tests   PDF → JSON parsing, validation, retry
test_retrieval.py        20 tests   age parsing, eligibility, chunking
test_scoring.py           9 tests   fence stripping, CoT extraction, retry
test_sanitizer.py         9 tests   injection patterns, truncation, dedup
test_biomarker_filter.py  6 tests   HER2/EGFR positive/negative conflicts
test_observability.py     6 tests   cost tracker, multi-stage aggregation
test_metrics.py          12 tests   recall@K, MRR, P/R/F1, confusion matrix
```

`tests/conftest.py` stubs `QdrantClient` and `SentenceTransformer` at import time so tests don't need the on-disk Qdrant lock or the 400MB PubMedBERT model.

---

## Roadmap

Done in this iteration:

- [x] Prompt-injection defense (sanitizer + delimited blocks)
- [x] Centralized prompt registry with versioning
- [x] CoT scoring with explicit exclusion-walk
- [x] Concurrent reranker (8 workers) and scorer (4 workers)
- [x] Hard biomarker filter before LLM scoring
- [x] Cost / token / latency telemetry per stage
- [x] LangSmith tracing wired in (zero-overhead when off)
- [x] Structured logging across the pipeline
- [x] Lazy-load embedding model
- [x] Configurable Qdrant URL (local-disk OR remote service)
- [x] Dockerfile + docker-compose with Qdrant as a service
- [x] Labeled ground-truth schema + retrieval & scoring metrics
- [x] 60 criteria-derived labels + **measured scoring results** (acc 0.833, n=54) — see [Measured results](#measured-results-2026-07-10)
- [x] Isolated, rate-limit-safe scoring eval (`eval/score_eval.py`) that excludes fallbacks from metrics
- [x] Env-configurable concurrency (`CT_SCORE_CONCURRENCY`, `CT_RERANK_CONCURRENCY`) for rate-limited tiers
- [x] Prompt-version A/B harness
- [x] 68-test suite with proper mocks

Open work:

- [ ] Raise PARTIAL recall (0.56) — the scorer over-rejects borderline-eligible trials; try softening the exclusion-forces-NO rule or adding an `UNCERTAIN` verdict
- [ ] Clinician review of `ground_truth.json` (currently criteria-derived, not clinician-reviewed)
- [ ] Real retrieval-recall study with independently-sourced eligible trials (current labels can't measure it without degeneracy)
- [ ] Make retrieval run-to-run deterministic (observed drift between runs on the same patients)
- [ ] OCR fallback for scanned PDFs (PyMuPDF returns empty on those — `pytesseract` would close the gap)
- [ ] Streaming UI updates per pipeline stage in Streamlit
- [ ] LLM response cache `(patient_hash, criteria_hash) → score` for production cost control
- [ ] Cross-encoder reranker (e.g., `ms-marco-MiniLM`) as an alternative to LLM rerank — higher quality at lower cost per call

---

## Reproducibility

Three knobs materially affect output — pin them when reporting numbers or comparing runs:

- LLM: Llama 3.3 70B via Groq (`llama-3.3-70b-versatile`)
- Embeddings: `pritamdeka/S-PubMedBert-MS-MARCO` (768-dim)
- Active prompt version: `SCORE_PROMPT_V3`

`temperature=0.1` is used at extract/score; for strict A/B reruns set it to `0.0` and pin `seed=` where supported, otherwise expect run-to-run variance on borderline trials.
