# pipeline/nodes.py

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from langsmith import traceable
from pipeline.state import PipelineState
from pipeline.prompts import SCORE_PROMPT
from extraction.pdf_loader import load_pdf
from extraction.extractor import extract_patient_profile
from trials.fetcher import fetch_trials
from rag.ingest import ingest_trials, client as qdrant_client, embedder as embedding_model
from rag.retrieve import retrieve_trials
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from observability import track_call

logger = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_API_KEY)

SCORE_CONCURRENCY = 4

# ── Node 1: Extract ──────────────────────────────────────
def extract_node(state: PipelineState) -> PipelineState:
    logger.info("node.extract.start")
    try:
        if state.get("patient_profile") is not None:
            logger.info("node.extract.skip reason=profile_already_provided")
            return {**state, "error": None}

        text = load_pdf(state["pdf_path"])

        # Retry loop for extraction
        profile = None
        for attempt in range(3):
            try:
                profile = extract_patient_profile(text)
                break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 2:
                    wait = (attempt + 1) * 15
                    logger.warning("node.extract.rate_limit attempt=%d wait_s=%d err=%s", attempt, wait, err_str[:80])
                    time.sleep(wait)
                else:
                    raise e

        return {
            **state,
            "raw_text": text,
            "patient_profile": profile,
            "error": None
        }
    except Exception as e:
        logger.error("node.extract.error err=%s", e)
        return {
            **state,
            "error": str(e),
            "retry_count": state.get("retry_count", 0) + 1
        }


# ── Node 2: Fetch ────────────────────────────────────────
def fetch_node(state: PipelineState) -> PipelineState:
    logger.info("node.fetch.start")
    try:
        profile = state["patient_profile"]
        cond = profile.to_condition_query()
        term = profile.to_term_query()

        trials = fetch_trials(query=cond, location=None, page_size=30, term=term)

        # Fall back to bare diagnosis if the rich query over-filtered to zero
        if not trials and (term or cond != profile.diagnosis):
            logger.info("node.fetch.fallback rich_query_returned_zero diagnosis=%s", profile.diagnosis)
            trials = fetch_trials(query=profile.diagnosis, location=None, page_size=30)

        logger.info("node.fetch.done count=%d", len(trials))
        return {**state, "fetched_trials": trials}
    except Exception as e:
        logger.error("node.fetch.error err=%s", e)
        return {**state, "error": str(e)}


# ── Node 3: Ingest ───────────────────────────────────────
def ingest_node(state: PipelineState) -> PipelineState:
    logger.info("node.ingest.start")
    try:
        trials = state["fetched_trials"]
        count = ingest_trials(trials)
        logger.info("node.ingest.done chunks_stored=%d", count)
        return {**state, "chunks_stored": count}
    except Exception as e:
        logger.error("node.ingest.error err=%s", e)
        return {**state, "error": str(e)}


# ── Node 4: Retrieve ─────────────────────────────────────
def retrieve_node(state: PipelineState) -> PipelineState:
    logger.info("node.retrieve.start")
    try:
        profile = state["patient_profile"]
        chunks = retrieve_trials(profile, qdrant_client, embedding_model, top_k=8)
        logger.info("node.retrieve.done chunks=%d", len(chunks))
        return {**state, "retrieved_chunks": chunks}
    except Exception as e:
        logger.error("node.retrieve.error err=%s", e)
        return {**state, "error": str(e)}


# ── Node 5: Score ────────────────────────────────────────
def score_node(state: PipelineState) -> PipelineState:
    logger.info("node.score.start")
    chunks = state["retrieved_chunks"]
    profile = state["patient_profile"]

    # Map nct_id to trial objects for quick location lookup
    trial_map = {}
    if state.get("fetched_trials"):
        trial_map = {t.nct_id: t for t in state["fetched_trials"] if t}

    seen = set()
    unique_chunks = []
    for c in chunks:
        if c["nct_id"] not in seen:
            seen.add(c["nct_id"])
            unique_chunks.append(c)

    def _score_one(chunk):
        trial_obj = trial_map.get(chunk["nct_id"])
        locations_data = (
            [loc.model_dump() for loc in trial_obj.locations] if trial_obj else []
        )
        result = _score_with_retry(chunk, profile)
        result["locations"] = locations_data
        return result

    scored = []
    workers = min(SCORE_CONCURRENCY, len(unique_chunks)) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_score_one, unique_chunks):
            scored.append(result)

    scored.sort(key=lambda x: x["score"], reverse=True)
    logger.info("node.score.done scored=%d", len(scored))
    return {**state, "scored_trials": scored}


@traceable(name="score_trial", run_type="llm", metadata={"stage": "scoring"})
def _score_with_retry(chunk: dict, profile, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            exclusion = chunk.get("exclusion_text") or "(none provided)"
            prompt = SCORE_PROMPT.format(
                patient=profile.model_dump_json(indent=2),
                criteria=chunk["text"],
                exclusion=exclusion,
                nct_id=chunk["nct_id"],
                title=chunk["title"]
            )

            with track_call("scoring", GROQ_MODEL) as ctx:
                ctx["response"] = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=800,
                    response_format={"type": "json_object"},
                )
            response = ctx["response"]

            raw = response.choices[0].message.content.strip()
            data = _extract_json(raw)
            return {
                "nct_id": chunk["nct_id"],
                "title": chunk["title"],
                "match_type": data.get("match_type", "PARTIAL"),
                "score": float(data.get("score", 0.5)),
                "reason": data.get("reason", ""),
                "phase": chunk.get("phase")
            }

        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 3
                logger.warning("score.retry attempt=%d wait_s=%d err=%s", attempt, wait, str(e)[:80])
                time.sleep(wait)
            else:
                logger.error("score.give_up nct_id=%s err=%s", chunk["nct_id"], str(e)[:80])
                return {
                    "nct_id": chunk["nct_id"],
                    "title": chunk["title"],
                    "match_type": "PARTIAL",
                    "score": 0.0,
                    "reason": f"Scoring failed: {str(e)[:80]}",
                    "phase": chunk.get("phase")
                }


def _extract_json(raw: str) -> dict:
    """Robust JSON extraction from a model response.

    Tries (in order):
      1. ```json ... ``` fenced block
      2. ```      ... ``` bare fence
      3. The LAST {...} block in the text (so a JSON example inside a
         REASONING preamble is not mistaken for the final verdict)
      4. Direct json.loads on the trimmed input
    Raises json.JSONDecodeError if every strategy fails.
    """
    import re
    text = raw.strip()

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    bare = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if bare:
        return json.loads(bare.group(1))

    # Walk the string and find balanced top-level JSON objects.
    candidates: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start:i + 1])
                    start = -1
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue

    return json.loads(text)