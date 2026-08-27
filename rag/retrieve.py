# rag/retrieve.py

import logging
import os
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langsmith import traceable
from models.patient_profile import PatientProfile
from observability import track_call  # noqa: F401 kept for future hooks

logger = logging.getLogger(__name__)

COLLECTION_NAME = "trials"


@lru_cache(maxsize=1)
def _get_flashranker():
    """
    Lazily load FlashRank reranker. Returns None if flashrank is not installed
    so the rest of retrieval still works (cosine-only fallback).
    """
    try:
        from flashrank import Ranker
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")
        logger.info("flashrank.loaded model=ms-marco-MiniLM-L-12-v2")
        return ranker
    except ImportError:
        logger.warning("flashrank not installed — falling back to cosine-only reranking. "
                       "Run: pip install flashrank")
        return None


def get_exclusion_text(client: QdrantClient, nct_id: str, max_chunks: int = 20) -> str:
    """Fetch all exclusion chunks for a trial and join them into a single block."""
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="nct_id", match=MatchValue(value=nct_id)),
                FieldCondition(key="criteria_type", match=MatchValue(value="exclusion")),
            ]
        ),
        limit=max_chunks,
        with_payload=["text"],
        with_vectors=False,
    )
    texts = [(p.payload or {}).get("text", "") for p in points]
    return "\n".join(t for t in texts if t)


def parse_age(age_str: str) -> float | None:
    """Parse ClinicalTrials age string to float years."""
    if not age_str or age_str.lower() in ("none", "n/a", "unknown"):
        return None

    parts = age_str.strip().split()
    try:
        val = float(parts[0])
        unit = parts[1].lower() if len(parts) > 1 else "years"
        if "month" in unit:
            return val / 12.0
        elif "week" in unit:
            return val / 52.0
        elif "day" in unit:
            return val / 365.0
        return val
    except (ValueError, IndexError):
        return None


# ── Hard biomarker filter ─────────────────────────────────
# Patterns that signal a trial REQUIRES a specific biomarker state. If the
# patient's biomarker state contradicts the trial chunk's stated requirement,
# we hard-reject before LLM scoring — both for cost and for correctness
# (LLMs are unreliable at exact-token-match logic).
_BIOMARKER_TOKENS = (
    "her2", "her-2", "egfr", "alk", "ros1", "kras", "braf", "pd-l1",
    "er", "pr", "brca1", "brca2", "msi", "tmb", "flt3", "idh1", "idh2",
)


def _patient_biomarker_state(profile: PatientProfile) -> dict[str, str]:
    """Map biomarker name → "positive"|"negative"|"mutated"|"unknown"."""
    state: dict[str, str] = {}
    for b in profile.biomarkers:
        text = b.lower()
        for token in _BIOMARKER_TOKENS:
            if token in text:
                if "negative" in text or "neg" in text or "wild" in text:
                    state[token] = "negative"
                elif (
                    "positive" in text or "pos" in text or "+" in text
                    or "mutat" in text or "amplif" in text or "express" in text
                ):
                    state[token] = "positive"
        # raw biomarker like "EGFR L858R" implies a mutation state
        if any(t in text for t in ("egfr", "alk", "kras", "braf", "flt3", "idh")):
            state.setdefault(text.split()[0], "positive")
    return state


def _trial_biomarker_conflict(patient_state: dict[str, str], chunk_text: str) -> str | None:
    """Return a human-readable reason if the trial chunk requires a biomarker
    state that conflicts with the patient. None means no conflict found
    (which is the safe default — the LLM still gets to decide)."""
    text = chunk_text.lower()
    for token, p_state in patient_state.items():
        if token not in text:
            continue
        # Look for "<token> negative/positive" within ~30 chars of the token
        idx = text.find(token)
        window = text[idx : idx + len(token) + 40]
        if p_state == "positive" and (
            "negative" in window or "wild-type" in window or "wild type" in window
        ):
            return f"trial requires {token.upper()} negative; patient is {token.upper()} positive"
        if p_state == "negative" and ("positive" in window and "negative" not in window[:window.find("positive")]):
            return f"trial requires {token.upper()} positive; patient is {token.upper()} negative"
    return None


def is_eligible(profile: PatientProfile, payload: dict) -> bool:
    """Basic demographic eligibility check (gender and age)."""
    # Gender check
    trial_gender = payload.get("gender")  # "MALE", "FEMALE", "ALL"
    patient_gender = profile.gender.value.lower()  # "male", "female"

    if trial_gender and trial_gender.lower() not in ("all", "all_active"):
        if trial_gender.lower() != patient_gender:
            return False

    # Age check
    min_age = parse_age(payload.get("minimum_age"))
    max_age = parse_age(payload.get("maximum_age"))

    if min_age is not None and profile.age < min_age:
        return False
    if max_age is not None and profile.age > max_age:
        return False

    return True


@traceable(name="rerank_results", run_type="chain", metadata={"stage": "retrieval"})
def rerank_results(results: list[dict], profile: PatientProfile) -> list[dict]:
    """
    Re-rank retrieved chunks using FlashRank (local cross-encoder).

    FlashRank runs entirely in-process — no API calls, no quota consumption.
    Falls back to cosine-only ranking if flashrank is not installed.
    Combined score: 0.4 * cosine + 0.6 * flashrank_score
    """
    ranker = _get_flashranker()
    patient_summary = profile.to_search_query()

    if ranker is None:
        # Cosine-only fallback
        for r in results:
            r["rerank_score"] = r["similarity_score"]
            r["combined_score"] = r["similarity_score"]
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results

    try:
        from flashrank import RerankRequest
        passages = [
            {"id": i, "text": r["text"][:512]}  # FlashRank max input
            for i, r in enumerate(results)
        ]
        rerank_req = RerankRequest(query=patient_summary, passages=passages)
        reranked = ranker.rerank(rerank_req)

        # Map index → FlashRank score
        score_map: dict[int, float] = {item["id"]: item["score"] for item in reranked}

        for i, r in enumerate(results):
            fr_score = score_map.get(i, 0.0)
            r["rerank_score"] = fr_score
            r["combined_score"] = 0.4 * r["similarity_score"] + 0.6 * fr_score

        logger.info("flashrank.reranked count=%d", len(results))

    except Exception as e:
        logger.warning("flashrank.error err=%s — falling back to cosine", str(e)[:80])
        for r in results:
            r["rerank_score"] = r["similarity_score"]
            r["combined_score"] = r["similarity_score"]

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results


def retrieve_trials(profile: PatientProfile, client: QdrantClient, embedder, top_k: int = 8) -> list[dict]:
    """
    Retrieve clinical trials matching patient profile.
    Embeds profile query, searches Qdrant inclusion chunks, filters demographics
    + hard biomarker conflicts, re-ranks with LLM, then attaches the trial's
    full exclusion text for scoring.
    """

    query_text = profile.to_search_query()
    query_vector = embedder.encode(query_text).tolist()

    # Fetch larger pool for filtering + reranking
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="criteria_type",
                    match=MatchValue(value="inclusion")
                )
            ]
        ),
        limit=top_k * 4
    )

    patient_biomarkers = _patient_biomarker_state(profile)

    results = []
    demographic_drops = 0
    biomarker_drops = 0
    for hit in response.points:
        payload = hit.payload or {}

        if not is_eligible(profile, payload):
            demographic_drops += 1
            continue

        conflict = _trial_biomarker_conflict(patient_biomarkers, payload.get("text", ""))
        if conflict:
            biomarker_drops += 1
            logger.debug("retrieve.biomarker_drop nct_id=%s reason=%s",
                         payload.get("nct_id"), conflict)
            continue

        results.append({
            "nct_id": payload.get("nct_id"),
            "title": payload.get("title"),
            "criteria_type": payload.get("criteria_type"),
            "text": payload.get("text"),
            "similarity_score": hit.score,
            "phase": payload.get("phase"),
            "minimum_age": payload.get("minimum_age"),
            "maximum_age": payload.get("maximum_age"),
            "gender": payload.get("gender"),
            "condition": payload.get("condition")
        })

    if demographic_drops or biomarker_drops:
        logger.info(
            "retrieve.filter_summary candidates=%d kept=%d demographic_drops=%d biomarker_drops=%d",
            len(response.points), len(results), demographic_drops, biomarker_drops,
        )

    # Re-rank with LLM
    if results:
        results = rerank_results(results, profile)

    top_results = results[:top_k]

    # Attach exclusion text for the trials that survived ranking
    for r in top_results:
        r["exclusion_text"] = get_exclusion_text(client, r["nct_id"])

    return top_results
