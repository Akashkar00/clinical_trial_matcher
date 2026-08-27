# pipeline/planner.py
"""
Planner Node — routes an incoming query to one of two paths:

  "technical"      → PDF-based full pipeline (extract → fetch → ingest → retrieve → score)
  "conversational" → Direct LLM answer for general clinical trial questions

Routing uses a fast heuristic pass first (keyword signals), then a Groq LLM
call when the intent is ambiguous.
"""

import logging
import re
from functools import lru_cache
from typing import Literal

from groq import Groq
from config import GROQ_API_KEY

logger = logging.getLogger(__name__)

_PLANNER_MODEL = "llama-3.1-8b-instant"

# Keywords that strongly signal a PDF / patient-specific technical request
_TECHNICAL_SIGNALS = re.compile(
    r"\b(pdf|upload|patient report|medical record|scan|document|"
    r"my patient|eligible|eligibility|match (me|my patient)|find trials for)\b",
    re.IGNORECASE,
)

# Keywords that strongly signal a general conversational question
_CONVERSATIONAL_SIGNALS = re.compile(
    r"\b(what is|how does|explain|tell me about|what are|"
    r"define|difference between|how do i|can you help|"
    r"what does .* mean|overview|summary)\b",
    re.IGNORECASE,
)

_PLANNER_SYSTEM_PROMPT = """
You are a routing classifier for a Clinical Trial Matcher assistant.

Classify the user's message as ONE of:
  "technical"      — the user wants to match a specific patient to trials
                     (they have a PDF, a patient profile, or specific eligibility data)
  "conversational" — the user is asking a general question about clinical trials,
                     how the system works, or wants explanatory information

Reply with ONLY the single word: technical  OR  conversational
"""


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


def plan(query: str, has_pdf: bool = False) -> Literal["technical", "conversational"]:
    """
    Determine the routing for a user query.

    Args:
        query:   The user's text message.
        has_pdf: True if the user also uploaded a PDF in this request.

    Returns:
        "technical" or "conversational"
    """
    # If there's a PDF, always go technical regardless of text
    if has_pdf:
        logger.info("planner.route reason=pdf_uploaded → technical")
        return "technical"

    # Fast heuristic pass
    if _TECHNICAL_SIGNALS.search(query):
        logger.info("planner.route reason=keyword_technical → technical")
        return "technical"

    if _CONVERSATIONAL_SIGNALS.search(query):
        logger.info("planner.route reason=keyword_conversational → conversational")
        return "conversational"

    # Ambiguous — ask the LLM
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=_PLANNER_MODEL,
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=5,
        )
        verdict = resp.choices[0].message.content.strip().lower()
        route: Literal["technical", "conversational"] = (
            "technical" if "technical" in verdict else "conversational"
        )
        logger.info("planner.route reason=llm_verdict verdict=%s → %s", verdict, route)
        return route

    except Exception as e:
        # Default to conversational on error — safer than trying to run
        # the PDF pipeline without a PDF
        logger.error("planner.error err=%s — defaulting to conversational", e)
        return "conversational"
