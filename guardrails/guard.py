# guardrails/guard.py
"""
Lightweight guardrail: keyword hard-block + Groq LLM topic classifier.

Usage:
    from guardrails.guard import check_guardrails

    blocked, reason = check_guardrails("find trials for lung cancer")
    if blocked:
        return {"error": reason}
"""

import logging
from functools import lru_cache
from groq import Groq
from config import GROQ_API_KEY
from guardrails.rules import hard_block_check, ALLOWED_TOPICS_DESCRIPTION

logger = logging.getLogger(__name__)

# Use a smaller, faster model for the guard — saves quota for the main pipeline.
_GUARD_MODEL = "llama-3.1-8b-instant"


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


def check_guardrails(user_message: str) -> tuple[bool, str]:
    """
    Run the two-layer guardrail check on a user message.

    Layer 1 — keyword hard-block (fast, no LLM call).
    Layer 2 — Groq LLM topic classifier (catches nuanced off-topic queries).

    Returns:
        (blocked: bool, reason: str)
        blocked=True  → caller should reject the request and return `reason`
        blocked=False → request is safe to proceed
    """
    message = user_message.strip()

    if not message:
        return True, "Empty query — please enter a question."

    if len(message) > 4000:
        return True, "Query too long (max 4000 characters)."

    # ── Layer 1: keyword check ─────────────────────────────
    blocked, reason = hard_block_check(message)
    if blocked:
        logger.warning("guardrail.hard_block reason=%s query_preview=%s", reason, message[:60])
        return True, reason

    # ── Layer 2: LLM topic classifier ─────────────────────
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=_GUARD_MODEL,
            messages=[
                {"role": "system", "content": ALLOWED_TOPICS_DESCRIPTION},
                {"role": "user", "content": message},
            ],
            temperature=0.0,
            max_tokens=40,
        )
        verdict = resp.choices[0].message.content.strip()
        logger.debug("guardrail.llm_verdict verdict=%s", verdict)

        if verdict.upper().startswith("ALLOW"):
            return False, ""

        # BLOCK: <reason> or just BLOCK
        if ":" in verdict:
            block_reason = verdict.split(":", 1)[1].strip()
        else:
            block_reason = "Query is outside the scope of this clinical trial assistant."

        logger.warning("guardrail.llm_block reason=%s query_preview=%s", block_reason, message[:60])
        return True, block_reason

    except Exception as e:
        # Fail open — if the guard itself errors, let the request through
        # and log the issue. Do NOT silently fail closed (that would break
        # legitimate queries whenever Groq has a hiccup).
        logger.error("guardrail.error err=%s — failing open", e)
        return False, ""
