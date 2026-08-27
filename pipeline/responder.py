# pipeline/responder.py
"""
Responder Node — generates a conversational answer for general clinical-trial
questions. Used when the Planner routes the request as "conversational".

Optionally receives conversation_history from the LangGraph MemorySaver so
multi-turn context is maintained across requests.
"""

import logging
from typing import Optional
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_RESPONDER_SYSTEM_PROMPT = """
You are a knowledgeable, empathetic Clinical Trial Matcher assistant.

Your role:
- Answer questions about clinical trials, eligibility criteria, trial phases,
  biomarkers, treatment histories, and how the matching pipeline works.
- Help users understand what information they need to find matching trials.
- Explain clinical terminology clearly and accessibly.

Boundaries:
- Do NOT diagnose medical conditions.
- Do NOT recommend specific treatments or medications.
- Do NOT provide opinions on a patient's personal prognosis.
- When relevant, always encourage users to consult a qualified oncologist.

Keep answers concise (2-4 paragraphs) unless more detail is needed.
"""


def respond(
    user_message: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Generate a conversational response to a clinical-trial question.

    Args:
        user_message:          The user's current message.
        conversation_history:  Prior turns as list of {"role": ..., "content": ...}.
                               Loaded from LangGraph MemorySaver when available.

    Returns:
        The assistant's response string.
    """
    client = Groq(api_key=GROQ_API_KEY)

    messages: list[dict] = [{"role": "system", "content": _RESPONDER_SYSTEM_PROMPT}]

    # Inject memory — last 10 turns to stay within context limits
    if conversation_history:
        messages.extend(conversation_history[-10:])

    messages.append({"role": "user", "content": user_message})

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        answer = resp.choices[0].message.content.strip()
        logger.info("responder.done chars=%d", len(answer))
        return answer

    except Exception as e:
        logger.error("responder.error err=%s", e)
        return (
            "I'm sorry, I encountered an error generating a response. "
            "Please try again or rephrase your question."
        )
