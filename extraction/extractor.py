# extraction/extractor.py

import json
import logging
import re
from groq import Groq
from langsmith import traceable
from models.patient_profile import PatientProfile
from config import GROQ_API_KEY, GROQ_MODEL
from pipeline.prompts import EXTRACTION_PROMPT
from extraction.sanitizer import sanitize_report, wrap_as_data
from observability import track_call
from pydantic import ValidationError

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)


def _extract_json(raw: str) -> dict:
    """Tolerant JSON extraction. Tries fenced ```json``` block, bare ```
    fence, the last balanced {...} object, then direct loads. Raises
    json.JSONDecodeError if every strategy fails.
    """
    text = raw.strip()

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    bare = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if bare:
        return json.loads(bare.group(1))

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


@traceable(name="extract_patient_profile", run_type="llm", metadata={"stage": "extraction"})
def extract_patient_profile(raw_text: str) -> PatientProfile:
    """
    Sanitize raw PDF text, send to Groq, parse response as PatientProfile.
    Raises ValueError if extraction or validation fails.
    """
    if not raw_text or len(raw_text.strip()) < 50:
        raise ValueError("Text too short for extraction")

    report = sanitize_report(raw_text)
    if report.flags:
        logger.warning(
            "extraction.injection_signals signals=%s original_len=%d truncated=%s",
            report.flags, report.original_length, report.truncated,
        )
    if report.truncated:
        logger.info(
            "extraction.truncated original_len=%d kept=%d",
            report.original_length, len(report.text),
        )

    wrapped = wrap_as_data(report.text)
    prompt = EXTRACTION_PROMPT.format(report_text=wrapped)

    with track_call("extraction", GROQ_MODEL) as ctx:
        ctx["response"] = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
    raw_response = ctx["response"].choices[0].message.content.strip()

    try:
        data = _extract_json(raw_response)
    except json.JSONDecodeError as e:
        raise ValueError(f"Groq returned invalid JSON: {e}\nRaw: {raw_response[:200]}")

    try:
        profile = PatientProfile(**data)
    except ValidationError as e:
        raise ValueError(f"Profile validation failed: {e}")

    return profile


if __name__ == "__main__":
    from extraction.pdf_loader import load_pdf
    
    test_files = [
        "tests/patient_1.pdf",
        "tests/patient_2.pdf",
        "tests/patient_3.pdf"
    ]
    
    for path in test_files:
        print(f"\n{'='*50}")
        print(f"File: {path}")
        try:
            raw_text = load_pdf(path)
            profile = extract_patient_profile(raw_text)
            print("Successfully Extracted Patient Profile:")
            print(profile.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error extracting from {path}: {e}")