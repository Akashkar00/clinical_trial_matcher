import json
import pytest
from unittest.mock import patch, MagicMock

from extraction.extractor import extract_patient_profile
from models.patient_profile import PatientProfile, Gender


VALID_PAYLOAD = {
    "age": 62,
    "gender": "female",
    "diagnosis": "Lung adenocarcinoma",
    "stage": "IV",
    "biomarkers": ["EGFR L858R"],
    "prior_treatments": ["carboplatin"],
    "current_status": "progressing",
    "location": "Mumbai",
    "ecog_status": 1,
    "comorbidities": [],
    "key_labs": None,
}

LONG_TEXT = "Patient medical report. " * 20  # comfortably > 50 chars


def _mock_groq_response(content: str):
    msg = MagicMock(); msg.content = content
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


def _patch_groq(content: str):
    return patch(
        "extraction.extractor.client.chat.completions.create",
        return_value=_mock_groq_response(content),
    )


def test_valid_json_returns_profile():
    with _patch_groq(json.dumps(VALID_PAYLOAD)):
        profile = extract_patient_profile(LONG_TEXT)
    assert isinstance(profile, PatientProfile)
    assert profile.age == 62
    assert profile.gender == Gender.FEMALE
    assert profile.diagnosis == "Lung adenocarcinoma"
    assert profile.biomarkers == ["EGFR L858R"]


def test_strips_markdown_fence_with_json_tag():
    fenced = "```json\n" + json.dumps(VALID_PAYLOAD) + "\n```"
    with _patch_groq(fenced):
        profile = extract_patient_profile(LONG_TEXT)
    assert profile.age == 62


def test_strips_bare_fence():
    fenced = "```\n" + json.dumps(VALID_PAYLOAD) + "\n```"
    with _patch_groq(fenced):
        profile = extract_patient_profile(LONG_TEXT)
    assert profile.diagnosis == "Lung adenocarcinoma"


def test_malformed_json_raises():
    with _patch_groq("not json at all"):
        with pytest.raises(ValueError, match="invalid JSON"):
            extract_patient_profile(LONG_TEXT)


def test_short_text_raises():
    with pytest.raises(ValueError, match="too short"):
        extract_patient_profile("hi")


def test_validation_failure_raises():
    bad = dict(VALID_PAYLOAD, age=999)  # ge=0, le=120 -> invalid
    with _patch_groq(json.dumps(bad)):
        with pytest.raises(ValueError, match="validation failed"):
            extract_patient_profile(LONG_TEXT)
