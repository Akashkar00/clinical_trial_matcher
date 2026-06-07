import pytest
from unittest.mock import patch, MagicMock

from pipeline.nodes import _score_with_retry
from models.patient_profile import PatientProfile, Gender


def _mock_response(content: str):
    msg = MagicMock(); msg.content = content
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


def _profile():
    return PatientProfile(age=58, gender=Gender.FEMALE, diagnosis="Lung adenocarcinoma")


def _chunk():
    return {
        "nct_id": "NCT99999999",
        "title": "Trial X",
        "text": "Patients ≥18 years with advanced NSCLC",
        "exclusion_text": "Active brain metastases",
        "phase": "PHASE2",
    }


def _patch_create(content: str):
    return patch(
        "pipeline.nodes.groq_client.chat.completions.create",
        return_value=_mock_response(content),
    )


def test_valid_score_parsed():
    payload = '{"match_type": "MATCH", "score": 0.85, "reason": "fits inclusion"}'
    with _patch_create(payload):
        result = _score_with_retry(_chunk(), _profile())
    assert result["nct_id"] == "NCT99999999"
    assert result["match_type"] == "MATCH"
    assert result["score"] == pytest.approx(0.85)
    assert result["reason"] == "fits inclusion"
    assert result["phase"] == "PHASE2"


def test_strips_markdown_fence():
    payload = '```json\n{"match_type": "PARTIAL", "score": 0.5, "reason": "borderline"}\n```'
    with _patch_create(payload):
        result = _score_with_retry(_chunk(), _profile())
    assert result["match_type"] == "PARTIAL"
    assert result["score"] == 0.5


def test_score_failure_returns_partial_fallback(monkeypatch):
    """All retries fail → returns a graceful PARTIAL with score 0.0 (no exception)."""
    monkeypatch.setattr("pipeline.nodes.time.sleep", lambda *a, **k: None)
    with patch(
        "pipeline.nodes.groq_client.chat.completions.create",
        side_effect=RuntimeError("boom"),
    ):
        result = _score_with_retry(_chunk(), _profile(), max_retries=2)
    assert result["nct_id"] == "NCT99999999"
    assert result["match_type"] == "PARTIAL"
    assert result["score"] == 0.0
    assert "Scoring failed" in result["reason"]


def test_invalid_json_falls_back(monkeypatch):
    """Malformed JSON across all retries → fallback dict, never raises."""
    monkeypatch.setattr("pipeline.nodes.time.sleep", lambda *a, **k: None)
    with _patch_create("not-json-at-all"):
        result = _score_with_retry(_chunk(), _profile(), max_retries=2)
    assert result["match_type"] == "PARTIAL"
    assert result["score"] == 0.0


def test_retry_succeeds_after_first_failure(monkeypatch):
    """Flaky API: first call fails, second succeeds → MATCH returned."""
    monkeypatch.setattr("pipeline.nodes.time.sleep", lambda *a, **k: None)
    good = _mock_response('{"match_type": "MATCH", "score": 0.9, "reason": "ok"}')
    with patch(
        "pipeline.nodes.groq_client.chat.completions.create",
        side_effect=[RuntimeError("flaky"), good],
    ):
        result = _score_with_retry(_chunk(), _profile(), max_retries=3)
    assert result["match_type"] == "MATCH"
    assert result["score"] == 0.9


def test_cot_response_with_reasoning_and_fenced_json():
    """CoT format: REASONING block + ```json fenced verdict."""
    payload = """REASONING:
1. Exclusions: "Active brain metastases" — patient profile does not mention brain mets, so this exclusion does not clearly apply.
2. Inclusions: patient is 58 with advanced NSCLC, clearly meets the age and disease criteria.
3. Verdict: no exclusion fires and inclusions are met → MATCH at high confidence.

JSON:
```json
{"match_type": "MATCH", "score": 0.88, "reason": "Meets NSCLC inclusion; brain mets exclusion does not apply"}
```"""
    with _patch_create(payload):
        result = _score_with_retry(_chunk(), _profile())
    assert result["match_type"] == "MATCH"
    assert result["score"] == pytest.approx(0.88)
    assert "NSCLC" in result["reason"]


def test_cot_response_unfenced_json_at_end():
    """Model returns reasoning then a bare JSON object (no fence) — last {...} wins."""
    payload = """REASONING: Patient profile lacks ECOG status and biomarker info.
Verdict: missing data, borderline.

{"match_type": "PARTIAL", "score": 0.55, "reason": "Missing ECOG"}"""
    with _patch_create(payload):
        result = _score_with_retry(_chunk(), _profile())
    assert result["match_type"] == "PARTIAL"
    assert result["score"] == pytest.approx(0.55)


def test_cot_response_picks_last_json_block():
    """If reasoning text contains an example JSON, the FINAL JSON should be the verdict."""
    payload = """REASONING:
For reference, a MATCH would look like {"match_type": "MATCH"} but here
the patient has a clear exclusion.

JSON:
```json
{"match_type": "NO", "score": 0.1, "reason": "Active brain mets exclusion"}
```"""
    with _patch_create(payload):
        result = _score_with_retry(_chunk(), _profile())
    assert result["match_type"] == "NO"
    assert result["score"] == pytest.approx(0.1)
