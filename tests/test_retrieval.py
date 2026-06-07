import pytest

from rag.retrieve import parse_age, is_eligible
from rag.ingest import chunk_criteria
from models.patient_profile import PatientProfile, Gender


# ── parse_age ──────────────────────────────────────────────
@pytest.mark.parametrize("inp,expected", [
    ("18 Years", 18.0),
    ("6 Months", 0.5),
    ("4 Weeks", 4 / 52.0),
    ("365 Days", 1.0),
    ("21", 21.0),  # missing unit defaults to years
    (None, None),
    ("", None),
    ("N/A", None),
    ("Unknown", None),
    ("garbage", None),
])
def test_parse_age(inp, expected):
    result = parse_age(inp)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, rel=1e-3)


# ── is_eligible ────────────────────────────────────────────
def _profile(age=50, gender=Gender.FEMALE):
    return PatientProfile(age=age, gender=gender, diagnosis="x")


def test_eligible_gender_match():
    assert is_eligible(
        _profile(gender=Gender.MALE),
        {"gender": "MALE", "minimum_age": "18 Years", "maximum_age": "65 Years"},
    )


def test_ineligible_gender_mismatch():
    assert not is_eligible(
        _profile(gender=Gender.MALE),
        {"gender": "FEMALE", "minimum_age": "18 Years", "maximum_age": "65 Years"},
    )


def test_eligible_gender_all():
    assert is_eligible(
        _profile(gender=Gender.MALE),
        {"gender": "ALL", "minimum_age": "18 Years", "maximum_age": "65 Years"},
    )


def test_ineligible_below_min_age():
    assert not is_eligible(
        _profile(age=10),
        {"gender": "ALL", "minimum_age": "18 Years", "maximum_age": "65 Years"},
    )


def test_ineligible_above_max_age():
    assert not is_eligible(
        _profile(age=70),
        {"gender": "ALL", "minimum_age": "18 Years", "maximum_age": "65 Years"},
    )


def test_eligible_when_no_age_bounds():
    assert is_eligible(_profile(age=70), {"gender": "ALL"})


# ── chunk_criteria ────────────────────────────────────────
def test_chunk_criteria_numbered_list():
    text = """
1. Adult patients >=18 years of age with biopsy-proven advanced solid tumor
2. ECOG performance status of 0 or 1 at study entry
3. Adequate organ function as defined by hematologic and hepatic labs
"""
    chunks = chunk_criteria(text, max_chunk_size=200)
    assert chunks
    joined = " ".join(chunks)
    assert "Adult patients" in joined
    assert "ECOG" in joined
    assert "Adequate organ function" in joined


def test_chunk_criteria_bullet_list():
    text = """
- Histologically confirmed metastatic non-small cell lung cancer
- Documented EGFR-activating mutation prior to enrollment
- No prior systemic therapy in the metastatic setting
"""
    chunks = chunk_criteria(text)
    joined = " ".join(chunks)
    assert "EGFR" in joined
    assert "metastatic" in joined


def test_chunk_criteria_filters_tiny_items():
    text = """
1. Yes
2. No
3. Adult patients with histologically confirmed advanced solid tumor
"""
    chunks = chunk_criteria(text)
    joined = " ".join(chunks)
    assert "Adult patients" in joined


def test_chunk_criteria_unstructured_returns_text():
    text = "First sentence about inclusion. Second sentence with more detail. Third sentence wraps it up."
    chunks = chunk_criteria(text)
    assert chunks
    joined = " ".join(chunks)
    assert "First sentence" in joined


def test_chunk_criteria_respects_max_size():
    long_items = "\n".join(
        f"{i}. " + ("Eligibility text item " * 15) for i in range(1, 6)
    )
    chunks = chunk_criteria(long_items, max_chunk_size=200)
    assert len(chunks) > 1
