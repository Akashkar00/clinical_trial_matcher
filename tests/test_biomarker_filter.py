"""Tests for the hard biomarker filter in retrieve."""
from rag.retrieve import _patient_biomarker_state, _trial_biomarker_conflict
from models.patient_profile import PatientProfile, Gender


def _profile(biomarkers):
    return PatientProfile(
        age=55, gender=Gender.FEMALE, diagnosis="Breast cancer",
        biomarkers=biomarkers,
    )


def test_her2_positive_patient_against_her2_negative_trial():
    p = _profile(["HER2 positive", "ER positive"])
    state = _patient_biomarker_state(p)
    assert state.get("her2") == "positive"
    conflict = _trial_biomarker_conflict(
        state, "HER2 negative breast cancer required."
    )
    assert conflict is not None
    assert "HER2" in conflict


def test_her2_negative_patient_against_her2_positive_trial():
    p = _profile(["HER2 negative"])
    state = _patient_biomarker_state(p)
    assert state.get("her2") == "negative"
    conflict = _trial_biomarker_conflict(
        state, "Confirmed HER2 positive disease required."
    )
    assert conflict is not None


def test_no_conflict_when_trial_unrelated():
    p = _profile(["EGFR L858R"])
    state = _patient_biomarker_state(p)
    conflict = _trial_biomarker_conflict(
        state, "Patients aged 18-75 with adequate organ function."
    )
    assert conflict is None


def test_no_conflict_when_trial_matches_patient():
    p = _profile(["HER2 positive"])
    state = _patient_biomarker_state(p)
    conflict = _trial_biomarker_conflict(
        state, "HER2 positive metastatic breast cancer."
    )
    assert conflict is None


def test_egfr_mutation_inferred_as_positive():
    p = _profile(["EGFR L858R"])
    state = _patient_biomarker_state(p)
    assert state.get("egfr") == "positive"


def test_empty_biomarkers_yields_empty_state():
    p = _profile([])
    state = _patient_biomarker_state(p)
    assert state == {}
