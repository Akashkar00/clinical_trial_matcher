# pipeline/state.py

from typing import TypedDict, Optional, Any
from models.patient_profile import PatientProfile
from trials.models import ClinicalTrial


class TrialMatch(TypedDict):
    nct_id: str
    title: str
    match_type: str        # MATCH / PARTIAL / NO
    score: float           # 0.0 to 1.0
    reason: str
    phase: Optional[str]
    locations: Optional[list]


class PipelineState(TypedDict):
    # inputs
    pdf_path: str

    # populated by nodes
    raw_text: Optional[str]
    patient_profile: Optional[PatientProfile]
    fetched_trials: Optional[list[ClinicalTrial]]
    chunks_stored: Optional[int]
    retrieved_chunks: Optional[list[dict]]
    scored_trials: Optional[list[TrialMatch]]

    # control
    error: Optional[str]
    retry_count: int

    # session / memory
    session_id: Optional[str]            # thread_id for LangGraph MemorySaver
    conversation_history: Optional[list[dict]]  # prior turns [{role, content}]
