# api/schemas.py
"""
Pydantic request/response models for the FastAPI /query endpoint.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming request to POST /query."""
    text_query: Optional[str] = Field(
        default=None,
        description="Conversational question about clinical trials.",
        max_length=4000,
    )
    pdf_path: Optional[str] = Field(
        default=None,
        description="Absolute path to a patient PDF (for the technical pipeline).",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session / thread ID for multi-turn memory. Auto-generated if omitted.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text_query": "What trials are available for EGFR+ lung cancer?",
                "session_id": "user-abc-123",
            }
        }


class TrialResult(BaseModel):
    nct_id: str
    title: str
    match_type: Literal["MATCH", "PARTIAL", "NO"]
    score: float
    reason: str
    phase: Optional[str] = None


class QueryResponse(BaseModel):
    """Successful response from POST /query."""
    route: Literal["conversational", "technical"]
    session_id: str
    # Conversational path
    answer: Optional[str] = None
    # Technical path
    trials: Optional[list[TrialResult]] = None
    patient_profile: Optional[dict] = None


class BlockedResponse(BaseModel):
    """Response when guardrails block the request."""
    blocked: bool = True
    reason: str
