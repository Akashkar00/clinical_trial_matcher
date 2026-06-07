# models/patient_profile.py

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class ECOGStatus(int, Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class PatientProfile(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: Gender
    diagnosis: str = Field(..., description="Primary diagnosis")
    stage: Optional[str] = Field(None, description="Cancer stage if applicable")
    biomarkers: list[str] = Field(default_factory=list, description="Molecular markers")
    prior_treatments: list[str] = Field(default_factory=list, description="Previous treatments")
    current_status: Optional[str] = Field(None, description="Current disease status")
    location: Optional[str] = Field(None, description="City or country")
    ecog_status: Optional[int] = Field(None, ge=0, le=4, description="ECOG performance score")
    comorbidities: list[str] = Field(default_factory=list, description="Other conditions")
    key_labs: Optional[dict] = Field(None, description="Important lab values")

    def to_search_query(self) -> str:
        """Rich query for embedding/semantic search."""
        parts = [self.diagnosis]
        if self.stage:
            parts.append(self.stage)
        if self.biomarkers:
            parts.append(" ".join(self.biomarkers))
        if self.prior_treatments:
            parts.append("prior treatment: " + ", ".join(self.prior_treatments))
        if self.ecog_status is not None:
            parts.append(f"ECOG {self.ecog_status}")
        if self.current_status:
            parts.append(self.current_status)
        return " ".join(parts)

    def to_condition_query(self) -> str:
        """
        Condition query for ClinicalTrials.gov `query.cond` field.
        Disease + stage + primary biomarker — narrow enough for relevance,
        broad enough not to zero out results.
        """
        parts = [self.diagnosis]
        if self.stage:
            parts.append(self.stage)
        if self.biomarkers:
            # Primary biomarker — disease subtype (e.g., "EGFR-mutated NSCLC")
            parts.append(self.biomarkers[0])
        return " ".join(parts)

    def to_term_query(self) -> Optional[str]:
        """
        Broader terms for ClinicalTrials.gov `query.term` field — secondary
        biomarkers and prior-treatment context. Returns None when there's
        nothing extra to add.
        """
        parts = []
        if len(self.biomarkers) > 1:
            parts.extend(self.biomarkers[1:])
        if self.prior_treatments:
            parts.extend(self.prior_treatments)
        return " ".join(parts) if parts else None