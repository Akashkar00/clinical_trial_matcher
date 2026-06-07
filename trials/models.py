# trials/models.py

from pydantic import BaseModel, Field
from typing import Optional


class TrialLocation(BaseModel):
    facility: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


class ClinicalTrial(BaseModel):
    nct_id: str
    title: str
    overall_status: str
    phase: Optional[str] = None
    condition: Optional[str] = None
    minimum_age: Optional[str] = None
    maximum_age: Optional[str] = None
    gender: Optional[str] = None
    eligibility_criteria: Optional[str] = None
    locations: list[TrialLocation] = Field(default_factory=list)
    brief_summary: Optional[str] = None

    def get_inclusion_criteria(self) -> str:
        """Extract inclusion section from eligibility text."""
        if not self.eligibility_criteria:
            return ""
        text = self.eligibility_criteria
        if "Inclusion Criteria" in text:
            parts = text.split("Exclusion Criteria")
            return parts[0].replace("Inclusion Criteria:", "").strip()
        return text

    def get_exclusion_criteria(self) -> str:
        """Extract exclusion section from eligibility text."""
        if not self.eligibility_criteria:
            return ""
        if "Exclusion Criteria" in self.eligibility_criteria:
            parts = self.eligibility_criteria.split("Exclusion Criteria")
            return parts[1].strip() if len(parts) > 1 else ""
        return ""
