# trials/fetcher.py

import requests
from trials.models import ClinicalTrial, TrialLocation


API_BASE = "https://clinicaltrials.gov/api/v2/studies"

FIELDS = (
    "NCTId,BriefTitle,OverallStatus,Phase,"
    "Condition,MinimumAge,MaximumAge,Gender,"
    "EligibilityCriteria,LocationFacility,"
    "LocationCity,LocationCountry,BriefSummary"
)


def fetch_trials(
    query: str,
    location: str = None,
    page_size: int = 20,
    status: str = "RECRUITING",
    term: str = None
) -> list[ClinicalTrial]:
    """
    Fetch recruiting trials from ClinicalTrials.gov.
    `query` -> query.cond (disease/stage/primary biomarker).
    `term`  -> query.term (secondary biomarkers, prior treatments).
    Returns list of ClinicalTrial objects.
    """
    params = {
        "query.cond": query,
        "filter.overallStatus": status,
        "fields": FIELDS,
        "pageSize": page_size,
        "format": "json"
    }

    if location:
        params["query.locn"] = location

    if term:
        params["query.term"] = term

    try:
        response = requests.get(API_BASE, params=params, timeout=15)
        response.raise_for_status()
    except requests.Timeout:
        raise ConnectionError("ClinicalTrials API timed out")
    except requests.HTTPError as e:
        raise ConnectionError(f"ClinicalTrials API error: {e}")

    data = response.json()
    studies = data.get("studies", [])

    if not studies:
        return []

    trials = []
    for study in studies:
        trial = _parse_study(study)
        if trial:
            trials.append(trial)

    return trials


def _parse_study(study: dict) -> ClinicalTrial | None:
    """Parse raw API response dict into ClinicalTrial."""
    try:
        proto = study.get("protocolSection", {})

        id_module = proto.get("identificationModule", {})
        status_module = proto.get("statusModule", {})
        design_module = proto.get("designModule", {})
        eligibility_module = proto.get("eligibilityModule", {})
        conditions_module = proto.get("conditionsModule", {})
        contacts_module = proto.get("contactsLocationsModule", {})
        desc_module = proto.get("descriptionModule", {})

        # locations
        locations = []
        for loc in contacts_module.get("locations", []):
            locations.append(TrialLocation(
                facility=loc.get("facility"),
                city=loc.get("city"),
                country=loc.get("country")
            ))

        # phase
        phases = design_module.get("phases", [])
        phase = phases[0] if phases else None

        # condition
        conditions = conditions_module.get("conditions", [])
        condition = conditions[0] if conditions else None

        return ClinicalTrial(
            nct_id=id_module.get("nctId", ""),
            title=id_module.get("briefTitle", ""),
            overall_status=status_module.get("overallStatus", ""),
            phase=phase,
            condition=condition,
            minimum_age=eligibility_module.get("minimumAge"),
            maximum_age=eligibility_module.get("maximumAge"),
            gender=eligibility_module.get("sex"),
            eligibility_criteria=eligibility_module.get("eligibilityCriteria"),
            locations=locations,
            brief_summary=desc_module.get("briefSummary")
        )

    except Exception:
        return None  # skip malformed trials silently


# python -m trials.fetcher

if __name__ == "__main__":
    from extraction.pdf_loader import load_pdf
    from extraction.extractor import extract_patient_profile

    # use patient_1 profile
    text = load_pdf("tests/patient_1.pdf")
    profile = extract_patient_profile(text)

    print(f"Search query: {profile.to_search_query()}")
    print(f"Fetching trials...\n")

    trials = fetch_trials(
        query=profile.to_search_query(),
        location=profile.location,
        page_size=10
    )

    print(f"Trials found: {len(trials)}\n")

    for t in trials[:3]:
        print(f"{'='*50}")
        print(f"ID: {t.nct_id}")
        print(f"Title: {t.title}")
        print(f"Phase: {t.phase}")
        print(f"Status: {t.overall_status}")
        print(f"Age range: {t.minimum_age} - {t.maximum_age}")
        print(f"Gender: {t.gender}")
        print(f"Locations: {len(t.locations)}")
        print(f"Has eligibility text: {bool(t.eligibility_criteria)}")
        inc = t.get_inclusion_criteria()
        print(f"Inclusion preview: {inc[:150]}")
