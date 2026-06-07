# main.py

import logging

import config  # noqa: F401  (configures logging on import)
from observability import tracker
from pipeline.graph import pipeline

logger = logging.getLogger(__name__)


def run(pdf_path: str):
    logger.info("pipeline.run pdf=%s", pdf_path)
    tracker.reset()

    result = pipeline.invoke({
        "pdf_path": pdf_path,
        "raw_text": None,
        "patient_profile": None,
        "fetched_trials": None,
        "chunks_stored": None,
        "retrieved_chunks": None,
        "scored_trials": None,
        "error": None,
        "retry_count": 0
    })

    if result.get("error"):
        logger.error("pipeline.failed err=%s", result["error"])
        return

    trials = result.get("scored_trials", [])
    profile = result.get("patient_profile")

    print(f"\nPatient: {profile.diagnosis}, Age {profile.age}")
    print(f"Trials scored: {len(trials)}")
    print(f"\n{'='*55}")
    print("RANKED TRIAL MATCHES")
    print(f"{'='*55}")

    for t in trials:
        badge = {
            "MATCH":   "✅ MATCH",
            "PARTIAL": "⚠️  PARTIAL",
            "NO":      "❌ NO"
        }.get(t["match_type"], "?")

        print(f"\n{badge}  |  Score: {t['score']:.2f}")
        print(f"ID:     {t['nct_id']}")
        print(f"Title:  {t['title'][:65]}...")
        print(f"Phase:  {t['phase']}")
        print(f"Reason: {t['reason']}")
        print(f"{'-'*55}")

    print("\nCOST + LATENCY (per pipeline run)")
    print(tracker.report())


if __name__ == "__main__":
    run("tests/patient_3.pdf")
    from rag.ingest import client as qdrant_client
    qdrant_client.close()