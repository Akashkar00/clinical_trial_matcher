"""
End-to-end evaluation runner.

Usage:
    python -m eval.run_eval                          # full eval
    python -m eval.run_eval --max-patients 5         # smoke test
    python -m eval.run_eval --dump-trials 3          # list retrieved trials for patient 3
    python -m eval.run_eval --compare-prompts        # A/B test SCORE_PROMPT_V2 vs V3

Reports retrieval recall@K + MRR and scoring precision/recall/F1 against
`eval/ground_truth.json`. Pipeline metrics (MATCH/PARTIAL/NO distribution
+ avg score) are kept under --honest-metrics for the previous behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import config  # noqa: F401  (configures logging)
from models.patient_profile import PatientProfile
from observability import tracker
from pipeline.graph import pipeline
from eval.metrics import (
    group_labels_by_patient,
    retrieval_metrics,
    scoring_metrics,
)

logger = logging.getLogger(__name__)


PROFILE_DIR = Path("eval/synthetic_patients")
GT_PATH = Path("eval/ground_truth.json")


def _load_profiles(max_patients: int | None) -> list[tuple[int, PatientProfile]]:
    files = sorted(PROFILE_DIR.glob("patient_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"No synthetic profiles in {PROFILE_DIR}")
    if max_patients:
        files = files[:max_patients]
    out: list[tuple[int, PatientProfile]] = []
    for f in files:
        with f.open() as fh:
            data = json.load(fh)
        out.append((int(data["id"]), PatientProfile(**data)))
    return out


def _load_ground_truth() -> list[dict]:
    if not GT_PATH.exists():
        return []
    with GT_PATH.open() as fh:
        gt = json.load(fh)
    return gt.get("labels", [])


def _run_pipeline(profile: PatientProfile) -> dict:
    return pipeline.invoke({
        "pdf_path": "",
        "raw_text": None,
        "patient_profile": profile,
        "fetched_trials": None,
        "chunks_stored": None,
        "retrieved_chunks": None,
        "scored_trials": None,
        "error": None,
        "retry_count": 0,
    })


def run_evaluation(max_patients: int | None = None) -> None:
    profiles = _load_profiles(max_patients)
    labels = _load_ground_truth()
    by_patient = group_labels_by_patient(labels)

    tracker.reset()
    retrieval_queries: list[dict] = []
    scoring_pairs: list[tuple[str, str]] = []
    per_patient_rows: list[dict] = []

    for pid, profile in profiles:
        logger.info("eval.patient.start id=%d diagnosis=%s", pid, profile.diagnosis)
        state = _run_pipeline(profile)
        if state.get("error"):
            logger.error("eval.patient.failed id=%d err=%s", pid, state["error"])
            continue

        scored = state.get("scored_trials") or []
        retrieved_ranked = [t["nct_id"] for t in scored]
        predicted_by_nct = {t["nct_id"]: t["match_type"] for t in scored}

        patient_labels = by_patient.get(pid, [])
        relevant = {row["nct_id"] for row in patient_labels
                    if row["expected_verdict"] in ("MATCH", "PARTIAL")}

        if relevant:
            retrieval_queries.append({
                "relevant": relevant,
                "retrieved_ranked": retrieved_ranked,
            })

        per_patient_pairs: list[tuple[str, str]] = []
        for row in patient_labels:
            pred = predicted_by_nct.get(row["nct_id"])
            if pred is None:
                # Labeled trial wasn't even retrieved for this patient.
                # Don't punish the scorer for the retriever's miss.
                continue
            per_patient_pairs.append((row["expected_verdict"], pred))

        scoring_pairs.extend(per_patient_pairs)

        per_patient_rows.append({
            "patient_id": pid,
            "diagnosis": profile.diagnosis,
            "retrieved": len(scored),
            "labeled_for_patient": len(patient_labels),
            "labeled_and_retrieved": len(per_patient_pairs),
            "relevant_in_retrieval": sum(1 for r in retrieved_ranked if r in relevant),
            "relevant_total": len(relevant),
        })

    _print_per_patient(per_patient_rows)

    print("\n" + "=" * 60)
    print("RETRIEVAL")
    print("=" * 60)
    if retrieval_queries:
        rmetrics = retrieval_metrics(retrieval_queries)
        print(rmetrics.summary())
    else:
        print("(no labeled relevant trials yet — populate eval/ground_truth.json)")

    print("\n" + "=" * 60)
    print("SCORING")
    print("=" * 60)
    if scoring_pairs:
        smetrics = scoring_metrics(scoring_pairs)
        print(smetrics.summary())
        print("\nConfusion matrix (rows=truth, cols=predicted):")
        header = f"{'':<10} " + " ".join(f"{p:>8}" for p in ("MATCH", "PARTIAL", "NO"))
        print(header)
        for t in ("MATCH", "PARTIAL", "NO"):
            row = f"{t:<10} " + " ".join(f"{smetrics.confusion[t][p]:>8d}" for p in ("MATCH", "PARTIAL", "NO"))
            print(row)
    else:
        print("(no labeled (patient, trial) pairs landed in retrieval — cannot score)")

    print("\n" + "=" * 60)
    print("COST + LATENCY")
    print("=" * 60)
    print(tracker.report())


def _print_per_patient(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n" + "=" * 90)
    print("PER-PATIENT BREAKDOWN")
    print("=" * 90)
    print(f"{'ID':<4} {'Diagnosis':<28} {'Retr':>5} {'Labeled':>8} {'L∩R':>5} {'Rel/Total':>10}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['patient_id']:<4} {r['diagnosis'][:28]:<28} "
            f"{r['retrieved']:>5} {r['labeled_for_patient']:>8} "
            f"{r['labeled_and_retrieved']:>5} "
            f"{r['relevant_in_retrieval']}/{r['relevant_total']:>3}"
        )


def dump_trials(patient_id: int) -> None:
    """Print every trial currently retrieved for a patient — feeds the
    annotation workflow described in eval/ground_truth.json."""
    profile_path = PROFILE_DIR / f"patient_{patient_id}.json"
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)
    with profile_path.open() as fh:
        profile = PatientProfile(**json.load(fh))

    state = _run_pipeline(profile)
    if state.get("error"):
        print(f"Pipeline failed: {state['error']}")
        return
    trials = state.get("scored_trials") or []
    print(f"\nPatient {patient_id} — {profile.diagnosis} ({profile.age}{profile.gender.value[0].upper()})")
    print(f"Retrieved {len(trials)} trials. Annotate each as MATCH / PARTIAL / NO:\n")
    for t in trials:
        print(f"  {t['nct_id']}  [{t['match_type']:<7}] score={t['score']:.2f}  {t['title'][:55]}")


def compare_prompts() -> None:
    """Run the eval twice with different SCORE_PROMPT versions and diff
    the resulting metrics. Bumps the alias in pipeline.prompts at runtime
    — careful: this mutates module state for the duration of the script."""
    from pipeline import prompts

    versions = [
        ("V2", prompts.SCORE_PROMPT_V2),
        ("V3", prompts.SCORE_PROMPT_V3),
    ]
    original = prompts.SCORE_PROMPT
    try:
        for name, template in versions:
            print(f"\n{'#' * 60}\n# SCORE_PROMPT = {name}\n{'#' * 60}")
            prompts.SCORE_PROMPT = template
            # nodes module read-time-bound the alias — re-bind it.
            from pipeline import nodes
            nodes.SCORE_PROMPT = template
            run_evaluation()
    finally:
        prompts.SCORE_PROMPT = original
        from pipeline import nodes
        nodes.SCORE_PROMPT = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--dump-trials", type=int, default=None,
                        help="Patient id; lists retrieved trials so you can label them")
    parser.add_argument("--compare-prompts", action="store_true",
                        help="A/B test SCORE_PROMPT_V2 vs V3 against ground truth")
    args = parser.parse_args()

    if args.dump_trials is not None:
        dump_trials(args.dump_trials)
    elif args.compare_prompts:
        compare_prompts()
    else:
        run_evaluation(max_patients=args.max_patients)
