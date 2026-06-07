# eval/evaluate.py

import json
import argparse
from pathlib import Path
from pipeline.graph import pipeline
from models.patient_profile import PatientProfile


def evaluate(max_patients: int = 3, country: str | None = None):
    """Run the pipeline on synthetic patient profiles and report MATCH/PARTIAL/NO breakdowns.

    If `country` is provided, additionally reports metrics restricted to trials with at
    least one location in that country (case-insensitive match on TrialLocation.country).
    """
    patient_dir = Path("eval/synthetic_patients")
    patient_files = sorted(list(patient_dir.glob("patient_*.json")), key=lambda p: int(p.stem.split("_")[1]))

    if not patient_files:
        print("No synthetic patient profiles found. Run generate_synthetic_patients.py first.")
        return

    patient_files = patient_files[:max_patients]
    print(f"Starting evaluation on {len(patient_files)} patients (honest pipeline metrics)...\n")

    country_norm = country.strip().lower() if country else None
    results_summary = []

    for pf in patient_files:
        print(f"\nEvaluating patient from {pf.name}...")
        with open(pf, "r") as f:
            patient_data = json.load(f)

        profile = PatientProfile(**patient_data)

        state = pipeline.invoke({
            "pdf_path": "",  # skipped because profile is provided
            "raw_text": None,
            "patient_profile": profile,
            "fetched_trials": None,
            "chunks_stored": None,
            "retrieved_chunks": None,
            "scored_trials": None,
            "error": None,
            "retry_count": 0
        })

        if state.get("error"):
            print(f"Pipeline failed for {pf.name}: {state['error']}")
            continue

        scored_trials = state.get("scored_trials", [])
        total_scored = len(scored_trials)
        if total_scored == 0:
            print(f"Patient {patient_data.get('id')} ({profile.diagnosis}): No trials retrieved.")
            continue

        global_metrics = _compute_metrics(scored_trials)

        country_metrics = None
        if country_norm:
            country_trials = [
                t for t in scored_trials
                if any(
                    (loc.get("country") or "").strip().lower() == country_norm
                    for loc in t.get("locations", [])
                )
            ]
            country_metrics = _compute_metrics(country_trials)

        row = {
            "patient_id": patient_data.get("id"),
            "diagnosis": profile.diagnosis,
            **{f"global_{k}": v for k, v in global_metrics.items()},
        }
        if country_metrics is not None:
            row.update({f"country_{k}": v for k, v in country_metrics.items()})
        results_summary.append(row)

        print(
            f"Patient {patient_data.get('id')} ({profile.diagnosis}):\n"
            f"  [Global] {_fmt_metrics(global_metrics)}"
        )
        if country_metrics is not None:
            print(f"  [{country}]  {_fmt_metrics(country_metrics)}")

    if not results_summary:
        print("No patients successfully evaluated.")
        return

    _print_summary_table(results_summary, country)


def _compute_metrics(trials: list[dict]) -> dict:
    total = len(trials)
    if total == 0:
        return {"total": 0, "match": 0, "partial": 0, "no": 0, "avg_score": 0.0}
    return {
        "total": total,
        "match": sum(1 for t in trials if t["match_type"] == "MATCH"),
        "partial": sum(1 for t in trials if t["match_type"] == "PARTIAL"),
        "no": sum(1 for t in trials if t["match_type"] == "NO"),
        "avg_score": sum(t["score"] for t in trials) / total,
    }


def _fmt_metrics(m: dict) -> str:
    return (
        f"Total: {m['total']} | MATCH: {m['match']} | PARTIAL: {m['partial']} | "
        f"NO: {m['no']} | Avg Score: {m['avg_score']:.2f}"
    )


def _print_summary_table(rows: list[dict], country: str | None):
    width = 85 if country is None else 105
    print("\n" + "=" * width)
    print("EVALUATION PIPELINE METRICS SUMMARY")
    print("=" * width)

    header = f"{'ID':<4} | {'Diagnosis':<25} | {'Scored':<6} | {'MATCH':<5} | {'PARTIAL':<7} | {'NO':<3} | {'Avg':<5}"
    if country:
        header += f" | {country + ' Scored':<14} | {country + ' Avg':<10}"
    print(header)
    print("-" * width)

    for r in rows:
        line = (
            f"{r['patient_id']:<4} | {r['diagnosis'][:25]:<25} | "
            f"{r['global_total']:<6} | {r['global_match']:<5} | "
            f"{r['global_partial']:<7} | {r['global_no']:<3} | "
            f"{r['global_avg_score']:<5.2f}"
        )
        if country:
            line += f" | {r['country_total']:<14} | {r['country_avg_score']:<10.2f}"
        print(line)
    print("=" * width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clinical trial matching quality.")
    parser.add_argument("--max-patients", type=int, default=3, help="Maximum number of patients to evaluate (default: 3)")
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Optional country name (case-insensitive) to additionally report metrics for trials with at least one location in that country.",
    )
    args = parser.parse_args()

    evaluate(max_patients=args.max_patients, country=args.country)
