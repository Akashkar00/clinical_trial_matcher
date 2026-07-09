"""
Isolated scoring evaluation.

`run_eval.py` measures the scorer *through* the live retrieval pipeline, which
couples the scoring metric to retrieval drift and — on rate-limited API tiers —
lets the reranker's per-trial calls exhaust the request budget so the scorer
falls back to a non-verdict PARTIAL@0.0. That poisons the confusion matrix.

This module isolates the scorer: it takes the labeled (patient, trial) pairs in
`eval/ground_truth.json`, fetches each trial's real eligibility criteria from
ClinicalTrials.gov, and calls the SAME production scoring function
(`pipeline.nodes._score_with_retry`, SCORE_PROMPT_V3) directly — one paced call
per pair — then compares the verdict to the label.

Why this is the right shape for a scoring metric:
  * every labeled pair is scored (no retrieval drift dropping pairs),
  * rate-limit fallbacks (scoring_ok=False) are excluded, not counted,
  * `--pace` keeps the call rate under the tier limit (Groq free tier ~30/min).

Usage:
    python -m eval.score_eval                     # all labeled pairs
    python -m eval.score_eval --patients 1,2,3    # subset
    python -m eval.score_eval --pace 2.5          # seconds between calls
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

import config  # noqa: F401  (configures logging)
from models.patient_profile import PatientProfile
from pipeline.nodes import _score_with_retry
from eval.metrics import scoring_metrics

logger = logging.getLogger(__name__)

PROFILE_DIR = Path("eval/synthetic_patients")
GT_PATH = Path("eval/ground_truth.json")
CACHE_PATH = Path("eval/.criteria_cache.json")
API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = "NCTId,BriefTitle,EligibilityCriteria,Phase"

VERDICTS = ("MATCH", "PARTIAL", "NO")


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _fetch_criteria(nct: str, cache: dict) -> dict | None:
    if nct in cache:
        return cache[nct]
    try:
        r = requests.get(API + f"/{nct}", params={"fields": FIELDS, "format": "json"}, timeout=20)
        r.raise_for_status()
        proto = r.json().get("protocolSection", {})
        elig = proto.get("eligibilityModule", {})
        ident = proto.get("identificationModule", {})
        design = proto.get("designModule", {})
        rec = {
            "title": ident.get("briefTitle", nct),
            "criteria": elig.get("eligibilityCriteria") or "",
            "phase": (design.get("phases") or [None])[0],
        }
        cache[nct] = rec
        return rec
    except Exception as e:
        logger.warning("score_eval.fetch_failed nct=%s err=%s", nct, str(e)[:80])
        return None


def _split_criteria(text: str) -> tuple[str, str]:
    """Split a ClinicalTrials.gov eligibility block into (inclusion, exclusion)
    on the 'Exclusion Criteria' header, mirroring how the ingest step chunks
    them by criteria_type."""
    low = text.lower()
    idx = low.find("exclusion criteria")
    if idx == -1:
        return text.strip(), "(none provided)"
    return text[:idx].strip(), text[idx:].strip()


def _load_profiles(only_ids: set[int] | None) -> dict[int, PatientProfile]:
    out: dict[int, PatientProfile] = {}
    for f in PROFILE_DIR.glob("patient_*.json"):
        pid = int(f.stem.split("_")[1])
        if only_ids and pid not in only_ids:
            continue
        out[pid] = PatientProfile(**json.loads(f.read_text()))
    return out


def run(only_ids: set[int] | None = None, pace: float = 2.5) -> None:
    labels = json.loads(GT_PATH.read_text()).get("labels", [])
    if only_ids:
        labels = [l for l in labels if l["patient_id"] in only_ids]
    profiles = _load_profiles(only_ids)
    cache = _load_cache()

    pairs: list[tuple[str, str]] = []
    fallbacks = 0
    fetch_fail = 0
    rows: list[dict] = []

    for i, lab in enumerate(labels, 1):
        pid, nct = lab["patient_id"], lab["nct_id"]
        truth = lab["expected_verdict"]
        profile = profiles.get(pid)
        if profile is None:
            continue
        rec = _fetch_criteria(nct, cache)
        if not rec or not rec["criteria"]:
            fetch_fail += 1
            continue

        inclusion, exclusion = _split_criteria(rec["criteria"])
        chunk = {
            "nct_id": nct, "title": rec["title"], "text": inclusion,
            "exclusion_text": exclusion, "phase": rec["phase"],
        }
        result = _score_with_retry(chunk, profile)
        if not result.get("scoring_ok", True):
            fallbacks += 1
            logger.warning("score_eval.fallback nct=%s (rate-limited)", nct)
        else:
            pred = result["match_type"]
            pairs.append((truth, pred))
            rows.append({"pid": pid, "nct": nct, "truth": truth,
                         "pred": pred, "score": result["score"]})

        logger.info("score_eval.progress %d/%d pid=%d nct=%s truth=%s pred=%s",
                    i, len(labels), pid, nct, truth,
                    result.get("match_type") if result.get("scoring_ok", True) else "FALLBACK")
        time.sleep(pace)

    CACHE_PATH.write_text(json.dumps(cache, indent=2))

    print("\n" + "=" * 64)
    print("SCORING EVALUATION (isolated — production scorer on labeled pairs)")
    print("=" * 64)
    print(f"labeled pairs={len(labels)}  scored={len(pairs)}  "
          f"rate_limit_fallbacks_excluded={fallbacks}  fetch_failures={fetch_fail}")

    if not pairs:
        print("\nNo real verdicts produced — increase --pace and retry.")
        return

    m = scoring_metrics(pairs)
    print(f"\naccuracy={m.accuracy:.3f}  (n={m.n})")
    print(f"{'class':<9} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}")
    for v in VERDICTS:
        c = m.per_class.get(v, {})
        print(f"{v:<9} {c.get('precision', 0):>6.3f} {c.get('recall', 0):>6.3f} "
              f"{c.get('f1', 0):>6.3f} {int(c.get('support', 0)):>8d}")

    print("\nConfusion matrix (rows=truth, cols=predicted):")
    print(f"{'':<10}" + "".join(f"{p:>9}" for p in VERDICTS))
    for t in VERDICTS:
        print(f"{t:<10}" + "".join(f"{m.confusion[t][p]:>9d}" for p in VERDICTS))

    # Binary collapse: treat MATCH+PARTIAL as "worth surfacing", NO as "reject".
    tp = sum(1 for t, p in pairs if t != "NO" and p != "NO")
    fp = sum(1 for t, p in pairs if t == "NO" and p != "NO")
    fn = sum(1 for t, p in pairs if t != "NO" and p == "NO")
    tn = sum(1 for t, p in pairs if t == "NO" and p == "NO")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"\nBinary (surface [MATCH|PARTIAL] vs reject [NO]): "
          f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}  (tp={tp} fp={fp} fn={fn} tn={tn})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=str, default=None)
    ap.add_argument("--pace", type=float, default=2.5,
                    help="Seconds to sleep between scoring calls (rate-limit safety)")
    ap.add_argument("--prompt-version", type=str, default=None,
                    help="Score against a specific SCORE_PROMPT_V<N> for A/B (e.g. V3, V4). "
                         "Defaults to the active alias.")
    args = ap.parse_args()

    if args.prompt_version:
        from pipeline import prompts, nodes
        template = getattr(prompts, f"SCORE_PROMPT_{args.prompt_version.upper()}")
        nodes.SCORE_PROMPT = template  # _score_with_retry reads this module global
        print(f"[A/B] scoring with SCORE_PROMPT_{args.prompt_version.upper()}")

    ids = {int(x) for x in args.patients.split(",")} if args.patients else None
    run(only_ids=ids, pace=args.pace)