"""
Metrics for the clinical trial matcher.

Two evaluation surfaces:

  1. Retrieval — given a patient, did the *known-eligible* trials make
     it into the top-K candidate set before LLM scoring?
       Reports recall@K and MRR.

  2. Scoring  — given a (patient, trial) pair the labelers reviewed,
     did the LLM's verdict (MATCH/PARTIAL/NO) match the human verdict?
       Reports per-class precision/recall/F1 plus an overall accuracy
       and a 3x3 confusion matrix.

Both metrics are reported separately so a regression in one is not
hidden by stability in the other.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


VERDICTS = ("MATCH", "PARTIAL", "NO")


@dataclass
class ScoringMetrics:
    accuracy: float
    per_class: dict[str, dict[str, float]]   # verdict -> {precision, recall, f1, support}
    confusion: dict[str, dict[str, int]]      # truth -> predicted -> count
    n: int

    def summary(self) -> str:
        lines = [
            f"n={self.n}  accuracy={self.accuracy:.3f}",
            f"{'class':<8} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}",
        ]
        for v in VERDICTS:
            m = self.per_class.get(v, {})
            lines.append(
                f"{v:<8} {m.get('precision', 0.0):>6.3f} "
                f"{m.get('recall', 0.0):>6.3f} {m.get('f1', 0.0):>6.3f} "
                f"{int(m.get('support', 0)):>8d}"
            )
        return "\n".join(lines)


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    n_queries: int
    n_relevant_total: int
    n_relevant_retrieved: int

    def summary(self) -> str:
        return (
            f"n_queries={self.n_queries} "
            f"recall@k={self.recall_at_k:.3f} "
            f"mrr={self.mrr:.3f} "
            f"({self.n_relevant_retrieved}/{self.n_relevant_total} relevant retrieved)"
        )


def scoring_metrics(pairs: list[tuple[str, str]]) -> ScoringMetrics:
    """`pairs` is a list of (truth_verdict, predicted_verdict)."""
    n = len(pairs)
    if n == 0:
        return ScoringMetrics(0.0, {}, {}, 0)

    confusion = {t: {p: 0 for p in VERDICTS} for t in VERDICTS}
    correct = 0
    for truth, pred in pairs:
        if truth not in VERDICTS or pred not in VERDICTS:
            continue
        confusion[truth][pred] += 1
        if truth == pred:
            correct += 1

    per_class = {}
    for v in VERDICTS:
        tp = confusion[v][v]
        fp = sum(confusion[t][v] for t in VERDICTS if t != v)
        fn = sum(confusion[v][p] for p in VERDICTS if p != v)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[v] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }

    return ScoringMetrics(
        accuracy=correct / n,
        per_class=per_class,
        confusion=confusion,
        n=n,
    )


def retrieval_metrics(
    queries: list[dict],
) -> RetrievalMetrics:
    """`queries` is a list of {"relevant": set[str], "retrieved_ranked": list[str]}.

    Every "relevant" id is one the labeler said should reach scoring.
    "retrieved_ranked" is the ordered list of NCT ids the retriever
    surfaced (post-filter, post-rerank), most relevant first.
    """
    if not queries:
        return RetrievalMetrics(0.0, 0.0, 0, 0, 0)

    total_relevant = 0
    total_relevant_retrieved = 0
    rr_sum = 0.0
    valid = 0

    for q in queries:
        relevant = set(q.get("relevant") or [])
        retrieved = list(q.get("retrieved_ranked") or [])
        if not relevant:
            continue
        valid += 1
        total_relevant += len(relevant)
        total_relevant_retrieved += sum(1 for r in retrieved if r in relevant)

        rr = 0.0
        for rank, nct in enumerate(retrieved, start=1):
            if nct in relevant:
                rr = 1.0 / rank
                break
        rr_sum += rr

    return RetrievalMetrics(
        recall_at_k=(total_relevant_retrieved / total_relevant) if total_relevant else 0.0,
        mrr=(rr_sum / valid) if valid else 0.0,
        n_queries=valid,
        n_relevant_total=total_relevant,
        n_relevant_retrieved=total_relevant_retrieved,
    )


def group_labels_by_patient(labels: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in labels:
        grouped[int(row["patient_id"])].append(row)
    return dict(grouped)
