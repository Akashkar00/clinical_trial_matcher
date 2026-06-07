"""Tests for eval.metrics — retrieval recall/MRR and scoring P/R/F1."""
from eval.metrics import (
    retrieval_metrics,
    scoring_metrics,
    group_labels_by_patient,
)


# ── Retrieval ────────────────────────────────────────────

def test_retrieval_perfect():
    queries = [
        {"relevant": {"NCT001"}, "retrieved_ranked": ["NCT001", "NCT999"]},
        {"relevant": {"NCT002"}, "retrieved_ranked": ["NCT002"]},
    ]
    m = retrieval_metrics(queries)
    assert m.recall_at_k == 1.0
    assert m.mrr == 1.0
    assert m.n_queries == 2


def test_retrieval_partial_recall():
    queries = [
        {"relevant": {"NCT001", "NCT002"}, "retrieved_ranked": ["NCT001", "NCT999"]},
    ]
    m = retrieval_metrics(queries)
    assert m.recall_at_k == 0.5  # 1 of 2 relevant retrieved
    assert m.mrr == 1.0           # first relevant at rank 1


def test_retrieval_mrr_lower_when_relevant_late():
    queries = [
        {"relevant": {"NCT003"}, "retrieved_ranked": ["NCT001", "NCT002", "NCT003"]},
    ]
    m = retrieval_metrics(queries)
    assert m.recall_at_k == 1.0
    assert abs(m.mrr - 1 / 3) < 1e-9


def test_retrieval_misses_relevant_entirely():
    queries = [
        {"relevant": {"NCT099"}, "retrieved_ranked": ["NCT001"]},
    ]
    m = retrieval_metrics(queries)
    assert m.recall_at_k == 0.0
    assert m.mrr == 0.0


def test_retrieval_skips_query_with_no_labels():
    queries = [
        {"relevant": set(), "retrieved_ranked": ["NCT001"]},
        {"relevant": {"NCT002"}, "retrieved_ranked": ["NCT002"]},
    ]
    m = retrieval_metrics(queries)
    assert m.n_queries == 1   # the empty-relevant query was skipped


def test_retrieval_empty_input():
    m = retrieval_metrics([])
    assert m.recall_at_k == 0.0
    assert m.mrr == 0.0
    assert m.n_queries == 0


# ── Scoring ──────────────────────────────────────────────

def test_scoring_perfect_classifier():
    pairs = [("MATCH", "MATCH"), ("NO", "NO"), ("PARTIAL", "PARTIAL")]
    m = scoring_metrics(pairs)
    assert m.accuracy == 1.0
    for v in ("MATCH", "PARTIAL", "NO"):
        assert m.per_class[v]["precision"] == 1.0
        assert m.per_class[v]["recall"] == 1.0


def test_scoring_all_wrong():
    pairs = [("MATCH", "NO"), ("NO", "MATCH")]
    m = scoring_metrics(pairs)
    assert m.accuracy == 0.0


def test_scoring_confusion_matrix_counts():
    pairs = [
        ("MATCH", "MATCH"),
        ("MATCH", "PARTIAL"),
        ("NO", "NO"),
        ("PARTIAL", "NO"),
    ]
    m = scoring_metrics(pairs)
    assert m.confusion["MATCH"]["MATCH"] == 1
    assert m.confusion["MATCH"]["PARTIAL"] == 1
    assert m.confusion["NO"]["NO"] == 1
    assert m.confusion["PARTIAL"]["NO"] == 1
    # Class supports
    assert m.per_class["MATCH"]["support"] == 2
    assert m.per_class["NO"]["support"] == 1


def test_scoring_handles_empty():
    m = scoring_metrics([])
    assert m.n == 0
    assert m.accuracy == 0.0


def test_scoring_summary_smoke():
    pairs = [("MATCH", "MATCH"), ("NO", "PARTIAL")]
    m = scoring_metrics(pairs)
    s = m.summary()
    assert "accuracy" in s
    assert "MATCH" in s


# ── Grouping ─────────────────────────────────────────────

def test_group_labels_by_patient():
    labels = [
        {"patient_id": 1, "nct_id": "NCT001", "expected_verdict": "MATCH"},
        {"patient_id": 2, "nct_id": "NCT002", "expected_verdict": "NO"},
        {"patient_id": 1, "nct_id": "NCT003", "expected_verdict": "PARTIAL"},
    ]
    grouped = group_labels_by_patient(labels)
    assert set(grouped.keys()) == {1, 2}
    assert len(grouped[1]) == 2
    assert len(grouped[2]) == 1
