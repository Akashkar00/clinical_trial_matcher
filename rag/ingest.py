# rag/ingest.py

import logging
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)
from sentence_transformers import SentenceTransformer
from trials.models import ClinicalTrial
from config import QDRANT_URL, QDRANT_PATH, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


COLLECTION_NAME = "trials"
VECTOR_SIZE = 768  # pritamdeka/S-PubMedBert-MS-MARCO output size


def _build_qdrant_client() -> QdrantClient:
    """Prefer a remote Qdrant when QDRANT_URL is set; fall back to on-disk."""
    if QDRANT_URL:
        logger.info("qdrant.connect mode=remote url=%s", QDRANT_URL)
        return QdrantClient(url=QDRANT_URL)
    logger.info("qdrant.connect mode=local path=%s", QDRANT_PATH)
    return QdrantClient(path=QDRANT_PATH)


client = _build_qdrant_client()


# Lazy-load the embedder so importing this module doesn't pay the
# ~400MB model-load tax. Tests stub `SentenceTransformer` at import time
# (conftest.py) so the lazy-load still gets the mock cleanly.
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("embedder.load model=%s", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


class _LazyEmbedder:
    """Thin proxy so legacy `from rag.ingest import embedder` still works
    without forcing the model load at import time."""

    def encode(self, *args, **kwargs):
        return _get_embedder().encode(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(_get_embedder(), name)


embedder = _LazyEmbedder()


def init_collection():
    """Create Qdrant collection if not exists."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )


def get_cached_nct_ids() -> set[str]:
    """Return the set of nct_ids already present in the collection."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        return set()

    cached: set[str] = set()
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=next_offset,
            with_payload=["nct_id"],
            with_vectors=False,
        )
        for p in points:
            nct_id = (p.payload or {}).get("nct_id")
            if nct_id:
                cached.add(nct_id)
        if next_offset is None:
            break
    return cached


def chunk_criteria(text: str, max_chunk_size: int = 500) -> list[str]:
    """Split eligibility text into semantic chunks by individual criteria items."""
    import re
    # Split on numbered items, bullet points, or newline-separated criteria
    items = re.split(r'\n\s*(?:\d+[\.\)]\s*|-\s*|•\s*|\*\s*)', text)
    items = [item.strip() for item in items if item.strip() and len(item.strip()) > 20]

    if not items:
        # Fallback: split by sentences into chunks
        sentences = re.split(r'(?<=[.;])\s+', text)
        items = [s.strip() for s in sentences if s.strip()]

    # Group small items into chunks up to max_chunk_size
    chunks = []
    current = ""
    for item in items:
        if len(current) + len(item) + 1 > max_chunk_size and current:
            chunks.append(current.strip())
            current = item
        else:
            current = current + "\n" + item if current else item
    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def ingest_trials(trials: list[ClinicalTrial]) -> int:
    """
    Chunk each trial's eligibility criteria into semantic chunks.
    Embed and store in Qdrant. Skips trials whose nct_id is already cached.
    Returns number of new chunks stored.
    """
    init_collection()

    cached_nct_ids = get_cached_nct_ids()
    new_trials = [t for t in trials if t.nct_id not in cached_nct_ids]
    skipped = len(trials) - len(new_trials)
    if skipped:
        logger.info("ingest.cache_hit skipped=%d embedding=%d", skipped, len(new_trials))

    if not new_trials:
        return 0

    # Collect every chunk + its payload first so we can batch-encode.
    pending: list[tuple[str, dict]] = []  # (chunk_text, payload-without-vector)

    for trial in new_trials:
        base_payload = {
            "nct_id": trial.nct_id,
            "title": trial.title,
            "phase": trial.phase,
            "minimum_age": trial.minimum_age,
            "maximum_age": trial.maximum_age,
            "gender": trial.gender,
            "condition": trial.condition,
        }

        inclusion = trial.get_inclusion_criteria()
        if inclusion:
            for chunk_text in chunk_criteria(inclusion):
                pending.append((
                    chunk_text,
                    {**base_payload, "criteria_type": "inclusion", "text": chunk_text},
                ))

        exclusion = trial.get_exclusion_criteria()
        if exclusion:
            for chunk_text in chunk_criteria(exclusion):
                pending.append((
                    chunk_text,
                    {**base_payload, "criteria_type": "exclusion", "text": chunk_text},
                ))

    if not pending:
        return 0

    texts = [t for t, _ in pending]
    vectors = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vec.tolist(), payload=payload)
        for vec, (_, payload) in zip(vectors, pending)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)

    return len(points)


def get_client():
    """Expose client for retriever."""
    return client


def get_embedder():
    """Expose embedder for retriever."""
    return embedder

# python -m rag.ingest

if __name__ == "__main__":
    from extraction.pdf_loader import load_pdf
    from extraction.extractor import extract_patient_profile
    from trials.fetcher import fetch_trials
    from rag.retrieve import retrieve_trials

    # full pipeline test
    print("Loading patient...")
    text = load_pdf("tests/patient_1.pdf")
    profile = extract_patient_profile(text)
    print(f"Patient: {profile.diagnosis}, {profile.age}F")

    # Check if database already has ingested trials
    db_client = get_client()
    try:
        point_count = db_client.count(collection_name=COLLECTION_NAME).count
    except Exception:
        point_count = 0

    if point_count == 0:
        print("\nDatabase empty. Fetching and embedding trials...")
        trials = fetch_trials(
            query=profile.to_search_query(),
            location=None,
            page_size=30
        )
        print(f"Fetched: {len(trials)} trials")
        chunks = ingest_trials(trials)
        print(f"Stored: {chunks} chunks")
    else:
        print(f"\nReusing {point_count} cached trial chunks from disk.")

    print("\nRetrieving top-8...")
    results = retrieve_trials(profile, client, embedder, top_k=8)
    print(f"Retrieved: {len(results)} chunks\n")

    for r in results:
        print(f"{'='*50}")
        print(f"Trial: {r['nct_id']}")
        print(f"Title: {r['title']}")
        print(f"Type: {r['criteria_type']}")
        print(f"Similarity: {r['similarity_score']}")
        print(f"Preview: {r['text'][:150]}")