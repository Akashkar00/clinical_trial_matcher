# config.py

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env")

# ── Qdrant ────────────────────────────────────────────────
# Two modes:
#   QDRANT_URL set      → talk to a Qdrant server (docker-compose, cloud)
#   QDRANT_URL unset    → fall back to local on-disk storage at QDRANT_PATH
# Production deployments should always use QDRANT_URL.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")

# ── Embedding model ───────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")

# ── LangSmith tracing (optional) ─────────────────────────
# When LANGSMITH_API_KEY is set, LangGraph nodes and any @traceable
# functions stream to https://smith.langchain.com automatically.
# Leave unset to run without tracing — the @traceable decorators
# become true no-ops and add zero overhead.
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "clinical-trial-matcher")

# ── Logging ───────────────────────────────────────────────
# Configure once at import time so every module that calls
# `logging.getLogger(__name__)` gets a consistent format.
import logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
