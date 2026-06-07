"""
Test setup: stub heavy externals BEFORE any test module imports the project,
so module-level QdrantClient / SentenceTransformer construction succeeds even
when streamlit holds the on-disk Qdrant lock, and so config.py doesn't crash
when GROQ_API_KEY is missing in CI.
"""
import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")

# Project root on path so `from extraction...` works regardless of cwd.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import qdrant_client as _qc
_qc.QdrantClient = MagicMock()

import sentence_transformers as _st
_st.SentenceTransformer = MagicMock()
