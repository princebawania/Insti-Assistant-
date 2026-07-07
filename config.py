"""
Central configuration for IITB Insti-Assist.

Every tunable knob lives here so the rest of the code stays clean.
Values can be overridden with environment variables (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load .env if present

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # drop your source documents here
INDEX_DIR = DATA_DIR / "index"       # FAISS index + metadata get saved here
INDEX_PATH = INDEX_DIR / "corpus.faiss"
META_PATH = INDEX_DIR / "corpus_meta.pkl"

# ---- Embedding model ----
# all-MiniLM-L6-v2 -> 384-dim, fast, great quality/size tradeoff.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = 384

# ---- Chunking ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # overlap between chunks

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", "4"))
# Cosine similarity below this => we treat the corpus as NOT containing the answer.
# This is the core "grounding" guardrail: below threshold we say "I don't know".
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))

# ---- LLM ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty => provider default (see rag/llm.py)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "4"))  # multi-turn memory window

# The refusal string the assistant must use when unsupported by context.
REFUSAL = "I don't know based on the documents I have."
