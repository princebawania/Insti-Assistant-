"""
Embedding: turn text into dense vectors with sentence-transformers.

Vectors are L2-normalised so that inner product == cosine similarity, which
is what the FAISS IndexFlatIP in vectorstore.py expects.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

import config


@lru_cache(maxsize=1)
def _get_model():
    # Imported lazily so `import rag.embedding` is cheap and doesn't require
    # torch to be installed until you actually embed something.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings -> (n, EMBED_DIM) float32, L2-normalised."""
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    model = _get_model()
    vecs = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,      # unit vectors -> IP == cosine
        show_progress_bar=len(texts) > 64,
    )
    return vecs.astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """Embed a single query string -> (EMBED_DIM,) float32."""
    prefixed = "Represent this sentence for searching relevant passages: " + text
    return embed_texts([prefixed])[0]
