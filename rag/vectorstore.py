"""
Vector store: a thin wrapper over a FAISS inner-product index plus the
per-chunk metadata (text + source), with save/load to disk.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import config


class VectorStore:
    def __init__(self, dim: int = config.EMBED_DIM):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # inner product == cosine (unit vecs)
        self.metadata: List[dict] = []        # aligned 1:1 with index vectors

    # ---- build ----
    def add(self, embeddings: np.ndarray, metadatas: List[dict]) -> None:
        if len(metadatas) == 0:
            return
        assert embeddings.shape[0] == len(metadatas), "vectors/metadata mismatch"
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(metadatas)

    # ---- query ----
    def search(self, query_vec: np.ndarray, k: int = config.TOP_K) -> List[dict]:
        if self.index.ntotal == 0:
            return []
        q = np.array([query_vec], dtype=np.float32)
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)   # cosine similarity in [-1, 1]
            results.append(item)
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal

    # ---- persistence ----
    def save(self, index_path: Path = config.INDEX_PATH,
             meta_path: Path = config.META_PATH) -> None:
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, index_path: Path = config.INDEX_PATH,
             meta_path: Path = config.META_PATH) -> Optional["VectorStore"]:
        import faiss

        if not Path(index_path).exists() or not Path(meta_path).exists():
            return None
        store = cls()
        store.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            store.metadata = pickle.load(f)
        return store

    def clone(self) -> "VectorStore":
        """Return an in-memory copy (used to add live-uploaded PDFs per session)."""
        import faiss

        new = VectorStore(self.dim)
        new.index = faiss.deserialize_index(faiss.serialize_index(self.index))
        new.metadata = list(self.metadata)
        return new
