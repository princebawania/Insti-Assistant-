"""
Chunking: split raw document text into overlapping, retrievable passages.

We use a character-based sliding window that tries to break on natural
boundaries (paragraph, then sentence, then whitespace) so chunks don't get
cut mid-word. Overlap preserves context that would otherwise be lost at the
seam between two chunks.
"""
from __future__ import annotations

from typing import List

import config


def _clean(text: str) -> str:
    # Normalise whitespace but keep paragraph breaks.
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def _best_split(text: str, hard_end: int) -> int:
    """Find a nice place to end a chunk at or before hard_end."""
    window = text[:hard_end]
    for sep in ("\n\n", ". ", "\n", " "):
        idx = window.rfind(sep)
        # only accept the boundary if it's reasonably far in (avoid tiny chunks)
        if idx != -1 and idx > hard_end * 0.5:
            return idx + len(sep)
    return hard_end


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[dict]:
    """
    Split one document's text into chunks.

    Returns a list of dicts: {"text", "source", "chunk_id"}.
    """
    text = _clean(text)
    if not text:
        return []

    chunks: List[dict] = []
    start = 0
    chunk_id = 0
    n = len(text)

    while start < n:
        hard_end = min(start + chunk_size, n)
        if hard_end < n:
            end = start + _best_split(text[start:], chunk_size)
        else:
            end = n

        piece = text[start:end].strip()
        if piece:
            chunks.append(
                {"text": piece, "source": source, "chunk_id": chunk_id}
            )
            chunk_id += 1

        if end >= n:
            break
        # step forward, keeping `overlap` characters of context
        start = max(end - overlap, start + 1)

    return chunks


def chunk_documents(documents: List[dict]) -> List[dict]:
    """
    documents: list of {"text": str, "source": str}
    returns: flat list of chunk dicts across all documents.
    """
    all_chunks: List[dict] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc["text"], doc["source"]))
    return all_chunks
