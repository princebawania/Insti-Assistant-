"""
Build (or rebuild) the FAISS index from everything in data/raw/.

Usage:
    python build_index.py
"""
from __future__ import annotations

import config
from rag.chunking import chunk_documents
from rag.embedding import embed_texts
from rag.ingest import load_directory
from rag.vectorstore import VectorStore


def main() -> None:
    print(f"Loading documents from {config.RAW_DIR} ...")
    docs = load_directory()
    if not docs:
        print("No documents found. Add .txt/.md/.pdf/.html files to data/raw/ first.")
        return
    print(f"  Loaded {len(docs)} document(s): {[d['source'] for d in docs]}")

    print("Chunking ...")
    chunks = chunk_documents(docs)
    print(f"  Produced {len(chunks)} chunks.")

    print(f"Embedding with {config.EMBED_MODEL_NAME} ...")
    vectors = embed_texts([c["text"] for c in chunks])

    print("Building FAISS index ...")
    store = VectorStore()
    store.add(vectors, chunks)
    store.save()
    print(f"Saved index ({store.size} vectors) to {config.INDEX_PATH}")


if __name__ == "__main__":
    main()
