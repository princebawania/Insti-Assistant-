"""
Headless CLI for quick testing without the UI.

Usage:
    python ask.py "What does WnCC's Learners' Space cover?"
"""
from __future__ import annotations

import sys

from rag.pipeline import RAGPipeline
from rag.vectorstore import VectorStore


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question"')
        return
    question = " ".join(sys.argv[1:])

    store = VectorStore.load()
    if store is None:
        print("No index found. Run `python build_index.py` first.")
        return

    result = RAGPipeline(store).answer(question)
    badge = "GROUNDED" if result["grounded"] else "NOT GROUNDED"
    print(f"\n[{badge} | confidence {result['confidence']:.2f}]\n")
    print(result["answer"])
    print("\n--- Sources ---")
    for i, s in enumerate(result["sources"], 1):
        print(f"[{i}] {s['source']}  (score {s['score']:.2f})")
        print(f"    {s['text'][:160].strip()}...")


if __name__ == "__main__":
    main()
