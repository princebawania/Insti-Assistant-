"""
The RAG pipeline: retrieve -> build grounded prompt -> generate -> package.

This is the heart of the assistant. The two things that make it *honest*:
  1. A hard retrieval-score gate (MIN_SCORE): if nothing relevant is found we
     refuse *before* calling the LLM (saves quota and prevents hallucination).
  2. A strict system prompt that instructs the model to answer ONLY from the
     provided context and to say "I don't know" otherwise.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import config
from rag import llm
from rag.embedding import embed_text
from rag.vectorstore import VectorStore

SYSTEM_PROMPT = (
    "You are IITB Insti-Assist, a factual assistant for IIT Bombay. "
    "Answer using ONLY the provided context passages; do not use outside knowledge. "
    "If the context does not clearly contain the answer, reply with exactly: "
    f'"{config.REFUSAL}". '
    "IMPORTANT: When a date or fact depends on a specific semester, term, or year "
    "(e.g. Autumn vs Spring vs Summer term), state which one it belongs to. "
    "If the context contains MULTIPLE possible answers (e.g. exam dates for several "
    "terms) and the question doesn't specify which, do NOT pick one — instead briefly "
    "list each option with its term, or ask the user to specify. "
    "Be concise and cite the [Source N] you used."
)


def build_prompt(
    question: str,
    chunks: List[dict],
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Assemble the final user prompt from context, chat history, and question."""
    context_blocks = []
    for i, ch in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i} | {ch['source']}]\n{ch['text']}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "(no context found)"

    convo = ""
    if history:
        recent = history[-config.MAX_HISTORY_TURNS:]
        turns = [f"User: {u}\nAssistant: {a}" for u, a in recent]
        convo = "Previous conversation (for context on follow-up questions):\n" + \
                "\n".join(turns) + "\n\n"

    return (
        f"{convo}"
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (use only the context above; cite [Source N]):"
    )


class RAGPipeline:
    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve(self, question: str, k: int = config.TOP_K) -> List[dict]:
        q_vec = embed_text(question)
        return self.store.search(q_vec, k=k)

    def answer(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        k: int = config.TOP_K,
    ) -> dict:
        """
        Returns a dict:
          {
            "answer": str,
            "sources": [chunk dicts with 'score'],
            "grounded": bool,
            "confidence": float,   # top cosine similarity
          }
        """
        chunks = self.retrieve(question, k=k)
        top_score = chunks[0]["score"] if chunks else 0.0
        grounded = top_score >= config.MIN_SCORE

        # Guardrail: nothing relevant -> refuse without calling the LLM.
        if not grounded:
            return {
                "answer": config.REFUSAL,
                "sources": chunks,
                "grounded": False,
                "confidence": top_score,
            }

        prompt = build_prompt(question, chunks, history)
        try:
            reply = llm.generate(prompt, system=SYSTEM_PROMPT)
        except llm.LLMError as e:
            return {
                "answer": f"[LLM error] {e}",
                "sources": chunks,
                "grounded": grounded,
                "confidence": top_score,
            }

        # If the model itself refused, reflect that in the grounded flag.
        model_refused = config.REFUSAL.lower()[:12] in reply.lower()
        return {
            "answer": reply,
            "sources": chunks,
            "grounded": grounded and not model_refused,
            "confidence": top_score,
        }
