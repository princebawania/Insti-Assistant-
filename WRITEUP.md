# IITB Insti-Assist — Project Write-up

*WnCC Machine Learning Learners' Space 2026 — Final Project*

---

## 1. Chosen scope and why

**Scope: General Insti Assistant** — a single assistant spanning academics,
hostel & campus life, student councils/clubs, and administrative processes.

We chose the general scope deliberately. A narrow scope (e.g. only grading
policy) makes retrieval easy but doesn't stress-test the part of RAG that
actually matters: distinguishing *relevant* context from *irrelevant* context
across heterogeneous documents. A general corpus forces the retriever to route
each question to the right document, and it makes the grounding guardrail
(refusing when nothing relevant is found) genuinely load-bearing rather than
decorative. The tradeoff — that breadth is harder to do well in a week — is
mitigated by keeping the corpus curated and topic-segmented (one document per
theme).

## 2. Data sources used

The corpus is organised as one document per insti theme:
1. WnCC & Learners' Space program overview
2. Grading & credit system (SPI/CPI, letter grades)
3. Hostel & campus life
4. Student Gymkhana & councils
5. Academic calendar & course registration
6. Examinations & attendance rules

> **Note on the submitted version:** the repository ships with six clearly
> labelled *sample* documents so the pipeline runs out of the box. For the
> graded submission these are replaced with **real, official IIT Bombay
> sources** — PDFs from the Academic Office / rulebook, official web pages
> (pulled via `rag.ingest.scrape_url`), and council pages — meeting the
> "at least 5 real documents" requirement.

## 3. Chunking strategy and why

We use a **character-based sliding window** of **600 characters with 120
characters of overlap** (`config.CHUNK_SIZE`, `config.CHUNK_OVERLAP`), breaking
on natural boundaries (paragraph → sentence → whitespace) so chunks never cut
mid-word.

Reasoning:
- **~600 chars (~100–130 words)** is large enough to hold a self-contained fact
  (e.g. a full grading-scale explanation) but small enough that the embedding
  vector stays semantically focused rather than averaging several unrelated
  ideas into mush.
- **Overlap** prevents the "lost at the seam" failure where a fact split across
  two chunks becomes unretrievable in both.
- **Boundary-aware splitting** keeps chunks readable, which matters because we
  display the exact chunk as a citation in the UI.

Embeddings use `all-MiniLM-L6-v2` (384-dim), L2-normalised so inner product in
the FAISS `IndexFlatIP` equals cosine similarity. Retrieval returns the top-4
chunks per query.

## 4. How grounding / honesty is enforced

Two independent mechanisms:
1. A **retrieval-score gate**: if the top chunk's cosine similarity is below
   `MIN_SCORE` (default 0.30), we return "I don't know" *before* calling the
   LLM — it physically cannot hallucinate context it was never given.
2. A **strict system prompt** instructing the model to answer only from the
   provided passages, cite `[Source N]`, and otherwise refuse.

The UI surfaces a grounded/not-grounded badge with the confidence score, and an
expander showing the exact retrieved chunks and their similarity scores.

## 5. Known limitations / what we'd improve with more time

- **Fixed-size chunking is naive.** A semantic or layout-aware chunker (respect
  headings, tables) would improve retrieval on structured PDFs like rulebooks.
- **Single-vector retrieval only.** Adding a lexical (BM25) retriever in a
  hybrid setup, plus a cross-encoder re-ranker, would sharpen precision on
  keyword-heavy queries (course codes, hostel names).
- **Threshold is global and hand-tuned.** A learned or per-query calibrated
  threshold would handle the varying difficulty of questions better.
- **No evaluation harness.** We'd add a small labelled Q&A set to measure
  answer accuracy, retrieval hit-rate, and false-"I don't know" rate.
- **Stale data.** Institute rules change; a scheduled re-scrape + re-index job
  would keep the corpus current.
- **Multi-turn memory is verbatim.** Long conversations bloat the prompt; we'd
  summarise older turns rather than pass them raw.
