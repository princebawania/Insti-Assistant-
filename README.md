# 🎓 IITB Insti-Assist — A RAG-Powered AI Assistant for IIT Bombay

A retrieval-augmented assistant that answers questions about IIT Bombay by
**grounding a Large Language Model in real institute documents**. It answers
*only* from retrieved context and clearly says **"I don't know"** when the
answer isn't in its knowledge base — no hallucination.

Final project for the WnCC Machine Learning Learners' Space 2026.

**Scope:** General Insti Assistant (academics, examinations, disciplinary rules,
campus/institute information).

---

## ✨ Features

**Core**
- Data ingestion from `.txt`, `.md`, `.pdf`, `.html` (plus a URL scraper)
- Chunking + embedding with `sentence-transformers` (`BAAI/bge-small-en-v1.5`)
- Vector search over a **FAISS** index (cosine similarity)
- LLM generation (Google **Gemini**) with retrieved context injected into the prompt
- **Grounded answering** — refuses when no relevant context is found
- Source display for every answer
- **Streamlit** web interface

**Stretch goals (all implemented)**
- 📚 **Source citation highlighting** — exact chunk + similarity score per answer
- 🧵 **Multi-turn memory** — follow-up questions use recent conversation history
- 📎 **Live PDF upload** — drop a PDF in the sidebar and query it instantly
- 🚦 **Grounded / confidence indicator** — a badge on every answer

---

## 📚 Data sources (knowledge base)

Nine real, official IIT Bombay documents (~100 pages) in `data/raw/`:

| Document | Covers |
|---|---|
| `UG_RULE_BOOK.pdf` | Undergraduate rules & regulations |
| `Academic_Calendar_2026-27_FINAL.pdf` | Semester dates, registration, exams |
| `Dissertation17june09-10.pdf` | Dissertation rules |
| `IDDDP_Guidelines_2025.pdf` | Interdisciplinary Dual Degree Programme |
| `RulesforAwardofMedalsandAcademicprizesforUGandPG.pdf` | Medals & academic prizes |
| `procedures201521July.pdf` | Disciplinary procedures |
| `punishments201521July.pdf` | Disciplinary punishments |
| `student-medical-rules.pdf` | Student medical rules |
| `01_iitb_overview.txt` | General institute overview (departments, campus, Gymkhana) |

---

## 🗂️ Project structure

```
iitb-insti-assist/
├── app.py                # Streamlit UI (chat + all 4 stretch goals)
├── build_index.py        # Build the FAISS index from data/raw/
├── ask.py                # Headless CLI for quick testing
├── config.py             # All tunable settings (chunk size, top-k, threshold…)
├── requirements.txt
├── .env.example          # Copy to .env and add your API key
├── rag/
│   ├── ingest.py         # Load txt/md/pdf/html; scrape_url() helper
│   ├── chunking.py       # Overlapping sliding-window chunker
│   ├── embedding.py      # sentence-transformers wrapper (BGE, normalised vectors)
│   ├── vectorstore.py    # FAISS index + metadata, save/load/clone
│   ├── llm.py            # Provider wrapper (Gemini default; OpenAI/Anthropic)
│   └── pipeline.py       # retrieve → build prompt → generate → grounding
└── data/
    ├── raw/              # source documents (9 included)
    └── index/            # generated FAISS index (rebuild with build_index.py)
```

---

## 🚀 Setup & run

### 1. Install dependencies
```bash
cd iitb-insti-assist
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your free LLM API key
Defaults to **Google Gemini** (free tier). Get a key:
https://aistudio.google.com/app/apikey

```bash
cp .env.example .env
```
Then edit `.env`:
```
GEMINI_API_KEY=your_key_here
LLM_MODEL=gemini-2.5-flash
```

> To switch providers, set `LLM_PROVIDER=openai` or `anthropic` in `.env` and add
> the matching key — no code changes needed.

### 3. Build the index
```bash
python build_index.py
```
(First run downloads the embedding model, ~130 MB.)

### 4. Run the assistant
```bash
python -m streamlit run app.py
```
Or test headless:
```bash
python ask.py "What are the rules for student medical leave?"
```

> **Tip:** always run through the venv. If you use Anaconda, launch with the
> venv's interpreter directly to avoid environment mix-ups:
> `.venv/bin/python -m streamlit run app.py`

---

## ⚙️ Key settings (`config.py`)

| Setting | Value | Meaning |
|---|---|---|
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model (384-dim) |
| `CHUNK_SIZE` | 1200 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `TOP_K` | 4 | Passages retrieved per query |
| `MIN_SCORE` | 0.30 | Grounding threshold (below → "I don't know") |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model |

Both `top-k` and the grounding threshold are also adjustable live from the app sidebar.

---

## 🔧 How it stays honest (no hallucination)

1. **Retrieval-score gate** (`MIN_SCORE`): if the best passage's cosine similarity
   is below the threshold, the assistant returns "I don't know" **without calling
   the LLM**.
2. **Strict system prompt**: the model answers only from provided context, states
   which semester/term a date belongs to, lists options when a question is
   ambiguous, cites `[Source N]`, and otherwise refuses.

---

## 📄 Attribution
Built for educational use as part of WnCC ML LS 2026. Source documents are
official IIT Bombay publications used for academic demonstration.
