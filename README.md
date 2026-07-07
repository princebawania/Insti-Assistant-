# 🎓 IITB Insti-Assist — A RAG-Powered AI Assistant for IIT Bombay

A retrieval-augmented assistant that answers questions about IIT Bombay by
**grounding a Large Language Model in real institute documents**. It answers
*only* from retrieved context and clearly says **"I don't know"** when the
answer isn't in its knowledge base — no hallucination.

Final project for the WnCC Machine Learning Learners' Space 2026.

**Scope:** General Insti Assistant (academics, hostels, clubs, campus life).

---

## ✨ Features

**Core (required)**
- Data ingestion from `.txt`, `.md`, `.pdf`, `.html` (plus a URL scraper)
- Chunking + embedding with `sentence-transformers` (`all-MiniLM-L6-v2`)
- Vector search over a **FAISS** index (cosine similarity)
- LLM generation with retrieved context injected into the prompt
- **Grounded answering** — refuses when no relevant context is found
- Source display for every answer
- **Streamlit** web interface

**Stretch goals (all implemented)**
- 📚 **Source citation highlighting** — exact chunk + similarity score per answer
- 🧵 **Multi-turn memory** — follow-up questions use recent conversation history
- 📎 **Live PDF upload** — drop a PDF in the sidebar and query it instantly
- 🚦 **Grounded / confidence indicator** — a badge on every answer

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
│   ├── embedding.py      # sentence-transformers wrapper (normalised vectors)
│   ├── vectorstore.py    # FAISS index + metadata, save/load/clone
│   ├── llm.py            # Provider wrapper (Gemini default; OpenAI/Anthropic)
│   └── pipeline.py       # retrieve → build prompt → generate → grounding
└── data/
    ├── raw/              # <-- put your source documents here (6 samples included)
    └── index/            # generated FAISS index lives here
```

---

## 🚀 Setup & run

### 1. Install dependencies
```bash
cd iitb-insti-assist
python -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
```

### 2. Add your free LLM API key
This project defaults to **Google Gemini** (generous free tier).
Get a free key: https://aistudio.google.com/app/apikey

```bash
cp .env.example .env
# then edit .env and set:
#   GEMINI_API_KEY=your_key_here
```

> Want a different provider? Set `LLM_PROVIDER=openai` or `anthropic` in `.env`
> and add the matching key. The rest of the code doesn't change.

### 3. Add documents (or use the included samples)
Six **sample** documents are already in `data/raw/`. They exist so the pipeline
runs immediately — **replace them with real, official IIT Bombay documents**
(PDFs, scraped pages, official text) before submitting. You need at least 5.

Tip — pull an official page straight into the corpus:
```python
from rag.ingest import scrape_url
doc = scrape_url("https://www.iitb.ac.in/en/education/academic-programmes")
open("data/raw/academics.txt", "w").write(doc["text"])
```

### 4. Build the index
```bash
python build_index.py
```

### 5. Run the assistant
```bash
streamlit run app.py
```
…or test it headless:
```bash
python ask.py "What does WnCC's Learners' Space cover?"
```

---

## 🔧 How it stays honest (no hallucination)

Two independent guardrails:

1. **Retrieval-score gate** (`config.MIN_SCORE`): if the best passage's cosine
   similarity is below the threshold, the assistant returns *"I don't know"*
   **without calling the LLM** — it can't hallucinate what it never sends.
2. **Strict system prompt**: the model is instructed to answer *only* from the
   provided context, cite `[Source N]`, and otherwise refuse.

Tune the threshold live from the sidebar or in `config.py`.

---

## ⚙️ Configuration knobs (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 600 | Characters per chunk |
| `CHUNK_OVERLAP` | 120 | Overlap between consecutive chunks |
| `TOP_K` | 4 | Passages retrieved per query |
| `MIN_SCORE` | 0.30 | Grounding threshold (below → "I don't know") |
| `EMBED_MODEL_NAME` | all-MiniLM-L6-v2 | sentence-transformers model |
| `LLM_PROVIDER` | gemini | `gemini` / `openai` / `anthropic` |

---

## 📄 License / attribution
Built for educational use as part of WnCC ML LS 2026. Sample documents are
placeholders and must be replaced with official sources.
