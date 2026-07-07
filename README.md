# IITB Insti-Assist

A chatbot that answers questions about IIT Bombay from actual institute documents instead of guessing. Ask it about the grading rules, exam dates, medical leave, disciplinary stuff, and it pulls the answer straight out of the official PDFs. If the answer isn't in there, it just says "I don't know" instead of making something up.

This is my final project for the WnCC Machine Learning Learners' Space 2026. The scope I picked is a General Insti Assistant, so the documents cover a bit of everything: academics, exams, discipline, campus info.

## Why RAG

A plain LLM like ChatGPT knows nothing about IITB's specific rules. It'll happily invent a grading policy that sounds right and is completely wrong. RAG fixes that by fetching the relevant document chunks at question time and handing them to the model, so the answer comes from real text and not from the model's imagination.

## What it does

The basics are all there: it loads the documents, splits them into chunks, embeds them, stores them in a FAISS index, and searches that index for whatever's relevant to your question. Then it feeds those chunks to Gemini and shows you the answer plus the exact passages it used.

I also built all four of the optional stretch goals:

- It shows the exact source chunks behind every answer, with their similarity scores, so you can check its work.
- It remembers earlier turns in the conversation, so follow-up questions work.
- You can upload a PDF from the sidebar and ask about it on the spot.
- Every answer gets a little badge saying whether it's actually grounded in the docs or not.

## The documents

Nine real IITB documents live in `data/raw/`, roughly 100 pages total:

| File | What's in it |
|---|---|
| UG_RULE_BOOK.pdf | Undergraduate rules and regulations |
| Academic_Calendar_2026-27_FINAL.pdf | Semester dates, registration, exam schedule |
| Dissertation17june09-10.pdf | Dissertation rules |
| IDDDP_Guidelines_2025.pdf | Interdisciplinary Dual Degree Programme |
| RulesforAwardofMedalsandAcademicprizesforUGandPG.pdf | Medals and academic prizes |
| procedures201521July.pdf | Disciplinary procedures |
| punishments201521July.pdf | Disciplinary punishments |
| student-medical-rules.pdf | Student medical rules |
| 01_iitb_overview.txt | General overview: departments, campus, Gymkhana, admissions |

## Folder layout

```
iitb-insti-assist/
├── app.py                # the Streamlit chat app
├── build_index.py        # builds the FAISS index from data/raw/
├── ask.py                # quick command-line tester
├── config.py             # all the settings live here
├── requirements.txt
├── .env.example          # copy this to .env and add your key
├── rag/
│   ├── ingest.py         # reads txt/md/pdf/html, plus a scraper
│   ├── chunking.py       # splits documents into chunks
│   ├── embedding.py      # turns text into vectors
│   ├── vectorstore.py    # the FAISS wrapper
│   ├── llm.py            # the Gemini call
│   └── pipeline.py       # ties retrieval + prompt + grounding together
└── data/
    ├── raw/              # the source documents
    └── index/            # the built index (you generate this)
```

## Getting it running

Install everything first. I'd use a virtual environment so it doesn't mess with your system Python.

```bash
cd iitb-insti-assist
python -m venv .venv
source .venv/bin/activate      # on Windows it's .venv\Scripts\activate
pip install -r requirements.txt
```

Then get a free Gemini key from https://aistudio.google.com/app/apikey. Copy the example env file and paste your key in:

```bash
cp .env.example .env
```

Open `.env` and set these two lines:

```
GEMINI_API_KEY=your_key_here
LLM_MODEL=gemini-2.5-flash
```

Build the index (the first run downloads the embedding model, about 130 MB, so give it a minute):

```bash
python build_index.py
```

And launch it:

```bash
python -m streamlit run app.py
```

If you'd rather test without the UI:

```bash
python ask.py "What are the rules for student medical leave?"
```

One thing that tripped me up a lot: if you're on Anaconda, plain `streamlit run app.py` sometimes grabs the wrong Python and throws a "no module named faiss" error. The fix that always worked for me was launching through the venv's Python directly:

```bash
.venv/bin/python -m streamlit run app.py
```

## Settings worth knowing

Everything's in `config.py`, but the ones I actually tuned:

| Setting | I used | What it does |
|---|---|---|
| EMBED_MODEL_NAME | BAAI/bge-small-en-v1.5 | the embedding model |
| CHUNK_SIZE | 1200 | characters per chunk |
| CHUNK_OVERLAP | 200 | overlap between chunks |
| TOP_K | 4 | how many chunks it retrieves |
| MIN_SCORE | 0.30 | grounding cutoff, below this it says "I don't know" |
| LLM_MODEL | gemini-2.5-flash | the Gemini model |

You can also change top-k and the grounding cutoff live from the sidebar while the app is running, which is handy for testing.

## How it avoids making things up

Two guards. First, before it even calls the LLM, it checks the best chunk's similarity score. If nothing clears the 0.30 cutoff, it refuses right there, so it can't hallucinate from context it never received. Second, the system prompt tells the model to answer only from the passages it's given, cite the source number, and say it doesn't know otherwise.

## A note on the sources

These are real IITB documents used for an academic project. Nothing here is official or maintained, so don't rely on the bot for actual rule-checking. Read the real documents for that.
