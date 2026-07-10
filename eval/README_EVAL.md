# Insti-Assist — Evaluation (add this to your repo)

This adds a real, measurable evaluation to Insti-Assist. It reports numbers you
can defend in an interview and put on your resume.

## What it measures
- **Retrieval** — Hit@k, MRR: does the correct source document appear in the top-k?
- **Threshold ablation** — sweeps `MIN_SCORE` to show the refusal tradeoff
  (out-of-scope refusal accuracy vs. wrongly refusing answerable questions).
  This empirically *justifies* your `MIN_SCORE=0.30` choice.
- **Faithfulness** — LLM-as-judge: is each answer actually supported by the
  retrieved context? (no gold answer needed)
- **Correctness** — LLM-as-judge vs. gold answers (only for items where you fill
  in `answer_key`; skipped otherwise).

## Setup (30 seconds)
1. Copy this folder into your repo as `eval/`:
   ```
   Insti-Assistant/
   ├── eval/
   │   ├── evaluate.py
   │   ├── eval_dataset.json
   │   └── README_EVAL.md
   ├── rag/ ...
   ├── config.py
   └── build_index.py
   ```
2. Make sure your index is built: `python build_index.py`

## Run it
```bash
# Retrieval + refusal ablation only — NO API key needed, ~10 seconds
python eval/evaluate.py --mode retrieval

# Full run: also faithfulness (+ correctness if you filled answer_keys)
# needs GEMINI_API_KEY in your .env  (uses your existing rag/llm.py)
python eval/evaluate.py --mode full
```
Outputs: prints a summary, and writes `eval/eval_results.json` + `eval/eval_report.md`.

## To also get a Correctness number (optional but strong)
Open `eval_dataset.json` and fill the `answer_key` for as many in-scope
questions as you can, using the REAL text from the docs in `data/raw/`.
Do NOT guess — leave it `""` if unsure; those items are simply skipped for
correctness (retrieval/faithfulness/refusal still run on them).

## What to send me
Paste the printed summary (or `eval_report.md`). I need:
- Hit@k and MRR at your configured k
- the k-sweep table
- the MIN_SCORE ablation table
- faithfulness % (and correctness % if you scored it)

I'll drop the real numbers into your resume bullets below.

---

## Resume bullets (fill the [X] once you run it)

**One-page version (3 bullets):**
- Built a Retrieval-Augmented Generation (RAG) chatbot answering IIT Bombay policy
  questions grounded in 9 official documents, with a FAISS retriever, bge embeddings,
  and Gemini generation
- Added an **evaluation suite** (35 labelled Qs): achieved **[X]% Hit@4 retrieval**,
  **[X]% faithfulness**, and **[X]% out-of-scope refusal accuracy**
- Tuned the grounding threshold via ablation, cutting hallucinated answers while
  keeping the false-refusal rate under **[X]%**

**Two-page version (add):**
- Ran a MIN_SCORE / top-k ablation across the labelled set to select operating point
  (MIN_SCORE=0.30, k=4), trading off refusal accuracy vs. answerable-question recall

> Note: only claim "eliminates hallucination" AFTER the faithfulness number backs it.
> If faithfulness is, say, 90%, say "grounded [90]% of answers in source context."
