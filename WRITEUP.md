# IITB Insti-Assist — Write-up

WnCC Machine Learning Learners' Space 2026, final project.

## The scope I picked, and why

I went with the General Insti Assistant, the broad one. The brief warns it's harder to pull off in a week, and honestly that's exactly why I chose it. A narrow bot that only knows the grading policy is easy, but it never really tests retrieval, because almost any question lands on the same document. With a mixed bag of documents the retriever actually has to decide which one is relevant, and the "I don't know" guardrail starts doing real work instead of being decoration. It felt like the version of the project that would teach me the most.

## What went into the knowledge base

Nine real IITB documents, around 100 pages between them. Eight are official PDFs and one is a plain-text overview I put together for the general campus stuff the rulebooks don't cover:

- UG rulebook (the big one, 52 pages)
- Academic calendar 2026-27
- Dissertation rules
- IDDDP guidelines
- Rules for medals and academic prizes
- Disciplinary procedures
- Disciplinary punishments
- Student medical rules
- A general institute overview (departments, campus, hostels, Gymkhana, admissions)

Before indexing anything I checked every PDF actually had selectable text and wasn't a scanned image, since a scanned PDF would just embed as nothing and quietly break the whole thing. All nine came back clean.

## How I chunked, and the bug that made me change it

I started with small chunks, 600 characters. That worked fine for the prose documents like the rulebook, but the academic calendar wrecked it. The calendar is basically a big table, and when it got sliced up the section headings ("Autumn Semester", "Summer Term") got separated from the actual dates. So I'd ask when the end-sem exams were, and the bot would confidently give me a date from the summer term instead, because the chunk it found just said "Term-end examination, 14 July" with no clue which term that even belonged to. It looked grounded. It was wrong.

Bumping the chunk size up to 1200 characters, with 200 characters of overlap, mostly fixed it, because now a term's heading tends to stay attached to its dates. That took the corpus down to 247 chunks. It's still not perfect, tables are just hard, but it's a lot better. The overlap matters too, otherwise a fact that happens to sit right on a chunk boundary can go missing from both chunks.

## Embeddings, search, and the model

I embed with BAAI/bge-small-en-v1.5. I actually started with all-MiniLM-L6-v2, which is the usual default, but the answers weren't great, so I swapped to the BGE model. Same size and speed roughly, noticeably better at finding the right chunk. One quirk: BGE wants a little instruction prefix stuck on the front of the query (not the documents), so I added that on the query side only.

The vectors are normalised and stored in a FAISS flat index, so search is just cosine similarity. It grabs the top 4 chunks per question. For generation I'm using Gemini 2.5-flash on the free tier, at a low temperature so it stays factual.

## Keeping it honest

Two things stop it hallucinating. The first is a score cutoff: if the best chunk it finds scores below 0.30, it doesn't even bother calling Gemini, it just says it doesn't know. You can't hallucinate from a document you never sent. The second is the prompt itself, which tells the model to stick to the passages it's given, to say which semester or term a date belongs to, to list the options when a question is ambiguous instead of picking one at random, and to cite its sources. The app shows a grounded-or-not badge and the confidence score on every answer, and you can expand the sources to see the exact chunks it used.

## Stretch goals

I did all four: showing the source chunks with scores, remembering previous turns for follow-ups, live PDF upload from the sidebar, and the grounded/confidence badge.

## What's still weak, and what I'd do with more time

The calendar is the obvious weak spot. Even at 1200-character chunks, dates buried in tables just don't embed as cleanly as normal sentences, so it's the least reliable part of the bot. A proper fix would be a chunker that understands table layout, or honestly just hand-cleaning the calendar into plain text with the term written on every row.

The bigger lesson was that "grounded" and "correct" aren't the same thing. A chunk can score high and still be the wrong section, like the summer-vs-autumn exam mix-up. The standard fix is a re-ranker: pull a bigger batch of candidates, then use a second model to reorder them and keep the best few. I didn't have time to add it but that's the first thing I'd build next.

A few other things I'd want: some lexical search (BM25) alongside the vector search, so exact terms like course codes or rule numbers match better; a small set of test questions with known answers so I could actually measure accuracy instead of eyeballing it; and summarising old conversation turns instead of stuffing them all into the prompt, which would help once a chat gets long.

The threshold and top-k I landed on (0.30 and 4) worked well for these documents, but they're hand-tuned to this corpus. Set the threshold too low and it starts answering off-topic questions; too high and it refuses real ones. Finding that line was most of the tuning.
