"""
IITB Insti-Assist — Streamlit UI.

Features (core + all four stretch goals):
  * Chat interface grounded in the document corpus
  * [Stretch] Source citation highlighting — exact chunk + similarity score
  * [Stretch] Multi-turn memory — follow-up questions use recent history
  * [Stretch] Live PDF upload — add a PDF at runtime, merged into the index
  * [Stretch] Grounded / confidence indicator — badge on every answer
"""
from __future__ import annotations

import streamlit as st

import config
from rag.chunking import chunk_documents
from rag.embedding import embed_texts
from rag.ingest import load_pdf
from rag.pipeline import RAGPipeline
from rag.vectorstore import VectorStore

st.set_page_config(page_title="IITB Insti-Assist", page_icon="🎓", layout="centered")


# ---------- resource loading (cached across reruns) ----------
@st.cache_resource(show_spinner="Loading the knowledge base…")
def load_base_store():
    return VectorStore.load()  # None if the index hasn't been built yet


def get_session_store():
    """A per-session copy of the base index so live PDF uploads don't persist
    globally or across users."""
    if "store" not in st.session_state:
        base = load_base_store()
        st.session_state.store = base.clone() if base is not None else None
    return st.session_state.store


# ---------- state ----------
if "history" not in st.session_state:
    st.session_state.history = []          # list of (user, assistant) tuples
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = set()


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    store = get_session_store()

    if store is None:
        st.error("No index found. Run `python build_index.py` after adding "
                 "documents to `data/raw/`.")
    else:
        st.success(f"Knowledge base: {store.size} chunks indexed")

    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, config.TOP_K)
    min_score = st.slider("Grounding threshold", 0.0, 0.9, config.MIN_SCORE, 0.05,
                          help="If the best passage scores below this, the "
                               "assistant refuses instead of guessing.")

    st.divider()
    st.subheader("📎 Add a PDF (this session)")
    uploaded = st.file_uploader("Upload a PDF to query it live", type=["pdf"])
    if uploaded is not None and uploaded.name not in st.session_state.uploaded_names:
        if store is None:
            store = VectorStore()
            st.session_state.store = store
        with st.spinner(f"Indexing {uploaded.name}…"):
            tmp = config.INDEX_DIR / f"_upload_{uploaded.name}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(uploaded.read())
            text = load_pdf(tmp)
            tmp.unlink(missing_ok=True)
            chunks = chunk_documents([{"text": text, "source": uploaded.name}])
            if chunks:
                vecs = embed_texts([c["text"] for c in chunks])
                store.add(vecs, chunks)
                st.session_state.uploaded_names.add(uploaded.name)
                st.success(f"Added {len(chunks)} chunks from {uploaded.name}")
            else:
                st.warning("Couldn't extract text from that PDF.")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.rerun()

    st.caption(f"LLM provider: **{config.LLM_PROVIDER}**")


# ---------- main ----------
st.title("🎓 IITB Insti-Assist")
st.caption("A retrieval-grounded assistant for IIT Bombay. It answers only "
           "from its documents — and says *\"I don't know\"* when it can't.")

# replay history
for user_msg, asst_msg in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(asst_msg)

question = st.chat_input("Ask about academics, hostels, clubs, campus life…")

if question:
    store = get_session_store()
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if store is None or store.size == 0:
            st.markdown("I don't have any documents indexed yet. Build the "
                        "index or upload a PDF from the sidebar.")
        else:
            # apply sidebar overrides
            config.MIN_SCORE = min_score
            pipe = RAGPipeline(store)
            with st.spinner("Retrieving and thinking…"):
                result = pipe.answer(
                    question, history=st.session_state.history, k=top_k
                )

            # [Stretch] grounded / confidence badge
            if result["grounded"]:
                st.markdown(f"✅ **Grounded** · confidence `{result['confidence']:.2f}`")
            else:
                st.markdown(f"⚠️ **Not grounded** · confidence `{result['confidence']:.2f}`")

            st.markdown(result["answer"])

            # [Stretch] source citation highlighting
            if result["sources"]:
                with st.expander(f"📚 Sources ({len(result['sources'])} passages)"):
                    for i, s in enumerate(result["sources"], 1):
                        st.markdown(
                            f"**[Source {i}] {s['source']}** — "
                            f"similarity `{s['score']:.3f}`"
                        )
                        st.info(s["text"])

            # [Stretch] multi-turn memory
            st.session_state.history.append((question, result["answer"]))
