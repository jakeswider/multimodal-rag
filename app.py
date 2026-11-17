import os

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from ingest.embedder import get_embedder
from ingest.store import get_vectorstore
from rag.generator import build_llm, generate_answer_stream
from rag.retriever import retrieve

load_dotenv()

st.set_page_config(page_title="Multimodal RAG", page_icon="📄")
st.title("Multimodal RAG (pgvector + LangChain)")


@st.cache_resource
def load_pipeline():
    database_url = os.environ["DATABASE_URL"]
    embedder = get_embedder()
    vectorstore = get_vectorstore(embedder, database_url)
    llm = build_llm(os.environ.get("OLLAMA_MODEL", "qwen2.5vl"))
    return embedder, vectorstore, llm


question = st.text_input("Ask a question about your ingested PDFs:")

if question:
    try:
        embedder, vectorstore, llm = load_pipeline()
    except Exception as exc:
        st.error(f"Could not connect to the vectorstore or Ollama: {exc}")
        st.stop()

    with st.spinner("Retrieving relevant context..."):
        try:
            context_docs = retrieve(vectorstore, embedder, question, k=4)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()

    st.subheader("Answer")
    try:
        st.write_stream(generate_answer_stream(llm, question, context_docs))
    except Exception as exc:
        st.error(
            f"Answer generation failed. Is Ollama running (`ollama serve`)? Details: {exc}"
        )
        st.stop()

    st.subheader("Sources")
    for doc in context_docs:
        source = doc.metadata.get("source_pdf", "unknown")
        page = doc.metadata.get("page_number", "?")
        if doc.metadata.get("element_type") == "text":
            st.markdown(f"**{source}, page {page}**")
            st.write(doc.page_content)
        elif doc.metadata.get("element_type") == "image":
            st.markdown(f"**{source}, page {page}** (image)")
            try:
                st.image(Image.open(doc.page_content), width=200)
            except Exception:
                st.warning(f"Image not available: {doc.page_content}")
