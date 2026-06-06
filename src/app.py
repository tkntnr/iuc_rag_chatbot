import streamlit as st
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from rag_engine import hybrid_search, build_context, ask

VECTORDB_DIR = r"C:\Users\hp\iuc-rag-chatbot\vectordb"

st.set_page_config(
    page_title="İÜC Akademik Asistan",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_system():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        VECTORDB_DIR,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    with open(os.path.join(VECTORDB_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(VECTORDB_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return vectorstore, bm25, chunks

# Sidebar
with st.sidebar:
    st.title("⚙️ Ayarlar")
    model_choice = st.selectbox(
        "Model",
        ["gemma3:4b", "llama3.2:3b", "llama3.1:8b-instruct-q4_K_M"],
        index=0
    )
    temperature = st.slider("Sıcaklık", 0.0, 1.0, 0.1, 0.05)
    st.divider()
    if st.button("🗑️ Geçmişi Sil"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("**İÜC Akademik Asistan**")
    st.markdown("RAG tabanlı yapay zeka sistemi")

# Ana başlık
st.title("🎓 İÜC Akademik Asistan")
st.caption("İstanbul Üniversitesi-Cerrahpaşa Bilgi Sistemi")

# Mesaj geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Kaynaklar"):
                for source in message["sources"]:
                    st.markdown(f"- `{source}`")

# Sistem yükle
with st.spinner("Sistem yükleniyor..."):
    vectorstore, bm25, chunks = load_system()
    llm = OllamaLLM(model=model_choice, temperature=temperature)

# Kullanıcı girişi
if query := st.chat_input("Sorunuzu yazın..."):
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Asistan yanıtı
    with st.chat_message("assistant"):
        with st.spinner("Yanıt aranıyor..."):
            result = ask(query, vectorstore, bm25, chunks, llm)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📚 Kaynaklar"):
                for source in result["sources"]:
                    st.markdown(f"- `{source}`")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })