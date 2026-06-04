import pickle
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi

VECTORDB_DIR = r"C:\Users\hp\iuc-rag-chatbot\vectordb"

SYSTEM_PROMPT = """Sen İstanbul Üniversitesi-Cerrahpaşa'nın resmi akademik asistanısın.
Sana verilen bağlam belgelerini kullanarak öğrencilerin sorularını yanıtla.
Yanıtların her zaman:
- Türkçe olmalı
- Yalnızca verilen belgelere dayanmalı
- Kısa, net ve anlaşılır olmalı
- Kaynak belirtmeli (hangi yönetmelik/yönerge)
Eğer bilgi belgelerinde yoksa "Bu konuda bilgim bulunmamaktadır, lütfen öğrenci işleri ile iletişime geçin." de.
"""

def load_indexes():
    print("İndeksler yükleniyor...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cuda"}
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
    print("İndeksler yüklendi!")
    return vectorstore, bm25, chunks

def hybrid_search(query, vectorstore, bm25, chunks, k=5, alpha=0.4):
    faiss_results = vectorstore.similarity_search_with_score(query, k=20)
    faiss_scores = {}
    for doc, score in faiss_results:
        chunk_id = doc.metadata.get("chunk_id", "")
        faiss_scores[chunk_id] = (1 - score, doc)

    tokenized_query = query.lower().split()
    bm25_scores_raw = bm25.get_scores(tokenized_query)
    max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
    bm25_normalized = bm25_scores_raw / max_bm25

    final_scores = {}
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["metadata"]["chunk_id"]
        faiss_score = faiss_scores.get(chunk_id, (0, None))[0]
        bm25_score = float(bm25_normalized[i])

        # Öncelikli PDF'lere bonus puan ver
        priority_sources = [
            "411.1y_iuc-onlisans",
            "iu-cerrahpasa-onlisans-ve-lisans-yonetmeligi-web"
        ]
        bonus = 0
        for ps in priority_sources:
            if ps in chunk["metadata"]["source"]:
                bonus = 0.3
                break

        final_scores[chunk_id] = (
            alpha * faiss_score + (1 - alpha) * bm25_score + bonus,
            chunk
        )

    sorted_results = sorted(final_scores.items(), key=lambda x: x[1][0], reverse=True)
    top_chunks = [item[1][1] for item in sorted_results[:k]]
    return top_chunks

    sorted_results = sorted(final_scores.items(), key=lambda x: x[1][0], reverse=True)
    top_chunks = [item[1][1] for item in sorted_results[:k]]
    return top_chunks

def build_context(chunks):
    context_parts = []
    for chunk in chunks:
        source = chunk["metadata"].get("source", "Bilinmiyor")
        content = chunk["content"]
        context_parts.append(f"[Kaynak: {source}]\n{content}")
    return "\n\n---\n\n".join(context_parts)

def ask(query, vectorstore, bm25, chunks, llm):
    top_chunks = hybrid_search(query, vectorstore, bm25, chunks)
    context = build_context(top_chunks)

    prompt = f"""{SYSTEM_PROMPT}

BAĞLAM BELGELERİ:
{context}

ÖĞRENCİ SORUSU: {query}

YANIT:"""

    response = llm.invoke(prompt)
    sources = list(set([c["metadata"].get("source", "") for c in top_chunks]))

    return {
        "answer": response,
        "sources": sources,
        "chunks": top_chunks
    }

if __name__ == "__main__":
    vectorstore, bm25, chunks = load_indexes()
    llm = OllamaLLM(model="gemma3:4b", temperature=0.1)

    print("\nİÜC Akademik Asistan hazır! (Çıkmak için 'quit' yazın)\n")
    while True:
        query = input("Sorunuz: ").strip()
        if query.lower() in ["quit", "exit", "çıkış"]:
            break
        if not query:
            continue

        print("\nYanıt aranıyor...")
        result = ask(query, vectorstore, bm25, chunks, llm)

        print(f"\n{'='*50}")
        print(f"YANIT:\n{result['answer']}")
        print(f"\nKAYNAKLAR:")
        for s in result['sources']:
            print(f"  - {s}")
        print(f"{'='*50}\n")