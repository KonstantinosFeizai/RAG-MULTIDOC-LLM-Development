import requests
import chromadb
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

# Connect to persistent Chroma database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# Load Cross-Encoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_db_chunk_count() -> int:
    """Returns the total number of indexed chunks currently in ChromaDB."""
    return collection.count()


def get_all_chroma_docs():
    """Fetches all documents and metadatas from ChromaDB to build the BM25 index."""
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    return documents, metadatas


# Build BM25 index dynamically from ChromaDB content at startup
all_docs, all_metadatas = get_all_chroma_docs()
tokenized_corpus = [doc.lower().split() for doc in all_docs] if all_docs else [[]]
bm25_index = BM25Okapi(tokenized_corpus) if all_docs else None
all_docs_with_meta = list(zip(all_docs, all_metadatas))


def get_embedding(text: str) -> list[float]:
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": text})
    return r.json()["embeddings"][0]


def rewrite_query(chat_history: list[dict], latest_question: str) -> str:
    """Rewrites follow-up questions into standalone search queries using chat history."""
    if not chat_history:
        return latest_question

    history_text = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]]
    )

    rewrite_prompt = f"""Given the conversation history below, rewrite the user's latest follow-up question into a clear, standalone search query that includes all necessary context (like specific topics, policies, or names mentioned earlier). Do not answer the question, just output the rewritten standalone query.

Conversation History:
{history_text}

Latest Follow-up Question: {latest_question}
Standalone Search Query:"""

    res = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": rewrite_prompt}],
        },
    )
    return res.json()["message"]["content"].strip()


def reciprocal_rank_fusion(vector_results, bm25_results, k_constant=60):
    """Combines vector search ranks and BM25 keyword ranks using RRF."""
    fused_scores = {}

    def add_ranks(ranked_items):
        for rank, (doc, meta) in enumerate(ranked_items):
            if doc not in fused_scores:
                fused_scores[doc] = {"score": 0.0, "meta": meta}
            fused_scores[doc]["score"] += 1.0 / (k_constant + (rank + 1))

    add_ranks(vector_results)
    add_ranks(bm25_results)

    sorted_fusion = sorted(fused_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return [(item[0], item[1]["meta"]) for item in sorted_fusion]


def retrieve_and_rerank(question: str, candidate_pool_size: int = 15, final_k: int = 3):
    """Performs Hybrid Search (Vector Search + BM25 Keyword Search) + RRF, then Reranks."""
    global bm25_index, all_docs, all_metadatas, all_docs_with_meta
    
    current_count = collection.count()
    if len(all_docs) != current_count:
        all_docs, all_metadatas = get_all_chroma_docs()
        all_docs_with_meta = list(zip(all_docs, all_metadatas))
        tokenized_corpus = [doc.lower().split() for doc in all_docs] if all_docs else [[]]
        bm25_index = BM25Okapi(tokenized_corpus) if all_docs else None

    if not all_docs:
        return [], [], []

    # 1. Vector Search (Semantic)
    q_embedding = get_embedding(question)
    vec_results = collection.query(query_embeddings=[q_embedding], n_results=min(candidate_pool_size, len(all_docs)))
    vec_chunks = vec_results["documents"][0]
    vec_metas = vec_results["metadatas"][0]
    vector_ranked = list(zip(vec_chunks, vec_metas))

    # 2. BM25 Search (Keyword)
    tokenized_query = question.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    best_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_pool_size]
    bm25_ranked = [(all_docs[i], all_metadatas[i]) for i in best_bm25_indices if bm25_scores[i] > 0]

    # 3. Reciprocal Rank Fusion
    fused_candidates = reciprocal_rank_fusion(vector_ranked, bm25_ranked)
    
    pool_chunks = [item[0] for item in fused_candidates[:candidate_pool_size]]
    pool_metas = [item[1] for item in fused_candidates[:candidate_pool_size]]

    if not pool_chunks:
        return [], [], []

    # 4. Stage 2: Cross-Encoder Reranking
    pairs = [[question, chunk] for chunk in pool_chunks]
    scores = reranker.predict(pairs)
    scored_results = sorted(
        zip(pool_chunks, pool_metas, scores), key=lambda x: x[2], reverse=True
    )

    top_chunks = [item[0] for item in scored_results[:final_k]]
    top_metadatas = [item[1] for item in scored_results[:final_k]]
    top_scores = [float(item[2]) for item in scored_results[:final_k]]
    
    return top_chunks, top_metadatas, top_scores


def generate_answer(chat_history: list[dict], latest_question: str):
    """Full Hybrid RAG pipeline returning answer, rewritten query, sources, and retrieved chunks."""
    search_query = rewrite_query(chat_history, latest_question)
    retrieved_chunks, metadatas, scores = retrieve_and_rerank(search_query)

    context = "\n\n".join(retrieved_chunks)
    system_prompt = f"""Answer the user's question based strictly on the provided context below.
If the answer is not found in the context, respond with "I don't know."

Context:
{context}"""

    messages_payload = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages_payload.append({"role": msg["role"], "content": msg["content"]})
    messages_payload.append({"role": "user", "content": latest_question})

    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "messages": messages_payload,
        },
    )
    answer = r.json()["message"]["content"]
    sources = sorted(
        list(set(m.get("source", "unknown") for m in metadatas if m and isinstance(m, dict)))
    )

    return {
        "answer": answer,
        "rewritten_query": search_query,
        "sources": sources,
        "chunks": list(zip(retrieved_chunks, metadatas, scores)),
    }