import requests
import chromadb
from sentence_transformers import CrossEncoder

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

# Connect to Chroma and load reranker
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_embedding(text):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": text})
    return r.json()["embeddings"][0]

def ask_llm(messages_list):
    """Sends the full conversation history to Ollama for context-aware responses."""
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, 
        "stream": False, 
        "messages": messages_list
    })
    return r.json()["message"]["content"]

def rewrite_query(chat_history, latest_question):
    """
    Takes vague follow-up questions (e.g., 'What about that policy?') and 
    turns them into clear, standalone search queries using conversation history.
    """
    if not chat_history:
        return latest_question

    # Format recent history for the rewriter
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-4:]])
    
    rewrite_prompt = f"""Given the conversation history below, rewrite the user's latest follow-up question into a clear, standalone search query that includes all necessary context (like specific topics, policies, or names mentioned earlier). Do not answer the question, just output the rewritten standalone query.

Conversation History:
{history_text}

Latest Follow-up Question: {latest_question}
Standalone Search Query:"""

    # Ask the LLM to rewrite the query
    res = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": rewrite_prompt}]
    })
    
    standalone_query = res.json()["message"]["content"].strip()
    print(dk_format := f"🔍 [Query Rewritten]: '{latest_question}' ➔ '{standalone_query}'")
    return standalone_query

def retrieve_and_rerank(question, initial_k=10, final_k=3):
    q_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=initial_k)
    
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    if not chunks:
        return [], []

    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    scored_results = sorted(zip(chunks, metadatas, scores), key=lambda x: x[2], reverse=True)
    
    top_chunks = [item[0] for item in scored_results[:final_k]]
    top_metadatas = [item[1] for item in scored_results[:final_k]]
    return top_chunks, top_metadatas

print(f"🚀 Conversational RAG with Memory & Query Rewriting Ready!\n")

# Maintain chat history across turns
chat_history = []

while True:
    question = input("Ask a question (or type 'quit' to exit): ")
    if question.lower() == 'quit':
        break
        
    # Step 1: Rewrite the query so it's standalone and avoids ambiguity
    search_query = rewrite_query(chat_history, question)

    # Step 2: Retrieve and rerank based on the clear search query
    retrieved_chunks, metadatas = retrieve_and_rerank(search_query, initial_k=10, final_k=3)
    context = "\n\n".join(retrieved_chunks)
    
    # Step 3: Build prompt containing context and add user question to history
    system_prompt = f"""Answer the user's question based strictly on the provided context below.
If the answer is not found in the context, respond with "I don't know."

Context:
{context}"""

    # Build the message payload for Ollama including prior history
    messages_payload = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages_payload.append(msg)
    
    # Add the current user question
    messages_payload.append({"role": "user", "content": question})

    # Step 4: Get final response from LLM
    answer = ask_llm(messages_payload)
    sources = set(m.get("source", "unknown") for m in metadatas if m and isinstance(m, dict))
    
    print(f"\n{answer}\n")
    print(f"📖 Sources: {list(sources)}")
    print("-" * 40)

    # Update chat history log
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})