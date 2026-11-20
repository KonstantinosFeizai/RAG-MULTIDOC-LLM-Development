import json
import requests
import chromadb
from sentence_transformers import CrossEncoder

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

# Connect to Chroma and load the reranker
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def get_embedding(text):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": text})
    return r.json()["embeddings"][0]

def ask_llm(prompt):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, 
        "stream": False, 
        "messages": [{"role": "user", "content": prompt}]
    })
    return r.json()["message"]["content"]

def retrieve_and_rerank(question, initial_k=10, final_k=3):
    q_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=initial_k)
    
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    if not chunks:
        return []

    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    scored_results = sorted(zip(chunks, metadatas, scores), key=lambda x: x[2], reverse=True)
    
    return [item[0] for item in scored_results[:final_k]]

def run_evaluations():
    # Load test dataset
    try:
        with open("eval_dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("❌ Error: eval_dataset.json not found!")
        return

    print(f"📊 Running evaluations on {len(dataset)} test cases...\n")
    
    passed = 0
    failed = 0

    for test in dataset:
        qid = test["id"]
        question = test["question"]
        keywords = test["required_keywords"]

        print(f"--- Test Case #{qid}: {question}")
        
        # Run pipeline
        retrieved_chunks = retrieve_and_rerank(question)
        context = "\n\n".join(retrieved_chunks)
        
        prompt = f"""Answer the question based on the context below.
If the answer is not found in the context, respond with "I don't know."

Context:
{context}

Question: {question}"""

        answer = ask_llm(prompt)
        print(f"🤖 Answer: {answer}")

        # Simple keyword-based validation (deterministic and fast)
        missing_keywords = [kw for kw in keywords if kw.lower() not in answer.lower()]

        if not missing_keywords:
            print("✅ Status: PASSED")
            passed += 1
        else:
            print(f"❌ Status: FAILED (Missing keywords: {missing_keywords})")
            failed += 1
        print("-" * 50)

    # Final Scorecard Summary
    total = passed + failed
    score = (passed / total) * 100 if total > 0 else 0
    print(f"\n🎯 EVALUATION RESULTS:")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Final Accuracy Score: {score:.1f}%")

if __name__ == "__main__":
    run_evaluations()