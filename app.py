import streamlit as st
from rag_engine import generate_answer, get_db_chunk_count

st.set_page_config(page_title="Local Knowledge Base RAG", page_icon="📚", layout="wide")

# Initialize session state for chat history and debug info
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_debug" not in st.session_state:
    st.session_state.last_debug = None

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Local RAG System")
    st.caption("Powered by Ollama, ChromaDB & Cross-Encoders")
    st.divider()

    st.metric("Indexed Chunks in DB", get_db_chunk_count())

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_debug = None
        st.rerun()

    st.divider()

    # Show retrieval diagnostics for the latest turn
    if st.session_state.last_debug:
        debug = st.session_state.last_debug
        st.subheader("🔍 Latest Retrieval Details")
        st.markdown(f"**Standalone Search Query:**\n> {debug['rewritten_query']}")
        st.markdown(f"**Sources Used:** `{', '.join(debug['sources'])}`")

        with st.expander("View Reranked Context Chunks"):
            for idx, (chunk, meta, score) in enumerate(debug["chunks"], start=1):
                src = meta.get("source", "unknown") if meta else "unknown"
                st.markdown(f"**#{idx} — `{src}`** (Rerank Score: `{score:.2f}`)")
                st.caption(chunk)
                st.divider()

# --- Main Chat Interface ---
st.header("Chat with Your Documents")

# Display prior messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"📖 Sources: {', '.join(message['sources'])}")

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching & reranking documents..."):
            result = generate_answer(st.session_state.messages, prompt)

        st.markdown(result["answer"])
        if result["sources"]:
            st.caption(f"📖 Sources: {', '.join(result['sources'])}")

    # Save turn to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
    st.session_state.last_debug = result
    st.rerun()