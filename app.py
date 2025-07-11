# Streamlit Chatbot Deployment with Streaming
import streamlit as st
import time
from src.rag_pipeline import RAGPipeline

# Reset chat session if flagged
if "clear_chat" in st.session_state and st.session_state.clear_chat:
    st.session_state.clear_chat = False
    st.session_state.messages = []
    st.session_state.source_chunks = [] 

# Initialize the RAG pipeline
@st.cache_resource
def init_rag():
    return RAGPipeline(vectordb_path="vectordb")

rag = init_rag()

# --- App Title ---
st.title("🤖 eBay Policy Assistant")
st.markdown(
    "Welcome! I'm your AI assistant trained on eBay's Terms & Conditions and User Agreement. "
    "Ask me anything about account usage, listing policies, buyer responsibilities, or platform policies."
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_chunks" not in st.session_state:
    st.session_state.source_chunks = []

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# --- User Input ---
user_input = st.chat_input("Type your question here...")
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # RAG pipeline operations
    top_chunks = rag.search(user_input)
    prompt_msgs, used_chunks = rag.build_prompt(user_input, top_chunks)
    st.session_state.source_chunks = used_chunks  

    # Typing + Streaming 
    with st.chat_message("assistant", avatar="🤖"):
        display = st.empty()

        # Typing animation
        for i in range(3):
            display.markdown("*Typing" + "." * (i + 1) + "*")
            time.sleep(0.3)

        # Stream model response
        full_response = ""
        for token in rag.stream_answer(prompt_msgs):
            full_response += token
            display.markdown(full_response + "▌")
        display.markdown(full_response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Sidebar Info ---
st.sidebar.title("🔧 Chatbot Settings")

# Model Info 
with st.sidebar.expander("**Model Info**", expanded=False):
    st.markdown("🟢 **Model Used:** `Meta-LLaMA-3-8B-Instruct`")
    st.markdown("**Purpose:** This model is fine-tuned for instruction-following and document-based Q&A.")
    st.markdown(f"**Chunks Indexed:** `{len(rag.chunks)} chunks from eBay policy docs`")
    st.markdown("🟢 _Answers are generated using the most relevant sections of the document retrieved through semantic search._")
    st.markdown("🛑 _Model may not answer correctly if context is insufficient._")

# --- Sidebar: Source highlights ---
if st.session_state.source_chunks:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Source highlights")
    for i, chunk in enumerate(st.session_state.source_chunks, 1):
        snippet = " ".join(chunk.strip().split())
        if len(snippet) > 300:
            snippet = snippet[:297] + "..."
        st.sidebar.markdown(f"**{i}.** {snippet}")

# --- New Chat Button ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 New Chat"):
    st.session_state.clear_chat = True
    st.rerun()
