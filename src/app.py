import streamlit as st
import os
import shutil
from src.core.agent import PrivateCFO
from src.ingestion.loader import IngestionManager
from src.ingestion.vector_db import VectorDB
import gc

# 1. Page Configuration
st.set_page_config(
    page_title="Private CFO Agent",
    page_icon="🔒",
    layout="wide"
)

# 2. Session State Management
# Streamlit refreshes the script on every click. 
# We use st.session_state to remember things between refreshes.

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    # Initialize the Brain once
    st.session_state.agent = PrivateCFO()

# 3. The Sidebar: Data Ingestion
with st.sidebar:
    st.header("📂 Secure Data Vault")
    st.info("Files uploaded here are processed LOCALLY. No data leaves your machine.")
    
    uploaded_file = st.file_uploader("Upload Financial Report (PDF/Excel)", type=["pdf", "xlsx"])
    
    if uploaded_file:
        # Save file temporarily
        temp_path = f"data/raw/{uploaded_file.name}"
        os.makedirs("data/raw", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🧠 Ingest & Memorize"):
            with st.status("Processing Document...", expanded=True) as status:
                try:
                    # CRITICAL STEP 1: Release the Lock
                    # If an agent exists, it's holding the DB open. Kill it.
                    if "agent" in st.session_state:
                        del st.session_state.agent
                        st.write("🧹 Clearing old memory locks...")
                    
                    gc.collect()
                    # A. Load
                    st.write("📖 Reading file...")
                    loader = IngestionManager()
                    docs = loader.load_file(temp_path)
                    
                    # B. Embed & Store
                    st.write("🔢 Vectorizing content...")
                    vdb = VectorDB(reset=True) 
                    vdb.ingest_documents(docs)
                    
                    # CRITICAL STEP 2: Force Reload
                    # We deleted the agent above. The next time the script runs (in 1ms),
                    # it will re-initialize the Agent with the NEW database.
                    
                    status.update(label="✅ Knowledge Base Updated!", state="complete", expanded=False)
                    st.success(f"Ingested {len(docs)} pages. The Agent is ready.")
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# 4. The Main Chat Interface
st.title("🔒 Private CFO Agent")
st.markdown("""
* **Sovereign Mode:** Answers sensitive questions using local Ollama (Mistral).
* **Cloud Mode:** Answers general math/coding questions using Cloud Brain.
""")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "source" in msg:
            st.caption(f"Source: {msg['source']}")

# Handle User Input
if prompt := st.chat_input("Ask about your data or financial theory..."):
    # A. Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the Orchestrator
            response_payload = st.session_state.agent.process_query(prompt)
            
            answer = response_payload["answer"]
            source = response_payload["source"]
            
            st.markdown(answer)
            st.caption(f"Source: {source}")
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "source": source
            })