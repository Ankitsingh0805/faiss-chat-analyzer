import streamlit as st
import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import tempfile

# Set environment variables to avoid torch._classes error
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from rag_chatbot import RAGChatbotEnhanced

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rag_chatbot_streamlit")

# Set page config
st.set_page_config(
    page_title="Document Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("🤖 Document Analyzer")
st.markdown("""
This app provides a conversational interface to a Retrieval-Augmented Generation (RAG) chatbot.
Upload your documents and start asking questions about them!
""")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "data_dir" not in st.session_state:
    st.session_state.data_dir = tempfile.mkdtemp()

if "ready" not in st.session_state:
    st.session_state.ready = False

if "vector_store_path" not in st.session_state:
    st.session_state.vector_store_path = None

# Sidebar for configurations and file uploads
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_path = st.text_input(
        "Model Path", 
        value="models/llama-2-7b-chat.Q5_K_M.gguf",
        help="Path to your LLM model file"
    )
    
    # Embedding model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        options=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ],
        help="Select the embedding model to use"
    )
    
    # Advanced options collapsible
    with st.expander("Advanced Options"):
        chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50, 
                              help="Size of text chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10,
                                 help="Overlap between chunks")
        max_tokens = st.slider("Max Tokens", 128, 1024, 512, 64,
                              help="Maximum tokens for the model to generate")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                               help="Controls randomness: 0 (deterministic) to 1 (creative)")
        top_k = st.slider("Top K Documents", 1, 10, 3, 1,
                         help="Number of relevant documents to retrieve")
    
    # File uploads
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload text files", 
                                     type=["txt", "md", "csv", "json"], 
                                     accept_multiple_files=True)
    
    # Save uploaded files
    if uploaded_files:
        st.text(f"Saving files to temporary directory...")
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            # Save the file to the data directory
            file_path = os.path.join(st.session_state.data_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        st.success(f"Saved {len(uploaded_files)} files to temporary directory!")
    
    # Vector store operations
    st.header("Vector Store")
    
    vector_col1, vector_col2 = st.columns(2)
    
    with vector_col1:
        save_vector_store = st.button("Save Vector Store")
    
    with vector_col2:
        load_vector_store = st.file_uploader("Load Vector Store", type=["faiss"])
    
    # Initialize button
    st.header("Initialize Chatbot")
    init_button = st.button("Initialize RAG Chatbot")
    
    if init_button:
        with st.spinner("Initializing RAG Chatbot..."):
            try:
                # Check if there are files in the data directory
                if not os.listdir(st.session_state.data_dir):
                    st.error("No files found! Please upload some documents first.")
                else:
                    # Initialize the chatbot
                    st.session_state.chatbot = RAGChatbotEnhanced(
                        data_dir=st.session_state.data_dir,
                        model_path=model_path,
                        embedding_model=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        verbose=True
                    )
                    
                    # Process documents and create vector store
                    st.session_state.chatbot.load_documents()
                    st.session_state.chatbot.chunk_documents()
                    st.session_state.chatbot.setup_vector_store()
                    st.session_state.chatbot.setup_llm()
                    st.session_state.chatbot.setup_rag_chain()
                    
                    st.session_state.ready = True
                    st.success("RAG Chatbot initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing chatbot: {str(e)}")
                logger.error(f"Error initializing chatbot: {str(e)}")
    
    # Save vector store if button clicked
    if save_vector_store and st.session_state.chatbot is not None:
        try:
            # Create a temporary file for the vector store
            with tempfile.TemporaryDirectory() as temp_dir:
                vector_store_path = os.path.join(temp_dir, "vector_store")
                st.session_state.chatbot.save_vector_store(vector_store_path)
                
                # Create a download button for the saved vector store
                import zipfile
                import io
                
                # Zip the vector store directory
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w') as zip_file:
                    for root, dirs, files in os.walk(vector_store_path):
                        for file in files:
                            zip_file.write(
                                os.path.join(root, file),
                                os.path.relpath(os.path.join(root, file), vector_store_path)
                            )
                
                buffer.seek(0)
                st.download_button(
                    label="Download Vector Store",
                    data=buffer,
                    file_name="vector_store.zip",
                    mime="application/zip"
                )
        except Exception as e:
            st.error(f"Error saving vector store: {str(e)}")
            logger.error(f"Error saving vector store: {str(e)}")
    
    # Load vector store if file uploaded
    if load_vector_store is not None:
        try:
            # Save the uploaded file to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                vector_store_path = os.path.join(temp_dir, "vector_store")
                os.makedirs(vector_store_path, exist_ok=True)
                
                # Extract the zip file
                import zipfile
                import io
                
                with zipfile.ZipFile(io.BytesIO(load_vector_store.getbuffer()), 'r') as zip_ref:
                    zip_ref.extractall(vector_store_path)
                
                # Initialize chatbot if not already done
                if st.session_state.chatbot is None:
                    st.session_state.chatbot = RAGChatbotEnhanced(
                        data_dir=st.session_state.data_dir,
                        model_path=model_path,
                        embedding_model=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        verbose=True
                    )
                
                # Load the vector store
                st.session_state.chatbot.load_vector_store(vector_store_path)
                st.session_state.chatbot.setup_llm()
                st.session_state.chatbot.setup_rag_chain()
                
                st.session_state.ready = True
                st.success("Vector store loaded successfully!")
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            logger.error(f"Error loading vector store: {str(e)}")
    
    # Show performance stats
    if st.session_state.chatbot is not None:
        if st.button("Show Performance Statistics"):
            current_mem, peak_mem = st.session_state.chatbot.get_memory_usage()
            st.write(f"Current Memory Usage: {current_mem:.2f} MB")
            st.write(f"Peak Memory Usage: {peak_mem:.2f} MB")
            
            # If there are memory and CPU usage metrics available
            if hasattr(st.session_state.chatbot, 'memory_usage') and hasattr(st.session_state.chatbot, 'cpu_usage'):
                if st.session_state.chatbot.memory_usage and st.session_state.chatbot.cpu_usage:
                    import numpy as np
                    import matplotlib.pyplot as plt
                    import pandas as pd
                    
                    # Create a DataFrame for the metrics
                    metrics_df = pd.DataFrame({
                        'Memory (MB)': st.session_state.chatbot.memory_usage,
                        'CPU (%)': st.session_state.chatbot.cpu_usage
                    })
                    
                    # Plot the metrics
                    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
                    ax[0].plot(metrics_df['Memory (MB)'])
                    ax[0].set_title('Memory Usage (MB)')
                    ax[0].set_ylabel('MB')
                    
                    ax[1].plot(metrics_df['CPU (%)'])
                    ax[1].set_title('CPU Usage (%)')
                    ax[1].set_ylabel('%')
                    
                    st.pyplot(fig)
    
    # Reset conversation button
    if st.session_state.chatbot is not None:
        if st.button("Reset Conversation"):
            st.session_state.chatbot.reset_conversation()
            st.session_state.messages = []
            st.success("Conversation reset!")

# Main chat interface
chat_container = st.container()

with chat_container:
    # Display chat messages from history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
    
    # Accept user input
    if st.session_state.ready:
        user_input = st.chat_input("Ask me anything about your documents...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(user_input)
            
                            # Process the user query and get response
            with st.spinner("Thinking..."):
                try:
                    # Get response from chatbot
                    result = st.session_state.chatbot.ask(user_input)
                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])
                    
                    # Format the response with source citations
                    response = answer
                    
                    if source_docs:
                        response += "\n\n**Sources:**"
                        for i, doc in enumerate(source_docs, 1):
                            # Get a snippet of the source document
                            snippet = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                            response += f"\n{i}. {snippet}"
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        
                except Exception as e:
                    error_msg = f"Error processing your query: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
    else:
        st.info("Please initialize the RAG chatbot from the sidebar before chatting!")

# Footer
st.markdown("---")
st.markdown("RAG Chatbot - Powered by LangChain, FAISS, and LLama")