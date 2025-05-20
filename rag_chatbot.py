import os
import sys
import logging
import argparse
import time
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("rag_chatbot_enhanced")

try:
    # Core dependencies
    from langchain.document_loaders import TextLoader, DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain, LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.llms import CTransformers
    from langchain.prompts import PromptTemplate
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain.schema import Document
    
    # For performance monitoring
    import psutil
    import numpy as np
    
except ImportError as e:
    logger.error(f"Error importing dependencies: {str(e)}")
    logger.error("Please install the required packages using: pip install -r requirements.txt")
    sys.exit(1)

class RAGChatbotEnhanced:
    
    def __init__(self, 
                 data_dir: str = "data",
                 model_path: str = "models/llama-2-7b-chat.Q5_K_M.gguf",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 top_k: int = 3,
                 verbose: bool = False):
        
        self.data_dir = data_dir
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.verbose = verbose
        
        # Set up logging based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")
            
        # Initialize components as None
        self.documents = None
        self.text_splitter = None
        self.chunks = None
        self.embeddings = None
        self.vector_store = None
        self.memory = None
        self.llm = None
        self.qa_chain = None
        
        # Performance monitoring
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []
        
        logger.info("RAG Chatbot initialized successfully")
        
    def log_resource_usage(self):
        """Log current resource usage"""
        mem_info = self.process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        
        self.memory_usage.append(mem_mb)
        self.cpu_usage.append(cpu_percent)
        
        logger.debug(f"Memory usage: {mem_mb:.2f} MB | CPU: {cpu_percent:.1f}%")
        
    def load_documents(self) -> List[Document]:

        logger.info(f"Loading documents from {self.data_dir}...")
        self.log_resource_usage()
        
        try:
            # Check if there are any files in the directory
            text_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
            if not text_files:
                raise ValueError(f"No text files found in {self.data_dir}")
            
            # Use the DirectoryLoader to load all text files
            loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader)
            self.documents = loader.load()
            
            logger.info(f"Loaded {len(self.documents)} documents")
            self.log_resource_usage()
            return self.documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    def chunk_documents(self) -> List[Document]:
        
        if self.documents is None:
            self.load_documents()
            
        logger.info("Splitting documents into chunks...")
        self.log_resource_usage()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        self.chunks = self.text_splitter.split_documents(self.documents)
        
        logger.info(f"Created {len(self.chunks)} chunks")
        self.log_resource_usage()
        return self.chunks
    
    def setup_vector_store(self) -> FAISS:
        """
        Create a vector store from document chunks using embeddings.
        
        Returns:
            FAISS vector store
        """
        if self.chunks is None:
            self.chunk_documents()
            
        logger.info("Setting up embeddings and vector store...")
        self.log_resource_usage()
        
        # Using a lightweight embedding model suitable for CPU
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Using FAISS for efficient similarity search with limited RAM
        self.vector_store = FAISS.from_documents(self.chunks, self.embeddings)
        
        logger.info("Vector store created successfully")
        self.log_resource_usage()
        return self.vector_store
    
    def setup_llm(self) -> CTransformers:
        """
        Set up the language model optimized for CPU.
        
        Returns:
            Configured LLM
        """
        logger.info("Setting up the language model...")
        self.log_resource_usage()
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            logger.error("Please download the model using the download_model.py script.")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Use CTransformers to load quantized models efficiently
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = CTransformers(
            model=self.model_path,
            model_type="llama",
            callback_manager=callback_manager,
            config={
                'max_new_tokens': self.max_tokens,
                'temperature': self.temperature,
                'context_length': 2048,
                'gpu_layers': 0  # Force CPU only
            }
        )
        
        logger.info("Language model loaded successfully")
        self.log_resource_usage()
        return self.llm
    
    def setup_rag_chain(self) -> ConversationalRetrievalChain:
        """
        Create the RAG chain for question answering.
        
        Returns:
            ConversationalRetrievalChain for QA
        """
        if self.vector_store is None:
            self.setup_vector_store()
            
        if self.llm is None:
            self.setup_llm()
            
        logger.info("Setting up RAG chain...")
        self.log_resource_usage()
        
        # Create a custom prompt template for better results
        template = """
        You are an AI assistant providing accurate answers based on the context provided.
        
        Chat History:
        {chat_history}
        
        Context information:
        {context}
        
        Question: {question}
        
        Please provide a comprehensive and accurate answer based on the given context.
        If the information is not in the context, politely state that you don't have that information.
        Answer:
        """
        
        qa_prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template=template,
        )
        
        # Create conversation memory with the right output key
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",  # Explicitly set the output key
            return_messages=True
        )
        
        # Create retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=self.verbose
        )
        
        logger.info("RAG chain set up successfully")
        self.log_resource_usage()
        return self.qa_chain
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the RAG chatbot.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if self.qa_chain is None:
            self.setup_rag_chain()
            
        logger.info(f"Question: {question}")
        self.log_resource_usage()
        
        # Track time for response generation
        start_time = time.time()
        
        # Get response from chain
        result = self.qa_chain({"question": question})
        
        # Log response time
        end_time = time.time()
        response_time = end_time - start_time
        logger.info(f"Response generated in {response_time:.2f} seconds")
        
        # Extract source documents
        source_docs = result.get("source_documents", [])
        if source_docs:
            logger.debug(f"Retrieved {len(source_docs)} source documents")
            for i, doc in enumerate(source_docs):
                logger.debug(f"Source {i+1}: {doc.page_content[:100]}...")
        
        self.log_resource_usage()
        return result
    
    def chat(self):
        """
        Interactive chat interface for the RAG chatbot.
        """
        if self.qa_chain is None:
            self.setup_rag_chain()
            
        print("\n===== RAG Chatbot =====")
        print("Type 'exit' to quit the chat")
        print("Type 'stats' to view performance statistics")
        print("Type 'reset' to reset the conversation history")
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Exiting chat... Goodbye!")
                    break
                elif user_input.lower() == "stats":
                    self.print_performance_stats()
                    continue
                elif user_input.lower() == "reset":
                    self.reset_conversation()
                    print("Conversation history has been reset.")
                    continue
                    
                result = self.ask(user_input)
                print("\nChatbot:", result["answer"])
                
                # Show source documents if available
                if "source_documents" in result and result["source_documents"]:
                    source_docs = result["source_documents"]
                    print("\nSources:")
                    for i, doc in enumerate(source_docs, 1):
                        print(f"{i}. {doc.page_content[:100]}...")
                        
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                print(f"Error: {str(e)}")
                
    def reset_conversation(self):
        """Reset the conversation history"""
        if self.memory is not None:
            self.memory.clear()
            logger.info("Conversation history reset")
    
    def print_performance_stats(self):
        """Print performance statistics"""
        if not self.memory_usage or not self.cpu_usage:
            print("No performance data available yet.")
            return
            
        # Calculate statistics
        avg_mem = np.mean(self.memory_usage)
        max_mem = np.max(self.memory_usage)
        avg_cpu = np.mean(self.cpu_usage)
        max_cpu = np.max(self.cpu_usage)
        
        # Total run time
        total_time = time.time() - self.start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n===== Performance Statistics =====")
        print(f"Run time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Memory usage: {avg_mem:.2f} MB (avg), {max_mem:.2f} MB (max)")
        print(f"CPU usage: {avg_cpu:.1f}% (avg), {max_cpu:.1f}% (max)")
        print(f"Documents loaded: {len(self.documents) if self.documents else 0}")
        print(f"Chunks created: {len(self.chunks) if self.chunks else 0}")
        print("=================================")
                
    def save_vector_store(self, path: str = "vector_store"):
        """
        Save the vector store for future use.
        
        Args:
            path: Path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
            
        logger.info(f"Saving vector store to {path}...")
        self.vector_store.save_local(path)
        logger.info(f"Vector store saved to {path}")
                
    def load_vector_store(self, path: str = "vector_store"):
        """
        Load a previously saved vector store.
        
        Args:
            path: Path to the saved vector store
            
        Returns:
            Loaded vector store
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store path not found: {path}")
            
        if self.embeddings is None:
            # Initialize embeddings if not already done
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
        logger.info(f"Loading vector store from {path}...")
        self.vector_store = FAISS.load_local(path, self.embeddings)
        logger.info("Vector store loaded successfully")
        return self.vector_store
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """
        Get current memory usage statistics
        
        Returns:
            Tuple of (current_memory_usage_mb, peak_memory_usage_mb)
        """
        current_mem = self.process.memory_info().rss / 1024 / 1024
        peak_mem = max(self.memory_usage) if self.memory_usage else current_mem
        return current_mem, peak_mem


def main():
    """Main function to run the enhanced RAG chatbot"""
    parser = argparse.ArgumentParser(description="Enhanced RAG Chatbot")
    parser.add_argument("--data", default="data", help="Directory containing text files")
    parser.add_argument("--model", default="models/llama-2-7b-chat.Q5_K_M.gguf", help="Path to LLM model file")
    parser.add_argument("--embeddings", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=500, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--load-vectors", help="Load vector store from path")
    parser.add_argument("--save-vectors", help="Save vector store to path")
    
    args = parser.parse_args()
    
    try:
        # Initialize chatbot
        chatbot = RAGChatbotEnhanced(
            data_dir=args.data,
            model_path=args.model,
            embedding_model=args.embeddings,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            verbose=args.verbose
        )
        
        # Load vector store if specified
        if args.load_vectors:
            chatbot.load_vector_store(args.load_vectors)
        else:
            # Process documents
            chatbot.load_documents()
            chatbot.chunk_documents()
            chatbot.setup_vector_store()
            
            # Save vector store if specified
            if args.save_vectors:
                chatbot.save_vector_store(args.save_vectors)
        
        # Setup language model and chain
        chatbot.setup_llm()
        chatbot.setup_rag_chain()
        
        # Start chat
        chatbot.chat()
        
    except Exception as e:
        logger.error(f"Error running chatbot: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())