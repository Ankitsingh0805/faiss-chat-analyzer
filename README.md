# RAG Document Analyzer Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot for analyzing documents with a conversational interface built with Streamlit, LangChain, FAISS, and Llama-2.

![Document Analyzer](Screenshot%20from%202025-05-20%2018-40-51.png)

## 📋 Features

- **Document Analysis**: Upload text files and ask questions about their content  
- **Conversational Interface**: Natural language interaction with your documents  
- **Efficient RAG Pipeline**: Documents are chunked, embedded, and retrieved based on relevance  
- **Source Citations**: Responses include citations to the source documents  
- **Streamlit Web Interface**: User-friendly interface for document uploads and interaction  
- **Performance Monitoring**: Track memory and CPU usage  
- **Vector Store Management**: Save and load vector stores for quick reuse  
- **Configurable Parameters**: Adjust chunk size, overlap, temperature, and more  

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Ankitsingh0805/faiss-chat-analyzer.git
cd rag-document-analyzer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Download the LLama-2 model:
```bash
# Create models directory
mkdir -p models

# Download the model
wget -O models/llama-2-7b-chat.Q5_K_M.gguf https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf
```

Alternatively, download the model directly from the browser by visiting:  
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf

## 📂 Project Structure

```
rag-document-analyzer/
├── data/                  # Directory for document storage
│   ├── ai_ethics.txt
│   ├── climate_change.txt
│   └── ...
├── models/                # Directory for model storage
│   └── llama-2-7b-chat.Q5_K_M.gguf
├── rag_chatbot.py         # Core RAG implementation
├── app.py                 # Streamlit application
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🚀 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Access the application in your web browser (typically at http://localhost:8501)

3. Upload your documents using the sidebar

4. Click "Initialize RAG Chatbot" to process the documents

5. Start asking questions about your documents!

## 💻 Command Line Usage

You can also use the RAG chatbot directly from the command line:

```bash
python rag_chatbot.py --data data --model models/llama-2-7b-chat.Q5_K_M.gguf
```

Optional arguments:
```
--data DIR               Directory containing text files
--model PATH             Path to model file
--embeddings MODEL       Embedding model name
--chunk-size SIZE        Size of text chunks
--chunk-overlap SIZE     Overlap between chunks
--max-tokens COUNT       Maximum tokens to generate
--temperature TEMP       Temperature for text generation
--top-k COUNT            Number of documents to retrieve
--verbose                Enable verbose output
--load-vectors PATH      Load vector store from path
--save-vectors PATH      Save vector store to path
```

## 📊 Screenshots

### Main Interface
![Main Interface](Screenshot%20from%202025-05-20%2018-40-51.png)

### Chatting with Documents
![Chat Example](Screenshot%20from%202025-05-20%2018-09-02.png)

### Storing and downloading vector store
![Performance Statistics](Screenshot%20from%202025-05-20%2018-07-44.png)

## 🔧 Advanced Configuration

The system can be customized through the Streamlit interface or command line arguments:

- **Chunk Size**: Controls how documents are split (smaller chunks for more precise retrieval)  
- **Chunk Overlap**: Overlap between chunks to maintain context  
- **Temperature**: Controls randomness of generation (lower = more deterministic)  
- **Max Tokens**: Maximum length of generated responses  
- **Top K**: Number of document chunks to retrieve for each query  

## 🧠 How It Works

1. **Document Processing**: Text files are loaded and split into manageable chunks  
2. **Embedding Creation**: Each chunk is converted into an embedding vector using sentence transformers  
3. **Vector Storage**: FAISS efficiently indexes these vectors for similarity search  
4. **Query Processing**: User questions are embedded and matched against document chunks  
5. **Response Generation**: LLama-2 generates natural language responses based on relevant chunks  
6. **Source Attribution**: Sources used to generate responses are tracked and displayed  

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework  
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search  
- [HuggingFace](https://huggingface.co/) for model hosting  
- [Llama 2](https://ai.meta.com/llama/) by Meta AI  
- [Streamlit](https://streamlit.io/) for the web interface  
