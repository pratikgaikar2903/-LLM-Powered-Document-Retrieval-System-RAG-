
# RAG System with LangChain, Groq, and Chroma

This project implements a Retrieval Augmented Generation (RAG) system using modern LangChain components, Groq for the Language Model (LLM), and ChromaDB for vector storage. It processes PDF documents to answer questions based on their content.

## Features
- Document ingestion from PDF files.
- Text splitting and embedding using HuggingFace embeddings.
- Vector storage and retrieval with ChromaDB.
- Question answering using Groq's `llama-3.1-8b-instant` model.
- Built with LangChain Expression Language (LCEL) for a streamlined RAG chain.

## Setup

### 1. Install Dependencies
First, install the necessary Python libraries. This is handled by the first cell in the notebook:

```bash
!pip install -q langchain-core langchain-groq langchain-huggingface langchain-chroma pypdf sentence-transformers langchain-text-splitters langchain-community
```

### 2. Set Up Groq API Key
This project requires an API key from Groq. You need to add your `GROQ_API_KEY` to the Colab Secrets (the 🔑 tab on the left sidebar).

```python
import os
from google.colab import userdata

try:
    os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')
    print("✅ API key loaded.")
except Exception:
    raise ValueError("Add 'GROQ_API_KEY' to the Secrets (🔑) tab in Colab.")
```

### 3. Provide PDF Document
Ensure your PDF document is accessible. By default, the system looks for `/content/Pratik Gaikar - ATS Data Resume.pdf`.

## Usage

The `RAGSystem` class encapsulates the entire RAG pipeline. Here's how to use it:

### 1. Initialize the RAG System
```python
rag = RAGSystem(
    file_path="/content/Pratik Gaikar - ATS Data Resume.pdf", # Path to your PDF
    persist_dir="/content/chroma_db" # Directory to store ChromaDB embeddings
)
```

### 2. Ingest Documents
If a vector store doesn't already exist, ingest your documents to create one.
```python
if rag.vector_store is None:
    rag.ingest_documents()
```

### 3. Query the System
Ask questions related to the document content.
```python
question = "What are the main skills or objectives mentioned in the resume?"
answer = rag.query(question)
print("
📝 Answer:
", answer)
```

## Project Structure

- `lk8ecCK2xA8P`: Installs required Python packages.
- `qGSF8hRwxlSF`: Contains the `RAGSystem` class definition and example usage.

