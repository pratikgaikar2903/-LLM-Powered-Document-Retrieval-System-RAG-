# LLM-Powered Document Retrieval

# System (RAG)

#### Building a Retrieval-Augmented Generation (RAG) system is an excellent project. It

#### bridges the gap between static Large Language Models (LLMs) and your proprietary,

#### dynamic data, effectively solving the "hallucination" problem by grounding the AI's

#### answers in facts.

#### This project runs a fully functional, free-tier RAG pipeline natively in Google Colab using

#### LangChain , HuggingFace Embeddings , ChromaDB , and the Groq API (Llama 3.1).

#### Below is the cell-by-cell execution of the Colab notebook.

## 📌 Cell 1: Clean Install with Modern Packages

## Code:

```
# Install all necessary Python libraries for the RAG system.
# - `langchain-core`: Core LangChain components.
# - `langchain-groq`: Integrates with Groq for LLM access.
# - `langchain-huggingface`: For using HuggingFace models, specifically
embeddings.
# - `langchain-chroma`: Integrates with ChromaDB for vector store
functionality.
# - `pypdf`: Used for loading PDF documents.
# - `sentence-transformers`: For creating embeddings using pre-trained
models.
# - `langchain-text-splitters`: For splitting documents into manageable
chunks.
# - `langchain-community`: Contains various community-contributed
LangChain components.
# The `-q` flag stands for 'quiet', suppressing most installation
output.
```

```
!pip install -q langchain-core langchain-groq langchain-huggingface
langchain-chroma pypdf sentence-transformers langchain-text-splitters
langchain-community
```
## 📌 Cell 2: Full RAG System (Modern LangChain LCEL)

### Code:

```
import os
from google.colab import userdata
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```
```
# 1. Load API Key
try:
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')
print("✅ API key loaded.")
except userdata.SecretNotFoundError:
raise ValueError("API Key not found in Colab Secrets.")
```
```
# 2. Initialize Free Embeddings
print("Loading HuggingFace embeddings (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```
```
# 3. Initialize Free LLM
print("Initializing Groq LLM (llama-3.1-8b-instant)...")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
```
```
# 4. Initialize Vector Store
persist_dir = "/content/chroma_db"
if os.path.exists(persist_dir) and os.listdir(persist_dir):
print("✅ Found existing vector store — loading it.")
vector_store = Chroma(persist_directory=persist_dir,
embedding_function=embeddings)
```

else:
print("No existing vector store found. Running ingestion...")
loader = PyPDFDirectoryLoader("/content/data")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vector_store = Chroma.from_documents(documents=splits,
embedding=embeddings, persist_directory=persist_dir)

# 5. Build LCEL Pipeline (No legacy chains)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

template = """
You are a helpful assistant. Use the following pieces of retrieved
context to answer the question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
return "\n\n".join(doc.page_content for doc in docs)

# Modern LCEL Routing
rag_pipeline = (
{"context": retriever | format_docs, "question":
RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

# 6. Execute Query
question = "What are the main skills or objectives mentioned in the
resume?"
print(f"\n🔍 Question: {question}")

answer = rag_pipeline.invoke(question)
print(f"\n📝 Answer:\n{answer}")


### Output:

```
✅ API key loaded.
Loading HuggingFace embeddings (all-MiniLM-L6-v2)...
Loading weights: 100%
```
```
103/103 [00:00<00:00, 2750.05it/s]
Initializing Groq LLM (llama-3.1-8b-instant)...
✅ Found existing vector store — loading it.
```
```
🔍 Question: What are the main skills or objectives mentioned in the
resume?
```
```
📝 Answer:
The main skills or objectives mentioned in the resume are:
```
1. Data Analysis
2. Machine Learning
3. Data Science
4. Predictive Analytics
5. Problem Solving
6. Analytical Thinking
7. Communication
8. Team Collaboration
9. Time Management

```
Additionally, the resume highlights the candidate's expertise in
various programming languages, tools, and platforms, including:
```
1. Python
2. SQL
3. Scikit-learn
4. Pandas
5. NumPy
6. Matplotlib
7. Seaborn
8. Jupyter Notebook
9. VS Code
10. Git
11. GitHub
