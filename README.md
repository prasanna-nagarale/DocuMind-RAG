# 🤖 Advanced RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that enables intelligent conversations with your documents and web content. Powered by Groq's high-performance LLMs and vector search for accurate, hallucination-free responses.

---

## ✨ Features

- **📄 Multi-Format Document Support** – Process PDF, DOCX, TXT, CSV, JSON, XLSX files
- **🌐 Intelligent Web Scraping** – Auto-crawls websites with internal link following
- **💬 Interactive Chat Interface** – User-friendly Streamlit UI with chat history
- **🔍 Vector-Based Retrieval** – FAISS-powered semantic search with HuggingFace embeddings
- **🚀 Powered by Groq** – Ultra-fast inference using LLaMA 3.1, Mixtral models
- **🔧 Highly Configurable** – Control model parameters, chunk size, retrieval depth

---

## 🛠️ Technologies

| Component | Technology |
|-----------|-----------|
| **Framework** | Streamlit |
| **LLM Provider** | Groq |
| **Orchestration** | LangChain |
| **Vector DB** | FAISS |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Scraping** | BeautifulSoup4 |
| **Document Parsers** | PyPDF, Unstructured |

---

## 📦 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/prasanna-nagarale/DocuMind-RAG
cd rag-chatbot
```

### 2️⃣ Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**Manual Installation:**
```bash
pip install streamlit langchain-community langchain-core langchain-text-splitters
pip install pypdf faiss-cpu sentence-transformers groq requests beautifulsoup4 python-dotenv
pip install unstructured python-docx openpyxl
```

---

## 🔑 Configuration

### Create `.env` file:
```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Model Settings
MODEL_NAME=llama-3.1-8b-instant
TEMPERATURE=0
MAX_COMPLETION_TOKENS=1024

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Text Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
RETRIEVAL_K=3

# Web Scraping
DEFAULT_SCRAPE_URL=https://www.icmr.gov.in/tenders
SCRAPE_TIMEOUT=10
MIN_TEXT_LENGTH=30
MAX_PAGES_TO_SCRAPE=5
```

🔗 **Get your Groq API key:** [https://console.groq.com/keys](https://console.groq.com/keys)

---

## ▶️ Usage

### Start the Application
```bash
streamlit run app.py
```
Access at: **http://localhost:8501**

### 📄 File Upload Mode
1. Select **"📄 File Upload"** in sidebar
2. Upload documents (PDF, DOCX, CSV, JSON, XLSX)
3. Click **"Process Files"**
4. Start chatting with your documents

### 🌐 Web Scraping Mode
1. Select **"🌐 Website Scraping"** in sidebar
2. Enter target URL
3. Click **"Scrape Website"**
4. Ask questions about scraped content

### ⚙️ Advanced Settings
Configure from sidebar:
- **Temperature** (0-1): Control response creativity
- **Max Tokens**: Set response length
- **Chunks to Retrieve**: Adjust context depth
- **Max Pages**: Control scraping scope

---

## 📊 Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | llama-3.1-8b-instant | LLM model selection |
| `TEMPERATURE` | 0 | Response randomness (0=deterministic) |
| `MAX_COMPLETION_TOKENS` | 1024 | Maximum response length |
| `CHUNK_SIZE` | 1000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `RETRIEVAL_K` | 3 | Number of chunks retrieved |
| `MAX_PAGES_TO_SCRAPE` | 5 | Web pages to crawl |

---

## 🎯 Key Design Principles

- **Strict Source Attribution**: Responses generated only from provided documents
- **Zero Hallucination**: Vector-based retrieval ensures factual accuracy
- **Scalable Architecture**: Handles multiple files and large documents
- **Production Ready**: Error handling, session management, progress tracking

---

## 🧪 Example Use Cases

- **📚 Research** – Analyze academic papers, reports
- **💼 Business** – Query financial documents, contracts
- **📰 News** – Extract insights from articles, blogs
- **📖 Documentation** – Build knowledge base chatbots
- **🎓 Education** – Study materials, course content Q&A

---

## 🔍 How It Works

1. **Document Processing** → Load and split documents into chunks
2. **Embedding Generation** → Convert chunks to vectors using HuggingFace
3. **Vector Storage** → Index embeddings in FAISS database
4. **Query Processing** → User question converted to vector
5. **Retrieval** → Find K most similar chunks
6. **Response Generation** → Groq LLM generates answer from context

---

## 🚧 Troubleshooting

**API Key Error:**
```bash
# Verify .env file exists and contains valid key
cat .env
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Slow Processing:**
- Reduce `CHUNK_SIZE`
- Decrease `MAX_PAGES_TO_SCRAPE`
- Use lighter model (llama-3.1-8b-instant)


---

## 🙏 Acknowledgments

- **Groq** – High-performance LLM inference
- **LangChain** – RAG orchestration framework
- **FAISS** – Efficient vector similarity search
- **HuggingFace** – State-of-the-art embeddings
- **Streamlit** – Rapid web app development

---

## 📧 Contact

For questions or support, reach out via:
- GitHub Issues
- Email: nagaraleprasanna@gmail.com

---

**⭐ Star this repo if you find it useful!**
