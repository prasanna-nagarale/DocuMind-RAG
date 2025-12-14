import streamlit as st
import requests
from bs4 import BeautifulSoup
from groq import Groq
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredExcelLoader, UnstructuredWordDocumentLoader, JSONLoader
)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import tempfile
import os
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from typing import List, Tuple
import time

load_dotenv()

# ================================
# CONFIGURATION
# ================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "1024"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
DEFAULT_SCRAPE_URL = os.getenv("DEFAULT_SCRAPE_URL", "https://www.icmr.gov.in/tenders")
SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", "10"))
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "30"))
MAX_PAGES_TO_SCRAPE = int(os.getenv("MAX_PAGES_TO_SCRAPE", "5"))

st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="🤖", layout="wide")

# ================================
# COMPACT CSS
# ================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main-header { font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0.3rem; }
.subtitle { text-align: center; color: #6b7280; font-size: 1.1rem; margin-bottom: 1.5rem; }
.feature-card { background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-left: 4px solid #667eea;
    padding: 1rem; border-radius: 10px; margin: 0.5rem 0; transition: transform 0.2s; }
.feature-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2); }
.stButton > button { border-radius: 8px; padding: 0.5rem 1.2rem; font-weight: 600; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
.stFileUploader { border: 2px dashed #667eea; border-radius: 10px; padding: 1.5rem;
    background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%); }
.stProgress > div > div { background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%); border-radius: 8px; }
hr { margin: 1.5rem 0; border: none; height: 1px; background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%); }
div[data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; color: #667eea; }
</style>
""", unsafe_allow_html=True)

# ================================
# SESSION STATE
# ================================
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'num_chunks' not in st.session_state:
    st.session_state.num_chunks = 0
if 'num_files' not in st.session_state:
    st.session_state.num_files = 0

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    
    groq_api_key = st.text_input("🔑 Groq API Key", type="password", value=GROQ_API_KEY)
    if groq_api_key:
        st.success("✅ API key configured")
    else:
        st.error("⚠️ API key required")
    
    model_name = st.selectbox("🤖 Model", 
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0 if MODEL_NAME not in ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"] 
        else ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"].index(MODEL_NAME))
    
    st.divider()
    
    with st.expander("🔧 Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.1)
        max_tokens = st.number_input("Max Tokens", 100, 8000, MAX_COMPLETION_TOKENS, 100)
        retrieval_k = st.number_input("Chunks to Retrieve", 1, 10, RETRIEVAL_K)
        max_pages = st.number_input("Max Pages", 1, 20, MAX_PAGES_TO_SCRAPE)
    
    st.divider()
    source_type = st.radio("📁 Data Source", ["📄 File Upload", "🌐 Website Scraping"])
    
    if st.session_state.data_loaded:
        st.divider()
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.num_chunks)
        col2.metric("Files", st.session_state.num_files)

# ================================
# HELPER FUNCTIONS
# ================================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def process_uploaded_files(uploaded_files: List) -> Tuple:
    all_documents = []
    successful_files = []
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((idx + 1) / len(uploaded_files))
        status.text(f"Processing: {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")
        
        suffix = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            loader = None
            if suffix == "pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix == "txt":
                loader = TextLoader(tmp_path, encoding='utf-8')
            elif suffix == "csv":
                loader = CSVLoader(tmp_path)
            elif suffix in ["xlsx", "xls"]:
                loader = UnstructuredExcelLoader(tmp_path, mode="elements")
            elif suffix in ["docx", "doc"]:
                loader = UnstructuredWordDocumentLoader(tmp_path, mode="elements")
            elif suffix == "json":
                try:
                    loader = JSONLoader(file_path=tmp_path, jq_schema=".", text_content=False)
                except:
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        import json
                        data = json.load(f)
                        all_documents.append(Document(page_content=json.dumps(data, indent=2),
                                                     metadata={"source": uploaded_file.name}))
                        successful_files.append(uploaded_file.name)
                        continue
            
            if loader:
                docs = loader.load()
                if docs:
                    all_documents.extend(docs)
                    successful_files.append(uploaded_file.name)
        except Exception as e:
            st.warning(f"⚠️ Error processing {uploaded_file.name}: {str(e)[:50]}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    progress_bar.empty()
    status.empty()
    
    if not all_documents:
        st.error("❌ No documents processed successfully")
        return None, 0
    
    st.success(f"✅ {len(successful_files)} files processed")
    
    with st.spinner("🔪 Splitting into chunks..."):
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(all_documents)
    
    with st.spinner("🧠 Creating embeddings..."):
        vectorstore = FAISS.from_documents(chunks, get_embeddings())
    
    return vectorstore, len(chunks)

def scrape_website_recursive(base_url: str, max_pages: int = 5) -> Tuple:
    visited = set()
    all_texts = []
    
    status = st.empty()
    
    def scrape_page(url: str, depth: int = 0):
        if url in visited or len(visited) >= max_pages or depth > 2:
            return
        
        visited.add(url)
        status.text(f"Scraping: {url} (Page {len(visited)}/{max_pages})")
        
        try:
            response = requests.get(url, timeout=SCRAPE_TIMEOUT,
                headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            page_texts = []
            for tag in soup.find_all(["p", "li", "h1", "h2", "h3", "h4", "td"]):
                text = tag.get_text(strip=True)
                if text and len(text) > MIN_TEXT_LENGTH:
                    page_texts.append(text)
            
            if page_texts:
                all_texts.extend(page_texts)
            
            if depth < 2 and len(visited) < max_pages:
                base_domain = urlparse(base_url).netloc
                for a_tag in soup.find_all("a", href=True):
                    link = urljoin(base_url, a_tag["href"]).split('#')[0]
                    if urlparse(link).netloc == base_domain and link not in visited:
                        scrape_page(link, depth + 1)
        
        except Exception as e:
            st.warning(f"⚠️ Could not access: {url}")
    
    scrape_page(base_url)
    status.empty()
    
    if not all_texts:
        st.error("❌ No content extracted")
        return None, 0
    
    st.success(f"✅ Scraped {len(visited)} pages • {len(all_texts)} sections")
    
    unique_texts = list(dict.fromkeys(all_texts))
    documents = [Document(page_content="\n\n".join(unique_texts),
                         metadata={"source": base_url, "pages_scraped": len(visited)})]
    
    with st.spinner("🔪 Splitting into chunks..."):
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)
    
    with st.spinner("🧠 Creating embeddings..."):
        vectorstore = FAISS.from_documents(chunks, get_embeddings())
    
    return vectorstore, len(chunks)

def chat_with_rag(question: str, vectorstore, api_key: str, model: str, temp: float, max_tok: int, k_retrieval: int) -> str:
    try:
        result = vectorstore.similarity_search(question, k=k_retrieval)
        context = "\n\n".join([f"Chunk {i}: {doc.page_content}" for i, doc in enumerate(result, 1)])
        
        system_prompt = """You are a smart, helpful AI assistant. Answer based ONLY on the provided knowledge base.

Guidelines:
- Provide clear, accurate responses
- Don't mention "chunks" or "sources"
- If not found, say "I cannot find this information in the documents"
- Be conversational and helpful"""
        
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\nKnowledge Base:\n{context}"},
                {"role": "user", "content": question}
            ],
            temperature=temp,
            max_completion_tokens=max_tok
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ================================
# MAIN APP
# ================================
st.markdown('<h1 class="main-header">🤖 Advanced RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Chat with your documents and websites using AI</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="feature-card">📄 <b>Multi-Format</b><br><small>PDF, DOCX, CSV, JSON</small></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-card">🌐 <b>Web Crawling</b><br><small>Auto-follows links</small></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="feature-card">🧠 <b>Vector Search</b><br><small>FAISS powered</small></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="feature-card">💬 <b>Smart Chat</b><br><small>Context-aware AI</small></div>', unsafe_allow_html=True)

st.divider()

# ================================
# DATA SOURCE SECTION
# ================================
if source_type == "📄 File Upload":
    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader("Choose files", 
        type=["pdf", "txt", "csv", "json", "xlsx", "xls", "docx", "doc"],
        accept_multiple_files=True)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        if st.button("🔄 Process Files", use_container_width=True, type="primary"):
            vectorstore, num_chunks = process_uploaded_files(uploaded_files)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.data_loaded = True
                st.session_state.chat_history = []
                st.session_state.num_chunks = num_chunks
                st.session_state.num_files = len(uploaded_files)
                st.success(f"🎉 Success! {num_chunks} chunks created")
                st.balloons()
                time.sleep(1)
                st.rerun()

else:  # Website Scraping
    st.markdown("### 🌐 Scrape Website")
    website_url = st.text_input("Enter Website URL", value=DEFAULT_SCRAPE_URL, placeholder="https://example.com")
    
    if st.button("🔄 Scrape Website", use_container_width=True, type="primary"):
        if website_url and website_url.startswith(('http://', 'https://')):
            vectorstore, num_chunks = scrape_website_recursive(website_url, max_pages)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.data_loaded = True
                st.session_state.chat_history = []
                st.session_state.num_chunks = num_chunks
                st.session_state.num_files = 1
                st.success(f"🎉 Success! {num_chunks} chunks created")
                st.balloons()
                time.sleep(1)
                st.rerun()
        else:
            st.error("❌ Please enter a valid URL")

st.divider()

# ================================
# CHAT SECTION
# ================================
if st.session_state.data_loaded and st.session_state.vectorstore:
    st.markdown("### 💬 Chat with Your Data")
    
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
    
    if user_question := st.chat_input("💭 Ask a question..."):
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = chat_with_rag(user_question, st.session_state.vectorstore,
                    groq_api_key, model_name, temperature, max_tokens, retrieval_k)
                st.markdown(response)
        
        st.session_state.chat_history.append((user_question, response))
    
    col1, col2 = st.columns([1, 1])
    if col1.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    if col2.button("🔄 Reset All"):
        for key in ['vectorstore', 'data_loaded', 'chat_history', 'num_chunks', 'num_files']:
            st.session_state[key] = [] if key == 'chat_history' else (None if key == 'vectorstore' else (False if key == 'data_loaded' else 0))
        st.rerun()
else:
    st.info("👆 Upload files or scrape a website to start chatting!")

st.divider()
st.caption("💡 Powered by Groq • LangChain • FAISS • HuggingFace")