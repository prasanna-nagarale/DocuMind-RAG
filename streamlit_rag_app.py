import streamlit as st
import requests
from bs4 import BeautifulSoup
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ================================
# LOAD CONFIGURATION FROM .ENV
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

# ================================
# SIDEBAR - CONFIGURATION
# ================================
st.sidebar.title("⚙️ Configuration")

# API Key Input - Load from .env
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=GROQ_API_KEY,
    help="Enter your Groq API key (loaded from .env if available)"
)

# Show warning if API key is not set
if not groq_api_key:
    st.sidebar.warning("⚠️ Please add your GROQ_API_KEY to the .env file or enter it above")

# Model Selection - Load from .env
model_name = st.sidebar.selectbox(
    "Model",
    ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    index=["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"].index(MODEL_NAME) if MODEL_NAME in ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"] else 0,
    help="Select the LLM model to use"
)

# Advanced Settings (Collapsible)
with st.sidebar.expander("🔧 Advanced Settings"):
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=TEMPERATURE,
        step=0.1,
        help="Controls randomness. 0 = deterministic, 1 = creative"
    )
    
    max_tokens = st.number_input(
        "Max Completion Tokens",
        min_value=100,
        max_value=8000,
        value=MAX_COMPLETION_TOKENS,
        step=100,
        help="Maximum length of generated response"
    )
    
    retrieval_k = st.number_input(
        "Retrieval K",
        min_value=1,
        max_value=10,
        value=RETRIEVAL_K,
        help="Number of relevant chunks to retrieve"
    )
    
    st.info(f"📊 Using embedding model: {EMBEDDING_MODEL}")
    st.info(f"📝 Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

# Source Selection
source_type = st.sidebar.radio(
    "Data Source",
    ["PDF Upload", "Website Scraping"],
    help="Choose your data source"
)

# ================================
# INITIALIZE SESSION STATE
# ================================
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# ================================
# HELPER FUNCTIONS
# ================================

@st.cache_resource
def get_embeddings():
    """Load and cache the embeddings model"""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def process_pdf(uploaded_file):
    """Process uploaded PDF file"""
    with st.spinner("📄 Processing PDF..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Split into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(docs)
            
            # Create vector store
            embeddings = get_embeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            return vectorstore, len(chunks)
        
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

def scrape_website(url):
    """Scrape website content"""
    with st.spinner(f"🌐 Scraping website: {url}"):
        try:
            response = requests.get(url, timeout=SCRAPE_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract text from various tags
            texts = []
            for tag in soup.find_all(["p", "li", "a", "td", "h1", "h2", "h3", "span"]):
                text = tag.get_text(strip=True)
                if text and len(text) > MIN_TEXT_LENGTH:
                    texts.append(text)
            
            full_text = "\n".join(texts)
            
            # Create document
            documents = [Document(
                page_content=full_text,
                metadata={"source": url}
            )]
            
            # Split into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = get_embeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            return vectorstore, len(chunks)
        
        except Exception as e:
            st.error(f"Error scraping website: {str(e)}")
            return None, 0

def chat_with_rag(question, vectorstore, api_key, model, temp, max_tok, k_retrieval):
    """Query the RAG system"""
    # Retrieve relevant documents
    result = vectorstore.similarity_search(question, k=k_retrieval)
    
    # Build context
    context = []
    for i, doc in enumerate(result, 1):
        context.append(f"Chunk {i}: {doc.page_content}")
    
    context_text = "\n\n".join(context)
    
    # Create prompt
    system_prompt = """You are a smart chatbot. Answer user questions based ONLY on the knowledge base provided below.
Provide well-structured, accurate responses without mentioning the chunks or sources.
If the answer is not in the knowledge base, say "I cannot find this information in the provided documents"."""
    
    final_prompt = f"{system_prompt}\n\nKnowledge Base:\n{context_text}"
    
    # Call Groq API
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": question}
            ],
            temperature=temp,
            max_completion_tokens=max_tok
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}"

# ================================
# MAIN APP
# ================================

st.title("🤖 RAG Chatbot")
st.markdown("Chat with your documents using Retrieval-Augmented Generation")

# ================================
# DATA SOURCE SECTION
# ================================

st.markdown("---")

if source_type == "PDF Upload":
    st.subheader("📄 Upload PDF Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to chat with"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        with col2:
            if st.button("🔄 Process PDF", use_container_width=True):
                vectorstore, num_chunks = process_pdf(uploaded_file)
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.data_loaded = True
                    st.session_state.chat_history = []
                    st.success(f"✅ PDF processed! Created {num_chunks} chunks")
                    st.rerun()

else:  # Website Scraping
    st.subheader("🌐 Scrape Website")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        website_url = st.text_input(
            "Website URL",
            value=DEFAULT_SCRAPE_URL,
            help="Enter the URL of the website to scrape"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("🔄 Scrape Website", use_container_width=True):
            vectorstore, num_chunks = scrape_website(website_url)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.data_loaded = True
                st.session_state.chat_history = []
                st.success(f"✅ Website scraped! Created {num_chunks} chunks")
                st.rerun()

# ================================
# CHAT SECTION
# ================================

st.markdown("---")

if st.session_state.data_loaded and st.session_state.vectorstore:
    st.subheader("💬 Chat with your data")
    
    # Display chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
    
    # Chat input
    user_question = st.chat_input("Ask a question about your data...")
    
    if user_question:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(user_question)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_with_rag(
                    user_question,
                    st.session_state.vectorstore,
                    groq_api_key,
                    model_name,
                    temperature,
                    max_tokens,
                    retrieval_k
                )
                st.write(response)
        
        # Save to history
        st.session_state.chat_history.append((user_question, response))
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.info("👆 Please upload a PDF or scrape a website to start chatting!")

# ================================
# FOOTER
# ================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built with Streamlit • Powered by Groq & LangChain
    </div>
    """,
    unsafe_allow_html=True
)