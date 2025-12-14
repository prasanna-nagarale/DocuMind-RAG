🤖 RAG Chatbot – PDF & Web Scraping

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that allows users to chat only from provided PDF documents or scraped web content.
The chatbot uses Groq’s high-performance LLMs along with vector search to deliver accurate, hallucination-free responses.

✨ Features

📄 PDF Document Processing
Upload PDF files and ask questions directly from the document content.

🌐 Web Scraping Support
Scrape website content and interact with the extracted information.

💬 Interactive Chat Interface
User-friendly Streamlit interface for natural conversations.

🔍 Vector-Based Retrieval
Efficient semantic search using FAISS and HuggingFace embeddings.

🚀 Powered by Groq LLMs
Ultra-fast inference using models like LLaMA 3.1 and Mixtral.

🔧 Highly Configurable
Control model selection, temperature, chunk size, and retrieval depth.

🛠️ Technologies Used

Streamlit – Web application interface

LangChain – Document processing and orchestration

FAISS – Vector database for similarity search

HuggingFace – Embeddings (all-MiniLM-L6-v2)

Groq – LLM inference engine

BeautifulSoup – Web scraping

PyPDF – PDF document parsing

📦 Installation
1️⃣ Clone the Repository
git clone <your-repo-url>
cd rag-chatbot

2️⃣ Create a Virtual Environment (Recommended)
Windows
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
Option 1: Install manually
pip install streamlit
pip install langchain-community langchain-core langchain-text-splitters
pip install pypdf faiss-cpu sentence-transformers
pip install groq requests beautifulsoup4 python-dotenv

Option 2: Install from requirements.txt
pip install -r requirements.txt

🔑 Environment Setup
1️⃣ Create .env file
cp .env.example .env

2️⃣ Add your Groq API Key
GROQ_API_KEY=your_actual_groq_api_key_here


🔗 Get your API key from:
https://console.groq.com/keys

▶️ Run the Application
streamlit run app.py


The app will open at:
👉 http://localhost:8501

🔧 Configuration

All configuration is managed using the .env file.

🔐 Required Configuration
Variable	Description
GROQ_API_KEY	Your Groq API key

🧠 Model Configuration
Variable	Default	Description
MODEL_NAME	llama-3.1-8b-instant	LLM model
TEMPERATURE	0	Controls randomness (0–1)
MAX_COMPLETION_TOKENS	1024	Max response length

📊 Embedding Configuration
Variable	Default	Description
EMBEDDING_MODEL	all-MiniLM-L6-v2	HuggingFace embedding model

✂️ Text Processing Configuration
Variable	Default	Description
CHUNK_SIZE	1000	Characters per chunk
CHUNK_OVERLAP	200	Overlap between chunks

🔎 Retrieval Configuration
Variable	Default	Description
RETRIEVAL_K	3	Number of chunks retrieved

🌐 Web Scraping Configuration
Variable	Default	Description
DEFAULT_SCRAPE_URL	https://www.icmr.gov.in/tenders	Default URL
SCRAPE_TIMEOUT	10	Request timeout (seconds)
MIN_TEXT_LENGTH	30	Minimum text length

🚀 Usage
📄 PDF Mode

Select PDF Upload from the sidebar
Upload a PDF file
Click Process PDF
Ask questions from the document

🌐 Web Scraping Mode

Select Website Scraping
Enter a URL (or use default)
Click Scrape Website
Ask questions from scraped content

⚙️ Advanced Settings

Use the sidebar to adjust:
Temperature (creativity control)
Max response tokens
Retrieval depth (K value)

🧠 Key Design Principle

The chatbot strictly answers only from the provided documents and web sources, ensuring high accuracy and zero hallucination.

📌 Future Improvements (Optional)

Multi-PDF support

Chat history persistence

Source citation highlighting

Cloud deployment (Streamlit Community Cloud)

⭐ Final Note

This project demonstrates a production-ready RAG pipeline combining document intelligence, vector search, and LLM inference — suitable for enterprise, research, and knowledge-base applications.
