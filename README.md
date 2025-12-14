🤖 RAG Chatbot - PDF & Web Scraping
A powerful Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that allows you to chat with your documents (PDFs) or scraped web content using Groq's LLM models.
✨ Features

📄 PDF Document Processing: Upload and chat with PDF documents
🌐 Web Scraping: Scrape and chat with website content
💬 Interactive Chat Interface: Natural conversation with your data
🔧 Configurable Settings: Adjust model parameters, temperature, and retrieval settings
🚀 Powered by Groq: Fast inference with state-of-the-art LLM models
🔍 Vector Search: Efficient semantic search using FAISS and HuggingFace embeddings

🛠️ Technologies Used

Streamlit - Web interface
LangChain - Document processing and text splitting
FAISS - Vector database for similarity search
HuggingFace - Embeddings (all-MiniLM-L6-v2)
Groq - LLM inference (Llama 3.1, Mixtral)
BeautifulSoup - Web scraping
PyPDF - PDF processing

📦 Installation
1. Clone the Repository
bashgit clone <your-repo-url>
cd rag-chatbot
2. Create Virtual Environment (Recommended)
bash# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bashpip install streamlit
pip install langchain-community langchain-core langchain-text-splitters
pip install pypdf faiss-cpu sentence-transformers
pip install groq requests beautifulsoup4 python-dotenv
Or install from requirements.txt (if provided):
bashpip install -r requirements.txt
4. Set Up Environment Variables

Copy the .env.example file to .env:

bashcp .env.example .env

Edit the .env file and add your Groq API key:

envGROQ_API_KEY=your_actual_groq_api_key_here
Get your Groq API Key: https://console.groq.com/keys
5. Run the Application
bashstreamlit run app.py
The app will open in your browser at http://localhost:8501
🔧 Configuration
All configuration is managed through the .env file. Here are the available options:
Required Configuration
VariableDescriptionExampleGROQ_API_KEYYour Groq API key (Required)gsk_xxxxx...
Model Configuration
VariableDefaultDescriptionMODEL_NAMEllama-3.1-8b-instantLLM model to useTEMPERATURE0Randomness (0-1)MAX_COMPLETION_TOKENS1024Max response length
Embedding Configuration
VariableDefaultDescriptionEMBEDDING_MODELall-MiniLM-L6-v2HuggingFace embedding model
Text Processing Configuration
VariableDefaultDescriptionCHUNK_SIZE1000Characters per chunkCHUNK_OVERLAP200Overlapping characters
Retrieval Configuration
VariableDefaultDescriptionRETRIEVAL_K3Number of chunks to retrieve
Web Scraping Configuration
VariableDefaultDescriptionDEFAULT_SCRAPE_URLhttps://www.icmr.gov.in/tendersDefault URLSCRAPE_TIMEOUT10Request timeout (seconds)MIN_TEXT_LENGTH30Minimum text length to include
🚀 Usage
PDF Mode

Select "PDF Upload" in the sidebar
Click "Browse files" and upload your PDF
Click "Process PDF" button
Wait for processing to complete
Start asking questions in the chat!

Web Scraping Mode

Select "Website Scraping" in the sidebar
Enter the URL you want to scrape (or use the default)
Click "Scrape Website" button
Wait for scraping to complete
Start asking questions in the chat!

Advanced Settings
Click on "Advanced Settings" in the sidebar to adjust:

Temperature: Control response creativity (0 = deterministic, 1 = creative)
Max Tokens: Limit response length
Retrieval K: Number of relevant chunks to use for context
