## 🚀 Live Demo
https://webchat-rag-lyy79esusjoc2xrwaclzta.streamlit.app/

# Website RAG Chatbot

A simple RAG (Retrieval-Augmented Generation) chatbot that lets you ask questions about the content of any website. Built with LangChain, FAISS, HuggingFace embeddings, and Groq's LLM, with a Streamlit web interface.

## How it works

1. **Scrape** – Loads the webpage content using `WebBaseLoader`
2. **Chunk** – Splits the text into 1000-character chunks (with 200-char overlap) using `RecursiveCharacterTextSplitter`
3. **Embed + Store** – Converts each chunk into vector embeddings using HuggingFace's `all-MiniLM-L6-v2` model, stored in a FAISS vector index
4. **Ask** – On a user question, retrieves the top 4 most relevant chunks via similarity search, sends them along with the question to Groq's `llama-3.1-8b-instant` model, and returns an answer

## Tech Stack

- **LangChain** – orchestration framework
- **FAISS** – vector similarity search
- **HuggingFace Sentence Transformers** – local embeddings (no API key needed)
- **Groq** – fast LLM inference (Llama 3.1)
- **Streamlit** – web UI

## Setup

1. Clone the repo

git clone https://github.com/Akarsh-42/webchat-rag.git

cd webchat-rag

2. Install dependencies

pip install -r requirements.txt

3. Create a `.env` file in the main folder:
GROQ_API_KEY=your_groq_api_key_here
   Get a free key at [console.groq.com](https://console.groq.com)

4. Run the app

streamlit run app.py

## Usage

1. Enter a website URL and click "Load Website"
2. Once loaded, type a question about the page's content
3. Click "Get Answer"

## Future Improvements

- Support for multiple URLs 
- Conversation history / follow-up questions
- Streaming responses for faster perceived speed
