import os
os.environ["USER_AGENT"] = "webchat-rag/1.0"
from langchain_community.document_loaders import WebBaseLoader

def load_website(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    return chunks

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vectorstore(url):
    docs = load_website(url)
    chunks = split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def ask_question(vs, question):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )
    docs = vs.similarity_search(question, k=4)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""Answer the question based only on the context below.

Context:
{context}

Question: {question}
Answer:"""
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    vs = build_vectorstore("https://en.wikipedia.org/wiki/Python_(programming_language)")
    answer = ask_question(vs, "What is Python used for?")
    print(answer)