import streamlit as st
from rag import build_vectorstore, ask_question

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 Website RAG Chatbot")
st.write("Ask questions about any website's content!")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

url = st.text_input("Enter website URL:", placeholder="https://en.wikipedia.org/wiki/Python_(programming_language)")

if st.button("Load Website"):
    with st.spinner("Scraping, chunking, and embedding... this may take a moment"):
        st.session_state.vectorstore = build_vectorstore(url)
    st.success("Website loaded! You can now ask questions.")

if st.session_state.vectorstore:
    question = st.text_input("Ask a question:")
    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = ask_question(st.session_state.vectorstore, question)
        st.write("### Answer")
        st.write(answer)