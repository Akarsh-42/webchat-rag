import streamlit as st
import time
import os
import tempfile
from rag import build_vectorstore, build_vectorstore_from_file, ask_question

st.set_page_config(
    page_title="RAG AI — Website Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #070b14 !important;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2a3a;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
.sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #818cf8 !important;
    padding: 0.5rem 0;
    border-bottom: 1px solid #1e2a3a;
    margin-bottom: 1rem;
}
.history-item {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: #94a3b8 !important;
    cursor: pointer;
    transition: border-color 0.2s;
}
.history-item:hover { border-color: #6366f1; }
.history-q { color: #c7d2fe !important; font-weight: 500; font-size: 0.78rem; }
.history-a { color: #64748b !important; font-size: 0.72rem; margin-top: 2px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid #4338ca;
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 0.75rem;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    max-width: 500px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}

/* ── Glass card ── */
.glass {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}

/* ── Mode tabs ── */
.mode-tabs {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    justify-content: center;
    margin-bottom: 1.5rem;
}
.mode-tab {
    background: #111827;
    border: 1px solid #1e2a3a;
    color: #64748b;
    border-radius: 10px;
    padding: 0.5rem 1.1rem;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}
.mode-tab.active {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-color: #6366f1;
    color: #ffffff;
}

/* ── Insight cards ── */
.insight-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.insight-card {
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem;
}
.insight-num {
    font-size: 1.6rem;
    font-weight: 700;
    color: #818cf8;
    font-family: 'Space Grotesk', sans-serif;
}
.insight-label {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* ── Chip questions ── */
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1rem 0;
}
.chip {
    background: #111827;
    border: 1px solid #1e2a3a;
    color: #a5b4fc;
    border-radius: 20px;
    padding: 0.4rem 0.9rem;
    font-size: 0.8rem;
    cursor: pointer;
}

/* ── Chat bubbles ── */
.chat-wrap { display: flex; flex-direction: column; gap: 1rem; margin: 1rem 0; }
.bubble-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    max-width: 75%;
    font-size: 0.9rem;
    line-height: 1.5;
}
.bubble-bot {
    align-self: flex-start;
    background: #111827;
    border: 1px solid #1e2a3a;
    color: #cbd5e1;
    border-radius: 16px 16px 16px 4px;
    padding: 0.9rem 1.2rem;
    max-width: 85%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.bubble-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    opacity: 0.6;
}

/* ── Agent debate ── */
.agent-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin: 1rem 0; }
.agent-card {
    border-radius: 12px;
    padding: 1rem;
    border: 1px solid;
}
.agent-researcher { background: rgba(16,185,129,0.07); border-color: rgba(16,185,129,0.25); }
.agent-developer  { background: rgba(59,130,246,0.07); border-color: rgba(59,130,246,0.25); }
.agent-critic     { background: rgba(239,68,68,0.07);  border-color: rgba(239,68,68,0.25); }
.agent-business   { background: rgba(245,158,11,0.07); border-color: rgba(245,158,11,0.25); }
.agent-name { font-weight: 600; font-size: 0.85rem; margin-bottom: 0.5rem; }
.agent-researcher .agent-name { color: #10b981; }
.agent-developer  .agent-name { color: #3b82f6; }
.agent-critic     .agent-name { color: #ef4444; }
.agent-business   .agent-name { color: #f59e0b; }
.agent-text { font-size: 0.82rem; color: #94a3b8; line-height: 1.5; }

/* ── Podcast ── */
.podcast-card {
    background: linear-gradient(135deg, #1e1b4b, #0f172a);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.podcast-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 700; color: #a5b4fc; margin-bottom: 0.5rem; }
.podcast-script { color: #94a3b8; font-size: 0.875rem; line-height: 1.8; white-space: pre-line; }

/* ── Source citations ── */
.citation {
    background: #0d1117;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.8rem;
    color: #64748b;
}

/* ── File upload ── */
.upload-zone {
    background: rgba(99,102,241,0.05);
    border: 2px dashed rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.file-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Avatar ── */
.avatar {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    cursor: pointer;
    z-index: 999;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 4px 20px rgba(99,102,241,0.4); }
    50%       { box-shadow: 0 4px 30px rgba(99,102,241,0.7); }
}

/* ── Typing dots ── */
.typing { display: flex; gap: 5px; align-items: center; padding: 0.5rem 0; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: #6366f1; animation: bounce 1.2s infinite; }
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-8px)} }

div[data-testid="stMarkdownContainer"] p { color: #94a3b8; }
.stSuccess { background: #064e3b !important; }
.stWarning { background: #78350f !important; }
.stError   { background: #7f1d1d !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in {
    "vectorstore": None,
    "loaded_url": "",
    "chat_history": [],
    "mode": "Chat",
    "insights": [],
    "suggested_qs": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 RAG AI</div>', unsafe_allow_html=True)

    st.markdown("**📚 Chat History**")
    if st.session_state.chat_history:
        for i, c in enumerate(reversed(st.session_state.chat_history[-10:])):
            st.markdown(f"""
            <div class="history-item">
                <div class="history-q">Q: {c['question'][:50]}{'…' if len(c['question'])>50 else ''}</div>
                <div class="history-a">{c['answer'][:60]}{'…' if len(c['answer'])>60 else ''}</div>
            </div>""", unsafe_allow_html=True)
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown('<p style="color:#374151;font-size:0.8rem;">No history yet.</p>', unsafe_allow_html=True)

    st.divider()
    if st.session_state.loaded_url:
        st.markdown("**🌐 Loaded Website**")
        st.markdown(f'<p style="color:#6366f1;font-size:0.75rem;word-break:break-all;">{st.session_state.loaded_url}</p>', unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ Powered by Groq + LangChain</div>
    <div class="hero-title">Turn Any Website Into<br>Your AI Knowledge Base</div>
    <div class="hero-sub">Load a URL and instantly chat, debate, summarize, quiz, or podcast-ify any content.</div>
</div>
""", unsafe_allow_html=True)

# ── Input Source Tabs ──────────────────────────────────────────────────────────
tab_url, tab_file = st.tabs(["🌐 Website URL", "📄 Upload PDF / TXT"])

with tab_url:
    col1, col2 = st.columns([4, 1])
    with col1:
        url = st.text_input("", placeholder="🔗  Paste any website URL here…", label_visibility="collapsed", key="url_input")
    with col2:
        load_btn = st.button("🚀 Load", key="load_url")

    if load_btn:
        if url:
            with st.spinner("Indexing website content…"):
                try:
                    st.session_state.vectorstore = build_vectorstore(url)
                    st.session_state.loaded_url = url
                    st.session_state.chat_history = []
                    insights_raw = ask_question(st.session_state.vectorstore,
                        "List exactly 5 key facts or insights from this page as short bullet points. Be concise.")
                    st.session_state.insights = [l.strip("•-– ").strip()
                        for l in insights_raw.split("\n") if l.strip()][:5]
                    qs_raw = ask_question(st.session_state.vectorstore,
                        "Suggest 6 interesting questions a curious reader might ask about this page. One per line, no numbers.")
                    st.session_state.suggested_qs = [l.strip()
                        for l in qs_raw.split("\n") if l.strip()][:6]
                    st.success("✅ Website loaded! Explore below.")
                except Exception as e:
                    st.error(f"❌ {e}")
        else:
            st.warning("⚠️ Please enter a URL.")

with tab_file:
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your file here",
        type=["pdf", "txt", "docx"],
        label_visibility="visible",
        key="file_upload"
    )
    st.markdown('<p class="upload-label">Supports PDF, TXT, DOCX files</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        st.markdown(f'<div class="file-badge">📄 {uploaded_file.name} ({round(uploaded_file.size/1024, 1)} KB)</div>', unsafe_allow_html=True)
        if st.button("📥 Process File", key="load_file"):
            with st.spinner(f"Reading {uploaded_file.name}…"):
                try:
                    suffix = "." + uploaded_file.name.split(".")[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    st.session_state.vectorstore = build_vectorstore_from_file(tmp_path, suffix)
                    st.session_state.loaded_url = f"📄 {uploaded_file.name}"
                    st.session_state.chat_history = []
                    os.unlink(tmp_path)
                    insights_raw = ask_question(st.session_state.vectorstore,
                        "List exactly 5 key facts or insights from this document as short bullet points. Be concise.")
                    st.session_state.insights = [l.strip("•-– ").strip()
                        for l in insights_raw.split("\n") if l.strip()][:5]
                    qs_raw = ask_question(st.session_state.vectorstore,
                        "Suggest 6 interesting questions a curious reader might ask about this document. One per line, no numbers.")
                    st.session_state.suggested_qs = [l.strip()
                        for l in qs_raw.split("\n") if l.strip()][:6]
                    st.success(f"✅ {uploaded_file.name} loaded! Explore below.")
                except Exception as e:
                    st.error(f"❌ {e}")

# ── Content area (only after load) ────────────────────────────────────────────
if st.session_state.vectorstore:

    # Key insights strip
    if st.session_state.insights:
        st.markdown("### 🔍 5 Key Insights")
        st.markdown('<div class="insight-grid">', unsafe_allow_html=True)
        for ins in st.session_state.insights:
            st.markdown(f'<div class="insight-card"><div class="insight-label">{ins}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Mode selector
    st.markdown("""
    <div class="mode-tabs">
        <span class="mode-tab" onclick="void(0)">💬 Chat</span>
        <span class="mode-tab" onclick="void(0)">🤖 AI Debate</span>
        <span class="mode-tab" onclick="void(0)">🎙️ Podcast</span>
        <span class="mode-tab" onclick="void(0)">📝 Summary</span>
        <span class="mode-tab" onclick="void(0)">🧪 Quiz</span>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("", ["💬 Chat", "🤖 AI Debate", "🎙️ Podcast", "📝 Summary", "🧪 Quiz"],
                    horizontal=True, label_visibility="collapsed")
    st.session_state.mode = mode

    st.divider()

    # ── CHAT MODE ──────────────────────────────────────────────────────────────
    if mode == "💬 Chat":
        # Suggested question chips
        if st.session_state.suggested_qs:
            st.markdown("**💡 Suggested questions — click to ask:**")
            cols = st.columns(3)
            for i, q in enumerate(st.session_state.suggested_qs):
                with cols[i % 3]:
                    if st.button(q, key=f"chip_{i}"):
                        with st.spinner("Thinking…"):
                            ans = ask_question(st.session_state.vectorstore, q)
                            st.session_state.chat_history.append({"question": q, "answer": ans})

        # Chat input
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        question = st.text_input("", placeholder="Ask anything about this website…",
                                  key="chat_q", label_visibility="collapsed")
        if st.button("📨 Send"):
            if question:
                with st.spinner(""):
                    st.markdown('<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>', unsafe_allow_html=True)
                    ans = ask_question(st.session_state.vectorstore, question)
                    st.session_state.chat_history.append({"question": question, "answer": ans})
        st.markdown('</div>', unsafe_allow_html=True)

        # Chat bubbles
        if st.session_state.chat_history:
            st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
            for c in reversed(st.session_state.chat_history):
                st.markdown(f"""
                <div class="bubble-user">
                    <div class="bubble-label">You</div>
                    {c['question']}
                </div>
                <div class="bubble-bot">
                    <div class="bubble-label">🤖 AI</div>
                    {c['answer']}
                </div>""", unsafe_allow_html=True)
                with st.expander("📋 Source citations"):
                    docs = st.session_state.vectorstore.similarity_search(c['question'], k=2)
                    for d in docs:
                        st.markdown(f'<div class="citation">{d.page_content[:200]}…</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── AI DEBATE MODE ─────────────────────────────────────────────────────────
    elif mode == "🤖 AI Debate":
        st.markdown("### 🤖 Multi-Agent Debate")
        st.markdown('<p style="color:#64748b;font-size:0.85rem;">Four AI agents debate your question from different perspectives.</p>', unsafe_allow_html=True)

        debate_q = st.text_input("", placeholder="What do you want the agents to debate?",
                                   key="debate_q", label_visibility="collapsed")
        if st.button("⚔️ Start Debate"):
            if debate_q:
                agents = [
                    ("🔬 Researcher", "researcher", "Answer as an academic researcher: cite facts, data, and evidence."),
                    ("💻 Developer",  "developer",  "Answer as a software developer: focus on technical implementation and code."),
                    ("🔍 Critic",     "critic",     "Play devil's advocate: find flaws, risks, and counterarguments."),
                    ("💼 Business",   "business",   "Answer as a business strategist: ROI, market, and practical impact."),
                ]
                st.markdown('<div class="agent-grid">', unsafe_allow_html=True)
                for name, cls, prompt in agents:
                    with st.spinner(f"{name} thinking…"):
                        full_prompt = f"{prompt}\n\nBased on the website content, answer in 3–4 sentences:\n{debate_q}"
                        resp = ask_question(st.session_state.vectorstore, full_prompt)
                    st.markdown(f"""
                    <div class="agent-card agent-{cls}">
                        <div class="agent-name">{name}</div>
                        <div class="agent-text">{resp}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Enter a question to debate.")

    # ── PODCAST MODE ───────────────────────────────────────────────────────────
    elif mode == "🎙️ Podcast":
        st.markdown("### 🎙️ 5-Minute Podcast Script")
        st.markdown('<p style="color:#64748b;font-size:0.85rem;">Converts this website into an engaging podcast episode.</p>', unsafe_allow_html=True)

        if st.button("🎙️ Generate Podcast"):
            with st.spinner("Writing your podcast script…"):
                script = ask_question(st.session_state.vectorstore, """
Write a natural, engaging 5-minute podcast script based on this website content.
Format:
[INTRO] Host introduction and hook
[SEGMENT 1] First major topic
[SEGMENT 2] Second major topic
[SEGMENT 3] Third major topic
[OUTRO] Closing thoughts and call to action
Use a conversational tone, as if two hosts are discussing it.""")

            st.markdown(f"""
            <div class="podcast-card">
                <div class="podcast-title">🎙️ Podcast Episode</div>
                <div class="podcast-script">{script}</div>
            </div>""", unsafe_allow_html=True)

    # ── SUMMARY MODE ───────────────────────────────────────────────────────────
    elif mode == "📝 Summary":
        st.markdown("### 📝 Website Summary")
        level = st.radio("Summary depth:", ["Quick (3 bullets)", "Standard (1 paragraph)", "Deep (full breakdown)"], horizontal=True)

        if st.button("✨ Summarize"):
            prompts = {
                "Quick (3 bullets)":       "Summarize this website in exactly 3 bullet points.",
                "Standard (1 paragraph)":  "Write a clear, concise 1-paragraph summary of this website.",
                "Deep (full breakdown)":   "Write a detailed structured summary with sections: Overview, Key Topics, Main Insights, and Conclusion.",
            }
            with st.spinner("Summarizing…"):
                summary = ask_question(st.session_state.vectorstore, prompts[level])
            st.markdown(f'<div class="glass"><p style="color:#cbd5e1;line-height:1.8;">{summary}</p></div>', unsafe_allow_html=True)

    # ── QUIZ MODE ──────────────────────────────────────────────────────────────
    elif mode == "🧪 Quiz":
        st.markdown("### 🧪 Knowledge Quiz")
        st.markdown('<p style="color:#64748b;font-size:0.85rem;">Test your understanding of this website.</p>', unsafe_allow_html=True)

        if st.button("🎲 Generate Quiz"):
            with st.spinner("Creating quiz questions…"):
                quiz = ask_question(st.session_state.vectorstore, """
Create a 5-question multiple choice quiz based on this website.
For each question use this format:
Q1: [question]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [letter]

Repeat for Q2–Q5.""")
            st.markdown(f'<div class="glass"><pre style="color:#cbd5e1;white-space:pre-wrap;font-family:Inter,sans-serif;font-size:0.88rem;line-height:1.8;">{quiz}</pre></div>', unsafe_allow_html=True)

# ── Floating avatar ────────────────────────────────────────────────────────────
st.markdown('<div class="avatar">🧠</div>', unsafe_allow_html=True)