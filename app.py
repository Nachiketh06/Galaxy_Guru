import streamlit as st
from backend import create_qa_agent, ask_question
import os
import time
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Path to your PDF
PDF_PATH = "/Users/nachikethnachu/Downloads/newcode_project/Astronomy-OP_oV0J80E.pdf"

# Set Streamlit page config for better UI
st.set_page_config(page_title="Astronomy AI Assistant", page_icon="🌌", layout="wide")

# Initialize session state
if "qa_agent" not in st.session_state:
    with st.spinner("Setting up the QA Agent..."):
        st.session_state.qa_agent = create_qa_agent(PDF_PATH)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Save past Q&A

# ----- Header -----
st.markdown("<h1 style='text-align: center; color: #4A4A4A;'>🌌 Astronomy AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask anything from Astronomy — I’ll guide you across the stars!</p>", unsafe_allow_html=True)
st.divider()

# ----- Main Input Area -----
with st.container():
    col1, col2 = st.columns([4, 1])

    with col1:
        question = st.chat_input("Type your question here...")

    with col2:
        education_level = st.selectbox(
            "Select Education Level",
            ["High School", "Undergraduate", "Graduate/Masters"],
            index=1,
            help="Choose your background for better answers!"
        )

# ----- Process the Question -----
new_answer = None
if question:
    with st.spinner("Thinking..."):
        result = ask_question(st.session_state.qa_agent, question, education_level)

        if result.get("error"):
            st.error(result["error"])
        else:
            st.session_state.chat_history.append((
                question,
                result["answer"],
                education_level,
                result.get("follow_up_questions", []),
                result.get("sources", [])
            ))
            new_answer = result["answer"]

# ----- Show Chat History -----
st.markdown("---")
st.markdown("### 📜 Conversation History")

# If new question was asked, stream it at top, then render others below
if new_answer:
    user_q, bot_a, level, followups, sources = st.session_state.chat_history[-1]

    # STREAM the latest one first (on top)
    with st.chat_message("user"):
        st.markdown(f"**You ({level}):** {user_q}")

    typing_placeholder = st.empty()
    typing_placeholder.markdown("_Astronomy AI is typing..._")
    time.sleep(1)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        for char in bot_a:
            full_text += char
            placeholder.markdown(f"**Astronomy AI:** {full_text}")
            time.sleep(random.uniform(0.008, 0.015))

        # Display follow-up questions
        if followups:
            st.markdown("#### 🤔 Follow-up Questions")
            for fq in followups:
                st.markdown(f"- {fq}")

        # Display source documents
        if sources:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:** {src[:500]}...")

    typing_placeholder.empty()
    rest_history = st.session_state.chat_history[:-1][::-1]
else:
    rest_history = st.session_state.chat_history[::-1]

# ----- Show previous messages
for user_q, bot_a, level, followups, sources in rest_history:
    with st.chat_message("user"):
        st.markdown(f"**You ({level}):** {user_q}")
    with st.chat_message("assistant"):
        st.markdown(f"**Astronomy AI:** {bot_a}")

        if followups:
            st.markdown("#### 🤔 Follow-up Questions")
            for fq in followups:
                st.markdown(f"- {fq}")

        if sources:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:** {src[:500]}...")

# ----- Footer
st.markdown("---")
st.caption("🚀 Powered by LangChain, Ollama, and Streamlit")
st.markdown("🌟 Created by Nachiketh")
