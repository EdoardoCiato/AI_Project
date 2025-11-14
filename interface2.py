# app.py
import io
import contextlib
import streamlit as st
from query_data import query_rag
from compare_table import compare_as_table

# --- PAGE SETUP (must be first Streamlit call) ---
st.set_page_config(page_title="Uni Assistant", page_icon=":university:", layout="centered")

ASSISTANT_AVATAR = "🧑‍🎓"
USER_AVATAR = "🙋‍♀️"

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = None              # "learn" or "compare"
if "awaiting_next" not in st.session_state:
    st.session_state.awaiting_next = False    # true after an answer
if "end_chat" not in st.session_state:
    st.session_state.end_chat = False

# --- HELPERS ---
def assistant_say(text: str):
    st.session_state.messages.append(("assistant", text))

def user_say(text: str):
    st.session_state.messages.append(("user", text))

def render_history():
    for role, content in st.session_state.messages:
        with st.chat_message(role, avatar=ASSISTANT_AVATAR if role == "assistant" else USER_AVATAR):
            st.markdown(content)

def end_chat_screen():
    st.title("🎓 Uni Assistant")
    st.success("Thanks for chatting with me! 👋 Have a great day.")
    st.stop()

def ask_continue_only():
    """Show ONLY the 'ask another / switch / done' options, then stop."""
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("Would you like to do anything else?")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Ask another"):
                st.session_state.awaiting_next = False
                st.experimental_rerun()
        with c2:
            if st.button("Switch mode"):
                st.session_state.mode = None
                st.session_state.awaiting_next = False
                st.experimental_rerun()
        with c3:
            if st.button("Done ✅"):
                st.session_state.end_chat = True
                st.experimental_rerun()
    st.stop()

# --- ROUTING ---
if st.session_state.end_chat:
    end_chat_screen()

# 1) Render history first
render_history()

# 2) If we are awaiting the next action, show ONLY the 3 buttons and stop.
if st.session_state.awaiting_next:
    ask_continue_only()

# 3) If no mode, show greeting + choice
if st.session_state.mode is None:
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("Hi! I'm **Uni Assistant** 🤖. What would you like to do?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Investigate one university"):
                st.session_state.mode = "learn"
                user_say("I want to investigate one university.")
                st.experimental_rerun()
        with col2:
            if st.button("📊 Compare universities"):
                st.session_state.mode = "compare"
                user_say("I want to compare universities.")
                st.experimental_rerun()
    st.stop()

# 4) Mode UIs (only shown when NOT awaiting_next)

# ---- Investigate one university ----
if st.session_state.mode == "learn":
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("Ask me a question about one university "
                    "(e.g., *What is Stanford tuition for 2024–2025?*).")
    q = st.chat_input("Type your question about a university…")
    if q:
        user_say(q)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Thinking…"):
                try:
                    answer = query_rag(q)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                st.text(answer)           # plain text to avoid markdown parsing
                assistant_say(answer)
        st.session_state.awaiting_next = True
        st.experimental_rerun()

# ---- Compare universities ----
elif st.session_state.mode == "compare":
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown(
            "Okay! I can compare the **same** question across multiple universities.\n\n"
            "Enter your question and a comma-separated list of universities."
        )
    with st.form("compare_form", clear_on_submit=False):
        q = st.text_input("Question", placeholder="e.g., What are the student populations?")
        uni_text = st.text_input(
            "Universities (comma-separated)",
            placeholder="e.g., Harvard University, University Of California-Berkeley, University Of California-Los Angeles",
        )
        submitted = st.form_submit_button("Compare 🔎")

    if submitted:
        user_say(f"Compare: {q} | Unis: {uni_text}")
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Comparing…"):
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf):
                        compare_as_table(q.strip(), [u.strip() for u in uni_text.split(",") if u.strip()])
                    table_out = buf.getvalue()
                except Exception as e:
                    table_out = f"⚠️ Error: {e}"
                st.code(table_out, language="text")
                assistant_say(f"Here’s your comparison:\n\n```text\n{table_out}\n```")
        st.session_state.awaiting_next = True
        st.experimental_rerun()
