# app.py
import io
import html
import contextlib
import streamlit as st

from query_data import query_rag
# ✅ use your UniGuide comparer (from comparing.py)
from comparing import compare_universities

# --- PAGE SETUP (must come first) ---
st.set_page_config(page_title="Uni Assistant", page_icon=":university:", layout="centered")

ASSISTANT_AVATAR = "🧑‍🎓"
USER_AVATAR = "🙋‍♀️"

# --- SESSION STATE INIT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = None  # "learn" or "compare"
if "awaiting_next" not in st.session_state:
    st.session_state.awaiting_next = False
if "end_chat" not in st.session_state:
    st.session_state.end_chat = False

# --- MESSAGE HELPERS ---
def assistant_say(text: str):
    st.session_state.messages.append(("assistant", text))

def user_say(text: str):
    st.session_state.messages.append(("user", text))

# --- END CHAT SCREEN ---
def end_chat_screen():
    st.title("🎓 Uni Assistant")
    st.success("Thanks for chatting with me! 👋 Have a great day.")
    st.stop()

# --- PARSER: turn comparing.py stdout into dict {uni: [bullets]} ---
def parse_comparison_stdout(text: str):
    """
    Expected format from comparing.py:
    Comparison:

    - University A
      - bullet
      - bullet

    - University B
      - bullet
    """
    unis = {}
    current = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.strip().lower().startswith("comparison"):
            continue
        if line.startswith("- ") and not line.startswith("- -"):
            current = line[2:].strip()
            unis[current] = []
            continue
        if current and line.lstrip().startswith("- "):
            bullet = line.lstrip()[2:].strip()
            if bullet:
                unis[current].append(bullet)
    return unis

# --- RENDER: show a friendly, integrated comparison (not a code block) ---
def render_comparison(unis_dict):
    st.markdown("Here’s your comparison:")
    for uni, bullets in unis_dict.items():
        st.markdown(f"**{uni}**")
        st.write("\n".join([f"- {b}" for b in bullets]))
        st.write("")  # spacer

def render_history():
    for role, content in st.session_state.messages:
        with st.chat_message(role, avatar=ASSISTANT_AVATAR if role == "assistant" else USER_AVATAR):
            st.markdown(content)

# --- ROUTING ---
if st.session_state.end_chat:
    end_chat_screen()

# --- RENDER CHAT HISTORY ---
render_history()

# --- STEP 1: Greeting & MODE SELECTION ---
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

# --- STEP 2: ASK/COMPARE FLOW ---
if st.session_state.mode == "learn":
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("Ask me a question about one university (e.g., *What is Harvard tuition for 2024–2025?*)")
    question = st.chat_input("Type your question about a university…")
    if question:
        user_say(question)
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Thinking…"):
                try:
                    answer = query_rag(question)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                # keep simple like your first version
                st.text(answer)
                assistant_say(answer)
        st.session_state.awaiting_next = True
        st.experimental_rerun()

elif st.session_state.mode == "compare":
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown(
            "Okay! I can compare a question across multiple universities. "
            "Enter your question and a comma-separated list of universities."
        )

    with st.form("compare_form", clear_on_submit=False):
        q = st.text_input("Question", placeholder="e.g., How does first-year advising work?")
        uni_text = st.text_input(
            "Universities (comma-separated)",
            placeholder="e.g., Harvard University, Princeton University, Brown University",
        )
        submitted = st.form_submit_button("Compare 🔎")

    if submitted:
        universities = [u.strip() for u in uni_text.split(",") if u.strip()]
        user_say(f"Compare: {q} | Unis: {', '.join(universities)}")
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Comparing…"):
                # capture stdout from comparing.compare_universities
                buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(buf):
                        compare_universities(q.strip(), universities)
                    raw = buf.getvalue()
                except Exception as e:
                    raw = f"⚠️ Error: {e}"

                if raw.strip().startswith("⚠️"):
                    # show errors plainly
                    st.text(raw)
                    assistant_say(raw)
                else:
                    data = parse_comparison_stdout(raw)
                    if data:
                        render_comparison(data)
                        # Store a compact text summary in history (not the raw terminal dump)
                        friendly_summary = "Here’s your comparison:\n\n" + "\n\n".join(
                            [f"**{u}**\n" + "\n".join([f"- {b}" for b in bs]) for u, bs in data.items()]
                        )
                        assistant_say(friendly_summary)
                    else:
                        # Fallback: show whatever came back
                        st.markdown(raw)
                        assistant_say(raw)

        st.session_state.awaiting_next = True
        st.experimental_rerun()

# --- STEP 3: CONTINUE OR END CHAT ---
if st.session_state.awaiting_next:
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("Would you like to do anything else?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Ask another"):
                st.session_state.awaiting_next = False
                st.experimental_rerun()
        with col2:
            if st.button("Switch mode"):
                st.session_state.mode = None
                st.session_state.awaiting_next = False
                st.experimental_rerun()
        with col3:
            if st.button("Done ✅"):
                st.session_state.end_chat = True
                st.experimental_rerun()
