# interface3.py
# Purpose: Streamlit frontend for UniScout. Provides three modes:
#   1) "learn"      → Ask about one university (RAG over brochures)
#   2) "compare"    → Compare the same question across multiple universities
#   3) "recommend"  → Get personalized school recommendations
# UX: Chat-style interface with session state to track messages and flow.

import io
import html
import contextlib
import streamlit as st

from query_data import query_rag                     # RAG: answers about a single university
# ✅ compare flow
from comparing import compare_universities           # CLI function that prints comparison bullets to stdout
# ✅ recommendation flow (returns (rec_text, top))
from final_recomm import recommend as recommend_universities  # Returns markdown recommendation text

# --- PAGE SETUP (must come first) ---
st.set_page_config(page_title="Uni Assistant", page_icon=":university:", layout="centered")

# --- GLOBAL STYLES (frame + chat vibe + logo + centered landing) ---
# Note: We inject CSS into Streamlit to create a card-like layout and chat bubbles.
st.markdown(
    """
    <style>
      /* Outer card for the whole app */
      .main .block-container {
        max-width: 820px;
        margin: 0 auto;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(10, 89, 255, 0.08);
        border: 1px solid #e2e8f0;
        background: #ffffff;
        padding: 32px 28px !important;
      }

      /* Chatbox-style inner area (conversation messages) */
      .stChat {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #fafafa;
        padding: 18px;
        height: 60vh;
        overflow-y: auto;
        margin-bottom: 14px;
      }

      /* Chat bubbles */
      [data-testid="stChatMessage"] {
        border: 1px solid #eef3ff;
        border-radius: 14px;
        padding: 12px;
        background: #fbfdff;
        margin-bottom: 8px;
      }

      /* Style the chat input area to visually match the box */
      [data-testid="stChatInput"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: #fafafa;
        padding: 8px 10px;
      }

      /* Landing layout: center logo + tagline nicely */
      .centered-landing {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 75vh;
          text-align: center;
      }

      .uniscout-logo {
          font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
          font-weight: 900;
          font-size: 50px;
          letter-spacing: 3px;
          color: #0a59ff;
          padding: 12px 28px;
          border: 3px solid #0a59ff;
          border-radius: 16px;
          background: linear-gradient(90deg, #e9f1ff 0%, #ffffff 100%);
          box-shadow: 0 6px 18px rgba(10, 89, 255, 0.15);
          margin-bottom: 12px;
          display: inline-block;
      }

      .tagline {
          font-size: 18px;
          color: #4b5563;
          margin-bottom: 30px;
      }

      /* Small badge styling for ranks */
      .rank-badge {
          display: inline-block;
          padding: 2px 8px;
          border-radius: 9999px;
          border: 1px solid #dbeafe;
          background: #eff6ff;
          font-size: 12px;
          color: #1d4ed8;
          margin-right: 6px;
      }

      /* Centered dashed separator for mode changes */
      .mode-sep {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #64748b;
        font-size: 13px;
        margin: 10px 0 14px;
        text-transform: uppercase;
        letter-spacing: .05em;
      }
      .mode-sep:before,
      .mode-sep:after {
        content: "";
        flex: 1;
        border-top: 1px dashed #cbd5e1;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

ASSISTANT_AVATAR = "🧑‍🎓"
USER_AVATAR = "🙋‍♀️"

# --- SESSION STATE INIT ---
# We maintain mode and message history across reruns to simulate a chat app.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = None  # "learn", "compare", "recommend"
if "awaiting_next" not in st.session_state:
    st.session_state.awaiting_next = False
if "end_chat" not in st.session_state:
    st.session_state.end_chat = False

# --- MESSAGE HELPERS ---
def assistant_say(text: str):
    # Append an assistant message to the chat history
    st.session_state.messages.append(("assistant", text))

def user_say(text: str):
    # Append a user message to the chat history
    st.session_state.messages.append(("user", text))

def _strip_wrapping_quotes(s: str) -> str:
    # Remove symmetrical wrapping quotes if present: “...”, "...", '...', «...»
    if not s:
        return s
    s = s.strip()
    pairs = [("“","”"), ('"','"'), ("'","'"), ("«","»")]
    for lq, rq in pairs:
        if s.startswith(lq) and s.endswith(rq):
            return s[len(lq):-len(rq)].strip()
    return s

def add_mode_separator(new_mode: str):
    # Insert a visual separator message when switching modes
    pretty = {
        "learn": "Switched to: Investigate one university",
        "compare": "Switched to: Compare universities",
        "recommend": "Switched to: Recommend me a university",
    }
    st.session_state.messages.append(("separator", pretty.get(new_mode, "Mode changed")))

# --- END CHAT SCREEN ---
def end_chat_screen():
    # Final “goodbye” screen; stops further Streamlit execution.
    st.title("🎓 UNISCOUT")
    st.success("Thanks for chatting with UNISCOUT! 👋 Have a great day exploring universities.")
    st.stop()

# --- PARSER: comparing.py stdout -> {uni: [bullets]} ---
def parse_comparison_stdout(text: str):
    """
    Parse the stdout produced by compare_universities() into a dict:
      { "<University Name>": ["bullet 1", "bullet 2", ...], ... }
    Expected format:
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

# --- RENDERERS ---
def render_comparison(unis_dict):
    # Render comparison bullets for each university
    st.markdown("Here’s your comparison:")
    for uni, bullets in unis_dict.items():
        st.markdown(f"**{uni}**")
        st.write("\n".join([f"- {b}" for b in bullets]))
        st.write("")  # spacer

def render_recommendation_markdown(md: str):
    # Render advisor-style markdown text (single block)
    st.markdown(md if md else "⚠️ No recommendation text was returned.")

def render_recommendations_struct(rec_text, top):
    """
    Render structured recommendations:
      rec_text: advisor paragraph (markdown ok)
      top: List[Tuple[str, float, str]] => (university, score, reason)
    """
    if rec_text:
        st.markdown(rec_text)
    if top:
        st.markdown("**Top matches**")
        for idx, (u, s, w) in enumerate(top, start=1):
            st.markdown(
                f"<span class='rank-badge'>#{idx}</span> **{u}** — {round(float(s),1)}/100",
                unsafe_allow_html=True,
            )
            if w:
                st.write(f"- {w}")
            st.write("")  # spacer

def render_history():
    # Repaint the chat history (assistant/user/separator messages)
    for role, content in st.session_state.messages:
        if role == "separator":
            st.markdown(f"<div class='mode-sep'>{content}</div>", unsafe_allow_html=True)
            continue
        with st.chat_message(role, avatar=ASSISTANT_AVATAR if role == "assistant" else USER_AVATAR):
            st.markdown(content)

# --- ROUTING ---
if st.session_state.end_chat:
    end_chat_screen()

# --- LANDING (mode chooser) ---
# The landing screen offers three buttons to select a mode and reruns the app to proceed.
# --- LANDING (mode chooser) ---
if st.session_state.mode is None:
    st.markdown(
        """
        <div class="centered-landing">
            <div class="uniscout-logo">UNISCOUT</div>
            <div class="tagline">Helping students choose where to go for college and investigate universities 🎓</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("Hi! I'm **UniScout** 🤖. What would you like to do?")
        col1, col2, col3 = st.columns(3, gap="large")
        st.markdown("""
            <style>
            div[data-testid="column"] div.stButton > button {
                width: 100%;
                height: 4.5rem;
                white-space: normal;
                line-height: 1.2;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)

    with col1:
        if st.button("🔍 Investigate one university", use_container_width=True):
            add_mode_separator("learn")
            st.session_state.mode = "learn"
            user_say("I want to investigate one university.")
            st.experimental_rerun()  # or st.rerun()

    with col2:
        if st.button("📊 Compare universities", use_container_width=True):
            add_mode_separator("compare")
            st.session_state.mode = "compare"
            user_say("I want to compare universities.")
            st.experimental_rerun()

    with col3:
        if st.button("🎯 Recommend me a university", use_container_width=True):
            add_mode_separator("recommend")
            st.session_state.mode = "recommend"
            user_say("Please recommend a university for me.")
            st.experimental_rerun()

    # ⬇️ keep the stop *inside* the landing branch
    st.stop()



# --- CHAT HISTORY (inside chat-style box) ---
# We render the chat area (messages) before the input widgets for each mode.
st.markdown('<div class="stChat">', unsafe_allow_html=True)
render_history()
st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: ASK / COMPARE / RECOMMEND FLOW ---
if st.session_state.mode == "learn":
    # Single-university Q&A (RAG). Show input only when not awaiting the next action.
    if not st.session_state.awaiting_next:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown("Ask me a question about one university (e.g., *What are some traditions at Harvard?*)")
        question = st.chat_input("Type your question about a university…")
        if question:
            user_say(question)
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                with st.spinner("Thinking…"):
                    try:
                        answer = query_rag(question)
                    except Exception as e:
                        answer = f"⚠️ Error: {e}"
                    answer_clean = _strip_wrapping_quotes(answer)
                    st.markdown(answer_clean)          # no quotes, normal text
                    assistant_say(answer_clean)        # store cleaned in history
            st.session_state.awaiting_next = True
            st.experimental_rerun()

elif st.session_state.mode == "compare":
    # Compare the same question across multiple universities using the CLI-style function.
    # We capture its stdout and parse it into a structured dict for display.
    if not st.session_state.awaiting_next:
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
                    buf = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(buf):
                            compare_universities(q.strip(), universities)
                        raw = buf.getvalue()
                    except Exception as e:
                        raw = f"⚠️ Error: {e}"

                    if raw.strip().startswith("⚠️"):
                        st.text(raw)
                        assistant_say(raw)
                    else:
                        data = parse_comparison_stdout(raw)
                        if data:
                            render_comparison(data)
                            # Store a compact text summary in history
                            friendly_summary = "Here’s your comparison:\n\n" + "\n\n".join(
                                [f"**{u}**\n" + "\n".join([f"- {b}" for b in bs]) for u, bs in data.items()]
                            )
                            assistant_say(friendly_summary)
                        else:
                            st.markdown(raw)
                            assistant_say(raw)

            st.session_state.awaiting_next = True
            st.experimental_rerun()

elif st.session_state.mode == "recommend":
    # Collect preference text → get markdown recommendation from the recommender.
    if not st.session_state.awaiting_next:
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(
                "Tell me a bit about you and what you’re looking for. "
                "For example: intended major(s), location preference, class size, budget/aid needs, campus vibe, athletics, etc."
            )

        with st.form("recommend_form", clear_on_submit=False):
            prefs = st.text_area(
                "Your profile & preferences",
                placeholder=(
                    "e.g., Interested in computer science and entrepreneurship, medium city on the East Coast, "
                    "strong internships, generous need-based aid, collaborative vibe, not too big."
                ),
                height=140,
            )
            submitted = st.form_submit_button("Recommend 🎯")

        if submitted:
            user_say(f"My preferences: {prefs}")
            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                with st.spinner("Finding good matches…"):
                    try:
                        md = recommend_universities(prefs.strip())   # ← single markdown string
                    except Exception as e:
                        md = f"⚠️ Error: {e}"

                    render_recommendation_markdown(md)
                    assistant_say(md)  # save markdown directly into chat history

            st.session_state.awaiting_next = True
            st.experimental_rerun()

# --- STEP 3: CONTINUE OR END CHAT ---
# After any action completes, offer follow-up options.
if st.session_state.awaiting_next:
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown("Would you like to do anything else?")

        # equal-width columns; tweak gap to taste ("small" | "medium" | "large")
        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            if st.button("Ask another", use_container_width=True):
                st.session_state.awaiting_next = False
                st.experimental_rerun()

        with col2:
            if st.button("Switch mode", use_container_width=True):
                st.session_state.mode = None
                st.session_state.awaiting_next = False
                st.experimental_rerun()

        with col3:
            if st.button("Done ✅", use_container_width=True):
                st.session_state.end_chat = True
                st.experimental_rerun()
