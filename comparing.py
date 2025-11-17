# comparing.py
import argparse
import string
from typing import List, Dict, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

from get_embedding_function import get_embedding_function

# ---- Config ----
CHROMA_PATH = "./chroma_db"
LLM_MODEL = "gemma:7b"  # e.g., "llama3.2:3b-instruct" for speed

UNIGUIDE_PROMPT = """
You are UniGuide — a knowledgeable, friendly student ambassador helping someone learn about ONE university.
Answer ONLY using the provided context. If a detail isn't in the context, say so politely.

University: {university}
Question: {question}

Context:
{context}

Your role:
1) Read the context and find the most relevant information that answers the question.
2) Write in a warm, confident, student-like tone — helpful but not salesy.
3) Output 2–3 short bullet points (no more than ~60 words total). 
   - Start DIRECTLY with bullets. 
   - Do NOT include intros or filler like “That’s a great question”, “I’m glad you asked”, “Here’s a quick overview/comparison”, or “Sure!”.
   - Bullet 1: The core answer for {university}.
   - Bullet 2–3: One-sentence supporting details (key resources, how it works, what to expect).
4) If the context doesn’t include the answer, write one bullet:
   - "The provided materials don’t mention that detail, but it might be available on the university’s website."

Do NOT add anything not supported by the context. Keep bullets crisp and readable.
"""

# ---- Name normalization / aliases ----
ALIASES = {
    "ucla": "University Of California-Los Angeles",
    "berkeley": "University Of California-Berkeley",
    "uc berkeley": "University Of California-Berkeley",
    "harvard": "Harvard University",
    "princeton": "Princeton University",
    "brown": "Brown University",
}
def canonicalize(name: str) -> str:
    n = name.strip().lower()
    n = n.strip(string.punctuation + " ")
    return ALIASES.get(n, name.strip().strip(string.punctuation + " "))


def build_prompt(question: str, context: str, university: str) -> str:
    tmpl = ChatPromptTemplate.from_template(
        """SYSTEM:
{system_prompt}

USER:
University: {university}
Question: {question}

Context:
{context}

ASSISTANT:
- """  # seed a bullet so models stay in bullet mode
    )
    return tmpl.format(system_prompt=UNIGUIDE_PROMPT, question=question, context=context, university=university)


def ask_model(question: str, context_text: str, university: str, model: Ollama) -> str:
    prompt = build_prompt(question, context_text, university)
    return (model.invoke(prompt) or "").strip()


def retrieve_context_for_university(
    db: Chroma, base_question: str, university_key: str, k: int = 10
) -> Tuple[str, list]:
    """
    Try filtered retrieval first (metadata["university"] == canonical).
    If too few hits, do a broader search and filter by source/title/university fields.
    """
    uk_canon = canonicalize(university_key)
    q = f"{base_question} {uk_canon}"

    # 1) Try strict metadata filter
    try:
        results = db.similarity_search_with_score(q, k=40, filter={"university": uk_canon})
    except Exception:
        results = []

    # 2) If not enough, broaden and filter heuristically
    if not results or len(results) < 6:
        broad = db.similarity_search_with_score(q, k=80)
        filtered = []
        uk_l = uk_canon.lower()
        for doc, score in broad:
            meta_uni = (doc.metadata.get("university") or "").lower()
            src = (doc.metadata.get("source") or "").lower()
            title = (doc.metadata.get("title") or "").lower()
            if uk_l in meta_uni or uk_l in src or uk_l in title:
                filtered.append((doc, score))
        results = filtered[:max(k, 12)] or broad[:k]

    # Build context
    context_parts, sources = [], []
    for doc, score in results[:k]:
        if not doc.page_content:
            continue
        context_parts.append(doc.page_content)
        sources.append(doc.metadata.get("id", doc.metadata.get("source", "unknown")))

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, sources


def _normalize_bullets(text: str) -> List[str]:
    """
    Make sure we return 2–3 clean bullets max, no empties, no numbering,
    and trim whitespace. If the model returns paragraphs, split by lines/sentences.
    """
    # Split by lines first
    lines = [ln.strip(" -*•\t") for ln in text.strip().splitlines() if ln.strip()]
    # If it looks like a single paragraph, split into sentences
    if len(lines) <= 1:
        raw = lines[0] if lines else ""
        # simple sentence split fallback
        parts = [p.strip() for p in raw.replace("•", ". ").split(". ") if p.strip()]
    else:
        parts = lines

    # Keep 1–3 bullets
    parts = parts[:3] if parts else []
    # Remove trailing punctuation dots for bullet cleanliness (optional)
    cleaned = []
    for p in parts:
        cleaned.append(p.strip())
    # Ensure at least one bullet if empty
    return cleaned or ["The provided materials don’t mention that detail, but it might be available on the university’s website."]


def compare_universities(question: str, universities: List[str]) -> None:
    """Print a UniGuide-style, bullet-point comparison (one section per university)."""
    universities = [canonicalize(u) for u in universities if u.strip()]

    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    model = Ollama(model=LLM_MODEL)

    print("\nComparison:\n")
    for uni in universities:
        context_text, _sources = retrieve_context_for_university(db, question, uni, k=10)
        q_for_uni = f"{question} (about {uni})"
        raw_answer = ask_model(q_for_uni, context_text, uni, model)
        bullets = _normalize_bullets(raw_answer)

        print(f"- {uni}")
        for b in bullets:
            print(f"  - {b}")
        print()  # blank line between schools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_question", type=str, help="Question to ask each university")
    parser.add_argument("universities", nargs="+", help="List of universities to compare")
    args = parser.parse_args()

    compare_universities(args.base_question, args.universities)


if __name__ == "__main__":
    main()
