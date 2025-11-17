# recommendation.py
"""
Simplified college recommender: Top 2 (Best fit + 1 good option)
- No numeric scoring.
- Picks two universities by context richness (how much indexed text we retrieve).
- Locks the selected 'Best fit' header to avoid drift.
- Bullets are deterministically extracted from retrieved text (no hallucinations).
- LLM is used only for the short "why this fits" paragraph, with guards to avoid invented numbers.

Usage:
  python recommendation.py "I want an urban campus in the Northeast, strong CS, small classes, housing, scholarships, ~60k total, club rowing."
"""

from __future__ import annotations
import argparse
import json
import re
from typing import Dict, List, Tuple

# === Non-deprecated packages ===
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document  # typing clarity

# ====== CONFIG ======
CHROMA_PATH = "./chroma_db"
LLM_MODEL = "llama3.1:latest"   # ollama pull llama3.1:latest

# ---- Embedding helper with robust fallback ----
def _fallback_embedding_function():
    """Prefer Ollama embeddings; fallback to sentence-transformers if needed."""
    try:
        return OllamaEmbeddings(model="nomic-embed-text")  # ollama pull nomic-embed-text
    except Exception:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(
                "No embedding function available. Install one of:\n"
                "- `langchain-ollama` and pull `nomic-embed-text` in Ollama, or\n"
                "- `sentence-transformers` for a local HF model."
            ) from e

try:
    from get_embedding_function import get_embedding_function as _ext_get_emb
    def get_embedding_function():
        return _ext_get_emb()
except Exception:
    def get_embedding_function():
        return _fallback_embedding_function()

# ====== PREFERENCES PARSER ======
SPORT_WORDS = [
    "rowing","crew","basketball","soccer","football","baseball","softball",
    "swimming","tennis","track","cross country","volleyball","hockey",
    "lacrosse","water polo","golf","fencing","wrestling"
]
DISCIPLINE_WORDS = [
    "computer science","cs","data science","statistics","engineering",
    "electrical engineering","mechanical engineering","economics","business",
    "finance","biology","biomedical","chemistry","physics","mathematics",
    "psychology","political science","international relations","english",
    "history","philosophy","art","design","architecture","neuroscience","public policy"
]

def _regex_number(text: str):
    m = re.search(r'(\$?\s*\d{1,3}(?:[,\s]\d{3})+|\$?\s*\d+(?:\.\d+)?\s*[kK]?)', text)
    if not m:
        return None
    raw = m.group(1).lower().replace("$", "").replace(" ", "")
    if raw.endswith("k"):
        return float(raw[:-1]) * 1000.0
    return float(raw.replace(",", ""))

def _regex_list(text: str, vocab: List[str]) -> List[str]:
    found = []
    t = text.lower()
    for w in vocab:
        if re.search(rf'\b{re.escape(w)}\b', t):
            found.append(w)
    return list(dict.fromkeys(found))

def parse_preferences(freetext: str) -> Dict:
    """Heuristic-only parse; fast and deterministic."""
    return {
        "budget_cap": _regex_number(freetext),
        "budget_mode": "total" if re.search(r"total|coa|cost of attendance|room\s*and\s*board", freetext, re.I) else "tuition",
        "requires_scholarships": True if re.search(r"scholarship|financial aid|grant|need[- ]based", freetext, re.I) else None,
        "needs_on_campus_housing": True if re.search(r"housing|dorm|on[- ]campus", freetext, re.I) else None,
        "location_preference": (
            "rural" if re.search(r"\brural\b", freetext, re.I)
            else "urban" if re.search(r"\burban\b|city|downtown|metro|subway", freetext, re.I)
            else None
        ),
        "class_size_preference": "small" if re.search(r"small class|small classes|low student[- ]faculty", freetext, re.I) else None,
        "discipline_keywords": _regex_list(freetext, DISCIPLINE_WORDS),
        "sports_interest": _regex_list(freetext, SPORT_WORDS),
        "ranking_preference": "top 20 percent" if re.search(r"top\s*20", freetext, re.I) else None,
    }

# ====== UNIVERSITY LISTING ======
def list_universities(db: Chroma) -> List[str]:
    """
    Collect unique 'university' names stored in metadatas (handles different key casings).
    Works with langchain_chroma backend.
    """
    items = db.get(include=["metadatas"])
    unis = set()
    for meta in items.get("metadatas", []):
        if not meta:
            continue
        uni = meta.get("university") or meta.get("University") or meta.get("category")
        if uni and isinstance(uni, str) and len(uni) > 3 and "Number of" not in uni:
            unis.add(uni.strip())
    return sorted(unis)

# ====== CONTEXT RETRIEVAL ======
RETRIEVAL_QUERY_TEMPLATE = (
    "2024 2025 tuition fees cost of attendance room board housing scholarships financial aid "
    "undergraduate majors enrollments acceptance rate student-faculty ratio class size athletics clubs sports "
    "{extras}"
)

def _similarity_search_with_filter(db: Chroma, q: str, uni: str, k: int) -> List[Tuple[Document, float]]:
    """Try multiple metadata keys for robustness."""
    for key in ("university", "University", "category"):
        try:
            hits = db.similarity_search_with_score(q, k=max(k, 30), filter={key: uni})
            if hits:
                return hits
        except Exception:
            pass
    return []

def retrieve_context_for_uni(db: Chroma, uni: str, prefs: Dict, k: int = 16) -> str:
    extras = " ".join(prefs.get("discipline_keywords") or []) + " " + " ".join(prefs.get("sports_interest") or [])
    q = RETRIEVAL_QUERY_TEMPLATE.format(extras=extras).strip()

    # 1) Strict filter by metadata
    results = _similarity_search_with_filter(db, q, uni, k)

    # 2) Broaden but keep the same university via metadata equality
    if not results or len(results) < 8:
        try:
            broad = db.similarity_search_with_score(f"{q} {uni}", k=120)
        except Exception:
            broad = []
        filtered = []
        for doc, score in broad:
            m = (doc.metadata or {})
            m_uni = m.get("university") or m.get("University") or m.get("category") or ""
            if isinstance(m_uni, str) and m_uni.strip().lower() == uni.strip().lower():
                filtered.append((doc, score))
        # If still nothing, last-resort: title contains the uni name
        if not filtered:
            for doc, score in broad:
                m = (doc.metadata or {})
                title = (m.get("title") or "").lower()
                if uni.lower() in title:
                    filtered.append((doc, score))
        results = filtered[:max(k, 16)]

    parts = []
    for doc, _ in results:
        txt = (doc.page_content or "").strip()
        if txt:
            parts.append(txt)
    return "\n\n---\n\n".join(parts)

# ====== Deterministic extraction (bullets from text only) ======
SETTING_WORDS = ["urban", "suburban", "rural", "college town", "town"]

def _find_first_case_insensitive(text: str, words: List[str]) -> str | None:
    t = text.lower()
    for w in words:
        if w in t:
            return w
    return None

def _clean_snippet(s: str | None, maxlen: int = 180) -> str | None:
    if not s:
        return None
    s = re.sub(r'\s+', ' ', s).strip()
    # remove noisy numeric tables / long runs
    s = re.sub(r'(\d{2,}[%]?\s*){3,}', ' ', s)
    s = re.sub(r'(?:\$?\d[\d,]*\s*){4,}', ' ', s)
    # smart truncate at sentence boundary
    if len(s) > maxlen:
        cut = s[:maxlen]
        last_dot = cut.rfind('.')
        if last_dot > 60:
            s = cut[:last_dot+1]
        else:
            s = cut + '…'
    return s

def extract_fields(ctx: str) -> dict:
    """
    Very literal extraction—only returns something if we can see it in the context text.
    """
    info = {
        "budget_aid": None,
        "setting": None,
        "academics": None,
        "ratio": None,
        "housing": None,
        "athletics": None,
    }
    if not ctx:
        return info

    # Budget & aid
    m = re.search(r"(financial aid|scholarship|grant|no[-\s]?loan|meets 100%|need[-\s]?blind|merit)", ctx, re.I)
    money = re.search(r"\$\s?\d[\d,]*(?:\.\d+)?\s*(?:per year|/year|annual|tuition|cost|coa|cost of attendance)?", ctx, re.I)
    if m or money:
        start = (m.start() if m else money.start())
        end = (m.end() if m else money.end())
        span = ctx[max(0, start-80): min(len(ctx), end+160)]
        info["budget_aid"] = _clean_snippet(" ".join(span.split()))

    # Setting
    w = _find_first_case_insensitive(ctx, SETTING_WORDS)
    if w:
        info["setting"] = w

    # Academics (CS/DS)
    m = re.search(r"(computer science|data science|statistics|cs department|school of engineering)", ctx, re.I)
    if m:
        span = ctx[max(0, m.start()-60): min(len(ctx), m.end()+120)]
        info["academics"] = _clean_snippet(" ".join(span.split()))

    # Student–faculty ratio / class size
    m = re.search(r"(\b\d{1,2}\s*:\s*\d{1,2}\b|\bstudent[-\s]?faculty ratio\b|class size)", ctx, re.I)
    if m:
        span = ctx[max(0, m.start()-30): min(len(ctx), m.end()+90)]
        info["ratio"] = _clean_snippet(" ".join(span.split()), 120)

    # Housing
    m = re.search(r"(on[-\s]?campus housing|residential college|residence hall|dorm|guaranteed housing)", ctx, re.I)
    if m:
        span = ctx[max(0, m.start()-40): min(len(ctx), m.end()+140)]
        info["housing"] = _clean_snippet(" ".join(span.split()))

    # Athletics / clubs (prefer rowing if present)
    m_row = re.search(r"(rowing|crew)", ctx, re.I)
    if m_row:
        span = ctx[max(0, m_row.start()-40): min(len(ctx), m_row.end()+120)]
        info["athletics"] = _clean_snippet(" ".join(span.split()))
    else:
        m_any = re.search(r"(athletics|varsity|club sports|intramural)", ctx, re.I)
        if m_any:
            span = ctx[max(0, m_any.start()-40): min(len(ctx), m_any.end()+140)]
            info["athletics"] = _clean_snippet(" ".join(span.split()))

    return info

# ====== Helpers ======
def _trim(txt: str, limit: int = 4000) -> str:
    return (txt or "")[:limit]

def _pick_top2_by_context(db: Chroma, prefs: Dict, all_unis: List[str]) -> List[Tuple[str, str]]:
    """Pick two universities with the richest retrieved context."""
    items: List[Tuple[str, str]] = []
    for uni in all_unis:
        ctx = retrieve_context_for_uni(db, uni, prefs, k=16)
        if ctx and ctx.strip():
            items.append((uni, ctx))
    items.sort(key=lambda x: len(x[1]), reverse=True)
    if len(items) >= 2 and items[0][0] == items[1][0]:
        for cand in items[2:]:
            if cand[0] != items[0][0]:
                items[1] = cand
                break
    return items[:2]

def _enforce_consistency(markdown: str, best_uni: str) -> str:
    """Only force the Best fit header to match the chosen university."""
    text = markdown or ""
    if re.search(r"^#\s*Best fit:.*$", text, flags=re.M):
        text = re.sub(r"^#\s*Best fit:.*$", f"# Best fit: {best_uni}", text, flags=re.M)
    else:
        text = f"# Best fit: {best_uni}\n\n" + text
    return text

def _strip_numbers_if_no_budget(text: str, has_budget: bool) -> str:
    """If no budget/aid evidence was found in context, scrub specific numbers from the 'why fit' prose."""
    if has_budget:
        return text
    return re.sub(r'(\$?\d[\d,]*(\.\d+)?\s*(k|K)?(%|\b))|(\$+\s?\d[\d,]*)', ' ', text)

# ====== MAIN RECOMMENDER ======
def recommend(freetext: str) -> str:
    prefs = parse_preferences(freetext)

    emb = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)

    all_unis = list_universities(db)
    if not all_unis:
        return "I couldn't find any universities in the database. Populate the DB first."

    picked = _pick_top2_by_context(db, prefs, all_unis)
    if not picked:
        return "No relevant universities found in the database."

    best_uni, best_ctx = picked[0]
    if len(picked) > 1:
        c1_uni, c1_ctx = picked[1]
    else:
        c1_uni, c1_ctx = ("—", "")

    # Deterministic extraction for bullets
    best_fields = extract_fields(best_ctx)
    c1_fields   = extract_fields(c1_ctx)

    # LLM writes only the short "why fit" paragraph, grounded by context
    WHY_PROMPT = ChatPromptTemplate.from_template(
        "You are a college advisor. In 2–3 sentences, explain why {uni} fits the student "
        "based ONLY on this context. If something is missing, explicitly say 'not found'. "
        "Avoid specific numbers or income thresholds unless they appear verbatim in the context.\n\n"
        "Context:\n{ctx}\n"
    )
    llm = OllamaLLM(model=LLM_MODEL, temperature=0.2)
    best_why = llm.invoke(WHY_PROMPT.format(uni=best_uni, ctx=_trim(best_ctx, 3500)))
    c1_why   = llm.invoke(WHY_PROMPT.format(uni=c1_uni,   ctx=_trim(c1_ctx,   3500))) if c1_uni != "—" else ""

    # Guard: strip precise numbers if we found no budget/aid snippet
    best_why = _strip_numbers_if_no_budget(best_why, bool(best_fields["budget_aid"])).strip()
    if c1_why:
        c1_why = _strip_numbers_if_no_budget(c1_why, bool(c1_fields["budget_aid"])).strip()

    def _bullet(v, fallback="not found"):
        return v if (v and str(v).strip()) else fallback

    # Build markdown deterministically
    md = []
    md.append(f"# Best fit: {best_uni}")
    md.append("**Why this university fits (2–3 sentences):**")
    md.append(best_why)
    md.append("")
    md.append("**Highlights from the indexed info:**")
    md.append(f"- **Budget & aid:** {_bullet(_clean_snippet(best_fields['budget_aid']))}")
    md.append(f"- **Setting & campus vibe:** {_bullet(_clean_snippet(best_fields['setting']))}")
    md.append(f"- **Academics/programs:** {_bullet(_clean_snippet(best_fields['academics']))}")
    md.append(f"- **Class size / student–faculty ratio:** {_bullet(_clean_snippet(best_fields['ratio']))}")
    md.append(f"- **Housing/residential life:** {_bullet(_clean_snippet(best_fields['housing']))}")
    md.append(f"- **Athletics & clubs (e.g., rowing):** {_bullet(_clean_snippet(best_fields['athletics']))}")
    md.append("")
    md.append("**Potential trade-offs / unknowns:**")
    missing = [k for k,v in best_fields.items() if v is None]
    if missing:
        for k in missing[:3]:
            md.append(f"* {k.replace('_',' ')} not explicitly found in the brochure text.")
    else:
        md.append("* none noted in the provided excerpts.")
    md.append("")
    md.append("# Other good option")
    if c1_uni != "—":
        md.append(f"1) **{c1_uni}** — {c1_why if c1_why else 'Another promising option based on the available context.'}")
    else:
        md.append("1) **—** — not enough context for a second option.")

    return _enforce_consistency("\n".join(md), best_uni)

# ====== CLI ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preferences", type=str, help="Free-text preferences from the student")
    args = ap.parse_args()
    md = recommend(args.preferences)
    print("\n=== Recommendation ===\n")
    print(md)

if __name__ == "__main__":
    main()
