# recommendation.py
"""
RAG-based recommender:
- Input: one free-text sentence describing the student's wants/identity.
- Output: friendly recommendation + short ranked list with reasons.

Usage:
  python recommendation.py "I want an urban campus near Boston with small classes, strong CS, budget 70k total, must offer scholarships, I love rowing."
"""

from __future__ import annotations
import argparse
import json
import re
from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma_db"
# Choose a responsive local model (pull it first via: ollama pull llama3.2:3b-instruct)
LLM_MODEL = "llama3.2:1b"
 # or "gemma:7b" if you want larger

# -------------------------------
# 1) Parse user preferences (fast heuristic first; optional LLM refine)
# -------------------------------

PREFS_PROMPT = ChatPromptTemplate.from_template(
    """Extract a clean JSON of preferences from the text below.

TEXT:
{text}

Return JSON with EXACT keys:
{{
  "budget_cap": number or null,                  // numeric annual budget (no currency symbols)
  "budget_mode": "total" | "tuition",            // "total" for total cost/COA; else "tuition"
  "requires_scholarships": true/false/null,
  "needs_on_campus_housing": true/false/null,
  "location_preference": string or null,         // e.g. "urban", "rural", "California", "Northeast", "near Boston"
  "class_size_preference": "small"|"medium"|"large"|null,
  "discipline_keywords": [string, ...],          // e.g. ["computer science","business","engineering"]
  "sports_interest": [string, ...],              // e.g. ["rowing","basketball","soccer"]
  "ranking_preference": string or null           // e.g. "top 20 percent", "highly ranked", "no preference"
}}

Rules:
- budget_cap: use numbers only, no $ or commas.
- If something isn't stated, set it to null.
Return ONLY the JSON.
"""
)

SPORT_WORDS = [
    "rowing","crew","basketball","soccer","football","baseball","softball","swimming",
    "tennis","track","cross country","volleyball","hockey","lacrosse","water polo",
    "golf","fencing","wrestling"
]

DISCIPLINE_WORDS = [
    "computer science","cs","data science","statistics","engineering","electrical engineering",
    "mechanical engineering","economics","business","finance","biology","biomedical",
    "chemistry","physics","mathematics","psychology","political science","international relations",
    "english","history","philosophy","art","design","architecture","neuroscience","public policy"
]

def _regex_number(text: str) -> Optional[float]:
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
        # allow partial “cs” etc. as whole word
        if re.search(rf'\b{re.escape(w)}\b', t):
            found.append(w)
    return list(dict.fromkeys(found))  # dedupe preserve order

def parse_preferences(freetext: str) -> Dict:
    """Heuristic parse first (fast), then try to refine with LLM JSON; never crash."""
    base = {
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

    # Try to refine with LLM (optional, will be skipped if Ollama not ready)
    try:
        llm = Ollama(model=LLM_MODEL)
        js = llm.invoke(PREFS_PROMPT.format(text=freetext))
        llm_dict = json.loads(js)
        for k in base.keys():
            if k in llm_dict and llm_dict[k] is not None:
                base[k] = llm_dict[k]
    except Exception:
        pass

    return base

# -------------------------------
# 2) Retrieve context per university (robust)
# -------------------------------

ALIASES = {
    "harvard": "Harvard University",
    "princeton": "Princeton University",
    "brown": "Brown University",
    "ucla": "University Of California-Los Angeles",
    "uc berkeley": "University Of California-Berkeley",
    "berkeley": "University Of California-Berkeley",
}

def _canon(name: str) -> str:
    n = (name or "").strip().lower()
    return ALIASES.get(n, name)

def list_universities(db: Chroma) -> List[str]:
    items = db.get(include=["metadatas"])
    unis = set()
    for meta in items["metadatas"]:
        if meta and meta.get("university"):
            unis.add(_canon(meta["university"]))
    return sorted(unis)

RETRIEVAL_QUERY_TEMPLATE = (
    "2024 2025 tuition fees total cost cost of attendance room board housing scholarships financial aid "
    "undergraduate enrollment acceptance rate class size student-faculty ratio programs majors athletics sports teams "
    "{extras}"
)

def retrieve_context_for_uni(db: Chroma, uni: str, prefs: Dict, k: int = 16) -> str:
    extras = " ".join(prefs.get("discipline_keywords") or []) + " " + " ".join(prefs.get("sports_interest") or [])
    q = RETRIEVAL_QUERY_TEMPLATE.format(extras=extras).strip()

    # Try strict metadata filter
    try:
        results = db.similarity_search_with_score(q, k=60, filter={"university": uni})
    except Exception:
        results = []

    # Broaden if needed: search then filter by filename/title/university fields
    if not results or len(results) < 8:
        broad = db.similarity_search_with_score(f"{q} {uni}", k=100)
        filtered = []
        uk_l = uni.lower()
        for doc, score in broad:
            meta_uni = (doc.metadata.get("university") or "").lower()
            src = (doc.metadata.get("source") or "").lower()
            title = (doc.metadata.get("title") or "").lower()
            if uk_l in meta_uni or uk_l in src or uk_l in title:
                filtered.append((doc, score))
        results = filtered[:max(k, 16)] or broad[:k]

    parts = [doc.page_content for doc, _ in results if (doc.page_content or "").strip()]
    return "\n\n---\n\n".join(parts)

# -------------------------------
# 3) Score each university (LLM JSON; fallback to rules if LLM fails)
# -------------------------------

SCORE_PROMPT = ChatPromptTemplate.from_template(
    """You are ranking how well a university matches a student's preferences.

Preferences (JSON):
{prefs}

Context from this university's official materials:
{context}

Tasks:
1) In 1–2 lines, identify concrete matches (budget fit, scholarships/aid, housing, location vibe, class size, notable programs, athletics).
2) Output JSON only:
{{
  "match_score": number 0..100,
  "why": "one-sentence reason that cites those matches"
}}
"""
)

def _rule_score(context: str, prefs: Dict) -> Tuple[float, str]:
    """Quick, deterministic scoring if the LLM is unavailable."""
    c = (context or "").lower()
    score, why = 0.0, []

    # Scholarships / aid
    if prefs.get("requires_scholarships"):
        if re.search(r"scholarship|financial aid|need[- ]based|grant", c):
            score += 20; why.append("strong scholarships/aid mentioned")
        else:
            score -= 10; why.append("aid not evident")

    # Housing
    if prefs.get("needs_on_campus_housing"):
        if re.search(r"on[- ]campus housing|residential (college|house)|dorm", c):
            score += 10; why.append("on-campus housing / residential system")
        else:
            score -= 6; why.append("housing not clear")

    # Class size (very rough)
    if prefs.get("class_size_preference") == "small":
        if re.search(r"student[- ]faculty\s*ratio\s*[:=]\s*\d+\s*:\s*\d+", c) or "small class" in c:
            score += 6; why.append("small classes / low student-faculty ratio")

    # Location vibe
    loc = prefs.get("location_preference")
    if loc == "urban" and re.search(r"urban|city|downtown|subway|metro|museum|stadium", c):
        score += 4; why.append("urban setting")
    if loc == "rural" and re.search(r"rural|suburban|wooded|campus town", c):
        score += 4; why.append("rural/suburban vibe")

    # Budget (naive): look for a few dollar-like numbers and compare with cap
    cap = prefs.get("budget_cap")
    if cap:
        nums = [int(n.replace(",", "")) for n in re.findall(r"\$?\b(\d{2,3}[,]?\d{3})\b", context)]
        if nums:
            m = min(nums)
            if m <= cap:
                score += 10; why.append(f"cost snippet ≲ {m} within budget")
            else:
                score -= 10; why.append(f"cost snippet ≳ {m} over budget")
        else:
            why.append("no explicit cost found")

    # Sports interest hints
    sports = prefs.get("sports_interest") or []
    if sports:
        if re.search(r"varsity|ncaa|club sport|intramural|athletics|recreation|fitness", c):
            score += 4; why.append("athletics/recreation opportunities")

    return score, "; ".join(why) or "basic match"

def score_university_with_llm(uni: str, context: str, prefs: Dict) -> Tuple[float, str]:
    try:
        llm = Ollama(model=LLM_MODEL)
        js = llm.invoke(SCORE_PROMPT.format(prefs=json.dumps(prefs), context=context[:4000]))
        data = json.loads(js)
        return float(data.get("match_score", 0.0)), str(data.get("why", ""))
    except Exception:
        return _rule_score(context, prefs)

# -------------------------------
# 4) Friendly final recommendation
# -------------------------------

FINAL_PROMPT = ChatPromptTemplate.from_template(
    """You are a warm, practical college advisor. Based on the ranked candidates below,
write a concise recommendation (3–6 sentences) naming the best fit first and why.
Blend in concrete facts (aid/housing/setting/class-size/programs/athletics) that appeared in the context.
Then suggest 1–2 concise alternatives with a quick reason.

Student Preferences:
{prefs}

Ranked Candidates (JSON array of {{"university","score","why"}}):
{ranked}
"""
)

def recommend(freetext: str, top_k: int = 5) -> Tuple[str, List[Tuple[str, float, str]]]:
    prefs = parse_preferences(freetext)

    emb = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)

    universities = list_universities(db)
    if not universities:
        return "I couldn't find any universities in the database. Run populate_database.py first.", []

    scored: List[Tuple[str, float, str]] = []
    for uni in universities:
        ctx = retrieve_context_for_uni(db, uni, prefs, k=16)
        if not ctx.strip():
            scored.append((uni, 0.0, "No context available."))
            continue
        s, why = score_university_with_llm(uni, ctx, prefs)
        scored.append((uni, s, why))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    try:
        llm = Ollama(model=LLM_MODEL)
        rec = llm.invoke(FINAL_PROMPT.format(
            prefs=json.dumps(prefs, ensure_ascii=False),
            ranked=json.dumps(
                [{"university": u, "score": round(s,1), "why": w} for (u,s,w) in top],
                ensure_ascii=False
            ),
        ))
    except Exception:
        # Simple fallback paragraph
        if not top:
            return "I couldn't produce a recommendation — no candidates were found.", []
        best = top[0][0]
        rec = f"{best} looks like the best fit based on your preferences. Here are the next options: " + ", ".join(u for (u, _, _) in top[1:])

    return rec, top

# -------------------------------
# 5) CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("preferences", type=str, help="Free-text preferences from the student")
    args = parser.parse_args()
    rec_text, top = recommend(args.preferences, top_k=5)

    print("\n=== Recommendation ===\n")
    print(rec_text)
    print("\n=== Top matches ===")
    for u, s, w in top:
        print(f"- {u}: {round(s,1)} — {w}")

if __name__ == "__main__":
    main()
