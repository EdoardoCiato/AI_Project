# load_documents_fixed.py
from langchain_core.documents import Document
import json
from pathlib import Path
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from ingest import add_to_chroma

FILE_PATH = Path("./data/colleges_long.jsonl")
CHROMA_PATH = "./chroma_db"

def _is_universityish(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in ["university", "college", "institute", "polytechnic"])

def _looks_like_header(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in ["academic year", "general information", "overview", "summary"])

def _normalize_mapping(r: dict) -> dict:
    """
    Fix records where 'university' contains a header (e.g., 'General information: Academic year 2024-25')
    and 'category' actually contains the school name.
    """
    uni = str(r.get("university", "") or "")
    cat = str(r.get("category", "") or "")

    if _looks_like_header(uni) and _is_universityish(cat):
        # swap
        uni, cat = cat, uni

    # Forward-fill 'variable' is not generally possible in JSONL alone,
    # but keep whatever is present.
    var = r.get("variable", None)
    det = r.get("detail", None)
    val = r.get("value", None)
    sheet = r.get("sheet", None)

    return {
        "university": str(uni),
        "category": str(cat) if cat is not None else "",
        "variable": "" if var is None else str(var),
        "detail": "" if det in (None, float("nan")) else str(det),
        "value": "" if val is None else str(val),
        "sheet": "" if sheet is None else str(sheet),
    }

import re

def _infer_variable(rr: dict) -> str:
    """
    Try to infer a useful variable name when it's blank.
    - Heuristic: rows in 'Institution Characteristics' + 'General information' + 5–9 digit numeric value → 'UnitId'
    - Otherwise return a generic label.
    """
    var = (rr.get("variable") or "").strip()
    if var:
        return var

    sheet = (rr.get("sheet") or "").lower()
    category = (rr.get("category") or "").lower()
    value = str(rr.get("value") or "").strip()

    # Looks like a UnitId (your sample case)

    # Add other tiny heuristics here if you like, e.g.:
    # if value.startswith("http"): return "Website"
    # if value.endswith("%"): return "Percentage"

    return "Unknown variable"


def load_documents(file_path=FILE_PATH):
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rr = _normalize_mapping(r)

            # NEW: ensure variable is never empty
            variable = _infer_variable(rr)

            # Build readable, self-contained embedding text
            page_content = " | ".join([
                f"University: {rr['university']}",
                f"Sheet: {rr['sheet']}",
                f"Category: {rr['category']}",
                f"Variable: {variable}",
                f"Detail: {rr['detail']}" if rr['detail'] else "Detail: (none)",
                f"Value: {rr['value']}",
            ])

            metadata = {
                "university": rr["university"],
                "sheet": rr["sheet"],
                "category": rr["category"],
                "variable": variable,          # <- store the inferred variable
                "detail": rr["detail"],
                "value": rr["value"],
            }

            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs



def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Crea un ID unico per ogni chunk, tipo:
    data/College Navigator - Harvard University.pdf:0:0
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

# --- verify ---
if __name__ == "__main__":
    docs = load_documents()
    print(f"\n✅ Loaded {len(docs)} cleaned facts from {FILE_PATH.name}\n")

    # Show 5 samples that should now have the correct university field
    shown = 0
    for d in docs:
        if "Academic year" not in d.metadata["university"]:  # sanity condition
            print("--------------------------------------------------")
            print(f"University: {d.metadata.get('university')}")
            print(f"Sheet: {d.metadata.get('sheet')}")
            print(f"Category: {d.metadata.get('category')}")
            print(f"Variable: {d.metadata.get('variable')}")
            print(f"Value: {d.metadata.get('value')}")
            print(f"Page Content: {d.page_content[:200]}...")
            shown += 1
            if shown >= 5:
                break

    add_to_chroma(docs)
