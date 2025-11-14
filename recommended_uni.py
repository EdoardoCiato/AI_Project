# catalog_builder.py

from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma_db"

# ========= PROMPT PER ESTRARRE IL CATALOGO =========

PROMPT_TEMPLATE_CATALOG = """
You are a data extraction assistant. Your task is to build a clean and structured
catalog entry for a single university using ONLY the information in the context.

University: {university}

Context:
{context}

For this university, produce exactly ONE entry with the following fields.
Use "unknown" if the information is not available in the text:

- name
- location_city
- location_state_or_region
- institution_type                    # e.g. "4-year, Private not-for-profit"
- tuition_2024_2025
- fees_2024_2025
- total_cost_on_campus_2024_2025
- books_supplies_2024_2025
- application_fee_2024_2025
- in_state_tuition_2024_2025          # if applicable
- out_of_state_tuition_2024_2025      # if applicable
- undergraduate_enrollment
- acceptance_rate                     # if available

- sat_ebrw_25th                       # SAT Evidence-Based Reading and Writing
- sat_ebrw_50th
- sat_ebrw_75th
- sat_math_25th
- sat_math_50th
- sat_math_75th

- act_composite_25th
- act_composite_50th
- act_composite_75th

- notable_programs_or_features        # optional short bullet list

Rules:
1. Extract ONLY facts explicitly mentioned in the context.
2. Do NOT guess or invent values.
3. Use numeric values when possible (no dollar signs, no commas).
4. If the same field appears multiple times, use the value for 2024–2025 when available.
5. Keep the style compact and uniform.

Output format (strict, valid JSON-like):

UNIVERSITY_CATALOG = [
  {{
    "name": "...",
    "location_city": "...",
    "location_state_or_region": "...",
    "institution_type": "...",
    "tuition_2024_2025": ...,
    "fees_2024_2025": ...,
    "total_cost_on_campus_2024_2025": ...,
    "books_supplies_2024_2025": ...,
    "application_fee_2024_2025": ...,
    "in_state_tuition_2024_2025": ...,
    "out_of_state_tuition_2024_2025": ...,
    "undergraduate_enrollment": ...,
    "acceptance_rate": "...",
    "sat_ebrw_25th": ...,
    "sat_ebrw_50th": ...,
    "sat_ebrw_75th": ...,
    "sat_math_25th": ...,
    "sat_math_50th": ...,
    "sat_math_75th": ...,
    "act_composite_25th": ...,
    "act_composite_50th": ...,
    "act_composite_75th": ...,
    "notable_programs_or_features": [...]
  }}
]

Return ONLY the UNIVERSITY_CATALOG block, nothing else.
"""

# ========= UTILS: LEGGERE LE UNIVERSITÀ DAL DB =========

def get_list_of_universities() -> List[str]:
    """Legge tutte le università distinte dal metadato 'university' in Chroma."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    all_items = db.get(include=["metadatas"])
    universities = set()

    for meta in all_items["metadatas"]:
        if meta is None:
            continue
        uni = meta.get("university")
        if uni:
            universities.add(uni)

    return sorted(list(universities))


def get_context_for_university(university: str, k: int = 12) -> str:
    """
    Recupera i chunk più rilevanti per costruire il catalogo di una singola università.
    Usa una query generica che punta a costi, tasse, iscrizione ecc.
    """
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    retrieval_question = (
        "tuition, fees, total expenses, total cost on campus, books, supplies, "
        "application fee, in-state tuition, out-of-state tuition, undergraduate enrollment, "
        "acceptance rate, programs, financial aid"
    )

    results = db.similarity_search_with_score(
        retrieval_question,
        k=k,
        filter={"university": university},
    )

    chunks = []
    for doc, score in results:
        chunks.append(doc.page_content)

    context_text = "\n\n---\n\n".join(chunks)
    return context_text


# ========= COSTRUTTORE DEL CATALOGO PER UNA UNIVERSITÀ =========

def build_university_catalog_entry(university: str) -> str:
    """Costruisce l'entry di catalogo per una singola università."""
    context = get_context_for_university(university)

    if not context.strip():
        return f"# {university}\nUNIVERSITY_CATALOG = []\n"

    model = Ollama(model="gemma:7b")

    prompt = PROMPT_TEMPLATE_CATALOG.format(
        university=university,
        context=context,
    )

    answer = model.invoke(prompt)
    return f"# {university}\n{answer}\n"


# ========= MAIN: COSTRUISCE IL CATALOGO PER TUTTE LE UNIVERSITÀ =========

def main():
    universities = get_list_of_universities()
    print("Found universities in DB:")
    for u in universities:
        print(" -", u)
    print("\nBuilding catalog...\n")

    full_catalog = ""

    for uni in universities:
        entry = build_university_catalog_entry(uni)
        full_catalog += entry + "\n"

    # Stampa su console
    print("===== UNIVERSITY CATALOG =====\n")
    print(full_catalog)

    # Opzionale: salva su file
    with open("university_catalog.txt", "w", encoding="utf-8") as f:
        f.write(full_catalog)

    print("\nCatalog saved to university_catalog.txt")


if __name__ == "__main__":
    main()