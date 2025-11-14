import argparse
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma_db"

SYSTEM_PROMPT = """You are UniScout, a careful admissions assistant.

Answer ONLY from the provided context.

- If the question asks about tuition or “tuition and fees” for a specific academic year (e.g. 2024-2025), and the context contains tables or rows with that year and tuition-related values, you MUST extract the relevant numbers from those rows.
- If there are multiple categories (e.g. in-state vs out-of-state, undergraduate vs graduate), list them all and clearly label each category.
- If it is genuinely impossible to identify any tuition-related value for that year in the context, then say you don't know and suggest where it might be (e.g., financial aid or cost of attendance page).

Always answer concisely and factually.

Include simple citations like:
[University — College Navigator (2025), p. X]
"""


def build_prompt(question: str, context: str) -> str:
    prompt_template = ChatPromptTemplate.from_template(
        """SYSTEM:
{system_prompt}

USER:
Question: {question}

Context:
{context}

Answer:"""
    )
    return prompt_template.format(
        system_prompt=SYSTEM_PROMPT,
        question=question,
        context=context,
    )


def retrieve_context_for_university(
    db: Chroma, question: str, university_key: str, k: int = 8
) -> Tuple[str, list]:
    """
    Recupera i chunk più rilevanti per una università.

    NON uso più il filtro di Chroma perché $contains non è supportato.
    Faccio:
    1. similarity_search_with_score senza filtro
    2. filtro in Python sui documenti il cui metadata["source"]
       contiene university_key (case-insensitive).
    """

    # Cerco un po’ più di k documenti, così il filtro ha spazio per lavorare
    raw_results = db.similarity_search_with_score(question, k=30)

    filtered_results = []
    uk = university_key.lower()

    for doc, score in raw_results:
        src = str(doc.metadata.get("source", "")).lower()
        if uk in src:
            filtered_results.append((doc, score))

    # Se il filtro non trova nulla, facciamo fallback ai risultati grezzi
    if not filtered_results:
        filtered_results = raw_results

    # Teniamo solo i primi k
    filtered_results = filtered_results[:k]

    context_parts = []
    sources = []
    for doc, score in filtered_results:
        context_parts.append(doc.page_content)
        src = doc.metadata.get("id", doc.metadata.get("source", "unknown"))
        sources.append(src)

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, sources


def ask_model(question: str, context: str, model: Ollama) -> str:
    prompt = build_prompt(question, context)
    response = model.invoke(prompt)
    return response


def compare_universities(base_question: str, universities: List[str]) -> None:
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    model = Ollama(model="gemma:7b")

    results = []

    for uni in universities:
        # Rendo la domanda leggermente specifica per l’uni
        question = f"{base_question} ({uni})"
        context_text, sources = retrieve_context_for_university(db, question, uni)

        answer = ask_model(question, context_text, model)

        results.append(
            {
                "university": uni,
                "answer": answer,
                "sources": sources,
            }
        )

    print("\n================ UNIVERSITY COMPARISON ================\n")
    print(f"Question: {base_question}\n")

    for res in results:
        print(f"▶ {res['university']}")
        print(res["answer"].strip())
        print(f"Sources: {res['sources']}")
        print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the same question across multiple universities."
    )
    parser.add_argument(
        "base_question",
        type=str,
        help='Domanda base, es. "What is the tuition for 2024-2025?"',
    )
    parser.add_argument(
        "universities",
        nargs="+",
        help='Lista di chiavi per le università, es. "Harvard" "Stanford" "Berkeley"',
    )

    args = parser.parse_args()
    compare_universities(args.base_question, args.universities)