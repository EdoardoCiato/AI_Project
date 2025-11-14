import argparse
from typing import List, Dict

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from get_embedding_function import get_embedding_function
import argparse
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma_db"
PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions about universities
using ONLY the context provided.

If the answer is not explicitly stated in the context, say:
"I cannot find this information in the provided documents."

Question: {question}

Context:
{context}

Answer in one short sentence.
"""
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
        system_prompt=PROMPT_TEMPLATE,
        question=question,
        context=context,
    )

def compare_as_table(question: str, universities: List[str]) -> None:
    """Stampa una tabella ASCII con una riga per università."""
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    answers: Dict[str, str] = {}

    for uni in universities:
        context = retrieve_context_for_university(db, question, uni, k=6)
        answer = ask_model(question, context, Ollama(model="gemma:7b"))
        answers[uni] = answer

    # 📊 Costruzione tabella
    col1 = "University"
    col2 = "Answer"
    col1_width = max(len(col1), max(len(u) for u in universities))
    col2_width = 80  # tronchiamo la risposta per stare su una riga

    header = f"{col1.ljust(col1_width)} | {col2}"
    sep = f"{'-' * col1_width}-+-{'-' * col2_width}"
    print()
    print(header)
    print(sep)

    for uni in universities:
        full_ans = answers[uni].replace("\n", " ")
        ans_short = (full_ans[: col2_width - 3] + "...") if len(full_ans) > col2_width else full_ans
        print(f"{uni.ljust(col1_width)} | {ans_short}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_question", type=str, help="Question to ask")
    parser.add_argument("universities", nargs="+", help="List of universities to compare")
    args = parser.parse_args()

    compare_as_table(args.base_question, args.universities)


if __name__ == "__main__":
    main()