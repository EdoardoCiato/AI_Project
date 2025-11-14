import argparse #to read command line arguments
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

PROMPT_TEMPLATE = '''You are helping a student compare universities using official College Navigator data.

You will be given:
- the name of ONE target university
- some context, which is guaranteed to come ONLY from that university's profile
- a question about that university

University: {university}
Question: {question}

Context:
{context}

Instructions:
1. Assume ALL of the context refers to the university given above.
2. Carefully read tables and lines that mention tuition, fees, expenses, student population, etc.
3. If the answer can be inferred from the context (even from a table, or even if the university name is not in the same line), extract the **single best number or fact** and answer concisely.
4. If there are both multiple years, pick the value that corresponds to the academic year in the question (e.g. 2024–2025).
5. Only if the context truly does NOT contain enough information to answer, say:
   "The provided text does not contain the information needed to answer this question."

Answer in one or two sentences, and do NOT mention these instructions. '''
CHROMA_PATH = "./chroma_db"

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def detect_university_from_question(question: str, db) -> str | None:
    all_items = db.get(include=["metadatas"])
    universities = set()

    for meta in all_items["metadatas"]:
        if meta is None:
            continue
        uni = meta.get("university")
        if uni:
            universities.add(uni)
    print(universities)
    q = question.lower()
    # Ordina per lunghezza decrescente, così "university of california, berkeley"
    # batte "berkeley"
    for uni in sorted(universities, key=len, reverse=True):
        uni_lower = uni.lower()
        # Spezza "stanford university" in ["stanford", "university"]
        tokens = [t for t in uni_lower.replace(",", " ").replace("-", " ").split() if len(t) > 3]
        print("Provo a matchare:", uni, "con tokens:", tokens)
        print(tokens)
        # Se almeno una parola "forte" è nella domanda → match
        if any(token in q for token in tokens):
            print("Detected university from question:", uni)
            return uni

    return None

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    uni = detect_university_from_question(query_text, db)
    print(uni)
    if uni:
        results = db.similarity_search_with_score(
            query_text,
            k=8,
            filter={"university": uni}

        )
    else:
        # fallback: niente filtro se non capiamo l'università
        results = db.similarity_search_with_score(query_text, k=8)
    chunks = []
    sources = []
    # Search the DB.
    for doc, score in results:
        title = doc.metadata.get("title", "Document")
        page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))

        citation = f"[{title} (p. {page})]"
        chunk_text = f"{citation}\n{doc.page_content}"

        chunks.append(chunk_text)
        sources.append(citation)

    context_text = "\n\n".join(chunks)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, university = uni if uni else "the target university")
    # print(prompt)

    model = Ollama(model="gemma:7b")
    print("===== CONTEXT SENT TO THE MODEL =====")
    print(context_text)
    print("=====================================")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()