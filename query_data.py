import argparse #to read command line arguments
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma



PROMPT_TEMPLATE = """
You are UniGuide — a knowledgeable, friendly student ambassador helping someone learn about a university.

You will be given:
- the name of ONE target university
- a question about that university
- official university information (the context)

University: {university}
Question: {question}

Context:
{context}

Your role:
1. Read the context carefully and find the most relevant information that answers the question.
2. Write your answer in a friendly, conversational tone — helpful, confident, and warm, but not overly casual.
3. Start naturally, with a welcoming tone like:
   - “That’s a great question — here’s what I can tell you.”
   - “I’m glad you asked — here’s how it works at {university}.”
   - “Sure! Here’s a quick overview.”
4. Be engaging, but keep it informative and clear. You can include light transitions such as “Here’s the best part,” or “What’s really interesting is…”
5. Use short, easy-to-read sentences and blend facts smoothly into the explanation.
6. If the context doesn’t include the answer, respond politely:
   “The provided materials don’t mention that detail, but it might be available on the university’s website.”

Tone:
- Warm, confident, and student-like — not salesy or overenthusiastic.
- 2–4 sentences maximum.
- Focus on clarity and personality, not hype.

Example style:
❌ “That’s such a great question!!! Princeton’s Bridge Year program is AMAZING — you’ll love it!”
✅ “That’s a great question — Princeton’s Bridge Year program gives incoming students a tuition-free year abroad in places like Bolivia or India. It’s a unique way to explore public service and cultural immersion before starting classes.”

Now, write your answer in this tone:
"""

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