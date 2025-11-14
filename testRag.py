import os
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_core.documents import Document 
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import pypdf

print(os.getcwd())
DATA_PATH = "./data"

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()
    return docs

def split_documents(documents):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 800,
      chunk_overlap = 80,
      length_function = len,
      is_separator_regex=False
  )
  return text_splitter.split_documents(documents)

def get_embedding_function():
  embeddings = OllamaEmbeddings(
      model = 'nomic-embed-text'
  )
  return embeddings

def add_to_chroma(chunks):
    db = Chroma(
      persist_directory="CHROMA PATH",
        embedding_function=get_embedding_function()
    )
    db.add_documents(chunks, ids =new_chunks_ids)
    db.persist()

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

    chunk.metadata["id"] = chunk_id

    existing_items = db.get(include = [])
    existing_ids = set(existing_items['ids'])
    print("Existing IDs in Chroma:", len(existing_ids))
    new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
    db.add_documents(new_chunks, ids=new_chunks_ids)
    db.persist()


def query_rag(query):
    PROMPT_TEMPLATE = """Usa i seguenti documenti per rispondere alla domanda in modo conciso e pertinente.
    Se non conosci la risposta, dì semplicemente "Non lo so"."""

    results = db.similarity_search_with_score(query, k=3)


    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)                     
    print(response_text)

if __name__ == "__main__":
    docs = load_documents()
    documents = load_documents()
    chunks = split_documents(documents)
    print(chunks[0])
    print("Documenti caricati:", len(docs))

