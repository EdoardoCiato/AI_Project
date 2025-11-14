from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
import os
from langchain_docling import DoclingLoader

from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from docling.document_converter import DocumentConverter


DIRECTORY_PATH = "./data"
CHROMA_PATH = "./chroma_db"

def load_documents():
    source = "./US News: Profile.pdf"  # file path or URL
    converter = DocumentConverter()
    doc = converter.convert(source).document
    tables = doc.tables
    for tab in tables:
        print(tab.export_to_markdown())

# populate_database.py

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader #to upload all the PDFs from a directory
from langchain_text_splitters import RecursiveCharacterTextSplitter #to split the documents into chunks
from langchain_core.documents import Document #to represent documents with metadata
from langchain_community.vectorstores import Chroma

from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"

def infer_university(metadata: dict) -> str:
    title = (metadata.get("title") or "").lower()
    source = (metadata.get("source") or "").lower()

    # Esempi di title tipo: "College Navigator - Harvard University"
    if "college navigator" in title and "-" in title:
        # prendi tutto dopo il trattino
        after_dash = title.split("-", 1)[1].strip()
        return after_dash.title()

    # fallback: usa il nome del file (senza estensione)
    if source:
        filename = os.path.basename(source)
        name_no_ext, _ = os.path.splitext(filename)
        # "College Navigator - Stanford" -> prendo dopo il trattino
        if "-" in name_no_ext:
            after_dash = name_no_ext.split("-", 1)[1].strip()
            return after_dash.title()
        return name_no_ext.replace("_", " ").title()

    return "Unknown"

def split_documents(documents: list[Document]) -> list[Document]:
    """Splitta i documenti in chunk sovrapposti."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len, #measurement done by counting the characters
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,)
    chunks = text_splitter.split_documents(documents)
    print(f"🔹 Chunk creati: {len(chunks)}")
    return chunks


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


def add_to_chroma(chunks: list[Document]) -> None:
    """Aggiunge solo i nuovi chunk al database Chroma persistente."""
    chunks = filter_complex_metadata(chunks)
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    # aggiunge gli ID ai chunk
    chunks_with_ids = calculate_chunk_ids(chunks)

    # recupera gli ID già presenti nel DB
    existing_items = db.get(include=[])  # gli ids sono sempre inclusi
    existing_ids = set(existing_items["ids"])
    print(f"📚 Documenti già presenti nel DB: {len(existing_ids)}")

    # filtra solo i nuovi chunk
    new_chunks = [ch for ch in chunks_with_ids if ch.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Aggiungo nuovi documenti: {len(new_chunks)}")
        new_chunk_ids = [ch.metadata["id"] for ch in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        print("✅ DB Chroma aggiornato e salvato.")
    else:
        print("✅ Nessun nuovo documento da aggiungere.")


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


if __name__ == "__main__":
    main()

