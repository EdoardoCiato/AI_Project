# populate_database.py

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Upload all PDFs from a directory
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Split documents into chunks
from langchain_core.documents import Document  # Represent documents with metadata
from langchain_community.vectorstores import Chroma

from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"

def infer_university(metadata: dict) -> str:
    title = (metadata.get("title") or "").lower()
    source = (metadata.get("source") or "").lower()

    # Example titles: "Brochure - Harvard University"
    if "brochure " in title and "-" in title:
        # Take everything after the dash
        after_dash = title.split("-", 1)[1].strip()
        return after_dash.title()

    # Fallback: use the filename (without extension)
    if source:
        filename = os.path.basename(source)
        name_no_ext, _ = os.path.splitext(filename)
        # Example: "Brochure - Brown" -> take text after the dash
        if "-" in name_no_ext:
            after_dash = name_no_ext.split("-", 1)[1].strip()
            return after_dash.title()
        return name_no_ext.replace("_", " ").title()

    return "Unknown"


def load_documents():
    """Load all PDFs from the data directory and assign university metadata."""
    loader = PyPDFDirectoryLoader(DATA_PATH)
    docs = loader.load()

    for doc in docs:
        uni = infer_university(doc.metadata)
        doc.metadata["university"] = uni
        # Optional: print for debugging
        # print(doc.metadata["source"], "->", uni)

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,  # Character-based length measurement
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f" Chunks created: {len(chunks)}")
    return chunks


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    """
    Create a unique ID for each chunk, like:
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
    """Add only new chunks to the persistent Chroma database."""
    embedding_function = get_embedding_function()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    # Add IDs to chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Retrieve IDs already present in the DB
    existing_items = db.get(include=[])  # IDs are always included
    existing_ids = set(existing_items["ids"])
    print(f" Documents already in the database: {len(existing_ids)}")

    # Keep only new chunks
    new_chunks = [ch for ch in chunks_with_ids if ch.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [ch.metadata["id"] for ch in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        print(" Chroma database updated and saved.")
    else:
        print(" No new documents to add.")


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


if __name__ == "__main__":
    main()
