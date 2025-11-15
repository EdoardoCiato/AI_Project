# ingest.py
import hashlib
from tqdm import tqdm
import chromadb
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma_db"
COLLECTION  = "colleges_2024_25"
BATCH_SIZE  = 200

def _make_fact_id(meta: dict) -> str:
    key = "|".join([
        str(meta.get("university","")),
        str(meta.get("sheet","")),
        str(meta.get("category","")),
        str(meta.get("variable","")),
        str(meta.get("detail","")),
        str(meta.get("value","")),
    ])
    import hashlib
    return hashlib.md5(key.encode("utf-8")).hexdigest()

def add_to_chroma(docs) -> None:
    embedding_function = get_embedding_function()

    # NEW: explicit persistent client (auto-persist; no .persist())
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    db = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embedding_function,
    )
    print(db)
    # Build texts/metas/ids
    texts, metas, ids = [], [], []
    for d in docs:
        texts.append(d.page_content or "")
        metas.append(d.metadata or {})
        ids.append(_make_fact_id(d.metadata or {}))

    # Dedup in-memory
    seen = set()
    uniq_texts, uniq_metas, uniq_ids = [], [], []
    for t, m, i in zip(texts, metas, ids):
        if i in seen: 
            continue
        seen.add(i)
        uniq_texts.append(t); uniq_metas.append(m); uniq_ids.append(i)
    print(f"🧹 Kept {len(uniq_ids)} unique docs.")

    # Skip already-in-DB
    existing = db.get(include=[])
    existing_ids = set(existing["ids"])
    keep_t, keep_m, keep_i = [], [], []
    for t, m, i in zip(uniq_texts, uniq_metas, uniq_ids):
        if i in existing_ids:
            continue
        keep_t.append(t); keep_m.append(m); keep_i.append(i)
    print(f"📦 Upserting {len(keep_i)} new docs in batches of {BATCH_SIZE}…")

    # Batch upsert (no .persist() call)
    for start in tqdm(range(0, len(keep_i), BATCH_SIZE), desc="Upserting"):
        end = min(start + BATCH_SIZE, len(keep_i))

        print("\n--- Example metadata before insert ---")
        for i in range(3):
            print(keep_m[i])

        db.add_texts(
            texts=keep_t[start:end],
            metadatas=keep_m[start:end],
            ids=keep_i[start:end],
        )

    print("✅ Done (auto-persisted via PersistentClient).")
