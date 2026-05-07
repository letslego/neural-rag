from pathlib import Path

import chromadb
from chromadb.config import Settings

from neural_rag.config import CHROMA_PATH
from neural_rag.embeddings import embed_texts


def get_client() -> chromadb.PersistentClient:
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(name: str):
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_files(paths: list[Path], collection_name: str, chunk_size: int = 800) -> int:
    """Read files, split into chunks, embed, upsert into Chroma."""
    collection = get_collection(collection_name)
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            doc_id = f"{path.as_posix()}::{i}"
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append({"source": str(path), "chunk": i})

    if not documents:
        return 0

    embeddings = embed_texts(documents)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(documents)


def _chunk_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks
