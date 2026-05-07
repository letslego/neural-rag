from openai import OpenAI

from neural_rag.config import (
    DEFAULT_TOP_K,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from neural_rag.embeddings import embed_texts
from neural_rag.store import get_collection


def retrieve(query: str, collection_name: str, top_k: int = DEFAULT_TOP_K) -> list[str]:
    collection = get_collection(collection_name)
    q_emb = embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents"],
    )
    docs = result.get("documents") or [[]]
    return docs[0] if docs else []


def answer(
    query: str,
    collection_name: str,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    contexts = retrieve(query, collection_name, top_k=top_k)
    context_block = "\n\n---\n\n".join(contexts) if contexts else "(no retrieved documents)"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    system = (
        "You are a helpful assistant. Answer using only the context provided below. "
        "If the context is insufficient, say so briefly."
    )
    user = f"Context:\n{context_block}\n\nQuestion:\n{query}"

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    choice = response.choices[0].message.content
    return choice or ""
