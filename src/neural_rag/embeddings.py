from functools import lru_cache

from sentence_transformers import SentenceTransformer

from neural_rag.config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()
