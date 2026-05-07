from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from neural_rag.config import LM_MODEL


@lru_cache(maxsize=1)
def get_lm() -> SentenceTransformer:
    return SentenceTransformer(LM_MODEL)


def encode_texts(texts: list[str]) -> torch.Tensor:
    """Encode texts to L2-normalized vectors (CPU tensor)."""
    model = get_lm()
    emb = model.encode(texts, normalize_embeddings=True)
    return torch.from_numpy(np.asarray(emb, dtype=np.float32))


def encode_question(question: str) -> torch.Tensor:
    return encode_texts([question])[0]
