from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"


@lru_cache(maxsize=1)
def get_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def encode_sentences(texts: list[str], *, config: EncoderConfig) -> np.ndarray:
    model = get_model(config.model_name)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors


def encode_sentence(text: str, *, config: EncoderConfig) -> np.ndarray:
    return encode_sentences([text], config=config)[0]
