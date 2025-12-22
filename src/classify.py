from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .encode import EncoderConfig, encode_sentence, encode_sentences
from .io_utils import read_json


@dataclass(frozen=True)
class Match:
    id: str
    text: str
    intent: str
    symptom: str
    score: float


@dataclass
class ClassifierConfig:
    encoder: EncoderConfig = EncoderConfig()
    top_k: int = 3
    threshold: float = 0.55


class ExampleIndex:
    def __init__(self, examples_path: str, *, config: ClassifierConfig):
        self.config = config
        obj: dict[str, Any] = read_json(examples_path)
        items = obj.get("items", [])
        self.items = items
        self.texts: list[str] = [it["text"] for it in items]
        self.vectors: np.ndarray = encode_sentences(self.texts, config=self.config.encoder)

    def query(self, text: str) -> tuple[list[Match], Match | None]:
        qv = encode_sentence(text, config=self.config.encoder)

        # vectors are normalized, so cosine similarity == dot product
        scores = (self.vectors @ qv).astype(float)
        order = np.argsort(-scores)
        top_k = min(self.config.top_k, len(order))

        matches: list[Match] = []
        for idx in order[:top_k]:
            it = self.items[int(idx)]
            matches.append(
                Match(
                    id=str(it.get("id", idx)),
                    text=it["text"],
                    intent=it.get("intent", "symptom_query"),
                    symptom=it.get("symptom", "unknown"),
                    score=float(scores[int(idx)]),
                )
            )

        best = matches[0] if matches else None
        if best and best.score >= self.config.threshold:
            return matches, best
        return matches, None
