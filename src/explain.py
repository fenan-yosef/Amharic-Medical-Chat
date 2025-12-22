from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .classify import ClassifierConfig, ExampleIndex
from .normalize import Normalizer


@dataclass
class PipelineConfig:
    lexicon_dir: str
    examples_path: str
    top_k: int = 3
    threshold: float = 0.55


class ExplainablePipeline:
    def __init__(self, *, config: PipelineConfig):
        self.config = config
        clf_cfg = ClassifierConfig(top_k=config.top_k, threshold=config.threshold)
        self.normalizer = Normalizer(config.lexicon_dir)
        self.index = ExampleIndex(config.examples_path, config=clf_cfg)
        self.embedding_model = clf_cfg.encoder.model_name

    def run(self, text: str) -> dict[str, Any]:
        norm = self.normalizer.normalize(text)
        canonical = norm.canonical_text

        matches, best = self.index.query(canonical)

        # If normalization says emergency, keep that even if similarity is low.
        intent = norm.features.get("intent", "symptom_query")
        severity = norm.features.get("severity", "routine")

        decided_symptom = norm.features.get("symptom", "unknown")
        accepted = False

        if intent == "emergency":
            accepted = True
        elif best is not None:
            # prefer embedding decision for symptom label
            decided_symptom = best.symptom
            accepted = True

        return {
            "input": norm.input_text,
            "tokens": norm.tokens,
            "rule_matches": norm.rule_matches,
            "features": norm.features,
            "canonical": canonical,
            "embedding_model": self.embedding_model,
            "top_matches": [
                {
                    "id": m.id,
                    "text": m.text,
                    "intent": m.intent,
                    "symptom": m.symptom,
                    "score": round(m.score, 4),
                }
                for m in matches
            ],
            "decision": {
                "intent": intent,
                "severity": severity,
                "symptom": decided_symptom,
                "threshold": self.config.threshold,
                "accepted": accepted,
            },
        }
