from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import read_json


def _strip_english_parentheses(text: str) -> str:
    # Removes trailing "(English...)" parts commonly used in the dataset.
    # Example: "ትኩሳት (Fever)" -> "ትኩሳት"
    return re.sub(r"\s*\([^\)]*\)\s*", " ", text).strip()


def _tokenize_simple(text: str) -> list[str]:
    # MVP tokenization: whitespace + keep original tokens.
    # (Amharic tokenization can be improved later; keep explainable & deterministic now.)
    return [t for t in re.split(r"\s+", text.strip()) if t]


def _find_phrase_matches(text: str, phrase_map: dict[str, Any]) -> list[dict[str, Any]]:
    # Longest-first matching so multi-word markers win over single tokens.
    phrases = sorted(phrase_map.keys(), key=len, reverse=True)
    hits: list[dict[str, Any]] = []
    for phrase in phrases:
        if phrase and phrase in text:
            hits.append({"match": phrase, "value": phrase_map[phrase]})
    return hits


@dataclass(frozen=True)
class NormalizationResult:
    input_text: str
    tokens: list[str]
    rule_matches: list[dict[str, Any]]
    features: dict[str, Any]
    canonical_text: str


class Normalizer:
    def __init__(self, lexicon_dir: str | Path):
        lexicon_dir = Path(lexicon_dir)
        self.modifiers: dict[str, Any] = read_json(lexicon_dir / "modifiers.json")
        self.body_parts: dict[str, str] = read_json(lexicon_dir / "body_parts.json")
        self.temporal_markers: dict[str, str] = read_json(lexicon_dir / "temporal_markers.json")
        self.emergency_phrases: dict[str, Any] = read_json(lexicon_dir / "emergency_phrases.json")
        self.symptom_keywords: dict[str, Any] = read_json(lexicon_dir / "symptom_keywords.json")

    def normalize(self, text: str) -> NormalizationResult:
        tokens = _tokenize_simple(text)

        rule_matches: list[dict[str, Any]] = []

        # Emergency detection (highest priority)
        emergency_hits = _find_phrase_matches(text, self.emergency_phrases)
        for h in emergency_hits:
            rule_matches.append({"type": "emergency_phrase", **h})

        # Temporal extraction
        temporal_hits = _find_phrase_matches(text, self.temporal_markers)
        for h in temporal_hits:
            rule_matches.append({"type": "temporal_marker", **h})

        # Body-part extraction (token/substring based)
        body_part: str | None = None
        for surface, canonical in sorted(self.body_parts.items(), key=lambda kv: len(kv[0]), reverse=True):
            if surface in text:
                body_part = canonical
                rule_matches.append({"type": "body_part", "match": surface, "value": canonical})
                break

        # Modifier extraction
        modifier_hits = _find_phrase_matches(text, self.modifiers)
        for h in modifier_hits:
            rule_matches.append({"type": "modifier", **h})

        # Symptom category inference via keywords
        symptom: str | None = None
        for symptom_key, obj in self.symptom_keywords.items():
            for kw in obj.get("keywords", []):
                if kw and kw in text:
                    symptom = symptom_key
                    rule_matches.append({"type": "symptom_keyword", "match": kw, "value": symptom_key})
                    break
            if symptom:
                break

        # Build features
        intensity: str | None = None
        pain_quality: str | None = None
        quality: str | None = None
        for h in modifier_hits:
            v = h["value"]
            if isinstance(v, dict):
                intensity = intensity or v.get("intensity")
                resolved_pain_quality = v.get("pain_quality")
                by_body_part = v.get("pain_quality_by_body_part")
                if isinstance(by_body_part, dict) and body_part and body_part in by_body_part:
                    resolved_pain_quality = by_body_part[body_part]
                pain_quality = pain_quality or resolved_pain_quality
                quality = quality or v.get("quality")

        temporal: str | None = temporal_hits[0]["value"] if temporal_hits else None

        # Determine intent + severity from emergency, otherwise default.
        if emergency_hits:
            intent = "emergency"
            severity = emergency_hits[0]["value"].get("severity", "critical")
        else:
            intent = "symptom_query"
            severity = "routine" if intensity in (None, "mild") else "moderate" if intensity == "moderate" else "high" if intensity == "severe" else "routine"

        features: dict[str, Any] = {
            "intent": intent,
            "severity": severity,
            "symptom": symptom or "unknown",
            "body_part": body_part or "unknown",
            "intensity": intensity or "unknown",
            "pain_quality": pain_quality or (quality or "unknown"),
            "temporal": temporal or "unspecified",
        }

        # Canonical sentence representation (used for embeddings)
        canonical_text = self._canonicalize(text=text, inferred_symptom=symptom)

        return NormalizationResult(
            input_text=text,
            tokens=tokens,
            rule_matches=rule_matches,
            features=features,
            canonical_text=canonical_text,
        )

    def _canonicalize(self, text: str, inferred_symptom: str | None) -> str:
        # Prefer symptom-specific canonical Amharic template.
        if inferred_symptom and inferred_symptom in self.symptom_keywords:
            canonical = self.symptom_keywords[inferred_symptom].get("canonical_am")
            if isinstance(canonical, str) and canonical.strip():
                return canonical.strip()

        # Fallback: attempt to strip any English parentheses already present.
        return _strip_english_parentheses(text)
