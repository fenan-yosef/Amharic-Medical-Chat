from __future__ import annotations

from pathlib import Path

from src.normalize import Normalizer


def test_normalize_headache_splitting() -> None:
    root = Path(__file__).resolve().parents[1]
    normalizer = Normalizer(root / "data" / "lexicons")

    res = normalizer.normalize("ራሴን ሰንጥቆ ያመኛል")

    assert res.features["body_part"] == "head"
    assert res.features["intensity"] in {"severe", "unknown"}
    assert res.features["symptom"] == "headache"
    assert "ራስ" in res.canonical_text
