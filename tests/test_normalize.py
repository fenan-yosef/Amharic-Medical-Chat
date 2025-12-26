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


def test_normalize_lower_back_pain_cutting() -> None:
    root = Path(__file__).resolve().parents[1]
    normalizer = Normalizer(root / "data" / "lexicons")

    res = normalizer.normalize("ወገቤ ተቆረጠ")

    assert res.features["body_part"] == "back"
    assert res.features["pain_quality"] in {"cramping", "unknown"}
    assert res.features["symptom"] == "musculoskeletal"
    assert "ወገብ" in res.canonical_text


def test_modifier_teqorete_is_cramping_for_abdomen() -> None:
    root = Path(__file__).resolve().parents[1]
    normalizer = Normalizer(root / "data" / "lexicons")

    res = normalizer.normalize("ሆድ ተቆረጠ")

    assert res.features["body_part"] == "abdomen"
    assert res.features["pain_quality"] == "cramping"


def test_modifier_teqorete_defaults_to_cutting_outside_abdomen_back() -> None:
    root = Path(__file__).resolve().parents[1]
    normalizer = Normalizer(root / "data" / "lexicons")

    res = normalizer.normalize("ደረት ተቆረጠ")

    assert res.features["body_part"] == "chest"
    assert res.features["pain_quality"] == "cutting"
