from __future__ import annotations

import json
from pathlib import Path
import sys

from .explain import ExplainablePipeline, PipelineConfig


def main() -> None:
    # Ensure Amharic text round-trips correctly in Windows terminals.
    # (Still recommend `chcp 65001` or PowerShell 7+ for best results.)
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    config = PipelineConfig(
        lexicon_dir=str(root / "data" / "lexicons"),
        examples_path=str(root / "data" / "examples.json"),
        top_k=3,
        threshold=0.55,
    )

    print("Explainable Amharic Symptom Chatbot (Prototype)")
    print("Disclaimer: Not medical advice. For emergencies, seek professional help.")

    pipeline = ExplainablePipeline(config=config)

    while True:
        try:
            text = input("\nAmharic input (or 'exit'): ").strip()
        except EOFError:
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit", ":q"}:
            break

        result = pipeline.run(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
