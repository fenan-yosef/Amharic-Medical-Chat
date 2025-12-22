# ADAMA SCIENCE AND TECHNOLOGY UNIVERSITY

## SCHOOL OF ELECTRICAL ENGINEERING AND COMPUTING

### DEPARTMENT OF SOFTWARE ENGINEERING

---

# An Explainable Amharic NLP Chatbot for Symptom Understanding

**Course:** Natural Language Processing
**Project Type:** NLP System Design & Implementation (Explainable Prototype)

**Team Members:**

* Bereket Melese
* Fenan Yosef

---

## 1. Introduction

Natural Language Processing systems are often evaluated not only by their performance, but also by how well their internal decisions can be explained. This is especially important in low‑resource languages such as Amharic, where large annotated datasets and pretrained domain‑specific models are limited.

This project presents an **explainable hybrid NLP chatbot prototype** that understands Amharic health‑related queries at a fundamental linguistic level. The goal is not to build a full medical diagnosis system, but to demonstrate **how raw Amharic text is transformed step‑by‑step into structured meaning** using well‑established NLP principles.

The system is intentionally designed to expose all intermediate representations — tokens, features, vectors, similarity scores, and final decisions — allowing every design choice to be justified and defended academically.

---

## 2. Problem Definition

Amharic speakers frequently use **colloquial and metaphorical expressions** to describe physical discomfort. For example:

> "ራሴን ሰንጥቆ ያመኛል"

While humans easily interpret this as a *severe headache*, most NLP systems fail because:

* The language is metaphorical rather than literal
* Amharic is a low‑resource language
* Direct translation loses clinical nuance

The core NLP challenge addressed in this project is:

> **How can we transform colloquial Amharic sentences into structured, machine‑understandable representations while preserving semantic and intensity information?**

---

## 3. Project Objectives

### Primary NLP Objectives

* Demonstrate text normalization, feature extraction, encoding, and similarity‑based decision making
* Preserve colloquial meaning such as intensity and quality of symptoms
* Build an interpretable NLP pipeline suitable for low‑resource languages

### Educational Objective

* Ensure that every component can be explained using fundamental NLP concepts such as:

  * Tokenization
  * Feature extraction
  * Vector representations
  * Similarity metrics
  * Decision thresholds

---

## 4. System Overview

The chatbot follows a **hybrid NLP architecture** that combines symbolic (rule‑based) processing with neural sentence embeddings.

```
User Input (Amharic)
        ↓
Colloquial Normalization & Feature Extraction (Rule‑Based)
        ↓
Canonical Sentence Representation
        ↓
Sentence Encoding (SBERT / mBERT)
        ↓
Vector Similarity & Classification
        ↓
Intent + Symptom Output
```

This separation ensures that each layer performs a clearly defined task and can be independently evaluated.

---

## 5. Colloquial Normalization and Feature Extraction

### Purpose of This Layer

This layer handles **linguistic creativity**, not statistical learning. It rewrites and decomposes colloquial expressions into structured semantic components.

### Example

**Input:**

```
"ራሴን ሰንጥቆ ያመኛል"
```

**Extracted Representation:**

```json
{
  "base_symptom": "headache",
  "body_part": "head",
  "pain_quality": "splitting",
  "intensity": "severe"
}
```

### Why Rule‑Based?

* Colloquial expressions are low‑frequency and high‑impact
* Dataset size is insufficient for reliable supervised learning
* Rules provide full interpretability and deterministic behavior

This approach follows classical NLP techniques such as **finite‑state processing** and **lexicon‑driven analysis**.

---

## 6. Linguistic Resources

### Modifier Lexicon (Excerpt)

```json
{
  "ሰንጥቆ": {"quality": "splitting", "intensity": "severe"},
  "በጣም": {"intensity": "severe"},
  "ትንሽ": {"intensity": "mild"}
}
```

### Symptom Lexicon (Excerpt)

```json
{
  "ራሴን": "head",
  "ሆዴን": "abdomen"
}
```

Rules are treated as **data**, not hard‑coded logic, enabling easy extension and inspection.

---

## 7. Sentence Encoding and Vector Representation

### Motivation

Once colloquial modifiers are extracted, the remaining canonical sentence expresses the **core semantic intent**.

Example canonical sentence:

```
"ራሴን አሞኛል"
```

This sentence is encoded using a **pretrained multilingual sentence embedding model (SBERT or mBERT with pooling)**.

### Encoding Principle

* Each sentence is mapped to a dense vector in (R^d)
* Similar meanings result in vectors that are close in vector space

This converts language into a numerical form suitable for mathematical comparison.

---

## 8. Similarity‑Based Classification

Rather than training a neural classifier, the system uses **nearest‑neighbor similarity**.

### Method

* Store vectors of labeled example sentences
* Compute cosine similarity between input vector and stored vectors
* Assign intent and symptom based on highest similarity

### Why This Method?

* No large labeled dataset required
* Fully explainable decisions
* Direct use of matrix operations and similarity measures

This aligns with fundamental NLP and information retrieval concepts.

---

## 9. Intent and Symptom Labels

Each example sentence is annotated with:

* Intent (e.g., symptom_query, informational_query)
* Symptom category (e.g., headache, stomach_pain)

Example:

```json
{
  "text": "ራሴን አሞኛል",
  "intent": "symptom_query",
  "symptom": "headache"
}
```

---

## 10. Explainability and Decision Transparency

At runtime, the system exposes:

* Tokenized input
* Extracted modifiers and features
* Sentence vector similarity scores
* Final classification decision

This ensures that every output can be traced back to explicit linguistic or mathematical reasoning.

---

## 11. Implementation Scope

To ensure feasibility, the prototype is limited to:

* 20–30 colloquial modifier rules
* 5 common symptom categories
* Text‑only interaction
* Pretrained models only (no fine‑tuning)

This scope is sufficient to demonstrate all core NLP principles.

---

## 12. Limitations

* Limited coverage of rare expressions
* Manual rule creation
* No speech input
* No clinical diagnosis or treatment

These limitations are acknowledged as structural constraints rather than design flaws.

---

## 13. Conclusion

This project demonstrates that robust and explainable NLP systems can be built for low‑resource languages using hybrid architectures. By combining symbolic linguistic knowledge with modern embedding techniques, the system achieves meaningful language understanding while remaining transparent and defensible.

The prototype serves both as a practical chatbot demonstration and as an educational artifact illustrating foundational NLP concepts in a real‑world Amharic context.

---

**End of Document**

---

## Prototype (Code) — How to Run

This repository includes a working CLI prototype that follows the design in this document.

### 1) Install dependencies

From the project folder:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

### 2) (Windows) Make the terminal show Amharic correctly

If Amharic characters display as `???`, use one of these:

- PowerShell (recommended):

```powershell
chcp 65001
$env:PYTHONUTF8 = "1"
```

- Or run Python with UTF‑8 mode:

```powershell
python -X utf8 -m src.app
```

### 3) Run the chatbot

```powershell
python -m src.app
```

Type an Amharic symptom sentence, e.g.:

```text
ራሴን ሰንጥቆ ያመኛል
```

The output is a single explainability JSON object containing:
- tokens
- rule matches + extracted features
- canonical sentence
- top-k similarity matches with scores
- final decision

### 4) Run tests

```powershell
python -m pytest -q
```

### Disclaimer

This prototype is **not** medical advice, diagnosis, or treatment.
