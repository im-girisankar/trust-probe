"""datasets.py — Dataset loaders for trust-probe evaluation.

Three loaders are provided:

1. ``load_ragtruth(path)`` — parses the RAGTruth benchmark JSON.
   Expected on-disk schema (each line or list element):
     {
       "prompt":    str,           # the user question
       "response":  str,           # model answer to evaluate
       "context":   str,           # retrieved passages
       "label":     int,           # 0 = faithful, 1 = hallucinated
       "per_token": [str, ...]     # optional; token-level text
     }

2. ``load_halueval(path)`` — parses the HaluEval JSON.
   Expected on-disk schema (each line or list element):
     {
       "question":          str,
       "right_answer":      str,
       "hallucinated_answer": str,
       "knowledge":         str    # supporting context
     }
   Both answers are included: right_answer → label=0, hallucinated_answer → label=1.

3. ``synthetic_dataset(n, seed)`` — fabricates deterministic records plus matching
   fake activation tensors (numpy arrays of shape (1, 16, 64, 256)) stored under
   record["activations"].  Useful for fully offline testing.

All three return ``list[dict]`` where every record has at minimum:
    prompt, response, context, label (int 0/1)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

Record = dict[str, Any]


def load_ragtruth(path: str | os.PathLike) -> list[Record]:
    """Load a RAGTruth-format file (JSON array or JSONL).

    Each record is expected to have keys: prompt, response, context, label.
    'per_token' is optional and passed through if present.

    Parameters
    ----------
    path:
        Path to a ``.json`` (array) or ``.jsonl`` (newline-delimited) file.

    Returns
    -------
    list[dict] with keys: prompt, response, context, label, per_token (optional).
    """
    path = str(path)
    records: list[Record] = []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    # Try as JSON array first, fall back to JSONL
    try:
        raw = json.loads(content)
        if isinstance(raw, list):
            items = raw
        else:
            items = [raw]
    except json.JSONDecodeError:
        items = [json.loads(line) for line in content.splitlines() if line.strip()]

    for item in items:
        rec: Record = {
            "prompt": str(item.get("prompt", "")),
            "response": str(item.get("response", "")),
            "context": str(item.get("context", "")),
            "label": int(item.get("label", 0)),
        }
        if "per_token" in item:
            rec["per_token"] = item["per_token"]
        records.append(rec)

    return records


def load_halueval(path: str | os.PathLike) -> list[Record]:
    """Load a HaluEval-format file (JSON array or JSONL).

    Each raw item is expanded to two records:
      - right_answer    → label=0 (faithful)
      - hallucinated_answer → label=1 (hallucinated)

    Parameters
    ----------
    path:
        Path to a ``.json`` or ``.jsonl`` file.

    Returns
    -------
    list[dict] with keys: prompt, response, context, label.
    """
    path = str(path)
    records: list[Record] = []
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()

    try:
        raw = json.loads(content)
        items = raw if isinstance(raw, list) else [raw]
    except json.JSONDecodeError:
        items = [json.loads(line) for line in content.splitlines() if line.strip()]

    for item in items:
        question = str(item.get("question", ""))
        knowledge = str(item.get("knowledge", ""))
        right = str(item.get("right_answer", ""))
        halluc = str(item.get("hallucinated_answer", ""))

        records.append({"prompt": question, "response": right, "context": knowledge, "label": 0})
        records.append(
            {"prompt": question, "response": halluc, "context": knowledge, "label": 1}
        )

    return records


def synthetic_dataset(n: int = 100, seed: int = 42) -> list[Record]:
    """Fabricate ``n`` deterministic records with matching fake activation tensors.

    The synthetic data is designed so that:
    - Positive (hallucinated) records have higher activation norms in later layers.
    - This makes the white-box signal non-trivially informative.
    - Black-box context faithfulness is also noisily correlated with the label.

    Each record contains:
        prompt, response, context, label (0/1), activations (np.ndarray shape (1,16,64,256))

    Parameters
    ----------
    n:
        Number of records (half positive, half negative, ±1 for odd n).
    seed:
        Random seed for full reproducibility.
    """
    rng = np.random.default_rng(seed)

    prompts = [
        "What is the capital of France?",
        "Explain quantum entanglement.",
        "Who wrote Hamlet?",
        "What is the boiling point of water?",
        "Describe the process of photosynthesis.",
    ]
    faithful_responses = [
        "The capital of France is Paris.",
        "Quantum entanglement is a physical phenomenon where two particles share quantum states.",
        "Hamlet was written by William Shakespeare.",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen.",
    ]
    hallucinated_responses = [
        "The capital of France is Lyon.",
        "Quantum entanglement means particles can communicate faster than light.",
        "Hamlet was written by Christopher Marlowe.",
        "Water boils at 50 degrees Celsius.",
        "Photosynthesis converts moonlight into starch using nitrogen gas.",
    ]
    contexts = [
        "France is a country in Western Europe. Its capital and largest city is Paris.",
        "Quantum entanglement is a quantum mechanical phenomenon where pairs of particles interact.",
        "Hamlet is a tragedy written by William Shakespeare, believed to have been written around 1600.",
        "The boiling point of water is 100 C (212 F) at 1 atm of pressure.",
        "Photosynthesis is a process used by plants using sunlight, water, and CO2 to produce oxygen.",
    ]

    records: list[Record] = []
    for i in range(n):
        label = i % 2  # alternating 0, 1, 0, 1 ...
        idx = i % len(prompts)

        # Build activation tensor: shape (1, L=16, T=64, D=256)
        # For label=1 (hallucinated), later layers have higher norms (discriminative signal)
        base = rng.standard_normal((1, 16, 64, 256)).astype(np.float32)
        if label == 1:
            # Amplify later layers (indices 8-15) for hallucinated samples
            base[0, 8:, :, :] += 0.8
        activations = base

        rec: Record = {
            "prompt": prompts[idx],
            "response": hallucinated_responses[idx] if label == 1 else faithful_responses[idx],
            "context": contexts[idx],
            "label": label,
            "activations": activations,
        }
        records.append(rec)

    return records
