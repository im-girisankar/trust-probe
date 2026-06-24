"""baselines.py — Simple baseline detectors for comparison in the leaderboard.

Each baseline is a function that takes a list of records and an injected LM
callable, and returns a numpy array of risk scores in [0, 1].

The LM callable interface
--------------------------
    lm(prompt: str) -> str
        Generate a response string for the given prompt.

    lm(prompt: str, n: int) -> list[str]
        Generate n sampled responses (for consistency-based baselines).

Mock LM (for offline testing)
-------------------------------
``MockLM`` is provided for tests — it returns deterministic responses without
any model loading.

Baselines
---------
logprob_baseline(records, lm) -> np.ndarray
    Proxy for log-probability: uses normalised unigram faithfulness as a
    stand-in (since true log-probs require model access).

ptrue_baseline(records, lm) -> np.ndarray
    Asks the LM "Is the following response faithful? Answer yes/no." and
    maps yes/no to 0.0/1.0 risk.

selfcheckgpt_baseline(records, lm, n_samples=3) -> np.ndarray
    SelfCheckGPT-style: samples n responses per prompt and computes
    consistency_risk across samples.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from trust_probe.blackbox import consistency_risk, faithfulness_risk

# Type alias for the LM callable
LMCallable = Callable[..., Any]


# ---------------------------------------------------------------------------
# Mock LM for offline testing
# ---------------------------------------------------------------------------


class MockLM:
    """Deterministic mock language model for offline tests.

    Returns pre-set responses based on a cycling pattern.
    Supports both single-response and multi-sample calls.

    Parameters
    ----------
    responses : list[str] | None
        Cycle through these responses.  Defaults to a small fixed set.
    seed : int
        Seed for any randomness (currently unused — responses are deterministic).
    """

    _DEFAULT_RESPONSES = [
        "The answer is correct and well-grounded.",
        "Yes, this is faithful.",
        "No, this does not match the context.",
        "The response appears to be accurate.",
        "This seems inconsistent with the provided information.",
    ]

    def __init__(
        self,
        responses: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self._responses = responses or list(self._DEFAULT_RESPONSES)
        self._seed = seed
        self._call_count = 0

    def __call__(self, prompt: str, n: int = 1) -> str | list[str]:
        """Return one response (n=1) or a list of n responses."""
        result = []
        for _ in range(n):
            idx = self._call_count % len(self._responses)
            result.append(self._responses[idx])
            self._call_count += 1
        return result[0] if n == 1 else result


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def logprob_baseline(
    records: list[dict[str, Any]],
    lm: LMCallable | None = None,  # noqa: ARG001 — interface for real LM, unused in proxy
) -> np.ndarray:
    """Proxy log-probability baseline: faithfulness_risk of response vs. context.

    In a real deployment this would use the LM's token log-probabilities.
    Here we use lexical faithfulness as a proxy (no model access needed).

    Parameters
    ----------
    records:
        List of dicts with 'response' and 'context' keys.
    lm:
        Language model callable (interface kept for consistency; not used here).

    Returns
    -------
    np.ndarray of shape (N,) — risk scores in [0, 1].
    """
    scores = []
    for rec in records:
        risk = faithfulness_risk(
            answer=rec.get("response", ""),
            context=rec.get("context", ""),
        )
        scores.append(risk)
    return np.array(scores, dtype=float)


def ptrue_baseline(
    records: list[dict[str, Any]],
    lm: LMCallable | None = None,
) -> np.ndarray:
    """P(True) baseline: ask LM if response is faithful; map to risk.

    Sends a prompt asking the LM to say 'yes' (faithful) or 'no' (not faithful).
    Maps 'yes' -> 0.0 risk, 'no' -> 1.0 risk, unclear -> 0.5.

    Parameters
    ----------
    records:
        List of dicts with 'prompt', 'response', and 'context' keys.
    lm:
        LM callable.  If None, falls back to lexical faithfulness proxy.

    Returns
    -------
    np.ndarray of shape (N,) — risk scores in [0, 1].
    """
    scores = []
    for rec in records:
        if lm is None:
            # Fallback: use lexical faithfulness
            risk = faithfulness_risk(rec.get("response", ""), rec.get("context", ""))
        else:
            prompt_text = (
                f"Context: {rec.get('context', '')}\n"
                f"Question: {rec.get('prompt', '')}\n"
                f"Response: {rec.get('response', '')}\n"
                "Is this response faithful to the context? Answer only 'yes' or 'no'."
            )
            answer = str(lm(prompt_text)).strip().lower()
            if answer.startswith("yes"):
                risk = 0.0
            elif answer.startswith("no"):
                risk = 1.0
            else:
                risk = 0.5
        scores.append(risk)
    return np.array(scores, dtype=float)


def selfcheckgpt_baseline(
    records: list[dict[str, Any]],
    lm: LMCallable | None = None,
    n_samples: int = 3,
) -> np.ndarray:
    """SelfCheckGPT-style baseline: consistency risk across sampled responses.

    Samples ``n_samples`` responses per prompt from the LM, then computes
    consistency_risk across the samples.

    Parameters
    ----------
    records:
        List of dicts with 'prompt' and 'response' keys.
    lm:
        LM callable supporting ``lm(prompt, n=k) -> list[str]``.
        If None, returns faithfulness-based proxy scores.
    n_samples:
        Number of samples to draw per prompt.

    Returns
    -------
    np.ndarray of shape (N,) — risk scores in [0, 1].
    """
    scores = []
    for rec in records:
        if lm is None:
            # Fallback to faithfulness proxy
            risk = faithfulness_risk(rec.get("response", ""), rec.get("context", ""))
        else:
            prompt_text = rec.get("prompt", "")
            sampled = lm(prompt_text, n=n_samples)
            if isinstance(sampled, str):
                sampled = [sampled] * n_samples
            risk = consistency_risk(sampled)
        scores.append(risk)
    return np.array(scores, dtype=float)
