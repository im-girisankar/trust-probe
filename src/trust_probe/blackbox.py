"""blackbox.py — Black-box reliability signals (pure Python, offline-capable).

Wraps the llm-reliability-kit's consistency and faithfulness modules.
All functions are CPU-only and have no heavy dependencies.

The faithfulness and consistency implementations here are SELF-CONTAINED
re-implementations that match the llm-reliability-kit interface but do not
import from it, so trust-probe has zero runtime dependency on that package.
(The reliability-kit is a sibling repo used only to understand the interface.)

Public API
----------
consistency_risk(samples) -> float
    1 - mean pairwise Jaccard similarity across samples.  Range [0, 1].

faithfulness_risk(answer, context, support_fn=None) -> float
    1 - fraction of answer sentences supported by context.  Range [0, 1].

selfcheck_consistency(samples, similarity_fn=None) -> float
    Alias for consistency_risk (SelfCheckGPT-style consistency signal).

hri(answer, context=None, samples=None) -> float
    Hallucination Risk Index: 0.5 * consistency_risk + 0.5 * faithfulness_risk.
    Matches the llm-reliability-kit locked design decisions exactly.
"""

from __future__ import annotations

from collections.abc import Callable

# ---------------------------------------------------------------------------
# Lexical helpers (no deps beyond stdlib)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Simple whitespace+punctuation tokenizer."""
    import re

    return set(re.findall(r"\b\w+\b", text.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity between two strings. Range [0, 1]."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union > 0 else 0.0


def normalized_containment(sentence: str, context: str) -> float:
    """Fraction of sentence tokens present in context tokens. Range [0, 1]."""
    ts = _tokenize(sentence)
    tc = _tokenize(context)
    if not ts:
        return 1.0
    return len(ts & tc) / len(ts)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?'."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def consistency_risk(
    samples: list[str],
    similarity_fn: Callable[[str, str], float] | None = None,
) -> float:
    """Estimate hallucination risk from self-consistency across sampled responses.

    Parameters
    ----------
    samples:
        Multiple sampled answers to the same prompt.
    similarity_fn:
        Callable f(a, b) -> float in [0, 1].  Defaults to Jaccard similarity.

    Returns
    -------
    float in [0, 1].  0 = all samples identical (low risk). 1 = no overlap (high risk).
    """
    if len(samples) <= 1:
        return 0.0  # single sample → no disagreement possible

    sim_fn = similarity_fn if similarity_fn is not None else jaccard_similarity

    total = 0.0
    count = 0
    n = len(samples)
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_fn(samples[i], samples[j])
            count += 1

    mean_sim = total / count if count > 0 else 1.0
    return 1.0 - mean_sim


def selfcheck_consistency(
    samples: list[str],
    similarity_fn: Callable[[str, str], float] | None = None,
) -> float:
    """SelfCheckGPT-style consistency signal.

    Alias for ``consistency_risk`` with identical semantics.

    Parameters
    ----------
    samples:
        Multiple sampled answers.
    similarity_fn:
        Optional callable for pairwise similarity.

    Returns
    -------
    float in [0, 1].  Higher = more inconsistent = higher hallucination risk.
    """
    return consistency_risk(samples, similarity_fn)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def faithfulness_risk(
    answer: str,
    context: str,
    support_fn: Callable[[str, str], bool] | None = None,
    threshold: float = 0.25,
) -> float:
    """Estimate hallucination risk from lack of grounding in context.

    Parameters
    ----------
    answer:
        The LLM-generated answer to evaluate.
    context:
        Retrieved / reference text the answer should be grounded in.
    support_fn:
        Optional callable f(sentence, context) -> bool.  Defaults to lexical
        containment (normalized_containment >= threshold).
    threshold:
        Minimum containment for a sentence to count as supported.  Default 0.25.
        Only used when support_fn is None.

    Returns
    -------
    float in [0, 1].  0 = fully grounded (low risk). 1 = no sentences supported (high risk).
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0  # empty answer → vacuously faithful
    if not context or not context.strip():
        return 1.0  # no context → cannot be grounded

    if support_fn is not None:

        def checker(s: str) -> bool:
            return support_fn(s, context)

    else:

        def checker(s: str) -> bool:  # type: ignore[misc]
            return normalized_containment(s, context) >= threshold

    supported = sum(1 for s in sentences if checker(s))
    faithfulness = supported / len(sentences)
    return 1.0 - faithfulness


# ---------------------------------------------------------------------------
# HRI composite scorer (matches llm-reliability-kit locked design)
# ---------------------------------------------------------------------------

_W_CONSISTENCY: float = 0.5
_W_FAITHFULNESS: float = 0.5


def hri(
    answer: str,
    context: str | None = None,
    samples: list[str] | None = None,
) -> float:
    """Hallucination Risk Index (HRI) matching llm-reliability-kit's locked design.

    HRI = 0.5 * consistency_risk + 0.5 * faithfulness_risk

    Parameters
    ----------
    answer:
        The primary LLM answer to evaluate.
    context:
        Retrieved / reference text.  If None, faithfulness component is skipped.
    samples:
        Multiple sampled answers (incl. answer or not).  If None or < 2, consistency
        component is skipped.

    Returns
    -------
    float in [0, 1].
        0.5 when neither context nor samples are provided (maximally uncertain).

    Notes
    -----
    Locked weights: consistency=0.5, faithfulness=0.5.
    Matches ``llm_reliability_kit.hallucination.hri`` semantics exactly.
    """
    has_context = context is not None and bool(context.strip())
    has_samples = samples is not None and len(samples) >= 2

    if not has_context and not has_samples:
        return 0.5  # maximally uncertain

    if has_context and has_samples:
        c_risk = consistency_risk(samples)  # type: ignore[arg-type]
        f_risk = faithfulness_risk(answer, context)  # type: ignore[arg-type]
        return _W_CONSISTENCY * c_risk + _W_FAITHFULNESS * f_risk

    if has_samples:
        return consistency_risk(samples)  # type: ignore[arg-type]

    # has_context only
    return faithfulness_risk(answer, context)  # type: ignore[arg-type]
