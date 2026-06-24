"""Tests for trust_probe.blackbox — pure-Python black-box risk signals.

No torch, no transformers, no GPU required.
"""

from __future__ import annotations

import pytest

from trust_probe.blackbox import (
    consistency_risk,
    faithfulness_risk,
    hri,
    jaccard_similarity,
    normalized_containment,
    selfcheck_consistency,
)

# ---------------------------------------------------------------------------
# Lexical helpers
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_strings(self):
        """Identical strings: Jaccard = 1.0."""
        assert jaccard_similarity("the cat sat", "the cat sat") == pytest.approx(1.0)

    def test_disjoint_strings(self):
        """No shared tokens: Jaccard = 0.0."""
        assert jaccard_similarity("apple orange", "banana mango") == pytest.approx(0.0)

    def test_partial_overlap(self):
        """'a b c' vs 'a b d' -> intersection={a,b}, union={a,b,c,d} -> 0.5."""
        assert jaccard_similarity("a b c", "a b d") == pytest.approx(0.5)

    def test_both_empty(self):
        """Both empty: Jaccard = 1.0 (vacuous identity)."""
        assert jaccard_similarity("", "") == pytest.approx(1.0)

    def test_case_insensitive(self):
        """Tokenizer should be case-insensitive."""
        assert jaccard_similarity("The Cat", "the cat") == pytest.approx(1.0)


class TestNormalizedContainment:
    def test_fully_contained(self):
        """All tokens of sentence appear in context."""
        assert normalized_containment("Paris is the capital", "Paris is the capital of France") == pytest.approx(1.0)

    def test_not_contained(self):
        """No tokens of sentence appear in context."""
        # 'xyz' not in 'abc def'
        result = normalized_containment("xyz uvw", "abc def")
        assert result == pytest.approx(0.0)

    def test_empty_sentence(self):
        """Empty sentence: containment = 1.0 (vacuously contained)."""
        assert normalized_containment("", "some context") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Consistency risk
# ---------------------------------------------------------------------------


class TestConsistencyRisk:
    def test_identical_samples_zero_risk(self):
        """All identical samples → consistency = 1.0 → risk = 0.0."""
        samples = ["Paris is the capital.", "Paris is the capital.", "Paris is the capital."]
        assert consistency_risk(samples) == pytest.approx(0.0)

    def test_single_sample_zero_risk(self):
        """Single sample → no pairs → risk = 0.0."""
        assert consistency_risk(["only one response"]) == pytest.approx(0.0)

    def test_empty_list_zero_risk(self):
        """Empty list → risk = 0.0."""
        assert consistency_risk([]) == pytest.approx(0.0)

    def test_disjoint_samples_high_risk(self):
        """Disjoint samples → Jaccard = 0.0 → risk = 1.0."""
        samples = ["apple orange banana", "gamma delta epsilon"]
        assert consistency_risk(samples) == pytest.approx(1.0)

    def test_partial_overlap_intermediate(self):
        """Samples with partial overlap → risk between 0 and 1."""
        samples = ["a b c d", "a b e f"]
        risk = consistency_risk(samples)
        assert 0.0 < risk < 1.0

    def test_custom_similarity_fn(self):
        """Custom similarity function is used."""
        def always_one(a, b):
            return 1.0

        samples = ["hello world", "foo bar"]
        # With similarity always 1.0, consistency_risk = 0.0
        assert consistency_risk(samples, similarity_fn=always_one) == pytest.approx(0.0)


class TestSelfcheckConsistency:
    def test_alias_matches_consistency_risk(self):
        """selfcheck_consistency should match consistency_risk exactly."""
        samples = ["the sky is blue", "the sky is green", "the sky is red"]
        assert selfcheck_consistency(samples) == pytest.approx(consistency_risk(samples))


# ---------------------------------------------------------------------------
# Faithfulness risk
# ---------------------------------------------------------------------------


class TestFaithfulnessRisk:
    def test_fully_faithful_answer(self):
        """Answer tokens fully covered by context → risk = 0.0."""
        context = "Paris is the capital of France. France is in Europe."
        answer = "Paris is the capital."
        result = faithfulness_risk(answer, context)
        assert result == pytest.approx(0.0)

    def test_unfaithful_answer(self):
        """Answer with no token overlap with context → risk = 1.0."""
        context = "France is a country in Western Europe."
        answer = "Quantum entanglement is bizarre phenomenon involving particles."
        result = faithfulness_risk(answer, context)
        assert result == pytest.approx(1.0)

    def test_empty_answer(self):
        """Empty answer → 0.0 risk (vacuously faithful)."""
        assert faithfulness_risk("", "some context") == pytest.approx(0.0)

    def test_empty_context(self):
        """Empty context → 1.0 risk (cannot be grounded)."""
        assert faithfulness_risk("Paris is the capital.", "") == pytest.approx(1.0)

    def test_custom_support_fn(self):
        """Custom support function overrides lexical check."""
        def always_supported(sentence, context):
            return True

        result = faithfulness_risk(
            "some answer",
            "some context",
            support_fn=always_supported,
        )
        assert result == pytest.approx(0.0)  # all sentences supported


# ---------------------------------------------------------------------------
# HRI composite
# ---------------------------------------------------------------------------


class TestHRI:
    def test_no_context_no_samples_returns_half(self):
        """HRI with no context and no samples returns 0.5 (maximally uncertain).
        Matches llm-reliability-kit locked design."""
        result = hri("any answer")
        assert result == pytest.approx(0.5)

    def test_context_only_faithful(self):
        """With only context: HRI = faithfulness_risk.
        Fully faithful answer → HRI = 0.0."""
        context = "Water boils at 100 degrees Celsius."
        answer = "Water boils at 100 degrees Celsius."
        result = hri(answer, context=context)
        assert result == pytest.approx(0.0)

    def test_context_only_unfaithful(self):
        """Unfaithful answer with no sample → HRI = 1.0."""
        context = "France is in Western Europe."
        answer = "Photosynthesis uses nitrogen."
        result = hri(answer, context=context)
        assert result == pytest.approx(1.0)

    def test_samples_only_consistent(self):
        """With only identical samples: HRI = consistency_risk = 0.0."""
        samples = ["same answer"] * 3
        result = hri("same answer", samples=samples)
        assert result == pytest.approx(0.0)

    def test_both_context_and_samples(self):
        """HRI = 0.5*consistency_risk + 0.5*faithfulness_risk."""
        samples = ["a b c d", "a b e f"]  # partial overlap
        c_risk = consistency_risk(samples)
        answer = "Paris is the capital of France."
        context = "France is a country. Paris is its capital."
        f_risk = faithfulness_risk(answer, context)
        expected = 0.5 * c_risk + 0.5 * f_risk
        result = hri(answer, context=context, samples=samples)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_hri_range(self):
        """HRI should always be in [0, 1]."""
        result = hri("test", context="test context", samples=["test", "other"])
        assert 0.0 <= result <= 1.0

    def test_single_sample_uses_context_only(self):
        """With < 2 samples, falls back to context-only mode."""
        context = "Paris is the capital of France."
        answer = "Paris is the capital of France."
        result_with_one_sample = hri(answer, context=context, samples=["only one"])
        result_context_only = hri(answer, context=context)
        assert result_with_one_sample == pytest.approx(result_context_only)
