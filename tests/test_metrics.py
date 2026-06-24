"""Tests for trust_probe.metrics — all hand-computed expected values.

No torch, no transformers, no GPU required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trust_probe.metrics import auprc, auroc, best_f1_threshold, bootstrap_ci, ece

# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


class TestAuroc:
    def test_perfect_classifier(self):
        """Perfect separation: all positives scored higher than all negatives.
        Expected AUROC = 1.0."""
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert auroc(y, scores) == pytest.approx(1.0)

    def test_chance_classifier(self):
        """A genuinely uninformative ranking gives AUROC 0.5.

        y=[0,1,1,0], scores=[0.1,0.2,0.3,0.4]: positives are 0.2 and 0.3,
        negatives 0.1 and 0.4. Of the four pos>neg pairs exactly two hold → 0.5.
        """
        y = np.array([0, 1, 1, 0])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        result = auroc(y, scores)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_inverted_perfect(self):
        """Inverted perfect separation gives AUROC = 0.0."""
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert auroc(y, scores) == pytest.approx(0.0)

    def test_single_class_returns_nan(self):
        """Single class label returns NaN (undefined AUROC)."""
        y = np.array([0, 0, 0])
        scores = np.array([0.1, 0.5, 0.9])
        result = auroc(y, scores)
        assert math.isnan(result)

    def test_larger_dataset(self):
        """AUROC of a clearly separable dataset should be >= 0.9."""
        rng = np.random.default_rng(0)
        y = np.array([0] * 50 + [1] * 50)
        scores = np.concatenate([rng.uniform(0, 0.4, 50), rng.uniform(0.6, 1.0, 50)])
        assert auroc(y, scores) > 0.9


# ---------------------------------------------------------------------------
# AUPRC
# ---------------------------------------------------------------------------


class TestAuprc:
    def test_perfect_classifier(self):
        """Perfect separation: AUPRC = 1.0."""
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert auprc(y, scores) == pytest.approx(1.0)

    def test_below_random_baseline(self):
        """Inverted scores: AUPRC should be less than class prevalence."""
        y = np.array([0, 0, 0, 1])
        scores = np.array([0.9, 0.7, 0.5, 0.1])
        result = auprc(y, scores)
        # Prevalence = 0.25; inverted should be well below 1.0
        assert result < 0.6

    def test_single_class_returns_nan(self):
        y = np.array([1, 1, 1])
        scores = np.array([0.1, 0.5, 0.9])
        result = auprc(y, scores)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


class TestEce:
    def test_perfectly_calibrated(self):
        """When predicted probability equals actual frequency in each bin, ECE = 0.
        Hand-constructed: 10 samples at score=0.4, all positive → ECE=0.6
        Wait: if label=1 and score=0.4, that's calibration error 0.6 for that bin.
        Let's use score=0.5 and label=0.5 (50% positive in bin) → ECE=0."""
        # 10 samples: 5 positive, 5 negative, all scored 0.5
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.full(10, 0.5)
        result = ece(y, scores, n_bins=1)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_high_ece_miscalibrated(self):
        """All positive labels but all scores near 0 → high ECE."""
        y = np.ones(10, dtype=int)
        scores = np.full(10, 0.1)
        # Mean pred = 0.1, mean true = 1.0 → ECE = |0.1 - 1.0| = 0.9
        result = ece(y, scores, n_bins=1)
        assert result == pytest.approx(0.9, abs=0.01)

    def test_ece_range(self):
        """ECE must always be in [0, 1]."""
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 100)
        scores = rng.uniform(0, 1, 100)
        result = ece(y, scores)
        assert 0.0 <= result <= 1.0

    def test_ece_bins_argument(self):
        """ECE with different bin counts should still return a valid float."""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        r5 = ece(y, scores, n_bins=5)
        r10 = ece(y, scores, n_bins=10)
        assert 0.0 <= r5 <= 1.0
        assert 0.0 <= r10 <= 1.0


# ---------------------------------------------------------------------------
# best_f1_threshold
# ---------------------------------------------------------------------------


class TestBestF1Threshold:
    def test_clear_threshold(self):
        """Clear gap at 0.5: threshold should be in [0.4, 0.6] and F1=1.0."""
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        t, f1 = best_f1_threshold(y, scores)
        assert f1 == pytest.approx(1.0)
        assert 0.05 <= t <= 0.95

    def test_f1_between_zero_and_one(self):
        """F1 should always be in [0, 1]."""
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 50)
        scores = rng.uniform(0, 1, 50)
        t, f1 = best_f1_threshold(y, scores)
        assert 0.0 <= f1 <= 1.0
        assert 0.05 <= t <= 0.95

    def test_threshold_improves_f1(self):
        """The searched threshold should be >= F1 at default 0.5."""
        from sklearn.metrics import f1_score

        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        t, f1 = best_f1_threshold(y, scores)
        f1_at_half = f1_score(y, (scores >= 0.5).astype(int), zero_division=0)
        assert f1 >= f1_at_half - 1e-6


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_ci_shape_and_order(self):
        """Bootstrap CI should have lower <= upper."""
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        lo, hi = bootstrap_ci(auroc, y, scores, n=200, seed=0)
        assert lo <= hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_ci_covers_point_estimate(self):
        """CI should contain the point estimate for a well-behaved metric."""
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])
        point = auroc(y, scores)
        lo, hi = bootstrap_ci(auroc, y, scores, n=500, seed=42)
        assert lo <= point <= hi

    def test_perfect_classifier_ci_near_one(self):
        """CI for a perfect classifier should be close to [1.0, 1.0]."""
        y = np.array([0] * 20 + [1] * 20)
        scores = np.concatenate([np.full(20, 0.1), np.full(20, 0.9)])
        lo, hi = bootstrap_ci(auroc, y, scores, n=200, seed=0)
        assert lo > 0.9

    def test_bootstrap_n_samples(self):
        """Running with different n should give different (but plausible) intervals."""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        scores = np.array([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.1, 0.9])
        lo100, hi100 = bootstrap_ci(auroc, y, scores, n=100, seed=5)
        lo500, hi500 = bootstrap_ci(auroc, y, scores, n=500, seed=5)
        # Both should be in valid range
        assert 0 <= lo100 <= hi100 <= 1
        assert 0 <= lo500 <= hi500 <= 1
