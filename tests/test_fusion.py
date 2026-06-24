"""Tests for the fusion meta-classifier."""

from __future__ import annotations

import numpy as np

from trust_probe.fusion import FusionClassifier


def test_predict_risk_in_unit_interval():
    rng = np.random.default_rng(0)
    y = (rng.random(60) < 0.5).astype(int)
    X = np.column_stack([y + rng.normal(0, 0.3, 60), y + rng.normal(0, 0.3, 60)])
    fc = FusionClassifier(method="lr", calibrate=False, random_state=0).fit(X, y)
    risk = fc.predict_risk(X)
    assert risk.shape == (60,)
    assert ((risk >= 0.0) & (risk <= 1.0)).all()


def test_fusion_separates_easy_case():
    rng = np.random.default_rng(1)
    y = (rng.random(80) < 0.5).astype(int)
    z = 2 * y - 1
    X = np.column_stack([z + rng.normal(0, 0.5, 80), z + rng.normal(0, 0.5, 80)])
    fc = FusionClassifier(method="lr", calibrate=False, random_state=1).fit(X, y)
    risk = fc.predict_risk(X)
    # risk should be higher on average for the positive (hallucinated) class
    assert risk[y == 1].mean() > risk[y == 0].mean()
