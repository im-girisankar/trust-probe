"""Tests for the CPU white-box probe path (no torch)."""

from __future__ import annotations

from trust_probe.activations import SyntheticActivations
from trust_probe.metrics import auroc
from trust_probe.probes import LogRegProbe


def test_logreg_probe_fits_and_predicts():
    sa = SyntheticActivations(n=120, seed=1)
    probe = LogRegProbe(max_iter=300)
    probe.fit(sa.X[:90], sa.y[:90])
    proba = probe.predict_proba(sa.X[90:])
    assert proba.shape == (30, 2)
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_logreg_probe_learns_the_signal():
    sa = SyntheticActivations(n=160, seed=2)
    probe = LogRegProbe(max_iter=400)
    probe.fit(sa.X[:120], sa.y[:120])
    scores = probe.predict_proba(sa.X[120:])[:, 1]
    # the synthetic activations carry a label-correlated signal
    assert auroc(sa.y[120:], scores) > 0.8
