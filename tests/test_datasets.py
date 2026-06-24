"""Tests for dataset loaders / synthetic data."""

from __future__ import annotations

from trust_probe.datasets import synthetic_dataset


def test_synthetic_dataset_shape_and_labels():
    recs = synthetic_dataset(n=50, seed=3)
    assert len(recs) == 50
    for r in recs:
        assert "label" in r
        assert r["label"] in (0, 1)


def test_synthetic_dataset_is_deterministic():
    a = synthetic_dataset(n=20, seed=7)
    b = synthetic_dataset(n=20, seed=7)
    assert [r["label"] for r in a] == [r["label"] for r in b]
