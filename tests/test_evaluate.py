"""Tests for the evaluation harness + the fusion demonstration."""

from __future__ import annotations

import numpy as np

from trust_probe.evaluate import compare, evaluate, synthetic_fusion_demo


def test_evaluate_returns_all_metrics():
    records = [{"label": i % 2} for i in range(20)]
    # perfect detector: score == label
    rep = evaluate(records, lambda recs: np.array([r["label"] for r in recs]))
    for key in ("name", "auroc", "auprc", "ece", "best_f1", "threshold",
                "auroc_ci", "auprc_ci", "n_total"):
        assert key in rep
    assert rep["auroc"] == 1.0
    assert 0.0 <= rep["ece"] <= 1.0
    assert rep["n_total"] == 20


def test_compare_returns_one_report_per_detector():
    records = [{"label": i % 2} for i in range(30)]
    reports = compare(
        {
            "perfect": lambda recs: np.array([r["label"] for r in recs]),
            "constant": lambda recs: np.full(len(recs), 0.5),
        },
        records,
        n_bootstrap=50,
    )
    assert len(reports) == 2
    names = {r["name"] for r in reports}
    assert names == {"perfect", "constant"}


def test_fusion_beats_each_part():
    """The headline claim: fused detector has the highest AUROC."""
    reports = synthetic_fusion_demo(n=400, seed=7)
    by_name = {r["name"]: r["auroc"] for r in reports}
    fused = by_name["fusion (WB + BB)"]
    wb = by_name["white-box probe (sim.)"]
    bb = by_name["black-box signal (sim.)"]
    assert fused >= wb and fused >= bb
    assert fused == max(by_name.values())
    # signals are individually imperfect (not a trivial 1.0 demo)
    assert wb < 0.95 and bb < 0.95
