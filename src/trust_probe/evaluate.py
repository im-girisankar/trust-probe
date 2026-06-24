"""evaluate.py — Evaluation harness for hallucination detectors.

Public API
----------
evaluate(records, detector) -> Report
    Run detector over records, compute AUROC/AUPRC/ECE/best-F1 + bootstrap CIs.

compare(detectors, records) -> list[Report]
    Run multiple detectors and return leaderboard-ordered Reports.

render_markdown(reports) -> str
    Render a leaderboard table as a markdown string.

cross_dataset_eval(train_records, test_records, detector_factory) -> Report
    Train detector on train_records, evaluate on test_records.

Types
-----
Report = TypedDict with keys:
    name, auroc, auprc, ece, best_f1, threshold,
    auroc_ci, auprc_ci, n_pos, n_neg, n_total
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np

from trust_probe.metrics import (
    auprc,
    auroc,
    best_f1_threshold,
    bootstrap_ci,
    ece,
)

# ---------------------------------------------------------------------------
# Report type
# ---------------------------------------------------------------------------


class Report(TypedDict):
    """Evaluation result for a single detector."""

    name: str
    auroc: float
    auprc: float
    ece: float
    best_f1: float
    threshold: float
    auroc_ci: tuple[float, float]
    auprc_ci: tuple[float, float]
    n_pos: int
    n_neg: int
    n_total: int


# ---------------------------------------------------------------------------
# Detector protocol (duck-typed)
# ---------------------------------------------------------------------------
# A detector is any object with:
#   .name : str   (optional, used as display name)
#   __call__(records) -> np.ndarray  (shape (N,), scores in [0, 1])
# OR a plain callable.


def _get_name(detector: Any) -> str:
    if hasattr(detector, "name"):
        return str(detector.name)
    if hasattr(detector, "__name__"):
        return str(detector.__name__)
    return repr(detector)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate(
    records: list[dict[str, Any]],
    detector: Any,
    name: str | None = None,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> Report:
    """Evaluate a detector over a set of records.

    Parameters
    ----------
    records:
        List of dicts with at least 'label' (int 0/1) key.
    detector:
        Callable: records -> np.ndarray of scores in [0, 1].
    name:
        Display name.  Inferred from detector if not provided.
    n_bootstrap:
        Number of bootstrap resamples for CIs.
    bootstrap_seed:
        Seed for bootstrap CIs.

    Returns
    -------
    Report dict with all evaluation metrics.
    """
    y = np.array([rec["label"] for rec in records], dtype=int)
    scores = np.asarray(detector(records), dtype=float)

    if len(scores) != len(y):
        raise ValueError(
            f"Detector returned {len(scores)} scores for {len(y)} records."
        )

    roc = auroc(y, scores)
    prc = auprc(y, scores)
    cal = ece(y, scores)
    thresh, f1 = best_f1_threshold(y, scores)

    roc_ci = bootstrap_ci(auroc, y, scores, n=n_bootstrap, seed=bootstrap_seed)
    prc_ci = bootstrap_ci(auprc, y, scores, n=n_bootstrap, seed=bootstrap_seed)

    return Report(
        name=name or _get_name(detector),
        auroc=roc,
        auprc=prc,
        ece=cal,
        best_f1=f1,
        threshold=thresh,
        auroc_ci=roc_ci,
        auprc_ci=prc_ci,
        n_pos=int(y.sum()),
        n_neg=int((1 - y).sum()),
        n_total=len(y),
    )


# ---------------------------------------------------------------------------
# Multi-detector comparison
# ---------------------------------------------------------------------------


def compare(
    detectors: dict[str, Any],
    records: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> list[Report]:
    """Run multiple detectors and return Reports sorted by AUROC (descending).

    Parameters
    ----------
    detectors:
        Dict mapping display name -> callable(records) -> scores.
    records:
        Evaluation records with 'label' key.
    n_bootstrap:
        Bootstrap resamples for CIs.
    bootstrap_seed:
        Seed.

    Returns
    -------
    list[Report] sorted by auroc descending.
    """
    reports = []
    for name, det in detectors.items():
        r = evaluate(records, det, name=name, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)
        reports.append(r)
    return sorted(reports, key=lambda r: r["auroc"], reverse=True)


# ---------------------------------------------------------------------------
# Cross-dataset evaluation
# ---------------------------------------------------------------------------


def cross_dataset_eval(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    detector_factory: Callable[[], Any],
    name: str = "cross_dataset",
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> Report:
    """Train a detector on train_records and evaluate on test_records.

    Parameters
    ----------
    train_records:
        Records for training (must have 'label' and the fields the detector needs).
    test_records:
        Records for evaluation.
    detector_factory:
        Callable() -> detector with a .fit(records) or .fit(X, y) method.
        The factory is called fresh; its .fit(train_records) is called first.
    name:
        Display name for the report.
    n_bootstrap:
        Bootstrap resamples.
    bootstrap_seed:
        Seed.

    Returns
    -------
    Report evaluated on test_records.
    """
    detector = detector_factory()
    if hasattr(detector, "fit"):
        detector.fit(train_records)
    return evaluate(test_records, detector, name=name, n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "Detector": 28,
    "AUROC": 8,
    "AUPRC": 8,
    "ECE": 7,
    "Best-F1": 8,
    "Threshold": 10,
    "N": 6,
}


def render_markdown(reports: list[Report]) -> str:
    """Render a leaderboard table as a markdown string.

    Parameters
    ----------
    reports:
        List of Report dicts (as returned by compare()).

    Returns
    -------
    str — a markdown table with header, separator, and one row per report.
    """
    header = (
        f"| {'Detector':<26} | {'AUROC':>6} | {'AUPRC':>6} | "
        f"{'ECE':>5} | {'Best-F1':>7} | {'Threshold':>9} | {'N':>5} |"
    )
    sep = "|" + "|".join(["-" * (w + 2) for w in [28, 8, 8, 7, 9, 11, 7]]) + "|"

    rows = []
    for r in reports:
        row = (
            f"| {r['name']:<26} | {r['auroc']:>6.3f} | {r['auprc']:>6.3f} | "
            f"{r['ece']:>5.3f} | {r['best_f1']:>7.3f} | {r['threshold']:>9.3f} | "
            f"{r['n_total']:>5} |"
        )
        rows.append(row)

    return "\n".join([header, sep] + rows)


def render_text_table(reports: list[Report]) -> str:
    """Render a plain-text leaderboard table (no markdown).

    Used by the CLI for terminal output.
    """
    if not reports:
        return "(no results)"

    header = (
        f"{'Detector':<28}  {'AUROC':>6}  {'AUPRC':>6}  "
        f"{'ECE':>5}  {'Best-F1':>7}  {'Threshold':>9}  {'N':>5}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in reports:
        row = (
            f"{r['name']:<28}  {r['auroc']:>6.3f}  {r['auprc']:>6.3f}  "
            f"{r['ece']:>5.3f}  {r['best_f1']:>7.3f}  {r['threshold']:>9.3f}  "
            f"{r['n_total']:>5}"
        )
        rows.append(row)
    rows.append(sep)
    top = max(reports, key=lambda r: r["auroc"])
    t_lo, t_hi = top["auroc_ci"]
    rows.append(f"AUROC 95% CI for top detector ({top['name']}): [{t_lo:.3f}, {t_hi:.3f}]")
    return "\n".join(rows)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def synthetic_fusion_demo(n: int = 400, seed: int = 7) -> list[Report]:
    """Controlled offline demo isolating the fusion claim.

    Simulates a noisy white-box probe score and a noisy black-box risk signal,
    each correlated with the label but individually imperfect, then trains the
    :class:`~trust_probe.fusion.FusionClassifier` on them. Returns the leaderboard
    so a caller/test can assert the fused detector beats either signal alone on
    AUROC. The signals are deterministically *simulated* (seeded) so the fusion
    benefit is reproducible; the real probe and black-box modules are exercised
    by their own tests.
    """
    from trust_probe.fusion import FusionClassifier

    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    z = 2.0 * y - 1.0
    # Deliberately noisy, individually-imperfect signals so the fusion gain is real.
    wb = _sigmoid(0.85 * z + rng.standard_normal(n))
    bb = _sigmoid(0.70 * z + rng.standard_normal(n))

    n_tr = (n * 3) // 4
    fusion = FusionClassifier(method="lr", calibrate=False, random_state=seed)
    fusion.fit(np.column_stack([wb[:n_tr], bb[:n_tr]]), y[:n_tr])

    wb_te, bb_te, y_te = wb[n_tr:], bb[n_tr:], y[n_tr:]
    fus_te = fusion.predict_risk(np.column_stack([wb_te, bb_te]))
    recs = [{"label": int(v)} for v in y_te]

    detectors: dict[str, Callable[[list[dict[str, Any]]], np.ndarray]] = {
        "white-box probe (sim.)": lambda r: wb_te[: len(r)],
        "black-box signal (sim.)": lambda r: bb_te[: len(r)],
        "fusion (WB + BB)": lambda r: fus_te[: len(r)],
    }
    return compare(detectors, recs, n_bootstrap=300, bootstrap_seed=seed)
