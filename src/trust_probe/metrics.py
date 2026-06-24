"""metrics.py — Evaluation metrics for hallucination detection.

All functions are deterministic and CPU-only.  sklearn is used where applicable;
ECE is hand-computed to avoid sklearn version inconsistencies.

Public API
----------
auroc(y_true, scores) -> float
    Area under the ROC curve.

auprc(y_true, scores) -> float
    Area under the precision-recall curve.

ece(y_true, scores, n_bins=10) -> float
    Expected Calibration Error.

best_f1_threshold(y_true, scores) -> tuple[float, float]
    (threshold, best_F1) maximising F1 over a grid.

bootstrap_ci(metric_fn, y_true, scores, n=1000, seed=42) -> tuple[float, float]
    95% bootstrap confidence interval for any scalar metric.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve.

    Parameters
    ----------
    y_true : array-like of int, shape (N,)
        Binary ground truth labels (0/1).
    scores : array-like of float, shape (N,)
        Predicted hallucination risk scores (higher = more likely positive).

    Returns
    -------
    float in [0, 1].  0.5 = random; 1.0 = perfect.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve.

    Parameters
    ----------
    y_true : array-like of int, shape (N,)
    scores : array-like of float, shape (N,)

    Returns
    -------
    float in [0, 1].
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def ece(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE).

    Bins predicted probabilities equally into ``n_bins`` buckets and measures
    the weighted mean absolute difference between mean predicted probability
    and mean actual positive rate within each non-empty bin.

    Parameters
    ----------
    y_true : array-like of int, shape (N,)
    scores : array-like of float, shape (N,)
        Should be in [0, 1] for meaningful calibration.
    n_bins : int
        Number of equal-width bins.  Default 10.

    Returns
    -------
    float >= 0.  0.0 = perfectly calibrated.
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=False):
        mask = (scores >= lo) & (scores <= hi if hi == 1.0 else scores < hi)
        if mask.sum() == 0:
            continue
        mean_pred = scores[mask].mean()
        mean_true = y_true[mask].mean()
        ece_val += mask.sum() / n * abs(mean_pred - mean_true)

    return float(ece_val)


def best_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find the threshold that maximises binary F1 score.

    Parameters
    ----------
    y_true : array-like of int, shape (N,)
    scores : array-like of float, shape (N,)
    grid : array-like of float, optional
        Threshold candidates.  Defaults to np.arange(0.05, 0.95, 0.01).

    Returns
    -------
    (threshold, best_f1) — both floats.
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.01)

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        preds = (scores >= t).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)

    return float(best_t), float(best_f1)


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    scores: np.ndarray,
    n: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for a scalar metric.

    Parameters
    ----------
    metric_fn:
        Callable taking (y_true, scores) and returning a float.
    y_true : array-like of int, shape (N,)
    scores : array-like of float, shape (N,)
    n : int
        Number of bootstrap resamples.  Default 1000.
    seed : int
        Random seed for reproducibility.
    alpha : float
        Significance level.  Default 0.05 → 95% CI.

    Returns
    -------
    (lower, upper) — the (alpha/2, 1-alpha/2) percentile interval.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)

    N = len(y_true)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        try:
            val = metric_fn(y_true[idx], scores[idx])
        except Exception:
            val = float("nan")
        samples.append(val)

    samples_arr = np.array(samples)
    valid = samples_arr[~np.isnan(samples_arr)]
    if len(valid) == 0:
        return (float("nan"), float("nan"))

    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return lo, hi
