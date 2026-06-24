"""fusion.py — Meta-classifier fusing white-box probe scores with black-box signals.

``FusionClassifier`` trains a calibrated sklearn meta-classifier that takes
as features:
  - One or more white-box probe risk scores (float in [0, 1])
  - One or more black-box reliability signals (float in [0, 1])

and outputs a calibrated hallucination risk score in [0, 1].

The meta-classifier is CPU-trainable with sklearn only (no torch).

Typical usage
-------------
>>> from trust_probe.fusion import FusionClassifier
>>> fc = FusionClassifier()
>>> feature_matrix = np.column_stack([probe_scores, hri_scores])  # (N, 2)
>>> fc.fit(feature_matrix, y)
>>> risks = fc.predict_risk(feature_matrix)   # (N,) in [0, 1]

Feature convention
------------------
The feature matrix passed to fit() / predict_risk() must be shape (N, K) where
K = number of signals.  Columns can be in any order; the caller is responsible
for consistent ordering between fit and predict.

Recommended column order (for the standard pipeline):
    [white_box_probe_score, consistency_risk, faithfulness_risk]
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FusionClassifier:
    """Meta-classifier fusing white-box and black-box hallucination signals.

    Internally uses a sklearn GradientBoostingClassifier with isotonic calibration,
    falling back to LogisticRegression if GBM is not available or if the
    dataset is too small for GBM (< 20 samples).

    Parameters
    ----------
    method : {'gbm', 'lr'}
        Meta-classifier backend.  'gbm' uses GradientBoostingClassifier (default);
        'lr' uses LogisticRegression (faster, better for tiny datasets).
    calibrate : bool
        Whether to apply isotonic calibration (CalibratedClassifierCV).
        Default True.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        method: str = "gbm",
        calibrate: bool = True,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.calibrate = calibrate
        self.random_state = random_state
        self._pipeline: Any = None
        self._fitted = False

    def _build_pipeline(self, n_samples: int) -> Any:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # GBM requires more samples than classes (use LR for tiny sets)
        use_gbm = self.method == "gbm" and n_samples >= 20
        if use_gbm:
            try:
                from sklearn.ensemble import GradientBoostingClassifier

                base = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    random_state=self.random_state,
                )
            except ImportError:
                use_gbm = False

        if not use_gbm:
            base = LogisticRegression(  # type: ignore[assignment]
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state,
            )

        if self.calibrate and n_samples >= 10:
            # Need at least enough samples for CV=2
            cv = min(3, n_samples // 2)
            if cv >= 2:
                estimator = CalibratedClassifierCV(base, method="isotonic", cv=cv)
            else:
                estimator = base
        else:
            estimator = base

        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> FusionClassifier:
        """Train the meta-classifier.

        Parameters
        ----------
        X : np.ndarray, shape (N, K)
            Feature matrix of K hallucination signals for N samples.
        y : np.ndarray, shape (N,)
            Binary labels: 0 = faithful, 1 = hallucinated.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self._pipeline = self._build_pipeline(len(y))
        self._pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated hallucination risk scores in [0, 1].

        Parameters
        ----------
        X : np.ndarray, shape (N, K)

        Returns
        -------
        np.ndarray, shape (N,) — risk scores in [0, 1].
        """
        if not self._fitted or self._pipeline is None:
            raise RuntimeError("FusionClassifier must be fitted before predict_risk.")
        X = np.asarray(X, dtype=np.float64)
        return self._pipeline.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions (0/1)."""
        return (self.predict_risk(X) >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability matrix for sklearn compatibility."""
        risks = self.predict_risk(X)
        return np.column_stack([1.0 - risks, risks])
