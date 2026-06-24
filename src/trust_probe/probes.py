"""probes.py — Probe adapters with a common sklearn-compatible interface.

Three probe implementations:

1. ``AttentionMLPProbe`` — wraps the thesis ``HallucinationClassifier``
   (attention-weighted MLP over the 16-layer activation tensor).
   LAZY-imports torch; requires trust-probe[gpu].

2. ``TinyConvProbeAdapter`` — wraps the thesis ``TinyConvProbe``
   (1D-conv over layer sequence).
   LAZY-imports torch; requires trust-probe[gpu].

3. ``LogRegProbe`` — pure sklearn logistic regression on compact features
   derived from the activation tensor (numpy only, CPU-friendly).
   This is the CPU-testable probe path.

Common interface (all three)
-----------------------------
    .fit(X, y)           X: np.ndarray (N, L, T, D), y: (N,) int array
    .predict_proba(X)    -> np.ndarray shape (N, 2): [[p_neg, p_pos], ...]

Shape convention (single source of truth, from classifiers.py):
    X: (N, 16, 64, 256) = (batch, layers, tokens, projected_dim)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Feature extraction (pure numpy — used by LogRegProbe)
# ---------------------------------------------------------------------------


def _token_pool_np(X: np.ndarray) -> np.ndarray:
    """(N, L, T, D) -> (N, L, 2D): per-layer mean & std over token axis."""
    mean = X.mean(axis=2)  # (N, L, D)
    std = X.std(axis=2)    # (N, L, D)
    return np.concatenate([mean, std], axis=-1)  # (N, L, 2D)


def _flatten_pool(X: np.ndarray) -> np.ndarray:
    """Flatten pooled tensor to (N, L*2D) for sklearn probes."""
    pooled = _token_pool_np(X)  # (N, L, 2D)
    N = pooled.shape[0]
    return pooled.reshape(N, -1)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseProbe(ABC):
    """Abstract base for all trust-probe probe adapters."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseProbe:
        """Train the probe.  X: (N, L, T, D), y: (N,) int."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability matrix [[p_neg, p_pos], ...]."""
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions (0/1) at the given threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# ---------------------------------------------------------------------------
# CPU probe: LogRegProbe (pure numpy + sklearn)
# ---------------------------------------------------------------------------


class LogRegProbe(BaseProbe):
    """Logistic regression probe on flattened mean+std of activations.

    Features: (N, L*2D) = (N, 16*512) = (N, 8192) flattened token-pool.
    Trained with sklearn LogisticRegression (CPU, no torch needed).

    Parameters
    ----------
    C : float
        Inverse regularisation strength.
    max_iter : int
        Maximum iterations for the solver.
    class_weight : str or None
        'balanced' to handle imbalanced datasets.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
    ) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight=class_weight,
                        solver="saga",
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogRegProbe:
        """Train on activation tensor X of shape (N, L, T, D)."""
        feats = _flatten_pool(X)
        self.pipeline.fit(feats, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 2) probability array."""
        if not self._fitted:
            raise RuntimeError("LogRegProbe must be fitted before predict_proba.")
        feats = _flatten_pool(X)
        return self.pipeline.predict_proba(feats)


# ---------------------------------------------------------------------------
# GPU probes: AttentionMLPProbe and TinyConvProbeAdapter (lazy torch)
# ---------------------------------------------------------------------------


class _TorchProbeBase(BaseProbe):
    """Base for torch-based probes; lazy-imports torch and the thesis classifiers."""

    def __init__(self, **kwargs: Any) -> None:
        self._model: Any = None
        self._kwargs = kwargs
        self._fitted = False

    def _build_model(self) -> Any:
        raise NotImplementedError

    def _ensure_torch(self) -> Any:
        try:
            import torch

            return torch
        except ImportError as e:
            raise ImportError(
                "torch is required for neural probes. "
                "Install with: pip install trust-probe[gpu]"
            ) from e

    def _to_tensor(self, X: np.ndarray) -> Any:
        torch = self._ensure_torch()
        return torch.tensor(X, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _TorchProbeBase:
        """Minimal sklearn-style fit: one epoch of Adam on BCE loss."""
        torch = self._ensure_torch()
        if self._model is None:
            self._model = self._build_model()

        X_t = self._to_tensor(X)
        y_t = torch.tensor(y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        self._model.train()
        for _ in range(20):  # 20 epochs (minimal for synthetic convergence)
            optimizer.zero_grad()
            logits = self._model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._model is None:
            raise RuntimeError("Probe must be fitted before predict_proba.")
        torch = self._ensure_torch()
        with torch.no_grad():
            self._model.eval()
            X_t = self._to_tensor(X)
            logits = self._model(X_t)
            probs_pos = torch.sigmoid(logits).numpy()
        probs_neg = 1.0 - probs_pos
        return np.stack([probs_neg, probs_pos], axis=1)


class AttentionMLPProbe(_TorchProbeBase):
    """Wraps the thesis ``HallucinationClassifier`` (attention MLP).

    Requires: trust-probe[gpu] (torch + the hallucination-detection-probing package
    or a local copy of classifiers.py on PYTHONPATH).

    Parameters
    ----------
    hidden : int
        Hidden dimension of the MLP head (default 64).
    dropout : float
        Dropout rate (default 0.5).
    """

    def __init__(self, hidden: int = 64, dropout: float = 0.5) -> None:
        super().__init__(hidden=hidden, dropout=dropout)

    def _build_model(self) -> Any:
        try:
            from models.classifiers import HallucinationClassifier
        except ImportError:
            # Fallback: inline minimal AttentionMLP (matches thesis exactly)
            return _InlineAttentionMLP(**self._kwargs)
        return HallucinationClassifier(**self._kwargs)


class TinyConvProbeAdapter(_TorchProbeBase):
    """Wraps the thesis ``TinyConvProbe`` (1D-conv over layer sequence).

    Requires: trust-probe[gpu].
    """

    def __init__(self, embed: int = 64, dropout: float = 0.3) -> None:
        super().__init__(embed=embed, dropout=dropout)

    def _build_model(self) -> Any:
        try:
            from models.classifiers import TinyConvProbe
        except ImportError:
            return _InlineTinyConv(**self._kwargs)
        return TinyConvProbe(**self._kwargs)


# ---------------------------------------------------------------------------
# Inline fallbacks (exact re-implementations for when thesis package not on path)
# ---------------------------------------------------------------------------


def _try_import_torch_nn() -> Any:
    """Return torch.nn or raise ImportError with clear message."""
    try:
        import torch.nn as nn

        return nn
    except ImportError as e:
        raise ImportError("torch is required for neural probes.") from e


class _InlineAttentionMLP:
    """Minimal inline version of HallucinationClassifier — matches thesis exactly."""

    def __init__(self, hidden: int = 64, dropout: float = 0.5) -> None:
        import torch
        import torch.nn as nn

        POOL_DIM = 512  # 2 * 256

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer_attn = nn.Linear(POOL_DIM, 1)
                self.layer_norm = nn.LayerNorm(POOL_DIM)
                self.linear1 = nn.Linear(POOL_DIM, hidden)
                self.bn1 = nn.BatchNorm1d(hidden)
                self.act = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(hidden, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                mean = x.mean(dim=2)
                std = x.std(dim=2)
                p = torch.cat([mean, std], dim=-1)  # (B, L, 512)
                scores = self.layer_attn(p)           # (B, L, 1)
                alpha = torch.softmax(scores, dim=1)
                z = (alpha * p).sum(dim=1)            # (B, 512)
                z = self.layer_norm(z)
                h = self.dropout(self.act(self.bn1(self.linear1(z))))
                return self.linear2(h).squeeze(-1)    # (B,)

        self._inner = _Model()

    def __call__(self, x: Any) -> Any:
        return self._inner(x)

    def train(self) -> None:
        self._inner.train()

    def eval(self) -> None:
        self._inner.eval()

    def parameters(self) -> Any:
        return self._inner.parameters()


class _InlineTinyConv:
    """Minimal inline version of TinyConvProbe — matches thesis exactly."""

    def __init__(self, embed: int = 64, dropout: float = 0.3) -> None:
        import torch
        import torch.nn as nn

        POOL_DIM = 512

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer_embed = nn.Linear(POOL_DIM, embed, bias=False)
                self.conv1 = nn.Conv1d(embed, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
                self.act = nn.GELU()
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(64, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                mean = x.mean(dim=2)
                std = x.std(dim=2)
                p = torch.cat([mean, std], dim=-1)  # (B, L, 512)
                e = self.layer_embed(p)              # (B, L, embed)
                e = e.transpose(1, 2)               # (B, embed, L)
                h = self.dropout(self.act(self.conv1(e)))
                h = self.act(self.conv2(h))
                mean_pool = h.mean(dim=2)
                max_pool = h.max(dim=2).values
                z = torch.cat([mean_pool, max_pool], dim=-1)
                return self.head(z).squeeze(-1)

        self._inner = _Model()

    def __call__(self, x: Any) -> Any:
        return self._inner(x)

    def train(self) -> None:
        self._inner.train()

    def eval(self) -> None:
        self._inner.eval()

    def parameters(self) -> Any:
        return self._inner.parameters()
