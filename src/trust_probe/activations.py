"""activations.py — Activation extraction from Llama-3.1-8B-Instruct (GPU-guarded).

This module has TWO paths:

GPU path (lazy-imported, GPU-guarded)
--------------------------------------
``extract_activations(records, model_id, layers)`` — loads the model with
``output_hidden_states=True``, runs a forward pass per record, projects each
hidden state from its native dimension D_model down to 256 using a fixed random
projection, and returns:
  - activations: np.ndarray of shape (N, L, T=64, 256)
  - projector:   np.ndarray of shape (D_model, 256)

The fixed random projection is seeded at 42 and can be stored/reloaded to
keep embeddings comparable across runs.

CPU/synthetic path (no torch required)
---------------------------------------
``SyntheticActivations(n, seed)`` — produces activation arrays of exactly the
same shape as the GPU path but filled with deterministic random values.
Label-correlated signal is injected so probes can actually train on it.

Activation tensor shape (single source of truth)
-------------------------------------------------
  (N, L, T, D) = (samples, n_layers, 64 tokens, 256 projected dim)

Where:
  N = number of records
  L = len(layers) — typically 16 (layers 8-23 of Llama-3.1-8B)
  T = 64 (token sequence length, padded/truncated)
  D = 256 (projected hidden dim)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass  # avoid top-level torch import; only used in type comments below

_PROJ_DIM = 256
_N_TOKENS = 64
_DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_DEFAULT_LAYERS = range(8, 24)  # 16 layers: indices 8-23


def _make_projector(d_model: int, seed: int = 42) -> np.ndarray:
    """Create a fixed random projection matrix (d_model, 256) with unit-norm columns."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((d_model, _PROJ_DIM)).astype(np.float32)
    # Normalize columns
    norms = np.linalg.norm(P, axis=0, keepdims=True)
    return P / (norms + 1e-8)


def extract_activations(
    records: list[dict[str, Any]],
    model_id: str = _DEFAULT_MODEL,
    layers: range | list[int] = _DEFAULT_LAYERS,
    max_new_tokens: int = 1,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract hidden-state activations from a Llama model.

    REQUIRES torch + transformers (GPU-guarded: not called by any test).

    Parameters
    ----------
    records:
        List of dicts with at least 'prompt' and 'response' keys.
    model_id:
        HuggingFace model identifier.
    layers:
        Which transformer hidden-state layer indices to extract (0-indexed
        from the model's hidden_states tuple, which includes the embedding).
        Default: range(8, 24) — 16 layers of Llama-3.1-8B.
    max_new_tokens:
        Only matters to prevent generation; set to 1 for pure encoding.
    device:
        'cpu' or 'cuda'.

    Returns
    -------
    activations : np.ndarray, shape (N, L, T=64, 256)
    projector   : np.ndarray, shape (D_model, 256)
        The fixed projection matrix used; store this to keep embeddings
        comparable across different calls.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "torch and transformers are required for extract_activations. "
            "Install with: pip install trust-probe[gpu]"
        ) from e

    layers_list = list(layers)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        output_hidden_states=True,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    # Infer D_model from first hidden state
    d_model: int | None = None
    projector: np.ndarray | None = None
    all_acts: list[np.ndarray] = []

    with torch.no_grad():
        for rec in records:
            text = f"[INST] {rec['prompt']} [/INST] {rec['response']}"
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=_N_TOKENS,
                padding="max_length",
            ).to(device)

            out = model(**enc)
            hidden_states = out.hidden_states  # tuple of (B, T, D_model), len = n_layers+1

            if d_model is None:
                d_model = hidden_states[0].shape[-1]
                projector = _make_projector(d_model)

            # Extract and project each requested layer
            layer_acts = []
            for li in layers_list:
                h = hidden_states[li][0]  # (T, D_model)
                # Pad or truncate to _N_TOKENS
                if h.shape[0] < _N_TOKENS:
                    pad = torch.zeros(_N_TOKENS - h.shape[0], d_model, device=device)
                    h = torch.cat([h, pad], dim=0)
                else:
                    h = h[:_N_TOKENS]
                h_np = h.cpu().numpy()  # (T, D_model)
                projected = h_np @ projector  # type: ignore[operator]  # (T, 256)
                layer_acts.append(projected)

            # Stack: (L, T, 256)
            sample_acts = np.stack(layer_acts, axis=0)
            all_acts.append(sample_acts)

    # (N, L, T, 256)
    activations = np.stack(all_acts, axis=0)
    assert projector is not None
    return activations, projector


class SyntheticActivations:
    """Generate fake activation tensors of shape (N, L, T=64, D=256).

    No torch required. The activations have label-correlated signal injected
    so that probes trained on them achieve non-trivial AUC.

    Usage
    -----
    >>> sa = SyntheticActivations(n=100, seed=42)
    >>> X, y = sa.X, sa.y      # (100, 16, 64, 256), (100,)
    """

    N_LAYERS: int = 16
    N_TOKENS: int = _N_TOKENS
    PROJ_DIM: int = _PROJ_DIM

    def __init__(self, n: int = 100, seed: int = 42) -> None:
        self.n = n
        self.seed = seed
        self.X, self.y = self._generate()

    def _generate(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.n, self.N_LAYERS, self.N_TOKENS, self.PROJ_DIM)).astype(
            np.float32
        )
        y = np.array([i % 2 for i in range(self.n)], dtype=np.int64)

        # Inject discriminative signal: hallucinated (y=1) samples have
        # elevated activations in later layers (indices 8-15)
        mask = y == 1
        X[mask, 8:, :, :] += 1.2

        return X, y
