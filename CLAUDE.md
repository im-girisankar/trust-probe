# trust-probe — working notes

White-box (activation probes) ⊕ black-box (consistency/faithfulness) hallucination detection,
with an eval harness. CPU core is tested offline on synthetic data; torch/transformers paths
are lazy-imported and GPU-only.

## Conventions
- Heavy deps (torch/transformers/sentence-transformers) lazy-imported inside functions only;
  the tested core needs just numpy + scikit-learn. ruff: line-length 100, E/F/I/UP/B, ignore E501.
- `trustprobe eval --synthetic` MUST run offline and show fusion beating each signal alone.
- Commits authored solely as Girisankar G — no co-author trailers.

## GPU-gated (not in CI): activations.extract_activations, AttentionMLPProbe, TinyConvProbeAdapter.
