# trust-probe

A **real-time hallucination trust layer** for open LLMs that fuses **white-box** internal
activation probes with **black-box** reliability signals (self-consistency + faithfulness),
with a full evaluation harness. This is the research core that feeds
[`llm-firewall`](https://github.com/im-girisankar/llm-firewall) as a runtime guardrail, built
on the activation-probing approach from my M.Tech thesis
([`hallucination-detection-probing`](https://github.com/im-girisankar/hallucination-detection-probing)).

> **Why this is different:** most hallucination detectors are *either* white-box (internal states)
> *or* black-box (output consistency). trust-probe **fuses both** and detects mid-generation —
> the combination is under-explored and is the novelty.

## Status
- **CPU core (done):** datasets, eval harness (AUROC/AUPRC/ECE/bootstrap CIs), baselines,
  the sklearn `LogRegProbe`, black-box signals, and the `FusionClassifier` — all run and are
  tested **offline on synthetic data**. The headline demo is reproducible with no GPU.
- **GPU-gated (code present, not run in CI):** real activation extraction from Llama-3.1-8B
  (`activations.extract_activations`), the torch probe heads (`AttentionMLPProbe`,
  `TinyConvProbeAdapter`), and validation on **RAGTruth / HaluEval** for real AUROC numbers.

## The headline demo (offline, no GPU)
```bash
pip install -e ".[dev]"
trustprobe eval --synthetic
```
Trains the real white-box probe on synthetic activations, then fuses a deliberately-noisy
white-box score with a noisy black-box signal and shows the **fused detector beats either
alone** on AUROC, with a bootstrap CI — the core claim of the project, in miniature:

```
Detector                  AUROC
fusion (WB + BB)          0.911   <- wins
black-box signal (sim.)   0.858
white-box probe (sim.)    0.835
```

## Validating for real (needs a GPU)
Point the loaders at RAGTruth/HaluEval, run `activations.extract_activations` on
Llama-3.1-8B, train the probe, and the same harness produces real AUROC/AUPRC/ECE vs
baselines (SelfCheckGPT / logprob / P(True)). See the methodology in the private
`portfolio-flagship-plan` repo.

## Layout
`datasets.py` (loaders + synthetic) · `activations.py` (GPU extraction + synthetic) ·
`probes.py` (LogReg + torch probe adapters) · `blackbox.py` (consistency + faithfulness) ·
`fusion.py` (meta-classifier) · `metrics.py` (AUROC/AUPRC/ECE/bootstrap) ·
`baselines.py` · `evaluate.py` (harness + leaderboard) · `cli.py`.

## License
MIT © 2026 Girisankar G
