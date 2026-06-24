# Results — white-box hallucination probing, validated

Real, reproduced measurement of the white-box half of the trust layer on a public
benchmark, with an honest accounting of where fusion helps and where it doesn't.

## Setup
- **Model:** Qwen2.5-7B-Instruct (8-bit), `output_hidden_states`.
- **Data:** TruthfulQA (generation) — each question yields a *correct* answer
  (faithful, label 0) and an *incorrect* answer (hallucination, label 1). **N = 1000**, balanced.
- **Feature:** last-token hidden state at each of 16 mid/late layers.
- **Probe:** standardized logistic regression, **5-fold out-of-fold** scoring (no train/test leak).
- **Metric:** AUROC with 1000-sample bootstrap 95% CIs.

## Headline result
A linear probe on **layer-19 activations detects TruthfulQA hallucinations at
AUROC 0.916 (95% CI [0.898, 0.931])**. The signal localizes to the **mid-late
layers** — AUROC climbs from 0.745 (layer 10) to a peak of 0.916 (layer 19),
then declines — reproducing the layer-localization finding from the M.Tech thesis
(`hallucination-detection-probing`) on a **different model and a public benchmark**,
and consistent with the "LLMs internally encode truthfulness" literature.

![per-layer AUROC](assets/layer_auroc.png)

## Leaderboard
| Detector | AUROC | 95% CI |
|---|---|---|
| **white-box probe (layer 19)** | **0.916** | [0.898, 0.931] |
| fusion (white-box + logprob) | 0.893 | [0.871, 0.913] |
| answer-logprob baseline | 0.460 | [0.425, 0.494] |

## Honest findings
- **The white-box probe is the result.** 0.916 with a tight CI on a public set is strong and defensible.
- **Fusion with the logprob baseline does *not* help** (0.893 < 0.916). The logprob signal is
  near-random (0.460), so fusing it only injects noise. A meta-classifier can't rescue a useless feature.
- **Fusion needs a *good* black-box signal.** The repo also evaluates a **P(True) self-evaluation**
  signal (ask the model whether its own answer is correct) and supports SelfCheckGPT-style
  consistency. Caveat: when one signal is already near the task ceiling (0.916), fusion has little
  headroom — fusion pays off most when *neither* signal alone is strong.

## Reproduce
GPU (Kaggle/Colab), ~10 min — one cell:
```bash
!wget -q -O validate.py "https://raw.githubusercontent.com/im-girisankar/trust-probe/main/kaggle/validate_kaggle.py" && python validate.py
```
Or automatically via CI (`.github/workflows/kaggle.yml`): a push to `kaggle/` runs the
validation on a Kaggle GPU through GitHub Actions and streams the leaderboard into the run log.

## Limitations
- TruthfulQA correct-vs-incorrect answers are a *proxy* for hallucination labels; RAGTruth /
  HaluEval give span-level RAG labels and are the natural next step.
- Results are model-specific (probes are trained per model) and English-only.
- White-box probing requires open-weights models with accessible hidden states.
