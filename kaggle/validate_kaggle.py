# =====================================================================
# trust-probe — real validation on Kaggle GPU (copy-paste one cell)
# =====================================================================
# Kaggle setup: Notebook -> Settings -> Accelerator = GPU (T4 or P100),
# Internet = ON. Then paste this whole file into one cell and Run.
#
# It produces a REAL hallucination-detection leaderboard:
#   - white-box activation probe (per-layer + best)   <- the thesis signal
#   - answer mean log-probability baseline            <- simple black-box-ish
#   - FUSION of the two                               <- the core claim
# with AUROC + bootstrap 95% CIs and a per-layer AUROC plot.
#
# Labels: TruthfulQA correct answers (faithful=0) vs incorrect answers
# (hallucination=1) — a reliable, public, license-free proxy. Swap DATASET/
# MODEL_ID below for RAGTruth + Llama-3.1-8B once you have HF access.
# =====================================================================

import subprocess, sys
def pip(*pkgs): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
pip("transformers>=4.46", "accelerate>=0.30", "datasets>=2.19", "scikit-learn>=1.3", "bitsandbytes>=0.43")

import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ---------------- config ----------------
# No-gating default (just works). For the thesis model use
# "meta-llama/Llama-3.1-8B-Instruct" (accept its license on HF + set HF_TOKEN
# as a Kaggle secret) and LAYERS = range(8, 24).
MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
LOAD_8BIT  = True
N_PAIRS    = 500          # examples per class-ish (kept small for speed)
LAYERS     = None         # None = auto-pick a 16-layer mid/late band
SEED       = 7
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ---------------- data: TruthfulQA -> labeled (Q, A, hallucination?) -------
print("Loading TruthfulQA ...")
ds = load_dataset("truthful_qa", "generation")["validation"]
examples = []
for row in ds:
    q = row["question"]
    correct = (row.get("correct_answers") or [row.get("best_answer")])
    wrong   = row.get("incorrect_answers") or []
    if correct and correct[0]:
        examples.append((q, correct[0], 0))
    if wrong and wrong[0]:
        examples.append((q, wrong[0], 1))
rng.shuffle(examples)
examples = examples[: 2 * N_PAIRS]
y = np.array([e[2] for e in examples])
print(f"{len(examples)} examples  |  positives(halluc)={int(y.sum())}  negatives={int((1-y).sum())}")

# ---------------- model ----------------
print(f"Loading {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
kw = dict(output_hidden_states=True, device_map="auto")
if LOAD_8BIT:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)  # modern API
else:
    kw["torch_dtype"] = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
model.eval()
DEV = next(model.parameters()).device
n_layers_total = model.config.num_hidden_layers
if LAYERS is None:
    lo = max(1, n_layers_total // 2 - 4); LAYERS = range(lo, min(n_layers_total, lo + 16))
LAYERS = list(LAYERS)
print(f"model layers={n_layers_total}, probing layers {LAYERS[0]}..{LAYERS[-1]}")

# ---------------- white-box activations + logprob baseline -----------------
@torch.no_grad()
def features(q, a):
    """Return (per-layer last-token hidden states [L, H], answer mean logprob)."""
    prompt = f"Question: {q}\nAnswer:"
    full = prompt + " " + a
    p_ids = tok(prompt, return_tensors="pt").input_ids
    f_ids = tok(full, return_tensors="pt").input_ids.to(DEV)
    out = model(f_ids, output_hidden_states=True)
    hs = out.hidden_states  # tuple (n_layers+1) of [1, seq, H]
    last = f_ids.shape[1] - 1
    vecs = np.stack([hs[li][0, last].float().cpu().numpy() for li in LAYERS])  # [L, H]
    # answer mean log-prob (teacher forcing over the answer tokens)
    logits = out.logits[0].float()                       # [seq, vocab]
    logp = torch.log_softmax(logits, dim=-1)
    ans_start = p_ids.shape[1]
    tok_lp = [logp[t - 1, f_ids[0, t]].item() for t in range(ans_start, f_ids.shape[1])]
    mean_lp = float(np.mean(tok_lp)) if tok_lp else -20.0
    return vecs, mean_lp

print("Extracting activations (this is the GPU-heavy part) ...")
X_layers, logprobs = [], []
for i, (q, a, _) in enumerate(examples):
    v, lp = features(q, a)
    X_layers.append(v); logprobs.append(lp)
    if (i + 1) % 100 == 0: print(f"  {i+1}/{len(examples)}")
X_layers = np.stack(X_layers)            # [N, L, H]
logprobs = np.array(logprobs)            # [N]  (lower = less likely)

# ---------------- evaluation helpers ----------------
def cv_scores(Xf, y, seed=SEED):
    """Out-of-fold probe scores via 5-fold CV (honest, no train/test leak)."""
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(Xf, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Xf[tr], y[tr]); oof[te] = clf.predict_proba(Xf[te])[:, 1]
    return oof

def boot_ci(y, s, n=1000, seed=SEED):
    r = np.random.default_rng(seed); aucs = []
    for _ in range(n):
        idx = r.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2: continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

# per-layer white-box AUROC (the thesis curve)
print("\nPer-layer white-box probe AUROC:")
layer_auc = []
for j, li in enumerate(LAYERS):
    s = cv_scores(X_layers[:, j, :], y)
    a = roc_auc_score(y, s); layer_auc.append(a)
    print(f"  layer {li:2d}: AUROC {a:.3f}")
best_j = int(np.argmax(layer_auc))
wb_scores = cv_scores(X_layers[:, best_j, :], y)        # best-layer probe
print(f"  -> best layer = {LAYERS[best_j]} (AUROC {layer_auc[best_j]:.3f})")

# black-box-ish baseline: higher hallucination risk when answer is less likely
bb_scores = (-logprobs)
bb_scores = (bb_scores - bb_scores.min()) / (bb_scores.ptp() + 1e-9)

# fusion of [white-box probe score, baseline]
Xf = np.column_stack([wb_scores, bb_scores])
fused = cv_scores(Xf, y)

# ---------------- leaderboard ----------------
rows = [
    ("white-box probe (best layer)", roc_auc_score(y, wb_scores), boot_ci(y, wb_scores)),
    ("answer-logprob baseline",       roc_auc_score(y, bb_scores), boot_ci(y, bb_scores)),
    ("FUSION (WB + baseline)",        roc_auc_score(y, fused),     boot_ci(y, fused)),
]
rows.sort(key=lambda r: r[1], reverse=True)
print("\n================ trust-probe REAL leaderboard ================")
print(f"model={MODEL_ID}  N={len(y)}  dataset=TruthfulQA")
print(f"{'Detector':<32} {'AUROC':>7}   95% CI")
for name, auc, (lo, hi) in rows:
    print(f"{name:<32} {auc:>7.3f}   [{lo:.3f}, {hi:.3f}]")
best = rows[0][0]
print(f"\nTop detector: {best}")
print("If FUSION is top, the white-box+black-box thesis holds on real data.")

# per-layer plot
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(LAYERS, layer_auc, marker="o")
    plt.xlabel("layer"); plt.ylabel("AUROC"); plt.title("White-box probe AUROC by layer")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig("layer_auroc.png", dpi=120)
    print("\nSaved per-layer plot -> layer_auroc.png")
except Exception as e:
    print("plot skipped:", e)
