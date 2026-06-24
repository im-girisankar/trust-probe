# =====================================================================
# trust-probe — real validation on a GPU (Kaggle or Colab). Copy-paste one cell.
# =====================================================================
# Setup: GPU runtime + internet ON, then run.
#
# RESUMABLE: caches features to `trustprobe_features_v2.npz` after extraction;
# re-running in the same session skips model load + extraction.
#
# Outputs per-layer white-box AUROC + a leaderboard comparing the white-box
# probe, a P(True) self-eval black-box signal, a logprob baseline, and their
# FUSION, each with bootstrap 95% CIs; saves a layer-AUROC plot.
# Labels: TruthfulQA correct (faithful=0) vs incorrect (hallucination=1).
# =====================================================================

import importlib.util
import subprocess, sys
def pip(*pkgs): subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
pip("transformers>=4.46", "accelerate>=0.30", "datasets>=2.19", "scikit-learn>=1.3")
# Keep the environment's CUDA-matched bitsandbytes (Kaggle ships one); installing
# over it causes a CUDA-symbol-mismatch segfault. Only install when missing.
if importlib.util.find_spec("bitsandbytes") is None:
    pip("bitsandbytes")

import os
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- config ----------------
MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"   # thesis model: "meta-llama/Llama-3.1-8B-Instruct" + LAYERS=range(8,24)
LOAD_8BIT = True
N_PAIRS   = 500
LAYERS    = None
SEED      = 7
CACHE     = os.environ.get("TP_CACHE", "trustprobe_features_v2.npz")
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

# ---------------- extract OR resume from cache ----------------
if os.path.exists(CACHE):
    print(f"Resuming from cache {CACHE} (skipping model load + extraction) ...")
    d = np.load(CACHE)
    X_layers, logprobs, ptrue, y = d["X"], d["logprobs"], d["ptrue"], d["y"]
    LAYERS = [int(v) for v in d["LAYERS"]]
    print(f"  {len(y)} examples | layers {LAYERS[0]}..{LAYERS[-1]}")
else:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading TruthfulQA ...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    examples = []
    for row in ds:
        q = row["question"]
        correct = (row.get("correct_answers") or [row.get("best_answer")])
        wrong = row.get("incorrect_answers") or []
        if correct and correct[0]:
            examples.append((q, correct[0], 0))
        if wrong and wrong[0]:
            examples.append((q, wrong[0], 1))
    rng.shuffle(examples)
    examples = examples[: 2 * N_PAIRS]
    y = np.array([e[2] for e in examples])
    print(f"{len(examples)} examples  |  positives={int(y.sum())}  negatives={int((1-y).sum())}")

    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    kw = dict(output_hidden_states=True, device_map="auto")
    if LOAD_8BIT:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model.eval()
    DEV = next(model.parameters()).device
    n_total = model.config.num_hidden_layers
    if LAYERS is None:
        lo = max(1, n_total // 2 - 4); LAYERS = range(lo, min(n_total, lo + 16))
    LAYERS = list(LAYERS)
    print(f"model layers={n_total}, probing layers {LAYERS[0]}..{LAYERS[-1]}")

    def _first_ids(words):
        ids = set()
        for w in words:
            for variant in (w, " " + w):
                t = tok(variant, add_special_tokens=False).input_ids
                if t:
                    ids.add(t[0])
        return list(ids)
    YES_IDS = _first_ids(["Yes", "yes", "YES"])
    NO_IDS = _first_ids(["No", "no", "NO"])

    @torch.no_grad()
    def features(q, a):
        prompt = f"Question: {q}\nAnswer:"
        full = prompt + " " + a
        p_ids = tok(prompt, return_tensors="pt").input_ids
        f_ids = tok(full, return_tensors="pt").input_ids.to(DEV)
        out = model(f_ids, output_hidden_states=True)
        hs = out.hidden_states
        last = f_ids.shape[1] - 1
        vecs = np.stack([hs[li][0, last].float().cpu().numpy() for li in LAYERS])
        logp = torch.log_softmax(out.logits[0].float(), dim=-1)
        ans_start = p_ids.shape[1]
        tok_lp = [logp[t - 1, f_ids[0, t]].item() for t in range(ans_start, f_ids.shape[1])]
        mean_lp = float(np.mean(tok_lp)) if tok_lp else -20.0
        # P(True): ask the model whether its own answer is correct.
        judge = (f"Question: {q}\nProposed answer: {a}\n"
                 "Is the proposed answer factually correct? Answer yes or no.\nAnswer:")
        j_ids = tok(judge, return_tensors="pt").input_ids.to(DEV)
        jp = torch.softmax(model(j_ids).logits[0, -1].float(), dim=-1)
        p_yes = float(jp[YES_IDS].sum()); p_no = float(jp[NO_IDS].sum())
        ptrue_risk = p_no / (p_yes + p_no + 1e-9)  # high -> model judges it wrong -> hallucination
        return vecs, mean_lp, ptrue_risk

    print("Extracting activations + P(True) (GPU-heavy) ...")
    X_layers, logprobs, ptrue = [], [], []
    for i, (q, a, _) in enumerate(examples):
        v, lp, pt = features(q, a)
        X_layers.append(v); logprobs.append(lp); ptrue.append(pt)
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(examples)}")
    X_layers = np.stack(X_layers); logprobs = np.array(logprobs); ptrue = np.array(ptrue)
    np.savez(CACHE, X=X_layers, logprobs=logprobs, ptrue=ptrue, y=y, LAYERS=np.array(LAYERS))
    print(f"Saved feature cache -> {CACHE}")

# ---------------- evaluation (fast CPU; runs cached or fresh) ----------------
def cv_scores(Xf, y, seed=SEED):
    Xf = np.atleast_2d(Xf.T).T if Xf.ndim == 1 else Xf
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xf, y):
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

print("\nPer-layer white-box probe AUROC:")
layer_auc = []
for j, li in enumerate(LAYERS):
    a = roc_auc_score(y, cv_scores(X_layers[:, j, :], y)); layer_auc.append(a)
    print(f"  layer {li:2d}: AUROC {a:.3f}")
best_j = int(np.argmax(layer_auc))
wb = cv_scores(X_layers[:, best_j, :], y)
print(f"  -> best layer = {LAYERS[best_j]} (AUROC {layer_auc[best_j]:.3f})")

# black-box signals (both are hallucination-risk scores, higher = worse)
lp = -np.asarray(logprobs); lp = (lp - lp.min()) / (np.ptp(lp) + 1e-9)
pt = np.asarray(ptrue)
fuse_pt = cv_scores(np.column_stack([wb, pt]), y)
fuse_all = cv_scores(np.column_stack([wb, pt, lp]), y)

board = [
    ("white-box probe (best layer)", wb),
    ("P(True) self-eval", pt),
    ("logprob baseline", lp),
    ("FUSION (WB + P(True))", fuse_pt),
    ("FUSION (WB + P(True) + logprob)", fuse_all),
]
rows = sorted(((n, roc_auc_score(y, s), boot_ci(y, s)) for n, s in board),
              key=lambda r: r[1], reverse=True)
print("\n================ trust-probe REAL leaderboard ================")
print(f"model={MODEL_ID}  N={len(y)}  dataset=TruthfulQA  best_layer={LAYERS[best_j]}")
print(f"{'Detector':<34} {'AUROC':>7}   95% CI")
for name, auc, (lo, hi) in rows:
    print(f"{name:<34} {auc:>7.3f}   [{lo:.3f}, {hi:.3f}]")
print(f"\nTop detector: {rows[0][0]}")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4)); plt.plot(LAYERS, layer_auc, marker="o")
    plt.xlabel("layer"); plt.ylabel("AUROC"); plt.title("White-box probe AUROC by layer")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig("layer_auroc.png", dpi=120)
    print("Saved per-layer plot -> layer_auroc.png")
except Exception as e:
    print("plot skipped:", e)
