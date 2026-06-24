"""cli.py — Command-line interface for trust-probe.

Entry point: ``trustprobe``

Commands
--------
trustprobe eval --synthetic
    Run the full evaluation pipeline on synthetic data and print the leaderboard.
    Works completely offline with just numpy + scikit-learn.

trustprobe eval --ragtruth PATH
    Evaluate on a RAGTruth dataset file.

trustprobe eval --halueval PATH
    Evaluate on a HaluEval dataset file.

Exit codes: 0 on success, 1 on error.
"""

from __future__ import annotations

import argparse
import sys


def _run_synthetic_eval() -> int:
    """Full offline pipeline on synthetic data. Returns exit code."""
    from trust_probe.activations import SyntheticActivations
    from trust_probe.evaluate import render_text_table, synthetic_fusion_demo
    from trust_probe.metrics import auroc
    from trust_probe.probes import LogRegProbe

    print("=" * 60)
    print("trust-probe  |  Synthetic Offline Evaluation")
    print("=" * 60)
    print()

    # --- 1. Real white-box probe path (sanity) --------------------------------
    # Train the actual LogRegProbe on synthetic activation tensors and report
    # its held-out AUROC, proving the white-box path runs end-to-end on CPU.
    print("Training LogRegProbe on synthetic activations (white-box path) ...")
    sa = SyntheticActivations(n=240, seed=42)
    n_tr = 180
    probe = LogRegProbe(C=1.0, max_iter=500)
    probe.fit(sa.X[:n_tr], sa.y[:n_tr])
    probe_test = probe.predict_proba(sa.X[n_tr:])[:, 1]
    probe_auroc = auroc(sa.y[n_tr:], probe_test)
    print(f"  white-box probe held-out AUROC: {probe_auroc:.3f}")
    print()

    # --- 2. Fusion-beats-parts demonstration ----------------------------------
    print("Fusing a noisy white-box score with a noisy black-box signal ...")
    reports = synthetic_fusion_demo(n=400, seed=7)

    print()
    print(render_text_table(reports))
    print()
    best = max(reports, key=lambda r: r["auroc"])
    parts = [r for r in reports if r["name"] != "fusion (WB + BB)"]
    fused = next(r for r in reports if r["name"] == "fusion (WB + BB)")
    verdict = "FUSION WINS" if fused is best else "fusion did NOT win (rerun seed)"
    print(
        f"{verdict}: fusion AUROC {fused['auroc']:.3f} vs "
        f"parts {[round(r['auroc'], 3) for r in parts]}"
    )
    print()
    print("GPU-guarded paths (extract_activations, AttentionMLPProbe,")
    print("TinyConvProbeAdapter) were NOT exercised — they need torch + a GPU.")
    print()
    return 0


def _run_file_eval(path: str, format: str) -> int:
    """Evaluate on a file-based dataset. Returns exit code."""
    from trust_probe.baselines import logprob_baseline
    from trust_probe.datasets import load_halueval, load_ragtruth
    from trust_probe.evaluate import compare, render_text_table

    try:
        if format == "ragtruth":
            records = load_ragtruth(path)
        else:
            records = load_halueval(path)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(records)} records from {path}")

    reports = compare(
        {"logprob-proxy": logprob_baseline},
        records,
        n_bootstrap=500,
    )
    print(render_text_table(reports))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the trustprobe CLI."""
    parser = argparse.ArgumentParser(
        prog="trustprobe",
        description="trust-probe: hallucination trust layer evaluation",
    )
    sub = parser.add_subparsers(dest="command")

    # eval sub-command
    eval_parser = sub.add_parser("eval", help="Run evaluation pipeline")
    eval_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run fully offline evaluation on synthetic data",
    )
    eval_parser.add_argument(
        "--ragtruth",
        metavar="PATH",
        help="Path to a RAGTruth dataset file",
    )
    eval_parser.add_argument(
        "--halueval",
        metavar="PATH",
        help="Path to a HaluEval dataset file",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "eval":
        if args.synthetic:
            return _run_synthetic_eval()
        elif args.ragtruth:
            return _run_file_eval(args.ragtruth, "ragtruth")
        elif args.halueval:
            return _run_file_eval(args.halueval, "halueval")
        else:
            print(
                "Error: specify one of --synthetic, --ragtruth PATH, or --halueval PATH",
                file=sys.stderr,
            )
            return 1

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
