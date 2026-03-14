"""
Step 2: Evaluate prediction CSVs for DAiSEE engagement classification.

Metrics: Accuracy, F1 Macro, Cohen's kappa (quadratic weighted), MSE

Usage:
  # Single run
  python scripts/02_evaluate.py --predictions results/qwen25vl_prompt1_run1.csv

  # Prompt sensitivity (all 3 prompts)
  python scripts/02_evaluate.py --prompt-sensitivity \\
    results/qwen25vl_prompt1_run1.csv \\
    results/qwen25vl_prompt2_run1.csv \\
    results/qwen25vl_prompt3_run1.csv

  # Self-consistency (3 runs of same prompt)
  python scripts/02_evaluate.py --consistency \\
    results/qwen25vl_prompt2_run1.csv \\
    results/qwen25vl_prompt2_run2.csv \\
    results/qwen25vl_prompt2_run3.csv

  # Full evaluation for a model (all prompts + consistency)
  python scripts/02_evaluate.py --full --model qwen25vl --results-dir results/
"""

import argparse
import csv
import glob
import os
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, mean_squared_error
)


def load_csv(path):
    preds, gts = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            preds.append(int(row["predicted"]))
            gts.append(int(row["ground_truth"]))
    return np.array(preds), np.array(gts)


def compute_metrics(preds, gts):
    acc   = accuracy_score(gts, preds)
    f1    = f1_score(gts, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(gts, preds, weights="quadratic")
    mse   = mean_squared_error(gts, preds)
    return {"accuracy": acc, "f1_macro": f1, "kappa_qw": kappa, "mse": mse}


def print_metrics(label, metrics):
    print(f"  {label}")
    print(f"    Accuracy : {metrics['accuracy']:.3f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"    F1 Macro : {metrics['f1_macro']:.3f}")
    print(f"    Kappa QW : {metrics['kappa_qw']:.3f}")
    print(f"    MSE      : {metrics['mse']:.3f}")


def print_distribution(preds, gts, label=""):
    if label:
        print(f"  {label}")
    dist_pred = Counter(preds.tolist())
    dist_gt   = Counter(gts.tolist())
    print(f"    Predicted : { {k: dist_pred.get(k,0) for k in range(4)} }")
    print(f"    GT        : { {k: dist_gt.get(k,0)   for k in range(4)} }")


def majority_vote(preds_list):
    """Column-wise majority vote across runs."""
    stacked = np.stack(preds_list, axis=1)
    result = []
    for row in stacked:
        counts = Counter(row.tolist())
        result.append(counts.most_common(1)[0][0])
    return np.array(result)


def evaluate_single(path):
    preds, gts = load_csv(path)
    m = compute_metrics(preds, gts)
    print(f"\n{'='*60}")
    print(f"File: {os.path.basename(path)}")
    print_metrics("Results", m)
    print_distribution(preds, gts)
    return m


def evaluate_prompt_sensitivity(paths):
    print(f"\n{'='*60}")
    print("PROMPT SENSITIVITY")
    print(f"{'='*60}")
    all_metrics = []
    for p in paths:
        preds, gts = load_csv(p)
        m = compute_metrics(preds, gts)
        name = os.path.basename(p)
        print_metrics(name, m)
        print_distribution(preds, gts)
        print()
        all_metrics.append((name, m))

    # Summary table
    print("\nSummary table:")
    print(f"{'File':<40} {'Acc':>6} {'F1':>6} {'Kappa':>7} {'MSE':>6}")
    print("-" * 70)
    for name, m in all_metrics:
        print(f"{name:<40} {m['accuracy']*100:5.1f}% {m['f1_macro']:6.3f} {m['kappa_qw']:7.3f} {m['mse']:6.3f}")


def evaluate_consistency(paths):
    print(f"\n{'='*60}")
    print("SELF-CONSISTENCY")
    print(f"{'='*60}")
    all_preds, gts = [], None
    for p in paths:
        preds, gt = load_csv(p)
        all_preds.append(preds)
        if gts is None:
            gts = gt
        m = compute_metrics(preds, gts)
        print_metrics(os.path.basename(p), m)

    voted = majority_vote(all_preds)
    m_voted = compute_metrics(voted, gts)
    print()
    print_metrics("Majority vote across runs", m_voted)


def evaluate_full(model, results_dir):
    print(f"\n{'='*60}")
    print(f"FULL EVALUATION: {model}")
    print(f"{'='*60}")

    # All prompts, run 1
    prompt_files = []
    for p in [1, 2, 3]:
        path = os.path.join(results_dir, f"{model}_prompt{p}_run1.csv")
        if os.path.exists(path):
            prompt_files.append(path)
        else:
            print(f"  [MISSING] {path}")

    if prompt_files:
        evaluate_prompt_sensitivity(prompt_files)

    # Self-consistency: find prompts with 3 runs
    for p in [1, 2, 3]:
        run_files = []
        for r in [1, 2, 3]:
            path = os.path.join(results_dir, f"{model}_prompt{p}_run{r}.csv")
            if os.path.exists(path):
                run_files.append(path)
        if len(run_files) == 3:
            print(f"\n--- Self-consistency: prompt {p} ---")
            evaluate_consistency(run_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="+", help="One or more prediction CSVs")
    ap.add_argument("--prompt-sensitivity", nargs=3, metavar="CSV",
                    help="3 CSVs (one per prompt variant)")
    ap.add_argument("--consistency", nargs="+", metavar="CSV",
                    help="CSVs for same prompt, different runs")
    ap.add_argument("--full", action="store_true",
                    help="Full evaluation for --model in --results-dir")
    ap.add_argument("--model", help="Model slug (e.g. qwen25vl)")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    if args.full:
        if not args.model:
            ap.error("--full requires --model")
        evaluate_full(args.model, args.results_dir)

    elif args.prompt_sensitivity:
        evaluate_prompt_sensitivity(args.prompt_sensitivity)

    elif args.consistency:
        evaluate_consistency(args.consistency)

    elif args.predictions:
        for p in args.predictions:
            evaluate_single(p)

    else:
        ap.print_help()


if __name__ == "__main__":
    main()
