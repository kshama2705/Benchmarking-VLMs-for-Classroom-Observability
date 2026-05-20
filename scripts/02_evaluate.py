"""
Step 2: Unified evaluation script for all models.

Input: CSV file(s) with columns: clip_id, predicted, ground_truth
  - For single run: one CSV
  - For self-consistency: 3 CSVs (run1, run2, run3)
  - For prompt sensitivity: CSVs for each prompt variant

Usage:
  # Single run evaluation
  python 02_evaluate.py --predictions results/gpt4o_prompt1_run1.csv

  # Self-consistency (3 runs of same prompt)
  python 02_evaluate.py --consistency results/gpt4o_prompt1_run1.csv results/gpt4o_prompt1_run2.csv results/gpt4o_prompt1_run3.csv

  # Prompt sensitivity (best run from each prompt)
  python 02_evaluate.py --prompt-sensitivity results/gpt4o_prompt1_run1.csv results/gpt4o_prompt2_run1.csv results/gpt4o_prompt3_run1.csv

  # Full evaluation (all at once)
  python 02_evaluate.py --full --model gpt4o --results-dir results/
"""

import argparse
import csv
import os
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from collections import Counter


def load_predictions(csv_path):
    """Load predictions CSV. Returns list of (clip_id, predicted, ground_truth)."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "clip_id": row["clip_id"],
                "predicted": int(row["predicted"]),
                "ground_truth": int(row["ground_truth"]),
            })
    return rows


def evaluate_single(rows, label=""):
    """Compute MSE, accuracy, kappa, per-class F1 for a single run."""
    y_true = [r["ground_truth"] for r in rows]
    y_pred = [r["predicted"] for r in rows]

    mse = mean_squared_error(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    results = {
        "MSE": mse,
        "Accuracy": acc,
        "Cohen_Kappa_Quadratic": kappa,
        "F1_Macro": f1_macro,
        "F1_per_class": {i: float(f1_per_class[i]) for i in range(4)},
        "Confusion_Matrix": cm.tolist(),
        "N": len(rows),
    }

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  N = {len(rows)}")
    print(f"  MSE:                {mse:.4f}")
    print(f"  Accuracy (4-class): {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Cohen's Kappa (QW): {kappa:.4f}")
    print(f"  F1 Macro:           {f1_macro:.4f}")
    print(f"  F1 per class:       0={f1_per_class[0]:.3f}  1={f1_per_class[1]:.3f}  2={f1_per_class[2]:.3f}  3={f1_per_class[3]:.3f}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"         Pred 0  Pred 1  Pred 2  Pred 3")
    for i in range(4):
        row_str = "  ".join(f"{cm[i][j]:6d}" for j in range(4))
        print(f"  True {i}  {row_str}")

    # Distribution analysis
    true_dist = Counter(y_true)
    pred_dist = Counter(y_pred)
    print(f"\n  Label distribution:")
    print(f"    True:      {dict(sorted(true_dist.items()))}")
    print(f"    Predicted: {dict(sorted(pred_dist.items()))}")

    return results


def evaluate_consistency(csv_paths):
    """Compute self-consistency across multiple runs."""
    all_runs = [load_predictions(p) for p in csv_paths]
    n_runs = len(all_runs)

    # Verify same clip_ids across runs
    ids_0 = set(r["clip_id"] for r in all_runs[0])
    for i, run in enumerate(all_runs[1:], 1):
        ids_i = set(r["clip_id"] for r in run)
        if ids_0 != ids_i:
            print(f"WARNING: Run {i} has different clip_ids than run 0")

    # Build prediction matrix: clip_id -> [pred_run1, pred_run2, ...]
    pred_by_clip = {}
    for run_idx, run in enumerate(all_runs):
        for row in run:
            cid = row["clip_id"]
            if cid not in pred_by_clip:
                pred_by_clip[cid] = []
            pred_by_clip[cid].append(row["predicted"])

    # Compute agreement
    full_agreement = 0
    total = 0
    pairwise_agreements = []

    for cid, preds in pred_by_clip.items():
        if len(preds) == n_runs:
            total += 1
            if len(set(preds)) == 1:
                full_agreement += 1
            # Pairwise agreement
            for i in range(n_runs):
                for j in range(i + 1, n_runs):
                    pairwise_agreements.append(1 if preds[i] == preds[j] else 0)

    full_rate = full_agreement / total if total > 0 else 0
    pairwise_rate = np.mean(pairwise_agreements) if pairwise_agreements else 0

    print(f"\n{'='*60}")
    print(f"  Self-Consistency ({n_runs} runs)")
    print(f"{'='*60}")
    print(f"  Clips evaluated:     {total}")
    print(f"  Full agreement:      {full_agreement}/{total} ({full_rate*100:.1f}%)")
    print(f"  Pairwise agreement:  {pairwise_rate*100:.1f}%")

    # Also evaluate each run individually
    for i, (run, path) in enumerate(zip(all_runs, csv_paths)):
        evaluate_single(run, label=f"Run {i+1}: {os.path.basename(path)}")

    return {
        "full_agreement_rate": full_rate,
        "pairwise_agreement_rate": pairwise_rate,
        "n_runs": n_runs,
        "n_clips": total,
    }


def evaluate_prompt_sensitivity(csv_paths):
    """Compare results across different prompt variants."""
    all_prompts = [load_predictions(p) for p in csv_paths]

    print(f"\n{'='*60}")
    print(f"  Prompt Sensitivity ({len(csv_paths)} variants)")
    print(f"{'='*60}")

    metrics_per_prompt = []
    for i, (prompt_data, path) in enumerate(zip(all_prompts, csv_paths)):
        y_true = [r["ground_truth"] for r in prompt_data]
        y_pred = [r["predicted"] for r in prompt_data]
        mse = mean_squared_error(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        metrics_per_prompt.append({"mse": mse, "acc": acc, "kappa": kappa})
        print(f"  Prompt {i+1} ({os.path.basename(path)}):")
        print(f"    MSE={mse:.4f}  Acc={acc*100:.1f}%  Kappa={kappa:.4f}")

    # Compute sensitivity (max - min)
    mse_vals = [m["mse"] for m in metrics_per_prompt]
    acc_vals = [m["acc"] for m in metrics_per_prompt]
    kappa_vals = [m["kappa"] for m in metrics_per_prompt]

    print(f"\n  Sensitivity (max - min):")
    print(f"    MSE:   {max(mse_vals) - min(mse_vals):.4f}")
    print(f"    Acc:   {(max(acc_vals) - min(acc_vals))*100:.1f}%")
    print(f"    Kappa: {max(kappa_vals) - min(kappa_vals):.4f}")

    return {
        "per_prompt": metrics_per_prompt,
        "sensitivity_mse": max(mse_vals) - min(mse_vals),
        "sensitivity_acc": max(acc_vals) - min(acc_vals),
        "sensitivity_kappa": max(kappa_vals) - min(kappa_vals),
    }


def full_evaluation(model_name, results_dir):
    """Run full evaluation for a model: find all its CSV files and compute everything."""
    # Expected file naming: {model}_prompt{N}_run{M}.csv
    import glob

    pattern = os.path.join(results_dir, f"{model_name}_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching {pattern}")
        return

    print(f"\nFound {len(files)} result files for {model_name}:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    # Group by prompt
    by_prompt = {}
    for f in files:
        base = os.path.basename(f)
        # Extract prompt number
        for pn in ["prompt1", "prompt2", "prompt3"]:
            if pn in base:
                by_prompt.setdefault(pn, []).append(f)
                break

    # Evaluate best run per prompt
    print(f"\n\n{'#'*60}")
    print(f"  FULL EVALUATION: {model_name.upper()}")
    print(f"{'#'*60}")

    # 1. Individual runs
    for prompt_name in sorted(by_prompt.keys()):
        for run_file in by_prompt[prompt_name]:
            rows = load_predictions(run_file)
            evaluate_single(rows, label=f"{model_name} / {prompt_name} / {os.path.basename(run_file)}")

    # 2. Self-consistency per prompt
    for prompt_name in sorted(by_prompt.keys()):
        if len(by_prompt[prompt_name]) >= 2:
            print(f"\n--- Consistency for {prompt_name} ---")
            evaluate_consistency(by_prompt[prompt_name])

    # 3. Prompt sensitivity (first run of each prompt)
    first_runs = [by_prompt[p][0] for p in sorted(by_prompt.keys()) if by_prompt[p]]
    if len(first_runs) >= 2:
        evaluate_prompt_sensitivity(first_runs)


def main():
    parser = argparse.ArgumentParser(description="DAiSEE VLM Evaluation")
    parser.add_argument("--predictions", type=str, help="Single predictions CSV")
    parser.add_argument("--consistency", nargs="+", help="Multiple run CSVs for self-consistency")
    parser.add_argument("--prompt-sensitivity", nargs="+", help="CSVs for different prompts")
    parser.add_argument("--full", action="store_true", help="Full evaluation for a model")
    parser.add_argument("--model", type=str, help="Model name (used with --full)")
    parser.add_argument("--results-dir", type=str, default="results/", help="Results directory")
    args = parser.parse_args()

    if args.predictions:
        rows = load_predictions(args.predictions)
        evaluate_single(rows, label=os.path.basename(args.predictions))
    elif args.consistency:
        evaluate_consistency(args.consistency)
    elif args.prompt_sensitivity:
        evaluate_prompt_sensitivity(args.prompt_sensitivity)
    elif args.full and args.model:
        full_evaluation(args.model, args.results_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
