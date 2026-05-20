"""
Pull all probe results into one comparison table.
Reads:
  results/linear_probe/probe_results.json             (single-frame CLIP)
  results/linear_probe/probe_results_multiframe.json  (multi-frame CLIP)
  results/linear_probe/probe_results_dinov2.json      (DINOv2 single-frame)
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR_DIR = os.path.join(BASE_DIR, "results", "linear_probe")

FILES = {
    "CLIP single (LogReg)":   ("probe_results.json", "logreg"),
    "CLIP single (Ridge)":    ("probe_results.json", "ridge_ordinal"),
    "CLIP multi-frame (LogReg)": ("probe_results_multiframe.json", "logreg"),
    "CLIP multi-frame (Ridge)":  ("probe_results_multiframe.json", "ridge_ordinal"),
    "DINOv2 (LogReg)":        ("probe_results_dinov2.json", "logreg"),
    "DINOv2 (Ridge)":         ("probe_results_dinov2.json", "ridge_ordinal"),
}


def fmt_ci(point, ci):
    return f"{point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def main():
    print(f"{'Method':32} {'Acc':28} {'κ_quad':28} {'F1 macro':28} {'MSE':28}  pred_dist")
    print("-" * 200)
    for label, (fname, key) in FILES.items():
        path = os.path.join(PR_DIR, fname)
        if not os.path.exists(path):
            print(f"{label:32}  (missing: {fname})")
            continue
        with open(path) as f:
            d = json.load(f)
        if key not in d:
            print(f"{label:32}  (key missing: {key})")
            continue
        m = d[key]["test_metrics"]
        c = d[key]["test_metrics_ci"]
        print(f"{label:32} {fmt_ci(m['accuracy'], c['accuracy_ci95']):28}"
              f" {fmt_ci(m['kappa_quadratic'], c['kappa_quadratic_ci95']):28}"
              f" {fmt_ci(m['f1_macro'], c['f1_macro_ci95']):28}"
              f" {fmt_ci(m['mse'], c['mse_ci95']):28}"
              f"  {m['pred_dist']}")


if __name__ == "__main__":
    main()
