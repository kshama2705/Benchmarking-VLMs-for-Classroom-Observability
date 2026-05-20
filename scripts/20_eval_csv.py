"""Compute metrics + bootstrap CIs for any predicted/ground_truth CSV."""
import sys, csv, json
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

RNG = np.random.default_rng(42)


def metrics(yt, yp):
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa_quadratic": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "kappa_linear": float(cohen_kappa_score(yt, yp, weights="linear")),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mse": float(mean_squared_error(yt, yp)),
        "pred_dist": {int(k): int((yp == k).sum()) for k in range(4)},
        "true_dist": {int(k): int((yt == k).sum()) for k in range(4)},
    }


def boot(yt, yp, n=1000):
    out = {"acc": [], "kq": [], "f1": [], "mse": []}
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        a, b = yt[idx], yp[idx]
        out["acc"].append(accuracy_score(a, b))
        out["kq"].append(cohen_kappa_score(a, b, weights="quadratic"))
        out["f1"].append(f1_score(a, b, average="macro", zero_division=0))
        out["mse"].append(mean_squared_error(a, b))
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
            for k, v in out.items()}


def main():
    paths = sys.argv[1:]
    for p in paths:
        rows = list(csv.DictReader(open(p)))
        # Filter out rows with empty predicted (refusals)
        keep = [r for r in rows if r.get("predicted", "").strip() not in ("", "None")]
        yt = np.array([int(r["ground_truth"]) for r in keep])
        yp = np.array([int(r["predicted"]) for r in keep])
        m = metrics(yt, yp)
        c = boot(yt, yp)
        print(f"\n{p}")
        print(f"  n={m['n']}  acc={m['accuracy']:.3f} [{c['acc'][0]:.3f},{c['acc'][1]:.3f}]"
              f"  κ_q={m['kappa_quadratic']:.3f} [{c['kq'][0]:.3f},{c['kq'][1]:.3f}]"
              f"  F1={m['f1_macro']:.3f}  MSE={m['mse']:.3f}")
        print(f"  pred_dist={m['pred_dist']}  true_dist={m['true_dist']}")


if __name__ == "__main__":
    main()
