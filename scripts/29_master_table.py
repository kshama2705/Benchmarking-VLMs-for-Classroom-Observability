"""
Master results aggregator. Pulls all completed experiments into a single table.
Reads:
  results/full/clip_full_metrics.json       (A1)
  results/full/calibration_results.json     (A6)
  results/full/gpt4o_full_metrics.json      (A3)
  results/full/llava_full_p{1,2,3}_run1.csv (A2)
  results/linear_probe/probe_results.json          (single-frame CLIP probe)
  results/linear_probe/probe_results_multiframe.json (multi-frame CLIP probe)
  results/linear_probe/probe_results_dinov2.json    (DINOv2 probe)
  results/idep/idep_finegrained.json   (N4)
  results/idep/idep_leace_results.json (N3)
  results/siep/siep_results.json       (N5)
  results/siep/siep_multiseed.json     (N6)
  results/siep/siep_v2_results.json    (N7, if exists)
"""
import os, json, csv, glob
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_squared_error

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(42)

def boot_kq(yt, yp, n=1000):
    out = []
    nt = len(yt)
    for _ in range(n):
        idx = RNG.integers(0, nt, size=nt)
        out.append(cohen_kappa_score(yt[idx], yp[idx], weights="quadratic"))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def csv_metrics(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    keep = [r for r in rows if r.get("predicted", "").strip() not in ("", "None")]
    if not keep:
        return None
    yt = np.array([int(r["ground_truth"]) for r in keep])
    yp = np.array([int(r["predicted"]) for r in keep])
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa_quadratic": float(cohen_kappa_score(yt, yp, weights="quadratic")),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mse": float(mean_squared_error(yt, yp)),
        "ci": boot_kq(yt, yp),
    }


def fmt(point, ci=None):
    if ci is not None:
        return f"{point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{point:.3f}"


def section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def main():
    rows = []  # (category, method, encoder/model, prompt/cfg, kq, ci, acc, n)

    # ----- Zero-shot full test set -----
    section("ZERO-SHOT (full 1,784-clip test)")
    # CLIP from clip_full_metrics.json
    p = os.path.join(BASE, "results", "full", "clip_full_metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        for k, m in d.items():
            ci = m.get("ci95", {}).get("kappa_quadratic", None) if "ci95" in m else None
            print(f"  CLIP zero-shot {k:4} κ_q={m['kappa_quadratic']:.3f}"
                  f"{' [' + f'{ci[0]:.3f}, {ci[1]:.3f}' + ']' if ci else ''}"
                  f"  acc={m['accuracy']:.3f}  pred_dist={m['pred_dist']}")
            rows.append(("zero-shot", "CLIP", k, m["kappa_quadratic"], ci, m["accuracy"]))

    # GPT-4o
    p = os.path.join(BASE, "results", "full", "gpt4o_full_metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        for k, m in d.items():
            kept = m.get("metrics_excluding_refusals", {})
            full = m.get("metrics_with_L2_fallback", {})
            r_rate = m.get("refusal_rate", 0)
            print(f"  GPT-4o   {k:4} refusal={r_rate*100:.1f}%  "
                  f"κ_q (kept)={kept.get('kappa_quadratic', float('nan')):.3f} (n={kept.get('n', 0)})  "
                  f"κ_q (L2-fallback)={full.get('kappa_quadratic', float('nan')):.3f}")
            rows.append(("zero-shot", "GPT-4o", k,
                         kept.get("kappa_quadratic"), kept.get("kappa_q_ci95"),
                         kept.get("accuracy")))

    # LLaVA
    for pid in [1, 2, 3]:
        p = os.path.join(BASE, "results", "full", f"llava_full_p{pid}_run1.csv")
        m = csv_metrics(p)
        if m:
            print(f"  LLaVA    P{pid}   κ_q={fmt(m['kappa_quadratic'], m['ci'])}  "
                  f"acc={m['accuracy']:.3f}  n={m['n']}")
            rows.append(("zero-shot", "LLaVA", f"P{pid}", m["kappa_quadratic"], m["ci"], m["accuracy"]))

    # ----- Linear probes -----
    section("LINEAR PROBES (subject-disjoint, full test)")
    for label, p in [("CLIP single-frame", "results/linear_probe/probe_results.json"),
                     ("CLIP multi-frame",  "results/linear_probe/probe_results_multiframe.json"),
                     ("DINOv2 single",     "results/linear_probe/probe_results_dinov2.json")]:
        full = os.path.join(BASE, p)
        if not os.path.exists(full):
            continue
        with open(full) as f:
            d = json.load(f)
        for kk in ["logreg", "ridge_ordinal"]:
            if kk in d:
                m = d[kk]["test_metrics"]
                ci = d[kk].get("test_metrics_ci", {}).get("kappa_quadratic_ci95", None)
                print(f"  {label:20} {kk:14}  κ_q={fmt(m['kappa_quadratic'], ci)}  "
                      f"acc={m['accuracy']:.3f}")

    # ----- Calibration -----
    section("CALIBRATION (CLIP)")
    p = os.path.join(BASE, "results", "full", "calibration_results.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        for prompt, vv in d.items():
            for method in ["uncalibrated", "class_prior_adjusted", "calibrate_before_use", "both"]:
                if method in vv:
                    m = vv[method]
                    ci = m.get("kappa_q_ci95", None)
                    print(f"  CLIP {prompt} {method:24}  κ_q={fmt(m['kappa_quadratic'], ci)}  acc={m['accuracy']:.3f}")

    # ----- IDEP fine-grained best -----
    section("IDEP — linear residualization (N4 best per encoder)")
    p = os.path.join(BASE, "results", "idep", "idep_finegrained.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        for enc, recs in d.items():
            # Find best LR and best Ridge across k
            best_lr = max(recs, key=lambda r: r["lr_kq"])
            best_rd = max(recs, key=lambda r: r["rd_kq"])
            print(f"  {enc:20} best_LR    k={best_lr['k']:3d}  κ_q={best_lr['lr_kq']:.3f} "
                  f"[{best_lr['lr_ci_lo']:.3f}, {best_lr['lr_ci_hi']:.3f}]  id_acc={best_lr['id_acc']:.3f}")
            print(f"  {enc:20} best_Ridge k={best_rd['k']:3d}  κ_q={best_rd['rd_kq']:.3f} "
                  f"[{best_rd['rd_ci_lo']:.3f}, {best_rd['rd_ci_hi']:.3f}]  id_acc={best_rd['id_acc']:.3f}")

    # ----- LEACE -----
    section("IDEP — LEACE concept erasure (N3)")
    p = os.path.join(BASE, "results", "idep", "idep_leace_results.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        for enc, vv in d.items():
            base_lr = vv["baseline"]["logreg"]
            leace_lr = vv["leace"]["logreg"]
            base_rd = vv["baseline"]["ridge_ordinal"]
            leace_rd = vv["leace"]["ridge_ordinal"]
            id_after = vv.get("subject_id_after_leace", None)
            print(f"  {enc:20} baseline LR     κ_q={base_lr['kappa_quadratic']:.3f}")
            print(f"  {enc:20} LEACE    LR     κ_q={leace_lr['kappa_quadratic']:.3f}  id_acc_after={id_after:.3f}")
            print(f"  {enc:20} baseline Ridge  κ_q={base_rd['kappa_quadratic']:.3f}")
            print(f"  {enc:20} LEACE    Ridge  κ_q={leace_rd['kappa_quadratic']:.3f}")

    # ----- SIEP v1 single-seed -----
    section("SIEP v1 (single-seed sweep, N5)")
    p = os.path.join(BASE, "results", "siep", "siep_results.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        for enc, runs in d.items():
            for r in runs:
                cfg = r["config"]
                kq = r["kappa_quadratic"]
                ci = r.get("kappa_q_ci95", None)
                tag = f"d={cfg['d_hidden']}, λ={cfg['lambda_max']}, layers={cfg['n_layers']}, drop={cfg['dropout']}"
                print(f"  {enc:20} {tag:50}  κ_q={fmt(kq, ci)}  acc={r['accuracy']:.3f}")

    # ----- SIEP v1 multi-seed -----
    section("SIEP v1 (multi-seed validation, N6)")
    p = os.path.join(BASE, "results", "siep", "siep_multiseed.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        for s in d:
            cfg = s["config"]
            tag = f"d={cfg['d_hidden']}, λ={cfg['lambda_max']}"
            print(f"  {s['encoder']:20} {tag:25}  mean κ_q={s['mean_kq']:.3f} ± {s['std_kq']:.3f}  "
                  f"(min={s['min_kq']:.3f}, max={s['max_kq']:.3f})")

    # ----- SIEP v2 -----
    section("SIEP v2 (stabilized + contrastive, N7) — if available")
    p = os.path.join(BASE, "results", "siep", "siep_v2_results.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        for s in d:
            cfg = s["config"]
            tag = f"mode={cfg['mode']}, λ={cfg.get('lambda_max',0)}, cw={cfg.get('contrastive_weight',0)}"
            print(f"  {s['encoder']:20} {tag:55}  mean κ_q={s['mean_kq']:.3f} ± {s['std_kq']:.3f}  "
                  f"(min={s['min_kq']:.3f}, max={s['max_kq']:.3f})")
    else:
        print("  (still running)")


if __name__ == "__main__":
    main()
