"""Plot the IDEP entanglement curve: identity recoverability vs engagement κ vs k."""
import os, csv
import matplotlib.pyplot as plt
import matplotlib as mpl

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "results", "idep", "idep_curve.csv")
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                     "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})

rows = list(csv.DictReader(open(CSV)))
encs = sorted({r["encoder"] for r in rows})

fig, axes = plt.subplots(1, len(encs), figsize=(6.0 * len(encs), 3.6), sharey=False)
if len(encs) == 1:
    axes = [axes]

for ax, enc in zip(axes, encs):
    rs = sorted([r for r in rows if r["encoder"] == enc], key=lambda r: int(r["k"]))
    ks = [int(r["k"]) for r in rs]
    id_acc = [float(r["id_acc"]) for r in rs]
    lr_kq = [float(r["lr_kq"]) for r in rs]
    lr_lo = [float(r["lr_ci_lo"]) for r in rs]
    lr_hi = [float(r["lr_ci_hi"]) for r in rs]
    rd_kq = [float(r["rd_kq"]) for r in rs]
    rd_lo = [float(r["rd_ci_lo"]) for r in rs]
    rd_hi = [float(r["rd_ci_hi"]) for r in rs]

    ax2 = ax.twinx()

    line_id = ax.plot(ks, id_acc, "-o", color="#cc3333", lw=2, ms=5, label="Subject-ID acc")[0]
    ax.set_xlabel("$k$ — top identity directions removed")
    ax.set_ylabel("Subject-ID accuracy", color="#cc3333")
    ax.tick_params(axis="y", labelcolor="#cc3333")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0/69, color="#cc3333", ls=":", lw=0.7, alpha=0.6)

    line_lr = ax2.plot(ks, lr_kq, "-s", color="#1f77b4", lw=2, ms=5, label="Engagement κ_q (LogReg)")[0]
    ax2.fill_between(ks, lr_lo, lr_hi, color="#1f77b4", alpha=0.15)
    line_rd = ax2.plot(ks, rd_kq, "-^", color="#2ca02c", lw=2, ms=5, label="Engagement κ_q (Ridge)")[0]
    ax2.fill_between(ks, rd_lo, rd_hi, color="#2ca02c", alpha=0.15)
    ax2.set_ylabel("Engagement κ (quadratic)", color="#1f4f7d")
    ax2.tick_params(axis="y", labelcolor="#1f4f7d")
    ax2.set_ylim(-0.04, max(max(lr_hi), max(rd_hi)) + 0.03)
    ax2.axhline(0, color="grey", lw=0.5, ls=":")

    ax.set_title(enc.replace("_", " "))
    ax.set_xscale("symlog", linthresh=2)

    # Combine legends
    lines = [line_id, line_lr, line_rd]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="lower left", framealpha=0.85)

fig.suptitle("The Identity-Engagement Entanglement: ID is high-rank redundant; engagement is low-rank fragile",
             fontsize=11, y=1.02)
fig.tight_layout()
out_pdf = os.path.join(OUT, "fig_idep_curve.pdf")
out_png = os.path.join(OUT, "fig_idep_curve.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Wrote {out_pdf}\nWrote {out_png}")
