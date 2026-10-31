"""scripts/plot_e2.py: E2 figures from results/e2_summary.csv only.

usage:  python scripts/plot_e2.py [--results results] [--alpha 0.05]
writes: results/figures/e2_anywrong_vs_mult.png   any-wrong (pooled over seeds x gamma, Clopper-Pearson 95%) vs assumed-delta multiplier, per arm
        results/figures/e2_power_by_gamma.png     disparity power (vs noise-free flags at the assumed delta) at mult = 1, Holm vs Shaffer vs Bonferroni vs uncorrected
Pooled units share cells, so the intervals are descriptive. Zero counts are drawn at the floor of the log axis and labelled.
"""
import argparse, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from cp_interval import clopper_pearson  # noqa: E402

ARMS = ["unc", "bonf2K", "holm2K", "shaffer"]
COL = {"unc": "#c0392b", "bonf2K": "#7b5ea7", "holm2K": "#1b6ca8", "shaffer": "#e08a1e"}
OFF = {"unc": -0.045, "bonf2K": -0.015, "holm2K": 0.015, "shaffer": 0.045}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    res = Path(a.results); out = res / "figures"; out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(res / "e2_summary.csv")
    Ns, taus, mults = sorted(s.N.unique()), sorted(s.tau.unique(), reverse=True), sorted(s.mult.unique())
    floor = 3e-5

    fig, axs = plt.subplots(len(taus), len(Ns), figsize=(5.2 * len(Ns), 3.9 * len(taus)), sharey=True, squeeze=False)
    for i, t in enumerate(taus):
        for j, N in enumerate(Ns):
            ax = axs[i, j]
            for arm in ARMS:
                xs, ys, lo, hi = [], [], [], []
                for m in mults:
                    g = s[(s.N == N) & (s.tau == t) & (s.arm == arm) & (s.mult == m)]
                    k, n = int(g.k_any.sum()), int(g.R.sum()); l, h = clopper_pearson(k, n)
                    xs.append(m + OFF[arm]); ys.append(max(k / n, floor)); lo.append(max(k / n - l, 0)); hi.append(h - k / n)
                yerr = np.array([np.minimum(lo, np.array(ys) - 1e-9), hi])
                ax.errorbar(xs, ys, yerr=yerr, fmt="-o", ms=4, lw=1.2, capsize=2, color=COL[arm], label=arm)
            ax.axhline(a.alpha, color="k", ls="--", lw=1); ax.axvline(1.0, color="grey", ls=":", lw=1)
            ax.set_yscale("log"); ax.set_ylim(floor * 0.7, 0.2); ax.set_xticks(mults)
            ax.set_title(f"N = {N:,},  tau = {t}", fontsize=10)
            if j == 0: ax.set_ylabel("P(any wrong certificate)")
            if i == len(taus) - 1: ax.set_xlabel("assumed delta / oracle delta")
    axs[0, 0].text(0.52, a.alpha * 1.15, f"alpha = {a.alpha}", fontsize=8)
    h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("E2: family-level error, pooled over seeds and gamma (Clopper-Pearson 95%); points at the floor = zero wrong certificates", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(out / "e2_anywrong_vs_mult.png", dpi=200); plt.close(fig)

    gam = sorted(s.gamma.unique())
    fig, axs = plt.subplots(1, len(Ns), figsize=(5.4 * len(Ns), 3.9), sharey=True, squeeze=False)
    for j, N in enumerate(Ns):
        ax = axs[0, j]; w = 0.2
        for q, arm in enumerate(ARMS):
            y = [s[(s.N == N) & (s.tau == 0.05) & (s.arm == arm) & (s.mult == 1.0) & (s.gamma == g)].power_disp.mean() for g in gam]
            ax.bar(np.arange(len(gam)) + (q - 1.5) * w, y, w, color=COL[arm], label=arm)
        ax.set_xticks(range(len(gam)), [f"{g:g}" for g in gam]); ax.set_xlabel("gamma"); ax.set_title(f"N = {N:,},  tau = 0.05,  mult = 1", fontsize=10)
        if j == 0: ax.set_ylabel("disparity power (mean over seeds)")
    axs[0, -1].legend(fontsize=8, frameon=False)
    fig.suptitle("Power is small: correction costs little only because little is certifiable at these N", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out / "e2_power_by_gamma.png", dpi=200); plt.close(fig)
    print(f"figures saved to {out}")


if __name__ == "__main__":
    main()