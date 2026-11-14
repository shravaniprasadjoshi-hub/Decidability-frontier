"""E1 coverage figures + pass-check table from the per-rep pooled CSV written by b8a_coverage.py.
usage: python plot_e1.py STATE SEED M_MIN [--data-dir DIR] [--out DIR]
reads  {data-dir}/b8a_pooled_{STATE}_{SEED}_m{M_MIN}_R*.csv  (default data-dir = smoke tests/results)
writes {out}/e1_{STATE}_s{SEED}_m{M_MIN}_passcheck.csv, _thinbins.csv and figures per gamma (default out = data-dir; all gitignored)
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"clt_comp": "CLT + composition term", "clt_nocomp": "CLT, no composition term", "exact": "Exact"}
COLORS = dict(zip(ARMS, ["steelblue", "salmon", "seagreen"]))
NG_LAB = ["[0.0, 50.0)", "[50.0, 100.0)", "[100.0, 200.0)", "[200.0, 500.0)", "[500.0, inf)"]
NG_SHOW = ["0-50", "50-100", "100-200", "200-500", "500+"]
NUN_LAB = ["0", "1", "2"]; NUN_SHOW = ["0 (clean)", "1", "2+"]
PASS = 0.94

ap = argparse.ArgumentParser()
ap.add_argument("state"); ap.add_argument("seed", type=int); ap.add_argument("m_min", type=int)
ap.add_argument("--data-dir", default=os.path.join(HERE, "results")); ap.add_argument("--out", default=None)
a = ap.parse_args(); out = a.out or a.data_dir
pat = os.path.join(a.data_dir, f"b8a_pooled_{a.state}_{a.seed}_m{a.m_min}_R*.csv")
files = glob.glob(pat); assert files, f"no file matches {pat}"
tab = pd.read_csv(files[0], dtype={"level": str}); os.makedirs(out, exist_ok=True)
tag = f"e1_{a.state}_s{a.seed}_m{a.m_min}"

# pass check: proposed arm (clt_comp), clean bin (nun=0), both endpoints, every N and gamma
pc = tab[(tab.arm == "clt_comp") & (tab.group == "nun_bin") & (tab.level == "0")]
pc = pc.assign(informative=pc.cell_reps >= 500)   # <500 clean cell-reps: MC-se too large, pass/fail is not evidence
pc.to_csv(os.path.join(out, f"{tag}_passcheck.csv"), index=False)
print(pc[["N", "gamma", "endpoint", "cov", "mc_se", "cell_reps", "passes", "informative"]].round(4).to_string(index=False))

# thin bins at N=20000: n_g 0-50, 50-100, and nun=1 (where undercoverage showed up); informative=False rows are NOT evidence
th = tab[(tab.N == 20000) & tab.arm.isin(["clt_comp", "clt_nocomp"]) &
         (((tab.group == "ng_bin") & tab.level.isin(["[0.0, 50.0)", "[50.0, 100.0)"])) | ((tab.group == "nun_bin") & (tab.level == "1")))]
th = th.assign(informative=th.cell_reps >= 500)
th.to_csv(os.path.join(out, f"{tag}_thinbins.csv"), index=False)
print(th.pivot_table(index=["gamma", "group", "level", "endpoint"], columns="arm", values="cov").round(3).to_string())
print("rows with informative=False (cell_reps<500):", int((~th.informative).sum()))

def panel(ax, t, labs, show, xlabel):
    x = np.arange(len(labs)); wd = 0.8 / len(ARMS)
    for k, arm in enumerate(ARMS):
        s = t[t.arm == arm].set_index("level").reindex(labs)
        ax.bar(x + (k - (len(ARMS) - 1) / 2) * wd, s["cov"], wd, yerr=1.96 * s["mc_se"], capsize=2,
               label=ARMS[arm], color=COLORS[arm])
    ax.axhline(0.95, color="red", ls="--", lw=1, label="nominal 0.95")
    ax.axhline(PASS, color="gray", ls=":", lw=1, label=f"pass {PASS}")
    ax.set_xticks(x); ax.set_xticklabels(show); ax.set_xlabel(xlabel); ax.set_ylim(0.85, 1.02)

for gam, tg in tab.groupby("gamma"):
    for group, labs, show, xl in (("ng_bin", NG_LAB, NG_SHOW, "pooled n_g bin"),
                                  ("nun_bin", NUN_LAB, NUN_SHOW, "# unexplored strata (nun)")):
        Ns = sorted(tg["N"].unique())
        fig, axs = plt.subplots(len(Ns), 2, figsize=(12, 4.2 * len(Ns)), squeeze=False)
        for i, N in enumerate(Ns):
            for j, ep in enumerate(("lower", "upper")):
                ax = axs[i, j]; panel(ax, tg[(tg.N == N) & (tg.group == group) & (tg.endpoint == ep)], labs, show, xl)
                ax.set_title(f"{ep} endpoint, N={N}, gamma={gam}, m_min={a.m_min}")
                if a.m_min >= 10:
                    ax.text(0.01, 0.02, f"m_min={a.m_min}: n_g<{a.m_min} bins conservative from truncation;\n"
                            "se in the 1-9 bins NOT tested (m_min=1 arm only)", transform=ax.transAxes, fontsize=7)
        axs[0, 0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"E1 coverage {a.state} seed {a.seed}; pooled over cell-reps; bars = 95% rep-clustered MC-se")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"{tag}_{group}_g{gam}.png"), dpi=200); plt.close(fig)
print(f"wrote {out}/{tag}_*")