"""plot_e3.py: E3 frontier figures from a12 CSVs (+ optional faithful-Holm panel from b8b npz logs).

Run from repo root:  python "smoke tests/results/plot_e3.py" ["smoke tests"]     (arg = folder holding the files; default 'smoke tests')
Reads : a12_controlled_CA_<seed>_<flip>.csv   (needs the *_exp columns, i.e. patched a12)
        zlog_CA_<seed>_<flip>_<cfg>_g<gamma>_N<N>_tau<tau>.npz   (b8b output; optional, Fig 3 skipped if none)
Writes: figs/e3_fig1_frac_certifiable.png, e3_fig2_tau_curves.png, e3_fig3_holm_power.png, e3_summary.csv, e3_holm_power_<exp|real>.csv
Env: DENOM=exp (default; d*_exp > delta defines certifiable cells in Fig 3) | real (realized-draw d*, as b8b used)

Each (seed, flip) is its own arm: thin line per arm, thick = mean across arms. Do NOT pool counts across arms.
x-axis = recomputed shared oracle delta (propensity, max cell x stratum), one point per gamma.
N=inf uses expected-count pool quantities (tau_*_min0_exp); finite N uses expected counts scaled by N/n_pool with m_min truncation
('unc' z=1.645; 'bon' = Bonferroni-2K z, a conservative ENVELOPE, not Holm's operating point; Holm is Fig 3).
Parity at tau is certifiable iff tau_par_min < tau; disparity iff tau_disp_max > tau.
"""
import sys, os, re, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ndtr

D = sys.argv[1] if len(sys.argv) > 1 else "smoke tests/results"
OUT = os.path.join(D, "figs"); os.makedirs(OUT, exist_ok=True)
TAUS = [0.05, 0.02]

# ---------- load arms ----------
arms = {}
for f in sorted(glob.glob(os.path.join(D, "a12_controlled_CA_*_*.csv"))):
    m = re.search(r"CA_(\d+)_(\d+)\.csv$", f)
    if not m: continue
    d = pd.read_csv(f)
    if "tau_par_min0_exp" not in d.columns:
        print(f"skip {f}: no *_exp columns (unpatched a12)"); continue
    arms[(int(m[1]), int(m[2]))] = d
assert arms, f"no patched a12 CSVs in {D}"
print("arms:", list(arms), "| git:", {k: v.git.iloc[0] for k, v in arms.items()})

# ---------- series: fraction of cells certifiable, per arm per gamma ----------
def series(d):
    rows = []
    for g, x in d.groupby("gamma"):
        r = dict(gamma=g, delta=x.delta_shared.iloc[0], K=len(x))
        for t in TAUS:
            for name, par, disp in [("inf", "tau_par_min0_exp", "tau_disp_max0_exp"),
                                    ("unc20k", "tau_par_min_exp_unc_20000", "tau_disp_max_exp_unc_20000"),
                                    ("bon20k", "tau_par_min_exp_bon_20000", "tau_disp_max_exp_bon_20000"),
                                    ("unc5k", "tau_par_min_exp_unc_5000", "tau_disp_max_exp_unc_5000")]:
                r[f"par_{name}@{t}"] = (x[par] < t).mean()
                r[f"disp_{name}@{t}"] = (x[disp] > t).mean()
        rows.append(r)
    return pd.DataFrame(rows).sort_values("delta")

S = {k: series(d) for k, d in arms.items()}
pd.concat([s.assign(seed=k[0], flip=k[1]) for k, s in S.items()]).to_csv(os.path.join(OUT, "e3_summary.csv"), index=False)

STY = {"inf": ("k", "-", "N=inf (expected counts)"), "unc20k": ("C0", "-", "N=20k, z=1.645"),
       "bon20k": ("C3", "--", "N=20k, Bonferroni-2K (envelope)"), "unc5k": ("C2", ":", "N=5k, z=1.645")}

def band(ax, kind, t):
    for name, (c, ls, lab) in STY.items():
        key = f"{kind}_{name}@{t}"
        for s in S.values(): ax.plot(s.delta, s[key], color=c, ls=ls, lw=0.6, alpha=0.35)
        mean = pd.concat([s.set_index("gamma")[key] for s in S.values()], axis=1).mean(axis=1)
        dl = pd.concat([s.set_index("gamma").delta for s in S.values()], axis=1).mean(axis=1)
        ax.plot(dl, mean, color=c, ls=ls, lw=2, marker="o", ms=3, label=lab)
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("assumed delta (oracle, shared)"); ax.grid(alpha=0.3)

# ---------- Fig 1 ----------
fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
for i, t in enumerate(TAUS):
    for j, kind in enumerate(["par", "disp"]):
        ax = axs[i, j]; band(ax, kind, t)
        ax.set_title(f"{'parity' if kind == 'par' else 'disparity'} certifiable, tau={t}")
        if j == 0: ax.set_ylabel("share of 31 cells")
axs[0, 0].legend(fontsize=8)
fig.suptitle(f"E3: share of cells certifiable vs delta ({len(S)} arms; thin = arm, thick = mean; no multiplicity beyond the stated z)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "e3_fig1_frac_certifiable.png"), dpi=150); plt.close(fig)

# ---------- Fig 2: tau_min curves across cells (median + IQR), pooled over arms at each gamma ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
spec = [("inf", "tau_par_min0_exp", "tau_disp_max0_exp"), ("unc20k", "tau_par_min_exp_unc_20000", "tau_disp_max_exp_unc_20000"),
        ("bon20k", "tau_par_min_exp_bon_20000", "tau_disp_max_exp_bon_20000")]
for ax, which, ttl in [(axs[0], 1, "tau_par_min: smallest tau at which parity is certifiable"),
                       (axs[1], 2, "tau_disp_max (clipped at 0 = not certifiable at any tau)")]:
    for name, *cols in spec:
        c, ls, lab = STY[name]; xs, q = [], []
        for g in sorted(set().union(*[set(d.gamma) for d in arms.values()])):
            v = np.concatenate([d.loc[d.gamma == g, cols[which - 1]].values for d in arms.values()])
            dl = np.mean([d.loc[d.gamma == g, "delta_shared"].iloc[0] for d in arms.values()])
            v = np.maximum(v, 0) if which == 2 else v
            xs.append(dl); q.append(np.nanpercentile(v, [25, 50, 75]))
        q = np.array(q); ax.plot(xs, q[:, 1], color=c, ls=ls, marker="o", ms=3, label=lab); ax.fill_between(xs, q[:, 0], q[:, 2], color=c, alpha=0.10)
    for t in TAUS: ax.axhline(t, color="grey", lw=0.8, ls=":")
    ax.set_title(ttl, fontsize=9); ax.set_xlabel("assumed delta"); ax.grid(alpha=0.3)
axs[0].set_ylabel("tau (median, IQR over cells x arms)"); axs[0].legend(fontsize=8)
axs[0].text(0.02, 0.97, "parity certifiable at tau where curve < tau", transform=axs[0].transAxes, va="top", fontsize=7)
axs[1].text(0.02, 0.97, "disparity certifiable at tau where curve > tau", transform=axs[1].transAxes, va="top", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "e3_fig2_tau_curves.png"), dpi=150); plt.close(fig)

# ---------- Fig 3: faithful Holm vs uncorrected power from b8b npz (optional) ----------
# denominators: cells certifiable at pool level under EXPECTED counts (d*_exp > delta), matching the audit DGP (D ~ Bern(propensity)).
# set DENOM=real to use the realized-draw d* instead (what b8b used).
DENOM = os.environ.get("DENOM", "exp"); SUF = "_exp" if DENOM == "exp" else ""
def holm_rej(p, a):
    rej = np.zeros(len(p), bool)
    for i, idx in enumerate(np.argsort(p)):
        if p[idx] <= a / (len(p) - i): rej[idx] = True
        else: break
    return rej

recs = []; warn = set(); found = {}
for (seed, flip), d in arms.items():
    for g, x in d.groupby("gamma"):
        dl = x.delta_shared.iloc[0]
        for N in (5000, 20000):
            for t in TAUS:
                fs = sorted(glob.glob(os.path.join(D, f"zlog_CA_{seed}_{flip}_*_g{g}_N{N}_tau{t}.npz")), key=os.path.getmtime)
                if not fs: continue
                found[(seed, flip)] = found.get((seed, flip), 0) + 1
                if len(fs) > 1: warn.add((seed, flip))
                z = np.load(fs[-1], allow_pickle=True); A = z["z"]; K = int(z["K"]); cells = list(z["cells"])
                pv = A[:, 4 * K:]                                    # rows: [dlo,dhi,slo,shi (4K), pv (2K)]
                idx = np.array([cells.index(c) for c in x.cell])     # align to a12 order
                dok = (x[f"d*_disp@{t}{SUF}"].values > dl); pok = (x[f"d*_par@{t}{SUF}"].values > dl)
                H = np.array([holm_rej(r, 0.05) for r in pv]); U = pv <= 0.05
                for kind, ok, off in [("disp", dok, 0), ("par", pok, K)]:
                    sel = idx[ok] + off
                    recs.append(dict(seed=seed, flip=flip, gamma=g, delta=dl, N=N, tau=t, kind=kind, n_cert=len(sel), R=len(pv),
                                     unc=U[:, sel].mean() if len(sel) else np.nan, holm=H[:, sel].mean() if len(sel) else np.nan))
print("npz files used per arm:", found, "| missing arms:", [k for k in arms if k not in found])
if recs:
    R = pd.DataFrame(recs); R.to_csv(os.path.join(OUT, f"e3_holm_power_{DENOM}.csv"), index=False)
    n_arms = R[["seed", "flip"]].drop_duplicates().shape[0]
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\nper-arm power, N=20000, tau=0.05, disparity (n_cert, uncorrected, Holm):")
    print(R[(R.N == 20000) & (R.tau == 0.05) & (R.kind == "disp")].pivot_table(index="gamma", columns=["seed", "flip"], values=["n_cert", "unc", "holm"]).round(3).to_string())
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for i, N in enumerate((20000, 5000)):
        for j, kind in enumerate(["disp", "par"]):
            ax = axs[i, j]; anyp = False
            for proc, c in [("unc", "C0"), ("holm", "C3")]:
                for t, ls in [(0.05, "-"), (0.02, "--")]:
                    r = R[(R.N == N) & (R.kind == kind) & (R.tau == t)].dropna(subset=[proc])
                    if r.empty: continue
                    anyp = True
                    for _, ra in r.groupby(["seed", "flip"]):            # thin line per arm
                        ra = ra.sort_values("gamma"); ax.plot(ra.delta, ra[proc], color=c, ls=ls, lw=0.7, alpha=0.4)
                    gg = r.groupby("gamma").agg(delta=("delta", "mean"), p=(proc, "mean"), ncmin=("n_cert", "min")).sort_values("delta")
                    ax.plot(gg.delta, gg.p, color=c, ls=ls, lw=2, label=f"{'uncorrected' if proc == 'unc' else 'Holm(2K)'}, tau={t} (mean)")
                    thin = gg.ncmin < 5                                   # hollow = fewer than 5 certifiable cells in some arm
                    ax.scatter(gg.delta[~thin], gg.p[~thin], color=c, s=16, zorder=3)
                    ax.scatter(gg.delta[thin], gg.p[thin], facecolors="none", edgecolors=c, s=26, zorder=3)
            nc = R[(R.N == N) & (R.kind == kind) & (R.tau == 0.05)].groupby("gamma").n_cert.mean()
            ax.set_title(f"{'disparity' if kind == 'disp' else 'parity'} power, N={N}  (cells certifiable at tau=.05 by gamma: " + "/".join(f"{v:.0f}" for v in nc.values) + ")", fontsize=8)
            if kind == "par" and anyp and np.nanmax(R[(R.N == N) & (R.kind == "par")][["unc", "holm"]].values) == 0:
                ax.text(0.5, 0.5, "0 rejections in every certifiable cell\n(parity unreachable at this N)", transform=ax.transAxes, ha="center", fontsize=9)
            ax.set_xlabel("assumed delta"); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)
    axs[0, 0].legend(fontsize=7); axs[0, 0].set_ylabel("rejection rate"); axs[1, 0].set_ylabel("rejection rate")
    fig.suptitle(f"E3: power, uncorrected vs Holm(2K); thin = arm, thick = mean of {n_arms} arms; denominator = d*>delta ({DENOM} counts); hollow = <5 certifiable cells", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "e3_fig3_holm_power.png"), dpi=150); plt.close(fig)
    if warn: print("WARNING: several npz cfg hashes for", sorted(warn), "-> used the newest; delete stale zlogs if unsure")
else:
    print("no zlog npz found -> Fig 3 skipped")
print("wrote figs to", OUT)