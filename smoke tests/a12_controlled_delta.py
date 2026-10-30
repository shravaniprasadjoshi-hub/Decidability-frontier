"""
A12: controlled-Delta semi-synthetic arm (pool level, N = infinity, noise-free propensity). 
Extends smoke test 3 and needs bounds.py and common.py to be in the same folder / PYTHONPATH.

python "smoke tests/a12_controlled_delta.py"

Env: STATE (CA), SEED (0), FLIP_SEED (123), SYNTH=1 (offline logic test only)
"""
import os
import numpy as np
import pandas as pd
import subprocess, json, hashlib
from common import build, J, M_MIN_DEFAULT, GRID

from bounds import Cell, endpoints, breakdown, gap, bonferroni_z
N_AUDIT = [5000, 20000]
Z_ZS = {"unc": 1.645, "bon": bonferroni_z(0.05, 2 * (C_ - 1))} if (C_ := 32) else None  # placeholder, see below

# CONFIG
STATE, YEAR = os.environ.get("STATE", "CA"), "2018"
SEED, FLIP_SEED = int(os.environ.get("SEED", 0)), int(os.environ.get("FLIP_SEED", 123))
GAMMAS, TAUS, M_MIN, MIN_PER_SIDE = [0.02, 0.03, 0.04, 0.05, 0.1], [0.05, 0.02], M_MIN_DEFAULT, 30

def _git():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return h[:10] + ("+dirty" if dirty else "")
    except Exception:
        return "nogit"

CFG = dict(script="a12", state=STATE, year=YEAR, seed=SEED, flip_seed=FLIP_SEED, gammas=GAMMAS, taus=TAUS,
           m_min=M_MIN, grid=GRID.tolist(), J=J)
CFG_HASH = hashlib.sha1(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:8]
GIT = _git()
TAG = f"{STATE}_{SEED}_{FLIP_SEED}_{CFG_HASH}"
print(f"git={GIT} cfg={CFG_HASH}")

# load everything from common.py
S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED)
Y, Y0, n = S.Y, S.Y0, S.n
codes, names, C = S.codes, S.names, S.C
ref, others, target = S.ref, S.others, S.target
theta, Dtrue = S.theta, S.Dtrue
p, r, eps0 = S.p, S.r, S.eps0
s5, key = S.s5, S.key
select, grouped, amax = S.select, S.grouped, S.amax

Z_ZS = {"unc": 1.645, "bon": bonferroni_z(0.05, 2 * len(others))}   # bon = Holm's first-step z (conservative)

def frontier_cols(cg, cref, dlt, n_pool):
    r = gap(cg, cref, dlt, 0.0, M_MIN)               # lo/hi include delta*S and u_g/u_ref; se_* at pool counts
    out = dict(Dlo=r["lo"], Dhi=r["hi"], se_lo0=r["se_lo"], se_hi0=r["se_hi"],
               tau_par_min0=max(r["hi"], -r["lo"]), tau_disp_max0=max(r["lo"], -r["hi"]))
    for zn, z in Z_ZS.items():
        for N in N_AUDIT:
            f = np.sqrt(n_pool / N)                  # se(N) = se_pool * sqrt(n_pool/N), APPROXIMATE
            out[f"tau_par_min_{zn}_{N}"]  = max(r["hi"] + z*r["se_hi"]*f, -r["lo"] + z*r["se_lo"]*f)
            out[f"tau_disp_max_{zn}_{N}"] = max(r["lo"] - z*r["se_lo"]*f, -r["hi"] - z*r["se_hi"]*f)
        for t in TAUS:                               # closed-form crossover N*, inf if slack <= 0
            ns = lambda slack, se: n_pool * (z*se/slack)**2 if slack > 0 else np.inf
            out[f"Nstar_disp_{zn}@{t}"] = min(ns(r["lo"] - t, r["se_lo"]), ns(-r["hi"] - t, r["se_hi"]))
            out[f"Nstar_par_{zn}@{t}"]  = max(ns(t - r["hi"], r["se_hi"]), ns(t + r["lo"], r["se_lo"]))
    return out

# compute natural theta0 (since S.theta is already flipped)
theta0 = np.array([Y0[codes == c].mean() for c in range(C)])

print(f"STATE={STATE} SEED={SEED} FLIP_SEED={FLIP_SEED} n={n} base rate {Y0.mean():.3f} -> {Y.mean():.3f}; ref='{names[ref]}' (n={int((codes==ref).sum())}), K={len(others)}")
print(f"target vs realized Delta_true: max|diff|={np.nanmax(np.abs(Dtrue[others]-target[others])):.4f} (clipping only); natural Delta median |.|={np.median(np.abs(theta0[others]-theta0[ref])):.3f} -> controlled {np.median(np.abs(Dtrue[others])):.3f}")
print(f"cells with |Delta_true|<=tau: " + ", ".join(f"tau={t}: {int((np.abs(Dtrue[others])<=t).sum())}/{len(others)}" for t in TAUS))

rows = []
for g in GAMMAS:
    D, prop = select(g, eps0); gr = grouped(key, C * J, D, prop)
    
    # use S.dgap instead of local dpop
    dlt = amax(S.dgap(gr)[1]); pooled = amax(S.dgap(grouped(s5, J, D, prop))[1])
    
    print(f"\n===== GAMMA={g}: approval {D.mean():.3f} | recomputed shared delta (propensity, max cell x stratum) = {dlt:.3f} (ratio to gamma {dlt/g:.2f}; pooled {pooled:.3f})")
    
    cl = lambda c: Cell(gr["n"].reshape(C, J)[c], gr["na"].reshape(C, J)[c], gr["ka"].reshape(C, J)[c])
    cref = cl(ref); er = endpoints(cref, 0.0, M_MIN); R_ = []
    
    for c in others:
        cg = cl(c); eg = endpoints(cg, 0.0, M_MIN)
        row = dict(gamma=g, cell=names[c], n=int(cg.n), target=target[c], Delta_true=Dtrue[c], Dc=eg["centre"] - er["centre"], S=eg["S"] + er["S"],
                   u_g=eg["u"], u_ref=er["u"], delta_shared=dlt)
    
        for t in TAUS:
            b = breakdown(cg, cref, t, 0.0, M_MIN); row[f"d*_disp@{t}"], row[f"d*_par@{t}"] = b["disparity"], b["parity"]
    
        row.update(frontier_cols(cg, cref, dlt, n))
        R_.append(row)
    
    R = pd.DataFrame(R_); rows += R_
    print(f"  rho_ref(explored)={er['S']:.3f}; per-cell S median {R.S.median():.2f} (range {R.S.min():.2f}-{R.S.max():.2f}); |Dc - Delta_true| median {np.median(np.abs(R.Dc-R.Delta_true)):.3f}")
    
    for t in TAUS:
        pr, ds = R[f"d*_par@{t}"], R[f"d*_disp@{t}"]
        print(f"  tau={t}: |Dc|<tau {(R.Dc.abs()<t).sum()}/{len(R)} | parity feasible at delta=0 {(pr>=0).sum()} | parity certifiable at delta={dlt:.3f}: {(pr>dlt).sum()} | "
              f"disparity certifiable: {(ds>dlt).sum()} | d*_disp in [0.02,0.10]: {((ds>=0.02)&(ds<=0.10)).sum()} | d*_par in [0.005,0.10]: {((pr>=0.005)&(pr<=0.10)).sum()}")

df = pd.DataFrame(rows)
df["cfg_hash"], df["git"] = CFG_HASH, GIT
outfile = f"smoke tests/a12_controlled_{STATE}_{SEED}_{FLIP_SEED}.csv"
df.to_csv(outfile, index=False)
print(f"\nsaved {outfile}")