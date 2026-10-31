"""
A12: controlled-Delta semi-synthetic arm (pool level, N = infinity, noise-free propensity). 
Extends smoke test 3 and needs bounds.py and common.py to be in the same folder / PYTHONPATH.

python "smoke tests/results/a12_controlled_delta.py"

Env: STATE (CA), SEED (0), FLIP_SEED (123), SYNTH=1 (offline logic test only)
"""
import os
import numpy as np
import pandas as pd
import subprocess, json, hashlib
from common import build, J, M_MIN_DEFAULT, GRID
from bounds import Cell, endpoints, breakdown, gap, verdict, bonferroni_z

# CONFIG
STATE, YEAR = os.environ.get("STATE", "CA"), "2018"
SEED, FLIP_SEED = int(os.environ.get("SEED", 0)), int(os.environ.get("FLIP_SEED", 123))
GAMMAS, TAUS, M_MIN, MIN_PER_SIDE = [0.02, 0.03, 0.04, 0.05, 0.1], [0.05, 0.02], M_MIN_DEFAULT, 30
N_AUDIT = [5000, 20000]

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

S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED)
Y, Y0, n = S.Y, S.Y0, S.n
codes, names, C = S.codes, S.names, S.C
ref, others, target = S.ref, S.others, S.target
theta, Dtrue = S.theta, S.Dtrue
p, r, eps0 = S.p, S.r, S.eps0
s5, key = S.s5, S.key
select, grouped, amax = S.select, S.grouped, S.amax

Z_ZS = {"unc": 1.645, "bon": bonferroni_z(0.05, 2 * len(others))}

NGRID = np.geomspace(1_000, 2_000_000, 240)

def _scaled(c, f): 
    return Cell(c.n_j * f, c.na_j * f, c.k_j * f)

def nstar_trunc_exp(cg, cref, delta, tau, kind, z, m_min, n_pool):
    want = "disparity" if kind == "disp" else "parity"
    good = np.array([verdict(_scaled(cg, N / n_pool), _scaled(cref, N / n_pool), delta, tau, z, m_min) == want for N in NGRID])
    bad = np.flatnonzero(~good)
    if len(bad) == 0: return float(NGRID[0])
    if bad[-1] == len(NGRID) - 1: return np.inf
    return float(NGRID[bad[-1] + 1])

def exp_cols(cg, cref, dlt, n_pool, taus, zs, m_min, Ns=(5000, 20000)):
    eg, er = endpoints(cg, 0.0, m_min), endpoints(cref, 0.0, m_min)
    rd = gap(cg, cref, dlt, 0.0, m_min)
    out = dict(Dc_exp=eg["centre"] - er["centre"], S_exp=eg["S"] + er["S"], u_g_exp=eg["u"], u_ref_exp=er["u"],
               tau_par_min0_exp=max(rd["hi"], -rd["lo"]), tau_disp_max0_exp=max(rd["lo"], -rd["hi"]))
    for t in taus:
        b = breakdown(cg, cref, t, 0.0, m_min)
        out[f"d*_disp@{t}_exp"], out[f"d*_par@{t}_exp"] = b["disparity"], b["parity"]
        for kind in ("disp", "par"):
            out[f"Nstar_trunc_exp_{kind}@{t}"] = nstar_trunc_exp(cg, cref, dlt, t, kind, 1.645, m_min, n_pool)
    for zn, z in zs.items():
        for N in Ns:
            rN = gap(_scaled(cg, N / n_pool), _scaled(cref, N / n_pool), dlt, z, m_min)
            out[f"tau_par_min_exp_{zn}_{N}"] = max(rN["hi_c"], -rN["lo_c"])
            out[f"tau_disp_max_exp_{zn}_{N}"] = max(rN["lo_c"], -rN["hi_c"])
    return out

def frontier_cols(cg, cref, dlt, n_pool):
    r = gap(cg, cref, dlt, 0.0, M_MIN)
    out = dict(Dlo=r["lo"], Dhi=r["hi"], se_lo0=r["se_lo"], se_hi0=r["se_hi"],
               tau_par_min0=max(r["hi"], -r["lo"]), tau_disp_max0=max(r["lo"], -r["hi"]))
    for zn, z in Z_ZS.items():
        for N in N_AUDIT:
            f = np.sqrt(n_pool / N)
            out[f"tau_par_min_{zn}_{N}"]  = max(r["hi"] + z*r["se_hi"]*f, -r["lo"] + z*r["se_lo"]*f)
            out[f"tau_disp_max_{zn}_{N}"] = max(r["lo"] - z*r["se_lo"]*f, -r["hi"] - z*r["se_hi"]*f)
        for t in TAUS:
            ns = lambda slack, se: n_pool * (z*se/slack)**2 if slack > 0 else np.inf
            out[f"Nstar_disp_{zn}@{t}"] = min(ns(r["lo"] - t, r["se_lo"]), ns(-r["hi"] - t, r["se_hi"]))
            out[f"Nstar_par_{zn}@{t}"]  = max(ns(t - r["hi"], r["se_hi"]), ns(t + r["lo"], r["se_lo"]))
    return out

theta0 = np.array([Y0[codes == c].mean() for c in range(C)])

print(f"STATE={STATE} SEED={SEED} FLIP_SEED={FLIP_SEED} n={n} base rate {Y0.mean():.3f} -> {Y.mean():.3f}; ref='{names[ref]}' (n={int((codes==ref).sum())}), K={len(others)}")
print(f"target vs realized Delta_true: max|diff|={np.nanmax(np.abs(Dtrue[others]-target[others])):.4f} (clipping only); natural Delta median |.|={np.median(np.abs(theta0[others]-theta0[ref])):.3f} -> controlled {np.median(np.abs(Dtrue[others])):.3f}")
print(f"cells with |Delta_true|<=tau: " + ", ".join(f"tau={t}: {int((np.abs(Dtrue[others])<=t).sum())}/{len(others)}" for t in TAUS))

rows = []
for g in GAMMAS:
    D, prop = select(g, eps0); gr = grouped(key, C * J, D, prop)
    
    dlt = amax(S.dgap(gr)[1]); pooled = amax(S.dgap(grouped(s5, J, D, prop))[1])
    
    print(f"\n===== GAMMA={g}: approval {D.mean():.3f} | recomputed shared delta (propensity, max cell x stratum) = {dlt:.3f} (ratio to gamma {dlt/g:.2f}; pooled {pooled:.3f})")
    
    cl = lambda c: Cell(gr["n"].reshape(C, J)[c], gr["na"].reshape(C, J)[c], gr["ka"].reshape(C, J)[c])
    cref = cl(ref); er = endpoints(cref, 0.0, M_MIN); R_ = []
    
    cl_e = lambda c: Cell(gr["n"].reshape(C, J)[c],
                          np.minimum(gr["A"].reshape(C, J)[c], gr["n"].reshape(C, J)[c]),
                          np.minimum(gr["Ay"].reshape(C, J)[c], gr["A"].reshape(C, J)[c]))
    cref_e = cl_e(ref)
    
    for c in others:
        cg = cl(c); eg = endpoints(cg, 0.0, M_MIN)
        row = dict(gamma=g, cell=names[c], n=int(cg.n), target=target[c], Delta_true=Dtrue[c], Dc=eg["centre"] - er["centre"], S=eg["S"] + er["S"],
                   u_g=eg["u"], u_ref=er["u"], delta_shared=dlt)
    
        for t in TAUS:
            b = breakdown(cg, cref, t, 0.0, M_MIN); row[f"d*_disp@{t}"], row[f"d*_par@{t}"] = b["disparity"], b["parity"]
    
        row.update(frontier_cols(cg, cref, dlt, n))
        row.update(exp_cols(cl_e(c), cref_e, dlt, n, TAUS, Z_ZS, M_MIN))
        
        for t in TAUS:
            assert (row["tau_par_min0_exp"] < t) == (row[f"d*_par@{t}_exp"] > dlt) or abs(row["tau_par_min0_exp"] - t) < 1e-9
            
        R_.append(row)
    
    R = pd.DataFrame(R_); rows += R_
    print(f"  rho_ref(explored)={er['S']:.3f}; per-cell S median {R.S.median():.2f} (range {R.S.min():.2f}-{R.S.max():.2f}); |Dc - Delta_true| median {np.median(np.abs(R.Dc-R.Delta_true)):.3f}")
    
    for t in TAUS:
        pr, ds = R[f"d*_par@{t}"], R[f"d*_disp@{t}"]
        pr_e, ds_e = R[f"d*_par@{t}_exp"], R[f"d*_disp@{t}_exp"]
        print(f"  tau={t}: |Dc|<tau {(R.Dc.abs()<t).sum()}/{len(R)} | parity feasible at delta=0 {(pr>=0).sum()} | parity certifiable at delta={dlt:.3f}: {(pr>dlt).sum()} (exp: {(pr_e>dlt).sum()}) | "
              f"disparity certifiable: {(ds>dlt).sum()} (exp: {(ds_e>dlt).sum()}) | d*_disp in [0.02,0.10]: {((ds>=0.02)&(ds<=0.10)).sum()} | d*_par in [0.005,0.10]: {((pr>=0.005)&(pr<=0.10)).sum()}")

df = pd.DataFrame(rows)
df["cfg_hash"], df["git"] = CFG_HASH, GIT
outfile = f"smoke tests/results/a12_controlled_{STATE}_{SEED}_{FLIP_SEED}.csv"
df.to_csv(outfile, index=False)
print(f"\nsaved {outfile}")