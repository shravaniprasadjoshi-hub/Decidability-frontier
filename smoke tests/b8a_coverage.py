"""
B8a: per-cell coverage of the identified-set ENDPOINTS, CLT margin (bounds.py) vs exact (Clopper-Pearson + union bound), natural cells.

usage from repo root:
    set PYTHONPATH=.;core
    python "smoke tests/results/b8a_coverage.py"

Env: STATE (CA), SEED (0), R (reps, 1000), M_MIN (10), SYNTH=1 (offline logic test only)
"""
import os, numpy as np, pandas as pd
from scipy.stats import beta
from bounds import Cell, endpoints
from common import build, J, M_MIN_DEFAULT, GRID
import subprocess, json, hashlib

# CONFIG 
STATE, YEAR = os.environ.get("STATE", "CA"), "2018"
SEED = int(os.environ.get("SEED", 0)); R = int(os.environ.get("R", 1000))
M_MIN = int(os.environ.get("M_MIN", M_MIN_DEFAULT))
GAMMAS, N_LIST = [0.02, 0.05, 0.1], [5000, 20000]
MIN_PER_SIDE = 30
Z, ALPHA, TAU = 1.645, 0.05, 0.05 # one-sided per endpoint, single-test z (NOT multiplicity-adjusted)
NBINS = [0, 50, 100, 200, 500, 10**9]

def _git():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return h[:10] + ("+dirty" if dirty else "")
    except Exception:
        return "nogit"

CFG = dict(script="b8a", state=STATE, year=YEAR, seed=SEED, R=R, gammas=GAMMAS, N_list=N_LIST,
           m_min=M_MIN, grid=GRID.tolist(), J=J, alpha=ALPHA, tau=TAU)
CFG_HASH = hashlib.sha1(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:8]
GIT = _git()
print(f"git={GIT} cfg={CFG_HASH}")

# load everything from common.py (natural Y, no flips)
S = build(state=STATE, seed=SEED)
Y, n = S.Y, S.n
codes, names, C = S.codes, S.names, S.C
ref, theta, Dtrue = S.ref, S.theta, S.Dtrue
p, r, eps0 = S.p, S.r, S.eps0
s5, key = S.s5, S.key
select, grouped, amax = S.select, S.grouped, S.amax

# audit-sample stream
rng2 = np.random.default_rng(10_000 + SEED)

# exact arm 
def cp_lo(k, m, a):
    with np.errstate(all="ignore"): return np.where(k > 0, beta.ppf(a, k, m - k + 1), 0.0)

def cp_up(k, m, a):
    with np.errstate(all="ignore"): return np.where(k < m, beta.ppf(1 - a, k + 1, m - k), 1.0)

def exact(nj, na, ka, d):
    """L = sum_j (w_j mu_j) - d*sum_j r_j (r_j = rejected share of the cell in stratum j); all terms >= 0, so bound each piece.
    3J one-sided CP events per endpoint, union bound => each at ALPHA/(3J). No m_min: na=0 gives mu in [0,1] automatically."""
    ng = nj.sum(1, keepdims=True); a = ALPHA / (3 * J)
    wl, wu, ru = cp_lo(nj, ng, a), cp_up(nj, ng, a), cp_up(nj - na, ng, a)
    ml, mu_ = cp_lo(ka, na, a), cp_up(ka, na, a)
    return (wl * ml).sum(1) - d * ru.sum(1), (wu * mu_).sum(1) + d * ru.sum(1)

# main
print(f"STATE={STATE} SEED={SEED} R={R} ref='{names[ref]}' K={C-1}  (MC se per pooled bin ~ small; per cell ~{100*np.sqrt(.95*.05/R):.1f}pp)")
allrows = []

for g in GAMMAS:
    D, prop = select(g, eps0); gr = grouped(key, C * J, D, prop)
    
    # use S.dgap for the oracle delta
    dlt = amax(S.dgap(gr)[1])
    
    nc = gr["n"].reshape(C, J); w = nc / np.maximum(nc.sum(1, keepdims=True), 1)
    with np.errstate(all="ignore"):
        mu = np.nan_to_num(gr["Ay"] / gr["A"]).reshape(C, J); rho = np.nan_to_num(gr["B"] / gr["n"]).reshape(C, J)
    Lt, Ut = (w * (mu - rho * dlt)).sum(1), (w * (mu + rho * dlt)).sum(1)
    
    print(f"\n######## gamma={g}  shared oracle delta={dlt:.3f}  sanity: theta in [L*,U*] for {int(((theta>=Lt-1e-9)&(theta<=Ut+1e-9)).sum())}/{C} cells")
    
    for N in N_LIST:
        rec = {k: np.full((R, C), np.nan) for k in "L U sL sU sL0 sU0 Le Ue ng nun".split()}
        
        for t in range(R):
            idx = rng2.integers(0, n, N); Di = rng2.random(N) < prop[idx]; k = key[idx]; yy = Y[idx] == 1
            bc = lambda m: np.bincount(k[m], minlength=C * J).reshape(C, J)
            nj, na, ka = bc(np.ones(N, bool)), bc(Di), bc(Di & yy)
            rec["Le"][t], rec["Ue"][t] = exact(nj, na, ka, dlt); rec["ng"][t] = nj.sum(1); rec["nun"][t] = (na < M_MIN).sum(1)
        
            for c in range(C):
                if nj[c].sum() == 0: continue
                cl = Cell(nj[c], na[c], ka[c]); e, e0 = endpoints(cl, dlt, M_MIN), endpoints(cl, dlt, M_MIN, between=False)
                rec["L"][t, c], rec["U"][t, c], rec["sL"][t, c], rec["sU"][t, c] = e["L"], e["U"], e["se_L"], e["se_U"]
                rec["sL0"][t, c], rec["sU0"][t, c] = e0["se_L"], e0["se_U"]
        
        L, U, sL, sU, Le, Ue = (rec[k] for k in ["L", "U", "sL", "sU", "Le", "Ue"])
        
        # disparity certificate vs the fixed reference (one-sided z, no multiplicity)
        dlo, dhi = L - U[:, [ref]], U - L[:, [ref]]
        slo, shi = np.sqrt(sL**2 + sU[:, [ref]]**2), np.sqrt(sU**2 + sL[:, [ref]]**2)
        cd_clt = (dlo - Z * slo > TAU) | (dhi + Z * shi < -TAU)
        cd_ex = ((Le - Ue[:, [ref]]) > TAU) | ((Ue - Le[:, [ref]]) < -TAU)
        
        df = pd.DataFrame({
            "cell": np.tile(names, R), "c": np.tile(np.arange(C), R), "ng": rec["ng"].ravel(), "nun": rec["nun"].ravel(),
            "cL": ((L - Z * sL) <= Lt).ravel(), "cU": ((U + Z * sU) >= Ut).ravel(),
            "cL_nb": ((L - Z * rec["sL0"]) <= Lt).ravel(), "cU_nb": ((U + Z * rec["sU0"]) >= Ut).ravel(),
            "eL": (Le <= Lt).ravel(), "eU": (Ue >= Ut).ravel(),
            "th": (((L - Z * sL) <= theta) & (theta <= U + Z * sU)).ravel(),
            "w_clt": (U + Z * sU - (L - Z * sL)).ravel(), "w_ex": (Ue - Le).ravel(),
            "cd_clt": cd_clt.ravel(), "cd_ex": cd_ex.ravel(), "Dtrue": np.tile(Dtrue, R), "ok": ~np.isnan(L).ravel()})
        
        df = df[(df.ok) & (df.c != ref)].drop(columns="ok"); df["gamma"], df["N"] = g, N
        allrows.append(df)
        df["ngbin"] = pd.cut(df.ng, NBINS, right=False); df["nun2"] = df.nun.clip(upper=2)
        agg = dict(share=("ng", "size"), cL=("cL", "mean"), cU=("cU", "mean"), cL_nb=("cL_nb", "mean"), cU_nb=("cU_nb", "mean"),
                   eL=("eL", "mean"), eU=("eU", "mean"), theta=("th", "mean"), w_clt=("w_clt", "mean"), w_ex=("w_ex", "mean"))
        pd.set_option("display.width", 220, "display.max_columns", 30)
        
        print(f"\n--- gamma={g} N={N}: by realized n_g (cL/cU = CLT endpoint coverage, _nb = no composition term, eL/eU = exact; nominal .95)")
        print(df.groupby("ngbin", observed=True).agg(**agg).assign(share=lambda t: t.share / len(df)).round(3).to_string())
        
        print(f"--- gamma={g} N={N}: by # unexplored strata (<{M_MIN} approved); 0 = clean CLT test")
        print(df.groupby("nun2").agg(**agg).assign(share=lambda t: t.share / len(df)).round(3).to_string())
        df["near"] = df.Dtrue.abs() <= TAU
        
        print(f"--- gamma={g} N={N}: disparity-certificate rate (z={Z}, no multiplicity) by whether |Delta_true|<=tau (those are WRONG certs)")
        print(df.groupby("near").agg(cells=("c", "nunique"), clt=("cd_clt", "mean"), exact=("cd_ex", "mean")).round(3).to_string())

out = pd.concat(allrows).drop(columns=["ngbin", "nun2"], errors="ignore").groupby(["gamma", "N", "cell"]).mean(numeric_only=True)
out["cfg_hash"], out["git"] = CFG_HASH, GIT
outfile = f"smoke tests/results/b8a_coverage_{STATE}_{SEED}_m{M_MIN}_R{R}.csv"
out.to_csv(outfile)
print(f"\nsaved {outfile} (per-cell means)")