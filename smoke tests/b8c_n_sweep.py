"""
B8c: N sweep for parity/disparity power on the controlled-Delta arm (A12 setup).

Env: STATE (CA), SEED (0), FLIP_SEED (123), R (200), GAMMA (0.02), SYNTH=1
"""
import os, numpy as np, pandas as pd
from scipy.special import ndtr
from bounds import Cell, endpoints, breakdown
from common import build, J, GRID
import subprocess, json, hashlib

# CONFIG
STATE, YEAR = os.environ.get("STATE", "CA"), "2018"
SEED, FLIP_SEED = int(os.environ.get("SEED", 0)), int(os.environ.get("FLIP_SEED", 123))
TAUS, M_MIN, MIN_PER_SIDE = [0.05, 0.02], 10, 30
ALPHA, Z, R = 0.05, 1.645, int(os.environ.get("R", 200))
G = float(os.environ.get("GAMMA", 0.02))
N_LIST = [20000, 50000, 100000, 195665]

def _git():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return h[:10] + ("+dirty" if dirty else "")
    except Exception:
        return "nogit"

CFG = dict(script="b8c", state=STATE, year=YEAR, seed=SEED, flip_seed=FLIP_SEED, gamma=G, R=R, taus=TAUS, m_min=M_MIN, grid=GRID.tolist(), J=J)
CFG_HASH = hashlib.sha1(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:8]
GIT = _git()
print(f"git={GIT} cfg={CFG_HASH}")

# load setup from common.py
S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED)
Y, n = S.Y, S.n
codes, names, C = S.codes, S.names, S.C
ref, others = S.ref, S.others
select, grouped, amax = S.select, S.grouped, S.amax

s5, key = S.s5, S.key
K = len(others)
rng2 = np.random.default_rng(30_000 + SEED)

def holm(p, a):
    rej = np.zeros(len(p), bool)
    for i, idx in enumerate(np.argsort(p)):
        if p[idx] <= a / (len(p) - i): rej[idx] = True
        else: break
    return rej

# Setup for this gamma
D, prop = select(G, S.eps0); gr = grouped(key, C * J, D, prop)
dlt = amax(S.dgap(gr)[1])
cl = lambda c: Cell(gr["n"].reshape(C, J)[c], gr["na"].reshape(C, J)[c], gr["ka"].reshape(C, J)[c])

# Compute pool-level breakdown and Nstar
dstar = {t: (np.array([breakdown(cl(c), cl(ref), t, 0.0, M_MIN)["disparity"] for c in others]),
             np.array([breakdown(cl(c), cl(ref), t, 0.0, M_MIN)["parity"] for c in others])) for t in TAUS}

# Nstar calculation (based on pool size n)
pool_E = {c: endpoints(cl(c), dlt, M_MIN) for c in others + [ref]}
er = pool_E[ref]
Nstar = {}
for tau in TAUS:
    for c_idx, c in enumerate(others):
        # For disparity via Delta_lo: margin = |Dc| - tau - delta*S - u_ref
        # We use the observed Dc from pool
        Dc = pool_E[c]["centre"] - er["centre"]
        S = pool_E[c]["S"] + er["S"]
        u = er["u"] # direction-specific (u_ref for disparity via L)
        
        margin_disp = abs(Dc) - tau - dlt * S - u
        se_pool_disp = np.sqrt(pool_E[c]["se_L"]**2 + er["se_U"]**2)
        Nstar[(c_idx, tau, "disp")] = (n * (Z * se_pool_disp / margin_disp)**2) if margin_disp > 0 else np.inf

        # For parity: margin = dstar_par - delta
        margin_par = dstar[tau][1][c_idx] - dlt
        se_pool_par = np.sqrt(pool_E[c]["se_U"]**2 + er["se_L"]**2)
        Nstar[(c_idx, tau, "par")] = (n * (Z * se_pool_par / margin_par)**2) if margin_par > 0 else np.inf

print(f"gamma={G} oracle delta={dlt:.3f} R={R} K={K}; parity-certifiable cells at assumed delta: " + ", ".join(f"tau={t}: {int((dstar[t][1]>dlt).sum())}" for t in TAUS))

# Track per-cell power over reps
cert_disp = {t: {proc: np.zeros((len(N_LIST), K)) for proc in ["none", "holm"]} for t in TAUS}
cert_par = {t: {proc: np.zeros((len(N_LIST), K)) for proc in ["none", "holm"]} for t in TAUS}

for iN, N in enumerate(N_LIST):
    for _ in range(R):
        idx = rng2.integers(0, n, N); Di = rng2.random(N) < prop[idx]; k = key[idx]; yy = Y[idx] == 1
        bc = lambda m: np.bincount(k[m], minlength=C * J).reshape(C, J)
        nj, na, ka = bc(np.ones(N, bool)), bc(Di), bc(Di & yy)
        
        E = {c: endpoints(Cell(nj[c], na[c], ka[c]), dlt, M_MIN) if nj[c].sum() > 0 else None for c in others + [ref]}
        er = E[ref]
        g_ = lambda key_: np.array([E[c][key_] if E[c] else np.nan for c in others])
        Lg, Ug, sLg, sUg = g_("L"), g_("U"), g_("se_L"), g_("se_U")
        dlo, dhi = Lg - er["U"], Ug - er["L"]
        slo, shi = np.maximum(np.sqrt(sLg**2 + er["se_U"]**2), 1e-9), np.maximum(np.sqrt(sUg**2 + er["se_L"]**2), 1e-9)
        
        for tau in TAUS:
            p_lo, p_hi = 1 - ndtr((dlo - tau) / slo), ndtr((dhi + tau) / shi)
            p_par = np.maximum(ndtr((dhi - tau) / shi), 1 - ndtr((dlo + tau) / slo))
            pv = np.nan_to_num(np.concatenate([np.minimum(p_lo, p_hi), p_par]), nan=1.0)
            
            for proc, rej in [("none", pv <= ALPHA), ("holm", holm(pv, ALPHA))]:
                cert_disp[tau][proc][iN] += (rej[:K] & (dstar[tau][0] > dlt))
                cert_par[tau][proc][iN] += (rej[K:] & (dstar[tau][1] > dlt))

# Compile results
rows = []
for tau in TAUS:
    for c_idx, c in enumerate(others):
        row = dict(gamma=G, tau=tau, cell=names[c], Nstar_disp=Nstar[(c_idx, tau, "disp")], Nstar_par=Nstar[(c_idx, tau, "par")])
        
        for proc in ["none", "holm"]:
            for iN, N in enumerate(N_LIST):
                row[f"pow_disp_{proc}_N{N}"] = cert_disp[tau][proc][iN, c_idx] / R
                row[f"pow_par_{proc}_N{N}"] = cert_par[tau][proc][iN, c_idx] / R
        
        # Observed crossover (first N with power >= 0.5)
        for proc in ["none", "holm"]:
            pd_arr = [row[f"pow_disp_{proc}_N{N}"] for N in N_LIST]
            pp_arr = [row[f"pow_par_{proc}_N{N}"] for N in N_LIST]
            row[f"obs_cross_disp_{proc}"] = next((N for N, p in zip(N_LIST, pd_arr) if p >= 0.5), np.inf)
            row[f"obs_cross_par_{proc}"] = next((N for N, p in zip(N_LIST, pp_arr) if p >= 0.5), np.inf)
            
        rows.append(row)

T = pd.DataFrame(rows)
T["cfg_hash"], T["git"] = CFG_HASH, GIT

print(f"\n--- Crossover Summary (Gamma={G}, Tau=.05) ---")
T_05 = T[T.tau == 0.05]
T_05_cert = T_05[T_05.Nstar_par != np.inf]
print(T_05_cert[["cell", "Nstar_par", "obs_cross_par_holm", "Nstar_disp", "obs_cross_disp_holm"]].head(10).to_string(index=False))

outfile = f"smoke tests/b8c_{STATE}_{SEED}_{FLIP_SEED}_g{G}_m{M_MIN}.csv"
T.to_csv(outfile, index=False)
print(f"\nsaved {outfile}")