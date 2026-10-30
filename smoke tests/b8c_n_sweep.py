"""
B8c: N sweep, observed per-cell crossover vs closed-form N*.
Needs bounds.py, common.py, multiplicity.py and the a12 CSV for the SAME (STATE, SEED, FLIP_SEED, GAMMA, M_MIN).
Env: STATE (CA), SEED (0), FLIP_SEED (123), GAMMA (0.02), R (200)
"""
import os, numpy as np, pandas as pd
from bounds import Cell, endpoints
from multiplicity import holm, cell_pvalues, family_pvalues
from common import build, J, M_MIN_DEFAULT

try: 
    GIT = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()[:10] + ("+dirty" if subprocess.check_output(["git","status","--porcelain"], text=True).strip() else "")
except Exception: 
    GIT = "nogit"
    
STATE, SEED, FLIP_SEED = os.environ.get("STATE", "CA"), int(os.environ.get("SEED", 0)), int(os.environ.get("FLIP_SEED", 123))
G, R, M_MIN, ALPHA = float(os.environ.get("GAMMA", 0.02)), int(os.environ.get("R", 200)), M_MIN_DEFAULT, 0.05   # M_MIN fixed at 10 (E4 decision)
TAUS = [0.05, 0.02]
N_LIST = [5000, 10000, 20000, 35000, 50000, 75000, 100000, 150000, 195665]
S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED)
Y, n, C, names, ref, others, Dtrue = S.Y, S.n, S.C, S.names, S.ref, S.others, S.Dtrue
key, K = S.key, len(others)
D, prop = S.select(G, S.eps0); gr = S.grouped(key, C * J, D, prop); dlt = S.amax(S.dgap(gr)[1])
print(f"gamma={G} oracle delta={dlt:.3f} R={R} K={K} M_MIN={M_MIN}")

rng2 = np.random.default_rng(30_000 + SEED)
rej = {(N, t, p): np.zeros((R, 2 * K), bool) for N in N_LIST for t in TAUS for p in ("unc", "holm")}
for N in N_LIST:
    for i in range(R):
        idx = rng2.integers(0, n, N); Di = rng2.random(N) < prop[idx]; k = key[idx]; yy = Y[idx] == 1
        bc = lambda m: np.bincount(k[m], minlength=C * J).reshape(C, J)
        nj, na, ka = bc(np.ones(N, bool)), bc(Di), bc(Di & yy)
        E = {c: endpoints(Cell(nj[c], na[c], ka[c]), dlt, M_MIN) if nj[c].sum() > 0 else None for c in others + [ref]}
        er = E[ref]; g_ = lambda f: np.array([E[c][f] if E[c] else np.nan for c in others])
        dlo, dhi = g_("L") - er["U"], g_("U") - er["L"]
        slo, shi = np.sqrt(g_("se_L")**2 + er["se_U"]**2), np.sqrt(g_("se_U")**2 + er["se_L"]**2)
        for t in TAUS:
            pv = family_pvalues(*cell_pvalues(dlo, dhi, slo, shi, t))
            rej[(N, t, "unc")][i] = pv <= ALPHA; rej[(N, t, "holm")][i] = holm(pv, ALPHA)
    print(f"N={N} done")

# per-cell power table
rows = []
for N in N_LIST:
    for t in TAUS:
        for ci, c in enumerate(others):
            r = dict(gamma=G, N=N, tau=t, cell=names[c], Dtrue=Dtrue[c])
            for p in ("unc", "holm"):
                m = rej[(N, t, p)].mean(0); r[f"pow_disp_{p}"], r[f"pow_par_{p}"] = m[ci], m[K + ci]
            rows.append(r)
P = pd.DataFrame(rows); P.to_csv(f"smoke tests/b8c_power_{STATE}_{SEED}_{FLIP_SEED}_g{G}.csv", index=False)

# observed crossover (power >= 0.5, log-N interpolation) vs N*
def cross(Ns, pw):
    pw = np.asarray(pw)
    if np.isnan(pw).any(): return np.nan, "nan"
    if pw[0] >= .5: return Ns[0], "left-censored" # already >=0.5 at smallest N: only an upper bound on N50
    j = np.flatnonzero(pw >= .5)
    if len(j) == 0: return np.inf, "right-censored"
    j = j[0]; a, b = np.log(Ns[j-1]), np.log(Ns[j])
    return float(np.exp(a + (.5 - pw[j-1]) / (pw[j] - pw[j-1]) * (b - a))), "ok"

A = pd.read_csv(f"smoke tests/a12_controlled_{STATE}_{SEED}_{FLIP_SEED}.csv"); A = A[np.isclose(A.gamma, G)]
out = []
for t in TAUS:
    for kind, col in (("disp", "pow_disp_unc"), ("par", "pow_par_unc")):
        for _, a in A.iterrows():
            Nstar = a[f"Nstar_{kind}_unc@{t}"]
            if not (N_LIST[0] <= Nstar <= N_LIST[-1]): continue            # only observable range
            sub = P[(P.tau == t) & (P.cell == a.cell)].sort_values("N")
            n50, flag = cross(sub.N.values, sub[col].values)
            out.append(dict(tau=t, kind=kind, cell=a.cell, Nstar=Nstar, N50=n50, flag=flag, ratio=n50 / Nstar,
                            ok=(flag == "ok") and (0.5 <= n50 / Nstar <= 2),
                            definite_fail=(flag == "right-censored") and Nstar < N_LIST[-1] / 2,))
O = pd.DataFrame(out); pd.set_option("display.width", 200)
print(O.round(2).to_string(index=False))
if len(O): 
    print("\npass (within 2x), flag=ok only:\n", O[O.flag == "ok"].groupby(["tau", "kind"]).ok.agg(["mean", "size"]).round(2))
    print("definite fails (right-censored, N* < pool/2):", int(O.definite_fail.sum()))

O["git"] = GIT
O.to_csv(f"smoke tests/b8c_crossover_{STATE}_{SEED}_{FLIP_SEED}_g{G}.csv", index=False)