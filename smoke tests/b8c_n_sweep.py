"""
B8c: N sweep, observed per-cell crossover vs closed-form N*.
Needs bounds.py, common.py, multiplicity.py and the a12 CSV for the SAME (STATE, SEED, FLIP_SEED, GAMMA, M_MIN).
Env: STATE (CA), SEED (0), FLIP_SEED (123), GAMMA (0.02), R (200)
Run: python "smoke tests/b8c_n_sweep.py"

Cell selection and pass rule use the CORRECTED N* (a12 column Nstar_trunc_exp_{kind}@tau: truncation-aware, expected counts, z=1.645 only).
The ORIGINAL closed-form N* (a12 column Nstar_{kind}_unc@tau) is written to the output as `Nstar`, so b8c_trunc_diag.py can still
compare the original rule against the corrections. Only cells with corrected N* in [N_LIST[0], N_LIST[-1]] are tested.
N* = 'last failing N on the grid + 1'; the observed crossover = 'first N with power >= 0.5'. They differ if the verdict is non-monotone in N (m_min truncation).
"""
import os, numpy as np, pandas as pd
from bounds import Cell, endpoints, definite_fail
from multiplicity import holm, cell_pvalues, family_pvalues
from common import build, J, M_MIN_DEFAULT, git_state as _git

STATE, SEED, FLIP_SEED = os.environ.get("STATE", "CA"), int(os.environ.get("SEED", 0)), int(os.environ.get("FLIP_SEED", 123))
G, R, M_MIN, ALPHA = float(os.environ.get("GAMMA", 0.02)), int(os.environ.get("R", 200)), M_MIN_DEFAULT, 0.05
TAUS = [0.05, 0.02]
N_BASE = [5000, 10000, 20000, 35000, 50000, 75000, 100000, 150000]

GIT = _git()
print(f"git={GIT}")

S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED)
N_LIST = N_BASE + [S.n] # last grid point = pool size, no hardcoded 195665
import json, hashlib
CFG = dict(script="b8c", state=STATE, seed=SEED, flip_seed=FLIP_SEED, gamma=G, R=R, m_min=M_MIN, taus=TAUS, N_list=N_LIST, alpha=ALPHA)
CFG_HASH = hashlib.sha1(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:8]
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

os.makedirs("smoke tests/results", exist_ok=True)

# per-cell power table
rows = []
for N in N_LIST:
    for t in TAUS:
        for ci, c in enumerate(others):
            r = dict(gamma=G, N=N, tau=t, cell=names[c], Dtrue=Dtrue[c])
            for p in ("unc", "holm"):
                m = rej[(N, t, p)].mean(0); r[f"pow_disp_{p}"], r[f"pow_par_{p}"] = m[ci], m[K + ci]
            rows.append(r)
P = pd.DataFrame(rows); P.to_csv(f"smoke tests/results/b8c_power_{STATE}_{SEED}_{FLIP_SEED}_g{G}.csv", index=False)

# observed crossover (power >= 0.5, log-N interpolation) vs N*
def cross(Ns, pw):
    pw = np.asarray(pw)
    if np.isnan(pw).any(): return np.nan, "nan"
    if pw[0] >= .5: return Ns[0], "left-censored"
    j = np.flatnonzero(pw >= .5)
    if len(j) == 0: return np.inf, "right-censored"
    j = j[0]; a, b = np.log(Ns[j-1]), np.log(Ns[j])
    return float(np.exp(a + (.5 - pw[j-1]) / (pw[j] - pw[j-1]) * (b - a))), "ok"

A = pd.read_csv(f"smoke tests/results/a12_controlled_{STATE}_{SEED}_{FLIP_SEED}.csv"); A = A[np.isclose(A.gamma, G)]
out = []
for t in TAUS:
    for kind, col in (("disp", "pow_disp_unc"), ("par", "pow_par_unc")):
        for _, a in A.iterrows():
            Nte = a[f"Nstar_trunc_exp_{kind}@{t}"]            # corrected N*: selection + pass rule
            if not (N_LIST[0] <= Nte <= N_LIST[-1]): continue
            Nor = a[f"Nstar_{kind}_unc@{t}"]                  # original closed-form N* (b8c_trunc_diag reads this as Nstar)
            sub = P[(P.tau == t) & (P.cell == a.cell)].sort_values("N")
            n50, flag = cross(sub.N.values, sub[col].values)
            out.append(dict(tau=t, kind=kind, cell=a.cell, Nstar=Nor, Nstar_trunc_exp=Nte, N50=n50, flag=flag, ratio=n50 / Nte,
                            ok=(flag == "ok") and (0.5 <= n50 / Nte <= 2),
                            definite_fail=definite_fail(flag, Nte, N_LIST[-1])))

O = pd.DataFrame(out); pd.set_option("display.width", 200)
print(O.round(2).to_string(index=False))

if len(O):
    # flag counts
    print("\nFlag counts by kind/tau:")
    print(O.groupby(["tau", "kind"]).agg(
        n=("ok", "size"),
        ok_flag=("flag", lambda x: (x == "ok").sum()),
        left=("flag", lambda x: (x == "left-censored").sum()),
        right=("flag", lambda x: (x == "right-censored").sum()),
        nan=("flag", lambda x: (x == "nan").sum())
    ))
    
    # distinct (seed, flip, cell) counts
    print("\nDistinct cells passing 2x rule (one seed/flip/gamma run; cells repeat across tau, so rows are not independent):")
    print(O[O.ok].groupby(["kind", "tau"]).agg(distinct_cells=("cell", "nunique")))
    
    # parity explanation
    print("\nNOTE: Parity ratios (N50/N*) are ~1.3-1.5 because parity is an IUT (intersection-union test). Both one-sided tests must reject, so power is ~0.5 at N*, meaning N50 must exceed N*.")
    
    pool_size = S.n  # removed hardcoded 195665
    print(f"\ndefinite fails (right-censored, N* < pool/2): {int(O.definite_fail.sum())} (pool/2 = {pool_size/2:.0f})")

O["git"] = GIT
O["cfg_hash"] = CFG_HASH
O.to_csv(f"smoke tests/results/b8c_crossover_{STATE}_{SEED}_{FLIP_SEED}_g{G}.csv", index=False)