"""
b8c_trunc_diag.py: 
POST-HOC diagnostic, not pre-registered. Arms: orig N*, truncation-aware N* (trunc), and the same two on EXPECTED counts under the audit DGP (exp: na=A, ka=Ay, since b8c redraws D ~ Bern(prop) instead of using the pool's realized D). Does a truncation-aware N* explain the b8c crossover failures?
Truncation-aware N*: scale the pool counts of (cell, ref) by N/n_pool, apply m_min=10 (expected-count truncation), and take the
smallest N from which the certificate condition (z=1.645, assumed delta) holds for all larger N on a log grid.
Reads smoke tests/results/b8c_crossover_{STATE}_{SEED}_{FLIP}_g{GAMMA}.csv (all that exist), writes smoke tests/results/b8c_trunc_diag_{STATE}.csv.
Needs bounds.py, common.py. Env: STATE (CA).  Run: python "smoke tests/b8c_trunc_diag.py"
"""
import os, re, glob, numpy as np, pandas as pd
pd.set_option("display.max_columns", 30)

# NOTE: Imported nstar_trunc_exp from bounds
from bounds import Cell, gap, nstar_trunc_exp
from common import build, J, M_MIN_DEFAULT, git_state

STATE, Z, M_MIN = os.environ.get("STATE", "CA"), 1.645, M_MIN_DEFAULT
built = {}
out = []
GIT = git_state()

for f in sorted(glob.glob(f"smoke tests/results/b8c_crossover_{STATE}_*_g*.csv")):
    m = re.search(rf"b8c_crossover_{STATE}_(\d+)_(\d+)_g(\d\.\d+)\.csv$", f.replace("\\", "/"))
    if not m: continue
    seed, flip, G = int(m[1]), int(m[2]), float(m[3])
    if (seed, flip) not in built: built[(seed, flip)] = build(state=STATE, seed=seed, flip_seed=flip)
    S = built[(seed, flip)]; n, C = S.n, S.C; names = list(S.names)
    D, prop = S.select(G, S.eps0); gr = S.grouped(S.key, C * J, D, prop); dlt = S.amax(S.dgap(gr)[1])
    cnt = {k: gr[k].reshape(C, J) for k in ("n", "na", "ka")}
    cexp = {"n": cnt["n"], "na": gr["A"].reshape(C, J), "ka": gr["Ay"].reshape(C, J)}   # expected counts under the audit DGP
    cell = lambda c, s=1.0, E=False: Cell(*[(cexp if E else cnt)[k][c] * s for k in ("n", "na", "ka")])
    ref = S.ref

    def nstar_cf(c, kind, t, E=False):  # closed form (E=False reproduces the a12 N*; E=True = same formula on expected counts)
        r = gap(cell(c, 1.0, E), cell(ref, 1.0, E), dlt, 0.0, M_MIN)
        ns = lambda sl, se: n * (Z * se / sl) ** 2 if sl > 0 else np.inf
        return (min(ns(r["lo"] - t, r["se_lo"]), ns(-r["hi"] - t, r["se_hi"])) if kind == "disp"
                else max(ns(t - r["hi"], r["se_hi"]), ns(t + r["lo"], r["se_lo"])))

    d = pd.read_csv(f)
    cfg_hash = d['cfg_hash'].iloc[0] if 'cfg_hash' in d.columns else "unknown"
    
    for _, r in d.iterrows():
        c = names.index(r.cell)
        nt = nstar_trunc_exp(...)
        nte = nstar_trunc_exp(...)
        
        out.append(dict(seed=seed, flip=flip, gamma=G, tau=r.tau, kind=r.kind, cell=r.cell, 
                        n_cell=int(cnt["n"][c].sum()), Nstar=r.Nstar,
                        rebuild_relerr=abs(nstar_cf(c, r.kind, r.tau) / r.Nstar - 1), 
                        ng_at_Nstar=cnt["n"][c].sum() * r.Nstar / n,
                        Nstar_trunc=nt, Nstar_exp=nstar_cf(c, r.kind, r.tau, True), 
                        Nstar_trunc_exp=nte, N50=r.N50, flag=r.flag,
                        git=GIT, cfg_hash=cfg_hash))
    print("done", f)

T = pd.DataFrame(out)
# add flags logic
T["ratio"] = T.N50 / T.Nstar
T["ratio_trunc"] = T.N50 / T.Nstar_trunc
T["ratio_exp"] = T.N50 / T.Nstar_exp
T["ratio_trunc_exp"] = T.N50 / T.Nstar_trunc_exp
T["ok"] = (T.flag == "ok") & T.ratio.between(0.5, 2)
T["ok_trunc"] = (T.flag == "ok") & T.ratio_trunc.between(0.5, 2)
T["ok_exp"] = (T.flag == "ok") & T.ratio_exp.between(0.5, 2)
T["ok_trunc_exp"] = (T.flag == "ok") & T.ratio_trunc_exp.between(0.5, 2)

T.to_csv(f"smoke tests/results/b8c_trunc_diag_{STATE}.csv", index=False)

print(f"\nrebuild check (closed form vs CSV Nstar): max rel err {T.rebuild_relerr[np.isfinite(T.rebuild_relerr)].max():.2e}")
print("\nNOTE: Selection is conditional on the corrected N* (Nstar_trunc_exp) from a12. This is not an independent test of all possible cells.")
print("\nNOTE: Parity ratios (N50/N*) are ~1.3-1.5 because parity is an IUT (both one-sided tests must reject).")

o = T[T.flag == "ok"]
print("\npass within 2x, flag=ok only:")
print(o.groupby(["kind", "tau"]).agg(
    n=("ok", "size"), 
    ok_flag=("flag", lambda x: (x == "ok").sum()),
    left=("flag", lambda x: (x == "left-censored").sum()),
    right=("flag", lambda x: (x == "right-censored").sum()),
    nan=("flag", lambda x: (x == "nan").sum()),
    orig=("ok", "mean"), trunc=("ok_trunc", "mean"), exp=("ok_exp", "mean"), trunc_exp=("ok_trunc_exp", "mean"),
    med_ratio=("ratio", "median"), med_trunc_exp=("ratio_trunc_exp", "median")
).round(2))

print("\nDistinct (seed, flip, cell) passing 2x rule:")
print(o[o.ok_trunc_exp].groupby(["kind", "tau"]).agg(distinct_cells=("cell", "nunique")))

c = T[T.flag != "ok"]
pool_size = S.n  # Removed hardcoded 195665
print(f"\nright-censored: {len(c)}; definite fails (Nstar < pool/2): {(c.Nstar < pool_size / 2).sum()}; with trunc_exp N*: {(c.Nstar_trunc_exp < pool_size / 2).sum()}")