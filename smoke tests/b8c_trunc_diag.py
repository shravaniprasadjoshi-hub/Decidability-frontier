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
from bounds import Cell, gap, nstar_trunc_exp, definite_fail
from common import build, J, M_MIN_DEFAULT, git_state

STATE, Z, M_MIN = os.environ.get("STATE", "CA"), 1.645, M_MIN_DEFAULT
built = {}
out = []
GIT = git_state()
in_gits, a12_gits = set(), set()

for f in sorted(glob.glob(f"smoke tests/results/b8c_crossover_{STATE}_*_g*.csv")):
    m = re.search(rf"b8c_crossover_{STATE}_(\d+)_(\d+)_g(\d\.\d+)\.csv$", f.replace("\\", "/"))
    if not m: continue
    seed, flip, G = int(m[1]), int(m[2]), float(m[3])
    if (seed, flip) not in built: built[(seed, flip)] = build(state=STATE, seed=seed, flip_seed=flip)
    S = built[(seed, flip)]; n, C = S.n, S.C; names = list(S.names)
    D, prop = S.select(G, S.eps0); gr = S.grouped(S.key, C * J, D, prop); dlt = S.amax(S.dgap(gr)[1])
    cnt = {k: gr[k].reshape(C, J) for k in ("n", "na", "ka")}
    A_, Ay_ = gr["A"].reshape(C, J), gr["Ay"].reshape(C, J)
    cexp = {"n": cnt["n"], "na": np.minimum(A_, cnt["n"]), "ka": np.minimum(Ay_, A_)}   # same clip as a12.cl_e
    cell = lambda c, s=1.0, E=False: Cell(*[(cexp if E else cnt)[k][c] * s for k in ("n", "na", "ka")])
    ref = S.ref

    def nstar_cf(c, kind, t, E=False):  # closed form (E=False reproduces the a12 N*; E=True = same formula on expected counts)
        r = gap(cell(c, 1.0, E), cell(ref, 1.0, E), dlt, 0.0, M_MIN)
        ns = lambda sl, se: n * (Z * se / sl) ** 2 if sl > 0 else np.inf
        return (min(ns(r["lo"] - t, r["se_lo"]), ns(-r["hi"] - t, r["se_hi"])) if kind == "disp"
                else max(ns(t - r["hi"], r["se_hi"]), ns(t + r["lo"], r["se_lo"])))

    d = pd.read_csv(f)
    missing = [c for c in ("cfg_hash", "git", "a12_git", "Nstar_trunc_exp") if c not in d.columns]
    assert not missing, f"{f}: stale crossover CSV (missing {missing}); delete it and rerun b8c"
    assert d["cfg_hash"].nunique() == 1 and d["git"].nunique() == 1 and d["a12_git"].nunique() == 1, f"{f}: mixed provenance inside one file"
    cfg_hash = d["cfg_hash"].iloc[0]; in_gits.add(d["git"].iloc[0]); a12_gits.add(d["a12_git"].iloc[0])
    
    for _, r in d.iterrows():
        c = names.index(r.cell)
        nt  = nstar_trunc_exp(cell(c), cell(ref), dlt, r.tau, r.kind, Z, M_MIN, n)                    # realized counts
        nte = nstar_trunc_exp(cell(c, 1.0, True), cell(ref, 1.0, True), dlt, r.tau, r.kind, Z, M_MIN, n)  # expected counts
        a12v = r.get("Nstar_trunc_exp", np.nan)                                                         # value b8c copied from a12
        chk = 0.0 if (np.isinf(nte) and np.isinf(a12v)) else abs(nte / a12v - 1)
        
        out.append(dict(seed=seed, flip=flip, gamma=G, tau=r.tau, kind=r.kind, cell=r.cell, 
                        n_cell=int(cnt["n"][c].sum()), Nstar=r.Nstar,
                        rebuild_relerr=abs(nstar_cf(c, r.kind, r.tau) / r.Nstar - 1), 
                        ng_at_Nstar=cnt["n"][c].sum() * r.Nstar / n,
                        Nstar_trunc=nt, Nstar_exp=nstar_cf(c, r.kind, r.tau, True), 
                        Nstar_trunc_exp=nte, N50=r.N50, flag=r.flag,
                        n_pool=n, rebuild_relerr_trunc=chk,
                        git=GIT, cfg_hash=cfg_hash))
    print("done", f)

assert len(in_gits) == 1 and in_gits == a12_gits, f"inputs come from different commits: b8c {sorted(in_gits)}, a12 {sorted(a12_gits)}"
assert not any("dirty" in g or g == "nogit" for g in in_gits), f"inputs are not from a clean commit: {sorted(in_gits)}"

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
T["sfc"] = T.seed.astype(str) + "_" + T.flip.astype(str) + "_" + T.cell   # distinct (seed, flip, cell) key
T["definite_fail"] = [definite_fail(f, x, m) for f, x, m in zip(T.flag, T.Nstar_trunc_exp, T.n_pool)]

T.to_csv(f"smoke tests/results/b8c_trunc_diag_{STATE}.csv", index=False)

print(f"\nrebuild check (closed form vs CSV Nstar): max rel err {T.rebuild_relerr[np.isfinite(T.rebuild_relerr)].max():.2e}")
print("\nNOTE: Selection is conditional on the corrected N* (Nstar_trunc_exp) from a12. This is not an independent test of all possible cells.")
print("\nNOTE: parity is an intersection-union test (both one-sided tests must reject), so power at N* is below 0.5 and N50/N* > 1 is expected. Observed range is printed below.")

o = T[T.flag == "ok"]
print("\nrebuild check (trunc_exp vs a12 column): max rel err",
      T.rebuild_relerr_trunc[np.isfinite(T.rebuild_relerr_trunc)].max())
flags = T.groupby(["kind", "tau"]).agg(
    rows=("flag", "size"), ok=("flag", lambda x: (x == "ok").sum()), left=("flag", lambda x: (x == "left-censored").sum()),
    right=("flag", lambda x: (x == "right-censored").sum()), nan=("flag", lambda x: (x == "nan").sum()),
    definite_fail=("definite_fail", "sum"))
rates = o.groupby(["kind", "tau"]).agg(
    orig=("ok", "mean"), trunc=("ok_trunc", "mean"), exp=("ok_exp", "mean"), trunc_exp=("ok_trunc_exp", "mean"),
    med_ratio=("ratio", "median"), med_trunc_exp=("ratio_trunc_exp", "median"),
    min_trunc_exp=("ratio_trunc_exp", "min"), max_trunc_exp=("ratio_trunc_exp", "max"))
print("\nflag counts (all rows) next to pass rates (flag=ok rows ONLY; censored rows are excluded from every rate):")
print(pd.concat([flags, rates], axis=1).round(2))
pr = o[o.kind == "par"].ratio_trunc_exp
print(f"parity N50/N*_trunc_exp over {len(pr)} ok rows: " + (f"min {pr.min():.2f}, median {pr.median():.2f}, max {pr.max():.2f}" if len(pr) else "none"))
d_all = T.groupby(["kind", "tau"]).sfc.nunique().rename("distinct_all")
d_pass = o[o.ok_trunc_exp].groupby(["kind", "tau"]).sfc.nunique().rename("distinct_pass")
d_fail = o[~o.ok_trunc_exp].groupby(["kind", "tau"]).sfc.nunique().rename("distinct_fail")
print("\ndistinct (seed, flip, cell) per group; rows share RNG streams across gamma and repeat across tau, so do not treat rows as independent:")
print(pd.concat([d_all, d_pass, d_fail], axis=1).fillna(0).astype(int))

print("\n=== per-arm split; (3, 321) is the confirmatory arm (corrections were fit on seeds 0-2) ===")
for (s_, f_), Ta in T.groupby(["seed", "flip"]):
    oa = Ta[Ta.flag == "ok"]
    print(f"\narm ({s_},{f_}): rows {len(Ta)}, ok {len(oa)}, left {(Ta.flag == 'left-censored').sum()}, right {(Ta.flag == 'right-censored').sum()}, nan {(Ta.flag == 'nan').sum()}, gammas {sorted(Ta.gamma.unique())}")
    print(oa.groupby(["kind", "tau"]).agg(n_ok=("ok_trunc_exp", "size"), pass_trunc_exp=("ok_trunc_exp", "mean"), pass_orig=("ok", "mean"),
          med=("ratio_trunc_exp", "median"), lo=("ratio_trunc_exp", "min"), hi=("ratio_trunc_exp", "max")).round(2))
print("\nright-censored rows:")
print(T[T.flag == "right-censored"][["seed", "flip", "gamma", "tau", "kind", "cell", "Nstar_trunc_exp", "N50"]].to_string(index=False))