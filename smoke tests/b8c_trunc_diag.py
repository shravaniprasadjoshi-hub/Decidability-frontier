"""
b8c_trunc_diag.py: POST-HOC diagnostic, not pre-registered. Does a truncation-aware N* explain the b8c crossover failures?
Truncation-aware N*: scale the pool counts of (cell, ref) by N/n_pool, apply m_min=10 (expected-count truncation), and take the
smallest N from which the certificate condition (z=1.645, assumed delta) holds for all larger N on a log grid.
Reads smoke tests/b8c_crossover_{STATE}_{SEED}_{FLIP}_g{GAMMA}.csv (all that exist), writes smoke tests/b8c_trunc_diag_{STATE}.csv.
Needs bounds.py, common.py. Env: STATE (CA).  Run: python "smoke tests/b8c_trunc_diag.py"
"""
import os, re, glob, numpy as np, pandas as pd
from bounds import Cell, gap
from common import build, J, M_MIN_DEFAULT

STATE, Z, M_MIN = os.environ.get("STATE", "CA"), 1.645, M_MIN_DEFAULT
NGRID = np.geomspace(2000, 400000, 150)
built = {}
out = []

for f in sorted(glob.glob(f"smoke tests/b8c_crossover_{STATE}_*_g*.csv")):
    m = re.search(rf"b8c_crossover_{STATE}_(\d+)_(\d+)_g(\d\.\d+)\.csv$", f.replace("\\", "/"))
    if not m: continue
    seed, flip, G = int(m[1]), int(m[2]), float(m[3])
    if (seed, flip) not in built: built[(seed, flip)] = build(state=STATE, seed=seed, flip_seed=flip)
    S = built[(seed, flip)]; n, C = S.n, S.C; names = list(S.names)
    D, prop = S.select(G, S.eps0); gr = S.grouped(S.key, C * J, D, prop); dlt = S.amax(S.dgap(gr)[1])
    cnt = {k: gr[k].reshape(C, J) for k in ("n", "na", "ka")}
    cell = lambda c, s=1.0: Cell(cnt["n"][c] * s, cnt["na"][c] * s, cnt["ka"][c] * s)
    ref = S.ref

    def passes(c, kind, t, s):
        r = gap(cell(c, s), cell(ref, s), dlt, 0.0, M_MIN)
        lo, hi = r["lo"] - Z * r["se_lo"], r["hi"] + Z * r["se_hi"]
        return (lo > t or hi < -t) if kind == "disp" else (hi <= t and lo >= -t)

    def nstar_trunc(c, kind, t):
        ok = np.array([passes(c, kind, t, N / n) for N in NGRID]); bad = np.flatnonzero(~ok)
        if len(bad) == 0: return NGRID[0]
        return np.inf if bad[-1] == len(NGRID) - 1 else NGRID[bad[-1] + 1]

    def nstar_cf(c, kind, t):          # original closed form, only to verify I rebuilt the same pool counts as a12
        r = gap(cell(c), cell(ref), dlt, 0.0, M_MIN)
        ns = lambda sl, se: n * (Z * se / sl) ** 2 if sl > 0 else np.inf
        return (min(ns(r["lo"] - t, r["se_lo"]), ns(-r["hi"] - t, r["se_hi"])) if kind == "disp"
                else max(ns(t - r["hi"], r["se_hi"]), ns(t + r["lo"], r["se_lo"])))

    d = pd.read_csv(f)
    for _, r in d.iterrows():
        c = names.index(r.cell); nt = nstar_trunc(c, r.kind, r.tau)
        out.append(dict(seed=seed, gamma=G, tau=r.tau, kind=r.kind, cell=r.cell, n_cell=int(cnt["n"][c].sum()), Nstar=r.Nstar,
                        rebuild_relerr=abs(nstar_cf(c, r.kind, r.tau) / r.Nstar - 1), ng_at_Nstar=cnt["n"][c].sum() * r.Nstar / n,
                        Nstar_trunc=nt, N50=r.N50, flag=r.flag))
    print("done", f)

T = pd.DataFrame(out)
T["ratio"] = T.N50 / T.Nstar; T["ratio_trunc"] = T.N50 / T.Nstar_trunc
T["ok"] = (T.flag == "ok") & T.ratio.between(0.5, 2); T["ok_trunc"] = (T.flag == "ok") & T.ratio_trunc.between(0.5, 2)
T.to_csv(f"smoke tests/b8c_trunc_diag_{STATE}.csv", index=False)
pd.set_option("display.width", 200)
print(f"\nrebuild check (closed form vs CSV Nstar): max rel err {T.rebuild_relerr[np.isfinite(T.rebuild_relerr)].max():.2e}  (should be ~0; else counts differ from the a12 run)")
o = T[T.flag == "ok"]
print("\npass within 2x, flag=ok only (orig N* vs truncation-aware N*):")
print(o.groupby(["kind", "tau"]).agg(n=("ok", "size"), orig=("ok", "mean"), trunc=("ok_trunc", "mean"),
                                    med_ratio=("ratio", "median"), med_ratio_trunc=("ratio_trunc", "median")).round(2))
print("\nby expected cell rows at the ORIGINAL N*:")
print(o.groupby(pd.cut(o.ng_at_Nstar, [0, 150, 300, 600, 1e9])).agg(n=("ok", "size"), orig=("ok", "mean"), trunc=("ok_trunc", "mean"),
                                                                       med_ratio=("ratio", "median"), med_ratio_trunc=("ratio_trunc", "median")).round(2))
print("\nrows failing under the original rule:")
print(o[~o.ok][["seed", "gamma", "tau", "cell", "n_cell", "ng_at_Nstar", "Nstar", "Nstar_trunc", "N50", "ratio", "ratio_trunc"]].round(2).to_string(index=False))
c = T[T.flag != "ok"]
print(f"\nright-censored: {len(c)}; definite fails (Nstar < pool/2): {(c.Nstar < 195665 / 2).sum()}; with trunc N*: {(c.Nstar_trunc < 195665 / 2).sum()}")