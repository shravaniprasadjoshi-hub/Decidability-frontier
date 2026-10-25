"""ST3: per-cell table + per-cell/propensity oracle delta + J-floor + clipped-width check (pool level, N = infinity).
Run (Colab):  pip install folktables scikit-learn pandas numpy scipy ; python st3_cell_table.py
Needs bounds.py in the same folder. Env overrides: STATE (default CA), SEED (default 0), SYNTH=1 (offline logic test only).
Regime = decision D3: gt50k, stratum mode, J = J_SEL = 5, noise 1.0, 60% approved. Selection rng stream matches ST2 (same seed => same D).
Answers: (1) how many cells have |Delta_obs| < tau at all?  (2) is per-cell delta larger than the pooled max|d_j| ST2 used?
(3) per-cell breakdown deltas vs the gamma grid  (4) is the J=10 gamma=0 floor just noise?  (5) does clipping tighten anything?"""
import os, numpy as np, pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier
from bounds import Cell, endpoints, clipped_bounds, breakdown

# ---------------- CONFIG ----------------
STATE, YEAR = os.environ.get("STATE", "CA"), "2018"
SEED = int(os.environ.get("SEED", 0))
Y_THR, NOISE_SD, J, J_SEL, APPROVAL = 50000, 1.0, 5, 5, 0.60
GAMMAS, SHOW_ROWS_FOR = [0.0, 0.02, 0.05, 0.1], [0.05]
TAUS, M_MIN, MIN_PER_SIDE = [0.05, 0.02], 10, 30
J_FLOOR_LIST, N_FLOOR_SEEDS = [3, 5, 10], 20
rng = np.random.default_rng(SEED)

def load():
    if os.environ.get("SYNTH"):
        n = 60000
        X = pd.DataFrame({"AGEP": rng.integers(18, 80, n), "SEX": rng.integers(1, 3, n),
                          "RAC1P": rng.choice([1, 2, 6, 8], n, p=[.6, .1, .15, .15]),
                          "SCHL": rng.integers(1, 25, n), "WKHP": rng.integers(1, 70, n), "MAR": rng.integers(1, 6, n)})
        lin = -3 + .04 * X.WKHP + .08 * X.SCHL - .01 * np.abs(X.AGEP - 45) + .4 * (X.SEX == 1)
        y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
        return X, np.where(y == 1, 50001 + rng.exponential(40000, n), rng.uniform(1000, 50000, n))
    from folktables import ACSDataSource, ACSIncome
    acs = ACSDataSource(survey_year=YEAR, horizon="1-Year", survey="person").get_data(states=[STATE], download=True)
    X, y, _ = ACSIncome.df_to_pandas(acs)
    pinc = ACSIncome._preprocess(acs)["PINCP"].values
    assert len(pinc) == len(X) and ((pinc > 50000).astype(int) == y.iloc[:, 0].values).all(), "PINCP misaligned"
    return X.reset_index(drop=True), pinc

X, pinc = load()
Y = (pinc > Y_THR).astype(int); n = len(Y)
print(f"STATE={STATE} SEED={SEED} n={n} base rate={Y.mean():.3f}")

# ---- cells, score, hidden residual: identical to ST2 ----
race = np.select([X.RAC1P == 1, X.RAC1P == 2, X.RAC1P == 6], ["W", "B", "A"], "O")
age = pd.cut(X.AGEP, [0, 29, 44, 59, 200], labels=["<30", "30-44", "45-59", "60+"]).astype(str)
cell = pd.Series(X.SEX.astype(str).values + "|" + race + "|" + age.values)
codes, names = pd.factorize(cell); C = len(names)
feats = X.drop(columns=[c for c in ["SEX", "RAC1P"] if c in X])
tr = rng.random(n) < 0.5
p = np.clip(HistGradientBoostingClassifier(max_iter=150, random_state=SEED).fit(feats[tr], Y[tr]).predict_proba(feats)[:, 1], 1e-3, 1 - 1e-3)
r = (Y - p) / np.sqrt(p * (1 - p)); r = (r - r.mean()) / r.std()
eps0 = rng.standard_normal(n)

def strata(Jx): return np.digitize(p, np.quantile(p, np.linspace(0, 1, Jx + 1)[1:-1]))
base = (lambda s: (s - s.mean()) / s.std())(strata(J_SEL).astype(float))

def select(gamma, eps):
    """D and the TRUE propensity P(D=1 | X, Y) = Phi((q - base - gamma r)/noise_sd), q = realized approval threshold."""
    lat = base + gamma * r + NOISE_SD * eps
    q = np.quantile(lat, APPROVAL)
    return (lat <= q).astype(int), ndtr((q - base - gamma * r) / NOISE_SD)

def grouped(key, size, D, prop):
    """per-group counts: n, approved, defaults|approved, rejected, defaults|rejected, and propensity-weighted sums."""
    bc = lambda m, w=None: np.bincount(key[m], weights=None if w is None else w[m], minlength=size)
    a, b, one = D == 1, D == 0, np.ones(len(D), bool)
    return dict(n=bc(one), na=bc(a), ka=bc(a & (Y == 1)), nr=bc(b), kr=bc(b & (Y == 1)),
                A=bc(one, prop), Ay=bc(one, prop * Y), B=bc(one, 1 - prop), By=bc(one, (1 - prop) * Y))

def dgap(g):
    with np.errstate(all="ignore"):
        real = np.where((g["na"] >= MIN_PER_SIDE) & (g["nr"] >= MIN_PER_SIDE), g["kr"] / g["nr"] - g["ka"] / g["na"], np.nan)
        pop = np.where((g["A"] >= MIN_PER_SIDE) & (g["B"] >= MIN_PER_SIDE), g["By"] / g["B"] - g["Ay"] / g["A"], np.nan)
    return real, pop

amax = lambda a: float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else np.nan

# ================= PART A: per-cell table =================
ref = int(np.bincount(codes).argmax())
print(f"cells={C}; reference = largest cell '{names[ref]}' (n={int((codes == ref).sum())}); K tested = {C - 1}")
s5 = strata(J); key = codes * J + s5
theta = np.array([Y[codes == c].mean() for c in range(C)])
rows = []
for g in GAMMAS:
    D, prop = select(g, eps0)
    gr = grouped(key, C * J, D, prop)
    d_real, d_pop = (a.reshape(C, J) for a in dgap(gr))
    pooled_real, pooled_pop = (amax(a) for a in dgap(grouped(s5, J, D, prop)))
    dsh_real, dsh_pop = amax(d_real), amax(d_pop)
    per_cell_pop = np.array([amax(d_pop[c]) for c in range(C)])
    print(f"\n===== GAMMA={g}: approval {D.mean():.3f}")
    print(f"  delta candidates: pooled max|d_j| realized {pooled_real:.3f} | pooled propensity {pooled_pop:.3f} | "
          f"max over cell x stratum realized {dsh_real:.3f} | propensity {dsh_pop:.3f}")
    print(f"  per-cell propensity delta: median {np.nanmedian(per_cell_pop):.3f}, max {np.nanmax(per_cell_pop):.3f}; "
          f"cells exceeding pooled propensity delta: {int((per_cell_pop > pooled_pop + 1e-9).sum())}/{C}")
    cl = lambda c: Cell(gr["n"].reshape(C, J)[c], gr["na"].reshape(C, J)[c], gr["ka"].reshape(C, J)[c])
    cref = cl(ref); dlt = dsh_pop if np.isfinite(dsh_pop) else pooled_pop
    for c in range(C):
        if c == ref: continue
        cg = cl(c); eg, er = endpoints(cg, 0.0, M_MIN), endpoints(cref, 0.0, M_MIN)
        row = dict(gamma=g, cell=names[c], n=int(cg.n), theta=theta[c], Delta_true=theta[c] - theta[ref],
                   Dc=eg["centre"] - er["centre"], S=eg["S"] + er["S"], u_g=eg["u"], u_ref=er["u"], delta_cell_pop=per_cell_pop[c])
        for t in TAUS:
            b = breakdown(cg, cref, t, 0.0, M_MIN); row[f"d*_disp@{t}"], row[f"d*_par@{t}"] = b["disparity"], b["parity"]
        lo, hi = endpoints(cg, dlt, M_MIN)["L"], endpoints(cg, dlt, M_MIN)["U"]; clo, chi = clipped_bounds(cg, dlt, M_MIN)
        row["clip_gain%"] = 100 * (1 - (chi - clo) / (hi - lo)) if hi - lo > 1e-3 else np.nan; row["delta_shared"] = dlt
        rows.append(row)
    R = pd.DataFrame([x for x in rows if x["gamma"] == g])
    for t in TAUS:
        print(f"  tau={t}: |Dc|<tau in {(R.Dc.abs() < t).sum()}/{len(R)} cells | parity feasible at delta=0 in {(R[f'd*_par@{t}'] >= 0).sum()} | "
              f"parity certifiable at shared delta={dlt:.3f}: {(R[f'd*_par@{t}'] > dlt).sum()} | disparity certifiable: {(R[f'd*_disp@{t}'] > dlt).sum()}")
    print(f"  clipped-vs-unclipped width gain at delta={dlt:.3f}: median {R['clip_gain%'].median():.2f}%  max {R['clip_gain%'].max():.2f}%")
    if g in SHOW_ROWS_FOR:
        pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
        print(R.drop(columns=["gamma", "delta_shared"]).to_string(index=False))
out = f"st3_cell_table_{STATE}_{SEED}.csv"; pd.DataFrame(rows).to_csv(out, index=False); print(f"\nsaved {out}")

# ================= PART B: J-floor at gamma=0 =================
print("\n===== J-FLOOR at gamma=0 (stratum selection on 5 bins; realized = mean+-sd over fresh selection noise; propensity = noise-free)")
D0, prop0 = select(0.0, eps0)
for Jx in J_FLOOR_LIST:
    sJ = strata(Jx); real = []
    for k in range(N_FLOOR_SEEDS):
        D, prop = select(0.0, np.random.default_rng(1000 + k).standard_normal(n))
        real.append(amax(dgap(grouped(sJ, Jx, D, prop))[0]))
    pop = amax(dgap(grouped(sJ, Jx, D0, prop0))[1])
    print(f"  J={Jx:2d}: realized max|d_j| {np.mean(real):.4f} +- {np.std(real):.4f} | propensity max|d_j| {pop:.4f}")
print("  Expect propensity ~0 for J=5,10 (bins nest in the 5 selection bins) and >0 for J=3 (coarser). If realized J=10 ~ 0.015-0.02 with propensity ~0, the floor is sampling noise.")
