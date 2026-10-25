"""
A12: controlled-Delta semi-synthetic arm (pool level, N = infinity, noise-free propensity). 
Extends smoke test 3 and needs bounds.py to be in the same folder.

usage: python a12_controlled_delta.py

Env: STATE (CA), SEED (0), FLIP_SEED (123), SYNTH=1 (offline logic test only)

Design: 
- ref = largest cell by count (Y-independent)
- Each non-ref cell gets a target Delta_g from GRID, assigned by a seeded permutation fixed ex ante (never looks at data)
- Y is thinned inside the cell to EXACT counts so theta_g = theta_ref + Delta_g (clipped to [.01,.99]), BEFORE the score is fitted
- Everything downstream (score p, strata, residual r, selection, oracle delta) is recomputed on the flipped Y, so delta is re-measured, not assumed ~gamma
- One arm, one grid, one seed set (decision from F9)

gives Delta_true (realized), Dc (observed-centre gap), S, u, breakdowns, certifiability at the recomputed shared delta
Selection rng stream matches st3/b8a up to the flip (flips use their own rng)

"""
import os, numpy as np, pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier
from bounds import Cell, endpoints, breakdown

# CONFIG
STATE, YEAR = os.environ.get("STATE", "CA"), "2018"
SEED, FLIP_SEED = int(os.environ.get("SEED", 0)), int(os.environ.get("FLIP_SEED", 123))
Y_THR, NOISE_SD, J, J_SEL, APPROVAL = 50000, 1.0, 5, 5, 0.60
GAMMAS, TAUS, M_MIN, MIN_PER_SIDE = [0.02, 0.05, 0.1], [0.05, 0.02], 10, 30
GRID = np.array([-.15, -.10, -.07, -.04, -.02, -.005, .005, .02, .04, .07, .10, .15])   # 6/12 within +-0.05
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

X, pinc = load(); Y0 = (pinc > Y_THR).astype(int); n = len(Y0)
race = np.select([X.RAC1P == 1, X.RAC1P == 2, X.RAC1P == 6], ["W", "B", "A"], "O")
age = pd.cut(X.AGEP, [0, 29, 44, 59, 200], labels=["<30", "30-44", "45-59", "60+"]).astype(str)
cell = pd.Series(X.SEX.astype(str).values + "|" + race + "|" + age.values)
codes, names = pd.factorize(cell); C = len(names)
ref = int(np.bincount(codes).argmax()); others = sorted([c for c in range(C) if c != ref], key=lambda c: names[c])

# controlled Delta: exact-count thinning inside each non-ref cell, fixed ex ante
frng = np.random.default_rng(FLIP_SEED)
perm = frng.permutation(len(others)); target = np.full(C, np.nan)
for i, c in enumerate(others): target[c] = GRID[perm[i] % len(GRID)]
Y = Y0.copy(); th_ref = Y0[codes == ref].mean()

for c in others:
    m = codes == c; nc = int(m.sum()); t = float(np.clip(th_ref + target[c], 0.01, 0.99)); k1 = int(round(t * nc))
    i1, i0 = np.flatnonzero(m & (Y0 == 1)), np.flatnonzero(m & (Y0 == 0))
    if k1 < len(i1):  Y[frng.choice(i1, len(i1) - k1, replace=False)] = 0
    elif k1 > len(i1): Y[frng.choice(i0, min(k1 - len(i1), len(i0)), replace=False)] = 1

theta0 = np.array([Y0[codes == c].mean() for c in range(C)]); theta = np.array([Y[codes == c].mean() for c in range(C)])
Dtrue = theta - theta[ref]

print(f"STATE={STATE} SEED={SEED} FLIP_SEED={FLIP_SEED} n={n} base rate {Y0.mean():.3f} -> {Y.mean():.3f}; ref='{names[ref]}' (n={int((codes==ref).sum())}), K={len(others)}")
print(f"target vs realized Delta_true: max|diff|={np.nanmax(np.abs(Dtrue[others]-target[others])):.4f} (clipping only); natural Delta median |.|={np.median(np.abs(theta0[others]-theta0[ref])):.3f} -> controlled {np.median(np.abs(Dtrue[others])):.3f}")
print(f"cells with |Delta_true|<=tau: " + ", ".join(f"tau={t}: {int((np.abs(Dtrue[others])<=t).sum())}/{len(others)}" for t in TAUS))

# score, hidden residual, selection: identical pipeline, on flipped Y
feats = X.drop(columns=[c for c in ["SEX", "RAC1P"] if c in X])
tr = rng.random(n) < 0.5
p = np.clip(HistGradientBoostingClassifier(max_iter=150, random_state=SEED).fit(feats[tr], Y[tr]).predict_proba(feats)[:, 1], 1e-3, 1 - 1e-3)
r = (Y - p) / np.sqrt(p * (1 - p)); r = (r - r.mean()) / r.std()
eps0 = rng.standard_normal(n)

def strata(Jx): return np.digitize(p, np.quantile(p, np.linspace(0, 1, Jx + 1)[1:-1]))
base = (lambda s: (s - s.mean()) / s.std())(strata(J_SEL).astype(float))

def select(gamma, eps):
    lat = base + gamma * r + NOISE_SD * eps; q = np.quantile(lat, APPROVAL)
    return (lat <= q).astype(int), ndtr((q - base - gamma * r) / NOISE_SD)

def grouped(key, size, D, prop):
    bc = lambda m, w=None: np.bincount(key[m], weights=None if w is None else w[m], minlength=size)
    a, b, one = D == 1, D == 0, np.ones(len(D), bool)
    return dict(n=bc(one), na=bc(a), ka=bc(a & (Y == 1)), A=bc(one, prop), Ay=bc(one, prop * Y), B=bc(one, 1 - prop), By=bc(one, (1 - prop) * Y))

def dpop(g):
    with np.errstate(all="ignore"):
        return np.where((g["A"] >= MIN_PER_SIDE) & (g["B"] >= MIN_PER_SIDE), g["By"] / g["B"] - g["Ay"] / g["A"], np.nan)

amax = lambda a: float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else np.nan

s5 = strata(J); key = codes * J + s5; rows = []
for g in GAMMAS:
    D, prop = select(g, eps0); gr = grouped(key, C * J, D, prop)
    dlt = amax(dpop(gr)); pooled = amax(dpop(grouped(s5, J, D, prop)))
    print(f"\n===== GAMMA={g}: approval {D.mean():.3f} | recomputed shared delta (propensity, max cell x stratum) = {dlt:.3f} (ratio to gamma {dlt/g:.2f}; pooled {pooled:.3f})")
    cl = lambda c: Cell(gr["n"].reshape(C, J)[c], gr["na"].reshape(C, J)[c], gr["ka"].reshape(C, J)[c])
    cref = cl(ref); er = endpoints(cref, 0.0, M_MIN); R_ = []
    
    for c in others:
        cg = cl(c); eg = endpoints(cg, 0.0, M_MIN)
        row = dict(gamma=g, cell=names[c], n=int(cg.n), target=target[c], Delta_true=Dtrue[c], Dc=eg["centre"] - er["centre"], S=eg["S"] + er["S"],
                   u_g=eg["u"], u_ref=er["u"], delta_shared=dlt)
    
        for t in TAUS:
            b = breakdown(cg, cref, t, 0.0, M_MIN); row[f"d*_disp@{t}"], row[f"d*_par@{t}"] = b["disparity"], b["parity"]
    
        R_.append(row)
    
    R = pd.DataFrame(R_); rows += R_
    print(f"  rho_ref(explored)={er['S']:.3f}; per-cell S median {R.S.median():.2f} (range {R.S.min():.2f}-{R.S.max():.2f}); |Dc - Delta_true| median {np.median(np.abs(R.Dc-R.Delta_true)):.3f}")
    
    for t in TAUS:
        pr, ds = R[f"d*_par@{t}"], R[f"d*_disp@{t}"]
        print(f"  tau={t}: |Dc|<tau {(R.Dc.abs()<t).sum()}/{len(R)} | parity feasible at delta=0 {(pr>=0).sum()} | parity certifiable at delta={dlt:.3f}: {(pr>dlt).sum()} | "
              f"disparity certifiable: {(ds>dlt).sum()} | d*_disp in [0.02,0.10]: {((ds>=0.02)&(ds<=0.10)).sum()} | d*_par in [0.005,0.10]: {((pr>=0.005)&(pr<=0.10)).sum()}")

pd.DataFrame(rows).to_csv(f"a12_controlled_{STATE}_{SEED}_{FLIP_SEED}.csv", index=False); print(f"\nsaved a12_controlled_{STATE}_{SEED}_{FLIP_SEED}.csv")
