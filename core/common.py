"""
shared setup for st3 / a12 / b8a / b8b / b8c

usage:  from common import build, J, M_MIN_DEFAULT
        S = build(state="CA", seed=0) # natural Y (st3, b8a)
        S = build(state="CA", seed=0, flip_seed=123)  # controlled-Delta arm (a12, b8b, b8c): Y thinned inside non-ref cells BEFORE the score is fitted

RNG ORDER is part of the contract: setup stream = default_rng(seed) -> [load (SYNTH only)] -> train mask -> eps0
DO NOT reorder these calls
Env SYNTH=1 gives an offline logic test only
returns a namespace: X, Y (flipped if flip_seed), Y0 (natural), n, codes, names, C, ref, others, target, theta, Dtrue, p, r, eps0, s5, key, strata(Jx), select(gamma, eps) -> (D, propensity), grouped(key, size, D, prop), dgap(g) -> (realized, propensity), amax(a)

"""

import os, numpy as np, pandas as pd
from types import SimpleNamespace
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier

Y_THR, NOISE_SD, J, J_SEL, APPROVAL = 50000, 1.0, 5, 5, 0.60
MIN_PER_SIDE, M_MIN_DEFAULT = 30, 10
GRID = np.array([-.15, -.10, -.07, -.04, -.02, -.005, .005, .02, .04, .07, .10, .15])   # 6/12 within +-0.05

def _load(state, year, rng):
    if os.environ.get("SYNTH"):
        n = 60000
        X = pd.DataFrame({"AGEP": rng.integers(18, 80, n), "SEX": rng.integers(1, 3, n),
                          "RAC1P": rng.choice([1, 2, 6, 8], n, p=[.6, .1, .15, .15]),
                          "SCHL": rng.integers(1, 25, n), "WKHP": rng.integers(1, 70, n), "MAR": rng.integers(1, 6, n)})
        
        lin = -3 + .04 * X.WKHP + .08 * X.SCHL - .01 * np.abs(X.AGEP - 45) + .4 * (X.SEX == 1)
        y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
        
        return X, np.where(y == 1, 50001 + rng.exponential(40000, n), rng.uniform(1000, 50000, n))
    
    from folktables import ACSDataSource, ACSIncome
    acs = ACSDataSource(survey_year=year, horizon="1-Year", survey="person").get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_pandas(acs)
    pinc = ACSIncome._preprocess(acs)["PINCP"].values
    
    assert len(pinc) == len(X) and ((pinc > 50000).astype(int) == y.iloc[:, 0].values).all(), "PINCP misaligned"
    return X.reset_index(drop=True), pinc

def build(state="CA", seed=0, flip_seed=None, grid=GRID, year="2018"):
    rng = np.random.default_rng(seed)
    X, pinc = _load(state, year, rng); Y0 = (pinc > Y_THR).astype(int); n = len(Y0)
    race = np.select([X.RAC1P == 1, X.RAC1P == 2, X.RAC1P == 6], ["W", "B", "A"], "O")
    age = pd.cut(X.AGEP, [0, 29, 44, 59, 200], labels=["<30", "30-44", "45-59", "60+"]).astype(str)
    codes, names = pd.factorize(pd.Series(X.SEX.astype(str).values + "|" + race + "|" + age.values)); C = len(names)
    ref = int(np.bincount(codes).argmax())                       # largest cell by count; never looks at Y
    others = sorted([c for c in range(C) if c != ref], key=lambda c: names[c])
    Y, target = Y0.copy(), np.full(C, np.nan)
    
    if flip_seed is not None:                                     # exact-count thinning inside each non-ref cell, targets fixed ex ante
        frng = np.random.default_rng(flip_seed); perm = frng.permutation(len(others)); th_ref = Y0[codes == ref].mean()
        
        for i, c in enumerate(others):
            target[c] = grid[perm[i] % len(grid)]
            m = codes == c; nc = int(m.sum()); k1 = int(round(float(np.clip(th_ref + target[c], 0.01, 0.99)) * nc))
            i1, i0 = np.flatnonzero(m & (Y0 == 1)), np.flatnonzero(m & (Y0 == 0))
            if k1 < len(i1): Y[frng.choice(i1, len(i1) - k1, replace=False)] = 0
            elif k1 > len(i1): Y[frng.choice(i0, min(k1 - len(i1), len(i0)), replace=False)] = 1
    
    theta = np.array([Y[codes == c].mean() for c in range(C)]); Dtrue = theta - theta[ref]
    feats = X.drop(columns=[c for c in ["SEX", "RAC1P"] if c in X])
    tr = rng.random(n) < 0.5
    p = np.clip(HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(feats[tr], Y[tr]).predict_proba(feats)[:, 1], 1e-3, 1 - 1e-3)
    r = (Y - p) / np.sqrt(p * (1 - p)); r = (r - r.mean()) / r.std()
    eps0 = rng.standard_normal(n)
    
    def strata(Jx): return np.digitize(p, np.quantile(p, np.linspace(0, 1, Jx + 1)[1:-1]))
    base = (lambda s: (s - s.mean()) / s.std())(strata(J_SEL).astype(float))
    
    def select(gamma, eps=eps0):
        """D and the TRUE propensity P(D=1|X,Y) = Phi((q - base - gamma r)/noise_sd), q = realized approval threshold."""
        lat = base + gamma * r + NOISE_SD * eps; q = np.quantile(lat, APPROVAL)
        return (lat <= q).astype(int), ndtr((q - base - gamma * r) / NOISE_SD)
    
    def grouped(key, size, D, prop):
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
    s5 = strata(J)
    
    return SimpleNamespace(X=X, Y=Y, Y0=Y0, n=n, codes=codes, names=names, C=C, ref=ref, others=others, target=target, theta=theta, Dtrue=Dtrue,
                           p=p, r=r, eps0=eps0, s5=s5, key=codes * J + s5, strata=strata, select=select, grouped=grouped, dgap=dgap, amax=amax)
