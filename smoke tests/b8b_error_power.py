"""
B8b: audit-level error control + power on the controlled-Delta arm (A12 setup).

usage: python b8b_error_power.py

Env: STATE (CA), SEED (0), FLIP_SEED (123), R (reps, 300), SYNTH=1

Setup block = a12_controlled_delta.py (same Y flips, score, selection)

Per rep: N rows iid with replacement from the pool, fresh D ~ Bern(true propensity), Y seen iff D=1. 
CLT certificate (bounds.endpoints, m_min=10, composition term)

Family: 2K hypotheses fixed ex ante, K=31 non-ref cells x {disparity null |Delta|<=tau, parity null |Delta|>tau}. 
p-values from the one-sided endpoint z-tests:
1. p_disp = min(p_lo, p_hi) (no x2: two-sidedness cost ~0 in thick cells, <1% in thin ones, see writeup s4)
2. p_par = max of the two endpoint p-values (intersection-union)

Procedures: none (per-cell, p<=alpha) | Bonferroni (alpha/2K) | Holm (m=2K; exactly one null per cell is true, but Holm cannot use that, so m=2K is conservative)
WRONG cert = disparity certified when |Delta_true|<=tau, or parity certified when |Delta_true|>tau (Delta_true = pool theta_g - theta_ref).
POWER = share of cells that are certifiable at the ASSUMED delta (pool-level breakdown > delta, noise-free) that the audit certifies.
MULTS: assumed delta = mult x oracle. mult=1 tests the valid regime; mult=0.5 is deliberate misspecification (true confounding > assumed) to see where error control breaks.
Pre-registered: at mult=1, Holm any-wrong rate <= 0.05 (+MC se ~1.3pp at R=300) for both tau; if none/Bonferroni/Holm are all ~0 the test is too conservative to say anything about error control and the mult=0.5 rows are the informative ones.

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


# B8b
ALPHA, R, N_LIST, MULTS = 0.05, int(os.environ.get("R", 300)), [5000, 20000], [1.0, 0.5]
rng2 = np.random.default_rng(20_000 + SEED); K = len(others)

def holm(p, a):
    rej = np.zeros(len(p), bool)
    for i, idx in enumerate(np.argsort(p)):
        if p[idx] <= a / (len(p) - i): rej[idx] = True
        else: break
    return rej
sf = lambda z: 1 - ndtr(z)
s5 = strata(J); key = codes * J + s5; out = []
print(f"R={R} K={K} family m=2K={2*K}; MC se of an error rate near .05: {100*np.sqrt(.05*.95/R):.1f}pp")

for g in GAMMAS:
    D, prop = select(g, eps0); gr = grouped(key, C * J, D, prop); dlt = amax(dpop(gr))
    cl = lambda c: Cell(gr["n"].reshape(C, J)[c], gr["na"].reshape(C, J)[c], gr["ka"].reshape(C, J)[c])
    dstar = {t: (np.array([breakdown(cl(c), cl(ref), t, 0.0, M_MIN)["disparity"] for c in others]),
                 np.array([breakdown(cl(c), cl(ref), t, 0.0, M_MIN)["parity"] for c in others])) for t in TAUS}
    Dt = Dtrue[others]
    
    print(f"\n##### gamma={g} oracle delta={dlt:.3f}")
    for N in N_LIST:
        acc = {}
        
        for t_ in range(R):
            idx = rng2.integers(0, n, N); Di = rng2.random(N) < prop[idx]; k = key[idx]; yy = Y[idx] == 1
            bc = lambda m: np.bincount(k[m], minlength=C * J).reshape(C, J)
            nj, na, ka = bc(np.ones(N, bool)), bc(Di), bc(Di & yy)
        
            for mult in MULTS:
                d = mult * dlt; E = {}
        
                for c in others + [ref]:
                    E[c] = endpoints(Cell(nj[c], na[c], ka[c]), d, M_MIN) if nj[c].sum() > 0 else None
        
                er = E[ref]; Lg = np.array([E[c]["L"] if E[c] else np.nan for c in others]); Ug = np.array([E[c]["U"] if E[c] else np.nan for c in others])
                sLg = np.array([E[c]["se_L"] if E[c] else np.nan for c in others]); sUg = np.array([E[c]["se_U"] if E[c] else np.nan for c in others])
                dlo, dhi = Lg - er["U"], Ug - er["L"]
                slo, shi = np.maximum(np.sqrt(sLg**2 + er["se_U"]**2), 1e-9), np.maximum(np.sqrt(sUg**2 + er["se_L"]**2), 1e-9)
        
                for tau in TAUS:
                    p_lo, p_hi = sf((dlo - tau) / slo), ndtr((dhi + tau) / shi)           # disparity: lower-side / upper-side
                    p_par = np.maximum(ndtr((dhi - tau) / shi), sf((dlo + tau) / slo))      # parity: both endpoints inside +-tau
                    pv = np.nan_to_num(np.concatenate([np.minimum(p_lo, p_hi), p_par]), nan=1.0)
                    wrong = np.concatenate([np.abs(Dt) <= tau, np.abs(Dt) > tau])
                    dok, pok = dstar[tau][0] > d, dstar[tau][1] > d
        
                    for proc, rej in [("none", pv <= ALPHA), ("bonf", pv <= ALPHA / (2 * K)), ("holm", holm(pv, ALPHA))]:
                        a = acc.setdefault((mult, tau, proc), dict(any=[], nw=[], pd=[], pp=[]))
                        a["any"].append((rej & wrong).any()); a["nw"].append((rej & wrong).sum())
                        a["pd"].append((rej[:K] & dok).sum() / dok.sum() if dok.any() else np.nan)
                        a["pp"].append((rej[K:] & pok).sum() / pok.sum() if pok.any() else np.nan)
        
        for (mult, tau, proc), a in acc.items():
            out.append(dict(gamma=g, N=N, mult=mult, tau=tau, proc=proc, any_wrong=np.mean(a["any"]), mean_wrong=np.mean(a["nw"]),
                            power_disp=np.nanmean(a["pd"]) if not np.all(np.isnan(a["pd"])) else np.nan,
                            power_par=np.nanmean(a["pp"]) if not np.all(np.isnan(a["pp"])) else np.nan,
                            n_disp_ok=int((dstar[tau][0] > mult * dlt).sum()), n_par_ok=int((dstar[tau][1] > mult * dlt).sum())))
        T = pd.DataFrame([o for o in out if o["gamma"] == g and o["N"] == N]).drop(columns=["gamma", "N"])
        pd.set_option("display.width", 220, "display.max_columns", 30)
        print(f"\n--- gamma={g} N={N}\n" + T.round(3).to_string(index=False))

pd.DataFrame(out).to_csv(f"b8b_{STATE}_{SEED}_{FLIP_SEED}.csv", index=False); print(f"\nsaved b8b_{STATE}_{SEED}_{FLIP_SEED}.csv")
