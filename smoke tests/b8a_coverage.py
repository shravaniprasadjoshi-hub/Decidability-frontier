"""
B8a: per-cell coverage of the identified-set ENDPOINTS, CLT margin (bounds.py) vs exact (Clopper-Pearson + union bound), natural cells.

usage from repo root:
    set PYTHONPATH=.;core
    python "smoke tests/b8a_coverage.py"
needs bounds.py

Env: STATE (CA), SEED (0), R (reps, 1000), SYNTH=1 (offline logic test only)

DGP per rep: draw N rows WITH replacement from the pool (iid, no fpc), fresh D ~ Bernoulli(true propensity), Y seen only if D=1.
Target: population endpoints L*_g = sum_j w_j (mu_j - rho_j d), U*_g = sum_j w_j (mu_j + rho_j d), all strata, UNCLIPPED, from pool propensities; d = shared oracle delta (propensity, max over cell x stratum, decision P8). 
theta_g in [L*, U*] holds by construction when d >= true gaps.
Coverage of theta_g itself is also logged but is NOT the test: it is ~1 when d exceeds the cell's true gaps, so it cannot detect a bad se.
Pre-registered read (fixed before running): H = CLT margin undercovers (>2pp below 0.95 per endpoint) in thin cells (n_g<100 or >=1 unexplored stratum) while exact stays >= 0.95. 
H refuted if CLT >= 0.93 in every n_g bin AND every n_unexplored bin. 
Clean CLT test = bin n_unexplored = 0 (no m_min truncation).

"""
import os, numpy as np, pandas as pd
from scipy.special import ndtr
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from core.bounds import Cell, endpoints

# CONFIG 
STATE, YEAR = os.environ.get("STATE", "CA"), "2018"
SEED = int(os.environ.get("SEED", 0)); R = int(os.environ.get("R", 1000))
Y_THR, NOISE_SD, J, J_SEL, APPROVAL = 50000, 1.0, 5, 5, 0.60
GAMMAS, N_LIST = [0.02, 0.05, 0.1], [5000, 20000]
M_MIN, MIN_PER_SIDE = int(os.environ.get("M_MIN", 10)), 30
Z, ALPHA, TAU = 1.645, 0.05, 0.05 # one-sided per endpoint, single-test z (NOT multiplicity-adjusted)
NBINS = [0, 50, 100, 200, 500, 10**9]
rng = np.random.default_rng(SEED) # setup stream (must stay identical to st3)
rng2 = np.random.default_rng(10_000 + SEED) # audit-sample stream

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
    acs = ACSDataSource(survey_year=YEAR, horizon="1-Year", survey="person", root_dir="./smoke tests/data").get_data(states=[STATE], download=True)
    X, y, _ = ACSIncome.df_to_pandas(acs)
    pinc = ACSIncome._preprocess(acs)["PINCP"].values
    assert len(pinc) == len(X) and ((pinc > 50000).astype(int) == y.iloc[:, 0].values).all(), "PINCP misaligned"
    return X.reset_index(drop=True), pinc

X, pinc = load(); Y = (pinc > Y_THR).astype(int); n = len(Y)
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
    lat = base + gamma * r + NOISE_SD * eps; q = np.quantile(lat, APPROVAL)
    return (lat <= q).astype(int), ndtr((q - base - gamma * r) / NOISE_SD)

def grouped(key, size, D, prop):
    bc = lambda m, w=None: np.bincount(key[m], weights=None if w is None else w[m], minlength=size)
    a, b, one = D == 1, D == 0, np.ones(len(D), bool)
    return dict(n=bc(one), na=bc(a), ka=bc(a & (Y == 1)), nr=bc(b), kr=bc(b & (Y == 1)),
                A=bc(one, prop), Ay=bc(one, prop * Y), B=bc(one, 1 - prop), By=bc(one, (1 - prop) * Y))

def dgap_pop(g):
    with np.errstate(all="ignore"):
        return np.where((g["A"] >= MIN_PER_SIDE) & (g["B"] >= MIN_PER_SIDE), g["By"] / g["B"] - g["Ay"] / g["A"], np.nan)

# exact arm 
def cp_lo(k, m, a):
    with np.errstate(all="ignore"): return np.where(k > 0, beta.ppf(a, k, m - k + 1), 0.0)

def cp_up(k, m, a):
    with np.errstate(all="ignore"): return np.where(k < m, beta.ppf(1 - a, k + 1, m - k), 1.0)

def exact(nj, na, ka, d):
    """L = sum_j (w_j mu_j) - d*sum_j r_j (r_j = rejected share of the cell in stratum j); all terms >= 0, so bound each piece.
    3J one-sided CP events per endpoint, union bound => each at ALPHA/(3J). No m_min: na=0 gives mu in [0,1] automatically (continuous treatment)."""
    ng = nj.sum(1, keepdims=True); a = ALPHA / (3 * J)
    wl, wu, ru = cp_lo(nj, ng, a), cp_up(nj, ng, a), cp_up(nj - na, ng, a)
    ml, mu_ = cp_lo(ka, na, a), cp_up(ka, na, a)
    return (wl * ml).sum(1) - d * ru.sum(1), (wu * mu_).sum(1) + d * ru.sum(1)

# main
s5 = strata(J); key = codes * J + s5
ref = int(np.bincount(codes).argmax()); theta = np.array([Y[codes == c].mean() for c in range(C)]); Dtrue = theta - theta[ref]

print(f"STATE={STATE} SEED={SEED} R={R} ref='{names[ref]}' K={C-1}  (MC se per pooled bin ~ small; per cell ~{100*np.sqrt(.95*.05/R):.1f}pp)")
allrows = []

for g in GAMMAS:
    D, prop = select(g, eps0); gr = grouped(key, C * J, D, prop)
    dlt = float(np.nanmax(np.abs(dgap_pop(gr).reshape(C, J))))
    nc = gr["n"].reshape(C, J); w = nc / np.maximum(nc.sum(1, keepdims=True), 1)
    with np.errstate(all="ignore"):
        mu = np.nan_to_num(gr["Ay"] / gr["A"]).reshape(C, J); rho = np.nan_to_num(gr["B"] / gr["n"]).reshape(C, J)
    Lt, Ut = (w * (mu - rho * dlt)).sum(1), (w * (mu + rho * dlt)).sum(1)
    
    print(f"\n######## gamma={g}  shared oracle delta={dlt:.3f}  sanity: theta in [L*,U*] for {int(((theta>=Lt-1e-9)&(theta<=Ut+1e-9)).sum())}/{C} cells")
    
    for N in N_LIST:
        rec = {k: np.full((R, C), np.nan) for k in "L U sL sU sL0 sU0 Le Ue ng nun".split()}
        
        for t in range(R):
            idx = rng2.integers(0, n, N); Di = rng2.random(N) < prop[idx]; k = key[idx]; yy = Y[idx] == 1
            bc = lambda m: np.bincount(k[m], minlength=C * J).reshape(C, J)
            nj, na, ka = bc(np.ones(N, bool)), bc(Di), bc(Di & yy)
            rec["Le"][t], rec["Ue"][t] = exact(nj, na, ka, dlt); rec["ng"][t] = nj.sum(1); rec["nun"][t] = (na < M_MIN).sum(1)
        
            for c in range(C):
                if nj[c].sum() == 0: continue
                cl = Cell(nj[c], na[c], ka[c]); e, e0 = endpoints(cl, dlt, M_MIN), endpoints(cl, dlt, M_MIN, between=False)
                rec["L"][t, c], rec["U"][t, c], rec["sL"][t, c], rec["sU"][t, c] = e["L"], e["U"], e["se_L"], e["se_U"]
                rec["sL0"][t, c], rec["sU0"][t, c] = e0["se_L"], e0["se_U"]
        
        L, U, sL, sU, Le, Ue = (rec[k] for k in ["L", "U", "sL", "sU", "Le", "Ue"])
        
        # disparity certificate vs the fixed reference (one-sided z, no multiplicity)
        dlo, dhi = L - U[:, [ref]], U - L[:, [ref]]
        slo, shi = np.sqrt(sL**2 + sU[:, [ref]]**2), np.sqrt(sU**2 + sL[:, [ref]]**2)
        cd_clt = (dlo - Z * slo > TAU) | (dhi + Z * shi < -TAU)
        cd_ex = ((Le - Ue[:, [ref]]) > TAU) | ((Ue - Le[:, [ref]]) < -TAU)
        
        df = pd.DataFrame({
            "cell": np.tile(names, R), "c": np.tile(np.arange(C), R), "ng": rec["ng"].ravel(), "nun": rec["nun"].ravel(),
            "cL": ((L - Z * sL) <= Lt).ravel(), "cU": ((U + Z * sU) >= Ut).ravel(),
            "cL_nb": ((L - Z * rec["sL0"]) <= Lt).ravel(), "cU_nb": ((U + Z * rec["sU0"]) >= Ut).ravel(),
            "eL": (Le <= Lt).ravel(), "eU": (Ue >= Ut).ravel(),
            "th": (((L - Z * sL) <= theta) & (theta <= U + Z * sU)).ravel(),
            "w_clt": (U + Z * sU - (L - Z * sL)).ravel(), "w_ex": (Ue - Le).ravel(),
            "cd_clt": cd_clt.ravel(), "cd_ex": cd_ex.ravel(), "Dtrue": np.tile(Dtrue, R), "ok": ~np.isnan(L).ravel()})
        
        df = df[(df.ok) & (df.c != ref)].drop(columns="ok"); df["gamma"], df["N"] = g, N
        allrows.append(df)
        df["ngbin"] = pd.cut(df.ng, NBINS, right=False); df["nun2"] = df.nun.clip(upper=2)
        agg = dict(share=("ng", "size"), cL=("cL", "mean"), cU=("cU", "mean"), cL_nb=("cL_nb", "mean"), cU_nb=("cU_nb", "mean"),
                   eL=("eL", "mean"), eU=("eU", "mean"), theta=("th", "mean"), w_clt=("w_clt", "mean"), w_ex=("w_ex", "mean"))
        pd.set_option("display.width", 220, "display.max_columns", 30)
        
        print(f"\n--- gamma={g} N={N}: by realized n_g (cL/cU = CLT endpoint coverage, _nb = no composition term, eL/eU = exact; nominal .95)")
        print(df.groupby("ngbin", observed=True).agg(**agg).assign(share=lambda t: t.share / len(df)).round(3).to_string())
        
        print(f"--- gamma={g} N={N}: by # unexplored strata (<{M_MIN} approved); 0 = clean CLT test")
        print(df.groupby("nun2").agg(**agg).assign(share=lambda t: t.share / len(df)).round(3).to_string())
        df["near"] = df.Dtrue.abs() <= TAU
        
        print(f"--- gamma={g} N={N}: disparity-certificate rate (z={Z}, no multiplicity) by whether |Delta_true|<=tau (those are WRONG certs)")
        print(df.groupby("near").agg(cells=("c", "nunique"), clt=("cd_clt", "mean"), exact=("cd_ex", "mean")).round(3).to_string())

pd.concat(allrows).drop(columns=["ngbin", "nun2"], errors="ignore").groupby(["gamma", "N", "cell"]).mean(numeric_only=True).to_csv(f"smoke tests/b8a_coverage_{STATE}_{SEED}_m{M_MIN}_R{R}.csv")
print(f"\nsaved b8a_coverage_{STATE}_{SEED}_m{M_MIN}_R{R}.csv (per-cell means)")
