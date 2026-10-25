"""ST2: conditional vs unconditional delta_true on Folktables with imposed selection.
Run: pip install folktables scikit-learn pandas numpy ; python st2_conditional_delta.py
Day-1-dependent choices are all in the CONFIG block."""
import os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# ---------------- CONFIG ----------------
STATE, YEAR      = "CA", "2018"
Y_MODE           = os.environ.get("Y_MODE", "gt50k")  # "gt50k": Y = PINCP>50000 (ACSIncome label, base rate ~0.41). "gt100k": Y = PINCP>100000 (rarer, closer to a default-like rate; check printed base rate)
NOISE_SD         = float(os.environ.get("NOISE_SD", 1.0))     # sd of selection noise relative to the unit-variance score. Larger = softer approval = more overlap. Try 0.5 (near-hard cutoff), 1.0, 2.0
J_LIST           = [3, 5, 10]       # strata (quantile bins of the score); J is part of the assumption
APPROVAL_RATE    = 0.60
GAMMA_LIST       = [float(x) for x in os.environ.get("GAMMAS", "0,0.02,0.05,0.1").split(",")]  # hidden-confounding strengths to run; 0.0 gives the binning floor per J
SELECTION        = os.environ.get("SELECTION", "stratum") # "score": approval uses the continuous score (floor from within-bin leakage); "stratum": approval depends on X only through J_SEL bins (delta_true = pure confounding for J >= J_SEL)
J_SEL            = 5
J_PROFILE        = 5        # J at which the per-stratum profile table is printed
MIN_PER_SIDE     = 30       # min approved and min rejected in a stratum to compute its gap
N_AUDIT_LIST     = [2000, 5000, 20000]  # audit-sample sizes used only for the cell-size check
SMALL_CELL       = 100
RATIO_PASS, SMALL_SHARE_PASS = 3.0, 0.3  # placeholders for "much smaller" / "good share": team must agree
SEED = 0
rng = np.random.default_rng(SEED)

def load():
    if os.environ.get("SYNTH"):   # offline logic test only
        n = 60000
        X = pd.DataFrame({"AGEP": rng.integers(18, 80, n), "SEX": rng.integers(1, 3, n),
                          "RAC1P": rng.choice([1, 2, 6, 8], n, p=[.6, .1, .15, .15]),
                          "SCHL": rng.integers(1, 25, n), "WKHP": rng.integers(1, 70, n), "MAR": rng.integers(1, 6, n)})
        lin = -3 + .04 * X.WKHP + .08 * X.SCHL - .01 * np.abs(X.AGEP - 45)
        y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(int)
        pinc = np.where(y == 1, 50001 + rng.exponential(40000, n), rng.uniform(1000, 50000, n))
        return X, pd.Series(y), pinc
    from folktables import ACSDataSource, ACSIncome
    acs = ACSDataSource(survey_year=YEAR, horizon="1-Year", survey="person").get_data(states=[STATE], download=True)
    X, y, _ = ACSIncome.df_to_pandas(acs)
    pinc = ACSIncome._preprocess(acs)["PINCP"].values      # same filter, same row order as X
    assert len(pinc) == len(X) and ((pinc > 50000).astype(int) == y.iloc[:, 0].values).all(), "PINCP misaligned with ACSIncome label"
    return X.reset_index(drop=True), y.iloc[:, 0].reset_index(drop=True), pinc


X, y, pinc = load()
Y = (pinc > (50000 if Y_MODE == "gt50k" else 100000)).astype(int)
print(f"n={len(Y)}  Y_MODE={Y_MODE}  base rate of Y = {Y.mean():.3f}   selection: {SELECTION}, noise_sd={NOISE_SD}")

# cells: SEX x race(4) x age(4); protected attrs are NOT in the score
race = np.select([X.RAC1P == 1, X.RAC1P == 2, X.RAC1P == 6], ["W", "B", "A"], "O")
age = pd.cut(X.AGEP, [0, 29, 44, 59, 200], labels=["<30", "30-44", "45-59", "60+"]).astype(str)
cell = pd.Series(X.SEX.astype(str).values + "|" + race + "|" + age.values)
feats = X.drop(columns=[c for c in ["SEX", "RAC1P"] if c in X])

tr = rng.random(len(Y)) < 0.5
p = HistGradientBoostingClassifier(max_iter=150, random_state=SEED).fit(feats[tr], Y[tr]).predict_proba(feats)[:, 1]
p = np.clip(p, 1e-3, 1 - 1e-3)
z = np.log(p / (1 - p)); z = (z - z.mean()) / z.std()
r = (Y - p) / np.sqrt(p * (1 - p)); r = (r - r.mean()) / r.std()
eps = rng.standard_normal(len(Y))          # same noise across gammas so runs are comparable

def strata(J): return np.digitize(p, np.quantile(p, np.linspace(0, 1, J + 1)[1:-1]))

def select(gamma):
    base = z if SELECTION == "score" else (lambda s: (s - s.mean()) / s.std())(strata(J_SEL).astype(float))
    lat = base + gamma * r + NOISE_SD * eps
    return (lat <= np.quantile(lat, APPROVAL_RATE)).astype(int)

def pooled(D, J):
    s = strata(J); d, w = [], []
    for j in range(J):
        a, m = (s == j) & (D == 1), (s == j) & (D == 0)
        if a.sum() >= MIN_PER_SIDE and m.sum() >= MIN_PER_SIDE:
            d.append(Y[m].mean() - Y[a].mean()); w.append((s == j).sum())
    d, w = np.abs(np.array(d)), np.array(w)
    return d.max(), np.average(d, weights=w), len(d)

summary = []
prof_rows = []   # per-stratum profile, dumped to CSV for ST1 profile mode
for g in GAMMA_LIST:
    D = select(g); unc = Y[D == 0].mean() - Y[D == 1].mean()
    print(f"\n===== GAMMA={g}: approval {D.mean():.3f}  Y|approved={Y[D==1].mean():.3f}  Y|rejected={Y[D==0].mean():.3f}  uncond gap={unc:+.3f}")
    for J in J_LIST:
        mx, wm, k = pooled(D, J); summary.append((g, J, unc, mx, wm, abs(unc) / mx, k))
    s = strata(J_PROFILE); mx, wm, _ = pooled(D, J_PROFILE)
    print(f"-- stratum profile at J={J_PROFILE} (delta_true max={mx:.3f}, w-mean={wm:.3f}). Exposure uses delta=w-mean and a typical cell of N_ref={N_AUDIT_LIST[1]} / {cell.nunique()} cells")
    print("  j |   n_j | pi_j  | mu_obs | mu_miss | d_j    | z_low=(mu-d)/sd | z_up=(1-d-mu)/sd | exposed (|z|<2 & rho>0.3)")
    for j in range(J_PROFILE):
        a, m = (s == j) & (D == 1), (s == j) & (D == 0)
        mo, mm = (Y[a].mean() if a.sum() else np.nan), (Y[m].mean() if m.sum() else np.nan)
        pj = a.sum() / (s == j).sum(); napp = N_AUDIT_LIST[1] / cell.nunique() * (1 / J_PROFILE) * pj
        sd = np.sqrt(mo * (1 - mo) / max(napp, 1e-9)); zl, zu = (mo - wm) / sd, (1 - wm - mo) / sd
        prof_rows.append(dict(y_mode=Y_MODE, noise=NOISE_SD, sel=SELECTION, gamma=g, J=J_PROFILE, j=j, n_j=int((s==j).sum()), pi=pj, mu_obs=mo, mu_miss=mm))
        print(f" {j:2d} | {(s==j).sum():6d} | {pj:.3f} | {mo:.3f}  |  {mm:.3f}  | {mm-mo:+.3f} |     {zl:+6.1f}      |      {zu:+6.1f}      |   {'YES' if (min(abs(zl), abs(zu)) < 2 and 1 - pj > 0.3) else 'no'}")
    for N in N_AUDIT_LIST:   # decidability floor: share of a cell's mass in strata with no approved applicant in that cell
        idx = rng.choice(len(Y), N, replace=False); u = []; th = []
        for c in cell.iloc[idx].unique():
            ci = idx[cell.iloc[idx].values == c]
            if len(ci) < 20: continue
            u.append(np.mean([((s[ci] == j) & (D[ci] == 1)).sum() == 0 for j in s[ci]]))
            th.append((np.bincount(s[ci][D[ci] == 1], minlength=J_PROFILE) < 10).mean())   # share of the J_PROFILE strata with <10 approved in this cell
        print(f"   audit N={N:6d}: median unexplored share u_g (cells with n_g>=20) = {np.median(u):.2f}  [max {np.max(u):.2f}]")
        print(f"   audit N={N:6d}: thin strata (<10 approved in cell): median share {np.median(th):.2f}  [max {np.max(th):.2f}]")

print("\n===== SUMMARY: ratio = |uncond gap| / max|d_j|;  gamma=0 rows are the binning floor (no hidden confounding)")
print(" gamma |  J | uncond | max|d_j| | w-mean|d_j| | ratio | strata used")
for g, J, u, mx, wm, ra, k in summary: print(f" {g:5.2f} | {J:2d} | {u:+.3f} |  {mx:.3f}   |    {wm:.3f}     | {ra:5.1f} | {k}/{J}")

pd.DataFrame(prof_rows).to_csv(f"profile_{Y_MODE}_{SELECTION}_noise{NOISE_SD}.csv", index=False)
print(f"\nprofile CSV written: profile_{Y_MODE}_{SELECTION}_noise{NOISE_SD}.csv")
print("\nCELL SIZES in an audit sample (independent of gamma)")
for N in N_AUDIT_LIST:
    sz = cell.iloc[rng.choice(len(cell), N, replace=False)].value_counts().reindex(cell.unique(), fill_value=0)
    print(f"N={N:6d}: cells={len(sz)}  share with n_g<{SMALL_CELL}: {(sz < SMALL_CELL).mean():.2f}   median n_g {int(sz.median())}")
print("\nNote: pass criterion should be decidability (delta_true*(rho_g+rho_ref) vs tau and z*se), not the ratio alone. Thresholds are the team's call.")