"""ST1: kink bias of plug-in clipped stratified bounds on theta_g (lower endpoint is where the clip bites).
Modes:
  python st1_kink_bias.py                          synthetic kink/ctrl DGPs (output identical to the original script)
  python st1_kink_bias.py adversarial             worst case: every stratum has mu_obs=delta, rejection 0.7, truth on lower endpoint
  python st1_kink_bias.py profile FILE.csv [g ..]  DGP = ST2's per-stratum profile (CSV dumped by st2_conditional_delta.py).
      Assumed delta = max|d_j| of the profile (valid sensitivity parameter). Truth is placed ON the lower endpoint
      (and separately on the upper one), i.e. the worst case for coverage. 'ctrl' = same profile with mu_obs clipped to
      [0.15, 0.85], which moves strata away from both clip points (delta and 1-delta) and leaves the rest alone."""
import numpy as np, itertools, sys
rng = np.random.default_rng(0)
DELTA, Z, R = 0.03, 1.645, 2000   # additive delta, one-sided z, reps
NS, JS = [50, 100, 200, 500, 2000, 10000], [1, 5, 20]
NS_PROF = [100, 200, 500, 2000, 10000]   # cell sizes; N_audit=5000 over 32 cells is ~156 per cell
THIN = 10                                # a stratum with < THIN approved in the cell is "thin"
BETWEEN = True   # add strata-composition variance (n_j multinomial) to se; False reproduces the OLD se that undercovers ~2-3pp at large n

def dgp(J, kink):
    w = np.ones(J) / J
    pi = np.linspace(0.9, 0.3, J) if J > 1 else np.array([0.6])
    # kink-active: low-risk strata sit at/near mu_obs = delta (where max(0, mu-delta) kinks); control: far from it
    mu = np.linspace(DELTA, 0.25, J) if kink else np.linspace(0.15, 0.40, J)
    if J == 1: mu = np.array([DELTA * 1.5 if kink else 0.25])
    return w, pi, mu

def endpoints(w, pi, mu, delta):     # true clipped endpoints when d_j = -delta (lower) / +delta (upper) in every stratum
    rho = 1 - pi
    return (np.sum(w * (pi * mu + rho * np.maximum(0, mu - delta))),
            np.sum(w * (pi * mu + rho * np.minimum(1, mu + delta))))

def one_setting(n, w, pi, mu, delta):
    L0, U0 = endpoints(w, pi, mu, delta)
    nj = rng.multinomial(n, w, size=R); na = rng.binomial(nj, pi); k = rng.binomial(na, mu)
    ok = na > 0; muh = np.where(ok, k / np.maximum(na, 1), 0.0)
    wh = nj / n; rho = np.where(nj > 0, 1 - na / np.maximum(nj, 1), 0.0)
    pih = np.where(nj > 0, na / np.maximum(nj, 1), 0.0)
    # plug-in bounds; strata with no approved applicants -> [0,1] for their whole mass
    lo = np.sum(wh * np.where(ok, pih * muh + rho * np.maximum(0, muh - delta), 0.0), 1)
    hi = np.sum(wh * np.where(ok, pih * muh + rho * np.minimum(1, muh + delta), 1.0), 1)
    # delta-method se (adjusted mu for the variance so mu_hat=0 doesn't give se=0); slope 1 if clip inactive else pi
    mut = (k + 1) / (na + 2); v = mut * (1 - mut) / np.maximum(na, 1)
    slope = np.where(muh - delta > 0, 1.0, pih); se_lo = np.sqrt(np.sum(np.where(ok, (wh * slope) ** 2 * v, 0), 1))
    slope_hi = np.where(muh + delta < 1, 1.0, pih); se_hi = np.sqrt(np.sum(np.where(ok, (wh * slope_hi) ** 2 * v, 0), 1))
    # (slope_hi was a hard-coded 1.0 before; identical output whenever mu+delta<1, which holds in the default DGPs)
    if BETWEEN:   # old se treated w_j as fixed; stratum sizes are random, adds ~Var_j(g_j)/n
        g_lo = np.where(ok, pih * muh + rho * np.maximum(0, muh - delta), 0.0)
        g_hi = np.where(ok, pih * muh + rho * np.minimum(1, muh + delta), 1.0)
        se_lo = np.sqrt(se_lo ** 2 + np.sum(wh * (g_lo - np.sum(wh * g_lo, 1, keepdims=True)) ** 2, 1) / n)
        se_hi = np.sqrt(se_hi ** 2 + np.sum(wh * (g_hi - np.sum(wh * g_hi, 1, keepdims=True)) ** 2, 1) / n)
    cover = lambda T: np.mean((lo - Z * se_lo <= T) & (T <= hi + Z * se_hi))
    return dict(cov_plug=np.mean((lo <= L0) & (L0 <= hi)), cov_lo=cover(L0), cov_hi=cover(U0),
                b_lo=(np.mean(lo) - L0) / max(np.std(lo), 1e-12), b_hi=(np.mean(hi) - U0) / max(np.std(hi), 1e-12),
                bias_lo=np.mean(lo) - L0, width=np.mean(hi - lo), empty=np.mean(~ok.all(1)),
                thin=np.mean((na < THIN).sum(1)))

def default_mode():
    print(f"delta={DELTA} z={Z} reps={R} BETWEEN={BETWEEN} (MC se of coverage ~{100*np.sqrt(.95*.05/R):.1f}pp)")
    print("dgp     J     n | cov_plug cov_marg | bias_lo  bias/sd | width  frac_with_empty_stratum")
    for kink, J, n in itertools.product([True, False], JS, NS):
        w, pi, mu = dgp(J, kink); r = one_setting(n, w, pi, mu, DELTA)
        print(f"{'kink' if kink else 'ctrl':5s} {J:3d} {n:6d} |  {r['cov_plug']:.3f}    {r['cov_lo']:.3f}  | "
              f"{r['bias_lo']:+.4f}  {r['b_lo']:+.2f}  | {r['width']:.3f}  {r['empty']:.2f}")

def adversarial_mode():
    print(f"adversarial: mu_obs=delta={DELTA} in every stratum, pi=0.3 (rejection 0.7), truth on endpoint, BETWEEN={BETWEEN}, reps={R}")
    print("  J      n | cov_lo cov_hi | b_lo   b_hi | width")
    for J, n in [(1, 10000), (5, 10000), (20, 10000), (50, 10000), (20, 50000)]:
        r = one_setting(n, np.ones(J) / J, np.full(J, 0.3), np.full(J, DELTA), DELTA)
        print(f"{J:3d} {n:6d} | {r['cov_lo']:.3f}  {r['cov_hi']:.3f} | {r['b_lo']:+.2f}  {r['b_hi']:+.2f} | {r['width']:.3f}")

def profile_mode(path, gammas):
    import pandas as pd
    d0 = pd.read_csv(path)
    assert d0[["y_mode", "noise", "sel"]].drop_duplicates().shape[0] == 1, "CSV mixes y_mode/noise/sel: one regime per file"
    for g in (gammas or sorted(d0.gamma.unique())):
        d = d0[np.isclose(d0.gamma, g)].sort_values("j")
        w = d.n_j.values / d.n_j.sum(); pi = d.pi.values
        mu = d.mu_obs.fillna(0.5).clip(0, 1).values            # NaN only where a stratum has no approved (pi=0)
        delta = max(float((d.mu_miss - d.mu_obs).abs().max()), 1e-3)   # NaN mu_miss (pi=1, nobody rejected) is skipped
        print(f"\n===== {path}  gamma={g}  J={len(d)}  assumed delta=max|d_j|={delta:.3f}  reps={R}")
        print("  j |  w_j  | pi_j  | mu_obs |  |mu-delta| | |1-delta-mu|")
        for j in range(len(d)):
            print(f" {j:2d} | {w[j]:.3f} | {pi[j]:.3f} | {mu[j]:.3f}  |   {abs(mu[j]-delta):.3f}    |   {abs(1-delta-mu[j]):.3f}")
        print("cov_lo/cov_hi = margin coverage with truth ON the lower/upper endpoint; b = bias/sd; thin = mean # strata with <10 approved")
        print("dgp       n | cov_lo cov_hi | b_lo   b_hi  | width | empty  thin")
        for n in NS_PROF:
            for name, m in [("profile", mu), ("ctrl", np.clip(mu, 0.15, 0.85))]:
                r = one_setting(n, w, pi, m, delta)
                print(f"{name:8s} {n:6d} | {r['cov_lo']:.3f}  {r['cov_hi']:.3f} | {r['b_lo']:+.2f}  {r['b_hi']:+.2f} | {r['width']:.3f} | {r['empty']:.2f}  {r['thin']:.1f}")

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "profile": profile_mode(sys.argv[2], [float(x) for x in sys.argv[3:]])
    elif len(sys.argv) > 1 and sys.argv[1] == "adversarial": adversarial_mode()
    else: default_mode()