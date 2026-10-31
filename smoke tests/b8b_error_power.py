"""
B8b: audit-level error control + power on the controlled-Delta arm (A12 setup).

usage from repo root:
    set PYTHONPATH=.;core
    python "smoke tests/results/b8b_error_power.py"

Env: STATE (CA), SEED (0), FLIP_SEED (123), R (reps, 300), SYNTH=1
"""
import os, numpy as np, pandas as pd
from scipy.special import ndtr
from scipy.stats import beta
from bounds import Cell, endpoints, breakdown
from multiplicity import holm, holm_shaffer
from common import build, J, M_MIN_DEFAULT, GRID
import subprocess, json, hashlib

# CONFIG 
STATE, YEAR = os.environ.get("STATE", "CA"), "2018"
SEED, FLIP_SEED = int(os.environ.get("SEED", 0)), int(os.environ.get("FLIP_SEED", 123))
GAMMAS, TAUS, M_MIN, MIN_PER_SIDE = [0.02, 0.03, 0.04, 0.05, 0.1], [0.05, 0.02], M_MIN_DEFAULT, 30
ALPHA, R, N_LIST, MULTS = 0.05, int(os.environ.get("R", 300)), [5000, 20000], [1.0, 0.5]

def _git():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return h[:10] + ("+dirty" if dirty else "")
    except Exception:
        return "nogit"

CFG = dict(script="b8b", state=STATE, year=YEAR, seed=SEED, flip_seed=FLIP_SEED, R=R, gammas=GAMMAS, taus=TAUS,
           m_min=M_MIN, grid=GRID.tolist(), J=J, alpha=ALPHA, N_list=N_LIST, mults=MULTS)
CFG_HASH = hashlib.sha1(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:8]
GIT = _git()
TAG = f"{STATE}_{SEED}_{FLIP_SEED}_{CFG_HASH}"
print(f"git={GIT} cfg={CFG_HASH}")

# load everything from common.py
S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED)
Y, Y0, n = S.Y, S.Y0, S.n
codes, names, C = S.codes, S.names, S.C
ref, others, target = S.ref, S.others, S.target
theta, Dtrue = S.theta, S.Dtrue
p, r, eps0 = S.p, S.r, S.eps0
s5, key = S.s5, S.key
select, grouped, amax = S.select, S.grouped, S.amax

# compute natural theta0 for printout
theta0 = np.array([Y0[codes == c].mean() for c in range(C)])

print(f"STATE={STATE} SEED={SEED} FLIP_SEED={FLIP_SEED} n={n} base rate {Y0.mean():.3f} -> {Y.mean():.3f}; ref='{names[ref]}' (n={int((codes==ref).sum())}), K={len(others)}")
print(f"target vs realized Delta_true: max|diff|={np.nanmax(np.abs(Dtrue[others]-target[others])):.4f} (clipping only); natural Delta median |.|={np.median(np.abs(theta0[others]-theta0[ref])):.3f} -> controlled {np.median(np.abs(Dtrue[others])):.3f}")
print(f"cells with |Delta_true|<=tau: " + ", ".join(f"tau={t}: {int((np.abs(Dtrue[others])<=t).sum())}/{len(others)}" for t in TAUS))

# B8b execution
rng2 = np.random.default_rng(20_000 + SEED); K = len(others)
sf = lambda z: 1 - ndtr(z)
out = []
print(f"R={R} K={K} family m=2K={2*K}; MC se of an error rate near .05: {100*np.sqrt(.05*.95/R):.1f}pp")

zlog = {}   # (gamma, N, tau) -> list of per-rep arrays
for g in GAMMAS:
    D, prop = select(g, eps0); gr = grouped(key, C * J, D, prop)
    
    # use S.dgap instead of local dpop
    dlt = amax(S.dgap(gr)[1])
    
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

                er = E[ref]
                Lg = np.array([E[c]["L"] if E[c] else np.nan for c in others]); Ug = np.array([E[c]["U"] if E[c] else np.nan for c in others])
                sLg = np.array([E[c]["se_L"] if E[c] else np.nan for c in others]); sUg = np.array([E[c]["se_U"] if E[c] else np.nan for c in others])
                dlo, dhi = Lg - er["U"], Ug - er["L"]
                slo, shi = np.maximum(np.sqrt(sLg**2 + er["se_U"]**2), 1e-9), np.maximum(np.sqrt(sUg**2 + er["se_L"]**2), 1e-9)

                for tau in TAUS:
                    p_lo, p_hi = sf((dlo - tau) / slo), ndtr((dhi + tau) / shi)
                    p_par = np.maximum(ndtr((dhi - tau) / shi), sf((dlo + tau) / slo))
                    pv = np.nan_to_num(np.concatenate([np.minimum(p_lo, p_hi), p_par]), nan=1.0)
                    if mult == 1.0:
                        zlog.setdefault((g, N, tau), []).append(np.concatenate([dlo, dhi, slo, shi, pv]).astype(np.float32))
                    wrong = np.concatenate([np.abs(Dt) <= tau, np.abs(Dt) > tau])
                    dok, pok = dstar[tau][0] > d, dstar[tau][1] > d

                    for proc, rej in [("none", pv <= ALPHA),
                                      ("bonf", pv <= ALPHA / (2 * K)),
                                      ("holm", holm(pv, ALPHA)),
                                      ("shaffer", holm_shaffer(pv, ALPHA, K))]:
                        a = acc.setdefault((mult, tau, proc), dict(any=[], nw=[], pd=[], pp=[]))
                        a["any"].append((rej & wrong).any()); a["nw"].append((rej & wrong).sum())
                        a["pd"].append((rej[:K] & dok).sum() / dok.sum() if dok.any() else np.nan)
                        a["pp"].append((rej[K:] & pok).sum() / pok.sum() if pok.any() else np.nan)

        for tau in TAUS:
            np.savez_compressed(f"smoke tests/results/zlog_{TAG}_g{g}_N{N}_tau{tau}.npz",
                                z=np.array(zlog[(g, N, tau)]), K=K, Dt=Dt,
                                cells=np.array([names[c] for c in others]), cfg=json.dumps(CFG), git=GIT)
            del zlog[(g, N, tau)]

        for (mult, tau, proc), a in acc.items():
            kk = int(round(np.sum(a["any"])))
            ci_lo = beta.ppf(0.025, kk, R - kk + 1) if kk > 0 else 0.0
            ci_hi = beta.ppf(0.975, kk + 1, R - kk) if kk < R else 1.0
            out.append(dict(gamma=g, N=N, mult=mult, tau=tau, proc=proc, R=R, any_wrong=np.mean(a["any"]), any_lo=ci_lo, any_hi=ci_hi,
                            mean_wrong=np.mean(a["nw"]),
                            power_disp=np.nanmean(a["pd"]) if not np.all(np.isnan(a["pd"])) else np.nan,
                            power_par=np.nanmean(a["pp"]) if not np.all(np.isnan(a["pp"])) else np.nan,
                            n_disp_ok=int((dstar[tau][0] > mult * dlt).sum()), n_par_ok=int((dstar[tau][1] > mult * dlt).sum())))
        T = pd.DataFrame([o for o in out if o["gamma"] == g and o["N"] == N]).drop(columns=["gamma", "N"])
        pd.set_option("display.width", 220, "display.max_columns", 30)
        print(f"\n--- gamma={g} N={N}\n" + T.round(3).to_string(index=False))

df = pd.DataFrame(out); df["cfg_hash"], df["git"] = CFG_HASH, GIT
df.to_csv(f"smoke tests/results/b8b_{TAG}.csv", index=False); print(f"saved b8b_{TAG}.csv")
json.dump(CFG, open(f"smoke tests/results/b8b_{TAG}.json", "w"), indent=1)