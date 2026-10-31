"""E2: multiplicity error control on the controlled-Delta arm (extends smoke tests/b8b_error_power.py).

run (repo root):   PYTHONPATH=core python experiments/e2_error_control.py --config configs/e2_error_control.yaml --results results
quick check:       SYNTH=1 PYTHONPATH=core python experiments/e2_error_control.py --quick       (one seed, small R, prints a timing projection)

What it does, per (seed, gamma, N, rep): draw ONE audit sample (iid with replacement from the pool, D redrawn from the true propensity),
then for every assumed delta = mult * oracle delta and every tau compute the 2K family of p-values (K disparity nulls, then K parity nulls)
and apply every arm to the same p-values (paired): unc, bonf2K (alpha/2K), bonfK (alpha/K), holm2K, shaffer (arm only, O4 not acked).
Records any-wrong, any-wrong on EDGE cells (| |target| - tau | <= edge_rel_width * tau), counts, and power against the noise-free
population-limit flags at the assumed delta.  Clopper-Pearson CIs are on any-wrong counts.

Outputs (results/):  e2_summary.csv  e2_paired.csv  e2_pool.csv  e2_meta.json   (each CSV carries cfg_hash and git)
  e2_summary : one row per (seed, flip_seed, gamma, N, mult, tau, arm)
  e2_paired  : Shaffer minus Holm, per (seed, gamma, N, mult, tau), paired by rep
  e2_pool    : per (seed, gamma, mult, tau) population-limit diagnostics (is a wrong certificate possible even as N -> infinity?)
Independent of core/frontier.py and core/provenance.py (inline hashes), so it does not depend on the E3/E4 branch.
"""
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np, pandas as pd, yaml
from scipy.special import ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
from bounds import Cell, endpoints, breakdown          # noqa: E402
from multiplicity import holm, holm_shaffer             # noqa: E402
from common import build, J, M_MIN_DEFAULT, GRID        # noqa: E402
from cp_interval import clopper_pearson                   # noqa: E402

ARMS_ALL = ("unc", "bonf2K", "bonfK", "holm2K", "shaffer")


def git_id():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=ROOT, stderr=subprocess.DEVNULL).strip())
        return h[:10] + ("+dirty" if dirty else "")
    except Exception:
        return "nogit"


def sf(z):
    return 1 - ndtr(z)


def audit_stats(nj, na, ka, ref, others, d, m_min):
    """Endpoints once per cell at assumed delta d -> identified-difference bounds and their se's (as b8b). None if the ref cell is empty."""
    E = {c: (endpoints(Cell(nj[c], na[c], ka[c]), d, m_min) if nj[c].sum() > 0 else None) for c in others + [ref]}
    er = E[ref]
    if er is None:
        return None
    g = lambda key: np.array([E[c][key] if E[c] else np.nan for c in others])
    Lg, Ug, sLg, sUg = g("L"), g("U"), g("se_L"), g("se_U")
    dlo, dhi = Lg - er["U"], Ug - er["L"]
    slo = np.maximum(np.sqrt(sLg ** 2 + er["se_U"] ** 2), 1e-9)
    shi = np.maximum(np.sqrt(sUg ** 2 + er["se_L"] ** 2), 1e-9)
    return dlo, dhi, slo, shi


def family_p(st, tau, K):
    dlo, dhi, slo, shi = st
    p_lo, p_hi = sf((dlo - tau) / slo), ndtr((dhi + tau) / shi)
    p_par = np.maximum(ndtr((dhi - tau) / shi), sf((dlo + tau) / slo))
    return np.nan_to_num(np.concatenate([np.minimum(p_lo, p_hi), p_par]), nan=1.0)


def apply_arms(pv, alpha, K, arms):
    out = {}
    for a in arms:
        if a == "unc": out[a] = pv <= alpha
        elif a == "bonf2K": out[a] = pv <= alpha / (2 * K)
        elif a == "bonfK": out[a] = pv <= alpha / K
        elif a == "holm2K": out[a] = holm(pv, alpha)
        elif a == "shaffer": out[a] = holm_shaffer(pv, alpha, K)
        else: raise ValueError(a)
    return out


def run_seed(cfg, seed, flip, alpha_arms, quick_t=None):
    state, year = cfg["state"], str(cfg["year"])
    taus, mults, N_list, R = cfg["taus"], cfg["mults"], cfg["N_list"], int(cfg["reps"])
    alpha, level, m_min, arms = float(cfg["alpha"]), float(cfg["ci_level"]), M_MIN_DEFAULT, list(cfg["arms"])
    S = build(state=state, seed=seed, flip_seed=flip)
    C, ref, others, key, Y, n, target, Dtrue = S.C, S.ref, S.others, S.key, S.Y, S.n, S.target, S.Dtrue
    K = len(others); Dt = Dtrue[others]
    summ, pair, pool = [], [], []
    print(f"seed={seed} flip={flip} n={n} K={K} ref='{S.names[ref]}'", flush=True)
    for g in cfg["gammas"]:
        D0, prop = S.select(g, S.eps0); gr = S.grouped(key, C * J, D0, prop)
        dlt = S.amax(S.dgap(gr)[1])
        cl = lambda c: Cell(gr["n"].reshape(C, J)[c], gr["na"].reshape(C, J)[c], gr["ka"].reshape(C, J)[c])
        dstar = {t: (np.array([breakdown(cl(c), cl(ref), t, 0.0, m_min)["disparity"] for c in others]),
                     np.array([breakdown(cl(c), cl(ref), t, 0.0, m_min)["parity"] for c in others])) for t in taus}
        edge = {t: np.abs(np.abs(target[others]) - t) <= cfg["edge_rel_width"] * t for t in taus}
        wrong = {t: np.concatenate([np.abs(Dt) <= t, np.abs(Dt) > t]) for t in taus}
        edge2 = {t: np.concatenate([edge[t], edge[t]]) for t in taus}
        for mult in mults:
            d = mult * dlt
            for t in taus:
                dok, pok = dstar[t][0] > d, dstar[t][1] > d
                pw = np.concatenate([(np.abs(Dt) <= t) & dok, (np.abs(Dt) > t) & pok])    # wrong cert even at N -> inf
                pool.append(dict(seed=seed, flip_seed=flip, gamma=g, mult=mult, tau=t, delta=dlt, K=K, n_edge=int(edge[t].sum()),
                                 n_dok=int(dok.sum()), n_pok=int(pok.sum()), n_pop_wrong=int(pw.sum()), n_pop_wrong_edge=int((pw & edge2[t]).sum())))
        for N in N_list:
            t0 = time.time()
            acc = {}
            for rep in range(R):
                rng = np.random.default_rng([seed, flip, int(round(g * 1e4)), N, rep, 2])
                idx = rng.integers(0, n, N); Di = rng.random(N) < prop[idx]; k = key[idx]; yy = Y[idx] == 1
                bc = lambda m: np.bincount(k[m], minlength=C * J).reshape(C, J)
                nj, na, ka = bc(np.ones(N, bool)), bc(Di), bc(Di & yy)
                for mult in mults:
                    d = mult * dlt
                    st = audit_stats(nj, na, ka, ref, others, d, m_min)
                    for t in taus:
                        dok, pok = dstar[t][0] > d, dstar[t][1] > d
                        pv = family_p(st, t, K) if st is not None else np.ones(2 * K)
                        for arm, rej in apply_arms(pv, alpha, K, arms).items():
                            w = rej & wrong[t]
                            a = acc.setdefault((mult, t, arm), {k_: [] for k_ in ("any", "edge", "nw", "pd", "pp", "nd", "np")})
                            a["any"].append(bool(w.any())); a["edge"].append(bool((w & edge2[t]).any())); a["nw"].append(int(w.sum()))
                            a["pd"].append((rej[:K] & dok).sum() / dok.sum() if dok.any() else np.nan)
                            a["pp"].append((rej[K:] & pok).sum() / pok.sum() if pok.any() else np.nan)
                            a["nd"].append(int(rej[:K].sum())); a["np"].append(int(rej[K:].sum()))
                if quick_t is not None and rep == 4:
                    quick_t.append((time.time() - t0) / 5)
            for (mult, t, arm), a in acc.items():
                d = mult * dlt; dok, pok = dstar[t][0] > d, dstar[t][1] > d
                ka_, ke_ = int(np.sum(a["any"])), int(np.sum(a["edge"])); lo, hi = clopper_pearson(ka_, R, level); elo, ehi = clopper_pearson(ke_, R, level)
                nm = lambda x: float(np.nanmean(x)) if not np.all(np.isnan(x)) else np.nan
                summ.append(dict(seed=seed, flip_seed=flip, gamma=g, N=N, mult=mult, tau=t, arm=arm, R=R, delta=dlt, k_any=ka_, any_wrong=ka_ / R,
                                 any_lo=lo, any_hi=hi, n_edge=int(edge[t].sum()), k_edge=ke_, edge_any=ke_ / R, edge_lo=elo, edge_hi=ehi,
                                 mean_wrong=float(np.mean(a["nw"])), power_disp=nm(a["pd"]), power_par=nm(a["pp"]),
                                 mean_n_disp=float(np.mean(a["nd"])), mean_n_par=float(np.mean(a["np"])), n_dok=int(dok.sum()), n_pok=int(pok.sum())))
            if "shaffer" in arms and "holm2K" in arms:
                for mult in mults:
                    for t in taus:
                        s_, h_ = acc[(mult, t, "shaffer")], acc[(mult, t, "holm2K")]
                        dd = lambda key_: np.array(s_[key_], float) - np.array(h_[key_], float)
                        tot_s = np.array(s_["nd"]) + np.array(s_["np"]); tot_h = np.array(h_["nd"]) + np.array(h_["np"])
                        pair.append(dict(seed=seed, flip_seed=flip, gamma=g, N=N, mult=mult, tau=t, R=R,
                                         d_power_disp=float(np.nanmean(dd("pd"))) if not np.all(np.isnan(dd("pd"))) else np.nan,
                                         d_power_par=float(np.nanmean(dd("pp"))) if not np.all(np.isnan(dd("pp"))) else np.nan,
                                         d_n_certs=float(np.mean(tot_s - tot_h)), share_shaffer_more=float(np.mean(tot_s > tot_h)),
                                         share_shaffer_fewer=float(np.mean(tot_s < tot_h)),
                                         d_any_wrong=float(np.mean(s_["any"]) - np.mean(h_["any"]))))
            print(f"  gamma={g} N={N}: {time.time() - t0:.1f}s", flush=True)
    return summ, pair, pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/e2_error_control.yaml")
    ap.add_argument("--results", default="results")
    ap.add_argument("--quick", action="store_true", help="1 seed, 1 gamma, N=5000, R=50; prints a timing projection")
    a = ap.parse_args()
    cfg_path = a.config if os.path.exists(a.config) else os.path.join(ROOT, a.config)
    cfg = yaml.safe_load(open(cfg_path)); cfg.pop("_status", None)
    full = json.loads(json.dumps(cfg))
    if a.quick:
        cfg.update(seeds=cfg["seeds"][:1], gammas=[0.05], N_list=[5000], reps=50)
    if not set(cfg["arms"]) <= set(ARMS_ALL):
        raise SystemExit(f"unknown arm in {cfg['arms']}")
    os.makedirs(a.results, exist_ok=True)
    cfgfull = dict(cfg, grid=GRID.tolist(), J=J, m_min=M_MIN_DEFAULT, script="e2")
    cfg_hash = hashlib.sha1(json.dumps(cfgfull, sort_keys=True).encode()).hexdigest()[:8]; git = git_id()
    print(f"git={git} cfg={cfg_hash}  seeds={cfg['seeds']} R={cfg['reps']}" + ("  [QUICK: not for results]" if a.quick else ""), flush=True)
    if "+dirty" in git and not a.quick:
        print("WARNING: working tree is dirty; commit (and keep results/ ignored) before the run you report.", flush=True)
    S_, P_, L_, qt = [], [], [], []
    t0 = time.time()
    for seed, flip in cfg["seeds"]:
        s, p, l = run_seed(cfg, int(seed), int(flip), None, quick_t=qt if a.quick else None)
        S_ += s; P_ += p; L_ += l
        for name, rows in (("summary", S_), ("paired", P_), ("pool", L_)):                       # flush after each seed
            df = pd.DataFrame(rows); df["cfg_hash"], df["git"] = cfg_hash, git
            df.to_csv(os.path.join(a.results, f"e2_{name}.csv"), index=False)
    meta = dict(cfg=cfgfull, cfg_hash=cfg_hash, git=git, quick=a.quick, seconds=time.time() - t0, seeds=cfg["seeds"], R=cfg["reps"])
    json.dump(meta, open(os.path.join(a.results, "e2_meta.json"), "w"), indent=1)
    if a.quick:
        per_rep = float(np.mean(qt)); f = full
        total = per_rep * f["reps"] * len(f["gammas"]) * len(f["N_list"]) * len(f["seeds"]) * (1.0 + 3.0 * 0)  # N-scaling is mild; this is per-rep at N=5000
        print(f"\ntiming: {per_rep * 1000:.0f} ms per rep (all mults x taus x arms) -> full run ~ {total / 60:.0f} min (N=20000 is a bit slower)")
    print(f"done in {time.time() - t0:.0f}s -> {a.results}/e2_*.csv")


if __name__ == "__main__":
    main()