"""scripts/report_e2.py: E2 tables from results/e2_*.csv only.

usage:  python scripts/report_e2.py [--results results] [--alpha 0.05]
writes: results/e2_report_pooled.csv    any-wrong pooled over seeds x gamma, per (N, mult, tau, arm), Clopper-Pearson CI, rep count, label
        results/e2_report_edge.csv      same, EDGE cells only
        results/e2_report_paired.csv    Shaffer - Holm paired power / certificate counts with per-seed spread
and prints them plus the pre-registered switch-rule line.

Labels (pass rule):
  mult >= 1, corrected arm : "validity not contradicted (assuming correctly specified delta)" if CP upper <= alpha else "not shown valid"
  mult <  1, corrected arm : "no degradation found in this DGP" if 0 wrong certificates in every rep, else "rare wrong certificates: k reps (see events, pop_wrong)"
  unc                      : "per-test only" (no family control expected; shown for contrast)
Pooling caveat: units (seed, gamma) share the same cells and pool, so they are not independent replications; the pooled CP interval is
descriptive. The per-(seed, gamma) worst case is printed too.
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from cp_interval import clopper_pearson  # noqa: E402

CORRECTED = ("bonf2K", "bonfK", "holm2K", "shaffer")
ORDER = ["unc", "bonf2K", "bonfK", "holm2K", "shaffer"]


def label(arm, mult, k, hi, alpha):
    if arm == "unc":
        return "per-test only"
    if mult >= 1:
        return "validity not contradicted (assuming correctly specified delta)" if hi <= alpha else "not shown valid"
    return "no degradation found in this DGP" if k == 0 else f"rare wrong certificates: {k} reps (attribution: see events, pop_wrong)"


def pool(df, kcol, by, alpha, level):
    rows = []
    for key, g in df.groupby(by):
        k, n = int(g[kcol].sum()), int(g.R.sum()); lo, hi = clopper_pearson(k, n, level)
        d = dict(zip(by, key)); arm, mult = d["arm"], d["mult"]
        rows.append(dict(**d, units=len(g), reps=n, k_wrong=k, any_wrong=k / n, cp_lo=lo, cp_hi=hi, label=label(arm, mult, k, hi, alpha),
                         worst_unit=float((g[kcol] / g.R).max())))
    r = pd.DataFrame(rows)
    r["arm"] = pd.Categorical(r["arm"], ORDER, ordered=True)
    return r.sort_values(["N", "tau", "mult", "arm"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--level", type=float, default=0.95)
    a = ap.parse_args()
    s = pd.read_csv(os.path.join(a.results, "e2_summary.csv")); pr = pd.read_csv(os.path.join(a.results, "e2_paired.csv"))
    pl = pd.read_csv(os.path.join(a.results, "e2_pool.csv"))
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    by = ["N", "mult", "tau", "arm"]
    prim = s[(s.gamma > 0) & (s.gamma <= 0.05)]                      # primary: confounded and certifiable (gamma = 0.1 has ~no power; gamma = 0 is the no-slack check)
    tab = pool(prim, "k_any", by, a.alpha, a.level); tab.to_csv(os.path.join(a.results, "e2_report_pooled.csv"), index=False)
    tab_all = pool(s[s.gamma > 0], "k_any", by, a.alpha, a.level); tab_all.to_csv(os.path.join(a.results, "e2_report_pooled_allgamma.csv"), index=False)
    z0 = s[(s.gamma == 0) & (s.mult == 1.0)]                          # oracle delta ~ 0, so every mult is the same test: show mult = 1 only
    tab0 = pool(z0, "k_any", by, a.alpha, a.level) if len(z0) else None
    etab = pool(prim[prim.n_edge > 0], "k_edge", by, a.alpha, a.level); etab.to_csv(os.path.join(a.results, "e2_report_edge.csv"), index=False)
    print(f"seeds={sorted(s.seed.unique())} gammas={sorted(s.gamma.unique())} reps/unit={int(s.R.iloc[0])}  cfg={s.cfg_hash.iloc[0]} git={s.git.iloc[0]}\n")
    f = lambda t: t.assign(any_wrong=t.any_wrong.round(4), cp_lo=t.cp_lo.round(4), cp_hi=t.cp_hi.round(4), worst_unit=t.worst_unit.round(3))
    print("== any-wrong, all cells, PRIMARY pooling: 0 < gamma <= 0.05 (CP 95%) ==\n" + f(tab).to_string(index=False))
    if tab0 is not None:
        print("\n== gamma = 0 (no confounding, oracle delta ~ 0, no identification slack): does correction matter? ==\n" + f(tab0).to_string(index=False))
    print("\n== any-wrong per gamma, mult >= 1 and mult < 1 (k wrong reps / reps, all N, tau pooled) ==")
    pg = s.groupby(["gamma", "arm", s.mult >= 1]).agg(k=("k_any", "sum"), reps=("R", "sum")).reset_index().rename(columns={"mult": "mult>=1"})
    pg["rate"] = (pg.k / pg.reps).round(5); print(pg.pivot_table(index=["arm", "mult>=1"], columns="gamma", values="rate").to_string())
    print("\n== any-wrong, all gamma > 0 pooled (includes gamma = 0.1, where almost nothing is certifiable; optimistic) ==\n" + f(tab_all).to_string(index=False))
    print("\n== any-wrong, EDGE cells only (planted at +-0.9 tau and +-1.1 tau) ==\n" + f(etab).to_string(index=False))
    z0e = s[(s.gamma == 0) & (s.mult == 1.0) & (s.n_edge > 0)]
    if len(z0e):
        e0 = pool(z0e, "k_edge", by, a.alpha, a.level)
        print("\n== gamma = 0, EDGE cells only (no identification slack; mult = 1) ==\n" + f(e0).to_string(index=False))
    print(f"\nedge cells per tau (should be 8: four planted values x ~2 cells): {sorted(s.n_edge.unique())}")

    ev_path = os.path.join(a.results, "e2_events.csv")
    ev = pd.read_csv(ev_path) if os.path.exists(ev_path) else pd.DataFrame()
    print("\n== every wrong certificate of a corrected arm (cell, N, tau, mult); pop_wrong = wrongly certifiable even at N -> infinity ==")
    if "seed" in ev.columns and len(ev):
        cols = ["seed", "gamma", "N", "mult", "tau", "arm", "test", "cell_name", "Dtrue", "edge", "pop_wrong", "p"]
        print(ev.groupby(["mult", "pop_wrong"]).size().rename("n_events").to_string())
        print("events at mult >= 1:\n" + (ev[ev.mult >= 1][cols].round(5).to_string(index=False) if (ev.mult >= 1).any() else "none"))
        print("events at mult < 1 (first 60):\n" + ev[ev.mult < 1][cols].round(5).head(60).to_string(index=False))
    else:
        print("none")
    print("\n== population-limit diagnostics (EXPECTED counts): wrong certificates possible even as N -> infinity? ==")
    dg = pl.groupby(["mult", "tau"]).agg(units=("seed", "size"), n_edge=("n_edge", "mean"), n_dok=("n_dok", "mean"), n_pok=("n_pok", "mean"),
                                           pop_wrong=("n_pop_wrong", "sum"), pop_wrong_edge=("n_pop_wrong_edge", "sum")).round(2)
    print(dg.to_string())
    if (pl.n_pop_wrong == 0).all():
        print("-> no cell is wrongly certifiable at N -> infinity at any mult: any-wrong here is sampling error only (identification assumption never violated).")

    # Shaffer vs Holm paired, per-seed spread
    pr = pr[pr.gamma > 0]
    pg = pr.groupby(["seed", "N", "mult", "tau"]).agg(dpd=("d_power_disp", "mean"), dpp=("d_power_par", "mean"), dn=("d_n_certs", "mean"),
                                                      more=("share_shaffer_more", "mean"), fewer=("share_shaffer_fewer", "mean"), dwrong=("d_any_wrong", "mean")).reset_index()
    rows = []
    for (N, mult, tau), g in pg.groupby(["N", "mult", "tau"]):
        r = dict(N=N, mult=mult, tau=tau, n_seeds=len(g))
        for c, nm in (("dpd", "d_power_disp"), ("dpp", "d_power_par"), ("dn", "d_n_certs"), ("dwrong", "d_any_wrong")):
            r[f"{nm}_mean"] = g[c].mean(); r[f"{nm}_min"] = g[c].min(); r[f"{nm}_max"] = g[c].max(); r[f"{nm}_sd"] = g[c].std(ddof=1) if len(g) > 1 else np.nan
        r["share_shaffer_more"] = g.more.mean(); r["share_shaffer_fewer"] = g.fewer.mean()
        rows.append(r)
    ptab = pd.DataFrame(rows); ptab.to_csv(os.path.join(a.results, "e2_report_paired.csv"), index=False)
    cols = ["N", "mult", "tau", "n_seeds", "d_power_disp_mean", "d_power_disp_min", "d_power_disp_max", "d_power_par_mean", "d_power_par_min",
            "d_power_par_max", "d_n_certs_mean", "d_n_certs_sd", "share_shaffer_more", "share_shaffer_fewer", "d_any_wrong_mean"]
    print("\n== Shaffer - Holm (paired by rep; per-seed mean, spread = min/max across seeds) ==\n" + ptab[cols].round(4).to_string(index=False))
    if (ptab.share_shaffer_fewer > 0).any():
        print("WARNING: Shaffer certified FEWER than Holm in some reps; it should dominate Holm. Check holm_shaffer.")

    # switch rule: report only
    sh = etab[(etab.arm == "shaffer") & (etab.mult >= 1)]
    print("\n== pre-registered switch rule (report only; do not act) ==")
    if len(sh):
        ok = bool((sh.cp_hi <= a.alpha).all())
        print(sh[["N", "mult", "tau", "reps", "k_wrong", "cp_hi"]].round(4).to_string(index=False))
        print(f"Shaffer edge-cell any-wrong CP upper <= alpha={a.alpha} at every mult >= 1: {ok}  -> switch criterion {'MET' if ok else 'NOT met'}"
              " (decision stays with the team; Shaffer is not acked as primary)")
    else:
        print("no edge-cell rows at mult >= 1")


if __name__ == "__main__":
    main()