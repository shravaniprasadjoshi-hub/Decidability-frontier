"""Run: pytest -q test_bounds.py"""
import numpy as np, pytest
from bounds import Cell, endpoints, clipped_bounds, gap, verdict, breakdown, bonferroni_z


def cell_from(n, w, pi, mu):
    """Exact-count cell: n_j = n w_j, na_j = pi_j n_j, k_j = mu_j na_j (pick values that give integers)."""
    n_j = np.array(w) * n; na_j = np.array(pi) * n_j; k_j = np.nan_to_num(np.array(mu, float)) * na_j
    return Cell(n_j, na_j, k_j)


# ---------- hand calculations from the writeups ----------
def test_shravani_ex1_conditional_interval():
    c = cell_from(10000, [.5, .5], [.9, .3], [.05, .30])
    e = endpoints(c, 0.02)
    assert e["centre"] == pytest.approx(0.175) and e["S"] == pytest.approx(0.4)
    assert (e["L"], e["U"]) == pytest.approx((0.167, 0.183))
    assert 0.1825 >= e["L"] and 0.1825 <= e["U"]            # truth 0.1825 inside


def test_shreya_worked_check_unexplored_mass():
    g = cell_from(10000, [.4, .4, .2], [.9, .5, 0], [.04, .20, np.nan])
    ref = cell_from(10000, [1.0], [.9], [.1])
    eg = endpoints(g, 0.02)
    assert eg["u"] == pytest.approx(0.2)
    assert (eg["L"], eg["U"]) == pytest.approx((0.0912, 0.3008))
    r = gap(g, ref, 0.02)
    assert (r["lo"], r["hi"]) == pytest.approx((-0.0108, 0.2028))
    assert verdict(g, ref, 0.02, tau=0.05) == "undecided"
    # decidability floor from u alone: at delta=0, Delta_hi = 0.196 > tau and Delta_lo = -0.004 < tau => no disparity;
    # parity needs Delta_hi <= tau, impossible. Neither verdict at any delta.
    b = breakdown(g, ref, tau=0.05)
    assert b["parity"] < 0 and b["disparity"] < 0


def test_shravani_ex2_J1_breakdown_and_undecided():
    g = cell_from(400, [1.0], [.5], [.20]); ref = cell_from(2000, [1.0], [.8], [.10])
    b = breakdown(g, ref, tau=0.05, z=0.0)
    assert b["S"] == pytest.approx(0.7) and b["disparity"] == pytest.approx(0.05 / 0.7)
    assert verdict(g, ref, 0.10, 0.05) == "undecided"
    assert verdict(g, ref, 0.05 / 0.7 - 1e-6, 0.05) == "disparity"
    assert verdict(g, ref, 0.05 / 0.7 + 1e-6, 0.05) == "undecided"
    # se with z=1.645: ~0.0293 per writeup (J=1 so no between term; adjusted-variance gives <1% difference)
    assert gap(g, ref, 0.0)["se_lo"] == pytest.approx(0.0293, rel=0.02)


def test_shravani_parity_example():
    g = cell_from(1000, [1.0], [.6], [.12]); ref = cell_from(2000, [1.0], [.8], [.10])
    b = breakdown(g, ref, tau=0.05, z=0.0)
    assert b["parity"] == pytest.approx(0.05)                    # (0.05-0.02)/0.6
    b1 = breakdown(g, ref, tau=0.05, z=1.645)
    assert b1["parity"] == pytest.approx(0.0082, abs=6e-4)       # writeup: ~0.0082 (J=1 formulas)


# ---------- algebraic properties ----------
def rand_cell(rng, J=5, n=3000):
    w = rng.dirichlet(np.ones(J) * 5); nj = rng.multinomial(n, w)
    na = rng.binomial(nj, rng.uniform(.2, .95, J)); k = rng.binomial(na, rng.uniform(.02, .6, J))
    return Cell(nj, na, k)


def test_clipped_inside_unclipped_and_equal_without_kink():
    rng = np.random.default_rng(1)
    for _ in range(50):
        c = rand_cell(rng); e = endpoints(c, 0.04); lo, hi = clipped_bounds(c, 0.04)
        assert lo >= e["L"] - 1e-12 and hi <= e["U"] + 1e-12 and lo <= hi
    c = cell_from(10000, [.5, .5], [.9, .3], [.3, .5])           # mu - delta > 0, mu + delta < 1 everywhere
    e = endpoints(c, 0.02); assert clipped_bounds(c, 0.02) == pytest.approx((e["L"], e["U"]))
    c = cell_from(10000, [1.0], [.5], [.01])                      # kink bites: mu < delta
    assert clipped_bounds(c, 0.03)[0] == pytest.approx(0.5 * 0.01) and endpoints(c, 0.03)["L"] == pytest.approx(0.01 - 0.5 * 0.03)


def test_bounds_linear_in_delta_and_width():
    rng = np.random.default_rng(2); c = rand_cell(rng); e0, e1 = endpoints(c, 0.0), endpoints(c, 0.07)
    assert e1["U"] - e1["L"] == pytest.approx(e0["U"] - e0["L"] + 2 * 0.07 * e1["S"])
    assert e1["L"] == pytest.approx(e0["L"] - 0.07 * e1["S"])


def test_m_min_moves_thin_strata_to_unexplored():
    c = Cell([1000, 1000], [900, 5], [90, 1])
    assert endpoints(c, 0.02, m_min=1)["u"] == 0 and endpoints(c, 0.02, m_min=10)["u"] == pytest.approx(0.5)


def test_verdict_flips_exactly_at_breakdown_z0():
    rng = np.random.default_rng(3); n_checked = 0
    for _ in range(300):
        g, ref = rand_cell(rng, n=800), rand_cell(rng, n=4000)
        for tau in (0.02, 0.05):
            b = breakdown(g, ref, tau, z=0.0)
            if b["disparity"] > 1e-6 and np.isfinite(b["disparity"]):
                assert verdict(g, ref, b["disparity"] * (1 - 1e-6), tau) == "disparity"
                assert verdict(g, ref, b["disparity"] * (1 + 1e-6), tau) != "disparity"; n_checked += 1
            if b["parity"] > 1e-6 and np.isfinite(b["parity"]):
                assert verdict(g, ref, b["parity"] * (1 - 1e-6), tau) == "parity"
                assert verdict(g, ref, b["parity"] * (1 + 1e-6), tau) != "parity"; n_checked += 1
            if b["parity"] < 0: assert verdict(g, ref, 0.0, tau) != "parity"
            if b["disparity"] < 0: assert verdict(g, ref, 0.0, tau) != "disparity"
    assert n_checked > 50                                          # make sure the property test actually exercised something


def test_disparity_and_parity_mutually_exclusive_and_monotone_in_delta():
    rng = np.random.default_rng(4)
    for _ in range(100):
        g, ref = rand_cell(rng, n=1500), rand_cell(rng, n=6000); seq = [verdict(g, ref, d, 0.05, z=1.0) for d in np.linspace(0, .2, 41)]
        for a, b in zip(seq, seq[1:]):     # as delta grows, a certified verdict can only be lost, never gained
            if a == "undecided": assert b == "undecided"


def test_parity_floor_from_unexplored_mass():
    # u_g + u_ref > 2 tau  =>  parity never certifiable
    g = Cell([1000, 1000], [900, 0], [90, 0]); ref = Cell([1000, 1000], [900, 0], [90, 0])   # u = 0.5 each
    assert breakdown(g, ref, 0.05)["parity"] < 0 and verdict(g, ref, 0.0, 0.05) != "parity"


def test_bonferroni_z():
    assert bonferroni_z(0.05, 1) == pytest.approx(1.6449, abs=1e-3) and bonferroni_z(0.05, 32) > 2.9


# ---------- se calibration (regression test for the ST1 composition-variance bug) ----------
def test_se_matches_empirical_sd_and_between_term_matters():
    rng = np.random.default_rng(0); J, n, R, delta = 5, 600, 3000, 0.03
    w = np.ones(J) / J; pi = np.linspace(.9, .3, J); mu = np.linspace(.03, .70, J)    # wide spread in mu => big composition term
    L, se_b, se_nb = [], [], []
    for _ in range(R):
        nj = rng.multinomial(n, w); na = rng.binomial(nj, pi); k = rng.binomial(na, mu)
        c = Cell(nj, na, k); e, e0 = endpoints(c, delta), endpoints(c, delta, between=False)
        L.append(e["L"]); se_b.append(e["se_L"]); se_nb.append(e0["se_L"])
    sd = np.std(L)
    assert np.mean(se_b) / sd == pytest.approx(1.0, abs=0.06)
    assert np.mean(se_nb) / sd < 0.94          # old se (fixed weights) must be visibly too small here
