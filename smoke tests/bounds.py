"""bounds.py: conditional-delta partial identification bounds, se, verdicts, breakdown values.

Model (Day-1 writeup v2): |mu_miss_j - mu_obs_j| <= delta in every EXPLORED stratum; unexplored strata get Manski [0,1].
Inference uses the UNCLIPPED bounds. Clipped form is a point-level refinement only (no se).

Per cell the data is three length-J count vectors over fixed global strata:
    n_j  = rows in cell & stratum j (approved + rejected, X observed for all)
    na_j = approved rows            k_j = defaults among approved
Known omissions (documented, not bugs): se ignores the sampling variance of rho_hat_j * delta (tiny for delta <~ 0.1);
the certificate is CLT-based, i.e. asymptotic only (see open item 13).
"""
import numpy as np
from statistics import NormalDist


class Cell:
    def __init__(self, n_j, na_j, k_j):
        self.n_j, self.na_j, self.k_j = (np.asarray(a, float) for a in (n_j, na_j, k_j))
        assert self.n_j.shape == self.na_j.shape == self.k_j.shape and self.n_j.ndim == 1
        assert (self.k_j >= 0).all() and (self.k_j <= self.na_j).all() and (self.na_j <= self.n_j).all(), "need 0<=k<=na<=n"
        assert self.n_j.sum() > 0, "empty cell"

    @property
    def n(self): return float(self.n_j.sum())

    @classmethod
    def from_arrays(cls, strata, D, Y, mask, J):
        """strata: stratum index per row; D approved (0/1); Y outcome (only read where D==1); mask: rows in this cell."""
        s, D, Y = np.asarray(strata)[mask], np.asarray(D)[mask], np.asarray(Y)[mask]
        n_j = np.bincount(s, minlength=J)
        na_j = np.bincount(s[D == 1], minlength=J)
        k_j = np.bincount(s[(D == 1) & (Y == 1)], minlength=J)
        return cls(n_j, na_j, k_j)


def _parts(c, m_min):
    w = c.n_j / c.n
    ex = c.na_j >= max(m_min, 1)                         # explored: at least m_min approved (hard cutoff)
    mu = np.where(ex, c.k_j / np.maximum(c.na_j, 1), 0.0)
    rho = np.where(c.n_j > 0, 1 - c.na_j / np.maximum(c.n_j, 1), 0.0)
    return w, ex, mu, rho


def endpoints(c, delta, m_min=1, between=True):
    """Unclipped L, U of theta_g, with se of each endpoint.
    L = sum_E w_j (mu_j - rho_j delta);  U = sum_E w_j (mu_j + rho_j delta) + u.  S = delta-slope = sum_E w_j rho_j."""
    w, ex, mu, rho = _parts(c, m_min)
    g_lo = np.where(ex, mu - rho * delta, 0.0)
    g_hi = np.where(ex, mu + rho * delta, 1.0)           # unexplored mass -> 1 (Manski upper)
    L, U = float(np.sum(w * g_lo)), float(np.sum(w * g_hi))
    mut = (c.k_j + 1) / (c.na_j + 2)                      # adjusted so mu_hat in {0,1} doesn't give se = 0
    v = mut * (1 - mut) / np.maximum(c.na_j, 1)
    var_within = float(np.sum(np.where(ex, w ** 2 * v, 0.0)))
    if between:                                           # random stratum sizes (ST1 bug fix, decision D2)
        vb = lambda g: float(np.sum(w * (g - np.sum(w * g)) ** 2) / c.n)
        var_lo, var_hi = var_within + vb(g_lo), var_within + vb(g_hi)
    else:
        var_lo = var_hi = var_within
    return dict(L=L, U=U, se_L=np.sqrt(var_lo), se_U=np.sqrt(var_hi),
                u=float(np.sum(w[~ex])), S=float(np.sum(w[ex] * rho[ex])), centre=float(np.sum(w * mu)))


def clipped_bounds(c, delta, m_min=1):
    """Sharp clipped endpoints (point refinement only): pi mu + rho (mu-delta)_+ and pi mu + rho min(1, mu+delta)."""
    w, ex, mu, rho = _parts(c, m_min)
    pi = 1 - rho
    lo = np.where(ex, pi * mu + rho * np.maximum(0, mu - delta), 0.0)
    hi = np.where(ex, pi * mu + rho * np.minimum(1, mu + delta), 1.0)
    return float(np.sum(w * lo)), float(np.sum(w * hi))


def gap(g, ref, delta, z=0.0, m_min=1, between=True):
    """Identified set for Delta = theta_g - theta_ref, plus z*se-widened version. g and ref are disjoint => se's add."""
    eg, er = endpoints(g, delta, m_min, between), endpoints(ref, delta, m_min, between)
    lo, hi = eg["L"] - er["U"], eg["U"] - er["L"]
    se_lo, se_hi = np.hypot(eg["se_L"], er["se_U"]), np.hypot(eg["se_U"], er["se_L"])
    return dict(lo=lo, hi=hi, se_lo=se_lo, se_hi=se_hi, lo_c=lo - z * se_lo, hi_c=hi + z * se_hi,
                S=eg["S"] + er["S"], u_g=eg["u"], u_ref=er["u"])


def verdict(g, ref, delta, tau, z=0.0, m_min=1, between=True):
    r = gap(g, ref, delta, z, m_min, between)
    if r["lo_c"] > tau or r["hi_c"] < -tau: return "disparity"
    if r["hi_c"] <= tau and r["lo_c"] >= -tau: return "parity"
    return "undecided"


def _ratio(x, S):
    return x / S if S > 0 else (np.inf if x > 0 else -np.inf)


def breakdown(g, ref, tau, z=0.0, m_min=1, delta_ref=0.0, between=True):
    """Largest delta at which each verdict is still certifiable (endpoints are linear in delta).
    disparity: delta <  max((a-tau)/S, (-tau-b)/S);  parity: delta <= min((tau-b)/S, (a+tau)/S).
    a = Delta_lo(0) - z se_lo, b = Delta_hi(0) + z se_hi, with se evaluated at delta_ref (se depends weakly on delta
    through the composition term, so set delta_ref near the breakdown for a self-consistent number).
    Negative => not certifiable at ANY delta >= 0 (e.g. parity when u_g + u_ref > 2 tau)."""
    r0 = gap(g, ref, 0.0, 0.0, m_min, between)
    rs = gap(g, ref, delta_ref, 0.0, m_min, between)
    a, b = r0["lo"] - z * rs["se_lo"], r0["hi"] + z * rs["se_hi"]
    S = rs["S"]
    return dict(disparity=max(_ratio(a - tau, S), _ratio(-tau - b, S)),
                parity=min(_ratio(tau - b, S), _ratio(a + tau, S)), S=S)


def bonferroni_z(alpha, K):
    """One-sided critical value at level alpha/K (baseline multiplicity; Holm/max-t are upgrades, not implemented)."""
    return NormalDist().inv_cdf(1 - alpha / K)
