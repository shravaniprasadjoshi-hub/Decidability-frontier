"""multiplicity + certificate p-values + error rate. used by e2 (and anything that certifies many cells).
family = 2K hypotheses: for each non-ref cell, (disparity null: |Delta|<=tau) and (parity null: |Delta|>tau).
first K entries = disparity nulls, last K = parity nulls (same order as b8b)."""
import numpy as np
from scipy.special import ndtr

sf = lambda z: 1 - ndtr(z)


def bonferroni(p, alpha, m=None):
    # reject if p <= alpha/m, m defaults to number of hypotheses (2K)
    p = np.asarray(p, float)
    return p <= alpha / (len(p) if m is None else m)


def holm(p, alpha):
    # step-down: sort p, compare the i-th smallest to alpha/(m-i), stop at first fail
    p = np.asarray(p, float)
    rej = np.zeros(len(p), bool)
    for i, idx in enumerate(np.argsort(p)):
        if p[idx] <= alpha / (len(p) - i):
            rej[idx] = True
        else:
            break
    return rej


def holm_shaffer(p, alpha, K):
    # NOT acked, optional. same as holm but the divisor is min(remaining, K) because only K of the 2K nulls
    # can be true (exactly one per cell). valid for FWER, rejects at least what holm rejects.
    p = np.asarray(p, float)
    rej = np.zeros(len(p), bool)
    for i, idx in enumerate(np.argsort(p)):
        if p[idx] <= alpha / min(len(p) - i, K):
            rej[idx] = True
        else:
            break
    return rej


def cell_pvalues(dlo, dhi, slo, shi, tau):
    """one-sided endpoint z-tests -> (p_disp, p_par) per cell. dlo/dhi = Delta endpoints, slo/shi = their se.
    p_disp = min of the two sides (no factor 2, see writeup s4), p_par = max (intersection-union)."""
    dlo, dhi, slo, shi = (np.asarray(a, float) for a in (dlo, dhi, slo, shi))
    slo, shi = np.maximum(slo, 1e-9), np.maximum(shi, 1e-9)    # avoid divide by 0
    p_lo, p_hi = sf((dlo - tau) / slo), ndtr((dhi + tau) / shi)
    p_par = np.maximum(ndtr((dhi - tau) / shi), sf((dlo + tau) / slo))
    return np.minimum(p_lo, p_hi), p_par


def family_pvalues(p_disp, p_par):
    # stack into the 2K family, nan (empty cell) -> 1 so it never rejects
    return np.nan_to_num(np.concatenate([p_disp, p_par]), nan=1.0)


def true_nulls(Dtrue, tau):
    # which of the 2K nulls are actually true. rejecting one of these = a wrong certificate
    Dtrue = np.asarray(Dtrue, float)
    return np.concatenate([np.abs(Dtrue) <= tau, np.abs(Dtrue) > tau])


def wrong_certs(rej, Dtrue, tau):
    # number of wrong certificates (disparity when |D|<=tau, or parity when |D|>tau)
    return int((np.asarray(rej) & true_nulls(Dtrue, tau)).sum())


def any_wrong(rej, Dtrue, tau):
    return wrong_certs(rej, Dtrue, tau) > 0


def power(rej, certifiable, K):
    """share of cells that are certifiable at the assumed delta (pool level) that the audit certified.
    returns (power_disp, power_par), nan if nothing is certifiable. certifiable = (dok, pok) bool arrays of length K."""
    rej = np.asarray(rej); dok, pok = certifiable
    pd_ = (rej[:K] & dok).sum() / dok.sum() if dok.any() else np.nan
    pp_ = (rej[K:] & pok).sum() / pok.sum() if pok.any() else np.nan
    return pd_, pp_
