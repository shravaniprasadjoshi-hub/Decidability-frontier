"""core/cp_interval.py: exact binomial (Clopper-Pearson) confidence interval.

Same convention as smoke tests/b8b_error_power.py: two-sided, equal-tailed, level 0.95 by default.
lower = 0 when k = 0, upper = 1 when k = n.
"""
from scipy.stats import beta


def clopper_pearson(k, n, level=0.95):
    """(lo, hi) for k successes out of n trials. n = 0 gives (0, 1)."""
    k, n = int(round(k)), int(n)
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    a = 1.0 - level
    lo = float(beta.ppf(a / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1 - a / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi
