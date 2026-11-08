import os, sys
import pytest
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from cp_interval import clopper_pearson  # noqa: E402


def test_known_values():
    lo, hi = clopper_pearson(5, 20)
    assert lo == pytest.approx(0.08657, abs=1e-5) and hi == pytest.approx(0.49105, abs=1e-5)


def test_zero_successes_upper_bound():
    lo, hi = clopper_pearson(0, 300)
    assert lo == 0.0 and hi == pytest.approx(0.01222, abs=1e-5)


def test_all_successes():
    lo, hi = clopper_pearson(30, 30)
    assert hi == 1.0 and 0.88 < lo < 0.89


def test_matches_scipy_exact():
    for n in (1, 7, 50, 300, 1000):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
            lo, hi = clopper_pearson(k, n)
            assert lo == pytest.approx(ci.low, abs=1e-9) and hi == pytest.approx(ci.high, abs=1e-9)


def test_empty_and_invalid():
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        clopper_pearson(5, 3)


def test_upper_bound_monotone_in_n():
    # with zero errors, more reps can only tighten the upper bound
    ups = [clopper_pearson(0, n)[1] for n in (100, 300, 1000, 3000)]
    assert ups == sorted(ups, reverse=True)