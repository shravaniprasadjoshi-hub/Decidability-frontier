"""
run: 
set PYTHONPATH=core 
pytest -q tests/test_multiplicity.py
"""

import numpy as np, pytest
from multiplicity import (bonferroni, holm, holm_shaffer, cell_pvalues, family_pvalues, true_nulls,
                          wrong_certs, any_wrong, power)


def holm_old(p, a):
    rej = np.zeros(len(p), bool)
    for i, idx in enumerate(np.argsort(p)):
        if p[idx] <= a / (len(p) - i): rej[idx] = True
        else: break
    return rej


def test_holm_hand_example():
    p = [0.001, 0.02, 0.04, 0.5]            # alpha .05, m=4: thresholds .0125, .0167, .025, .05
    assert holm(p, 0.05).tolist() == [True, False, False, False]
    p = [0.001, 0.01, 0.02, 0.5]            # .001<.0125, .01<.0167, .02<.025, .5>.05
    assert holm(p, 0.05).tolist() == [True, True, True, False]


def test_holm_matches_b8b_copy():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.uniform(0, 0.2, 62) ** 2
        assert (holm(p, 0.05) == holm_old(p, 0.05)).all()


def test_ordering_bonf_holm_shaffer():
    rng = np.random.default_rng(1)
    for _ in range(300):
        p = rng.uniform(0, 1, 62) ** 4
        b, h, s = bonferroni(p, 0.05), holm(p, 0.05), holm_shaffer(p, 0.05, 31)
        assert (b <= h).all() and (h <= s).all()       # each one rejects at least the one before


def test_bonferroni_divisor():
    p = np.full(62, 0.05 / 62 + 1e-9)
    assert not bonferroni(p, 0.05).any()
    assert bonferroni(p, 0.05, m=31).all()


def test_pvalues_directions():
    # delta interval way above tau -> disparity p tiny, parity p ~1
    pd_, pp = cell_pvalues([0.20], [0.25], [0.01], [0.01], 0.05)
    assert pd_[0] < 1e-6 and pp[0] > 0.999
    # interval well inside +-tau -> parity p tiny, disparity p ~1
    pd_, pp = cell_pvalues([-0.01], [0.01], [0.005], [0.005], 0.05)
    assert pp[0] < 1e-6 and pd_[0] > 0.999


def test_wrong_cert_bookkeeping():
    Dt = np.array([0.01, 0.20])             # cell 0 inside tau, cell 1 outside (tau .05), K=2
    rej = np.array([False, True, True, False])   # disparity cert for cell 1 (ok), parity cert for cell 0 (ok)
    assert not any_wrong(rej, Dt, 0.05)
    rej = np.array([True, False, False, True])   # disparity for cell 0 (wrong), parity for cell 1 (wrong)
    assert wrong_certs(rej, Dt, 0.05) == 2
    assert true_nulls(Dt, 0.05).tolist() == [True, False, False, True]


def test_power_helper():
    rej = np.array([True, False, True, False]); dok = np.array([True, True]); pok = np.array([True, False])
    assert power(rej, (dok, pok), 2) == (0.5, 1.0)
    assert np.isnan(power(rej, (np.array([False, False]), pok), 2)[0])


@pytest.mark.parametrize("proc", ["holm", "shaffer"])
def test_fwer_under_truth(proc):
    # K cells, exactly one true null per cell (uniform p), the false nulls get p=0 (worst case, they never block).
    # any-wrong rate should be <= alpha (+ mc noise)
    rng = np.random.default_rng(2); K, R, a = 31, 4000, 0.05; bad = 0
    for _ in range(R):
        p = np.zeros(2 * K); which = rng.integers(0, 2, K)
        for c in range(K): p[c + K * which[c]] = rng.uniform()
        rej = holm(p, a) if proc == "holm" else holm_shaffer(p, a, K)
        true_idx = np.arange(K) + K * which
        bad += rej[true_idx].any()
    assert bad / R <= a + 0.015
