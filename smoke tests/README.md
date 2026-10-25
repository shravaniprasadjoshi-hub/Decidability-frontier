# Smoke tests and simulations

Empirical side of *Partial Identification and Optimal Certification for Intersectional Fairness Auditing under Selective Labels*
Everything here is **Folktables ACSIncome, CA 2018, Y = income > 50K, seed 0**, stratum-mode selection, J = 5, noise 1.0, 60% approved

## Setup
```
pip install folktables scikit-learn pandas numpy scipy pytest
```
`bounds.py` (certificate: bounds, se, verdicts, breakdowns) is imported by `st3`, `se_check`, `a12`, `b8*`. `SYNTH=1` runs an offline logic test (no real numbers).

## Run order
| Step | Script | What it answers | Output |
|---|---|---|---|
| 1 | `st2_conditional_delta.py` | conditional vs unconditional confounding gap | `st2_*.txt`, `profile_*.csv` |
| 2 | `st1_kink_bias.py [adversarial \| profile FILE.csv]` | does the clip kink undercover? | `st1_*.txt` |
| 3 | `st3_cell_table.py` | per-cell table, per-cell/propensity oracle delta, J-floor, clipping gain | `st3_cell_table_<STATE>_<SEED>.csv` |
| 4 | `se_check.py` | se calibration at delta = 0.03 / 0.10 | stdout |
| 5 | `b8a_coverage.py` | coverage of identified-set endpoints, CLT vs exact, natural cells | `b8a_coverage_*.csv` |
| 6 | `a12_controlled_delta.py` | controlled-Delta arm (pool level): realized Delta, Dc, S, breakdowns | `a12_controlled_*.csv` |
| 7 | `b8b_error_power.py` | any-wrong-certificate rate and power (none / Bonferroni / Holm) on the A12 arm | `b8b_*.csv` |
| 8 | `b8c_n_sweep.py` | power vs audit N; margin vs z*se vs unexplored mass | `b8c_*_m{1,10}.csv` |

Tests: `pytest test_bounds.py` (12; they check the code matches the writeup math)
Env vars: `STATE` (CA), `SEED` (0), `FLIP_SEED` (123), `R` (reps), `GAMMA` (b8c), `M_MIN` (b8c, default 10)
`common.py` has the shared setup (`build(...)`)

## Findings so far (single state, seed, J = 5)
- **ST1 kink:** no detectable undercoverage on Folktables profiles; the adversarial case (J >= 20, high rejection near the clip) does undercover. Footnote + appendix, not a contribution
- **se:** must include the strata-composition term (without it ~0.90-0.96 of the empirical sd). Calibrated to ~3% with it
- **ST2/ST3:** conditioning on strata shrinks the gap from 0.26-0.47 to max|d_j| 0.02-0.10. Natural cells: only 3/31 have |Delta_obs| < 0.05. Oracle delta = propensity max over cell x stratum, ~ gamma. J = 10 floor is sampling noise
- **B8a:** CLT endpoint coverage 0.942-0.968 when no stratum is unexplored; thin cells are conservative through `m_min` truncation (not a clean se test). The Clopper-Pearson + union arm is valid but 2-2.4x wider and certifies far less
- **A12 arm:** 16/31 cells within +-0.05 (9/31 within 0.02); recomputed delta = 0.024 / 0.063 / 0.123 at gamma 0.02 / 0.05 / 0.1 (delta/gamma ~ 1.22-1.25, flips dilute)
- **B8b:** no wrong certificates under Holm (also at half the oracle delta), so error control is untested at the edge. Power is the result
- **B8c:** disparity is sampling-limited (power 0.31 -> 0.98 uncorrected, 0.08 -> 0.76 Holm over N = 20k -> 195k). Parity at tau = 0.05 reaches 0.39 uncorrected and ~0.01 under Holm at N = 195k; at tau = 0.02 it is not certifiable. `m_min` = 1 vs 10 leaves parity unchanged and moves disparity by about +-8pp

## Limitations 
- one state/year/seed/score model; stratum-mode selection is an idealization (no binning leakage); Y is income, not default
- A12 raises the base rate to 0.61 (cells moved toward the reference), so it is not default-like
- the certificate is CLT-based
- thin-cell finite-sample validity is not established (B8a validates only the 0-unexplored bin); `m_min` is not tuned
- oracle delta is a propensity-based, finite-population quantity; delta misspecification was tested only at 0.5x
