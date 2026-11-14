# Decidability Frontier: Intersectional Fairness Auditing under Selective Labels

Partial identification + certification of intersectional disparity verdicts (disparity / parity / undecided) when outcomes are observed only for approved applicants.

## What this repo does

Per intersectional cell g (sex x race x age, 32 cells, fixed reference = largest cell), under the conditional sensitivity model |mu_miss_j - mu_obs_j| <= delta in every explored score stratum j:

- computes unclipped identified-set endpoints for theta_g and for Delta = theta_g - theta_ref, with an se that includes the random-stratum-size (composition) term;
- issues a three-way verdict at tolerance tau (certified disparity / certified parity / undecided);
- controls the probability of any wrong certificate across the 2K family (31 non-ref cells x {disparity null, parity null}) with Holm (and Shaffer-type as an arm);
- computes breakdown values (largest delta at which each verdict is still certifiable) and closed-form N* (audit size needed for a verdict).

Data: Folktables ACSIncome, CA 2018, Y = income > 50K (stand-in outcome, not default). Selection is imposed: stratum mode, J = 5, noise 1.0, 60% approval, hidden confounding strength gamma.

## Repo layout

```
core/
  bounds.py         Cell, endpoints, clipped_bounds, gap, verdict, breakdown, N* (nstar_trunc_exp), definite_fail
  common.py         build(): shared setup (data, cells, score, selection, controlled-Delta flips); GRID, J, M_MIN_DEFAULT
  multiplicity.py   Bonferroni, Holm, Holm-Shaffer, certificate p-values, wrong-cert counting, power
smoke tests/
  st1_kink_bias.py, st2_conditional_delta.py, st3_cell_table.py, se_check.py   early smoke tests
  a12_controlled_delta.py   controlled-Delta arm + frontier / breakdown / N* columns (E3)
  b8a_coverage.py           coverage of endpoints, CLT vs exact (E1)
  b8b_error_power.py        any-wrong-certificate rate + power under multiplicity procedures (E2)
  b8c_n_sweep.py            observed crossover N50 vs closed-form N* (E4)
  b8c_trunc_diag.py         post-hoc diagnostic: truncation-aware / expected-count N*
  plot_e1.py, plot_e3.py    figures and summary tables
  results/                  CSV / JSON / npz outputs (PNGs are gitignored; regenerate)
experiments/
  check_breakdown_identity.py
tests/
  test_bounds.py
```

## Setup

```
pip install folktables scikit-learn pandas numpy scipy matplotlib pytest
```

Run everything from the repo root. Paths contain a space, so quote them.

```
# bash
export PYTHONPATH=.:core
# Windows cmd
set PYTHONPATH=.;core
```

`SYNTH=1` runs a script on synthetic data (offline logic test only, no real numbers).
First real run downloads ACS data for CA 2018 via folktables.

Common env vars: `STATE` (CA), `SEED` (0), `FLIP_SEED` (123), `R` (reps).
Arms (seed:flip) used: 0:123, 1:456, 2:789, 3:321.

## Reproducing results

Run the simulation scripts at one clean commit (no `+dirty` in the git hash). The plotting/diagnostic scripts assert that the input hashes match each other (a12 == b8b zlogs; b8c == a12).

### Unit tests
```
PYTHONPATH=core pytest -q tests/test_bounds.py
```

### E1: coverage (Madhu)
```
STATE=CA SEED=0 R=1000 M_MIN=1  python "smoke tests/b8a_coverage.py"
STATE=CA SEED=0 R=1000 M_MIN=10 python "smoke tests/b8a_coverage.py"
STATE=CA SEED=0 R=1000 M_MIN=30 python "smoke tests/b8a_coverage.py"
python "smoke tests/plot_e1.py" CA 0 1        # STATE SEED M_MIN
```
Repeat for SEED = 1, 2. Writes to `smoke tests/results/`:
- `b8a_coverage_{STATE}_{SEED}_m{M_MIN}_R{R}.csv` per-cell means (gitignored)
- `b8a_pooled_{STATE}_{SEED}_m{M_MIN}_R{R}.csv` pooled by n_g bin (0-50, 50-100, 100-200, 200-500, 500+) and nun (0, 1, 2+), with rep-clustered MC-se (committed; 288 rows per file)

Pass check: CLT + composition arm, clean bin (nun = 0), coverage >= 0.94. `informative` is a precision threshold (>= 500 cell-reps), not a test. At m_min >= 10 the thin bins are conservative from truncation, so only the m_min = 1 arm tests the se there. Rows across gamma and N within a seed share one population, so report coverage with `mc_se`, not pass counts.

### E2: error control + power (Shravani)
```
STATE=CA SEED=0 FLIP_SEED=123 R=300 python "smoke tests/b8b_error_power.py"
```
Evaluates none / Bonferroni / Holm / Holm-Shaffer over the 2K = 62 family at N in {5000, 20000}, tau in {0.05, 0.02}, and delta multiplier in {1.0, 0.5}. Outputs `b8b_*.csv`, a config JSON, and per-rep z/p logs `zlog_*.npz` (used by Fig 3 in `plot_e3.py`). Any-wrong rates come with Clopper-Pearson CIs.

### E3: frontier (Shreya)
```
for a in "0 123" "1 456" "2 789" "3 321"; do set -- $a
  SEED=$1 FLIP_SEED=$2 python "smoke tests/a12_controlled_delta.py" || exit 1
done
for a in "0 123" "1 456" "2 789" "3 321"; do set -- $a
  SEED=$1 FLIP_SEED=$2 python experiments/check_breakdown_identity.py || exit 1
done
python "smoke tests/plot_e3.py" "smoke tests/results"      # default arms 0:123 1:456 2:789 3:321
```
`a12` writes `a12_controlled_{STATE}_{SEED}_{FLIP}.csv` (pool-level quantities from expected counts, plus `*_real` realized versions). `plot_e3.py` writes `figs/e3_fig1_frac_certifiable.png`, `e3_fig2_tau_curves.png`, `e3_fig3_holm_power_exp.png` and `e3_summary.csv`, `e3_holm_power_exp.csv`. Fig 3 needs the b8b zlogs; set `DENOM=real` to use realized-draw d*.

### E4: N sweep vs closed-form N* (Shreya)
Needs the a12 CSV for the same (STATE, SEED, FLIP_SEED) at the same commit.
```
for g in 0.02 0.03 0.04 0.05; do
  SEED=0 FLIP_SEED=123 GAMMA=$g R=200 python "smoke tests/b8c_n_sweep.py"
done
python "smoke tests/b8c_trunc_diag.py"
```
Smoke-run b8c at small R first (`R=5`) before a full run. Outputs `b8c_power_*.csv`, `b8c_crossover_*.csv`, `b8c_trunc_diag_CA.csv`.
Pass rule: observed N50 within 2x of N*. Only flag=ok rows count; censored rows are reported separately.

### Early smoke tests (reference)
```
python "smoke tests/st1_kink_bias.py"                 # kink bias, synthetic
python "smoke tests/st1_kink_bias.py" adversarial     # worst case table
python "smoke tests/st2_conditional_delta.py"         # conditional vs unconditional delta
python "smoke tests/st3_cell_table.py"                # per-cell table, J-floor
python "smoke tests/se_check.py"                      # se calibration at delta 0.03 / 0.10
```

## Settings (team-acked)

- tau = 0.05 primary, 0.02 sensitivity. 
- m_min = 10 primary (1 and 30 as sensitivity).
- Holm over 2K = 62 hypotheses, per-cell uncorrected as secondary.
- delta = propensity-based max over cell x stratum.
- J = 5 primary.
- Largest cell = reference.
- Clipped bounds = footnote only; Option C (active sampling) cut.
- Flip seed 321 was adopted as the confirmatory arm

## Headline findings (CA 2018, single pool)

- **Coverage (E1):** in the clean bin (no unexplored strata), CLT + composition term covers at about 0.94-0.97 across seeds 0-2 and m_min {1, 10, 30}. Without the composition term, coverage is about 0.92. Near-0.94 values are within MC noise. Thin-cell coverage is only tested by the m_min = 1 arm.
- **Error control (E2):** no wrong certificates under Holm or Shaffer at delta multiplier 1.0 in any arm (CP upper bound about 0.004 at R = 1000). This means validity was not contradicted, not that the procedure is tight. The multiplier-0.5 arm showed no degradation in this DGP, which is not evidence of robustness to misspecification.
- **Frontier (E3):** disparity is certifiable but power-limited under Holm. Parity is practically unreachable at audit N, and at tau = 0.02 at any N.
- **N sweep (E4):** the closed-form N* ignoring truncation is optimistic in small cells. A truncation-aware, expected-count N* fixes this, but both corrections were found post hoc. Arm 3:321 is the fresh-seed check. Under the truncation-aware expected-count N*, the pass rate is 1.0 across all 174 observed ok rows (0 failures), with 4 right-censored rows reported separately. The confirmatory arm (3, 321) has 27 ok rows and 0 failures.

## Limitations

- One state, one year, one score model; Y is income, not default.
- Stratum-mode selection is an idealization (no binning leakage).
- Oracle delta is used in simulation; delta misspecification and delta calibration on real data are not addressed.
- The certificate is CLT-based. Thin-cell validity is not established beyond the m_min = 1 arm.
- The corrected N* uses the same `endpoints()`/m_min code as the simulator, so E4 tests the truncation mechanism, not an independent model.
- Breakdown values are not novel by themselves (Rambachan-Coston-Kennedy, Masten-Poirier, Byun et al.).

## Reproducibility notes

- Every results CSV carries `git` and `cfg_hash`. Final E3/E4 results were produced at run commit `0a92c7e4b3`. Later commits only changed docstrings or post-processing. Confirmed via AST comparison that the simulation code is identical to 2b8aaa52c1 outside of docstrings.
- Results are committed once, after all runs. PNGs are gitignored; regenerate with the plot scripts.

## Contributors

- **Shreyashree**: Core statistical bounds and tests, controlled-Delta simulation arm, decidability frontier figures, N-sweep and truncation-aware diagnostic, provenance verification.
- **Shravani**: Error-control and power experiments, multiplicity corrections (Holm/Shaffer).
- **Madhu**: Coverage validation and pooling analysis across seeds and truncation thresholds.

## References

- Rambachan, Coston & Kennedy, arXiv 2212.09844
- Liu, Molinari & Velez (2026), arXiv 2606.14977
- Molina & Loiseau, NeurIPS 2022
- Yan & Zhang, ICML 2022
- Byun et al., AISTATS 2024
- Masten & Poirier (2020)
- Das et al., NeurIPS 2025
