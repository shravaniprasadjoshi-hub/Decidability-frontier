# E1 coverage validation

Data and scripts live in `smoke tests/`. Run from the repo root:

    export PYTHONPATH=.:core
    STATE=CA SEED=0 R=1000 M_MIN=1 python "smoke tests/b8a_coverage.py"   # also M_MIN=10, 30
    python "smoke tests/plot_e1.py" CA 0 1                                   # STATE SEED M_MIN

`b8a_coverage.py` writes (to `smoke tests/results/`; only the pooled CSV is committed, the rest is gitignored):
- `b8a_coverage_{STATE}_{SEED}_m{M_MIN}_R{R}.csv`: per-cell means over reps
- `b8a_pooled_{STATE}_{SEED}_m{M_MIN}_R{R}.csv`: per-rep pooled coverage by n_g bin (0-50, 50-100, 100-200, 200-500, 500+) and nun (0, 1, 2+), rep-clustered MC-se

`plot_e1.py` reads the pooled CSV from `smoke tests/results/` and regenerates figures, the pass-check table and a thin-bin table there (gitignored).

Pass check: CLT+composition arm, clean bin (nun = 0), coverage >= 0.94. `informative` is a threshold (>= 500 cell-reps) on MC-se precision, not a statistical test; `informative=False` rows are not evidence either way.
At m_min >= 10 the n_g 1-9 bins are conservative from truncation; only the m_min=1 arm tests the se there.
