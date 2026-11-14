"""Verify E5's realised-count baseline against A12 before reporting E5 tables.

E5's report rows use expected counts by design. This check separately rebuilds
the A12 realised arm (noise=1, J=5, gamma=.05, m_min=10) and verifies that its
delta and uncorrected endpoints match the committed A12 CSV cell by cell.
"""
import os
import numpy as np
import pandas as pd

from bounds import Cell, gap
from common import J, M_MIN_DEFAULT, build

STATE = os.environ.get("STATE", "CA")
SEED = int(os.environ.get("SEED", 0))
FLIP = {0: 123, 1: 456, 2: 789, 3: 321}.get(SEED)
if FLIP is None:
    raise ValueError("SEED must be 0, 1, 2, or 3")
GAMMA = 0.05
path = os.environ.get("A12_CSV", f"smoke tests/results/a12_controlled_{STATE}_{SEED}_{FLIP}.csv")

S = build(state=STATE, seed=SEED, flip_seed=FLIP, noise_sd=1.0)
D, prop = S.select(GAMMA, S.eps0)
gr = S.grouped(S.key, S.C * J, D, prop)
delta = S.amax(S.dgap(gr)[1])
cell = lambda c: Cell(gr["n"].reshape(S.C, J)[c], gr["na"].reshape(S.C, J)[c], gr["ka"].reshape(S.C, J)[c])
ref = cell(S.ref)

a12 = pd.read_csv(path)
a12 = a12[np.isclose(a12.gamma, GAMMA)].set_index("cell")
if len(a12) != len(S.others):
    raise AssertionError(f"Expected {len(S.others)} A12 cells at gamma=.05; found {len(a12)}")

errors = []
for c in S.others:
    observed = gap(cell(c), ref, delta, z=0.0, m_min=M_MIN_DEFAULT)
    target = a12.loc[S.names[c]]
    errors.append((abs(observed["lo"] - target.Dlo), abs(observed["hi"] - target.Dhi),
                   abs(delta - target.delta_shared)))
max_lo, max_hi, max_delta = np.max(np.asarray(errors), axis=0)
print(f"A12 realised-arm check: cells={len(errors)} max_abs(Dlo)={max_lo:.3e} "
      f"max_abs(Dhi)={max_hi:.3e} max_abs(delta)={max_delta:.3e}")
if max(max_lo, max_hi, max_delta) > 1e-10:
    raise AssertionError("E5/A12 realised-count baseline mismatch")
print("PASS: E5's realised baseline reproduces A12 endpoints and verdict inputs.")
