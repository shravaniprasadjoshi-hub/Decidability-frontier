"""E5 (owner: Rohini): one-axis sensitivity analysis for the decidability certificate.

This is a deterministic, finite-pool analysis: it reports the identified-set
width and the share of cells certifiable at each setting.  It deliberately does
not claim a new coverage result--that belongs to E1.  The script writes a
provenance-bearing CSV so that plots and paper tables need not scrape stdout.

Examples (run each seed independently):
  PYTHONPATH=.:core SEED=0 python "smoke tests/e5_sensitivities.py"
  PYTHONPATH=.:core SEED=0 SYNTH=1 python "smoke tests/e5_sensitivities.py"
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from bounds import Cell, gap, verdict
from common import M_MIN_DEFAULT, build, git_state

STATE = os.environ.get("STATE", "CA")
SEED = int(os.environ.get("SEED", 0))
FLIP_SEED = int(os.environ.get("FLIP_SEED", [123, 456, 789][SEED % 3]))
GAMMA = float(os.environ.get("GAMMA", 0.05))
TAUS = (0.05, 0.02)
M_MINS = (1, M_MIN_DEFAULT, 30)
NOISES = (1.0, 0.5)
J10_GATE_PASSED = os.environ.get("J10_GATE_PASSED", "0") == "1"
JS = (5, 10) if J10_GATE_PASSED else (5,)


def cells_for(S, J, noise_sd):
    """Return observed cell-by-stratum counts and the propensity-oracle delta."""
    strata = S.strata(J)
    key = S.codes * J + strata
    D, prop = S.select(GAMMA, S.eps0)
    grouped = S.grouped(key, S.C * J, D, prop)
    delta = S.amax(S.dgap(grouped)[1])
    def cell(c):
        return Cell(grouped["n"].reshape(S.C, J)[c], grouped["na"].reshape(S.C, J)[c],
                    grouped["ka"].reshape(S.C, J)[c])
    return cell, delta


cfg = dict(script="e5_sensitivities", state=STATE, seed=SEED, flip_seed=FLIP_SEED,
           gamma=GAMMA, taus=TAUS, m_mins=M_MINS, noises=NOISES, J=JS,
           j10_gate_passed=J10_GATE_PASSED)
cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
git = git_state()
rows = []

# noise_sd affects the selection law in build(); all comparisons below retain the
# same data seed and flip assignment.  This isolates the declared axis.
for noise_sd in NOISES:
    S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED, noise_sd=noise_sd)
    for J in JS:
        cell, delta = cells_for(S, J, noise_sd)
        for m_min in M_MINS:
            ref = cell(S.ref)
            for c in S.others:
                g = cell(c)
                r = gap(g, ref, delta, z=0.0, m_min=m_min)
                for tau in TAUS:
                    rows.append(dict(seed=SEED, flip_seed=FLIP_SEED, gamma=GAMMA,
                                     noise_sd=noise_sd, J=J, m_min=m_min, tau=tau,
                                     cell=S.names[c], delta_oracle=delta,
                                     identified_width=r["hi"] - r["lo"],
                                     unexplored_mass=r["u_g"] + r["u_ref"],
                                     sensitivity_slope=r["S"],
                                     j10_gate_passed=J10_GATE_PASSED,
                                     verdict=verdict(g, ref, delta, tau, z=0.0, m_min=m_min),
                                     git=git, cfg_hash=cfg_hash))

out = pd.DataFrame(rows)
os.makedirs("smoke tests/results", exist_ok=True)
path = f"smoke tests/results/e5_sensitivities_{STATE}_{SEED}_{FLIP_SEED}.csv"
out.to_csv(path, index=False)

summary = (out.assign(certified=out.verdict.ne("undecided"))
           .groupby(["noise_sd", "J", "m_min", "tau"], as_index=False)
           .agg(cells=("cell", "size"), certified_share=("certified", "mean"),
                median_width=("identified_width", "median"), median_delta=("delta_oracle", "median"),
                median_unexplored=("unexplored_mass", "median")))
print(f"git={git} cfg={cfg_hash}; wrote {path}")
print(summary.round(4).to_string(index=False))
