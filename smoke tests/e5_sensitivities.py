"""E5 (owner: Rohini): pool-level sensitivity table for the certificate.

Primary rows use A12-compatible expected approval/positive counts (A, Ay), not
one realized approval draw. Every row is either identification-only (z=0) or a
CLT certificate at a stated audit size (z=1.645, N=20k by default). Noise rows
must be compared against their reported delta_oracle: fixed gamma does not hold
delta fixed when the selection-noise scale changes.

Primary arms: 0:123, 1:456, 2:789. Arm 3:321 is held out and may be run
separately. m_min=30 and J=10 are proposed/gated and excluded by default.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from bounds import Cell, _scaled, gap, verdict
from common import M_MIN_DEFAULT, MIN_PER_SIDE, build, git_state

ARM_FLIPS = {0: 123, 1: 456, 2: 789, 3: 321}
STATE = os.environ.get("STATE", "CA")
SEED = int(os.environ.get("SEED", 0))
if SEED not in ARM_FLIPS:
    raise ValueError(f"Unknown SEED={SEED}; expected one of {sorted(ARM_FLIPS)}")
FLIP_SEED = int(os.environ.get("FLIP_SEED", ARM_FLIPS[SEED]))
if FLIP_SEED != ARM_FLIPS[SEED]:
    raise ValueError(f"SEED={SEED} is paired with FLIP_SEED={ARM_FLIPS[SEED]}, not {FLIP_SEED}")

GAMMAS = tuple(float(x) for x in os.environ.get("GAMMAS", "0.03,0.04,0.05").split(","))
TAUS = (0.05, 0.02)
NOISES = (1.0, 0.5)
M_MINS = (1, M_MIN_DEFAULT) + ((30,) if os.environ.get("INCLUDE_PROPOSED_M30") == "1" else ())
AUDIT_N = int(os.environ.get("AUDIT_N", 20_000))
Z_MODES = (("identification_only", 0.0, None), ("audit_clt", 1.645, AUDIT_N))
SYNTHETIC = os.environ.get("SYNTH") == "1"
SOURCE = "synthetic" if SYNTHETIC else "folktables_acs_ca_2018"


def gate_artifact():
    """Require an ST1-generated J=10 diagnostic artifact; never accept a bare env flag."""
    path = os.environ.get("J10_GATE_ARTIFACT")
    if not path:
        return None
    with open(path) as f:
        artifact = json.load(f)
    if artifact.get("kind") != "st1_profile_diagnostic" or artifact.get("j_profile") != 10:
        raise ValueError("J10_GATE_ARTIFACT is not an ST1 J=10 profile diagnostic")
    if artifact.get("status") != "diagnostic-complete":
        raise ValueError("J10_GATE_ARTIFACT did not complete successfully")
    return dict(path=str(Path(path)), cfg_hash=artifact.get("cfg_hash"), git=artifact.get("git"))


GATE = gate_artifact()
INCLUDE_J10 = os.environ.get("INCLUDE_J10") == "1"
if INCLUDE_J10 and GATE is None:
    raise ValueError("J=10 needs INCLUDE_J10=1 and a completed J10_GATE_ARTIFACT")
JS = (5, 10) if INCLUDE_J10 else (5,)


def expected_cell(grouped, C, J, c):
    """A12's expected-count convention, including feasibility clipping."""
    n = grouped["n"].reshape(C, J)[c]
    A = np.minimum(grouped["A"].reshape(C, J)[c], n)
    Ay = np.minimum(grouped["Ay"].reshape(C, J)[c], A)
    return Cell(n, A, Ay)


def oracle_delta_info(grouped, C, J):
    """Propensity delta and entries omitted by MIN_PER_SIDE."""
    A = grouped["A"].reshape(C, J)
    B = grouped["B"].reshape(C, J)
    Ay = grouped["Ay"].reshape(C, J)
    By = grouped["By"].reshape(C, J)
    valid = (A >= MIN_PER_SIDE) & (B >= MIN_PER_SIDE)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = By / B - Ay / A
    delta = float(np.nanmax(np.abs(np.where(valid, d, np.nan))))
    return delta, int((~valid).sum()), int(valid.sum())


def row_for(cell, ref, *, delta, z, audit_n, m_min, S, metadata):
    g, r = cell, ref
    if audit_n is not None:
        scale = audit_n / S.n
        g, r = _scaled(g, scale), _scaled(r, scale)
    interval = gap(g, r, delta, z=z, m_min=m_min)
    v = verdict(g, r, delta, metadata["tau"], z=z, m_min=m_min)
    return dict(**metadata, z=z, audit_n=audit_n, identified_width=interval["hi"] - interval["lo"],
                unexplored_mass=interval["u_g"] + interval["u_ref"], sensitivity_slope=interval["S"],
                verdict=v, certified_disparity=int(v == "disparity"), certified_parity=int(v == "parity"))


cfg = dict(script="e5_sensitivities", state=STATE, seed=SEED, flip_seed=FLIP_SEED,
           gammas=GAMMAS, taus=TAUS, m_mins=M_MINS, noises=NOISES, J=JS,
           audit_n=AUDIT_N, z_modes=Z_MODES, count_mode="expected", data_source=SOURCE,
           proposed_m30=(30 in M_MINS), j10_gate=GATE)
cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
git = git_state()
rows = []

print(f"building score/data for seed={SEED}, flip_seed={FLIP_SEED}; source={SOURCE}", flush=True)
S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED, noise_sd=1.0)
print(f"evaluating gamma={GAMMAS}, noise={NOISES}, J={JS}, m_min={M_MINS}", flush=True)
for noise_sd in NOISES:
    for J in JS:
        strata = S.strata(J)
        key = S.codes * J + strata
        for gamma in GAMMAS:
            D, prop = S.select(gamma, S.eps0, selection_noise_sd=noise_sd)
            grouped = S.grouped(key, S.C * J, D, prop)
            delta, delta_dropped, delta_used = oracle_delta_info(grouped, S.C, J)
            ref = expected_cell(grouped, S.C, J, S.ref)
            for m_min in M_MINS:
                for c in S.others:
                    cell = expected_cell(grouped, S.C, J, c)
                    for tau in TAUS:
                        common = dict(seed=SEED, flip_seed=FLIP_SEED, gamma=gamma, noise_sd=noise_sd,
                                      J=J, m_min=m_min, tau=tau, cell=S.names[c], count_mode="expected",
                                      data_source=SOURCE, delta_oracle=delta, delta_entries_dropped=delta_dropped,
                                      delta_entries_used=delta_used, j10_gate_artifact=None if GATE is None else GATE["path"],
                                      j10_gate_cfg=None if GATE is None else GATE["cfg_hash"], git=git, cfg_hash=cfg_hash)
                        for _, z, audit_n in Z_MODES:
                            rows.append(row_for(cell, ref, delta=delta, z=z, audit_n=audit_n, m_min=m_min,
                                                S=S, metadata=common))

out = pd.DataFrame(rows)
Path("smoke tests/results").mkdir(exist_ok=True)
gamma_tag = "-".join(f"{g:.2f}" for g in GAMMAS)
source_tag = "SYNTHETIC" if SYNTHETIC else "ACS"
path = Path(f"smoke tests/results/e5_sensitivities_{source_tag}_{STATE}_s{SEED}_f{FLIP_SEED}_g{gamma_tag}.csv")
out.to_csv(path, index=False)

summary = (out.groupby(["gamma", "noise_sd", "J", "m_min", "tau", "z", "audit_n"], dropna=False, as_index=False)
           .agg(cells=("cell", "size"), certified_disparity_share=("certified_disparity", "mean"),
                certified_parity_share=("certified_parity", "mean"), median_width=("identified_width", "median"),
                median_delta=("delta_oracle", "median"), median_unexplored=("unexplored_mass", "median"),
                delta_entries_dropped=("delta_entries_dropped", "first")))
print(f"git={git} cfg={cfg_hash} source={SOURCE}; wrote {path}")
print(summary.round(4).to_string(index=False))
