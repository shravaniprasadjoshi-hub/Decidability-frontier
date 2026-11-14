"""E6 (owner: Rohini): reproducible adversarial clipped-bound kink check (appendix only).

All strata are placed at the lower clipping point mu_obs=delta.  The output
contains both the empirical standardized bias and the pre-specified heuristic
0.41*sqrt(J); it is a stress test, not a validity theorem.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from common import git_state

DELTA = float(os.environ.get("DELTA", 0.03))
PI = float(os.environ.get("PI", 0.30))
R = int(os.environ.get("R", 2000))
SEED = int(os.environ.get("SEED", 0))
SETTINGS = ((1, 10_000), (5, 10_000), (10, 10_000), (20, 10_000), (50, 10_000), (20, 50_000))


def run(J, n, rng):
    w = np.full(J, 1 / J)
    nj = rng.multinomial(n, w, size=R)
    na = rng.binomial(nj, PI)
    k = rng.binomial(na, DELTA)
    mu = k / np.maximum(na, 1)
    rho = 1 - na / np.maximum(nj, 1)
    # True lower endpoint equals pi*delta because every stratum is at the kink.
    truth = PI * DELTA
    lo = np.sum((nj / n) * (PI * mu + rho * np.maximum(0, mu - DELTA)), axis=1)
    bias, sd = float(lo.mean() - truth), float(lo.std(ddof=1))
    return dict(J=J, n=n, reps=R, empirical_bias=bias, empirical_sd=sd,
                bias_over_sd=bias / sd if sd else np.nan,
                predicted_bias_over_sd=0.41 * np.sqrt(J),
                prediction_ratio=(bias / sd) / (0.41 * np.sqrt(J)) if sd else np.nan)


cfg = dict(script="e6_adversarial_kink", delta=DELTA, pi=PI, reps=R, seed=SEED, settings=SETTINGS)
cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
git = git_state()
rng = np.random.default_rng(SEED)
df = pd.DataFrame([run(J, n, rng) for J, n in SETTINGS])
df["delta"], df["pi"], df["git"], df["cfg_hash"] = DELTA, PI, git, cfg_hash
os.makedirs("smoke tests/results", exist_ok=True)
path = f"smoke tests/results/e6_adversarial_kink_s{SEED}.csv"
df.to_csv(path, index=False)
print(f"git={git} cfg={cfg_hash}; wrote {path}")
print(df.round(4).to_string(index=False))
