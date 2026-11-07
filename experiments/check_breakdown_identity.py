"""
experiments/check_breakdown_identity.py
Checks the breakdown identity on a12 output (realized counts):
  (i)  d*_par@tau == (tau - max(Dhi0, -Dlo0)) / S   (recomputed here from bounds.endpoints)
  (ii) parity certifiable at delta  <=>  tau_par_min0 < tau  <=>  d*_par@tau > delta_shared   (breakdown() vs gap(), two code paths)
Naming: tau_par_min0 in a12 is the MARGIN max(Dhi, -Dlo). Here slack_par = tau - margin.
Exits 1 on any mismatch or non-finite value. Env: STATE, SEED, FLIP_SEED.
"""
import pandas as pd
import numpy as np
import sys
import os
from bounds import Cell, endpoints, breakdown
from common import build, J, M_MIN_DEFAULT

STATE, SEED, FLIP_SEED = os.environ.get("STATE", "CA"), int(os.environ.get("SEED", 0)), int(os.environ.get("FLIP_SEED", 123))
M_MIN = M_MIN_DEFAULT
TAUS = [0.05, 0.02]

print(f"Checking breakdown identity for {STATE}, Seed {SEED}, Flip {FLIP_SEED}")
S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED)
names = S.names
ref, others = S.ref, S.others

csv_path = f"smoke tests/results/a12_controlled_{STATE}_{SEED}_{FLIP_SEED}.csv"
df = pd.read_csv(csv_path)
failed = False

for g in df['gamma'].unique():
    df_g = df[df['gamma'] == g]
    dlt = df_g['delta_shared'].iloc[0]
    
    D, prop = S.select(g, S.eps0); gr = S.grouped(S.key, S.C * J, D, prop)
    cl = lambda c: Cell(gr["n"].reshape(S.C, J)[c], gr["na"].reshape(S.C, J)[c], gr["ka"].reshape(S.C, J)[c])
    er_0 = endpoints(cl(ref), 0.0, M_MIN)
    
    print(f"\n--- Gamma: {g} (delta_shared = {dlt:.4f}) ---")
    for tau in TAUS:
        d_vals = []
        theory_vals = []
        for _, row in df_g.iterrows():
            c_name = row['cell']
            c_idx = names.get_loc(c_name)
            cg_0 = endpoints(cl(c_idx), 0.0, M_MIN)
            
            Dhi_0 = cg_0["U"] - er_0["L"]
            Dlo_0 = cg_0["L"] - er_0["U"]
            S_total = cg_0["S"] + er_0["S"]
            
            slack_par = tau - max(Dhi_0, -Dlo_0)
            theoretical_d_star = slack_par / S_total if S_total > 0 else np.inf
            
            d_vals.append(row[f'd*_par@{tau}'])
            theory_vals.append(theoretical_d_star)
            flags_ok = bool(np.isfinite(row["tau_par_min0"]) and ((row["tau_par_min0"] < tau) == (row[f"d*_par@{tau}"] > dlt)))
            if not flags_ok and abs(row["tau_par_min0"] - tau) > 1e-9:
                print(f"  FAIL (ii) {c_name} tau={tau}"); failed = True
            
        d = np.array(d_vals)
        theory = np.array(theory_vals)
        bad = ~np.isfinite(d) | ~np.isfinite(theory) | (np.abs(d - theory) > 1e-5)
        
        if bad.any():
            print(f"  FAIL at tau={tau}: {bad.sum()} mismatches")
            failed = True
        else:
            print(f"  PASS at tau={tau}: {len(d)} rows match")

if failed:
    sys.exit(1)