"""
experiments/check_breakdown_identity.py
Verifies the breakdown identity: d*_par > dlt iff tau_minus_margin < tau.
tau_minus_margin = tau - max(Dhi, -Dlo).
Fails with exit code 1 if any mismatch (including NaN) is found.
"""
import pandas as pd
import numpy as np
import sys
from bounds import Cell, endpoints, breakdown
from common import build, J, M_MIN_DEFAULT

STATE, SEED, FLIP_SEED = "CA", 0, 123
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
            
            tau_minus_margin = tau - max(Dhi_0, -Dlo_0)
            theoretical_d_star = tau_minus_margin / S_total if S_total > 0 else np.inf
            
            d_vals.append(row[f'd*_par@{tau}'])
            theory_vals.append(theoretical_d_star)
            
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