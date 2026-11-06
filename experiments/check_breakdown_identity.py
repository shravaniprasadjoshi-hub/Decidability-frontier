"""
experiments/check_breakdown_identity.py

to verify that d*_par > dlt is mathematically equivalent to tau_par_min0 < tau (where tau_par_min0 = tau - max(Dhi, -Dlo)
"""
import pandas as pd
import numpy as np
from bounds import Cell, endpoints, breakdown
from common import build, J, M_MIN_DEFAULT
import os

STATE, SEED, FLIP_SEED = "CA", 0, 123
M_MIN = M_MIN_DEFAULT
TAUS = [0.05, 0.02]

print(f"Checking breakdown identity for {STATE}, Seed {SEED}, Flip {FLIP_SEED}")
S = build(state=STATE, seed=SEED, flip_seed=FLIP_SEED)
Y, n = S.Y, S.n
codes, C, J = S.codes, S.C, J
ref, others, names = S.ref, S.others, S.names
s5, key = S.s5, S.key
select, grouped, amax = S.select, S.grouped, S.amax

# FIXED PATH
csv_path = f"smoke tests/results/a12_controlled_{STATE}_{SEED}_{FLIP_SEED}.csv"
df = pd.read_csv(csv_path)

for g in df['gamma'].unique():
    df_g = df[df['gamma'] == g]
    dlt = df_g['delta_shared'].iloc[0]
    
    D, prop = select(g, S.eps0); gr = grouped(key, C * J, D, prop)
    cl = lambda c: Cell(gr["n"].reshape(C, J)[c], gr["na"].reshape(C, J)[c], gr["ka"].reshape(C, J)[c])
    er_0 = endpoints(cl(ref), 0.0, M_MIN)
    
    print(f"\n--- Gamma: {g} (delta_shared = {dlt:.4f}) ---")
    for tau in TAUS:
        mismatches = 0
        for _, row in df_g.iterrows():
            c_name = row['cell']
            c_idx = names.get_loc(c_name)
            cg_0 = endpoints(cl(c_idx), 0.0, M_MIN)
            
            Dhi_0 = cg_0["U"] - er_0["L"]
            Dlo_0 = cg_0["L"] - er_0["U"]
            S_total = cg_0["S"] + er_0["S"]
            
            # RENAMED to tau_minus_margin
            tau_minus_margin = tau - max(Dhi_0, -Dlo_0)
            theoretical_d_star = tau_minus_margin / S_total if S_total > 0 else np.inf
            
            d_star_par = row[f'd*_par@{tau}']
            
            # NaN-safe assertion check
            bad = not np.isfinite(d_star_par) or not np.isfinite(theoretical_d_star) or (np.abs(d_star_par - theoretical_d_star) > 1e-5)
            
            if bad:
                mismatches += 1
                if mismatches <= 3:
                    diff_val = np.abs(d_star_par - theoretical_d_star) if np.isfinite(d_star_par) and np.isfinite(theoretical_d_star) else "NaN"
                    print(f"  MISMATCH {c_name} @tau={tau}: d*_par={d_star_par:.5f}, theory={theoretical_d_star:.5f}, diff={diff_val}")
                    
        print(f"  Tau={tau}: {'PASS' if mismatches == 0 else f'FAIL ({mismatches} mismatches)'}")