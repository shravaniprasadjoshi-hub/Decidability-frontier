"""
checks core/common.py build() gives the same numbers as the old inline a12 script.

usage: 
set PYTHONPATH=core 
python experiments/check_common_vs_a12.py "smoke tests/a12_controlled_CA_0_123.csv"

compares per-cell Delta_true and n, and the shared oracle delta per gamma. if this fails the RNG order in common.py is off

"""
import sys, numpy as np, pandas as pd
from common import build, J

csv = sys.argv[1] if len(sys.argv) > 1 else "a12_controlled_CA_0_123.csv"
state, seed, flip = "CA", 0, 123
b = build(state=state, seed=seed, flip_seed=flip)
ref = pd.read_csv(csv)
ok = True

# per cell: Delta_true and n
cells = ref.drop_duplicates("cell").set_index("cell")
for c in b.others:
    r = cells.loc[b.names[c]]
    dn = abs(r.Delta_true - b.Dtrue[c]); nn = int(r.n) - int((b.codes == c).sum())
    if dn > 1e-9 or nn != 0: ok = False; print("MISMATCH cell", b.names[c], dn, nn)

# shared oracle delta per gamma (propensity, max over cell x stratum)
for g in sorted(ref.gamma.unique()):
    D, prop = b.select(g)
    gr = b.grouped(b.key, b.C * J, D, prop)
    d = b.amax(b.dgap(gr)[1])
    want = ref[np.isclose(ref.gamma, g)].delta_shared.iloc[0]
    print(f"gamma {g}: common {d:.6f} vs a12 csv {want:.6f}")
    if abs(d - want) > 1e-6: ok = False

print("PASS: common.build matches a12" if ok else "FAIL: common.build differs from a12")
sys.exit(0 if ok else 1)
