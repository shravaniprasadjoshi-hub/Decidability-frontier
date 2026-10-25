"""MC check of the certificate se at larger delta (open item: only checked at delta=0.03).
Run: python se_check.py   (needs bounds.py, numpy). Prints mean(se)/empirical sd for both endpoints; want ~1.00 (+-0.05)."""
import numpy as np
from bounds import Cell, endpoints
rng = np.random.default_rng(0); J, R = 5, 3000
w = np.ones(J) / J; pi = np.linspace(.9, .3, J); mu = np.linspace(.03, .70, J)
print(" delta |    n | se_L/sd_L  se_U/sd_U | (no composition term: L, U)")
for delta in (0.03, 0.10):
    for n in (200, 600, 2000):
        L, U, sL, sU, sL0, sU0 = [], [], [], [], [], []
        for _ in range(R):
            nj = rng.multinomial(n, w); na = rng.binomial(nj, pi); k = rng.binomial(na, mu); c = Cell(nj, na, k)
            e, e0 = endpoints(c, delta), endpoints(c, delta, between=False)
            L.append(e["L"]); U.append(e["U"]); sL.append(e["se_L"]); sU.append(e["se_U"]); sL0.append(e0["se_L"]); sU0.append(e0["se_U"])
        print(f" {delta:.2f}  | {n:4d} |  {np.mean(sL)/np.std(L):.3f}      {np.mean(sU)/np.std(U):.3f}    | {np.mean(sL0)/np.std(L):.3f} {np.mean(sU0)/np.std(U):.3f}")
