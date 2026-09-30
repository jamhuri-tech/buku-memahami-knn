"""Bab 9: galat k-NN terhadap n pada data gelombang untuk lima cara
memilih k: k = 1, k = 15 tetap, k = sqrt(n), k = n^(2/3), dan
k = n/5 (k/n tidak menuju nol). Rata-rata 10 data latih per n.
"""
import numpy as np
from bab04_sintetis import bangkitkan, eta
from bab07_loo import ramal_semua_k

BENIH = 20260929
U = np.random.default_rng(BENIH + 1).random((5_000, 2))
eU = eta(U)[:, None]
NS = (100, 400, 1600, 6400, 25600)


def ganjil(v):
    return int(v) | 1


ATURAN = {"k=1": lambda n: 1, "k=15": lambda n: 15,
          "sqrt(n)": lambda n: ganjil(np.sqrt(n)),
          "n^(2/3)": lambda n: ganjil(n ** (2 / 3)),
          "n/5": lambda n: ganjil(n / 5)}


def galat_aturan(n, ulang=10):
    rng = np.random.default_rng(BENIH + n)
    ks = {nm: a(n) for nm, a in ATURAN.items()}
    kmaks = max(ks.values())
    hasil = {nm: 0.0 for nm in ks}
    for _ in range(ulang):
        X, y = bangkitkan(n, rng)
        rm = ramal_semua_k(X, y, U, kmaks)
        g = np.mean(np.where(rm == 1, 1 - eU, eU), axis=0)
        for nm, k in ks.items():
            hasil[nm] += g[k - 1] / ulang
    return hasil


if __name__ == "__main__":
    print("galat Bayes 0.1356")
    print("     n   " + "  ".join(f"{nm:>7s}" for nm in ATURAN))
    for n in NS:
        h = galat_aturan(n)
        print(f"{n:6d}   " + "  ".join(f"{h[nm]:7.4f}" for nm in ATURAN))
