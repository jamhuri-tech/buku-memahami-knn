"""Bab 7: k terbaik (menurut galat sebenarnya) pada data gelombang
untuk berbagai ukuran data n, dibandingkan dengan aturan sqrt(n) dan
dengan n^(2/3) (= n^(4/(4+p)) untuk p = 2, Samworth 2012).
Galat dirata-ratakan atas 20 data latih per n.
"""
import numpy as np
from bab04_sintetis import bangkitkan, eta
from bab07_loo import ramal_semua_k

BENIH = 20260929
U = np.random.default_rng(BENIH + 1).random((20_000, 2))
e = eta(U)[:, None]
print("    n   k*    galat(k*)  galat(sqrt n)  sqrt(n)  n^(2/3)")
for n in (100, 400, 1600, 6400):
    kmaks = min(n - 1, 1001)
    kurva = np.zeros(kmaks)
    for r in range(20):
        X, y = bangkitkan(n, np.random.default_rng(BENIH + 1000 * r + n))
        rm = ramal_semua_k(X, y, U, kmaks)
        kurva += np.mean(np.where(rm == 1, 1 - e, e), axis=0) / 20
    ganjil = np.arange(1, kmaks + 1, 2)
    kb = ganjil[np.argmin(kurva[ganjil - 1])]
    ka = int(np.sqrt(n)) | 1                  # ganjil terdekat ke atas
    print(f"{n:5d}  {kb:4d}   {kurva[kb - 1]:.4f}     {kurva[ka - 1]:.4f}"
          f"         {np.sqrt(n):4.0f}     {n ** (2 / 3):4.0f}")
