"""Bab 9: galat asimtotik k-NN (k tetap, ganjil) pada data gelombang,
R_k = E[r + (1 - 2r) P(Bin(k, r) > k/2)], r = min(eta, 1 - eta),
dibandingkan dengan batas R* + 1/sqrt(k e) dan dengan simulasi pada
n = 20000 titik latih.
"""
import numpy as np
from scipy.stats import binom
from bab04_sintetis import bangkitkan, eta
from bab07_loo import ramal_semua_k

BENIH = 20260929
t = (np.arange(1001) + 0.5) / 1001
G = np.column_stack([np.repeat(t, len(t)), np.tile(t, len(t))])
e = eta(G)
r = np.minimum(e, 1 - e)
rb = r.mean()
ks = (1, 3, 5, 11, 21, 51)
rk = {k: np.mean(r + (1 - 2 * r) * binom.sf(k // 2, k, r)) for k in ks}

rng = np.random.default_rng(BENIH)
U = np.random.default_rng(BENIH + 1).random((20_000, 2))
eU = eta(U)[:, None]
sim = np.zeros(51)
for _ in range(5):
    X, y = bangkitkan(20_000, rng)
    rm = ramal_semua_k(X, y, U, 51)
    sim += np.mean(np.where(rm == 1, 1 - eU, eU), axis=0) / 5
print(f"R* = {rb:.4f}")
print("  k   R_k (rumus)   simulasi n=20000   R* + 1/sqrt(k e)")
for k in ks:
    print(f"{k:3d}   {rk[k]:.4f}        {sim[k - 1]:.4f}"
          f"             {rb + 1 / np.sqrt(k * np.e):.4f}")
