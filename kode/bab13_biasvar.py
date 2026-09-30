"""Bab 13: galat kuadrat taksiran peluang eta_hat = k_1/k, dihitung
EKSAK terhadap label acak untuk satu data latih tetap (posisi tetap):
  E(eta_hat - eta(x))^2 = (eta_bar - eta(x))^2 + sum_i eta_i(1-eta_i)/k^2,
dengan eta_i posterior di tetangga ke-i. Data gelombang, n = 400.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from bab04_sintetis import bangkitkan, eta

BENIH = 20260929
rng = np.random.default_rng(BENIH)
X, _ = bangkitkan(400, rng)
U = np.random.default_rng(BENIH + 1).random((20_000, 2))
_, idx = NearestNeighbors(n_neighbors=201).fit(X).kneighbors(U)
eX, eU = eta(X), eta(U)
print("  k    bias^2    varians   jumlah")
terbaik = None
for k in (1, 5, 15, 31, 61, 121, 201):
    en = eX[idx[:, :k]]
    b2 = np.mean((en.mean(1) - eU) ** 2)
    v = np.mean((en * (1 - en)).sum(1) / k**2)
    print(f"{k:3d}   {b2:.5f}   {v:.5f}   {b2 + v:.5f}")
ks = np.arange(1, 202)
tot = [np.mean((eX[idx[:, :k]].mean(1) - eU) ** 2)
       + np.mean((eX[idx[:, :k]] * (1 - eX[idx[:, :k]])).sum(1) / k**2)
       for k in ks]
print(f"k terbaik untuk peluang: {ks[int(np.argmin(tot))]}")
