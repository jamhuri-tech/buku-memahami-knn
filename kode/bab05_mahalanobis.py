"""Bab 5: jarak Mahalanobis sama dengan jarak Euclid sesudah
transformasi L, dengan M = L^T L (Cholesky), dan sama dengan
metric="mahalanobis" milik scikit-learn.
"""
import numpy as np
from sklearn.metrics import pairwise_distances

BENIH = 20260929
S = np.array([[2.0, 1.0], [1.0, 2.0]])
M = np.linalg.inv(S)
print("Sigma^-1 * 3 =\n", np.round(3 * M, 10))
for v in ([1, 1], [1, -1]):
    v = np.array(v, dtype=float)
    print(f"v = {v}: d_M^2 = {v @ M @ v:.6f}, ||v||^2 = {v @ v:.1f}")
C = np.linalg.cholesky(M)          # M = C C^T, jadi L = C^T
L = C.T
print("L =\n", np.round(L, 6))
rng = np.random.default_rng(BENIH)
Z = rng.standard_normal((200, 2))
D1 = pairwise_distances(Z, metric="mahalanobis", VI=M)
D2 = pairwise_distances(Z @ L.T)
print(f"selisih terbesar Mahalanobis vs Euclid(Lx): "
      f"{np.abs(D1 - D2).max():.2e}")
