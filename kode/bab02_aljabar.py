"""Bab 2: hasil kali dalam dan sudut, dekomposisi eigen, akar invers,
Cholesky, dan fungsi gamma, dengan NumPy dan SciPy.
"""
import numpy as np
from scipy.special import gamma

u = np.array([1.0, 2.0, 2.0])
v = np.array([2.0, 0.0, 1.0])
c = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
print(f"u.v = {u @ v:.0f}, ||u|| = {np.linalg.norm(u):.0f}, "
      f"||v|| = {np.linalg.norm(v):.4f}")
print(f"cos sudut = {c:.4f}, sudut = {np.degrees(np.arccos(c)):.2f} derajat")

S = np.array([[2.0, 1.0], [1.0, 2.0]])
lam, V = np.linalg.eigh(S)
print("nilai eigen:", lam, " vektor eigen (kolom):")
print(np.round(V, 4))
akar_inv = V @ np.diag(lam ** -0.5) @ V.T
print("Sigma^(-1/2) =")
print(np.round(akar_inv, 4))
print("periksa: S^(-1/2) S S^(-1/2) =",
      np.round(akar_inv @ S @ akar_inv, 10).tolist())

A = np.array([[4.0, 2.0], [2.0, 5.0]])
print("Cholesky [[4,2],[2,5]]:", np.linalg.cholesky(A).tolist())
for s in (0.5, 1, 1.5, 2, 2.5, 5):
    print(f"Gamma({s}) = {gamma(s):.6f}")
print(f"sqrt(pi) = {np.sqrt(np.pi):.6f}, 3 sqrt(pi)/4 = "
      f"{3 * np.sqrt(np.pi) / 4:.6f}")
