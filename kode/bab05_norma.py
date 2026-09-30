"""Bab 5: norma L_q untuk q -> tak hingga mendekati norma maksimum,
dan ketaksamaan antarnorma diperiksa pada vektor acak.
"""
import numpy as np

BENIH = 20260929
v = np.array([3.0, -4.0, 1.0])
for q in (1, 2, 4, 8, 16, 64):
    print(f"q = {q:2d}: ||v||_q = {np.sum(np.abs(v)**q)**(1/q):.6f}")
print(f"max |v_j|    = {np.abs(v).max():.6f}")

rng = np.random.default_rng(BENIH)
p = 50
V = rng.standard_normal((100_000, p))
n1 = np.abs(V).sum(1)
n2 = np.sqrt((V**2).sum(1))
ni = np.abs(V).max(1)
print(f"p = {p}, 100000 vektor acak:")
print(f"  ||v||_inf <= ||v||_2      : {np.all(ni <= n2)}")
print(f"  ||v||_2 <= ||v||_1        : {np.all(n2 <= n1)}")
print(f"  ||v||_1 <= sqrt(p) ||v||_2: {np.all(n1 <= np.sqrt(p) * n2)}")
print(f"  rasio ||v||_1/||v||_2 terbesar {np.max(n1 / n2):.3f}"
      f", batas sqrt(p) = {np.sqrt(p):.3f}")
