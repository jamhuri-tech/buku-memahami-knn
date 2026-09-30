"""Pemeriksa Contoh Soal Bab 18."""
import math

import numpy as np
from bab18_nca import nca_f_grad

# ---- Contoh Soal 18.1: pembobotan fitur dan pangkat M ----
L = np.diag([2, 0.5])
M = L.T @ L
assert np.allclose(M, np.diag([4, 0.25]))
for v in ([1, 1], [1, -1]):
    assert math.isclose(np.array(v) @ M @ np.array(v), 4.25)
L2 = np.array([[1, 0, 0], [0, 1, 0]])
assert np.allclose(L2 @ [0, 0, 5], 0)
assert np.linalg.matrix_rank(L2.T @ L2) == 2

# ---- Contoh Soal 18.2: tiga titik di garis ----
X = np.array([[0.0], [1.0], [2.0]])
y = np.array([0, 0, 1])
f = lambda l: nca_f_grad(np.array([[l]]), X, y)[0]
for l in (0.3, 1.0, 2.0):
    assert math.isclose(f(l), 1 / (1 + math.exp(-3 * l * l)) + 0.5)
assert round(math.exp(-3), 5) == 0.04979
assert round(1 / 1.04979, 4) == 0.9526 and round(f(1.0), 4) == 1.4526
df = 6 * math.exp(-3) / (1 + math.exp(-3)) ** 2
assert round(df, 4) == 0.2711
assert round(6 * 0.04979, 5) == 0.29874 and round(1.04979**2, 5) == 1.10206
assert math.isclose(f(1e-9), 1.0) and abs(f(10.0) - 1.5) < 1e-12

# ---- Contoh Soal 18.3: gradien rumus di l = 1 ----
_, G = nca_f_grad(np.array([[1.0]]), X, y)
assert round(G[0, 0], 4) == round(df, 4)
p12, p13 = 0.9526, 0.0474
a = p12 * (p12 * 1 + p13 * 4)
assert round(p12 * 1 + p13 * 4, 4) == 1.1422 and round(a, 4) == 1.0881
assert round(a - p12, 4) == 0.1355 and round(2 * (a - p12), 3) == 0.271

# ---- Contoh Soal 18.4: banyaknya parameter dan perkalian ----
assert 39 * 39 == 1521 and 1521 / 178 > 8 and 2 * 39 == 78
assert 1000 * 2 * 64 == 128_000
assert round(1000 * 1347 * 64 / 1e7, 1) == 8.6
assert round(1000 * 1347 * 2 / 1e6, 1) == 2.7

print("Contoh Soal Bab 18: semua bilangan cocok")
