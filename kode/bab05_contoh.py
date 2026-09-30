"""Pemeriksa Contoh Soal Bab 5."""
import math
from fractions import Fraction as F

import numpy as np

# ---- Contoh Soal 5.1: tiga metrik, tiga tetangga ----
P = {"a": (6, 0), "b": (4, 4), "c": (2, 5)}
L1 = {k: abs(x) + abs(y) for k, (x, y) in P.items()}
L2 = {k: x * x + y * y for k, (x, y) in P.items()}
Li = {k: max(abs(x), abs(y)) for k, (x, y) in P.items()}
assert L1 == {"a": 6, "b": 8, "c": 7}
assert L2 == {"a": 36, "b": 32, "c": 29}
assert Li == {"a": 6, "b": 4, "c": 5}
assert min(L1, key=L1.get) == "a"
assert min(L2, key=L2.get) == "c"
assert min(Li, key=Li.get) == "b"
assert round(math.sqrt(29), 3) == 5.385

# ---- Contoh Soal 5.2: q = 1/2 bukan metrik ----
def n_half(v):
    return sum(math.sqrt(abs(t)) for t in v) ** 2
assert n_half((1, 1)) == 4 and n_half((1, 0)) == 1 and n_half((0, 1)) == 1
assert n_half((1, 1)) > n_half((1, 0)) + n_half((0, 1))
assert math.isclose(n_half((0.5, 0.5)), 2)
assert math.isclose(n_half((3 * 0.5, 3 * 0.5)), 3 * n_half((0.5, 0.5)))

# ---- Contoh Soal 5.3: Mahalanobis ----
S = np.array([[2, 1], [1, 2]])
assert round(np.linalg.det(S)) == 3
M = [[F(2, 3), F(-1, 3)], [F(-1, 3), F(2, 3)]]
def quad(v):
    return sum(v[i] * M[i][j] * v[j] for i in range(2) for j in range(2))
assert quad((1, 1)) == F(2, 3) and quad((1, -1)) == 2
assert np.allclose(np.linalg.inv(S), np.array(M, dtype=float))
l11, l12, l22 = math.sqrt(2 / 3), -1 / math.sqrt(6), 1 / math.sqrt(2)
L = np.array([[l11, l12], [0, l22]])
assert np.allclose(L.T @ L, np.array(M, dtype=float))
assert (round(l11, 4), round(l12, 4), round(l22, 4)) == (0.8165, -0.4082,
                                                        0.7071)
assert F(2, 3) - F(1, 6) == F(1, 2)
assert np.allclose(S @ [1, 1], [3, 3]) and np.allclose(S @ [1, -1],
                                                       [1, -1])

# ---- Contoh Soal 5.4: kosinus dan sudut ----
u, v, w = np.array([1, 0]), np.array([0, 1]), np.array([1, 1]) / 2**0.5
cos = lambda a, b: a @ b
assert cos(u, v) == 0 and math.isclose(cos(u, w), 1 / 2**0.5)
dk = lambda a, b: 1 - cos(a, b)
assert round(dk(u, w), 4) == 0.2929
assert dk(u, w) + dk(w, v) < dk(u, v)
assert round(2 - 2**0.5, 4) == 0.5858
ang = lambda a, b: math.acos(max(-1, min(1, cos(a, b)))) / math.pi
assert math.isclose(ang(u, v), 0.5) and math.isclose(ang(u, w), 0.25)
assert math.isclose(ang(u, w) + ang(w, v), ang(u, v))
assert math.isclose(((u - w) ** 2).sum(), 2 - 2**0.5)

# ---- Contoh Soal 5.5: Gower ----
d = [abs(30 - 50) / 40, 1, 0, abs(5 - 8) / 10]
assert d == [0.5, 1, 0, 0.3] and math.isclose(sum(d) / 4, 0.45)
e2 = 20**2 + 1 + 0 + 3**2
assert e2 == 410 and round(math.sqrt(410), 2) == 20.25
assert round(400 / 410 * 100, 1) == 97.6

# ---- Contoh Soal 5.6: Malang - Surabaya ----
R, r = 6371, 0.017453
dy = R * 0.73 * r
dx = R * 0.12 * r * 0.9912
assert round(dy, 2) == 81.17 and round(dx, 2) == 13.23
assert round(math.cos(math.radians(7.615)), 4) == 0.9912
assert round(81.17**2, 1) == 6588.6 and round(13.23**2, 1) == 175.0
assert round(math.sqrt(6588.6 + 175.0), 2) == 82.24
assert round(math.hypot(0.73, 0.12), 2) == 0.74
assert round(math.cos(math.radians(37)), 2) == 0.80

print("Contoh Soal Bab 5: semua bilangan cocok")
