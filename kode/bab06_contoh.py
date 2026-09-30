"""Pemeriksa Contoh Soal Bab 6."""
import math
from fractions import Fraction as F

import numpy as np
from scipy.stats import norm

BENIH = 20260929

# ---- Contoh Soal 6.1: E||X - Z||^2 = 2 tr Sigma ----
assert F(10000, 10004) > F(9995, 10000)
assert round(100 * 10000 / 10004, 2) == 99.96
assert round(100 * 4 / 10004, 2) == 0.04
assert 2 * 10004 == 20008
assert round(math.sqrt(2) * 100) == 141 and round(math.sqrt(2) * 2, 1) == 2.8
rng = np.random.default_rng(BENIH)
S = np.array([[4.0, 30.0], [30.0, 10000.0]])        # korelasi tidak berperan
Xs = rng.multivariate_normal([0, 0], S, size=(400_000, 2))
m = np.mean(((Xs[:, 0] - Xs[:, 1]) ** 2).sum(1))
assert abs(m / (2 * np.trace(S)) - 1) < 0.01

# ---- Contoh Soal 6.2: lima nilai dengan pencilan ----
v = np.array([2, 4, 6, 8, 30], dtype=float)
mm = (v - 2) / 28
assert [F(int(t) - 2, 28) for t in v] == [0, F(1, 14), F(1, 7), F(3, 14), 1]
assert v.mean() == 10 and ((v - 10) ** 2).sum() == 520 and v.var() == 104
z = (v - 10) / math.sqrt(104)
assert np.allclose(np.round(z, 3), [-0.784, -0.588, -0.392, -0.196, 1.961])
assert np.percentile(v, 25) == 4 and np.percentile(v, 75) == 8
assert np.median(v) == 6
assert list((v - 6) / 4) == [-1, -0.5, 0, 0.5, 6]
from sklearn.preprocessing import RobustScaler, StandardScaler
assert np.allclose(RobustScaler().fit_transform(v[:, None]).ravel(),
                   (v - 6) / 4)
assert np.allclose(StandardScaler().fit_transform(v[:, None]).ravel(), z)
assert round(6 / 28, 3) == 0.214 and round(6 / math.sqrt(104), 3) == 0.588
assert 6 / 4 == 1.5

# ---- Contoh Soal 6.3: tetangga berganti ----
x, a, b = np.array([2, 5000]), np.array([3, 5200]), np.array([9, 5050])
assert ((x - a) ** 2).sum() == 40001 and ((x - b) ** 2).sum() == 2549
s = np.array([2, 100])
assert np.isclose((((x - a) / s) ** 2).sum(), 4.25)
assert np.isclose((((x - b) / s) ** 2).sum(), 12.5)

# ---- Contoh Soal 6.4: peluang bertukar ----
# momen normal baku: E u^2 = 1, E u^4 = 3
assert 3 - 2 + 3 == 4 and 4 * 1 * 2 == 8
assert 4 + 8 == 12 and math.sqrt(12 * 12) == 12
assert round(norm.cdf(-0.5), 4) == 0.3085
# (e - u)^2 = 2 chi^2_1, varians 4 * 2 = 8; dua bebas -> 16 m
assert 4 * 2 == 8
e, u, w = rng.standard_normal((3, 2_000_000))
t = (e - u) ** 2 - (e - w) ** 2
assert abs(t.mean()) < 0.01 and abs(t.var() - 12) < 0.1

print("Contoh Soal Bab 6: semua bilangan cocok")
