"""Pemeriksa Contoh Soal Bab 17."""
import math
from fractions import Fraction as F

import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
from bab17_kondensasi import kondensasi

# ---- Contoh Soal 17.1: kondensasi Hart di garis bilangan ----
x = np.arange(1, 9, dtype=float)[:, None]
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])        # A = 0, B = 1
S, lintasan = kondensasi(x, y)
assert sorted((S + 1).tolist()) == [1, 3, 4, 5, 6] and lintasan == 2
assert list(S + 1) == [1, 4, 3, 5, 6]         # urutan masuk
from sklearn.neighbors import KNeighborsClassifier
yh = KNeighborsClassifier(1).fit(x[S], y[S]).predict(x)
assert np.all(yh == y)

# ---- Contoh Soal 17.2: penyuntingan Wilson di satu daerah ----
e = F(3, 10)
lolos1 = 3 * e**2 * (1 - e) + e**3
lolos0 = (1 - e) ** 3 + 3 * e * (1 - e) ** 2
assert lolos1 == F(216, 1000) and lolos0 == F(784, 1000)
e2 = e * lolos1 / (e * lolos1 + (1 - e) * lolos0)
assert e * lolos1 == F(648, 10000) and (1 - e) * lolos0 == F(5488, 10000)
assert round(float(e2), 4) == 0.1056
g0 = 2 * e * (1 - e)
g1 = e2 * (1 - e) + (1 - e2) * e
assert g0 == F(42, 100) and round(float(g1), 4) == 0.3422
assert round(0.1056 * 0.7, 4) == 0.0739 and round(0.8944 * 0.3, 4) == 0.2683

# ---- Contoh Soal 17.3: akurasi seimbang ----
assert F(3, 10) / F(1, 10) == 3 and round(0.7 / 0.9, 2) == 0.78
assert F(3, 10) < F(7, 10)

# ---- Contoh Soal 17.4: jarak nan_euclidean ----
a = np.array([[1, np.nan, 3, 4]])
b = np.array([[2, 5, np.nan, 6]])
d = nan_euclidean_distances(a, b)[0, 0]
assert math.isclose(d, math.sqrt(10)) and round(d, 4) == 3.1623
assert round(math.sqrt(5), 2) == 2.24

print("Contoh Soal Bab 17: semua bilangan cocok")
