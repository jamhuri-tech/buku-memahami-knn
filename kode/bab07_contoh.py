"""Pemeriksa Contoh Soal Bab 7."""
from fractions import Fraction as F
from math import comb

import numpy as np

# ---- Contoh Soal 7.1: bias-varians, f(x) = |x|, sigma^2 = 6 ----
x = np.array([1, -2, 3, -4])
o = np.argsort(np.abs(x), kind="stable")
fv = [abs(int(v)) for v in x[o]]
assert fv == [1, 2, 3, 4]
galat = {}
for k in (1, 2, 3, 4):
    fb = F(sum(fv[:k]), k)
    b2, var = fb ** 2, F(6, k)
    galat[k] = 6 + b2 + var
assert galat == {1: 13, 2: F(45, 4), 3: 12, 4: F(55, 4)}
assert min(galat, key=galat.get) == 2
g1 = {k: F(sum(fv[:k]), k) ** 2 + F(1, k) for k in (1, 2, 3, 4)}
assert g1[1] == 2 and g1[2] == F(11, 4) and round(float(g1[3]), 2) == 4.33
assert g1[4] == F(13, 2) and min(g1, key=g1.get) == 1

# ---- Contoh Soal 7.2: galat di titik, eta = 0,7 ----
def q(k, e):
    return sum(comb(k, j) * e**j * (1 - e) ** (k - j)
               for j in range((k - 1) // 2 + 1))
e = F(7, 10)
assert q(1, e) == F(3, 10) and q(3, e) == F(216, 1000)
assert q(5, e) == F(16308, 100000)
assert F(3, 10) + F(4, 10) * q(3, e) == F(3864, 10000)
assert F(3, 10) + F(4, 10) * q(5, e) == F(365232, 1000000)
assert round(0.365232, 5) == 0.36523
assert F(3, 10) + F(4, 10) * q(1, e) == 2 * e * (1 - e) == F(42, 100)

# ---- Contoh Soal 7.3: LOO pada enam titik ----
xs = np.array([1, 2, 4, 5, 10, 12])
ys = np.array([0, 0, 1, 0, 1, 1])                 # A = 0, B = 1
D = np.abs(xs[:, None] - xs[None]).astype(float)
np.fill_diagonal(D, np.inf)
salah = {}
for k in (1, 3, 5):
    s = 0
    for i in range(6):
        d = np.sort(D[i])[:4]
        assert len(set(d)) == 4                     # tanpa seri
        o = np.argsort(D[i])[:k]
        s += int(ys[o].sum() * 2 > k) != ys[i]
    salah[k] = s
assert salah == {1: 2, 3: 1, 5: 6}
assert list(np.sort(np.abs(xs[[2, 1, 0, 4, 5]] - 5))) == [1, 3, 4, 5, 7]

# ---- Contoh Soal 7.4: matriks penghalus ----
xp = np.array([1, 2, 4, 7])
S = np.zeros((4, 4))
for i in range(4):
    o = np.argsort(np.abs(xp - xp[i]), kind="stable")[:2]
    S[i, o] = 0.5
assert np.allclose(S, [[.5, .5, 0, 0], [.5, .5, 0, 0], [0, .5, .5, 0],
                       [0, 0, .5, .5]])
assert np.trace(S) == 2 and np.allclose(S.sum(1), 1)
assert S[2, 1] == 0.5 and S[1, 2] == 0

print("Contoh Soal Bab 7: semua bilangan cocok")
