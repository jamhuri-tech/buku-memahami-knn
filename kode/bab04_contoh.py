"""Pemeriksa Contoh Soal Bab 4 (bilangan bulat dan pecahan eksak)."""
from fractions import Fraction as F
from itertools import product
from math import comb

import numpy as np
from bab04_data import X, y

# ---- Contoh Soal 4.1: titik uji (4, 3) ----
q = np.array([4, 3])
d2 = ((X - q) ** 2).sum(1)
o = np.argsort(d2, kind="stable")
assert list(d2[o]) == [1, 2, 5, 8, 9, 10, 13, 17]
assert len(set(d2)) == 8                     # urutan tunggal
assert [tuple(X[i]) for i in o[:3]] == [(5, 3), (3, 2), (2, 4)]
for k, (a, b) in {1: (0, 1), 3: (2, 1), 5: (3, 2)}.items():
    s = np.bincount(y[o[:k]], minlength=2)
    assert tuple(s) == (a, b) and a + b == k
from sklearn.neighbors import KNeighborsClassifier
for k, ramal in ((1, 1), (3, 0), (5, 0)):
    m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    assert m.predict([q])[0] == ramal

# ---- Contoh Soal 4.2: garis bagi dan titik sudut Voronoi ----
a, b, c = np.array([3, 2]), np.array([5, 3]), np.array([2, 4])
assert tuple(b - a) == (2, 1) and b @ b == 34 and a @ a == 13
# 2 (b - a) . x <= |b|^2 - |a|^2  ->  4 x1 + 2 x2 <= 21
assert 2 * 4 + 2.5 == 10.5                   # titik tengah (4; 2,5)
assert 2 * 4 + 3 == 11 > 10.5                # (4, 3) di sisi b
assert tuple(c - a) == (-1, 2) and c @ c - a @ a == 7
A = np.array([[2, 1], [-1, 2]], dtype=float)
v = np.linalg.solve(A, [10.5, 3.5])
assert np.allclose(v, [3.5, 3.5])
for t in (a, b, c):
    assert np.isclose(((v - t) ** 2).sum(), 2.5)

# ---- Contoh Soal 4.3: seri di (4, 0) ----
q = np.array([4, 0])
d2 = ((X - q) ** 2).sum(1)
tabel = {tuple(X[i]): int(d2[i]) for i in range(8)}
assert tabel == {(3, 2): 5, (6, 1): 5, (1, 1): 10, (5, 3): 10,
                 (2, 4): 20, (8, 2): 20, (4, 6): 36, (7, 4): 25}
bb, aa = np.array([6, 1]), np.array([3, 2])
assert tuple(bb - aa) == (3, -1) and bb @ bb - aa @ aa == 24
assert 3 * 4 - 0 == 12                       # 3 x1 - x2 = 12
urut = sorted(d2)
assert urut[0] == urut[1] and urut[2] == urut[3]   # seri di k=1 dan k=3

# ---- Contoh Soal 4.4: peluang seri suara ----
def pseri(k, eta):
    return comb(k, k // 2) * eta ** (k // 2) * (1 - eta) ** (k // 2)
assert pseri(4, F(1, 2)) == F(3, 8)
assert abs(pseri(4, 0.3) - 0.2646) < 1e-12
beda = sum(1 for t in product(range(3), repeat=3) if len(set(t)) == 3)
assert beda == 6 and F(beda, 27) == F(2, 9)
assert round(2 / 9, 3) == 0.222

print("Contoh Soal Bab 4: semua bilangan cocok")
