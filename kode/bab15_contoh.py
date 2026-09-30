"""Pemeriksa Contoh Soal Bab 15."""
import math

import numpy as np
from bab04_data import X
from bab15_pohon import Pohon

# ---- Contoh Soal 15.1: bangun pohon k-d, daun 2 ----
T = Pohon(X, daun=2, bola=False)
a = T.akar
assert list(a.lo) == [1, 1] and list(a.hi) == [8, 6]
kiri, kanan = a.kiri, a.kanan
assert (list(kiri.lo), list(kiri.hi)) == ([1, 1], [4, 6])
assert (list(kanan.lo), list(kanan.hi)) == ([5, 1], [8, 4])
daun = [kiri.kiri, kiri.kanan, kanan.kiri, kanan.kanan]
isi = [sorted(tuple(int(v) for v in X[i]) for i in d.idx) for d in daun]
assert isi == [[(1, 1), (3, 2)], [(2, 4), (4, 6)], [(5, 3), (6, 1)],
               [(7, 4), (8, 2)]]
kotak = [(list(d.lo), list(d.hi)) for d in daun]
assert kotak == [([1, 1], [3, 2]), ([2, 4], [4, 6]), ([5, 1], [6, 3]),
                 ([7, 2], [8, 4])]

# ---- Contoh Soal 15.2: telusur kueri (7, 1) ----
jejak = []
d, j = T.query([7, 1], k=1, jejak=jejak)
assert T.n_jarak == 2 and d[0] == 1 and tuple(X[j[0]]) == (6, 1)
assert [e[0] for e in jejak] == ["daun", "lewati", "lewati"]
assert math.isclose(jejak[1][2], 1) and math.isclose(jejak[2][2], 3)
d2 = ((X - [7, 1]) ** 2).sum(1)
assert sorted(d2) == [1, 2, 8, 9, 17, 34, 34, 36]
assert list(d2) == [36, 34, 17, 34, 8, 1, 9, 2]

# ---- Contoh Soal 15.3: batas bawah bola ----
P = X[[0, 1, 2, 3]].astype(float)
c = P.mean(0)
assert np.allclose(c, [2.5, 3.25])
r2 = ((P - c) ** 2).sum(1)
assert np.allclose(r2, [7.3125, 0.8125, 1.8125, 9.8125])
r = math.sqrt(9.8125)
dq = math.sqrt(4.5**2 + 2.25**2)
assert round(r, 3) == 3.132 and round(dq, 3) == 5.031
assert round(dq - r, 3) == 1.899
B = Pohon(X, daun=2, bola=True)
jj = []
B.query([7, 1], k=1, jejak=jj)
assert any(e[0] == "lewati" and round(e[2], 3) == 1.899 for e in jj)
assert round(math.sqrt(17), 3) == 4.123

# ---- Contoh Soal 15.4: kedalaman pohon ----
assert 2**14 < 25_000 <= 2**15 and 10**6 / 40 == 25_000
assert round(10**6 / 2**15, 1) == 30.5
assert round(math.log2(10**6 / 40), 1) == 14.6

print("Contoh Soal Bab 15: semua bilangan cocok")
