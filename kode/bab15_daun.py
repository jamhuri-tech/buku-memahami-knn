"""Bab 15: pengaruh ukuran daun pada pohon k-d (10000 titik seragam di
R^4, k = 5): jarak ke titik, simpul yang dikunjungi, dan banyaknya
simpul pohon.
"""
import numpy as np
from bab15_pohon import Pohon

BENIH = 20260929


def banyak_simpul(s):
    return 1 if s.idx is not None else 1 + banyak_simpul(s.kiri) + \
        banyak_simpul(s.kanan)


rng = np.random.default_rng(BENIH)
X = rng.random((10_000, 4))
Q = rng.random((100, 4))
print(" daun   simpul pohon   jarak/kueri   simpul dikunjungi/kueri")
for daun in (1, 5, 20, 80, 320):
    T = Pohon(X, daun, False)
    j = s = 0
    for q in Q:
        T.query(q, k=5)
        j += T.n_jarak
        s += T.n_simpul
    print(f"{daun:5d}   {banyak_simpul(T.akar):8d}       {j / 100:8.1f}"
          f"        {s / 100:8.1f}")
