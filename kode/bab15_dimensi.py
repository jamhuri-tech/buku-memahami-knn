"""Bab 15: banyaknya jarak yang dihitung pohon k-d dan ball tree
(k = 5, daun = 20) terhadap dimensi, pada 10000 titik seragam, lalu pada
angka tulisan tangan (dimensi 64, dimensi intrinsik sekitar 7).
Untuk ball tree dicatat juga jarak ke pusat simpul.
"""
import numpy as np
from sklearn.datasets import load_digits
from bab15_pohon import Pohon

BENIH = 20260929


def rata(T, Q, k=5):
    j, pu = 0, 0
    for q in Q:
        T.query(q, k=k)
        j += T.n_jarak
        pu += T.n_pusat
    return j / len(Q), pu / len(Q)


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    print("data              n     p   k-d: titik   ball: titik + pusat")
    for p in (2, 4, 8, 16, 32):
        X = rng.random((10_000, p))
        Q = rng.random((100, p))
        kd, _ = rata(Pohon(X, 20, False), Q)
        bt, pu = rata(Pohon(X, 20, True), Q)
        print(f"seragam        10000  {p:3d}   {kd:8.0f}      "
              f"{bt:8.0f} + {pu:4.0f}")
    D, _ = load_digits(return_X_y=True)
    X, Q = D[100:], D[:100]
    kd, _ = rata(Pohon(X, 20, False), Q)
    bt, pu = rata(Pohon(X, 20, True), Q)
    print(f"angka           {len(X)}   64   {kd:8.0f}      "
          f"{bt:8.0f} + {pu:4.0f}")
