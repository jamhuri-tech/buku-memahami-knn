"""Bab 17: prototipe K-Means per kelas pada angka tulisan tangan.
Untuk m prototipe per kelas, K-Means dijalankan pada titik latih setiap
kelas, lalu 1-NN memakai 10 m pusat itu saja. Dibandingkan dengan
kondensasi Hart dan 1-NN pada semua titik latih.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from bab17_kondensasi import kondensasi, suntingan

BENIH = 20260929


def prototipe(X, y, m):
    P, L = [], []
    for c in np.unique(y):
        km = KMeans(m, n_init=4, random_state=BENIH).fit(X[y == c])
        P.append(km.cluster_centers_)
        L.append(np.full(m, c))
    return np.vstack(P), np.concatenate(L)


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    Xl, Xu, yl, yu = train_test_split(X, y, test_size=450, stratify=y,
                                      random_state=BENIH)
    nilai = lambda A, b: KNeighborsClassifier(1).fit(A, b).score(Xu, yu)
    print("himpunan                     titik   akurasi 1-NN")
    print(f"semua titik latih             {len(Xl):4d}    {nilai(Xl, yl):.4f}")
    for m in (1, 2, 5, 10, 20, 40):
        P, L = prototipe(Xl, yl, m)
        print(f"K-Means, {m:2d} per kelas          {len(P):4d}    "
              f"{nilai(P, L):.4f}")
    c, _ = kondensasi(Xl, yl)
    print(f"kondensasi Hart               {len(c):4d}    "
          f"{nilai(Xl[c], yl[c]):.4f}")
    s = suntingan(Xl, yl, 3)
    print(f"penyuntingan Wilson (k=3)     {len(s):4d}    "
          f"{nilai(Xl[s], yl[s]):.4f}")
