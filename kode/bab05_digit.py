"""Bab 5: akurasi k-NN pada angka tulisan tangan dengan lima metrik."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

BENIH = 20260929
X, y = load_digits(return_X_y=True)
X_lt, X_uji, y_lt, y_uji = train_test_split(
    X, y, test_size=450, stratify=y, random_state=BENIH)
pilihan = [("manhattan (q=1)", dict(metric="minkowski", p=1)),
           ("euclidean (q=2)", dict(metric="minkowski", p=2)),
           ("minkowski q=4", dict(metric="minkowski", p=4)),
           ("chebyshev (q=inf)", dict(metric="chebyshev")),
           ("cosine", dict(metric="cosine"))]
print("metrik              k=1     k=3     k=5")
for nama, arg in pilihan:
    aku = []
    for k in (1, 3, 5):
        m = KNeighborsClassifier(n_neighbors=k, algorithm="brute", **arg)
        aku.append(m.fit(X_lt, y_lt).score(X_uji, y_uji))
    print(f"{nama:18s}  " + "  ".join(f"{a:.4f}" for a in aku))
