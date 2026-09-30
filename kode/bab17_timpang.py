"""Bab 17: kelas tak seimbang. Data gelombang dengan titik berlabel 1
hanya disimpan dengan peluang 0,1 (data latih n kira-kira 1150, 10%
kelas 1). Dibandingkan: k-NN biasa, dan k-NN yang membagi proporsi
tetangga dengan prior kelas data latih (aturan untuk akurasi
seimbang). Label uji juga diambil dari sebaran yang tak seimbang.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from bab04_sintetis import bangkitkan

BENIH = 20260929


def timpang(n, rng, simpan=0.1):
    X, y = bangkitkan(n, rng)
    jaga = (y == 0) | (rng.random(n) < simpan)
    return X[jaga], y[jaga]


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    X, y = timpang(2000, rng)
    U, yu = timpang(40_000, rng)
    pi = np.bincount(y) / len(y)
    print(f"data latih {len(y)} titik, bagian kelas 1: {pi[1]:.3f}")
    print("        ------ biasa ------   ---- dikoreksi ----")
    print("  k    recall 1   seimbang    recall 1   seimbang")
    for k in (1, 5, 15, 45):
        m = KNeighborsClassifier(k).fit(X, y)
        p = m.predict_proba(U)
        a = p.argmax(1)
        b = (p / pi).argmax(1)
        print(f"{k:3d}     {recall_score(yu, a):.3f}      "
              f"{balanced_accuracy_score(yu, a):.3f}       "
              f"{recall_score(yu, b):.3f}      "
              f"{balanced_accuracy_score(yu, b):.3f}")
