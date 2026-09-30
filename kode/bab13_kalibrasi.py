"""Bab 13: kalibrasi peluang k-NN terhadap posterior yang sebenarnya.
Titik uji dikelompokkan menurut peluang yang dilaporkan (10 selang
sama lebar); di setiap selang dibandingkan rata-rata peluang yang
dilaporkan dengan rata-rata eta. ECE = rata-rata berbobot selisihnya.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from bab04_sintetis import bangkitkan, eta
from bab13_logloss import dihaluskan

BENIH = 20260929


def keandalan(q, e, selang=10):
    b = np.minimum((q * selang).astype(int), selang - 1)
    tengah, benar, bobot = [], [], []
    for j in range(selang):
        m = b == j
        if m.any():
            tengah.append(q[m].mean())
            benar.append(e[m].mean())
            bobot.append(m.mean())
    tengah, benar, bobot = map(np.array, (tengah, benar, bobot))
    return tengah, benar, np.sum(bobot * np.abs(tengah - benar))


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(400, rng)
    U = np.random.default_rng(BENIH + 1).random((50_000, 2))
    e = eta(U)
    print("  k   ECE seragam   ECE 1/d   ECE alpha=0,5")
    for k in (5, 15, 31, 61):
        a = KNeighborsClassifier(k).fit(X, y).predict_proba(U)[:, 1]
        b = KNeighborsClassifier(k, weights="distance").fit(
            X, y).predict_proba(U)[:, 1]
        c = dihaluskan(a, k, 0.5)
        print(f"{k:3d}     {keandalan(a, e)[2]:.4f}      "
              f"{keandalan(b, e)[2]:.4f}     {keandalan(c, e)[2]:.4f}")
        if k == 5:
            ekstrem = (e[a == 0].mean(), e[a == 1].mean())
    print(f"k = 5: rata-rata eta di p = 0: {ekstrem[0]:.4f}, "
          f"di p = 1: {ekstrem[1]:.4f}")
