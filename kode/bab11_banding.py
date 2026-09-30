"""Bab 11: bobot seragam, bobot 1/d, dan bobot kernel Epanechnikov
(K(d / d_{k+1})) pada data gelombang (galat sebenarnya) dan pada angka
tulisan tangan (akurasi data uji), serta tetangga dalam radius tetap.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import (KNeighborsClassifier,
                               RadiusNeighborsClassifier)
from bab04_sintetis import bangkitkan, eta

BENIH = 20260929


def epanechnikov(d):
    """Bobot kernel dengan bandwidth adaptif = jarak tetangga ke-(k+1).
    scikit-learn memberi d untuk k tetangga; kita pakai d terbesar
    dikali 1,0001 supaya tetangga ke-k tidak berbobot nol."""
    h = d.max(axis=1, keepdims=True) * 1.0001
    return 0.75 * (1 - (d / h) ** 2)


BOBOT = {"seragam": "uniform", "1/d": "distance",
         "Epanechnikov": epanechnikov}

if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(400, rng)
    U = np.random.default_rng(BENIH + 1).random((50_000, 2))
    e = eta(U)
    print("gelombang n=400, galat sebenarnya")
    print("  k    " + "  ".join(f"{b:>12s}" for b in BOBOT))
    for k in (1, 5, 15, 31, 61, 121):
        g = []
        for w in BOBOT.values():
            yh = KNeighborsClassifier(k, weights=w).fit(X, y).predict(U)
            g.append(np.mean(np.where(yh == 1, 1 - e, e)))
        print(f"{k:4d}  " + "  ".join(f"{v:12.4f}" for v in g))
    print("radius tetap:")
    for r in (0.05, 0.1, 0.2):
        m = RadiusNeighborsClassifier(radius=r, outlier_label=0)
        yh = m.fit(X, y).predict(U)
        _, ind = m.radius_neighbors(U)
        kosong = np.mean([len(i) == 0 for i in ind])
        print(f"  r = {r:.2f}: galat {np.mean(np.where(yh == 1, 1 - e, e)):.4f}"
              f", bola kosong {kosong:.4f}")

    Xd, yd = load_digits(return_X_y=True)
    Xl, Xu, yl, yu = train_test_split(Xd, yd, test_size=450, stratify=yd,
                                      random_state=BENIH)
    print("angka tulisan tangan, akurasi data uji")
    print("  k    " + "  ".join(f"{b:>12s}" for b in BOBOT))
    for k in (1, 5, 15):
        a = [KNeighborsClassifier(k, weights=w).fit(Xl, yl).score(Xu, yu)
             for w in BOBOT.values()]
        print(f"{k:4d}  " + "  ".join(f"{v:12.4f}" for v in a))
