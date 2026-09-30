"""Bab 16: proyeksi acak Johnson-Lindenstrauss.

(1) 300 titik normal baku di R^5000 diproyeksikan dengan matriks R
    berukuran k x p, unsur N(0, 1/k). Dicatat rasio jarak sesudah dan
    sebelum proyeksi untuk semua 44850 pasangan.
(2) KNN pada data berdimensi tinggi (10 kelas, p = 5000) sesudah
    proyeksi ke k dimensi.
"""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier

BENIH = 20260929


def proyeksi(X, k, rng):
    R = rng.standard_normal((X.shape[1], k)) / np.sqrt(k)
    return X @ R


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    X = rng.standard_normal((300, 5000))
    D = pairwise_distances(X)
    iu = np.triu_indices(300, 1)
    print("   k    rasio min   rasio maks   bagian |rasio-1| > 0,1")
    for k in (50, 200, 1000, 4000):
        Dk = pairwise_distances(proyeksi(X, k, rng))
        r = Dk[iu] / D[iu]
        print(f"{k:5d}     {r.min():.3f}       {r.max():.3f}         "
              f"{np.mean(np.abs(r - 1) > 0.1):.4f}")

    # Angka tulisan tangan disisipkan ke R^5000 dengan matriks
    # berkolom ortonormal: semua jarak tetap, tetapi p = 5000.
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    Xd, yd = load_digits(return_X_y=True)
    Q, _ = np.linalg.qr(rng.standard_normal((5000, 64)))
    Z = Xd @ Q.T
    Zl, Zu, yl, yu = train_test_split(Z, yd, test_size=450,
                                      stratify=yd, random_state=BENIH)
    a = KNeighborsClassifier(5).fit(Zl, yl).score(Zu, yu)
    print(f"angka di R^5000, 5-NN tanpa proyeksi: akurasi {a:.4f}")
    for k in (5, 10, 20, 50, 200):
        R = rng.standard_normal((5000, k)) / np.sqrt(k)
        a = KNeighborsClassifier(5).fit(Zl @ R, yl).score(Zu @ R, yu)
        print(f"  sesudah proyeksi ke k = {k:3d}: akurasi {a:.4f}")

    # Sepuluh kelas yang pusatnya hanya sedikit berbeda di R^5000.
    p, n = 5000, 2000
    pusat = rng.standard_normal((10, p)) * 0.08
    y = rng.integers(0, 10, n)
    Z = pusat[y] + rng.standard_normal((n, p))
    lt, uji = slice(0, 1500), slice(1500, None)
    a = KNeighborsClassifier(5).fit(Z[lt], y[lt]).score(Z[uji], y[uji])
    print(f"kelas berdekatan di R^5000, 5-NN: akurasi {a:.3f}")
    for k in (100, 1000):
        Zk = proyeksi(Z, k, rng)
        a = KNeighborsClassifier(5).fit(Zk[lt], y[lt]).score(Zk[uji],
                                                               y[uji])
        print(f"  sesudah proyeksi ke k = {k:4d}: akurasi {a:.3f}")
