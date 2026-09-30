"""Bab 10: penaksir dimensi intrinsik Levina-Bickel (2005).

Untuk titik x dengan jarak ke k tetangga T_1 <= ... <= T_k,
  m(x) = (k - 1) / sum_{j<k} log(T_k / T_j),
lalu dirata-ratakan atas titik (rata-rata harmonik kebalikannya, sesuai
MacKay & Ghahramani 2005). Diuji pada data yang dimensi intrinsiknya
diketahui, lalu dipakai pada angka tulisan tangan.
"""
import numpy as np
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.neighbors import NearestNeighbors

BENIH = 20260929


def levina_bickel(X, k=10):
    T, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    T = T[:, 1:]                                   # buang diri sendiri
    Y = np.log(T[:, -1:] / T[:, :-1]).sum(1)       # sum log(T_k / T_j)
    return (k - 1) / np.mean(Y)                    # = 1/rata-rata(1/m)


# Kontras relatif jarak: angka tulisan tangan lawan titik seragam
# berdimensi 64 dengan banyak titik yang sama.
def kontras(D, m=200):
    idx = np.random.default_rng(BENIH).choice(len(D), m, replace=False)
    hasil = []
    for i in idx:
        d = np.sqrt(((D - D[i]) ** 2).sum(1))
        d = np.delete(d, i)
        hasil.append((d.max() - d.min()) / d.min())
    return np.median(hasil)



if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    n = 2000
    Q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    data = {
        "kubus p=2": rng.random((n, 2)),
        "kubus p=5": rng.random((n, 5)),
        "kubus p=10": rng.random((n, 10)),
        "bidang di R^64": np.hstack([rng.random((n, 2)),
                                     np.zeros((n, 62))]) @ Q,
        "gulungan (p=3)": make_swiss_roll(n, random_state=BENIH)[0],
    }
    X, _ = load_digits(return_X_y=True)
    data["angka (p=64)"] = X
    print("data              dimensi taksiran (k=10)  (k=20)")
    for nama, D in data.items():
        print(f"{nama:17s} {levina_bickel(D, 10):8.2f}        "
              f"{levina_bickel(D, 20):8.2f}")

    S = rng.random(X.shape)
    print(f"kontras (max-min)/min: angka {kontras(X):.3f}, "
          f"seragam p=64 {kontras(S):.3f}")
