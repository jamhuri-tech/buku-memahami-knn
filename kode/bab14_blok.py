"""Bab 14: pencarian k tetangga per blok dengan batas memori, dicocokkan
dengan NearestNeighbors(algorithm="brute"). Setiap blok titik kueri
dihitung jarak kuadratnya dengan rumus Gram (data dipusatkan lebih
dahulu), lalu k terkecil dipilih dengan argpartition.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors


def knn_blok(X, Q, k, memori_mb=8):
    """Indeks dan jarak k tetangga; matriks jarak per blok tidak lebih
    dari memori_mb megabita (float64)."""
    m = X.mean(0)
    X = X - m
    Q = Q - m
    nx = (X ** 2).sum(1)
    baris = max(1, int(memori_mb * 2**20 / (8 * len(X))))
    idx = np.empty((len(Q), k), dtype=np.int64)
    jarak = np.empty((len(Q), k))
    for a in range(0, len(Q), baris):
        B = Q[a:a + baris]
        d2 = (B ** 2).sum(1)[:, None] + nx[None, :] - 2 * B @ X.T
        np.maximum(d2, 0, out=d2)
        i = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(d2, i, 1)
        u = np.argsort(dk, axis=1, kind="stable")
        idx[a:a + baris] = np.take_along_axis(i, u, 1)
        jarak[a:a + baris] = np.sqrt(np.take_along_axis(dk, u, 1))
    return jarak, idx, baris


if __name__ == "__main__":
    X, _ = load_digits(return_X_y=True)
    Q, Xl = X[:500], X[500:]
    d, i, baris = knn_blok(Xl, Q, 10, memori_mb=1)
    ds, js = NearestNeighbors(n_neighbors=10, algorithm="brute").fit(
        Xl).kneighbors(Q)
    print(f"{len(Q)} kueri x {len(Xl)} titik latih; blok {baris} baris "
          f"({baris * len(Xl) * 8 / 2**20:.2f} MB)")
    print(f"selisih jarak terbesar dengan sklearn: {np.abs(d - ds).max():.2e}")
    # Piksel bulat membuat jarak sering seri; himpunan tetangga dapat
    # berbeda walaupun daftar jaraknya sama persis.
    beda = np.any(np.sort(i, 1) != np.sort(js, 1), axis=1)
    print(f"kueri dengan himpunan tetangga berbeda: {beda.sum()}")
    print(f"daftar jarak kueri itu sama: {np.allclose(d[beda], ds[beda])}")
    n, m = 10**6, 10**4
    print(f"matriks jarak penuh n = {n:.0e}, m = {m:.0e}: "
          f"{n * m * 8 / 2**30:.1f} GiB")
