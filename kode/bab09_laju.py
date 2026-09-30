"""Bab 9: laju galat regresi k-NN terhadap n untuk p = 1, 2, 4.

x seragam di [0, 1]^p, f(x) = p^(-1/2) sum_j sin(2 pi x_j), sigma = 0,5.
Untuk setiap n, k dipilih yang memberi galat kuadrat rata-rata
E(fhat - f)^2 terkecil (rata-rata 5 data latih), lalu kemiringan
log-galat terhadap log-n dibandingkan dengan -2/(2+p) (batas Lipschitz)
dan -4/(4+p) (f mulus, titik di dalam domain).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

BENIH = 20260929
NS = (250, 1000, 4000, 16000)


def f(X):
    return np.sin(2 * np.pi * X).sum(1) / np.sqrt(X.shape[1])


def kurva(n, p, rng, U, kmaks=400, ulang=5):
    kmaks = min(kmaks, n // 2)
    fu = f(U)
    tot = np.zeros(kmaks)
    for _ in range(ulang):
        X = rng.random((n, p))
        y = f(X) + 0.5 * rng.standard_normal(n)
        _, idx = NearestNeighbors(n_neighbors=kmaks).fit(X).kneighbors(U)
        rata = np.cumsum(y[idx], 1) / np.arange(1, kmaks + 1)
        tot += np.mean((rata - fu[:, None]) ** 2, axis=0) / ulang
    return tot


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    print("p      n    k*    galat     | kemiringan  -2/(2+p)  -4/(4+p)")
    for p in (1, 2, 4):
        U = rng.random((4000, p))
        g = []
        for n in NS:
            c = kurva(n, p, rng, U)
            g.append(c.min())
            print(f"{p}  {n:6d}  {c.argmin() + 1:4d}   {c.min():.5f}   |")
        miring = np.polyfit(np.log(NS), np.log(g), 1)[0]
        print(f"{'':27s}|  {miring:7.3f}    {-2 / (2 + p):6.3f}    "
              f"{-4 / (4 + p):6.3f}")
