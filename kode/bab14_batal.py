"""Bab 14: galat pembatalan pada rumus ||x||^2 + ||z||^2 - 2 x.z.

1000 titik di R^3 berpusat di (10^4, 10^4, 10^4) dengan sebaran
simpangan baku 1. Jarak kuadrat dihitung dengan: selisih langsung
(float64, acuan), rumus Gram float64, rumus Gram float32, selisih
langsung float32, dan rumus Gram float32 sesudah data dipusatkan.
"""
import numpy as np

BENIH = 20260929


def gram(A, B):
    return ((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :]
            - 2 * A @ B.T)


def langsung(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


rng = np.random.default_rng(BENIH)
X = 1e4 + rng.standard_normal((1000, 3))
Q = 1e4 + rng.standard_normal((200, 3))
acuan = langsung(Q, X)
nn_acuan = acuan.argmin(1)
cara = {
    "Gram float64": gram(Q, X),
    "Gram float32": gram(Q.astype(np.float32), X.astype(np.float32)),
    "langsung float32": langsung(Q.astype(np.float32),
                                 X.astype(np.float32)),
}
from sklearn.metrics.pairwise import euclidean_distances
cara["sklearn float32"] = euclidean_distances(
    Q.astype(np.float32), X.astype(np.float32), squared=True)
m = X.mean(0)
cara["Gram float32 pusat"] = gram((Q - m).astype(np.float32),
                                  (X - m).astype(np.float32))
print(f"||x||^2 sekitar {np.mean((X**2).sum(1)):.3e}, "
      f"jarak^2 rata-rata {acuan.mean():.2f}")
print("cara                 galat rel. median   D^2 < 0   1-NN beda")
for nama, D in cara.items():
    D = D.astype(np.float64)
    rel = np.median(np.abs(D - acuan) / acuan)
    print(f"{nama:20s}   {rel:10.2e}     {np.sum(D < 0):5d}"
          f"     {np.sum(D.argmin(1) != nn_acuan):3d}")
