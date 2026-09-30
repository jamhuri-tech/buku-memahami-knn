"""Bab 11: penaksir kepadatan k-NN satu dimensi,
p(x) = k / (n * 2 r_k(x)), dibandingkan dengan taksiran kepadatan
kernel Gaussian (scipy.stats.gaussian_kde), pada campuran
0,7 N(0, 1) + 0,3 N(3; 0,5^2) dengan n = 500 dan k = 25.
"""
import numpy as np
from scipy.stats import gaussian_kde, norm

BENIH = 20260929
rng = np.random.default_rng(BENIH)
n, k = 500, 25
komp = rng.random(n) < 0.7
x = np.where(komp, rng.normal(0, 1, n), rng.normal(3, 0.5, n))
xs = np.sort(x)


def p_benar(t):
    return 0.7 * norm.pdf(t, 0, 1) + 0.3 * norm.pdf(t, 3, 0.5)


def p_knn(t, k=k, koreksi=False):
    r = np.sort(np.abs(t[:, None] - xs[None, :]), axis=1)[:, k - 1]
    return (k - 1 if koreksi else k) / (n * 2 * r)


if __name__ == "__main__":
    t = np.linspace(-4, 6, 2001)
    kde = gaussian_kde(x)
    for nama, ph in (("k-NN (k)", p_knn(t)),
                     ("k-NN (k - 1)", p_knn(t, koreksi=True)),
                     ("kernel Gaussian", kde(t))):
        ise = np.trapezoid((ph - p_benar(t)) ** 2, t)
        print(f"{nama:16s}: galat kuadrat terintegral {ise:.5f}")
    print("integral taksiran k-NN di [-L, L]:")
    for L in (5, 50, 500, 5000):
        tt = np.linspace(-L, L, 400_001)
        print(f"  L = {L:5d}: {np.trapezoid(p_knn(tt), tt):.3f}")
