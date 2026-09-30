"""Bab 8: model satu dimensi yang galat Bayes dan galat 1-NN
asimtotiknya dapat dihitung eksak. x seragam di [0, 1].

  linear   : eta(x) = x
  tetap    : eta(x) = 0,1 di mana-mana
  separuh  : eta(x) = 0 untuk x < 1/2, eta(x) = 1/2 untuk x >= 1/2
Dipakai ulang oleh Bab 9.
"""
import numpy as np

MODEL = {
    "linear": lambda x: x,
    "tetap": lambda x: np.full_like(x, 0.1),
    "separuh": lambda x: np.where(x < 0.5, 0.0, 0.5),
}


def galat_asimtotik(eta, m=200_001):
    """(R*, R_NN) dengan aturan titik tengah pada m titik."""
    x = (np.arange(m) + 0.5) / m
    e = eta(x)
    return np.mean(np.minimum(e, 1 - e)), np.mean(2 * e * (1 - e))


def bangkitkan(eta, n, rng):
    x = rng.random(n)
    return x, (rng.random(n) < eta(x)).astype(int)


def galat_1nn(eta, n, rng, nuji=20_000):
    """Galat harapan 1-NN (terhadap eta) untuk satu data latih."""
    x, y = bangkitkan(eta, n, rng)
    urut = np.argsort(x)
    xs, ys = x[urut], y[urut]
    u = rng.random(nuji)
    j = np.clip(np.searchsorted(xs, u), 1, n - 1)
    kiri = np.abs(u - xs[j - 1]) <= np.abs(xs[j] - u)
    yh = np.where(kiri, ys[j - 1], ys[j])
    e = eta(u)
    return np.mean(np.where(yh == 1, 1 - e, e))
