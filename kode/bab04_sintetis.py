"""Bab 4: data bangkitan "gelombang" yang posteriornya diketahui.

Titik x seragam di persegi [0, 1]^2. Label y = 1 dengan peluang
eta(x) = 1 / (1 + exp(-A (x2 - g(x1)))), g(x1) = 0,5 + 0,2 sin(2 pi x1).
Batas Bayes adalah kurva x2 = g(x1). Karena eta diketahui, galat Bayes
dan galat 1-NN asimtotik dapat dihitung dengan integral (Bab 8).
Modul ini dipakai ulang oleh Bab 7, 8, 9, dan 13.
"""
import numpy as np

A = 10.0


def g(x1):
    return 0.5 + 0.2 * np.sin(2 * np.pi * x1)


def eta(X):
    X = np.atleast_2d(X)
    return 1.0 / (1.0 + np.exp(-A * (X[:, 1] - g(X[:, 0]))))


def bangkitkan(n, rng):
    X = rng.random((n, 2))
    y = (rng.random(n) < eta(X)).astype(int)
    return X, y


def galat_bayes(m=2001):
    """R* = E[min(eta, 1 - eta)] dengan aturan trapesium."""
    t = np.linspace(0, 1, m)
    X1, X2 = np.meshgrid(t, t)
    e = eta(np.column_stack([X1.ravel(), X2.ravel()])).reshape(X1.shape)
    return np.trapezoid(np.trapezoid(np.minimum(e, 1 - e), t), t)


if __name__ == "__main__":
    print(f"galat Bayes model gelombang: {galat_bayes():.4f}")
