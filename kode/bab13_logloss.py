"""Bab 13: peluang 0 dan 1, log-loss, Brier score, dan penghalusan
(k_1 + alpha) / (k + 2 alpha) pada data gelombang (n = 400, k = 15).
Label uji dibangkitkan dari eta; galat kalibrasi dihitung terhadap eta
yang sebenarnya.
"""
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neighbors import KNeighborsClassifier
from bab04_sintetis import bangkitkan, eta

BENIH = 20260929


def dihaluskan(p1, k, alpha):
    """p1 = k_1/k -> (k_1 + alpha) / (k + 2 alpha)."""
    return (p1 * k + alpha) / (k + 2 * alpha)


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(400, rng)
    rng2 = np.random.default_rng(BENIH + 1)
    U = rng2.random((50_000, 2))
    e = eta(U)
    yu = (rng2.random(len(U)) < e).astype(int)
    k = 15
    p1 = KNeighborsClassifier(k).fit(X, y).predict_proba(U)[:, 1]
    ekstrem = (p1 == 0) | (p1 == 1)
    salah = ekstrem & (np.round(p1) != yu)
    print(f"k = {k}: peluang 0 atau 1 pada {ekstrem.mean():.4f} titik uji,"
          f" salah {salah.mean():.4f}")
    print("penghalusan   log-loss   Brier    rata |p - eta|")
    for alpha in (0, 0.5, 1, 2, 5):
        q = dihaluskan(p1, k, alpha)
        ll = log_loss(yu, np.clip(q, 1e-15, 1 - 1e-15))
        print(f"alpha = {alpha:3}   {ll:8.4f}   {brier_score_loss(yu, q):.4f}"
              f"   {np.mean(np.abs(q - e)):.4f}")
    print(f"posterior sebenarnya: log-loss "
          f"{log_loss(yu, e):.4f}, Brier {brier_score_loss(yu, e):.4f}")
