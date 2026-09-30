"""Bab 8: tiga kelas. x seragam di [0, 1] dan posterior kelas
(eta_1, eta_2, eta_3) = ((1 - x)^2, 2x(1 - x), x^2). Galat Bayes,
galat 1-NN asimtotik, batas Cover-Hart K kelas, dan simulasi 1-NN.
"""
import numpy as np

BENIH = 20260929
K = 3


def posterior(x):
    return np.column_stack([(1 - x) ** 2, 2 * x * (1 - x), x ** 2])


m = 200_001
x = (np.arange(m) + 0.5) / m
P = posterior(x)
rb = np.mean(1 - P.max(1))
rnn = np.mean(1 - (P ** 2).sum(1))
print(f"R* = {rb:.4f}, R_NN = {rnn:.4f}")
print(f"batas R*(2 - K R*/(K - 1)) = {rb * (2 - K * rb / (K - 1)):.4f},"
      f" 2R* = {2 * rb:.4f}")
rng = np.random.default_rng(BENIH)
g = []
for _ in range(50):
    xl = rng.random(10_000)
    yl = (rng.random(len(xl))[:, None]
          > np.cumsum(posterior(xl), 1)).sum(1)
    urut = np.argsort(xl)
    xs, ys = xl[urut], yl[urut]
    u = rng.random(20_000)
    j = np.clip(np.searchsorted(xs, u), 1, len(xs) - 1)
    yh = np.where(np.abs(u - xs[j - 1]) <= np.abs(xs[j] - u),
                  ys[j - 1], ys[j])
    g.append(np.mean(1 - posterior(u)[np.arange(len(u)), yh]))
print(f"simulasi 1-NN, n = 10000 (50 data latih): {np.mean(g):.4f}")
