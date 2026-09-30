"""Bab 7: dekomposisi bias-varians regresi k-NN pada rancangan tetap.

x_i = (i - 0,5)/n, i = 1..n, n = 100; f(x) = sin(2 pi x); noise
N(0, sigma^2), sigma = 0,3. Galat harapan di titik uji x0 (500 titik
seragam di [0, 1]) dihitung dengan rumus eksak, lalu diperiksa dengan
simulasi 2000 ulangan noise untuk k = 10.
"""
import numpy as np

BENIH = 20260929
n, sigma = 100, 0.3
x = (np.arange(1, n + 1) - 0.5) / n
f = lambda t: np.sin(2 * np.pi * t)
x0 = (np.arange(500) + 0.5) / 500
urut = np.argsort(np.abs(x0[:, None] - x[None, :]), axis=1,
                  kind="stable")


def bias2_var(k):
    rata_f = f(x[urut[:, :k]]).mean(axis=1)     # (1/k) sum f(x_(l))
    return np.mean((f(x0) - rata_f) ** 2), sigma**2 / k


print("  k    bias^2    varians   galat harapan (+ sigma^2)")
for k in (1, 3, 5, 10, 20, 40, 80):
    b2, v = bias2_var(k)
    print(f"{k:3d}   {b2:.5f}   {v:.5f}   {sigma**2 + b2 + v:.5f}")

rng = np.random.default_rng(BENIH)
k, ulang = 10, 2000
galat = []
for _ in range(ulang):
    y = f(x) + sigma * rng.standard_normal(n)
    fhat = y[urut[:, :k]].mean(axis=1)
    y0 = f(x0) + sigma * rng.standard_normal(len(x0))
    galat.append(np.mean((y0 - fhat) ** 2))
b2, v = bias2_var(k)
print(f"k = 10: rumus {sigma**2 + b2 + v:.5f}, "
      f"simulasi {np.mean(galat):.5f}")
