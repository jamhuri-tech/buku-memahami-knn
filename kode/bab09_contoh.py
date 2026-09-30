"""Pemeriksa Contoh Soal Bab 9."""
import math
from fractions import Fraction as F
from math import comb

import numpy as np

BENIH = 20260929


# ---- Contoh Soal 9.1: q_{k+2} - q_k ----
def q(k, r):
    return sum(comb(k, j) * r**j * (1 - r) ** (k - j)
               for j in range(k // 2 + 1, k + 1))


r = F(3, 10)
for k in (1, 3, 5, 7, 9, 21):
    m = (k - 1) // 2
    assert q(k + 2, r) - q(k, r) == comb(k, m) * (r * (1 - r)) ** (m + 1) \
        * (2 * r - 1)
assert q(1, r) == F(3, 10) and q(3, r) == F(216, 1000)
assert q(5, r) == F(16308, 100000)
assert comb(1, 0) * F(21, 100) * F(-4, 10) == F(-84, 1000)
assert comb(3, 1) * F(21, 100) ** 2 * F(-4, 10) == F(-5292, 100000)

# ---- Contoh Soal 9.2: maksimum t exp(-k t^2 / 2) ----
for k in (1, 11, 51):
    t = np.linspace(0, 1, 200_001)
    h = t * np.exp(-k * t**2 / 2)
    assert abs(h.max() - 1 / math.sqrt(k * math.e)) < 1e-8
    assert abs(t[h.argmax()] - 1 / math.sqrt(k)) < 1e-4
assert round(11 * math.e, 2) == 29.90
assert round(1 / math.sqrt(11 * math.e), 3) == 0.183
assert round(1 / math.sqrt(11), 2) == 0.30
assert round((1 - 1 / math.sqrt(11)) / 2, 2) == 0.35

# ---- Contoh Soal 9.3: ketaksamaan sisipan ----
eta, et = 0.6, 0.45
assert math.isclose(eta - (1 - eta), 0.2)
assert math.isclose(abs(2 * eta - 1), 0.2)
assert math.isclose(2 * abs(et - eta), 0.3)

# ---- Contoh Soal 9.4: E U_(k) = k / (n + 1) ----
rng = np.random.default_rng(BENIH)
Uu = np.sort(rng.random((200_000, 9)), axis=1)
assert abs(Uu[:, 2].mean() - 3 / 10) < 0.002
D = np.sort(np.abs(rng.random((200_000, 9)) - 0.5), axis=1)
assert abs(D[:, 2].mean() - 0.15) < 0.001
assert F(3, 2 * 10) == F(3, 20)
assert F(1, 2 * 10) == F(1, 20)

# ---- Contoh Soal 9.5: keseimbangan a/k + b (k/n)^(2/p) ----
def optimum(a, b, n, p):
    k = (p * a / (2 * b)) ** (p / (p + 2)) * n ** (2 / (p + 2))
    return k, a / k + b * (k / n) ** (2 / p)
k2, g2 = optimum(1, 1, 10_000, 2)
assert math.isclose(k2, 100) and math.isclose(g2, 0.02)
k4, g4 = optimum(1, 1, 10_000, 4)
assert round(k4, 1) == 34.2 and round(g4, 4) == 0.0877
assert round(2 ** (2 / 3), 3) == 1.587 and round(10_000 ** (1 / 3), 2) == 21.54
assert round(1 / 34.2, 4) == 0.0292 and round(math.sqrt(34.2 / 10_000), 4) \
    == 0.0585
ks = np.linspace(1, 500, 499_001)
g = 1 / ks + (ks / 10_000) ** 0.5
assert abs(ks[g.argmin()] - k4) < 0.01
assert round((0.0877 / 0.02) ** 3) == 84

print("Contoh Soal Bab 9: semua bilangan cocok")
