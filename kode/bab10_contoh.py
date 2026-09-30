"""Pemeriksa Contoh Soal Bab 10."""
import math
from fractions import Fraction as F

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

BENIH = 20260929
V = lambda p: math.pi ** (p / 2) / gamma(p / 2 + 1)

# ---- Contoh Soal 10.1: rekursi V_p = (2 pi / p) V_{p-2} ----
for p in range(3, 30):
    assert math.isclose(V(p), 2 * math.pi / p * V(p - 2))
assert math.isclose(V(4), math.pi**2 / 2) and round(V(4), 3) == 4.935
assert math.isclose(V(5), 8 * math.pi**2 / 15) and round(V(5), 3) == 5.264
assert math.isclose(V(6), math.pi**3 / 6) and round(V(6), 3) == 5.168
assert max(range(1, 40), key=V) == 5
assert 6 < 2 * math.pi < 7

# ---- Contoh Soal 10.2: jari-jari bola sepuluh tetangga ----
r10 = (10 / (10_000 * 2.55)) ** 0.1
assert round(10 / (10_000 * 2.55), 6) == 0.000392
assert round(math.log(3.92e-4) / 10, 3) == -0.784
assert round(r10, 3) == 0.456
r2 = math.sqrt(10 / (10_000 * math.pi))
assert round(10 / (10_000 * math.pi) * 1e4, 2) == 3.18 and round(r2, 4) \
    == 0.0178
assert round(r10 / r2) == 26 or round(r10 / r2, 1) == 25.6
assert round(V(10), 2) == 2.55

# ---- Contoh Soal 10.3: momen T = X - Z ----
ET2, _ = quad(lambda t: t**2 * (1 - abs(t)), -1, 1)
ET4, _ = quad(lambda t: t**4 * (1 - abs(t)), -1, 1)
assert math.isclose(ET2, 1 / 6) and math.isclose(ET4, 1 / 15)
assert 2 * (F(1, 3) - F(1, 4)) == F(1, 6)
assert 2 * (F(1, 5) - F(1, 6)) == F(1, 15)
assert F(1, 15) - F(1, 36) == F(7, 180)
assert F(1, 12) + F(1, 12) == F(1, 6)
rng = np.random.default_rng(BENIH)
T = rng.random(2_000_000) - rng.random(2_000_000)
assert abs((T**2).var() - 7 / 180) < 5e-4
assert math.isclose(math.sqrt(7 * 36 / 180), math.sqrt(1.4))

# ---- Contoh Soal 10.4: Levina-Bickel ----
Tj = [0.5, 0.7, 0.8, 0.9]
logs = [round(math.log(1 / t), 4) for t in Tj]
assert logs == [0.6931, 0.3567, 0.2231, 0.1054]
assert round(sum(logs), 4) == 1.3783
assert round(4 / 1.3783, 2) == 2.90
assert round(4 / 3, 2) == 1.33

print("Contoh Soal Bab 10: semua bilangan cocok")
