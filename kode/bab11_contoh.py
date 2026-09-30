"""Pemeriksa Contoh Soal Bab 11."""
import math
from fractions import Fraction as F

import numpy as np
from scipy.integrate import quad
from scipy.special import beta as B

BENIH = 20260929

# ---- Contoh Soal 11.1: bobot 1/d di (4, 3) ----
w = [1, 1 / math.sqrt(2), 1 / math.sqrt(5), 1 / math.sqrt(8), 1 / 3]
assert [round(v, 4) for v in w] == [1, 0.7071, 0.4472, 0.3536, 0.3333]
A3, B3 = 0.7071 + 0.4472, 1
assert round(A3, 4) == 1.1543 and round(A3 / (A3 + B3), 4) == 0.5358
A5, B5 = A3 + 0.3333, B3 + 0.3536
assert round(A5, 4) == 1.4876 and round(B5, 4) == 1.3536
assert round(A5 / (A5 + B5), 4) == 0.5236
assert round(sum(w[i] for i in (1, 2, 4)), 4) == 1.4877

# ---- Contoh Soal 11.2: E[1/U] = n/(k - 1) untuk Beta(k, n-k+1) ----
for n, k in ((10, 2), (10, 3), (500, 25)):
    v, _ = quad(lambda u: u ** (k - 2) * (1 - u) ** (n - k)
                / B(k, n - k + 1), 0, 1)
    assert math.isclose(v, n / (k - 1), rel_tol=1e-8)
rng = np.random.default_rng(BENIH)
U = rng.beta(25, 476, 1_000_000)
assert abs(np.mean(24 / (500 * U)) - 1) < 0.002
assert F(25, 24) == 1 + F(1, 24) and round(25 / 24, 3) == 1.042

# ---- Contoh Soal 11.3: ekor k/(2 n |x|) ----
assert round(25 / 500 * math.log(10), 3) == 0.115
assert round(math.log(10), 4) == 2.3026

# ---- Contoh Soal 11.4: posterior k_c / k ----
V = F(1)                                    # volume sembarang
fA, fB, f = F(7, 60) / V, F(3, 40) / V, F(10, 100) / V
assert F(6, 10) * fA / f == F(7, 10)
assert F(4, 10) * fB / f == F(3, 10)
pa = F(1, 2) * fA / (F(1, 2) * fA + F(1, 2) * fB)
assert pa == F(14, 23) and round(float(pa), 3) == 0.609
assert F(3, 40) == F(9, 120) and F(7, 60) == F(14, 120)

# ---- Contoh Soal 11.5: Nadaraya-Watson Epanechnikov ----
u = np.array([0.2, 0.5, 0.8])
k = 1 - u**2
assert np.allclose(k, [0.96, 0.75, 0.36]) and math.isclose(k.sum(), 2.07)
yv = np.array([2, 4, 6])
assert math.isclose((k * yv).sum(), 7.08)
assert round((k * yv).sum() / k.sum(), 3) == 3.420
assert np.allclose(np.round(k / k.sum(), 3), [0.464, 0.362, 0.174])

print("Contoh Soal Bab 11: semua bilangan cocok")
