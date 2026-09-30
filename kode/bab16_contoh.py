"""Pemeriksa Contoh Soal Bab 16."""
import math

import numpy as np
from scipy.stats import chi2

BENIH = 20260929

# ---- Contoh Soal 16.1: E dan Var ||Rv||^2 ----
rng = np.random.default_rng(BENIH)
v = rng.standard_normal(40)
k = 50
R = rng.standard_normal((20_000, k, 40)) / math.sqrt(k)
q = ((R @ v) ** 2).sum(1)
assert abs(q.mean() / (v @ v) - 1) < 0.01
assert abs(q.var() / (2 * (v @ v) ** 2 / k) - 1) < 0.05
assert math.isclose(math.sqrt(2 / 50), 0.2)
assert round(math.sqrt(2 / 1000), 3) == 0.045
assert round(math.sqrt(1 / 100), 2) == 0.1 and round(math.sqrt(1 / 2000),
                                                        3) == 0.022

# ---- Lema 16.1 (diperiksa numerik) ----
for kk, e in ((100, 0.2), (400, 0.1)):
    atas = chi2.sf((1 + e) * kk, kk)
    bawah = chi2.cdf((1 - e) * kk, kk)
    assert atas <= math.exp(-kk * (e**2 / 2 - e**3 / 3) / 2)
    assert bawah <= math.exp(-kk * e**2 / 4)

# ---- Contoh Soal 16.2: k Johnson-Lindenstrauss ----
L = math.log(10_000)
assert round(L, 3) == 9.210
k1 = 4 * L / (0.1**2 / 2 - 0.1**3 / 3)
k2 = 4 * L / (0.2**2 / 2 - 0.2**3 / 3)
assert abs(k1 - 7894) < 1 and abs(k2 - 2125) < 1
assert round(36.84 / 0.004667) == 7894 and round(36.84 / 0.017333) == 2125
assert round(k1 / k2, 1) == 3.7
k300 = 4 * math.log(300) / (0.21**2 / 2 - 0.21**3 / 3)
assert 1150 < k300 < 1250
assert math.isclose(1.1**2 - 1, 0.21)

# ---- Contoh Soal 16.3: peluang 1 - theta/pi ----
assert math.isclose(1 - (math.pi / 3) / math.pi, 2 / 3)
assert math.isclose(1 - (math.pi / 2) / math.pi, 0.5)

# ---- Contoh Soal 16.4: penguatan AND-OR ----
assert round(0.9**8, 4) == 0.4305 and round(0.6**8, 4) == 0.0168
p9 = 1 - (1 - 0.9**8) ** 10
p6 = 1 - (1 - 0.6**8) ** 10
assert round(p9, 4) == 0.9964 and round(p6, 3) == 0.156
assert round(math.log(0.5695), 4) == -0.5630
assert round(math.exp(-5.630), 4) == 0.0036
assert round(math.log(0.9832), 5) == -0.01694
assert round(math.exp(-0.1694), 4) == 0.8442

print("Contoh Soal Bab 16: semua bilangan cocok")
