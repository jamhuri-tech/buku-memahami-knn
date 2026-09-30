"""Pemeriksa Contoh Soal Bab 3."""
import math
from fractions import Fraction as F

import numpy as np
from scipy.integrate import quad
from scipy.special import beta
from scipy.stats import binom

# ---- Contoh Soal 3.1: E D = int P(D > r) dr ----
a = 3.0
v, _ = quad(lambda r: 1 - r / a, 0, a)
assert math.isclose(v, a / 2)

# ---- Contoh Soal 3.2: statistik terurut ----
n, k = 7, 3
c = math.factorial(n) / (math.factorial(k - 1) * math.factorial(n - k))
assert math.isclose(c, 1 / beta(k, n - k + 1))
m, _ = quad(lambda u: u * c * u ** (k - 1) * (1 - u) ** (n - k), 0, 1)
assert math.isclose(m, k / (n + 1))
rng = np.random.default_rng(20260929)
assert abs(np.sort(rng.random((200_000, n)), 1)[:, k - 1].mean() - 3 / 8) \
    < 0.002

# ---- Contoh Soal 3.3: Chebyshev, Hoeffding, Jensen ----
assert 100 * 0.25 == 25 and 25 / 100 == 0.25
assert round(math.exp(-2), 4) == 0.1353 and round(2 * math.exp(-2), 4) \
    == 0.2707
eksak = binom.cdf(40, 100, 0.5) + binom.sf(59, 100, 0.5)
assert round(eksak, 3) == 0.057
assert round(2 / 3, 3) == 0.667 and round(math.sqrt(0.5), 3) == 0.707

# ---- Contoh Soal 3.4: lipatan ----
assert 178 == 5 * 35 + 3 and 3 * 36 + 2 * 35 == 178
from sklearn.model_selection import KFold
uk = sorted(len(u) for _, u in KFold(5).split(np.zeros(178)))
assert uk == [35, 35, 36, 36, 36]
print("Contoh Soal Bab 3: semua bilangan cocok")
