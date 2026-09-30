"""Pemeriksa Contoh Soal Bab 13."""
import math
from fractions import Fraction as F

import numpy as np

BENIH = 20260929

# ---- Contoh Soal 13.1: bias dan varians taksiran peluang ----
e = [F(6, 10), F(7, 10), F(8, 10)]
assert sum(e) / 3 == F(7, 10)
v = sum(t * (1 - t) for t in e) / 9
assert v == F(61, 900) and round(float(v), 4) == 0.0678
assert round(math.sqrt(0.0678), 2) == 0.26
assert 3 * F(21, 100) / 9 == F(7, 100)
rng = np.random.default_rng(BENIH)
lab = rng.random((1_000_000, 3)) < np.array([0.6, 0.7, 0.8])
assert abs(np.mean((lab.mean(1) - 0.7) ** 2) - 61 / 900) < 5e-4

# ---- Contoh Soal 13.2: prior beta ----
def rata_posterior(k1, k, a):
    return F(k1 + a, k + 2 * a)
assert rata_posterior(5, 5, 0) == 1 and rata_posterior(5, 5, 1) == F(6, 7)
assert round(6 / 7, 3) == 0.857 and round(math.log(7), 3) == 1.946
from scipy.stats import beta
assert math.isclose(beta(5 + 1, 0 + 1).mean(), 6 / 7)

# ---- Contoh Soal 13.3: aturan skor yang tepat ----
eta = 0.7
bs = lambda q: (q - eta) ** 2 + eta * (1 - eta)
assert math.isclose(bs(0.7), 0.21) and math.isclose(bs(1), 0.30)
assert math.isclose(bs(0.5), 0.25)
assert math.isclose(eta * (1 - 1) ** 2 + (1 - eta) * 1, 0.3)
q = np.linspace(0.001, 0.999, 9999)
ll = -eta * np.log(q) - (1 - eta) * np.log(1 - q)
assert abs(q[ll.argmin()] - 0.7) < 1e-3

# ---- Contoh Soal 13.4: selang Wald dan Wilson ----
z2, k, p = 3.8416, 10, 0.7
assert round(1.96 * math.sqrt(0.21 / 10), 3) == 0.284
assert round(math.sqrt(0.021), 4) == 0.1449
pusat = (p + z2 / (2 * k)) / (1 + z2 / k)
h = math.sqrt(z2) / (1 + z2 / k) * math.sqrt(p * (1 - p) / k
                                             + z2 / (4 * k * k))
assert round(pusat, 4) == 0.6445 and round(h, 3) == 0.248
assert (round(pusat - h, 3), round(pusat + h, 3)) == (0.397, 0.892)
assert round(1.96 / 1.38416, 3) == 1.416 and round(math.sqrt(0.0306), 4) \
    == 0.1749
assert math.isclose(pusat, (7 + z2 / 2) / (10 + z2))

print("Contoh Soal Bab 13: semua bilangan cocok")
