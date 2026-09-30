"""Pemeriksa Contoh Soal Bab 8 (pecahan eksak dan integral)."""
from fractions import Fraction as F

import numpy as np
from scipy.integrate import quad

BENIH = 20260929

# ---- Contoh Soal 8.1: jarak tetangga terdekat di [0, 1] ----
for n in (1, 9, 99):
    v, _ = quad(lambda r: (1 - 2 * r) ** n, 0, 0.5)
    assert abs(v - 1 / (2 * (n + 1))) < 1e-12
assert F(1, 2 * 10) == F(1, 20)
rng = np.random.default_rng(BENIH)
d = np.abs(rng.random((200_000, 9)) - 0.5).min(1)
assert abs(d.mean() - 0.05) < 0.001

# ---- Contoh Soal 8.2: eta(x) = x ----
rb, _ = quad(lambda x: min(x, 1 - x), 0, 1, points=[0.5])
rnn, _ = quad(lambda x: 2 * x * (1 - x), 0, 1)
assert abs(rb - 0.25) < 1e-12 and abs(rnn - 1 / 3) < 1e-12
assert 2 * (F(1, 2) - F(1, 3)) == F(1, 3)
assert 2 * F(1, 4) * F(3, 4) == F(3, 8)

# ---- Contoh Soal 8.3: model tetap dan separuh ----
assert abs(2 * 0.1 * 0.9 - 0.18) < 1e-15
assert F(1, 2) * 0 + F(1, 2) * F(1, 2) == F(1, 4)
assert F(1, 2) * 0 + F(1, 2) * (2 * F(1, 2) * F(1, 2)) == F(1, 4)
assert 2 * F(1, 4) * F(3, 4) == F(3, 8)

# ---- Contoh Soal 8.4: posterior tiga kelas di satu titik ----
def titik(eta):
    eta = [F(e) for e in eta]
    r = 1 - max(eta)
    return r, 1 - sum(e * e for e in eta), 2 * r - F(3, 2) * r * r
r, nn, b = titik(["0.5", "0.3", "0.2"])
assert (r, nn, b) == (F(1, 2), F(62, 100), F(625, 1000))
r, nn, b = titik(["0.5", "0.25", "0.25"])
assert nn == b == F(625, 1000)
r, nn, b = titik(["0.5", "0.5", "0"])
assert nn == r == F(1, 2)

# ---- Contoh Soal 8.5: model binomial tiga kelas ----
x13, x23 = F(1, 3), F(2, 3)
assert (1 - x13) ** 2 == 2 * x13 * (1 - x13)
a = (1 - (1 - x13) ** 3) / 3
assert a == F(19, 81)
G = lambda t: t * t - F(2, 3) * t ** 3
assert G(x23) == F(20, 81) and G(x13) == F(7, 81)
assert G(x23) - G(x13) == F(13, 81)
tot = 2 * a + F(13, 81)
assert tot == F(51, 81) == F(17, 27)
Rb = 1 - tot
assert Rb == F(10, 27)
assert F(1, 3) - F(2, 4) + F(1, 5) == F(1, 30)
s = F(1, 5) + 4 * F(1, 30) + F(1, 5)
assert s == F(8, 15) and 1 - s == F(7, 15)
bat = Rb * (2 - F(3, 2) * Rb)
assert bat == F(390, 729) and 2 - F(3, 2) * Rb == F(39, 27)
assert (round(float(Rb), 4), round(7 / 15, 4), round(float(bat), 4)) == \
    (0.3704, 0.4667, 0.535)

print("Contoh Soal Bab 8: semua bilangan cocok")
