"""Pemeriksa Contoh Soal Bab 14."""
import math
from decimal import Context, Decimal

import numpy as np
from bab14_seleksi import knuth

# ---- Contoh Soal 14.1: operasi dan memori ----
m, n, p = 450, 1347, 64
assert 2 * m * n * p == 77_587_200
assert m * n * p == 38_793_600 and 8 * m * n * p == 310_348_800
assert round(8 * m * n * p / 2**20) == 296
assert 8 * m * n == 4_849_200 and round(8 * m * n / 2**20, 1) == 4.6

# ---- Contoh Soal 14.2: pembatalan dengan enam angka bermakna ----
c6 = Context(prec=6)
x, z = Decimal("1000.1"), Decimal("1000")
x2 = c6.multiply(x, x)
assert x2 == Decimal("1.00020E+6")
dua_xz = c6.multiply(Decimal(2), c6.multiply(x, z))
hasil = c6.subtract(c6.add(x2, c6.multiply(z, z)), dua_xz)
assert hasil == 0
assert (x - z) ** 2 == Decimal("0.01")
xp, zp = Decimal("0.05"), Decimal("-0.05")
g = c6.subtract(c6.add(c6.multiply(xp, xp), c6.multiply(zp, zp)),
                c6.multiply(Decimal(2), c6.multiply(xp, zp)))
assert g == Decimal("0.01")
kappa = (1000.1**2 + 1000**2) / 0.01
assert round(kappa / 1e8, 1) == 2.0 and round(5e-6 * kappa) == 1000

# ---- Contoh Soal 14.3: quickselect <= 4n ----
for nn in (2, 3, 10, 11, 100, 101):
    s = sum(max(i - 1, nn - i) for i in range(1, nn + 1))
    assert s <= 3 * nn * nn / 4
    if nn % 2 == 0:
        assert 4 * s == 3 * nn * nn - 2 * nn
C = np.zeros(301)
for nn in range(2, 301):          # batas atas rekursi terburuk
    C[nn] = nn - 1 + sum(C[max(i - 1, nn - i)] for i in range(1, nn + 1)) / nn
    assert C[nn] <= 4 * nn
H = lambda q: sum(1 / j for j in range(1, q + 1))
assert math.isclose(knuth(2000, 1), 2 * (2000 - H(2000)))

# ---- Contoh Soal 14.4: ukuran blok ----
b = 2**30 // (8 * 10**6)
assert b == 134 and math.ceil(10_000 / b) == 75
assert 75 * 134 == 10_050 and 74 * 134 == 9916
b32 = 2**30 // (4 * 10**6)
assert b32 == 268 and math.ceil(10_000 / b32) == 38

print("Contoh Soal Bab 14: semua bilangan cocok")
