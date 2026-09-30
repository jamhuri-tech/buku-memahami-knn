"""Pemeriksa Contoh Soal Bab 19."""
import math

assert round(111.2 * 0.813, 1) == 90.4
assert round(math.cos(math.radians(35.63)), 3) == 0.813
d2 = 10**2 + (20 * 1.5) ** 2
assert d2 == 1000 and round(math.sqrt(d2), 1) == 31.6 and 900 / d2 == 0.9
assert round(0.719**2, 4) == 0.5170 and round(1.154**2, 4) == 1.3317
assert round(1 - 0.719**2 / 1.154**2, 3) == 0.612
print("Contoh Soal Bab 19: semua bilangan cocok")
