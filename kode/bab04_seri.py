"""Bab 4: peluang suara seri pada dua kelas bila setiap tetangga
berlabel 1 dengan peluang 1/2, dibandingkan hampiran Stirling
sqrt(2 / (pi k)).
"""
from math import comb, pi, sqrt

print(" k   P(seri) eksak   sqrt(2/(pi k))")
for k in (2, 4, 10, 50, 100):
    eksak = comb(k, k // 2) / 2**k
    print(f"{k:3d}   {eksak:.4f}          {sqrt(2 / (pi * k)):.4f}")
