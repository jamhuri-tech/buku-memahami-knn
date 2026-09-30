"""Bab 3: peluang Bin(k, 0,7) < k/2 (suara terbanyak memilih kelas yang
kurang mungkin) dibandingkan batas Chebyshev dan batas Hoeffding.
"""
import numpy as np
from scipy.stats import binom

eta = 0.7
print("  k    eksak       Chebyshev   Hoeffding")
for k in (5, 11, 21, 51, 101):
    eksak = binom.cdf((k - 1) // 2, k, eta)
    t = k * (eta - 0.5)                       # simpangan dari harapan
    cheb = min(1.0, k * eta * (1 - eta) / t**2)
    hoef = np.exp(-2 * t**2 / k)
    print(f"{k:3d}   {eksak:.2e}    {cheb:.2e}    {hoef:.2e}")
