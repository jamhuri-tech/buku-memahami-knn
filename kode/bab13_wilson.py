"""Bab 13: selang kepercayaan 95% untuk eta dari k_1 tetangga berkelas
1 di antara k: selang Wald dan selang Wilson, dan peluang cakupannya
yang eksak (dijumlahkan atas sebaran binomial).
"""
import numpy as np
from scipy.stats import binom

Z = 1.959964


def wald(k1, k):
    p = k1 / k
    h = Z * np.sqrt(p * (1 - p) / k)
    return p - h, p + h


def wilson(k1, k):
    p = k1 / k
    pusat = (p + Z**2 / (2 * k)) / (1 + Z**2 / k)
    h = Z / (1 + Z**2 / k) * np.sqrt(p * (1 - p) / k + Z**2 / (4 * k**2))
    return pusat - h, pusat + h


def cakupan(selang, eta, k):
    k1 = np.arange(k + 1)
    lo, hi = selang(k1, k)
    return np.sum(binom.pmf(k1, k, eta) * ((lo <= eta) & (eta <= hi)))


if __name__ == "__main__":
    lo, hi = wald(7, 10)
    print(f"k1 = 7, k = 10: Wald   [{lo:.3f}, {hi:.3f}]")
    lo, hi = wilson(7, 10)
    print(f"                Wilson [{lo:.3f}, {hi:.3f}]")
    print("cakupan yang sebenarnya (target 0,95):")
    print("   k   eta    Wald    Wilson")
    for k in (10, 30):
        for e in (0.5, 0.7, 0.9, 0.97):
            print(f"{k:4d}  {e:.2f}   {cakupan(wald, e, k):.3f}"
                  f"   {cakupan(wilson, e, k):.3f}")
