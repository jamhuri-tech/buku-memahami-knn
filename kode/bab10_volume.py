"""Bab 10: volume bola satuan V_p = pi^(p/2) / Gamma(p/2 + 1), bagian
kubus [-1, 1]^p yang diisi bola itu, massa kulit setebal 5%, dan
panjang sisi kubus kecil yang memuat 1% data seragam.
"""
import numpy as np
from scipy.special import gamma

print("  p     V_p        V_p/2^p     1-0.95^p   0.01^(1/p)")
for p in (1, 2, 3, 5, 10, 20, 50, 100):
    V = np.pi ** (p / 2) / gamma(p / 2 + 1)
    print(f"{p:3d}   {V:9.4g}   {V / 2**p:9.3g}    {1 - 0.95**p:.4f}"
          f"     {0.01 ** (1 / p):.4f}")
p_maks = max(range(1, 30), key=lambda p: np.pi ** (p / 2)
             / gamma(p / 2 + 1))
print(f"V_p terbesar di p = {p_maks}")
