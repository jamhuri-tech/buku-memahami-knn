"""Bab 8: jarak tetangga terdekat menyusut bila n bertambah.
Titik x = pusat kubus [0, 1]^p, n titik latih seragam, 2000 ulangan.
Untuk p = 1, harapannya eksak 1 / (2 (n + 1)).
"""
import numpy as np

BENIH = 20260929
rng = np.random.default_rng(BENIH)
print("    n   p=1: simulasi  1/(2(n+1))   p=2      p=10")
for n in (9, 99, 999, 9999):
    hasil = []
    for p in (1, 2, 10):
        d = [np.sqrt(((rng.random((n, p)) - 0.5) ** 2).sum(1)).min()
             for _ in range(2000)]
        hasil.append(np.mean(d))
    print(f"{n:5d}      {hasil[0]:.6f}    {1 / (2 * (n + 1)):.6f}"
          f"    {hasil[1]:.4f}   {hasil[2]:.4f}")
