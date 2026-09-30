"""Bab 7: galat k-NN di satu titik bila semua tetangga mempunyai
posterior yang sama eta = 0,7 (bias nol). Galat = min(eta, 1 - eta)
+ |2 eta - 1| * P(suara terbanyak berpihak ke kelas yang salah).
"""
from math import comb

eta = 0.7


def q(k, eta):
    """P(Bin(k, eta) < k/2), k ganjil: suara terbanyak memilih 0."""
    return sum(comb(k, j) * eta**j * (1 - eta) ** (k - j)
               for j in range(0, (k - 1) // 2 + 1))


print("  k   P(suara salah)   galat di titik")
for k in (1, 3, 5, 11, 21, 51, 101):
    qk = q(k, eta)
    print(f"{k:3d}   {qk:.5f}          {0.3 + 0.4 * qk:.5f}")
print(f"galat Bayes di titik ini: {min(eta, 1 - eta):.5f}")
