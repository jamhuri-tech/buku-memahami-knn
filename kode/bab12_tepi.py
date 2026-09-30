"""Bab 12: bias regresi k-NN di tepi domain. x seragam di [0, 1],
f(x) = x tanpa noise, n = 99. Di x0 = 0 tetangga hanya di satu sisi,
sehingga ramalan rata-rata (k + 1) / (2 (n + 1)); di x0 = 1/2 simetris.
"""
import numpy as np

BENIH = 20260929
rng = np.random.default_rng(BENIH)
n, ulang = 99, 20_000
X = np.sort(rng.random((ulang, n)), axis=1)
print("  k   tepi x0=0: simulasi  (k+1)/(2(n+1))   tengah x0=1/2")
for k in (1, 9, 19, 49):
    tepi = X[:, :k].mean(1).mean()             # k titik terkecil
    d = np.abs(X - 0.5)
    tengah = np.take_along_axis(X, np.argsort(d, 1)[:, :k], 1).mean(1)
    print(f"{k:3d}      {tepi:.4f}          {(k + 1) / (2 * (n + 1)):.4f}"
          f"          {tengah.mean() - 0.5:+.4f}")
