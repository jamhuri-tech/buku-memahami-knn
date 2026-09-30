"""Bab 3: siaran (broadcasting) untuk matriks jarak, argpartition,
take_along_axis, dan bincount untuk suara terbanyak.
"""
import numpy as np

BENIH = 20260929
rng = np.random.default_rng(BENIH)
X = rng.integers(0, 10, (6, 2)).astype(float)     # 6 titik latih
y = np.array([0, 1, 0, 1, 1, 0])
Q = rng.integers(0, 10, (3, 2)).astype(float)     # 3 titik kueri
selisih = Q[:, None, :] - X[None, :, :]
print("bentuk Q[:, None, :]:", Q[:, None, :].shape,
      " X[None, :, :]:", X[None, :, :].shape)
print("bentuk selisih:", selisih.shape)
D2 = (selisih ** 2).sum(axis=-1)
print("matriks jarak kuadrat (3 x 6):")
print(D2.astype(int))
k = 3
idx = np.argpartition(D2, k - 1, axis=1)[:, :k]
d = np.take_along_axis(D2, idx, axis=1)
idx = np.take_along_axis(idx, np.argsort(d, axis=1, kind="stable"), 1)
print("3 tetangga (argpartition):", idx.tolist())
print("3 tetangga (argsort)     :",
      np.argsort(D2, axis=1, kind="stable")[:, :k].tolist())
for i in range(3):
    print(f"kueri {i}: suara per kelas {np.bincount(y[idx[i]], minlength=2)}")
