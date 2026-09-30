"""Pemeriksa Contoh Soal Bab 12."""
from fractions import Fraction as F

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# ---- Contoh Soal 12.1: ramalan di x0 = 4,8 ----
x = np.array([1, 2, 4, 7, 8]); y = np.array([3, 5, 4, 9, 10])
d = np.abs(x - 4.8)
assert np.allclose(np.sort(d), [0.8, 2.2, 2.8, 3.2, 3.8])
for k, v in ((1, 4), (2, 6.5), (3, 6)):
    m = KNeighborsRegressor(k).fit(x[:, None], y)
    assert np.isclose(m.predict([[4.8]])[0], v)

# ---- Contoh Soal 12.2: identitas LOO ----
x = np.array([1, 2, 4, 7]); y = np.array([1, 3, 2, 6])
yh = np.array([np.mean(y[np.argsort(np.abs(x - xi), kind="stable")[:2]])
               for xi in x])
assert np.allclose(yh, [2, 2, 2.5, 4])
r = y - yh
assert np.allclose(r, [-1, 1, -0.5, 2])
loo = []
for i in range(4):
    lain = np.delete(np.arange(4), i)
    j = lain[np.argmin(np.abs(x[lain] - x[i]))]
    loo.append(y[i] - y[j])
assert np.allclose(loo, 2 * r)
assert np.isclose(np.mean(r**2), 1.5625) and np.isclose(np.mean(
    np.array(loo) ** 2), 6.25)

# ---- Contoh Soal 12.3: bias tepi ----
n, k = 99, 9
assert sum(F(i, n + 1) for i in range(1, k + 1)) / k == F(k + 1, 2 * (n + 1))
assert F(10, 200) == F(1, 20)

# ---- Contoh Soal 12.4: rata-rata lawan median ----
lab = np.array([2.1, 1.9, 2.0, 2.2, 15])
assert np.isclose(lab.mean(), 4.64) and np.isclose(np.median(lab), 2.1)
assert np.isclose(lab[:4].mean(), 2.05)
rng = np.random.default_rng(20260929)
for _ in range(200):
    biasa = rng.normal(2, 0.1, 8)
    pencil = rng.choice([-1e6, 1e6], 7)
    med = np.median(np.concatenate([biasa, pencil]))
    assert biasa.min() <= med <= biasa.max()

print("Contoh Soal Bab 12: semua bilangan cocok")
