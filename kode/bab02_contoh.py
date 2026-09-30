"""Pemeriksa Contoh Soal Bab 2."""
import math

import numpy as np
from scipy.special import gamma

u, v = np.array([1, 2, 2]), np.array([2, 0, 1])
assert u @ v == 4 and u @ u == 9 and v @ v == 5
assert round(4 / (3 * math.sqrt(5)), 4) == 0.5963
assert round(math.degrees(math.acos(4 / (3 * math.sqrt(5)))), 1) == 53.4
assert ((u - v) ** 2).sum() == 6 == 9 - 8 + 5
assert round(3 * math.sqrt(5), 2) == 6.71

S = np.array([[2, 1], [1, 2]])
lam, V = np.linalg.eigh(S)
assert np.allclose(lam, [1, 3])
a, b = np.array([1, 1]) / 2**0.5, np.array([-1, 1]) / 2**0.5
M = 3**-0.5 * np.outer(a, a) + np.outer(b, b)
assert np.allclose(M @ S @ M, np.eye(2))
assert np.allclose(np.round(M, 4), [[0.7887, -0.2113], [-0.2113, 0.7887]])
assert np.isclose(lam.sum(), np.trace(S)) and np.isclose(lam.prod(), 3)

C = np.array([[2, 0], [1, 2]])
assert np.allclose(C @ C.T, [[4, 2], [2, 5]])
assert np.allclose(np.linalg.cholesky([[4, 2], [2, 5]]), C)

assert math.isclose(gamma(1.5), math.sqrt(math.pi) / 2)
assert math.isclose(gamma(2.5), 3 * math.sqrt(math.pi) / 4)
assert round(gamma(1.5), 4) == 0.8862 and round(gamma(2.5), 4) == 1.3293
assert math.isclose(math.pi**1.5 / gamma(2.5), 4 * math.pi / 3)
print("Contoh Soal Bab 2: semua bilangan cocok")
