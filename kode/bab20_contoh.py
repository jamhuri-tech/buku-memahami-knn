"""Pemeriksa Contoh Soal Bab 20."""
import math

import numpy as np
from scipy.sparse.csgraph import shortest_path

s = math.sin(math.radians(22.5))
assert round(s, 5) == 0.38268 and round(2 * s, 5) == 0.76537
assert round(4 * 2 * s, 4) == 3.0615
sudut = np.radians([0, 45, 90, 135, 180])
P = np.column_stack([np.cos(sudut), np.sin(sudut)])
W = np.zeros((5, 5))
for i in range(4):
    W[i, i + 1] = W[i + 1, i] = np.linalg.norm(P[i] - P[i + 1])
assert math.isclose(shortest_path(W, directed=False)[0, 4], 8 * s)
assert round((math.pi - 8 * s) / math.pi * 100, 1) == 2.6
assert round((math.pi - 2) / math.pi * 100) == 36
m = 10_000
assert abs(2 * (m - 1) * math.sin(math.pi / (2 * (m - 1))) - math.pi) < 1e-6
print("Contoh Soal Bab 20: semua bilangan cocok")
