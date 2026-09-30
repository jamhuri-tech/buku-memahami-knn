"""Bab 12: rata-rata lawan median tetangga bila 10% label tercemar
pencilan. x seragam di [0, 1], f(x) = sin(2 pi x), noise N(0; 0,3^2),
dan 10% label ditambah 5 (pencilan). Galat diukur terhadap f.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

BENIH = 20260929


def ramal(X, y, T, k, fungsi):
    _, idx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(T)
    return fungsi(y[idx], axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    f = lambda t: np.sin(2 * np.pi * t)
    T = np.linspace(0, 1, 1001)[:, None]
    print("  k    rata-rata   median   (galat kuadrat terhadap f)")
    hasil = {k: [0.0, 0.0] for k in (5, 15, 31)}
    for _ in range(200):
        X = rng.random((300, 1))
        y = f(X[:, 0]) + 0.3 * rng.standard_normal(300)
        cemar = rng.random(300) < 0.1
        y[cemar] += 5
        for k in hasil:
            for j, fn in enumerate((np.mean, np.median)):
                g = np.mean((ramal(X, y, T, k, fn) - f(T[:, 0])) ** 2)
                hasil[k][j] += g / 200
    for k, (a, b) in hasil.items():
        print(f"{k:3d}    {a:.4f}      {b:.4f}")
