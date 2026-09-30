"""Bab 14: waktu menghitung matriks jarak dengan tiga cara. Hasilnya
bergantung pada komputer, sehingga tidak dicocokkan dengan naskah.
# cek_keluaran: lewati
"""
import time

import numpy as np

BENIH = 20260929


def perulangan(A, B):
    D = np.empty((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            s = 0.0
            for t in range(A.shape[1]):
                s += (A[i, t] - B[j, t]) ** 2
            D[i, j] = s
    return D


def siaran(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)


def gram(A, B):
    return ((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :]
            - 2 * A @ B.T)


def ukur(f, A, B, ulang=3):
    t = []
    for _ in range(ulang):
        t0 = time.perf_counter()
        f(A, B)
        t.append(time.perf_counter() - t0)
    return min(t)


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    A = rng.random((100, 64))
    B = rng.random((1000, 64))
    tp = ukur(perulangan, A[:10], B, ulang=1) * 10
    ts, tg = ukur(siaran, A, B), ukur(gram, A, B)
    print(f"perulangan {tp:.3f} s, siaran {ts:.4f} s, Gram {tg:.5f} s")
    print(f"rasio perulangan/Gram {tp / tg:.0f}, siaran/Gram {ts / tg:.0f}")
