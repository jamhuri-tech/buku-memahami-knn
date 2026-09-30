"""Bab 16: LSH hyperplane acak (Charikar 2002) untuk kemiripan kosinus.

(1) Peluang dua vektor bersudut theta mendapat bit hash yang sama:
    simulasi lawan 1 - theta/pi.
(2) Indeks LSH dengan b tabel, masing-masing r bit, pada angka tulisan
    tangan yang dipusatkan dan dinormalkan: recall 10 tetangga (kosinus)
    dan bagian data yang diperiksa ulang secara tepat.
"""
from collections import defaultdict

import numpy as np

BENIH = 20260929


class IndeksLSH:
    def __init__(self, X, r, b, rng):
        self.X = X
        self.R = rng.standard_normal((b, r, X.shape[1]))
        self.bobot = 1 << np.arange(r)
        self.tabel = []
        for t in range(b):
            kode = ((X @ self.R[t].T) > 0) @ self.bobot
            ember = defaultdict(list)
            for i, c in enumerate(kode):
                ember[c].append(i)
            self.tabel.append(ember)

    def calon(self, q):
        c = set()
        for t, ember in enumerate(self.tabel):
            kode = ((self.R[t] @ q) > 0) @ self.bobot
            c.update(ember.get(kode, []))
        return np.fromiter(c, dtype=int)

    def query(self, q, k):
        c = self.calon(q)
        if len(c) == 0:
            return c, 0
        s = self.X[c] @ q                      # kosinus (sudah normal)
        return c[np.argsort(-s)[:k]], len(c)


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    print("sudut   simulasi   1 - theta/pi")
    for der in (10, 30, 60, 90, 135):
        t = np.radians(der)
        u = np.array([1.0, 0.0] + [0.0] * 48)
        v = np.array([np.cos(t), np.sin(t)] + [0.0] * 48)
        R = rng.standard_normal((200_000, 50))
        sama = np.mean(np.sign(R @ u) == np.sign(R @ v))
        print(f"{der:4d}    {sama:.4f}     {1 - t / np.pi:.4f}")

    from sklearn.datasets import load_digits
    X, _ = load_digits(return_X_y=True)
    X = X - X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q, D = X[:200], X[200:]
    benar = np.argsort(-(Q @ D.T), axis=1)[:, :10]
    print("  r   b   recall@10   bagian data diperiksa")
    for r, b in ((4, 1), (8, 1), (8, 5), (8, 20), (12, 20), (16, 50)):
        idx = IndeksLSH(D, r, b, rng)
        rec, frak = 0.0, 0.0
        for i, q in enumerate(Q):
            j, nc = idx.query(q, 10)
            rec += len(set(j) & set(benar[i])) / 10
            frak += nc / len(D)
        print(f"{r:3d} {b:3d}     {rec / len(Q):.3f}        "
              f"{frak / len(Q):.3f}")
