"""Bab 20: pencarian rakus di graf tetangga (gagasan dasar NSW/HNSW).

Setiap titik latih dihubungkan dengan m tetangga terdekatnya (graf
dibuat simetris). Pencarian mulai dari titik acak, berpindah ke tetangga
graf yang paling dekat ke kueri, dan berhenti bila tidak ada tetangga
yang lebih dekat. Diulang dari beberapa titik awal. Dicatat recall@1 dan
banyaknya jarak yang dihitung, pada angka tulisan tangan yang dibakukan.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BENIH = 20260929


def graf_tetangga(X, m):
    _, idx = NearestNeighbors(n_neighbors=m + 1).fit(X).kneighbors(X)
    tetangga = [set(b[1:]) for b in idx]
    for i, b in enumerate(idx):
        for j in b[1:]:
            tetangga[j].add(i)
    return [np.array(sorted(t)) for t in tetangga]


def cari_rakus(X, graf, q, awal):
    dihitung = {}

    def jarak(i):
        if i not in dihitung:
            dihitung[i] = np.sum((X[i] - q) ** 2)
        return dihitung[i]

    terbaik = None
    for a in awal:
        i = a
        while True:
            t = graf[i]
            j = t[np.argmin([jarak(u) for u in t])]
            if jarak(j) < jarak(i):
                i = j
            else:
                break
        if terbaik is None or jarak(i) < jarak(terbaik):
            terbaik = i
    return terbaik, len(dihitung)


if __name__ == "__main__":
    X, _ = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    Q, D = X[:200], X[200:]
    benar = NearestNeighbors(n_neighbors=1).fit(D).kneighbors(Q)[1][:, 0]
    rng = np.random.default_rng(BENIH)
    print(" m   awal   recall@1   jarak/kueri (brute force 1597)")
    for m in (5, 10, 20):
        graf = graf_tetangga(D, m)
        for r in (1, 5):
            hit = jum = 0
            for i, q in enumerate(Q):
                awal = rng.integers(0, len(D), r)
                j, n = cari_rakus(D, graf, q, awal)
                hit += j == benar[i]
                jum += n
            print(f"{m:2d}    {r}      {hit / len(Q):.3f}       "
                  f"{jum / len(Q):6.1f}")
