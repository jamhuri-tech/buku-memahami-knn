"""Bab 15: pohon k-d dan ball tree dari nol, dengan penghitung jarak.

Keduanya membelah titik pada median koordinat yang sebarannya terbesar
sampai sebuah simpul berisi paling banyak `daun` titik. Pohon k-d
menyimpan kotak pembatas setiap simpul; ball tree menyimpan pusat
(rata-rata) dan jari-jari. Pencarian: telusuri anak yang lebih dekat
lebih dahulu, lewati simpul yang batas bawah jaraknya tidak lebih kecil
daripada jarak tetangga ke-k terbaik sejauh ini.
"""
import heapq

import numpy as np


class Simpul:
    __slots__ = ("idx", "kiri", "kanan", "lo", "hi", "pusat", "r")


def _bangun(X, idx, daun, bola):
    s = Simpul()
    P = X[idx]
    s.lo, s.hi = P.min(0), P.max(0)
    if bola:
        s.pusat = P.mean(0)
        s.r = np.sqrt(((P - s.pusat) ** 2).sum(1)).max()
    if len(idx) <= daun:
        s.idx, s.kiri, s.kanan = idx, None, None
        return s
    j = int(np.argmax(s.hi - s.lo))
    urut = idx[np.argsort(P[:, j], kind="stable")]
    h = len(idx) // 2
    s.idx = None
    s.kiri = _bangun(X, urut[:h], daun, bola)
    s.kanan = _bangun(X, urut[h:], daun, bola)
    return s


class Pohon:
    """bola=False: pohon k-d; bola=True: ball tree."""

    def __init__(self, X, daun=2, bola=False):
        self.X = np.asarray(X, dtype=float)
        self.bola = bola
        self.akar = _bangun(self.X, np.arange(len(self.X)), daun, bola)

    def batas_bawah(self, s, q):
        """Batas bawah jarak dari q ke titik mana pun di simpul s."""
        if self.bola:
            self.n_pusat += 1
            return max(0.0, np.sqrt(((q - s.pusat) ** 2).sum()) - s.r)
        g = np.maximum(0, np.maximum(s.lo - q, q - s.hi))
        return np.sqrt((g ** 2).sum())

    def query(self, q, k=1, jejak=None):
        """k tetangga terdekat q; mencatat banyaknya jarak ke titik."""
        q = np.asarray(q, dtype=float)
        self.n_jarak, self.n_pusat, self.n_simpul = 0, 0, 0
        terbaik = []                    # tumpukan (-jarak, indeks)

        def ke(s, lb):
            batas = -terbaik[0][0] if len(terbaik) == k else np.inf
            if lb >= batas:
                if jejak is not None:
                    jejak.append(("lewati", s, lb, batas))
                return
            self.n_simpul += 1
            if s.idx is not None:
                d = np.sqrt(((self.X[s.idx] - q) ** 2).sum(1))
                self.n_jarak += len(s.idx)
                for dj, ij in zip(d, s.idx):
                    if len(terbaik) < k:
                        heapq.heappush(terbaik, (-dj, ij))
                    elif dj < -terbaik[0][0]:
                        heapq.heapreplace(terbaik, (-dj, ij))
                if jejak is not None:
                    jejak.append(("daun", s, lb, list(s.idx)))
                return
            a = (s.kiri, self.batas_bawah(s.kiri, q))
            b = (s.kanan, self.batas_bawah(s.kanan, q))
            for anak, lba in sorted((a, b), key=lambda t: t[1]):
                ke(anak, lba)

        ke(self.akar, self.batas_bawah(self.akar, q))
        hasil = sorted((-d, i) for d, i in terbaik)
        return (np.array([d for d, _ in hasil]),
                np.array([i for _, i in hasil]))


if __name__ == "__main__":
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(20260929)
    X = rng.random((3000, 5))
    Q = rng.random((200, 5))
    ds, js = NearestNeighbors(n_neighbors=5).fit(X).kneighbors(Q)
    for nama, bola in (("pohon k-d", False), ("ball tree", True)):
        T = Pohon(X, daun=20, bola=bola)
        cocok, jarak = 0, 0
        for i, q in enumerate(Q):
            d, j = T.query(q, k=5)
            cocok += np.array_equal(j, js[i]) and np.allclose(d, ds[i])
            jarak += T.n_jarak
        print(f"{nama}: cocok dengan sklearn {cocok}/200")
        print(f"   rata-rata {jarak / 200:.0f} jarak per kueri "
              f"(brute force 3000)")
