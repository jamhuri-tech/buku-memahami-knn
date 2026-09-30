"""Bab 17: kondensasi Hart (1968) dan penyuntingan Wilson (1972).

kondensasi(X, y): mulai dari satu titik per kelas; berulang-ulang,
setiap titik yang salah diklasifikasi 1-NN pada himpunan terpilih
ditambahkan, sampai satu lintasan penuh tidak menambah apa pun.
suntingan(X, y, k): buang setiap titik yang salah diklasifikasi k-NN
pada titik-titik lain (tanpa dirinya sendiri).
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


def kondensasi(X, y, urutan=None):
    n = len(X)
    urutan = np.arange(n) if urutan is None else urutan
    pilih = []
    for c in np.unique(y):                      # satu titik per kelas
        pilih.append(urutan[np.flatnonzero(y[urutan] == c)[0]])
    lintasan = 0
    while True:
        lintasan += 1
        tambah = 0
        for i in urutan:
            if i in pilih:
                continue
            P = X[pilih]
            j = np.argmin(((P - X[i]) ** 2).sum(1))
            if y[pilih[j]] != y[i]:
                pilih.append(i)
                tambah += 1
        if tambah == 0:
            return np.array(pilih), lintasan


def suntingan(X, y, k=3):
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    tet = np.array([[j for j in b if j != i][:k] for i, b in
                    enumerate(idx)])
    suara = np.array([np.bincount(y[t], minlength=y.max() + 1)
                      for t in tet])
    return np.flatnonzero(suara.argmax(1) == y)


if __name__ == "__main__":
    from bab04_sintetis import bangkitkan, eta
    BENIH = 20260929
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(1000, rng)
    U = np.random.default_rng(BENIH + 1).random((50_000, 2))
    e = eta(U)

    def galat(Xs, ys, k=1):
        yh = KNeighborsClassifier(k).fit(Xs, ys).predict(U)
        return np.mean(np.where(yh == 1, 1 - e, e))

    s = suntingan(X, y, 3)
    c, lc = kondensasi(X, y)
    sc, lsc = kondensasi(X[s], y[s])
    print("himpunan latih          titik   galat 1-NN sebenarnya")
    print(f"semua                    {len(X):4d}      {galat(X, y):.4f}")
    print(f"kondensasi Hart          {len(c):4d}      "
          f"{galat(X[c], y[c]):.4f}")
    print(f"penyuntingan Wilson      {len(s):4d}      "
          f"{galat(X[s], y[s]):.4f}")
    print(f"suntingan + kondensasi   {len(sc):4d}      "
          f"{galat(X[s][sc], y[s][sc]):.4f}")
    print(f"(15-NN pada semua data: {galat(X, y, 15):.4f}; "
          f"galat Bayes 0.1356)")
    # Kondensasi selalu konsisten dengan himpunan asalnya:
    yh = KNeighborsClassifier(1).fit(X[c], y[c]).predict(X)
    print(f"1-NN pada hasil kondensasi: {np.sum(yh != y)} titik latih "
          f"salah; {lc} lintasan")
