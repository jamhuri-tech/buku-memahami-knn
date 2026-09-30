"""Bab 7: validasi silang tinggalkan-satu (LOO) untuk SEMUA k sekaligus
dengan satu pencarian tetangga, dicocokkan dengan LeaveOneOut
scikit-learn, lalu dibandingkan dengan 10 lipatan dan galat harapan
yang sebenarnya pada data gelombang (n = 400).
"""
import numpy as np
from sklearn.model_selection import (KFold, LeaveOneOut,
                                     cross_val_score)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from bab04_sintetis import bangkitkan, eta

BENIH = 20260929
KMAKS = 151


def ramal_semua_k(X, y, Xq, kmaks, buang_diri=False):
    """Ramalan k-NN (label 0/1) untuk k = 1..kmaks, bentuk (nq, kmaks).
    Satu pencarian kmaks tetangga; suara dijumlahkan kumulatif."""
    m = kmaks + int(buang_diri)
    nn = NearestNeighbors(n_neighbors=m).fit(X)
    _, idx = nn.kneighbors(Xq)
    if buang_diri:                 # Xq = X: buang titik itu sendiri
        idx = np.array([[j for j in baris if j != i][:kmaks]
                        for i, baris in enumerate(idx)])
    satu = np.cumsum(y[idx], axis=1)          # suara kelas 1
    k = np.arange(1, kmaks + 1)
    return (2 * satu > k).astype(int)         # seri -> kelas 0


def kurva(X, y, kmaks=KMAKS):
    """Galat LOO, galat 10 lipatan, dan galat sebenarnya, k = 1..kmaks."""
    loo = np.mean(ramal_semua_k(X, y, X, kmaks, True) != y[:, None], 0)
    g10 = np.zeros(kmaks)
    for lt, uji in KFold(10, shuffle=True,
                         random_state=BENIH).split(X):
        r = ramal_semua_k(X[lt], y[lt], X[uji], kmaks)
        g10 += np.sum(r != y[uji, None], axis=0)
    g10 /= len(X)
    U = np.random.default_rng(BENIH + 1).random((50_000, 2))
    e = eta(U)[:, None]
    r = ramal_semua_k(X, y, U, kmaks)
    benar = np.mean(np.where(r == 1, 1 - e, e), axis=0)
    return loo, g10, benar


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(400, rng)
    loo, g10, benar = kurva(X, y)
    for k in (1, 15):
        sk = 1 - cross_val_score(KNeighborsClassifier(k), X, y,
                                 cv=LeaveOneOut()).mean()
        print(f"k = {k:2d}: LOO cepat {loo[k - 1]:.4f}, "
              f"LeaveOneOut sklearn {sk:.4f}")
    ganjil = np.arange(1, KMAKS + 1, 2)
    print("k ganjil terbaik menurut:")
    for nama, g in (("LOO", loo), ("10 lipatan", g10),
                    ("galat sebenarnya", benar)):
        i = ganjil[np.argmin(g[ganjil - 1])]
        print(f"  {nama:16s}: k = {i:3d}, taksiran {g[i - 1]:.4f},"
              f" benar {benar[i - 1]:.4f}")
