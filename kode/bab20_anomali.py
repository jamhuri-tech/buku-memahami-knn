"""Bab 20: deteksi anomali dengan jarak ke tetangga ke-k dan dengan LOF
(Local Outlier Factor, Breunig dkk. 2000). Data: satu gumpalan padat,
satu gumpalan jarang, dan 20 pencilan yang dekat ke gumpalan padat.
Diukur dengan AUC (skor tinggi = anomali).
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

BENIH = 20260929


def data(rng):
    padat = rng.normal([0, 0], 0.3, (300, 2))
    jarang = rng.normal([5, 0], 1.5, (300, 2))
    sudut = rng.uniform(0, 2 * np.pi, 20)
    pencil = np.column_stack([1.5 * np.cos(sudut), 1.5 * np.sin(sudut)])
    X = np.vstack([padat, jarang, pencil])
    y = np.r_[np.zeros(600), np.ones(20)]
    return X, y


if __name__ == "__main__":
    X, y = data(np.random.default_rng(BENIH))
    k = 10
    d = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, -1]
    lof = LocalOutlierFactor(n_neighbors=k).fit(X)
    skor = -lof.negative_outlier_factor_
    print(f"AUC jarak ke tetangga ke-{k}: {roc_auc_score(y, d):.3f}")
    print(f"AUC LOF (k = {k})            : {roc_auc_score(y, skor):.3f}")
    atas = np.argsort(-d)[:20]
    atas_lof = np.argsort(-skor)[:20]
    print(f"pencilan di antara 20 skor tertinggi: jarak {int(y[atas].sum())},"
          f" LOF {int(y[atas_lof].sum())}")
