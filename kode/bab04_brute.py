"""Bab 4: KNN brute force dari nol, dicocokkan dengan scikit-learn
pada angka tulisan tangan (load_digits).
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

BENIH = 20260929


class TetanggaBrute:
    """Pengklasifikasi k-NN: fit menyimpan, predict menghitung."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.kelas_ = np.unique(self.y_)
        return self

    def kneighbors(self, X):
        X = np.asarray(X, dtype=float)
        d2 = (np.sum(X**2, axis=1)[:, None]
              - 2 * X @ self.X_.T
              + np.sum(self.X_**2, axis=1)[None, :])
        np.maximum(d2, 0, out=d2)           # galat pembulatan (Bab 14)
        idx = np.argpartition(d2, self.k - 1, axis=1)[:, :self.k]
        baris = np.arange(len(X))[:, None]
        urut = np.argsort(d2[baris, idx], axis=1, kind="stable")
        idx = idx[baris, urut]
        return np.sqrt(d2[baris, idx]), idx

    def predict(self, X):
        _, idx = self.kneighbors(X)
        label = np.searchsorted(self.kelas_, self.y_[idx])
        suara = np.zeros((len(idx), len(self.kelas_)), dtype=int)
        np.add.at(suara, (np.arange(len(idx))[:, None], label), 1)
        return self.kelas_[suara.argmax(axis=1)]   # seri: kelas terkecil


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    X_lt, X_uji, y_lt, y_uji = train_test_split(
        X, y, test_size=450, stratify=y, random_state=BENIH)
    print(f"data latih {X_lt.shape}, data uji {X_uji.shape}")
    for k in (1, 3, 5, 15):
        kita = TetanggaBrute(k).fit(X_lt, y_lt).predict(X_uji)
        sk = KNeighborsClassifier(n_neighbors=k, algorithm="brute")
        sk = sk.fit(X_lt, y_lt).predict(X_uji)
        print(f"k = {k:2d}: akurasi kita {np.mean(kita == y_uji):.4f},"
              f" sklearn {np.mean(sk == y_uji):.4f}, "
              f"beda {np.sum(kita != sk)}")
    n, p = X_lt.shape
    print(f"perkalian per titik uji: n*p = {n}*{p} = {n * p}")
    print(f"untuk {len(X_uji)} titik uji: {n * p * len(X_uji)}")
