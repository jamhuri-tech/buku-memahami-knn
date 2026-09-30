"""Bab 18: angka tulisan tangan diproyeksikan ke 2 dimensi dengan PCA
dan dengan NCA kita (dilatih pada data latih yang dibakukan), lalu
akurasi 5-NN pada data uji di ruang dua dimensi itu.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from bab18_nca import latih_nca

BENIH = 20260929


def proyeksi_dua():
    X, y = load_digits(return_X_y=True)
    Xl, Xu, yl, yu = train_test_split(X, y, test_size=450, stratify=y,
                                      random_state=BENIH)
    sc = StandardScaler().fit(Xl)
    Xl, Xu = sc.transform(Xl), sc.transform(Xu)
    pca = PCA(2).fit(Xl)
    L, _ = latih_nca(Xl, yl, d=2, iterasi=300)
    return {"PCA": (pca.transform(Xl), pca.transform(Xu)),
            "NCA": (Xl @ L.T, Xu @ L.T)}, yl, yu


if __name__ == "__main__":
    hasil, yl, yu = proyeksi_dua()
    for nama, (A, B) in hasil.items():
        a = KNeighborsClassifier(5).fit(A, yl).score(B, yu)
        print(f"{nama} ke 2 dimensi: akurasi 5-NN data uji {a:.4f}")
