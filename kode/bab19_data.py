"""Bab 19: data harga rumah California (fetch_california_housing).

20.640 kelompok blok sensus AS tahun 1990; label = median nilai rumah
dalam ratusan ribu dolar, dipotong di 5,00001. Fungsi muat()
mengembalikan fitur, label, koordinat dalam km (hampiran
equirectangular, Bab 5), dan kelompok wilayah (sel 0,5 x 0,5 derajat)
untuk validasi silang per wilayah.
"""
import numpy as np
from sklearn.datasets import fetch_california_housing

R_BUMI = 6371.0


def muat():
    d = fetch_california_housing()
    X, y = d.data, d.target
    lat, lon = X[:, 6], X[:, 7]
    f0 = np.radians(lat.mean())
    km = np.column_stack([R_BUMI * np.radians(lon) * np.cos(f0),
                          R_BUMI * np.radians(lat)])
    wilayah = (np.floor(lat / 0.5) * 1000 + np.floor(lon / 0.5))
    return X, y, km, wilayah.astype(int), list(d.feature_names)


if __name__ == "__main__":
    X, y, km, wilayah, nama = muat()
    print(f"{X.shape[0]} kelompok blok, {X.shape[1]} fitur")
    print(f"label: median {np.median(y):.3f}, "
          f"simpangan baku {y.std():.3f}")
    print(f"label yang terpotong di 5.00001: {np.sum(y >= 5)}")
    print("fitur        median      persentil 99    maksimum")
    for j in (0, 2, 4, 5):
        print(f"{nama[j]:11s} {np.median(X[:, j]):9.2f}   "
              f"{np.percentile(X[:, j], 99):11.2f}   {X[:, j].max():10.2f}")
    print(f"lintang rata-rata {X[:, 6].mean():.2f}, "
          f"cos = {np.cos(np.radians(X[:, 6].mean())):.3f}")
    print(f"wilayah 0,5 x 0,5 derajat: {len(np.unique(wilayah))}")
    ada, hit = np.unique(X[:, 6:8], axis=0, return_counts=True)
    print(f"koordinat yang dipakai lebih dari satu kelompok blok: "
          f"{np.sum(hit > 1)}")
