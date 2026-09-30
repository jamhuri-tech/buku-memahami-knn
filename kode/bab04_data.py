"""Bab 4: delapan titik rekaan berkoordinat bulat, dipakai di Contoh
Soal sepanjang buku. Kelas 0 (A, biru) di kiri, kelas 1 (B, jingga) di
kanan. Tidak ada unsur acak.
"""
import numpy as np

X = np.array([[1, 1], [2, 4], [3, 2], [4, 6],      # kelas A
              [5, 3], [6, 1], [7, 4], [8, 2]])     # kelas B
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
NAMA = np.array(["A", "B"])

if __name__ == "__main__":
    for xi, yi in zip(X, y):
        print(f"({xi[0]}, {xi[1]})  kelas {NAMA[yi]}")
