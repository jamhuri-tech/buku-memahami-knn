"""Bab 5: tiga titik latih, tiga metrik, tiga tetangga terdekat yang
berbeda untuk titik uji (0, 0).
"""
import numpy as np

X = np.array([[6, 0], [4, 4], [2, 5]])
nama = ["a=(6,0)", "b=(4,4)", "c=(2,5)"]
L1 = np.abs(X).sum(1)
L2 = np.sqrt((X**2).sum(1))
Li = np.abs(X).max(1)
print("titik      L1     L2      Linf")
for i in range(3):
    print(f"{nama[i]}  {L1[i]:2d}   {L2[i]:.4f}   {Li[i]}")
for m, d in (("L1", L1), ("L2", L2), ("Linf", Li)):
    print(f"tetangga terdekat menurut {m:4s}: {nama[int(d.argmin())]}")
