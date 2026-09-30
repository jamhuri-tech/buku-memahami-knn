"""Bab 4: tabel jarak kuadrat dari titik uji (4, 3) dan (4, 0) ke
delapan titik rekaan, lalu ramalan k-NN untuk beberapa k.
"""
import numpy as np
from bab04_data import X, y, NAMA

for q in ([4, 3], [4, 0]):
    q = np.array(q)
    d2 = ((X - q) ** 2).sum(axis=1)
    urut = np.argsort(d2, kind="stable")
    print(f"titik uji ({q[0]}, {q[1]}):")
    for r, i in enumerate(urut, 1):
        print(f"  {r}. ({X[i, 0]}, {X[i, 1]}) {NAMA[y[i]]}"
              f"  jarak^2 = {d2[i]:2d}")
    for k in (1, 3, 5):
        suara = np.bincount(y[urut[:k]], minlength=2)
        print(f"  k = {k}: suara A:B = {suara[0]}:{suara[1]}")

# scikit-learn di titik (4, 0), tempat setiap jarak berpasangan seri.
from sklearn.neighbors import KNeighborsClassifier


def titik(j):
    return f"({X[j, 0]}, {X[j, 1]})"


q = np.array([[4, 0]])
for algo in ("brute", "kd_tree", "ball_tree"):
    m = KNeighborsClassifier(n_neighbors=1, algorithm=algo).fit(X, y)
    _, i1 = m.kneighbors(q, n_neighbors=1)
    _, i2 = m.kneighbors(q, n_neighbors=2)
    print(f"{algo:9s}  k=1: {titik(i1[0, 0])}  k=2: "
          f"{titik(i2[0, 0])} {titik(i2[0, 1])}  "
          f"ramalan {NAMA[m.predict(q)[0]]}")
