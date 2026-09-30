"""Bab 6: data anggur (load_wine). Satu fitur menguasai jarak Euclid,
dan pembakuan di dalam pipeline memulihkan akurasi.
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import StandardScaler

BENIH = 20260929
data = load_wine()
X, y = data.data, data.target
print(f"data anggur: {X.shape[0]} x {X.shape[1]}, kelas "
      f"{np.bincount(y)}")
v = X.var(axis=0)
urut = np.argsort(v)[::-1][:3]
print("fitur      rentang            bagian varians (%)")
for j in urut:
    print(f"{data.feature_names[j][:9]:9s}  {X[:, j].min():7.2f} - "
          f"{X[:, j].max():7.2f}   {100 * v[j] / v.sum():.5f}")
j = list(data.feature_names).index("hue")
print(f"{'hue':9s}  {X[:, j].min():7.2f} - {X[:, j].max():7.2f}   "
      f"{100 * v[j] / v.sum():.5f}")

cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
print("akurasi validasi silang 5 lipatan:")
print("penskalaan         k=1     k=5     k=15")
for nama, sc in (("tanpa", None), ("proline saja", "proline"),
                 ("pembakuan (z)", StandardScaler()),
                 ("min-max", MinMaxScaler()),
                 ("robust", RobustScaler())):
    hasil = []
    for k in (1, 5, 15):
        knn = KNeighborsClassifier(n_neighbors=k)
        if sc is None:
            a = cross_val_score(knn, X, y, cv=cv)
        elif sc == "proline":
            a = cross_val_score(knn, X[:, [12]], y, cv=cv)
        else:
            a = cross_val_score(make_pipeline(sc, knn), X, y, cv=cv)
        hasil.append(a.mean())
    print(f"{nama:17s}  " + "  ".join(f"{a:.4f}" for a in hasil))
