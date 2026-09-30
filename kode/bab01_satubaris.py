"""Bab 1: KNN untuk angka tulisan tangan dalam beberapa baris
scikit-learn: validasi silang 5 lipatan untuk beberapa k, lalu ramalan
dan tetangga satu citra.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

BENIH = 20260929
X, y = load_digits(return_X_y=True)
print(f"{X.shape[0]} citra, {X.shape[1]} piksel, {len(np.unique(y))} kelas")
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
for k in (1, 3, 5, 15):
    model = KNeighborsClassifier(n_neighbors=k)
    akurasi = cross_val_score(model, X, y, cv=cv)
    print(f"k = {k:2d}: akurasi validasi silang {akurasi.mean():.4f}")
model = KNeighborsClassifier(n_neighbors=3).fit(X[1:], y[1:])
jarak, idx = model.kneighbors(X[:1])
print(f"citra pertama (label {y[0]}): ramalan {model.predict(X[:1])[0]}, "
      f"label tetangga {y[1:][idx[0]]}")
print(f"jarak ke tiga tetangga: {np.round(jarak[0], 2)}")
