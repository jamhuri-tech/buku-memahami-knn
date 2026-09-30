"""Bab 11: suara berbobot 1/d di titik (4, 3), dicocokkan dengan
weights="distance" milik scikit-learn, dan perilaku bila d = 0.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from bab04_data import X, y, NAMA

q = np.array([[4, 3]])
d = np.sqrt(((X - q) ** 2).sum(1))
urut = np.argsort(d, kind="stable")
for k in (3, 5):
    tet = urut[:k]
    w = 1 / d[tet]
    suara = [w[y[tet] == c].sum() for c in (0, 1)]
    m = KNeighborsClassifier(k, weights="distance").fit(X, y)
    pr = m.predict_proba(q)[0]
    print(f"k = {k}: bobot " + " ".join(f"{v:.4f}" for v in w))
    print(f"   suara A {suara[0]:.4f}, B {suara[1]:.4f}; "
          f"predict_proba {pr.round(4)}")
# Titik uji yang sama dengan titik latih (3, 2): jaraknya nol.
m = KNeighborsClassifier(3, weights="distance").fit(X, y)
print("titik uji (3, 2):", m.predict_proba([[3, 2]])[0])
