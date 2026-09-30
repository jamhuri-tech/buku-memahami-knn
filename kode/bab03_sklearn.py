"""Bab 3: antarmuka scikit-learn untuk KNN: fit, predict, predict_proba,
kneighbors, pipeline dengan pembakuan, dan beberapa cara validasi
silang pada data anggur.
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import (GroupKFold, KFold, LeaveOneOut,
                                     StratifiedKFold, cross_val_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BENIH = 20260929
X, y = load_wine(return_X_y=True)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
model.fit(X[1:], y[1:])
print("ramalan untuk contoh pertama:", model.predict(X[:1]))
print("peluang:", model.predict_proba(X[:1]).round(2))
knn = model[-1]
jarak, idx = knn.kneighbors(model[0].transform(X[:1]))
print("indeks tetangga:", idx[0])
print("jarak:", jarak[0].round(3))
kelompok = np.arange(len(y)) % 10                  # contoh kelompok
for nama, cv in (("KFold 5", KFold(5, shuffle=True, random_state=BENIH)),
                 ("StratifiedKFold 5",
                  StratifiedKFold(5, shuffle=True, random_state=BENIH)),
                 ("LeaveOneOut", LeaveOneOut()),
                 ("GroupKFold 5", GroupKFold(5))):
    g = kelompok if nama.startswith("Group") else None
    s = cross_val_score(model, X, y, cv=cv, groups=g)
    print(f"{nama:18s}: {len(s):3d} lipatan, akurasi {s.mean():.4f}")
