"""Bab 17: nilai hilang. (1) jarak nan_euclidean dengan tangan dan
dengan scikit-learn. (2) Data anggur dengan 20% nilai dihapus acak:
imputasi rata-rata lawan KNNImputer (k = 5), galat nilai yang
diimputasi (dalam satuan simpangan baku) dan akurasi 5-NN sesudahnya.
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BENIH = 20260929
x = np.array([[1, np.nan, 3, 4]])
z = np.array([[2, 5, np.nan, 6]])
print(f"nan_euclidean((1,?,3,4), (2,5,?,6)) = "
      f"{nan_euclidean_distances(x, z)[0, 0]:.4f}")

rng = np.random.default_rng(BENIH)
X, y = load_wine(return_X_y=True)
Xz = StandardScaler().fit_transform(X)
hilang = rng.random(X.shape) < 0.2
Xh = np.where(hilang, np.nan, Xz)
cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
print("imputasi        galat nilai (sb)   akurasi 5-NN")
for nama, imp in (("rata-rata", SimpleImputer()),
                  ("KNNImputer k=5", KNNImputer(n_neighbors=5))):
    Xi = imp.fit_transform(Xh)
    g = np.sqrt(np.mean((Xi[hilang] - Xz[hilang]) ** 2))
    a = cross_val_score(make_pipeline(imp, KNeighborsClassifier(5)),
                        Xh, y, cv=cv).mean()
    print(f"{nama:15s}      {g:.3f}            {a:.4f}")
a = cross_val_score(KNeighborsClassifier(5), Xz, y, cv=cv).mean()
print(f"tanpa nilai hilang                      {a:.4f}")
