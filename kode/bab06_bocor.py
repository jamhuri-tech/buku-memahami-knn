"""Bab 6: pembakuan di luar pipeline (bocor) lawan di dalam pipeline,
pada 50 pembagian validasi silang data anggur, k = 5.
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BENIH = 20260929
X, y = load_wine(return_X_y=True)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                             random_state=BENIH)
knn = KNeighborsClassifier(n_neighbors=5)
Xz = StandardScaler().fit_transform(X)          # melihat seluruh data
bocor = cross_val_score(knn, Xz, y, cv=cv)
jujur = cross_val_score(make_pipeline(StandardScaler(), knn), X, y,
                        cv=cv)
print(f"bocor : rata-rata {bocor.mean():.4f}")
print(f"jujur : rata-rata {jujur.mean():.4f}")
print(f"selisih rata-rata {bocor.mean() - jujur.mean():+.4f}, "
      f"lipatan yang berbeda {np.sum(bocor != jujur)} dari {len(bocor)}")

# Kebocoran yang jauh lebih parah: memilih fitur dengan label SELURUH
# data sebelum validasi silang. Fitur dan label di sini murni acak,
# jadi akurasi yang jujur semestinya sekitar 0,5.
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold

rng = np.random.default_rng(BENIH)
Xa = rng.standard_normal((100, 2000))
ya = rng.integers(0, 2, 100)
cv5 = StratifiedKFold(5, shuffle=True, random_state=BENIH)
pilih = SelectKBest(f_classif, k=20).fit(Xa, ya)     # memakai semua label
bocor = cross_val_score(knn, pilih.transform(Xa), ya, cv=cv5).mean()
jujur = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), knn),
                        Xa, ya, cv=cv5).mean()
print("data acak 100 x 2000, pilih 20 fitur, 5-NN:")
print(f"  pilih di luar validasi silang : {bocor:.4f}")
print(f"  pilih di dalam pipeline       : {jujur:.4f}")
