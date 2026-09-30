"""Bab 4: jebakan menilai 1-NN pada data latihnya sendiri."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

BENIH = 20260929
X, y = load_digits(return_X_y=True)
X_lt, X_uji, y_lt, y_uji = train_test_split(
    X, y, test_size=450, stratify=y, random_state=BENIH)
for k in (1, 3, 5):
    m = KNeighborsClassifier(n_neighbors=k).fit(X_lt, y_lt)
    print(f"k = {k}: akurasi data latih {m.score(X_lt, y_lt):.4f}, "
          f"data uji {m.score(X_uji, y_uji):.4f}")
# Berapa banyak citra yang sama persis muncul lebih dari sekali?
_, hit = np.unique(X, axis=0, return_counts=True)
print(f"citra kembar di seluruh data: {np.sum(hit > 1)} pola")
