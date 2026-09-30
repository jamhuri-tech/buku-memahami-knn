"""Pemeriksa Contoh Soal Bab 1."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[7, 8], [8, 7], [6, 9], [7, 3], [8, 2], [6, 4]])
y = np.array([0, 0, 0, 1, 1, 1])                 # 0 jeruk, 1 apel
q = np.array([7, 6])
d2 = ((X - q) ** 2).sum(1)
assert list(d2) == [4, 2, 10, 9, 17, 5]
assert sorted(d2) == [2, 4, 5, 9, 10, 17]
for k, suara in ((1, (1, 0)), (3, (2, 1)), (5, (3, 2))):
    o = np.argsort(d2)[:k]
    assert tuple(np.bincount(y[o], minlength=2)) == suara
    assert KNeighborsClassifier(k).fit(X, y).predict([q])[0] == 0
print("Contoh Soal Bab 1: semua bilangan cocok")
