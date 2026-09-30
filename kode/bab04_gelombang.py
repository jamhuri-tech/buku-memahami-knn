"""Bab 4: galat k-NN pada data gelombang untuk k = 1, 15, 151,
dihitung terhadap posterior yang sebenarnya (bukan terhadap label uji
yang acak), sehingga hasilnya dapat dibandingkan dengan galat Bayes.
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from bab04_sintetis import bangkitkan, eta, galat_bayes

BENIH = 20260929
rng = np.random.default_rng(BENIH)
X, y = bangkitkan(400, rng)          # sama dengan Gambar 4.2
U = np.random.default_rng(BENIH + 1).random((200_000, 2))
e = eta(U)
print(f"galat Bayes            : {galat_bayes():.4f}")
for k in (1, 15, 151):
    yh = KNeighborsClassifier(n_neighbors=k).fit(X, y).predict(U)
    galat = np.mean(np.where(yh == 1, 1 - e, e))
    print(f"galat harapan k = {k:3d} : {galat:.4f}")
