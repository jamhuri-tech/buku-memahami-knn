"""Bab 12: identitas validasi silang tinggalkan-satu untuk regresi k-NN.

Bila yhat_i adalah rata-rata k tetangga TERMASUK titik i sendiri, maka
galat LOO regresi (k-1)-NN sama dengan (k / (k - 1))^2 kali galat
kuadrat data latih yhat. Diperiksa pada data diabetes (dibakukan)
dengan LeaveOneOut scikit-learn.
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
_, idx = NearestNeighbors(n_neighbors=41).fit(X).kneighbors(X)
assert np.all(idx[:, 0] == np.arange(len(X)))     # tetangga 1 = diri
print("  k   latih (dengan diri)  x (k/(k-1))^2   LOO (k-1)-NN")
for k in (2, 6, 11, 21, 41):
    yhat = y[idx[:, :k]].mean(1)
    latih = np.mean((y - yhat) ** 2)
    loo = -cross_val_score(KNeighborsRegressor(k - 1), X, y,
                           cv=LeaveOneOut(),
                           scoring="neg_mean_squared_error").mean()
    print(f"{k:3d}   {latih:10.2f}          {latih * (k / (k - 1))**2:10.2f}"
          f"     {loo:10.2f}")
