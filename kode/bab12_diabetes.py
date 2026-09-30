"""Bab 12: regresi k-NN lawan regresi linear pada data diabetes
(442 x 10), validasi silang 10 lipatan diulang 5 kali, dengan dan tanpa
pembakuan. Skor R^2 dan akar galat kuadrat rata-rata (RMSE).
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BENIH = 20260929
X, y = load_diabetes(return_X_y=True)
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=BENIH)


def nilai(model):
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
    mse = -cross_val_score(model, X, y, cv=cv,
                           scoring="neg_mean_squared_error").mean()
    return r2, np.sqrt(mse)


print(f"simpangan baku y: {y.std():.2f}")
print("model                    R^2      RMSE")
r2, rm = nilai(LinearRegression())
print(f"regresi linear          {r2:.4f}   {rm:.2f}")
for k in (1, 5, 10, 20, 40, 80):
    r2, rm = nilai(make_pipeline(StandardScaler(), KNeighborsRegressor(k)))
    print(f"k-NN k = {k:2d} (dibakukan) {r2:.4f}   {rm:.2f}")
r2, rm = nilai(make_pipeline(StandardScaler(),
                             KNeighborsRegressor(20, weights="distance")))
print(f"k-NN k = 20, bobot 1/d  {r2:.4f}   {rm:.2f}")
