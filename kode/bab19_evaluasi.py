"""Bab 19: RMSE validasi silang 5 lipatan dengan dua cara membagi data:
acak (KFold) dan per wilayah (GroupKFold, sel 0,5 x 0,5 derajat).
Model: rata-rata, regresi linear, k-NN semua fitur (dibakukan), k-NN
geografis (haversine), dan k-NN pada (x_km, y_km, s * MedInc).
Untuk setiap cara membagi, k dan s dipilih dari kisi.
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bab19_data import muat

BENIH = 20260929
KS = (5, 10, 20, 40)
SS = (0, 5, 10, 20, 40)


def rmse(model, Z, y, cv, g):
    return -cross_val_score(model, Z, y, cv=cv, groups=g,
                            scoring="neg_root_mean_squared_error").mean()


def km_pendapatan(X, km, s):
    return np.column_stack([km, s * X[:, 0]])


def semua_skema():
    X, y, km, wil, _ = muat()
    skema = {"acak": (KFold(5, shuffle=True, random_state=BENIH), None),
             "wilayah": (GroupKFold(5), wil)}
    hasil = {}
    for nama, (cv, g) in skema.items():
        h = {"rata-rata": rmse(DummyRegressor(), X, y, cv, g),
             "regresi linear": rmse(LinearRegression(), X, y, cv, g)}
        h["k-NN semua fitur"] = min(
            (rmse(make_pipeline(StandardScaler(), KNeighborsRegressor(k)),
                  X, y, cv, g), k) for k in KS)
        rad = np.radians(X[:, [6, 7]])
        h["k-NN haversine"] = min(
            (rmse(KNeighborsRegressor(k, metric="haversine",
                                      algorithm="ball_tree"),
                  rad, y, cv, g), k) for k in KS)
        kisi = {(s, k): rmse(KNeighborsRegressor(k),
                             km_pendapatan(X, km, s), y, cv, g)
                for s in SS for k in KS}
        h["k-NN km + s MedInc"] = min((v, sk) for sk, v in kisi.items())
        hasil[nama] = (h, kisi)
    return hasil


if __name__ == "__main__":
    hasil = semua_skema()
    print("model                  acak           wilayah")
    for m in hasil["acak"][0]:
        sel = []
        for nama in ("acak", "wilayah"):
            v = hasil[nama][0][m]
            if isinstance(v, tuple):
                par = v[1] if not isinstance(v[1], tuple) else \
                    f"s={v[1][0]},k={v[1][1]}"
                par = f"k={par}" if isinstance(v[1], int) else par
                sel.append(f"{v[0]:.3f} ({par})")
            else:
                sel.append(f"{v:.3f}")
        print(f"{m:20s} {sel[0]:15s} {sel[1]}")
