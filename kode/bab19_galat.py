"""Bab 19: analisis galat model k-NN (x_km, y_km, 20 MedInc), k = 10,
dengan ramalan validasi silang per wilayah. Galat dikelompokkan menurut
jarak ke tetangga ke-10 (dalam ruang fitur model) dan menurut label
yang terpotong.
"""
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from bab19_data import muat
from bab19_evaluasi import km_pendapatan


def ramalan_wilayah(s=20, k=10):
    X, y, km, wil, _ = muat()
    Z = km_pendapatan(X, km, s)
    yh = np.empty_like(y)
    dk = np.empty_like(y)
    for lt, uji in GroupKFold(5).split(Z, y, wil):
        m = KNeighborsRegressor(k).fit(Z[lt], y[lt])
        yh[uji] = m.predict(Z[uji])
        dk[uji] = m.kneighbors(Z[uji])[0][:, -1]
    return X, y, yh, dk


if __name__ == "__main__":
    X, y, yh, dk = ramalan_wilayah()
    r = y - yh
    print(f"RMSE {np.sqrt(np.mean(r**2)):.3f}, "
          f"R^2 {1 - np.mean(r**2) / y.var():.3f}")
    q = np.quantile(dk, [0, 0.25, 0.5, 0.75, 1])
    print("jarak ke tetangga ke-10      titik    RMSE")
    for a, b in zip(q[:-1], q[1:]):
        m = (dk >= a) & (dk <= b)
        print(f"{a:7.1f} - {b:7.1f}             {m.sum():5d}    "
              f"{np.sqrt(np.mean(r[m]**2)):.3f}")
    c = y >= 5
    print(f"label terpotong (5.00001): {c.sum()} titik, "
          f"RMSE {np.sqrt(np.mean(r[c]**2)):.3f}")
    print(f"   rata-rata ramalan untuk titik itu {yh[c].mean():.2f}")
    print(f"label biasa: RMSE {np.sqrt(np.mean(r[~c]**2)):.3f}")
