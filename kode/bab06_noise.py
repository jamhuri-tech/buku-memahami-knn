"""Bab 6: fitur noise menutupi sinyal.

(1) Data anggur ditambah m fitur N(0, 1), lalu dibakukan di dalam
    pipeline; akurasi dirata-ratakan atas 10 matriks noise.
(2) Simulasi peluang dua calon tetangga bertukar urutan: selisih jarak
    kuadrat sinyal Delta, lalu m fitur noise N(0, 1) ditambahkan ke
    titik uji dan kedua calon. Teori: Phi(-Delta / sqrt(12 m)).
"""
import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BENIH = 20260929
ULANG = 10


def noise(r, n, m):
    """Matriks noise ke-r (r = 0, ..., 9), n x m, benih tetap."""
    return np.random.default_rng(BENIH + 100 + r).standard_normal((n, m))


if __name__ == "__main__":
    X, y = load_wine(return_X_y=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
    print("m fitur noise   akurasi 5-NN (rata-rata 10 ulangan)")
    for m in (0, 5, 13, 26, 52, 104):
        a = [cross_val_score(model, np.hstack([X, noise(r, len(X), m)]),
                             y, cv=cv).mean() for r in range(ULANG)]
        print(f"{m:4d}            {np.mean(a):.4f}")

    print("peluang bertukar urutan, Delta = 6:")
    print("  m    simulasi   teori Phi(-Delta/sqrt(12 m))")
    rng = np.random.default_rng(BENIH)
    delta, n = 6.0, 200_000
    for m in (3, 12, 48, 192):
        ex = rng.standard_normal((n, m))
        ea = rng.standard_normal((n, m))
        eb = rng.standard_normal((n, m))
        # calon a lebih dekat secara sinyal: D_s^2(b) = D_s^2(a) + Delta
        da = ((ex - ea) ** 2).sum(1)
        db = delta + ((ex - eb) ** 2).sum(1)
        teori = norm.cdf(-delta / np.sqrt(12 * m))
        print(f"{m:4d}   {np.mean(da > db):.4f}     {teori:.4f}")
