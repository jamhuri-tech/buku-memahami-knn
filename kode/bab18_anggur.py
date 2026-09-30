"""Bab 18: NCA pada data anggur, tanpa dan dengan 26 fitur noise (matriks
noise ke-0 dari Bab 6). Validasi silang 5 lipatan, 5-NN sesudah
pembakuan dan salah satu transformasi. Juga bagian bobot metrik yang
jatuh pada 13 fitur asli: jumlah kuadrat kolom L untuk fitur asli
dibagi jumlah kuadrat seluruh kolom (13/39 = 0,333 bila merata).
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bab06_noise import noise
from bab06_whitening import Whitening
from bab18_nca import latih_nca

BENIH = 20260929


class NCAKita(BaseEstimator, TransformerMixin):
    def __init__(self, d=None):
        self.d = d

    def fit(self, X, y):
        self.L_, self.f_ = latih_nca(X, y, d=self.d)
        return self

    def transform(self, X):
        return X @ self.L_.T


def bagian_asli(L, p_asli=13):
    n = (L ** 2).sum(0)
    return n[:p_asli].sum() / n.sum()


if __name__ == "__main__":
    X, y = load_wine(return_X_y=True)
    Xn = np.hstack([X, noise(0, len(X), 26)])
    cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
    cara = [("pembakuan saja", None), ("whitening", Whitening()),
            ("NCA penuh (39 x 39)", NCAKita()),
            ("NCA ke 2 dimensi", NCAKita(2)),
            ("NCA ke 5 dimensi", NCAKita(5)),
            ("sklearn NCA, 5 dimensi",
             NeighborhoodComponentsAnalysis(5, random_state=BENIH))]
    print("transformasi              anggur   anggur + 26 noise")
    for nama, t in cara:
        a = []
        for D in (X, Xn):
            langkah = [StandardScaler()] + ([t] if t is not None else [])
            m = make_pipeline(*langkah, KNeighborsClassifier(5))
            a.append(cross_val_score(m, D, y, cv=cv).mean())
        print(f"{nama:24s}  {a[0]:.4f}   {a[1]:.4f}")
    D = StandardScaler().fit_transform(Xn)
    print("bagian bobot pada 13 fitur asli (anggur + 26 noise):")
    for d in (None, 5, 2):
        L, f = latih_nca(D, y, d=d)
        nm = "penuh" if d is None else f"{d} dimensi"
        print(f"  NCA {nm:10s}: {bagian_asli(L):.3f}  (f(L) = {f:.1f})")
