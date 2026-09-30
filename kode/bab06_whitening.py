"""Bab 6: pembakuan lawan whitening (Mahalanobis dengan kovarians
seluruh data) pada data anggur dan data anggur + 26 fitur noise.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bab06_noise import ULANG, noise

BENIH = 20260929


class Whitening(BaseEstimator, TransformerMixin):
    """x -> Sigma^{-1/2} (x - rata-rata), Sigma dari data latih."""

    def fit(self, X, y=None):
        self.mu_ = X.mean(0)
        lam, V = np.linalg.eigh(np.cov(X, rowvar=False))
        self.W_ = V @ np.diag(lam ** -0.5) @ V.T     # simetris (ZCA)
        return self

    def transform(self, X):
        return (X - self.mu_) @ self.W_


if __name__ == "__main__":
    X, y = load_wine(return_X_y=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)


    def akurasi(D, t):
        m = make_pipeline(t, KNeighborsClassifier(5))
        return cross_val_score(m, D, y, cv=cv).mean()


    print("data              pembakuan  whitening")
    a = [akurasi(X, t) for t in (StandardScaler(), Whitening())]
    print(f"{'anggur':17s} {a[0]:.4f}     {a[1]:.4f}")
    # Sama dengan baris m = 26 di bab06_noise.py (10 matriks noise).
    a = np.mean([[akurasi(np.hstack([X, noise(r, len(X), 26)]), t)
                  for t in (StandardScaler(), Whitening())]
                 for r in range(ULANG)], axis=0)
    print(f"{'anggur + 26 noise':17s} {a[0]:.4f}     {a[1]:.4f}")
    lam = np.linalg.eigvalsh(np.cov(StandardScaler().fit_transform(X),
                                    rowvar=False))
    print(f"nilai eigen (dibakukan): terbesar {lam.max():.3f}, "
          f"terkecil {lam.min():.4f}")
