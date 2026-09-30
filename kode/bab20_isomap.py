"""Bab 20: graf tetangga untuk jarak di sepanjang manifold. Pada data
gulungan (make_swiss_roll), korelasi (mutlak, Spearman) antara posisi
sebenarnya di sepanjang gulungan dan komponen pertama PCA, dibandingkan
dengan komponen pertama Isomap (jarak terpendek di graf 10 tetangga).
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

BENIH = 20260929

if __name__ == "__main__":
    X, t = make_swiss_roll(1500, random_state=BENIH)
    z_pca = PCA(2).fit_transform(X)[:, 0]
    z_iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X)[:, 0]
    print(f"korelasi dengan posisi di gulungan: PCA "
          f"{abs(spearmanr(t, z_pca)[0]):.4f}, Isomap "
          f"{abs(spearmanr(t, z_iso)[0]):.4f}")
