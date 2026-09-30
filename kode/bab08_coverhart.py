"""Bab 8: galat Bayes, galat 1-NN asimtotik, dan batas Cover-Hart
untuk tiga model satu dimensi dan data gelombang, lalu galat 1-NN
hasil simulasi untuk n yang makin besar.
"""
import numpy as np
from bab04_sintetis import bangkitkan as bangkit2, eta as eta2
from bab08_model import MODEL, galat_1nn, galat_asimtotik
from sklearn.neighbors import KNeighborsClassifier

BENIH = 20260929
print("model      R*       R_NN     2R*(1-R*)")
for nama, eta in MODEL.items():
    rb, rnn = galat_asimtotik(eta)
    print(f"{nama:9s}  {rb:.4f}   {rnn:.4f}   {2 * rb * (1 - rb):.4f}")
t = (np.arange(2001) + 0.5) / 2001
G = np.column_stack([np.repeat(t, len(t)), np.tile(t, len(t))])
e2 = eta2(G)
rb, rnn = np.mean(np.minimum(e2, 1 - e2)), np.mean(2 * e2 * (1 - e2))
print(f"{'gelombang':9s}  {rb:.4f}   {rnn:.4f}   {2 * rb * (1 - rb):.4f}")

print("galat 1-NN (rata-rata 200 data latih; gelombang 20):")
print("     n    linear   tetap    separuh  gelombang")
rng = np.random.default_rng(BENIH)
U = np.random.default_rng(BENIH + 1).random((20_000, 2))
eU = eta2(U)
for n in (10, 100, 1000, 10000):
    baris = [np.mean([galat_1nn(eta, n, rng) for _ in range(200)])
             for eta in MODEL.values()]
    g = []
    for _ in range(20):
        X, y = bangkit2(n, rng)
        yh = KNeighborsClassifier(1).fit(X, y).predict(U)
        g.append(np.mean(np.where(yh == 1, 1 - eU, eU)))
    baris.append(np.mean(g))
    print(f"{n:6d}   " + "   ".join(f"{v:.4f}" for v in baris))
