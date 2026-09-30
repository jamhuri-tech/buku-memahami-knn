"""Bab 10: pemusatan jarak. X, Z seragam di [0, 1]^p:
E ||X - Z||^2 = p/6, Var ||X - Z||^2 = 7p/180, sehingga simpangan baku
relatifnya sqrt(1,4 / p). Juga kontras relatif (max - min)/min jarak
dari satu titik kueri ke 1000 titik.
"""
import numpy as np

BENIH = 20260929
rng = np.random.default_rng(BENIH)
print("        E D^2              sb relatif D^2    kontras")
print("   p    p/6     simulasi   teori   simulasi  (max-min)/min")
for p in (2, 10, 100, 1000):
    X = rng.random((20_000, p))
    Z = rng.random((20_000, p))
    D2 = ((X - Z) ** 2).sum(1)
    kontras = []
    for _ in range(50):
        A = rng.random((1000, p))
        q = rng.random(p)
        d = np.sqrt(((A - q) ** 2).sum(1))
        kontras.append((d.max() - d.min()) / d.min())
    print(f"{p:4d} {p / 6:8.3f}  {D2.mean():8.3f}   "
          f"{np.sqrt(1.4 / p):.4f}  {D2.std() / D2.mean():.4f}"
          f"   {np.median(kontras):8.3f}")
