"""Bab 5: jarak Malang-Surabaya dengan haversine, hampiran
equirectangular, dan haversine_distances scikit-learn.
"""
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

R = 6371.0                                   # jari-jari Bumi, km
malang = np.array([-7.98, 112.63])           # lintang, bujur (derajat)
surabaya = np.array([-7.25, 112.75])
f1, l1 = np.radians(malang)
f2, l2 = np.radians(surabaya)
hav = (np.sin((f2 - f1) / 2) ** 2
       + np.cos(f1) * np.cos(f2) * np.sin((l2 - l1) / 2) ** 2)
d_hav = 2 * R * np.arcsin(np.sqrt(hav))
fm = (f1 + f2) / 2
dx = R * (l2 - l1) * np.cos(fm)
dy = R * (f2 - f1)
d_eq = np.hypot(dx, dy)
d_sk = R * haversine_distances(np.radians([malang, surabaya]))[0, 1]
print(f"haversine          : {d_hav:.3f} km")
print(f"equirectangular    : {d_eq:.3f} km (dx {dx:.2f}, dy {dy:.2f})")
print(f"scikit-learn       : {d_sk:.3f} km")
print(f"derajat sebagai Euclid: {np.hypot(*(surabaya - malang)):.4f}"
      f" derajat")
