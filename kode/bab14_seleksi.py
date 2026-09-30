"""Bab 14: banyaknya perbandingan quickselect (Hoare 1961) untuk
mencari unsur terkecil ke-k dari n bilangan, dibandingkan rumus harapan
Knuth: C(n, k) = 2[(n+1) H_n - (n+3-k) H_{n+1-k} - (k+2) H_k + n + 3],
dan dengan batas umum 4n. Poros dipilih acak; n = 2000.
"""
import numpy as np

BENIH = 20260929


def quickselect(a, k, rng):
    """Unsur terkecil ke-k (k mulai dari 1) dan banyaknya perbandingan.
    Setiap langkah membandingkan semua unsur (selain poros) dengan
    poros, seperti analisis Knuth."""
    banding = 0
    while True:
        if len(a) == 1:
            return a[0], banding
        poros = a[rng.integers(len(a))]
        kiri = a[a < poros]
        kanan = a[a > poros]
        banding += len(a) - 1
        if k <= len(kiri):
            a = kiri
        elif k == len(kiri) + 1:
            return poros, banding
        else:
            k -= len(kiri) + 1
            a = kanan


def H(m):
    return np.sum(1.0 / np.arange(1, m + 1))


def knuth(n, k):
    return 2 * ((n + 1) * H(n) - (n + 3 - k) * H(n + 1 - k)
                - (k + 2) * H(k) + n + 3)


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    n = 2000
    print("   k    simulasi/n   rumus Knuth/n   batas 4n/n")
    for k in (1, 10, 100, 500, 1000):
        c = []
        for _ in range(3000):
            a = rng.permutation(n).astype(float)
            v, b = quickselect(a, k, rng)
            assert v == k - 1
            c.append(b)
        print(f"{k:5d}     {np.mean(c) / n:6.3f}        "
              f"{knuth(n, k) / n:6.3f}          4")
    print(f"median, n besar: 2(1 + ln 2) = {2 * (1 + np.log(2)):.3f}")
