# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Benih acak selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENIH = 20260929  # sama dengan seluruh kode/bab*.py
GBR = Path("gbr")

# Sebagian gambar memakai kelas yang sudah ditulis di kode/, supaya
# logikanya tidak terduplikasi di dua tempat.
sys.path.insert(0, str(Path(__file__).parent / "kode"))

# Warna mengikuti preamble.tex.
BIRU = "#1B3B6F"
HIJAU = "#1E6F5C"
JINGGA = "#B85C00"
MERAH = "#9B1B30"
ABU = "#5A6472"
ABU_GARIS = "#C9CED6"
BIRU_MUDA = "#E8EEF7"
HIJAU_MUDA = "#E6F2EF"
JINGGA_MUDA = "#FDF0E3"
MERAH_MUDA = "#FBE9EC"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": ABU_GARIS,
    "axes.labelcolor": ABU,
    "axes.titlesize": 8,
    "axes.titlecolor": BIRU,
    "xtick.color": ABU,
    "ytick.color": ABU,
    "text.color": ABU,
    "grid.color": ABU_GARIS,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def angka(v, n=3):
    """Angka dengan koma desimal, sesuai kaidah bahasa Indonesia."""
    return f"{v:.{n}f}".replace(".", ",")


def angka_mat(v, n=3):
    """Seperti angka(), untuk mode matematika: koma tanpa spasi."""
    return angka(v, n).replace(",", "{,}")


def _koma(fig):
    """Mengubah pemisah desimal pada label sumbu menjadi koma.

    Sumbu berskala logaritmik dilewati, karena labelnya berupa pangkat
    sepuluh dan tidak memuat pemisah desimal.
    """
    from matplotlib.ticker import FuncFormatter
    rapi = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        kunci = getattr(ax, "_label_terkunci", set())
        if "x" not in kunci and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(rapi)
        if "y" not in kunci and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(rapi)


def kunci_label(ax, *sumbu):
    """Menandai sumbu yang labelnya kita tetapkan sendiri."""
    ax._label_terkunci = getattr(ax, "_label_terkunci",
                                 set()) | set(sumbu)


def simpan(fig, nama):
    _koma(fig)
    GBR.mkdir(exist_ok=True)
    fig.savefig(GBR / f"{nama}.pdf", bbox_inches="tight")
    fig.savefig(GBR / f"{nama}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("gbr/" + nama)


def _rapikan(ax):
    """Gaya sumbu seri: tanpa bingkai atas dan kanan."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------
#  Gambar per bab ditambahkan di bawah ini sebagai fungsi babNN_nama().



# ============================ Bab 4 ==================================

def bab04_voronoi():
    """Sel Voronoi delapan titik rekaan, diwarnai menurut kelas."""
    from scipy.spatial import Voronoi
    from bab04_data import X, y
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    t1 = np.linspace(0, 9, 721)
    t2 = np.linspace(0, 7, 561)
    G1, G2 = np.meshgrid(t1, t2)
    G = np.column_stack([G1.ravel(), G2.ravel()])
    d2 = ((G[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    lab = y[d2.argmin(1)].reshape(G1.shape)
    from matplotlib.colors import ListedColormap
    ax.contourf(G1, G2, lab, levels=[-0.5, 0.5, 1.5],
                cmap=ListedColormap([BIRU_MUDA, JINGGA_MUDA]))
    vor = Voronoi(np.vstack([X, [[-20, -20], [30, -20], [-20, 30],
                                  [30, 30]]]))
    for (i, j), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if i >= 8 or j >= 8 or -1 in rv:
            continue
        a, b = vor.vertices[rv]
        beda = y[i] != y[j]
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=MERAH if beda else ABU_GARIS,
                lw=1.4 if beda else 0.7, zorder=2)
    for c, w, m in ((0, BIRU, "o"), (1, JINGGA, "s")):
        ax.scatter(*X[y == c].T, color=w, marker=m, s=26, zorder=3,
                   label="kelas A" if c == 0 else "kelas B")
    for xi in X:
        ax.annotate(f"({xi[0]},{xi[1]})", xi, xytext=(3, 3),
                    textcoords="offset points", fontsize=5.8, color=ABU)
    ax.scatter([4], [3], marker="*", s=90, color=HIJAU, zorder=4)
    ax.annotate("(4,3)", (4, 3), xytext=(-18, 5),
                textcoords="offset points", fontsize=6.3, color=HIJAU)
    ax.add_patch(plt.Circle((4, 3), np.sqrt(5), fill=False, ls="--",
                            color=HIJAU, lw=0.8))
    ax.scatter([4], [0], marker="D", s=26, color=HIJAU, zorder=4)
    ax.annotate("(4,0)", (4, 0), xytext=(4, 3),
                textcoords="offset points", fontsize=6.3, color=HIJAU)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.3, 7)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=6.3)
    _rapikan(ax)
    simpan(fig, "bab04-voronoi")


def bab04_batas():
    """Batas keputusan k-NN untuk k = 1, 15, 151 pada data gelombang."""
    from sklearn.neighbors import KNeighborsClassifier
    from bab04_sintetis import bangkitkan, g
    from matplotlib.colors import ListedColormap
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(400, rng)
    t = np.linspace(0, 1, 301)
    G1, G2 = np.meshgrid(t, t)
    G = np.column_stack([G1.ravel(), G2.ravel()])
    fig, axs = plt.subplots(1, 3, figsize=(6.2, 2.25), sharey=True)
    for ax, k in zip(axs, (1, 15, 151)):
        m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        z = m.predict(G).reshape(G1.shape)
        ax.contourf(G1, G2, z, levels=[-0.5, 0.5, 1.5],
                    cmap=ListedColormap([BIRU_MUDA, JINGGA_MUDA]))
        ax.scatter(*X[y == 0].T, s=2.5, color=BIRU, lw=0)
        ax.scatter(*X[y == 1].T, s=2.5, color=JINGGA, lw=0)
        ax.plot(t, g(t), color=HIJAU, lw=1.0, ls="--")
        ax.set_title(f"$k = {k}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
    axs[0].set_ylabel("$x_2$")
    simpan(fig, "bab04-batas")


def bab04_digit():
    """Dua citra uji dan lima tetangga terdekatnya di data latih."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
    X, y = load_digits(return_X_y=True)
    X_lt, X_uji, y_lt, y_uji = train_test_split(
        X, y, test_size=450, stratify=y, random_state=BENIH)
    nn = NearestNeighbors(n_neighbors=5).fit(X_lt)
    d, idx = nn.kneighbors(X_uji)
    salah = np.where(y_lt[idx[:, 0]] != y_uji)[0]
    benar = np.where(y_lt[idx[:, :5]].min(1) == y_uji)[0]
    pilih = [benar[0], salah[0]]
    fig, axs = plt.subplots(2, 6, figsize=(5.2, 2.2))
    for r, i in enumerate(pilih):
        axs[r, 0].imshow(X_uji[i].reshape(8, 8), cmap="Greys")
        axs[r, 0].set_title(f"uji: {y_uji[i]}", fontsize=6.5,
                            color=HIJAU)
        for c in range(5):
            j = idx[i, c]
            axs[r, c + 1].imshow(X_lt[j].reshape(8, 8), cmap="Greys")
            w = BIRU if y_lt[j] == y_uji[i] else MERAH
            axs[r, c + 1].set_title(f"{y_lt[j]}  d={angka(d[i, c], 1)}",
                                    fontsize=6, color=w)
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(ABU_GARIS)
    simpan(fig, "bab04-digit")



# ============================ Bab 5 ==================================

def _bola(q, m=721):
    t = np.linspace(0, 2 * np.pi, m)
    c, s = np.cos(t), np.sin(t)
    if np.isinf(q):
        r = 1 / np.maximum(np.abs(c), np.abs(s))
    else:
        r = (np.abs(c) ** q + np.abs(s) ** q) ** (-1 / q)
    return r * c, r * s


def bab05_bola():
    """Kiri: bola satuan L_q. Kanan: bola tiap metrik yang menyentuh
    tetangga terdekat titik (0, 0) di antara a, b, c."""
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.8),
                            gridspec_kw=dict(wspace=0.55))
    ax = axs[0]
    warna = {0.5: MERAH, 1: BIRU, 2: HIJAU, 4: ABU, np.inf: JINGGA}
    for q, w in warna.items():
        x, yy = _bola(q)
        lab = "$q=\\infty$" if np.isinf(q) else f"$q={angka(q, 1) if q < 1 else q}$"
        ax.plot(x, yy, color=w, lw=1.1, label=lab)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axhline(0, color=ABU_GARIS, lw=0.5)
    ax.axvline(0, color=ABU_GARIS, lw=0.5)
    ax.legend(fontsize=6, loc="center left", handlelength=1.2,
              bbox_to_anchor=(1.0, 0.5))
    ax.set_title("bola satuan $\\|\\mathbf{v}\\|_q \\leq 1$")
    _rapikan(ax)
    ax = axs[1]
    P = np.array([[6, 0], [4, 4], [2, 5]])
    for (q, w, r) in ((1, BIRU, 6), (2, HIJAU, np.sqrt(29)),
                      (np.inf, JINGGA, 4)):
        x, yy = _bola(q)
        ax.plot(r * x, r * yy, color=w, lw=1.0,
                label={1: "$L_1$, jari-jari 6", 2: "$L_2$, jari-jari "
                       "$\\sqrt{29}$", np.inf: "$L_\\infty$, jari-jari 4"}[q])
    ax.scatter(*P.T, color=ABU, s=18, zorder=3)
    for nm, pt in zip("abc", P):
        ax.annotate(f"{nm}=({pt[0]},{pt[1]})", pt, xytext=(3, 3),
                    textcoords="offset points", fontsize=6)
    ax.scatter([0], [0], marker="*", color=MERAH, s=40, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 7.5)
    ax.legend(fontsize=5.8, loc="upper right")
    ax.set_title("bola yang menyentuh tetangga pertama")
    _rapikan(ax)
    simpan(fig, "bab05-bola")


def bab05_voronoi():
    """Wilayah keputusan 1-NN delapan titik rekaan untuk tiga metrik."""
    from bab04_data import X, y
    from matplotlib.colors import ListedColormap
    t1 = np.linspace(0, 9, 541)
    t2 = np.linspace(0, 7, 421)
    G1, G2 = np.meshgrid(t1, t2)
    G = np.column_stack([G1.ravel(), G2.ravel()])
    D = G[:, None, :] - X[None, :, :]
    fig, axs = plt.subplots(1, 3, figsize=(6.2, 1.95), sharey=True)
    for ax, (nama, d) in zip(axs, (("$L_1$", np.abs(D).sum(-1)),
                                   ("$L_2$", (D ** 2).sum(-1)),
                                   ("$L_\\infty$", np.abs(D).max(-1)))):
        idx = d.argmin(1).reshape(G1.shape)
        lab = y[idx]
        ax.contourf(G1, G2, lab, levels=[-0.5, 0.5, 1.5],
                    cmap=ListedColormap([BIRU_MUDA, JINGGA_MUDA]))
        ax.contour(G1, G2, lab, levels=[0.5], colors=MERAH,
                   linewidths=1.1)
        ax.scatter(*X[y == 0].T, color=BIRU, s=10, zorder=3)
        ax.scatter(*X[y == 1].T, color=JINGGA, marker="s", s=10,
                   zorder=3)
        ax.scatter([4], [3], marker="*", color=HIJAU, s=40, zorder=4)
        ax.set_title(nama)
        ax.set_aspect("equal")
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 7)
        ax.set_xlabel("$x_1$")
    axs[0].set_ylabel("$x_2$")
    simpan(fig, "bab05-voronoi")


def bab05_mahalanobis():
    """Awan titik berkorelasi, lingkaran Euclid dan elips Mahalanobis."""
    rng = np.random.default_rng(BENIH)
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    Z = rng.multivariate_normal([0, 0], S, size=400)
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.7))
    t = np.linspace(0, 2 * np.pi, 400)
    lam, V = np.linalg.eigh(S)
    for ax, judul in zip(axs, ("jarak Euclid", "jarak Mahalanobis")):
        ax.scatter(*Z.T, s=2, color=ABU_GARIS)
        for r in (1, 2):
            if judul == "jarak Euclid":
                e = r * np.column_stack([np.cos(t), np.sin(t)])
            else:
                e = (r * np.column_stack([np.cos(t), np.sin(t)])
                     * np.sqrt(lam)) @ V.T
            ax.plot(*e.T, color=BIRU if r == 1 else HIJAU, lw=1.0)
        for v, w in (([1, 1], JINGGA), ([1, -1], MERAH)):
            ax.annotate("", xy=v, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=w, lw=1.2))
        ax.text(1.05, 1.15, "(1,1)", color=JINGGA, fontsize=6.5)
        ax.text(1.05, -1.35, "(1,$-$1)", color=MERAH, fontsize=6.5)
        ax.set_aspect("equal")
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-4.2, 4.2)
        ax.set_title(judul)
        _rapikan(ax)
    simpan(fig, "bab05-mahalanobis")



# ============================ Bab 6 ==================================

def bab06_dominasi():
    """Data anggur, flavanoid dan proline: satuan asli (sumbu sama
    skala) dan sesudah pembakuan."""
    from sklearn.datasets import load_wine
    d = load_wine()
    X, y = d.data, d.target
    j1 = list(d.feature_names).index("flavanoids")
    j2 = list(d.feature_names).index("proline")
    A = X[:, [j1, j2]]
    Z = (A - A.mean(0)) / A.std(0)
    warna = [BIRU, JINGGA, HIJAU]
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.9),
                            gridspec_kw=dict(width_ratios=[1, 1.6]))
    for c in range(3):
        axs[0].scatter(*A[y == c].T, s=4, color=warna[c])
        axs[1].scatter(*Z[y == c].T, s=4, color=warna[c],
                       label=f"kelas {c}")
    axs[0].set_xlim(-700, 700)
    axs[0].set_ylim(200, 1750)
    axs[0].set_aspect("equal")
    axs[0].set_xlabel("flavanoid (satuan asli)")
    axs[0].set_ylabel("proline (satuan asli)")
    axs[0].set_title("skala sama: jarak = selisih proline")
    axs[1].set_aspect("equal")
    axs[1].set_xlabel("flavanoid (dibakukan)")
    axs[1].set_ylabel("proline (dibakukan)")
    axs[1].set_title("sesudah pembakuan")
    axs[1].legend(fontsize=6, loc="upper left")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-dominasi")


def bab06_skala():
    """Lima nilai dengan satu pencilan di bawah empat penskalaan."""
    v = np.array([2.0, 4.0, 6.0, 8.0, 30.0])
    baris = [("asli", v),
             ("min-max", (v - v.min()) / (v.max() - v.min())),
             ("pembakuan (z)", (v - v.mean()) / v.std()),
             ("robust", (v - np.median(v)) / 4.0)]
    fig, axs = plt.subplots(4, 1, figsize=(4.8, 2.6))
    for ax, (nama, w) in zip(axs, baris):
        ax.axhline(0, color=ABU_GARIS, lw=0.8)
        ax.scatter(w[:4], np.zeros(4), color=BIRU, s=16, zorder=3)
        ax.scatter(w[4:], np.zeros(1), color=MERAH, s=16, zorder=3)
        for t in w:
            ax.annotate(angka(t, 2).rstrip("0").rstrip(","), (t, 0),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=5.5)
        ax.set_yticks([])
        ax.set_ylabel(nama, rotation=0, ha="right", va="center",
                      fontsize=6.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6)
        pad = 0.06 * (w.max() - w.min())
        ax.set_xlim(w.min() - pad, w.max() + pad)
    fig.tight_layout(h_pad=0.4)
    simpan(fig, "bab06-skala")


def bab06_noise():
    """Kiri: akurasi 5-NN anggur terhadap banyaknya fitur noise. Kanan:
    peluang dua calon tetangga bertukar urutan, simulasi dan teori."""
    from scipy.stats import norm
    from sklearn.datasets import load_wine
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from bab06_noise import ULANG, noise
    X, y = load_wine(return_X_y=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=BENIH)
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
    ms = [0, 5, 13, 26, 39, 52, 78, 104, 156]
    aku = [np.mean([cross_val_score(
        model, np.hstack([X, noise(r, len(X), m)]), y, cv=cv).mean()
        for r in range(ULANG)]) for m in ms]
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.3))
    axs[0].plot(ms, aku, "o-", color=BIRU, ms=3)
    axs[0].set_xlabel("banyaknya fitur noise $m$")
    axs[0].set_ylabel("akurasi 5-NN")
    axs[0].set_title("anggur (13 fitur) + noise")
    rng = np.random.default_rng(BENIH)
    mm = np.array([1, 2, 3, 6, 12, 24, 48, 96, 192])
    sim = []
    for m in mm:
        ex, ea, eb = (rng.standard_normal((40_000, m)) for _ in range(3))
        sim.append(np.mean(((ex - ea) ** 2).sum(1)
                           > 6 + ((ex - eb) ** 2).sum(1)))
    t = np.geomspace(1, 200, 200)
    axs[1].plot(t, norm.cdf(-6 / np.sqrt(12 * t)), color=HIJAU,
                label="teori $\\Phi(-\\Delta/\\sqrt{12m})$")
    axs[1].plot(mm, sim, "o", color=JINGGA, ms=3, label="simulasi")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("banyaknya fitur noise $m$")
    axs[1].set_ylabel("peluang bertukar urutan")
    axs[1].set_title("dua calon, selisih sinyal $\\Delta = 6$")
    axs[1].legend(fontsize=6)
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-noise")



# ============================ Bab 7 ==================================

def bab07_biasvarians():
    """Kiri: regresi k-NN pada satu data untuk tiga k. Kanan: bias^2,
    varians, dan galat harapan terhadap k (rancangan tetap)."""
    n, sigma = 100, 0.3
    x = (np.arange(1, n + 1) - 0.5) / n
    f = lambda t: np.sin(2 * np.pi * t)
    rng = np.random.default_rng(BENIH)
    y = f(x) + sigma * rng.standard_normal(n)
    t = np.linspace(0, 1, 1001)
    urut_t = np.argsort(np.abs(t[:, None] - x[None, :]), axis=1,
                        kind="stable")
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.4))
    ax = axs[0]
    ax.scatter(x, y, s=3, color=ABU_GARIS)
    ax.plot(t, f(t), color=ABU, lw=0.8, ls="--", label="$f$")
    for k, w in ((1, MERAH), (10, BIRU), (80, JINGGA)):
        ax.plot(t, y[urut_t[:, :k]].mean(1), color=w, lw=0.9,
                label=f"$k={k}$", drawstyle="steps-mid")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(fontsize=5.8, ncol=2, loc="lower left")
    _rapikan(ax)
    x0 = (np.arange(500) + 0.5) / 500
    urut = np.argsort(np.abs(x0[:, None] - x[None, :]), axis=1,
                      kind="stable")
    ks = np.arange(1, 91)
    b2 = np.array([np.mean((f(x0) - f(x[urut[:, :k]]).mean(1)) ** 2)
                   for k in ks])
    v = sigma ** 2 / ks
    ax = axs[1]
    ax.plot(ks, b2, color=JINGGA, label="bias$^2$")
    ax.plot(ks, v, color=BIRU, label="varians $\\sigma^2/k$")
    ax.plot(ks, b2 + v, color=HIJAU, lw=1.4, label="jumlah")
    kb = ks[np.argmin(b2 + v)]
    ax.axvline(kb, color=ABU_GARIS, ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_title(f"terkecil di $k = {kb}$")
    ax.legend(fontsize=5.8)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-biasvarians")


def bab07_titik():
    """Galat k-NN di satu titik tanpa bias, terhadap k."""
    from math import comb
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ks = np.arange(1, 102, 2)
    for eta, w in ((0.6, MERAH), (0.7, BIRU), (0.9, HIJAU)):
        q = [sum(comb(k, j) * eta ** j * (1 - eta) ** (k - j)
                 for j in range((k - 1) // 2 + 1)) for k in ks]
        g = min(eta, 1 - eta) + abs(2 * eta - 1) * np.array(q)
        ax.plot(ks, g, "o-", ms=1.8, lw=0.9, color=w,
                label=f"$\\eta = {angka_mat(eta, 1)}$")
        ax.axhline(min(eta, 1 - eta), color=w, lw=0.6, ls=":")
    ax.set_xlabel("$k$ (ganjil)")
    ax.set_ylabel("peluang salah di titik itu")
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1, 0.93))
    _rapikan(ax)
    simpan(fig, "bab07-titik")


def bab07_validasi():
    """Galat LOO, 10 lipatan, dan galat sebenarnya terhadap k."""
    from bab04_sintetis import bangkitkan
    from bab07_loo import kurva
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(400, rng)
    loo, g10, benar = kurva(X, y)
    ks = np.arange(1, len(loo) + 1)
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.plot(ks, loo, color=BIRU, lw=0.8, label="LOO")
    ax.plot(ks, g10, color=JINGGA, lw=0.8, label="10 lipatan")
    ax.plot(ks, benar, color=HIJAU, lw=1.4, label="galat sebenarnya")
    ax.axhline(0.1356, color=ABU, ls=":", lw=0.8)
    ax.text(150, 0.1375, "galat Bayes", ha="right", fontsize=6,
            color=ABU)
    ax.set_xscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("galat")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab07-validasi")



# ============================ Bab 8 ==================================

def bab08_titik():
    """Kiri: galat bersyarat di satu titik terhadap eta. Kanan: jarak
    tetangga terdekat terhadap n untuk p = 1, 2, 10."""
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.4))
    e = np.linspace(0, 1, 401)
    r = np.minimum(e, 1 - e)
    ax = axs[0]
    ax.plot(e, r, color=BIRU, label="Bayes")
    ax.plot(e, 2 * e * (1 - e), color=JINGGA, label="1-NN")
    ax.plot(e, 2 * r, color=ABU, ls=":", lw=0.9, label="2 $\\times$ Bayes")
    ax.set_xlabel("$\\eta(\\mathbf{x})$")
    ax.set_ylabel("peluang salah di $\\mathbf{x}$")
    ax.legend(fontsize=5.8, loc="upper left")
    _rapikan(ax)
    rng = np.random.default_rng(BENIH)
    ns = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    ax = axs[1]
    for p, w in ((1, BIRU), (2, HIJAU), (10, MERAH)):
        d = [np.mean([np.sqrt(((rng.random((n, p)) - 0.5) ** 2)
                              .sum(1)).min() for _ in range(300)])
             for n in ns]
        ax.plot(ns, d, "o-", ms=2.5, color=w, label=f"$p = {p}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel("rata-rata jarak ke tetangga pertama")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-titik")


def bab08_batas():
    """Bidang (R*, R_NN): daerah yang diizinkan Cover-Hart dan empat
    model."""
    fig, ax = plt.subplots(figsize=(3.9, 3.0))
    r = np.linspace(0, 0.5, 201)
    ax.fill_between(r, r, 2 * r * (1 - r), color=HIJAU_MUDA)
    ax.plot(r, r, color=BIRU, lw=1, label="$R_{\\mathrm{NN}} = R^*$")
    ax.plot(r, 2 * r * (1 - r), color=JINGGA, lw=1,
            label="$R_{\\mathrm{NN}} = 2R^*(1-R^*)$")
    ax.plot(r, 2 * r, color=ABU, lw=0.7, ls=":", label="$2R^*$")
    titik = {"linear": (0.25, 1 / 3), "tetap": (0.1, 0.18),
             "separuh": (0.25, 0.25), "gelombang": (0.1356, 0.1941)}
    geser = {"linear": (4, 3), "tetap": (-20, 4), "separuh": (4, -8),
             "gelombang": (5, -7)}
    for nama, (a, b) in titik.items():
        ax.plot(a, b, "o", color=MERAH, ms=3.5)
        ax.annotate(nama, (a, b), xytext=geser[nama],
                    textcoords="offset points", fontsize=6.3)
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("galat Bayes $R^*$")
    ax.set_ylabel("galat 1-NN asimtotik $R_{\\mathrm{NN}}$")
    ax.legend(fontsize=6, loc="upper left")
    _rapikan(ax)
    simpan(fig, "bab08-batas")


def bab08_konvergen():
    """Galat 1-NN terhadap n untuk model linear dan gelombang."""
    from bab08_model import MODEL, galat_1nn
    from bab04_sintetis import bangkitkan, eta
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(BENIH)
    ns = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    lin = [np.mean([galat_1nn(MODEL["linear"], n, rng)
                    for _ in range(200)]) for n in ns]
    U = np.random.default_rng(BENIH + 1).random((20_000, 2))
    eU = eta(U)
    gel = []
    for n in ns:
        g = []
        for _ in range(20):
            X, y = bangkitkan(n, rng)
            yh = KNeighborsClassifier(1).fit(X, y).predict(U)
            g.append(np.mean(np.where(yh == 1, 1 - eU, eU)))
        gel.append(np.mean(g))
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.3), sharex=True)
    for ax, g, (rb, rnn), judul in (
            (axs[0], lin, (0.25, 1 / 3), "model linear, $p = 1$"),
            (axs[1], gel, (0.1356, 0.1941), "gelombang, $p = 2$")):
        ax.plot(ns, g, "o-", ms=2.5, color=BIRU, label="1-NN")
        ax.axhline(rnn, color=JINGGA, ls="--", lw=0.9,
                   label="$R_{\\mathrm{NN}}$")
        ax.axhline(rb, color=HIJAU, ls=":", lw=0.9, label="$R^*$")
        ax.axhline(2 * rb * (1 - rb), color=ABU, ls="-.", lw=0.8,
                   label="$2R^*(1-R^*)$")
        ax.set_xscale("log")
        ax.set_xlabel("$n$")
        ax.set_title(judul)
        _rapikan(ax)
    axs[0].set_ylabel("galat")
    axs[1].legend(fontsize=5.8, loc="upper right")
    fig.tight_layout()
    simpan(fig, "bab08-konvergen")



# ============================ Bab 9 ==================================

def bab09_rk():
    """Galat asimtotik k-NN (k tetap) pada data gelombang dan batas
    R* + 1/sqrt(k e)."""
    from scipy.stats import binom
    from bab04_sintetis import eta
    t = (np.arange(801) + 0.5) / 801
    G = np.column_stack([np.repeat(t, len(t)), np.tile(t, len(t))])
    e = eta(G)
    r = np.minimum(e, 1 - e)
    rb = r.mean()
    ks = np.arange(1, 102, 2)
    rk = [np.mean(r + (1 - 2 * r) * binom.sf(k // 2, k, r)) for k in ks]
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ax.plot(ks, rk, "o-", ms=2, lw=0.9, color=BIRU,
            label="$R_k$ (data tak hingga)")
    ax.plot(ks, rb + 1 / np.sqrt(ks * np.e), color=JINGGA, lw=0.9,
            label="$R^* + 1/\\sqrt{ke}$")
    ax.axhline(rb, color=HIJAU, ls=":", lw=0.9, label="$R^*$")
    ax.set_ylim(0.12, 0.3)
    ax.set_xlabel("$k$ (ganjil)")
    ax.set_ylabel("galat")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab09-rk")


def bab09_konsisten():
    """Galat k-NN terhadap n untuk lima aturan memilih k."""
    from bab09_konsisten import ATURAN, galat_aturan
    ns = (100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600)
    h = [galat_aturan(n) for n in ns]
    warna = {"k=1": MERAH, "k=15": JINGGA, "sqrt(n)": BIRU,
             "n^(2/3)": HIJAU, "n/5": ABU}
    label = {"k=1": "$k = 1$", "k=15": "$k = 15$",
             "sqrt(n)": "$k = \\sqrt{n}$", "n^(2/3)": "$k = n^{2/3}$",
             "n/5": "$k = n/5$"}
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for nm in ATURAN:
        ax.plot(ns, [x[nm] for x in h], "o-", ms=2.2, lw=0.9,
                color=warna[nm], label=label[nm])
    ax.axhline(0.1356, color=ABU, ls=":", lw=0.9)
    ax.text(100, 0.1370, "$R^* = 0{,}1356$", ha="left", fontsize=6.3,
            color=ABU)
    ax.set_xscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel("galat")
    ax.legend(fontsize=6, ncol=2)
    _rapikan(ax)
    simpan(fig, "bab09-konsisten")


def bab09_laju():
    """Galat kuadrat regresi k-NN (k terbaik) terhadap n, p = 1, 2, 4."""
    from bab09_laju import NS, kurva
    rng = np.random.default_rng(BENIH)
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for p, w in ((1, BIRU), (2, HIJAU), (4, MERAH)):
        U = rng.random((4000, p))
        g = [kurva(n, p, rng, U).min() for n in NS]
        ax.plot(NS, g, "o-", ms=3, color=w, label=f"$p = {p}$")
        ns = np.array(NS, dtype=float)
        ax.plot(ns, g[0] * (ns / ns[0]) ** (-4 / (4 + p)), color=w,
                ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel("$\\mathrm{E}(\\hat f - f)^2$ dengan $k$ terbaik")
    ax.legend(fontsize=6, title="titik: kemiringan $-4/(4+p)$",
              title_fontsize=5.8)
    _rapikan(ax)
    simpan(fig, "bab09-laju")



# ============================ Bab 10 =================================

def bab10_volume():
    """Volume bola satuan, massa kulit, dan panjang sisi kubus kecil."""
    from scipy.special import gamma
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.1))
    p = np.arange(1, 31)
    V = np.pi ** (p / 2) / gamma(p / 2 + 1)
    ax = axs[0]
    ax.plot(p, V, "o-", ms=2, color=BIRU)
    ax.set_xlabel("$p$")
    ax.set_title("volume bola satuan $V_p$")
    ax = axs[1]
    pp = np.arange(1, 201)
    for eps, w in ((0.01, BIRU), (0.05, HIJAU), (0.1, JINGGA)):
        ax.plot(pp, 1 - (1 - eps) ** pp, color=w,
                label=f"kulit {angka(100 * eps, 0)}%")
    ax.set_xlabel("$p$")
    ax.set_title("massa bola di kulit luarnya")
    ax.legend(fontsize=5.8)
    ax = axs[2]
    sfr = np.linspace(0, 1, 201)
    for d, w in ((1, BIRU), (2, HIJAU), (3, JINGGA), (10, MERAH)):
        ax.plot(sfr, sfr ** (1 / d), color=w, label=f"$p = {d}$")
    ax.set_xlabel("bagian data yang dicakup")
    ax.set_title("panjang sisi kubus kecil")
    ax.legend(fontsize=5.8)
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-volume")


def bab10_pemusatan():
    """Sebaran jarak antartitik seragam, dibagi rata-ratanya."""
    rng = np.random.default_rng(BENIH)
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    for p, w in ((2, BIRU), (10, HIJAU), (100, JINGGA), (1000, MERAH)):
        X = rng.random((5000, p))
        Z = rng.random((5000, p))
        d = np.sqrt(((X - Z) ** 2).sum(1))
        ax.hist(d / d.mean(), bins=np.linspace(0, 2.6, 105),
                density=True, histtype="step", color=w, lw=1.0,
                label=f"$p = {p}$")
    ax.set_xlabel("jarak dibagi rata-ratanya")
    ax.set_ylabel("kepadatan")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab10-pemusatan")


def bab10_intrinsik():
    """Taksiran Levina-Bickel terhadap k untuk data berdimensi
    intrinsik diketahui dan untuk angka tulisan tangan."""
    from sklearn.datasets import load_digits, make_swiss_roll
    from bab10_intrinsik import levina_bickel
    rng = np.random.default_rng(BENIH)
    n = 2000
    Q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    data = [("kubus $p=2$", rng.random((n, 2)), 2, BIRU),
            ("kubus $p=5$", rng.random((n, 5)), 5, HIJAU),
            ("kubus $p=10$", rng.random((n, 10)), 10, JINGGA),
            ("bidang di $\\mathbb{R}^{64}$",
             np.hstack([rng.random((n, 2)), np.zeros((n, 62))]) @ Q,
             2, ABU),
            ("angka ($p = 64$)", load_digits().data, None, MERAH)]
    ks = [5, 8, 10, 15, 20, 30, 40]
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    for nama, D, benar, w in data:
        ax.plot(ks, [levina_bickel(D, k) for k in ks], "o-", ms=2.5,
                color=w, label=nama)
        if benar:
            ax.axhline(benar, color=w, ls=":", lw=0.7)
    ax.set_xlabel("$k$")
    ax.set_ylabel("taksiran dimensi intrinsik")
    ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(1, 1))
    _rapikan(ax)
    simpan(fig, "bab10-intrinsik")



# ============================ Bab 11 =================================

def bab11_kepadatan():
    """Kepadatan sebenarnya, taksiran k-NN, dan taksiran kernel."""
    from scipy.stats import gaussian_kde
    from bab11_kepadatan import p_benar, p_knn, x
    t = np.linspace(-4, 6, 1601)
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.3))
    ax = axs[0]
    ax.plot(t, p_benar(t), color=ABU, lw=1.4, label="sebenarnya")
    ax.plot(t, p_knn(t), color=BIRU, lw=0.9, label="$k$-NN, $k = 25$")
    ax.plot(t, gaussian_kde(x)(t), color=JINGGA, lw=0.9, label="kernel")
    ax.plot(x, np.full_like(x, -0.012), "|", color=ABU_GARIS, ms=3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("kepadatan")
    ax.legend(fontsize=6)
    ax = axs[1]
    tt = np.linspace(-40, 40, 4000)          # tanpa titik x = 0
    ax.plot(tt, p_benar(tt), color=ABU, lw=1.4)
    ax.plot(tt, p_knn(tt), color=BIRU, lw=0.9)
    ax.plot(tt, 25 / (2 * 500 * np.abs(tt)), color=MERAH, ls=":",
            lw=0.9, label="$k/(2n|x|)$")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("$x$")
    ax.set_title("ekor taksiran $k$-NN (skala log)")
    ax.legend(fontsize=6)
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab11-kepadatan")


def bab11_radius():
    """Bola k-NN (k = 8) dan bola radius tetap di daerah padat dan
    jarang."""
    rng = np.random.default_rng(BENIH)
    padat = rng.normal([0.3, 0.5], 0.08, (150, 2))
    jarang = rng.uniform([0.55, 0.05], [1.0, 0.95], (25, 2))
    X = np.vstack([padat, jarang])
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.7), sharey=True)
    for ax, judul in zip(axs, ("$k$ tetangga ($k = 8$)",
                               "radius tetap ($r = 0{,}1$)")):
        ax.scatter(*X.T, s=3, color=ABU_GARIS)
        for q, w in (([0.3, 0.5], BIRU), ([0.8, 0.5], JINGGA)):
            d = np.sqrt(((X - q) ** 2).sum(1))
            if judul.startswith("$k$"):
                r = np.sort(d)[7]
            else:
                r = 0.1
            dalam = d <= r
            ax.scatter(*X[dalam].T, s=5, color=w)
            ax.add_patch(plt.Circle(q, r, fill=False, color=w, lw=0.9))
            ax.plot(*q, "*", color=w, ms=6)
            ax.annotate(f"{dalam.sum()} titik", q, xytext=(0, 30),
                        textcoords="offset points", ha="center",
                        fontsize=6, color=w)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)
        ax.set_title(judul)
        _rapikan(ax)
    simpan(fig, "bab11-radius")


def bab11_bobot():
    """Galat sebenarnya terhadap k untuk tiga cara pembobotan."""
    from sklearn.neighbors import KNeighborsClassifier
    from bab04_sintetis import bangkitkan, eta
    from bab11_banding import BOBOT
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(400, rng)
    U = np.random.default_rng(BENIH + 1).random((20_000, 2))
    e = eta(U)
    ks = [1, 3, 5, 9, 15, 21, 31, 45, 61, 91, 121, 181]
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    for (nama, w), c in zip(BOBOT.items(), (BIRU, JINGGA, HIJAU)):
        g = [np.mean(np.where(KNeighborsClassifier(k, weights=w).fit(
            X, y).predict(U) == 1, 1 - e, e)) for k in ks]
        ax.plot(ks, g, "o-", ms=2.3, lw=0.9, color=c, label=nama)
    ax.axhline(0.1356, color=ABU, ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("galat sebenarnya")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab11-bobot")



# ============================ Bab 12 =================================

def bab12_S():
    """Matriks penghalus 5-NN untuk 30 titik terurut di [0, 1]."""
    rng = np.random.default_rng(BENIH)
    x = np.sort(rng.random(30))
    k = 5
    S = np.zeros((30, 30))
    for i in range(30):
        S[i, np.argsort(np.abs(x - x[i]), kind="stable")[:k]] = 1 / k
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.6),
                            gridspec_kw=dict(width_ratios=[1, 1.25]))
    axs[0].imshow(S, cmap="Blues", vmin=0, vmax=0.3)
    axs[0].set_title("$\\mathbf{S}$ untuk 5-NN, 30 titik terurut")
    axs[0].set_xlabel("kolom $j$")
    axs[0].set_ylabel("baris $i$")
    ax = axs[1]
    ax.plot(x, np.zeros_like(x), "|", color=ABU, ms=8)
    for i, w in ((0, JINGGA), (14, BIRU), (29, HIJAU)):
        tet = np.where(S[i] > 0)[0]
        ax.plot(x[tet], np.full(k, 0.2 + 0.1 * (i // 14)), "o", ms=3,
                color=w)
        ax.plot(x[i], 0.2 + 0.1 * (i // 14), "*", ms=7, color=w)
    ax.set_ylim(-0.1, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_title("tetangga baris 1, 15, dan 30")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-S")


def bab12_tepi():
    """Rata-rata ramalan k-NN atas 2000 data untuk f(x) = x, n = 99."""
    rng = np.random.default_rng(BENIH)
    n, ulang = 99, 2000
    t = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.plot(t, t, color=ABU, lw=1.2, ls="--", label="$f(x) = x$")
    for k, w in ((9, BIRU), (29, JINGGA), (59, MERAH)):
        rata = np.zeros_like(t)
        for _ in range(ulang):
            X = rng.random(n)
            idx = np.argsort(np.abs(t[:, None] - X[None, :]), 1)[:, :k]
            rata += X[idx].mean(1) / ulang
        ax.plot(t, rata, color=w, lw=1, label=f"$k = {k}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("rata-rata ramalan")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab12-tepi")


def bab12_median():
    """Satu data dengan 10% pencilan: rata-rata dan median 15 tetangga."""
    from bab12_median import ramal
    rng = np.random.default_rng(BENIH)
    f = lambda t: np.sin(2 * np.pi * t)
    X = rng.random((300, 1))
    y = f(X[:, 0]) + 0.3 * rng.standard_normal(300)
    y[rng.random(300) < 0.1] += 5
    T = np.linspace(0, 1, 501)[:, None]
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ax.scatter(X[:, 0], y, s=3, color=ABU_GARIS)
    ax.plot(T[:, 0], f(T[:, 0]), color=ABU, ls="--", lw=1, label="$f$")
    ax.plot(T[:, 0], ramal(X, y, T, 15, np.mean), color=MERAH, lw=1,
            label="rata-rata 15 tetangga")
    ax.plot(T[:, 0], ramal(X, y, T, 15, np.median), color=BIRU, lw=1,
            label="median 15 tetangga")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(fontsize=6, loc="center right", bbox_to_anchor=(1, 0.55))
    _rapikan(ax)
    simpan(fig, "bab12-median")



# ============================ Bab 13 =================================

def bab13_biasvar():
    """Bias^2, varians, dan jumlahnya untuk taksiran peluang k_1/k."""
    from sklearn.neighbors import NearestNeighbors
    from bab04_sintetis import bangkitkan, eta
    rng = np.random.default_rng(BENIH)
    X, _ = bangkitkan(400, rng)
    U = np.random.default_rng(BENIH + 1).random((20_000, 2))
    _, idx = NearestNeighbors(n_neighbors=201).fit(X).kneighbors(U)
    eX, eU = eta(X), eta(U)
    ks = np.arange(1, 202)
    b2, v = [], []
    for k in ks:
        en = eX[idx[:, :k]]
        b2.append(np.mean((en.mean(1) - eU) ** 2))
        v.append(np.mean((en * (1 - en)).sum(1) / k ** 2))
    b2, v = np.array(b2), np.array(v)
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ax.plot(ks, b2, color=JINGGA, label="bias$^2$")
    ax.plot(ks, v, color=BIRU, label="varians")
    ax.plot(ks, b2 + v, color=HIJAU, lw=1.4, label="jumlah")
    kb = ks[np.argmin(b2 + v)]
    ax.axvline(kb, color=HIJAU, ls=":", lw=0.8)
    ax.axvline(17, color=ABU, ls="--", lw=0.8)
    ax.text(kb * 1.08, 0.05, f"$k = {kb}$\n(peluang)", fontsize=6,
            color=HIJAU)
    ax.text(17 * 0.93, 0.05, "$k = 17$\n(klasifikasi)", fontsize=6,
            color=ABU, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("$\\mathrm{E}(\\hat\\eta - \\eta)^2$")
    ax.legend(fontsize=6, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, 1.0))
    _rapikan(ax)
    simpan(fig, "bab13-biasvar")


def bab13_keandalan():
    """Diagram keandalan terhadap posterior sebenarnya."""
    from sklearn.neighbors import KNeighborsClassifier
    from bab04_sintetis import bangkitkan, eta
    from bab13_kalibrasi import keandalan
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(400, rng)
    U = np.random.default_rng(BENIH + 1).random((50_000, 2))
    e = eta(U)
    fig, axs = plt.subplots(1, 3, figsize=(6.2, 2.2), sharey=True)
    for ax, (k, w, judul) in zip(axs, ((5, "uniform", "$k = 5$"),
                                       (15, "uniform", "$k = 15$"),
                                       (31, "distance",
                                        "$k = 31$, $1/d$"))):
        q = KNeighborsClassifier(k, weights=w).fit(X, y).predict_proba(
            U)[:, 1]
        t, b, ece = keandalan(q, e)
        ax.plot([0, 1], [0, 1], color=ABU_GARIS, lw=0.8)
        ax.plot(t, b, "o-", ms=2.5, color=BIRU)
        ax.set_title(f"{judul}, ECE {angka(ece, 3)}")
        ax.set_xlabel("peluang yang dilaporkan")
        ax.set_aspect("equal")
        _rapikan(ax)
    axs[0].set_ylabel("rata-rata $\\eta$ sebenarnya")
    simpan(fig, "bab13-keandalan")


def bab13_wilson():
    """Peluang cakupan selang Wald dan Wilson, k = 10 dan 30."""
    from bab13_wilson import cakupan, wald, wilson
    e = np.linspace(0.01, 0.99, 491)
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.2), sharey=True)
    for ax, k in zip(axs, (10, 30)):
        ax.plot(e, [cakupan(wald, v, k) for v in e], color=MERAH, lw=0.8,
                label="Wald")
        ax.plot(e, [cakupan(wilson, v, k) for v in e], color=BIRU,
                lw=0.8, label="Wilson")
        ax.axhline(0.95, color=ABU, ls=":", lw=0.8)
        ax.set_ylim(0.2, 1.01)
        ax.set_xlabel("$\\eta$")
        ax.set_title(f"$k = {k}$")
        _rapikan(ax)
    axs[0].set_ylabel("peluang cakupan")
    axs[0].legend(fontsize=6, loc="lower center")
    simpan(fig, "bab13-wilson")



# ============================ Bab 14 =================================

def bab14_batal():
    """Galat relatif jarak kuadrat rumus Gram terhadap bilangan kondisi
    kappa = (||x||^2 + ||z||^2) / ||x - z||^2."""
    rng = np.random.default_rng(BENIH)
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for tipe, w, u in ((np.float64, BIRU, 2.0 ** -53),
                       (np.float32, MERAH, 2.0 ** -24)):
        kap, rel = [], []
        for pusat in np.geomspace(1e-1, 1e6, 36):
            A = pusat + rng.standard_normal((40, 3))
            B = pusat + rng.standard_normal((40, 3))
            acuan = ((A[:, None] - B[None]) ** 2).sum(-1)
            a, b = A.astype(tipe), B.astype(tipe)
            g = ((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None]
                 - 2 * a @ b.T).astype(float)
            k = ((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None]) / acuan
            kap.append(k.ravel())
            rel.append(np.abs(g - acuan).ravel() / acuan.ravel())
        kap, rel = np.concatenate(kap), np.concatenate(rel)
        rel = np.maximum(rel, 1e-18)
        ax.scatter(kap, rel, s=0.6, color=w, alpha=0.3, lw=0)
        t = np.geomspace(1, 1e14, 50)
        ax.plot(t, u * t, color=w, lw=0.9,
                label=("float64" if tipe is np.float64 else "float32")
                + ", garis $u\\kappa$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e3)
    ax.set_xlabel("bilangan kondisi $\\kappa$")
    ax.set_ylabel("galat relatif $D^2$")
    ax.legend(fontsize=6, loc="upper left")
    _rapikan(ax)
    simpan(fig, "bab14-batal")


def bab14_seleksi():
    """Perbandingan quickselect per n terhadap k/n: simulasi dan rumus."""
    from bab14_seleksi import knuth, quickselect
    rng = np.random.default_rng(BENIH)
    n = 1000
    kk = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    ax.plot(kk / n, [knuth(n, k) / n for k in kk], color=BIRU,
            label="rumus Knuth")
    ks = [1, 50, 125, 250, 375, 500, 625, 750, 875, 950, 1000]
    sim = []
    for k in ks:
        c = [quickselect(rng.permutation(n).astype(float), k, rng)[1]
             for _ in range(600)]
        sim.append(np.mean(c) / n)
    ax.plot(np.array(ks) / n, sim, "o", ms=3, color=JINGGA,
            label="simulasi (600 ulangan)")
    ax.axhline(4, color=ABU, ls=":", lw=0.8)
    ax.text(0.02, 3.85, "batas $4n$", fontsize=6, color=ABU, va="top")
    ax.set_ylim(0, 4.3)
    ax.set_xlabel("$k/n$")
    ax.set_ylabel("perbandingan / $n$")
    ax.legend(fontsize=6, loc="lower center")
    _rapikan(ax)
    simpan(fig, "bab14-seleksi")


def bab14_blok():
    """Skema pencarian per blok: blok kueri kali seluruh data latih."""
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    # matriks jarak m x n
    ax.add_patch(Rectangle((1.2, 0), 4.0, 2.4, facecolor=BIRU_MUDA,
                           edgecolor=ABU_GARIS))
    for j in range(6):
        y0 = 2.4 - (j + 1) * 0.4
        w = JINGGA_MUDA if j == 2 else BIRU_MUDA
        ax.add_patch(Rectangle((1.2, y0), 4.0, 0.4, facecolor=w,
                               edgecolor="white", lw=1))
    ax.add_patch(Rectangle((0.2, 0), 0.7, 2.4, facecolor=HIJAU_MUDA,
                           edgecolor=ABU_GARIS))
    ax.add_patch(Rectangle((0.2, 1.2), 0.7, 0.4, facecolor=JINGGA,
                           alpha=0.6))
    ax.add_patch(Rectangle((1.2, 2.55), 4.0, 0.5, facecolor=HIJAU_MUDA,
                           edgecolor=ABU_GARIS))
    ax.add_patch(Rectangle((5.5, 1.2), 0.5, 0.4, facecolor=JINGGA,
                           alpha=0.6))
    ax.annotate("", xy=(5.45, 1.4), xytext=(5.25, 1.4),
                arrowprops=dict(arrowstyle="->", color=JINGGA))
    ax.text(0.55, -0.2, "kueri $Q$\n($m \\times p$)", ha="center",
            va="top", fontsize=6.3)
    ax.text(3.2, 3.15, "data latih $X^{\\top}$ ($p \\times n$)",
            ha="center", fontsize=6.3)
    ax.text(3.2, 1.4, "satu blok: $b \\times n$ jarak", ha="center",
            va="center", fontsize=6.3, color=JINGGA)
    ax.text(3.2, -0.2, "matriks jarak penuh $m \\times n$ tidak pernah "
            "disimpan", ha="center", va="top", fontsize=6.3)
    ax.text(5.75, 1.0, "$b \\times k$\nindeks", ha="center", va="top",
            fontsize=6.3, color=JINGGA)
    ax.set_xlim(0, 6.3)
    ax.set_ylim(-0.75, 3.4)
    ax.axis("off")
    simpan(fig, "bab14-blok")



# ============================ Bab 15 =================================

def _daun_semua(s):
    if s.idx is not None:
        return [s]
    return _daun_semua(s.kiri) + _daun_semua(s.kanan)


def bab15_kd():
    """Kotak-kotak daun pohon k-d pada 80 titik, dan kotak yang
    diperiksa (biru) atau dilewati (abu-abu) untuk satu kueri."""
    from matplotlib.patches import Rectangle
    from bab15_pohon import Pohon
    rng = np.random.default_rng(BENIH)
    X = rng.random((80, 2))
    T = Pohon(X, daun=4, bola=False)
    q = np.array([0.52, 0.47])
    jejak = []
    d, j = T.query(q, k=3, jejak=jejak)
    diperiksa = {id(e[1]) for e in jejak if e[0] == "daun"}
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.9))
    for ax, bola in zip(axs, (False, True)):
        T = Pohon(X, daun=4, bola=bola)
        jejak = []
        T.query(q, k=3, jejak=jejak)
        diperiksa = {id(e[1]) for e in jejak if e[0] == "daun"}
        for s in _daun_semua(T.akar):
            w = BIRU if id(s) in diperiksa else ABU_GARIS
            if bola:
                ax.add_patch(plt.Circle(s.pusat, s.r, fill=id(s) in
                             diperiksa, facecolor=BIRU_MUDA,
                             edgecolor=w, lw=0.7, alpha=0.8))
            else:
                ax.add_patch(Rectangle(s.lo, *(s.hi - s.lo), fill=id(s)
                             in diperiksa, facecolor=BIRU_MUDA,
                             edgecolor=w, lw=0.7))
        ax.scatter(*X.T, s=4, color=ABU, zorder=3)
        ax.scatter(*X[j].T, s=10, color=JINGGA, zorder=4)
        ax.plot(*q, "*", color=MERAH, ms=7, zorder=5)
        ax.add_patch(plt.Circle(q, d[-1], fill=False, color=MERAH,
                                ls="--", lw=0.8))
        ax.set_aspect("equal")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(("ball tree" if bola else "pohon k-d") +
                     f": {T.n_jarak} jarak ke titik")
        _rapikan(ax)
    simpan(fig, "bab15-kd")


def bab15_dimensi():
    """Banyaknya jarak per kueri terhadap dimensi, data seragam."""
    from bab15_pohon import Pohon
    from bab15_dimensi import rata
    rng = np.random.default_rng(BENIH)
    ps = [2, 3, 4, 6, 8, 12, 16, 24, 32]
    kd, bt, bp = [], [], []
    for p in ps:
        X = rng.random((10_000, p))
        Q = rng.random((60, p))
        kd.append(rata(Pohon(X, 20, False), Q)[0])
        a, b = rata(Pohon(X, 20, True), Q)
        bt.append(a)
        bp.append(a + b)
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.plot(ps, kd, "o-", ms=2.5, color=BIRU, label="pohon k-d")
    ax.plot(ps, bp, "s-", ms=2.5, color=HIJAU,
            label="ball tree (titik + pusat)")
    ax.axhline(10_000, color=MERAH, ls="--", lw=0.9, label="brute force")
    ax.set_yscale("log")
    ax.set_xlabel("dimensi $p$")
    ax.set_ylabel("jarak per kueri")
    ax.legend(fontsize=6, loc="lower right")
    _rapikan(ax)
    simpan(fig, "bab15-dimensi")



# ============================ Bab 16 =================================

def bab16_jl():
    """Rasio jarak sesudah/sebelum proyeksi acak untuk tiga k."""
    from sklearn.metrics import pairwise_distances
    from bab16_jl import proyeksi
    rng = np.random.default_rng(BENIH)
    X = rng.standard_normal((300, 5000))
    D = pairwise_distances(X)
    iu = np.triu_indices(300, 1)
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    for k, w in ((50, MERAH), (200, JINGGA), (1000, BIRU)):
        r = pairwise_distances(proyeksi(X, k, rng))[iu] / D[iu]
        ax.hist(r, bins=np.linspace(0.55, 1.5, 120), density=True,
                histtype="step", color=w, lw=1, label=f"$k = {k}$")
    ax.axvline(1, color=ABU, ls=":", lw=0.8)
    ax.set_xlabel("jarak sesudah / jarak sebelum proyeksi")
    ax.set_ylabel("kepadatan")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab16-jl")


def bab16_bidang():
    """Kiri: bukti peluang 1 - theta/pi. Kanan: simulasi."""
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.6))
    ax = axs[0]
    t = np.radians(60)
    u = np.array([1, 0])
    v = np.array([np.cos(t), np.sin(t)])
    busur = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(busur), np.sin(busur), color=ABU_GARIS, lw=0.8)
    # arah r yang membuat tanda berbeda: normalnya di dua busur
    for a0 in (np.pi / 2, 3 * np.pi / 2):
        bb = np.linspace(a0, a0 + t, 60)
        ax.fill(np.r_[0, np.cos(bb)], np.r_[0, np.sin(bb)],
                color=JINGGA_MUDA)
    for w, vek, nama in ((BIRU, u, "$\\mathbf{u}$"),
                         (HIJAU, v, "$\\mathbf{v}$")):
        ax.annotate("", xy=vek, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=w, lw=1.2))
        ax.text(*(1.12 * vek), nama, color=w, ha="center", va="center")
    r = np.array([np.cos(np.radians(120)), np.sin(np.radians(120))])
    ax.annotate("", xy=0.8 * r, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=MERAH, lw=1))
    ax.text(*(0.95 * r), "$\\mathbf{r}$", color=MERAH, ha="center")
    ax.plot([-1.2 * r[1], 1.2 * r[1]], [1.2 * r[0], -1.2 * r[0]],
            color=MERAH, ls="--", lw=0.8)
    ax.text(0.28, 0.1, "$\\theta$", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    ax.set_title("$\\mathbf{r}$ di daerah jingga: tanda berbeda")
    ax = axs[1]
    rng = np.random.default_rng(BENIH)
    der = np.arange(0, 181, 15)
    sim = []
    for d in der:
        a = np.radians(d)
        uu = np.array([1.0, 0, 0, 0])
        vv = np.array([np.cos(a), np.sin(a), 0, 0])
        R = rng.standard_normal((40_000, 4))
        sim.append(np.mean(np.sign(R @ uu) == np.sign(R @ vv)))
    ax.plot(der, 1 - np.radians(der) / np.pi, color=BIRU,
            label="$1 - \\theta/\\pi$")
    ax.plot(der, sim, "o", ms=2.5, color=JINGGA, label="simulasi")
    ax.set_xlabel("sudut $\\theta$ (derajat)")
    ax.set_ylabel("peluang bit sama")
    ax.legend(fontsize=6)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab16-bidang")


def bab16_amplifikasi():
    """Kurva S 1 - (1 - s^r)^b terhadap sudut, beberapa (r, b)."""
    th = np.linspace(0, 180, 361)
    s = 1 - th / 180
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    for (r, b), w in (((1, 1), ABU), ((4, 1), MERAH), ((8, 5), JINGGA),
                      ((8, 20), BIRU), ((16, 50), HIJAU)):
        ax.plot(th, 1 - (1 - s ** r) ** b, color=w, lw=1,
                label=f"$r = {r}$, $b = {b}$")
    ax.set_xlabel("sudut $\\theta$ (derajat)")
    ax.set_ylabel("peluang menjadi calon")
    ax.legend(fontsize=5.8)
    _rapikan(ax)
    simpan(fig, "bab16-amplifikasi")


def bab16_recall():
    """Recall@10 terhadap bagian data yang diperiksa, indeks LSH."""
    from sklearn.datasets import load_digits
    from bab16_lsh import IndeksLSH
    rng = np.random.default_rng(BENIH)
    X, _ = load_digits(return_X_y=True)
    X = X - X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q, D = X[:150], X[200:]
    benar = np.argsort(-(Q @ D.T), axis=1)[:, :10]
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for r, w in ((4, MERAH), (8, JINGGA), (12, BIRU), (16, HIJAU)):
        fr, rc = [], []
        for b in (1, 2, 5, 10, 20, 50):
            idx = IndeksLSH(D, r, b, rng)
            a = c = 0.0
            for i, q in enumerate(Q):
                j, nc = idx.query(q, 10)
                a += len(set(j) & set(benar[i])) / 10
                c += nc / len(D)
            fr.append(c / len(Q))
            rc.append(a / len(Q))
        ax.plot(fr, rc, "o-", ms=2.5, lw=0.9, color=w, label=f"$r = {r}$")
    ax.set_xscale("log")
    ax.set_xlabel("bagian data yang diperiksa ulang")
    ax.set_ylabel("recall@10")
    ax.legend(fontsize=6, title="$b = 1, \\ldots, 50$", title_fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab16-recall")



# ============================ Bab 17 =================================

def bab17_kondensasi():
    """Data gelombang: semua titik, kondensasi Hart, suntingan +
    kondensasi, dengan batas keputusan 1-NN masing-masing."""
    from matplotlib.colors import ListedColormap
    from sklearn.neighbors import KNeighborsClassifier
    from bab04_sintetis import bangkitkan, g
    from bab17_kondensasi import kondensasi, suntingan
    rng = np.random.default_rng(BENIH)
    X, y = bangkitkan(1000, rng)
    c, _ = kondensasi(X, y)
    s = suntingan(X, y, 3)
    sc, _ = kondensasi(X[s], y[s])
    himp = [("semua (1000)", X, y), (f"kondensasi ({len(c)})", X[c], y[c]),
            (f"suntingan + kondensasi ({len(sc)})", X[s][sc], y[s][sc])]
    t = np.linspace(0, 1, 251)
    G1, G2 = np.meshgrid(t, t)
    G = np.column_stack([G1.ravel(), G2.ravel()])
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)
    for ax, (judul, A, b) in zip(axs, himp):
        z = KNeighborsClassifier(1).fit(A, b).predict(G).reshape(G1.shape)
        ax.contourf(G1, G2, z, levels=[-0.5, 0.5, 1.5],
                    cmap=ListedColormap([BIRU_MUDA, JINGGA_MUDA]))
        ax.scatter(*A[b == 0].T, s=2 if len(A) > 500 else 5, color=BIRU,
                   lw=0)
        ax.scatter(*A[b == 1].T, s=2 if len(A) > 500 else 5,
                   color=JINGGA, lw=0)
        ax.plot(t, g(t), color=HIJAU, ls="--", lw=0.8)
        ax.set_title(judul, fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("$x_1$")
    axs[0].set_ylabel("$x_2$")
    simpan(fig, "bab17-kondensasi")


def bab17_prototipe():
    """Akurasi 1-NN angka tulisan tangan terhadap banyaknya prototipe."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from bab17_prototipe import prototipe
    from bab17_kondensasi import kondensasi
    X, y = load_digits(return_X_y=True)
    Xl, Xu, yl, yu = train_test_split(X, y, test_size=450, stratify=y,
                                      random_state=BENIH)
    ms = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60]
    a = []
    for m in ms:
        P, L = prototipe(Xl, yl, m)
        a.append(KNeighborsClassifier(1).fit(P, L).score(Xu, yu))
    c, _ = kondensasi(Xl, yl)
    ac = KNeighborsClassifier(1).fit(Xl[c], yl[c]).score(Xu, yu)
    semua = KNeighborsClassifier(1).fit(Xl, yl).score(Xu, yu)
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ax.plot(10 * np.array(ms), a, "o-", ms=2.5, color=BIRU,
            label="pusat K-Means per kelas")
    ax.plot(len(c), ac, "s", ms=4, color=JINGGA, label="kondensasi Hart")
    ax.axhline(semua, color=HIJAU, ls="--", lw=0.9,
               label=f"semua {len(Xl)} titik latih")
    ax.set_xscale("log")
    ax.set_xlabel("banyaknya prototipe")
    ax.set_ylabel("akurasi 1-NN data uji")
    ax.legend(fontsize=6, loc="lower right")
    _rapikan(ax)
    simpan(fig, "bab17-prototipe")


def bab17_timpang():
    """Akurasi seimbang k-NN biasa dan terkoreksi prior terhadap k."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from bab17_timpang import timpang
    rng = np.random.default_rng(BENIH)
    X, y = timpang(2000, rng)
    U, yu = timpang(40_000, rng)
    pi = np.bincount(y) / len(y)
    ks = [1, 3, 5, 9, 15, 25, 45, 75, 121]
    a, b = [], []
    for k in ks:
        p = KNeighborsClassifier(k).fit(X, y).predict_proba(U)
        a.append(balanced_accuracy_score(yu, p.argmax(1)))
        b.append(balanced_accuracy_score(yu, (p / pi).argmax(1)))
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    ax.plot(ks, a, "o-", ms=2.5, color=MERAH, label="$k$-NN biasa")
    ax.plot(ks, b, "s-", ms=2.5, color=BIRU,
            label="$k$-NN dibagi prior")
    ax.set_xscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("akurasi seimbang")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab17-timpang")



# ============================ Bab 18 =================================

def bab18_satu():
    """Contoh tiga titik di garis: p_1(l) dan f(l) = p_1 + p_2 + p_3."""
    l = np.linspace(0, 2.5, 301)
    p1 = 1 / (1 + np.exp(-3 * l ** 2))
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.plot(l, p1, color=BIRU, label="$p_1(l)$")
    ax.plot(l, np.full_like(l, 0.5), color=HIJAU, label="$p_2 = 1/2$")
    ax.plot(l, np.zeros_like(l), color=JINGGA, label="$p_3 = 0$")
    ax.plot(l, p1 + 0.5, color=MERAH, lw=1.4, label="$f(l)$")
    ax.axvline(1, color=ABU_GARIS, ls=":", lw=0.8)
    ax.set_xlabel("skala $l$")
    ax.legend(fontsize=6, loc="center right")
    _rapikan(ax)
    simpan(fig, "bab18-satu")


def bab18_bobot():
    """Jumlah kuadrat kolom L NCA dua dimensi pada anggur + 26 noise."""
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    from bab06_noise import noise
    from bab18_nca import latih_nca
    X, y = load_wine(return_X_y=True)
    D = StandardScaler().fit_transform(np.hstack([X, noise(0, len(X),
                                                              26)]))
    L, _ = latih_nca(D, y, d=2)
    n = (L ** 2).sum(0)
    fig, ax = plt.subplots(figsize=(5.2, 2.1))
    ax.bar(np.arange(13), n[:13], color=BIRU, label="13 fitur asli")
    ax.bar(np.arange(13, 39), n[13:], color=ABU_GARIS,
           label="26 fitur noise")
    ax.set_xlabel("fitur")
    ax.set_ylabel("$\\sum_i L_{ij}^2$")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab18-bobot")


def bab18_dua():
    """Angka tulisan tangan di dua dimensi: PCA dan NCA."""
    from bab18_digit2d import proyeksi_dua
    hasil, yl, _ = proyeksi_dua()
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.8))
    for ax, (nama, (A, _)) in zip(axs, hasil.items()):
        for c in range(10):
            ax.scatter(*A[yl == c].T, s=2, color=cmap(c), lw=0)
            m = A[yl == c].mean(0)
            ax.text(*m, str(c), fontsize=8, ha="center", va="center",
                    weight="bold", color="black")
        ax.set_title(nama)
        ax.set_xticks([])
        ax.set_yticks([])
    simpan(fig, "bab18-dua")



# ============================ Bab 19 =================================

def bab19_peta():
    """Peta median nilai rumah dan kisi wilayah 0,5 derajat."""
    from bab19_data import muat
    X, y, _, _, _ = muat()
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    urut = np.argsort(y)
    sc = ax.scatter(X[urut, 7], X[urut, 6], c=y[urut], s=0.6,
                    cmap="viridis", lw=0)
    for v in np.arange(-124.5, -113.9, 0.5):
        ax.axvline(v, color=ABU_GARIS, lw=0.2)
    for v in np.arange(32.5, 42.1, 0.5):
        ax.axhline(v, color=ABU_GARIS, lw=0.2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("median nilai rumah (100 ribu dolar)")
    ax.set_xlabel("bujur")
    ax.set_ylabel("lintang")
    ax.set_aspect(1 / np.cos(np.radians(35.6)))
    kunci_label(ax, "x", "y")
    simpan(fig, "bab19-peta")


def bab19_s():
    """RMSE validasi silang terhadap skala pendapatan s, k = 10."""
    from bab19_evaluasi import semua_skema, SS
    hasil = semua_skema()
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    for nama, w in (("acak", BIRU), ("wilayah", JINGGA)):
        kisi = hasil[nama][1]
        ax.plot(SS, [kisi[(s, 10)] for s in SS], "o-", ms=3, color=w,
                label=f"validasi {nama}")
        ax.axhline(hasil[nama][0]["regresi linear"], color=w, ls=":",
                   lw=0.8)
    ax.set_xlabel("skala pendapatan $s$ (km per 10 ribu dolar)")
    ax.set_ylabel("RMSE, $k = 10$")
    ax.legend(fontsize=6, loc="upper right")
    _rapikan(ax)
    simpan(fig, "bab19-s")


def bab19_galat():
    """Ramalan lawan label (validasi per wilayah), dan RMSE terhadap
    jarak ke tetangga ke-10."""
    from bab19_galat import ramalan_wilayah
    X, y, yh, dk = ramalan_wilayah()
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.5))
    ax = axs[0]
    ax.scatter(y, yh, s=0.5, color=BIRU, alpha=0.3, lw=0)
    ax.plot([0, 5.2], [0, 5.2], color=MERAH, lw=0.8)
    ax.set_xlabel("label")
    ax.set_ylabel("ramalan")
    ax.set_aspect("equal")
    ax = axs[1]
    tepi = np.quantile(dk, np.linspace(0, 1, 11))
    r = y - yh
    tengah, nilai = [], []
    for a, b in zip(tepi[:-1], tepi[1:]):
        m = (dk >= a) & (dk <= b)
        tengah.append(np.median(dk[m]))
        nilai.append(np.sqrt(np.mean(r[m] ** 2)))
    ax.plot(tengah, nilai, "o-", ms=3, color=JINGGA)
    ax.set_xlabel("jarak ke tetangga ke-10 (median desil)")
    ax.set_ylabel("RMSE")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab19-galat")



# ============================ Bab 20 =================================

def bab20_graf():
    """Recall@1 terhadap banyaknya jarak, pencarian rakus di graf."""
    from sklearn.datasets import load_digits
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from bab20_graf import cari_rakus, graf_tetangga
    X, _ = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    Q, D = X[:150], X[200:]
    benar = NearestNeighbors(n_neighbors=1).fit(D).kneighbors(Q)[1][:, 0]
    rng = np.random.default_rng(BENIH)
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    for m, w in ((5, MERAH), (10, JINGGA), (20, BIRU), (40, HIJAU)):
        graf = graf_tetangga(D, m)
        xs, ys = [], []
        for r in (1, 2, 5, 10):
            hit = jum = 0
            for i, q in enumerate(Q):
                j, n = cari_rakus(D, graf, q, rng.integers(0, len(D), r))
                hit += j == benar[i]
                jum += n
            xs.append(jum / len(Q))
            ys.append(hit / len(Q))
        ax.plot(xs, ys, "o-", ms=2.5, color=w, label=f"$m = {m}$")
    ax.set_xscale("log")
    ax.set_xlabel("jarak yang dihitung per kueri")
    ax.set_ylabel("recall@1")
    ax.legend(fontsize=6, title="titik: 1, 2, 5, 10 awal",
              title_fontsize=6, loc="lower right")
    _rapikan(ax)
    simpan(fig, "bab20-graf")


def bab20_anomali():
    """Skor jarak ke tetangga ke-10 dan skor LOF."""
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from bab20_anomali import data
    X, y = data(np.random.default_rng(BENIH))
    d = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)[0][:, -1]
    lof = -LocalOutlierFactor(n_neighbors=10).fit(X) \
        .negative_outlier_factor_
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.4), sharey=True)
    for ax, sk, judul in ((axs[0], d, "jarak ke tetangga ke-10"),
                          (axs[1], lof, "LOF")):
        atas = np.argsort(-sk)[:20]
        ax.scatter(*X.T, s=2, color=ABU_GARIS)
        ax.scatter(*X[atas].T, s=12, facecolor="none", edgecolor=MERAH,
                   lw=0.8, label="20 skor tertinggi")
        ax.scatter(*X[y == 1].T, s=4, color=BIRU, label="pencilan")
        ax.set_title(judul)
        ax.set_aspect("equal")
        _rapikan(ax)
    axs[0].legend(fontsize=5.8, loc="lower left")
    simpan(fig, "bab20-anomali")


def bab20_isomap():
    """Gulungan, komponen pertama PCA, dan komponen pertama Isomap."""
    from sklearn.datasets import make_swiss_roll
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    X, t = make_swiss_roll(1500, random_state=BENIH)
    zp = PCA(2).fit_transform(X)[:, 0]
    zi = Isomap(n_neighbors=10, n_components=2).fit_transform(X)[:, 0]
    fig, axs = plt.subplots(1, 3, figsize=(6.2, 2.1))
    axs[0].scatter(X[:, 0], X[:, 2], c=t, s=1.5, cmap="viridis", lw=0)
    axs[0].set_title("gulungan ($x_1$, $x_3$)")
    axs[0].set_aspect("equal")
    axs[1].scatter(t, zp, s=1.5, color=ABU, lw=0)
    axs[1].set_title("PCA, komponen 1")
    axs[2].scatter(t, zi, s=1.5, color=BIRU, lw=0)
    axs[2].set_title("Isomap, komponen 1")
    for ax in axs[1:]:
        ax.set_xlabel("posisi di gulungan $t$")
    for ax in axs:
        _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab20-isomap")



# ============================ Bab 1 ==================================

def bab01_ide():
    """Jeruk dan apel rekaan: titik uji dan lingkaran 1, 3, 5 tetangga."""
    J = np.array([[7, 8], [8, 7], [6, 9]])
    A = np.array([[7, 3], [8, 2], [6, 4]])
    q = np.array([7, 6])
    P = np.vstack([J, A])
    d = np.sqrt(((P - q) ** 2).sum(1))
    r = np.sort(d)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.scatter(*J.T, color=JINGGA, s=30, label="jeruk", zorder=3)
    ax.scatter(*A.T, color=MERAH, marker="s", s=26, label="apel",
               zorder=3)
    ax.plot(*q, "*", color=BIRU, ms=10, zorder=4, label="buah baru")
    for k, ls in ((1, "-"), (3, "--"), (5, ":")):
        ax.add_patch(plt.Circle(q, r[k - 1] + 0.05, fill=False, color=BIRU,
                                ls=ls, lw=0.8))
        ax.text(q[0] - (r[k - 1] + 0.1) * 0.72, q[1] + (r[k - 1] + 0.1)
                * 0.72, f"$k={k}$", fontsize=6.5, color=BIRU,
                ha="right")
    ax.set_xlim(3.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_aspect("equal")
    ax.set_xlabel("diameter (cm)")
    ax.set_ylabel("kekasaran kulit (0-10)")
    ax.legend(fontsize=6, loc="lower left")
    _rapikan(ax)
    simpan(fig, "bab01-ide")


def bab01_sejarah():
    """Garis waktu gagasan tetangga terdekat (satu baris per peristiwa)."""
    peristiwa = [(1951, "Fix & Hodges", "aturan k tetangga terdekat"),
                 (1967, "Cover & Hart", "batas galat 1-NN dua kali galat Bayes"),
                 (1968, "Hart", "kondensasi data latih"),
                 (1972, "Wilson", "penyuntingan data latih"),
                 (1975, "Bentley", "pohon k-d"),
                 (1977, "Stone", "k-NN konsisten universal"),
                 (1977, "Friedman, Bentley & Finkel", "pencarian log n"),
                 (1989, "Omohundro", "ball tree"),
                 (1998, "Indyk & Motwani", "locality-sensitive hashing"),
                 (2004, "Goldberger dkk.", "NCA, belajar metrik"),
                 (2016, "Malkov & Yashunin", "graf navigasi HNSW")]
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    n = len(peristiwa)
    ax.plot([0.9, 0.9], [0.5, n + 0.5], color=ABU_GARIS, lw=1)
    for i, (t, siapa, apa) in enumerate(peristiwa):
        yy = n - i
        ax.plot(0.9, yy, "o", color=BIRU, ms=3)
        ax.text(0.75, yy, str(t), ha="right", va="center", fontsize=6.5,
                color=BIRU)
        ax.text(1.05, yy, f"{siapa}: {apa}", ha="left", va="center",
                fontsize=6.5, color=ABU)
    ax.set_xlim(0, 6)
    ax.set_ylim(0.3, n + 0.7)
    ax.axis("off")
    simpan(fig, "bab01-sejarah")


def bab01_peta():
    """Peta enam bagian buku."""
    bagian = [("I Fondasi", "1-3", BIRU_MUDA),
              ("II Aturan tetangga terdekat", "4-7", HIJAU_MUDA),
              ("III Mengapa tetangga bekerja", "8-10", JINGGA_MUDA),
              ("IV Bobot, regresi, peluang", "11-13", MERAH_MUDA),
              ("V Mencari dengan cepat", "14-16", BIRU_MUDA),
              ("VI Menyempurnakan, menerapkan", "17-20", HIJAU_MUDA)]
    fig, ax = plt.subplots(figsize=(5.8, 2.2))
    pos = [(0, 1), (1, 1), (2, 1), (0, 0), (1, 0), (2, 0)]
    for (nama, bab, w), (c, r) in zip(bagian, pos):
        _kotak_teks(ax, c * 2.0, r * 1.1, 1.85, 0.9,
                    f"{nama}\nBab {bab}", w, ukuran=5.8)
    ax.set_xlim(-0.05, 5.95)
    ax.set_ylim(-0.05, 2.05)
    ax.axis("off")
    simpan(fig, "bab01-peta")


def _kotak_teks(ax, x, y, w, h, teks, warna, ukuran=6.5):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=warna, edgecolor=ABU_GARIS))
    ax.text(x + w / 2, y + h / 2, teks, ha="center", va="center",
            fontsize=ukuran, color=BIRU)



# ============================ Bab 2 ==================================

def bab02_proyeksi():
    """Dua vektor, sudutnya, dan proyeksi u ke arah v."""
    u = np.array([1.0, 2.2])
    v = np.array([2.6, 0.6])
    pr = (u @ v) / (v @ v) * v
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for w, vek, nama in ((BIRU, u, "$\\mathbf{u}$"), (HIJAU, v,
                                                       "$\\mathbf{v}$")):
        ax.annotate("", xy=vek, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=w, lw=1.3))
        ax.text(*(vek * 1.07), nama, color=w, fontsize=8)
    ax.plot([u[0], pr[0]], [u[1], pr[1]], color=ABU, ls="--", lw=0.8)
    ax.plot(*pr, "o", color=JINGGA, ms=3)
    ax.text(pr[0], pr[1] - 0.35, "proyeksi $\\mathbf{u}$ ke $\\mathbf{v}$",
            color=JINGGA, fontsize=6.5, ha="center")
    t = np.linspace(np.arctan2(v[1], v[0]), np.arctan2(u[1], u[0]), 30)
    ax.plot(0.5 * np.cos(t), 0.5 * np.sin(t), color=MERAH, lw=0.8)
    ax.text(0.55, 0.45, "$\\theta$", color=MERAH, fontsize=8)
    ax.set_xlim(-0.2, 3.1)
    ax.set_ylim(-0.6, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    simpan(fig, "bab02-proyeksi")


def bab02_elips():
    """Kurva x^T A x = 1 untuk A = Sigma dan A = Sigma^{-1}, dengan
    vektor eigen Sigma = [[2, 1], [1, 2]]."""
    S = np.array([[2.0, 1.0], [1.0, 2.0]])
    lam, V = np.linalg.eigh(S)
    t = np.linspace(0, 2 * np.pi, 400)
    lingk = np.column_stack([np.cos(t), np.sin(t)])
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for A, w, nama in ((S, BIRU, "$\\mathbf{x}^\\top\\Sigma\\mathbf{x}=1$"),
                       (np.linalg.inv(S), JINGGA,
                        "$\\mathbf{x}^\\top\\Sigma^{-1}\\mathbf{x}=1$")):
        la, Va = np.linalg.eigh(A)
        e = (lingk / np.sqrt(la)) @ Va.T
        ax.plot(*e.T, color=w, lw=1.1, label=nama)
    for j in range(2):
        ax.annotate("", xy=V[:, j] * np.sqrt(lam[j]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=HIJAU))
        ax.text(*(V[:, j] * np.sqrt(lam[j]) * 1.15),
                f"$\\lambda = {lam[j]:.0f}$", color=HIJAU, fontsize=6.5,
                ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.axhline(0, color=ABU_GARIS, lw=0.5)
    ax.axvline(0, color=ABU_GARIS, lw=0.5)
    ax.legend(fontsize=6, loc="lower right")
    _rapikan(ax)
    simpan(fig, "bab02-elips")


def bab02_gamma():
    """Fungsi gamma dan faktorial."""
    from scipy.special import gamma
    s = np.linspace(0.2, 5.5, 400)
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    ax.plot(s, gamma(s), color=BIRU, label="$\\Gamma(s)$")
    n = np.arange(1, 6)
    ax.plot(n, [1, 1, 2, 6, 24], "o", color=JINGGA, ms=3.5,
            label="$(s - 1)!$")
    ax.plot([0.5, 1.5, 2.5], gamma(np.array([0.5, 1.5, 2.5])), "s",
            color=HIJAU, ms=3, label="$s = 1/2, 3/2, 5/2$")
    ax.set_ylim(0, 26)
    ax.set_xlabel("$s$")
    ax.legend(fontsize=6, loc="upper left")
    _rapikan(ax)
    simpan(fig, "bab02-gamma")



# ============================ Bab 3 ==================================

def bab03_binomial():
    """Sebaran Bin(k, 0,7) untuk k = 5 dan 21; batang merah = suara
    terbanyak memilih kelas yang kurang mungkin."""
    from scipy.stats import binom
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 2.2))
    for ax, k in zip(axs, (5, 21)):
        j = np.arange(k + 1)
        pm = binom.pmf(j, k, 0.7)
        w = [MERAH if v < k / 2 else BIRU for v in j]
        ax.bar(j, pm, color=w, width=0.8)
        ax.set_title(f"$k = {k}$, peluang merah "
                     f"{angka(binom.cdf((k - 1) // 2, k, 0.7), 3)}")
        ax.set_xlabel("banyaknya tetangga kelas 1")
        _rapikan(ax)
    axs[0].set_ylabel("peluang")
    fig.tight_layout()
    simpan(fig, "bab03-binomial")


def bab03_batas():
    """Peluang ekor binomial, batas Chebyshev, dan batas Hoeffding."""
    from scipy.stats import binom
    ks = np.arange(1, 152, 2)
    eksak = binom.cdf((ks - 1) // 2, ks, 0.7)
    t = ks * 0.2
    cheb = np.minimum(1, ks * 0.21 / t ** 2)
    hoef = np.exp(-2 * t ** 2 / ks)
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ax.plot(ks, eksak, color=BIRU, label="eksak")
    ax.plot(ks, cheb, color=JINGGA, label="Chebyshev")
    ax.plot(ks, hoef, color=HIJAU, label="Hoeffding")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1.5)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$P(\\mathrm{Bin}(k; 0{,}7) < k/2)$")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab03-batas")


def bab03_validasi():
    """Skema validasi silang 5 lipatan: acak dan per kelompok."""
    from matplotlib.patches import Rectangle
    rng = np.random.default_rng(BENIH)
    fig, axs = plt.subplots(1, 2, figsize=(5.8, 1.9))
    n = 40
    kelompok = np.repeat(np.arange(8), 5)
    lipat_acak = rng.permutation(np.repeat(np.arange(5), 8))
    lipat_kel = kelompok % 5
    for ax, lipat, judul in ((axs[0], lipat_acak, "acak (KFold)"),
                             (axs[1], lipat_kel,
                              "per kelompok (GroupKFold)")):
        for f in range(5):
            for i in range(n):
                w = JINGGA if lipat[i] == f else BIRU_MUDA
                ax.add_patch(Rectangle((i, 4 - f), 0.9, 0.8,
                                       facecolor=w, edgecolor="white",
                                       lw=0.3))
        for b in range(1, 8):
            ax.axvline(5 * b - 0.05, color=ABU, lw=0.4)
        ax.set_xlim(-0.5, n)
        ax.set_ylim(-0.3, 5)
        ax.set_title(judul)
        ax.axis("off")
    simpan(fig, "bab03-validasi")



def bab15_daun():
    """Jarak ke titik dan simpul yang dikunjungi terhadap ukuran daun."""
    from bab15_pohon import Pohon
    rng = np.random.default_rng(BENIH)
    X = rng.random((10_000, 4))
    Q = rng.random((100, 4))
    daun = [1, 2, 5, 10, 20, 40, 80, 160, 320]
    j, s_ = [], []
    for d in daun:
        T = Pohon(X, d, False)
        a = b = 0
        for q in Q:
            T.query(q, k=5)
            a += T.n_jarak
            b += T.n_simpul
        j.append(a / len(Q))
        s_.append(b / len(Q))
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ax.plot(daun, j, "o-", ms=2.5, color=BIRU, label="jarak ke titik")
    ax.plot(daun, s_, "s-", ms=2.5, color=JINGGA,
            label="simpul dikunjungi")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ukuran daun")
    ax.set_ylabel("per kueri")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab15-daun")


# ---------------------------------------------------------------------
#  Fungsi gambar bab baru disisipkan DI ATAS blok ini.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    pola = re.compile(r"^bab\d\d_")
    pilihan = sys.argv[1:]
    fungsi = [(k, v) for k, v in sorted(globals().items())
              if pola.match(k) and callable(v)]
    if pilihan:
        fungsi = [(k, v) for k, v in fungsi
                  if any(p in k for p in pilihan)]
    if not fungsi:
        print("tidak ada gambar yang cocok")
    for _, f in fungsi:
        f()
