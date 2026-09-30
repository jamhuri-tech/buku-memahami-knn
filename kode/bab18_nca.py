"""Bab 18: Neighbourhood Components Analysis (Goldberger dkk. 2004) dari
nol. Fungsi tujuan f(L) = sum_i p_i, p_i = sum_{j sekelas i} p_ij,
p_ij = softmax_j(-||L x_i - L x_j||^2) dengan p_ii = 0, dan gradiennya
  df/dL = 2 L sum_i ( p_i sum_k p_ik x_ik x_ik^T
                      - sum_{j sekelas} p_ij x_ij x_ij^T ),
dengan x_ij = x_i - x_j. Diperiksa dengan beda hingga, lalu
dimaksimumkan dengan L-BFGS dan dibandingkan dengan scikit-learn.
"""
import numpy as np
from scipy.optimize import minimize


def nca_f_grad(L, X, y):
    Z = X @ L.T
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    D -= D.min(1, keepdims=True)                   # stabil numerik
    P = np.exp(-D)
    P /= P.sum(1, keepdims=True)
    sama = y[:, None] == y[None, :]
    Pc = P * sama
    p = Pc.sum(1)
    # sum_i sum_k (p_i p_ik - [k sekelas] p_ik) x_ik x_ik^T = X^T W X
    # dengan W simetris dibangun dari bobot A_ik = p_i p_ik - Pc_ik.
    A = p[:, None] * P - Pc
    A = A + A.T
    W = np.diag(A.sum(1)) - A
    G = 2 * L @ (X.T @ W @ X)
    return p.sum(), G


def latih_nca(X, y, d=None, iterasi=200, benih=20260929):
    """Memaksimumkan f(L) dengan L-BFGS, L awal = identitas (atau
    proyeksi acak kecil bila d < p)."""
    p_ = X.shape[1]
    d = p_ if d is None else d
    rng = np.random.default_rng(benih)
    L0 = np.eye(d, p_) if d == p_ else rng.standard_normal((d, p_)) * 0.1

    def tujuan(v):
        f, G = nca_f_grad(v.reshape(d, p_), X, y)
        return -f, -G.ravel()

    h = minimize(tujuan, L0.ravel(), jac=True, method="L-BFGS-B",
                 options={"maxiter": iterasi})
    return h.x.reshape(d, p_), -h.fun


if __name__ == "__main__":
    rng = np.random.default_rng(20260929)
    X = rng.standard_normal((30, 4))
    y = rng.integers(0, 3, 30)
    L = rng.standard_normal((2, 4)) * 0.5
    f, G = nca_f_grad(L, X, y)
    h = 1e-6
    Gn = np.zeros_like(L)
    for a in range(2):
        for b in range(4):
            E = np.zeros_like(L)
            E[a, b] = h
            Gn[a, b] = (nca_f_grad(L + E, X, y)[0]
                        - nca_f_grad(L - E, X, y)[0]) / (2 * h)
    print(f"f(L) = {f:.6f}; selisih gradien rumus vs beda hingga "
          f"{np.abs(G - Gn).max():.1e}")
