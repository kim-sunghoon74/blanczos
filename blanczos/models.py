"""검증용 예제 모델 생성기 (K, M 반환)."""

from __future__ import annotations

import itertools

import numpy as np


def spring_chain(n: int = 200, k: float = 1.0e4, m: float = 1.0):
    """고정-자유 스프링-질량 체인. 해석해: lam_j = 4k/m sin^2((2j-1)pi/(2(2n+1)))."""
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k
        if i + 1 < n:
            K[i, i] += k
            K[i, i + 1] = K[i + 1, i] = -k
    M = m * np.eye(n)
    j = np.arange(1, n + 1)
    exact = 4 * k / m * np.sin((2 * j - 1) * np.pi / (2 * (2 * n + 1))) ** 2
    return K, M, exact


def beam2d_free(n_elem: int = 99, length: float = 10.0, EI: float = 1.0e6,
                rhoA: float = 10.0, lumped: bool = True):
    """자유-자유 2D Euler-Bernoulli 빔 (DOF: 절점당 [w, theta]).

    lumped=True 이면 회전 DOF 질량이 0 -> M 특이 (무질량 회전 DOF). 강체 모드 2개.
    """
    nn = n_elem + 1
    n = 2 * nn
    Le = length / n_elem
    ke = EI / Le ** 3 * np.array([[12, 6 * Le, -12, 6 * Le],
                                  [6 * Le, 4 * Le ** 2, -6 * Le, 2 * Le ** 2],
                                  [-12, -6 * Le, 12, -6 * Le],
                                  [6 * Le, 2 * Le ** 2, -6 * Le, 4 * Le ** 2]])
    if lumped:
        me = rhoA * Le / 2 * np.diag([1.0, 0.0, 1.0, 0.0])
    else:
        me = rhoA * Le / 420 * np.array([[156, 22 * Le, 54, -13 * Le],
                                         [22 * Le, 4 * Le ** 2, 13 * Le, -3 * Le ** 2],
                                         [54, 13 * Le, 156, -22 * Le],
                                         [-13 * Le, -3 * Le ** 2, -22 * Le, 4 * Le ** 2]])
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    for e in range(n_elem):
        idx = np.arange(2 * e, 2 * e + 4)
        K[np.ix_(idx, idx)] += ke
        M[np.ix_(idx, idx)] += me
    return K, M


def frame3d_free(nx: int = 3, ny: int = 3, nz: int = 4, spacing: float = 1.0,
                 E: float = 2.1e11, G: float = 8.0e10, A: float = 1.0e-3,
                 I: float = 1.0e-6, J: float = 2.0e-6, rho: float = 7850.0,
                 rot_inertia: bool = False):
    """자유-자유 3D 격자 프레임 (절점당 6 DOF).

    - 구속이 없으므로 강체 모드 6개
    - x-y 단면이 정사각 대칭이고 Iy = Iz 이므로 굽힘 모드에 중근이 생긴다
    - rot_inertia=False 이면 회전 DOF 질량 0 (집중 질량만) -> M 특이
    기본값 3x3x4 = 36 절점 -> 216 DOF
    """
    nodes = np.array([(i * spacing, j * spacing, k * spacing)
                      for k in range(nz) for j in range(ny) for i in range(nx)], dtype=float)
    nid = {tuple(np.round(p / spacing).astype(int)): a for a, p in enumerate(nodes)}
    elems = []
    for (i, j, k), a in nid.items():
        for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            b = nid.get((i + d[0], j + d[1], k + d[2]))
            if b is not None:
                elems.append((a, b))

    n = 6 * len(nodes)
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    for a, b in elems:
        L = np.linalg.norm(nodes[b] - nodes[a])
        T = _frame_transform(nodes[a], nodes[b])
        ke = T.T @ _frame_local_k(E, G, A, I, I, J, L) @ T
        idx = np.r_[6 * a:6 * a + 6, 6 * b:6 * b + 6]
        K[np.ix_(idx, idx)] += ke
        m_half = rho * A * L / 2
        for nd in (a, b):
            M[6 * nd:6 * nd + 3, 6 * nd:6 * nd + 3] += m_half * np.eye(3)
            if rot_inertia:
                M[6 * nd + 3:6 * nd + 6, 6 * nd + 3:6 * nd + 6] += m_half * (I / A) * np.eye(3)
    return K, M


def random_pencil(n: int = 200, seed: int = 1, cluster: bool = True):
    """임의 SPD K, SPD M (중근 군집 포함 옵션). 정답 고유치를 함께 반환."""
    rng = np.random.default_rng(seed)
    lam = np.sort(rng.uniform(1.0, 1.0e4, n))
    if cluster:
        lam[10:13] = lam[10]                    # 3중근
        lam[30:32] = lam[30] * (1 + 1e-9)       # 근접 군집
    X = np.linalg.qr(rng.standard_normal((n, n)))[0] @ np.diag(rng.uniform(0.5, 2.0, n))
    # K = X^-T diag(lam) X^-1, M = X^-T X^-1  ->  K x = lam M x  with x = columns of X
    Xi = np.linalg.inv(X)
    K = Xi.T @ np.diag(lam) @ Xi
    M = Xi.T @ Xi
    return 0.5 * (K + K.T), 0.5 * (M + M.T), lam


def _frame_local_k(E, G, A, Iy, Iz, J, L):
    k = np.zeros((12, 12))
    EA, GJ = E * A / L, G * J / L
    k[0, 0] = k[6, 6] = EA
    k[0, 6] = k[6, 0] = -EA
    k[3, 3] = k[9, 9] = GJ
    k[3, 9] = k[9, 3] = -GJ
    for (v, t, Ib, s) in ((1, 5, Iz, 1.0), (2, 4, Iy, -1.0)):
        c = E * Ib / L ** 3
        vv, tt = np.array([v, v + 6]), np.array([t, t + 6])
        sub = c * np.array([[12, s * 6 * L, -12, s * 6 * L],
                            [s * 6 * L, 4 * L ** 2, -s * 6 * L, 2 * L ** 2],
                            [-12, -s * 6 * L, 12, -s * 6 * L],
                            [s * 6 * L, 2 * L ** 2, -s * 6 * L, 4 * L ** 2]])
        ids = [vv[0], tt[0], vv[1], tt[1]]
        for r, c_ in itertools.product(range(4), range(4)):
            k[ids[r], ids[c_]] = sub[r, c_]
    return k


def _frame_transform(pa, pb):
    e1 = (pb - pa) / np.linalg.norm(pb - pa)
    ref = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e3 = np.cross(e1, ref)
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    lam = np.vstack([e1, e2, e3])
    T = np.zeros((12, 12))
    for i in range(4):
        T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = lam
    return T


def reference_eigs(K, M, delta: float | None = None):
    """특이 M 에도 쓸 수 있는 밀집 기준 해: eigh(M, K + delta M) -> mu = 1/(lam + delta)."""
    from scipy.linalg import eigh
    if delta is None:
        delta = 1e-3 * np.median(np.diag(K)[np.diag(M) > 0] / np.diag(M)[np.diag(M) > 0])
    mu, X = eigh(M, K + delta * M)
    keep = mu > 1e-12 * mu.max()
    lam = 1.0 / mu[keep] - delta
    order = np.argsort(lam)
    return lam[order], X[:, keep][:, order]
