"""단일 이동점 Shifted Block Lanczos.

연산자 OP = (K - sigma M)^{-1} M 은 M-내적에 대해 대칭이므로 모든 직교화는 M-내적으로 한다.
블록 3항 점화식:  OP Q_j = Q_{j-1} B_j^T + Q_j A_j + Q_{j+1} B_{j+1}

재직교화 모드
- "full": 매 스텝 모든 이전 블록에 대해 2-pass 재직교화 (기준 구현)
- "pro" : 블록 omega 점화식으로 직교성 손실을 추정해 sqrt(eps) 초과 시에만 재직교화
          (Simon 의 Partial Reorthogonalization, Grimes-Lewis-Simon 1994 블록판)
          + 선택적 직교화 (Parlett-Scott): 잘 수렴한 리츠 벡터 방향을 매 스텝 제거
- "none": 국부(직전 두 블록) 재직교화만 -> 유령 고유치 시연용
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .orth import block_mqr, project_out

EPS = np.finfo(float).eps


@dataclass
class LanczosResult:
    sigma: float
    lam: np.ndarray            # 수렴한 리츠 값 -> 원 문제 고유치 (sigma + 1/theta)
    vecs: np.ndarray           # 정제된 고유벡터 (스케일 공간, M-정규화)
    bounds: np.ndarray         # theta 의 오차 한계 ||B_{j+1} s_last||
    theta: np.ndarray
    n_blocks: int
    n_reorth: int              # 전역 재직교화 발생 횟수
    n_deflated: int
    orth_loss: float           # 실제 max|Q^T M Q - I|
    all_theta: np.ndarray = None   # T 의 모든 리츠 값 (미수렴 포함) - 다음 이동점 선택용
    all_bounds: np.ndarray = None
    omega_hist: list = field(default_factory=list)   # (step, 추정 max omega, 실제 max) - debug
    stop_reason: str = ""


def block_lanczos(solve: Callable[[np.ndarray], np.ndarray], M: np.ndarray,
                  sigma: float, p: int, rng: np.random.Generator,
                  locked: tuple[np.ndarray, np.ndarray] | None = None,
                  max_blocks: int = 20, min_dim: int = 30, n_eff: int | None = None,
                  reorth: str = "pro", selective: bool = True, tol: float = 1e-10,
                  need: int | None = None, accept: Callable[[np.ndarray], np.ndarray] | None = None,
                  debug: bool = False) -> LanczosResult:
    n = M.shape[0]
    if reorth not in ("full", "pro", "none"):
        raise ValueError(f"unknown reorth mode {reorth}")
    n_eff = n if n_eff is None else n_eff
    p = max(1, min(p, n_eff))
    Lq, Lm = locked if locked is not None and locked[0].shape[1] else (None, None)
    sqeps, eta = np.sqrt(EPS), EPS ** 0.75

    def op_from_m(MX):                      # OP X = (K - sigma M)^{-1} (M X)
        return solve(MX)

    def purify(x):                          # range(OP) 로 투영 + 잠금 벡터 제거
        y = solve(M @ x)
        if Lq is not None:
            y = project_out(y[:, None] if y.ndim == 1 else y, Lq, Lm)
            y = y[:, 0] if x.ndim == 1 else y
        return y

    def all_basis(Qall, MQall):
        if Lq is None:
            return Qall, MQall
        return np.hstack([Lq, Qall]), np.hstack([Lm, MQall])

    # --- 시작 블록: 난수 -> OP 적용(무질량/무한대 고유치 성분 제거) -> M-QR
    R0 = purify(rng.standard_normal((n, p)))
    Q1, MQ1, _, n_defl, exhausted = block_mqr(R0, M, 0.0, (Lq, Lm) if Lq is not None else None,
                                              rng)
    Qs, MQs = [Q1], [MQ1]
    As: list[np.ndarray] = []
    Bs: list[np.ndarray] = [np.zeros((p, p))]   # Bs[i]: R_{i-1} = Q_i Bs[i]
    Qall, MQall = Q1.copy(), MQ1.copy()

    tnorm = 0.0
    n_reorth = 0
    forced: set[int] = set()
    Wp: list[np.ndarray] = []                    # W[k][j-1]
    Wc: list[np.ndarray] = [np.eye(p)]           # W[k][j]
    omega_hist = []
    Y = Ym = None                                # 선택적 직교화용 '좋은' 리츠 벡터
    stop_reason = "max_blocks"
    final_pass = False
    T = S = theta = bounds = Bn = Qn = None

    def noise():
        return EPS * np.sqrt(n) * rng.standard_normal((p, p)) / np.sqrt(p)

    for j in range(n_eff // p + 2):
        Qj, MQj = Qs[j], MQs[j]
        R = op_from_m(MQj)
        if j:
            R -= Qs[j - 1] @ Bs[j].T
        A = MQj.T @ R
        A = 0.5 * (A + A.T)
        R -= Qj @ A
        # 국부 재직교화 (항상): 직전 두 블록
        for k in (j - 1, j):
            if k >= 0:
                R -= Qs[k] @ (MQs[k].T @ R)
        if Lq is not None:
            project_out(R, Lq, Lm)
        if selective and Y is not None:
            project_out(R, Y, Ym, passes=1)
        As.append(A)
        tnorm = max(tnorm, np.linalg.norm(A, 2), np.linalg.norm(Bs[j], 2))

        if reorth == "full":
            project_out(R, Qall, MQall)
        elif reorth == "pro" and forced:
            idx = sorted(forced)
            Qf = np.hstack([Qs[k] for k in idx]); MQf = np.hstack([MQs[k] for k in idx])
            project_out(R, Qf, MQf)

        bq, bm = all_basis(Qall, MQall)
        Qn, MQn, Bn, nd, exhausted = block_mqr(R, M, tnorm, (bq, bm), rng)
        n_defl += nd

        if reorth == "pro":
            Wn = _omega_update(j, As, Bs, Bn, Wc, Wp, tnorm, noise)
            trigger: set[int] = set()
            if Wn is None or nd:
                trigger = set(range(max(j - 1, 0)))       # B 특이 -> 전 블록
            else:
                om = [np.linalg.norm(Wn[k], 2) for k in range(max(j - 1, 0))]
                if om and max(om) > sqeps:
                    trigger = {k for k, w in enumerate(om) if w > eta}
            if trigger:
                n_reorth += 1
                idx = sorted(trigger)
                Qf = np.hstack([Qs[k] for k in idx]); MQf = np.hstack([MQs[k] for k in idx])
                project_out(R, Qf, MQf)
                Qn, MQn, Bn, nd2, exhausted = block_mqr(R, M, tnorm, (bq, bm), rng)
                n_defl += nd2
                if Wn is None:
                    Wn = [noise() for _ in range(j + 1)] + [np.eye(p)]
                for k in trigger:
                    Wn[k] = noise()
            for k in forced:                                # 직전 스텝 강제 재직교화 반영
                Wn[k] = noise()
            forced = trigger                                # Simon 규칙: 다음 스텝도 같은 집합
            Wp, Wc = Wc, Wn
            if debug:
                Qtmp = np.hstack([Qall, Qn])
                actual = np.max(np.abs(Qtmp.T @ (M @ Qtmp) - np.eye(Qtmp.shape[1])))
                est = max((np.linalg.norm(Wn[k], 2) for k in range(j + 1)), default=0.0)
                omega_hist.append((j + 1, est, actual))

        # --- 수렴 검사:  T_{j+1} (크기 (j+1)p) 의 리츠 쌍
        T = _assemble_T(As, Bs)
        theta, S = np.linalg.eigh(T)
        bounds = np.linalg.norm(Bn @ S[-p:, :], axis=0)
        ok = (np.abs(theta) > 1e-14 * tnorm) & (bounds <= tol * np.abs(theta))
        if selective and reorth == "pro":
            good = (np.abs(theta) > 1e-14 * tnorm) & (bounds <= sqeps * np.abs(theta))
            if np.any(good):
                Y = Qall @ S[:, good]
                Ym = MQall @ S[:, good]
            else:
                Y = Ym = None

        lam = sigma + 1.0 / theta[ok]
        cnt = int(np.sum(accept(lam))) if accept is not None else len(lam)
        if need is not None and cnt >= need:
            stop_reason = "need_reached"
            break
        if final_pass:
            stop_reason = "subspace_full"
            break
        if exhausted or (j + 2) * p > n_eff:
            # 남은 유효 차원이 p 의 배수가 아니면 마지막 블록은 일부 열만 독립이다.
            # 그 부분 블록까지 T 에 넣어야 T 가 공간 전체를 덮어 모든 리츠 값이 수렴한다.
            if not Qn.any():
                stop_reason = "subspace_full"
                break
            final_pass = True
        # max_blocks 는 소프트 상한: 수렴 모드가 min(need, p)개 미만이거나 크릴로프 차원이
        # min_dim 미만이면 계속 진행 (분해 1회당 얻는 모드 수를 확보). 최종 상한은 유효 차원 소진.
        min_conv = min(need, p) if need is not None else 0
        if j + 1 >= max_blocks and cnt >= min_conv and (j + 1) * p >= min(min_dim, n_eff):
            break

        Qs.append(Qn); MQs.append(MQn); Bs.append(Bn)
        Qall = np.hstack([Qall, Qn]); MQall = np.hstack([MQall, MQn])

    # --- 수렴한 리츠 벡터 복원 + 정제: phi = OP(Q s) / theta  (역반복 1회)
    #     점화식이 정확하면 Q s + Q_{j+1} B_{j+1} s_last / theta 와 같지만, 재직교화/잠금/
    #     블록 대체가 섞이면 근사가 되므로 OP 를 직접 적용한다. M 이 특이할 때 M-노름으로 보이지
    #     않는 무질량 방향 오차를 이 단계가 제거한다.
    m = T.shape[0]
    ok = (np.abs(theta) > 1e-14 * tnorm) & (bounds <= tol * np.abs(theta))
    Sk = S[:, ok]
    th = theta[ok]
    X = solve(MQall[:, :m] @ Sk) / th
    if Lq is not None:
        project_out(X, Lq, Lm)
    MX = M @ X
    nrm = np.sqrt(np.maximum(np.einsum("ij,ij->j", X, MX), 0.0))
    X = X / nrm

    live = np.any(Qall[:, :m] != 0.0, axis=0)        # 소진으로 비워 둔 열 제외
    G = Qall[:, :m][:, live].T @ MQall[:, :m][:, live]
    orth_loss = float(np.max(np.abs(G - np.eye(G.shape[0]))))
    return LanczosResult(sigma=sigma, lam=sigma + 1.0 / th, vecs=X, bounds=bounds[ok],
                         theta=th, n_blocks=len(As), n_reorth=n_reorth, n_deflated=n_defl,
                         orth_loss=orth_loss, all_theta=theta, all_bounds=bounds,
                         omega_hist=omega_hist, stop_reason=stop_reason)


def _assemble_T(As, Bs) -> np.ndarray:
    nb, p = len(As), As[0].shape[0]
    T = np.zeros((nb * p, nb * p))
    for i, A in enumerate(As):
        T[i * p:(i + 1) * p, i * p:(i + 1) * p] = A
        if i:
            B = Bs[i]
            T[i * p:(i + 1) * p, (i - 1) * p:i * p] = B
            T[(i - 1) * p:i * p, i * p:(i + 1) * p] = B.T
    return T


def _omega_update(j, As, Bs, Bn, Wc, Wp, tnorm, noise):
    """W[k][j+1] = (B_{k+1}^T W[k+1][j] + A_k W[k][j] + B_k W[k-1][j]
                    - W[k][j] A_j - W[k][j-1] B_j^T + 잡음) B_{j+1}^{-1}

    W[a][b] ~ Q_a^T M Q_b 의 추정치. 반올림 오차를 모사하는 eps 크기 잡음을 주입한다.
    B_{j+1} 이 특이하면 None.
    """
    p = Bn.shape[0]
    try:
        cond = np.linalg.cond(Bn)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(cond) or cond > 1.0 / EPS:
        return None
    Binv = np.linalg.inv(Bn)
    Wn = []
    for k in range(j - 1):
        term = Bs[k + 1].T @ Wc[k + 1] + As[k] @ Wc[k] - Wc[k] @ As[j] - Wp[k] @ Bs[j].T
        if k:
            term += Bs[k] @ Wc[k - 1]
        Wn.append((term + tnorm * noise()) @ Binv)
    Wn += [noise() for _ in range(min(j + 1, 2))]      # W[j-1][j+1], W[j][j+1]: 국부 재직교화
    Wn.append(np.eye(p))
    return Wn
