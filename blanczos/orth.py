"""M-내적 기반 블록 직교화.

- Two-pass Modified Gram-Schmidt ("twice is enough")
- 열 피벗팅 (M-노름이 가장 큰 열부터) 으로 랭크 결핍 감지
- 결핍 열은 기존 기저와 M-직교인 난수 벡터로 대체 (블록 축소, Deflation)
"""

from __future__ import annotations

import numpy as np

EPS = np.finfo(float).eps


def project_out(X: np.ndarray, Q: np.ndarray | None, MQ: np.ndarray | None,
                passes: int = 2) -> np.ndarray:
    """X 에서 M-정규직교 기저 Q 성분을 제거한다 (제자리 수정 후 반환)."""
    if Q is None or Q.shape[1] == 0:
        return X
    for _ in range(passes):
        X -= Q @ (MQ.T @ X)
    return X


def m_norm(x: np.ndarray, mx: np.ndarray) -> float:
    return float(np.sqrt(max(x @ mx, 0.0)))


def block_mqr(R: np.ndarray, M: np.ndarray, ref_norm: float,
              basis: tuple[np.ndarray, np.ndarray] | None,
              rng: np.random.Generator,
              defl_tol: float = 1e-12):
    """R = Q @ B,  Q^T M Q = I  가 되도록 분해한다.

    피벗팅 때문에 B 는 '열이 치환된' 상삼각 행렬이다 (블록 삼중대각 T 를 만드는 데는 무관).

    Returns
    -------
    Q, MQ, B, n_deflated, exhausted
        exhausted=True 이면 대체 난수 벡터조차 기존 기저와 독립이 되지 않음
        (크릴로프 공간이 유효 차원을 모두 채움).
    """
    n, p = R.shape
    W = R.copy()
    MW = M @ W
    norms0 = np.sqrt(np.maximum(np.einsum("ij,ij->j", W, MW), 0.0))
    ref = max(ref_norm, float(np.max(norms0)) if p else 0.0, np.finfo(float).tiny)
    Bq, Bm = (basis if basis is not None else (None, None))

    Q = np.zeros((n, p))
    MQ = np.zeros((n, p))
    B = np.zeros((p, p))
    remaining = list(range(p))
    n_defl = 0
    exhausted = False

    for k in range(p):
        cur = [m_norm(W[:, c], MW[:, c]) for c in remaining]
        c = remaining.pop(int(np.argmax(cur)))
        v, mv = W[:, c], MW[:, c]
        # 2nd pass (left-looking): 이미 만든 q 들에 대해 한 번 더
        if k:
            coef = MQ[:, :k].T @ v
            v -= Q[:, :k] @ coef
            mv -= MQ[:, :k] @ coef
            B[:k, c] += coef
        nk = m_norm(v, mv)

        if nk <= max(defl_tol * ref, 1e2 * EPS * norms0[c]):
            # 랭크 결핍: B 의 대각을 0 으로 두고 직교하는 새 난수 벡터로 대체
            n_defl += 1
            B[k, c] = 0.0
            # 대체 벡터는 OP 로 정제하지 않는다: shift 근처 방향이 증폭되어 남는 성분이
            # 지나치게 작아지기 때문. 무질량 방향의 잔여 성분은 마지막 고유벡터 정제가 제거한다.
            w = None
            for _ in range(3):
                cand = rng.standard_normal(n)
                n_before = m_norm(cand, M @ cand)
                for _ in range(2):
                    cand = project_out(cand[:, None], Bq, Bm)[:, 0]
                    cand = project_out(cand[:, None], Q[:, :k], MQ[:, :k])[:, 0]
                mc = M @ cand
                nc = m_norm(cand, mc)
                if n_before > 0 and nc > 1e-8 * n_before:
                    w, mw = cand / nc, mc / nc
                    break
            if w is None:
                exhausted = True
                continue
            Q[:, k], MQ[:, k] = w, mw
        else:
            B[k, c] = nk
            Q[:, k], MQ[:, k] = v / nk, mv / nk

        # 1st pass (right-looking): 남은 열에서 q_k 성분 제거
        if remaining and not (exhausted and not Q[:, k].any()):
            idx = np.array(remaining)
            coef = MQ[:, k] @ W[:, idx]
            W[:, idx] -= np.outer(Q[:, k], coef)
            MW[:, idx] -= np.outer(MQ[:, k], coef)
            B[k, idx] += coef

    return Q, MQ, B, n_defl, exhausted
