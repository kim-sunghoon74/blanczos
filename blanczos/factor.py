"""K - sigma*M 의 LDL^T 분해, 블록 해법, Sylvester 관성(Sturm 개수).

scipy.linalg.ldl 은 Bunch-Kaufman 대칭 피벗팅을 사용하므로 부정부호 행렬도 안정적으로 분해한다.
A = lu @ D @ lu.T,  lu[perm] 은 단위 하삼각,  D 는 1x1 / 2x2 블록 대각.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import ldl, solve_triangular

EPS = np.finfo(float).eps


class SingularShiftError(RuntimeError):
    """이동점이 고유치에 너무 가까워 분해가 사실상 특이한 경우."""


class Factorization:
    def __init__(self, K: np.ndarray, M: np.ndarray, sigma: float,
                 sing_tol: float = 1e-13):
        self.sigma = float(sigma)
        A = K - self.sigma * M
        lu, D, perm = ldl(A, lower=True, hermitian=True)
        self._L = lu[perm]
        self._perm = perm
        self._D = D

        # D 의 블록 구조 파악 + 블록별 고유치로 관성 계산
        n = A.shape[0]
        blocks, d_eigs = [], []
        i = 0
        while i < n:
            if i + 1 < n and D[i + 1, i] != 0.0:
                blk = D[i:i + 2, i:i + 2]
                blocks.append((i, 2, np.linalg.inv(blk)))
                d_eigs.extend(np.linalg.eigvalsh(blk))
                i += 2
            else:
                blocks.append((i, 1, D[i, i]))
                d_eigs.append(D[i, i])
                i += 1
        self._blocks = blocks
        d_eigs = np.asarray(d_eigs)
        self.n_negative = int(np.sum(d_eigs < 0))   # = sigma 보다 작은 고유치 개수
        amax = np.max(np.abs(d_eigs))
        self.pivot_ratio = float(np.min(np.abs(d_eigs)) / amax) if amax > 0 else 0.0
        if not np.all(np.isfinite(d_eigs)) or self.pivot_ratio < sing_tol:
            raise SingularShiftError(
                f"sigma={self.sigma:.6e}: 최소/최대 피벗 비 {self.pivot_ratio:.2e} < {sing_tol:.0e}")

    def solve(self, B: np.ndarray) -> np.ndarray:
        """(K - sigma M) X = B 를 블록 우변 B (n x p) 에 대해 푼다."""
        vec = B.ndim == 1
        if vec:
            B = B[:, None]
        Z = solve_triangular(self._L, B[self._perm], lower=True, unit_diagonal=True,
                             check_finite=False)
        for i, size, inv in self._blocks:
            if size == 1:
                Z[i] /= inv
            else:
                Z[i:i + 2] = inv @ Z[i:i + 2]
        W = solve_triangular(self._L, Z, lower=True, trans="T", unit_diagonal=True,
                             check_finite=False)
        X = np.empty_like(W)
        X[self._perm] = W
        return X[:, 0] if vec else X


def factor_near(K: np.ndarray, M: np.ndarray, sigma: float, scale: float,
                log: list[str] | None = None, max_tries: int = 8) -> Factorization:
    """sigma 에서 분해하되, 특이하면 이동점을 조금씩 옮겨 다시 분해한다.

    Nastran 은 피벗을 섭동하지만, 여기서는 이동점 자체를 옮긴다. 이렇게 하면 Sturm 개수가
    항상 '옮긴 sigma' 에 대해 정확하게 유지된다.
    """
    step = 1e-6 * max(abs(sigma), scale)
    for k in range(max_tries):
        # sigma, sigma+h, sigma-4h, sigma+16h, ...
        s = sigma if k == 0 else sigma + (1 if k % 2 else -1) * step
        try:
            F = Factorization(K, M, s)
            if k and log is not None:
                log.append(f"  특이 이동점 회피: sigma {sigma:.6e} -> {s:.6e}")
            return F
        except SingularShiftError:
            step *= 4.0
    raise SingularShiftError(f"sigma={sigma:.6e} 근처에서 정칙 분해를 얻지 못했습니다")
