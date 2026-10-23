"""입력 검사와 대각 스케일링.

대칭 대각 스케일링  K~ = D K D,  M~ = D M D  (D = diag(1/sqrt(K_ii)))
은 합동 변환이므로 고유치와 Sturm 개수를 바꾸지 않으며,  phi = D psi  로 복원한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = np.finfo(float).eps


class InputError(ValueError):
    pass


@dataclass
class Problem:
    K: np.ndarray            # 스케일링된 강성
    M: np.ndarray            # 스케일링된 질량
    d: np.ndarray            # 스케일 계수 (phi = d * psi)
    K_orig: np.ndarray
    M_orig: np.ndarray
    rank_M: int
    massless_dofs: np.ndarray        # M_ii == 0 인 DOF (0-based)
    lam_scale: float                 # 고유치 크기 척도 (median K_ii/M_ii)
    warnings: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.K.shape[0]


def _symmetrize(A: np.ndarray, name: str, warnings: list[str]) -> np.ndarray:
    norm = np.max(np.abs(A))
    if norm == 0.0:
        return A
    asym = np.max(np.abs(A - A.T)) / norm
    if asym > 1e-8:
        raise InputError(f"{name} 가 대칭이 아닙니다 (상대 비대칭 {asym:.2e})")
    if asym > 0.0:
        warnings.append(f"{name}: 미소 비대칭 {asym:.1e} -> (A+A^T)/2 로 대칭화")
    return 0.5 * (A + A.T)


def prepare(K: np.ndarray, M: np.ndarray) -> Problem:
    warnings: list[str] = []
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise InputError("K 는 정사각 행렬이어야 합니다")
    if M.shape != K.shape:
        raise InputError(f"K{K.shape} 와 M{M.shape} 의 크기가 다릅니다")
    K = _symmetrize(K, "K", warnings)
    M = _symmetrize(M, "M", warnings)
    n = K.shape[0]

    kd, md = np.diag(K).copy(), np.diag(M).copy()
    if np.any(kd < 0) or np.any(md < 0):
        raise InputError("K 또는 M 의 대각항에 음수가 있습니다")

    # 스케일 계수: K_ii > 0 이면 1/sqrt(K_ii), 아니면 M 기준, 둘 다 0 이면 1
    d = np.ones(n)
    pos_k = kd > 0
    d[pos_k] = 1.0 / np.sqrt(kd[pos_k])
    only_m = ~pos_k & (md > 0)
    d[only_m] = 1.0 / np.sqrt(md[only_m])
    if np.any(~pos_k):
        warnings.append(f"K 대각항이 0 인 DOF {int(np.sum(~pos_k))}개 (구속/연결 확인 필요)")

    Ks = d[:, None] * K * d[None, :]
    Ms = d[:, None] * M * d[None, :]

    # 정부호성 검사 (n ~ 200 이므로 고유치로 직접 확인)
    ek = np.linalg.eigvalsh(Ks)
    em = np.linalg.eigvalsh(Ms)
    if em[0] < -1e-10 * max(em[-1], EPS):
        raise InputError(f"M 이 양의 준정부호가 아닙니다 (최소 고유치 {em[0]:.3e})")
    if em[-1] <= 0:
        raise InputError("M 이 영행렬입니다")
    if ek[0] < -1e-8 * ek[-1]:
        warnings.append(f"K 가 음의 고유치를 가집니다 ({ek[0]:.3e}): 불안정 구조일 수 있음")
    rank_M = int(np.sum(em > 1e-12 * em[-1]))

    massless = np.where(md == 0.0)[0]
    if rank_M < n:
        warnings.append(
            f"M 이 특이합니다: rank(M)={rank_M} < n={n} "
            f"(질량 0 대각 DOF {len(massless)}개) -> 유한 고유치는 최대 {rank_M}개")

    # 결합되지 않은 DOF (K, M 행이 모두 0) 는 고유치가 정의되지 않는다
    dead = np.where(~np.any(K != 0, axis=1) & ~np.any(M != 0, axis=1))[0]
    if len(dead):
        raise InputError(f"K, M 행이 모두 0 인 DOF 가 있습니다 (1-based): {list(dead + 1)[:10]}")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = kd[md > 0] / md[md > 0]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    lam_scale = float(np.median(ratio)) if ratio.size else 1.0

    return Problem(K=Ks, M=Ms, d=d, K_orig=K, M_orig=M, rank_M=rank_M,
                   massless_dofs=massless, lam_scale=lam_scale, warnings=warnings)
