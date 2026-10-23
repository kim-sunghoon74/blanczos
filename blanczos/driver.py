"""구간 [f1, f2] 내 모든 모드 추출 드라이버 (Nastran EIGRL V1/V2 방식).

1. 구간 양 끝에서 LDL^T 분해 -> Sturm 개수로 구간 내 고유치 총수 N 을 확정
2. 왼쪽 끝에서부터 이동점을 옮겨 가며 Block Lanczos 수행 (이미 찾은 모드는 잠금/직교화)
3. 다음 이동점 x 에서의 Sturm 개수로 [L, x) 를 '신뢰 구간' 으로 확정
4. 개수가 모자라면 이분법으로 누락 위치를 좁혀 그곳에 이동점을 두고 재탐색
5. 찾은 모든 벡터로 Rayleigh-Ritz -> M-직교성 보장, 원 좌표 복원, 잔차/개수 최종 검증
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .factor import Factorization, factor_near
from .lanczos import LanczosResult, block_lanczos
from .orth import project_out
from .preprocess import Problem, prepare

TWO_PI = 2.0 * np.pi


@dataclass
class EigenOptions:
    f2: float                       # 상한 주파수 [Hz] (필수)
    f1: float | None = None         # 하한 주파수 [Hz]; None/0 이하 -> 강체 모드 포함 (음수 미세 이동)
    block: int = 7                  # 블록 크기 p
    reorth: str = "pro"             # pro | full | none
    selective: bool = True
    tol: float = 1e-10              # 리츠 값 수렴 허용치 (bound/|theta|)
    max_blocks: int = 20            # 이동점 하나당 최대 블록 스텝
    max_shifts: int = 40
    seed: int = 0
    debug: bool = False


@dataclass
class ShiftRecord:
    sigma: float
    n_below: int                    # Sturm 개수 nu(sigma)
    purpose: str
    blocks: int = 0
    new_modes: int = 0
    reorth: int = 0
    deflated: int = 0
    orth_loss: float = 0.0
    stop: str = ""


@dataclass
class EigenResult:
    lam: np.ndarray                 # 고유치 omega^2
    phi: np.ndarray                 # 원 좌표 모드 형상 (질량 정규화, n x m)
    extraction_order: np.ndarray
    gen_mass: np.ndarray
    gen_stiff: np.ndarray
    residual: np.ndarray            # ||K phi - lam M phi|| / ((||K|| + |lam| ||M||) ||phi||)
    lam_range: tuple[float, float]  # 실제 Sturm 검사에 쓰인 구간
    sturm_count: int
    shifts: list[ShiftRecord]
    runs: list[LanczosResult]
    ghosts_removed: int
    relaxed: int                    # 완화 기준(잔차 1e-6)으로 채택된 모드 수
    m_orth_error: float             # max|Phi^T M Phi - I|
    problem: Problem
    log: list[str] = field(default_factory=list)

    @property
    def omega(self):
        return np.sqrt(np.abs(self.lam))

    @property
    def freq(self):
        return self.omega / TWO_PI

    @property
    def complete(self) -> bool:
        return len(self.lam) == self.sturm_count


class _ModeSet:
    """찾은 모드 집합 (스케일 공간, M-정규직교 유지)."""

    def __init__(self, prob: Problem):
        self.K, self.M = prob.K, prob.M
        self.knorm, self.mnorm = np.linalg.norm(prob.K, 2), np.linalg.norm(prob.M, 2)
        n = prob.n
        self.Phi = np.zeros((n, 0))
        self.MPhi = np.zeros((n, 0))
        self.lam = np.zeros(0)
        self.order = np.zeros(0, dtype=int)
        self.ghosts = 0
        self.relaxed = 0
        self._next = 1

    def count(self, lo: float, hi: float) -> int:
        return int(np.sum((self.lam >= lo) & (self.lam < hi)))

    @property
    def locked(self):
        return (self.Phi, self.MPhi)

    def merge(self, lam_c: np.ndarray, X: np.ndarray, lo: float, hi: float,
              res_tol: float = 1e-10, res_fallback: float = 1e-6, dup_cos: float = 0.7) -> int:
        """수렴 리츠 쌍을 모드 집합에 추가한다.

        잔차가 res_tol 보다 큰 후보(주로 이동점에서 먼 모드)는 보류한다. 이후 그 근처의
        새 이동점에서 더 정확하게 다시 구해진다. 단, 한 개도 채택되지 않으면 무한 반복을
        막기 위해 res_fallback 이하 후보를 받아들인다.
        """
        added = self._merge(lam_c, X, lo, hi, res_tol, dup_cos)
        if added == 0:
            added = self._merge(lam_c, X, lo, hi, res_fallback, dup_cos)
            if added:
                self.relaxed += added
        return added

    def _merge(self, lam_c, X, lo, hi, res_tol, dup_cos) -> int:
        added = 0
        for i in np.argsort(lam_c):
            if not (lo <= lam_c[i] <= hi):
                continue
            x = X[:, i].copy()
            mx = self.M @ x
            if self.Phi.shape[1]:
                o = self.MPhi.T @ x
                if np.max(np.abs(o)) > dup_cos:
                    self.ghosts += 1          # 이미 찾은 모드의 복제 (유령/중복)
                    continue
                x = project_out(x[:, None], self.Phi, self.MPhi)[:, 0]
                mx = self.M @ x
            nrm = np.sqrt(max(x @ mx, 0.0))
            if nrm == 0.0:
                continue
            x, mx = x / nrm, mx / nrm
            kx = self.K @ x
            lam = float(x @ kx)                 # Rayleigh 몫 (x^T M x = 1)
            # 행렬 노름 기준 상대 잔차 (강체 모드 lam~0 에서도 의미 있음)
            res = np.linalg.norm(kx - lam * mx) / ((self.knorm + abs(lam) * self.mnorm)
                                                  * np.linalg.norm(x))
            if res > res_tol or not (lo <= lam <= hi):
                continue
            self.Phi = np.hstack([self.Phi, x[:, None]])
            self.MPhi = np.hstack([self.MPhi, mx[:, None]])
            self.lam = np.append(self.lam, lam)
            self.order = np.append(self.order, self._next)
            self._next += 1
            added += 1
        return added


def solve_interval(K: np.ndarray, M: np.ndarray, opts: EigenOptions) -> EigenResult:
    prob = prepare(K, M)
    log = [f"[경고] {w}" for w in prob.warnings]
    rng = np.random.default_rng(opts.seed)
    Ks, Ms = prob.K, prob.M

    if opts.f2 <= 0:
        raise ValueError("f2 는 양수여야 합니다")
    lam_b = (TWO_PI * opts.f2) ** 2
    if opts.f1 is None or opts.f1 <= 0:
        # 강체 모드 대비: 음수 미세 이동점 (K + delta M 은 정칙)
        delta = 1e-2 * lam_b
        lam_a = -delta
        log.append(f"하한 미지정/0 -> 음수 이동점 sigma = -{delta:.4e} 로 강체 모드 포함")
    else:
        lam_a = (TWO_PI * opts.f1) ** 2
        if opts.f1 >= opts.f2:
            raise ValueError("f1 < f2 이어야 합니다")
    scale = max(abs(lam_b), abs(lam_a))

    shifts: list[ShiftRecord] = []
    runs: list[LanczosResult] = []

    def factor(sigma: float, purpose: str) -> Factorization:
        F = factor_near(Ks, Ms, sigma, scale, log)
        shifts.append(ShiftRecord(sigma=F.sigma, n_below=F.n_negative, purpose=purpose))
        return F

    Fa = factor(lam_a, "구간 하한")
    Fb = factor(lam_b, "구간 상한")
    lam_a, lam_b = Fa.sigma, Fb.sigma
    N = Fb.n_negative - Fa.n_negative
    log.append(f"Sturm 검사: nu({lam_a:.6e})={Fa.n_negative}, nu({lam_b:.6e})={Fb.n_negative}"
               f" -> 구간 내 고유치 {N}개")

    modes = _ModeSet(prob)
    L, nuL = lam_a, Fa.n_negative
    U, nuU = lam_b, Fb.n_negative
    F_shift, rec = Fa, shifts[0]

    for it in range(opts.max_shifts):
        if modes.count(L, U) >= nuU - nuL:
            break
        need = (nuU - nuL) - modes.count(L, U)
        n_eff = prob.rank_M - modes.Phi.shape[1]
        if n_eff <= 0:
            log.append("[경고] 유효 차원 소진: 더 찾을 수 있는 모드가 없습니다")
            break

        res = block_lanczos(F_shift.solve, Ms, F_shift.sigma, opts.block, rng,
                            locked=modes.locked, max_blocks=opts.max_blocks, n_eff=n_eff,
                            reorth=opts.reorth, selective=opts.selective, tol=opts.tol,
                            need=need, accept=lambda lam: (lam >= L) & (lam <= U),
                            debug=opts.debug)
        runs.append(res)
        added = modes.merge(res.lam, res.vecs, lam_a, lam_b)
        if rec.blocks:          # 같은 이동점 재사용 시 새 기록
            rec = ShiftRecord(sigma=F_shift.sigma, n_below=F_shift.n_negative, purpose="재실행")
            shifts.append(rec)
        rec.blocks, rec.new_modes, rec.reorth = res.n_blocks, added, res.n_reorth
        rec.deflated, rec.orth_loss, rec.stop = res.n_deflated, res.orth_loss, res.stop_reason
        if rec.purpose in ("구간 하한", "구간 상한"):
            rec.purpose += " + Lanczos"

        if modes.count(L, U) >= nuU - nuL:
            break

        x = _next_shift(res, modes, L, U)
        if x is not None and x < U:
            Fx = factor(x, "신뢰 구간 검증")
            cnt, true_cnt = modes.count(L, Fx.sigma), Fx.n_negative - nuL
            if cnt == true_cnt:
                log.append(f"신뢰 구간 확정 [{L:.6e}, {Fx.sigma:.6e}) : {cnt}개")
                L, nuL = Fx.sigma, Fx.n_negative
                F_shift, rec = Fx, shifts[-1]
                rec.purpose = "신뢰 구간 검증 + Lanczos"
                continue
            if cnt > true_cnt:
                log.append(f"[경고] [{L:.6e}, {Fx.sigma:.6e}) 에서 찾은 개수 {cnt} > Sturm {true_cnt}")
                L, nuL = Fx.sigma, Fx.n_negative
                F_shift, rec = Fx, shifts[-1]
                continue
            hi, nu_hi = Fx.sigma, Fx.n_negative
        else:
            hi, nu_hi = U, nuU
        log.append(f"누락 모드 {nu_hi - nuL - modes.count(L, hi)}개 감지 in [{L:.6e}, {hi:.6e}) -> 이분 탐색")
        F_shift = _locate_missing(L, nuL, hi, nu_hi, modes, factor)
        rec = shifts[-1]
        rec.purpose = "누락 모드 탐색 + Lanczos"
    else:
        log.append(f"[경고] 최대 이동점 횟수 {opts.max_shifts} 도달")

    return _finalize(prob, modes, (lam_a, lam_b), N, shifts, runs, log)


def _next_shift(res: LanczosResult, modes: _ModeSet, L: float, U: float) -> float | None:
    """수렴 모드가 연속으로 이어진 끝과 그 다음 (미수렴 리츠 값 / 다음 모드) 사이의 중간점."""
    sigma = res.sigma
    th, bd = res.all_theta, res.all_bounds
    unconv = (th > 0) & (bd > 1e-10 * np.abs(th))
    u_min = float(np.min(sigma + 1.0 / th[unconv])) if np.any(unconv) else np.inf
    lam = np.sort(modes.lam)
    c = lam[(lam >= max(L, sigma)) & (lam < u_min)]
    if c.size == 0:
        return None
    c_max = float(c[-1])
    above = lam[lam > c_max]
    nxt = min(u_min, float(above[0]) if above.size else np.inf, U)
    if not np.isfinite(nxt) or nxt <= c_max:
        return None
    return c_max + 0.5 * (nxt - c_max)


def _locate_missing(lo, nu_lo, hi, nu_hi, modes, factor, depth=8):
    """Sturm 개수 이분법으로 누락 모드가 있는 좁은 구간을 찾아 그 중앙에서 분해한다."""
    width0 = hi - lo
    for _ in range(depth):
        if hi - lo < 1e-4 * width0:
            break
        mid = 0.5 * (lo + hi)
        Fm = factor(mid, "이분 탐색")
        miss_left = (Fm.n_negative - nu_lo) - modes.count(lo, Fm.sigma)
        if miss_left > 0:
            hi, nu_hi = Fm.sigma, Fm.n_negative
        else:
            lo, nu_lo = Fm.sigma, Fm.n_negative
        if modes.count(lo, hi) == 0:      # 찾은 모드가 없는 순수 누락 구간
            break
    return factor(0.5 * (lo + hi), "누락 모드 이동점")


def _finalize(prob, modes, lam_range, N, shifts, runs, log) -> EigenResult:
    Phi, lam0, order0 = modes.Phi, modes.lam, modes.order
    m = Phi.shape[1]
    if m:
        # 찾은 전체 부분공간에서 Rayleigh-Ritz -> 중근 내부까지 M-직교
        from scipy.linalg import eigh
        Kr = Phi.T @ prob.K @ Phi
        Mr = Phi.T @ prob.M @ Phi
        lam, Y = eigh(0.5 * (Kr + Kr.T), 0.5 * (Mr + Mr.T))
        Psi = Phi @ Y
        order = order0[np.argmax(np.abs(Y), axis=0)]
    else:
        lam, Psi, order = lam0, Phi, order0

    phi = prob.d[:, None] * Psi                       # 원 좌표 복원
    K0, M0 = prob.K_orig, prob.M_orig
    Kp, Mp = K0 @ phi, M0 @ phi
    gm = np.einsum("ij,ij->j", phi, Mp)
    gk = np.einsum("ij,ij->j", phi, Kp)
    r = Kp - Mp * lam
    residual = np.linalg.norm(r, axis=0) / ((np.linalg.norm(K0, 2) + np.abs(lam) * np.linalg.norm(M0, 2))
                                           * np.linalg.norm(phi, axis=0))
    m_orth = float(np.max(np.abs(phi.T @ Mp - np.eye(m)))) if m else 0.0
    if m != N:
        log.append(f"[경고] 찾은 모드 {m}개 != Sturm 개수 {N}개")
    else:
        log.append(f"Sturm 검증 통과: 찾은 모드 {m}개 = 구간 내 고유치 {N}개")
    return EigenResult(lam=lam, phi=phi, extraction_order=order, gen_mass=gm, gen_stiff=gk,
                       residual=residual, lam_range=lam_range, sturm_count=N, shifts=shifts,
                       runs=runs, ghosts_removed=modes.ghosts, relaxed=modes.relaxed,
                       m_orth_error=m_orth, problem=prob, log=log)
