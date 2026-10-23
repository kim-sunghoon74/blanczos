"""결과 출력: Nastran 형식 고유치 표, 모드 형상 텍스트 파일, 해석 요약."""

from __future__ import annotations

import os

import numpy as np

from .driver import EigenResult


def eigen_table(res: EigenResult) -> str:
    lines = [
        "                                    R E A L   E I G E N V A L U E S",
        "   MODE    EXTRACTION      EIGENVALUE        RADIANS          CYCLES        GENERALIZED      GENERALIZED"
        "     RESIDUAL",
        "    NO.       ORDER                                                            MASS           STIFFNESS",
    ]
    for i in range(len(res.lam)):
        lines.append(
            f"{i + 1:7d} {res.extraction_order[i]:10d}   {res.lam[i]: .6E}  {res.omega[i]: .6E}"
            f"  {res.freq[i]: .6E}  {res.gen_mass[i]: .6E}  {res.gen_stiff[i]: .6E}  {res.residual[i]:.1E}")
    return "\n".join(lines)


def summary(res: EigenResult) -> str:
    prob = res.problem
    out = []
    out.append("=" * 100)
    out.append(" Shifted Block Lanczos 해석 요약")
    out.append("=" * 100)
    out.append(f" 자유도 n = {prob.n},  rank(M) = {prob.rank_M},  질량 0 대각 DOF = {len(prob.massless_dofs)}개")
    lo, hi = res.lam_range
    out.append(f" 고유치 구간 [{lo:.6E}, {hi:.6E}]  (주파수 {_hz(lo)} ~ {_hz(hi)} Hz)")
    out.append(f" Sturm 개수 (구간 내 실제 고유치 수) = {res.sturm_count}")
    out.append(f" 찾은 모드 수                        = {len(res.lam)}   "
               f"{'[OK] 누락 없음' if res.complete else '[경고] 불일치'}")
    if len(res.lam):
        out.append(f" 최대 상대 잔차 ||K phi - lam M phi||/((||K||+|lam|||M||)||phi||) = {res.residual.max():.2E}")
        out.append(f" 질량 직교성 오차 max|Phi^T M Phi - I|                          = {res.m_orth_error:.2E}")
    out.append(f" 유령/중복 모드 제거 = {res.ghosts_removed},  완화 기준(잔차<=1e-6)으로 채택 = {res.relaxed}")
    out.append("")
    out.append(" 이동점 이력")
    out.append("   no.        sigma   nu(sigma)  용도                      블록  신규모드  재직교화  블록축소  직교성손실  종료사유")
    for k, s in enumerate(res.shifts, 1):
        out.append(f"  {k:4d} {s.sigma: .5E}  {s.n_below:6d}   {s.purpose:24s} {s.blocks:4d}  {s.new_modes:6d}"
                   f"  {s.reorth:7d}  {s.deflated:7d}   {s.orth_loss:9.1E}  {s.stop}")
    out.append("")
    out.append(" 진행 기록")
    out.extend("   " + line for line in res.log)
    return "\n".join(out)


def write_results(res: EigenResult, outdir: str, labels: list[str] | None = None) -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    paths = []

    p = os.path.join(outdir, "eigenvalues.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(eigen_table(res) + "\n")
    paths.append(p)

    p = os.path.join(outdir, "modes.txt")
    n = res.phi.shape[0]
    labels = labels or [f"DOF{i + 1}" for i in range(n)]
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("# 모드 형상 (질량 정규화: phi^T M phi = 1)\n")
        for k in range(len(res.lam)):
            fh.write(f"\nMODE {k + 1}   EIGENVALUE {res.lam[k]: .8E}   FREQ {res.freq[k]: .8E} Hz\n")
            for i in range(n):
                fh.write(f"  {labels[i]:>10s}  {res.phi[i, k]: .10E}\n")
    paths.append(p)

    p = os.path.join(outdir, "summary.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(summary(res) + "\n")
    paths.append(p)

    p = os.path.join(outdir, "results.npz")
    np.savez(p, eigenvalues=res.lam, frequencies_hz=res.freq, modes=res.phi,
             generalized_mass=res.gen_mass, generalized_stiffness=res.gen_stiff,
             residual=res.residual)
    paths.append(p)
    return paths


def _hz(lam: float) -> str:
    return f"{np.sign(lam) * np.sqrt(abs(lam)) / (2 * np.pi):.4E}"
