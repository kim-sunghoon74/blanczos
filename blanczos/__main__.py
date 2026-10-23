"""명령행 실행:  python -m blanczos K.txt M.txt --f2 100 [--f1 0] [--out results]"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .driver import EigenOptions, solve_interval
from .matio import read_matrix
from .report import eigen_table, summary, write_results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m blanczos",
        description="Shifted Block Lanczos 로 K phi = lam M phi 의 [f1, f2] 구간 내 모든 모드를 구한다.")
    ap.add_argument("K", help="강성행렬 텍스트 파일")
    ap.add_argument("M", help="질량행렬 텍스트 파일")
    ap.add_argument("--f2", type=float, required=True, help="상한 주파수 [Hz]")
    ap.add_argument("--f1", type=float, default=None,
                    help="하한 주파수 [Hz] (생략 또는 0 -> 강체 모드 포함)")
    ap.add_argument("--block", type=int, default=7, help="블록 크기 p (기본 7)")
    ap.add_argument("--reorth", choices=["pro", "full", "none"], default="pro",
                    help="재직교화 방식 (기본 pro)")
    ap.add_argument("--no-selective", action="store_true", help="선택적 직교화 끄기")
    ap.add_argument("--max-blocks", type=int, default=20, help="이동점당 최대 블록 스텝")
    ap.add_argument("--tol", type=float, default=1e-10, help="수렴 허용치")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="결과 저장 폴더")
    ap.add_argument("--check", action="store_true",
                    help="밀집 행렬 기준 해(scipy)와 비교")
    args = ap.parse_args(argv)

    K = read_matrix(args.K).matrix
    M = read_matrix(args.M).matrix
    opts = EigenOptions(f2=args.f2, f1=args.f1, block=args.block, reorth=args.reorth,
                        selective=not args.no_selective, max_blocks=args.max_blocks,
                        tol=args.tol, seed=args.seed)
    t0 = time.perf_counter()
    res = solve_interval(K, M, opts)
    elapsed = time.perf_counter() - t0

    print(eigen_table(res))
    print()
    print(summary(res))
    print(f"\n 계산 시간 {elapsed:.2f} s")

    if args.check:
        from .models import reference_eigs
        ref, _ = reference_eigs(K, M)
        lo, hi = res.lam_range
        refin = ref[(ref >= lo) & (ref <= hi)]
        print("\n 기준 해 비교 (scipy.linalg.eigh, 밀집)")
        print(f"   기준 해 구간 내 개수 {len(refin)}, 찾은 개수 {len(res.lam)}")
        if len(refin) == len(res.lam) and len(refin):
            scale = max(abs(ref).max(), 1.0)
            err = np.abs(res.lam - refin)
            print(f"   max|lam - lam_ref| / max|lam_ref| = {err.max() / scale:.2E}")

    if args.out:
        for p in write_results(res, args.out):
            print(f" 저장: {p}")
    return 0 if res.complete else 2


if __name__ == "__main__":
    sys.exit(main())
