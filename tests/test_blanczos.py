import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blanczos import EigenOptions, solve_interval  # noqa: E402
from blanczos.__main__ import main  # noqa: E402
from blanczos.factor import Factorization  # noqa: E402
from blanczos.matio import MatrixFileError, read_matrix, write_matrix  # noqa: E402
from blanczos.models import (beam2d_free, frame3d_free, random_pencil,  # noqa: E402
                             reference_eigs, spring_chain)
from blanczos.orth import block_mqr  # noqa: E402

TWO_PI = 2 * np.pi


def hz(lam):
    return np.sqrt(lam) / TWO_PI


def f_between(ref, i):
    """ref[i] 과 ref[i+1] 사이 주파수 (중근 경계 회피)."""
    while ref[i + 1] - ref[i] < 1e-6 * abs(ref[i + 1]):
        i += 1
    return hz(0.5 * (ref[i] + ref[i + 1])), i


def check(res, ref_in, lam_scale):
    assert res.complete
    assert len(res.lam) == len(ref_in)
    assert np.max(np.abs(res.lam - ref_in)) / lam_scale < 1e-11
    assert res.residual.max() < 1e-10
    assert res.m_orth_error < 1e-10


# ---------------------------------------------------------------- 입출력
def test_matrix_roundtrip(tmp_path):
    K, M = beam2d_free(20)
    p = tmp_path / "K.txt"
    write_matrix(str(p), K, "K", comment="test")
    assert np.array_equal(read_matrix(str(p)).matrix, K)


@pytest.mark.parametrize("body,msg", [
    ("1 1 2.0\n1 1 3.0\n", "중복"),
    ("1 2 2.0\n2 1 3.0\n", "모두"),
    ("3 1 2.0\n", "범위"),
    ("1 1\n", "3개"),
])
def test_matrix_errors(tmp_path, body, msg):
    p = tmp_path / "bad.txt"
    p.write_text(f"MATRIX K\nSIZE 2\nSTORAGE SYMMETRIC\n{body}END\n", encoding="utf-8")
    with pytest.raises(MatrixFileError, match=msg):
        read_matrix(str(p))


def test_matrix_missing_end(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("SIZE 2\nSTORAGE FULL\n1 1 1.0\n", encoding="utf-8")
    with pytest.raises(MatrixFileError, match="END"):
        read_matrix(str(p))


# ---------------------------------------------------------------- 분해 / 직교화
def test_sturm_count_matches_eigenvalues():
    K, M, lam = random_pencil(60, seed=3, cluster=False)
    for sigma in (0.5 * (lam[9] + lam[10]), 0.5 * (lam[40] + lam[41]), -1.0):
        F = Factorization(K, M, sigma)
        assert F.n_negative == np.sum(lam < sigma)
        b = np.random.default_rng(0).standard_normal((60, 3))
        x = F.solve(b)
        assert np.allclose((K - sigma * M) @ x, b, atol=1e-8 * np.abs(b).max())


def test_block_mqr_rank_deficient():
    rng = np.random.default_rng(0)
    n = 50
    M = np.diag(rng.uniform(1, 2, n))
    R = rng.standard_normal((n, 4))
    R[:, 2] = R[:, 0] - 2 * R[:, 1]                 # 선형 종속 열
    Q, MQ, B, nd, exh = block_mqr(R, M, 1.0, None, rng)
    assert nd == 1 and not exh
    assert np.allclose(Q.T @ M @ Q, np.eye(4), atol=1e-13)
    assert np.allclose(Q @ B, R, atol=1e-12)


# ---------------------------------------------------------------- 고유치 해석
@pytest.mark.parametrize("reorth", ["pro", "full"])
def test_chain_analytic(reorth):
    K, M, exact = spring_chain(200)
    f2, i = f_between(exact, 39)
    res = solve_interval(K, M, EigenOptions(f2=f2, reorth=reorth))
    check(res, exact[:i + 1], exact[-1])


@pytest.mark.parametrize("lumped", [True, False])
def test_free_beam_rigid_modes(lumped):
    K, M = beam2d_free(lumped=lumped)
    ref, _ = reference_eigs(K, M)
    f2, i = f_between(ref, 49)
    res = solve_interval(K, M, EigenOptions(f2=f2))
    check(res, ref[:i + 1], abs(ref).max())
    assert np.sum(np.abs(res.lam) < 1e-6 * res.lam[2]) == 2        # 강체 모드 2개
    if lumped:
        assert res.problem.rank_M == 100                              # 회전 DOF 질량 0


def test_free_frame_rigid_and_repeated():
    K, M = frame3d_free()
    ref, _ = reference_eigs(K, M)
    f2, i = f_between(ref, 60)
    res = solve_interval(K, M, EigenOptions(f2=f2))
    check(res, ref[:i + 1], abs(ref).max())
    assert np.sum(np.abs(res.lam) < 1e-6 * res.lam[6]) == 6         # 강체 모드 6개
    el = res.lam[6:]
    assert np.sum(np.diff(el) < 1e-8 * el[1:]) >= 3                 # 중근 존재


@pytest.mark.parametrize("block", [1, 3, 7])
def test_random_with_triple_root(block):
    K, M, lam = random_pencil(200)
    f2, i = f_between(lam, 45)
    res = solve_interval(K, M, EigenOptions(f2=f2, block=block))
    check(res, lam[:i + 1], lam[-1])


def test_interval_with_f1_and_short_runs():
    """f1 > 0 구간 + 이동점당 블록 수를 작게 -> 신뢰 구간 여러 개로 분할."""
    K, M = frame3d_free()
    ref, _ = reference_eigs(K, M)
    f1, i1 = f_between(ref, 19)
    f2, i2 = f_between(ref, 80)
    res = solve_interval(K, M, EigenOptions(f1=f1, f2=f2, block=3, max_blocks=3))
    check(res, ref[i1 + 1:i2 + 1], abs(ref).max())
    assert sum(1 for s in res.shifts if s.blocks) > 1


def test_no_reorth_is_worse():
    """재직교화 없이는 유령 모드가 생기거나 누락이 남는다 (pro 는 완전)."""
    K, M = beam2d_free(lumped=False)
    ref, _ = reference_eigs(K, M)
    f2, _ = f_between(ref, 95)
    bad = solve_interval(K, M, EigenOptions(f2=f2, reorth="none"))
    good = solve_interval(K, M, EigenOptions(f2=f2, reorth="pro"))
    assert good.complete
    assert bad.ghosts_removed > 0 or not bad.complete or len(bad.shifts) > len(good.shifts)


def test_cli(tmp_path, capsys):
    K, M = beam2d_free(30)
    write_matrix(str(tmp_path / "K.txt"), K, "K")
    write_matrix(str(tmp_path / "M.txt"), M, "M")
    code = main([str(tmp_path / "K.txt"), str(tmp_path / "M.txt"), "--f2", "500",
                 "--out", str(tmp_path / "out"), "--check"])
    out = capsys.readouterr().out
    assert code == 0
    assert "R E A L   E I G E N V A L U E S" in out
    for f in ("eigenvalues.txt", "modes.txt", "summary.txt", "results.npz"):
        assert (tmp_path / "out" / f).exists()
