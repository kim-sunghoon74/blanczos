"""사람이 읽기 쉬운 텍스트 행렬 파일 읽기/쓰기.

파일 형식 예 ::

    # 주석은 '#' 뒤에 쓴다 (줄 중간에서도 가능)
    MATRIX  K              # 이름 (K, M 등, 정보용)
    SIZE    200            # 행렬 차수 n
    STORAGE SYMMETRIC      # SYMMETRIC: 대각 포함 한쪽 삼각만 기입 / FULL: 0 아닌 모든 항 기입
    #  행    열    값        (1부터 시작하는 번호)
       1     1    2.0E+06
       2     1   -1.0E+06
    END

- 기입하지 않은 항은 0 이다.
- SYMMETRIC 에서 (i,j) 와 (j,i) 를 둘 다 쓰면 오류로 처리한다 (중복 입력 실수 방지).
- 같은 (i,j) 를 두 번 쓰는 것도 오류이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MatrixFileError(ValueError):
    pass


@dataclass
class MatrixFile:
    name: str
    matrix: np.ndarray  # dense, symmetric (n x n)


def read_matrix(path: str) -> MatrixFile:
    name = "?"
    size = None
    storage = None
    entries: dict[tuple[int, int], float] = {}
    ended = False

    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ended:
                raise MatrixFileError(f"{path}:{lineno}: END 뒤에 내용이 있습니다")
            tokens = line.split()
            key = tokens[0].upper()
            where = f"{path}:{lineno}"

            if key == "MATRIX":
                name = tokens[1] if len(tokens) > 1 else "?"
            elif key == "SIZE":
                size = _parse_int(tokens, 1, where)
                if size <= 0:
                    raise MatrixFileError(f"{where}: SIZE 는 양수여야 합니다")
            elif key == "STORAGE":
                if len(tokens) < 2 or tokens[1].upper() not in ("SYMMETRIC", "FULL"):
                    raise MatrixFileError(f"{where}: STORAGE 는 SYMMETRIC 또는 FULL")
                storage = tokens[1].upper()
            elif key == "END":
                ended = True
            else:
                if size is None or storage is None:
                    raise MatrixFileError(f"{where}: 항 입력 전에 SIZE 와 STORAGE 가 필요합니다")
                if len(tokens) != 3:
                    raise MatrixFileError(f"{where}: '행 열 값' 3개 항목이 필요합니다: {raw.rstrip()}")
                i = _parse_int(tokens, 0, where)
                j = _parse_int(tokens, 1, where)
                try:
                    v = float(tokens[2])
                except ValueError:
                    raise MatrixFileError(f"{where}: 값을 읽을 수 없습니다: {tokens[2]}") from None
                if not (1 <= i <= size and 1 <= j <= size):
                    raise MatrixFileError(f"{where}: 번호 ({i},{j}) 가 SIZE={size} 범위를 벗어났습니다")
                if not np.isfinite(v):
                    raise MatrixFileError(f"{where}: 유한하지 않은 값 {v}")
                if (i, j) in entries:
                    raise MatrixFileError(f"{where}: ({i},{j}) 항이 중복 입력되었습니다")
                if storage == "SYMMETRIC" and i != j and (j, i) in entries:
                    raise MatrixFileError(
                        f"{where}: SYMMETRIC 인데 ({i},{j}) 와 ({j},{i}) 가 모두 입력되었습니다")
                entries[(i, j)] = v

    if size is None or storage is None:
        raise MatrixFileError(f"{path}: SIZE/STORAGE 헤더가 없습니다")
    if not ended:
        raise MatrixFileError(f"{path}: END 가 없습니다 (파일이 잘렸을 수 있음)")

    A = np.zeros((size, size))
    for (i, j), v in entries.items():
        A[i - 1, j - 1] = v
        if storage == "SYMMETRIC":
            A[j - 1, i - 1] = v
    return MatrixFile(name=name, matrix=A)


def write_matrix(path: str, A: np.ndarray, name: str, comment: str = "",
                 drop_tol: float = 0.0) -> None:
    """대칭 행렬을 SYMMETRIC(하삼각) 형식으로 저장한다."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    scale = np.max(np.abs(A)) if A.size else 0.0
    with open(path, "w", encoding="utf-8") as fh:
        for c in comment.splitlines():
            fh.write(f"# {c}\n")
        fh.write(f"MATRIX  {name}\n")
        fh.write(f"SIZE    {n}\n")
        fh.write("STORAGE SYMMETRIC\n")
        fh.write("#   행     열            값\n")
        for i in range(n):              # 행 순서, 하삼각 (j <= i)
            for j in range(i + 1):
                v = A[i, j]
                if v != 0.0 and abs(v) > drop_tol * scale:
                    fh.write(f"{i + 1:5d}  {j + 1:5d}  {v: .16E}\n")
        fh.write("END\n")


def _parse_int(tokens: list[str], k: int, where: str) -> int:
    try:
        return int(tokens[k])
    except (IndexError, ValueError):
        raise MatrixFileError(f"{where}: 정수가 필요합니다") from None
