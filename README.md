# blanczos — Python Shifted Block Lanczos 고유치 해석기

작성일: 2026-10-23

`K φ = λ M φ` (λ = ω²) 의 **주파수 구간 [f1, f2] 안의 모든 모드**를 구합니다.
MSC Nastran `EIGRL`(Grimes–Lewis–Simon Shifted Block Lanczos)의 안정화 기법을 200 DOF 규모의 밀집 행렬로 구현했습니다.

## 설치 / 실행

```bash
pip install numpy scipy pytest
python examples/make_examples.py                     # 예제 K/M 파일 생성
python -m blanczos examples/frame/K.txt examples/frame/M.txt --f2 300 --out results --check
python -m pytest -q tests                            # 테스트
```

| 옵션 | 의미 | 기본값 |
|---|---|---|
| `--f2` | 상한 주파수 [Hz] (필수) | |
| `--f1` | 하한 주파수 [Hz]. 생략하거나 0이면 강체 모드를 포함 (음수 미세 이동점) | 없음 |
| `--block` | 블록 크기 p | 7 |
| `--reorth` | `pro`(부분+선택적), `full`, `none`(유령 모드 시연용) | pro |
| `--max-blocks` | 이동점당 블록 스텝의 소프트 상한 | 20 |
| `--tol` | 리츠 값 수렴 허용치 (bound/\|θ\|) | 1e-10 |
| `--out` | 결과 폴더 | |
| `--check` | scipy 밀집 기준 해와 비교 | |

종료 코드는 Sturm 개수와 찾은 개수가 같으면 0, 다르면 2입니다.

## 입력 파일 형식 (텍스트)

```text
# '#' 뒤는 주석
MATRIX  K
SIZE    200
STORAGE SYMMETRIC        # SYMMETRIC: 한쪽 삼각만 기입 / FULL: 0이 아닌 모든 항 기입
#   행     열            값        (1부터 시작)
    1      1   1.1643588000000000E+10
    2      1   5.8806000000000000E+08
    2      2   3.9600000000000000E+07
END
```

- 적지 않은 항은 0입니다. 한 줄에 한 항(`행 열 값`)씩 씁니다.
- 실수를 막기 위해 다음은 오류로 처리합니다:
  - 같은 항을 두 번 입력
  - SYMMETRIC인데 (i,j)와 (j,i)를 둘 다 입력
  - 번호가 SIZE 범위를 벗어남
  - `END`가 없음 (파일이 잘린 경우)

## 출력 (`--out` 폴더)

- `eigenvalues.txt`: Nastran `R E A L   E I G E N V A L U E S` 형식 표. 모드 번호, 추출 순서, 고유치, rad/s, Hz, 일반화 질량/강성, 잔차가 들어 있습니다.
- `modes.txt`: 모드별 DOF 값 (질량 정규화 `φᵀMφ = 1`)
- `summary.txt`:
  - Sturm 검증 결과
  - 최대 잔차, 질량 직교성
  - 이동점 이력 (σ, ν(σ), 용도, 블록 수, 재직교화/블록 축소 횟수)
  - 진행 기록
- `results.npz`: numpy 배열

## Nastran 안정화 기법과 코드 위치

| 기법 | 구현 | 위치 |
|---|---|---|
| Shift-invert, LDLᵀ | Bunch–Kaufman LDLᵀ 한 번 분해 후 블록 전방/후방 대입 | `factor.py` |
| ① 부분 + 선택적 재직교화 | 블록 ω 점화식(잡음 주입)으로 `‖Q_kᵀMQ_{j+1}‖`을 추정. √ε를 넘으면 ε^¾을 넘는 블록에 대해 재직교화하고, 다음 스텝에도 같은 집합에 다시 적용(Simon 규칙). 잘 수렴한 리츠 벡터 방향은 매 스텝 제거(Parlett–Scott) | `lanczos.py` |
| ② 특이 M 대응 | M-내적 two-pass MGS. 시작 블록에 OP를 적용해 무질량 성분 제거. 최종 고유벡터는 `φ = OP(Qs)/θ`로 정제 | `orth.py`, `lanczos.py` |
| ③ Sturm 검사, 신뢰 구간 | 구간 양 끝의 ν(σ)로 총개수 확정. 다음 이동점의 ν로 `[L, x)`를 신뢰 구간으로 확정. 모자라면 Sturm 이분법으로 누락 위치를 찾아 이동점 배치 | `driver.py` |
| ④ 강체 모드 | f1을 생략하면 σ = −0.01·λ(f2). 분해가 특이하면 이동점을 미세하게 옮겨 재분해(Sturm 개수는 옮긴 σ 기준으로 정확) | `driver.py`, `factor.py` |
| ⑤ 블록 축소 | 열 피벗팅 MGS로 랭크 결핍을 감지하고, 기존 기저와 M-직교인 난수 벡터로 대체. 남은 차원이 p의 배수가 아닐 때 마지막 부분 블록도 처리 | `orth.py`, `lanczos.py` |
| ⑥ 오차 한계 | `‖B_{j+1} s_last‖ ≤ tol·\|θ\|`로 수렴 판정. 원 좌표에서 실제 잔차를 다시 검증하고, 1e-10을 넘으면 보류했다가 더 가까운 이동점에서 다시 구함 | `lanczos.py`, `driver.py` |
| 추가 | 대각 스케일링 `DKD, DMD`, 이미 찾은 모드 잠금(locking), 마지막 Rayleigh–Ritz로 중근 내부까지 M-직교화 | `preprocess.py`, `driver.py` |

## 검증 (`tests/`, scipy 밀집 기준 해와 비교)

- **모델:**
  - 스프링 체인 (해석해와 비교)
  - 자유-자유 빔: 강체 2개, 회전 DOF 질량 0
  - 자유-자유 3D 프레임: 강체 6개, 중근, M 특이
  - 3중근이 있는 임의 행렬
- **조건 조합:** f1>0 구간, 블록 크기 1/3/7, 이동점당 블록 3~20, `pro`/`full`
- **합격 기준:**
  - 찾은 개수 = Sturm 개수
  - 고유치 오차 < 1e-11 (최대 고유치 기준)
  - 잔차 < 1e-10
  - `max|ΦᵀMΦ − I|` < 1e-10

## 한계

- 밀집 행렬(n ≈ 수백)을 전제로 합니다. 희소 대형 모델로 가려면 `factor.Factorization`의 `solve()`, `n_negative`를 관성(inertia)을 주는 희소 LDLᵀ(MUMPS, CHOLMOD 등)로 바꿔야 합니다.
- 기본 검사(`preprocess.py`)에서 고유치를 직접 계산하므로 O(n³)입니다. 대형 모델에서는 이 검사를 빼야 합니다.
