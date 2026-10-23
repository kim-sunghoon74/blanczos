"""예제 K, M 텍스트 파일 생성:  python examples/make_examples.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blanczos.matio import write_matrix  # noqa: E402
from blanczos.models import beam2d_free, frame3d_free, spring_chain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

cases = {
    "chain": (spring_chain(200)[:2],
              "고정-자유 스프링-질량 체인 200 DOF (k=1e4, m=1)"),
    "beam": (beam2d_free(99),
             "자유-자유 2D 빔 100절점 x [w, theta] = 200 DOF, 집중질량 (회전 DOF 질량 0), 강체 모드 2개"),
    "frame": (frame3d_free(3, 3, 4),
              "자유-자유 3D 격자 프레임 3x3x4 = 36절점 x 6 = 216 DOF, 강체 모드 6개, 대칭 -> 중근"),
}

for name, ((K, M), desc) in cases.items():
    d = os.path.join(HERE, name)
    os.makedirs(d, exist_ok=True)
    write_matrix(os.path.join(d, "K.txt"), K, "K", comment=f"{desc}\n강성행렬")
    write_matrix(os.path.join(d, "M.txt"), M, "M", comment=f"{desc}\n질량행렬")
    print(f"{name}: {d}/K.txt, M.txt  ({K.shape[0]} DOF)")
