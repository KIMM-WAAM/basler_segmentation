"""edge2 하이라이트 강건 모드 (BestFA 오른쪽 와이어) — 옆면 직선 맞춤과 윗변 연장."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeller"))
from edge2 import SIDE_TOL, find_tip2, fit_side  # noqa: E402


def test_fit_side_drops_jump_near_start_instead_of_cutting_everything_above():
    # O 바로 위 첫 점이 튀면 기존 trim 은 위쪽을 통째로 버렸다 — 직선 맞춤은 그 점만 뺀다
    pts = [(30.0, 300.0)] + [(20.0 + 0.05 * (297 - s), float(s)) for s in range(297, 150, -3)]
    inliers, (a, b) = fit_side(pts)
    assert (30.0, 300.0) not in inliers
    assert len(inliers) == len(pts) - 1
    assert abs(b - (-0.05)) < 1e-6
    assert all(abs(c - (a + b * s)) <= SIDE_TOL for c, s in inliers)


def test_fit_side_rejects_lines_far_from_axis_and_too_few_points():
    assert fit_side([(0.0, 0.0), (1.0, 1.0)]) == ([], None)
    steep = [(float(s), float(s)) for s in range(0, 60, 3)]   # 기울기 1 — 옆면일 수 없음
    assert fit_side(steep) == ([], None)


def test_highlight_mode_follows_wire_edge_up_to_top_instead_of_obb_corner(tmp_path):
    # 와이어(40px)보다 살짝 밝은 어두운 배경 (실제 오른쪽 와이어처럼 어두운 쪽 옆면은 안 보임) · 왼쪽 옆면의
    # 밝은 하이라이트 줄 · 아래 아크. OBB 는 와이어보다 넓다.
    img = Image.new("L", (400, 600), 20)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 400, 399, 599), fill=255)            # 아크
    d.rectangle((180, 0, 220, 399), fill=12)             # 와이어 몸통
    d.ellipse((180, 375, 220, 415), fill=12)             # 둥근 끝
    d.rectangle((172, 150, 179, 385), fill=200)          # 왼쪽 옆면 하이라이트
    path = tmp_path / "wire.png"
    img.save(path)
    w, h = img.size
    # 윗변을 y=20 에 둔다 (정확히 0 이면 부동소수 오차로 시작점이 이미지 밖으로 판정됨)
    obb = [(165 / w, 20 / h), (235 / w, 20 / h), (235 / w, 490 / h), (165 / w, 490 / h)]

    plain = find_tip2(path, obb)
    robust = find_tip2(path, obb, highlight=True)
    assert plain["polygon"][0][0] == 165.0               # 기본: OBB 윗변 왼쪽 꼭짓점
    assert robust["score"]["side_lines"][1]              # 왼쪽 옆면 직선을 찾음
    assert abs(robust["polygon"][0][0] - 180) <= 3        # 윗변 왼쪽 = 와이어 왼쪽 경계
    assert robust["polygon"][1][0] == 235.0              # 오른쪽은 안 보여(배경 ≈ 와이어) OBB 꼭짓점 그대로
