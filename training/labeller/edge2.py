"""
와이어 끝 경계 찾기 — "compute edge 2" (BestFA 용으로 맞춤. edge 1 = edge.py).

BestFA 와이어는 KIMM 보다 가늘고 끝이 뾰족한 U/V 이며, 아크 빛을 받은 옆면이 밝다.
edge 1(열별 50% 지점 + 2차 곡선 하나)은 여기서 두 가지로 틀린다 — V 자 끝은 포물선에 안 맞아
잔차가 커서 "낮음"이 되고, 박스 가장자리 열이 빛 받은 옆면·배경에 걸려 옆면을 타고 올라간다.

edge 2 는 끝의 둘레를 직접 딴다.
  1. OBB 를 양옆 여유(MARGIN × 두께)까지 펴서(edge 1 과 같은 affine) 블러.
  2. 끝 꼭짓점(apex) = 행마다 박스 안 최소 밝기(= 와이어)가 "와이어 밝기와 아크 밝기의 중간"을
     처음 넘는 행 — 박스 가장자리 열이 무엇이든 와이어 자체로 정해진다.
  3. 끝 바로 위 행에서 와이어 폭·가운데를 재고, 끝 안쪽 중심 O(apex 에서 반폭 위)를 잡는다.
  4. 끝 둘레 = O 에서 아래 반원으로 쏜 광선(RAYS 개), 옆면 = O 높이에서 위로 행마다 좌우 탐색.
     둘 다 국소 문턱 — 가까운 쪽 최소(와이어)와 짧은 거리 안 최대(바깥)의 중간을 처음 넘는 곳.
     옆이 회색 배경이어도 와이어 옆면에 걸리고, 옆이 와이어만큼 어두우면(안 보임) 그 점은 없음.
  5. 옆면은 O 에서 위로 끊김 없이 이어지는 구간만 쓴다(배경으로 튄 갈고리 제거).
  6. 뾰족한 끝(주로 오른쪽 와이어) — 끝이 가늘어지며 헤일로에 회색으로 밝아지면 광선은 쐐기 중간에서
     끝을 평평하게 자른다. O 에서 아래로 행마다 좌우 국소 문턱으로 쐐기를 끝까지 따라가서, 광선 끝보다
     와이어 폭의 TIP_EXTRA_W 배 넘게 더 내려가면 그 행들의 좌우 경계로 V 를 만들고 맨 아래는 작은
     광선으로 둥글게 닫는다. (둥근 끝은 행이 늘 5~9px 정도만 더 내려가서 원래 광선 둘레 그대로.)
결과 폴리곤 = OBB 윗변 2점 + 오른쪽 옆면 → 끝 둘레 → 왼쪽 옆면. 옆면이 안 보이는 구간은
OBB 윗변 꼭짓점까지 직선.

판정("ok" / "low" / "no_tip") — 곡선 모양(U·V)과 무관한 기준만 쓴다.
  arc     끝 둘레 광선 중 경계를 찾은 비율          ≥ OK_ARC
  radius  끝 둘레 반지름의 최대 벗어남(중앙값 대비)  ≤ OK_RADIUS — 새는 광선만 멀리 튄다
  side    옆면 점들의 직선 잔차 RMS(px)             ≤ OK_SIDE   — 옆면은 축과 나란한 직선
  smooth  둘레 점 ↔ 양 이웃 평균 거리 RMS(px)        ≤ OK_SMOOTH
옆면이 안 보여 빠진 점은 감점하지 않는다(어두운 배경 앞에선 원래 안 보임).

BestFA 261026_115301 무작위 300장(와이어 600개)에서 edge 1 "낮음" 75개 → edge 2 20개.
뾰족한 끝으로 처리된 와이어: 오른쪽 117/300, 왼쪽 3/300.
남은 "낮음"은 박스가 실제 와이어에서 벗어났거나 끝이 가려진 장 — 사람이 봐야 하는 장.
"""
from __future__ import annotations

import math
from pathlib import Path
from statistics import median

from PIL import Image, ImageFilter

from edge import _exit_distance, obb_pose

BLUR = 2
REF_PX = (8, 60)
MIN_STEP = 20
MARGIN = 0.6          # 펼칠 때 양옆 여유 (두께 배수)
SIDE_ROWS = 1.2       # 옆면을 따는 높이 (O 위로, 두께 배수)
RAYS = 25             # 아래 반원 광선 수
SIDE_STEP = 3         # 옆면 행 간격(px)
SIDE_JUMP = 2.5       # 옆면 이웃 점의 가로 차가 이보다 크면 거기서 끊음(px)
TIP_STEP = 1.5        # 뾰족한 끝: O 아래로 행 간격(px)
TIP_MIN_W = 2.0       # 뾰족한 끝: 쐐기 폭이 이보다 좁아지면 거기가 끝(px)
CAP_RAYS = 11         # 뾰족한 끝: 맨 아래를 둥글게 닫는 광선 수 (양 끝 수평 포함)
TIP_EXTRA = 3.0       # 쐐기가 광선 끝보다 이만큼(px) 그리고
TIP_EXTRA_W = 0.3     # 와이어 폭의 이 비율보다 더 내려가야 뾰족한 끝으로 봄
OK_SMOOTH = 1.5      # 등고선 매끄러움 (점 ↔ 이웃 평균 거리 RMS, px)
OK_ARC = 0.9          # 끝 둘레 광선 중 경계를 찾은 비율
OK_RADIUS = 0.8       # 끝 둘레 반지름의 최대 벗어남 (중앙값 대비)
OK_SIDE = 2.0         # 옆면 점들의 직선 잔차 RMS (px)


def find_tip2(image_path: Path, points: list) -> dict:
    """
    이미지 + OBB 라벨(정규화 4점) → 끝 경계. 반환 형식은 edge.find_tip 과 같다 (status,
    polygon, edges, inliers/columns = 찾은 광선/전체 광선, rms = 매끄러움, tip = 끝까지 축 거리)
    + score (판정에 쓴 값들).
    """
    with Image.open(image_path) as im:
        gray = im.convert("L")
    w, h = gray.size
    pose = obb_pose([(x * w, y * h) for x, y in points])
    px, py = pose["pivot"]
    ux, uy = math.cos(pose["angle"]), math.sin(pose["angle"])
    rx, ry = uy, -ux
    thick, length = pose["thickness"], pose["length"]
    m = int(round(MARGIN * thick))

    base = {"pose": pose, "polygon": None, "edges": [], "inliers": 0, "columns": 0,
            "rms": None, "tip": None, "score": {}}

    # 펼친 띠: 열 c ↔ t = c − m (t = 왼쪽 변에서 두께 방향), 행 = s (피벗에서 축 방향)
    s_max = min(_exit_distance(px + m * rx, py + m * ry, ux, uy, w, h),
                _exit_distance(px - (thick + m) * rx, py - (thick + m) * ry, ux, uy, w, h),
                3 * length)
    tw, sh = int(round(thick)) + 2 * m, int(s_max)
    y0, y1 = REF_PX
    if sh <= y1 + 10:
        return {**base, "status": "no_tip"}
    ox, oy = px - (thick + m) * rx, py - (thick + m) * ry
    strip = gray.transform((tw, sh), Image.AFFINE, (rx, ux, ox, ry, uy, oy),
                           resample=Image.BILINEAR).filter(ImageFilter.GaussianBlur(BLUR))
    buf = strip.tobytes()

    def P(c: int, s: int) -> int:
        return buf[s * tw + c]

    def B(c: float, s: float) -> float:
        """양선형 보간 밝기 (띠 밖이면 None)."""
        if not (0 <= c <= tw - 1 and 0 <= s <= sh - 1):
            return None
        c0, s0 = int(c), int(s)
        c1, s1 = min(c0 + 1, tw - 1), min(s0 + 1, sh - 1)
        fc, fs = c - c0, s - s0
        return ((P(c0, s0) * (1 - fc) + P(c1, s0) * fc) * (1 - fs) +
                (P(c0, s1) * (1 - fc) + P(c1, s1) * fc) * fs)

    def to_img(t: float, s: float) -> list:
        return [px + (t - thick) * rx + s * ux, py + (t - thick) * ry + s * uy]

    box0, box1 = m + 2, m + int(round(thick)) - 2
    center = range(m + int(0.3 * thick), m + int(0.7 * thick) + 1)
    wire = median(P(c, s) for s in range(y0, y1) for c in center)
    valley = [min(buf[s * tw + box0: s * tw + box1 + 1]) for s in range(sh)]
    peak = max(valley[y1:])
    if peak - wire < MIN_STEP:
        return {**base, "status": "no_tip"}
    thr = wire + 0.5 * (peak - wire)

    # 끝 꼭짓점 — 박스 안 최소 밝기가 문턱을 넘는 첫 행
    apex = None
    for s in range(y1, sh):
        if valley[s] >= thr:
            prev = valley[s - 1]
            apex = s - 1 + (thr - prev) / (valley[s] - prev) if valley[s] > prev else float(s)
            break
    if apex is None:
        return {**base, "status": "no_tip"}

    def cross(c: float, s: float, dc: float, ds: float, limit: float):
        """
        (c, s) 에서 (dc, ds) 방향 limit px 안의 밝기 프로필에서 국소 문턱(가까운 쪽 최소 =
        와이어, limit 안 최대 = 바깥의 중간)을 처음 넘는 곳. 옆이 회색 배경이어도 와이어 옆면에
        걸린다. 바깥이 와이어보다 MIN_STEP 이상 밝지 않으면(옆면이 안 보임) None.
        """
        prof = []
        d = 0.0
        while d <= limit:
            v = B(c + dc * d, s + ds * d)
            if v is None:
                break
            prof.append(v)
            d += 0.5
        if len(prof) < 4:
            return None
        lo = min(prof[: max(2, len(prof) // 3)])
        hi = max(prof)
        if hi - lo < MIN_STEP:
            return None
        t_ = lo + 0.5 * (hi - lo)
        i0 = prof.index(lo)
        for i in range(i0 + 1, len(prof)):
            if prof[i] >= t_:
                prev = prof[i - 1]
                k = (t_ - prev) / (prof[i] - prev) if prof[i] > prev else 1.0
                dd = 0.5 * (i - 1 + k)
                return (c + dc * dd, s + ds * dd)
        return None

    def row_sides(s: float, c_hint: float, half: float):
        """행 s 에서 c_hint 근처 가장 어두운 곳부터 좌우로 국소 문턱 넘는 곳."""
        lo, hi = max(0, int(c_hint - half)), min(tw - 1, int(c_hint + half))
        si = int(round(s))
        cmin = min(range(lo, hi + 1), key=lambda c: P(c, si))
        lim = half + 10
        return cmin, cross(cmin, s, -1, 0, lim), cross(cmin, s, 1, 0, lim)

    # 끝 바로 위에서 폭·가운데 추정 → 끝 안쪽 중심 O
    s_probe = max(y1, apex - 0.6 * thick)
    cmin, L, R = row_sides(s_probe, m + thick / 2, thick / 2)
    if L and R:
        cc, half = (L[0] + R[0]) / 2, (R[0] - L[0]) / 2
    else:
        cc, half = cmin, thick / 3
    half = max(4.0, min(half, thick * 0.7))
    o_s = max(y1, apex - half)
    _, L, R = row_sides(o_s, cc, half)
    if L and R:
        cc = (L[0] + R[0]) / 2
    o_c = cc

    # 옆면 — O 높이에서 위로. 옆이 어두운 배경이면 안 보이는 게 정상이라 빠져도 감점 없음
    right_side, left_side = [], []
    s_top = max(y1, o_s - SIDE_ROWS * thick)
    s = o_s - SIDE_STEP
    hint = o_c
    while s >= s_top:
        _, L, R = row_sides(s, hint, half)
        if R:
            right_side.append(R)
        if L:
            left_side.append(L)
        if L and R:
            hint = (L[0] + R[0]) / 2
        s -= SIDE_STEP

    # 옆면은 O 에서 위로 끊김 없이 이어지는 구간만 — 위쪽에서 배경으로 튄 점(갈고리) 제거
    def trim(pts):
        kept = []
        for q in pts:
            if kept and abs(q[0] - kept[-1][0]) > SIDE_JUMP:
                break
            kept.append(q)
        return kept
    right_side, left_side = trim(right_side), trim(left_side)

    # 끝 둘레 — O 에서 아래 반원 광선
    arc = []
    lim = 1.3 * half + 10
    for i in range(RAYS):   # θ: 0 = 오른쪽 → π/2 = 아래(축 방향) → π = 왼쪽
        th = math.pi * i / (RAYS - 1)
        q = cross(o_c, o_s, math.cos(th), math.sin(th), lim)
        if q:
            arc.append(q)
    if len(arc) < RAYS // 2:
        return {**base, "status": "no_tip", "columns": RAYS, "inliers": len(arc)}

    # 뾰족한 끝 — 끝이 가늘어지며 헤일로에 회색으로 밝아지면, 와이어 밝기↔아크 밝기 중간(광선)은
    # 쐐기 중간에서 끝을 평평하게 자른다. 쐐기도 양옆보다는 어두우므로 O 에서 아래로 행마다 좌우
    # 국소 문턱으로 쐐기를 끝까지 따라가고, 그게 광선 끝보다 더 내려가면 그 행들로 V 를 만든다.
    # 광선 끝보다 두께만큼 더 내려가도 안 끝나면 쐐기가 아니라 와이어 아래 어두운 줄 → 쓰지 않음
    arc_bottom = max(q[1] for q in arc)
    rows_down, hint, s, ended = [], o_c, o_s, False
    while s < min(sh - 2, arc_bottom + thick):
        _, L, R = row_sides(s, hint, half)
        mid = (L[0] + R[0]) / 2 if (L and R) else None
        if mid is None or R[0] - L[0] < TIP_MIN_W or (rows_down and abs(mid - hint) > SIDE_JUMP):
            ended = True
            break
        rows_down.append((L, R))
        hint = mid
        s += TIP_STEP
    extra = (rows_down[-1][0][1] - arc_bottom) if (ended and rows_down) else 0.0
    # 행 문턱은 광선 문턱보다 늘 몇 px 더 내려가므로(둥근 끝도 5~9px) 폭에 비례한 여유를 넘어야 뾰족한 끝
    pointed = extra > max(TIP_EXTRA, TIP_EXTRA_W * 2 * half)

    # ── 점수 ──
    # 1) 끝 둘레를 다 찾았나 (광선이 놓치면 끝이 흐리거나 열려 있음)
    arc_found = len(arc) / RAYS
    # 2) 끝 둘레 반지름이 고른가 — 새는 곳이 있으면 그 광선만 멀리 튄다
    radii = [math.hypot(c - o_c, s_ - o_s) for c, s_ in arc]
    r_med = median(radii)
    r_out = max(abs(r - r_med) for r in radii) / max(r_med, 1.0)
    # 3) 옆면이 축과 나란한 직선인가 — 옆면을 타고 배경으로 새면 휘거나 벌어진다
    def side_dev(pts):
        if len(pts) < 4:
            return 0.0
        n = len(pts)
        ms = sum(p[1] for p in pts) / n
        mc = sum(p[0] for p in pts) / n
        sxx = sum((p[1] - ms) ** 2 for p in pts)
        b = sum((p[1] - ms) * (p[0] - mc) for p in pts) / sxx if sxx else 0.0
        return math.sqrt(sum((p[0] - (mc + b * (p[1] - ms))) ** 2 for p in pts) / n)
    side = max(side_dev(right_side), side_dev(left_side))
    if pointed:
        # V 끝: 오른쪽 경계 내려가며 → 끝점(마지막 행 가운데, 반 행 아래) → 왼쪽 경계 올라가며.
        # 광선 기준(끊김·튐)은 쓰지 않는다 — 쐐기를 끝까지 따라왔다는 것 자체가 끝을 찾은 것.
        # 맨 아래는 마지막 행 가운데에서 아래 반원 광선으로 둥글게 닫는다 (행은 대비가 약해지면
        # 폭이 남은 채 멈춰서, 그대로 이으면 바닥이 평평하게 잘림)
        (lL, lR) = rows_down[-1]
        c_c, c_s, c_w = (lL[0] + lR[0]) / 2, lL[1], lR[0] - lL[0]
        cap = []
        for i in range(1, CAP_RAYS - 1):   # 양 끝(수평)은 마지막 행의 R·L 과 같으므로 뺌
            th = math.pi * i / (CAP_RAYS - 1)
            q = cross(c_c, c_s, math.cos(th), math.sin(th), c_w + 8)
            if q and q[1] > c_s:
                cap.append(q)
        if len(cap) < (CAP_RAYS - 2) // 2:
            cap = [(c_c, c_s + TIP_STEP / 2)]
        arc = [R for _, R in rows_down] + cap + [L for L, _ in rows_down[::-1]]
        arc_found, r_out = 1.0, 0.0
    # 4) 등고선 매끄러움 — 각 점과 양 이웃 평균 사이 거리
    contour = right_side[::-1] + arc + left_side
    dev = [math.hypot(b[0] - (a[0] + c[0]) / 2, b[1] - (a[1] + c[1]) / 2)
           for a, b, c in zip(contour, contour[1:], contour[2:])]
    smooth = math.sqrt(sum(d * d for d in dev) / len(dev)) if dev else 0.0
    # 걸린 기준 — 라벨러가 "낮음" 옆에 이유로 보여 준다
    fail = [name for name, bad in (("끝 둘레 끊김", arc_found < OK_ARC), ("끝 둘레 튐", r_out > OK_RADIUS),
                                   ("옆면 휨", side > OK_SIDE), ("들쭉날쭉", smooth > OK_SMOOTH)) if bad]
    ok = not fail

    pts_t = [(c - m, s_) for c, s_ in contour]
    polygon = [to_img(0, 0), to_img(thick, 0)] + [to_img(t, s_) for t, s_ in pts_t]
    return {
        **base,
        "status": "ok" if ok else "low",
        "polygon": polygon,
        "edges": [to_img(t, s_) + [True] for t, s_ in pts_t],
        "inliers": len(arc), "columns": RAYS, "rms": smooth,
        "tip": max(apex, max(q[1] for q in arc)),
        "score": {"arc": round(arc_found, 2), "radius": round(r_out, 2), "side": round(side, 2),
                  "smooth": round(smooth, 2), "width": round(2 * half, 1),
                  "sides": [len(right_side), len(left_side)], "pointed": pointed, "extra": round(extra, 1), "fail": fail},
    }
