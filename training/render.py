"""
추론 결과 그림 — infer.py(사진·동영상)와 epoch_overlay.py(학습 중 격자)가 같이 쓴다.

Ultralytics 의 r.plot() 대신: 축 정렬 bbox 는 안 그리고, 마스크는 옅게 채우고 윤곽선
(OBB 모델은 회전 박스 윤곽), 클래스·점수는 그림 모서리에 한 줄씩.

와이어 끝점(tip) — seg 모델은 mask_tip (앱 · WAAM_Monitoring 의 WireTip.FromMask 와 같은 정의,
TIP_METHOD): 마스크 끝 부분 모양에서 잡는다. OBB 모델은 마스크가 없어 회전 박스 아랫변(짧은 변 중
화면 아래쪽)의 중점.
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics.utils.plotting import colors

MASK_ALPHA = 0.25   # 마스크 채움 불투명도 (Ultralytics 기본 0.5 는 원본을 가림)
TIP_METHOD = "end-region-pca"   # WireTip.Method (앱 · WAAM) 와 같은 이름
ELONGATED_RATIO = 1.5           # WireTip.ElongatedRatio


def bottom_mid(corners: np.ndarray) -> tuple:
    """회전 사각형 꼭짓점 4개(순서대로) → 아랫변(짧은 변 중 y 큰 것)의 중점 (x, y). OBB 모델 끝점."""
    c = np.asarray(corners, np.float64).reshape(4, 2)
    mids = (c + np.roll(c, -1, axis=0)) / 2               # 변 i = 꼭짓점 i → i+1
    lens = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    short = (0, 2) if lens[0] + lens[2] <= lens[1] + lens[3] else (1, 3)
    i = max(short, key=lambda k: mids[k, 1])
    return float(mids[i, 0]), float(mids[i, 1])


def _tip_axis(corners) -> tuple:
    """
    회전 사각형 꼭짓점 4개(순서대로) → (와이어 끝 쪽 단위 벡터, 끝 변 길이 = 폭, 길쭉한지).
    길쭉하면(길이/폭 ≥ ELONGATED_RATIO) 짧은 변 쌍이 끝, 아니면 더 수직인 축의 끝 쌍. 둘 중 화면 아래쪽 끝.
    """
    c = np.asarray(corners, np.float64).reshape(4, 2)
    even = (np.linalg.norm(c[0] - c[1]) + np.linalg.norm(c[2] - c[3])) / 2   # 끝 = 변 0 · 2
    odd = (np.linalg.norm(c[1] - c[2]) + np.linalg.norm(c[3] - c[0])) / 2    # 끝 = 변 1 · 3
    even_axis = (c[2] + c[3]) / 2 - (c[0] + c[1]) / 2
    odd_axis = (c[3] + c[0]) / 2 - (c[1] + c[2]) / 2
    elongated = max(even, odd) >= ELONGATED_RATIO * max(min(even, odd), 1e-6)
    vertical = lambda d: abs(d[1]) / max(np.linalg.norm(d), 1e-6)
    use_even = even <= odd if elongated else vertical(even_axis) >= vertical(odd_axis)
    axis = even_axis if use_even else odd_axis
    if axis[1] < 0:
        axis = -axis
    return axis / max(np.linalg.norm(axis), 1e-6), (even if use_even else odd), elongated


def mask_tip(mask: np.ndarray) -> tuple | None:
    """
    bool 마스크(원본 px) → 와이어 끝점 (x, y), 비면 None. 앱 · WAAM 의 WireTip.FromMask 와 같은 정의:
      1. 가장 큰 8-연결 덩어리만 (끝 근처에 붙은 아크 빛 · 스패터 조각 무시)
      2. 축 = 픽셀 주축(PCA). 어느 끝인지는 최소 면적 회전 사각형으로 (_tip_axis), 짧게만 보이면 그 수직 축
      3. 끝점 = 축 방향 맨 끝 픽셀까지, 거기서 폭의 절반 깊이 안 픽셀들의 가로 중심
    예전 정의(사각형 아랫변 중점)는 마스크 전체 방향에 묶여 긴 와이어에서 옆으로 튀었다.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    ys, xs = np.nonzero(lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    # 행마다 좌우 끝 픽셀의 픽셀-모서리 4점 (픽셀 x 는 [x, x+1]) → 최소 면적 회전 사각형
    first = np.full(mask.shape[0], mask.shape[1])
    last = np.full(mask.shape[0], -1)
    np.minimum.at(first, ys, xs)
    np.maximum.at(last, ys, xs)
    rows = np.unique(ys)
    a, b = first[rows], last[rows] + 1
    pts = np.concatenate([np.stack(p, 1) for p in ((a, rows), (a, rows + 1), (b, rows), (b, rows + 1))])
    axis, width, elongated = _tip_axis(cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32))))
    P = np.stack([xs + 0.5, ys + 0.5], 1)
    if elongated:
        d = P - P.mean(0)
        angle = 0.5 * np.arctan2(2 * (d[:, 0] * d[:, 1]).sum(), (d[:, 0] ** 2).sum() - (d[:, 1] ** 2).sum())
        pca = np.array([np.cos(angle), np.sin(angle)])
        axis = pca if pca @ axis >= 0 else -pca
    s = P @ axis
    smax = s.max()
    perp = np.array([-axis[1], axis[0]])
    lat = (P[s >= smax - max(2.0, 0.5 * width)] @ perp).mean()
    tip = axis * (smax + 0.5) + perp * lat        # smax + 0.5 = 맨 끝 픽셀의 바깥 가장자리
    return float(tip[0]), float(tip[1])


def mask_tips(r) -> list:
    """seg 결과 인스턴스마다 mask_tip — 마스크를 원본 크기로 되돌려서 (앱도 원본 크기 마스크를 받는다)."""
    from ultralytics.utils import ops
    full = ops.scale_masks(r.masks.data[None].float(), r.orig_shape[:2])[0] > 0.5
    return [mask_tip(m) for m in full.cpu().numpy()]


def detections(r, tips: bool = True) -> list:
    """점수 높은 순 [{name, conf, color, tip}] — tip 은 seg = mask_tip, OBB = 아랫변 중점 (없거나 tips=False 면 None)."""
    out = []
    if r.obb is not None and len(r.obb):
        for pts, c, s in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.int().tolist(), r.obb.conf.tolist()):
            out.append({"name": r.names[c], "conf": s, "color": colors(c, True), "tip": bottom_mid(pts)})
    elif r.boxes is not None and len(r.boxes):
        found = mask_tips(r) if tips and r.masks is not None else [None] * len(r.boxes)
        for tip, c, s in zip(found, r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            out.append({"name": r.names[c], "conf": s, "color": colors(c, True), "tip": tip})
    return sorted(out, key=lambda d: -d["conf"])


def shapes(r, alpha: float = MASK_ALPHA) -> np.ndarray:
    """원본(r.orig_img) 위에 마스크 / OBB 윤곽만 그린 그림."""
    im = r.orig_img.copy()
    lw = max(1, im.shape[1] // 600)
    if r.masks is not None and len(r.masks):
        fill = im.copy()
        for poly, c in zip(r.masks.xy, r.boxes.cls.int().tolist()):
            if len(poly) >= 3:
                cv2.fillPoly(fill, [poly.astype(np.int32)], colors(c, True))
        im = cv2.addWeighted(fill, alpha, im, 1 - alpha, 0)
        for poly, c in zip(r.masks.xy, r.boxes.cls.int().tolist()):
            if len(poly) >= 3:
                cv2.polylines(im, [poly.astype(np.int32)], True, colors(c, True), lw, cv2.LINE_AA)
    elif r.obb is not None and len(r.obb):
        for pts, c in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.cls.int().tolist()):
            cv2.polylines(im, [pts.astype(np.int32)], True, colors(c, True), lw * 2, cv2.LINE_AA)
    return im


def labels(r) -> list:
    """[(글, BGR 색)] — 점수 높은 순. 검출이 없으면 빈 목록."""
    return [(f"{d['name']} {d['conf']:.2f}", d["color"]) for d in detections(r, tips=False)]


def put_labels(im: np.ndarray, items: list, scale: float | None = None, top: bool = False) -> np.ndarray:
    """
    한 줄씩 — 색 네모 + 흰 글씨(검정 바탕). 색이 None 이면 네모 없이 회색 글씨.
    top=False: 왼쪽 아래부터 위로, top=True: 왼쪽 위부터 아래로. 비어 있으면 'no detection'.
    """
    items = items or [("no detection", None)]
    scale = scale or max(0.45, im.shape[1] / 1400)
    th_px = max(1, round(scale * 2))
    (_, h), base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, th_px)
    step = h + base + 6
    if top:
        ys = [6 + h + 4 + k * step for k in range(len(items))]
    else:
        ys = [im.shape[0] - 8 - base - k * step for k in range(len(items))][::-1]
    for (text, color), y in zip(items, ys):
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th_px)
        x = 6 if color is None else 6 + h + 6
        cv2.rectangle(im, (0, y - h - 4), (x + w + 6, y + base + 2), (0, 0, 0), -1)
        if color is not None:
            cv2.rectangle(im, (6, y - h), (6 + h, y), color, -1)
        cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255) if color is not None else (180, 180, 180), th_px, cv2.LINE_AA)
    return im


def draw_tip(im: np.ndarray, xy: tuple, color: tuple) -> None:
    """끝점 — 흰 테두리 원 + 가운데 색 점 (어두운·밝은 배경 모두에서 보이게)."""
    r = max(3, im.shape[1] // 200)
    p = (round(xy[0]), round(xy[1]))
    cv2.circle(im, p, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(im, p, r, color, -1, cv2.LINE_AA)


def render(r) -> np.ndarray:
    """shapes + 왼쪽 아래 라벨·점수."""
    return put_labels(shapes(r), labels(r))
