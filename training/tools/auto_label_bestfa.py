"""BestFA 자동 1차 라벨 — 세션마다 와이어 축을 찾아 OBB(labels/)를 쓰고, compute edge 2 로 끝을 따서
두 와이어 모두 "ok" 인 장만 seg(labels_seg/)까지 쓴다. 사람이 라벨러에서 검수하는 전제의 초안.

왜 세션 단위인가: 카메라가 토치에 붙어 있어(eye-in-hand) 한 세션 안에서는 와이어 축이 거의 움직이지
않고 끝 길이만 변한다. 한 장에서는 와이어 한쪽이 어두운 배경이라 안 보이는 일이 흔하지만, 세션 100장을
모으면 축이 안정적으로 나온다. (기준 이미지 reference_wire.png 는 10/26 구도라 11/11~13 세션과 맞지 않음)

  1. 축   오른쪽: 각 장에서 "양옆이 충분히 밝은 어두운 띠" 의 행별 중심·폭을 사전 축(PRIOR) 근처에서
          모아 RANSAC 직선 (사전 축에서 TOL 이상 벗어난 직선은 버림).
          왼쪽: 왼편이 어두운 배경이라 그 띠가 짧게만 보여서, 세션 90백분위 이미지의 반사 줄기(와이어
          오른쪽을 따라가는 가는 밝은 선) 직선을 축으로 쓰고 경계는 줄기 + RIDGE_OFFSET (아래 설명).
  2. 윗변  세션 90백분위 이미지에서 축을 따라 위로, 띠 오른쪽(빛 받는 쪽)이 띠보다 밝은 동안 올라간 끝
          = 와이어가 처음 보이는 곳. 라벨러 규약의 피벗(우측 상단 꼭짓점 = 튜브 출구 쪽)에 해당.
          윗변이 TOP_MAX 보다 아래면(위쪽에서 와이어가 안 보임 = 엉뚱한 선) 그 축은 실패 → 세션 실패.
  3. 끝   장마다 그 OBB 로 labeller/edge2.find_tip2 (라벨러 "compute edge 2" 와 같은 함수).
          두 와이어 모두 ok → labels/ + labels_seg/,  아니면 labels/ 만 (라벨러에서 seg 없는 장 = 검수 대상).
          축을 못 찾은 와이어가 있는 세션은 아무것도 쓰지 않는다 (와이어 하나만 라벨하면 다른 하나가
          학습에서 오답이 되므로). 이미 labels/ 나 labels_seg/ 가 있는 장은 건드리지 않는다.

  uv run --with numpy tools/auto_label_bestfa.py --axes-only        # 축만 → 검수 시트·요약 (쓰지 않음)
  uv run --with numpy tools/auto_label_bestfa.py                    # 라벨 쓰기
  uv run --with numpy tools/auto_label_bestfa.py --sessions 261113_163718 ...
산출물(검수용): datasets/bestfa/auto_label/  — axes_*.jpg (세션별 90백분위 + 찾은 OBB), sessions.csv, frames.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parents[1]                      # training/
sys.path[:0] = [str(HERE), str(HERE / "labeller")]
from edge2 import find_tip2  # noqa: E402

RAW = HERE.parent / "datasets" / "bestfa" / "raw"
OUT = HERE.parent / "datasets" / "bestfa" / "auto_label"
CLASS_ID = 0
ROI = (900, 150, 1800, 1050)            # 두 와이어가 있는 영역 (원본 px) — 90백분위 이미지를 이 영역만
YM = 460                                 # 축 x 를 이 행에서 잰다
# 사전 축 (11/11~13 세션에서 측정): x(YM), 수직에서 기운 각도(°, + = 아래로 갈수록 오른쪽)
PRIOR = {"left": (1180.0, 22.0), "right": (1455.0, 8.0)}
HALF = 70                                # 사전 축 좌우 탐색 폭(px)
STEP_MIN = {"left": 25, "right": 40}     # 띠 양옆이 띠보다 이만큼 밝아야 그 행을 씀 (0~255)
W_RANGE = (14, 70)                       # 행별 띠 폭 허용(px, 가로)
Y_RANGE = (280, 720)                     # 띠를 찾는 행 범위
TOL = {"left": (10.0, 60.0), "right": (7.0, 45.0)}   # RANSAC 직선이 사전 축에서 벗어나도 되는 (°, px)
RANSAC_TOL = 3.0                         # 직선까지 거리(px) — 인라이어
MIN_INLIERS = 20
LIT_STEP = 8                             # 윗변 찾기: 빛 받는 쪽이 띠보다 이만큼 밝으면 "보임"
TOP_MAX = 400                            # 윗변(와이어가 처음 보이는 행)이 이보다 아래면 축 실패
LENGTH_MARGIN = 10                       # OBB 길이 = edge2 끝 + 이만큼
# 왼쪽 와이어 — 왼편이 어두운 배경이라 "양옆이 밝은 띠" 가 아크 근처에만 잠깐 보이는 세션이 많다.
# 대신 90백분위 이미지에 와이어 오른쪽을 따라 반사 줄기(가는 밝은 선)가 위에서 아크까지 곧게 보인다.
# 축은 그 줄기 직선으로 잡고, 와이어 오른쪽 경계 = 줄기 + RIDGE_OFFSET (축이 확실한 82세션에서 잰
# 중앙값 −8.2px). 아크 근처 띠 점이 줄기 근처에 충분하면 그 세션 값으로 경계·폭을 다시 맞춘다.
RIDGE_ROWS = (300, 520)
RIDGE_MIN = 10                           # 줄기가 양옆(7~14px 바깥)보다 이만큼 밝아야 그 행을 씀
RIDGE_OFFSET = -8.0
LEFT_WIDTH = 26.4                        # 왼쪽 와이어 가로 폭 기본값(px) — 띠 점이 모자랄 때
REFINE_NEAR = 10.0                       # 띠 오른쪽 경계가 예상 위치에서 이 안이면 보정에 씀
REFINE_MIN = 10
WIDTH_TOL = 5.0                          # 보정한 폭이 기본값에서 이보다 벗어나면 기본값 (141세션 중 23~30px)


def band_points(a: np.ndarray, x0: float, k: float, step_min: float) -> list:
    """블러한 원본(float) 한 장 → 사전 축 근처의 깨끗한 어두운 띠 [(y, xl, xr)]."""
    pts = []
    for y in range(Y_RANGE[0], Y_RANGE[1], 3):
        xc = x0 + k * (y - YM)
        lo, hi = int(xc - HALF), int(xc + HALF + 1)
        p = a[y, lo:hi]
        cm = int(np.argmin(p))
        if cm < 5 or cm > len(p) - 6:
            continue
        v = p[cm]
        lmax, rmax = p[:cm].max(), p[cm + 1:].max()
        if min(lmax, rmax) - v < step_min:
            continue
        tl, tr = v + 0.5 * (lmax - v), v + 0.5 * (rmax - v)
        il = cm
        while il > 0 and p[il - 1] < tl:
            il -= 1
        ir = cm
        while ir < len(p) - 1 and p[ir + 1] < tr:
            ir += 1
        if il == 0 or ir == len(p) - 1:
            continue
        xl = lo + il - 1 + (tl - p[il - 1]) / (p[il] - p[il - 1]) if p[il] != p[il - 1] else lo + il
        xr = lo + ir + (tr - p[ir]) / (p[ir + 1] - p[ir]) if p[ir + 1] != p[ir] else lo + ir + 1
        if W_RANGE[0] <= xr - xl <= W_RANGE[1]:
            pts.append((y, xl, xr))
    return pts


def ransac_line(ys, xs, prior, tol, min_inliers: int = MIN_INLIERS, seed: int = 0):
    """x = a0 + b·y. 사전 축에서 크게 벗어난 후보는 버린다. → (a0, b, inlier mask) 또는 None."""
    rng = np.random.default_rng(seed)
    ys, xs = np.asarray(ys, float), np.asarray(xs, float)
    px0, pdeg = prior
    ang_tol, x_tol = tol
    best = None
    for _ in range(600):
        i, j = rng.choice(len(ys), 2, replace=False)
        if abs(ys[i] - ys[j]) < 60:
            continue
        b = (xs[j] - xs[i]) / (ys[j] - ys[i])
        a0 = xs[i] - b * ys[i]
        if abs(math.degrees(math.atan(b)) - pdeg) > ang_tol or abs(a0 + b * YM - px0) > x_tol:
            continue
        inl = np.abs(xs - (a0 + b * ys)) <= RANSAC_TOL
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < min_inliers:
        return None
    for _ in range(2):
        b, a0 = np.polyfit(ys[best], xs[best], 1)
        best = np.abs(xs - (a0 + b * ys)) <= RANSAC_TOL
    return a0, b, best


def ridge_points(p90: np.ndarray, prior) -> list:
    """90백분위 이미지에서 사전 축 근처 행마다 가장 뚜렷한 가는 밝은 선 [(y, x)] (원본 px)."""
    x0, deg = prior
    k = math.tan(math.radians(deg))
    pts = []
    for y in range(RIDGE_ROWS[0], RIDGE_ROWS[1], 2):
        row = p90[y - ROI[1]]
        lo = int(x0 + k * (y - YM) - ROI[0] - HALF)
        best, bx = RIDGE_MIN, None
        for i in range(max(lo, 14), min(lo + 2 * HALF + 1, len(row) - 15)):
            v = row[i]
            if v >= 240:                                         # 아크(포화)는 줄기가 아님
                continue
            side = max(row[i - 14:i - 7].mean(), row[i + 8:i + 15].mean())
            if v - side >= best:
                best, bx = v - side, i
        if bx is not None:
            pts.append((y, ROI[0] + bx))
    return pts


def band_axis(pts: list, prior, tol) -> dict:
    """양옆이 밝은 띠 점들 → 축 (오른쪽 와이어)."""
    ax = {"ok": False, "points": len(pts), "method": "band"}
    if len(pts) < MIN_INLIERS:
        ax["why"] = f"띠 점 {len(pts)}개"
        return ax
    fit = ransac_line([p[0] for p in pts], [(p[1] + p[2]) / 2 for p in pts], prior, tol)
    if fit is None:
        ax["why"] = "사전 축 근처 직선 없음"
        return ax
    a0, b, inl = fit
    yin = np.asarray([p[0] for p in pts])[inl]
    ax.update(a0=float(a0), b=float(b), inliers=int(inl.sum()),
              wh=float(np.median(np.asarray([p[2] - p[1] for p in pts])[inl])),
              ymin=int(yin.min()), ymax=int(yin.max()))
    return ax


def ridge_axis(p90: np.ndarray, band: list, prior, tol) -> dict:
    """반사 줄기 직선 → 축 (왼쪽 와이어). 줄기 근처의 띠 점으로 경계 위치·폭을 세션별로 보정."""
    rp = ridge_points(p90, prior)
    ax = {"ok": False, "points": len(rp), "method": "ridge"}
    fit = ransac_line([p[0] for p in rp], [p[1] for p in rp], prior, tol, min_inliers=15) if len(rp) >= 15 else None
    if fit is None:
        ax["why"] = f"반사 줄기 없음 (점 {len(rp)}개)"
        return ax
    r0, b, inl = fit
    yr = np.asarray([p[0] for p in rp])[inl]
    offset, wh, near = RIDGE_OFFSET, LEFT_WIDTH, []
    for y, xl, xr in band:
        if abs(xr - (r0 + b * y + RIDGE_OFFSET)) <= REFINE_NEAR:
            near.append((y, xl, xr))
    if len(near) >= REFINE_MIN:
        offset = float(np.median([xr - (r0 + b * y) for y, _, xr in near]))
        wh = float(np.median([xr - xl for _, xl, xr in near]))
        if abs(wh - LEFT_WIDTH) > WIDTH_TOL:                # 띠에 반사 줄기까지 섞이면 폭이 튄다
            wh = LEFT_WIDTH
    ax.update(a0=float(r0 + offset - wh / 2), b=float(b), inliers=int(inl.sum()), wh=wh,
              offset=round(offset, 1), refined=len(near),
              ymin=int(yr.min()), ymax=int(max([yr.max()] + [y for y, _, _ in near])))
    return ax


def session_axes(frames: list) -> tuple[dict, np.ndarray]:
    pooled = {w: [] for w in PRIOR}
    crops = []
    for fp in frames:
        g = Image.open(fp).convert("L")
        a = np.asarray(g.filter(ImageFilter.GaussianBlur(1.5)), np.float32)
        crops.append(np.asarray(g.crop(ROI)))
        for w, (x0, deg) in PRIOR.items():
            pooled[w] += band_points(a, x0, math.tan(math.radians(deg)), STEP_MIN[w])
    stack = np.stack(crops)
    del crops
    kth = int(round(0.9 * (len(stack) - 1)))                 # 90백분위 (uint8 그대로 — float 사본 없이)
    p90 = np.partition(stack, kth, axis=0)[kth].astype(np.float32)
    del stack
    axes = {"left": ridge_axis(p90, pooled["left"], PRIOR["left"], TOL["left"]),
            "right": band_axis(pooled["right"], PRIOR["right"], TOL["right"])}
    for ax in axes.values():
        if "a0" not in ax:
            continue
        ax["deg"] = math.degrees(math.atan(ax["b"]))
        ax["top"] = visible_top(p90, ax)
        if ax["top"] > TOP_MAX:
            ax["why"] = f"와이어가 위쪽에서 안 보임 (top {ax['top']})"
            continue
        ax["ok"] = True
    return axes, p90


def visible_top(p90: np.ndarray, ax: dict) -> int:
    """축을 따라 위로 — 띠 오른쪽(빛 받는 쪽)이 띠보다 LIT_STEP 이상 밝은 동안 (끊김 4행까지 허용)."""
    top, miss = ax["ymin"], 0
    for y in range(ax["ymin"], ROI[1] + 10, -2):
        xc = ax["a0"] + ax["b"] * y - ROI[0]
        row = p90[y - ROI[1]]
        core = row[int(xc - ax["wh"] / 4):int(xc + ax["wh"] / 4) + 1].min()
        lit = row[int(xc + ax["wh"] / 2 - 4):int(xc + ax["wh"] / 2 + 25)].max()
        if lit - core >= LIT_STEP:
            top, miss = y, 0
        else:
            miss += 1
            if miss > 4:
                break
    return top


def obb_corners(ax: dict, length: float) -> list:
    """라벨러 corners() 와 같은 순서 [좌상, 우상(피벗), 우하, 좌하] — 원본 px."""
    b = ax["b"]
    ux, uy = b / math.hypot(b, 1), 1 / math.hypot(b, 1)
    thick = ax["wh"] * uy                                       # 가로 폭 → 축에 수직인 두께
    y = ax["top"]
    px, py = ax["a0"] + b * y + ax["wh"] / 2, float(y)          # 윗변 행의 오른쪽 변 = 피벗
    rx, ry = uy, -ux
    return [(px - thick * rx, py - thick * ry), (px, py), (px + length * ux, py + length * uy),
            (px + length * ux - thick * rx, py + length * uy - thick * ry)]


def fmt_points(points, w: int, h: int) -> str:
    return " ".join(f"{max(0.0, min(1.0, x / w)):.6f} {max(0.0, min(1.0, y / h)):.6f}" for x, y in points)


def label_session(session: Path, write: bool) -> dict:
    frames = sorted((session / "Mono").glob("*.jpg"))
    axes, p90 = session_axes(frames)
    summary = {"session": session.name, "frames": len(frames)}
    for w in PRIOR:
        a = axes[w]
        summary[f"{w}_axis"] = "ok" if a["ok"] else a.get("why", "")
        summary[f"{w}_deg"] = round(a["deg"], 1) if "deg" in a else ""
        summary[f"{w}_width"] = round(a["wh"], 1) if "wh" in a else ""
        summary[f"{w}_top"] = a.get("top", "")
        summary[f"{w}_inliers"] = f"{a.get('inliers', 0)}/{a['points']}"
    summary["left_offset"] = axes["left"].get("offset", "")
    summary["left_refined"] = axes["left"].get("refined", "")
    rows = []
    counts = Counter()
    if all(a["ok"] for a in axes.values()):
        for fp in frames:
            lbl = session / "labels" / f"{fp.stem}.txt"
            seg = session / "labels_seg" / f"{fp.stem}.txt"
            if lbl.exists() or seg.exists():
                counts["kept_existing"] += 1
                rows.append({"session": session.name, "image": fp.name, "result": "kept_existing"})
                continue
            with Image.open(fp) as im:
                w, h = im.size
            boxes, polys, stats = [], [], []
            for name in ("left", "right"):
                ax = axes[name]
                long_box = obb_corners(ax, 900.0)                # 끝을 넉넉히 지나도록 — edge2 탐색 범위
                r = find_tip2(fp, [[x / w, y / h] for x, y in long_box])
                stats.append(r["status"])
                tip = r.get("tip") if r["status"] != "no_tip" else None
                length = (tip + LENGTH_MARGIN) if tip else (ax["ymax"] - ax["top"]) / (1 / math.hypot(ax["b"], 1))
                boxes.append(obb_corners(ax, length))
                polys.append(r.get("polygon"))
            both_ok = stats == ["ok", "ok"]
            result = "obb+seg" if both_ok else "obb_only"
            counts[result] += 1
            rows.append({"session": session.name, "image": fp.name, "result": result,
                         "left": stats[0], "right": stats[1]})
            if write:
                lbl.parent.mkdir(exist_ok=True)
                lbl.write_text("".join(f"{CLASS_ID} {fmt_points(b, w, h)}\n" for b in boxes), encoding="utf-8")
                if both_ok:
                    seg.parent.mkdir(exist_ok=True)
                    seg.write_text("".join(f"{CLASS_ID} {fmt_points(p, w, h)}\n" for p in polys),
                                   encoding="utf-8")
    else:
        counts["session_skipped"] = len(frames)
    summary.update({k: counts.get(k, 0) for k in ("obb+seg", "obb_only", "kept_existing", "session_skipped")})
    # 검수 썸네일: 90백분위 + 찾은 OBB (빨강 = 실패한 축)
    vis = Image.fromarray(p90.astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(vis)
    for ax in axes.values():
        if "a0" in ax:
            c = obb_corners(ax, (ax["ymax"] - ax["top"]) * math.hypot(ax["b"], 1))
            d.polygon([(x - ROI[0], y - ROI[1]) for x, y in c], outline=(0, 255, 0) if ax["ok"] else (255, 0, 0),
                      width=3)
    thumb = vis.resize((300, 300))
    ImageDraw.Draw(thumb).text((4, 4), session.name + ("" if all(a["ok"] for a in axes.values()) else "  FAIL"),
                               fill=(255, 255, 0))
    return {"summary": summary, "rows": rows, "thumb": thumb}


def merge_csv(path: Path, rows: list, sessions: set, fields: list | None = None) -> None:
    """path 의 기존 행 중 sessions 에 속한 것만 새 rows 로 바꿔 세션 순으로 다시 쓴다."""
    old = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fp:
            old = [r for r in csv.DictReader(fp) if r["session"] not in sessions]
    fields = fields or list(rows[0])
    merged = sorted(old + rows, key=lambda r: (r["session"], r.get("image", "")))
    with path.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(merged)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--axes-only", action="store_true", help="축만 찾아 검수 시트·요약 (라벨 안 씀)")
    ap.add_argument("--sessions", nargs="*", help="이 세션만 (raw 폴더 이름)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 4))   # 작업자당 수백 MB
    args = ap.parse_args()
    sessions = sorted(p for p in RAW.glob("[0-9]*") if (p / "Mono").is_dir())
    if args.sessions:
        sessions = [p for p in sessions if p.name in set(args.sessions)]
    OUT.mkdir(parents=True, exist_ok=True)
    write = not args.axes_only
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(label_session, s, write): s for s in sessions}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            s = r["summary"]
            results.append(r)
            print(f"[{i}/{len(sessions)}] {s['session']}  L:{s['left_axis']}  R:{s['right_axis']}  "
                  f"seg {s['obb+seg']} / obb만 {s['obb_only']} / 건너뜀 {s['session_skipped']}", flush=True)
    results.sort(key=lambda r: r["summary"]["session"])
    tag = "axes" if args.axes_only else "run"
    done = {r["summary"]["session"] for r in results}
    # 요약은 세션 단위로 합친다 — --sessions 로 일부만 다시 돌려도 나머지 세션 행은 남는다
    merge_csv(OUT / f"sessions_{tag}.csv", [r["summary"] for r in results], done)
    if write:
        merge_csv(OUT / "frames_run.csv", [row for r in results for row in r["rows"]], done,
                  fields=["session", "image", "result", "left", "right"])
    cols, per = 6, 36
    prefix = tag if not args.sessions else f"{tag}_partial"
    for n in range(0, len(results), per):
        chunk = results[n:n + per]
        sheet = Image.new("RGB", (cols * 300, ((len(chunk) + cols - 1) // cols) * 300))
        for i, r in enumerate(chunk):
            sheet.paste(r["thumb"], ((i % cols) * 300, (i // cols) * 300))
        sheet.save(OUT / f"{prefix}_{n // per + 1:02d}.jpg", quality=85)
    total = Counter()
    for r in results:
        for k in ("obb+seg", "obb_only", "session_skipped"):
            total[k] += r["summary"][k]
    failed = [r["summary"]["session"] for r in results if r["summary"]["session_skipped"]]
    print(f"세션 {len(results)}개 — 축 실패 {len(failed)}개: {failed}")
    if write:
        print(f"장: seg까지 {total['obb+seg']}, OBB만 {total['obb_only']}, 세션 실패로 건너뜀 {total['session_skipped']}")
    print(f"검수 자료 → {OUT}")


if __name__ == "__main__":
    main()
