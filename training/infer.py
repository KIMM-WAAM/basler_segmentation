"""
학습한 모델로 추론 — 사진 한 장(크롭 → 예측 그림) 또는 세션 폴더(전 프레임 → 동영상).

    python infer.py --weights runs/kimm/segment/kimm-yolo26s-seg-split/weights/best.pt \\
                    --source ../datasets/kimm/raw/261019_161818/Mono/000100__cap_261019_161818.png
    python infer.py --weights runs/kimm/segment/kimm-yolo26s-seg-split/weights/best.pt \\
                    --source ../datasets/kimm/raw/261019_161818 --fps 5

- 크롭은 학습 전처리와 같은 규약(preprocess.crop_rect: W×H 픽셀, 위치 0~100%).
  --crop auto(기본)면 세션 crop.yaml, 없으면 오류, "none" 이면 크롭 안 함.
  crop.yaml 에 ROI 가 여럿이면(BestFA) 앱처럼 ROI 마다 따로 추론해 가로로 이어 붙인다.
- 결과는 runs/<환경>/infer/<YYMMDD_HHMMSS>/ — 사진은 <이름>_pred.jpg, 세션은 <세션>.mp4.
- 경로는 적은 그대로 먼저, 없으면 training/ 기준으로 찾는다. weights 는 타임스탬프를 뺀
  runs/<…>/<name>/weights/best.pt 로 적으면 그 name 의 최신 실행 (resolve_weights).
- seg / obb 모델 모두 — 그림은 render.py (bbox 없이 마스크·OBB 윤곽). 왼쪽 위에 파일명과
  검출마다 클래스·점수·와이어 끝점(render.mask_tip — 앱 · WAAM WireTip 과 같은 정의, OBB 모델은 아랫변 중점, 그림에 점) 픽셀 좌표 — 크롭 기준, 크롭했으면
  원본(풀 프레임) 좌표도. 기본 검출 수는 크롭 창(ROI)마다 wire_count / roi_count (겹친 중복 검출 제거).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import HERE, IMG_EXTS, load_yaml, raw_image_dir, resolve, resolve_weights, run_stamp
from dataset_paths import DATASETS, check_source_dataset, crop_windows, default_crop, preprocess_config
from preprocess import crop_rect

BATCH = 16


def find(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.exists() else resolve(path_str)


def parse_crop(text: str, dataset: str = "kimm") -> dict | None:
    """'' / 'auto' → 선택한 환경의 크롭, 'none' → 크롭 안 함, 'W H CX CY' → 그 값."""
    text = (text or "").strip()
    if not text or text.lower() == "auto":
        return default_crop(dataset)
    if text.lower() == "none":
        return None
    try:
        w, h, cx, cy = (float(v) for v in text.replace(",", " ").split())
    except ValueError:
        raise SystemExit(f"--crop 은 'W H CX CY' (예: 1280 1280 50 50), auto, 또는 none: {text!r}") from None
    return {"width": int(w), "height": int(h), "center_x": cx, "center_y": cy}


def frames_of(source: Path) -> list:
    """사진 한 장 → [그 사진], 세션 폴더(또는 그 Mono/) → 정렬된 이미지 전부."""
    if source.is_file():
        return [source]
    raw = raw_image_dir(source)
    d = raw if raw.is_dir() else source
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"이미지가 없습니다: {d}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="학습한 모델로 추론 (사진 → 그림, 세션 → 동영상)")
    ap.add_argument("--weights", required=True, help="best.pt (training/ 기준 또는 절대경로)")
    ap.add_argument("--dataset", choices=DATASETS, default="kimm", help="크롭 설정·결과 저장 환경")
    ap.add_argument("--source", required=True, help="사진 파일 또는 세션 폴더(datasets/<환경>/raw/<세션>)")
    ap.add_argument("--crop", default="auto", help="'W H CX CY' / auto = 세션 크롭 (없으면 오류) / none")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-det", type=int,
                    help="크롭 창(ROI)마다 최대 검출 수 (기본: 환경별 wire_count / roi_count)")
    ap.add_argument("--fps", type=float, default=5, help="세션 동영상 fps")
    ap.add_argument("--out", default="", help="결과 폴더 (기본 runs/<환경>/infer/<시각>/)")
    args = ap.parse_args()
    if args.max_det is None:
        env = load_yaml(preprocess_config(args.dataset))
        args.max_det = max(1, -(-int(env.get("wire_count", 1)) // int(env.get("roi_count", 1))))

    weights = Path(args.weights) if Path(args.weights).exists() else resolve_weights(args.weights)
    if not weights.exists():
        raise SystemExit(f"가중치가 없습니다: {weights}")
    source = find(args.source)
    if not source.exists():
        raise SystemExit(f"입력이 없습니다: {source}")
    try:
        check_source_dataset(source, args.dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    crop = (default_crop(args.dataset, source) if not args.crop.strip() or args.crop.lower() == "auto"
            else parse_crop(args.crop, args.dataset))
    files = frames_of(source)
    out = find(args.out) if args.out else HERE / "runs" / args.dataset / "infer" / run_stamp()
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    import render

    model = YOLO(str(weights))
    print(f"[infer] 가중치: {weights}")
    windows = crop_windows(crop) if crop else []
    print(f"[infer] 입력: {source} ({len(files)}장)  크롭: "
          + (" · ".join(f"{w['width']}×{w['height']} @ ({w['center_x']}%, {w['center_y']}%)"
                        for w in windows) or "없음"))

    def load(p: Path) -> list:
        """[(크롭 그림, 크롭 왼쪽 위의 원본 좌표)] — 크롭 창(ROI)마다 하나, 크롭이 없으면 원본 하나."""
        im = cv2.imread(str(p))
        if not windows:
            return [(im, (0, 0))]
        parts = []
        for window in windows:
            x, y, w, h = crop_rect(im.shape[1], im.shape[0], window)
            parts.append((im[y:y + h, x:x + w], (x, y)))
        return parts

    def draw(r, name: str, offset: tuple):
        im = render.shapes(r)
        items = [(name, None)]
        for d in render.detections(r):
            items.append((f"{d['name']} {d['conf']:.2f}", d["color"]))
            if d["tip"] is not None:
                # 끝점 좌표는 한 줄씩 (색 네모 없는 줄) — crop 은 크롭 그림 기준, full 은 원본 기준
                x, y = d["tip"]
                render.draw_tip(im, d["tip"], d["color"])
                if crop:
                    items.append((f"crop ({x:.1f}, {y:.1f})", None))
                items.append((f"full ({x + offset[0]:.1f}, {y + offset[1]:.1f})", None))
        if len(items) == 1:
            items.append(("no detection", None))
        return render.put_labels(im, items, top=True)

    def tile(images: list):
        """ROI 그림들을 가로로 — 높이가 다르면 아래를 검게 채운다."""
        if len(images) == 1:
            return images[0]
        h = max(im.shape[0] for im in images)
        return cv2.hconcat([cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                            for im in images])

    kw = dict(imgsz=args.imgsz, conf=args.conf, max_det=args.max_det, verbose=False)

    def predict(chunk: list) -> list:
        """파일들 → 프레임 그림들. ROI 들은 한 배치로 같이 추론한다."""
        parts = [(p, part) for p in chunk for part in load(p)]
        results = model.predict([im for _, (im, _) in parts], **kw)
        frames: dict = {}
        for k, ((p, (_, offset)), r) in enumerate(zip(parts, results)):
            roi = k % max(len(windows), 1)
            name = p.name + (f"  ROI {roi + 1}" if len(windows) > 1 else "")
            frames.setdefault(p, []).append(draw(r, name, offset))
        return [tile(frames[p]) for p in chunk]

    if len(files) == 1:
        dst = out / f"{files[0].stem}_pred.jpg"
        cv2.imwrite(str(dst), predict(files)[0])
        print(f"[infer] → {dst}")
        return

    session = source.parent.name if source.name in ("Mono", "images") else source.name
    dst = out / f"{session}.mp4"
    writer = None
    for i in range(0, len(files), BATCH):
        chunk = files[i:i + BATCH]
        for frame in predict(chunk):
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(frame)
        print(f"[infer] {min(i + BATCH, len(files))}/{len(files)}", end="\r")
    writer.release()
    print(f"\n[infer] → {dst}  ({len(files)}프레임, {args.fps:g} fps)")


if __name__ == "__main__":
    main()
