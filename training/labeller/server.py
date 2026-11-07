"""
raw/<source>/Mono/* 에 OBB 라벨(단일 클래스)을 붙이는 브라우저 라벨러.

카메라가 eye-in-hand 라 기구부(금속 튜브 + 와이어)는 화면에서 늘 같은 자리에 있고,
와이어는 튜브 출구를 축으로 휘거나 길이만 변합니다. 그래서 라벨러는 앱(BaslerLiveView)
과 같은 것을 보여주고, 와이어 박스를 "그리는" 대신 기준 자세에서 맞춰 옮깁니다:
  · 선택한 환경의 reference/ 누끼·윤곽선을 전처리 설정의 불투명도로 겹쳐 표시
  · 와이어 박스 = 기준 와이어 OBB(reference_tube/wire 레이어, wire_template) 에서 시작해
    우측 상단 꼭짓점(튜브 출구, 피벗)을 기준으로 이동·회전 (+ 두께·길이)
  · 크롭 창을 앱처럼 W×H · 위치(%) 로 조절하고 환경별 전처리 설정에 저장 (save_crop)
한 녹화 세션 안에서는 위치가 거의 같으므로, 1장 맞춰 "일괄 복사"로 그 세션의 나머지에
뿌린 뒤 어긋난 장만 고쳐 저장하는 흐름을 가정합니다 (복사는 지금 이미지의 세션 안에서만).

소스는 configs/preprocess.<환경>.yaml 의 소스 이름입니다. 그 path 가 글롭(`raw/[0-9]*`)이면
앱 REC 세션 폴더 전부를 이어서 보여줍니다 (Source 참고). 산출물은 preprocess.py 가
labels_seg 를 읽습니다. labels 는 수동 OBB 편집용으로 따로 보관합니다:
    <세션 폴더>/labels/<이미지이름>.txt   "0 x1 y1 x2 y2 x3 y3 x4 y4" (정규화 4점)
    <세션 폴더>/data.yaml                  없으면 해당 환경 설정의 최종 names 로 자동 생성
    <세션 폴더>/session.yaml               없으면 id · env 만 채운 세션 명세 틀을 자동 생성
    <세션 폴더>/labels_seg/<이미지이름>.txt  와이어 폴리곤 "0 x1 y1 … xn yn" (compute edge 1 /
                                            OBB as edge 결과를 저장한 것 — labels/ 는 그대로)

실행 (training/ 또는 저장소 루트의 실행 진입점):
    uv run labeller/server.py [--port 8765]
    브라우저에서 http://localhost:8765 접속 후 KIMM / BestFA 프리셋 선택.
    원격 GPU 머신에서 돌린다면 로컬에서: ssh -L 8765:localhost:8765 <머신>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, unquote, urlparse

import yaml
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
REPO = TRAINING.parent
PREPROCESS_CFG = TRAINING / "configs" / "preprocess.kimm.yaml"
# training/tools/make_reference_cutout.py --tube/--wire 의 입력 레이어 (풀 프레임 RGBA)
REF_TUBE = REPO / "datasets" / "kimm" / "reference" / "reference_tube.png"
REF_WIRE = REPO / "datasets" / "kimm" / "reference" / "reference_wire.png"

sys.path.insert(0, str(TRAINING))
from edge import find_tip, obb_pose  # noqa: E402 — 같은 폴더 (labeller/)
from edge2 import find_tip2  # noqa: E402 — compute edge 2 (BestFA 용)
from reference import red_wire_templates  # noqa: E402

from common import raw_image_dir  # noqa: E402
from dataset_paths import (  # noqa: E402
    DATASETS,
    SESSION_GLOB,
    dataset_root,
    ensure_session_meta,
    preprocess_config,
    session_crop,
)
from preprocess import min_area_rect  # noqa: E402 — training/ 를 path 에 넣은 뒤라야 import 됨

IMG_EXTS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
           ".bmp": "image/bmp", ".webp": "image/webp"}
INDEX_HTML = HERE / "index.html"
CROP_KEYS = ("enabled", "width", "height", "center_x", "center_y")
# "완전히 검은 이미지" — 가장 밝은 픽셀이 이 값 이하 (0~255). 2026-11 KIMM·BestFA 원본에서 빈 프레임은
# 전부 ≤ 10 이었고, 그다음 무리는 15 이상(어둡지만 장면이 있음)이라 그 사이로 잡았다.
BLACK_MAX = 10


def brightness(fp: Path) -> tuple[int, int]:
    """(p99, 최댓값) 밝기 — JPEG 은 1/8 축소 디코딩으로 빠르게 (최댓값 판정에는 충분)."""
    with Image.open(fp) as im:
        im.draft("L", (max(1, im.width // 8), max(1, im.height // 8)))
        g = im.convert("L")
        hist = g.histogram()
    total, acc, p99 = sum(hist), 0, 255
    for v, c in enumerate(hist):
        acc += c
        if acc >= total * 0.99:
            p99 = v
            break
    return p99, max(v for v, c in enumerate(hist) if c)


# ── 기구부 오버레이 + 기준 와이어 ─────────────────────────────────────────────

def source_reference_settings(source) -> dict:
    """선택한 환경의 기준 파일만 읽는다. 파일이 없어도 다른 환경으로 대체하지 않는다."""
    settings = source.cfg.get("reference") or {}
    directory = Path(settings.get("directory") or (dataset_root(source.name) / "reference"))
    if not directory.is_absolute():
        directory = TRAINING / directory
    return {"enabled": settings.get("enabled", True),
            "opacity": float(settings.get("opacity", 0.35)),
            "outline_opacity": float(settings.get("outline_opacity", 0.9)),
            "white_background": bool(settings.get("white_background", False)),
            "wire_mask": settings.get("wire_mask", "alpha"),
            "cutout": directory / "reference_cutout.png",
            "outline": directory / "reference_outline.png",
            "tube": directory / "reference_tube.png",
            "wire": directory / "reference_wire.png"}


def wire_template(tube_path: Path = REF_TUBE, wire_path: Path = REF_WIRE) -> dict | None:
    """
    기준 와이어 OBB — 새 라벨의 초기값. 앱 오버레이의 빨간 와이어 박스와 같은 계산
    (training/tools/make_reference_cutout.py: 튜브 밖에 실제로 보이는 와이어 픽셀만의 회전 최소
    외접 사각형. 그쪽은 cv2 + 스무딩한 튜브 윤곽, 여기는 PIL + 튜브 알파 그대로라
    위 변이 몇 px 다를 수 있음). 가리는 튜브를 빼지 않으면 위 변이 튜브 속까지
    올라가 피벗이 튜브 출구에서 어긋납니다. 레이어가 없으면 None.
    """
    if not (tube_path.exists() and wire_path.exists()):
        return None

    def mask(p: Path):
        with Image.open(p) as im:
            return im.convert("RGBA").getchannel("A").point(lambda v: 255 if v else 0)

    wire, tube = mask(wire_path), mask(tube_path)
    if wire.size != tube.size:
        return None
    visible = ImageChops.subtract(wire, tube)
    w, h = visible.size
    pts = []
    for y in range(h):               # 행마다 양 끝 픽셀만 — 볼록 껍질에는 이것으로 충분
        bb = visible.crop((0, y, w, y + 1)).getbbox()
        if bb:
            pts += [(bb[0], y), (bb[2] - 1, y)]
    if len(pts) < 3:
        return None
    rect, area = min_area_rect(pts)
    if area <= 0:
        return None
    return {**obb_pose(rect), "size": [w, h]}


# ── preprocess.kimm.yaml 의 crop 제자리 수정 ───────────────────────────────────────

def save_crop(cfg_path: Path, source: str, values: dict) -> None:
    """
    preprocess.kimm.yaml 에서 sources[name=source].crop 의 enabled/width/height/center_x/
    center_y 값만 그 줄에서 고쳐 씁니다 — 주석·순서·다른 소스는 그대로 (yaml.safe_dump
    로 다시 쓰면 주석이 전부 날아갑니다). auto_crop 등 하위 블록은 건드리지 않고,
    없는 키는 crop: 바로 아래에 추가합니다.

    고친 텍스트를 다시 파싱해 "원래 내용 + 이 다섯 값"과 정확히 같을 때만 저장합니다
    (아니면 ValueError — 파일은 그대로).
    """
    want = {"enabled": bool(values["enabled"]),
            "width": int(values["width"]), "height": int(values["height"]),
            "center_x": float(values["center_x"]), "center_y": float(values["center_y"])}
    if want["width"] < 0 or want["height"] < 0:
        raise ValueError("width/height 는 0 이상이어야 합니다 (0 = 그 축 원본 전체)")
    for k in ("center_x", "center_y"):
        if not 0 <= want[k] <= 100:
            raise ValueError(f"{k} 는 0~100(%) 이어야 합니다: {want[k]}")

    def fmt(k: str) -> str:
        v = want[k]
        return ("true" if v else "false") if k == "enabled" else f"{v:g}"

    raw = cfg_path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.splitlines(keepends=True)

    def text(i: int) -> str:
        return lines[i].rstrip("\r\n")

    def lead(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    def is_code(s: str) -> bool:
        t = s.strip()
        return bool(t) and not t.startswith("#")

    item_re = re.compile(r"^(\s*)-\s+name:\s*['\"]?([^'\"#\s]+)['\"]?\s*(#.*)?$")
    start = dash = None
    for i in range(len(lines)):
        m = item_re.match(text(i))
        if m and m.group(2) == source:
            start, dash = i, len(m.group(1))
            break
    if start is None:
        raise ValueError(f"{cfg_path.name}: sources 에 {source} 가 없습니다")
    key_indent = text(start).index("name")       # 이 항목의 키들이 서는 열
    end = next((j for j in range(start + 1, len(lines))
                if is_code(text(j)) and lead(text(j)) <= dash), len(lines))

    crop_at = next((j for j in range(start + 1, end)
                    if is_code(text(j)) and lead(text(j)) == key_indent
                    and text(j).strip().startswith("crop:")), None)
    if crop_at is None:
        raise ValueError(f"{source}: crop: 블록이 없습니다 — preprocess.kimm.yaml 에 직접 추가하세요")
    if is_code(text(crop_at).strip()[len("crop:"):]):
        raise ValueError(f"{source}: crop 이 한 줄(flow) 형식이라 고칠 수 없습니다 — "
                         f"여러 줄 블록으로 바꿔 주세요")
    block_end = next((j for j in range(crop_at + 1, end)
                      if is_code(text(j)) and lead(text(j)) <= key_indent), end)
    children = [j for j in range(crop_at + 1, block_end) if is_code(text(j))]
    child_indent = min((lead(text(j)) for j in children), default=key_indent + 2)

    line_re = re.compile(r"^(\s*)(\w+):(\s*)([^#\r\n]*?)(\s*#[^\r\n]*)?(\r?\n)?$")
    found = set()
    for j in children:
        m = line_re.match(lines[j])
        if m and lead(text(j)) == child_indent and m.group(2) in CROP_KEYS:
            ind, key, sp, _, comment, nl = m.groups()
            lines[j] = f"{ind}{key}:{sp or ' '}{fmt(key)}{comment or ''}{nl or ''}"
            found.add(key)
    missing = [k for k in CROP_KEYS if k not in found]
    lines[crop_at + 1:crop_at + 1] = [f"{' ' * child_indent}{k}: {fmt(k)}{eol}" for k in missing]

    new = "".join(lines)
    expect = yaml.safe_load(raw) or {}
    for s in expect.get("sources") or []:
        if s.get("name") == source:
            if not isinstance(s.get("crop"), dict):
                s["crop"] = {}
            s["crop"].update(want)
    if (yaml.safe_load(new) or {}) != expect:
        raise ValueError(f"{cfg_path.name} 를 고친 결과가 의도와 달라 저장하지 않았습니다 — "
                         f"{source} 의 crop 블록을 직접 확인해 주세요")
    tmp = cfg_path.with_name(cfg_path.name + ".tmp")
    tmp.write_bytes(new.encode("utf-8"))
    os.replace(tmp, cfg_path)


class Source:
    """
    라벨링 대상 소스 — 경로 · 클래스 정의 · crop 미리보기 설정.

    `--source` 는 configs/preprocess.kimm.yaml 의 소스 **이름**입니다. 그 소스의 path 가
    글롭이면(앱 REC 는 녹화 세션마다 raw/YYMMDD_HHMMSS/ 를 새로 만듭니다
    — FrameRecorder) 매칭되는 세션 폴더 전부를 한 소스처럼 이어서 라벨링합니다.
    preprocess.py 의 source_dirs 와 같은 규약이라, 여기서 라벨한 것이 그대로
    전처리 대상이 됩니다.

    라벨은 각 이미지가 있던 세션 폴더의 labels/ 에 씁니다 (preprocess.py 가 읽는 자리).
    앱 캡처 파일명(000001__cap_YYMMDD_HHMMSS.png)은 세션 타임스탬프가 들어가 전역으로
    유일하므로, 브라우저 API 는 파일명만 주고받고 소속 폴더는 서버가 기억합니다.

    설정에 소스가 없으면 datasets/<환경>/raw/[0-9]* 를 읽습니다.
    """

    def __init__(self, name: str, config_path: Path | None = None) -> None:
        self.name = name
        self.config_path = config_path or (PREPROCESS_CFG if name == "kimm" else preprocess_config(name))
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        if self.cfg.get("dataset") and self.cfg["dataset"] != name:
            raise ValueError(f"{self.config_path.name}: 촬영 환경이 {name} 과 다릅니다")
        self.names = self._final_names()
        self.class_id = min(self.names) if self.names else 0
        self.wire_count = int(self.cfg.get("wire_count", 1))
        self.dirs = self._resolve_dirs()

        # 이미지 파일명 → 그 이미지가 있는 세션 폴더
        self._owner: dict = {}
        for d in self.dirs:
            (d / "labels").mkdir(parents=True, exist_ok=True)
            self._ensure_data_yaml(d)
            ensure_session_meta(d, self.cfg.get("dataset") or self.name, wire_count=self.wire_count)
            for p in sorted(raw_image_dir(d).iterdir()):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                prev = self._owner.get(p.name)
                if prev is not None:
                    # 파일명으로만 주고받는 구조라 겹치면 라벨이 엉뚱한 쪽에 붙습니다.
                    raise SystemExit(
                        f"파일명이 겹칩니다 — 한쪽을 지우거나 이름을 바꾸세요.\n"
                        f"  {raw_image_dir(prev) / p.name}\n  {p}")
                self._owner[p.name] = d

    def _source_entry(self) -> dict | None:
        """preprocess.kimm.yaml 의 sources 에서 이 이름의 소스 항목."""
        for src in self.cfg.get("sources") or []:
            if src.get("name") == self.name:
                return src
        return None

    def _resolve_dirs(self) -> list:
        """
        읽을 세션 폴더들 (이름순). preprocess.py 의 source_dirs 와 같은 규약 —
        글롭은 경로 마지막 칸에만, Mono/ 가 있는 폴더만 셉니다.
        """
        entry = self._source_entry() or {}
        raw = str(entry.get("path") or (dataset_root(self.name) / "raw" / SESSION_GLOB))
        base = Path(raw)
        base = base if base.is_absolute() else (TRAINING / base)

        if not any(c in base.name for c in "*?["):
            if not raw_image_dir(base).is_dir():
                raise SystemExit(f"이미지 폴더가 없습니다: {raw_image_dir(base)}")
            return [base]

        dirs = sorted(p for p in base.parent.glob(base.name)
                      if p.is_dir() and raw_image_dir(p).is_dir())
        if not dirs:
            raise SystemExit(f"'{base}' 에 맞는 폴더가 없습니다 "
                             f"(Mono/ 또는 이전 images/ 를 가진 폴더만 셉니다).")
        return dirs

    def _final_names(self) -> dict:
        """configs/preprocess.kimm.yaml 의 최종 클래스(names) — 없으면 기본 {0: wire}."""
        names = self.cfg.get("names")
        if names:
            return {int(k): str(v) for k, v in names.items()}
        return {0: "wire"}

    def _ensure_data_yaml(self, d: Path) -> None:
        """raw 소스 data.yaml 이 없으면 preprocess.py 가 요구하는 최소 형식으로 생성."""
        dy = d / "data.yaml"
        if dy.exists():
            return
        doc = {"names": self.names, "train": raw_image_dir(d).name}
        dy.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")

    def crop_state(self, name: str | None = None) -> dict | None:
        """미설정 세션은 중앙 1280 크롭을 제안한다. 저장 전에는 미설정 상태다."""
        src = self._source_entry()
        if src is None:
            return None
        configured = name is not None and (self.dir_of(name) / "crop.yaml").is_file()
        crop = session_crop(self.dir_of(name)) if configured else {
            "enabled": True, "width": 1280, "height": 1280,
            "center_x": 50, "center_y": 50,
        }
        return {"configured": configured, "enabled": bool(crop.get("enabled")),
                "width": int(crop.get("width") or 0), "height": int(crop.get("height") or 0),
                "center_x": float(crop.get("center_x", 50)),
                "center_y": float(crop.get("center_y", 50)),
                "auto": bool((crop.get("auto_crop") or {}).get("enabled"))}

    def save_crop(self, values: dict) -> dict:
        """현재 이미지가 속한 세션에 고정 크롭을 저장한다."""
        name = safe_name(values.get("name", ""))
        directory = self.dir_of(name)
        crop = {"enabled": bool(values["enabled"]),
                "width": int(values["width"]), "height": int(values["height"]),
                "center_x": float(values["center_x"]), "center_y": float(values["center_y"]),
                "auto_crop": {"enabled": False}}
        if crop["width"] < 0 or crop["height"] < 0:
            raise ValueError("크롭 크기는 0 이상이어야 합니다")
        if not all(math.isfinite(crop[k]) and 0 <= crop[k] <= 100 for k in ("center_x", "center_y")):
            raise ValueError("크롭 위치는 0~100%여야 합니다")
        path = directory / "crop.yaml"
        previous = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
        previous = previous if isinstance(previous, dict) else {}
        previous.update(crop)
        crop = previous
        path = directory / "crop.yaml"
        temp = path.with_suffix(".yaml.tmp")
        temp.write_text(yaml.safe_dump(crop, sort_keys=False), encoding="utf-8")
        os.replace(temp, path)
        return self.crop_state(name)

    def read_seg(self, name: str) -> list:
        path = self.seg_path(name)
        if not path.exists():
            return []
        with Image.open(self.image_path(name)) as image:
            width, height = image.size
        polygons = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = line.split()
            coords = [float(v) for v in values[1:]]
            if len(coords) < 6 or len(coords) % 2 or not all(math.isfinite(v) for v in coords):
                raise ValueError(f"잘못된 seg 라벨: {path.name}")
            polygons.append([[coords[i] * width, coords[i + 1] * height]
                             for i in range(0, len(coords), 2)])
        return polygons

    def images(self) -> list:
        # 세션 폴더 순 → 그 안에서 파일명 순 (_owner 의 삽입 순서). 파일명으로만
        # 정렬하면 세션들이 프레임 단위로 뒤섞여 "세션 하나 그리고 → 그 세션에 복사"
        # 흐름이 깨집니다.
        return list(self._owner)

    def dir_of(self, name: str) -> Path:
        d = self._owner.get(name)
        if d is None:
            raise ValueError(f"이 소스에 없는 이미지: {name!r}")
        return d

    def image_path(self, name: str) -> Path:
        return raw_image_dir(self.dir_of(name)) / name

    def label_path(self, name: str) -> Path:
        return self.dir_of(name) / "labels" / (Path(name).stem + ".txt")

    def read_labels(self, name: str) -> list:
        p = self.label_path(name)
        if not p.exists():
            return []
        instances = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 9 or int(parts[0]) != self.class_id:
                raise ValueError(f"OBB 라벨 형식 오류: {p.name}")
            coords = [float(v) for v in parts[1:]]
            points = [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]
            validate_polygon(points, 4, exact=True)
            instances.append(points)
        return instances

    def read_label(self, name: str) -> list | None:
        instances = self.read_labels(name)
        return instances[0] if instances else None

    def write_label(self, name: str, points: list | None) -> None:
        self.write_labels(name, [points] if points else [])

    def write_labels(self, name: str, instances: list) -> None:
        lines = []
        for points in instances:
            validate_polygon(points, 4, exact=True)
            flat = " ".join(f"{max(0.0, min(1.0, v)):.6f}" for xy in points for v in xy)
            lines.append(f"{self.class_id} {flat}\n")
        p = self.label_path(name)
        # seg 폴리곤은 이 OBB 의 윗변·옆변으로 만든 것이라, OBB 가 바뀌면 낡는다 → 같이 지움
        self.seg_path(name).unlink(missing_ok=True)
        if not lines:
            p.unlink(missing_ok=True)
            return
        p.write_text("".join(lines), encoding="utf-8")

    def delete_image(self, name: str, bin_name: str = "_removed_manual") -> Path:
        """
        이미지 + 짝 라벨(labels/, labels_seg/)을 <raw>/<bin_name>/<세션>/ 으로 옮긴다 —
        글롭([0-9]*)에 안 걸려 라벨러·전처리에서 바로 빠지고, 잘못 지웠으면 되돌릴 수 있음.
        """
        d = self.dir_of(name)
        trash = d.parent / bin_name / d.name
        stem = Path(name).stem
        for sub, fname in ((raw_image_dir(d).name, name), ("labels", stem + ".txt"),
                           ("labels_seg", stem + ".txt")):
            src = d / sub / fname
            if src.exists():
                (trash / sub).mkdir(parents=True, exist_ok=True)
                os.replace(src, trash / sub / fname)
        del self._owner[name]
        return trash

    def find_black(self, names: list | None = None) -> list:
        """
        완전히 검은 이미지 (가장 밝은 픽셀 ≤ BLACK_MAX) — [(이름, p99, max)]. 카메라가 노출 전·셔터
        닫힘으로 남긴 빈 프레임. 이미지가 많아 스레드로 나눠 읽는다 (PIL 디코딩은 GIL 을 놓음).
        """
        from concurrent.futures import ThreadPoolExecutor

        names = list(names if names is not None else self.images())
        with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4))) as ex:
            stats = list(ex.map(lambda n: brightness(self.image_path(n)), names))
        return [(n, p99, mx) for n, (p99, mx) in zip(names, stats) if mx <= BLACK_MAX]

    def remove_black(self, names: list | None = None) -> list:
        """
        find_black 결과를 <raw>/_removed_black/<세션>/ 으로 옮기고 _removed_black/manifest.csv 에
        한 줄씩 남긴다 (예전 정리와 같은 형식: session,image,p99,label). 옮긴 [(세션, 이름)].
        """
        moved = []
        for name, p99, _ in self.find_black(names):
            d = self.dir_of(name)
            label = d / "labels" / (Path(name).stem + ".txt")
            label_rel = f"{d.name}\\labels\\{label.name}" if label.exists() else ""
            trash = self.delete_image(name, "_removed_black")
            manifest = trash.parent / "manifest.csv"
            new = not manifest.exists()
            with manifest.open("a", encoding="utf-8", newline="") as f:
                if new:
                    f.write("session,image,p99,label\n")
                f.write(f"{d.name},{name},{p99},{label_rel}\n")
            moved.append((d.name, name))
        return moved

    def seg_path(self, name: str) -> Path:
        """seg 폴리곤 라벨 — 손으로 잡은 OBB(labels/) 와 따로 둔다."""
        return self.dir_of(name) / "labels_seg" / (Path(name).stem + ".txt")

    def write_seg(self, name: str, polygon: list) -> None:
        self.write_seg_polygons(name, [polygon])

    def write_seg_polygons(self, name: str, polygons: list) -> None:
        """폴리곤(이미지 px) → labels_seg/<이름>.txt (YOLO seg: 클래스 + 정규화 x y …)."""
        if not polygons:
            raise ValueError(f"{name}: 저장할 폴리곤이 없습니다")
        for polygon in polygons:
            validate_polygon(polygon, 3)
        labels = self.read_labels(name)
        if labels and len(polygons) != len(labels):
            raise ValueError(f"{name}: OBB {len(labels)}개와 폴리곤 {len(polygons)}개가 다릅니다")
        with Image.open(self.image_path(name)) as im:   # 헤더만 읽음
            w, h = im.size
        lines = []
        for polygon in polygons:
            flat = " ".join(f"{max(0.0, min(1.0, x / w)):.6f} {max(0.0, min(1.0, y / h)):.6f}"
                            for x, y in polygon)
            lines.append(f"{self.class_id} {flat}\n")
        p = self.seg_path(name)
        p.parent.mkdir(exist_ok=True)
        p.write_text("".join(lines), encoding="utf-8")


def validate_polygon(points, minimum: int, exact: bool = False) -> None:
    if not isinstance(points, list) or len(points) < minimum or (exact and len(points) != minimum):
        raise ValueError(f"좌표 점 개수가 잘못되었습니다 (기준 {minimum})")
    for pair in points:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pair)):
            raise ValueError("좌표는 유한한 x, y 숫자 쌍이어야 합니다")


def body_instances(body: dict) -> list:
    if "instances" in body:
        instances = body["instances"]
        if not isinstance(instances, list):
            raise ValueError("instances 는 목록이어야 합니다")
        return instances
    return [body["points"]] if body.get("points") else []


def wire_center_x(points: list) -> float:
    """와이어 순서 = 박스 중심 x (왼쪽이 1번) — 라벨러·앱 표시 순서와 같은 기준."""
    return sum(x for x, _ in points) / len(points)


def safe_name(raw: str) -> str:
    name = Path(raw).name  # 디렉터리 탈출 방지 — 파일명만 남긴다
    if not name or name != raw:
        raise ValueError(f"잘못된 파일명: {raw!r}")
    return name


def load_reference(source: Source) -> dict:
    overlay = source_reference_settings(source)
    if overlay["wire_mask"] == "red":
        templates = red_wire_templates(overlay["wire"])
        if overlay["wire"].exists() and len(templates) != source.wire_count:
            raise ValueError(f"빨간 기준 와이어 {len(templates)}개를 찾았습니다. 설정은 {source.wire_count}개입니다.")
    else:
        template = wire_template(overlay["tube"], overlay["wire"])
        templates = [template] if template else []
    return {"overlay": overlay, "templates": templates}


class Presets:
    """처음 선택할 때 환경을 읽고 보관한다. 선택 상태는 브라우저의 요청마다 지정한다."""

    def __init__(self, source: Source | None = None, ref: dict | None = None):
        self._loaded = {source.name: (source, ref or {})} if source else {}
        self._lock = Lock()

    def get(self, name: str) -> tuple[Source, dict]:
        if name not in DATASETS:
            raise ValueError(f"알 수 없는 프리셋: {name}")
        with self._lock:
            if name not in self._loaded:
                try:
                    source = Source(name)
                    ref = load_reference(source)
                except (SystemExit, OSError, yaml.YAMLError) as exc:
                    raise ValueError(f"{name} 프리셋을 읽을 수 없습니다: {exc}") from exc
                self._loaded[name] = (source, ref)
            return self._loaded[name]


def make_handler(source: Source | None = None, ref: dict | None = None, *, default_dataset: str = "kimm"):
    presets = Presets(source, ref)
    default_dataset = source.name if source else default_dataset

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:
            pass  # 조용히 — 필요하면 print(fmt % args)

        def _json(self, obj, status: int = 200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def _file(self, fp: Path, content_type: str) -> None:
            if not fp.exists():
                self.send_error(404)
                return
            data = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            path, query = url.path, parse_qs(url.query)
            try:
                if path == "/":
                    # 요청마다 읽는다 — index.html 을 고친 뒤 서버 재시작 없이 새로고침만으로 반영
                    self._file(INDEX_HTML, "text/html; charset=utf-8")
                    return
                if path == "/api/presets":
                    self._json({"default": default_dataset, "presets": [
                        {"id": name, "label": "KIMM" if name == "kimm" else "BestFA"} for name in DATASETS]})
                    return
                source, ref = presets.get(query.get("dataset", [default_dataset])[0])
                overlay = ref.get("overlay")
                templates = ref.get("templates") or ([ref["template"]] if ref.get("template") else [])
                if path == "/api/config":
                    self._json({
                        "source": source.name,
                        "config": source.config_path.name,
                        "class_name": source.names.get(source.class_id, "?"),
                        "crop": source.crop_state(), "total": len(source.images()),
                        "dirs": len(source.dirs), "templates": templates,
                        "template": templates[0] if templates else None,
                        "wire_count": source.wire_count,
                        "overlay": overlay and {
                            "enabled": overlay["enabled"], "opacity": overlay["opacity"],
                            "outline_opacity": overlay["outline_opacity"],
                            "white_background": overlay.get("white_background", False),
                            "cutout": overlay["cutout"].exists(),
                            "outline": overlay["outline"].exists()},
                    })
                elif path in ("/ref/cutout.png", "/ref/outline.png") and overlay:
                    self._file(overlay[path[len("/ref/"):-len(".png")]], "image/png")
                elif path == "/api/images":
                    self._json({"images": [{"name": n, "dir": source.dir_of(n).name,
                                            "labeled": source.label_path(n).exists(),
                                            "seg": source.seg_path(n).exists()}
                                           for n in source.images()]})
                elif path == "/api/crop":
                    name = safe_name(query.get("name", [""])[0])
                    self._json({"crop": source.crop_state(name)})
                elif path == "/api/seg":
                    name = safe_name(query.get("name", [""])[0])
                    self._json({"polygons": source.read_seg(name)})
                elif path == "/api/label":
                    name = safe_name(query.get("name", [""])[0])
                    instances = source.read_labels(name)
                    self._json({"instances": instances, "points": instances[0] if instances else None})
                elif path == "/api/edge":
                    # 와이어 폴리곤 — method=1: 저장된 OBB 의 윗변·옆변 + 영상에서 찾은 끝
                    # (edge.find_tip, "compute edge 1"), method=2: 끝 둘레를 직접 따는 BestFA 용
                    # (edge2.find_tip2, "compute edge 2"), method=obb: OBB 4점을 그대로
                    name = safe_name(query.get("name", [""])[0])
                    method = query.get("method", ["1"])[0]
                    # 와이어 1 = 왼쪽 — "와이어 1: ok · 와이어 2: low" 표시가 화면 번호와 맞게
                    instances = sorted(source.read_labels(name), key=wire_center_x)
                    fp = source.image_path(name)
                    if not instances:
                        self._json({"status": "no_label", "method": method})
                    elif method in ("obb", "1", "2"):
                        with Image.open(fp) as im:
                            w, h = im.size
                        results = []
                        for points in instances:
                            if method == "obb":
                                result = {"status": "ok", "edges": [],
                                          "polygon": [[x * w, y * h] for x, y in points]}
                            elif method == "2":
                                result = find_tip2(fp, points)
                            else:
                                result = find_tip(fp, points)
                            results.append(result)
                        statuses = {r["status"] for r in results}
                        status = ("ok" if statuses == {"ok"} else
                                  "no_tip" if "no_tip" in statuses else "low")
                        polygons = [r["polygon"] for r in results] if all(r.get("polygon") for r in results) else []
                        payload = dict(results[0]) if len(results) == 1 else {}
                        self._json({**payload, "status": status, "method": method,
                                    "instances": results, "polygons": polygons})
                    else:
                        raise ValueError(f"모르는 method: {method}")
                elif path.startswith("/image/"):
                    # 브라우저가 encodeURIComponent 로 보내므로 되돌린다 (공백 등)
                    name = safe_name(unquote(path[len("/image/"):]))
                    fp = source.image_path(name)
                    self._file(fp, IMG_EXTS[fp.suffix.lower()])
                else:
                    self.send_error(404)
            except ValueError as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self) -> None:
            url = urlparse(self.path)
            try:
                query = parse_qs(url.query)
                source, _ = presets.get(query.get("dataset", [default_dataset])[0])
                if url.path == "/api/label":
                    body = self._body()
                    name = safe_name(body["name"])
                    source.write_labels(name, body_instances(body))
                    self._json({"ok": True, "labeled": sum(
                        1 for n in source.images() if source.label_path(n).exists())})
                elif url.path == "/api/copy":
                    body = self._body()
                    instances = body_instances(body)
                    if not instances:
                        raise ValueError("복사할 와이어 라벨이 없습니다.")
                    for points in instances:
                        validate_polygon(points, 4, exact=True)
                    overwrite = bool(body.get("overwrite"))
                    # wire: 와이어 하나만 복사 (0부터, 왼쪽부터 센 번호) — 대상 이미지의 그 와이어만
                    # 바꾸고 나머지 와이어는 그대로. 와이어가 다 갖춰지지 않은 대상(라벨 없음 등)은
                    # 한 와이어만 쓰면 반쪽 라벨이 되므로 지금 이미지의 와이어 전부를 쓴다.
                    wire = body.get("wire")
                    if wire is not None:
                        wire = int(wire)
                        if not (0 <= wire < len(instances) == source.wire_count):
                            raise ValueError(f"와이어 번호 {wire + 1} 이(가) 맞지 않습니다 "
                                             f"(이 환경 와이어 {source.wire_count}개).")
                        instances = sorted(instances, key=wire_center_x)
                    # 복사 범위는 지금 보고 있는 이미지의 세션 폴더 안으로 한정 —
                    # "타깃 위치가 거의 같다"는 전제는 한 녹화 세션 안에서만 성립합니다
                    # (세션마다 장면·지그가 다름). 소스가 폴더 하나면 기존과 같습니다.
                    current = safe_name(body.get("name", ""))
                    here = source.dir_of(current)
                    names = source.images()
                    # 범위 (목록 순서 = 세션 → 파일명). from_here: 지금 이미지부터 세션 끝까지,
                    # to_name: 지금 이미지와 그 이미지 사이(양 끝 포함, 순서 무관)
                    lo, hi = 0, len(names)
                    if body.get("to_name"):
                        i, j = names.index(current), names.index(safe_name(body["to_name"]))
                        lo, hi = min(i, j), max(i, j) + 1
                    elif body.get("from_here"):
                        lo = names.index(current)
                    written, skipped, filled = [], 0, 0
                    for n in names[lo:hi]:
                        if source.dir_of(n) != here:
                            continue
                        if not overwrite and source.label_path(n).exists():
                            skipped += 1
                            continue
                        out = instances
                        if wire is not None:
                            existing = source.read_labels(n)
                            if len(existing) == source.wire_count:
                                out = sorted(existing, key=wire_center_x)
                                out[wire] = instances[wire]
                            else:
                                filled += 1
                        source.write_labels(n, out)
                        written.append(n)
                    # names: 실제로 OBB 가 바뀐 장 — 라벨러가 이 장들만 경계를 다시 계산
                    # filled: 와이어 하나만 복사하려 했지만 라벨이 다 없어서 전부 쓴 장 수
                    self._json({"ok": True, "written": len(written), "skipped": skipped,
                                "filled": filled, "names": written})
                elif url.path == "/api/delete":
                    trash = source.delete_image(safe_name(self._body().get("name", "")))
                    self._json({"ok": True, "moved_to": str(trash)})
                elif url.path == "/api/black":
                    # {"apply": false} → 검은 이미지 목록만 (세션별 장수) / {"apply": true, "names": […]}
                    # → 그 장들을 다시 확인한 뒤 _removed_black/ 으로 (목록과 실제가 어긋나지 않게)
                    body = self._body()
                    if body.get("apply"):
                        names = [safe_name(n) for n in body.get("names") or []]
                        moved = source.remove_black([n for n in names if n in source._owner])
                        self._json({"ok": True, "moved": len(moved), "names": [n for _, n in moved]})
                    else:
                        found = source.find_black()
                        per: dict = {}
                        for n, _, _ in found:
                            per[source.dir_of(n).name] = per.get(source.dir_of(n).name, 0) + 1
                        self._json({"ok": True, "threshold": BLACK_MAX, "names": [n for n, _, _ in found],
                                    "sessions": per, "total": len(source.images())})
                elif url.path == "/api/seg":
                    # {"items": [{"name", "polygon": [[x, y], …] (이미지 px)}, …]} → labels_seg/
                    items = self._body().get("items") or []
                    for it in items:
                        polygons = it.get("polygons") if "polygons" in it else [it["polygon"]]
                        source.write_seg_polygons(safe_name(it["name"]), polygons)
                    self._json({"ok": True, "written": len(items)})
                elif url.path == "/api/crop":
                    self._json({"ok": True, "crop": source.save_crop(self._body())})
                else:
                    self.send_error(404)
            except (ValueError, KeyError, TypeError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def main() -> None:
    # 한국어 Windows 콘솔(cp949)은 '—' 같은 글자를 못 찍어 print 가 예외로 죽는다 — 못 찍는
    # 글자만 ? 로 바꾼다 (common.py 와 같은 처리; 라벨러는 common 을 import 하지 않음)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser(description="와이어 OBB·seg 라벨러 (raw/<세션>/Mono 대상)")
    ap.add_argument("--source", "--dataset", default="kimm", choices=DATASETS,
                    help="처음 표시할 프리셋 (기본 kimm). 실행 후 GUI에서 바꿀 수 있습니다.")
    ap.add_argument("--port", type=int, default=8765)
    # 인증이 없고 라벨 파일을 쓰는 서버라 기본은 로컬만 — 원격 머신이면 ssh -L 로 터널.
    ap.add_argument("--host", default="127.0.0.1", help="바인드 주소 (기본: 127.0.0.1)")
    args = ap.parse_args()

    handler = make_handler(default_dataset=args.source)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        raise SystemExit(f"포트 {args.port}에 서버를 열 수 없습니다. 기존 라벨러를 종료하거나 "
                         f"--port 로 다른 번호를 지정하세요. ({exc})") from None
    print(f"[labeller] http://{args.host}:{args.port}  (브라우저에서 접속, Ctrl+C 로 종료)")
    print("[labeller] 화면 상단의 프리셋에서 KIMM / BestFA를 선택하세요.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
