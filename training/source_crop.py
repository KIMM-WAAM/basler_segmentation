"""원본 크롭 증강 설정과 전처리 인덱스. torch 없이 사전 검증 가능."""
from __future__ import annotations

import json
import math
from pathlib import Path

from common import load_yaml
from provenance import sha256_file

INDEX = "crop_sources.json"


def validate_options(options: dict) -> dict:
    out = dict(options)
    if "center_x" in out or "center_y" in out:
        raise ValueError("source_crop: 절대 위치 대신 offset_x/offset_y 이동 범위를 설정하세요")
    for key in ("offset_x", "offset_y"):
        values = out.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"source_crop.{key}: [최소, 최대]가 필요합니다")
        lo, hi = map(float, values)
        if not (math.isfinite(lo) and math.isfinite(hi) and -100 <= lo <= hi <= 100):
            raise ValueError(f"source_crop.{key}: -100~100 이동 범위여야 합니다")
        out[key] = [lo, hi]
    return out


def read_index(root: Path, verify_images: bool = False) -> dict:
    path = root / INDEX
    if not path.is_file():
        raise ValueError(f"원본 크롭 인덱스가 없습니다: {path}. 새 출력 폴더에 전처리를 다시 실행하세요.")
    manifest = load_yaml(root / "manifest.yaml")
    expected = (manifest.get("processing") or {}).get("crop_sources_sha256")
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"원본 크롭 인덱스 지문 불일치: {path}. 전처리를 다시 실행하세요.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("version") != 1:
        raise ValueError(f"지원하지 않는 원본 크롭 인덱스: {path}")
    entries = doc["images"]
    if verify_images:
        from tqdm import tqdm
        unique = {entry["image"]: entry["raw_sha256"] for entry in entries.values()}
        for name, digest in tqdm(unique.items(), desc="verify crop sources", unit="img", ascii=True):
            image = (root / name).resolve()
            if not image.is_file() or sha256_file(image) != digest:
                raise ValueError(f"전처리 이후 원본 이미지가 변경되었거나 없습니다: {image}")
    return entries
