"""BestFA의 <세션>/Mono 원본을 복사해 프로젝트 라벨링 구조로 가져온다.

python tools/import_bestfa.py --source "C:/.../261026_115301/Mono"
--source 에 DB 루트를 주면 바로 아래 각 세션의 Mono 를 가져온다.
원본은 수정하지 않는다. 재실행 시 동일 파일만 건너뛰고 충돌은 오류로 알린다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_RAW = Path(__file__).resolve().parents[2] / "datasets" / "bestfa" / "raw"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        return checksum.hexdigest()


def mono_dirs(source: Path) -> list[Path]:
    source = source.resolve(strict=True)
    if source.name.lower() == "mono" and source.is_dir():
        return [source]
    if (source / "Mono").is_dir():
        return [source / "Mono"]
    found = sorted(p / "Mono" for p in source.iterdir() if (p / "Mono").is_dir())
    if not found:
        raise ValueError(f"Mono 폴더가 없습니다: {source}")
    return found


def import_session(mono: Path, raw: Path = DEFAULT_RAW) -> dict:
    mono, raw = mono.resolve(strict=True), raw.resolve()
    session = mono.parent.name
    if not re.fullmatch(r"\d{6}_\d{6}(?:_[\w-]+)?", session):
        raise ValueError(f"세션 이름은 YYMMDD_HHMMSS[_메모] 형식이어야 합니다: {session}")
    destination = raw / f"bestfa_{session}"
    if destination == mono or mono in destination.parents or destination in mono.parents:
        raise ValueError("원본과 대상 경로가 겹칩니다")
    images = sorted(p for p in mono.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise ValueError(f"이미지가 없습니다: {mono}")
    # 파일명에 세션을 넣어 서로 다른 Mono 의 동일 시각 이름이 충돌하지 않게 한다.
    if (destination / "images").is_dir():
        raise ValueError(f"기존 images 폴더를 Mono 로 바꾼 후 다시 실행하세요: {destination}")
    plan = [(p, destination / "Mono" / f"bestfa_{session}__{p.name}") for p in images]
    metadata = destination / "import.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if Path(previous["source"]).resolve() != mono:
            raise FileExistsError(f"다른 원본에서 가져온 세션입니다: {destination}")
    # 충돌을 먼저 전부 검사한다. 기존 이미지·라벨은 덮어쓰지 않는다.
    for src, dst in plan:
        if dst.exists() and (src.stat().st_size != dst.stat().st_size or digest(src) != digest(dst)):
            raise FileExistsError(f"내용이 다른 파일이 이미 있습니다: {dst}")
    for folder in ("Mono", "labels", "labels_seg"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    copied = 0
    for src, dst in plan:
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    data_yaml = destination / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text("names: {0: wire}\ntrain: Mono\n", encoding="utf-8")
    manifest = {"source": str(mono), "session": destination.name,
                "files": [{"source": src.name, "image": dst.name} for src, dst in plan]}
    metadata.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"session": destination.name, "images": len(images), "copied": copied,
            "skipped": len(images) - copied, "destination": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="Mono / 세션 / DB 루트")
    parser.add_argument("--out", type=Path, default=DEFAULT_RAW, help="대상 raw 폴더")
    args = parser.parse_args()
    for mono in mono_dirs(args.source):
        result = import_session(mono, args.out)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
