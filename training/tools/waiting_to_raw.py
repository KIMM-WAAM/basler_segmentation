"""datasets/<환경>/waiting/ 에 넣어 둔 세션 폴더를 raw/ 규약으로 옮긴다 (data.yaml · session.yaml 자동).

    uv run tools/waiting_to_raw.py                       # 사용법 + 환경별 waiting 목록 (아무것도 안 바꿈)
    uv run tools/waiting_to_raw.py --dataset bestfa      # 세션마다 계획을 보여 주고 물어본 뒤 옮김
    uv run tools/waiting_to_raw.py --dataset bestfa --yes   # 묻지 않고 전부

waiting/<YYMMDD_HHMMSS[_메모]>/ 하나가 세션 하나:
· 폴더 이름 → raw/<YYMMDD_HHMMSS>/. 메모는 `_` 로 나눠 미리 정의된 항목(dataset_paths.SESSION_NAME_ITEMS,
  예: layer3 → layer: 3)은 그 필드로, 나머지는 scene 으로 session.yaml 에 적는다.
· 이미지 폴더 Mono/ · Image/ · images/ (하나만) → Mono/. 파일명에 세션 id 가 없으면 `<id>__` 를 붙인다
  — 세션마다 시각만으로 된 이름(00_02_23_544.jpg)이 겹치면 라벨러 · 전처리가 멈추기 때문.
· labels/ · labels_seg/ 가 있으면 같은 이름 규칙으로 함께 옮긴다. 그 밖의 파일은 그대로 옮긴다.
· data.yaml(names · train: Mono)이 없으면 만들고, session.yaml 틀을 만든다 (wire_count 는 환경 설정에서).
  session.yaml 끝에는 옮긴 기록(imported: 시각 · 원래 이름 · 이미지 폴더 · 장수 · 이름 바꾼 장수)을 남긴다.
· 옮기기(같은 드라이브 안 이동)라 waiting/ 에서는 사라진다. raw/ 에 같은 id 가 있으면 그 세션은 건너뛴다.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # training/ — 스크립트로 실행할 때
from common import IMG_EXTS, load_yaml, normalize_names  # noqa: E402
from dataset_paths import (  # noqa: E402
    DATASETS,
    SESSION_META,
    SESSION_NAME_ITEMS,
    dataset_root,
    ensure_session_meta,
    parse_session_name,
    preprocess_config,
    split_memo,
)

IMAGE_DIRS = ("Mono", "Image", "images")
LABEL_DIRS = ("labels", "labels_seg")


def usage() -> None:
    print(__doc__)
    print("미리 정의된 이름 항목 (메모의 `_` 조각):")
    for _, key, _, desc in SESSION_NAME_ITEMS:
        print(f"  {key:<8} {desc}")
    print("  (그 밖의 조각)  → scene. 예: 261026_145202_layer3_조명실험 → layer: 3, scene: 조명실험\n")
    for env in DATASETS:
        waiting = dataset_root(env) / "waiting"
        items = sorted(p.name for p in waiting.iterdir() if p.is_dir()) if waiting.is_dir() else []
        print(f"[{env}] {waiting}  — {len(items)}개" + "".join(f"\n    {n}" for n in items))


class Plan:
    """waiting 세션 하나를 raw 로 옮기는 계획 — 실제로 옮기기 전에 문제를 전부 찾는다."""

    def __init__(self, src: Path, raw: Path, taken: set) -> None:
        self.src = src
        self.problems: list = []
        try:
            self.id, memo = parse_session_name(src.name)
        except ValueError as e:
            self.id, memo = None, None
            self.problems.append(str(e))
        self.meta = split_memo(memo)
        self.dst = raw / self.id if self.id else None
        if self.dst is not None and self.dst.exists():
            self.problems.append(f"raw 에 이미 있음: {self.dst.name}")

        found = [d for d in IMAGE_DIRS if (src / d).is_dir()]
        if len(found) != 1:
            self.problems.append(f"이미지 폴더는 {'/'.join(IMAGE_DIRS)} 중 하나여야 합니다 (지금: {found or '없음'})")
        self.image_dir = found[0] if len(found) == 1 else None
        self.images = sorted(p for p in (src / self.image_dir).iterdir()
                             if p.suffix.lower() in IMG_EXTS) if self.image_dir else []
        if self.image_dir and not self.images:
            self.problems.append("이미지가 0장")

        # 새 이름 — 세션 id 가 이미 들어 있으면(앱 캡처 000001__cap_<id>.png) 그대로
        self.rename = {p.name: (p.name if self.id and self.id in p.name else f"{self.id}__{p.name}")
                       for p in self.images}
        clash = [n for n in self.rename.values() if n in taken]
        if clash:
            self.problems.append(f"다른 세션과 파일명이 겹침 {len(clash)}개 (예: {clash[0]})")
        taken.update(self.rename.values())
        self.labels = {d: sorted((src / d).glob("*.txt")) for d in LABEL_DIRS if (src / d).is_dir()}
        self.others = sorted(p for p in src.iterdir()
                             if p.name not in (self.image_dir, *LABEL_DIRS))

    def record(self) -> dict:
        """session.yaml 끝에 남길 옮긴 기록 — 언제 · 어디서 · 무엇을 · 이름을 어떻게 바꿨는지."""
        rec = {"at": datetime.now().astimezone().isoformat(timespec="seconds"),
               "from": f"waiting/{self.src.name}",
               "image_dir": self.image_dir,
               "images": len(self.images),
               "renamed": sum(1 for a, b in self.rename.items() if a != b)}
        if self.labels:
            rec["labels"] = {d: len(files) for d, files in self.labels.items()}
        if self.others:
            rec["other_files"] = [p.name for p in self.others]
        return rec

    def describe(self) -> str:
        prefixed = sum(1 for a, b in self.rename.items() if a != b)
        lines = [f"{self.src.name}  →  raw/{self.id or '?'}"]
        if self.meta:
            lines.append("    session.yaml: " + ", ".join(f"{k}: {v}" for k, v in self.meta.items()))
        if self.image_dir:
            lines.append(f"    {self.image_dir}/ {len(self.images)}장 → Mono/"
                         + (f"  (파일명 앞에 {self.id}__ — {prefixed}장)" if prefixed else ""))
        for d, files in self.labels.items():
            lines.append(f"    {d}/ {len(files)}개 (같은 이름 규칙)")
        if self.others:
            lines.append("    그 밖에 그대로: " + ", ".join(p.name for p in self.others))
        lines += [f"    ✗ {p}" for p in self.problems]
        return "\n".join(lines)

    def apply(self, env: str, names: dict, wires) -> None:
        (self.dst / "Mono").mkdir(parents=True)
        for p in self.images:
            os.replace(p, self.dst / "Mono" / self.rename[p.name])
        stems = {Path(a).stem: Path(b).stem for a, b in self.rename.items()}
        for d, files in self.labels.items():
            (self.dst / d).mkdir(exist_ok=True)
            for f in files:
                os.replace(f, self.dst / d / f"{stems.get(f.stem, f.stem)}.txt")
        for d in LABEL_DIRS:
            (self.dst / d).mkdir(exist_ok=True)
        for p in self.others:
            os.replace(p, self.dst / p.name)
        if not (self.dst / "data.yaml").exists():
            (self.dst / "data.yaml").write_text(
                yaml.safe_dump({"names": names, "train": "Mono"}, allow_unicode=True, sort_keys=False),
                encoding="utf-8")
        ensure_session_meta(self.dst, env, wire_count=wires, **self.meta)
        append_import_record(self.dst, self.record())
        # 비었으면 waiting 쪽 폴더 정리 (남은 게 있으면 그대로 두고 알림)
        for d in (self.image_dir, *self.labels):
            try:
                (self.src / d).rmdir()
            except OSError:
                pass
        try:
            self.src.rmdir()
        except OSError:
            print(f"    waiting 에 남은 파일이 있어 폴더를 두었습니다: {self.src}")


def append_import_record(session: Path, rec: dict) -> None:
    """session.yaml 끝에 imported: 블록 (자동 기록 — 사람이 적는 위쪽 명세와 구분). 이미 있으면 그대로."""
    p = session / SESSION_META
    text = p.read_text(encoding="utf-8")
    if "\nimported:" in text:
        return
    block = yaml.safe_dump({"imported": rec}, allow_unicode=True, sort_keys=False)
    p.write_text(text.rstrip("\n") + "\n\n# ── waiting_to_raw.py 가 옮길 때 남긴 기록 (자동) ──\n" + block,
                 encoding="utf-8")


def ask(plan: Plan) -> str:
    """y / n / e(메모 고치기) / a(이후 전부) / q."""
    while True:
        a = input("  진행? [y]es / [n]o / [e]dit 메모 / [a]ll / [q]uit: ").strip().lower()[:1]
        if a in ("y", "n", "a", "q"):
            return a
        if a == "e":
            for key in ("scene", "layer", "notes"):
                cur = plan.meta.get(key)
                new = input(f"    {key} [{'' if cur is None else cur}] (Enter=그대로, -=지움): ").strip()
                if new == "-":
                    plan.meta.pop(key, None)
                elif new:
                    plan.meta[key] = int(new) if key == "layer" and new.isdigit() else new
            print(plan.describe())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=DATASETS, help="환경 (없으면 사용법과 waiting 목록만)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--ask", action="store_true", help="세션마다 물어봄 (기본 — VS Code task 의 모드 선택용)")
    mode.add_argument("--yes", action="store_true", help="묻지 않고 문제 없는 세션 전부")
    mode.add_argument("--dry-run", action="store_true", help="계획만 보여 줌")
    args = ap.parse_args()
    if not args.dataset:
        usage()
        return

    root = dataset_root(args.dataset)
    waiting, raw = root / "waiting", root / "raw"
    sessions = sorted(p for p in waiting.iterdir() if p.is_dir()) if waiting.is_dir() else []
    if not sessions:
        print(f"{waiting} 에 세션 폴더가 없습니다.")
        return
    cfg = load_yaml(preprocess_config(args.dataset))
    names = normalize_names(cfg.get("names") or {0: "wire"})
    wires = cfg.get("wire_count")

    raw.mkdir(parents=True, exist_ok=True)
    taken = {p.name for p in raw.glob("*/Mono/*")}   # 이미 raw 에 있는 파일명
    plans = [Plan(s, raw, taken) for s in sessions]
    ids = [p.id for p in plans if p.id]
    for p in plans:
        if p.id and ids.count(p.id) > 1:
            p.problems.append(f"waiting 안에서 같은 id 가 여럿: {p.id}")

    done = 0
    all_yes = args.yes
    for p in plans:
        print(p.describe())
        if p.problems:
            print("    → 건너뜀 (위 문제를 고친 뒤 다시 실행)\n")
            continue
        if args.dry_run:
            print()
            continue
        if not all_yes:
            a = ask(p)
            if a == "q":
                break
            if a == "n":
                print()
                continue
            all_yes = a == "a"
        p.apply(args.dataset, names, wires)
        done += 1
        print(f"    ✓ raw/{p.id}\n")
    print(f"{done}개 옮김." + ("" if args.dry_run else f"  다음: 라벨러에서 {args.dataset} 프리셋으로 라벨링"))


if __name__ == "__main__":
    main()
