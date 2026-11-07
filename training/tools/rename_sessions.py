"""예전 세션 폴더 이름(kimm_YYMMDD_HHMMSS[_메모])을 새 규약(YYMMDD_HHMMSS)으로 바꾼다.

    uv run tools/rename_sessions.py --dataset kimm            # 바꿀 목록만 보여줌 (아무것도 안 바꿈)
    uv run tools/rename_sessions.py --dataset kimm --apply    # 실제로 바꿈

· raw/<세션>/ 과 raw/_removed_*/<세션>/ 둘 다 바꾼다 (지운 것을 되돌릴 때 이름이 맞도록).
· 이름의 _메모 는 그 세션의 session.yaml 의 scene 으로 옮긴다 (이미 session.yaml 이 있으면 그대로).
· raw/_removed_*/ 의 .csv (manifest.csv 등) 안의 예전 세션 이름도 새 이름으로 고친다.
· 같은 id 로 겹치는 세션이 있으면 아무것도 바꾸지 않고 중단한다.
· configs/preprocess.<환경>.yaml 의 val_sessions/test_sessions 는 직접 고쳐야 한다 — 예전 이름이
  남아 있으면 끝에 알려 준다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # training/ — 스크립트로 실행할 때
from common import load_yaml  # noqa: E402
from dataset_paths import (  # noqa: E402
    DATASETS,
    SESSION_RE,
    dataset_root,
    ensure_session_meta,
    parse_session_name,
    preprocess_config,
)


def plan(raw: Path) -> list:
    """(예전 폴더, 새 폴더, 메모) — 이미 새 이름인 세션은 빼고."""
    parents = [raw] + sorted(p for p in raw.glob("_removed_*") if p.is_dir())
    out = []
    for parent in parents:
        for d in sorted(p for p in parent.iterdir() if p.is_dir() and not p.name.startswith("_")):
            try:
                sid, memo = parse_session_name(d.name)
            except ValueError:
                print(f"  건너뜀 (세션 이름 형식 아님): {d.relative_to(raw)}")
                continue
            if d.name != sid:
                out.append((d, d.with_name(sid), memo))
    return out


def check(moves: list) -> None:
    targets: dict = {}
    for old, new, _ in moves:
        if new.exists():
            raise SystemExit(f"이미 있는 폴더와 겹칩니다: {old.name} → {new}")
        if new in targets:
            raise SystemExit(f"같은 id 로 겹칩니다: {targets[new].name} · {old.name} → {new.name}")
        targets[new] = old


def rewrite_csvs(raw: Path, names: dict) -> list:
    """_removed_*/ 의 csv 안 예전 세션 이름 → 새 이름 (긴 이름부터 바꿔 부분 겹침을 피함)."""
    changed = []
    for csv in sorted(raw.glob("_removed_*/*.csv")):
        text = csv.read_text(encoding="utf-8")
        new = text
        for old in sorted(names, key=len, reverse=True):
            new = new.replace(old, names[old])
        if new != text:
            csv.write_text(new, encoding="utf-8")
            changed.append(csv)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--raw", type=Path, help="raw 폴더 (기본 datasets/<환경>/raw)")
    ap.add_argument("--apply", action="store_true", help="실제로 바꿈 (없으면 목록만)")
    args = ap.parse_args()

    raw = (args.raw or dataset_root(args.dataset) / "raw").resolve()
    moves = plan(raw)
    check(moves)
    for old, new, memo in moves:
        print(f"  {old.relative_to(raw)}  →  {new.relative_to(raw)}" + (f"   (scene: {memo})" if memo else ""))
    if not moves:
        print("바꿀 세션이 없습니다.")
    if not args.apply:
        print(f"\n{len(moves)}개 — 실제로 바꾸려면 --apply")
        return

    wires = load_yaml(preprocess_config(args.dataset)).get("wire_count")
    for old, new, memo in moves:
        old.rename(new)
        ensure_session_meta(new, args.dataset, scene=memo, wire_count=wires)
    # 이미 새 이름이던 세션에도 session.yaml 틀을 둔다
    for d in raw.iterdir():
        m = SESSION_RE.match(d.name)
        if d.is_dir() and m and not m.group(2):
            ensure_session_meta(d, args.dataset, wire_count=wires)
    for csv in rewrite_csvs(raw, {old.name: new.name for old, new, _ in moves}):
        print(f"  csv 고침: {csv.relative_to(raw)}")
    print(f"\n{len(moves)}개 바꿈.")

    cfg = preprocess_config(args.dataset)
    left = [old.name for old, _, _ in moves if old.name in cfg.read_text(encoding="utf-8")]
    if left:
        print(f"경고: {cfg.name} 에 예전 이름이 남아 있습니다 — 따옴표로 감싼 새 이름으로 고치세요:\n  "
              + "\n  ".join(left))


if __name__ == "__main__":
    main()
