"""
KIMM · BestFA split/all 학습을 순서대로 끝까지 돌리는 무인 대기열.

  1. KIMM 분리 실험 A·B·C — 이미 돌고 있으면 끝나길 기다리고, 멈췄으면 남은 것부터 이어서 돌린다
  2. A·B·C 중 best.pt 의 mask mAP50-95 가 가장 높은 조건을 고른다
  3. BestFA split 학습
  4. KIMM all · BestFA all 학습 + ONNX export

all 학습의 epochs 는 해당 split 의 best epoch 를 [15, 60] 으로 제한한 값이다.
결과 요약은 runs/train_queue/report.md, 진행 로그는 runs/train_queue/queue.log.

    .venv/Scripts/python.exe tools/train_queue.py          # 실행
    .venv/Scripts/python.exe tools/train_queue.py --plan   # 현재 상태와 할 일만 출력
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CONFIGS = HERE / "configs"
QUEUE_DIR = HERE / "runs" / "train_queue"
ABLATION_LOGS = HERE / "runs" / "kimm" / "segment" / "ablation_logs"
KIMM_ROOT = HERE / "runs" / "kimm" / "segment"
BESTFA_ROOT = HERE / "runs" / "bestfa" / "segment"

ABLATION = {
    "a": "yolo26s-seg-ablation-a-baseline",
    "b": "yolo26s-seg-ablation-b-source-crop",
    "c": "yolo26s-seg-ablation-c-gamma-brightness",
}
SPLIT_EPOCHS = 60
MIN_ALL_EPOCHS = 15
STALE_SECONDS = 20 * 60  # 로그·results.csv 가 이만큼 멈추면 기존 실험이 죽은 것으로 본다
POLL_SECONDS = 30


def log(message: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with (QUEUE_DIR / "queue.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_rows(run: Path) -> list[dict]:
    path = run / "results.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def complete_run(root: Path, name: str, epochs: int) -> Path | None:
    """epochs 만큼 끝난 실행 중 가장 최근 폴더."""
    folder = root / name
    if not folder.is_dir():
        return None
    runs = sorted(d for d in folder.iterdir() if d.is_dir())
    return next((r for r in reversed(runs) if len(read_rows(r)) >= epochs), None)


def best_row(run: Path) -> dict:
    """best.pt 선택 기준(box + mask mAP50-95)으로 최고인 epoch 의 행."""
    return max(read_rows(run),
               key=lambda r: float(r["metrics/mAP50-95(B)"]) + float(r["metrics/mAP50-95(M)"]))


def mask_map(row: dict) -> float:
    return float(row["metrics/mAP50-95(M)"])


def all_epochs(row: dict) -> int:
    return max(MIN_ALL_EPOCHS, min(SPLIT_EPOCHS, int(float(row["epoch"]))))


def train(config: str, log_name: str) -> int:
    env = {**os.environ, "PYTHONUTF8": "1", "YOLO_OFFLINE": "true", "NO_ALBUMENTATIONS_UPDATE": "1"}
    log(f"시작: {config}")
    with (QUEUE_DIR / log_name).open("ab") as out:
        code = subprocess.run([sys.executable, "train.py", "--config", config],
                              cwd=HERE, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
    log(f"종료: {config} (exit={code})")
    return code


def newest_activity() -> float:
    paths = list(ABLATION_LOGS.glob("*"))
    for name in ABLATION.values():
        paths += list((KIMM_ROOT / name).glob("*/results.csv"))
    return max((p.stat().st_mtime for p in paths if p.is_file()), default=0.0)


def wait_for_running_ablation() -> None:
    summary = ABLATION_LOGS / "summary.txt"
    while True:
        if summary.is_file() and "c exit=" in summary.read_text(encoding="utf-8", errors="replace"):
            log("기존 KIMM 실험 대기열 종료 확인")
            return
        idle = time.time() - newest_activity()
        if idle > STALE_SECONDS:
            log(f"기존 KIMM 실험이 {idle / 60:.0f}분간 멈춤 — 남은 학습을 이어서 실행")
            return
        time.sleep(POLL_SECONDS)


def run_ablation() -> dict[str, dict]:
    wait_for_running_ablation()
    results = {}
    for key, name in ABLATION.items():
        run = complete_run(KIMM_ROOT, name, SPLIT_EPOCHS)
        if run is None:
            train(f"train.kimm.ablation_{key}.yaml", f"ablation_{key}.log")
            run = complete_run(KIMM_ROOT, name, SPLIT_EPOCHS)
        if run is None:
            log(f"KIMM 실험 {key}: 완료된 결과 없음")
            continue
        results[key] = {"run": run, "best": best_row(run)}
    return results


def write_all_config(filename: str, base: str, data: str, epochs: int, name: str) -> None:
    (CONFIGS / filename).write_text(
        "# tools/train_queue.py 가 생성. split 결과의 증강·optimizer 로 전체 데이터를 학습한다.\n"
        f"base: {base}\n\n"
        "stages:\n  preview_aug: false\n  train: true\n  export: true\n\n"
        f"train:\n  data: {data}\n  epochs: {epochs}\n  patience: 0\n  name: {name}\n",
        encoding="utf-8")


def report_line(label: str, root: Path, name: str, epochs: int) -> str:
    run = complete_run(root, name, epochs) if epochs else None
    if run is None:
        return f"| {label} | 실패 또는 미실행 | - | - | - |"
    row = best_row(run)
    return (f"| {label} | {int(float(row['epoch']))} | {float(row['metrics/mAP50(M)']):.3f} "
            f"| {mask_map(row):.3f} | `{run.relative_to(HERE).as_posix()}` |")


def write_report(lines: list[str]) -> None:
    text = "\n".join(["# 학습 대기열 결과", "",
                      "| 학습 | best epoch | mask mAP50 | mask mAP50-95 | 폴더 |",
                      "|---|---|---|---|---|", *lines, "",
                      "all 학습의 val 은 train 에 포함되어 있어 점수를 split 과 비교하면 안 된다.", ""])
    (QUEUE_DIR / "report.md").write_text(text, encoding="utf-8")


def plan() -> None:
    for key, name in ABLATION.items():
        run = complete_run(KIMM_ROOT, name, SPLIT_EPOCHS)
        print(f"KIMM {key}: {'완료 ' + run.name if run else '미완료'}")
    idle = time.time() - newest_activity()
    print(f"기존 KIMM 실험 마지막 활동: {idle:.0f}초 전 (멈춤 판정 {STALE_SECONDS}초)")
    run = complete_run(BESTFA_ROOT, "bestfa-yolo26s-seg-adamw-split", SPLIT_EPOCHS)
    print(f"BestFA split: {'완료 ' + run.name if run else '미완료'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", action="store_true", help="학습 없이 현재 상태만 출력")
    if parser.parse_args().plan:
        plan()
        return

    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: 이 프로세스가 도는 동안 절전 진입을 막는다
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    log(f"대기열 시작 (pid {os.getpid()})")

    results = run_ablation()
    lines = [report_line(f"KIMM split {k.upper()}", KIMM_ROOT, n, SPLIT_EPOCHS) for k, n in ABLATION.items()]

    bestfa_split = "bestfa-yolo26s-seg-adamw-split"
    if complete_run(BESTFA_ROOT, bestfa_split, SPLIT_EPOCHS) is None:
        train("train.bestfa.adamw_split.yaml", "bestfa_split.log")
    lines.append(report_line("BestFA split", BESTFA_ROOT, bestfa_split, SPLIT_EPOCHS))
    write_report(lines)

    kimm_epochs = 0
    if results:
        winner = max(results, key=lambda k: mask_map(results[k]["best"]))
        kimm_epochs = all_epochs(results[winner]["best"])
        log(f"KIMM 선택: {winner.upper()} (mask mAP50-95 {mask_map(results[winner]['best']):.3f}), "
            f"all epochs={kimm_epochs}")
        write_all_config("train.kimm.adamw_all.yaml", f"train.kimm.ablation_{winner}.yaml",
                         "../datasets/kimm/processed_seg/data.all.yaml", kimm_epochs, "yolo26s-seg-adamw-all")
        train("train.kimm.adamw_all.yaml", "kimm_all.log")
    else:
        log("KIMM all 건너뜀: 완료된 split 실험 없음")
    lines.append(report_line("KIMM all", KIMM_ROOT, "yolo26s-seg-adamw-all", kimm_epochs))
    write_report(lines)

    bestfa_epochs = 0
    run = complete_run(BESTFA_ROOT, bestfa_split, SPLIT_EPOCHS)
    if run is not None:
        bestfa_epochs = all_epochs(best_row(run))
        log(f"BestFA all epochs={bestfa_epochs}")
        write_all_config("train.bestfa.adamw_all.yaml", "train.bestfa.adamw_split.yaml",
                         "../datasets/bestfa/processed_seg/data.all.yaml", bestfa_epochs,
                         "bestfa-yolo26s-seg-adamw-all")
        train("train.bestfa.adamw_all.yaml", "bestfa_all.log")
    else:
        log("BestFA all 건너뜀: 완료된 split 학습 없음")
    lines.append(report_line("BestFA all", BESTFA_ROOT, "bestfa-yolo26s-seg-adamw-all", bestfa_epochs))
    write_report(lines)
    log("대기열 완료")


if __name__ == "__main__":
    main()
