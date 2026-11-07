"""
모델 출처 기록 — 어떤 코드 · 설정 · 데이터 · 환경으로 만든 모델인지.

  전처리   (preprocess.py) →  <환경>/processed_<task>/manifest.yaml  세션별 split · 장수 · 지문
                              + manifest.files.tsv                 파일 하나하나의 지문
  학습 시작 (train.py)     →  <실행 폴더>/run_info.yaml  커밋 · config · 데이터(manifest) · 환경
                              + config.yaml   재현용 config (상속을 푼 것 + 시작 가중치 · batch 고정)
                              + data/<소스>.manifest.yaml · .files.tsv   그때의 manifest 사본
                              + 커밋 안 된 수정이 있으면 code.patch (git diff HEAD)
  export   (train.py)     →  .onnx 메타데이터에 train_commit · train_dirty · train_run · train_config
  보관     (archive.py)   →  ../weights/<name>/<시각>/ 에 위 기록째로 복사 (git 에 커밋)

지문(hash)은 원본 이미지 바이트 + 최종 라벨 + split 으로 계산한다 — 같으면 학습이 본 데이터가
똑같고, 라벨 한 줄만 고쳐도 달라진다. 전체 지문에는 전처리 설정(크롭 · 클래스 등)도 들어간다.

Ultralytics 도 .pt 안에 git 정보를 넣지만(ckpt["git"]) 커밋 안 된 수정은 모르고, .venv 가
저장소 안에 있을 때만 우리 저장소를 잡는다 — 그래서 여기서 직접 남긴다. run_info.yaml 이
없는 예전 실행은 그 ckpt["git"] 으로 대신한다 (from_checkpoint).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from common import HERE, IMG_EXTS

REPO = HERE.parent
RUN_INFO = "run_info.yaml"
PATCH = "code.patch"
RUN_CONFIG = "config.yaml"
MANIFEST = "manifest.yaml"
MANIFEST_FILES = "manifest.files.tsv"
RUN_DATA = "data"          # 실행 폴더 안 manifest 사본 폴더

# 환경 기록에 버전을 남길 패키지 (설치된 것만)
PACKAGES = ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless",
            "albumentations", "pillow", "pyyaml", "onnx", "onnxruntime-gpu", "onnxslim")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              encoding="utf-8", check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_state() -> dict:
    """현재 커밋 · 브랜치 · 커밋 안 된 파일(추적 중인 수정 + 새 파일)."""
    changed = [line[3:] for line in _git("status", "--porcelain").splitlines() if line.strip()]
    return {
        "commit": _git("rev-parse", "HEAD").strip() or None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip() or None,
        "subject": _git("log", "-1", "--format=%s").strip() or None,
        "dirty": bool(changed),
        "uncommitted": changed,
    }


# ── 지문 ──────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tagged(digest: str) -> str:
    return f"sha256:{digest}"


# ── 전처리 manifest ───────────────────────────────────────────────────────

def write_manifest(out_dir: Path, header: dict, processing: dict, rows: list,
                   raw_counts: dict, meta: dict | None = None) -> dict:
    """
    소스 하나의 산출물 옆에 manifest.yaml + manifest.files.tsv 를 쓴다.

    rows: (split, 세션, 산출 파일명, 원본 이미지 sha256, 최종 라벨 sha256, 복제 수) — 이미지 한 장당 한 줄
    raw_counts: 세션 → 원본 이미지 수 (라벨이 없거나 크롭으로 빠진 장을 skipped 로 남기려고)
    processing: 산출물을 바꾸는 전처리 설정 (크롭 · 리사이즈 · 클래스 · task · 시드 등) — 전체 지문에 들어감
    meta: 세션 → session.yaml — scene · tags · layer 를 읽기 편하게 옮겨 적는다 (지문에는 안 들어감)
    """
    rows = sorted(rows, key=lambda r: (r[1], r[2]))
    sessions: dict = {}
    lines: dict = {}
    for split, session, name, img_sha, lbl_sha, reps in rows:
        s = sessions.setdefault(session, {"split": split, "images": 0})
        if s["split"] != split:
            s["split"] = "mixed"            # group_by: file 이면 세션이 split 에 나뉠 수 있다
        s["images"] += 1
        lines.setdefault(session, []).append(f"{split}\t{name}\t{img_sha}\t{lbl_sha}\t{reps}")
    for session, s in sessions.items():
        s["hash"] = _tagged(sha256_text("\n".join(lines[session])))
        s.update({k: v for k, v in ((meta or {}).get(session) or {}).items()
                  if k in ("scene", "tags", "layer") and v not in (None, "", [])})

    images = {"train": 0, "val": 0, "test": 0}
    for split, *_ in rows:
        images[split] += 1
    whole = json.dumps({"processing": processing,
                        "sessions": {k: v["hash"] for k, v in sorted(sessions.items())}},
                       sort_keys=True, ensure_ascii=False, default=str)
    skipped = {k: n - sessions.get(k, {}).get("images", 0) for k, n in sorted(raw_counts.items())}

    doc = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        **header,
        "git": {k: v for k, v in git_state().items() if k in ("commit", "dirty")},
        "hash": _tagged(sha256_text(whole)),
        "images": {k: v for k, v in images.items() if v},
        "sessions": dict(sorted(sessions.items())),
        "skipped": {k: v for k, v in skipped.items() if v},   # 라벨 없음 · 클래스 없음 · 크롭으로 빠진 장
        "processing": processing,
    }
    out_dir = Path(out_dir)
    (out_dir / MANIFEST).write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                                    encoding="utf-8")
    with (out_dir / MANIFEST_FILES).open("w", encoding="utf-8", newline="\n") as f:
        f.write("split\tsession\timage\traw_sha256\tlabel_sha256\treps\n")
        for split, session, name, img_sha, lbl_sha, reps in rows:
            f.write(f"{split}\t{session}\t{name}\t{img_sha}\t{lbl_sha}\t{reps}\n")
    return doc


def manifest_dirs(data_yaml: Path) -> list:
    """
    data.yaml 이 가리키는 산출물 폴더들 (<산출물>/images/<split> → <산출물>). manifest 를 찾는 자리.
    """
    try:
        d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    except OSError:
        return []
    root = Path(d.get("path") or Path(data_yaml).parent)
    found = []
    for split in ("train", "val", "test"):
        dirs = d.get(split) or []
        for s in dirs if isinstance(dirs, list) else [dirs]:
            p = (root / s).resolve()
            src = p.parent.parent if p.parent.name == "images" else p
            if src not in found:
                found.append(src)
    return found


def read_manifests(data_yaml: Path) -> dict:
    """
    소스 이름 → (산출물 폴더, manifest). manifest 가 없으면 (폴더, None) 이고 이름은 폴더 이름
    — 이 기능 이전에 구운 산출물.
    """
    out = {}
    for d in manifest_dirs(data_yaml):
        p = d / MANIFEST
        m = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None
        out[(m or {}).get("source") or d.name] = (d, m)
    return out


def data_splits(data_yaml: Path) -> dict:
    """data.yaml 의 split → 폴더 목록 — 학습이 실제로 무엇을 train/val 로 썼는지 (data.all 은 train 에 전부)."""
    try:
        d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    out = {}
    for split in ("train", "val", "test"):
        v = d.get(split)
        if v:
            out[split] = v if isinstance(v, list) else [v]
    return out


def _count_images(data_yaml: Path) -> dict:
    """data.yaml 의 split 별 이미지 수 (폴더 목록이면 합)."""
    try:
        d = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except OSError:
        return {}
    root = Path(d.get("path") or data_yaml.parent)
    out = {}
    for split in ("train", "val", "test"):
        dirs = d.get(split)
        if not dirs:
            continue
        dirs = dirs if isinstance(dirs, list) else [dirs]
        out[split] = sum(1 for s in dirs if (root / s).is_dir()
                         for p in (root / s).rglob("*") if p.suffix.lower() in IMG_EXTS)
    return out


# ── 실행 환경 ─────────────────────────────────────────────────────────────

def environment() -> dict:
    """파이썬 · 패키지 · CUDA · GPU · uv.lock 지문 — 같은 결과를 내려면 맞춰야 하는 것들."""
    from importlib import metadata

    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "packages": packages}
    lock = HERE / "uv.lock"
    if lock.exists():
        env["uv_lock"] = _tagged(sha256_file(lock))
    try:
        import torch

        env["cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.split()
        if drv:
            env["nvidia_driver"] = drv[0]
    except (OSError, subprocess.CalledProcessError):
        pass
    return env


def weights_info(weights: str) -> dict:
    """시작 가중치 — 파일이면 경로 + 지문, 이름(yolo26s-seg.pt)이면 Ultralytics 가 받은 파일의 지문."""
    p = Path(weights)
    if not p.exists():
        cand = HERE / "weights" / p.name        # train.py 가 Ultralytics weights_dir 을 여기로 둔다
        p = cand if cand.exists() else p
    info = {"model": str(weights)}
    if p.exists():
        info["file"] = _rel(p)
        info["sha256"] = sha256_file(p)
        src_run = p.parent.parent / RUN_INFO       # 이전 실행의 best.pt 에서 이어 학습한 경우
        if p.parent.name == "weights" and src_run.exists():
            info["from_run"] = (yaml.safe_load(src_run.read_text(encoding="utf-8")) or {}).get("run")
    return info


# ── 실행 기록 ─────────────────────────────────────────────────────────────

def _rel(path: Path) -> str:
    """training/ 기준 상대경로 — 저장소를 옮기거나 폴더 이름을 바꿔도 기록이 맞도록 (다른 드라이브면 그대로)."""
    try:
        return Path(os.path.relpath(path, HERE)).as_posix()
    except ValueError:
        return str(path)


def write_run_info(save_dir: Path, config_path: Path, cfg: dict, data_yaml: Path,
                   weights: str | None = None) -> dict:
    """
    학습 시작 시 실행 폴더에 run_info.yaml (+ dirty 면 code.patch, manifest 사본, 재현용 config.yaml).
    """
    save_dir = Path(save_dir)
    git = git_state()
    data = {"yaml": _rel(data_yaml), "images": _count_images(data_yaml),
            "splits": data_splits(data_yaml), "manifest": {}}
    for name, (src_dir, m) in read_manifests(data_yaml).items():
        if m is None:
            data["manifest"][name] = {"note": "manifest 없음 — 이 기능 이전에 구운 산출물. 다시 전처리하면 생김"}
            continue
        data["manifest"][name] = {k: m.get(k) for k in ("hash", "created", "env", "task", "images",
                                                        "sessions", "skipped")}
        (save_dir / RUN_DATA).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_dir / MANIFEST, save_dir / RUN_DATA / f"{name}.{MANIFEST}")
        crop_index = src_dir / "crop_sources.json"
        if crop_index.exists():
            shutil.copyfile(crop_index, save_dir / RUN_DATA / f"{name}.crop_sources.json")
        if (src_dir / MANIFEST_FILES).exists():
            shutil.copyfile(src_dir / MANIFEST_FILES, save_dir / RUN_DATA / f"{name}.{MANIFEST_FILES}")
    info = {
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": _rel(save_dir),
        "config": config_path.name,
        "git": git,
        "data": data,
        "start_weights": weights_info(weights) if weights else None,
        "environment": environment(),
        "config_resolved": cfg,   # base 상속까지 합친 config 전체 (blur · brightness 등 포함)
    }
    if git["dirty"]:
        patch = _git("diff", "HEAD", "--binary")
        if patch:
            (save_dir / PATCH).write_text(patch, encoding="utf-8")
            info["git"]["patch"] = PATCH
    write_info(save_dir, info)
    return info


def write_info(save_dir: Path, info: dict) -> None:
    (Path(save_dir) / RUN_INFO).write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False),
                                           encoding="utf-8")


def write_run_config(save_dir: Path, cfg: dict, weights_file: str | None, batch: int | None,
                     info: dict) -> Path:
    """
    재현용 config.yaml — 상속(base)을 푼 config 에, 실행 때 정해진 값을 고정한다:
      · train.model  "최신 실행" 해석(타임스탬프 없는 경로)이 아니라 실제로 쓴 파일
      · train.batch  -1(AutoBatch)이면 GPU 마다 달라지므로 실제로 잡힌 값
      · reproduce    데이터 지문 · 커밋 — train.py 가 다시 돌릴 때 대조한다
    """
    import copy

    out = copy.deepcopy(cfg)
    t = out.setdefault("train", {})
    if weights_file:
        t["model"] = weights_file
    if batch:
        t["batch"] = int(batch)
    out["reproduce"] = {
        "run": info.get("run"),
        "commit": (info.get("git") or {}).get("commit"),
        "patch": (info.get("git") or {}).get("patch"),
        "data_hash": {k: v.get("hash") for k, v in ((info.get("data") or {}).get("manifest") or {}).items()
                      if v.get("hash")},
    }
    p = Path(save_dir) / RUN_CONFIG
    head = ("# 재현용 config — train.py 가 이 실행에서 실제로 쓴 값 (base 상속을 풀고 시작 가중치 · batch 고정).\n"
            "#   uv run train.py --config <이 파일>\n"
            "# reproduce 블록: 시작할 때 데이터 지문이 다르면 중단하고, 커밋이 다르면 경고한다.\n")
    p.write_text(head + yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def from_checkpoint(weights: Path) -> dict:
    """run_info.yaml 이 없는 예전 실행 — Ultralytics 가 .pt 에 넣은 git 정보로 대신."""
    import torch

    ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    g = ck.get("git") or {}
    return {
        "started": None,
        "saved": ck.get("date"),
        "run": None,
        "config": None,
        "git": {"commit": g.get("commit"), "branch": g.get("branch"), "subject": g.get("message"),
                "dirty": None, "uncommitted": None,
                "note": "run_info.yaml 이전 실행 — Ultralytics ckpt['git'] (커밋 안 된 수정 여부 모름)"},
    }


def read(weights: Path) -> dict:
    """<실행 폴더>/weights/x.pt → 그 실행의 run_info (없으면 .pt 의 git 정보)."""
    weights = Path(weights)
    run_dir = weights.parent.parent
    p = run_dir / RUN_INFO
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    info = from_checkpoint(weights)
    info["run"] = _rel(run_dir)
    return info


def stamp_onnx(onnx_path: Path, info: dict) -> None:
    """.onnx 메타데이터에 출처를 덧붙인다 (Ultralytics 메타데이터는 그대로 둠)."""
    import onnx

    git = info.get("git") or {}
    manifests = (info.get("data") or {}).get("manifest") or {}
    extra = {
        "train_commit": git.get("commit") or "unknown",
        "train_dirty": {True: "true", False: "false"}.get(git.get("dirty"), "unknown"),
        "train_run": info.get("run") or "unknown",
        "train_config": info.get("config") or "unknown",
        "train_data": ", ".join(f"{k}={v['hash']}" for k, v in manifests.items() if v.get("hash"))
                      or "unknown",
    }
    model = onnx.load(str(onnx_path))
    keep = [p for p in model.metadata_props if p.key not in extra]
    del model.metadata_props[:]
    model.metadata_props.extend(keep)
    for k, v in extra.items():
        model.metadata_props.add(key=k, value=str(v))
    onnx.save(model, str(onnx_path))
