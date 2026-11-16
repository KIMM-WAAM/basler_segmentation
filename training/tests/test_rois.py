"""세션 crop.yaml 의 ROI 여러 개(BestFA: ROI 하나에 와이어 하나) — 검증 · 전처리 · 원본 크롭 인덱스."""
import random
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from dataset_paths import crop_windows, session_crop
from preprocess import Preprocessor
from source_crop import read_index

# 120×100 원본, 왼쪽 와이어 x 10~20 · 오른쪽 와이어 x 90~100 (y 20~80)
LEFT = "0 .0833 .2 .1667 .2 .1667 .8 .0833 .8"
RIGHT = "0 .75 .2 .8333 .2 .8333 .8 .75 .8"
ROIS = [{"width": 60, "height": 60, "center_x": 0, "center_y": 50},
        {"width": 60, "height": 60, "center_x": 100, "center_y": 50}]


def write_session(root: Path, name: str, crop: dict, labels: str = f"{LEFT}\n{RIGHT}\n") -> Path:
    directory = root / name
    (directory / "Mono").mkdir(parents=True)
    (directory / "labels_seg").mkdir()
    yy, xx = np.indices((100, 120), dtype=np.uint8)
    Image.fromarray(np.stack([xx, yy, np.zeros_like(xx)], axis=-1)).save(directory / "Mono" / f"{name}.png")
    (directory / "labels_seg" / f"{name}.txt").write_text(labels, encoding="utf-8")
    (directory / "crop.yaml").write_text(yaml.safe_dump(crop), encoding="utf-8")
    return directory


def config(root: Path, sessions: list, roi_count: int = 2) -> Path:
    cfg = {"dataset": "bestfa", "names": {0: "wire"}, "task": "seg", "out": str(root / "out"),
           "roi_count": roi_count,
           "sources": [{"name": "bestfa", "path": str(root / "[0-9]*"), "labels": "labels_seg",
                        "names": {0: "wire"}, "class_map": {"wire": "wire"}, "type": "seg",
                        "group_by": "session", "val_sessions": [sessions[1]]}]}
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_crop_windows_shapes(tmp_path):
    single = {"enabled": True, "width": 640, "height": 640, "center_x": 65, "center_y": 10}
    assert crop_windows(single) == [single]
    assert crop_windows({"enabled": False}) == [{"enabled": False}]
    multi = {"enabled": True, "min_area": 0.2, "rois": ROIS}
    windows = crop_windows(multi)
    assert [w["center_x"] for w in windows] == [0, 100]
    assert all(w["enabled"] and w["min_area"] == 0.2 and "rois" not in w for w in windows)


@pytest.mark.parametrize("crop,match", [
    ({"enabled": True, "rois": []}, "rois"),
    ({"enabled": True, "rois": [{"width": 60, "height": 60, "center_x": 50}]}, r"rois\[0\]\.center_y"),
    ({"enabled": True, "rois": [{**ROIS[0], "center_x": 120}]}, "범위"),
    ({"enabled": True, "rois": ROIS, "auto_crop": {"enabled": True}}, "auto_crop"),
])
def test_session_crop_rejects_bad_rois(tmp_path, crop, match):
    (tmp_path / "crop.yaml").write_text(yaml.safe_dump(crop), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        session_crop(tmp_path)


def test_preprocess_one_sample_per_roi(tmp_path, monkeypatch):
    sessions = ["261101_120000", "261102_120000"]
    for name in sessions:
        write_session(tmp_path, name, {"enabled": True, "rois": ROIS})
    Preprocessor(config(tmp_path, sessions)).run()
    out = tmp_path / "out"

    train = sorted(p.name for p in (out / "images/train").iterdir())
    assert train == [f"bestfa__{sessions[0]}_r1.png", f"bestfa__{sessions[0]}_r2.png"]
    # ROI 1 = x 0~60 (왼쪽 와이어만), ROI 2 = x 60~120 (오른쪽 와이어만)
    for roi, x0 in (("r1", 0), ("r2", 60)):
        stem = f"bestfa__{sessions[0]}_{roi}"
        with Image.open(out / "images/train" / f"{stem}.png") as im:
            assert im.size == (60, 60) and im.getpixel((0, 0))[0] == x0
        lines = (out / "labels/train" / f"{stem}.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    entries = read_index(out, verify_images=True)
    first, second = (entries[f"bestfa__{sessions[0]}_r{k}.png"] for k in (1, 2))
    assert (first["roi"], second["roi"]) == (0, 1)
    assert first["crop"]["center_x"] == 0 and second["crop"]["center_x"] == 100
    assert "rois" not in first["crop"] and len(first["labels"]) == 2   # 원본 라벨 전부 (랜덤 크롭용)

    manifest = yaml.safe_load((out / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["images"] == {"train": 2, "val": 2}
    assert not manifest["skipped"]


def test_source_crop_jitter_uses_roi_window(tmp_path, monkeypatch):
    from ultralytics.cfg import get_cfg

    from crop_dataset import SourceCropDataset

    sessions = ["261101_120000", "261102_120000"]
    for name in sessions:
        write_session(tmp_path, name, {"enabled": True, "rois": ROIS})
    Preprocessor(config(tmp_path, sessions)).run()
    out = tmp_path / "out"
    hyp = get_cfg(overrides={"imgsz": 64, "task": "segment", "mosaic": 0., "mixup": 0.,
                             "copy_paste": 0., "translate": 0., "scale": 0., "degrees": 0.,
                             "fliplr": 0., "flipud": 0., "hsv_h": 0., "hsv_s": 0., "hsv_v": 0.})
    dataset = SourceCropDataset(
        img_path=str(out / "images/train"), imgsz=60, batch_size=2, augment=True, hyp=hyp,
        task="segment", data={"names": {0: "wire"}, "nc": 1, "channels": 3},
        crop_entries=read_index(out), crop_root=out,
        crop_options={"offset_x": [-25, 25], "offset_y": [0, 0]},
    )
    monkeypatch.setattr(random, "uniform", lambda lo, hi: hi)
    names = [Path(f).name for f in dataset.im_files]
    r1 = dataset.get_image_and_label(names.index(f"bestfa__{sessions[0]}_r1.png"))
    r2 = dataset.get_image_and_label(names.index(f"bestfa__{sessions[0]}_r2.png"))
    # ROI 1 은 0% + 25%p → x0 = 60*.25 = 15, ROI 2 는 100% 에서 더 못 감 → x0 = 60. BGR 의 R = x.
    assert r1["img"][0, 0, 2] == 15 and r2["img"][0, 0, 2] == 60


def test_roi_count_mismatch_fails_before_writing(tmp_path):
    sessions = ["261101_120000", "261102_120000"]
    write_session(tmp_path, sessions[0], {"enabled": True, "rois": ROIS})
    write_session(tmp_path, sessions[1], {"enabled": True, "width": 60, "height": 60,
                                          "center_x": 0, "center_y": 50})
    with pytest.raises(SystemExit):
        Preprocessor(config(tmp_path, sessions)).run()
    assert not (tmp_path / "out" / "images").exists()
