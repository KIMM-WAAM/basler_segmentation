"""원본 크롭 좌표·분할·Windows worker 호환성 회귀 검사. 실제 데이터는 수정하지 않는다."""
import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset

from crop_dataset import SourceCropDataset, SourceCropSegTrainer
from preprocess import Preprocessor
from source_crop import INDEX, read_index, validate_options


@pytest.fixture
def prepared(tmp_path):
    sessions = ["261101_120000", "261102_120000", "261103_120000"]
    for i, session in enumerate(sessions):
        directory = tmp_path / session
        (directory / "Mono").mkdir(parents=True)
        (directory / "labels_seg").mkdir()
        # 좌표를 픽셀값으로 인코딩하여 원본의 어느 영역을 읽었는지 검사.
        yy, xx = np.indices((100, 120), dtype=np.uint8)
        image = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
        Image.fromarray(image).save(directory / "Mono" / f"image{i}.png")
        (directory / "labels_seg" / f"image{i}.txt").write_text(
            "0 .4 .3 .6 .3 .6 .7 .4 .7\n", encoding="utf-8")
        (directory / "crop.yaml").write_text(yaml.safe_dump({
            "enabled": True, "width": 60, "height": 60, "center_x": 50, "center_y": 50,
        }), encoding="utf-8")
    cfg = {"dataset": "kimm", "names": {0: "wire"}, "task": "seg", "out": str(tmp_path / "out"),
           "sources": [{"name": "kimm", "path": str(tmp_path / "[0-9]*"), "labels": "labels_seg",
                        "names": {0: "wire"}, "class_map": {"wire": "wire"}, "type": "seg",
                        "group_by": "session", "val_sessions": [sessions[1]], "test_sessions": [sessions[2]],
                        "oversample": 2}]}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    Preprocessor(config).run()
    return tmp_path / "out"


def make_dataset(root, task="segment"):
    hyp = get_cfg(overrides={"imgsz": 64, "task": task, "mosaic": 0., "mixup": 0.,
                             "copy_paste": 0., "translate": 0., "scale": 0., "degrees": 0.,
                             "fliplr": 0., "flipud": 0., "hsv_h": 0., "hsv_s": 0., "hsv_v": 0.})
    dataset = SourceCropDataset(
        img_path=str(root / "images/train"), imgsz=60, batch_size=2, augment=True, hyp=hyp,
        task=task, data={"names": {0: "wire"}, "nc": 1, "channels": 3},
        crop_entries=read_index(root), crop_root=root,
        crop_options={"offset_x": [-25, 25], "offset_y": [-25, 25]},
    )
    return dataset


def test_original_pixels_and_polygon_transform(prepared, monkeypatch):
    entries = read_index(prepared, verify_images=True)
    assert len(entries) == 4  # oversample 항목 포함
    dataset = make_dataset(prepared)
    monkeypatch.setattr(random, "uniform", lambda lo, hi: lo)
    low = dataset.get_image_and_label(0)
    # (120-60)*.25=15, (100-60)*.25=10. BGR에서 R=x, G=y.
    assert low["img"][0, 0].tolist() == [0, 10, 15]
    np.testing.assert_allclose(low["instances"].bboxes[0], [.75, 2 / 3, .4, 2 / 3], atol=1e-5)
    monkeypatch.setattr(random, "uniform", lambda lo, hi: hi)
    high = dataset.get_image_and_label(0)
    assert high["img"][0, 0].tolist() == [0, 30, 45]
    assert high["img"].shape == low["img"].shape == (60, 60, 3)
    assert not np.array_equal(high["instances"].bboxes, low["instances"].bboxes)


def test_empty_crop_and_two_wires(prepared, monkeypatch):
    dataset = make_dataset(prepared)
    entry = dataset.crop_entries[Path(dataset.im_files[0]).name]
    monkeypatch.setattr(random, "uniform", lambda lo, hi: lo)
    entry["labels"] *= 2
    assert len(dataset.get_image_and_label(0)["cls"]) == 2
    entry["labels"] = ["0 .95 .9 .99 .9 .99 .99 .95 .99"]
    sample = dataset.get_image_and_label(0)
    assert sample["cls"].shape == (0, 1)
    assert sample["instances"].bboxes.shape == (0, 4)
    monkeypatch.undo()
    assert dataset[0]["masks"].numel() > 0  # 표준 YOLO Format이 빈 마스크도 처리


def test_windows_worker_and_obb(prepared):
    dataset = make_dataset(prepared)
    loader = DataLoader(dataset, batch_size=2, num_workers=1,
                        multiprocessing_context="spawn", collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    assert batch["img"].shape[0] == 2
    assert len(batch["cls"]) == 2
    obb = make_dataset(prepared, task="obb")
    assert obb[0]["bboxes"].shape == (1, 5)


def test_validation_is_standard_fixed_dataset(prepared):
    # build_dataset의 train/val 분기를 실제 메서드로 검사 (GPU 학습은 실행하지 않음).
    trainer = SourceCropSegTrainer.__new__(SourceCropSegTrainer)
    trainer.crop_root = prepared
    trainer.crop_options = {"offset_x": [-25, 25], "offset_y": [-25, 25]}
    trainer.crop_entries = read_index(prepared)
    trainer.args = get_cfg(overrides={"task": "segment", "imgsz": 64, "mosaic": 0., "translate": 0.})
    trainer.model = SimpleNamespace(stride=torch.tensor([32]))
    trainer.data = {"names": {0: "wire"}, "nc": 1, "channels": 3}
    val = trainer.build_dataset(str(prepared / "images/val"), "val", 1)
    assert type(val) is YOLODataset
    before = val[0]
    after = val[0]
    assert torch.equal(before["img"], after["img"])
    assert torch.equal(before["bboxes"], after["bboxes"])
    train = trainer.build_dataset([str(prepared / f"images/{s}") for s in ("train", "val", "test")], "train", 2)
    assert isinstance(train, SourceCropDataset) and len(train) == 4


def test_integrity_and_options(prepared):
    entries = read_index(prepared)
    image = (prepared / next(iter(entries.values()))["image"]).resolve()
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="원본 이미지"):
        read_index(prepared, verify_images=True)
    index = prepared / INDEX
    doc = json.loads(index.read_text(encoding="utf-8"))
    doc["images"] = copy.deepcopy(doc["images"])
    next(iter(doc["images"].values()))["labels"] = []
    index.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="지문 불일치"):
        read_index(prepared)
    with pytest.raises(ValueError):
        validate_options({"offset_x": [-101, 25], "offset_y": [-25, 25]})


def test_cpu_training_smoke(prepared):
    from ultralytics import YOLO

    from crop_dataset import trainer_factory

    # 사전학습 다운로드 없이 임시 이미지로 실제 forward/backward + val 한 번.
    model = YOLO("yolo26n-seg.yaml")
    factory = trainer_factory("segment", prepared, {"offset_x": [-25, 25], "offset_y": [-25, 25]})
    model.train(trainer=factory, data=str(prepared / "data.yaml"), device="cpu", epochs=1,
                batch=2, imgsz=64, workers=0, amp=False, pretrained=False, optimizer="SGD",
                plots=False, save=False, cache=False, mosaic=0., translate=0., scale=0.,
                project=str(prepared.parent / "smoke"), name="crop", verbose=False)
    assert isinstance(model.trainer.train_loader.dataset, SourceCropDataset)
    assert type(model.trainer.test_loader.dataset) is YOLODataset


@pytest.mark.parametrize("center,offset,expected", [
    (10, -25, [0, 0, 0]),
    (90, 25, [0, 40, 60]),
    (10, 25, [0, 14, 21]),
    (90, -25, [0, 26, 39]),
])
def test_session_relative_crop_clamped(prepared, monkeypatch, center, offset, expected):
    dataset = make_dataset(prepared)
    entry = dataset.crop_entries[Path(dataset.im_files[0]).name]
    entry["crop"].update(center_x=center, center_y=center)
    monkeypatch.setattr(random, "uniform", lambda lo, hi: offset)
    sample = dataset.get_image_and_label(0)
    assert sample["img"][0, 0].tolist() == expected
    assert sample["img"].shape == (60, 60, 3)
    assert entry["crop"]["center_x"] == center


def test_legacy_absolute_options_rejected():
    with pytest.raises(ValueError, match="offset_x"):
        validate_options({"center_x": [25, 75], "center_y": [25, 75]})
