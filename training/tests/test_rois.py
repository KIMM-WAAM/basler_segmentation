"""preprocess 의 rois (이미지 한 장 → 고정 창 여러 개) 회귀 검사. torch 없이 돈다."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from preprocess import Preprocessor


def make_source(tmp_path, rois, sessions=("261101_120000", "261102_120000", "261103_120000")):
    for i, session in enumerate(sessions):
        directory = tmp_path / "raw" / session
        (directory / "Mono").mkdir(parents=True)
        (directory / "labels_seg").mkdir()
        # 좌표를 픽셀값으로 인코딩 (R = x, G = y) — 어느 영역을 잘랐는지 산출 이미지로 검사
        yy, xx = np.indices((100, 200), dtype=np.uint8)
        Image.fromarray(np.stack([xx, yy, np.zeros_like(xx)], axis=-1)).save(directory / "Mono" / f"img{i}.png")
        # 왼쪽 와이어 x 30~40, 오른쪽 와이어 x 150~160 (원본 200×100)
        (directory / "labels_seg" / f"img{i}.txt").write_text(
            "0 .15 .1 .2 .1 .2 .6 .15 .6\n0 .75 .1 .8 .1 .8 .6 .75 .6\n", encoding="utf-8")
    cfg = {"dataset": "bestfa", "names": {0: "wire"}, "task": "seg", "out": str(tmp_path / "out"),
           "sources": [{"name": "bestfa", "path": str(tmp_path / "raw" / "[0-9]*"), "labels": "labels_seg",
                        "names": {0: "wire"}, "class_map": {"wire": "wire"}, "type": "seg",
                        "group_by": "session", "val_sessions": [sessions[1]], "oversample": 1,
                        "rois": rois}]}
    config = tmp_path / "preprocess.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config


ROIS = [{"name": "r1", "width": 80, "height": 80, "center_x": 0, "center_y": 0},     # x 0~80
        {"name": "r2", "width": 80, "height": 80, "center_x": 100, "center_y": 0}]   # x 120~200


def test_each_roi_becomes_its_own_sample(tmp_path):
    config = make_source(tmp_path, ROIS)        # 세션 crop.yaml 없이 돈다
    Preprocessor(config).run()
    out = tmp_path / "out"
    train = sorted(p.name for p in (out / "images/train").iterdir())
    assert train == ["bestfa__img0__r1.png", "bestfa__img0__r2.png",
                     "bestfa__img2__r1.png", "bestfa__img2__r2.png"]
    assert sorted(p.name for p in (out / "images/val").iterdir()) == ["bestfa__img1__r1.png",
                                                                       "bestfa__img1__r2.png"]
    # r1 은 왼쪽 와이어 하나만, r2 는 오른쪽 와이어 하나만 (다른 와이어는 창 밖)
    r1 = (out / "labels/train/bestfa__img0__r1.txt").read_text().split("\n")
    r2 = (out / "labels/train/bestfa__img0__r2.txt").read_text().split("\n")
    assert len([ln for ln in r1 if ln]) == 1 and len([ln for ln in r2 if ln]) == 1
    x_r1 = [float(v) for v in r1[0].split()[1::2]]
    x_r2 = [float(v) for v in r2[0].split()[1::2]]
    assert min(x_r1) == pytest.approx(30 / 80) and max(x_r1) == pytest.approx(40 / 80)
    assert min(x_r2) == pytest.approx(30 / 80) and max(x_r2) == pytest.approx(40 / 80)   # 150-120, 160-120
    # 산출 이미지는 그 창의 원본 픽셀
    with Image.open(out / "images/train/bestfa__img0__r2.png") as im:
        assert im.size == (80, 80)
        assert np.asarray(im)[0, 0].tolist()[:2] == [120, 0]
    # 학습 중 원본 크롭 이동이 ROI 창 기준으로 움직이도록 crop_sources.json 에 그 창이 들어간다
    index = json.loads((out / "crop_sources.json").read_text(encoding="utf-8"))["images"]
    entry = index["bestfa__img0__r2.png"]
    assert entry["crop"] == {"enabled": True, "width": 80, "height": 80, "center_x": 100.0,
                             "center_y": 0.0, "min_area": 0.1}
    assert len(entry["labels"]) == 2           # 원본 라벨 전체 — 이동한 창에 걸치는 것은 학습에 들어감
    assert Path(entry["image"]).name == "img0.png"


def test_roi_without_target_is_skipped_but_others_kept(tmp_path):
    rois = ROIS + [{"name": "r3", "width": 40, "height": 20, "center_x": 50, "center_y": 100}]  # 아래 가운데, 와이어 없음
    Preprocessor(make_source(tmp_path, rois)).run()
    names = sorted(p.name for p in (tmp_path / "out/images/train").iterdir())
    assert not any("__r3" in n for n in names) and len(names) == 4


@pytest.mark.parametrize("bad, message", [
    ([], "창 목록"),
    ([{"name": "r1", "width": 0, "height": 80, "center_x": 0, "center_y": 0}], "양의 정수"),
    ([{"name": "r1", "width": 80, "height": 80, "center_x": 120, "center_y": 0}], "0~100"),
    ([ROIS[0], ROIS[0]], "겹치지 않는"),
    ([{**ROIS[0], "angle": 3}], "모르는 키"),
])
def test_invalid_rois_rejected(tmp_path, bad, message):
    with pytest.raises(SystemExit, match=message):
        Preprocessor(make_source(tmp_path, bad)).run()
