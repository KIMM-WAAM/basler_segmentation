"""환경 분리와 외부 원본 가져오기: 교차 참조·덮어쓰기 방지 회귀 검증."""
import sys

import pytest
import yaml
from PIL import Image

from common import HERE, load_yaml
from dataset_paths import DATASETS, check_source_dataset, dataset_root, default_crop, preprocess_config
from preprocess import Preprocessor
from tools.import_bestfa import digest, import_session, mono_dirs


def test_dataset_profiles():
    for dataset in DATASETS:
        st = Preprocessor(dataset=dataset)
        assert st.out.resolve() == dataset_root(dataset) / "processed" / "seg"
        assert {s["name"] for s in st.all_sources()} == {dataset}
        assert Preprocessor(dataset=dataset, task="obb").out == dataset_root(dataset) / "processed" / "obb"
        for kind in ("single", "seg_all", "obb_all"):
            name = f"train.{kind}.yaml" if dataset == "kimm" else f"train.bestfa.{kind}.yaml"
            cfg = load_yaml(HERE / "configs" / name)
            assert f"datasets/{dataset}/processed/" in cfg["train"]["data"]
            assert f"runs/{dataset}/" in cfg["train"]["project"]
    assert default_crop("kimm")["width"] == 1280
    assert default_crop("bestfa") is None
    with pytest.raises(SystemExit, match="다릅니다"):
        Preprocessor(preprocess_config("kimm"), dataset="bestfa")
    with pytest.raises(ValueError):
        dataset_root("../kimm")
    with pytest.raises(ValueError, match="--dataset bestfa"):
        check_source_dataset(dataset_root("bestfa") / "raw" / "photo.jpg", "kimm")
    check_source_dataset(dataset_root("bestfa") / "raw" / "photo.jpg", "bestfa")


def test_processing_and_labeller_isolate_environments(tmp_path):
    sys.path.insert(0, str(HERE / "labeller"))
    import server as lab

    configs = {}
    for dataset, width in (("kimm", 32), ("bestfa", 48)):
        root = tmp_path / dataset
        for index in range(3):
            session = root / "raw" / f"{dataset}_260101_00000{index}"
            (session / "Mono").mkdir(parents=True)
            (session / "labels_seg").mkdir()
            Image.new("RGB", (64, 64), "white").save(session / "Mono" / f"frame{index}.png")
            (session / "labels_seg" / f"frame{index}.txt").write_text(
                "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
        cfg = load_yaml(preprocess_config(dataset))
        cfg["out"] = str(root / "processed" / "seg")
        cfg["reference"] = {"directory": str(root / "reference")}
        source = cfg["sources"][0]
        source["path"] = str(root / "raw" / f"{dataset}_*")
        source["val_sessions"] = [f"{dataset}_260101_000001"]
        source["test_sessions"] = [f"{dataset}_260101_000002"]
        source["crop"] = {"enabled": True, "width": width, "height": width}
        configs[dataset] = tmp_path / f"preprocess.{dataset}.yaml"
        configs[dataset].write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        Preprocessor(configs[dataset]).run()
        data = load_yaml(root / "processed" / "seg" / "data.yaml")
        assert data["train"] == [f"{dataset}/images/train"]
        for split in ("train", "val", "test"):
            image = next((root / "processed" / "seg" / dataset / "images" / split).glob("*.png"))
            with Image.open(image) as im:
                assert im.size == (width, width)
        selected = lab.Source(dataset, configs[dataset])
        reference = lab.source_reference_settings(selected)
        assert reference["tube"] == root / "reference" / "reference_tube.png"
        assert not reference["tube"].exists()  # 다른 환경의 기존 기준 이미지로 대체하지 않음

    before = configs["kimm"].read_bytes()
    selected = lab.Source("bestfa", configs["bestfa"])
    selected.save_crop({"enabled": True, "width": 40, "height": 40, "center_x": 30, "center_y": 60})
    assert configs["kimm"].read_bytes() == before
    assert lab.Source("bestfa", configs["bestfa"]).crop_state()["width"] == 40


def test_import_preserves_originals_and_is_repeatable(tmp_path):
    db, raw = tmp_path / "DB", tmp_path / "raw"
    for session in ("261026_115301", "261026_144002"):
        mono = db / session / "Mono"
        mono.mkdir(parents=True)
        Image.new("RGB", (12, 8), "white").save(mono / "00_00_00_020.jpg")
    originals = {p: digest(p) for p in db.rglob("*.jpg")}
    for mono in mono_dirs(db):
        assert import_session(mono, raw)["copied"] == 1
        assert import_session(mono, raw)["copied"] == 0
        destination = raw / f"bestfa_{mono.parent.name}"
        assert (destination / "labels").is_dir()
        assert (destination / "labels_seg").is_dir()
        assert (destination / "import.json").is_file()
    images = list(raw.glob("*/Mono/*.jpg"))
    assert len({p.name for p in images}) == 2
    assert {p: digest(p) for p in originals} == originals
    assert len(list(db.rglob("*"))) == 6  # 원본에는 라벨이나 manifest를 생성하지 않음
    images[0].write_bytes(b"different")
    with pytest.raises(FileExistsError, match="내용이 다른"):
        import_session(db / images[0].parents[1].name.removeprefix("bestfa_") / "Mono", raw)
