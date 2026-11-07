"""Mono 원본에서 여러 와이어의 OBB·seg를 손실 없이 편집·전처리한다."""
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import yaml
from PIL import Image, ImageDraw

from common import HERE, raw_image_dir
from preprocess import Preprocessor

sys.path.insert(0, str(HERE / "labeller"))
import reference  # noqa: E402
import server as lab  # noqa: E402

BOXES = [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.7], [0.1, 0.7]],
         [[0.6, 0.1], [0.7, 0.1], [0.7, 0.8], [0.6, 0.8]]]


@pytest.fixture
def source(tmp_path):
    for index in range(3):
        session = tmp_path / "raw" / f"bestfa_{index}"
        (session / "Mono").mkdir(parents=True)
        for frame in range(2):
            Image.new("RGB", (100, 100), "white").save(session / "Mono" / f"{index}_{frame}.png")
    cfg = {"dataset": "bestfa", "wire_count": 2, "names": {0: "wire"}, "task": "seg",
           "out": str(tmp_path / "processed/seg"), "sources": [
               {"name": "bestfa", "path": str(tmp_path / "raw/bestfa_*"),
                "names": {0: "wire"}, "class_map": {"wire": "wire"}, "labels": "labels_seg",
                "type": "seg", "group_by": "session", "val_sessions": ["bestfa_1"],
                "test_sessions": ["bestfa_2"], "crop": {"enabled": False}}]}
    cfg_path = tmp_path / "preprocess.bestfa.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return lab.Source("bestfa", cfg_path)


def test_red_reference_separates_wires(tmp_path):
    path = tmp_path / "reference_wire.png"
    image = Image.new("RGBA", (200, 150), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 40, 120), outline="red", width=2)
    draw.rectangle((130, 10, 142, 130), outline="red", width=2)
    draw.line((0, 5, 199, 5), fill="blue", width=3)
    image.save(path)
    templates = reference.red_wire_templates(path)
    assert len(templates) == 2
    assert templates[0]["pivot"][0] < templates[1]["pivot"][0]
    assert templates[0]["thickness"] == pytest.approx(10)
    assert templates[1]["length"] == pytest.approx(120)


def test_multiwire_http_roundtrip_and_copy(source):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.make_handler(source))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def request(path, data=None):
        req = urllib.request.Request(base + path, data=json.dumps(data).encode() if data else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())

    try:
        name = "0_0.png"
        assert request("/api/config")["wire_count"] == 2
        request("/api/label", {"name": name, "instances": BOXES})
        assert request(f"/api/label?name={name}")["instances"] == BOXES
        assert len(source.label_path(name).read_text().splitlines()) == 2
        result = request(f"/api/edge?name={name}&method=obb")
        assert result["status"] == "ok" and len(result["polygons"]) == 2
        request("/api/seg", {"items": [{"name": name, "polygons": result["polygons"]}]})
        assert len(source.seg_path(name).read_text().splitlines()) == 2
        copied = request("/api/copy", {"name": name, "instances": BOXES, "overwrite": False})
        assert copied["written"] == 1
        assert source.read_labels("0_1.png") == BOXES
        assert not source.label_path("1_0.png").exists()  # 다른 세션은 변경하지 않음
        with urllib.request.urlopen(base + "/image/0_0.png") as response:
            assert response.status == 200
        # 두 번째 와이어가 없는 결과를 덮어써도 기존 두 줄 seg는 보존한다.
        before = source.seg_path(name).read_bytes()
        with pytest.raises(ValueError, match="다릅니다"):
            source.write_seg_polygons(name, result["polygons"][:1])
        assert source.seg_path(name).read_bytes() == before
        invalid = [BOXES[0], [[float("nan"), 0]] * 4]
        with pytest.raises(ValueError, match="유한"):
            source.write_labels(name, invalid)
        assert source.read_labels(name) == BOXES
        assert source.seg_path(name).read_bytes() == before
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_preprocess_keeps_both_wires_from_mono(source):
    for name in source.images():
        source.write_labels(name, BOXES)
        source.write_seg_polygons(name, [[[x * 100, y * 100] for x, y in p] for p in BOXES])
    processor = Preprocessor(source.config_path)
    processor.run()
    labels = list((processor.out / "bestfa/labels").rglob("*.txt"))
    assert len(labels) == 6
    assert all(len(p.read_text().splitlines()) == 2 for p in labels)
    for session in source.dirs:
        assert raw_image_dir(session).name == "Mono"
        assert yaml.safe_load((session / "data.yaml").read_text())["train"] == "Mono"


def test_ambiguous_raw_directory_is_rejected(tmp_path):
    (tmp_path / "Mono").mkdir()
    (tmp_path / "images").mkdir()
    with pytest.raises(ValueError, match="함께"):
        raw_image_dir(tmp_path)


def test_unlabeled_mono_session_in_fixed_split(source):
    for name in source.images():
        source.write_seg_polygons(name, [[[x * 100, y * 100] for x, y in p] for p in BOXES])
    session = source.dirs[0].parent / "bestfa_unlabeled"
    (session / "Mono").mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(session / "Mono" / "unlabeled.png")
    cfg = yaml.safe_load(source.config_path.read_text())
    cfg["sources"][0]["val_sessions"].append(session.name)
    source.config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    processor = Preprocessor(source.config_path)
    processor.run()
    assert len(list((processor.out / "bestfa/labels/val").glob("*.txt"))) == 2
