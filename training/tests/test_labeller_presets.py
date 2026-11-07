"""한 서버에서 프리셋을 바꿔도 이미지·라벨·크롭·기준 레이어가 섞이지 않는다."""
import io
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import yaml
from PIL import Image, ImageDraw

from common import HERE

sys.path.insert(0, str(HERE / "labeller"))
import server as lab  # noqa: E402

BOX = [[0.1, 0.1], [0.2, 0.1], [0.2, 0.8], [0.1, 0.8]]


@pytest.fixture
def presets(tmp_path, monkeypatch):
    configs = {}
    for name, count in (("kimm", 1), ("bestfa", 2)):
        root = tmp_path / name
        session = root / "raw" / f"{name}_session"
        (session / "Mono").mkdir(parents=True)
        size = (100 * count, 150)
        Image.new("RGB", size, "gray").save(session / "Mono/frame.png")
        refs = root / "reference"
        refs.mkdir()
        image = Image.new("RGBA", size, "white")
        draw = ImageDraw.Draw(image)
        for index in range(count):
            draw.rectangle((20 + index * 100, 20, 30 + index * 100, 120), outline="red", width=2)
        image.save(refs / "reference_wire.png")
        image.save(refs / "reference_outline.png")
        cfg = {"dataset": name, "wire_count": count, "names": {0: "wire"},
               "reference": {"directory": str(refs), "wire_mask": "red", "white_background": True},
               "sources": [{"name": name, "path": str(root / f"raw/{name}_*"),
                            "crop": {"enabled": False, "width": 0, "height": 0,
                                     "center_x": 50, "center_y": 50}}]}
        configs[name] = root / f"preprocess.{name}.yaml"
        configs[name].write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(lab, "PREPROCESS_CFG", configs["kimm"])
    monkeypatch.setattr(lab, "preprocess_config", configs.__getitem__)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lab.make_handler())
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def request(path, data=None, binary=False):
        req = Request(base + path, data=None if data is None else json.dumps(data).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req) as response:
            return response.read() if binary else json.load(response)

    try:
        yield request, configs
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_presets_keep_reads_and_writes_separate(presets):
    request, configs = presets
    menu = request("/api/presets")
    assert menu["default"] == "kimm"
    assert [p["id"] for p in menu["presets"]] == ["kimm", "bestfa"]
    assert request("/api/config")["source"] == "kimm"
    for name, count in (("bestfa", 2), ("kimm", 1), ("bestfa", 2)):
        cfg = request(f"/api/config?dataset={name}")
        assert cfg["source"] == name and cfg["wire_count"] == count
        assert len(cfg["templates"]) == count
        assert request(f"/api/images?dataset={name}")["images"][0]["dir"] == f"{name}_session"
        for path in ("/image/frame.png", "/ref/outline.png"):
            with Image.open(io.BytesIO(request(f"{path}?dataset={name}", binary=True))) as image:
                assert image.size == (100 * count, 150)

    request("/api/label?dataset=bestfa", {"name": "frame.png", "instances": [BOX, BOX]})
    assert request("/api/label?dataset=kimm&name=frame.png")["instances"] == []
    request("/api/label?dataset=kimm", {"name": "frame.png", "instances": [BOX]})
    for name, count in (("kimm", 1), ("bestfa", 2)):
        assert len(request(f"/api/label?dataset={name}&name=frame.png")["instances"]) == count
        edge = request(f"/api/edge?dataset={name}&name=frame.png&method=obb")
        request(f"/api/seg?dataset={name}", {"items": [{"name": "frame.png", "polygons": edge["polygons"]}]})
        seg = configs[name].parent / f"raw/{name}_session/labels_seg/frame.txt"
        assert len(seg.read_text().splitlines()) == count

    before = configs["kimm"].read_bytes()
    crop = {"enabled": True, "width": 160, "height": 140, "center_x": 50, "center_y": 50}
    request("/api/crop?dataset=bestfa", crop)
    assert request("/api/config?dataset=bestfa")["crop"]["width"] == 160
    assert configs["kimm"].read_bytes() == before
    for data in (None, {"name": "frame.png", "instances": []}):
        with pytest.raises(HTTPError) as exc:
            request("/api/label?dataset=unknown&name=frame.png", data)
        assert exc.value.code == 400
    assert len(request("/api/label?dataset=bestfa&name=frame.png")["instances"]) == 2


def test_unavailable_preset_does_not_block_menu_or_other_preset(presets):
    request, configs = presets
    contents = configs["kimm"].read_bytes()
    configs["kimm"].unlink()
    assert b'id="preset"' in request("/", binary=True)
    assert len(request("/api/presets")["presets"]) == 2
    with pytest.raises(HTTPError) as exc:
        request("/api/config?dataset=kimm")
    assert exc.value.code == 400
    assert request("/api/config?dataset=bestfa")["wire_count"] == 2
    # 실패한 프리셋은 캐시하지 않아 파일을 준비한 뒤 새로고침하면 다시 읽는다.
    configs["kimm"].write_bytes(contents)
    assert request("/api/config?dataset=kimm")["wire_count"] == 1
