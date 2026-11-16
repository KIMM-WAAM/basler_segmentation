"""train.py export.weights — 내보낼 체크포인트 선택 (best / last)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import Trainer  # noqa: E402


def make(tmp_path, weights_line: str) -> Trainer:
    cfg = tmp_path / "train.t.yaml"
    cfg.write_text("dataset: kimm\nstages: {export: true}\ntrain: {data: d.yaml, project: runs/t, name: t}\n"
                   f"export:\n  format: onnx\n{weights_line}", encoding="utf-8")
    return Trainer(str(cfg))


def test_default_is_best_only(tmp_path):
    assert make(tmp_path, "").export_weights() == ["best"]


def test_list_keeps_order_and_drops_duplicates(tmp_path):
    assert make(tmp_path, "  weights: [best, last, best]\n").export_weights() == ["best", "last"]
    assert make(tmp_path, "  weights: last\n").export_weights() == ["last"]


def test_unknown_name_fails(tmp_path):
    with pytest.raises(SystemExit):
        make(tmp_path, "  weights: [best, epoch50]\n").export_weights()


def test_run_weights_points_at_this_run(tmp_path):
    t = make(tmp_path, "")
    t._save_dir = tmp_path / "run"
    assert t.run_weights("last") == tmp_path / "run" / "weights" / "last.pt"
    assert t.best_weights() == tmp_path / "run" / "weights" / "best.pt"


def test_base_configs_export_best_and_last():
    for name in ("base/train.kimm.yaml", "base/train.bestfa.yaml"):
        assert Trainer(name).export_weights() == ["best", "last"]
