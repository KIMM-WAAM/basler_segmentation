import pickle
import random
from pathlib import Path

import albumentations as A
import numpy as np
import pytest

from augment import Brightness, Gamma, albumentations_list, apply_gamma
from common import load_yaml
from epoch_overlay import _augment


def test_gamma_pixels_and_independent_brightness():
    img = np.array([0, 64, 128, 255], dtype=np.uint8).reshape(2, 2, 1)
    np.testing.assert_array_equal(apply_gamma(img, 1), img)
    assert apply_gamma(img, 0.5)[0, 1, 0] > 64
    assert apply_gamma(img, 2)[0, 1, 0] < 64
    for gp, bp in [(0, 1), (1, 0), (1, 1), (0, 0)]:
        cfg = {"gamma": {"p": gp, "gain": [2, 2]},
               "brightness": {"p": bp, "gain": [0.5, 0.5]}}
        albumentations_list(cfg)
        assert set(cfg) == {"augmentations"}
        assert [type(t) for t in cfg["augmentations"]] == [Gamma, Brightness]
        pipeline = pickle.loads(pickle.dumps(A.Compose(cfg["augmentations"])))
        mask = np.ones((2, 2), dtype=np.uint8)
        result = pipeline(image=img, mask=mask)
        expected = apply_gamma(img, 2) if gp else img
        if bp:
            expected = (expected * 0.5).astype(np.uint8)
        np.testing.assert_array_equal(result["image"], expected)
        np.testing.assert_array_equal(result["mask"], mask)


def test_log_uniform_and_invalid_ranges():
    transform = Gamma()
    transform.set_random_seed(123)
    samples = np.array([transform.get_params()["g"] for _ in range(10000)])
    assert samples.min() >= 0.5 and samples.max() <= 2
    assert abs(np.log(samples).mean()) < 0.02
    for gain in [(0, 2), (2, 1), (float("nan"), 2)]:
        with pytest.raises(ValueError):
            Gamma(gain=gain)


def test_config_inheritance_and_overlay():
    for path in Path("configs").glob("train.*.yaml"):
        cfg = load_yaml(path)["train"]
        assert cfg["gamma"] == {"p": 0.8, "gain": [0.5, 2.0]}
        albumentations_list(cfg)
        assert isinstance(cfg["augmentations"][0], Gamma)
        assert "gamma" not in cfg
    hyp = dict.fromkeys(["degrees", "scale", "translate", "shear", "perspective", "hsv_v"], 0)
    hyp.update(imgsz=8, gamma={"p": 1, "gain": [2, 2]})
    img = np.full((8, 8, 3), 128, dtype=np.uint8)
    out, _, text = _augment(img, [], hyp, random.Random(0), False)
    np.testing.assert_array_equal(out, apply_gamma(img, 2))
    assert "gamma 2.00" in text
