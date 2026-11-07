"""학습에만 원본 좌표 랜덤 크롭을 적용하는 Ultralytics 데이터셋/트레이너."""
from __future__ import annotations

import math
import random
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.utils.ops import segments2boxes
from ultralytics.utils.patches import imread
from ultralytics.utils.torch_utils import unwrap_model

from preprocess import crop_rect, polygon_to_obb, transform_label
from source_crop import read_index, validate_options


class SourceCropDataset(YOLODataset):
    def __init__(self, *args, crop_entries: dict, crop_root: Path, crop_options: dict, **kwargs):
        self.crop_entries = crop_entries
        self.crop_root = Path(crop_root)
        self.crop_options = validate_options(crop_options)
        hyp = kwargs["hyp"]
        if any(getattr(hyp, key, 0) for key in ("mosaic", "mixup", "copy_paste", "cutmix")):
            raise ValueError("원본 크롭은 mosaic/mixup/copy_paste/cutmix를 끄고 사용하세요")
        super().__init__(*args, **kwargs)
        for name in self.im_files:
            entry = self.crop_entries.get(Path(name).name)
            if entry is None:
                raise ValueError(f"원본 크롭 인덱스에 이미지가 없습니다: {name}")
            crop = entry["crop"]
            if not crop.get("enabled") or (crop.get("auto_crop") or {}).get("enabled"):
                raise ValueError(f"랜덤 크롭은 enabled 고정 크롭이 필요합니다: {name}")
            if entry.get("resize"):
                raise ValueError("랜덤 원본 크롭은 preprocess resize: null을 사용하세요")

    def get_image_and_label(self, index: int) -> dict:
        if not self.augment:
            return super().get_image_and_label(index)
        name = self.im_files[index]
        entry = self.crop_entries[Path(name).name]
        image = imread(str((self.crop_root / entry["image"]).resolve()), flags=self.cv2_flag)
        if image is None:
            raise FileNotFoundError(entry["image"])
        src_h, src_w = image.shape[:2]
        crop = {**entry["crop"],
                "center_x": min(100.0, max(0.0, float(entry["crop"].get("center_x", 50))
                                           + random.uniform(*self.crop_options["offset_x"]))),
                "center_y": min(100.0, max(0.0, float(entry["crop"].get("center_y", 50))
                                           + random.uniform(*self.crop_options["offset_y"])))}
        rect = crop_rect(src_w, src_h, crop)
        x, y, width, height = rect
        lines = [line for raw in entry["labels"]
                 if (line := transform_label(raw, src_w, src_h, rect, crop.get("min_area", 0.1)))]
        if self.use_obb:
            lines = [result[0] for line in lines if (result := polygon_to_obb(line, width, height))]
        classes, segments = [], []
        for line in lines:
            values = line.split()
            classes.append([float(values[0])])
            segments.append(np.asarray(values[1:], dtype=np.float32).reshape(-1, 2))
        bboxes = segments2boxes(segments) if segments else np.zeros((0, 4), dtype=np.float32)
        image = image[y:y + height, x:x + width].copy()
        ratio = self.imgsz / max(height, width)
        size = (min(math.ceil(width * ratio), self.imgsz), min(math.ceil(height * ratio), self.imgsz))
        if size != (width, height):
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        if image.ndim == 2:
            image = image[..., None]
        # YOLO 표준 변환에 전달하는 normalized xywh + polygon.
        label = {"im_file": name, "img": image, "ori_shape": (height, width),
                 "resized_shape": image.shape[:2], "ratio_pad": (size[1] / height, size[0] / width),
                 "cls": np.asarray(classes, dtype=np.float32).reshape(-1, 1),
                 "bboxes": bboxes, "segments": segments, "normalized": True, "bbox_format": "xywh"}
        return self.update_labels_info(label)


class SourceCropTrainerMixin:
    def __init__(self, *args, crop_root: Path, crop_options: dict, **kwargs):
        self.crop_root = Path(crop_root)
        self.crop_options = crop_options
        self.crop_entries = read_index(self.crop_root)
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        args = self.args
        if args.rect or args.cache or args.classes is not None or args.single_cls:
            raise ValueError("원본 크롭 학습은 rect/cache/classes/single_cls 옵션을 지원하지 않습니다")
        return SourceCropDataset(
            img_path=img_path, imgsz=args.imgsz, batch_size=batch, augment=True, hyp=args,
            rect=False, cache=None, single_cls=False, stride=max(int(unwrap_model(self.model).stride.max()), 32),
            pad=0.0, prefix="train: ", task=args.task, classes=None, data=self.data, fraction=args.fraction,
            crop_root=self.crop_root, crop_options=self.crop_options, crop_entries=self.crop_entries,
        )


class SourceCropSegTrainer(SourceCropTrainerMixin, SegmentationTrainer):
    pass


class SourceCropOBBTrainer(SourceCropTrainerMixin, OBBTrainer):
    pass


def trainer_factory(task: str, root: Path, options: dict):
    classes = {"segment": SourceCropSegTrainer, "obb": SourceCropOBBTrainer}
    if task not in classes:
        raise ValueError(f"원본 크롭을 지원하지 않는 task: {task}")
    return partial(classes[task], crop_root=root, crop_options=options)
