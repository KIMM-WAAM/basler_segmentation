"""촬영 환경별 데이터·설정 경로. 상대경로는 training/ 기준."""
from pathlib import Path

from common import CONFIG_DIR, HERE, load_yaml

DATASETS = ("kimm", "bestfa")


def dataset_root(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"알 수 없는 촬영 환경: {dataset!r} (kimm / bestfa)")
    return HERE.parent / "datasets" / dataset


def preprocess_config(dataset: str) -> Path:
    dataset_root(dataset)  # 경로로 쓰기 전에 이름 검증
    return CONFIG_DIR / f"preprocess.{dataset}.yaml"


def check_source_dataset(source: Path, dataset: str) -> None:
    """프로젝트 안의 입력이 선택한 환경과 다르면 다른 크롭을 적용하기 전에 중단."""
    resolved = source.resolve()
    for candidate in DATASETS:
        if resolved.is_relative_to(dataset_root(candidate).resolve()) and candidate != dataset:
            raise ValueError(f"입력은 {candidate} 데이터입니다. --dataset {candidate} 를 사용하세요")


def default_crop(dataset: str = "kimm") -> dict | None:
    cfg = load_yaml(preprocess_config(dataset))
    for source in cfg.get("sources", []):
        if source["name"] == dataset:
            crop = source.get("crop") or {}
            if crop.get("enabled") and not (crop.get("auto_crop") or {}).get("enabled"):
                return crop
    return None
