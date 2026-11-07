"""촬영 환경별 데이터·설정 경로 + 세션 폴더 규약. 상대경로는 training/ 기준."""
import math
import re
from pathlib import Path

import yaml

from common import CONFIG_DIR, HERE, load_yaml

DATASETS = ("kimm", "bestfa")

# 세션 폴더 규약:  datasets/<환경>/raw/<YYMMDD_HHMMSS>/  (Mono/ · labels/ · labels_seg/ ·
# data.yaml · session.yaml). 환경은 상위 폴더가 말해 주므로 이름에 접두어를 붙이지 않고,
# 장면 메모 같은 명세는 이름 대신 session.yaml 에 적는다.
# 숫자로 시작하지 않는 폴더(_removed_* 등)는 SESSION_GLOB 에 안 걸려 라벨러·전처리에서 빠진다.
SESSION_GLOB = "[0-9]*"
SESSION_RE = re.compile(r"^(\d{6}_\d{6})(?:_(.+))?$")   # 예전 이름의 _메모 꼬리까지 받는다
SESSION_META = "session.yaml"


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


def session_crop(session: Path) -> dict:
    """세션별 크롭 필수. 공통 config나 묵시적 원본 사용으로 대체하지 않는다."""
    path = Path(session) / "crop.yaml"
    if not path.is_file():
        raise ValueError(f"세션 크롭 설정이 없습니다: {path}. 라벨러에서 크롭을 저장하세요 "
                         "(크롭하지 않을 때도 사용을 끄고 저장).")
    crop = load_yaml(path)
    if not isinstance(crop, dict) or not isinstance(crop.get("enabled"), bool):
        raise ValueError(f"{path}: enabled는 true 또는 false로 지정하세요")
    if crop["enabled"] and not (crop.get("auto_crop") or {}).get("enabled"):
        for key in ("width", "height", "center_x", "center_y"):
            value = crop.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{path}: {key}에 유한한 숫자가 필요합니다")
            if value < 0 or (key.startswith("center_") and value > 100):
                raise ValueError(f"{path}: {key} 범위가 잘못되었습니다")
    return crop


def default_crop(dataset: str = "kimm", input_path: Path | None = None) -> dict | None:
    """자동 추론 크롭: 입력 세션의 명시적 설정만 사용한다."""
    dataset_root(dataset)
    if input_path is None:
        raise ValueError("자동 크롭은 입력 세션 경로가 필요합니다")
    current = Path(input_path).resolve()
    current = current if current.is_dir() else current.parent
    session = current.parent if current.name in ("Mono", "images") else current
    crop = session_crop(session)
    if crop.get("enabled") and (crop.get("auto_crop") or {}).get("enabled"):
        raise ValueError(f"{session}/crop.yaml: 라벨 기반 auto_crop은 추론에 사용할 수 없습니다")
    return crop if crop["enabled"] else None


def parse_session_name(name: str) -> tuple[str, str | None]:
    """
    세션 폴더 이름 → (id, 메모). 예전 이름(`kimm_261019_155904_격자실험3`)의 환경 접두어도
    벗긴다 — 이름 정리(tools/rename_sessions.py)와 waiting → raw(tools/waiting_to_raw.py)가 함께 쓴다.
    """
    for prefix in DATASETS:
        name = name.removeprefix(f"{prefix}_")
    m = SESSION_RE.match(name)
    if not m:
        raise ValueError(f"세션 이름은 YYMMDD_HHMMSS[_메모] 형식이어야 합니다: {name}")
    return m.group(1), m.group(2)


def read_session_meta(session: Path) -> dict:
    """<세션>/session.yaml (없으면 빈 dict). 모든 항목이 선택이다."""
    p = Path(session) / SESSION_META
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


# session.yaml 틀 — id · env 만 채워 두고 나머지는 사람이 필요할 때 적는다 (전부 선택).
SESSION_META_FIELDS = (
    ("scene", "장면 메모 (예: 격자실험3, hollow 정사각형1)"),
    ("tags", "검색용 태그 목록 (예: [격자])"),
    ("wire_count", "화면 속 와이어 수"),
    ("notes", "그 밖의 메모 (조명 · 재료 · 문제 프레임 등)"),
)


# 폴더 이름 메모 안의 미리 정의된 항목 — `_` 로 나눈 조각이 패턴에 맞으면 session.yaml 의 그 필드로,
# 나머지 조각은 scene 으로 (tools/waiting_to_raw.py). 항목을 늘리려면 여기에 한 줄.
#   (패턴, 필드, 값 변환, 설명)
SESSION_NAME_ITEMS = (
    (re.compile(r"^layer(\d+)$", re.I), "layer", int, "_layer3 → layer: 3  (적층 층 번호)"),
)


def split_memo(memo: str | None) -> dict:
    """폴더 이름 메모 → session.yaml 필드. 예: 'layer3_조명실험' → {'layer': 3, 'scene': '조명실험'}."""
    out: dict = {}
    rest = []
    for part in (memo or "").split("_"):
        if not part:
            continue
        for pat, key, conv, _ in SESSION_NAME_ITEMS:
            m = pat.match(part)
            if m and key not in out:
                out[key] = conv(m.group(1))
                break
        else:
            rest.append(part)
    if rest:
        out["scene"] = "_".join(rest)
    return out


def ensure_session_meta(session: Path, env: str, **fields) -> Path:
    """
    session.yaml 이 없으면 틀을 만든다 — id · env 와 주어진 항목(예: 예전 이름의 메모 → scene)만
    값을 채우고, 나머지 항목은 빈 칸 + 설명 주석. 틀에 없는 항목(예: layer)은 env 뒤에 적는다.
    이미 있으면 사람이 적은 명세라 건드리지 않는다.
    """
    p = Path(session) / SESSION_META
    if p.exists():
        return p

    def value(v) -> str:
        return yaml.safe_dump(v, allow_unicode=True, default_flow_style=True).strip().removesuffix("...").strip()

    lines = [f"id: {value(Path(session).name)}", f"env: {value(env)}"]
    known = {k for k, _ in SESSION_META_FIELDS}
    lines += [f"{k}: {value(v)}" for k, v in fields.items() if k not in known and v not in (None, "", [])]
    for key, hint in SESSION_META_FIELDS:
        v = fields.get(key)
        lines.append(f"{key}: {value(v)}" if v not in (None, "", []) else f"{key + ':':<14}# {hint}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
