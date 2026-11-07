# training

와이어 seg / obb 모델(yolo26s) 학습 파이프라인 — 라벨링부터 앱 배포까지. uv 프로젝트 하나.

```
labeller/ → preprocess.py → train.py (single) → eval.py → train.py (*_all) → archive.py → weights/deploy.yaml
 라벨링       전처리          학습 · best 선택    test 평가   배포용 전체 학습    보관          앱 모델 선택
               processed_obb/
```

## 구성

| 경로 | 역할 |
|---|---|
| `labeller/` | 브라우저 라벨러 (`server.py` · `index.html` · 끝 경계 `edge.py`) |
| `preprocess.py` | 원본 → 학습 데이터셋 (크롭 · 클래스 통일 · 세션 단위 분할 · seg→obb 변환) |
| `train.py` | 학습 · 증강 미리보기 · ONNX export · 앱 배포 |
| `eval.py` | checkpoint 를 지정 split 에서 평가 + 정답/추론 겹친 그림 |
| `infer.py` | 사진 한 장 → 예측 그림, 세션 폴더 → 동영상 |
| `epoch_overlay.py` | 학습 중 매 val 마다 원본 4장 + 증강 12장 추론 격자 |
| `render.py` | 추론 그림 (bbox 없이 옅은 마스크·윤곽, 왼쪽 아래 클래스·점수) — infer · overlay 공용 |
| `augment.py` | Ultralytics 키로 못 내는 증강 — 밝기 배율 · 블러 |
| `common.py` | config 로드(+ `base:` 상속) · 경로 해석 · 로그 |
| `dataset_paths.py` | 환경별 경로 · 세션 폴더 규약(`YYMMDD_HHMMSS`) · `session.yaml` |
| `provenance.py` | 출처 기록 — 전처리 manifest · run_info · 재현용 config · .onnx 메타데이터 |
| `archive.py` | 학습 실행 → 저장소 루트 `weights/` 보관 (기록째로) |
| `tools/make_reference_cutout.py` | 기구부 기준 이미지 → 누끼·윤곽선 (앱·라벨러 오버레이) |
| `tools/waiting_to_raw.py` | `datasets/<환경>/waiting/` 의 세션 → `raw/` (이름 정리 · Mono/ · 파일명 접두어 · data.yaml · session.yaml) |
| `tools/rename_sessions.py` | 예전 세션 이름(`kimm_…_메모`) → `YYMMDD_HHMMSS` + session.yaml |
| `configs/` | `preprocess.kimm.yaml`, `train.*.yaml` (+ 공통 `base/train.yaml`), `eval.*.yaml` |

## 설치 · 실행

환경은 uv 하나 — `pyproject.toml` + `uv.lock`, 파이썬은 `.python-version` 의 3.12.
학습 머신은 Blackwell(B200 · RTX 5090)이라 **CUDA 12.8 빌드 torch** 를 쓴다 (pyproject 가 인덱스 고정).

```powershell
cd training
uv sync --extra train --extra export        # torch(cu128) · ultralytics · ONNX + dev(ruff)
uv run labeller/server.py                   # 라벨러 → http://localhost:8765
uv run preprocess.py                        # → ../datasets/kimm/processed_seg/
uv run preprocess.py --task obb             # → ../datasets/kimm/processed_obb/ (OBB 학습할 때만)
uv run train.py --config train.kimm.single.yaml
uv run eval.py --config eval.single.yaml
uv run infer.py --dataset kimm --weights runs/kimm/segment/yolo26s-seg-finetune/weights/best.pt --source ../datasets/kimm/raw/261019_161818
```

> **`uv sync` 는 항상 `--extra train --extra export` 와 함께.** extra 없이 돌리면 uv 가 환경을
> 정확히 맞추느라 torch 등을 지운다 (라벨링만 하는 머신이면 그게 의도).

VS Code: `Ctrl+Shift+P → Tasks: Run Task`에서 `run labeller`, `train`, `inference — picture`,
`inference — movie`, `tensorboard`를 선택한다 (`.vscode/tasks.json`).

## 라벨링 (`labeller/`)

저장소 루트와 `training/` 양쪽에서 아래 명령을 실행할 수 있다. 루트의 실행기는
`training/`의 uv 환경을 사용한다. 실행 후 브라우저에서 `http://127.0.0.1:8765`를 연다.

```powershell
uv run labeller/server.py
```

화면 상단의 **프리셋**에서 **KIMM / BestFA**를 고른다. 이미지 목록·크롭·기준 이미지와 와이어
개수가 함께 바뀐다. 저장하지 않은 라벨·크롭 변경이 있으면 전환 전에 확인하며, 저장·경계 계산
중에는 완료 후 전환할 수 있다. 새로고침해도 주소에 담긴 선택이 유지된다.
`--dataset`은 생략해도 되며, 지정하면 처음 표시할 프리셋만 바뀐다 (기본 KIMM).

VS Code의 **run labeller** task는 바로 서버를 실행하고 프리셋은 같은 화면에서 고른다.
같은 포트의 이전 서버가 켜져 있으면
그 터미널에서 `Ctrl+C`로 종료한 뒤 다시 실행하거나 `--port 8766`으로 다른 포트를 쓴다.

카메라가 기구에 붙어(eye-in-hand) 튜브·와이어가 화면에서 늘 같은 자리라, 와이어 박스를 그리지 않고
기준 자세에서 **우측 상단 꼭짓점(튜브 출구)을 축으로 옮기고 돌려** 맞춘다.

- KIMM은 와이어 1개, BestFA는 **와이어 2개**. 선택 메뉴·숫자 `1`/`2`·박스 클릭으로 편집할 와이어를 고른다.
  OBB 저장·복사는 두 박스를 함께 처리하고, 끝 경계도 두 와이어를 모두 계산한 후 seg로 저장한다.
- 세션별 보기 (↑↓ 세션, ←→ 이미지), 기구부 오버레이, 크롭 창 조절 → 선택한 세션의 `crop.yaml`에 저장
- 일괄 덮어쓰기: 세션 전체 / 이 번호부터 끝까지 / 이 번호부터 N번까지
- 끝 경계: **Compute edge 1** (OBB 윗변·옆변 + 영상에서 찾은 끝 곡선) 또는 **OBB as edge** →
  `labels_seg/` 에 와이어 폴리곤 저장. OBB 를 바꾼 장은 seg 가 지워지고 edge 1 로 다시 계산
- 이미지 삭제(<kbd>Delete</kbd>, 묻지 않고 바로) → `datasets/<환경>/raw/_removed_manual/` 로 이동 (되돌릴 수 있음).
  라벨만 지우기는 <kbd>Shift</kbd>+<kbd>Delete</kbd>
- **빈(검은) 이미지 전부 삭제** — 이 환경의 모든 세션에서 가장 밝은 픽셀이 10 이하(`BLACK_MAX`)인 빈 프레임을
  찾아 세션별 장수를 보여 주고, 확인하면 `raw/_removed_black/<세션>/` 으로 옮긴다 (`manifest.csv` 에 기록, 되돌릴 수 있음)

산출물 (세션 폴더마다): `labels/` = 손으로 잡은 OBB, `labels_seg/` = 학습에 쓰는 와이어 폴리곤.
라벨러는 세션을 열 때 `session.yaml` 이 없으면 틀(id · env · wire_count)을 만들어 둔다.

## 데이터 · 전처리 (`preprocess.py`, `configs/preprocess.kimm.yaml`)

```text
datasets/<환경>/                환경 = kimm 또는 bestfa
  raw/<YYMMDD_HHMMSS>/           Mono/ + labels/ + labels_seg/ + data.yaml + session.yaml
  raw/_removed_*/               빼 둔 프레임 — 숫자로 시작하지 않아 글롭([0-9]*)에서 빠짐
  reference/                    그 환경의 기구부·와이어 기준 이미지
  processed_seg/                preprocess.py 산출물 (config 하나 = 소스 하나 = 이 폴더 하나)
    images/{train,val,test}/
    labels/{train,val,test}/    전처리한 이미지 좌표의 YOLO seg 라벨
    preview/                    라벨을 그린 산출물 몇 장 (크롭 확인용)
    data.yaml                   train/val/test 분할 그대로 (train.single · eval)
    data.all.yaml               배포용 전체 데이터 목록 (train.*_all.yaml)
    manifest.yaml               세션별 split · 장수 · 지문, 전체 지문, 전처리 설정
    manifest.files.tsv          파일 하나하나: split · 세션 · 산출 이름 · 원본 sha256 · 라벨 sha256
  processed_obb/                같은 소스를 --task obb 로 변환한 결과 (구조 동일)
```

원본 `labels/`는 OBB, `labels_seg/`는 분할 폴리곤이다. `processed_*/`에는 변환된 이미지와
그에 맞는 라벨을 함께 저장한다. 학습 입력은 seg/obb 모두 YOLO의 `images/`·`labels/` 규약을 쓴다.

### 세션 이름 · 명세

세션 폴더 이름은 `YYMMDD_HHMMSS` 하나로 통일한다 — 환경은 상위 폴더(`datasets/<환경>/`)가 말하므로
접두어를 붙이지 않고, 장면 메모 같은 명세는 이름 대신 `session.yaml` 에 적는다 (전부 선택):

```yaml
id: '261019_155904'     # 따옴표 필수 — 없으면 yaml 이 숫자 261019155904 로 읽는다
env: kimm
scene: 격자실험3
tags: [격자]
wire_count: 1
notes: ""
```

`scene` · `tags` · `layer` 는 manifest 의 세션 항목에도 옮겨 적혀, 모델 기록에서 어떤 장면으로 학습했는지 보인다.
config 의 `val_sessions` / `test_sessions` 에 세션을 적을 때도 따옴표로 감싼다 (`- '261019_160703'`).
예전 이름(`kimm_YYMMDD_HHMMSS[_메모]`)은 `uv run tools/rename_sessions.py --dataset kimm` 으로 목록을
확인하고 `--apply` 로 바꾼다 (`_removed_*` 안의 세션 · csv 까지).

### 받아 온 세션 → raw (`tools/waiting_to_raw.py`)

외부에서 받은 세션 폴더는 `datasets/<환경>/waiting/<YYMMDD_HHMMSS[_메모]>/` 에 넣고:

```powershell
uv run tools/waiting_to_raw.py                    # 사용법 · 미리 정의된 이름 항목 · 환경별 대기 목록
uv run tools/waiting_to_raw.py --dataset bestfa   # 세션마다 계획을 보여 주고 y/n/e(메모 고치기)/a/q
uv run tools/waiting_to_raw.py --dataset bestfa --dry-run   # 계획만
```

- 폴더 이름 → `raw/<YYMMDD_HHMMSS>/`. 메모를 `_` 로 나눠 미리 정의된 항목(`layer3` → `layer: 3`)은 그 필드로,
  나머지는 `scene` 으로 session.yaml 에. 항목 추가는 `dataset_paths.SESSION_NAME_ITEMS` 에 한 줄.
- 이미지 폴더 `Mono/` · `Image/` · `images/` → `Mono/`. 파일명에 세션 id 가 없으면 `<id>__` 를 붙인다
  (BestFA 의 `00_02_23_544.jpg` 처럼 시각만으로 된 이름은 세션끼리 겹쳐 라벨러 · 전처리가 멈춤).
  `labels/` · `labels_seg/` 가 있으면 같은 규칙으로 함께 옮긴다.
- data.yaml · session.yaml(wire_count 는 환경 설정에서)을 만든다. 같은 id 가 raw 에 있거나 파일명이 겹치면
  그 세션은 건너뛰고 이유를 보여 준다. 옮기기(이동)라 waiting 에서는 사라진다.
- session.yaml 끝에 옮긴 기록을 자동으로 남긴다 (사람이 적는 위쪽 명세와 구분):

  ```yaml
  imported:
    at: '2026-11-07T15:05:11+09:00'    # 옮긴 시각
    from: waiting/261026_145202_layer3 # 원래 폴더 이름
    image_dir: Image                   # 원래 이미지 폴더 (→ Mono/)
    images: 3144
    renamed: 3144                      # 파일명 앞에 <id>__ 를 붙인 장수
  ```

### 촬영 환경 선택

| 작업 | KIMM | BestFA |
|---|---|---|
| 전처리 설정 | `configs/preprocess.kimm.yaml` | `configs/preprocess.bestfa.yaml` |
| 라벨러 (`uv run labeller/server.py`) | GUI 프리셋 **KIMM** | GUI 프리셋 **BestFA** |
| seg 전처리 | `uv run preprocess.py --dataset kimm` | `uv run preprocess.py --dataset bestfa` |
| obb 전처리 | 위 명령에 `--task obb` 추가 | 위 명령에 `--task obb` 추가 |
| 측정 학습 | `uv run train.py --config train.kimm.single.yaml` | `uv run train.py --config train.bestfa.single.yaml` |
| 평가 | `eval.single.yaml` (val) / `eval.final.yaml` (test) | `eval.bestfa.single.yaml` (val만) |

라벨러는 GUI에서 환경을 고른다. 전처리·추론은 `--dataset` 생략 시 KIMM을 사용한다.
학습/평가는 항상 `--config`로 선택한다.
학습·추론 결과는 `runs/<환경>/`, 평가 그림은 `eval/<환경>/`에 저장한다.
BestFA는 KIMM 기준 이미지와 크롭을 물려받지 않는다. 초기 크롭은 꺼져 있다.
`datasets/bestfa/reference/reference_wire.png`의 **빨간 영역 두 개**를 각각 와이어의 기준 박스로 읽고,
`reference_outline.png`를 기구부 윤곽 오버레이로 쓴다. 두 PNG의 흰 배경은 라벨러에서 표시할 때만
투명하게 처리하며 원본 파일은 보존한다. BestFA에는 `reference_tube.png`가 없어도 된다.
라벨러에서 크롭을 저장하면 선택한 BestFA 세션의 `crop.yaml`에 기록한다.

### BestFA 원본 받기

BestFA DB 의 세션 폴더(`<YYMMDD_HHMMSS[_메모]>/Mono` 또는 `Image`)를 `datasets/bestfa/waiting/` 에 넣고
`waiting_to_raw.py` (위)로 raw 에 분류한다. 라벨은 만들지 않는다 — `labels/`, `labels_seg/`를 라벨러로
채운 다음 전처리한다.
검증·테스트를 분리하려면 라벨된 세션을 여러 개 준비해야 한다. 전체 재학습 epochs는 BestFA 측정
학습의 best epoch로 정한다.

`sources:` 에는 항목이 **하나** — 산출물이 `processed_<task>/` 에 바로 쌓이므로 원천이 다르면 config 를 따로 둔다:

| 인자 | 뜻 |
|---|---|
| `path` | 소스 폴더. 마지막 칸 글롭(`../datasets/kimm/raw/[0-9]*`)이면 세션 폴더 전부를 한 소스로 — 세션이 늘어도 config 그대로. 빼고 싶은 세션은 폴더 이름이 숫자로 시작하지 않게 (예: `_removed_manual/` 로 옮기기) |
| `labels` | 라벨 폴더 이름 (기본 `labels`). kimm 은 `labels_seg` |
| `names` | 폴더에 `data.yaml` 이 없을 때 쓰는 원본 클래스 이름 |
| `class_map` | 원본 클래스명 → 최종 클래스명. 여기 없는 클래스는 버림 |
| `type` | 원본 라벨 형식 `seg` / `obb` — `task: obb` 로 구울 때 seg 폴리곤에 OBB 를 다시 맞춤 |
| `group_by` | `session` = 세션 폴더 통째로 한 split 에 (2fps 연속 프레임끼리 근중복이라 파일 단위로 나누면 val 이 train 의 복사본이 됨) |
| `val_sessions` / `test_sessions` | 세션 직접 지정 — 지금은 장면 묶음(10/14 · 10/19 격자 · 10/20 hollow)이 train 에 모두 들어가게 골라 둠 |
| `val_ratio` / `test_ratio` | 세션 지정이 없을 때 쓰는 비율 (세션 단위라 근사) |
| `oversample` · `crop` · `resize` | train 복제 배수 · 크롭(아래) · 크롭 후 리사이즈 |

- 원본에서 **이미지만 지우면** 다음 실행 때 짝 라벨도 지운다 (이미지 0장 폴더는 경로 실수로 보고 보호).
- 크롭은 이미지와 폴리곤을 **함께** 자른다. 위치 규약(0% = 왼쪽/위 끝, 100% = 오른쪽/아래 끝)과 크기
  `0` = "그 축 원본 전체" 는 앱 `FrameCropper.cs` 와 같다. `auto_crop` 을 켜면 이미지마다 라벨에서 창을
  잡는다. 창 밖으로 나간 인스턴스는 잘리고, 남은 면적이 `min_area` 미만이면 버린다.
- `preview` 를 켜면 소스별 `preview/` 에 라벨을 그린 산출물 몇 장 — 크롭이 라벨을 따라갔는지 확인용.

### 세션별 크롭과 저장된 seg 확인

- 라벨러에서 세션을 선택하고 크롭을 조정한 뒤 **이 세션의 크롭 저장**을 누르면
  `raw/<YYMMDD_HHMMSS>/crop.yaml`에 저장한다. 다른 세션은 바꾸지 않는다.
- 세션을 바꾸거나 라벨러를 다시 열면 저장된 크롭을 불러온다. `crop.yaml`이 없는 세션은
  미설정으로 표시하며 전처리·자동 크롭 추론에서 오류를 낸다. 크롭이 불필요해도 사용을 끄고 저장한다.
  전처리는 모든 세션의 크롭을 검증한 뒤 산출물을 갱신한다. 미저장 크롭을 두고 세션을 바꾸면 확인한다.
- 전처리와 `infer.py --crop auto`도 세션 크롭을 우선 사용한다. 학습 머신에 데이터를 옮길 때
  세션의 `crop.yaml`도 함께 복사한다. 적용된 세션별 크롭은 전처리 manifest와 지문에 포함된다.
- 이미지 열기 시 저장된 `labels_seg` 폴리곤을 자동으로 표시한다. **저장된 seg 불러오기** 버튼으로
  디스크의 라벨을 다시 읽을 수도 있다. 저장된 결과에는 계산 당시 신뢰도 정보가 없으므로 별도로 표시한다.
- OBB를 저장하면 기존 seg가 삭제되는 동작은 동일하다. 크롭만 저장할 때는 원본 좌표의 seg를 유지한다.

### 학습 중 원본 크롭 위치 이동

KIMM과 BestFA의 `configs/base/kimm.yaml`, `configs/base/bestfa.yaml`에 `source_crop`이 켜져 있다. 학습 샘플을 읽을 때마다
원본 이미지에서 세션에 저장한 가로·세로 위치에 각각 **±25%p**를 더하고, 크롭 크기는 유지한다.
`offset_x: [-25, 25]`, `offset_y: [-25, 25]`로 이동 범위를 설정한다. 합산 후 0~100으로 제한한다.
예: 저장 위치 10에서 -25 이동하면 0, 90에서 +25 이동하면 100이다. 원본 픽셀의 ±25%를 뜻하지 않는다.
폴리곤도 같은 창으로 잘라서 좌표를 변환한다. 창에서 사라진 객체는 정답에서 제외하며,
객체가 하나도 남지 않은 샘플은 배경 이미지로 학습한다.

- train 로더만 랜덤 크롭한다. val/test 및 일반 추론은 저장된 고정 크롭을 쓴다.
  배포용 all은 학습 입력으로 합친 세션 전체에 적용하고, 검증 로더는 고정 크롭을 유지한다.
- 중복 이동을 피하려고 KIMM과 BestFA 모두 `translate`는 0이다.
- 전처리가 `crop_sources.json`에 원본 경로·해시, 원본 좌표 라벨 사본, 세션 크롭을 기록한다.
  이 파일의 해시도 manifest에 포함하고 학습 이력에 사본을 보관한다. 원본 이미지가 변경되면 학습 시작 전에 중단한다.
- **이 기능 이전에 전처리한 데이터는 다시 전처리해야 한다.** 현재 학습이 같은 산출물을 읽고 있다면
  종료 후 전처리한다. 동시에 준비하려면 `preprocess.py --dataset kimm --out <별도 폴더>`로 만들고
  다음 학습 config의 `train.data`를 그 폴더의 `data.yaml` 또는 `data.all.yaml`로 지정한다.
- 학습 머신에는 processed 폴더뿐 아니라 원본 이미지도 같은 상대경로 구조로 옮긴다.
  원본 라벨은 전처리 당시 사본을 사용하므로, 라벨을 수정했으면 전처리도 다시 해야 한다.
- 실제 랜덤 크롭 결과는 `stages.preview_aug` 또는 학습의 `train_batch*.jpg`에서 확인한다.
  전처리 preview와 기존 epoch overlay는 고정 크롭 기준이다.
- 현재 단일 device의 seg/obb 학습을 지원한다. `rect`, 이미지 cache, 클래스 필터, mosaic/mixup/
  copy-paste/cutmix는 끄고 사용한다. `resume` 대신 `train.model`에 체크포인트를 지정해 새 실행을 시작한다.

## 학습 (`train.py`, `configs/train.*.yaml`)

| config | 모델 | 데이터 | 종료 | 앱 |
|---|---|---|---|---|
| `train.kimm.single.yaml` | seg | train 세션 / val·test 분리 | early stop (patience 50) | 평가 후 배포 |
| `train.bestfa.single.yaml` | seg | BestFA train / val 세션 분리 (test 없음) | early stop (patience 50) | 평가 후 배포 |

- **배포 순서**: `single`로 학습 설정을 결정하고 test로 평가한 뒤, `*_all`로 전체 데이터를 학습해 배포한다.
  `single`의 평가 결과와 `all` 배포 모델의 학습 기록은 구분한다. `all`의 val 점수는 독립 평가가 아니다.
- **배포용 설정**: KIMM은 `train.seg_all.yaml` / `train.obb_all.yaml`, BestFA는
  `train.bestfa.seg_all.yaml` / `train.bestfa.obb_all.yaml`을 사용한다. OBB는 전처리 `--task obb`가 필요하다.
  `epochs`는 기존 값(KIMM 15, BestFA 100)이므로 실행 전에 결정한다. 자동 갱신되지 않으며
  `patience: 0`으로 early stopping은 끈다.
- **공통 설정**은 `configs/base/train.yaml`이며 환경별 기하 증강은 `base/kimm.yaml`과
  `base/bestfa.yaml`에 둔다. 각 실행 설정은 환경별 base를 상속한다. 같은 환경의 single과 all은 같은 증강을 사용한다.
- **KIMM 증강**: 세션 저장 크롭 위치에서 ±25%p 이동, 회전 ±5°, 확대·축소 0.9~1.1배.
  별도 이동·기울이기·원근·좌우 반전은 끈다.
- **BestFA 증강**: 세션 저장 크롭 위치에서 ±25%p 이동, 회전 ±25°, 확대·축소 0.8~1.2배, 기울이기 ±2°, 원근 0.0002,
  좌우 반전 확률 50%. 좌우 반전의 효과는 val에서 비교한다.
- **공통 증강**: 상하 반전·mosaic·mixup·copy-paste는 끈다. 감마 0.5~2.0(로그 균등, 확률 80%) → 밝기 0.5~2배(로그 균등, 확률 80%)를 독립 적용하고,
  블러(확률 20%, sigma 1~3)는 기존 값을 유지한다. preview에서 와이어 끝이 사라지거나 포화되는지 확인한다.
- **보기**: TensorBoard(`uv run tensorboard --logdir runs` 또는 task) 와, 매 val 마다
  `<실행 폴더>/overlays/epoch_NNN.jpg` (원본 4장 + 증강 12장, 증강 값 표시, 초록 = 정답, 파랑 = 추론 마스크(옅게)). 끝나면 `best.jpg`.
- **증강 미리보기**: `stages: {preview_aug: true, train: false, export: false}` → `runs/preview/aug/`.
- **export 만**: `stages: {preview_aug: false, train: false, export: true}` — 최신 실행의 best.pt 를 ONNX 로.
- **배포**: 학습은 앱을 건드리지 않는다. 쓸 실행을 `uv run archive.py runs/<task>/<name>/<시각>` (또는 task
  `archive`) 로 저장소 루트 `weights/` 에 보관(Git LFS)하고, `weights/deploy.yaml` 에서
  `yolo26s-seg.onnx: yolo26s-seg-all/<시각>` 처럼 고른 뒤 커밋 → 앱 빌드가 그 `best.onnx` 를 실행 파일 옆
  `Models/` 로 복사한다. 이력은 deploy.yaml 의 git 이력. 앱은 모델 파일을 보고 seg/obb 를 알아서 고른다.
- **기록 · 재현**: 실행 폴더에 아래가 남고, `archive.py` 가 `weights/` 로 함께 보관한다 (`provenance.py`).

  | 파일 | 내용 |
  |---|---|
  | `run_info.yaml` | 코드(커밋 · 커밋 안 된 수정) · config(상속 푼 것) · 데이터(manifest — 세션별 split · 장수 · 지문) · 시작 가중치 sha256 · 환경(파이썬 · 패키지 · CUDA · GPU · 드라이버 · uv.lock 지문) · 실제 batch |
  | `config.yaml` | 재현용 config — 시작 가중치를 실제로 쓴 파일로, `batch: -1`(AutoBatch)을 실제 값으로 고정 + `reproduce` 블록(데이터 지문 · 커밋) |
  | `data/<소스>.manifest.yaml` · `.files.tsv` | 그때의 manifest 사본 — processed 를 다시 구워도 남는다 |
  | `code.patch` | 커밋 안 된 수정이 있었으면 그 diff |

  같은 학습을 다시 돌리려면 `uv run train.py --config <실행 폴더>/config.yaml`. 시작할 때 데이터 지문이
  기록과 다르면 중단하고(어느 세션이 다른지 표시), 커밋이 다르면 `git switch --detach <커밋>` 안내와 함께
  경고한다. GPU 연산의 비결정성 때문에 **같은 GPU · 드라이버 · 패키지**에서도 마지막 자리 수치까지 같다는
  보장은 없다 (데이터 · 코드 · 설정 · 시드는 같음 — Ultralytics `deterministic: true` 기본).
- 결과는 `runs/<task>/<name>/<YYMMDD_HHMMSS>/` — 실행마다 쌓인다. config 에서 타임스탬프를 뺀
  `runs/<task>/<name>/weights/best.pt` 는 그 name 의 최신 실행으로 해석된다.
  `weights/` 에는 best.pt · last.pt 외에 50 epoch 마다 `epoch0.pt` · `epoch50.pt` … 도 남는다
  (`save_period`, Ultralytics 가 epoch 을 0부터 세서 `epoch50.pt` = 51번째 epoch 끝).

## 평가 · 추론

- `eval.single.yaml` — val 세션 (학습 중 best 선택에 쓰인 것).
- `eval.final.yaml` — **test 세션, 측정용 모델에 딱 한 번.** 이 결과를 보고 설정을 다시 만지면 test 가
  오염된다.
- 그림: `eval/<시각>__<가중치>__<데이터셋>-<split>/` — 초록 = 정답, 빨강 = 추론(seg 는 마스크 윤곽),
  점선 = 가장 아래 두 점의 y (와이어 끝 높이 오차).
- `infer.py --dataset <환경>` — 크롭(`auto` = 선택한 환경의 설정, `W H CX CY`, `none`) 후 예측.
  사진이면 `<이름>_pred.jpg`, 세션 폴더면 `<세션>.mp4` (`runs/<환경>/infer/<시각>/`).

## config 규약 — 조용한 기본값 없음

설정은 아래 역할별로 수정한다. 파일명과 실행 결과 경로는 기존 명령과 호환된다.

| 바꿀 내용 | 파일 |
|---|---|
| 원본 경로, 세션 분할, 크롭, 기준 이미지 | `configs/preprocess.kimm.yaml` / `preprocess.bestfa.yaml` |
| 시작 모델, 학습 데이터, epochs/patience, 결과 경로 | `configs/train.kimm.single.yaml` / `train.bestfa.single.yaml` |
| GPU, batch, 입력 크기, 밝기·블러, ONNX export | `configs/base/train.yaml` |
| 환경별 회전·이동·스케일·반전 | `configs/base/kimm.yaml` / `base/bestfa.yaml` |
| 평가 가중치, 데이터, val/test, 이미지 저장 경로 | `configs/eval.*.yaml` |
| 공통 평가 해상도, GPU, 시각화 | `configs/base/eval.yaml` |

- `base:` 경로는 해당 YAML 파일 기준이다. dict는 키별 병합하고 리스트·단일 값은 교체한다.
  학습과 평가는 각각 공통 base를 상속하며, 전처리는 라벨러에서 직접 읽고 수정하므로 독립 파일로 둔다.
- KIMM의 val/test는 세션 목록으로 고정되어 있다. 새 세션은 train으로 들어가므로 새 촬영 조건을
  평가하려면 분할 목록을 먼저 정한다. 현재 hollow 세션은 train에만 있어 별도 평가가 필요하다.
- BestFA는 세션 단위 고정 분할: `261026_115301`(1층)·`261026_145202`(3층)은 train,
  `261026_144002`(2층)는 val이며 test는 만들지 않는다. 새 세션은 train으로 들어간다.
  한 세션의 프레임은 여러 split에 나뉘지 않는다. BestFA는 val로 선택·평가 후 all로 배포용 학습한다.
- 평가 가중치 경로에서 시각 폴더를 생략하면 최신 실행을 사용한다. 최종 평가·비교에는 시각까지 지정한다.

- 학습·평가는 `--config` 없이 돌지 않는다. 전처리는 `--dataset`으로 선택하며 생략 시 KIMM이다.
  읽은 config 경로는 항상 로그 첫 줄에 찍힌다.
- `stages` 블록·키 오타·전부 false, `train.data`·`train.model` 누락은 추측하지 않고 오류.
- `train.model` 에 경로 구분자가 있으면 체크포인트로 보고 존재를 확인, 없으면(`yolo26s-seg.pt`)
  Ultralytics 가 받아올 이름.
- yaml 안의 경로는 `training/` 기준.

## 린트

```powershell
uv run ruff check .      # 설정은 pyproject.toml (포맷터는 안 씀 — 줄 끝 주석 정렬 유지)
```

CI(`.github/workflows/ci.yml`)가 push 마다 돌리고, 앱 빌드·포맷도 확인한다.

## 같은 규칙이 여러 곳에 있는 것

언어가 달라 공유할 수 없어 같은 규칙을 따로 구현한 곳들 — **하나를 바꾸면 나머지도 맞춘다.**

| 규칙 | 위치 |
|---|---|
| 크롭 창 계산 (크기 0 = 전체, 위치 0~100%) | `preprocess.crop_rect` · 앱 `FrameCropper.cs` · 라벨러 `index.html` `cropRect()` |
| 와이어 자세 (긴 변 = 축, 우측 상단 꼭짓점 = 피벗) | `labeller/edge.obb_pose` · 라벨러 `index.html` `poseFromPoints()` |
| 증강 범위 | `configs/base/train.yaml` → 학습(Ultralytics)과 `epoch_overlay.py`(값 표시용 재현) |
