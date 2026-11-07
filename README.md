# basler_segmentation

Basler 카메라(eye-in-hand)로 WAAM 와이어를 실시간 분할하는 프로젝트. 세 덩어리로 나뉜다.

```
basler_segmentation/
├─ datasets/   데이터 (git 밖)          앱이 녹화하고, training 이 라벨·전처리·학습에 쓴다
├─ training/   학습 Python (uv 프로젝트) 라벨러 · 전처리 · 학습 · 평가 · 추론
└─ app/        실증 C# (.NET 8 WPF)      라이브 뷰 · 녹화 · 학습한 모델로 실시간 추론
```

흐름: `app` 으로 녹화 → `training/labeller` 로 라벨 → `training` 으로 전처리·학습 → 쓸 모델을 `weights/` 에
보관하고 `weights/deploy.yaml` 에서 고름 → `app` 빌드가 그 ONNX 를 넣어 실증.

## datasets/

KIMM과 BestFA는 촬영 구도·기구부가 달라 원본, 기준 이미지, 크롭 설정과 전처리 결과를 분리한다.
데이터는 git 밖에 두며, KIMM 샘플 1장과 환경별 기준 레이어는 예외로 추적한다.
샘플로 라벨러를 확인할 수 있고, 학습·검증 분할에는 실제 라벨된 세션들이 필요하다.

```text
datasets/
├─ kimm/
│  ├─ raw/kimm_<날짜>/        Mono/ · labels/ · labels_seg/ · data.yaml
│  ├─ reference/             KIMM 기구부·와이어 기준 이미지
│  └─ processed/{seg,obb}/   크롭·리사이즈·분할 후 학습 데이터
└─ bestfa/
   ├─ raw/bestfa_<날짜>/      Mono/ · labels/ · labels_seg/ · data.yaml
   ├─ reference/             BestFA 전용 기준 이미지
   └─ processed/{seg,obb}/   BestFA 전처리 결과
```

아래 경로는 `datasets/<환경>/` 기준이다.

| 경로 | 내용 | 만드는 쪽 |
|---|---|---|
| `raw/<환경>_YYMMDD_HHMMSS[_메모]/Mono/` | 녹화 세션 원본 | 앱 REC · BestFA 가져오기 |
| `raw/<세션>/labels/` · `labels_seg/` | 원본 좌표의 OBB · 와이어 seg 폴리곤 | 라벨러 |
| `raw/_removed_black/` · `_removed_manual/` | 빼 둔 프레임 (되돌리기용) | 정리 작업 · 라벨러 삭제 |
| `processed/seg/` · `processed/obb/` | 변환된 이미지와 라벨, train/val/test 분할 | `training/preprocess.py` |
| `reference/` | 기구부(튜브+와이어) 기준 이미지·누끼·윤곽선 | `training/tools/make_reference_cutout.py` |

## training/

```powershell
cd training
uv sync --extra train --extra export      # 첫 설치 (torch cu128 · ultralytics · ONNX · pytest · ruff)
uv run labeller/server.py                 # 라벨러 → http://localhost:8765
uv run preprocess.py --dataset kimm        # raw → kimm/processed/seg
uv run train.py --config train.single.yaml
uv run pytest                             # 파이프라인 테스트 (GPU·데이터 불필요)
```

라벨러는 `uv run labeller/server.py`로 실행하고 화면 상단 프리셋에서 **KIMM / BestFA**를 고른다.
BestFA 라벨링 후에는 `uv run preprocess.py --dataset bestfa`,
`uv run train.py --config train.bestfa.single.yaml` 순서로 실행한다.
외부 `<세션>/Mono` 가져오기와 환경별 명령은 [training/README.md](training/README.md)에 있다.
앱은 `BaslerLiveView.exe --dataset bestfa`로 BestFA 설정을 읽는다 (기본 KIMM).

자세한 건 [training/README.md](training/README.md). VS Code 에서는 `Ctrl+Shift+P → Tasks: Run Task` 의
`run labeller` / `train` / `inference — …` / `tensorboard`로 돌릴 수 있다.
라벨러는 저장소 루트에서도 같은 `uv run labeller/server.py` 명령을 지원한다.
BestFA는 빨간 기준 영역 두 개를 와이어 1·2로 읽어 각각 편집하고 함께 저장한다.

## app/

```powershell
# app/external/YoloDotNet — 빌드에 쓰는 두 폴더만 받는다 (~1MB).
# 그냥 `git submodule update --init` 하면 테스트용 ONNX 모델까지 ~500MB 를 받는다.
git submodule init
git clone --filter=blob:none --no-checkout https://github.com/NickSwardh/YoloDotNet.git app/external/YoloDotNet
git -C app/external/YoloDotNet sparse-checkout set YoloDotNet YoloDotNet.ExecutionProvider.DirectML
git submodule update --force
git submodule absorbgitdirs

dotnet build app/basler_segmentation.sln
```

`app/external/` 에 Basler pylon SDK(필요한 부분만 벤더링)와 YoloDotNet(submodule)이 있다.
설정은 [app/src/BaslerLiveView/config/config.xml](app/src/BaslerLiveView/config/config.xml) —
녹화 폴더·기준 이미지 같은 상대경로는 저장소 루트(`.git` 이 있는 폴더) 기준.
추론 모델은 `<Inference><Model>` — seg / obb 모델을 파일에서 알아서 구분한다. 그 파일은 빌드가
[weights/deploy.yaml](weights/deploy.yaml) 에서 고른 보관본을 복사한 것 (clone 뒤 `git lfs pull` 필요).
자세한 건 [app/src/BaslerLiveView/README.md](app/src/BaslerLiveView/README.md).
`<Camera><hardware_roi>false</hardware_roi></Camera>`가 기본이며, `true`이면 `<Crop>` 영역을
카메라 하드웨어 ROI로 수신·저장합니다. 설정 변경 후 앱을 다시 실행합니다.

## 저장소 규칙

- 줄바꿈 `.gitattributes`, 들여쓰기·인코딩 `.editorconfig`, Python 린트 `training/pyproject.toml` (ruff).
- CI(`.github/workflows/ci.yml`): push 마다 `training` ruff · pytest, `app` 빌드 · `dotnet format` 확인.
