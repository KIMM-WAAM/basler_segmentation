# BaslerLiveView

Basler pylon 카메라 실시간 영상 + 녹화 + 학습한 YOLO 모델(seg 또는 obb)로 실시간 추론하는 WPF 앱.

촬영 환경은 실행할 때 선택한다. 기본 KIMM은 `config/config.xml`, BestFA는
`config/config.bestfa.xml`을 읽으며 녹화 경로·세션 접두어·기준 이미지·크롭·모델을 따로 둔다.

```powershell
BaslerLiveView.exe --dataset kimm
BaslerLiveView.exe --dataset bestfa
# 카메라 없이 확인: 위 명령에 --emulate 추가
```

BestFA는 크롭을 끈 상태로 시작한다. 전용 기준 이미지와 학습 모델을 준비한 뒤 해당 설정에 연결한다.
BestFA 모델 배포 시 `weights/deploy.yaml`에 `bestfa-yolo26s-seg.onnx`를 추가한다.

## 구성

| 파일 | 역할 |
|------|------|
| `CameraService.cs` | pylon `Camera` 래퍼 — 연결/연속 그랩/`PixelDataConverter`(→BGRA) |
| `FrameCropper.cs` | 크롭 창 계산/실행 (위치는 축별 0~100%) — 추론 입력 + 화면 오버레이 좌표 |
| `FrameRecorder.cs` | REC 버튼 저장 — 저장 fps 샘플링 + 큐 기반 백그라운드 PNG 인코딩/기록 |
| `InferenceService.cs` | YoloDotNet(DirectML) 추론 — 모델 파일을 보고 seg(마스크) / obb(회전 박스) 자동 선택 |
| `CameraConfig.cs` | `config\config.xml` 로드 |
| `MainWindow.xaml(.cs)` | 카메라 선택·연결·시작/정지 · 크롭 · 기구부 오버레이 · REC · Inference UI |
| `App.xaml.cs` | `--dataset` 촬영 환경 선택 · `--emulate` 소프트웨어 카메라 에뮬레이터 |
| `BaslerLiveView.csproj` | `Basler.Pylon.dll` 참조 + `external\pylon\Runtime\x64` 네이티브 DLL·`Models\*.onnx` 를 출력 폴더로 복사 |

## 동작 흐름

```
CameraFinder.Enumerate()                    // 카메라 열거
  → new Camera(info) + Configuration.AcquireContinuous
  → Open()
  → StreamGrabber.Start(LatestImages, ProvidedByStreamGrabber)
  → ImageGrabbed 이벤트 (grab 스레드)
      → PixelDataConverter.Convert → BGRA byte[]
      → FrameRecorder.Submit         // REC 중이면 수신 원본(전체 센서 또는 하드웨어 ROI)을 큐에 복사
      → (Inference 켠 경우)
          FrameCropper.Crop          // 크롭본 = 모델 입력
          → InferenceService.Submit  // 최신 프레임만 처리 (밀리면 버림)
          → Compose()                // 풀 프레임 위 크롭 위치에 최신 추론 결과를 다시 얹음
      → Dispatcher.Invoke → WriteableBitmap.WritePixels  // 수신 프레임을 표시
```

기본 소프트웨어 크롭 모드에서는 화면이 풀 프레임이고, 크롭 영역은 그 위에 빨간 네모(`OverlayCanvas`)로 표시됩니다.
바깥이 보여야 프레임을 어디로 옮길지 판단할 수 있기 때문입니다. 모델은 학습 때와 같은 화각을
보도록 크롭본만 받고, 그 결과 오버레이는 네모 안쪽에 되돌려 그립니다.

- 그랩은 pylon이 제공하는 별도 스레드에서 돌고, 프레임마다 `ImageGrabbed`가 발생합니다.
- `PixelDataConverter`가 센서 포맷(Mono8/Bayer/YUV 등)을 WPF `Bgra32`에 그대로 맞는 BGRA8로 변환합니다.
- WPF 오브젝트는 UI 스레드에서만 만져야 하므로 `Dispatcher.Invoke`로 마샬링합니다.

## 추론 모델

`config\config.xml` 의 `<Inference><Model>` 파일을 `Models\` 에서 읽습니다 (기본 `yolo26s-seg.onnx`).
YoloDotNet 이 모델의 종류를 파일에서 알아내므로 seg 모델이면 마스크, obb 모델이면 회전 박스를 그립니다.

- 학습 쪽 `training/configs/train.seg_all.yaml` / `train.obb_all.yaml` 의 export 가
  `src\BaslerLiveView\Models\yolo26s-seg.onnx` / `yolo26s-obb.onnx` 를 덮어써 배포하고,
  이전 것은 `Models\history\` 에 남깁니다 (빌드는 history 를 복사하지 않음).
- obb 모델로 바꾸려면 `<Model>yolo26s-obb.onnx</Model>`.
- 프레임당 검출 수는 `<MaxDetections>` (점수 높은 순으로 그 개수만, = 와이어 수 — 지금 2),
  최소 점수는 `<Confidence>` (기본 0.24). 앱을 다시 켜면 반영됩니다.

## 학습 데이터 수집 (크롭 + 레코딩)

**기본값 `hardware_roi=false`에서는 REC가 크롭 전 풀사이즈 원본을 저장합니다.** 크롭은 추론 입력에만 실제로 걸리고,
화면에는 네모로 표시만 됩니다. 학습용 크롭은 `training/preprocess.py` 가 담당합니다 —
이미지만이 아니라 라벨까지 같이 잘라야 하고, 여기서 버린 픽셀은 되돌릴 수 없기 때문입니다.

- **툴바 2번째 줄**: `Crop` 체크 · 크기(px) · `↔`/`↕` 슬라이더(0~100%) · `Center`(둘 다 50%로 리셋)
  - 0% = 왼쪽/위 끝, 100% = 오른쪽/아래 끝, 50% = 중앙.
  - 네모 위에 `1280×1280 (40%, 40%)` 처럼 현재 값이 표시됩니다.
  - 크롭 크기가 센서 이미지보다 크면 원본 크기로 클램프되고, 네모는 사라집니다(= 잘릴 게 없음).
  - 학습 쪽 크롭은 `training/configs/preprocess.kimm.yaml` 의 `crop` (또는 라벨러의 크롭 칸) — 같은 0~100% 규약.
- **`● REC` 버튼**: 한 번 누르면 저장 시작, 다시 누르면 중지. 저장 현황은 상태바 오른쪽에 표시.
  - REC 를 켤 때마다 세션 폴더가 새로 생깁니다: `datasets/<환경>/raw/<환경>_YYMMDD_HHMMSS/Mono/`.
    학습 쪽은 이 세션 폴더들을 글롭(`kimm_*`)으로 한꺼번에 읽습니다.
  - 저장 속도는 `<Recording><Fps>`. 연속 프레임은 거의 똑같아서 전부 저장하면 디스크와 라벨링
    시간만 낭비됩니다. 그랩은 `<FrameRate>` 그대로 돌고 저장만 샘플링합니다 (`0` = 모든 프레임).
  - PNG 인코딩은 별도 워커 스레드 + 바운디드 큐라서 그랩 루프와 UI를 막지 않습니다.
    디스크가 못 따라가면 최신 프레임을 버리고(`dropped`), 파일 번호에 구멍으로 남습니다.
  - 저장되는 건 **카메라 원본**입니다 (추론 오버레이가 찍히지 않음).
- 크기·위치 시작값과 저장 경로·저장 fps 는 `config\config.xml` 의 `<Crop>` / `<Recording>` 에서 바꿉니다.
  상대경로는 저장소 루트(= `.git` 이 있는 폴더) 기준입니다.

## 하드웨어 ROI 전환

`config/config.xml`에서 `<Camera><hardware_roi>`를 `true` 또는 `false`로 설정하고 **앱을 다시 실행**합니다.
두 모드 모두 `<Crop>`의 `Enabled`, `Width`, `Height`, `CenterX`, `CenterY`를 사용합니다.

| 설정 | `false` (기본, 기존 train) | `true` (하드웨어 ROI) |
|---|---|---|
| 카메라 수신·화면·REC | 센서 전체 | 지정한 ROI 영역만 |
| 추론 | 수신 영상에서 소프트웨어 크롭 | 수신 ROI를 그대로 사용, 이중 크롭 없음 |
| 크롭 툴바 | 실행 중 조절 가능 | 비활성화, XML 수정 후 재실행 |
| 기준 이미지 오버레이 | 전체 기준 이미지 | 실제 카메라 ROI에 맞춰 기준 이미지도 크롭 |
| 와이어 `full` 좌표 | 센서 기준 | 하드웨어 ROI의 실제 오프셋을 더한 센서 기준 |

크기·위치는 카메라의 유효 범위와 증분에 맞춰 내림 정렬됩니다. `Width` 또는 `Height`가 `0`이면
해당 축 전체를 사용하고, `<Crop><Enabled>false</Enabled>`이면 두 축 모두 전체를 사용합니다.
`hardware_roi=false`로 돌아오면 카메라에 남아 있는 ROI를 전체 센서로 복원합니다.
ROI 적용에 실패하면 잘못된 영역으로 촬영하지 않도록 연결을 실패 처리합니다.
하드웨어 모드의 기준 이미지는 센서 전체 크기와 일치해야 하며, 다르면 해당 레이어를 표시하지 않습니다.

이전 `data_collection`의 촬영 설정은 다음과 같이 재현합니다 (`FrameRate`는 목표값).

```xml
<Camera>
  <FrameRate>120</FrameRate>
  <hardware_roi>true</hardware_roi>
</Camera>
<Crop>
  <Enabled>true</Enabled>
  <Width>640</Width>
  <Height>1280</Height>
  <CenterX>45</CenterX>
  <CenterY>10</CenterY>
</Crop>
```

하드웨어 ROI로 녹화한 PNG에는 바깥 영역이 없습니다. 해당 세션을 학습할 때는 전처리 소스를 따로
지정하고 `crop.enabled: false`로 두어 같은 크롭을 반복하지 마세요. 전체 센서 촬영과 구분하려면
`Recording/Directory`를 별도 폴더로 지정할 수 있습니다.

설정·좌표·ROI 전환·실제 프레임 수신·PNG 저장 회귀 검증 (물리 카메라 대신 pylon 에뮬레이터 사용):

```powershell
dotnet run --project app/tests/BaslerLiveView.Tests -c Release
```

## 실행

```powershell
git submodule update --init --recursive              # app/external/YoloDotNet (처음 한 번)
dotnet run --project app\src\BaslerLiveView          # 실제 카메라
dotnet run --project app\src\BaslerLiveView -- --emulate   # 카메라 없이 에뮬레이터로 테스트
```

또는 빌드 후 `bin\Debug\net8.0-windows\BaslerLiveView.exe` 직접 실행.

## 참고

- pylon이 시스템에 설치돼 있지 않아, 네이티브 런타임(`PylonBase`, `PylonC`, 전송 계층 `*_TL.dll`)을
  `app\external\pylon\Runtime\x64`에서 실행 파일 옆으로 복사합니다. 앱 실행 폴더는 항상 네이티브 DLL 검색 경로에 포함됩니다.
- `Basler.Pylon.dll`은 x64 전용이므로 프로세스도 x64로 강제(`PlatformTarget=x64`)합니다.
