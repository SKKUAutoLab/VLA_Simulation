# tools/ — 트랙 차선 GT 시각화 도구

트랙 차선 중심선 Ground-Truth(GT) 데이터를 눈으로 확인하기 위한 스크립트 모음.
(GT 데이터의 의미와 코드 위치는 상위 [`README.md` 부록 A](../README.md) 참고.)

| 스크립트 | 방식 | 출력 |
|----------|------|------|
| `gt_render.py` | matplotlib | 월드좌표(X/Y, m) 평면 플롯 |
| `gt_overlay.py` | OpenCV | `track.png` 위 GT 중심선 오버레이 |
| `gt_gui_shot.py` | PySide6 GUI | 어노테이터 창 스크린샷 (중심선 / 도로·차선 마스킹) |

모든 명령은 **워크스페이스 루트**(`~/VLA_simulation`)에서 실행합니다.

## 공통 입력
- `GT_JSON` (기본 `~/track_gt_manual.json`): `centerline_pixels`가 있으면 그대로,
  없으면 `centerline_world`를 `world_to_pixel()`로 변환해 사용합니다.
- 트랙 배경: `src/simulation_pkg/models/race_track/materials/textures/track.png`

## (A) 월드좌표 평면 플롯
```bash
python3 tools/gt_render.py ~/track_gt_manual.json -o /tmp/gt_render.png
python3 tools/gt_render.py                 # 기본 GT + /tmp/gt_render.png
```

## (B) 트랙 이미지 위 오버레이
```bash
python3 tools/gt_overlay.py ~/track_gt_manual.json -o /tmp/gt_overlay.png
```
빨간 점/선 = GT 중심선, 초록 원 = 시작점.

## (C) 실제 GUI 창 스크린샷
X 디스플레이가 필요합니다(헤드리스 서버면 `DISPLAY=:1 QT_QPA_PLATFORM=xcb`).

```bash
# GT 중심선만
DISPLAY=:1 QT_QPA_PLATFORM=xcb python3 tools/gt_gui_shot.py ~/track_gt_manual.json -o /tmp/gt_gui.png

# 도로/차선 마스킹까지 (1차선 빨강 / 2차선 파랑)
DISPLAY=:1 QT_QPA_PLATFORM=xcb python3 tools/gt_gui_shot.py ~/track_gt_manual.json --mask -o /tmp/gt_mask.png
```

`--mask`는 V-range 도로 검출 + 극좌표 차선 분리를 실행한 뒤 캡처합니다.

## 의존성
```bash
pip install matplotlib opencv-python PySide6
```
`gt_gui_shot.py`는 `src/gui_pkg/gui_pkg/gt_annotator.py`를 import하므로 저장소 안에서 실행해야 합니다.

## 참고 — 함정 노트
- `gt_annotator`의 `set_show_lane()` 인자는 `"both" | "inner" | "outer"` **문자열**입니다
  (`True` 등을 넣으면 마스크가 그려지지 않음).
- `*_demo.json`(월드좌표만)과 `track_gt_manual.json`(픽셀좌표 포함)은 포맷이 다릅니다 — 스크립트가 자동 처리.
