# doosan_autorobot
두산 로봇팔

## 수정이 필요한 경로

### yolo_inference_0922.py
- **line 37** : YOLO 모델 path 변경

### move_and_grip_web_0925.py
- **line 120** : 캘리브레이션 보정 파일 경로 (`xy_correction.json`)
- **line 121** : 기존 캘리브레이션 파일 경로 (`T_base_camera.npy`)

## 파일 설명

| 파일 | 설명 |
|------|------|
| `T_base_camera.npy` | 기존 캘리브레이션 |
| `xy_correction.json` | 캘리브레이션 보정 |
