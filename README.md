# doosan_autorobot
두산 로봇팔

## 세팅
```bash
ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=110.120.1.13 model:=e0509
ros2 run dsr_gripper gripper_service
python3 -m http.server 8080  # 서버 올리기
python3 move_and_grip_web_0925.py  # 실행
```

## 파일 설명

| 파일 | 설명 |
|------|------|
| `yolo_inference_0922.py` | YOLO로 객체 검출 |
| `move_and_grip_web_0925.py` | 로봇팔 구동 (웹으로 구동) |
| `food_pick_console.html` | 웹 화면 |
| `T_base_camera.npy` | 기존 캘리브레이션 |
| `xy_correction.json` | 캘리브레이션 보정 |

## 수정이 필요한 경로

### yolo_inference_0922.py
- **line 37** : YOLO 모델 path 변경

### move_and_grip_web_0925.py
- **line 120** : 캘리브레이션 보정 파일 경로 (`xy_correction.json`)
- **line 121** : 기존 캘리브레이션 파일 경로 (`T_base_camera.npy`)
