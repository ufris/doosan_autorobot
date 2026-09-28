###################### 테스트 세팅 ######################
# 터미널 1(bringup)
# $ jazzy
# $ cd /path/to/doosan_ws(두산 폴더로 이동)
# $ source ./install/local_setup.bash
# $ ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=real host:=110.120.1.13 model:=e0509
#
# 터미널 2(gripper 작동)  ※ 방식 A 를 쓸 것 — 로봇 모션과 DRL 충돌이 없음
# $ jazzy
# $ cd /path/to/doosan_ws(두산 폴더로 이동)
# $ source ./install/local_setup.bash
# $ ros2 run dsr_gripper gripper_service
#
# 터미널 3(이 스크립트 — 8000번 포트로 API 서버가 같이 뜬다)
# $ pip install fastapi uvicorn   # 처음 한 번만
# $ python3 move_and_grip_web_0923.py
#
# 화면
#   브라우저에서 food_pick_console.html 을 열면 자동으로 붙는다.
#   ※ 로봇 없이 화면만 테스트하려면 이 스크립트 대신 mock_server.py 를 띄울 것.
#     (둘 다 8000번이라 동시에는 못 띄운다)
#######################################################

import sys
import json
import socket
from pathlib import Path
import threading
import queue as queue_mod
import numpy as np
import cv2
import scipy
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from dsr_gripper import gripper_open, gripper_close, gripper_cmd

import rclpy
import time

import yolo_inference_0922 as my_yolo

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

######################################## 로봇 세팅
ROBOT_ID = "dsr01"
ROBOT_MODEL = "e0509"

#### jjw : rclpy 를 실행하고 진행해야함
import DR_init
DR_init.__dsr__id    = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

rclpy.init()
node = rclpy.create_node("vision_to_robot_web", namespace=ROBOT_ID)
DR_init.__dsr__node = node

#### jjw : rclpy 를 실행하고 진행해야함(ros를 시작하고 두산을 시작해야 연결이 가능함. 순서중요)
import DSR_ROBOT2 as dsr
from DSR_ROBOT2 import (
    movej, movel, get_current_posj, get_current_posx,
    set_robot_mode, get_robot_mode,
    ROBOT_MODE_AUTONOMOUS, ROBOT_MODE_MANUAL,
    DR_BASE, DR_MV_MOD_ABS, DR_MV_MOD_REL,
)
from DSR_ROBOT2 import posj, posx
from scipy.spatial.transform import Rotation as Rot


# 서비스가 발견될 때까지 잠깐 대기 — 노드를 만들자마자 바로 호출하면
# 요청이 유실되어 '응답 없이 무한 대기'에 빠질 수 있습니다.
assert dsr._ros2_get_robot_mode.wait_for_service(timeout_sec=10.0), (
    "로봇 서비스를 찾지 못했습니다 — bringup 실행 여부와 "
    "ROS_DOMAIN_ID(주피터 터미널 vs bringup 터미널)를 확인하세요.")

set_robot_mode(ROBOT_MODE_AUTONOMOUS)
x, sol = get_current_posx()
print("로봇이 연결되었습니다. 현재 좌표 [x,y,z,rx,ry,rz]:", [round(v, 1) for v in x])


# =====================================================================
#  설정값 — 동작을 바꾸려면 여기만 고치면 됩니다
# =====================================================================

######## 안전 ########
DRY_RUN = False   # True 면 좌표만 계산/출력하고 집는 동작은 하지 않음 (비전 확인용)

######## 웹 서버 ########
SERVER_PORT = 8000

######## 로봇 속도 / 가속도 ########
VEL_J, ACC_J = 140, 80   # movej — 관절 이동(자세 전환). 크게 휘두르므로 주변 확인 필수
VEL_L, ACC_L = 120, 80   # movel — 수평 이동(물체/그릇 위로 가는 구간)
VEL_Z, ACC_Z = 120, 80   # movel — 수직 하강/상승(집고 놓는 구간). 가장 느리게 두는 게 안전

######## 로봇 자세 (관절 각도, 도) ########
POSE_CAMERA_CLEAR = (90, 0, 90, 0, 90, 0)   # 촬영 전 팔을 시야 밖으로
POSE_READY        = (0, 0, 90, 0, 90, 0)    # 특이점 회피 준비자세

######## 그리퍼 ########
TCP_Z_MM     = 50.0   # 플랜지→그리퍼 끝 거리(mm). 불확실하면 실제보다 '길게'(덜 내려가서 안전)
Z_MIN_MM     = 50.0   # 그리퍼 끝이 이 높이(mm) 아래로 내려가는 명령은 거부
GRIP_CURRENT = 400    # 잡는 힘(전류 제한)
GRIP_WAIT    = 1.0    # 개폐 후 대기(초). DRL 내부 wait 만 1.2초라 넉넉히 둬야 함
                      #   ※ 짧으면 다음 movel 이 DRL 을 죽여 시리얼 포트가 열린 채 남고,
                      #      그 뒤로 그리퍼가 통째로 먹통이 됩니다.

######## 이동량 (mm) ########
H_PICK          = 175                 # 물체 잡으러 내려갈 양
H_PLACE         = 100                 # 그릇 위에서 놓으러 내려갈 양
CALIB_OFFSET_MM = [0.0, 0.0, 0.0]     # homography 가 이미 보정을 포함하므로 0.
                                      #   여기에 값을 넣으면 보정 위에 또 더해진다.
REACH_MIN_MM    = 150                 # e0509 도달 가능 수평거리 하한
REACH_MAX_MM    = 850                 # 상한 (작업반경 약 900mm)

######## 카메라 / 비전 ########
# XY 는 homography 로 구한다 (correction_coord_0925_2.py fit 결과).
# depth 와 hand-eye 는 참고 출력에만 쓴다.
XY_CORRECTION_PATH = "/home/jaewoo/doosan_ws/src/dsr_study-main/xy_correction.json"
CALIB_PATH  = "/home/jaewoo/HamdEyeCal/T_base_camera.npy"
N_WARMUP    = 15     # 자동노출 안정화용으로 버리는 프레임 수 (프로그램 시작 시 1회만)
N_FLUSH     = 5      # 주문마다 팔을 파킹한 직후, 오래된 버퍼 프레임을 버리는 수
N_COLOR     = 10     # YOLO 투표용 컬러 프레임 수
N_DEPTH     = 5      # 깊이 시간축 중앙값용 프레임 수
DEPTH_WIN   = 4      # 깊이 샘플 창 반경 → (2*win+1)^2 픽셀
DEPTH_MIN_M = 0.2    # 유효 깊이 하한(m)
DEPTH_MAX_M = 1.5    # 유효 깊이 상한(m)
DETECT_SHOW = False  # YOLO 검출 결과 창 표시 (웹 모드에선 매 주문마다 뜨면 번거로워서 기본 끔)

######## 라벨 ########
BOWL_ID = 5   # bowl — 놓을 자리. 웹 콘솔의 선택 대상에서는 제외한다.
# =====================================================================


# =====================================================================
#  함수 정의 (로봇/좌표 계산 — move_and_grip_0922_3.py 와 동일)
# =====================================================================

def pixel_to_camera_point(u, v, depth_frames, win=DEPTH_WIN):
    """(u,v) 주변 (2*win+1)^2 픽셀 × 여러 프레임의 깊이 중앙값으로 3D 점(m)을 계산."""
    if not isinstance(depth_frames, (list, tuple)):
        depth_frames = [depth_frames]
    pts = []
    for df in depth_frames:
        d_intr = df.profile.as_video_stream_profile().get_intrinsics()
        for du in range(-win, win + 1):
            for dv in range(-win, win + 1):
                uu, vv = int(u) + du, int(v) + dv
                if 0 <= uu < d_intr.width and 0 <= vv < d_intr.height:
                    z = df.get_distance(uu, vv)
                    if DEPTH_MIN_M <= z <= DEPTH_MAX_M:
                        pts.append(rs.rs2_deproject_pixel_to_point(
                            d_intr, [float(uu), float(vv)], z))
    assert pts, (f"({u}, {v}) 에서 유효한 깊이가 없습니다 — 반사가 심하거나 "
                 f"너무 가깝거나({DEPTH_MIN_M}m 미만) 너무 먼({DEPTH_MAX_M}m 초과) 지점입니다.")
    return np.median(np.array(pts), axis=0)   # [X, Y, Z] (m)


def load_xy_correction(path):
    """픽셀 → 로봇 XY homography 를 읽는다."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    doc["_H"] = np.asarray(doc["matrix_3x3"], dtype=float)
    doc["_hull"] = np.asarray(doc["pixel_convex_hull"], dtype=np.float32)
    err = doc.get("fit_error_mm", {})
    print(f"XY 보정: {path}")
    print(f"  해상도 {doc['camera']['width']}x{doc['camera']['height']}, "
          f"{doc.get('sample_count', '?')}점, 평면 Z {doc.get('plane_z_mm', float('nan')):.1f}mm")
    print(f"  오차({err.get('kind', '?')}) rms {err.get('rms', float('nan')):.2f} / "
          f"max {err.get('max', float('nan')):.2f} mm")
    return doc


def pixel_to_robot_xy(doc, u, v):
    """픽셀 (u,v) → 로봇 base XY (mm). 보정 범위 밖이면 예외."""
    inside = cv2.pointPolygonTest(doc["_hull"], (float(u), float(v)), False) >= 0
    assert inside, (f"픽셀 ({u:.0f}, {v:.0f}) 이 보정한 영역 밖입니다 — "
                    "이 위치는 보정값이 없어 외삽이 됩니다. 물체를 작업영역 안으로 옮기세요.")
    t = doc["_H"] @ np.array([float(u), float(v), 1.0])
    off = doc.get("manual_offset_mm", {})
    x = float(t[0] / t[2]) + float(off.get("x", 0.0))
    y = float(t[1] / t[2]) + float(off.get("y", 0.0))
    b = doc["robot_xy_bounds_mm"]
    assert b["x_min"] <= x <= b["x_max"] and b["y_min"] <= y <= b["y_max"], (
        f"계산된 XY ({x:.0f}, {y:.0f}) 가 보정 측정범위 "
        f"X {b['x_min']:.0f}~{b['x_max']:.0f}, Y {b['y_min']:.0f}~{b['y_max']:.0f} 밖입니다.")
    return x, y


def coord_uv(u, v, tip, depth_frames, label=""):
    x, y = pixel_to_robot_xy(XY_CORRECTION, u, v)       # ← XY 는 여기서 결정된다
    plane_z = float(XY_CORRECTION.get("plane_z_mm", float("nan")))
    print(f"\n{label} 픽셀 ({u:.0f}, {v:.0f}) → base XY (mm): x={x:.0f}, y={y:.0f}")

    # depth + hand-eye 는 참고 출력용. 실패해도 진행한다.
    try:
        T_base_camera = np.load(CALIB_PATH)
        p_cam = pixel_to_camera_point(u, v, depth_frames)
        old_mm = (T_base_camera @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3] * 1000.0
        print(f"  (참고) 예전 depth+handeye 방식: x={old_mm[0]:.0f}, y={old_mm[1]:.0f}, "
              f"z={old_mm[2]:.0f}  → XY 차이 {np.hypot(old_mm[0] - x, old_mm[1] - y):.0f} mm")
        target_mm = [x, y, float(old_mm[2])]
    except AssertionError as exc:
        print(f"  (참고) depth 읽기 실패: {exc}")
        target_mm = [x, y, plane_z]

    r = float(np.hypot(x, y))
    print(f"  base에서 수평거리: {r:.0f} mm",
          "(OK)" if REACH_MIN_MM < r < REACH_MAX_MM else " 도달범위 밖! 좌표를 다시 확인하세요")
    assert REACH_MIN_MM < r < REACH_MAX_MM, (
        f"목표 수평거리 {r:.0f}mm 가 도달범위({REACH_MIN_MM}~{REACH_MAX_MM}mm) 밖 — 캘리브/좌표를 확인하세요.")

    new_tip = [x, y, float(tip[2])]
    return new_tip, target_mm


def gripper_tip():
    pose, _ = get_current_posx()
    R = Rot.from_euler("ZYZ", pose[3:6], degrees=True).as_matrix()
    tip = np.array(pose[:3]) + R @ np.array([0.0, 0.0, TCP_Z_MM])
    return tip, list(pose)


def flange_target_for_tip(tip_xyz, rxyz):
    R = Rot.from_euler("ZYZ", rxyz, degrees=True).as_matrix()
    f = np.array(tip_xyz) - R @ np.array([0.0, 0.0, TCP_Z_MM])
    return posx(float(f[0]), float(f[1]), float(f[2]), *[float(a) for a in rxyz])


ARRIVE_TOL_MM = 5.0   # 이보다 더 어긋나면 "못 간 것"으로 본다


def move_coord_offset(offset, target_tip, pose):
    tip_fix = [target_tip[0] + offset[0],
               target_tip[1] + offset[1],
               target_tip[2] + offset[2]]
    ret = movel(flange_target_for_tip(tip_fix, pose[3:6]),
                vel=VEL_L, acc=ACC_L, ref=DR_BASE, mod=DR_MV_MOD_ABS)
    # ★ 반환값을 반드시 본다. 예전에는 버렸는데, 그러면 도달 못 한 자리에서
    #   그대로 move_updown 이 175mm 를 내려가 버린다 (제자리 상하 반복의 원인).
    assert ret == 0, (
        f"movel 실패 (ret={ret}) — 목표 ({tip_fix[0]:.0f}, {tip_fix[1]:.0f}, {tip_fix[2]:.0f}). "
        "이 자세로는 도달할 수 없는 위치일 수 있습니다.")
    reached, _ = gripper_tip()
    gap = float(np.linalg.norm(np.array(reached) - np.array(tip_fix)))
    assert gap <= ARRIVE_TOL_MM, (
        f"목표에 못 갔습니다: 목표 ({tip_fix[0]:.0f}, {tip_fix[1]:.0f}, {tip_fix[2]:.0f}) / "
        f"실제 ({reached[0]:.0f}, {reached[1]:.0f}, {reached[2]:.0f}), 차이 {gap:.0f}mm. "
        "고정된 손목 방향으로는 도달 불가한 자리입니다 — 물체를 안쪽으로 옮기거나 "
        "그 위치는 건너뛰세요. (하강하지 않고 중단합니다)")


def move_above(target_tip, pose, label="목표"):
    # movej(posj(*POSE_READY), vel=VEL_J, acc=ACC_J)
    print(f"{label} 위로 이동: x={target_tip[0]:.0f}, y={target_tip[1]:.0f} "
          f"(높이 {target_tip[2]:.0f}mm 유지)")
    move_coord_offset(CALIB_OFFSET_MM, target_tip, pose)


def grip_open(wait=GRIP_WAIT):
    gripper_open()
    time.sleep(wait)


def grip_close(current=GRIP_CURRENT, wait=GRIP_WAIT):
    gripper_close(current=current)
    time.sleep(wait)


def gripper(current=GRIP_CURRENT):
    grip_open()
    grip_close(current)


def move_updown(mm, target_mm, return_position=True, gripper_bool=True):
    STEP_MM = mm

    if STEP_MM == 0:
        if gripper_bool:
            gripper(GRIP_CURRENT)
        return

    tip, _ = gripper_tip()
    new_z = tip[2] - STEP_MM
    assert new_z >= Z_MIN_MM, (f"그리퍼 끝 {tip[2]:.0f}mm 에서 더 내리면 최저높이({Z_MIN_MM}mm) 보다 "
                               "낮아집니다 — 정말 필요하면 Z_MIN_MM 를 신중히 낮추세요.")
    assert movel(posx(0, 0, -STEP_MM, 0, 0, 0), vel=VEL_Z, acc=ACC_Z,
                 ref=DR_BASE, mod=DR_MV_MOD_REL) == 0, "하강 movel 실패"
    tip, _ = gripper_tip()
    print(f"  그리퍼 끝 z = {tip[2]:.0f} mm | 대상 z(카메라 추정) = {target_mm[2]:.0f} mm")

    if gripper_bool:
        gripper(GRIP_CURRENT)

    if return_position:
        assert movel(posx(0, 0, STEP_MM, 0, 0, 0), vel=VEL_Z, acc=ACC_Z,
                     ref=DR_BASE, mod=DR_MV_MOD_REL) == 0, "상승 movel 실패"


def endpoint():
    pipeline.stop()
    node.destroy_node()
    rclpy.shutdown()
    print("종료 완료")



# =====================================================================
#  소스 특정 위치(옆으로 집기)
# =====================================================================

def sauce():
    base = [24.408, 83.762, -102.200, 129.727, -132.987, 36.662]
    movej(posj(41.289, 93.049, -10.838, 94.654, -131.822, -0.001), vel=120, acc=80)
    movel(
        posx(0, 0, -20, 0, 0, 0),
        vel=10,
        acc=20,
        ref=DR_BASE,
        mod=DR_MV_MOD_REL
    )
    # time.sleep(1.5)
    gripper_cmd(60, current=400)
    time.sleep(1.5)
    print("완료")

    # pouring 위치
    movej(posj(base), vel=120, acc=20)
    time.sleep(1.5)
    print("완료")
    #
    # # ==========================================
    # # 기준 위치로 이동
    # # ==========================================
    #
    # base = [24.408, 83.762, -102.200, 129.727, -132.987, 36.662]
    #
    # movej(
    #     posj(*base),
    #     vel=10,
    #     acc=20
    # )

    # ==========================================
    # 손목 털기 - J6 ±5도, 3회
    # ==========================================

    SHAKE_ANGLE = 15.0
    SHAKE_COUNT = 3

    SHAKE_VEL = 40
    SHAKE_ACC = 80

    for _ in range(SHAKE_COUNT):
        # J6 한쪽으로
        movej(
            posj(
                base[0],
                base[1],
                base[2],
                base[3],
                base[4],
                base[5] - SHAKE_ANGLE
            ),
            vel=SHAKE_VEL,
            acc=SHAKE_ACC
        )

        # J6 반대쪽으로
        movej(
            posj(
                base[0],
                base[1],
                base[2],
                base[3],
                base[4],
                base[5] + SHAKE_ANGLE
            ),
            vel=SHAKE_VEL,
            acc=SHAKE_ACC
        )

    # ==========================================
    # 원래 자세로 복귀
    # ==========================================

    movej(
        posj(*base),
        vel=SHAKE_VEL,
        acc=SHAKE_ACC
    )

    movej(posj(41.289, 93.049, -10.838, 94.654, -131.822, -0.001), vel=120, acc=80)
    # time.sleep(1.5)
    movel(
        posx(0, 0, -20, 0, 0, 0),
        vel=10,
        acc=20,
        ref=DR_BASE,
        mod=DR_MV_MOD_REL
    )
    gripper_cmd(750, current=400)
    time.sleep(1.5)
    print("완성!")



# =====================================================================
#  카메라 / YOLO — 주문마다 다시 찍고 다시 검출한다
# =====================================================================

def capture_and_detect(targets):
    """팔이 POSE_CAMERA_CLEAR 에 있다고 가정하고, 새로 촬영 후 검출한다.

    이전 주문이 물체를 옮겼을 수 있으므로, 주문 하나를 시작할 때마다
    반드시 이 함수를 다시 호출해서 그 순간의 실제 배치를 다시 읽는다.
    """
    for _ in range(N_FLUSH):   # 팔이 막 움직인 직후의 낡은 프레임 버퍼를 흘려보냄
        align.process(pipeline.wait_for_frames())

    color_frames = []
    depth_frames = []
    for i in range(N_COLOR):
        fs = align.process(pipeline.wait_for_frames())
        color_frames.append(np.asanyarray(fs.get_color_frame().get_data()))
        if i >= N_COLOR - N_DEPTH:
            d = fs.get_depth_frame()
            d.keep()
            depth_frames.append(d)

    centers = my_yolo.detect(color_frames, targets=targets, show=DETECT_SHOW)

    with state_lock:
        dets = []
        for lbl, points in centers.items():
            for (u, v) in points:
                dets.append({"cls": lbl, "name": my_yolo.class_name(lbl), "conf": 1.0})
        state["dets"] = dets
        state["ts"] = time.time()

    return centers, depth_frames


# =====================================================================
#  웹 서버 (FastAPI) — 큐에 라벨 목록만 쌓는다. 좌표 계산은 워커가 한다.
# =====================================================================

state = {"dets": [], "ts": 0.0}
state_lock = threading.Lock()
job_queue: "queue_mod.Queue[list[int]]" = queue_mod.Queue()

PICKABLE_IDS = tuple(sorted(i for i in my_yolo.CLASS_NAMES if i != BOWL_ID))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                    allow_methods=["*"], allow_headers=["*"])


@app.get("/classes")
def get_classes():
    return {str(i): my_yolo.CLASS_NAMES[i] for i in PICKABLE_IDS}


@app.get("/labels")
def get_labels():
    with state_lock:
        return {"dets": state["dets"], "ts": state["ts"]}


@app.post("/pick")
def post_pick(req: dict):
    indices = [int(i) for i in req.get("indices", [])]
    bad = [i for i in indices if i not in PICKABLE_IDS]
    if bad:
        return {"ok": False, "reason": f"알 수 없는 라벨: {bad}"}
    if not indices:
        return {"ok": False, "reason": "빈 주문입니다"}

    job_queue.put(indices)
    print(f"\n[web] 주문 접수: {indices}  (대기열 {job_queue.qsize()}건)")
    return {"ok": True, "count": len(indices), "queue_position": job_queue.qsize()}


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")


# =====================================================================
#  로봇 워커 — 큐에서 주문을 하나씩 꺼내 "재검출 → 집기"를 반복한다
# =====================================================================

def process_job(indices):
    print(f"\n===== 새 주문 시작: 라벨 {indices} =====")
    gripper_cmd(position=100, current=100)
    time.sleep(2)
    grip_open(wait=0)
    #### jjw : 촬영 전엔 항상 팔을 시야 밖으로
    # movej(posj(*POSE_CAMERA_CLEAR), vel=VEL_J, acc=ACC_J)

    targets = tuple(sorted(set(indices) | {BOWL_ID}))
    centers, depth_frames = capture_and_detect(targets)

    pick_items = [(lbl, u, v) for lbl in indices for (u, v) in centers.get(lbl, [])]
    for lbl in indices:
        if not centers.get(lbl):
            print(f"  ! 라벨 {lbl} ({my_yolo.class_name(lbl)}) 이 안 보여서 건너뜁니다")

    if not centers.get(BOWL_ID):
        print("  ! bowl 을 못 찾아서 이 주문을 건너뜁니다")
        return
    if not pick_items:
        print("  ! 집을 물체가 하나도 안 보여서 이 주문을 건너뜁니다")
        return

    bowl_uv = centers[BOWL_ID][0]

    print("준비자세로 이동 (movej)…")
    # movej(posj(*POSE_READY), vel=VEL_J, acc=ACC_J)
    tip, pose = gripper_tip()
    print(f"현재 그리퍼 끝(계산값): x={tip[0]:.0f}, y={tip[1]:.0f}, z={tip[2]:.0f} (mm)")

    targets_xyz = []
    for lbl, u, v in pick_items:
        name = my_yolo.class_name(lbl)
        targets_xyz.append((name,) + coord_uv(u, v, tip, depth_frames, name))

    bowl_tip, bowl_mm = coord_uv(bowl_uv[0], bowl_uv[1], tip, depth_frames, "bowl")

    if DRY_RUN:
        print("DRY_RUN=True — 좌표만 확인하고 이 주문은 실제로 집지 않습니다.")
        return

    for i, (name, obj_tip, obj_mm) in enumerate(targets_xyz, 1):
        print(f"\n----- {i}/{len(targets_xyz)}  {name} -----")
        move_above(obj_tip, pose, name)
        move_updown(H_PICK, obj_mm)

        print("bowl 로 옮겨 놓기")
        move_above(bowl_tip, pose, "bowl")
        move_updown(H_PLACE, bowl_mm, False, False)
        grip_open()
        move_updown(-H_PLACE, bowl_mm, False, False)
        print(f"{i}/{len(targets_xyz)} 완료")

    sauce()
    #### jjw : 촬영 전엔 항상 팔을 시야 밖으로
    #### jjw : 처음 clear 상태로 초기화
    movej(posj(*POSE_CAMERA_CLEAR), vel=VEL_J, acc=ACC_J)
    print(f"===== 주문 완료: {indices} =====")


def robot_worker():
    while True:
        indices = job_queue.get()
        try:
            process_job(indices)
        except Exception as e:
            movej(posj(*POSE_CAMERA_CLEAR), vel=VEL_J, acc=ACC_J)  # 다음 주문을 위해 복귀
            print(f"! 주문 처리 중 오류: {e} — 이 주문은 중단하고 다음 주문으로 넘어갑니다")


# =====================================================================
#  실행
# =====================================================================

#### jjw : 포트 선점 확인 — mock_server.py 를 안 끄고 이걸 띄우는 실수를 먼저 잡는다.
#          uvicorn 은 백그라운드 스레드라 포트가 막혀 있어도 메인 루프는 그대로 돌고
#          웹서버만 조용히 죽어서, 로봇은 멀쩡한데 브라우저만 "서버 없음"이 된다.
with socket.socket() as _s:
    assert _s.connect_ex(("127.0.0.1", SERVER_PORT)) != 0, (
        f"{SERVER_PORT} 번 포트가 이미 사용 중입니다 — "
        "mock_server.py 나 이 스크립트가 이미 떠 있지 않은지 확인하세요.")

pipeline = rs.pipeline()
profile = None

# ★ 보정한 해상도와 반드시 같아야 한다. 1280x720 은 4:3 이 아니라 화각 자체가
#   달라서 배율로 환산할 수도 없다. 그래서 보정 해상도를 첫 번째로 둔다.
XY_CORRECTION = load_xy_correction(XY_CORRECTION_PATH)
_CW = int(XY_CORRECTION["camera"]["width"])
_CH = int(XY_CORRECTION["camera"]["height"])
modes = [((_CW, _CH, 30), (_CW, _CH, 30)),
         ((_CW, _CH, 15), (_CW, _CH, 15))]
for (cw, ch, cf), (dw, dh, df) in modes:
    try:
        config = rs.config()
        config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cf)
        config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, df)
        profile = pipeline.start(config)
        print(f"카메라 시작: color {cw}x{ch}@{cf} / depth {dw}x{dh}@{df}")
        break
    except RuntimeError:
        continue
assert profile is not None, (
    "카메라 시작 실패 — 다른 프로세스가 카메라를 잡고 있지 않은지 먼저 확인하세요 "
    "(주피터 커널, realsense-viewer, 이전에 띄운 스크립트). 그 다음 USB 연결 확인.")

try:
    movej(posj(*POSE_CAMERA_CLEAR), vel=VEL_J, acc=ACC_J)

    align = rs.align(rs.stream.color)
    for _ in range(N_WARMUP):
        align.process(pipeline.wait_for_frames())

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"intrinsics: fx={intr.fx:.1f}, fy={intr.fy:.1f}, ppx={intr.ppx:.1f}, ppy={intr.ppy:.1f}")
    assert (intr.width, intr.height) == (_CW, _CH), (
        f"컬러 해상도 {intr.width}x{intr.height} 가 보정 해상도 {_CW}x{_CH} 와 다릅니다 — "
        "픽셀 좌표가 달라져서 XY 보정을 그대로 쓸 수 없습니다.")

    threading.Thread(target=run_server, daemon=True).start()
    print(f"\n웹 서버 시작: http://localhost:{SERVER_PORT}")
    print("아티팩트(음식 집기 콘솔)에서 이 주소로 연결하세요.\n")

    threading.Thread(target=robot_worker, daemon=True).start()
    print("주문 대기 중… (Ctrl+C 로 종료)")

    while True:
        time.sleep(1.0)

except KeyboardInterrupt:
    print("\n종료 신호 받음")
finally:
    endpoint()
