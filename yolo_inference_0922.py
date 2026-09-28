#!/usr/bin/env python3
"""학습된 샐러드 YOLO 모델로 지정 라벨의 박스 중심 좌표를 뽑아내는 모듈.

이 파일은 **카메라를 열지 않습니다.** 프레임(numpy BGR 배열)을 인자로 받습니다.
move_and_grip_0921.py 처럼 이미 pyrealsense pipeline 을 들고 있는 쪽에서
그 color 배열을 그대로 넘기면 됩니다 — 장치 점유 충돌도 없고, uv 좌표가
depth_frame 과 같은 해상도·같은 시점이라 좌표 변환이 그대로 맞습니다.

검출 전략: N프레임 누적 투표 + 중앙값
    여러 프레임을 추론해 같은 물체끼리 묶은 뒤, min_hits 회 이상 잡힌
    클러스터만 채택합니다. 좌표는 그 클러스터 중심들의 중앙값이라
    프레임 간 흔들림(1~3px)과 깜빡이는 오검출이 함께 걸러집니다.

사용 예:
    import yolo_inference_0922 as yolo

    frames = [np.asanyarray(align.process(pipeline.wait_for_frames())
                            .get_color_frame().get_data()) for _ in range(10)]
    centers = yolo.detect(frames)
    # {0: [(334, 194), ...], 1: [...], 3: [...], 5: [...]}

단독 확인용:
    python3 yolo_inference_0922.py sample1.jpg sample2.jpg --show
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


WEIGHTS_ROOT = Path("/home/jaewoo/doosan_ws/src/mini_project/runs/detect/train-2/weights")
DEFAULT_MODEL = WEIGHTS_ROOT / "best.pt"
WINDOW_NAME = "Salad YOLO - detection result"

########## label index
CLASS_NAMES: dict[int, str] = {
    0: "tomato",
    1: "cheese",
    2: "lettuce",
    3: "berry",
    4: "onion",
    5: "bowl",
    6: "brocoli",
    7: "carrot",
    8: "bacon",
}
##########

# 좌표를 뽑을 대상 라벨
TARGET_IDS: tuple[int, ...] = (0, 1, 3, 5)

# 기본 투표 파라미터
DEFAULT_FRAMES = 10      # 권장 프레임 수 (호출하는 쪽에서 이 개수만큼 넘겨주세요)
DEFAULT_MIN_HITS = 3     # 이 횟수 이상 잡힌 물체만 채택
DEFAULT_MERGE_PX = 40.0  # 중심이 이 거리 안이면 같은 물체로 간주

_MODEL_CACHE: dict[str, YOLO] = {}


def load_model(model_path: Path | str = DEFAULT_MODEL) -> YOLO:
    """모델을 한 번만 로딩하고 캐시해서 재사용한다."""
    key = str(Path(model_path).expanduser().resolve())
    if key not in _MODEL_CACHE:
        if not Path(key).is_file():
            raise FileNotFoundError(f"학습 모델이 없습니다: {key}")
        print("모델 로딩:", key)
        model = YOLO(key)
        print("모델 클래스:", model.names)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def class_name(label: int, model: YOLO | None = None) -> str:
    if model is not None:
        names = model.names
        if isinstance(names, dict) and label in names:
            return str(names[label])
        if isinstance(names, list) and 0 <= label < len(names):
            return str(names[label])
    return CLASS_NAMES.get(label, str(label))


@dataclass
class _Cluster:
    """여러 프레임에 걸쳐 같은 물체로 묶인 검출들."""

    label: int
    us: list[float] = field(default_factory=list)
    vs: list[float] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)
    frame_ids: set[int] = field(default_factory=set)

    def add(self, u: float, v: float, conf: float, frame_id: int) -> None:
        self.us.append(u)
        self.vs.append(v)
        self.confs.append(conf)
        self.frame_ids.add(frame_id)

    @property
    def center(self) -> tuple[int, int]:
        return int(round(float(np.median(self.us)))), int(round(float(np.median(self.vs))))

    @property
    def conf(self) -> float:
        return float(np.median(self.confs))

    @property
    def hits(self) -> int:
        return len(self.frame_ids)


def _frame_detections(
    result: object, targets: tuple[int, ...]
) -> list[tuple[int, float, float, float]]:
    """한 프레임의 결과에서 (label, conf, cx, cy) 목록을 뽑는다."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return []

    xywh = boxes.xywh.detach().cpu().numpy()          # (cx, cy, w, h) 픽셀
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()

    dets: list[tuple[int, float, float, float]] = []
    for (cx, cy, _w, _h), label, score in zip(xywh, cls, conf):
        label = int(label)
        if label in targets:
            dets.append((label, float(score), float(cx), float(cy)))
    return dets


def _assign_to_clusters(
    clusters: list[_Cluster],
    dets: list[tuple[int, float, float, float]],
    frame_id: int,
    merge_px: float,
) -> None:
    """한 프레임의 검출들을 기존 클러스터에 1:1로 배정한다 (가까운 쌍 우선)."""
    pairs: list[tuple[float, int, int]] = []
    for di, (label, _score, cx, cy) in enumerate(dets):
        for ci, cluster in enumerate(clusters):
            if cluster.label != label or frame_id in cluster.frame_ids:
                continue
            cu, cv = cluster.center
            dist = math.hypot(cx - cu, cy - cv)
            if dist <= merge_px:
                pairs.append((dist, di, ci))
    pairs.sort()

    used_dets: set[int] = set()
    used_clusters: set[int] = set()
    for _dist, di, ci in pairs:
        if di in used_dets or ci in used_clusters:
            continue
        label, score, cx, cy = dets[di]
        clusters[ci].add(cx, cy, score, frame_id)
        used_dets.add(di)
        used_clusters.add(ci)

    # 어디에도 안 붙은 검출은 새 물체로 등록
    for di, (label, score, cx, cy) in enumerate(dets):
        if di in used_dets:
            continue
        cluster = _Cluster(label=label)
        cluster.add(cx, cy, score, frame_id)
        clusters.append(cluster)


def annotate(
    frame: np.ndarray,
    centers: dict[int, list[tuple[int, int]]],
    model: YOLO | None = None,
    result: object | None = None,
) -> np.ndarray:
    """확정된 중심 좌표를 프레임 위에 표시한 이미지를 만든다."""
    vis = result.plot(labels=True, conf=True, boxes=True) if result is not None else frame.copy()
    for label, points in centers.items():
        name = class_name(label, model)
        for rank, (u, v) in enumerate(points):
            cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
            cv2.putText(
                vis,
                f"{label}:{name}#{rank}",
                (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return vis


def detect(
    frames: np.ndarray | list[np.ndarray],
    model_path: Path | str = DEFAULT_MODEL,
    targets: tuple[int, ...] = TARGET_IDS,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
    min_hits: int = DEFAULT_MIN_HITS,
    merge_px: float = DEFAULT_MERGE_PX,
    show: bool = False,
    save_path: Path | str | None = None,
    verbose: bool = True,
) -> dict[int, list[tuple[int, int]]]:
    """프레임(들)에서 대상 라벨의 박스 중심 좌표를 뽑는다.

    Args:
        frames: BGR numpy 배열 한 장, 또는 여러 장의 리스트(투표에 사용).
        targets: 좌표를 뽑을 라벨 인덱스. 기본 (0, 1, 3, 5).
        min_hits: 몇 프레임 이상에서 잡혀야 채택할지. 프레임 수보다 크면 자동으로 줄임.
        merge_px: 프레임 간 같은 물체로 묶을 중심 거리 임계값(픽셀).
        show: True 면 결과 창을 띄우고 키 입력을 기다린다. 기본 False.
        save_path: 주면 결과 이미지를 그 경로에 저장한다 (show 없이 확인할 때 유용).

    Returns:
        {label: [(u, v), ...]} — 각 라벨의 중심 좌표를 신뢰도 내림차순으로.
        대상 라벨은 검출이 없어도 빈 리스트로 항상 키가 들어있다.
    """
    if isinstance(frames, np.ndarray):
        frames = [frames]
    frames = list(frames)
    if not frames:
        raise ValueError("프레임이 비어 있습니다.")
    if not 0.0 < conf <= 1.0:
        raise ValueError("conf는 0보다 크고 1 이하여야 합니다.")
    if imgsz < 32:
        raise ValueError("imgsz 값을 확인하세요.")
    if merge_px <= 0:
        raise ValueError("merge_px는 0보다 커야 합니다.")

    targets = tuple(dict.fromkeys(int(t) for t in targets))
    effective_min_hits = max(1, min(int(min_hits), len(frames)))
    if verbose and effective_min_hits != min_hits:
        print(f"min_hits={min_hits} → 프레임 수({len(frames)})에 맞춰 {effective_min_hits} 로 조정")

    model = load_model(model_path)

    clusters: list[_Cluster] = []
    last_result = None
    for frame_id, frame in enumerate(frames):
        result = model.predict(
            source=frame, conf=conf, imgsz=imgsz, device=device, verbose=False
        )[0]
        last_result = result
        _assign_to_clusters(
            clusters, _frame_detections(result, targets), frame_id, merge_px
        )

    accepted = [c for c in clusters if c.hits >= effective_min_hits]
    accepted.sort(key=lambda c: -c.conf)

    centers: dict[int, list[tuple[int, int]]] = {label: [] for label in targets}
    for cluster in accepted:
        centers[cluster.label].append(cluster.center)

    if verbose:
        print(f"프레임 {len(frames)}장 / 채택 기준 {effective_min_hits}회 이상")
        for cluster in accepted:
            u, v = cluster.center
            print(
                f"  검출 [{cluster.label}] {class_name(cluster.label, model):<8}"
                f" uv=({u:4d}, {v:4d})  conf={cluster.conf:.2f}  hits={cluster.hits}/{len(frames)}"
            )
        rejected = [c for c in clusters if c.hits < effective_min_hits]
        for cluster in rejected:
            u, v = cluster.center
            print(
                f"  (제외) [{cluster.label}] {class_name(cluster.label, model):<8}"
                f" uv=({u:4d}, {v:4d})  conf={cluster.conf:.2f}  hits={cluster.hits}/{len(frames)}"
            )
        for label in targets:
            if not centers[label]:
                print(f"  ! 라벨 {label} ({class_name(label, model)}) 미검출")

    if show or save_path is not None:
        vis = annotate(frames[-1], centers, model, last_result)
        if save_path is not None:
            save_path = Path(save_path).expanduser()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(save_path), vis):
                raise RuntimeError(f"이미지 저장에 실패했습니다: {save_path}")
            print("결과 이미지 저장:", save_path)
        if show:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            print("결과 창 — 아무 키나 누르면 닫힙니다.")
            cv2.waitKey(0)
            cv2.destroyWindow(WINDOW_NAME)

    return centers


def flatten(
    centers: dict[int, list[tuple[int, int]]], order: tuple[int, ...] = TARGET_IDS
) -> list[list[int]]:
    """{label: [(u,v)]} → [[u, v], ...] (move_and_grip 의 uv_list 형식)."""
    uv_list: list[list[int]] = []
    for label in order:
        uv_list.extend([u, v] for u, v in centers.get(label, []))
    return uv_list


def _main() -> int:
    """단독 확인용 — 이미지 파일을 받아 검출 결과를 출력한다 (카메라 사용 안 함)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="이미지 파일로 검출 좌표를 확인합니다. 카메라는 열지 않습니다."
    )
    parser.add_argument("images", nargs="+", type=Path, help="같은 장면을 찍은 이미지들")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    parser.add_argument("--merge-px", type=float, default=DEFAULT_MERGE_PX)
    parser.add_argument("--targets", type=int, nargs="+", default=list(TARGET_IDS))
    parser.add_argument("--show", action="store_true", help="결과 창을 띄웁니다 (기본: 안 띄움)")
    parser.add_argument("--save", type=Path, default=None, help="결과 이미지 저장 경로")
    args = parser.parse_args()

    try:
        frames = []
        for path in args.images:
            frame = cv2.imread(str(path.expanduser()))
            if frame is None:
                raise FileNotFoundError(f"이미지를 읽지 못했습니다: {path}")
            frames.append(frame)

        centers = detect(
            frames,
            model_path=args.model,
            targets=tuple(args.targets),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            min_hits=args.min_hits,
            merge_px=args.merge_px,
            show=args.show,
            save_path=args.save,
        )
        print("\n결과:", centers)
        print("uv_list:", flatten(centers))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
