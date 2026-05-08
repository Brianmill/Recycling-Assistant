import argparse
import importlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO


STATUS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "recyclable": (0, 180, 0),      # green
    "not_recyclable": (0, 0, 220),  # red
    "unknown": (0, 180, 255),       # orange
}

DEFAULT_CLASS_THRESHOLDS = {
    "paper": 0.55,
    "plastic": 0.45,
    "glass": 0.55,
    "metal": 0.50,
    "trash": 0.35,
}


@dataclass
class TrackState:
    id: int
    box: List[float]
    last_seen: int
    label: str
    display_box: Optional[List[float]] = None
    history: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    recycle_scores: Deque[float] = field(default_factory=lambda: deque(maxlen=8))
    unknown_streak: int = 0


class SerialSorterController:
    def __init__(self, port: str, baud_rate: int = 115200, cooldown_seconds: float = 1.5) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.cooldown_seconds = cooldown_seconds
        self.last_command: Optional[str] = None
        self.last_sent_at = 0.0
        self._serial = None

    def open(self) -> None:
        try:
            serial_module = importlib.import_module("serial")
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Install requirements.txt before using --serial-port."
            ) from exc

        self._serial = serial_module.Serial(self.port, self.baud_rate, timeout=1)
        time.sleep(2.0)
        self.send("CENTER", force=True)

    def send(self, command: str, force: bool = False) -> bool:
        normalized = command.strip().upper()
        if not normalized or self._serial is None:
            return False

        now = time.time()
        if not force and normalized == self.last_command and (now - self.last_sent_at) < self.cooldown_seconds:
            return False

        try:
            self._serial.write(f"{normalized}\n".encode("utf-8"))
            self._serial.flush()
            self.last_command = normalized
            self.last_sent_at = now
            print(f"Sent ESP32 command: {normalized}")
            return True
        except Exception as exc:
            print(f"Warning: failed to send '{normalized}' to {self.port}: {exc}")
            return False

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None


def load_recyclability_map(map_path: Path) -> Dict[str, str]:
    if not map_path.exists():
        raise FileNotFoundError(
            f"Could not find mapping file: {map_path}. "
            "Create it or pass --mapping with a valid path."
        )

    with map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {str(k).strip().lower(): str(v).strip().lower() for k, v in data.items()}


def load_class_thresholds(path: Optional[Path]) -> Dict[str, float]:
    if path is None:
        return dict(DEFAULT_CLASS_THRESHOLDS)

    if not path.exists():
        raise FileNotFoundError(f"Could not find threshold file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
    for key, value in data.items():
        thresholds[str(key).strip().lower()] = float(value)
    return thresholds


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def smooth_box(previous: Optional[List[float]], current: List[float], alpha: float = 0.65) -> List[float]:
    if previous is None:
        return list(current)

    return [alpha * prev + (1.0 - alpha) * curr for prev, curr in zip(previous, current)]


def assign_tracks(tracks: Dict[int, TrackState], detections: List[Dict[str, object]], frame_index: int, iou_threshold: float, max_missed_frames: int) -> None:
    unmatched_tracks = set(tracks.keys())

    for det in detections:
        det_box = det["box"]
        best_id = None
        best_iou = 0.0

        for track_id in list(unmatched_tracks):
            current_iou = iou_xyxy(tracks[track_id].box, det_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_id = track_id

        if best_id is not None and best_iou >= iou_threshold:
            track = tracks[best_id]
            track.box = det_box
            track.display_box = smooth_box(track.display_box, det_box)
            track.last_seen = frame_index
            track.label = str(det["label"])
            det["track_id"] = best_id
            unmatched_tracks.remove(best_id)
        else:
            new_id = (max(tracks.keys()) + 1) if tracks else 1
            tracks[new_id] = TrackState(
                id=new_id,
                box=det_box,
                last_seen=frame_index,
                label=str(det["label"]),
                display_box=list(det_box),
            )
            det["track_id"] = new_id

    stale = [
        track_id
        for track_id, track in tracks.items()
        if frame_index - track.last_seen > max_missed_frames
    ]
    for track_id in stale:
        del tracks[track_id]


def get_recyclability(label: str, mapping: Dict[str, str]) -> str:
    normalized = label.strip().lower()
    return mapping.get(normalized, mapping.get("default", "unknown"))


def detector_recycle_probability(base_status: str, confidence: float) -> float:
    if base_status == "recyclable":
        return confidence
    if base_status == "not_recyclable":
        return 1.0 - confidence
    return 0.5


def verifier_recycle_probability(verifier: Optional[YOLO], frame, box: List[float], default_probability: float = 0.5) -> float:
    if verifier is None:
        return default_probability

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return default_probability

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return default_probability

    results = verifier.predict(source=crop, verbose=False)
    if not results:
        return default_probability

    result = results[0]
    probs = getattr(result, "probs", None)
    if probs is None or probs.top1conf is None:
        return default_probability

    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf.item())
    class_name = str(result.names.get(top1_idx, "")).strip().lower()

    if class_name in {"recyclable", "recycle", "yes"}:
        return top1_conf
    if class_name in {"not_recyclable", "trash", "non_recyclable", "no"}:
        return 1.0 - top1_conf
    return default_probability


def fuse_decision(track: TrackState, detector_prob: float, verifier_prob: float, recyclable_accept: float, trash_accept: float, unknown_frames: int) -> Tuple[str, float]:
    temporal_prob = sum(track.recycle_scores) / len(track.recycle_scores) if track.recycle_scores else 0.5
    final_prob = 0.5 * detector_prob + 0.25 * verifier_prob + 0.25 * temporal_prob

    if final_prob >= recyclable_accept:
        status = "recyclable"
        track.unknown_streak = 0
    elif final_prob <= trash_accept:
        status = "not_recyclable"
        track.unknown_streak = 0
    else:
        track.unknown_streak += 1
        status = "not_recyclable" if track.unknown_streak >= unknown_frames else "unknown"

    track.recycle_scores.append(final_prob)
    track.history.append(status)
    return status, final_prob


def select_sort_command(detections: List[Dict[str, object]]) -> Optional[str]:
    best_detection: Optional[Dict[str, object]] = None
    best_score = -1.0

    for det in detections:
        status = str(det.get("status", "unknown"))
        if status not in {"recyclable", "not_recyclable"}:
            continue

        score = float(det.get("final_prob", 0.0))
        if score > best_score:
            best_score = score
            best_detection = det

    if best_detection is None:
        return None

    return "LEFT" if str(best_detection.get("status")) == "recyclable" else "RIGHT"



def draw_detection(frame, box: List[float], label: str, confidence: float, status: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {confidence:.2f} | {status}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20

    cv2.rectangle(frame, (x1, text_y - th - 6), (x1 + tw + 4, text_y + 4), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 2, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run_webcam_detector(
    model_path: str,
    verifier_path: Optional[str],
    mapping_path: Path,
    threshold_path: Optional[Path],
    camera_index: int,
    conf_threshold: float,
    recyclable_accept: float,
    trash_accept: float,
    unknown_frames: int,
    tracker_iou: float,
    tracker_max_missed: int,
    serial_port: Optional[str],
    serial_baud: int,
    serial_cooldown: float,
) -> None:
    
    model = YOLO(model_path)
    verifier = YOLO(verifier_path) if verifier_path else None
    mapping = load_recyclability_map(mapping_path)
    class_thresholds = load_class_thresholds(threshold_path)
    sorter: Optional[SerialSorterController] = None

    if serial_port:
        sorter = SerialSorterController(serial_port, baud_rate=serial_baud, cooldown_seconds=serial_cooldown)
        try:
            sorter.open()
        except Exception as exc:
            print(f"Warning: could not open ESP32 serial port '{serial_port}': {exc}")
            sorter = None

    tracks: Dict[int, TrackState] = {}
    frame_index = 0
    t0 = time.time()
    fps = 0.0

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. "
            "Try --camera 1 or check if another app is using it."
        )

    print("Starting webcam trash detection. Press 'q' to quit.")

    try:
        while True:
            frame_index += 1
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from webcam.")
                break

            results = model.predict(source=frame, conf=conf_threshold, verbose=False)
            detections: List[Dict[str, object]] = []

            if results:
                result = results[0]
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    det_conf = float(box.conf.item())
                    label = result.names.get(cls_id, str(cls_id))
                    xyxy = box.xyxy[0].tolist()

                    class_threshold = class_thresholds.get(str(label).strip().lower(), conf_threshold)
                    if det_conf < class_threshold:
                        continue

                    detections.append({
                        "label": str(label),
                        "conf": det_conf,
                        "box": xyxy,
                    })

            assign_tracks(tracks=tracks, detections=detections, frame_index=frame_index, iou_threshold=tracker_iou, max_missed_frames=tracker_max_missed)

            for det in detections:
                label = str(det["label"])
                det_conf = float(det["conf"])
                xyxy = det["box"]
                track_id = int(det["track_id"])
                track = tracks[track_id]

                base_status = get_recyclability(label, mapping)
                detector_prob = detector_recycle_probability(base_status, det_conf)
                verifier_prob = verifier_recycle_probability(verifier, frame, xyxy)
                status, final_prob = fuse_decision(track=track, detector_prob=detector_prob, verifier_prob=verifier_prob, recyclable_accept=recyclable_accept, trash_accept=trash_accept, unknown_frames=unknown_frames,)

                det["status"] = status
                det["final_prob"] = final_prob

                if status == "recyclable":
                    show_label = f"{label}#{track_id}"
                elif status == "not_recyclable":
                    show_label = f"trash#{track_id}"
                else:
                    show_label = f"unknown#{track_id}"
                draw_detection(frame, track.display_box or xyxy, show_label, final_prob, status)

            if sorter is not None:
                command = select_sort_command(detections)
                if command is not None:
                    sorter.send(command)

            dt = time.time() - t0
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            t0 = time.time()

            cv2.putText(
                frame,
                f"q: quit | fps: {fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("YOLOv8 Trash Recyclability Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if sorter is not None:
            sorter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect trash with YOLOv8 and classify recyclability in real time."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="taco_best.pt",
        help=(
            "Path to your YOLOv8 weights (.pt). "
            "Defaults to the retrained TACO model in runs/detect/.../best.pt."
        ),
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="kaggle_best.pt",
        help=(
            "Optional secondary classifier model path (.pt) trained on "
            "recyclable vs not_recyclable for crop verification."
        ),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("recyclability_map.json"),
        help="Path to JSON mapping labels to recyclable/not_recyclable/unknown.",
    )
    parser.add_argument(
        "--class-thresholds",
        type=Path,
        default=Path("class_thresholds.json"),
        help="Optional JSON file with per-class confidence thresholds.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam index, usually 0 for default camera.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Minimum YOLO confidence before class-specific threshold filtering.",
    )
    parser.add_argument(
        "--recycle-accept",
        type=float,
        default=0.62,
        help="Final recyclable score threshold.",
    )
    parser.add_argument(
        "--trash-accept",
        type=float,
        default=0.40,
        help="Final trash score threshold (below this means trash).",
    )
    parser.add_argument(
        "--unknown-frames",
        type=int,
        default=4,
        help="How many consecutive unknown frames before falling back to trash.",
    )
    parser.add_argument(
        "--tracker-iou",
        type=float,
        default=0.35,
        help="IoU threshold to associate detections across frames.",
    )
    parser.add_argument(
        "--tracker-max-missed",
        type=int,
        default=12,
        help="How many frames a track can be missing before being dropped.",
    )
    parser.add_argument(
        "--serial-port",
        type=str,
        default="COM3",
        help="Optional ESP32 serial port for servo commands, for example COM5 or /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--serial-baud",
        type=int,
        default=115200,
        help="Baud rate for the ESP32 serial link.",
    )
    parser.add_argument(
        "--serial-cooldown",
        type=float,
        default=1.5,
        help="Minimum seconds between repeated servo commands.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_webcam_detector(
        model_path=args.model,
        verifier_path=args.verifier,
        mapping_path=args.mapping,
        threshold_path=args.class_thresholds,
        camera_index=args.camera,
        conf_threshold=args.conf,
        recyclable_accept=args.recycle_accept,
        trash_accept=args.trash_accept,
        unknown_frames=args.unknown_frames,
        tracker_iou=args.tracker_iou,
        tracker_max_missed=args.tracker_max_missed,
        serial_port=args.serial_port or None,
        serial_baud=args.serial_baud,
        serial_cooldown=args.serial_cooldown,
    )


if __name__ == "__main__":
    main()
