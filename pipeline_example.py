from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from trash_detector import (
    TrackState,
    detector_recycle_probability,
    fuse_decision,
    get_recyclability,
    load_class_thresholds,
    load_recyclability_map,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = BASE_DIR / "togo.jpeg"
DEFAULT_OUTPUT_DIR = BASE_DIR / "pipeline_example_outputs"
MODEL_PATH = BASE_DIR / "runs/detect/runs/detect/taco_basic_fixed/weights/best.pt"
VERIFIER_PATH = BASE_DIR / "yolov8n-cls.pt"
MAPPING_PATH = BASE_DIR / "recyclability_map.json"
THRESHOLD_PATH = BASE_DIR / "class_thresholds.json"


STATUS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "recyclable": (0, 180, 0),
    "not_recyclable": (0, 0, 220),
    "unknown": (0, 180, 255),
}


def _resolve_input_image(image_path: Path) -> Path:
    if image_path.exists():
        return image_path

    candidates = sorted(BASE_DIR.glob("pipeline_example.*"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find pipeline example image at {image_path} or any pipeline_example.* file in {BASE_DIR}."
    )


def _draw_box(frame, box: List[float], text: str, status: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
    cv2.rectangle(frame, (x1, text_y - th - 6), (x1 + tw + 4, text_y + 4), color, -1)
    cv2.putText(frame, text, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _save_image(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write image: {path}")


def _predict_verifier_probability_on_crop(verifier: Optional[YOLO], crop) -> float:
    if verifier is None:
        return 0.5

    results = verifier.predict(source=crop, verbose=False)
    if not results:
        return 0.5

    result = results[0]
    probs = getattr(result, "probs", None)
    if probs is None or probs.top1conf is None:
        return 0.5

    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf.item())
    class_name = str(result.names.get(top1_idx, "")).strip().lower()

    if class_name in {"recyclable", "recycle", "yes"}:
        return top1_conf
    if class_name in {"not_recyclable", "trash", "non_recyclable", "no"}:
        return 1.0 - top1_conf
    return 0.5


def run_generic_yolo_pass(image_path: Path, output_dir: Path, model_path: Optional[Path] = None, conf: float = 0.25) -> Optional[str]:
    """Run a generic YOLOv8 model on the original image and save an annotated image.

    If `model_path` is not provided, attempt to use `yolov8n.pt` in the repo root.
    Returns the path to the saved image or None on failure.
    """
    try:
        model_file = Path(model_path) if model_path else BASE_DIR / "yolov8n.pt"
        generic = YOLO(str(model_file))
    except Exception:
        # Fallback to default model string (may download)
        generic = YOLO("yolov8n.pt")

    frame = cv2.imread(str(image_path))
    if frame is None:
        return None

    annotated = frame.copy()
    try:
        results = generic.predict(source=frame, conf=conf, verbose=False)
    except Exception:
        results = []

    if results:
        res = results[0]
        for box in getattr(res, "boxes", []):
            try:
                cls_id = int(box.cls.item())
                det_conf = float(box.conf.item())
                if str(res.names.get(cls_id, str(cls_id))).strip() == "person":
                    continue  # skip person detections in generic pass
                label = "Container"
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
            except Exception:
                continue

            x1, y1, x2, y2 = xyxy
            color = (255, 128, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            txt = f"{label} {det_conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.rectangle(annotated, (x1, ty - th - 6), (x1 + tw + 4, ty + 4), color, -1)
            cv2.putText(annotated, txt, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = output_dir / "00_generic.jpg"
    try:
        _save_image(out_path, annotated)
        return str(out_path)
    except Exception:
        return None


def run_image_pipeline(
    image_path: Path = DEFAULT_IMAGE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    conf_threshold: float = 0.3,
) -> Dict[str, object]:
    image_path = _resolve_input_image(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(str(MODEL_PATH))
    verifier = YOLO(str(VERIFIER_PATH)) if VERIFIER_PATH.exists() else None
    mapping = load_recyclability_map(MAPPING_PATH)
    class_thresholds = load_class_thresholds(THRESHOLD_PATH)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Run a generic YOLOv8 pass and save annotated image as 00_generic.jpg
    generic_output = run_generic_yolo_pass(image_path, output_dir)

    detector_frame = frame.copy()
    detector_results = detector.predict(source=frame, conf=conf_threshold, verbose=False)

    detections: List[Dict[str, object]] = []
    if detector_results:
        result = detector_results[0]
        for box in result.boxes:
            cls_id = int(box.cls.item())
            det_conf = float(box.conf.item())
            label = str(result.names.get(cls_id, str(cls_id)))
            class_threshold = class_thresholds.get(label.strip().lower(), conf_threshold)
            if det_conf < class_threshold:
                continue

            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            base_status = get_recyclability(label, mapping)
            detector_prob = detector_recycle_probability(base_status, det_conf)
            detections.append(
                {
                    "label": label,
                    "confidence": det_conf,
                    "box": xyxy,
                    "base_status": base_status,
                    "detector_prob": detector_prob,
                }
            )

            annotated_label = f"{label} {det_conf:.2f}"
            _draw_box(detector_frame, xyxy, annotated_label, base_status)

    detector_output = output_dir / "01_detector.jpg"
    _save_image(detector_output, detector_frame)

    if not detections:
        final_frame = frame.copy()
        cv2.putText(
            final_frame,
            "No detections found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        final_output = output_dir / "03_final.jpg"
        _save_image(final_output, final_frame)
        return {
            "image_path": str(image_path),
            "generic_output": generic_output,
            "detector_output": str(detector_output),
            "crop_output": None,
            "final_output": str(final_output),
            "detections": [],
        }

    best_detection = max(detections, key=lambda det: float(det["confidence"]))
    x1, y1, x2, y2 = [int(v) for v in best_detection["box"]]
    x1 = max(0, min(x1, frame.shape[1] - 1))
    y1 = max(0, min(y1, frame.shape[0] - 1))
    x2 = max(0, min(x2, frame.shape[1] - 1))
    y2 = max(0, min(y2, frame.shape[0] - 1))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Selected detection box is invalid after clipping.")

    crop = frame[y1:y2, x1:x2].copy()
    crop_with_border = crop.copy()
    cv2.rectangle(
        crop_with_border,
        (0, 0),
        (crop_with_border.shape[1] - 1, crop_with_border.shape[0] - 1),
        (0, 180, 255),
        2,
    )
    cv2.putText(
        crop_with_border,
        f"crop: {best_detection['label']}",
        (8, min(28, crop_with_border.shape[0] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    crop_output = output_dir / "02_crop.jpg"
    _save_image(crop_output, crop_with_border)

    verifier_prob = _predict_verifier_probability_on_crop(verifier, crop)

    track = TrackState(
        id=1,
        box=best_detection["box"],
        last_seen=0,
        label=str(best_detection["label"]),
    )
    final_status, final_prob = fuse_decision(
        track=track,
        detector_prob=float(best_detection["detector_prob"]),
        verifier_prob=verifier_prob,
        recyclable_accept=0.62,
        trash_accept=0.40,
        unknown_frames=4,
    )

    final_frame = frame.copy()
    final_text = f"{best_detection['label']} | {final_status} | {final_prob:.2f}"
    _draw_box(final_frame, best_detection["box"], final_text, final_status)
    final_output = output_dir / "03_final.jpg"
    _save_image(final_output, final_frame)

    return {
        "image_path": str(image_path),
        "generic_output": generic_output,
        "detector_output": str(detector_output),
        "crop_output": str(crop_output),
        "final_output": str(final_output),
        "best_detection": {
            "label": best_detection["label"],
            "confidence": round(float(best_detection["confidence"]), 3),
            "base_status": best_detection["base_status"],
            "detector_prob": round(float(best_detection["detector_prob"]), 3),
            "verifier_prob": round(float(verifier_prob), 3),
            "final_status": final_status,
            "final_prob": round(float(final_prob), 3),
            "box": [round(float(v), 1) for v in best_detection["box"]],
        },
        "detections": detections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recycling pipeline on pipeline_example.jpeg and save each stage.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Path to the input image.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for saved pipeline images.")
    parser.add_argument("--conf", type=float, default=0.3, help="Detector confidence threshold.")
    args = parser.parse_args()

    result = run_image_pipeline(image_path=args.image, output_dir=args.output_dir, conf_threshold=args.conf)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
