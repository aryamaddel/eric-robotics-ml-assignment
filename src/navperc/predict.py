"""Detect cones / barriers / stop signs and annotate each with its estimated distance.

    python -m navperc.predict --weights best.pt --camera configs/camera_nuscenes_front.yaml \
        --source path/to/images_or_video --out runs/predict

Runs the baseline .pt weights or optimized OpenVINO model on CPU.
Labels are drawn as "Cone, 1.5m". GPU comparisons use the separate Modal workflow.
"""

import argparse
import json
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

from navperc.classes import CLASSES, DISPLAY
from navperc.distance import Camera, DistanceEstimator

COLORS = {"cone": (0, 140, 255), "barrier": (0, 220, 255), "stop_sign": (40, 40, 230)}


def load_camera(path: Path, width: int, height: int) -> Camera:
    """Camera yaml: either K (3x3) with calib_width/height, or hfov_deg. Optional height_m."""
    cfg = yaml.safe_load(path.read_text())
    if "K" in cfg:
        cam = Camera.from_K(cfg["K"], cfg.get("height_m"))
        sx, sy = width / cfg["calib_width"], height / cfg["calib_height"]
        return cam.scaled(sx, sy) if (sx, sy) != (1, 1) else cam
    return Camera.from_fov(width, height, cfg["hfov_deg"], cfg.get("height_m"))


def annotate(frame, detections):
    scale = max(0.6, frame.shape[1] / 1600 * 0.7)  # keep labels readable on large images
    thick = max(2, round(scale * 2))
    for d in detections:
        x0, y0, x1, y1 = map(int, d["box"])
        color = COLORS[d["cls"]]
        label = f"{DISPLAY[d['cls']]}, {d['range']:.1f}m"
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thick)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y0 - 4 if y0 - th - 8 > 0 else y1 + th + 4
        cv2.rectangle(frame, (x0, ty - th - 4), (x0 + tw + 4, ty + base - 2), color, -1)
        cv2.putText(frame, label, (x0 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return frame


def detect(model, estimator_for, frame, conf, imgsz):
    h, w = frame.shape[:2]
    est = estimator_for(w, h)
    r = model.predict(frame, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
    out = []
    for cls_id, score, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        cls = CLASSES[int(cls_id)]
        d = est.estimate(cls, box, image_size=(w, h))
        out.append({"cls": cls, "conf": round(score, 3), "box": [round(v, 1) for v in box],
                    "depth": round(d["depth"], 2), "range": round(d["range"], 2), "method": d["method"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--camera", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="image, directory of images, or video")
    ap.add_argument("--out", type=Path, default=Path("runs/predict"))
    ap.add_argument("--imgsz", type=int, choices=[640, 960], default=960,
                    help="960 for accuracy, 640 for the speed comparison")
    args = ap.parse_args()

    model = YOLO(args.weights, task="detect")
    cache = {}

    def estimator_for(w, h):
        if (w, h) not in cache:
            cache[(w, h)] = DistanceEstimator(load_camera(args.camera, w, h))
        return cache[(w, h)]

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(str(args.source))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {args.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detect(model, estimator_for, frame, 0.35, args.imgsz)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(args.out / f"{args.source.stem}_annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotate(frame, dets))
        cap.release()
        if writer:
            writer.release()
    else:
        files = sorted(args.source.glob("*.jpg")) + sorted(args.source.glob("*.png")) if args.source.is_dir() else [args.source]
        for f in files:
            frame = cv2.imread(str(f))
            if frame is None:
                raise ValueError(f"Cannot read image: {f}")
            dets = detect(model, estimator_for, frame, 0.35, args.imgsz)
            cv2.imwrite(str(args.out / f.name), annotate(frame, dets))
            results[f.name] = dets
        (args.out / "detections.json").write_text(json.dumps(results, indent=2))
    print(f"wrote results to {args.out}")


if __name__ == "__main__":
    main()
