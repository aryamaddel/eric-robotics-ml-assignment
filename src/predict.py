"""Detect cones / barriers / stop signs and annotate each with its estimated distance.

    python src/predict.py --weights best.pt \
        --source path/to/images_or_video --out runs/predict

Runs the baseline .pt weights or optimized OpenVINO model on CPU.
Labels are drawn as "Cone, 1.5m". GPU inference uses src/modal_app.py::infer.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from classes import CLASSES, DISPLAY
from distance import Camera, DistanceEstimator

COLORS = {"cone": (0, 140, 255), "barrier": (0, 220, 255), "stop_sign": (40, 40, 230)}


def load_camera(width: int, height: int) -> Camera:
    """Fixed nuScenes front-camera calibration, scaled to the input resolution."""
    # First CAM_FRONT calibration in nuScenes v1.0-mini, at 1600 x 900.
    # Height above ground enables the ground-plane truncation fallback.
    cam = Camera.from_K([
        [1266.417203046554, 0.0, 816.2670197447984],
        [0.0, 1266.417203046554, 491.50706579294757],
        [0.0, 0.0, 1.0],
    ], height_m=1.51)
    return cam.scaled(width / 1600, height / 900)


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


def detect(model, estimator_for, frame, conf, imgsz, device="cpu"):
    h, w = frame.shape[:2]
    est = estimator_for(w, h)
    r = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    out = []
    for cls_id, score, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        cls = CLASSES[int(cls_id)]
        d = est.estimate(cls, box, image_size=(w, h))
        out.append({"cls": cls, "conf": round(score, 3), "box": [round(v, 1) for v in box],
                    "depth": round(d["depth"], 2), "range": round(d["range"], 2), "method": d["method"]})
    return out


def run(weights, source: Path, out: Path, imgsz=960, device="cpu"):
    """Shared annotated image/video inference for local CPU and Modal GPU."""
    started = time.perf_counter()
    model = YOLO(weights, task="detect")
    cache = {}

    def estimator_for(w, h):
        if (w, h) not in cache:
            cache[(w, h)] = DistanceEstimator(load_camera(w, h))
        return cache[(w, h)]

    out.mkdir(parents=True, exist_ok=True)
    results = {}
    if source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        writer = None
        frame_times = []
        loop_started = time.perf_counter()
        while True:
            frame_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            dets = detect(model, estimator_for, frame, 0.35, imgsz, device)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out / f"{source.stem}_annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("Cannot open output video writer")
            writer.write(annotate(frame, dets))
            frame_times.append(time.perf_counter() - frame_started)
        cap.release()
        if writer:
            writer.release()
        elapsed = time.perf_counter() - loop_started
        total = time.perf_counter() - started
        n = len(frame_times)
        if not n:
            raise ValueError(f"No frames decoded: {source}")
        steady = frame_times[10:]
        metrics = {
            "source": str(source), "device": str(device), "imgsz": imgsz,
            "frames": n, "source_fps": fps, "video_duration_s": n / fps,
            "pipeline_seconds": elapsed, "pipeline_fps": n / elapsed,
            "including_model_load_seconds": total, "including_model_load_fps": n / total,
            "realtime_speedup": (n / fps) / elapsed,
            "warmup_frames_excluded_from_steady": min(10, n),
            "steady_pipeline_fps": len(steady) / sum(steady) if steady else None,
            "timing_scope": "decode, detect, distance, draw, encode; pipeline includes first-frame setup and writer finalization; excludes cloud startup, upload and download",
        }
        (out / "performance.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2))
    else:
        files = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png")) if source.is_dir() else [source]
        for f in files:
            frame = cv2.imread(str(f))
            if frame is None:
                raise ValueError(f"Cannot read image: {f}")
            dets = detect(model, estimator_for, frame, 0.35, imgsz, device)
            cv2.imwrite(str(out / f.name), annotate(frame, dets))
            results[f.name] = dets
        (out / "detections.json").write_text(json.dumps(results, indent=2))
    print(f"wrote results to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--source", type=Path, required=True, help="image, directory of images, or video")
    ap.add_argument("--out", type=Path, default=Path("runs/predict"))
    ap.add_argument("--imgsz", type=int, choices=[640, 960], default=960,
                    help="960 for accuracy, 640 for the speed comparison")
    args = ap.parse_args()

    run(args.weights, args.source, args.out, args.imgsz)


if __name__ == "__main__":
    main()
