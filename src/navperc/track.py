"""Track cones / barriers / stop signs across frames with sparse optical flow.

The detector runs only every `--detect-every` frames. In between, each track's box is
propagated with pyramidal Lucas-Kanade optical flow: corner features inside the box are
tracked from the previous frame and the box is shifted by their median motion and rescaled
by the change in their spread. On detection frames, propagated tracks are matched to new
detections by IoU (greedy), which keeps identities stable and corrects drift.

This cuts detector calls by ~N x (an edge-compute saving) and gives each object a distance
history, from which a closing speed is estimated by a linear fit over the last second.

    python -m navperc.track --weights best.pt --nusc data/nuscenes --scene scene-0553 --out runs/track
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from navperc.classes import CLASSES, DISPLAY
from navperc.distance import Camera, DistanceEstimator
from navperc.eval_distance import load_tables

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
COLORS = {"cone": (0, 140, 255), "barrier": (0, 220, 255), "stop_sign": (40, 40, 230)}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def flow_box(prev_gray, gray, box):
    """Propagate a box with LK flow; returns the new box or None if tracking failed."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = gray.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    mask = np.zeros_like(prev_gray)
    mask[y0:y1, x0:x1] = 255
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=30, qualityLevel=0.01, minDistance=2, mask=mask)
    if pts is None or len(pts) < 3:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK)
    good = st.ravel() == 1
    if good.sum() < 3:
        return None
    p0, p1 = pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
    dx, dy = np.median(p1 - p0, axis=0)
    s0 = np.median(np.linalg.norm(p0 - p0.mean(0), axis=1))
    s1 = np.median(np.linalg.norm(p1 - p1.mean(0), axis=1))
    scale = float(np.clip(s1 / s0, 0.8, 1.25)) if s0 > 1 else 1.0
    cx, cy = (box[0] + box[2]) / 2 + dx, (box[1] + box[3]) / 2 + dy
    bw, bh = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
    return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]


class Tracker:
    def __init__(self, iou_thr=0.3, max_age=6):
        self.tracks, self.next_id, self.iou_thr, self.max_age = [], 1, iou_thr, max_age

    def propagate(self, prev_gray, gray):
        for t in self.tracks:
            nb = flow_box(prev_gray, gray, t["box"])
            if nb is None:
                t["age"] += 1
            else:
                t["box"] = nb

    def update(self, detections):
        pairs = sorted(((iou(t["box"], d["box"]), ti, di) for ti, t in enumerate(self.tracks)
                        for di, d in enumerate(detections) if t["cls"] == d["cls"]), reverse=True)
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < self.iou_thr or ti in used_t or di in used_d:
                continue
            self.tracks[ti].update(box=detections[di]["box"], age=0)
            used_t.add(ti)
            used_d.add(di)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["age"] += 1
        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks.append({"id": self.next_id, "cls": d["cls"], "box": d["box"], "age": 0, "history": []})
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]


def closing_speed(history, window_s=1.0):
    """m/s, positive when approaching; linear fit of range vs time over the last window."""
    if len(history) < 4:
        return None
    t_end = history[-1][0]
    pts = np.array([(t, r) for t, r in history if t >= t_end - window_s])
    if len(pts) < 4 or np.ptp(pts[:, 0]) < 0.3:
        return None
    return float(-np.polyfit(pts[:, 0], pts[:, 1], 1)[0])


def scene_frames(t, scene_name, channel="CAM_FRONT"):
    scene = next(s for s in t["scene"].values() if s["name"] == scene_name)
    samples = {tok for tok, s in t["sample"].items() if s["scene_token"] == scene["token"]}
    frames = [sd for sd in t["sample_data"] if sd["sample_token"] in samples and f"/{channel}/" in sd["filename"]]
    return sorted(frames, key=lambda sd: sd["timestamp"])


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--nusc", type=Path, required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/track"))
    ap.add_argument("--detect-every", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    t = load_tables(args.nusc, args.version)
    frames = [sd for sd in scene_frames(t, args.scene) if (args.nusc / sd["filename"]).exists()][: args.max_frames]
    if not frames:
        raise SystemExit(f"no frames found for {args.scene}; extract samples/ and sweeps/ CAM_FRONT")
    model = YOLO(args.weights)
    tracker = Tracker()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = frames[0]["timestamp"]
    duration = (frames[-1]["timestamp"] - t0) / 1e6
    fps = max(1.0, (len(frames) - 1) / duration) if duration > 0 else 10.0
    writer, prev_gray, detector_calls, log = None, None, 0, []

    for i, sd in enumerate(frames):
        cs = t["calibrated_sensor"][sd["calibrated_sensor_token"]]
        est = DistanceEstimator(Camera.from_K(cs["camera_intrinsic"], height_m=float(cs["translation"][2])))
        img = cv2.imread(str(args.nusc / sd["filename"]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ts = (sd["timestamp"] - t0) / 1e6
        if prev_gray is not None:
            tracker.propagate(prev_gray, gray)
        if i % args.detect_every == 0:
            r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            detector_calls += 1
            tracker.update([{"cls": CLASSES[int(c)], "box": b} for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())])
        for tr in tracker.tracks:
            if tr["age"] > 0:
                continue
            d = est.estimate(tr["cls"], tr["box"], image_size=(img.shape[1], img.shape[0]))
            tr["history"].append((ts, d["range"]))
            v = closing_speed(tr["history"])
            x0, y0, x1, y1 = map(int, tr["box"])
            label = f"{DISPLAY[tr['cls']]} #{tr['id']}, {d['range']:.1f}m" + (f", {v:+.1f}m/s" if v is not None else "")
            cv2.rectangle(img, (x0, y0), (x1, y1), COLORS[tr["cls"]], 2)
            cv2.putText(img, label, (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[tr["cls"]], 2, cv2.LINE_AA)
            log.append({"t": round(ts, 3), "id": tr["id"], "cls": tr["cls"], "range": round(d["range"], 2),
                        "closing_speed": None if v is None else round(v, 2)})
        tag = "DETECT" if i % args.detect_every == 0 else "flow"
        cv2.putText(img, f"{args.scene}  t={ts:5.2f}s  {tag}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(str(args.out / f"{args.scene}_tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (img.shape[1], img.shape[0]))
        writer.write(img)
        prev_gray = gray
    writer.release()
    summary = {"scene": args.scene, "frames": len(frames), "detector_calls": detector_calls,
               "video_fps": round(fps, 1), "tracks": tracker.next_id - 1}
    (args.out / f"{args.scene}_tracks.json").write_text(json.dumps({"summary": summary, "log": log}, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
