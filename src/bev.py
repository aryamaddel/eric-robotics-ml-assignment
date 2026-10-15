"""Bird's-eye view via the ground-plane homography (inverse perspective mapping).

For a calibrated camera with rotation R and position t (camera -> ego frame, ego z up,
ground at z = 0), a ground point (x, y, 0) maps to camera coordinates R^T (p - t), so

    s [u v 1]^T = K [ R^T e1 | R^T e2 | -R^T t ] [x y 1]^T  =  H [x y 1]^T

H is a 3x3 homography between the ground plane and the image. H^-1 sends a pixel on the
ground (e.g. the bottom-centre of a detection box, where the object touches the road) to
metric ground coordinates, giving a size-independent distance estimate that we compare with
the pinhole box-height estimate. Warping the image with H gives the top-down view.

    python src/bev.py --weights best.pt --nusc data/nuscenes --out runs/bev --n 6
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from distance import Camera, DistanceEstimator
from eval_distance import load_tables, quat_to_rot


def ground_homography(K, R_cam_to_ego, t_cam_in_ego):
    """Homography mapping ego ground-plane coords (x fwd, y left, metres) to image pixels."""
    Rt = np.asarray(R_cam_to_ego).T
    t = np.asarray(t_cam_in_ego, dtype=float)
    return np.asarray(K) @ np.column_stack([Rt[:, 0], Rt[:, 1], -Rt @ t])


class BEV:
    def __init__(self, H, x_range=(0.0, 40.0), y_range=(-15.0, 15.0), res=0.05):
        self.H, self.res = H, res
        self.x_range, self.y_range = x_range, y_range
        self.w = int((y_range[1] - y_range[0]) / res)
        self.h = int((x_range[1] - x_range[0]) / res)
        # BEV pixel (col, row) -> ego ground (x, y): forward is up, left is left
        self.A = np.array([[0, -res, x_range[1]], [-res, 0, y_range[1]], [0, 0, 1]], dtype=float)

    def warp(self, img):
        return cv2.warpPerspective(img, self.H @ self.A, (self.w, self.h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    def pixel_to_ground(self, u, v):
        p = np.linalg.inv(self.H) @ np.array([u, v, 1.0])
        return p[:2] / p[2]

    def ground_to_bev(self, x, y):
        p = np.linalg.inv(self.A) @ np.array([x, y, 1.0])
        return int(p[0]), int(p[1])

    def draw_grid(self, bev):
        for d in range(5, int(self.x_range[1]) + 1, 5):
            r = self.ground_to_bev(d, 0)[1]
            cv2.line(bev, (0, r), (self.w, r), (255, 255, 255), 1)
            cv2.putText(bev, f"{d} m", (5, r - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return bev


def front_camera_frames(t, n):
    frames = [sd for sd in t["sample_data"] if sd["is_key_frame"] and sd["filename"].startswith("samples/CAM_FRONT/")]
    frames.sort(key=lambda sd: sd["timestamp"])
    step = max(1, len(frames) // n)
    return frames[::step][:n]


def main():
    from ultralytics import YOLO

    from classes import CLASSES
    from predict import COLORS, annotate

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--nusc", type=Path, required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", type=Path, default=Path("runs/bev"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--frames", nargs="*", help="specific sample_data filenames instead of evenly spaced ones")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    t = load_tables(args.nusc, args.version)
    model = YOLO(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)
    frames = ([sd for sd in t["sample_data"] if sd["filename"] in set(args.frames)] if args.frames
              else front_camera_frames(t, args.n))
    records = []
    for sd in frames:
        cs = t["calibrated_sensor"][sd["calibrated_sensor_token"]]
        K, R, tc = np.array(cs["camera_intrinsic"]), quat_to_rot(cs["rotation"]), np.array(cs["translation"])
        bev = BEV(ground_homography(K, R, tc))
        est = DistanceEstimator(Camera.from_K(K, height_m=float(tc[2])))

        img = cv2.imread(str(args.nusc / sd["filename"]))
        r = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = []
        top = bev.draw_grid(bev.warp(img))
        for cls_id, box in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            cls = CLASSES[int(cls_id)]
            d = est.estimate(cls, box, image_size=(img.shape[1], img.shape[0]))
            # Ground contact = bottom-centre of the box. Meaningless for stop signs (on poles above ground).
            gx, gy = bev.pixel_to_ground((box[0] + box[2]) / 2, box[3])
            ground_range = float(np.hypot(gx - tc[0], gy - tc[1]))
            dets.append({"cls": cls, "box": box, "range": d["range"], "depth": d["depth"], "method": d["method"],
                         "ground_xy": [round(float(gx), 2), round(float(gy), 2)], "ground_range": round(ground_range, 2)})
            if cls != "stop_sign" and bev.x_range[0] < gx < bev.x_range[1] and bev.y_range[0] < gy < bev.y_range[1]:
                c = bev.ground_to_bev(gx, gy)
                cv2.circle(top, c, 7, COLORS[cls], -1)
                cv2.putText(top, f"{d['range']:.1f}m", (c[0] + 9, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COLORS[cls], 2, cv2.LINE_AA)
        cam = annotate(img.copy(), dets)
        scale = top.shape[0] / cam.shape[0]
        cam = cv2.resize(cam, (int(cam.shape[1] * scale), top.shape[0]))
        name = Path(sd["filename"]).stem
        cv2.imwrite(str(args.out / f"{name}_bev.jpg"), np.hstack([cam, top]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        records.append({"file": sd["filename"], "detections": dets})
    (args.out / "bev_detections.json").write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} BEV composites to {args.out}")


if __name__ == "__main__":
    main()
