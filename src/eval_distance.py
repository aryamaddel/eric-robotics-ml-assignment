"""Evaluate monocular distance estimates against nuScenes 3D ground truth.

For every key-frame camera image in nuScenes (mini), each traffic cone / barrier 3D box is
transformed global -> ego -> camera, giving the true depth and range. Its 8 corners are
projected with the camera intrinsics to obtain the 2D box a perfect detector would output,
and our pinhole estimator is run on that box. This isolates the *geometry* error from
detector error; with `--weights` the detector's own boxes are matched to GT by IoU instead
(end-to-end error).

The nuScenes boxes also carry real object heights, so the script also reports class height
priors fitted from data, evaluated leave-one-scene-out (no fitting on the evaluated scene).

    python src/eval_distance.py --nusc data/nuscenes --version v1.0-mini --out runs/distance_eval
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from distance import DEFAULT_HEIGHTS, Camera, DistanceEstimator

NUSC_TO_OURS = {"movable_object.trafficcone": "cone", "movable_object.barrier": "barrier"}
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def box_corners(center, size, rotation):
    """8 corners (3, 8) of a nuScenes box; size = (width, length, height), x = length axis."""
    w, l, h = size
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    return quat_to_rot(rotation) @ np.vstack([x, y, z]) + np.asarray(center)[:, None]


def load_tables(root: Path, version: str):
    d = root / version
    load = lambda n: json.loads((d / f"{n}.json").read_text())
    by_token = lambda n: {r["token"]: r for r in load(n)}
    return {
        "category": by_token("category"),
        "instance": by_token("instance"),
        "sensor": by_token("sensor"),
        "calibrated_sensor": by_token("calibrated_sensor"),
        "ego_pose": by_token("ego_pose"),
        "sample": by_token("sample"),
        "scene": by_token("scene"),
        "sample_data": load("sample_data"),
        "sample_annotation": load("sample_annotation"),
    }


def gt_objects(t, min_visibility=2, width=1600, height=900):
    """Yield one record per (camera image, cone/barrier) with GT distance and projected 2D box."""
    anns = defaultdict(list)
    for a in t["sample_annotation"]:
        name = t["category"][t["instance"][a["instance_token"]]["category_token"]]["name"]
        if name in NUSC_TO_OURS and int(a["visibility_token"]) >= min_visibility:
            anns[a["sample_token"]].append((NUSC_TO_OURS[name], a))

    for sd in t["sample_data"]:
        if not sd["is_key_frame"] or not sd["filename"].startswith("samples/CAM"):
            continue
        cs = t["calibrated_sensor"][sd["calibrated_sensor_token"]]
        ep = t["ego_pose"][sd["ego_pose_token"]]
        K = np.array(cs["camera_intrinsic"])
        R_ep, t_ep = quat_to_rot(ep["rotation"]), np.array(ep["translation"])
        R_cs, t_cs = quat_to_rot(cs["rotation"]), np.array(cs["translation"])
        scene = t["sample"][sd["sample_token"]]["scene_token"]
        for cls, a in anns[sd["sample_token"]]:
            corners = box_corners(a["translation"], a["size"], a["rotation"])
            pts = R_cs.T @ (R_ep.T @ (corners - t_ep[:, None]) - t_cs[:, None])
            center = R_cs.T @ (R_ep.T @ (np.array(a["translation"]) - t_ep) - t_cs)
            if (pts[2] < 0.5).any():
                continue  # partly behind the camera
            uv = (K @ pts)[:2] / pts[2]
            x0, y0, x1, y1 = uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max()
            if x1 < 0 or y1 < 0 or x0 >= width or y0 >= height:
                continue
            box = (max(x0, 0), max(y0, 0), min(x1, width - 1), min(y1, height - 1))
            # Fully-visible-in-frame flag: truncated boxes have unreliable pixel height.
            in_frame = x0 >= 0 and y0 >= 0 and x1 < width and y1 < height
            yield {
                "file": sd["filename"], "scene": scene, "cls": cls, "box": box, "K": K,
                "cam_height": float(cs["translation"][2]), "in_frame": in_frame,
                "gt_depth": float(center[2]), "gt_range": float(np.linalg.norm(center)),
                "gt_height": float(a["size"][2]),
            }


def summarize(rows, key="range"):
    if not rows:
        return {}
    est = np.array([r[f"est_{key}"] for r in rows])
    gt = np.array([r[f"gt_{key}"] for r in rows])
    err = est - gt
    rel = np.abs(err) / gt
    return {
        "n": len(rows),
        "MAE_m": round(float(np.abs(err).mean()), 2),
        "median_abs_err_m": round(float(np.median(np.abs(err))), 2),
        "bias_m": round(float(err.mean()), 2),
        "median_rel_err": round(float(np.median(rel)), 3),
        "within_10pct": round(float((rel < 0.10).mean()), 3),
        "within_20pct": round(float((rel < 0.20).mean()), 3),
    }


def estimate_rows(objs, heights, camera="calibrated"):
    rows = []
    for o in objs:
        cam = (Camera.from_fov(1600, 900, 70) if camera == "generic"
               else Camera.from_K(o["K"], o["cam_height"]))
        est = DistanceEstimator(cam, heights)
        e = est.estimate(o["cls"], o["box"], image_size=(1600, 900))
        rows.append({**o, "est_range": e["range"], "est_depth": e["depth"], "method": e["method"]})
    return rows


def report(rows, key="range"):
    out = {}
    for cls in ("cone", "barrier"):
        cr = [r for r in rows if r["cls"] == cls]
        out[cls] = {"all": summarize(cr, key)}
        for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 1e9)):
            out[cls][f"{lo}-{hi if hi < 1e9 else 'inf'}m"] = summarize([r for r in cr if lo <= r[f"gt_{key}"] < hi], key)
    return out


def fit_heights(objs):
    heights = dict(DEFAULT_HEIGHTS)
    for cls in ("cone", "barrier"):
        hs = [o["gt_height"] for o in objs if o["cls"] == cls]
        if hs:
            heights[cls] = round(float(np.median(hs)), 3)
    return heights


def match_detections(objs, nusc: Path, weights, imgsz=960, conf=0.25, iou_thr=0.3):
    """Replace each GT box by the best-overlapping same-class detector box (greedy IoU matching).

    GT 2D boxes are projected 3D cuboids, which are looser than a detector's tight box, hence the
    modest IoU threshold. Returns (matched objects, number of GT objects).
    """
    import cv2
    from ultralytics import YOLO

    from classes import CLASSES

    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

    model = YOLO(str(weights))
    by_file = defaultdict(list)
    for o in objs:
        by_file[o["file"]].append(o)
    matched = []
    for f, gts in by_file.items():
        r = model.predict(cv2.imread(str(nusc / f)), conf=conf, imgsz=imgsz, verbose=False)[0]
        preds = [(CLASSES[int(c)], b) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
        pairs = sorted(((iou(g["box"], b), gi, pi) for gi, g in enumerate(gts) for pi, (c, b) in enumerate(preds)
                        if c == g["cls"]), reverse=True)
        used_g, used_p = set(), set()
        for score, gi, pi in pairs:
            if score < iou_thr or gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            matched.append({**gts[gi], "box": tuple(preds[pi][1]), "match_iou": round(score, 3)})
    return matched, len(objs)


def loso(objs, t, rows_fn):
    """Leave-one-scene-out: height priors for each scene are fitted on the other scenes only."""
    rows, per_scene = [], {}
    for sc in sorted({o["scene"] for o in objs}):
        heights = fit_heights([o for o in objs if o["scene"] != sc])
        per_scene[t["scene"][sc]["name"]] = heights
        rows += rows_fn([o for o in objs if o["scene"] == sc], heights)
    return rows, per_scene


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nusc", type=Path, required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", type=Path, default=Path("runs/distance_eval"))
    ap.add_argument("--weights", help="also evaluate end-to-end with this detector's boxes")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    t = load_tables(args.nusc, args.version)
    objs = [o for o in gt_objects(t) if o["in_frame"]]
    loso_rows, per_scene_heights = loso(objs, t, estimate_rows)

    result = {
        "objects": len(objs),
        "scenes": len({o["scene"] for o in objs}),
        "height_priors": {"default": DEFAULT_HEIGHTS, "fitted_all_scenes": fit_heights(objs),
                          "fitted_leave_one_scene_out": per_scene_heights},
        "gt_boxes": {
            "default_priors": report(estimate_rows(objs, DEFAULT_HEIGHTS)),
            "generic_70deg_default_priors": report(estimate_rows(objs, DEFAULT_HEIGHTS, "generic")),
            "fitted_priors_leave_one_scene_out": report(loso_rows),
        },
    }

    if args.weights:
        # End-to-end: detector boxes (all GT objects, incl. ones cut off by the image border).
        all_objs = list(gt_objects(t))
        matched, n_gt = match_detections(all_objs, args.nusc, args.weights, args.imgsz)
        det_loso, _ = loso(matched, t, estimate_rows)
        result["detector_boxes"] = {
            "weights": str(args.weights),
            "gt_objects": n_gt,
            "matched": len(matched),
            "match_rate": round(len(matched) / max(n_gt, 1), 3),
            "default_priors": report(estimate_rows(matched, DEFAULT_HEIGHTS)),
            "generic_70deg_default_priors": report(estimate_rows(matched, DEFAULT_HEIGHTS, "generic")),
            "fitted_priors_leave_one_scene_out": report(det_loso),
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "distance_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
