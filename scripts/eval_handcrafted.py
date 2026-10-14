"""Compare hand-crafted detectors / verifier with the CNN on the per-source test splits.

  - standalone colour+shape detectors: cones on nuImages test, stop signs on MTSD test
  - CNN stop-sign detections with and without the hand-crafted stop-sign verifier (MTSD test)

AP50 is computed VOC-style (all-point interpolated precision/recall, IoU >= 0.5, one match
per ground-truth box), identically for every method.

    python scripts/eval_handcrafted.py --data data/yolo --weights weights/best.pt --out results/handcrafted
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from navperc.classes import CLASS_ID
from navperc.handcrafted import detect_cones, detect_stop_signs, verify_stop_sign


def load_gt(label_path: Path, cls: int, w: int, h: int):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        c, cx, cy, bw, bh = map(float, line.split())
        if int(c) == cls:
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def ap50(dets, gts):
    """dets: list of (image, score, box); gts: {image: [box, ...]}."""
    n_gt = sum(len(v) for v in gts.values())
    used = {k: [False] * len(v) for k, v in gts.items()}
    tp = []
    for img, score, box in sorted(dets, key=lambda d: -d[1]):
        best, j = 0.0, -1
        for i, g in enumerate(gts.get(img, [])):
            o = iou(box, g)
            if o > best and not used[img][i]:
                best, j = o, i
        if best >= 0.5:
            used[img][j] = True
            tp.append(1)
        else:
            tp.append(0)
    if not tp or n_gt == 0:
        return {"AP50": 0.0, "n_gt": n_gt, "n_det": len(dets)}
    tp = np.array(tp)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / (ctp + cfp)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"AP50": round(ap, 4), "max_recall": round(float(rec[-1]), 4), "n_gt": n_gt, "n_det": len(dets)}


def run_handcrafted(images, detector, label_dir, cls):
    dets, gts, t = [], {}, 0.0
    for p in images:
        img = cv2.imread(str(p))
        gts[p.name] = load_gt(label_dir / f"{p.stem}.txt", cls, img.shape[1], img.shape[0])
        t0 = time.perf_counter()
        found = detector(img)
        t += time.perf_counter() - t0
        dets += [(p.name, s, b) for s, b in found]
    return dets, gts, 1000 * t / len(images)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", type=Path, default=Path("results/handcrafted"))
    args = ap.parse_args()
    test_imgs, test_lbls = args.data / "images/test", args.data / "labels/test"
    nuim = sorted(test_imgs.glob("nuim_*.jpg"))
    mtsd = sorted(test_imgs.glob("mtsd_*.jpg"))
    out = {}

    dets, gts, ms = run_handcrafted(mtsd, detect_stop_signs, test_lbls, CLASS_ID["stop_sign"])
    out["handcrafted_stop_sign_mtsd"] = {**ap50(dets, gts), "ms_per_image": round(ms, 1)}
    dets, gts, ms = run_handcrafted(nuim, detect_cones, test_lbls, CLASS_ID["cone"])
    out["handcrafted_cone_nuimages"] = {**ap50(dets, gts), "ms_per_image": round(ms, 1)}

    from ultralytics import YOLO

    model = YOLO(args.weights)
    cnn, cnn_verified, gts = [], [], {}
    for p in mtsd:
        img = cv2.imread(str(p))
        gts[p.name] = load_gt(test_lbls / f"{p.stem}.txt", CLASS_ID["stop_sign"], img.shape[1], img.shape[0])
        r = model.predict(img, conf=0.01, imgsz=args.imgsz, classes=[CLASS_ID["stop_sign"]], verbose=False)[0]
        for s, b in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            cnn.append((p.name, s, b))
            x0, y0, x1, y1 = map(int, b)
            if verify_stop_sign(img[max(0, y0):y1, max(0, x0):x1]):
                cnn_verified.append((p.name, s, b))
    out["cnn_stop_sign_mtsd"] = ap50(cnn, gts)
    out["cnn_plus_verifier_stop_sign_mtsd"] = ap50(cnn_verified, gts)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "handcrafted_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
