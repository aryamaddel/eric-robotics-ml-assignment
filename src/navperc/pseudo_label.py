"""Fill in classes that a source dataset does not annotate, using a teacher detector.

Merging datasets with disjoint label sets creates false negatives: nuImages has stop signs
in frame that nobody labeled, so training would actively teach the model to ignore them.
We run a teacher over those images and add its confident detections as labels.

  - stop signs in nuImages: COCO-pretrained YOLO (COCO class 11 "stop sign")
  - cones/barriers in MTSD: our own first-round model (self-training), if time permits

Only boxes that do not overlap an existing label of the same class are added, and only
the train split is modified; val/test stay human-labelled and are evaluated per source.
"""

import argparse
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from navperc.classes import CLASS_ID


def read_labels(path: Path):
    if not path.exists():
        return np.zeros((0, 5))
    rows = [list(map(float, l.split())) for l in path.read_text().splitlines() if l.strip()]
    return np.array(rows).reshape(-1, 5)


def iou_xywh(a, b):
    """IoU between one box a and boxes b, all normalized (cx, cy, w, h)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    return inter / (a[2] * a[3] + b[:, 2] * b[:, 3] - inter + 1e-9)


def pseudo_label(weights, image_dir: Path, label_dir: Path, class_map: dict, conf: float, imgsz: int,
                 prefix: str = "", batch: int = 32, device=None):
    """class_map: teacher class id -> our class id."""
    model = YOLO(weights)
    images = sorted(p for p in image_dir.glob(f"{prefix}*.jpg"))
    added, touched = 0, 0
    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        results = model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz, classes=list(class_map), batch=batch,
                                device=device, verbose=False)
        for img_path, r in zip(chunk, results):
            if len(r.boxes) == 0:
                continue
            lbl_path = label_dir / f"{img_path.stem}.txt"
            existing = read_labels(lbl_path)
            new_lines = []
            for cls, box in zip(r.boxes.cls.cpu().numpy(), r.boxes.xywhn.cpu().numpy()):
                ours = class_map[int(cls)]
                same = existing[existing[:, 0] == ours][:, 1:]
                if len(same) and iou_xywh(box, same).max() > 0.3:
                    continue
                new_lines.append(f"{ours} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
            if new_lines:
                with lbl_path.open("a") as f:
                    f.write("\n".join(new_lines) + "\n")
                added += len(new_lines)
                touched += 1
    print(f"pseudo-labelled {added} boxes in {touched}/{len(images)} images")
    return added


COCO_STOP_SIGN = 11


def mask_regions(weights, image_dir: Path, label_dir: Path, teacher_classes, protect_classes, conf: float,
                 imgsz: int, prefix: str = "", batch: int = 16, device=None, pad: float = 0.15):
    """Grey out teacher detections instead of labelling them ("ignore regions").

    Used when the teacher is not precise enough to provide labels: a COCO stop-sign detector
    also fires on no-entry signs, sign backs and round mirrors. Blanking every candidate
    region removes the unlabelled-stop-sign false negatives *and* avoids teaching wrong
    labels. Regions overlapping a protected (labelled cone/barrier) box are left alone.
    Any stale labels of the masked class are removed first. Returns the list of masked boxes.
    """
    import cv2

    model = YOLO(weights)
    images = sorted(image_dir.glob(f"{prefix}*.jpg"))
    masked = []
    for i in range(0, len(images), batch):
        chunk = images[i : i + batch]
        results = model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz, classes=list(teacher_classes),
                                batch=batch, device=device, verbose=False)
        for img_path, r in zip(chunk, results):
            lbl_path = label_dir / f"{img_path.stem}.txt"
            labels = read_labels(lbl_path)
            protected = labels[np.isin(labels[:, 0], protect_classes)][:, 1:]
            if len(r.boxes) == 0:
                continue
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            changed = False
            for box, score in zip(r.boxes.xywhn.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if len(protected) and iou_xywh(box, protected).max() > 0:
                    continue
                cx, cy, bw, bh = box
                bw, bh = bw * (1 + pad), bh * (1 + pad)
                x0, y0 = int(max(0, (cx - bw / 2) * w)), int(max(0, (cy - bh / 2) * h))
                x1, y1 = int(min(w, (cx + bw / 2) * w)), int(min(h, (cy + bh / 2) * h))
                img[y0:y1, x0:x1] = 114
                masked.append({"image": img_path.name, "box_xywhn": [round(float(v), 5) for v in box],
                               "conf": round(float(score), 3)})
                changed = True
            if changed:
                cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"masked {len(masked)} regions in {len({m['image'] for m in masked})}/{len(images)} images")
    return masked


def apply_review(review: list, image_dir: Path, label_dir: Path, stop_cls: int, pad: float = 0.15):
    """Apply a visual review of teacher candidates (see results/stop_sign_audit/).

    stop_sign -> added as a label; not_stop -> left untouched, i.e. a hard negative such as a
    no-entry sign or the back of a sign; uncertain -> greyed out (ignored in training).
    """
    import cv2

    by_image = {}
    for r in review:
        by_image.setdefault(r["image"], []).append(r)
    added = masked = 0
    for name, regions in by_image.items():
        img_path, lbl_path = image_dir / name, label_dir / f"{Path(name).stem}.txt"
        stops = [r for r in regions if r["review"] == "stop_sign"]
        unsure = [r for r in regions if r["review"] == "uncertain"]
        if stops:
            with lbl_path.open("a") as f:
                for r in stops:
                    f.write(f"{stop_cls} " + " ".join(f"{v:.6f}" for v in r["box_xywhn"]) + "\n")
            added += len(stops)
        if unsure:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            for r in unsure:
                cx, cy, bw, bh = r["box_xywhn"]
                bw, bh = bw * (1 + pad), bh * (1 + pad)
                img[int(max(0, (cy - bh / 2) * h)):int(min(h, (cy + bh / 2) * h)),
                    int(max(0, (cx - bw / 2) * w)):int(min(w, (cx + bw / 2) * w))] = 114
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            masked += len(unsure)
    print(f"review applied: {added} stop-sign labels added, {masked} uncertain regions masked")


def strip_class(label_dir: Path, cls: int, prefix: str = ""):
    n = 0
    for p in label_dir.glob(f"{prefix}*.txt"):
        lines = p.read_text().splitlines()
        keep = [l for l in lines if l.strip() and int(float(l.split()[0])) != cls]
        if len(keep) != len([l for l in lines if l.strip()]):
            n += 1
            p.write_text("\n".join(keep) + ("\n" if keep else ""))
    print(f"removed class {cls} labels from {n} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="YOLO dataset root (images/, labels/)")
    ap.add_argument("--mode", choices=["coco-stop", "self"], required=True)
    ap.add_argument("--weights", default="yolo26x.pt", help="teacher weights")
    ap.add_argument("--prefix", required=True, help="only images whose name starts with this, e.g. nuim_")
    ap.add_argument("--split", default="train")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.mode == "coco-stop":
        class_map = {COCO_STOP_SIGN: CLASS_ID["stop_sign"]}
    else:
        class_map = {CLASS_ID["cone"]: CLASS_ID["cone"], CLASS_ID["barrier"]: CLASS_ID["barrier"]}
    pseudo_label(args.weights, args.data / "images" / args.split, args.data / "labels" / args.split,
                 class_map, args.conf, args.imgsz, args.prefix, device=args.device)


if __name__ == "__main__":
    main()
