"""Convert the stop-sign subset of the Mapillary Traffic Sign Dataset (MTSD v2) to YOLO format.

MTSD only labels traffic signs, so cones/barriers in these images are unlabeled;
they are filled in later by pseudo-labelling (see pseudo_label.py).

Splits: MTSD train -> train; MTSD val is divided deterministically into val/test
(MTSD's own test split has no public labels).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from navperc.classes import CLASS_ID
from navperc.datasets.yolo import split_by_hash, write_sample

STOP_PREFIX = "regulatory--stop--"


def stop_boxes(ann):
    for o in ann["objects"]:
        if o["label"].startswith(STOP_PREFIX):
            b = o["bbox"]
            yield CLASS_ID["stop_sign"], b["xmin"], b["ymin"], b["xmax"], b["ymax"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtsd-root", type=Path, required=True, help="dir with annotations/ and splits/")
    ap.add_argument("--images", type=Path, required=True, help="dir of downloaded <key>.jpg images")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--test-fraction", type=float, default=0.5, help="share of MTSD val used as test")
    args = ap.parse_args()

    train_keys = set((args.mtsd_root / "splits" / "train.txt").read_text().split())
    val_keys = set((args.mtsd_root / "splits" / "val.txt").read_text().split())

    stats, skipped = Counter(), Counter()
    for img_path in sorted(args.images.glob("*.jpg")):
        key = img_path.stem
        ann = json.loads((args.mtsd_root / "annotations" / f"{key}.json").read_text())
        if ann.get("ispano"):
            skipped["panorama"] += 1  # equirectangular: pinhole geometry doesn't apply
            continue
        if key in train_keys:
            split = "train"
        elif key in val_keys:
            split = "test" if split_by_hash(key, args.test_fraction) else "val"
        else:
            skipped["no split"] += 1
            continue
        n = write_sample(args.out, split, f"mtsd_{key}", img_path, list(stop_boxes(ann)), args.max_side)
        if n is None:
            skipped["unreadable"] += 1
            continue
        stats[f"{split} images"] += 1
        stats[f"{split} stop signs"] += n
    print(dict(stats), "skipped:", dict(skipped))


if __name__ == "__main__":
    main()
