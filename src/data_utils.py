"""Helpers for writing images + labels in Ultralytics YOLO format."""

import hashlib
from pathlib import Path

import cv2


def split_by_hash(key: str, test_fraction: float) -> bool:
    """Deterministic pseudo-random assignment that doesn't depend on iteration order."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < test_fraction


def write_sample(out_root: Path, split: str, name: str, img_path: Path, boxes, max_side: int):
    """Resize the image so its long side is <= max_side and write normalized YOLO labels.

    boxes: iterable of (class_id, xmin, ymin, xmax, ymax) in source pixel coordinates.
    Returns the number of boxes written, or None if the image could not be read.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)

    lines = []
    for cls, x0, y0, x1, y1 in boxes:
        x0, x1 = max(0.0, x0), min(float(w), x1)
        y0, y1 = max(0.0, y0), min(float(h), y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        cx, cy, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    img_dir, lbl_dir = out_root / "images" / split, out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def write_data_yamls(root: Path):
    """Write data.yaml (train/val) plus per-source eval yamls.

    Each source only has human labels for some classes, so cones/barriers are scored on
    nuImages images and stop signs on MTSD images (see README, "Evaluation protocol").
    """
    import yaml

    from classes import CLASSES

    root = root.resolve()
    names = dict(enumerate(CLASSES))
    base = {"path": str(root), "names": names}
    (root / "data.yaml").write_text(yaml.safe_dump({**base, "train": "images/train", "val": "images/val",
                                                    "test": "images/test"}, sort_keys=False))
    for source, prefix in (("nuimages", "nuim_"), ("mtsd", "mtsd_")):
        for split in ("val", "test"):
            imgs = sorted((root / "images" / split).glob(f"{prefix}*.jpg"))
            (root / f"{split}_{source}.txt").write_text("\n".join(f"./images/{split}/{p.name}" for p in imgs) + "\n")
        (root / f"eval_{source}.yaml").write_text(yaml.safe_dump(
            {**base, "train": f"val_{source}.txt", "val": f"val_{source}.txt", "test": f"test_{source}.txt"},
            sort_keys=False))
    print("wrote yamls in", root)


if __name__ == "__main__":
    import sys

    write_data_yamls(Path(sys.argv[1]))
