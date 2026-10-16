"""Apply the saved visual review of nuImages stop-sign candidates to training data.

True stop signs become labels, look-alikes stay visible as hard negatives, and
uncertain regions are masked. Modal staging removes stale stop labels first.
Review decisions are retained in results/stop_sign_audit/teacher_candidates_review.json.
"""

from pathlib import Path


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
