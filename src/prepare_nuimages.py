"""Convert a cone/barrier subset of nuImages to YOLO format.

nuImages definitions (which we adopt):
  movable_object.trafficcone -> cone
  movable_object.barrier     -> barrier ("temporary metal, concrete or water barrier
                                 placed to redirect traffic")
Stop signs are not labeled in nuImages; they are added later by pseudo-labelling.

Splits: nuImages train -> train; nuImages val is divided into val/test *by driving log*
so that near-duplicate frames from one drive never end up on both sides.

Two-step usage, so only the selected images need extracting from the samples tar:
  python src/prepare_nuimages.py select  --root data/nuimages --out data/nuimages/selection.json
  tar -xf nuimages-v1.0-all-samples.tar -C data/nuimages -T data/nuimages/selection.txt
  python src/prepare_nuimages.py convert --root data/nuimages --selection data/nuimages/selection.json --out data/yolo
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from classes import CLASS_ID
from data_utils import split_by_hash, write_sample

CATEGORY_MAP = {
    "movable_object.trafficcone": CLASS_ID["cone"],
    "movable_object.barrier": CLASS_ID["barrier"],
}


def load_split(root: Path, version: str):
    d = root / f"v1.0-{version}"
    load = lambda name: json.loads((d / f"{name}.json").read_text())
    cats = {c["token"]: c["name"] for c in load("category")}
    samples = {s["token"]: s for s in load("sample")}
    frames = {s["token"]: s for s in load("sample_data") if s["is_key_frame"]}
    boxes = defaultdict(list)
    for a in load("object_ann"):
        cls = CATEGORY_MAP.get(cats[a["category_token"]])
        if cls is not None and a["sample_data_token"] in frames:
            boxes[a["sample_data_token"]].append([cls, *a["bbox"]])
    return frames, samples, boxes


def pick(rng, tokens, n):
    tokens = sorted(tokens)
    return tokens if n < 0 or len(tokens) <= n else rng.sample(tokens, n)


def select(args):
    rng = random.Random(args.seed)
    selection, counts = [], Counter()
    for version in ("train", "val"):
        frames, samples, boxes = load_split(args.root, version)
        pos = [t for t in frames if boxes.get(t)]
        neg = [t for t in frames if not boxes.get(t)]
        if version == "train":
            groups = {"train": (pos, neg)}
        else:
            log_of = lambda t: samples[frames[t]["sample_token"]]["log_token"]
            in_test = lambda t: split_by_hash(log_of(t), 0.5)
            groups = {
                "val": ([t for t in pos if not in_test(t)], [t for t in neg if not in_test(t)]),
                "test": ([t for t in pos if in_test(t)], [t for t in neg if in_test(t)]),
            }
        for split, (p, n) in groups.items():
            n_pos = args.train_pos if split == "train" else args.eval_pos
            chosen = pick(rng, p, n_pos)
            for t in chosen + pick(rng, n, round(len(chosen) * args.neg_ratio)):
                selection.append({"split": split, "token": t, "file": frames[t]["filename"], "boxes": boxes.get(t, [])})
                counts[f"{split} images"] += 1
                counts[f"{split} boxes"] += len(boxes.get(t, []))
    args.out.write_text(json.dumps(selection))
    args.out.with_suffix(".txt").write_text("\n".join(s["file"] for s in selection) + "\n", newline="\n")
    print(dict(counts))


def convert(args):
    counts, missing = Counter(), 0
    for s in json.loads(args.selection.read_text()):
        n = write_sample(args.out, s["split"], f"nuim_{s['token']}", args.root / s["file"], s["boxes"], args.max_side)
        if n is None:
            missing += 1
            continue
        counts[f"{s['split']} images"] += 1
        counts[f"{s['split']} boxes"] += n
    print(dict(counts), "missing images:", missing)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--train-pos", type=int, default=8000, help="train images with cones/barriers (-1 = all)")
    s.add_argument("--eval-pos", type=int, default=1500, help="per val/test split")
    s.add_argument("--neg-ratio", type=float, default=0.1, help="background images per positive")
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("convert")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--selection", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--max-side", type=int, default=1600)
    args = ap.parse_args()
    {"select": select, "convert": convert}[args.cmd](args)


if __name__ == "__main__":
    main()
