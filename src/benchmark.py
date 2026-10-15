"""Speed / size / accuracy benchmark of the detector across export formats and input sizes.

For each (format, imgsz):
  1. export the trained .pt model (quantized variants are calibrated on the val split),
  2. latency: batch-1, end-to-end (pre-process + inference + post-process) on a fixed set of
     test images after warm-up, timed over 3 rounds; FPS = 1000 / median latency,
  3. accuracy: per-source test metrics (cones/barriers on nuImages, stop signs on MTSD),
  4. model file size.

CPU: python src/benchmark.py
GPU: src/modal_app.py::benchmark_gpu (existing Modal workflow)
"""

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from classes import CLASSES

FORMATS = {
    # name: (ultralytics export kwargs or None for the raw .pt, predict kwargs)
    "torch": (None, {}),
    "torch-fp16": (None, {"quantize": 16}),
    "openvino": ({"format": "openvino"}, {}),
    "openvino-int8": ({"format": "openvino", "quantize": 8}, {}),
    "engine-fp16": ({"format": "engine", "quantize": 16}, {}),
    "engine-int8": ({"format": "engine", "quantize": 8}, {}),
}


def dir_size_mb(p: Path) -> float:
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return round(sum(f.stat().st_size for f in files) / 1e6, 2)


def export(weights: Path, fmt: str, imgsz: int, data_yaml: Path, out_dir: Path, device) -> Path:
    from ultralytics import YOLO

    kwargs, _ = FORMATS[fmt]
    work = out_dir / f"{fmt}_{imgsz}"
    work.mkdir(parents=True, exist_ok=True)
    pt = work / "model.pt"
    shutil.copyfile(weights, pt)
    if kwargs is None:
        return pt
    # int8 calibration uses the dataset's val split (representative of deployment data)
    extra = {"data": str(data_yaml), "fraction": 0.3} if kwargs.get("quantize") == 8 else {}
    path = YOLO(str(pt)).export(imgsz=imgsz, batch=1, device=device, **kwargs, **extra)
    return Path(path)


def latency(model_path: Path, images, imgsz: int, device, predict_kwargs, warmup=10, rounds=3):
    """Per-frame latency. Desktop CPUs (hybrid P/E cores, OS scheduling) are noisy, so the set is
    timed `rounds` times; we report the median over all frames and the spread of round medians."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    if not images:
        raise ValueError("No benchmark images found")
    if warmup < 1 or rounds < 1:
        raise ValueError("warmup and rounds must be positive")
    frames = [cv2.imread(str(p)) for p in images]
    for path, frame in zip(images, frames):
        if frame is None:
            raise ValueError(f"Cannot read benchmark image: {path}")
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, device=device, verbose=False, **(predict_kwargs or {}))
    all_times, round_medians, infer = [], [], []
    for _ in range(rounds):
        times = []
        for f in frames:
            t0 = time.perf_counter()
            r = model.predict(f, imgsz=imgsz, device=device, verbose=False, **(predict_kwargs or {}))
            times.append((time.perf_counter() - t0) * 1000)
            infer.append(r[0].speed["inference"])  # network forward pass only
        all_times += times
        round_medians.append(float(np.median(times)))
    t = np.array(all_times)
    med = float(np.median(t))
    return {"median_ms": round(med, 2), "inference_median_ms": round(float(np.median(infer)), 2),
            "mean_ms": round(float(t.mean()), 2),
            "p90_ms": round(float(np.percentile(t, 90)), 2),
            "round_medians_ms": [round(x, 2) for x in round_medians], "fps": round(1000 / med, 1)}


def accuracy(model_path: Path, data_root: Path, imgsz: int, device, predict_kwargs=None, batch=1):
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    out = {}
    for source, classes in (("nuimages", ["cone", "barrier"]), ("mtsd", ["stop_sign"])):
        m = model.val(data=str(data_root / f"eval_{source}.yaml"), split="test", imgsz=imgsz, batch=batch,
                      device=device, classes=[CLASSES.index(c) for c in classes], plots=False,
                      verbose=False, **(predict_kwargs or {}))
        for c in classes:
            idx = list(m.box.ap_class_index).index(CLASSES.index(c))
            p, r, ap50, ap = m.box.class_result(idx)
            out[c] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                      "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}
    out["mean_mAP50"] = round(float(np.mean([out[c]["mAP50"] for c in CLASSES])), 4)
    return out


def run(weights, data_root, formats, sizes, device, n_images, out_dir, work_dir=None):
    from data_utils import write_data_yamls

    if n_images < 1:
        raise ValueError("n_images must be positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_data_yamls(data_root)
    test_imgs = sorted((data_root / "images" / "test").glob("*.jpg"))
    if not test_imgs:
        raise ValueError(f"No test JPG images in {data_root / 'images/test'}")
    rng = np.random.default_rng(0)
    images = [test_imgs[i] for i in rng.choice(len(test_imgs), size=min(n_images, len(test_imgs)), replace=False)]

    results = []
    for imgsz in sizes:
        for fmt in formats:
            print(f"== {fmt} @ {imgsz}")
            path = export(Path(weights), fmt, imgsz, data_root / "data.yaml", work_dir or out_dir, device)
            _, pk = FORMATS[fmt]
            row = {"format": fmt, "imgsz": imgsz, "size_mb": dir_size_mb(path), **latency(path, images, imgsz, device, pk)}
            row.update(accuracy(path, data_root, imgsz, device, pk))
            print(row)
            results.append(row)
            (out_dir / "results.json").write_text(json.dumps({"env": environment(device), "results": results}, indent=2))
    return results


def environment(device):
    import torch
    import ultralytics

    env = {"platform": platform.platform(), "processor": platform.processor(), "torch": torch.__version__,
           "ultralytics": ultralytics.__version__, "torch_threads": torch.get_num_threads(), "device": str(device)}
    try:
        import psutil

        env["cpu_affinity"] = psutil.Process().cpu_affinity()
    except Exception:
        pass
    if torch.cuda.is_available() and str(device) != "cpu":
        env["gpu"] = torch.cuda.get_device_name(0)
    try:
        import openvino

        env["openvino"] = openvino.__version__
    except ImportError:
        pass
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=Path("weights/best.pt"))
    ap.add_argument("--data", type=Path, default=Path("data/yolo"))
    ap.add_argument("--out", type=Path, default=Path("results/benchmark_cpu"))
    args = ap.parse_args()
    if not args.weights.is_file():
        ap.error(f"Missing weights: {args.weights}")
    run(args.weights, args.data.resolve(), ["torch", "openvino", "openvino-int8"],
        [640, 960], "cpu", 100, args.out,
        work_dir=Path("runs/exports") / args.out.name)


if __name__ == "__main__":
    main()
