"""Speed / size / accuracy benchmark of the detector across export formats and input sizes.

For each (format, imgsz):
  1. export the trained .pt model (quantized variants are calibrated on the val split),
  2. latency: batch-1, end-to-end (pre-process + inference + post-process) on a fixed set of
     test images after warm-up, timed over 3 rounds; FPS = 1000 / median latency,
  3. accuracy: per-source test metrics (cones/barriers on nuImages, stop signs on MTSD),
  4. model file size.

CPU:  python -m navperc.benchmark --weights best.pt --data data/yolo --device cpu \
          --formats torch onnx openvino openvino-int8 --imgsz 640 960
GPU:  run via cloud/modal_app.py::benchmark_gpu (formats torch torch-fp16 engine-fp16 engine-int8)
"""

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from navperc.classes import CLASSES

FORMATS = {
    # name: (ultralytics export kwargs or None for the raw .pt, predict kwargs)
    "torch": (None, {}),
    "torch-fp16": (None, {"half": True}),
    "onnx": ({"format": "onnx", "simplify": True}, {}),
    "openvino": ({"format": "openvino"}, {}),
    "openvino-int8": ({"format": "openvino", "int8": True}, {}),
    "engine-fp16": ({"format": "engine", "half": True}, {}),
    "engine-int8": ({"format": "engine", "int8": True}, {}),
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
    extra = {"data": str(data_yaml), "fraction": 0.3} if kwargs.get("int8") else {}
    path = YOLO(str(pt)).export(imgsz=imgsz, batch=1, device=device, **kwargs, **extra)
    return Path(path)


def latency(model_path: Path, images, imgsz: int, device, predict_kwargs, warmup=10, rounds=3):
    """Per-frame latency. Desktop CPUs (hybrid P/E cores, OS scheduling) are noisy, so the set is
    timed `rounds` times; we report the median over all frames and the spread of round medians."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    frames = [cv2.imread(str(p)) for p in images]
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, device=device, verbose=False, **predict_kwargs)
    all_times, round_medians, infer = [], [], []
    for _ in range(rounds):
        times = []
        for f in frames:
            t0 = time.perf_counter()
            r = model.predict(f, imgsz=imgsz, device=device, verbose=False, **predict_kwargs)
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


def accuracy(model_path: Path, data_root: Path, imgsz: int, device, predict_kwargs):
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    out = {}
    for source, classes in (("nuimages", ["cone", "barrier"]), ("mtsd", ["stop_sign"])):
        m = model.val(data=str(data_root / f"eval_{source}.yaml"), split="test", imgsz=imgsz, batch=1,
                      device=device, classes=[CLASSES.index(c) for c in classes], plots=False,
                      verbose=False, half=predict_kwargs.get("half", False))
        for c in classes:
            idx = list(m.box.ap_class_index).index(CLASSES.index(c))
            p, r, ap50, ap = m.box.class_result(idx)
            out[c] = {"mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}
    out["mean_mAP50"] = round(float(np.mean([out[c]["mAP50"] for c in CLASSES])), 4)
    return out


def restrict_cores(spec: str):
    """Pin this process to a subset of cores to emulate a smaller edge CPU.

    Assumes the i5-13400F layout: logical 0-11 = 6 performance cores x 2 hyper-threads,
    12-15 = 4 efficiency cores. "p<N>": N performance cores (one thread each),
    "e<N>": N efficiency cores, "all": no restriction.
    """
    import psutil
    import torch

    if spec == "all":
        return None
    kind, n = spec[0], int(spec[1:])
    cores = [2 * i for i in range(n)] if kind == "p" else [12 + i for i in range(n)]
    psutil.Process().cpu_affinity(cores)
    torch.set_num_threads(n)
    import cv2

    cv2.setNumThreads(n)  # OpenCV's pool (resize/letterbox) also defaults to all logical cores
    limit_runtime_threads(n, kind)
    return cores


def limit_runtime_threads(n: int, kind: str):
    """Size OpenVINO / ONNX Runtime thread pools to the emulated core count.

    Both runtimes size their pools from the whole machine by default, which oversubscribes a
    pinned subset of cores. Ultralytics does not expose these options, so the constructors are
    wrapped here (benchmark harness only); on a real device you would set the same properties.
    """
    import onnxruntime as ort
    import openvino as ov

    ov_extra = {"INFERENCE_NUM_THREADS": n, "ENABLE_HYPER_THREADING": False,
                "SCHEDULING_CORE_TYPE": "ECORE_ONLY" if kind == "e" else "PCORE_ONLY"}
    orig_compile = ov.Core.compile_model

    def compile_model(self, model, device_name=None, config=None, **kw):
        cfg = dict(config or {})
        if device_name in (None, "CPU"):
            cfg.update(ov_extra)
        return orig_compile(self, model, device_name, cfg, **kw)

    ov.Core.compile_model = compile_model

    orig_session = ort.InferenceSession

    class InferenceSession(orig_session):
        def __init__(self, path, sess_options=None, *args, **kw):
            sess_options = sess_options or ort.SessionOptions()
            sess_options.intra_op_num_threads = n
            sess_options.inter_op_num_threads = 1
            # idle workers busy-wait by default and starve the pre-processing threads on few cores
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            super().__init__(path, sess_options, *args, **kw)

    ort.InferenceSession = InferenceSession


def run(weights, data_root, formats, sizes, device, n_images, out_dir, skip_accuracy=False, work_dir=None):
    from navperc.datasets.yolo import write_data_yamls

    write_data_yamls(data_root)
    test_imgs = sorted((data_root / "images" / "test").glob("*.jpg"))
    rng = np.random.default_rng(0)
    images = [test_imgs[i] for i in rng.choice(len(test_imgs), size=min(n_images, len(test_imgs)), replace=False)]

    results = []
    for imgsz in sizes:
        for fmt in formats:
            print(f"== {fmt} @ {imgsz}")
            path = export(Path(weights), fmt, imgsz, data_root / "data.yaml", work_dir or out_dir, device)
            _, pk = FORMATS[fmt]
            row = {"format": fmt, "imgsz": imgsz, "size_mb": dir_size_mb(path), **latency(path, images, imgsz, device, pk)}
            if not skip_accuracy:
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
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", type=Path, required=True, help="YOLO dataset root with images/test")
    ap.add_argument("--formats", nargs="+", default=["torch", "onnx", "openvino", "openvino-int8"], choices=list(FORMATS))
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 960])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("results/benchmark_cpu"), help="results.json goes here")
    ap.add_argument("--work", type=Path, help="exported models go here (default: runs/exports/<out name>)")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--cores", default="all", help="emulate a smaller CPU: all | p1 | p2 | p4 | e4 ...")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pinned = restrict_cores(args.cores)
    if pinned:
        print(f"pinned to logical cores {pinned}")
    work = args.work or Path("runs/exports") / args.out.name
    run(args.weights, args.data.resolve(), args.formats, args.imgsz, args.device, args.n_images, args.out,
        args.skip_accuracy, work)


if __name__ == "__main__":
    main()
