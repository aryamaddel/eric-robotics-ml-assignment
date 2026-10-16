"""Cloud steps on Modal: label handling, GPU training, evaluation and GPU benchmarks.

Everything lives on one Modal Volume ("navperc-data"):
    /data/nuimages      selected nuImages frames (streamed by src/fetch_app.py, never stored locally)
    /data/yolo          merged YOLO dataset as individual files (MTSD part uploaded from the local machine)
    /data/yolo_v3.tar   the same dataset, prepared for training (see stage_dataset), as ONE file
    /data/runs          training runs / weights / benchmark results

Network volumes are slow for many small files, so jobs work on a local-disk copy: the first job
builds it with a parallel copy, prepares the labels and saves a single archive; later jobs copy and
unpack that archive.

Typical sequence (final model = data variant v3):
    modal volume put navperc-data data/yolo /yolo                             # MTSD stop-sign subset
    modal run --detach src/fetch_app.py --url "<nuImages samples tgz URL>"  # CPU only
    modal run --detach src/modal_app.py::train --batch 64
    modal run src/modal_app.py::benchmark_gpu                               # TensorRT etc. on an L4
"""

import os
from pathlib import Path

import modal

APP_DIR = Path(__file__).resolve().parent.parent
app = modal.App("navperc")
vol = modal.Volume.from_name("navperc-data", create_if_missing=True)

CPUS = int(os.environ.get("NAVPERC_CPUS", "8"))

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.177", "pyyaml")
    .env({"NAVPERC_CPUS": str(CPUS)})
)
image = (
    base_image.env({"PYTHONPATH": "/root/src"})
    .add_local_dir(APP_DIR / "src", "/root/src", copy=True, ignore=["**/__pycache__/**"])
    .add_local_file(APP_DIR / "results/stop_sign_audit/teacher_candidates_review.json", "/assets/review.json")
)
trt_image = (
    base_image.uv_pip_install("tensorrt-cu12", "nvidia-modelopt[onnx]", "onnx", "onnxslim", "onnxruntime-gpu")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(APP_DIR / "src", "/root/src", copy=True, ignore=["**/__pycache__/**"])
)

DATA = Path("/data")
LOCAL = Path("/tmp/yolo")
ARCHIVE = DATA / "yolo_v3.tar"


def parallel_copy(src: Path, dst: Path, workers: int = 64):
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    files = [Path(root) / f for root, _, fs in os.walk(src) for f in fs]
    for d in {dst / f.parent.relative_to(src) for f in files}:
        d.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda f: shutil.copyfile(f, dst / f.relative_to(src)), files))
    print(f"copied {len(files)} files")


def stage_dataset() -> Path:
    """Stage the final reviewed-label dataset on local disk."""
    import json
    import shutil
    import tarfile
    import time

    from classes import CLASS_ID
    from data_utils import write_data_yamls
    from pseudo_label import apply_review, strip_class

    archive = ARCHIVE
    t0 = time.time()
    if archive.exists():
        shutil.copyfile(archive, "/tmp/staged.tar")
        with tarfile.open("/tmp/staged.tar") as tf:
            tf.extractall("/tmp", filter="data")
        print(f"staged v3 from archive in {time.time() - t0:.0f}s")
    else:
        parallel_copy(DATA / "yolo", LOCAL)
        print(f"copied dataset in {time.time() - t0:.0f}s; preparing reviewed labels")
        stop = CLASS_ID["stop_sign"]
        # nuImages has no human stop-sign labels, so any class-2 line on a nuim_ frame is a stale pseudo-label
        strip_class(LOCAL / "labels/train", stop, prefix="nuim_")
        review = json.loads(Path("/assets/review.json").read_text())["regions"]
        apply_review(review, LOCAL / "images/train", LOCAL / "labels/train", stop)
        with tarfile.open("/tmp/staged.tar", "w") as tf:
            tf.add(LOCAL, arcname="yolo")
        shutil.copyfile("/tmp/staged.tar", archive)
        vol.commit()
        print(f"built {archive} in {time.time() - t0:.0f}s")
    write_data_yamls(LOCAL)
    return LOCAL


@app.function(image=image, volumes={"/data": vol}, gpu="H100", cpu=CPUS, memory=49152, timeout=10 * 3600)
def train(epochs: int = 60, batch: int = 32, name: str = "yolo26n_960_v3"):
    """Fine-tune the final YOLO26n recipe at 960 px on reviewed v3 labels."""
    from ultralytics import YOLO

    root = stage_dataset()
    YOLO("yolo26n.pt").train(
        data=str(root / "data.yaml"), epochs=epochs, imgsz=960, batch=batch, workers=CPUS,
        project=str(DATA / "runs"), name=name, exist_ok=True, patience=15, cos_lr=True,
        close_mosaic=min(10, epochs // 2), seed=0, deterministic=False)
    vol.commit()
    return run_eval(root, name, 960)


@app.function(image=image, volumes={"/data": vol}, gpu="H100", cpu=CPUS, memory=16384, timeout=3600)
def evaluate(name: str, imgsz: int = 960, weights: str = "best.pt"):
    return run_eval(stage_dataset(), name, imgsz, weights)


def run_eval(root: Path, name: str, imgsz: int, weights: str = "best.pt"):
    """Per-source test metrics: cones/barriers on nuImages test, stop signs on MTSD test."""
    import json

    from benchmark import accuracy

    out = accuracy(DATA / "runs" / name / "weights" / weights, root, imgsz, 0, batch=16)
    (DATA / "runs" / name / f"test_metrics_{imgsz}.json").write_text(json.dumps(out, indent=2))
    vol.commit()
    print(json.dumps(out, indent=2))
    return out



@app.function(image=trt_image, volumes={"/data": vol}, gpu="L4", cpu=8, memory=32768, timeout=3 * 3600)
def benchmark_gpu(name: str = "yolo26n_960_v3", sizes: str = "640,960", n_images: int = 200):
    """GPU speed/accuracy on an L4 (closer to deployment hardware than the H100 used for training)."""
    from benchmark import run

    root = stage_dataset()
    out = DATA / "runs" / name / "benchmark_gpu"
    out.mkdir(parents=True, exist_ok=True)
    results = run(DATA / "runs" / name / "weights" / "best.pt", root,
                  ["torch", "torch-fp16", "engine-fp16", "engine-int8"],
                  [int(s) for s in sizes.split(",")], 0, n_images, out)
    vol.commit()
    return results


@app.function(image=image, volumes={"/data": vol}, gpu="H100", cpu=CPUS, memory=16384, timeout=3600)
def infer(source: str, name: str = "yolo26n_960_v3"):
    """Annotate a source image, image folder or video uploaded to navperc-data."""
    from predict import run

    path = DATA / source.replace("\\", "/").lstrip("/")
    if not path.exists():
        raise FileNotFoundError(f"Upload the source to navperc-data first: {path}")
    out = DATA / "runs" / name / "inference" / path.stem
    run(str(DATA / "runs" / name / "weights" / "best.pt"), path, out, device=0)
    vol.commit()
    return str(out.relative_to(DATA))
