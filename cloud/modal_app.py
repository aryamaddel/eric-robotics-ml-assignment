"""Cloud steps on Modal: label handling, GPU training, evaluation and GPU benchmarks.

Everything lives on one Modal Volume ("navperc-data"):
    /data/nuimages      selected nuImages frames (streamed by cloud/fetch_app.py, never stored locally)
    /data/yolo          merged YOLO dataset as individual files (MTSD part uploaded from the local machine)
    /data/yolo_v3.tar   the same dataset, prepared for training (see stage_dataset), as ONE file
    /data/runs          training runs / weights / benchmark results

Network volumes are slow for many small files, so jobs work on a local-disk copy: the first job
builds it with a parallel copy, prepares the labels and saves a single archive; later jobs copy and
unpack that archive.

Typical sequence (final model = data variant v3):
    modal volume put navperc-data data/yolo /yolo                             # MTSD stop-sign subset
    modal run --detach cloud/fetch_app.py --url "<nuImages samples tgz URL>"  # CPU only
    NAVPERC_GPU=H100 NAVPERC_CPUS=16 modal run --detach cloud/modal_app.py::train --batch 64 --workers 16
    modal run cloud/modal_app.py::benchmark_gpu                               # TensorRT etc. on an L4
"""

import os
from pathlib import Path

import modal

APP_DIR = Path(__file__).resolve().parent.parent
app = modal.App("navperc")
vol = modal.Volume.from_name("navperc-data", create_if_missing=True)

# GPU for training/eval, e.g. NAVPERC_GPU=H100 modal run ... (baked into the image so the
# container sees the same function definitions)
GPU = os.environ.get("NAVPERC_GPU", "L4")
CPUS = int(os.environ.get("NAVPERC_CPUS", "8"))

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.177", "pyyaml")
    .env({"NAVPERC_GPU": GPU, "NAVPERC_CPUS": str(CPUS)})
)
image = (base_image.add_local_file(APP_DIR / "results/stop_sign_audit/teacher_candidates_review.json", "/assets/review.json")
         .add_local_python_source("navperc"))
# Local sources must be the last layer, so the TensorRT image branches from the base.
trt_image = (base_image.uv_pip_install("tensorrt-cu12", "nvidia-modelopt[onnx]", "onnx", "onnxslim", "onnxruntime-gpu")
             .add_local_python_source("navperc"))

DATA = Path("/data")
LOCAL = Path("/tmp/yolo")
# v1: COCO-teacher stop-sign detections added as labels (turned out to include many no-entry signs)
# v2: the same teacher detections are greyed out instead (ignore regions), see pseudo_label.mask_regions
# v3: the teacher candidates were reviewed visually; stop signs -> labels, look-alikes kept as
#     hard negatives, uncertain -> greyed out (results/stop_sign_audit/teacher_candidates_review.json)
ARCHIVES = {"v1": DATA / "yolo_pl.tar", "v2": DATA / "yolo_v2.tar", "v3": DATA / "yolo_v3.tar"}


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


def stage_dataset(variant: str = "v3") -> Path:
    """Local-disk copy of the dataset with the chosen handling of unlabelled stop signs."""
    import json
    import shutil
    import tarfile
    import time

    from navperc.classes import CLASS_ID
    from navperc.datasets.yolo import write_data_yamls
    from navperc.pseudo_label import COCO_STOP_SIGN, apply_review, mask_regions, pseudo_label, strip_class

    archive = ARCHIVES[variant]
    t0 = time.time()
    if archive.exists():
        shutil.copyfile(archive, "/tmp/staged.tar")
        with tarfile.open("/tmp/staged.tar") as tf:
            tf.extractall("/tmp", filter="data")
        print(f"staged {variant} from archive in {time.time() - t0:.0f}s")
    else:
        parallel_copy(DATA / "yolo", LOCAL)
        print(f"copied dataset in {time.time() - t0:.0f}s; preparing variant {variant}")
        stop = CLASS_ID["stop_sign"]
        # nuImages has no human stop-sign labels, so any class-2 line on a nuim_ frame is a stale pseudo-label
        strip_class(LOCAL / "labels/train", stop, prefix="nuim_")
        if variant == "v1":
            pseudo_label("yolo26x.pt", LOCAL / "images/train", LOCAL / "labels/train", {COCO_STOP_SIGN: stop},
                         conf=0.5, imgsz=1280, prefix="nuim_", batch=16, device=0)
        elif variant == "v3":
            review = json.loads(Path("/assets/review.json").read_text())["regions"]
            apply_review(review, LOCAL / "images/train", LOCAL / "labels/train", stop)
        else:
            masked = mask_regions("yolo26x.pt", LOCAL / "images/train", LOCAL / "labels/train", [COCO_STOP_SIGN],
                                  [CLASS_ID["cone"], CLASS_ID["barrier"]], conf=0.25, imgsz=1280,
                                  prefix="nuim_", batch=16, device=0)
            (DATA / "masked_regions_v2.json").write_text(json.dumps(masked))
        with tarfile.open("/tmp/staged.tar", "w") as tf:
            tf.add(LOCAL, arcname="yolo")
        shutil.copyfile("/tmp/staged.tar", archive)
        vol.commit()
        print(f"built {archive} in {time.time() - t0:.0f}s")
    write_data_yamls(LOCAL)
    return LOCAL


@app.function(image=image, volumes={"/data": vol}, gpu=GPU, cpu=CPUS, memory=49152, timeout=10 * 3600)
def train(model: str = "yolo26n.pt", epochs: int = 60, imgsz: int = 960, batch: int = 32,
          name: str = "yolo26n_960_v3", fraction: float = 1.0, workers: int = 8, data_variant: str = "v3"):
    from ultralytics import YOLO

    root = stage_dataset(data_variant)
    y = YOLO(model)
    y.train(data=str(root / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
            fraction=fraction, project=str(DATA / "runs"), name=name, exist_ok=True,
            patience=15, cos_lr=True, close_mosaic=min(10, epochs // 2), seed=0, deterministic=False)
    vol.commit()
    return run_eval(root, name, imgsz)


@app.function(image=image, volumes={"/data": vol}, gpu=GPU, cpu=CPUS, memory=16384, timeout=3600)
def evaluate(name: str, imgsz: int = 960, weights: str = "best.pt"):
    return run_eval(stage_dataset("v1"), name, imgsz, weights)  # test splits are identical in v1/v2


def run_eval(root: Path, name: str, imgsz: int, weights: str = "best.pt"):
    """Per-source test metrics: cones/barriers on nuImages test, stop signs on MTSD test."""
    import json

    from ultralytics import YOLO

    from navperc.classes import CLASSES

    model = YOLO(str(DATA / "runs" / name / "weights" / weights))
    out = {}
    for source, classes in (("nuimages", ["cone", "barrier"]), ("mtsd", ["stop_sign"])):
        m = model.val(data=str(root / f"eval_{source}.yaml"), split="test", imgsz=imgsz, batch=16,
                      classes=[CLASSES.index(c) for c in classes], plots=False, verbose=False)
        for c in classes:
            i = CLASSES.index(c)
            p, r, ap50, ap = m.box.class_result(list(m.box.ap_class_index).index(i))
            out[c] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                      "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}
    (DATA / "runs" / name / f"test_metrics_{imgsz}.json").write_text(json.dumps(out, indent=2))
    vol.commit()
    print(json.dumps(out, indent=2))
    return out



@app.function(image=trt_image, volumes={"/data": vol}, gpu="L4", cpu=8, memory=32768, timeout=3 * 3600)
def benchmark_gpu(name: str = "yolo26n_960_v3", sizes: str = "640,960", n_images: int = 200):
    """GPU speed/accuracy on an L4 (closer to deployment hardware than the H100 used for training)."""
    from navperc.benchmark import run

    root = stage_dataset("v1")  # test splits are identical in v1/v2; v1 archive already exists
    out = DATA / "runs" / name / "benchmark_gpu"
    out.mkdir(parents=True, exist_ok=True)
    results = run(DATA / "runs" / name / "weights" / "best.pt", root,
                  ["torch", "torch-fp16", "engine-fp16", "engine-int8"],
                  [int(s) for s in sizes.split(",")], 0, n_images, out)
    vol.commit()
    return results
