"""Structured channel pruning of the trained detector + fine-tuning (Modal).

Only *structured* pruning (removing whole channels) reduces compute on a CPU; unstructured
sparsity leaves dense kernels unchanged. Channels are ranked per dependency group (Torch-Pruning
DepGraph, L2 magnitude) and removed from the backbone/neck; the detection head and the
attention block are left intact. The pruned network is then fine-tuned with the normal
Ultralytics trainer (a trainer subclass returns the pruned module instead of rebuilding the
model from its yaml, which would silently undo the pruning).

    modal run src/prune_app.py::inspect --ratio 0.3        # CPU: pruning only, report MACs/params
    modal run --detach src/prune_app.py::prune_finetune --ratio 0.3 --epochs 30
"""

from pathlib import Path

import modal

APP_DIR = Path(__file__).resolve().parent.parent
app = modal.App("navperc-prune")
vol = modal.Volume.from_name("navperc-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.177", "pyyaml", "torch-pruning")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(APP_DIR / "src", "/root/src", copy=True, ignore=["**/__pycache__/**"])
)
DATA = Path("/data")
BASE_RUN = "yolo26n_960_v3"


def prune(weights: str, ratio: float, imgsz: int = 640):
    import torch
    import torch_pruning as tp
    from ultralytics import YOLO

    from prune_utils import remove_splits

    model = YOLO(weights).model.float().cpu()
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    example = torch.randn(1, 3, imgsz, imgsz)
    # Compare the feature maps entering the head: the final output is a top-k sorted list, which
    # reorders on near-tied scores and is useless for an equivalence check.
    feats = []
    hook = model.model[-1].register_forward_pre_hook(lambda mod, inp: feats.append([t.clone() for t in inp[0]]))
    with torch.no_grad():
        model(example)
        n_split = remove_splits(model)
        model(example)
    hook.remove()
    diff = max((a - b).abs().max().item() for a, b in zip(*feats))
    print(f"rewrote {n_split} split blocks, max output difference {diff:.2e}")

    ignored = [model.model[-1]]  # detection head: output channels are fixed by #classes / box params
    for m in model.modules():  # attention reshapes channels into heads; keep it intact
        if type(m).__name__ in {"Attention", "PSABlock"}:
            ignored.append(m)

    base_macs, base_params = tp.utils.count_ops_and_params(model, example)
    pruner = tp.pruner.MetaPruner(model, example, importance=tp.importance.GroupMagnitudeImportance(p=2),
                                  pruning_ratio=ratio, ignored_layers=ignored, round_to=8)
    pruner.step()
    macs, params = tp.utils.count_ops_and_params(model, example)
    with torch.no_grad():
        out = model(example)
    stats = {"ratio": ratio, "macs_before_G": round(base_macs / 1e9, 3), "macs_after_G": round(macs / 1e9, 3),
             "params_before_M": round(base_params / 1e6, 3), "params_after_M": round(params / 1e6, 3),
             "forward_ok": out is not None, "split_blocks_rewritten": n_split, "rewrite_max_abs_diff": diff}
    return model, stats


@app.function(image=image, volumes={"/data": vol}, cpu=4, memory=8192, timeout=1200)
def inspect(ratio: float = 0.3):
    _, stats = prune(str(DATA / "runs" / BASE_RUN / "weights" / "best.pt"), ratio)
    print(stats)
    return stats


@app.function(image=image, volumes={"/data": vol}, gpu="H100", cpu=16, memory=49152, timeout=4 * 3600)
def prune_finetune(ratio: float = 0.3, epochs: int = 30):
    import json
    import shutil
    import tarfile

    from ultralytics.models.yolo.detect import DetectionTrainer

    from data_utils import write_data_yamls

    shutil.copyfile(DATA / "yolo_v3.tar", "/tmp/staged.tar")  # same data as the base model
    with tarfile.open("/tmp/staged.tar") as tf:
        tf.extractall("/tmp", filter="data")
    root = Path("/tmp/yolo")
    write_data_yamls(root)

    pruned, stats = prune(str(DATA / "runs" / BASE_RUN / "weights" / "best.pt"), ratio)
    print(stats)

    class PrunedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return pruned

    from ultralytics import YOLO

    name = f"{BASE_RUN}_pruned{int(ratio * 100)}"
    y = YOLO(str(DATA / "runs" / BASE_RUN / "weights" / "best.pt"))
    y.train(trainer=PrunedTrainer, data=str(root / "data.yaml"), epochs=epochs, imgsz=960, batch=64,
            workers=16, project=str(DATA / "runs"), name=name, exist_ok=True, lr0=0.002, warmup_epochs=1,
            cos_lr=True, close_mosaic=min(10, epochs // 3), patience=epochs, seed=0, deterministic=False, amp=True)
    (DATA / "runs" / name / "prune_stats.json").write_text(json.dumps(stats, indent=2))
    vol.commit()

    from benchmark import accuracy

    out = {"prune": stats, **accuracy(DATA / "runs" / name / "weights" / "best.pt", root, 960, 0, batch=16)}
    (DATA / "runs" / name / "test_metrics_960.json").write_text(json.dumps(out, indent=2))
    vol.commit()
    print(json.dumps(out, indent=2))
    return out
