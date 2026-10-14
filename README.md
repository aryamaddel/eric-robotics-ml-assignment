# ERIC Robotics ML Assignment | Arya Maddel

## Contact Information

- **Name:** Arya Maddel
- **Phone:** 8806642286
- **Email:** aryamaddel@gmail.com

Detects **cones, barriers and stop signs**, labels each with its **distance from the camera** (`Cone, 6.4m`), and runs fast on limited hardware. The full write-up (data, method, every table, extras, reproduction) is in **[REPORT.md](REPORT.md)**.

![Stop sign detected, No Entry sign ignored, cones with distances](results/demo/stop_sign_nuimages_0.jpg)

| Task | Approach | Result |
|---|---|---|
| Detection | COCO-pretrained **YOLO26n** fine-tuned on nuImages (cones, barriers) + Mapillary (stop signs) | test mAP50 **0.863 / 0.769 / 0.817** |
| Distance | pinhole geometry `Z = f·H/h` (focal length, pixel size, class height), ground-plane fallback | vs nuScenes 3D ground truth: median error **6.5% (cones), 9.6% (barriers)** |
| Edge optimization | INT8 quantization (OpenVINO), TensorRT, structured pruning, input size; small-device CPU emulation | CPU 960 px **54 → 24 ms** (−0.010 mAP50); 4 slow cores **34 FPS**; GPU (L4) **16.3 → 5.6 ms** |
| Extras | bird's-eye view (homography), optical-flow tracking, stereo disparity → depth derivation | REPORT §6 |

Key findings (details in REPORT.md):
- The first model confused **No Entry** with **stop** signs (47% correct on Singapore scenes) because of noisy automatic labels. Reviewing those labels fixed it (**94%**) and improved every class.
- **INT8 costs only ~0.01 mAP50** for a 2–3× CPU speed-up. Lowering the input size beats pruning for this nano model. Hand-crafted colour/shape detectors were tested and fell far short (AP50 0.16 vs 0.81).

## Quick start

```bash
uv sync
uv run python -m navperc.predict --weights weights/best.pt --camera configs/camera_generic_70deg.yaml --source <image|folder|video>
```

| Command | Purpose |
|---|---|
| `uv run python -m navperc.benchmark --weights weights/best.pt --data data/yolo --imgsz 512 640 960` | CPU speed + accuracy for PyTorch / ONNX / OpenVINO FP32 / INT8 |
| `NAVPERC_GPU=H100 uv run modal run --detach cloud/modal_app.py::train --batch 64 --workers 16` | train on a cloud GPU (Modal) |
| `uv run modal run cloud/modal_app.py::benchmark_gpu` | GPU speed (PyTorch vs TensorRT) |
| `uv run python -m navperc.eval_distance --nusc data/nuscenes --weights weights/best.pt` | distance error vs nuScenes 3D ground truth |

Dataset preparation, all commands, and a table of where every setting lives (model, GPU, hyperparameters, quantization, pruning) are in [REPORT.md §7](REPORT.md#7-reproducing-and-where-everything-is-set).

## Layout

- `src/navperc/`: package (`predict`, `distance`, `benchmark`, `eval_distance`, `bev`, `track`, `handcrafted`, `datasets/`)
- `cloud/`: Modal jobs (data fetch, training, GPU benchmark, pruning)
- `configs/`: camera intrinsics
- `weights/`: final model + pruned model
- `results/`: metrics JSON, figures, demo images/video

Data licences (nuScenes/nuImages, Mapillary) are non-commercial; no dataset files are included.
