"""Render the README.md result tables (Markdown) from the JSON files in results/.

    python scripts/report_tables.py > tables.md
"""

import json
from pathlib import Path

R = Path("results")
NAMES = {"torch": "PyTorch FP32", "onnx": "ONNX Runtime FP32", "openvino": "OpenVINO FP32",
         "openvino-int8": "OpenVINO INT8", "torch-fp16": "PyTorch FP16", "engine-fp16": "TensorRT FP16",
         "engine-int8": "TensorRT INT8"}


def load(name):
    return json.loads((R / name / "results.json").read_text())


def acc_index(name):
    return {(r["format"], r["imgsz"]): r for r in load(name)["results"]}


def speed_table(name, acc, title):
    d = load(name)
    base = {r["imgsz"]: r for r in d["results"] if r["format"] == "torch"}
    rows = [f"**{title}**\n",
            "| Input | Format | End-to-end (ms) | Inference (ms) | FPS | Speed-up vs PyTorch | Mean mAP50 |",
            "|---|---|---|---|---|---|---|"]
    for r in d["results"]:
        a = acc.get((r["format"], r["imgsz"]))
        rows.append(f"| {r['imgsz']} | {NAMES[r['format']]} | {r['median_ms']:.1f} | {r['inference_median_ms']:.1f} | "
                    f"{r['fps']:.0f} | {base[r['imgsz']]['median_ms'] / r['median_ms']:.2f}× | "
                    f"{a['mean_mAP50']:.3f} |" if a else "| — |")
    return "\n".join(rows)


def accuracy_table(acc):
    rows = ["| Input | Format | Size (MB) | Cone | Barrier | Stop sign | Mean mAP50 |", "|---|---|---|---|---|---|---|"]
    for (fmt, s), r in sorted(acc.items(), key=lambda kv: (kv[0][1], list(NAMES).index(kv[0][0]))):
        rows.append(f"| {s} | {NAMES[fmt]} | {r['size_mb']:.1f} | {r['cone']['mAP50']:.3f} | {r['barrier']['mAP50']:.3f} | "
                    f"{r['stop_sign']['mAP50']:.3f} | {r['mean_mAP50']:.3f} |")
    return "\n".join(rows)


def gpu_table():
    d = load("benchmark_gpu")
    base = {r["imgsz"]: r for r in d["results"] if r["format"] == "torch"}
    rows = ["| Input | Format | End-to-end (ms) | Inference (ms) | FPS | Speed-up | Cone | Barrier | Stop | Mean mAP50 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["results"]:
        rows.append(f"| {r['imgsz']} | {NAMES[r['format']]} | {r['median_ms']:.1f} | {r['inference_median_ms']:.2f} | {r['fps']:.0f} | "
                    f"{base[r['imgsz']]['median_ms'] / r['median_ms']:.2f}× | {r['cone']['mAP50']:.3f} | {r['barrier']['mAP50']:.3f} | "
                    f"{r['stop_sign']['mAP50']:.3f} | {r['mean_mAP50']:.3f} |")
    return d["env"].get("gpu", "?"), "\n".join(rows)


def distance_table():
    d = json.loads((R / "distance_eval/distance_eval.json").read_text())["detector_boxes"]["default_priors"]
    rows = ["| Class | True range | n | Median rel. error | Median abs. error | Within ±10% | Within ±20% |",
            "|---|---|---|---|---|---|---|"]
    for cls in ("cone", "barrier"):
        for b, a in d[cls].items():
            if a:
                label = "all" if b == "all" else b.replace("-inf", "+").replace("m", " m")
                rows.append(f"| {cls} | {label} | {a['n']} | {a['median_rel_err'] * 100:.1f}% | {a['median_abs_err_m']:.2f} m | "
                            f"{a['within_10pct'] * 100:.0f}% | {a['within_20pct'] * 100:.0f}% |")
    return "\n".join(rows)


def main():
    acc = acc_index("benchmark_cpu")
    pruned_acc = acc_index("benchmark_cpu_pruned_accuracy")
    print("### accuracy per export\n" + accuracy_table(acc) + "\n")
    for name, title in (("benchmark_cpu_p6", "Desktop: 6 performance cores (i5-13400F)"),
                        ("benchmark_cpu_e4", "Small-device proxy: 4 efficiency cores")):
        print(speed_table(name, acc, title) + "\n")
    print(speed_table("benchmark_cpu_pruned_p6", pruned_acc, "Pruned 30%, 6 performance cores") + "\n")
    print(speed_table("benchmark_cpu_pruned_e4", pruned_acc, "Pruned 30%, 4 efficiency cores") + "\n")
    gpu, t = gpu_table()
    print(f"### GPU ({gpu})\n" + t + "\n")
    print("### distance\n" + distance_table())


if __name__ == "__main__":
    main()
