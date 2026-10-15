"""Print current CPU and GPU benchmark tables: python src/report_tables.py."""

import json
from pathlib import Path

NAMES = {
    "torch": "PyTorch FP32", "torch-fp16": "PyTorch FP16",
    "openvino": "OpenVINO FP32", "openvino-int8": "OpenVINO INT8",
    "engine-fp16": "TensorRT FP16", "engine-int8": "TensorRT INT8",
}


def benchmark_table(path):
    data = json.loads(path.read_text())
    rows = data["results"]
    baseline = {r["imgsz"]: r["median_ms"] for r in rows if r["format"] == "torch"}
    lines = [
        "| Input | Format | Size (MB) | Latency (ms) | FPS | Speed-up | Cone AP50 | Barrier AP50 | Stop AP50 | Mean AP50 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        scores = " | ".join(f"{r[c]['mAP50']:.3f}" for c in ("cone", "barrier", "stop_sign"))
        lines.append(
            f"| {r['imgsz']} | {NAMES[r['format']]} | {r['size_mb']:.2f} | {r['median_ms']:.2f} | "
            f"{r['fps']:.1f} | {baseline[r['imgsz']] / r['median_ms']:.2f}x | {scores} | {r['mean_mAP50']:.3f} |")
    return "\n".join(lines)


def main():
    for device in ("cpu", "gpu"):
        path = Path("results") / f"benchmark_{device}" / "results.json"
        print(f"### {device.upper()}\n\nSource: `{path.as_posix()}`\n")
        print(benchmark_table(path) + "\n")


if __name__ == "__main__":
    main()
