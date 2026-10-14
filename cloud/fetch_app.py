"""CPU-only Modal job: stream the nuImages samples archive and keep only the selected frames.

Kept separate from modal_app.py so it can run without GPU access. The ~16 GB archive is
streamed through tar inside Modal's network, so it never touches the local machine.

    modal run --detach cloud/fetch_app.py --url "https://.../nuimages-v1.0-all-samples.tgz"
"""

import subprocess
from pathlib import Path

import modal

APP_DIR = Path(__file__).resolve().parent.parent
app = modal.App("navperc-fetch")
vol = modal.Volume.from_name("navperc-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "curl")
    .uv_pip_install("ultralytics==8.4.177", "pyyaml")
    .add_local_file(APP_DIR / "data/nuimages/selection.json", "/assets/selection.json")
    .add_local_file(APP_DIR / "data/nuimages/selection.txt", "/assets/selection.txt")
    .add_local_python_source("navperc")
)

DATA = Path("/data")


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=16384, timeout=4 * 3600)
def fetch_nuimages(url: str):
    """Stream the (tens of GB) nuImages samples tar and keep only the selected frames."""
    out = DATA / "nuimages"
    out.mkdir(parents=True, exist_ok=True)
    # Strip CR in case the list was written on Windows; tar would otherwise look for "name\r".
    Path("/tmp/selection.txt").write_text(Path("/assets/selection.txt").read_text().replace("\r", ""))
    cmd = f'curl -sSL --retry 5 "{url}" | tar -xz -C {out} -T /tmp/selection.txt'
    # tar exits non-zero only if some listed member was missing; report instead of failing.
    rc = subprocess.run(["bash", "-o", "pipefail", "-c", cmd]).returncode
    n = sum(1 for _ in (out / "samples").rglob("*.jpg")) if (out / "samples").exists() else 0
    print(f"tar exit code {rc}; extracted {n} of {len(Path('/assets/selection.txt').read_text().split())} frames")
    vol.commit()

    from navperc.datasets.nuimages import convert
    from types import SimpleNamespace

    convert(SimpleNamespace(root=out, selection=Path("/assets/selection.json"), out=DATA / "yolo", max_side=1600))
    vol.commit()


@app.local_entrypoint()
def main(url: str):
    fetch_nuimages.remote(url)
