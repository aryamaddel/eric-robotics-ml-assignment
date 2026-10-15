"""Download selected members from large remote zip archives using HTTP range requests.

MTSD ships its images as ~10 GB zips, but we only need the few thousand images
that contain stop signs. A zip's central directory (at the end of the file)
lists every member's offset, so we read that, then fetch just the byte ranges
of the members we want.

Usage:
    python src/fetch_remote_zip_members.py --urls urls.txt --keys keys.txt --out data/mtsd/images
where urls.txt has one (signed) zip URL per line and keys.txt one image key per
line (member basename without extension).
"""

import argparse
import io
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

LOCAL_HEADER = struct.Struct("<4s5H3L2H")  # 30 bytes


def fetch_range(session, url, start, end):
    r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"server ignored range request ({r.status_code})")
    return r.content


class RangeFile:
    """Minimal seekable file over HTTP so zipfile can parse the central directory."""

    def __init__(self, session, url):
        self.session, self.url, self.pos = session, url, 0
        self.size = int(session.head(url, allow_redirects=True, timeout=60).headers["Content-Length"])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = fetch_range(self.session, self.url, self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data


def list_members(session, url):
    with zipfile.ZipFile(RangeFile(session, url)) as zf:
        return zf.infolist()


def download_member(session, url, info, dest):
    # Local header has its own filename/extra lengths, so over-fetch a little and parse.
    start = info.header_offset
    end = start + LOCAL_HEADER.size + len(info.orig_filename.encode()) + 1024 + info.compress_size
    blob = fetch_range(session, url, start, end)
    sig, *_, name_len, extra_len = LOCAL_HEADER.unpack_from(blob)
    if sig != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {info.filename}")
    off = LOCAL_HEADER.size + name_len + extra_len
    data = blob[off : off + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError(f"unsupported compression {info.compress_type}")
    if zlib.crc32(data) != info.CRC:
        raise RuntimeError(f"CRC mismatch for {info.filename}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", required=True, type=Path)
    ap.add_argument("--keys", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    urls = [u.strip() for u in args.urls.read_text().splitlines() if u.strip()]
    wanted = set(args.keys.read_text().split())
    args.out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    jobs = []
    for i, url in enumerate(urls):
        members = list_members(session, url)
        hits = [m for m in members if not m.is_dir() and Path(m.filename).stem in wanted]
        print(f"zip {i}: {len(members)} members, {len(hits)} wanted")
        jobs += [(url, m) for m in hits]

    todo = [(u, m, args.out / Path(m.filename).name) for u, m in jobs]
    todo = [t for t in todo if not t[2].exists()]
    found = {Path(m.filename).stem for _, m in jobs}
    print(f"{len(found)}/{len(wanted)} keys located, {len(todo)} to download")

    failures = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(download_member, session, u, m, d): m.filename for u, m, d in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # keep going; rerunning resumes
                failures += 1
                print(f"failed {futs[fut]}: {e}")
            if n % 200 == 0:
                print(f"{n}/{len(todo)}")
    print(f"done, {failures} failures")


if __name__ == "__main__":
    main()
