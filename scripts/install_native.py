"""Download pinned release assets in ranges; verify official SHA-256 before extraction."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import zlib
import httpx

ROOT = Path(__file__).resolve().parents[1]
TAG = "master-1127-79dc7ca"
ASSETS = {
    "vulkan": ("sd-master-79dc7ca-bin-win-vulkan-x64.zip", 30956796, "cc504eabda2001367ab21d752b27da60a884e7c5c61f84d79e006b5ea4e5dc9f"),
    "cuda": ("sd-master-79dc7ca-bin-win-cuda12-x64.zip", 413084402, "984b6e22803df7cf8ddff968c6e7abbb026b06f326e1d00471b484e59f891f2b"),
    "cudart": ("cudart-sd-bin-win-cu12-x64.zip", 563452046, "fe20366827d357c00797eebb58244dddab7fd9a348d70090c3871004c320f38d"),
}

def download(kind: str, workers: int, mirror: bool):
    name, size, digest = ASSETS[kind]
    folder = ROOT / "runtime" / "downloads"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / name
    if archive.exists() and hashlib.file_digest(archive.open("rb"), "sha256").hexdigest() == digest:
        print(f"Already verified: {name}", flush=True)
        return archive
    parts = folder / (name + ".parts")
    parts.mkdir(exist_ok=True)
    official = f"https://github.com/leejet/stable-diffusion.cpp/releases/download/{TAG}/{name}"
    url = "https://ghfast.top/" + official if mirror else official
    chunk = 2 * 1024 * 1024
    count = (size + chunk - 1) // chunk
    def part(index):
        start, end = index * chunk, min(size, (index + 1) * chunk) - 1
        target = parts / f"{index:05d}"
        if target.exists() and target.stat().st_size == end - start + 1:
            return
        for attempt in range(10):
            try:
                with httpx.Client(timeout=httpx.Timeout(150, connect=25), follow_redirects=True) as client:
                    # A distinct query prevents proxies from reusing a cached response to a different range.
                    with client.stream("GET", url + f"?rp_part={index}", headers={"Range": f"bytes={start}-{end}"}) as response:
                        if response.status_code == 429:
                            delay = max(30, min(120, int(response.headers.get("retry-after", "60"))))
                            print(f"Rate limited on range {index}; backing off {delay}s", flush=True)
                            time.sleep(delay)
                            continue
                        response.raise_for_status()
                        expected = f"bytes {start}-{end}/{size}"
                        if response.status_code != 206 or response.headers.get("content-range") != expected:
                            raise RuntimeError(f"Unexpected range response: {response.status_code} {response.headers.get('content-range')}")
                        tmp = target.with_suffix(".tmp")
                        with tmp.open("wb") as out:
                            for data in response.iter_bytes():
                                out.write(data)
                                if out.tell() > end - start + 1:
                                    raise RuntimeError("Range response exceeded expected length")
                if tmp.stat().st_size != end - start + 1:
                    raise RuntimeError("Truncated range")
                tmp.replace(target)
                return
            except Exception:
                if attempt == 9:
                    raise
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Could not download range {index}; retry later with fewer workers")
    print(f"Downloading {name}, {size / 2**20:.1f} MiB, {workers} workers; source={url}", flush=True)
    last = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for done, future in enumerate(concurrent.futures.as_completed([pool.submit(part, i) for i in range(count)]), 1):
            future.result()
            if time.monotonic() - last > 15 or done == count:
                print(f"{name}: {done}/{count} ranges ({done * 100 // count}%)", flush=True)
                last = time.monotonic()
    partial = archive.with_suffix(".assembling")
    hasher = hashlib.sha256()
    with partial.open("wb") as out:
        for i in range(count):
            with (parts / f"{i:05d}").open("rb") as piece:
                while data := piece.read(2 * 1024 * 1024):
                    hasher.update(data)
                    out.write(data)
    if partial.stat().st_size != size or hasher.hexdigest() != digest:
        raise RuntimeError(f"Official release digest mismatch for {name}; archive will not be executed")
    partial.replace(archive)
    print(f"Verified official SHA-256: {name} {digest}", flush=True)
    return archive

def extract(archive: Path, kind: str):
    destination = (ROOT / "runtime" / ("cuda" if kind == "cudart" else kind)).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError("Archive path escaped the runtime directory")
        for item in bundle.infolist():
            target = destination / item.filename
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == item.file_size:
                crc = 0
                with target.open("rb") as existing:
                    while data := existing.read(2 * 1024 * 1024): crc = zlib.crc32(data, crc)
                if crc & 0xffffffff == item.CRC:
                    continue  # Reinstalling the same pinned runtime need not overwrite a loaded DLL.
            bundle.extract(item, destination)
    print(f"Extracted to {destination}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["vulkan", "cuda", "cudart", "all"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--mirror", action="store_true", help="Use a download relay, always checked against the official release digest")
    args = parser.parse_args()
    kinds = ["vulkan", "cuda", "cudart"] if args.kind == "all" else [args.kind]
    for kind in kinds:
        extract(download(kind, max(1, min(args.workers, 24)), args.mirror), kind)
    (ROOT / "runtime" / "pin.json").write_text(json.dumps({"tag": TAG, "assets": ASSETS}, indent=2), encoding="utf-8")
