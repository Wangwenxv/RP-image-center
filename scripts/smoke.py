"""Generate real local pixels and record elapsed time and sampled GPU memory."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--reference", action="append", default=[])
parser.add_argument("--role", action="append", default=[])
parser.add_argument("--size", default="768x1024")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--prompt", default="A natural phone photograph of an empty Chinese university dorm bathroom, a rectangular wall mirror, white ceramic sink, warm indoor fluorescent light, tiled walls, no people.")
parser.add_argument("--url", default="http://127.0.0.1:7865")
parser.add_argument("--prompt-file")
parser.add_argument("--multipart", action="store_true")
args = parser.parse_args()
if args.prompt_file: args.prompt = Path(args.prompt_file).read_text(encoding="utf-8")
def gpu_memory():
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      timeout=5, text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()
        return max(int(line) for line in raw.splitlines())
    except Exception: return None
refs = []
for name in args.reference:
    file = Path(name)
    mime = "image/png" if file.suffix.lower() == ".png" else "image/webp" if file.suffix.lower() == ".webp" else "image/jpeg"
    refs.append(f"data:{mime};base64," + base64.b64encode(file.read_bytes()).decode())
body = {"model": "qwen-image-2.1-turbo-local", "prompt": args.prompt, "size": args.size,
        "seed": args.seed, "n": 1, "image_data_urls": refs, "reference_roles": args.role}
start = time.monotonic()
peak = gpu_memory() or 0
with httpx.Client(base_url=args.url, timeout=30, trust_env=False) as client:
    if args.multipart:
        fields = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else str(v)
                  for k, v in body.items() if k != "image_data_urls"}
        files = [("image[]", (Path(name).name, Path(name).read_bytes())) for name in args.reference]
        response = client.post("/v1/jobs", data=fields, files=files)
    else:
        response = client.post("/v1/jobs", json=body)
    response.raise_for_status()
    job = response.json()
    print(json.dumps(job, ensure_ascii=False), flush=True)
    last = ""
    last_log = 0
    while job["status"] not in {"completed", "failed", "cancelled", "interrupted"}:
        time.sleep(2)
        peak = max(peak, gpu_memory() or 0)
        response = client.get(f"/v1/jobs/{job['id']}")
        response.raise_for_status()
        job = response.json()
        progress = job.get("progress") or {}
        text = f"{job['status']} {job['phase']} {progress.get('completed')}/{progress.get('total')}"
        if text != last or time.monotonic() - last_log > 15:
            print(f"{time.monotonic()-start:.1f}s | {text} | sampled peak {peak} MiB", flush=True)
            last, last_log = text, time.monotonic()
        if time.monotonic() - start > 1800:
            client.post(f"/v1/jobs/{job['id']}/cancel")
            raise RuntimeError("Smoke timeout; cancellation requested")
    record = {"job": job, "elapsed_seconds": round(time.monotonic()-start, 2), "sampled_gpu_peak_mib": peak,
              "reference_files": args.reference, "scope": "real local model inference; sampled memory is not an exact peak"}
    (ROOT / "logs" / f"{job['id']}-smoke-report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    if job["status"] != "completed": raise RuntimeError(job.get("error") or job["status"])
