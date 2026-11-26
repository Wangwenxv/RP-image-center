"""Verify cancellation reaches the real native engine and the next job can run."""
import json
from pathlib import Path
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
with httpx.Client(base_url="http://127.0.0.1:7865", timeout=30, trust_env=False) as client:
    job = client.post("/v1/jobs", json={"prompt": "A phone photograph of an empty dorm bathroom with a wall mirror and ceramic sink, warm indoor light, no people.", "size": "1024x1024", "seed": 45}).json()
    start = time.monotonic()
    while not job.get("native_id"):
        time.sleep(0.1)
        job = client.get(f"/v1/jobs/{job['id']}").json()
        if job["status"] in {"failed", "cancelled", "completed"}: raise RuntimeError(f"Unexpected early state: {job}")
        if time.monotonic()-start > 120: raise RuntimeError("Job never reached the native engine")
    cancelled = client.post(f"/v1/jobs/{job['id']}/cancel").json()
    print(json.dumps(cancelled), flush=True)
    while job["status"] not in {"completed", "failed", "cancelled"}:
        time.sleep(0.2)
        job = client.get(f"/v1/jobs/{job['id']}").json()
        if time.monotonic()-start > 90: raise RuntimeError("Cancellation did not finish")
    assert job["status"] == "cancelled", job
    assert not (ROOT / "outputs" / f"{job['id']}.png").exists()
    following = client.post("/v1/jobs", json={"prompt": "An empty dorm bathroom with a ceramic sink and wall mirror, natural phone photo, warm indoor light.", "size": "256x256", "seed": 46}).json()
    while following["status"] not in {"completed", "failed", "cancelled"}:
        time.sleep(1)
        following = client.get(f"/v1/jobs/{following['id']}").json()
        if time.monotonic()-start > 240: raise RuntimeError("Following job did not finish")
    assert following["status"] == "completed", following
    report = {"cancelled": job, "following": following, "scope": "real native cancellation and subsequent real generation"}
    (ROOT / "logs" / "native-cancel-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
