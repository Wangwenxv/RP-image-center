"""Controlled identity QA: same prompt, output size and seed; vary references and LoRA only."""
import base64
import json
from pathlib import Path
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT.parent / "RP" / "card" / "suqixue"
PROMPT = "生成一张自然真实的手机前置摄像头近景自拍照片。画面中只有主身份参考中的同一位年轻成年女性，脸部占画面主要区域，取景到胸部。保留主身份参考的脸型、眼型、眼距、鼻部与嘴唇的相对关系，发型、发色和妆容沿用主身份参考，不美化重造脸部。她在普通宿舍房间里，温暖柔和室内灯光，平静自然的表情，略微侧倾手机拍摄。一只手在镜头外持机。她穿服装参考里的白色红心睡衣，服装只借款式、颜色、图案与材质。摄影机位是手机前置近景自拍。参考图的旧衣服、首饰、雨伞、街景、麦克风和手势不自动继承。"
CASES = [
    ("A_three_identity_lora07", ["sqx1.jpg", "sqx2.jpg", "sqx3.jpg", "yifu.webp"], ["identity"]*3+["clothing"], 0.7),
    ("B_main_identity_lora07", ["sqx1.jpg", "yifu.webp"], ["identity", "clothing"], 0.7),
    ("C_main_identity_no_lora", ["sqx1.jpg", "yifu.webp"], ["identity", "clothing"], 0.0),
]
rows = []
with httpx.Client(base_url="http://127.0.0.1:7865", timeout=30, trust_env=False) as client:
    for label, names, roles, strength in CASES:
        files = [("image[]", (name, (REFERENCE_DIR/name).read_bytes())) for name in names]
        fields = {"model": "qwen-image-2.1-turbo-local", "prompt": PROMPT, "size": "1152x768", "seed": "47", "n": "1",
                  "reference_roles": json.dumps(roles), "consistency_strength": str(strength)}
        response = client.post("/v1/jobs", data=fields, files=files)
        response.raise_for_status()
        job = response.json()
        started = time.monotonic()
        print(f"{label}: {job['id']}", flush=True)
        while job["status"] not in {"completed", "failed", "cancelled"}:
            time.sleep(2)
            job = client.get(f"/v1/jobs/{job['id']}").json()
            if time.monotonic()-started > 1800:
                client.post(f"/v1/jobs/{job['id']}/cancel")
                raise RuntimeError("Identity comparison timed out")
        row = {"case": label, "job": job, "seconds": round(time.monotonic()-started, 2), "references": names, "lora_strength": strength}
        rows.append(row)
        (ROOT/"logs"/"identity-comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if job["status"] != "completed": raise RuntimeError(job.get("error") or job["status"])
