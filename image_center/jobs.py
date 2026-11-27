from __future__ import annotations
import asyncio
import base64
from contextlib import closing
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import secrets
import sqlite3
import time
import uuid
from PIL import Image
from .backend import NativeMemoryError, SystemMemoryError
from .workflow import MODEL_ID, scaled_size

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}

@dataclass
class Job:
    id: str
    body: dict
    status: str = "queued"
    phase: str = "queued"
    native_id: str | None = None
    error: str | None = None
    progress: dict | None = None
    reference_count: int = 0
    attempt: int = 1
    render_size: tuple[int, int] | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def snapshot(self):
        return {"id": self.id, "status": self.status, "phase": self.phase, "native_id": self.native_id,
                "created": self.created, "started": self.started, "finished": self.finished,
                "progress": self.progress, "error": self.error, "width": self.body.get("width"), "height": self.body.get("height"),
                "mode": "edit" if self.reference_count else "t2i", "seed": self.body.get("seed"),
                "model": self.body.get("model", MODEL_ID),
                "steps": self.body.get("sample_params", {}).get("sample_steps"), "attempt": self.attempt, "render_size": self.render_size,
                "error_code": "gpu_memory" if self.error and self.error.startswith("GPU 显存不足")
                else "system_memory" if self.error and self.error.startswith("系统可用内存") else None,
                "loras": [{"name": Path(entry["path"]).name, "strength": entry["multiplier"]}
                          for entry in self.body.get("lora", [])],
                "result_url": f"/v1/jobs/{self.id}/result" if self.status == "completed" else None,
                "image_url": f"/v1/jobs/{self.id}/image" if self.status == "completed" else None}

class JobManager:
    def __init__(self, workflow, backend):
        self.workflow = workflow
        self.backend = backend
        self.settings = workflow.settings
        self.jobs = {}
        self.legacy_keys = {}
        self.queue = asyncio.Queue(maxsize=self.settings.queue_size)
        self.worker = None
        self.closing = False
        self.settings.outputs_dir.mkdir(exist_ok=True)
        self.database = self.settings.outputs_dir / "_history.sqlite3"
        self.schema_ready = False
        registry = self.settings.outputs_dir / "idempotency.json"
        if registry.exists():
            self.legacy_keys = json.loads(registry.read_text(encoding="utf-8"))
        for record in self.history():
            if record["status"] not in TERMINAL:
                record.update(status="interrupted", phase="interrupted", finished=time.time(), error="生图服务已重启，请重新提交任务。")
                self.save(record)

    def connect(self):
        db = sqlite3.connect(self.database)
        if not self.schema_ready:
            try:
                db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, created REAL NOT NULL, state TEXT NOT NULL,
                    request TEXT, request_key TEXT UNIQUE, fingerprint TEXT
                )""")
                db.execute("CREATE INDEX IF NOT EXISTS jobs_created ON jobs (created DESC)")
            except BaseException:
                db.close()
                raise
            self.schema_ready = True
        return closing(db)

    def save(self, record, request=None, key=None, fingerprint=None):
        with self.connect() as db, db:
            db.execute("""INSERT INTO jobs (id, created, state, request, request_key, fingerprint)
                VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET state = excluded.state""",
                (record["id"], record["created"], json.dumps(record, ensure_ascii=False),
                 json.dumps(request, ensure_ascii=False) if request is not None else None, key, fingerprint))

    def history(self, limit=None):
        records = {}
        if self.database.is_file():
            with self.connect() as db:
                rows = db.execute("""SELECT state FROM jobs WHERE json_extract(state, '$.status') <> 'deleted'
                                  ORDER BY created DESC LIMIT ?""", (-1 if limit is None else limit,))
                for row in rows:
                    record = json.loads(row[0])
                    records[record["id"]] = record
        # Continue to read galleries made by earlier versions.
        for filename in self.settings.outputs_dir.glob("img_*/state.json"):
            try:
                identifier = filename.parent.name
                if identifier not in records:
                    records[identifier] = self.get(identifier)
            except KeyError:
                continue
        for identifier in records.keys() & self.jobs.keys():
            records[identifier] = self.jobs[identifier].snapshot()
        return sorted(records.values(), key=lambda record: record["created"], reverse=True)[:limit]

    def start(self):
        self.worker = asyncio.create_task(self.run())

    def create(self, body, prepared, roles, key=None):
        if self.closing:
            raise RuntimeError("Service is stopping")
        fingerprint = None
        if key:
            if len(key) > 200:
                raise ValueError("Idempotency-Key too long")
            fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            key = hashlib.sha256(key.encode()).hexdigest()
            previous = self.legacy_keys.get(key)
            if previous is None and self.database.is_file():
                with self.connect() as db:
                    previous = db.execute("SELECT fingerprint, id FROM jobs WHERE request_key = ?", (key,)).fetchone()
            if previous:
                prior_hash, prior_id = previous
                if prior_hash != fingerprint:
                    raise ValueError("Idempotency-Key already used for a different request")
                prior = self.jobs.get(prior_id)
                if prior is None:
                    record = self.get(prior_id)
                    saved_body = {**body, "seed": record.get("seed", body["seed"])}
                    prior = Job(prior_id, saved_body, status=record["status"],
                                phase=record["phase"], native_id=record.get("native_id"), error=record.get("error"),
                                reference_count=len(prepared), attempt=record.get("attempt", 1), created=record["created"], started=record.get("started"),
                                finished=record.get("finished"))
                    if prior.status in TERMINAL:
                        prior.body.pop("ref_images", None)
                        prior.done.set()
                    prior.render_size = record.get("render_size")
                return prior
        if self.queue.full():
            raise OverflowError("Image queue is full")
        identifier = "img_" + uuid.uuid4().hex
        public_request = {k: v for k, v in body.items() if k != "ref_images"}
        public_request.update(reference_roles=roles, reference_sizes=[size for _, size in prepared],
                              workflow_sha256=self.workflow.sha256)
        job = Job(identifier, body, reference_count=len(prepared))
        self.save(job.snapshot(), public_request, key, fingerprint)
        self.jobs[identifier] = job
        self.queue.put_nowait(job)
        return job

    def get(self, identifier, *, include_deleted=False):
        if identifier in self.jobs:
            return self.jobs[identifier].snapshot()
        if not identifier.startswith("img_") or len(identifier) != 36 or any(c not in "0123456789abcdef" for c in identifier[4:]):
            raise KeyError(identifier)
        if self.database.is_file():
            with self.connect() as db:
                row = db.execute("SELECT state FROM jobs WHERE id = ?", (identifier,)).fetchone()
            if row:
                record = json.loads(row[0])
                if record["status"] == "deleted" and not include_deleted:
                    raise KeyError(identifier)
                return record
        filename = self.settings.outputs_dir / identifier / "state.json"
        try:
            record = json.loads(filename.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("id") != identifier or not {"status", "created"}.issubset(record):
                raise ValueError("Invalid job state")
            return record
        except (FileNotFoundError, ValueError) as error:
            raise KeyError(identifier) from error

    def delete_image(self, identifier):
        record = self.get(identifier, include_deleted=True)
        if record["status"] not in {"completed", "deleted"} or identifier in self.jobs:
            raise ValueError("Only completed images can be deleted")
        # Keep a deletion record so legacy galleries and idempotent requests cannot restore the image.
        if record["status"] == "completed":
            self.save({**record, "status": "deleted", "phase": "deleted", "result_url": None, "image_url": None})
        (self.settings.outputs_dir / f"{identifier}.png").unlink(missing_ok=True)
        (self.settings.outputs_dir / identifier / "image.png").unlink(missing_ok=True)

    def cancel(self, identifier):
        if identifier not in self.jobs:
            return self.get(identifier)
        job = self.jobs[identifier]
        if job.status not in TERMINAL:
            job.cancelled.set()
            job.status = "cancelled" if job.status == "queued" else "cancelling"
            job.phase = job.status
            if job.status == "cancelled":
                job.finished = time.time()
                job.done.set()
            self.save(job.snapshot())
        return job.snapshot()

    async def execute(self, job):
        job.status, job.phase, job.started = "generating", "loading", time.time()
        if job.body["seed"] == -1:
            job.body["seed"] = secrets.randbits(32)
        self.save(job.snapshot())
        try:
            await self.execute_attempt(job, job.body)
        except NativeMemoryError:
            if job.cancelled.is_set():
                raise asyncio.CancelledError
            if not self.settings.memory_retry:
                raise
            job.attempt = 2
            job.phase, job.native_id, job.progress = "retrying_memory", None, None
            self.save(job.snapshot())
            await self.backend.stop()
            body = self.retry_body(job)
            job.render_size = (body["width"], body["height"])
            await self.execute_attempt(job, body, low_memory=True)

    def retry_body(self, job):
        body = dict(job.body)
        # The native Qwen pipeline expands every reference to the output area.
        # Shrink the sampling canvas directly, then scale the final PNG back to the requested size.
        pixel_budget = min(768 * 768, 1_750_000 / (job.reference_count + 1))
        body["width"], body["height"] = scaled_size(body["width"], body["height"], pixel_budget, minimum=256, upscale=False)
        return body

    async def execute_attempt(self, job, body, *, low_memory=False):
        if low_memory:
            await self.backend.ensure_ready(job.cancelled, low_memory=True)
        else:
            await self.backend.ensure_ready(job.cancelled)
        if job.cancelled.is_set(): raise asyncio.CancelledError
        job.native_id = await self.backend.submit(body)
        job.phase = "generating"
        self.save(job.snapshot())
        deadline = time.monotonic() + self.settings.timeout_seconds
        cancel_sent = False
        cancel_deadline = None
        timed_out = False
        while True:
            if time.monotonic() > deadline and not timed_out:
                timed_out = True
                job.cancelled.set()
            if job.cancelled.is_set() and not cancel_sent:
                try:
                    await self.backend.cancel(job.native_id)
                except Exception as error:
                    if timed_out:
                        raise TimeoutError("Generation timed out while cancelling the native job") from error
                    raise
                cancel_sent = True
                cancel_deadline = time.monotonic() + 60
                job.status, job.phase = "cancelling", "cancelling"
            native = await self.backend.status(job.native_id)
            job.progress = native.get("progress")
            if not cancel_sent:
                job.phase = (job.progress or {}).get("stage", native["status"])
            if native["status"] in {"completed", "failed", "cancelled"}:
                if timed_out:
                    raise TimeoutError("Generation timed out; native execution has stopped")
                if job.cancelled.is_set() or native["status"] == "cancelled":
                    raise asyncio.CancelledError
                if native["status"] == "failed":
                    raise RuntimeError((native.get("error") or {}).get("message", "Native generation failed"))
                items = (native.get("result") or {}).get("images") or []
                if len(items) != 1 or not items[0].get("b64_json"):
                    raise RuntimeError("Native engine did not return one image")
                pixels = await asyncio.to_thread(self.prepare_result, job, body, items[0]["b64_json"])
                if job.cancelled.is_set():
                    raise asyncio.CancelledError
                image = self.settings.outputs_dir / f"{job.id}.png"
                temp = image.with_suffix(".tmp")
                try:
                    temp.write_bytes(pixels)
                    temp.replace(image)
                finally:
                    temp.unlink(missing_ok=True)
                job.status, job.phase = "completed", "completed"
                return
            # If the native engine cannot reach a cancellation checkpoint, stop the owned process.
            if cancel_sent and time.monotonic() > cancel_deadline:
                raise TimeoutError("Native cancellation stalled; owned native process was stopped")
            await asyncio.sleep(1)

    def prepare_result(self, job, body, encoded):
        pixels = base64.b64decode(encoded, validate=True)
        output_size = (job.body["width"], job.body["height"])
        with Image.open(io.BytesIO(pixels)) as image:
            image.load()
            if image.format != "PNG" or image.size != (body["width"], body["height"]):
                raise RuntimeError("Native result format/dimensions do not match the request")
            if image.size != output_size:
                output = io.BytesIO()
                image.resize(output_size, Image.Resampling.LANCZOS).save(output, format="PNG")
                pixels = output.getvalue()
        return pixels

    async def run(self):
        while not self.closing or not self.queue.empty():
            try:
                job = await asyncio.wait_for(self.queue.get(), timeout=self.settings.idle_shutdown_seconds)
            except TimeoutError:
                await self.backend.stop()
                continue
            try:
                if job.status == "cancelled": continue
                await self.execute(job)
            except asyncio.CancelledError:
                job.status, job.phase = "cancelled", "cancelled"
            except Exception as error:
                # Unknown native state must not overlap the next job. Stop the owned engine first.
                await self.backend.stop()
                if job.status == "cancelling" and job.cancelled.is_set() and not isinstance(error, (TimeoutError, SystemMemoryError)):
                    job.status, job.phase = "cancelled", "cancelled"
                    continue
                if isinstance(error, NativeMemoryError):
                    retry = "，低显存模式已重试一次" if job.attempt > 1 else ""
                    error = RuntimeError(f"GPU 显存不足或设备重置{retry}。请缩小画幅或减少参考图后重试。详情见 logs/native.log（{error}）")
                job.status, job.phase, job.error = "failed", "failed", str(error) or type(error).__name__
            finally:
                job.finished = time.time()
                job.body.pop("ref_images", None)
                self.save(job.snapshot())
                self.jobs.pop(job.id, None)
                job.done.set()
                self.queue.task_done()

    async def close(self):
        self.closing = True
        for job in self.jobs.values():
            if job.status not in TERMINAL: self.cancel(job.id)
        await self.backend.stop()
        if self.worker:
            self.worker.cancel()
            try: await self.worker
            except asyncio.CancelledError: pass
        await self.backend.close()

    def result(self, identifier):
        snapshot = self.get(identifier)
        if snapshot["status"] != "completed":
            raise ValueError("Job is not completed")
        data = self.image_path(identifier).read_bytes()
        return {"created": int(snapshot["finished"]), "data": [{"b64_json": base64.b64encode(data).decode()}],
                "job_id": identifier, "model": snapshot.get("model", MODEL_ID)}

    def image_path(self, identifier):
        output = self.settings.outputs_dir / f"{identifier}.png"
        if output.is_file():
            return output
        legacy = self.settings.outputs_dir / identifier / "image.png"
        if legacy.is_file():
            return legacy
        raise KeyError(identifier)
