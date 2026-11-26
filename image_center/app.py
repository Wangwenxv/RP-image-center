from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from .backend import NativeBackend
from .jobs import JobManager, TERMINAL
from .workflow import MODEL_ID, Settings, Workflow, build_request, decode_data_url

STUDIO_DIR = Path(__file__).with_name("studio")

class BodyLimit:
    def __init__(self, app): self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http": return await self.app(scope, receive, send)
        total = 0
        async def limited():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > 36 * 1024 * 1024: raise HTTPException(413, "Request exceeds 36 MiB")
            return message
        await self.app(scope, limited, send)

def create_app(settings=None, backend_factory=NativeBackend):
    settings = settings or Settings.load()
    workflow = Workflow(settings)
    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = backend_factory(workflow)
        app.state.manager = JobManager(workflow, app.state.backend)
        app.state.manager.start()
        yield
        await app.state.manager.close()
    app = FastAPI(title="RP Image Center", version="1.0.0", lifespan=lifespan)
    app.add_middleware(BodyLimit)
    # RP can run from a local file, localhost, or a user-hosted HTTP(S) page. Bind the service to loopback.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["Content-Type", "Authorization", "Idempotency-Key"], allow_credentials=False)
    app.state.workflow = workflow
    app.mount("/studio-assets", StaticFiles(directory=STUDIO_DIR), name="studio-assets")

    @app.exception_handler(ValueError)
    async def invalid(request, error): return JSONResponse({"error": {"message": str(error)}}, status_code=400)
    @app.exception_handler(KeyError)
    async def missing(request, error): return JSONResponse({"error": {"message": "Unknown job"}}, status_code=404)
    @app.exception_handler(OverflowError)
    async def full(request, error): return JSONResponse({"error": {"message": str(error)}}, status_code=429)

    @app.get("/")
    async def index():
        return {"service": "RP Image Center", "studio": "/studio", "docs": "/docs", "health": "/health", "workflow": "/v1/workflow"}

    @app.get("/studio", include_in_schema=False)
    async def studio():
        return FileResponse(STUDIO_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    async def health(request: Request):
        backend = request.app.state.backend
        return {"ok": True, "service": "rp-image-center", "pid": os.getpid(), "root": str(settings.root),
                "model_files_validated": True, "native_ready": await backend.reachable(),
                "backend": backend.kind, "device": backend.device, "queue_size": request.app.state.manager.queue.qsize(),
                "active": [job.id for job in request.app.state.manager.jobs.values() if job.status in {"generating", "cancelling"}]}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": ident, "object": "model", "owned_by": "local"}
                                           for ident in workflow.model_ids()]}

    @app.get("/v1/workflow")
    async def describe(): return workflow.describe()

    async def enqueue(request: Request, editing=None):
        images = []
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            async with request.form(max_files=settings.max_references, max_fields=16, max_part_size=8*1024*1024) as form:
                data = {key: value for key, value in form.items() if not isinstance(value, UploadFile)}
                files = form.getlist("image[]") or form.getlist("image")
                for upload in files:
                    if not isinstance(upload, UploadFile): raise ValueError("image[] must contain uploaded files")
                    images.append(await upload.read(6*1024*1024+1))
                if "reference_roles" in data:
                    data["reference_roles"] = json.loads(data["reference_roles"])
                if "seed" in data:
                    data["seed"] = int(data["seed"])
        else:
            data = await request.json()
            if not isinstance(data, dict): raise ValueError("Request must be an object")
            image_fields = set(data) & {"image_data_urls", "image_urls"}
            if len(image_fields) > 1: raise ValueError("Use only one reference image field")
            inline = data.pop(next(iter(image_fields))) if image_fields else []
            if not isinstance(inline, list): raise ValueError("Reference images must be a list")
            images = [decode_data_url(item) for item in inline]
        allowed = {"model", "prompt", "size", "n", "response_format", "seed", "reference_roles",
                   "consistency_strength", "character_strength"}
        unknown = set(data) - allowed
        if unknown: raise ValueError(f"Unsupported request fields: {sorted(unknown)}")
        model = data.get("model", MODEL_ID)
        if model not in (*workflow.model_ids(), "sd-cpp-local", "qwen-image-2.1-turbo"):
            raise ValueError("Unknown model")
        if str(data.get("n", 1)) != "1": raise ValueError("Only n=1 is supported; GPU jobs are queued separately")
        if data.get("response_format", "b64_json") not in {"b64_json", "auto"}: raise ValueError("Only Base64 output is supported")
        roles = data.get("reference_roles", [])
        if not isinstance(roles, list) or any(not isinstance(role, str) for role in roles):
            raise ValueError("reference_roles must be a string list")
        if editing is True and not images: raise ValueError("Image editing requires at least one reference")
        if editing is False and images: raise ValueError("Use /v1/images/edits for reference images")
        body, prepared, roles = await asyncio.to_thread(
            build_request, workflow, data.get("prompt"), images, roles, size=data.get("size", "auto"),
            seed=data.get("seed", -1), consistency_strength=data.get("consistency_strength"),
            model=model, character_strength=data.get("character_strength"))
        manager = request.app.state.manager
        try:
            return manager.create(body, prepared, roles, request.headers.get("Idempotency-Key"))
        except ValueError as error:
            if "Idempotency-Key already" in str(error): raise HTTPException(409, str(error))
            raise

    @app.post("/v1/jobs", status_code=202)
    async def submit(request: Request):
        job = await enqueue(request)
        return job.snapshot()

    async def synchronous(request, editing):
        job = await enqueue(request, editing)
        try:
            while not job.done.is_set():
                if await request.is_disconnected():
                    request.app.state.manager.cancel(job.id)
                    raise HTTPException(499, "Client disconnected; native cancellation requested")
                await asyncio.sleep(0.5)
            if job.status == "cancelled": raise HTTPException(409, "Image generation cancelled")
            if job.status != "completed": raise HTTPException(503, job.error or "Image generation did not complete")
            return request.app.state.manager.result(job.id)
        except asyncio.CancelledError:
            request.app.state.manager.cancel(job.id)
            raise

    @app.post("/v1/images/generations")
    async def generate(request: Request): return await synchronous(request, False)
    @app.post("/v1/images/edits")
    async def edit(request: Request): return await synchronous(request, True)

    @app.get("/v1/jobs")
    async def jobs(request: Request):
        manager = request.app.state.manager
        return {"data": manager.history(limit=100)}

    @app.get("/v1/jobs/{identifier}")
    async def status(identifier: str, request: Request): return request.app.state.manager.get(identifier)
    @app.post("/v1/jobs/{identifier}/cancel")
    async def cancel(identifier: str, request: Request): return request.app.state.manager.cancel(identifier)
    @app.get("/v1/jobs/{identifier}/result")
    async def result(identifier: str, request: Request): return request.app.state.manager.result(identifier)
    @app.get("/v1/jobs/{identifier}/image")
    async def image(identifier: str, request: Request):
        manager = request.app.state.manager
        record = manager.get(identifier)
        if record["status"] != "completed": raise HTTPException(409, "Image not completed")
        return FileResponse(manager.image_path(identifier), media_type="image/png")

    @app.post("/admin/stop-native")
    async def stop_native(request: Request):
        manager = request.app.state.manager
        for job in manager.jobs.values():
            if job.status not in TERMINAL: manager.cancel(job.id)
        await request.app.state.backend.stop()
        return {"ok": True}
    return app

app = create_app()
