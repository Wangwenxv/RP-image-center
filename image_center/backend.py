from __future__ import annotations
import asyncio
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import httpx
from .workflow import Workflow

class NativeMemoryError(RuntimeError):
    pass


class SystemMemoryError(RuntimeError):
    pass


def available_memory_gib():
    if os.name != "nt":
        return None
    # MEMORYSTATUSEX starts with two DWORDs, followed by seven DWORDLONGs.
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                    ("total_phys", ctypes.c_uint64), ("avail_phys", ctypes.c_uint64),
                    ("total_page", ctypes.c_uint64), ("avail_page", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64), ("avail_virtual", ctypes.c_uint64),
                    ("avail_extended", ctypes.c_uint64)]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return min(status.avail_phys, status.avail_page) / 1024**3
    return None


class NativeBackend:
    RETRY_VRAM_GIB = 5.0

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.settings = workflow.settings
        self.process = None
        self.log_file = None
        self.job_log_start = 0
        self.kind = None
        self.device = None
        self.low_memory = False
        self.monitor = None
        self.memory_stopped = False
        self.lock = asyncio.Lock()
        self.client = httpx.AsyncClient(base_url=self.settings.native_url, timeout=30, trust_env=False)

    def executable(self):
        windows = os.name == "nt"
        filename = "sd-server.exe" if windows else "sd-server"
        kinds = ["cuda", "vulkan"] if self.settings.backend == "auto" else [self.settings.backend]
        if self.low_memory and self.settings.backend == "auto":
            kinds.reverse()
        for kind in kinds:
            directory = self.settings.root / "runtime" / kind
            if windows and kind == "cuda" and not any(directory.rglob("cudart64*.dll")):
                continue
            for executable in directory.rglob(filename):
                if executable.is_file() and (windows or os.access(executable, os.X_OK)):
                    self.kind = kind
                    return executable
        if windows:
            raise RuntimeError("Native runtime missing. Run scripts/install_native.py; CUDA requires both cuda and cudart assets")
        raise RuntimeError("Native runtime missing. Run bash scripts/install_native_linux.sh for CUDA")

    def command(self):
        executable = self.executable()
        environment = os.environ.copy()
        if os.name == "nt":
            dll_dirs = {str(item.parent) for item in executable.parent.rglob("*.dll")}
            dll_dirs.update(str(item.parent) for item in (self.settings.root / "runtime" / self.kind).rglob("cudart64*.dll"))
            environment["PATH"] = os.pathsep.join(sorted(dll_dirs)) + os.pathsep + environment.get("PATH", "")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        listing = subprocess.run([str(executable), "--list-devices"], capture_output=True, text=True,
                                 errors="replace", env=environment, creationflags=flags, timeout=45)
        devices = listing.stdout + "\n" + listing.stderr
        candidates = []
        for line in devices.splitlines():
            match = re.match(r"^((?:CUDA|Vulkan)\d+)(?:\t|\\t)(.*)", line, re.I)
            if match and "NVIDIA" in match[2].upper():
                candidates.append(match[1].lower())
        if not candidates:
            raise RuntimeError(f"No NVIDIA {self.kind} compute device found: {devices[-1500:]}")
        self.device = candidates[0]
        budget = min(self.RETRY_VRAM_GIB, self.settings.max_vram_gib) if self.low_memory else self.settings.max_vram_gib
        vae_backend = "cpu" if self.settings.vae_on_cpu else self.device
        command = [str(executable), "--listen-ip", "127.0.0.1", "--listen-port", str(self.settings.native_port),
                   "--diffusion-model", str(self.workflow.diffusion), "--llm", str(self.workflow.encoder),
                   "--vae", str(self.workflow.vae), "--lora-model-dir", str(self.settings.root / "runtime" / "model-links"),
                   "--backend", f"diffusion={self.device},te={'cpu' if self.settings.encoder_on_cpu else self.device},vae={vae_backend}",
                   "--auto-fit", "on", "--max-vram", str(budget), "--threads", str(self.settings.cpu_threads), "--lora-apply-mode", "at_runtime",
                   "--model-args", f"qwen_image_2_1_prefix_cache_type={self.settings.prefix_cache_type},qwen_image_2_1_prefix_cache={str(self.settings.prefix_cache_enabled).lower()}",
                   "--sampling-method", "euler", "--cfg-scale", "1", "--steps", "8",
                   "--sigmas", "qwen-image-2.1-turbo", "--weight-staging-ttl", "30", "--weight-residency-ttl", "90"]
        if self.settings.params_backend != "auto":
            command.extend(["--params-backend", self.settings.params_backend])
        if self.low_memory:
            command.append("--disable-prefetch")
        return command, environment

    def lora_alias(self, source, name):
        # The native process gets ASCII file names even on systems with legacy ANSI argv handling.
        directory = self.settings.root / "runtime" / "model-links"
        directory.mkdir(parents=True, exist_ok=True)
        alias = directory / name
        if alias.exists():
            if not alias.samefile(source):
                source_stat, alias_stat = source.stat(), alias.stat()
                if (alias_stat.st_size, alias_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
                    raise RuntimeError(f"LoRA alias differs from its source: {alias}")
        else:
            try: os.link(source, alias)
            except OSError: shutil.copy2(source, alias)
        return alias

    async def reachable(self, force=False):
        if not force and self.process is None:
            return False
        try:
            response = await self.client.get("/sdcpp/v1/capabilities", timeout=0.5)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure_ready(self, cancel_event=None, *, low_memory=False):
        async with self.lock:
            if cancel_event and cancel_event.is_set():
                raise asyncio.CancelledError
            if self.low_memory == low_memory and self.process and self.process.poll() is None and await self.reachable():
                available = available_memory_gib()
                if available is None or available >= self.settings.start_ram_gib:
                    return
                # Drop the previous model caches before deciding whether the next job can start.
            await self.stop()
            self.memory_stopped = False
            self.check_system_memory(self.settings.start_ram_gib)
            if await self.reachable(force=True):
                raise RuntimeError(f"Native port {self.settings.native_port} is already occupied; stop the existing instance first")
            self.low_memory = low_memory
            self.lora_alias(self.workflow.lora, "qwen21-consistency.safetensors")
            command, environment = await asyncio.to_thread(self.command)
            log_dir = self.settings.root / "logs"
            log_dir.mkdir(exist_ok=True)
            self.log_file = (log_dir / "native.log").open("ab", buffering=0)
            self.job_log_start = os.fstat(self.log_file.fileno()).st_size
            self.process = subprocess.Popen(command, cwd=str(Path(command[0]).parent), env=environment,
                                            stdout=self.log_file, stderr=subprocess.STDOUT,
                                            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0)
            self.monitor = asyncio.create_task(self.monitor_memory())
            record = {"pid": self.process.pid, "executable": command[0], "kind": self.kind, "device": self.device, "command": command}
            (self.settings.root / "runtime" / "native-process.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            deadline = time.monotonic() + self.settings.native_start_timeout
            while time.monotonic() < deadline:
                if cancel_event and cancel_event.is_set():
                    await self.stop()
                    raise asyncio.CancelledError
                if self.process.poll() is not None:
                    raise self.failure(f"Native process exited ({self.process.returncode}); see logs/native.log")
                if await self.reachable():
                    return
                await asyncio.sleep(1)
            await self.stop()
            raise RuntimeError("Native startup timed out; see logs/native.log")

    def check_system_memory(self, minimum):
        if self.memory_stopped:
            raise SystemMemoryError("系统可用内存低于保护阈值，本机生图已停止并释放推理进程。")
        available = available_memory_gib()
        if available is not None and available < minimum:
            raise SystemMemoryError(f"系统可用内存仅 {available:.1f} GiB，低于本机生图的 {minimum:g} GiB 阈值。请释放内存后重试。")

    async def monitor_memory(self):
        minimum = self.settings.reserve_ram_gib
        if minimum <= 0:
            return
        low_samples = 0
        while self.process and self.process.poll() is None:
            available = available_memory_gib()
            low_samples = low_samples + 1 if available is not None and available < minimum else 0
            if low_samples >= 2:
                self.memory_stopped = True
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
                return
            await asyncio.sleep(1)

    async def submit(self, body):
        native_body = dict(body)
        native_body.pop("model", None)
        if native_body.get("lora"):
            native_body["lora"] = await asyncio.to_thread(self.resolve_loras, native_body["lora"])
        self.job_log_start = os.fstat(self.log_file.fileno()).st_size if self.log_file else 0
        try:
            response = await self.client.post("/sdcpp/v1/img_gen", json=native_body)
        except httpx.TransportError as error:
            raise self.failure(f"Lost connection while submitting to the native engine ({type(error).__name__}); see logs/native.log") from error
        if not response.is_success:
            raise self.failure(f"Native submission HTTP {response.status_code}: {response.text[:500]}")
        return response.json()["id"]

    def resolve_loras(self, entries):
        names = {str(self.workflow.lora): "qwen21-consistency.safetensors"}
        if self.workflow.sqx_lora:
            names[str(self.workflow.sqx_lora)] = self.workflow.sqx_lora.name
        return [{**entry, "path": self.lora_alias(Path(entry["path"]), names[entry["path"]]).name}
                for entry in entries]

    async def status(self, job_id):
        try:
            response = await self.client.get(f"/sdcpp/v1/jobs/{job_id}")
        except httpx.TransportError as error:
            raise self.failure(f"Lost connection to the native engine ({type(error).__name__}); see logs/native.log") from error
        response.raise_for_status()
        state = response.json()
        if state["status"] == "failed":
            error = state.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            raise self.failure(message or "Native generation failed; see logs/native.log")
        return state

    def failure(self, message):
        if self.memory_stopped:
            return SystemMemoryError("系统可用内存低于保护阈值，本机生图已停止并释放推理进程。")
        diagnostic = self.read_failure() or message
        if re.search(r"out of (?:device )?memory|error_out_of_device_memory|cannot make enough memory available|"
                     r"failed to allocate .*?(?:buffer|memory)|device lost|error_device_lost", diagnostic, re.I):
            return NativeMemoryError(diagnostic)
        return RuntimeError(message)

    def read_failure(self):
        filename = self.settings.root / "logs" / "native.log"
        try:
            with filename.open("rb") as source:
                end = source.seek(0, os.SEEK_END)
                source.seek(max(self.job_log_start, end - 65536))
                tail = source.read(65536)
        except OSError:
            return None
        markers = [(b"cuda error: out of memory", "CUDA out of memory"),
                   (b"cannot make enough memory available", "GPU memory allocation failed: cannot make enough memory available"),
                   (b"ggml_vulkan: device lost", "Vulkan device lost"),
                   (b"error_out_of_device_memory", "Vulkan out of device memory")]
        tail = tail.lower()
        position, message = max((tail.rfind(marker), message) for marker, message in markers)
        return message if position >= 0 else None

    async def cancel(self, job_id):
        response = await self.client.post(f"/sdcpp/v1/jobs/{job_id}/cancel")
        response.raise_for_status()
        return response.json()

    async def stop(self):
        if self.monitor:
            self.monitor.cancel()
            await asyncio.gather(self.monitor, return_exceptions=True)
            self.monitor = None
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: await asyncio.to_thread(self.process.wait, 10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
        self.process = None
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    async def close(self):
        await self.stop()
        await self.client.aclose()
