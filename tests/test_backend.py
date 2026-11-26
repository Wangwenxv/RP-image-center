import asyncio
from types import SimpleNamespace

import httpx
import pytest
from image_center.backend import NativeBackend, SystemMemoryError
from image_center.workflow import ROOT, Settings, Workflow

def test_auto_fit_is_not_disabled_and_nvidia_is_selected(monkeypatch):
    workflow = Workflow(Settings())
    backend = NativeBackend(workflow)
    def exe():
        backend.kind = "vulkan"
        return ROOT / "runtime" / "vulkan" / "sd-server.exe"
    monkeypatch.setattr(backend, "executable", exe)
    monkeypatch.setattr("image_center.backend.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="Vulkan0\tIntel GPU\nVulkan1\tNVIDIA GeForce RTX 4060", stderr=""))
    command, _ = backend.command()
    assert backend.device == "vulkan1"
    assert "--params-backend" not in command
    assert command[command.index("--backend")+1] == "diffusion=vulkan1,te=vulkan1,vae=vulkan1"
    assert command[command.index("--auto-fit")+1] == "on"
    assert command[command.index("--model-args")+1] == "qwen_image_2_1_prefix_cache_type=q8_0,qwen_image_2_1_prefix_cache=true"
    assert "--type" not in command  # No implicit requantization of model weights.

def test_explicit_cpu_fallback_is_available(monkeypatch):
    workflow = Workflow(Settings(encoder_on_cpu=True, vae_on_cpu=True, params_backend="cpu"))
    backend = NativeBackend(workflow)
    def exe():
        backend.kind = "cuda"
        return ROOT / "runtime" / "cuda" / "sd-server.exe"
    monkeypatch.setattr(backend, "executable", exe)
    monkeypatch.setattr("image_center.backend.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="CUDA0\tNVIDIA RTX 4060", stderr=""))
    command, _ = backend.command()
    assert command[command.index("--backend")+1] == "diffusion=cuda0,te=cpu,vae=cpu"
    assert command[command.index("--params-backend")+1] == "cpu"


def test_native_lora_paths_keep_both_adapters(tmp_path):
    settings = Settings(root=tmp_path)
    gh = tmp_path / "一致性.safetensors"
    sqx = tmp_path / "sqx-test.safetensors"
    gh.write_bytes(b"gh")
    sqx.write_bytes(b"sqx")
    backend = NativeBackend(SimpleNamespace(settings=settings, lora=gh, sqx_lora=sqx))
    entries = [{"path": str(gh), "multiplier": 0.7}, {"path": str(sqx), "multiplier": 0.7}]
    assert backend.resolve_loras(entries) == [{"path": "qwen21-consistency.safetensors", "multiplier": 0.7},
                                              {"path": sqx.name, "multiplier": 0.7}]
    assert (tmp_path / "runtime" / "model-links" / sqx.name).read_bytes() == b"sqx"


def test_native_disconnect_reports_oom_only_for_current_job(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path)
    backend = NativeBackend(SimpleNamespace(settings=settings))
    log = tmp_path / "logs" / "native.log"
    log.parent.mkdir()
    oom = b"[E] ggml: CUDA error: out of memory\n"
    log.write_bytes(oom)
    backend.job_log_start = log.stat().st_size

    async def disconnected(*args, **kwargs):
        raise httpx.ReadError("")

    monkeypatch.setattr(backend.client, "get", disconnected)

    async def check():
        try:
            with pytest.raises(RuntimeError, match="Lost connection.*ReadError"):
                await backend.status("job_new")
            with log.open("ab") as target:
                target.write(oom)
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                await backend.status("job_new")
        finally:
            await backend.close()

    asyncio.run(check())


def test_system_memory_guard_rejects_start_and_terminates_only_owned_process(tmp_path, monkeypatch):
    backend = NativeBackend(SimpleNamespace(settings=Settings(root=tmp_path)))
    available = 6
    monkeypatch.setattr('image_center.backend.available_memory_gib', lambda: available)

    class Process:
        terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

    async def check():
        nonlocal available
        try:
            with pytest.raises(SystemMemoryError, match='10 GiB'):
                await backend.ensure_ready()
            assert backend.process is None
            process = backend.process = Process()
            available = 3
            await backend.monitor_memory()
            assert process.terminated
            assert isinstance(backend.failure('generic connection failure'), SystemMemoryError)
        finally:
            await backend.close()

    asyncio.run(check())
