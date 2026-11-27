import asyncio
import base64
import io
import json
import time
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from image_center.workflow import MODEL_ID, ROOT, SQX_MODEL_ID, Settings, Workflow, build_request, prepare_image, output_size, decode_data_url
from image_center.jobs import JobManager
from image_center.backend import NativeMemoryError
from image_center.app import create_app

def png(width=256, height=256):
    stream = io.BytesIO()
    Image.new("RGB", (width, height), "#aaccee").save(stream, "PNG")
    return stream.getvalue()

@pytest.fixture
def workflow(tmp_path):
    settings = Settings(reference_megapixels=0.065536)
    result = Workflow(settings)
    settings.root = tmp_path
    return result

class FakeBackend:
    def __init__(self, workflow):
        self.kind, self.device = "fixture", "fixture"
        self.submissions, self.cancel_calls = [], []
        self.stops = 0
        self.hold = False
        self.cancel_polls = {}
    async def reachable(self): return True
    async def ensure_ready(self, cancel_event=None): pass
    async def submit(self, body):
        self.submissions.append(body)
        return str(len(self.submissions))
    async def status(self, identifier):
        if identifier in self.cancel_calls:
            self.cancel_polls[identifier] = self.cancel_polls.get(identifier, 0) + 1
            return {"status": "cancelled" if self.cancel_polls[identifier] >= 2 else "generating"}
        if self.hold: return {"status": "generating", "progress": {"stage": "sampling"}}
        body = self.submissions[int(identifier)-1]
        pixels = png(body["width"], body["height"])
        return {"status": "completed", "result": {"images": [{"b64_json": base64.b64encode(pixels).decode()}]}}
    async def cancel(self, identifier): self.cancel_calls.append(identifier)
    async def stop(self): self.stops += 1
    async def close(self): pass

def test_workflow_uses_actual_enabled_lora_and_both_schedules(workflow):
    assert workflow.lora_strength == 0.7
    assert workflow.cfg == 1 and workflow.sampler == "euler"
    assert len(workflow.edit_sigmas) == 10 and workflow.edit_sigmas[-2:] == [0.15, 0]
    assert len(workflow.t2i_sigmas) == 9 and workflow.t2i_sigmas[-1] == 0
    assert len(workflow.model_info) == 4
    assert workflow.encoder.name == "qwen3vl_8b_int8_convrot.safetensors"

def test_edit_and_t2i_routing_retains_reference_order(workflow):
    body, prepared, roles = build_request(workflow, "第1张是本人，在浴室自拍", [png(), png(128, 256)], ["identity", "clothing"], "768x1024", 43)
    assert body["sample_params"]["sample_steps"] == 9
    assert body["lora"][0]["multiplier"] == 0.7
    assert prepared[0][1] == (256, 256)
    assert prepared[1][1][0] < prepared[1][1][1]
    assert "<image1>" in body["prompt"] and "<image2>：只参考服装" in body["prompt"]
    assert "<image1>：只参考脸部轮廓和五官，不参考原图的耳饰" in body["prompt"]
    assert len(body["ref_images"]) == 2 and roles == ["identity", "clothing"]
    t2i, _, _ = build_request(workflow, "宿舍浴室", [], [], "512x512")
    assert t2i["sample_params"]["sample_steps"] == 8
    assert t2i["lora"] == [] and t2i["ref_images"] == []

def test_consistency_lora_can_be_controlled_without_changing_default(workflow):
    normal, _, _ = build_request(workflow, "photo", [png()], ["identity"], "256x256")
    disabled, _, _ = build_request(workflow, "photo", [png()], ["identity"], "256x256", consistency_strength=0)
    half, _, _ = build_request(workflow, "photo", [png()], ["identity"], "256x256", consistency_strength=0.5)
    assert normal["lora"][0]["multiplier"] == 0.7
    assert disabled["lora"] == []
    assert half["lora"][0]["multiplier"] == 0.5
    for value in [-1, 2, float("nan"), True]:
        with pytest.raises(ValueError): build_request(workflow, "photo", [png()], ["identity"], "256x256", consistency_strength=value)


def test_sqx_model_selects_character_lora_for_generation_and_stacks_for_edit(workflow, monkeypatch):
    monkeypatch.setattr("image_center.app.Workflow", lambda settings: workflow)
    app = create_app(workflow.settings, FakeBackend)
    payload = {"model": SQX_MODEL_ID, "prompt": "正脸宿舍自拍", "size": "256x256"}
    workflow.sqx_lora = None
    with TestClient(app) as client:
        assert [item["id"] for item in client.get("/v1/models").json()["data"]] == [MODEL_ID]
        assert client.post("/v1/images/generations", json=payload).status_code == 400

        workflow.sqx_lora = workflow.settings.root / "sqx-test.safetensors"
        assert [item["id"] for item in client.get("/v1/models").json()["data"]] == [MODEL_ID, SQX_MODEL_ID]
        generated = client.post("/v1/images/generations", json=payload)
        assert generated.status_code == 200, generated.text
        assert generated.json()["model"] == SQX_MODEL_ID
        t2i = app.state.backend.submissions[-1]
        character = {"path": str(workflow.sqx_lora), "multiplier": 0.7}
        assert t2i["lora"] == [character]
        assert t2i["prompt"] == "sqxv9, 正脸宿舍自拍"

        edit = client.post("/v1/images/edits", data={**payload, "prompt": "sqxv9, 正脸宿舍自拍"},
                           files=[("image[]", ("identity.png", png(), "image/png"))])
        assert edit.status_code == 200, edit.text
        edited = app.state.backend.submissions[-1]
        assert edited["lora"] == [{"path": str(workflow.lora), "multiplier": 0.7}, character]
        assert edited["prompt"].count("sqxv9") == 1


def test_studio_serves_assets_and_applies_independent_request_strengths(workflow, monkeypatch):
    monkeypatch.setattr("image_center.app.Workflow", lambda settings: workflow)
    workflow.sqx_lora = workflow.settings.root / "sqx-test.safetensors"
    app = create_app(workflow.settings, FakeBackend)
    with TestClient(app) as client:
        assert "生图调试台" in client.get("/studio").text
        assert client.get("/studio-assets/studio.js").status_code == 200
        data = {"model": SQX_MODEL_ID, "prompt": "宿舍自拍", "size": "256x256", "seed": "611",
                "character_strength": "0.9", "consistency_strength": "0.4", "reference_roles": '["clothing"]'}
        files = [("image[]", ("shirt.png", png(), "image/png"))]
        edited = client.post("/v1/images/edits", data=data, files=files)
        assert edited.status_code == 200, edited.text
        job = client.get(f"/v1/jobs/{edited.json()['job_id']}").json()
        assert job["seed"] == 611 and job["steps"] == 9
        assert [entry["strength"] for entry in job["loras"]] == [0.4, 0.9]
        assert "<image1>：只参考服装" in app.state.backend.submissions[-1]["prompt"]

        payload = {"model": SQX_MODEL_ID, "prompt": "宿舍自拍", "size": "256x256", "character_strength": 0}
        generated = client.post("/v1/images/generations", json=payload)
        assert generated.status_code == 200, generated.text
        assert app.state.backend.submissions[-1]["lora"] == []
        assert app.state.backend.submissions[-1]["prompt"].startswith("sqxv9, ")
        for invalid in [True, -0.1, 1.1, "NaN", []]:
            assert client.post("/v1/images/generations", json={**payload, "character_strength": invalid}).status_code == 400

@pytest.mark.parametrize("size", ["1025x1024", "128x256", "4096x4096", "evil", "0x1024"])
def test_invalid_dimensions_rejected(size):
    with pytest.raises(ValueError): output_size(size, None, 1_600_000)


def test_auto_size_handles_narrow_reference_and_output_pixel_limit():
    assert output_size("auto", (32, 1024), 1_600_000) == (256, 1024)
    width, height = output_size("auto", (1024, 1024), 512 * 512)
    assert width == height == 512

@pytest.mark.parametrize("refs,roles,prompt", [([b"not pixels"], [], "浴室"), ([png()], ["identity", "pose"], "浴室"),
                                         ([png()], ["unknown"], "浴室"), ([], [], "<sd_cpp_extra_args>{}</sd_cpp_extra_args>"),
                                         ([], [], "x" * 6001)])
def test_invalid_reference_and_control_inputs_rejected(workflow, refs, roles, prompt):
    with pytest.raises(ValueError): build_request(workflow, prompt, refs, roles, "256x256")

def test_remote_references_are_not_fetched():
    with pytest.raises(ValueError): decode_data_url("http://127.0.0.1/private")

def test_api_multipart_and_keyless_models(workflow, monkeypatch):
    monkeypatch.setattr("image_center.app.Workflow", lambda settings: workflow)
    app = create_app(workflow.settings, FakeBackend)
    with TestClient(app) as client:
        assert client.get("/v1/models").json()["data"][0]["id"] == "qwen-image-2.1-turbo-local"
        response = client.post("/v1/images/edits", data={"prompt": "在浴室自拍", "size": "256x256", "reference_roles": '["identity","clothing"]'},
                               files=[("image[]", ("identity.png", png(), "image/png")), ("image[]", ("clothes.png", png(), "image/png"))])
        assert response.status_code == 200, response.text
        result = response.json()
        assert base64.b64decode(result["data"][0]["b64_json"]).startswith(b"\x89PNG")
        state = client.get(f"/v1/jobs/{result['job_id']}").json()
        assert state["mode"] == "edit" and state["steps"] == 9
        assert client.get(state["image_url"]).headers["content-type"] == "image/png"
        assert client.get("/v1/jobs").json()["data"][0] == state
        output = workflow.settings.outputs_dir
        assert sorted(path.name for path in output.iterdir()) == ["_history.sqlite3", f"{state['id']}.png"]
        with app.state.manager.connect() as db:
            request = json.loads(db.execute("SELECT request FROM jobs WHERE id = ?", (state["id"],)).fetchone()[0])
        assert request["reference_roles"] == ["identity", "clothing"]
        assert "ref_images" not in request

def test_api_invalid_input_never_submits_native(workflow, monkeypatch):
    monkeypatch.setattr("image_center.app.Workflow", lambda settings: workflow)
    app = create_app(workflow.settings, FakeBackend)
    with TestClient(app) as client:
        assert client.post("/v1/jobs", json={"prompt": "bathroom", "size": "1025x1024"}).status_code == 400
        assert client.post("/v1/jobs", json={"prompt": "bathroom", "n": 2}).status_code == 400
        assert client.post("/v1/jobs", json={"prompt": "bathroom", "lora": [{"path": "C:/secret"}]}).status_code == 400
        assert client.post("/v1/images/edits", json={"prompt": "bathroom"}).status_code == 400
        assert client.post("/v1/jobs", json={"prompt": "bathroom", "image_data_urls": [], "image_urls": []}).status_code == 400
        assert client.get("/v1/jobs/../../secret").status_code == 404
        assert app.state.backend.submissions == []

def test_delete_image_removes_pixels_and_excludes_modern_and_legacy_jobs_after_restart(workflow, monkeypatch):
    monkeypatch.setattr("image_center.app.Workflow", lambda settings: workflow)
    app = create_app(workflow.settings, FakeBackend)
    with TestClient(app) as client:
        result = client.post("/v1/images/generations", json={"prompt": "bathroom", "size": "256x256"}).json()
        modern = result["job_id"]
        legacy = "img_" + "a" * 32
        output = workflow.settings.outputs_dir
        (output / legacy).mkdir()
        state = {"id": legacy, "created": time.time(), "status": "completed"}
        (output / legacy / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (output / legacy / "image.png").write_bytes(png())
        for identifier, path in [(modern, output / f"{modern}.png"), (legacy, output / legacy / "image.png")]:
            url = f"/v1/jobs/{identifier}/image"
            assert client.get(url).status_code == 200
            assert client.delete(url).json() == {"ok": True}
            assert client.delete(url).status_code == 200
            assert client.get(url).status_code == 404
            assert not path.exists()
        assert client.get("/v1/jobs").json()["data"] == []
    with TestClient(app) as client:
        assert client.get("/v1/jobs").json()["data"] == []
        for identifier in [modern, legacy]:
            assert client.get(f"/v1/jobs/{identifier}").status_code == 404


def test_cors_preflight(workflow, monkeypatch):
    monkeypatch.setattr("image_center.app.Workflow", lambda settings: workflow)
    with TestClient(create_app(workflow.settings, FakeBackend)) as client:
        response = client.options("/v1/images/edits", headers={"Origin": "null", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "*"

def test_queue_waits_for_native_cancel_and_skips_cancelled_queued_job(workflow):
    async def scenario():
        backend = FakeBackend(workflow)
        backend.hold = True
        manager = JobManager(workflow, backend)
        manager.start()
        body, prepared, roles = build_request(workflow, "bathroom", [], [], "256x256")
        first = manager.create(dict(body), prepared, roles)
        while not first.native_id: await asyncio.sleep(0.01)
        second = manager.create(dict(body), prepared, roles)
        third = manager.create(dict(body), prepared, roles)
        manager.cancel(third.id)
        manager.cancel(first.id)
        await asyncio.sleep(0.15)
        assert len(backend.submissions) == 1 and second.status == "queued"
        backend.hold = False
        await asyncio.wait_for(second.done.wait(), 5)
        assert first.status == "cancelled" and second.status == "completed" and third.status == "cancelled"
        assert len(backend.submissions) == 2 and backend.cancel_calls == [first.native_id]
        await manager.close()
    asyncio.run(scenario())


def test_memory_retry_keeps_job_seed_canvas_roles_and_loras(workflow):
    workflow.settings.reference_megapixels = 1

    class MemoryBackend(FakeBackend):
        async def ensure_ready(self, cancel_event=None, *, low_memory=False):
            if low_memory:
                assert self.stops == 1

        async def status(self, identifier):
            if identifier == '1':
                raise NativeMemoryError('Vulkan device lost')
            return await super().status(identifier)

        async def submit(self, body):
            return await super().submit(dict(body))

    async def scenario():
        backend = MemoryBackend(workflow)
        manager = JobManager(workflow, backend)
        manager.start()
        body, prepared, roles = build_request(workflow, '场景', [png()] * 4, ['background'] * 4, '1024x1024')
        assert sum(width * height for _, (width, height) in prepared) <= 2_400_000
        job = manager.create(body, prepared, roles)
        await asyncio.wait_for(job.done.wait(), 5)
        assert job.status == 'completed' and job.attempt == 2
        first, retry = backend.submissions
        assert {k: v for k, v in first.items() if k not in {'width', 'height'}} == {k: v for k, v in retry.items() if k not in {'width', 'height'}}
        assert len(retry['ref_images']) == 4 and job.body['seed'] >= 0
        assert retry['width'] == retry['height'] == 576
        assert job.snapshot()['render_size'] == (576, 576)
        assert sorted(path.name for path in workflow.settings.outputs_dir.iterdir()) == ['_history.sqlite3', f'{job.id}.png']
        with Image.open(manager.image_path(job.id)) as image:
            assert image.size == (1024, 1024)
        await manager.close()

    asyncio.run(scenario())

def test_idempotency_and_restart_do_not_regenerate(workflow):
    async def scenario():
        backend = FakeBackend(workflow)
        manager = JobManager(workflow, backend)
        manager.start()
        body, prepared, roles = build_request(workflow, "bathroom", [], [], "256x256")
        job = manager.create(dict(body), prepared, roles, "same-key")
        assert manager.create(dict(body), prepared, roles, "same-key") is job
        with pytest.raises(ValueError): manager.create({**body, "prompt": "other"}, prepared, roles, "same-key")
        await asyncio.wait_for(job.done.wait(), 3)
        await manager.close()
        backend2 = FakeBackend(workflow)
        second = JobManager(workflow, backend2)
        restored = second.create(dict(body), prepared, roles, "same-key")
        assert restored.id == job.id and restored.status == "completed" and restored.done.is_set()
        assert second.queue.empty() and backend2.submissions == []
        assert second.result(job.id)["data"][0]["b64_json"]
        await second.close()
    asyncio.run(scenario())
