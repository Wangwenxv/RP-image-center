from __future__ import annotations
import base64
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen-image-2.1-turbo-local"
SQX_MODEL_ID = MODEL_ID + "-sqx"
SQX_LORA_FILE = "sqx-1326-all44-transformer.safetensors"
SQX_TRIGGER = "sqxv9"
SQX_STRENGTH = 0.7
PURPOSES = {"identity", "clothing", "background", "pose", "style", "object", "reference"}

@dataclass
class Settings:
    root: Path = ROOT
    backend: str = "auto"
    native_port: int = 7866
    port: int = 7865
    reference_megapixels: float = 1.0
    max_output_pixels: int = 1_600_000
    max_references: int = 10
    queue_size: int = 8
    timeout_seconds: int = 1800
    native_start_timeout: int = 300
    prefix_cache_type: str = "q8_0"
    prefix_cache_enabled: bool = True
    encoder_on_cpu: bool = False
    vae_on_cpu: bool = False
    params_backend: str = "auto"
    max_vram_gib: float = 5.5
    reference_total_megapixels: float = 2.4
    memory_retry: bool = True
    start_ram_gib: float = 10.0
    reserve_ram_gib: float = 4.0
    cpu_threads: int = 8
    idle_shutdown_seconds: int = 45

    @classmethod
    def load(cls):
        filename = ROOT / "config.local.json"
        data = json.loads(filename.read_text(encoding="utf-8")) if filename.exists() else {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if "root" in data: unknown.add("root")
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        result = cls(**data)
        if result.backend not in {"auto", "cuda", "vulkan"}:
            raise ValueError("backend must be auto, cuda or vulkan")
        if not 0 < result.reference_megapixels <= 4:
            raise ValueError("reference_megapixels must be within (0,4]")
        if not 0 < result.reference_total_megapixels <= 40:
            raise ValueError("reference_total_megapixels must be within (0,40]")
        if result.params_backend not in {"auto", "cpu", "disk"}:
            raise ValueError("params_backend must be auto, cpu or disk")
        if result.prefix_cache_type not in {"auto", "f16", "f32", "q8_0", "q4_K"}:
            raise ValueError("Unsupported prefix_cache_type")
        if not 1 <= result.max_vram_gib <= 128:
            raise ValueError("max_vram_gib must be between 1 and 128")
        if not 0 <= result.reserve_ram_gib <= result.start_ram_gib <= 128:
            raise ValueError("RAM thresholds must satisfy 0 <= reserve_ram_gib <= start_ram_gib <= 128")
        if isinstance(result.cpu_threads, bool) or not isinstance(result.cpu_threads, int) or not 1 <= result.cpu_threads <= 64:
            raise ValueError("cpu_threads must be an integer between 1 and 64")
        if not 1 <= result.idle_shutdown_seconds <= 3600:
            raise ValueError("idle_shutdown_seconds must be between 1 and 3600")
        return result

    @property
    def models_dir(self): return self.root / "qwen2.1-turbo"

    @property
    def native_url(self): return f"http://127.0.0.1:{self.native_port}"

    @property
    def outputs_dir(self): return self.root / "outputs"

def inspect_safetensors(filename: Path):
    with filename.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        if header_size > 16 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header: {filename.name}")
        header = json.loads(source.read(header_size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    expected = 8 + header_size + max(v["data_offsets"][1] for v in tensors.values())
    if filename.stat().st_size != expected:
        raise ValueError(f"Truncated model: {filename.name}")
    return {"name": filename.name, "bytes": expected, "tensors": len(tensors), "metadata": header.get("__metadata__", {})}, tensors

class Workflow:
    def __init__(self, settings: Settings):
        self.settings = settings
        candidates = sorted(settings.models_dir.glob("*.json"))
        if len(candidates) != 1:
            raise ValueError("qwen2.1-turbo must contain exactly one workflow JSON")
        self.path = candidates[0]
        raw = self.path.read_bytes()
        graph = json.loads(raw)
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        def values(node_id, expected_type):
            node = self.nodes[node_id]
            if node["type"] != expected_type:
                raise ValueError(f"Workflow node {node_id} is not {expected_type}")
            return node.get("widgets_values") or []
        self.diffusion = settings.models_dir / Path(values(460, "UNETLoader")[0].replace("\\", "/")).name
        self.encoder = settings.models_dir / Path(values(452, "CLIPLoader")[0].replace("\\", "/")).name
        self.vae = settings.models_dir / Path(values(453, "VAELoader")[0].replace("\\", "/")).name
        enabled = [item for item in values(459, "Power Lora Loader (rgthree)") if isinstance(item, dict) and item.get("on") and item.get("lora")]
        if len(enabled) != 1:
            raise ValueError("Expected one enabled consistency LoRA in editing workflow")
        self.lora = settings.models_dir / Path(enabled[0]["lora"].replace("\\", "/")).name
        self.lora_strength = float(enabled[0]["strength"])
        self.edit_sigmas = [float(x.strip()) for x in values(467, "ManualSigmas")[0].split(",")]
        self.t2i_sigmas = [float(x.strip()) for x in values(489, "ManualSigmas")[0].split(",")]
        self.cfg = float(values(450, "CFGGuider")[0])
        self.sampler = values(443, "KSamplerSelect")[0]
        if values(437, "ComfySwitchNode")[0]:
            raise ValueError("Prompt enhancer is enabled; this implementation intentionally requires its workflow switch off")
        for sigmas in [self.edit_sigmas, self.t2i_sigmas]:
            if sigmas[-1] != 0 or any(a <= b for a, b in zip(sigmas, sigmas[1:])):
                raise ValueError("Invalid workflow sigma schedule")
        self.model_info = []
        for filename in [self.diffusion, self.encoder, self.vae, self.lora]:
            info, _ = inspect_safetensors(filename)
            self.model_info.append(info)
        sqx = settings.models_dir / SQX_LORA_FILE
        self.sqx_lora = sqx if sqx.is_file() else None
        self.sqx_info = inspect_safetensors(sqx)[0] if self.sqx_lora else None

    def model_ids(self):
        return [MODEL_ID, SQX_MODEL_ID] if self.sqx_lora else [MODEL_ID]

    def describe(self):
        return {"model": MODEL_ID, "workflow": self.path.name, "workflow_sha256": self.sha256,
                "models": self.model_info, "edit": {"steps": len(self.edit_sigmas)-1, "sigmas": self.edit_sigmas,
                "lora": self.lora.name, "lora_strength": self.lora_strength},
                "t2i": {"steps": len(self.t2i_sigmas)-1, "sigmas": self.t2i_sigmas, "lora": None},
                "character_models": {SQX_MODEL_ID: {"lora": self.sqx_info, "trigger": SQX_TRIGGER,
                                                     "strength": SQX_STRENGTH}} if self.sqx_lora else {},
                "cfg": self.cfg, "sampler": self.sampler, "prompt_enhancer": False,
                "reference_megapixels": self.settings.reference_megapixels,
                "reference_total_megapixels": self.settings.reference_total_megapixels, "alignment": 32,
                "max_references": self.settings.max_references, "max_output_pixels": self.settings.max_output_pixels,
                "runtime": {"backend": self.settings.backend, "params_backend": self.settings.params_backend,
                            "encoder_on_cpu": self.settings.encoder_on_cpu, "vae_on_cpu": self.settings.vae_on_cpu,
                            "prefix_cache_type": self.settings.prefix_cache_type, "prefix_cache_enabled": self.settings.prefix_cache_enabled,
                            "max_vram_gib": self.settings.max_vram_gib, "memory_retry": self.settings.memory_retry,
                            "start_ram_gib": self.settings.start_ram_gib, "reserve_ram_gib": self.settings.reserve_ram_gib,
                            "cpu_threads": self.settings.cpu_threads}}

def scaled_size(width, height, pixels, *, minimum=32, upscale=True):
    scale = math.sqrt(pixels / (width * height))
    if not upscale:
        scale = min(1, scale)
    width = max(minimum, round(width * scale / 32) * 32)
    height = max(minimum, round(height * scale / 32) * 32)
    if width * height > pixels and max(width, height) > minimum:
        if width >= height:
            width = max(minimum, int(pixels / height / 32) * 32)
        else:
            height = max(minimum, int(pixels / width / 32) * 32)
    return width, height

def prepare_image(data: bytes, megapixels: float):
    if len(data) > 6 * 1024 * 1024:
        raise ValueError("Single reference exceeds 6 MiB")
    with Image.open(io.BytesIO(data)) as source:
        if source.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Reference must be PNG, JPEG or WebP")
        if source.width * source.height > 40_000_000:
            raise ValueError("Reference exceeds 40 megapixels")
        image = ImageOps.exif_transpose(source).convert("RGBA" if "A" in source.getbands() or "transparency" in source.info else "RGB")
        width, height = scaled_size(*image.size, megapixels * 1_000_000)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG")
    return output.getvalue(), (width, height)

def image_data_url(data):
    return "data:image/png;base64," + base64.b64encode(data).decode()

def decode_data_url(value: str):
    if not isinstance(value, str) or not re.fullmatch(r"data:image/(png|jpeg|webp);base64,[A-Za-z0-9+/]*={0,2}", value):
        raise ValueError("References must be inline image data URLs; remote image downloads are not performed")
    try:
        return base64.b64decode(value.split(",", 1)[1], validate=True)
    except Exception as error:
        raise ValueError("Invalid reference Base64") from error

def output_size(size: str, first_size, max_pixels: int):
    if size == "auto":
        width, height = scaled_size(*(first_size or (1024, 1024)), max_pixels, minimum=256, upscale=False)
    else:
        if not isinstance(size, str) or not re.fullmatch(r"[1-9]\d{1,4}x[1-9]\d{1,4}", size):
            raise ValueError("size must be auto or WIDTHxHEIGHT")
        width, height = map(int, size.split("x"))
    if min(width, height) < 256 or width % 32 or height % 32:
        raise ValueError("Output dimensions must be at least 256 and divisible by 32")
    if width * height > max_pixels:
        raise ValueError(f"Output exceeds configured {max_pixels} pixel limit")
    return width, height

def compile_prompt(prompt: str, roles: list[str]):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 6000:
        raise ValueError("prompt must contain 1–6000 characters")
    if "<sd_cpp_extra_args>" in prompt or "<lora:" in prompt:
        raise ValueError("Prompt cannot contain backend control tags")
    if not roles:
        return prompt.strip()
    for role in roles:
        if role not in PURPOSES:
            raise ValueError(f"Unknown reference role: {role}")
    text = re.sub(r"第([1-9]\d*)张", lambda match: f"<image{match[1]}>", prompt.strip())
    names = {"identity": "只参考脸部轮廓和五官，不参考原图的耳饰、妆发、衣着或背景",
             "clothing": "只参考服装款式、颜色、图案与材质", "background": "只参考环境布局",
             "pose": "只参考动作与姿态", "style": "只参考视觉风格", "object": "只参考指定物品",
             "reference": "按本轮提示指定用途参考"}
    instructions = "；".join(f"<image{i+1}>：{names[role]}" for i, role in enumerate(roles))
    return text + "\n" + instructions + "。"

def parse_strength(value, default, field):
    if value is None:
        return default
    message = f"{field} must be a number within 0–1"
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        strength = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError(message)
    return strength


def build_request(workflow: Workflow, prompt: str, images: list[bytes], roles: list[str], size="auto", seed=-1,
                  consistency_strength=None, model=MODEL_ID, character_strength=None):
    if len(images) > workflow.settings.max_references:
        raise ValueError(f"At most {workflow.settings.max_references} reference images")
    if roles and len(roles) != len(images):
        raise ValueError("Reference role count must match image count")
    if sum(map(len, images)) > 24 * 1024 * 1024:
        raise ValueError("References exceed 24 MiB total")
    if isinstance(seed, bool) or not isinstance(seed, int) or not -1 <= seed <= 2**53-1:
        raise ValueError("seed must be -1 or an integer within JavaScript's exact range")
    roles = roles or ["reference"] * len(images)
    strength = parse_strength(consistency_strength, workflow.lora_strength, "consistency_strength")
    if character_strength is not None and model != SQX_MODEL_ID:
        raise ValueError("character_strength requires a character model")
    character_strength = parse_strength(character_strength, SQX_STRENGTH, "character_strength")
    target = workflow.settings.reference_megapixels
    if images:
        target = min(target, workflow.settings.reference_total_megapixels / len(images))
    try:
        prepared = [prepare_image(image, target) for image in images]
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Invalid or oversized reference pixels") from error
    first_size = prepared[0][1] if prepared else None
    if size == "auto" and first_size and target < workflow.settings.reference_megapixels:
        first_size = scaled_size(*first_size, workflow.settings.reference_megapixels * 1_000_000)
    width, height = output_size(size, first_size, workflow.settings.max_output_pixels)
    sigmas = workflow.edit_sigmas if images else workflow.t2i_sigmas
    body = {"prompt": compile_prompt(prompt, roles), "negative_prompt": "", "width": width, "height": height,
            "seed": seed, "batch_count": 1, "ref_images": [image_data_url(data) for data, _ in prepared],
            "sample_params": {"sample_method": workflow.sampler, "sample_steps": len(sigmas)-1,
                              "custom_sigmas": sigmas, "guidance": {"txt_cfg": workflow.cfg}},
            "lora": [{"path": str(workflow.lora), "multiplier": strength}] if images and strength > 0 else [],
            "output_format": "png", "preview": "none", "embed_image_metadata": True,
            "vae_tiling_params": {"enabled": True, "tile_size_w": 256, "tile_size_h": 256, "target_overlap": 0.25}}
    if model == SQX_MODEL_ID:
        if not workflow.sqx_lora:
            raise ValueError("SQX character LoRA is unavailable")
        if not re.search(rf"\b{SQX_TRIGGER}\b", body["prompt"], re.ASCII):
            body["prompt"] = f"{SQX_TRIGGER}, " + body["prompt"]
        if character_strength > 0:
            body["lora"].append({"path": str(workflow.sqx_lora), "multiplier": character_strength})
        body["model"] = model
    return body, prepared, roles
