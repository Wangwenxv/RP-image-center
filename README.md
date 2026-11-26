# RP Image Center

本机 Qwen-Image-2.1-Turbo 生图服务，直接调用 stable-diffusion.cpp，不需要 ComfyUI。

**当前状态：本地推理与 RP 接口已跑通，人物换场景的身份保持未通过效果验收。** 实测存在脸部漂移、参考物件残留与原图构图过度继承；单主图或关闭一致性 LoRA 均未解决全部问题。可继续作为场景、物品生成及简单编辑的试验选项，不能作为已验证的角色锁脸替代。对照与记录见 [实测验收](docs/verification.md)。

## 使用

1. 双击 `start.cmd`。服务绑定 `http://127.0.0.1:7865`，API 文档在 `/docs`。
2. 刷新 RP 网页，进入“设置 → 生图服务（RP / 微信共用）”，选择 **本机 Qwen 2.1 Turbo**。
3. 无需 API Key。原有本人/地点/服装参考继续使用；无参考请求自动走八步文生图，有参考请求走九步编辑。
4. 第一次生图按需加载模型。RP 正文会显示排队、加载和采样进度。在 `/studio` 可查看结果、取消任务，刷新后也可继续查看原任务。成图直接保存为 `outputs/img_<任务ID>.png`，所有任务共用一个 `outputs/_history.sqlite3`。
5. 双击 `stop.cmd`，会停止本项目的服务和推理进程。

当前配置是显式选择本地服务。不会因云服务出错而自动切换或重复付费。

## 文件

`qwen2.1-turbo/` 中保留用户提供的原始工作流与四份权重：

- `qwen_image_2.1_turbo_int8_convrot.safetensors`
- `qwen3vl_8b_int8_convrot.safetensors`（包括视觉部分）
- `qwen_image_2.1_vae_bf16.safetensors`
- `Qwen2.1 一致性无偏移_GH_V1.0.safetensors`

不会改写或重新量化这些权重。运行时在 `runtime/model-links/` 为 LoRA 创建英文文件名的硬链接；不支持硬链接时才复制。C++ API 使用相对于 LoRA 目录的路径 ID，而不是绝对路径。

`image_center/workflow.py` 从 JSON 的启用节点读取实际参数：编辑 Euler / CFG=1 / 九步 / LoRA 0.7；文生图八步 / 不挂编辑 LoRA。PE 改写关闭。生成提示由 RP 本轮传入，服务不会套用 JSON 中“背景和姿态完全不变”的固定换装示例。

## 环境与配置

已使用独立 `.venv`，依赖见 `requirements.txt`。不需要安装 PyTorch。

`scripts/install_native.py` 固定运行时 commit `79dc7ca`，校验官方发行资产 SHA-256 后才解压。支持 CUDA 和 Vulkan；`backend=auto` 优先选择完整 CUDA 安装，否则选择 Vulkan，自动选取 NVIDIA 显卡，避免误用 Intel 核显。通过下载中继下载时仍校验官方摘要；中继不参与推理或接收图片。

首次重新安装可用 `install.cmd`；已有运行时无需重复安装。国内下载较慢时可执行：

```powershell
.\.venv\Scripts\python.exe scripts\install_native.py cuda --workers 3 --mirror
.\.venv\Scripts\python.exe scripts\install_native.py cudart --workers 3 --mirror
```

复制 `config.example.json` 为 `config.local.json` 可以调整端口、后端、参考图像素数、超时和输出像素上限。更改后重启服务。

- 上传的参考图尺寸对齐32，默认每张最多约 1MP，总计约 2.4MP，保留原始比例与输入顺序。原生 Qwen 推理还会按渲染面积重采样每张参考图，因此多图任务的实际显存开销也取决于输出画幅。
- `size=auto` 有图时跟随第一张参考图的比例，无图时默认 1024×1024，并应用输出像素上限。指定尺寸必须至少256且为32的倍数。
- 默认最大输出1,600,000像素，最多10张参考；同时实际执行一张图，最多排队8张。
- 默认三个阶段都使用 NVIDIA GPU 计算。原生 auto-fit 会在显存和系统内存之间放置 INT8 权重并分段执行。默认管理预算为 5.5 GiB，实际显存占用还包含驱动及预算外的缓冲区。指定 `params_backend` 会关闭 auto-fit 的权重放置规划。
- 引擎报告显存分配失败或 Vulkan 设备丢失时，服务会在同一个任务中重启引擎并重试一次。低显存模式进一步缩小预算、关闭分段预取、把 VAE 移到 CPU，并按参考图数量限制渲染面积。需要时把成图放大到原请求的尺寸，调试台和任务记录会显示实际渲染尺寸。重试保留种子、参考图及 LoRA，会增加耗时，放大后的图片细节也会少一些。`memory_retry` 可关闭自动重试。
- `encoder_on_cpu`、`vae_on_cpu` 可以分别切换到 CPU；`params_backend=cpu` 是更保守的显式参数卸载方案，已用于首次基线验证，但图片编码与VAE解码可能明显变慢。
- 本机使用 `params_backend=disk`，引擎从模型文件按段加载，Windows 可回收文件缓存给其他应用。这样可降低推理进程的物理内存压力，冷启动和采样可能多花时间。`params_backend=auto` 会优先考虑 GPU 和系统内存，适合空闲内存更充裕时使用。
- VAE 使用256像素分块以控制内存；prefix cache 默认使用q8_0（INT8缓存），主模型/编码器权重保持用户的 INT8 文件不变。测试中f16缓存不足会退回每步重复计算，因此对8GB机器降低缓存体积。可在配置中改回f16作数值对照；缓存量化及跨引擎计算不保证逐像素复现 Comfy。
- 当前本机 `config.local.json` 使用 `q8_0` 的 prefix cache。早期关闭缓存的画质对照见 `docs/verification.md`。
- 本机当前指定 `backend=vulkan`。同一份 1024×1024、两张参考图、双 LoRA 的请求在 CUDA 首步发生显存不足，用 Vulkan 在同一块 RTX 4060 上完成。后端切换后应固定后端再做像素或观感对照。
- Windows 启动推理引擎前至少要有 10 GiB 可用内存。运行中连续两次采样低于 4 GiB 时，停止推理进程并释放资源。阈值同时考虑物理内存与 Windows 提交余量。CPU 默认限制为 8 线程，推理进程使用较低的调度优先级。队列空闲 45 秒后退出引擎，下一次请求重新加载。本机配置为 6 线程、最多等待 3 个任务、空闲 30 秒释放。

## 苏祈雪角色 LoRA

将 `sqx-1326-all44-transformer.safetensors` 放到 `qwen2.1-turbo/` 后重启服务。在 RP 的本机 Qwen 设置中点击「读取可用模型」，然后选择 `qwen-image-2.1-turbo-local-sqx`。RP 和微信会在苏祈雪本人入镜时使用这份权重。

服务端自动在提示词中加上训练触发词 `sqxv9`，按 0.70 加载角色 LoRA。无参考图时使用工作流的八步文生图，有参考图时使用九步编辑，默认同时挂载 GH 0.70。直接调用 API 也可通过 `model` 字段选择 SQX 模型，当前角色和 `selfPortrait` 的判断由 RP 适配器完成。`GET /v1/workflow` 可查实际登记的角色权重和强度。

本机对照结果见 [苏祈雪角色 LoRA 本机验证](docs/sqx-rp-verification.md)。

## 生图调试台

启动服务后打开 <http://127.0.0.1:7865/studio>。填写提示词，按需添加参考图并给每张图选用途。可以只传服装、道具或背景。画幅、种子、角色 LoRA 和 GH 一致性 LoRA 按次调整。A/B 预览支持 100% 原像素查看和原 PNG 下载。结果保存在本机 `outputs/`，页面会列出最近完成的请求。

API 支持可选 `character_strength`（0–1，适用于 SQX 模型，0 关闭角色 LoRA）。选择 SQX 时自动补 `sqxv9`，便于固定提示词比较强度。`consistency_strength` 控制编辑分支的 GH 强度。两者省略时采用各自的工作流默认值。

## API

兼容现有 RP Images 接口：

| 路径 | 用途 |
| --- | --- |
| `GET /health` | 服务、权重检查、当前后端与队列状态 |
| `GET /v1/models` | 读取模型，不加载或生成 |
| `GET /v1/workflow` | 当前工作流摘要、采样列表、权重文件信息 |
| `POST /v1/images/generations` | JSON，等待无参考生图，返回 `data[].b64_json` |
| `POST /v1/images/edits` | multipart `image[]`，等待参考图编辑，返回 Base64 |
| `POST /v1/jobs` | 异步提交，返回任务ID；支持JSON内嵌参考图或multipart |
| `GET /v1/jobs/{id}` | 任务状态与当前阶段进度，不包含参考图Base64 |
| `POST /v1/jobs/{id}/cancel` | 请求取消；等待引擎停止后才开始下一个任务 |
| `GET /v1/jobs/{id}/result` | 完成后的 OpenAI 格式 Base64 结果 |
| `GET /v1/jobs/{id}/image` | 查看生成的PNG |

请求示例：

```json
{
  "model": "qwen-image-2.1-turbo-local",
  "prompt": "宿舍浴室，镜子与白色洗手池，暖色室内荧光灯，自然手机照片，画面没有人物。",
  "size": "768x1024",
  "n": 1,
  "seed": 42
}
```

编辑的可选字段 `reference_roles` 是与图像顺序等长的列表，支持 `identity/clothing/background/pose/style/object/reference`。multipart 时该字段为 JSON 字符串。RP 适配器自动填写，提示中的“第N张”会映射为 `<imageN>`，原始本轮事实继续保留。

本地服务不读取云 API Key，也不下载远程参考 URL。同步客户端断开会请求取消；原生引擎确认终止之前不释放队列执行权。取消超过60秒无法确认时停止本项目拥有的推理进程，防止新旧任务重叠。

可选 `Idempotency-Key` 在同一个键与同一个请求下返回已有任务，跨重启保留。服务重启会把未完成的任务标记为 `interrupted`，手动重试使用新键。旧版的 `request.json` 记录提示词、参数及参考图用途，`state.json` 记录任务状态、错误和结果地址。新版把这些信息集中在 `_history.sqlite3`。参考图像素处理后仅在队列与推理期间使用，输出目录只保留生成的 PNG 和任务数据库。清空整个输出目录前先停止服务。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe scripts\smoke.py --size 512x512
.\.venv\Scripts\python.exe scripts\smoke.py --reference 路径\本人.jpg --role identity --size 768x1024 --prompt "在宿舍浴室镜前自拍，当前衣服和机位按本轮要求。"
```

单元测试使用虚拟引擎，仅验证接口、队列、取消、输入检查及重启行为。`smoke.py` 调用实际模型，记录结果、耗时和间隔采样的显存读数；采样最高值不等于精确峰值。实际出图验证记录另见 `docs/verification.md`。
