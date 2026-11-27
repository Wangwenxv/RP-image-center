# AutoDL 单卡测试

首轮使用单张 RTX 4090D、Ubuntu 22.04 和 CUDA 12.8。将项目放在实例数据盘（例如 `~/autodl-tmp/RP-image-center`），模型、编译产物和生成结果都保存在这里。

## 准备引擎和模型

在项目目录检查工具。

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
git --version
cmake --version
c++ --version
nvcc --version
```

需要 C/C++ 编译工具、CMake 和带 `nvcc` 的 CUDA Toolkit。如果镜像未将 `nvcc` 放进 `PATH`，可以设置 `CUDACXX` 为它的绝对路径。`nvidia-smi` 所示 CUDA 版本代表驱动支持范围。

```bash
bash scripts/install_native_linux.sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt huggingface_hub
```

编译脚本固定使用 `stable-diffusion.cpp` 的 `79dc7ca` 版本，默认编译 Ada GPU（SM 89）。编译完成会列出可用设备。模型下载可与编译并行。

```bash
.venv/bin/hf download Comfy-Org/Qwen-Image-2.1 \
  diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors \
  text_encoders/qwen3vl_8b_int8_convrot.safetensors \
  vae/qwen_image_2.1_vae_bf16.safetensors \
  --local-dir runtime/qwen-download
for model in runtime/qwen-download/{diffusion_models,text_encoders,vae}/*.safetensors; do
  ln -s "../$model" "qwen2.1-turbo/$(basename "$model")"
done
```

三份公开权重共约 17.3 GB。将已有的 `Qwen2.1 一致性无偏移_GH_V1.0.safetensors` 放到 `qwen2.1-turbo/`，服务启动时按工作流中的文件名校验。测试 SQX 时也放入 `sqx-1326-all44-transformer.safetensors`。

## 启动和验证

新建云端 `config.local.json`，首轮保存以下配置。

```json
{
  "backend": "cuda",
  "params_backend": "auto",
  "max_vram_gib": 20,
  "idle_shutdown_seconds": 300
}
```

```bash
.venv/bin/python -m uvicorn image_center.app:app --host 127.0.0.1 --port 7865
```

用 AutoDL 提供的 SSH 主机和端口建立本地转发，在自己的电脑打开 `http://127.0.0.1:7865/studio`。如果本地端口已占用，可以换一个本地端口。

```bash
ssh -N -L 127.0.0.1:7865:127.0.0.1:7865 -p <SSH端口> <SSH用户>@<SSH主机>
```

先检查 `/health` 和 `/v1/models`，再测 1024×1024 文生图、单图编辑和两图叠加 GH 与 SQX。固定提示词和种子，连续提交冷、热两次，记录耗时、任务的实际渲染尺寸和 `nvidia-smi` 的采样显存。低显存重试可能缩小实际画幅再放大，详情可在 `/studio` 看到。测试 RP-Hub 时，将 OpenAI Images 基础地址设为转发后的 `http://127.0.0.1:7865/v1`，参考图方式选 multipart。
