#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$root/runtime/stable-diffusion.cpp"
build_dir="$root/runtime/cuda/build"
tag=master-1127-79dc7ca
revision=79dc7ca9074d79419a46380cdb95a4524db9b74c

for tool in git cmake; do
    if ! command -v "$tool" >/dev/null; then
        printf 'Missing required command: %s\n' "$tool" >&2
        exit 1
    fi
done

if [[ -z "${CUDACXX:-}" ]] && ! command -v nvcc >/dev/null; then
    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        export CUDACXX=/usr/local/cuda/bin/nvcc
    else
        printf '%s\n' 'CUDA Toolkit compiler nvcc is required. Install the toolkit or set CUDACXX to its nvcc.' >&2
        exit 1
    fi
fi

mkdir -p "$root/runtime"
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --depth 1 --branch "$tag" https://github.com/leejet/stable-diffusion.cpp.git "$source_dir"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
    printf 'Expected stable-diffusion.cpp commit %s in %s\n' "$revision" "$source_dir" >&2
    exit 1
fi
git -C "$source_dir" submodule update --init --recursive --depth 1 -- \
    ggml thirdparty/libwebp thirdparty/libwebm thirdparty/cutlass

cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${RP_IMAGE_CUDA_ARCH:-89}" \
    -DSD_BUILD_EXAMPLES=ON -DSD_SERVER_BUILD_FRONTEND=OFF
cmake --build "$build_dir" --target sd-server --parallel "${RP_IMAGE_BUILD_JOBS:-4}"
"$build_dir/bin/sd-server" --list-devices
