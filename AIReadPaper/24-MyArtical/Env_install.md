安装 miniforge3
```bash

curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/Miniforge3.sh

bash /tmp/Miniforge3.sh -b -p ~/.cache/miniforge3


source ~/.cache/miniforge3/etc/profile.d/conda.sh

```

创建qwen2.5vl本地服务
```bash
# conda创建环境
conda create -n qwen25-transformers python=3.11 -y
conda activate qwen25-transformers

# 升级安装工具
python -m pip install -U pip setuptools wheel
python -m pip install \
	--only-binary=:all: \
	--index-url https://pypi.org/simple \
	--extra-index-url https://download.pytorch.org/whl/cu124 \
	torch==2.6.0+cu124 \
	torchvision==0.21.0+cu124 \
	transformers==4.51.3 \
	tokenizers==0.21.1 \
	huggingface-hub==0.30.2 \
	accelerate==1.6.0 \
	fastapi==0.115.12 \
	uvicorn==0.34.2 \
	numpy==1.26.4 \
	Pillow==12.2.0 \
	safetensors==0.5.3

# 测试torch
python - <<'PY'
import torch
import transformers
import fastapi

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
PY

# 下载模型。配置本地位置
mkdir -p /apdcephfs/private_xt/dev_workspace/model_cache/huggingface
export HF_HOME=/apdcephfs/private_xt/dev_workspace/model_cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
hf auth login

cd /apdcephfs/private_xt/dev_workspace/demo

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/apdcephfs/private_xt/dev_workspace/model_cache/huggingface \
HUGGINGFACE_HUB_CACHE=/apdcephfs/private_xt/dev_workspace/model_cache/huggingface/hub \
MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct \
python -m uvicorn qwen25_openai_server:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1


```

install flash attn
```bash
mkdir -p /dockerdata/root/flashattn-install/tmp
mkdir -p /dockerdata/root/flashattn-install/cache

TMPDIR=/dockerdata/root/flashattn-install/tmp \
PIP_CACHE_DIR=/dockerdata/root/flashattn-install/cache \
MAX_JOBS=8 \
python -m pip install flash-attn==2.8.3.post2 --no-build-isolation


python -c "import flash_attn; from flash_attn import flash_attn_func; print(flash_attn.__version__)"




# ==========================================
mkdir -p /dockerdata/root/flashattn-build/tmp

TMPDIR=/dockerdata/root/flashattn-build/tmp \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
TORCH_CUDA_ARCH_LIST="8.0" \
MAX_JOBS=8 \
python -m pip install \
  flash-attn==2.8.3.post2 \
  --no-build-isolation \
  --no-cache-dir
  
- 8.0 = A100 / A800
- 8.6 = RTX3090、A10
- 9.0 = H100/H800
```

单卡的demo
```bash
cd /apdcephfs/private_xt/dev_workspace/demo/hyworld2/worldgen
source ~/.cache/miniforge3/etc/profile.d/conda.sh


# ================================= stage 2 ====================================
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
traj_render.py \
	--target_path /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001 \
	--llm_addr 127.0.0.1 \
	--llm_port 8000 \
	--llm_name Qwen/Qwen2.5-VL-7B-Instruct \
> /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001/log_2.log 2>&1



# ================================= stage 3 ====================================
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
video_gen.py \
	--target_path /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001 \
	--max_reference 4 \
	--downsampled_pts 1000000 \
	--skip_exist \
> /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001/log_3.log 2>&1


cd /apdcephfs/private_xt/dev_workspace/demo/hyworld2/worldgen
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nnodes=1 --nproc-per-node=1 \
video_gen.py \
	--target_path /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001 \
	--max_reference 4 \
	--downsampled_pts 1000000 \
	--skip_exist \
> /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001/log_3_3.log 2>&1



# ================================= stage 4 ====================================
cd /apdcephfs/private_xt/dev_workspace/demo/hyworld2/worldgen
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
torchrun --nproc_per_node 1 gen_gs_data.py \
	--root_path /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001 \
	--save_normal --split_sky \
> /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001/log_4.log 2>&1



# ================================= stage 5 ====================================
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -m world_gs_trainer default \
	--data_dir /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001/gs_data \
	--result_dir /apdcephfs/private_xt/dev_workspace/demo/examples/output/worldgen/case001 \
	--max_steps 8000 --save_steps 1500 --eval_steps 1500 --ply_steps 1500 \
	--disable-viewer \
	--save_ply --convert_to_spz --disable_video \
	--use_scale_regularization --antialiased \
	--depth_loss --normal_loss --sky_depth_from_pcd \
	--use_mask_gaussian --mask_export_stochastic \
	--no-mask-export-anchor-protection --use_anchor_protection --export_mesh \
	--strategy.refine-start-iter 150 --strategy.refine-stop-iter 750 \
	--strategy.refine-every 100 --strategy.refine-scale2d-stop-iter 750 \
	--strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1 \
> /apdcephfs/private_xt/dev_workspace/demo/examples/worldgen/case001/log_5.log 2>&1

```

conda 环境打包
```bash

# base env
conda install -n base -c conda-forge conda-pack zstandard -y



# -n：conda env list 下的名字
# -p：环境绝对路径
conda run -n base conda-pack \
	-n qwen25-transformers \
	-o ./demo_env/qwen25-transformers-conda.tar.zst \
	--format tar.zst \
	--compress-level 1 \
	--n-threads -1 \
	--force

# 如果有-e 安装的第三方库，例如 gsplat，可以加上 `--ignore-editable-packages` 忽略可编辑包，但是重启之后要重新 pip install -e . 安装

# 或者

CONDA_BASE="$(conda info --base)"
"$CONDA_BASE/bin/conda-pack" \
	-n qwen25-transformers \
	-o ./demo_env/qwen25-transformers-conda.tar.zst \
	--format tar.zst \
	--compress-level 1 \
	--n-threads -1 \
	--force

# --------------------- 重启使用 -----------------------------

ARCHIVE="/dockerdata/root/conda-packages/demo-conda.tar.zst"
DEST="/root/.cache/env/demo-conda-restored"

mkdir -p "$DEST"
tar --zstd -xf "$ARCHIVE" -C "$DEST"

source "$DEST/bin/activate"
conda-unpack


```

scene_create
```bash
conda create -n scene_create python=3.11 -y
conda activate scene_create

# nvcc 为 11.8
# install torch
pip install loguru
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install plyfile pyyaml scipy trimesh

python main.py --help
```
