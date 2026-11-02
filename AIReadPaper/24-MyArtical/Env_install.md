安装 miniforge3
```bash

curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/Miniforge3.sh

bash /tmp/Miniforge3.sh -b -p /root/miniforge3
source /root/miniforge3/etc/profile.d/conda.sh

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
