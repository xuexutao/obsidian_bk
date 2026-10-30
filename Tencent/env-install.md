```python
# uv 安装环境
uv venv --python 3.11 ~/.cache/env/demo

# nvcc 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# 安装 torch
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

uv pip install -r requirements.txt "onnx==1.19.0"


cd hyworld2/worldgen/third_party/gsplat_maskgaussian
uv pip install -e . --no-build-isolation
uv pip install flash-attn --no-build-isolation

uv pip install --no-build-isolation -r requirements_git.txt "pandas==2.2.3"
git submodule update --init --recursive
cd hyworld2/worldgen/third_party/navmesh
uv pip install . --no-build-isolation
cd ../../../../
```
