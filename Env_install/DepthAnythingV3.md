# DepthAnythingV3

conda环境

```bash

### cuda 12.1  python 3.11 ###
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

pip install -e .
# pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70 # for gaussian head
pip install gsplat
pip install -e ".[app]" --no-deps
pip install -e ".[all]" --no-deps

```

# Wan2.2

conda 环境
```bash
### cuda 12.1  python 3.11 ###
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121


```
