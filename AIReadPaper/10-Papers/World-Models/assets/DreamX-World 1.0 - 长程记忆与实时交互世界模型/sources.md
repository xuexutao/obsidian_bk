# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 / 版本 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2606.16993 | arXiv:2606.16993v1，2026-06-15，21 页 | 2026-10-28 | 精读主来源 |
| `figures/01-teaser.png` | 能力总览图 | https://arxiv.org/pdf/2606.16993 | Figure 1，p.1 | 2026-10-28 | 展示跨写实、游戏、风格化域的交互生成能力 |
| `figures/02-system-overview.png` | 系统总览图 | https://arxiv.org/pdf/2606.16993 | Figure 2，p.2 | 2026-10-28 | 数据、训练、推理全栈总览 |
| `figures/03-memory-persistence.png` | 记忆训练框架图 | https://arxiv.org/pdf/2606.16993 | Figure 6，p.7 | 2026-10-28 | 几何检索、token packing 与 residual recycling |
| `figures/04-dmd-forcing.png` | 长视频蒸馏图 | https://arxiv.org/pdf/2606.16993 | Figure 7，p.8 | 2026-10-28 | DMD-forcing 与局部时间窗监督 |
| `figures/05-qualitative-results.png` | 定性结果图 | https://arxiv.org/pdf/2606.16993 | Figure 10，p.11 | 2026-10-28 | 多风格、多相机轨迹的长视频关键帧 |
| `figures/06-memory-results.png` | 记忆定量表 | https://arxiv.org/pdf/2606.16993 | Table 5，p.14 | 2026-10-28 | 重访一致性评测 |
| `sources/hf-dreamx-world-5b-config.json` | 官方权重配置 | https://huggingface.co/GD-ML/DreamX-World-5B/resolve/main/config.json | 下载时 main | 2026-10-28 | AR 5B 层数、宽度、patch 与 attention 配置 |
| `sources/hf-dreamx-world-5b-cam-config.json` | 官方权重配置 | https://huggingface.co/GD-ML/DreamX-World-5B-Cam/resolve/main/config.json | 下载时 main | 2026-10-28 | 双向 5B-Cam PRoPE 配置 |
| `sources/dreamx-ar-config.json` | 代码仓库配置 | https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/configs/dreamx-ar/config.json | commit `a1f4c6e` | 2026-10-28 | AR 模型结构与 KV cache 参数 |
| `sources/causal-camera-forcing-5b.yaml` | 推理配置 | https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/configs/dreamx-ar/causal_camera_forcing_5b.yaml | commit `a1f4c6e` | 2026-10-28 | few-step timestep、chunk 与 context-noise |
| `sources/wan-ti2v-5b.yaml` | 基座配置 | https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/configs/wan2.2/wan_ti2v_5b.yaml | commit `a1f4c6e` | 2026-10-28 | VAE 压缩率、UMT5 与 scheduler |
| `sources/inference-README.md` | 官方推理说明 | https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/inference_README.md | commit `a1f4c6e` | 2026-10-28 | 推荐运行命令与硬件需求 |
| `sources/inference-dreamx-5b.sh` | 双向模型入口 | https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/inference_dreamx_5b.sh | commit `a1f4c6e` | 2026-10-28 | 5B-Cam shell 实际默认参数 |
| `sources/inference-ar-forcing.sh` | AR 模型入口 | https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/inference_ar_forcing.sh | commit `a1f4c6e` | 2026-10-28 | 长视频 shell 实际默认参数 |

## 官方链接

- arXiv：https://arxiv.org/abs/2606.16993
- 论文 HTML：https://arxiv.org/html/2606.16993
- 官方项目页：https://amap-ml.github.io/DreamX_World/
- 在线体验：https://www.dreamx-world.com/
- GitHub：https://github.com/AMAP-ML/DreamX-World
- DreamX-World-5B：https://huggingface.co/GD-ML/DreamX-World-5B
- DreamX-World-5B-Cam：https://huggingface.co/GD-ML/DreamX-World-5B-Cam

## 代码审阅记录

- 审阅 commit：`a1f4c6e5e45600718e5236955f2e0702e53fc275`
- commit 时间：2026-07-23 17:44:29 +0800
- 许可证：GitHub 代码仓库为 Apache-2.0；Hugging Face 模型卡标注 MIT；组合使用还依赖 Wan2.2 组件，应分别核对基础模型条款。
- 已人工复核的主要文件：
  - `inference_dreamx5b.py`
  - `inference_ar_forcing.py`
  - `pipeline/pipeline_dreamxworld.py`
  - `pipeline/pipeline_causal_camera.py`
  - `models/wan_transformer3d.py`
  - `models/prope_utils.py`
  - `wan/modules/causal_camera_model_2_2_prope_infinity.py`
  - `wan/modules/camera_prope.py`
  - `utils/inference_utils.py`
  - `utils/trajectory_processor.py`
  - `utils/wan_wrapper.py`
  - `utils/scheduler.py`
- 开源范围结论：官方已开源权重与两条推理路径；训练数据、训练入口、E-PRoPE 的论文版空间 token 压缩、Memory-Conditioned Scene Persistence、Event Instruction Tuning、DMD/causal-forcing 训练、RL 对齐与 16 FPS 完整部署栈未开源。
