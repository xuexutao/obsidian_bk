# 溯源记录：Hy-Nomas - 自由交互式世界生成

> 本文比较特殊：**尚无论文**（技术报告标注 Coming Soon，实际 PDF 404，无 arXiv 条目）。以下信息均来自官方渠道，获取日期均为 2026-10-24。

## 官方来源

| 材料 | URL | 用途 | 获取日期 |
|---|---|---|---|
| HF 模型卡（英文） | https://huggingface.co/tencent/Hy-Nomas | 架构配置表、条件通道细节、部署配置、Bench 描述的唯一权威来源；快照存 `sources/HF_README.md` | 2026-10-24 |
| GitHub README（英/中） | https://github.com/Tencent-Hunyuan/Hy-Nomas | 训练/推理/微调命令、资产清单；快照存 `sources/README.md`、`sources/README_CN.md` | 2026-10-24 |
| Hy-Nomas-Bench README | https://huggingface.co/datasets/tencent/Hy-Nomas-Bench | 四基准用例数/轮数/时长、相机动作设计；快照存 `sources/BENCH.md` | 2026-10-24 |
| 官方架构图 | https://huggingface.co/tencent-Hy-Nomas 原图 `assets/method_framework_h.png`（2910×1200） | 笔记图 1；存 `figures/01-framework-dual-stream-mmdit.png` | 2026-10-24 |
| 在线 Demo | https://hy-nomas.hy.tencent.com/ | 交互体验入口 | 2026-10-24 |

## 关键数字出处

- 54 双流 MMDiT / hidden 2048 / 16×128 头 / FFN 8192 / 3D RoPE [16,56,56] theta 256 / flow velocity + Euler：HF 模型卡 "Model Architecture" 表（可追溯到仓库 `hymm/models/dit/nomas_cpt.py` 与 `configs/experiment/inference.yaml`）。
- DCAE-V `f32t4c32`（32 通道、空间 32×、时间 4×）、PixelUMM（36 层 / 4096 / 32q+8kv 头 / 1 步解码 / 窗口 10 重叠 5 latent 帧）：HF 模型卡 "Video Latents and PixelUMM Decoder"。
- 8 步去噪（支持 4/6/8）、720×1280@24FPS、CFG 1.0、标准 8 卡 / 单卡 4090 配置（INT8+FP8+CPU offload、18 latent 帧历史预算）：HF 模型卡与仓库 README。
- 权重体积与 **~8.5B 参数推算**：fp32 DiT 34.2 GB → 每参数 4 字节，**第三方（OrcaRouter）算术，非官方口径**，笔记中已注明。
- 发布时间线（权重 2026-10-23 11:18 UTC 等）：OrcaRouter 博客（https://www.orcarouter.ai/blog/hy-nomas-interactive-world-model ），2026-10-24 获取。
- 混元世界模型 1.5（WorldPlay）的对比信息：百度百科 WorldPlay 词条，2026-10-24 获取。

## 未知信息（截至 2026-10-24）

- 技术报告未发布：训练数据、训练 recipe、蒸馏细节、消融、benchmark 成绩全部空缺。
- 官方未公布：总参数量、推理延迟、与竞品对比。
- 待技术报告发布后补全笔记 §5.1 并回填评级。
