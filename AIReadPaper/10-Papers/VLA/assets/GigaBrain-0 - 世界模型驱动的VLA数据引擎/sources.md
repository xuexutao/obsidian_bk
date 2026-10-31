# GigaBrain-0 来源清单

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取/更新日期 | 用途 |
|---|---|---|---|---|---|
| `real2real.png` | Real2Real 数据迁移图 | arXiv e-print 2510.19430，`images/real2real.pdf` | Figure，`sections/GigaBrain_Data.tex` | 2026-10-16 | 生成数据路线说明；由官方矢量 PDF 以 2400 px 渲染，替换 400×227 低分辨率图 |
| `viewtransfer.png` | 视角迁移图 | arXiv e-print 2510.19430，`images/viewtransfer.pdf` | Figure，`sections/GigaBrain_Data.tex` | 2026-10-16 | View Transfer 路线说明；由官方矢量 PDF 以 2400 px 渲染 |
| `sim2real.png` | Sim2Real 图 | arXiv e-print 2510.19430，`images/sim2real.pdf` | Figure，`sections/GigaBrain_Data.tex` | 2026-10-16 | 仿真到真实迁移说明；由官方矢量 PDF 以 2400 px 渲染 |
| `human_transfer.png` | 人类视频到机器人迁移图 | arXiv e-print 2510.19430，`images/human_transfer.pdf` | Figure，`sections/GigaBrain_Data.tex` | 2026-10-16 | Human Video Transfer 说明；由官方矢量 PDF 以 2400 px 渲染 |
| `videogen.png` | 视频生成与 IDM 图 | arXiv e-print 2510.19430，`images/videogen.pdf` | Figure，`sections/GigaBrain_Data.tex` | 2026-10-16 | Video Generation + IDM 说明；由官方矢量 PDF 以 2400 px 渲染 |
| `framework.png` | GigaBrain-0 总体架构图 | arXiv 2510.19430 官方 PDF / e-print | 总体 framework figure | 2026-10-31 | 说明 RGBD/指令/状态、Embodied CoT 与动作专家的数据流 |
| `multiview.png` | 多视图生成示意 | arXiv 2510.19430 官方 PDF / e-print | Multiview generation figure | 2026-10-31 | 说明 GigaWorld 生成数据的跨视角增强路线 |
| `main_exp.png` | 主要实验结果图表 | arXiv 2510.19430 官方 PDF / e-print | Main experiments | 2026-10-31 | 汇总主基准结果 |
| `generalization_exp.png` | 泛化实验结果图表 | arXiv 2510.19430 官方 PDF / e-print | Generalization experiments | 2026-10-31 | 汇总外观、摆放与视角泛化结果 |

- 论文页：<https://arxiv.org/abs/2510.19430>
- 官方 PDF：<https://arxiv.org/pdf/2510.19430>
- 官方源码：<https://arxiv.org/e-print/2510.19430>
- 正文引用的九张图片均已登记；其中五张数据迁移图由官方矢量 PDF 渲染，另外四张 framework/实验图来自官方论文 PDF/e-print。

## 官方代码与固定 revision（代码级实现核查）

| 材料 | 固定 revision | 核查文件 | 用途 |
|---|---|---|---|
| GigaBrain-0 主仓库 | `open-gigaai/giga-brain-0@19328ff7d16e5047b7401a00aa6677d08dc160c7` | `giga_brain_0_trainer.py`、`giga_brain_0_loss.py`、`giga_brain_0_transforms.py`、`configs/giga_brain_0_from_scratch.py` | 固定到 2025-11-28 初代开源状态，避免用后来的 GigaBrain-0.7 代码替代 |
| GigaModels 官方模型库 | `open-gigaai/giga-models@b5e69a53e4b6efa965b615cc648d229a9f4c7413` | `modeling_giga_brain_0.py`、`paligemma2_with_expert.py`、`giga_brain_0_utils.py`、`pipeline_giga_brain_0.py` | 核查真正的 policy forward、SigLIP/PaliGemma2/action expert、mask、flow denoise 与推理后处理 |
| GigaBrain-0-3.5B-Base | HF revision `6b169fe5d01353a91974a0cf418de30bb04c9394` | `config.json` | 核对公开 checkpoint 默认：RGB 3 通道、50×32 action、10-step flow、KI 开启、轨迹头关闭 |
| 当前主仓库（仅状态对照） | `open-gigaai/giga-brain-0@5f450a9d3368876ddb6e4b5bd7134afba1801fb9` | `README.md` | 确认当前 HEAD 已转为 GigaBrain-0.7，不用于初代 shape 推导 |

### 开源边界结论

- 初代 policy 架构、训练/微调入口和权重已官方开源，但主仓库依赖官方 `giga-models`、`giga-train`、`giga-datasets`，不是单仓自包含。
- 公开 base checkpoint 明确排除了 depth image 与中间 2D trajectory；这两项是代码可选能力/论文完整模型描述，不是公开权重默认路径。
- GigaWorld 的生成 backbone、筛选和多视角一致性训练未在上述初代 policy 仓库中完整提供；不能用第三方 world-model 实现冒充官方。
