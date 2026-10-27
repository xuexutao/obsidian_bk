# Sources

## 论文与图表

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v2） | https://arxiv.org/pdf/2512.10942v2 | 全文，共 15 页 | 2026-10-27 | 精读主来源 |
| `figures/01-architecture-overview.png` | 架构概览 | https://arxiv.org/pdf/2512.10942v2 | Figure 1, p.1 | 2026-10-27 | 解释四模块抽象关系 |
| `figures/02-method-and-applications.png` | 完整方法与任务接口图 | https://arxiv.org/pdf/2512.10942v2 | Figure 2, p.3 | 2026-10-27 | 核心方法与多任务推理 |
| `figures/03-embedding-vs-token.png` | 受控实验与参数/延迟对比图 | https://arxiv.org/pdf/2512.10942v2 | Figure 3, p.7 | 2026-10-27 | 嵌入预测与 token 预测对比 |
| `figures/04-selective-decoding.png` | 选择性解码示意与曲线 | https://arxiv.org/pdf/2512.10942v2 | Figure 4, p.8 | 2026-10-27 | 解释 2.85× 解码削减及其边界 |
| `figures/05-ablation-table.png` | 完整消融表 | https://arxiv.org/pdf/2512.10942v2 | Table 7, p.9 | 2026-10-27 | 分析预训练、损失、Predictor 与 Y-Encoder |

- arXiv 摘要页：https://arxiv.org/abs/2512.10942v2
- DOI：https://doi.org/10.48550/arXiv.2512.10942
- OpenReview / ICLR 2026：https://openreview.net/forum?id=tjimrqc2BU
- 作者主页：https://chendelong.world/

> 未发现独立 Supplementary Material；论文的实验与实现补充均包含在 15 页 v2 PDF 中。

## 官方开源状态核查

核查日期：2026-10-27。

### 结论

**仅有第三方复现；官方代码、官方推理代码与可下载权重均未发布。**

论文、arXiv 和作者主页均未提供官方代码仓库。作者 GitHub 与 Meta 组织下也未发现 VL-JEPA 仓库。Hugging Face 的两个作者账号仓库采用人工访问审批并声明 FAIR Noncommercial Research License，但当前文件树都只有 `README.md` 和 `.gitattributes`，`usedStorage=0`，不存在模型权重、`config.json`、tokenizer、训练代码或推理代码。因此不能表述为“官方权重已开源”。

### Hugging Face 元数据快照

| 本地文件 | 原始 URL | 核查要点 |
|---|---|---|
| `sources/hf-vl-jepa-api.json` | https://huggingface.co/api/models/delong-chen/VL-JEPA | BASE 仓库 API 元数据；gated=manual，usedStorage=0 |
| `sources/hf-vl-jepa-tree.json` | https://huggingface.co/api/models/delong-chen/VL-JEPA/tree/main?recursive=true&expand=true | 仅 `.gitattributes` 与 `README.md` |
| `sources/hf-vl-jepa-sft-api.json` | https://huggingface.co/api/models/delong-chen/VL-JEPA-SFT | SFT 仓库 API 元数据；gated=manual，usedStorage=0 |
| `sources/hf-vl-jepa-sft-tree.json` | https://huggingface.co/api/models/delong-chen/VL-JEPA-SFT/tree/main?recursive=true&expand=true | 仅 `.gitattributes` 与 `README.md` |

- BASE 页面：https://huggingface.co/delong-chen/VL-JEPA
- BASE revision：`20f7f0e71f43e00e579df9a1814c5b9f519b45ea`
- SFT 页面：https://huggingface.co/delong-chen/VL-JEPA-SFT
- SFT revision：`1c0eb0bee29b05c9bf793843937f47af1c602fc5`
- 许可证说明：https://ai.meta.com/resources/models-and-libraries/cwm-license/

### 官方仓库搜索

- 作者 GitHub：https://github.com/ChenDelong1999
- 作者账号搜索：https://api.github.com/search/repositories?q=VL-JEPA+user%3AChenDelong1999
- Meta FAIR 搜索：https://api.github.com/search/repositories?q=VL-JEPA+org%3Afacebookresearch
- Meta Llama 搜索：https://api.github.com/search/repositories?q=VL-JEPA+org%3Ameta-llama

以上搜索在核查日均未发现官方 VL-JEPA 代码仓库。

## 第三方代码复现

### 本次代码级阅读对象

- 仓库：https://github.com/Soumya-Chakraborty/VL-JEPA
- 定位：第三方 `Paper-Derived Implementation`，非论文作者或 Meta 官方实现
- commit：`7400361ecbe520504f285d4c7e906e1205b68469`
- commit 时间：2026-02-20T23:19:53+05:30
- 许可证：仓库未提供 LICENSE，复用与分发边界不明确
- 阅读日期：2026-10-27

主要阅读文件：

| 文件 | 用途 |
|---|---|
| `README.md` | 安装、推荐命令、paper-style/tiny 路径与作者自述边界 |
| `vljepa/models/vljepa.py` | X-Encoder、query embedding、Predictor、Y-Encoder 与 projection |
| `vljepa/models/losses.py` | 双向 InfoNCE |
| `vljepa/data/dataset.py` | 图像/视频读取、帧采样、batch 形状 |
| `vljepa/train/build.py` | DataLoader、AdamW、LR group 与 scheduler |
| `vljepa/train/trainer.py` | AMP、反向传播、梯度裁剪、checkpoint |
| `vljepa/inference/decoder.py` | 固定 text bank 最近邻“解码” |
| `vljepa/inference/selective.py` | Ward + 时间连通约束的离线分段 |
| `scripts/train_pretrain.py` | 预训练入口 |
| `scripts/train_sft.py` | SFT 入口 |
| `scripts/run_inference.py` | caption / discriminative VQA / selective 入口 |
| `vljepa/configs/pretrain.yaml` | paper-style 默认配置 |
| `vljepa/configs/pretrain_tiny.yaml` | 离线 tiny 配置与可核验具体形状 |

### 其他第三方实现

- https://github.com/dion-jy/open-vljepa — 明确标注 unofficial、small-scale
- https://github.com/xieliaing/VL-JEPA — from-scratch 架构与 benchmark 实验
- https://github.com/JosefAlbers/vl-jepa — Apple MLX / PaliGemma 教学实现

这些第三方实现的配置、权重与实验结果都不能替代官方实现，也不能用于证明论文指标可复现。

## 数字溯源说明

- 模型结构、训练规模、实验数字均以 `sources/paper.pdf` 为准。
- 论文未披露的视觉 token 数、Predictor 隐藏宽度、Y-Decoder 架构、InfoNCE 温度与跨卡负样本实现均标为“未披露”，未用第三方代码补成论文事实。
- 第三方代码章节中的具体张量形状只描述 `Soumya-Chakraborty/VL-JEPA` 在对应配置下的行为，并明确与论文模型隔离。
- “约 129,024 H200 GPU-hours”由论文“4 周 × 24 节点 × 每节点 8 张 H200”推导，属于估算，不是论文直接报告值。
