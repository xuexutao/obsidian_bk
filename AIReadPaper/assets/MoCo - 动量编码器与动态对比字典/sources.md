# MoCo 来源与溯源

- 核对日期：2026-10-15
- 论文：*Momentum Contrast for Unsupervised Visual Representation Learning*
- 作者：Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick
- arXiv ID：1911.05722
- 会议：CVPR 2020，pp. 9729–9738
- DOI：10.1109/CVPR42600.2020.00975

## 1. 官方论文来源

1. arXiv 摘要页：<https://arxiv.org/abs/1911.05722>
2. arXiv 官方 PDF（本地 `sources/paper.pdf` 的直接来源）：<https://arxiv.org/pdf/1911.05722>
3. CVPR 2020 Open Access 页面：<https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html>
4. CVPR 2020 官方 PDF：<https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf>
5. CVPR 官方 Supplementary：<https://openaccess.thecvf.com/content_CVPR_2020/supplemental/He_Momentum_Contrast_for_CVPR_2020_supplemental.pdf>

## 2. 本地 PDF 完整性

- 文件：`sources/paper.pdf`
- 下载 URL：<https://arxiv.org/pdf/1911.05722>
- 下载日期：2026-10-15
- 页数：12
- SHA-256：`ad31ea26ed2d9ba3e2f767a2dcb0961a5b70ff505b1c3fb18d07201140e4e1fe`
- 用途：方法、公式、表格、附录、图号与 PDF 页码的主核对依据。

## 3. 官方代码来源

1. 官方仓库：<https://github.com/facebookresearch/moco>
2. 初始公开实现快照（commit `5df3287ecca83af5efc65515627009bb026dcce5`）：<https://github.com/facebookresearch/moco/tree/5df3287ecca83af5efc65515627009bb026dcce5>
3. 核心构建器：<https://github.com/facebookresearch/moco/blob/5df3287ecca83af5efc65515627009bb026dcce5/moco/builder.py>
4. MoCo v1 训练入口：<https://github.com/facebookresearch/moco/blob/5df3287ecca83af5efc65515627009bb026dcce5/main_moco.py>
5. README：<https://github.com/facebookresearch/moco/blob/5df3287ecca83af5efc65515627009bb026dcce5/README.md>

从官方代码核对的关键实现：

- `dim=128`、`K=65536`、`m=0.999`、`T=0.07`。
- `encoder_k` 从 `encoder_q` 初始化，`requires_grad=False`。
- key encoder 更新为 `param_k = m * param_k + (1-m) * param_q`。
- queue 是形状 `C×K` 的归一化 buffer，当前全局 batch 的 keys 入队，旧 keys 出队。
- 正样本 logit 是逐样本 `q·k`，负样本 logits 是 `q` 与 queue 的矩阵乘积。
- 正样本固定放在第 0 列，因此交叉熵标签全为 0。
- Shuffle BN 在 DDP 下先全局 gather、随机打乱 key 图像、分发到各 GPU，编码后再恢复顺序。
- 官方公开实现只支持多 GPU DistributedDataParallel；README 报告 8×V100、200 epochs、约 53 小时。

## 4. 原图来源与验图记录

下列图片不是网页截图，也不是 PDF 页面截图；均来自 arXiv 官方 TeX 源包 `https://arxiv.org/e-print/1911.05722` 内的原始矢量 PDF，并以 4×矩阵高分辨率渲染为 PNG。2026-10-15 已逐张使用图像读取工具检查：边界完整、文字可读、曲线与图例未截断。

| 本地文件 | 官方源文件 | 论文标号 | PDF 页 | 内容 | 验图 |
|---|---|---:|---:|---|---|
| `figures/fig1_queue_mechanism.png` | `fig/teaser.pdf` | Figure 1 | 1 | query encoder、momentum encoder、queue 与 contrastive loss 总体机制 | 通过 |
| `figures/fig2_contrastive_mechanisms.png` | `fig/mechanisms.pdf` | Figure 2 | 3 | end-to-end、memory bank、MoCo 三类字典机制 | 通过 |
| `figures/fig3_dictionary_ablation.png` | `fig/results_mechanisms.pdf` | Figure 3 | 5 | 字典大小 K 与三种机制的 ImageNet 线性准确率 | 通过 |
| `figures/fig4_imagenet_linear.png` | `fig/results_imgnet.pdf` | Table 1 内嵌结果图 | 6 | ImageNet 线性评估的准确率—参数量比较 | 通过 |
| `figures/fig5_shuffle_bn_ablation.png` | `fig/shufflebn.pdf` | Figure A.1 | 10 | 有/无 Shuffle BN 的训练查表准确率与 kNN 验证曲线 | 通过 |

原图对应的官方论文链接统一为：<https://arxiv.org/pdf/1911.05722>。

## 5. 数值核对索引

- InfoNCE 与动量公式：论文 §3.1–§3.2，PDF pp. 3–4。
- Shuffle BN：论文 §3.3，PDF p. 4；附录 Figure A.1，PDF p. 10。
- 训练配置：论文 §4，PDF p. 5。
- 字典机制和 K 消融：Figure 3，PDF p. 5。
- momentum 消融：论文 §4.1，PDF p. 5。
- ImageNet 线性评估：Table 1，PDF p. 6。
- PASCAL VOC：Tables 2–4，PDF pp. 6–7。
- COCO 检测/实例分割：Table 5，PDF p. 8。
- keypoint、DensePose、LVIS、Cityscapes、VOC semantic segmentation：Table 6，PDF p. 9。
- COCO 6× schedule 与 Shuffle BN 附录：Table A.1 / Figure A.1，PDF p. 10。

## 6. 关系工作核对入口

用于笔记中的横向关系讨论，不用于替代 MoCo 原文数值：

- SimCLR：<https://arxiv.org/abs/2002.05709>
- BYOL：<https://arxiv.org/abs/2006.07733>
- DINO：<https://arxiv.org/abs/2104.14294>
- MoCo v2：<https://arxiv.org/abs/2003.04297>

## 7. 口径说明

- “MoCo”默认指本文 MoCo v1；MoCo v2 的 67.5/67.7 或 71.1 仅在演化关系与官方仓库复现说明中出现，不与本文 v1 的 60.6 混为一谈。
- ImageNet 线性评估数值以论文 Table 1 为主；官方仓库 5 次复现实验给出 `60.8±0.2`，属于代码复现统计。
- 下游比较严格区分 IN-1M 与 IG-1B 预训练、1×/2×/6× schedule、C4/FPN/dilated-C5 backbone。
- 论文所谓“7 个任务超过监督预训练”指任务层面的归纳；并不意味着每个数据集、骨干、指标与训练 schedule 都全面领先。
