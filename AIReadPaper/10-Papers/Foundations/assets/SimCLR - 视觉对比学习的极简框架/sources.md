# SimCLR 精读来源清单

- **统一获取/核验日期**：2026-10-15
- **论文**：A Simple Framework for Contrastive Learning of Visual Representations
- **arXiv ID**：2002.05709
- **核对版本**：v3，2020-07-01

## 1. 论文与元信息

### 1.1 arXiv 摘要页

- **原始 URL**：https://arxiv.org/abs/2002.05709
- **Figure/页码**：不适用
- **获取日期**：2026-10-15
- **用途**：核对标题、作者、ICML 2020、版本历史、摘要、arXiv ID、DOI，以及官方代码链接。

### 1.2 arXiv 官方 PDF

- **原始 URL**：https://arxiv.org/pdf/2002.05709
- **Figure/页码**：全文，PDF 共 20 页；重点核对 pp.1–8、13–19
- **获取日期**：2026-10-15
- **本地文件**：`sources/paper.pdf`
- **SHA-256**：`e1b10db76adeb0d21a014b9abcf42f6a6af928540c223f010fa1d8c293d9d105`
- **用途**：正文、公式、算法、主表、附录、页码与图注的最终权威来源。

### 1.3 arXiv 官方源码包

- **原始 URL**：https://arxiv.org/e-print/2002.05709
- **Figure/页码**：`main.tex` 与 `figures/*.pdf`
- **获取日期**：2026-10-15
- **用途**：核对公式、Algorithm 1、表格原始数值、增强伪代码；提取 Figure 1/5/8/9 的完整矢量原图。

### 1.4 arXiv 官方 HTML

- **原始 URL**：https://arxiv.org/html/2002.05709v3
- **Figure/页码**：全文 HTML；对应 PDF pp.1–20
- **获取日期**：2026-10-15
- **用途**：交叉核对方法、表格、附录实验与可搜索文本；所有数字最终仍与 PDF/TeX 对照。

## 2. 正文保留图片

### 2.1 总体框架

- **本地文件**：`figures/fig2_framework_page2.png`
- **原始 URL**：https://arxiv.org/pdf/2002.05709
- **Figure/页码**：Figure 2，PDF p.2
- **获取日期**：2026-10-15
- **获取方式**：从官方 PDF 以 300 DPI 裁取完整图与图注。
- **用途**：说明双视图增强、共享 encoder、projection head、$h/z$ 分工以及训练后丢弃 $g$。
- **完整性检查**：已逐图查看；左右分支、顶部 agreement 箭头、底部输入与完整图注均未切边。

### 2.2 数据增强示例

- **本地文件**：`figures/fig4_augmentations_page4.png`
- **原始 URL**：https://arxiv.org/pdf/2002.05709
- **Figure/页码**：Figure 4，PDF p.4
- **获取日期**：2026-10-15
- **获取方式**：从官方 PDF 以 300 DPI 裁取完整双排增强示例与图注。
- **用途**：区分论文研究的候选增强与最终默认增强策略。
- **完整性检查**：已逐图查看；10 个子图、子标题与完整图注均未切边。

### 2.3 增强组合消融

- **本地文件**：`figures/fig5_augmentation_pairwise_page4.png`
- **原始 URL**：https://arxiv.org/e-print/2002.05709
- **源文件**：`figures/heatmap_da_pairwise.pdf`
- **Figure/页码**：Figure 5，PDF p.4
- **获取日期**：2026-10-15
- **获取方式**：直接从官方 arXiv 源码中的完整矢量 PDF 以 300 DPI 渲染。
- **用途**：展示单项增强与两两组合的 ImageNet linear Top-1，支持 crop + color 为关键组合的结论。
- **完整性检查**：已逐图查看；行列标签、全部数值、Average 列与色条完整。

### 2.4 Projection head 消融

- **本地文件**：`figures/fig8_projection_head_page6.png`
- **原始 URL**：https://arxiv.org/e-print/2002.05709
- **源文件**：`figures/critic_dim.pdf`
- **Figure/页码**：Figure 8，PDF p.6
- **获取日期**：2026-10-15
- **获取方式**：直接从官方 arXiv 源码中的完整矢量 PDF 以 300 DPI 渲染。
- **用途**：比较 linear/non-linear/no projection 及不同输出维度。
- **完整性检查**：已逐图查看；坐标轴、所有柱、图例与 32–2048 维标签完整。

### 2.5 Batch size 与训练时长消融

- **本地文件**：`figures/fig9_batch_epochs_page7.png`
- **原始 URL**：https://arxiv.org/e-print/2002.05709
- **源文件**：`figures/bar_bsstep_top1.pdf`
- **Figure/页码**：Figure 9，PDF p.7
- **获取日期**：2026-10-15
- **获取方式**：直接从官方 arXiv 源码中的完整矢量 PDF 以 300 DPI 渲染。
- **用途**：说明大 batch 在短训练时优势明显，而长训练缩小 batch 差距。
- **完整性检查**：已逐图查看；100–1000 epochs、全部 batch size 柱、坐标轴和图例完整。

### 2.6 ImageNet 线性评估关键结果

- **本地文件**：`figures/fig1_linear_eval_sota_page1.png`
- **原始 URL**：https://arxiv.org/e-print/2002.05709
- **源文件**：`figures/sota_figure.pdf`
- **Figure/页码**：Figure 1，PDF p.1
- **获取日期**：2026-10-15
- **获取方式**：直接从官方 arXiv 源码中的完整矢量 PDF 以 300 DPI 渲染。
- **用途**：展示参数量与 ImageNet linear Top-1 的 SOTA 对比。
- **完整性检查**：已逐图查看；坐标轴、全部方法标签、SimCLR 三个宽度点与 supervised 标记完整。

## 3. 官方代码

### 3.1 官方仓库主页与 README

- **原始 URL**：https://github.com/google-research/simclr
- **Figure/页码**：README 的 Environment setup、Pretraining、Linear eval、Semi-supervised、Known issues
- **获取日期**：2026-10-15
- **用途**：核对 TPU/单 GPU 支持、ImageNet/CIFAR-10 命令、batch size、temperature、linear evaluation、少标签 fine-tuning 与已知问题。
- **注意**：仓库当前同时覆盖 SimCLRv1/SimCLRv2；不能把 master 默认 flags 直接等同于论文 SimCLRv1 设置。

### 3.2 `run.py`

- **原始 URL**：https://raw.githubusercontent.com/google-research/simclr/master/run.py
- **Figure/页码**：flags `temperature`、`proj_head_mode`、`proj_out_dim`、`num_proj_layers`、`train_batch_size`、`train_epochs`
- **获取日期**：2026-10-15
- **用途**：核对官方实现暴露的训练参数与当前 master 默认值，并识别 SimCLRv2 默认 head 深度与论文 v1 的差异。

### 3.3 `objective.py`

- **原始 URL**：https://raw.githubusercontent.com/google-research/simclr/master/objective.py
- **Figure/页码**：`add_contrastive_loss`
- **获取日期**：2026-10-15
- **用途**：核对 $\ell_2$ normalization、双视图拆分、cross-replica concat、self-mask、双向 logits 与 softmax cross entropy 实现。

### 3.4 `data_util.py`

- **原始 URL**：https://raw.githubusercontent.com/google-research/simclr/master/data_util.py
- **Figure/页码**：`color_jitter`、`random_color_jitter`、`gaussian_blur`、`random_crop_with_resize`、`preprocess_for_train`
- **获取日期**：2026-10-15
- **用途**：交叉核对 color jitter 强度、0.8/0.2 概率、blur sigma、kernel size、crop/flip 数据流。

### 3.5 `model.py`

- **原始 URL**：https://raw.githubusercontent.com/google-research/simclr/master/model.py
- **Figure/页码**：pretrain 分支调用 projection head 与 contrastive loss 的代码段
- **获取日期**：2026-10-15
- **用途**：核对 projection head 输出进入 NT-Xent，以及 temperature/global TPU context 的传递。

### 3.6 `model_util.py`

- **原始 URL**：https://raw.githubusercontent.com/google-research/simclr/master/model_util.py
- **Figure/页码**：learning-rate schedule、LARS、projection head
- **获取日期**：2026-10-15
- **用途**：核对 linear/sqrt learning-rate scaling、warmup、cosine decay、LARS 和 projection head 的具体结构。

## 4. 关联工作官方来源

### 4.1 MoCo

- **原始 URL**：https://arxiv.org/abs/1911.05722
- **Figure/页码**：官方摘要与方法概述
- **获取日期**：2026-10-15
- **用途**：核对 MoCo 的 queue、dynamic dictionary 与 moving-averaged encoder，支撑其与 SimCLR in-batch negatives 的对比。

### 4.2 BYOL

- **原始 URL**：https://arxiv.org/abs/2006.07733
- **Figure/页码**：官方摘要与方法概述
- **获取日期**：2026-10-15
- **用途**：核对 online/target networks、EMA target 以及不使用 negative pairs 的官方表述。

### 4.3 DINO

- **原始 URL**：https://arxiv.org/abs/2104.14294
- **Figure/页码**：官方摘要与方法概述
- **获取日期**：2026-10-15
- **用途**：核对 self-distillation with no labels、momentum encoder、multi-crop 与 ViT 的关系。

## 5. 数字口径备注

- **76.5%**：ResNet-50 $4\times$，ImageNet linear evaluation **Top-1**，Table 6。
- **93.2%**：同一模型的 **Top-5**，Table 6。
- **69.3%**：标准 ResNet-50，ImageNet linear evaluation **Top-1**，Table 6。
- **85.8%**：1% 标签 fine-tuning 的 **Top-5**，ResNet-50 $4\times$，Table 7；不是 Top-1。
- **63.0%**：同一 1% 标签设置的 **Top-1**，Table B.4。
- **64.5%**：默认增强下，ResNet-50、100 epochs 的消融口径；不是 1000-epoch 最终模型。
- **1.5 小时**：128 TPU v3 cores、batch 4096、ResNet-50、100 epochs，论文 Section 2.2 脚注。
- **6 小时**：32 TPU v3、batch 4096、100 epochs，当前官方 README；与论文的 128-core 口径不同。
- **weight decay $10^{-6}$**：论文 v3 默认消融设置；当前仓库 README 命令/master defaults 可能不同，复现时以目标版本与显式 flags 为准。
