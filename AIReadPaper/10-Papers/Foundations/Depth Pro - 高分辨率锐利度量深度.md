---
type: paper-note
status: complete
domain: Foundations
paper: "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second"
year: 2024
arxiv: "2410.02073"
source: https://arxiv.org/abs/2410.02073
code: https://github.com/apple/ml-depth-pro
tags:
  - metric-depth
  - high-resolution
  - camera-calibration
  - depth-boundary
created: 2026-10-15
updated: 2026-10-15
---

# Depth Pro：高分辨率锐利度量深度

## 基本信息

| 项目 | 内容 |
|---|---|
| 作者/机构 | Aleksei Bochkovskii 等，Apple |
| 输入输出 | 单张 RGB → 1536² canonical inverse depth + 预测焦距 → metric depth |
| 速度 | V100 上约 0.3 秒，2.25MP 原生输出 |
| 核心 | 多尺度共享 ViT、两阶段真实+合成课程、导数损失、独立焦距头 |
| 重要性 | ★★★★☆：把高分辨率、绝对尺度、边界锐度和低延迟同时纳入设计目标 |

> **一句话结论：** Depth Pro 的优势不是“更高分辨率”单点，而是系统性组合：共享权重的多尺度 ViT patch encoder 提取局部高频信息，独立 image encoder 提供全局尺度；第一阶段混合真实与合成数据学泛化，第二阶段只用精确合成深度和一/二阶导数损失锐化边界；最后由独立焦距头把 canonical inverse depth 转成 metric depth。

---

## 1. 任务与表示

输入仅 RGB，不需要 EXIF。网络直接预测 canonical inverse depth $C$，再通过预测焦距恢复米制深度：

$$D_m=\frac{f_{px}}{wC}$$

inverse depth 强调近景区域，适合新视角合成和编辑中对前景边界的高要求。焦距错误会线性传播到绝对尺度，这是 Depth Pro 的主要耦合点。

![](assets/Depth%20Pro%20-%20高分辨率锐利度量深度/figures/01-qualitative-comparison.png)

> 图 1 子集：同一 AM-2k 样例的输入与 Depth Pro、Depth Anything V2、Marigold、Metric3D V2 输出。Depth Pro 对兔子毛发等遮挡边界更锐利。来源：论文 Figure 1 官方源码图组合。

---

## 2. 多尺度 ViT 架构

### 2.1 为什么不能直接跑 1536² 全局 ViT

全局 self-attention 随 token 数二次增长。Depth Pro 把固定 $1536\times1536$ 输入拆为多尺度 $384\times384$ patch：最细 25、次细 9、最粗 1，共 35 个 patch，全部通过共享 plain ViT 编码并沿 batch 并行。

### 2.2 局部与全局双编码

- patch encoder：处理多尺度局部块，恢复毛发、植物、细杆和深度边界；
- image encoder：把整图缩到 $384\times384$，提供场景布局和全局尺度；
- DPT 风格 decoder：拼接各尺度特征、上采样并融合，输出 1536² inverse depth。

最细尺度还抽取两个中间层特征，补足浅层高频结构。

![](assets/Depth%20Pro%20-%20高分辨率锐利度量深度/figures/02-architecture.png)

> 图 3：多尺度切块、共享 patch encoder、全局 image encoder 与 DPT decoder。来源：论文 Figure 3。

---

## 3. 两阶段训练课程

### 3.1 阶段一：泛化

混合真实、合成、metric 与 non-metric 数据。Metric 数据在 canonical inverse depth 上用 MAE；non-metric 数据先做 median/MAD scale-shift normalization。真实 metric 标签会去掉误差最大的 20% 像素，降低传感器边界错位和离群点影响。

论文发现，对所有数据直接加 gradient loss 会减慢甚至妨碍收敛；第一阶段只在精确合成数据上使用 scale-shift-invariant gradient supervision 最好。

### 3.2 阶段二：锐化

只用合成数据，联合：

$$L_{MAE},\quad L_{MAGE}=L_{S,1,6},\quad L_{MALE}=L_{L,1,6},\quad L_{MSGE}=L_{S,2,6}$$

其中 $S$ 为 Scharr 一阶梯度、$L$ 为 Laplace 二阶导数，在 6 个逐级下采样尺度上监督。第二阶段的目标不是重新学场景，而是用干净边界修复第一阶段被真实噪声抹平的高频结构。

这与常见“合成预训练 → 真实微调”相反：Depth Pro 是“真实+合成学分布 → 合成收尾做锐化”。

---

## 4. 独立焦距头

焦距头接收：

1. 冻结的深度网络特征；
2. 独立 ViT image encoder 的任务专用特征。

它在深度训练后单独训练，用 L2 预测水平 angular FOV。分开训练避免深度与焦距损失配比冲突，也允许使用“只有 EXIF、没有深度”的大规模摄影数据。

六个焦距测试集上，Depth Pro 的 $\delta_{25\%}$ 为 64.6%–84.2%，显著超过 UniDepth、SPEC、im2pcl；例如 DDDP 为 66.9%，第二名仅 14.6%。

---

## 5. 边界指标

传统 AbsRel/δ 指标对边界位置不敏感。Depth Pro 定义相邻像素深度比超过阈值 $t$ 的有向遮挡边界：

$$c_d(i,j)=\left[\frac{d(j)}{d(i)}>1+\frac{t}{100}\right]$$

在 $t=5$ 到 25 之间加权平均 F1，并对预测边界做 non-maximum suppression，惩罚模糊宽边。对于 matting/segmentation 数据只能计算 recall，因为 mask 不包含全部内部深度边界。

![](assets/Depth%20Pro%20-%20高分辨率锐利度量深度/figures/03-boundary-runtime.png)

> 图 2：边界召回—运行时间。Depth Pro 同时位于高召回与低延迟区域；Marigold、PatchFusion 等高细节方法慢 1–2 个数量级。来源：论文 Figure 2。

---

## 6. 主要结果

### 6.1 Zero-shot metric depth

六个数据集 $\delta_1$ 平均排名：Depth Pro 2.5，Metric3D v2 3.7，UniDepth 4.2，PatchFusion 5.2，Depth Anything V1/V2 5.7/5.8。Depth Pro 不在每个数据集第一，但跨域最稳定：Middlebury 60.5、Sintel 40.0 为最佳；SUN-RGBD 89.0 接近 UniDepth 95.8。

### 6.2 边界

| 方法 | Sintel F1 | Spring F1 | iBims F1 | AM-2k Recall | P3M Recall | DIS Recall |
|---|---:|---:|---:|---:|---:|---:|
| DA V2 | 0.228 | 0.056 | 0.111 | 0.107 | 0.131 | 0.056 |
| Marigold | 0.068 | 0.032 | 0.149 | 0.064 | 0.101 | 0.049 |
| **Depth Pro** | **0.409** | **0.079** | **0.176** | **0.173** | **0.168** | **0.077** |

六项全部第一，说明其优势确实集中在高频几何边界，而不只是整体深度排名。

### 6.3 效率

原生 1536² 输出在 V100 上约 0.3 s。固定内部尺寸使 4K 输入也不会按原始分辨率无限增大显存，但最终细节仍受 1536² 上限约束。

---

## 7. 消融与关键判断

- 高分辨率是边界准确的必要条件，但单纯提高分辨率不够；
- plain ViT 多尺度切块兼顾局部细节与可替换 backbone；
- inverse depth 对近景更准，适合 NVS；
- 第一阶段 gradient loss 只用于合成数据优于全数据；
- 两阶段课程优于单阶段，也优于“合成 → 真实”反向顺序；
- 焦距与深度解耦训练优于联合训练。

---

## 8. 局限与评价

- 半透明面与体积散射没有唯一单值深度，模型表示本身不足；
- metric scale 依赖焦距头，焦距错误会直接缩放深度；
- 固定 1536² 对超细 4K 结构仍有上限；
- 论文训练数据混合部分包含非公开/许可不同的来源，严格公平比较困难；
- 未给出显式 uncertainty，实际部署缺少逐像素可信度。

**个人判断**：Depth Pro 把“边界”从视觉观感变成正式评测目标，这是重要贡献。它也说明单目深度路线已经从平均误差竞争，转向数据质量、相机尺度、边界定位和部署延迟的多目标权衡。

---

## 9. 复现建议

优先使用官方权重；若复现训练，先做固定 1536²、多尺度 patch、合成-only sharpen 三项。用于测量前必须单独验证焦距；用于 NVS 时应保留 inverse depth 和边界，不要过早低分辨率下采样。

---

## 10. 材料来源

见 [sources.md](assets/Depth%20Pro%20-%20高分辨率锐利度量深度/sources.md)。
