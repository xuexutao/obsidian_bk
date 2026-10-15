# BYOL 来源与图表溯源

- 论文：*Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning*
- arXiv：2006.07733v3
- 会议：NeurIPS 2020 Oral
- 统一访问/核验日期：2026-10-15
- 本地 PDF：`sources/paper.pdf`
- PDF 页数：35
- PDF 大小：4,837,213 bytes
- SHA-256：`873e7b74d58e1e17806cf7a3cf6b80ead914559f36ed19313c749672eed0aa94`

## 1. 核心原始资料

1. **arXiv 摘要页**
   - URL：https://arxiv.org/abs/2006.07733
   - 用途：标题、作者、摘要、arXiv 版本与提交历史。
   - 核验日期：2026-10-15。

2. **arXiv 官方 PDF**
   - URL：https://arxiv.org/pdf/2006.07733
   - 用途：正文、公式、实验数字、附录、全部本地图表裁剪。
   - 对应本地文件：`sources/paper.pdf`。
   - 下载日期：2026-10-15。

3. **NeurIPS 2020 官方论文页**
   - URL：https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html
   - 用途：会议收录信息、作者列表、摘要、Paper/Supplemental/Review/MetaReview 官方入口。
   - 核验日期：2026-10-15。

4. **NeurIPS 2020 Virtual Oral 页面**
   - URL：https://nips.cc/virtual/2020/oral/18558
   - 用途：确认该工作为 NeurIPS 2020 Oral，并复核摘要。
   - 核验日期：2026-10-15。

5. **Google DeepMind 官方实现仓库**
   - URL：https://github.com/google-deepmind/deepmind-research/tree/master/byol
   - 论文中原始 URL：https://github.com/deepmind/deepmind-research/tree/master/byol
   - 用途：官方复现说明、训练入口、ImageNet 预期精度、Cloud TPU 配置、公开 checkpoint、BatchNorm 初始化注意事项、Imagenette 小规模配置。
   - 核验日期：2026-10-15。

## 2. 本地图表逐项溯源

所有图表均由 `sources/paper.pdf` 直接以 300 DPI 裁剪；未经过内容重绘、数值改写或 AI 生成。

### 2.1 `figures/fig01_imagenet_performance.png`

- 原始编号：**Figure 1**。
- PDF 页码：**第 1 页**。
- 原始 URL：https://arxiv.org/pdf/2006.07733
- 内容：ImageNet 线性评估中，BYOL、监督基线及同期自监督方法随参数量变化的 top-1 精度。
- 笔记位置：第 9.2 节“ImageNet 线性评估”。
- 裁剪与验图日期：2026-10-15。
- 验图结果：坐标轴、方法标签、数据点和完整 Figure 1 图注均清晰可见。

### 2.2 `figures/fig02_online_target_architecture.png`

- 原始编号：**Figure 2**。
- PDF 页码：**第 4 页**。
- 原始 URL：https://arxiv.org/pdf/2006.07733
- 内容：BYOL online/target 双分支架构，包含 encoder、projector、online predictor、stop-gradient 与 loss 流向。
- 笔记位置：第 4.1 节“整体框架”。
- 裁剪与验图日期：2026-10-15。
- 验图结果：两条分支、全部模块、箭头、符号及完整 Figure 2 图注均清晰可见。

### 2.3 `figures/fig03_batch_augmentation_robustness.png`

- 原始编号：**Figure 3**。
- PDF 页码：**第 8 页**。
- 原始 URL：https://arxiv.org/pdf/2006.07733
- 内容：BYOL 与 SimCLR 对 batch size 缩小和增强逐步删除的性能退化对比。
- 笔记位置：第 10.1—10.2 节。
- 裁剪与验图日期：2026-10-15。
- 验图结果：左右两个子图、图例、坐标轴、数据曲线和完整 Figure 3 图注均清晰可见。

### 2.4 `figures/table05_target_predictor_collapse_ablation.png`

- 原始编号：**Table 5**。
- PDF 页码：**第 9 页**。
- 原始 URL：https://arxiv.org/pdf/2006.07733
- 内容：不同 target 模式、EMA 衰减率、predictor/target/负样本项组合的 top-1 消融，是论文讨论 collapse 的关键证据。
- 笔记位置：第 7.2 节“论文直接提供的经验证据”。
- 裁剪与验图日期：2026-10-15。
- 验图结果：左右两个子表、脚注、勾选项、数值及完整 Table 5 图注均清晰可见。

## 3. 后续解释与方法关系资料

以下资料只用于笔记中“后续工作解释”和方法谱系对比，不替代 BYOL 原论文证据。

1. **SimSiam：Exploring Simple Siamese Representation Learning**
   - URL：https://arxiv.org/abs/2011.10566
   - 用途：说明后续工作在无负样本、无大 batch、无 momentum encoder 时，仍可借助 predictor 与 stop-gradient 学到非坍塌表示；用于区分“EMA 是重要稳定器”和“EMA 是唯一防坍塌条件”。
   - 核验日期：2026-10-15。

2. **DINO：Emerging Properties in Self-Supervised Vision Transformers**
   - URL：https://arxiv.org/abs/2104.14294
   - 用途：比较后续 teacher-student 自蒸馏方法中的 EMA teacher、multi-crop、centering、sharpening 与 ViT 表征特性。
   - 核验日期：2026-10-15。

3. **SimCLR：A Simple Framework for Contrastive Learning of Visual Representations**
   - URL：https://arxiv.org/abs/2002.05709
   - 用途：对比 InfoNCE、batch 内负样本、projection head 与增强依赖；BYOL 原文参考文献 [8]。
   - 核验日期：2026-10-15。

4. **MoCo：Momentum Contrast for Unsupervised Visual Representation Learning**
   - URL：https://arxiv.org/abs/1911.05722
   - 用途：比较 momentum encoder 在对比式 queue/key 一致性与 BYOL 回归目标稳定化中的不同角色；BYOL 原文参考文献 [9]。
   - 核验日期：2026-10-15。

## 4. 数字与公式核验位置

- online/target 定义、EMA 更新、单向损失与对称损失：PDF 第 3—4 页，Section 3.1，Eq. (1)—(3)。
- 条件方差与近最优 predictor 直觉：PDF 第 4—5 页，Section 3.2，Eq. (4)—(5)。
- 主训练配置：PDF 第 5 页，Section 3.3。
- ImageNet 线性与半监督结果：PDF 第 6 页，Table 1—2。
- 分类迁移与视觉任务迁移：PDF 第 7 页，Table 3—4。
- batch size 与增强鲁棒性：PDF 第 8 页，Figure 3。
- target/predictor/collapse 消融：PDF 第 9—10 页，Table 5 与正文。
- 完整算法：PDF 第 16 页，Algorithm 1。
- 增强参数：PDF 第 16—17 页，Table 6。
- 大模型线性评估：PDF 第 20 页，Table 9。
- projector/predictor、学习率、weight decay、batch size、增强消融：PDF 第 25—27 页，Table 14—18。
- stop-gradient 与稳定滞后 target 的拆分：PDF 第 28 页，Table 19。
- 输出归一化消融：PDF 第 29—30 页，Figure 7、Table 20。
- 小 batch 训练配方：PDF 第 30 页，Appendix G。
- predictor 学习率与最优 predictor 实验：PDF 第 30—31 页，Appendix H—I，Table 21—22。
- JAX 伪代码与超参数：PDF 第 33—35 页，Appendix J。

## 5. 口径说明

- “不使用负样本”均指**不使用显式的实例级负样本项**；标准实现仍包含 BatchNorm 的跨样本统计。
- “不坍塌”是论文在报告配置下的经验结果，不表述为一般理论保证。
- BYOL 对增强“更鲁棒”不等于“无需增强”；只用 crop 时仍有显著性能下降。
- 74.3% 是标准 ImageNet 线性评估协议；Appendix C 修改评估增强和 logits regularization 后的 74.8% 另行标注，不混用。
- 79.6% 对应 ResNet-200 2×，不是标准 ResNet-50。
