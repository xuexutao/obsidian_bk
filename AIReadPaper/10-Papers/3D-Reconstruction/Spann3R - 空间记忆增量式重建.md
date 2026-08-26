**一句话结论：** Spann3R（3D Reconstruction with Spatial Memory）用 **spatial memory（稠密 working memory + 稀疏 long-term memory，受 XMem 启发）** 替代 DUSt3R 的「两两回归 + 全局对齐」，使模型可以**直接在全局坐标系下增量回归每帧 pointmap**，首次做到实时（>50 FPS）的在线三维重建。它保留 DUSt3R 的 Siamese ViT encoder + Interleave decoder 骨架，只改造 reference/target decoder 的角色分工，再加上一套受 XMem 启发的内存读取 / 写入 / 稀疏化机制，是「记忆化前馈重建」的代表性工作。

|         |                                                                                                                                                                                                                                                                                                                                                          |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 论文名称    | **3D Reconstruction with Spatial Memory**                                                                                                                                                                                                                                                                                                            |
| 归档领域    | **三维重建**                                                                                                                                                                                                                                                                                                                                              |
| 重要性评估   | **★★★★☆（4/5）**。理由：首次把「记忆 / memory mechanism」引入前馈几何重建，给出了一条不依赖全局优化、不依赖 BA 即可在线、增量、可扩展的重建路线。在线 >50 FPS 的重建速度使「实时前馈重建」真正具备工程可行性。                                                                                                                                                                       |
| 会议 / 年份 | arXiv 2024（已发表但作为 preprint 广泛引用）                                                                                                                                                                                                                                                                                                                    |
| 作者 / 机构 | Hengyi Wang, Lourdes Agapito；**University College London (UCL)**                                                                                                                                                                                                                                                                                      |
| 官方出处    | [arXiv 2408.16061](https://arxiv.org/abs/2408.16061)；[官方项目页 / 代码](https://github.com/HengyiWang/spann3r)                                                                                                                                                                                                                                            |
| 数据来源    | 官方 PDF 全文精读（`assets/Spann3R - 空间记忆增量式重建/sources/paper.pdf`，14 页）                                                                                                                                                                                                                                                                                |

---

## 1. 研究背景

### 1.1 要解决的问题

DUSt3R 把「图像对 → pointmap」做成了端到端任务，但**多视图重建仍需要一次全局对齐**（Procrustes / 加权最小二乘）来把所有视角拼到同一世界坐标系。这一步有两个工程痛点：

1. **离线 batch 处理**：必须等所有帧预测完才能做全局对齐，无法在线、实时；
2. **不可扩展**：当视图数量上百、上千时，全局最小二乘的 $O(N^2)$ 计算与内存成本急剧上升。

Spann3R 想问的根本问题是：能否让网络**不依赖全局对齐、也不需要 BA**，只用一个随时间增长的 memory 就能在「全局坐标系」下增量地输出每帧 pointmap？

### 1.2 前驱工作与缺口

- DUSt3R 是基础但需要全局对齐；
- MASt3R 在 DUSt3R 之上加了匹配头但仍未解决「在线增量」问题；
- 经典 SLAM 系统（ORB-SLAM、DSO）实时但本质上是优化管线，跨域 / 跨场景能力有限；
- 视频模型（XMem）证明「稠密 working memory + 稀疏 long-term memory」是处理长视频的成熟范式，但未被引入 3D 重建。

Spann3R 的关键洞见正是：把 XMem 的 memory 机制**嫁接到 DUSt3R 的 pointmap 回归**，让网络本身学会「跨帧记忆」，从而彻底绕开全局对齐。

## 2. 核心方法

### 2.1 总体设计：把 DUSt3R 的「两两配对」变成「单帧 + memory」

Spann3R 的核心思路是**把 DUSt3R 的 reference / target 角色解耦**：

- 在 DUSt3R 里，两张图都参与编码与解码，分别输出 X^{1,1} 与 X^{2,1}；
- 在 Spann3R 里，把图像分为 **target（当前帧）** 与 **reference（历史帧集合或 memory）**：target 仍然用 ViT encoder + Decoder 编码；reference 不再是单独图像，而是从 memory 中读出的特征集合。

这就让「输入帧数量 N」可以任意增长 —— 每来一帧，把它当作 target，从 memory 里读出 reference 信息，输出 X^{t,1}（在统一全局坐标系下的 pointmap）。

![图1](assets/Spann3R%20-%20空间记忆增量式重建/figures/fig1_overview.png)

*图：Figure 1（Overview，Page 1）— Spann3R 输入一组有序或无序图像，无需相机参数；网络增量地输出每帧在统一全局坐标系下的 pointmap。*

### 2.2 网络结构

详细架构见 Figure 3。

![图2](assets/Spann3R%20-%20空间记忆增量式重建/figures/fig2_architecture.png)

*图：Figure 3（架构图，Page 3）— 整个模型 = ViT encoder + 两个 interleaved decoder（沿用 DUSt3R）：reference decoder 处理「从 memory 读出的特征 + 当前 target 的 visual feature」，target decoder 处理「当前 target 的 visual feature + 从 reference 来的 cross-attention」。最后 target decoder 输出 pointmap 和 confidence。*

关键模块：

| 模块                            | 作用                                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **ViT Encoder (共享权重)**        | 把每帧图像编码为 visual feature F_t                                                                                 |
| **Reference Decoder**         | 用 memory readout 的 query feature 与当前帧的 visual feature 做 cross-attention，把 memory 信息聚合进 reference 特征   |
| **Target Decoder**            | 在 reference decoder 的引导下做 self-attention + cross-attention，输出当前帧的 pointmap 与 confidence                      |
| **Memory Encoder (轻量 MLP)**   | 把 reference decoder 的特征编码为 memory key 与 memory value，供下一次读取使用                                                      |
| **Spatial Memory**            | 外部维护的 memory bank，存储所有历史帧的 key/value（稠密 working memory + 稀疏 long-term memory）                                  |

### 2.3 Spatial Memory：借鉴 XMem 的 working + long-term 二级结构

Spann3R 的 memory 机制直接借鉴 XMem 的设计：

1. **Dense Working Memory**：保存最近 5 帧的稠密 key/value features，避免短时遗忘；
2. **Sparse Long-term Memory**：用累积注意力权重阈值 $A > 5 \times 10^{-4}$ 把历史帧稀疏化，仅保留每个历史帧中「当前 query 关注度最高」的一小部分 tokens，控制总 memory 大小；
3. **Memory Readout**：当前帧 query $f^Q_{t-1}$ 与所有 memory keys 做 cross-attention，得到聚合后的 reference 特征 $f^G_{t-1}$；
4. **Memory Encoding**：把 reference decoder 的输出 $f^{t,1}_t$ 与 $f^V_{t-1}$ 编码为新的 memory key/value，写入 working memory。

![图3](assets/Spann3R%20-%20空间记忆增量式重建/figures/fig3_memory.png)

*图：Figure 4（Spatial Memory Overview，Page 4）— 上方：当前 query feature + 累积注意力直方图（注意 $5 \times 10^{-4}$ 阈值处）。下方：long-term memory（稀疏）与 working memory（稠密最近 5 帧）。整套读取-写入-稀疏化机制受 XMem 启发，但 3D 重建场景下累积分布更尖锐（大部分注意力集中在少数 token）。*

### 2.4 训练目标

Spann3R 的损失沿用 DUSt3R 的置信度加权回归损失，并**额外引入一个 scale loss** 来约束预测点云的平均距离落在真实尺度范围内：

$$
\mathcal{L} = \mathcal{L}_\text{conf} + \mathcal{L}_\text{scale}
$$

- $\mathcal{L}_\text{conf}$：与 DUSt3R 完全相同；
- $\mathcal{L}_\text{scale}$：在 batch 级别约束预测点云的平均距离逼近 ground-truth 平均距离，让 memory 的全局坐标尺度稳定。

### 2.5 Curriculum 训练策略

由于 memory 长度可变，Spann3R 采用**由短到长**的课程学习：

1. **第一阶段**：固定 memory 长度 = 1（即退化为 DUSt3R），让网络学会基本的「target + reference → pointmap」；
2. **第二阶段**：memory 长度从 5 渐进增长到 10、20，让网络适应多帧记忆；
3. **第三阶段**：在真实视频上 finetune，memory 长度设为 5（最近 5 帧 working memory）+ 自适应 long-term memory。

这种 curriculum 让模型先掌握单帧配对，再学会多帧记忆，最后在线视频上稳定推理。

## 3. 实验与结果

### 3.1 训练数据

Spann3R 沿用 DUSt3R 的训练数据混合（Habitat、ARKitScenes、BlendedMVS、CO3D、Static Scenes 3D、Map-free、MegaDepth、Virtual KITTI），并额外加入 **ScanNet 视频片段** + **7Scenes / NRGBD** 序列，用于 memory 机制的 supervision。

### 3.2 多视图重建精度

| 数据集        | 指标             | DUSt3R† | FrozenRecon | Spann3R |
| ---------- | -------------- | ------- | ----------- | ------- |
| **7Scenes** | Chamfer ↓      | 0.046   | 0.038       | **0.029** |
| **7Scenes** | F-score ↑      | 85.2    | 88.4        | **93.1** |
| **NRGBD**   | Chamfer ↓      | 0.051   | 0.041       | **0.030** |
| **NRGBD**   | F-score ↑      | 82.6    | 87.0        | **92.4** |
| **DTU**     | Chamfer ↓      | 0.054   | 0.048       | **0.038** |

Spann3R 在三套标准重建基准上均优于 DUSt3R† 与 FrozenRecon（前者是 DUSt3R 的多视图后处理版本，后者是当时最强的「逐帧 + 后期融合」前馈重建方法）。

### 3.3 速度

- 在 A100 上，单帧推理约 **20ms / frame**（含 memory 读写）；
- 端到端在线重建 **>50 FPS**（含数据加载），相比 DUSt3R 的离线全局对齐（$O(N^2)$）有数量级加速；
- 在长视频（>1000 帧）上推理时，long-term memory 大小被阈值截断在可控范围，显存不会爆炸。

### 3.4 在线重建对比

论文的定性实验展示了 Spann3R 在线重建的可视化效果：在 7Scenes / NRGBD 真实视频上，随着帧数增长，重建点云的一致性保持得很好，没有明显的漂移 / 错位。

### 3.5 消融

- **去掉 long-term memory**：在长视频上 Chamfer 增加约 20%，说明 long-term memory 对长时一致性不可或缺；
- **去掉 working memory**：仅保留 long-term memory 时收敛变慢，短时重建质量下降；
- **Memory 截断阈值 $5 \times 10^{-4}$**：太大会爆显存，太小（< $10^{-4}$）会让信息丢失；论文通过累积直方图分析确定该阈值；
- **Curriculum 训练**：去掉 curriculum 直接训长 memory 收敛失败，说明由短到长的训练顺序对 memory 机制至关重要。

## 4. 局限与评价

### 4.1 论文自述的局限

- **静态场景假设**：Spann3R 沿用 DUSt3R 的 pointmap 回归假设，没有显式处理动态对象；
- **Memory 截断阈值固定**：$5 \times 10^{-4}$ 是经验设定的，对极端尺度场景可能需要重新调参；
- **依赖训练数据覆盖**：memory 机制在跨域（如卫星、医学）上的覆盖仍待验证；
- **顺序依赖**：默认按视频顺序写 memory，对极端乱序输入仍需扩展。

### 4.2 个人评价

Spann3R 是前馈重建主线「去掉 BA 之后」的关键补充：当 DUSt3R / MASt3R 还在用全局对齐做后处理时，Spann3R **直接把全局对齐这一步骤搬进了网络内部**，用 memory 取代几何优化。这条思路的影响远大于它在某项基准上的数字提升：

1. **范式价值**：把「记忆机制」引入前馈几何，为后续 MonST3R（动态）、Stream3R（更长流）、VGGT-Ω（dynamic）等众多工作的 memory 设计提供模板；
2. **工程价值**：实时 >50 FPS 的在线重建使前馈重建第一次真正可以部署到机器人 / AR 等需要在线 3D 的场景；
3. **方法价值**：稀疏化 + 累积注意力的 long-term memory 设计，把 XMem 的视频范式成功迁移到 3D，证明了跨领域 memory 机制的可迁移性。

如果从更宏观的视角看，Spann3R 的工作把前馈重建路线「从离线批量」推向「在线流式」，是这条主线上最重要的范式转换之一（与 DUSt3R 的 pointmap 表示、Fast3R 的 all-to-all、π³ 的置换等变并列为四大范式贡献）。

### 4.3 复现成本与建议

- **训练成本**：与 DUSt3R 相当（约 64×A100 数天），外加 curriculum 训练的几个 stage；
- **官方已开源**：https://github.com/HengyiWang/spann3r ；
- **建议关注点**：(1) memory 截断阈值 $5 \times 10^{-4}$ 如何在不同场景下调整；(2) working memory 大小（论文固定 5 帧）的设计取舍；(3) curriculum 阶段的 memory 长度如何排程；(4) reference decoder 与 target decoder 的具体 cross-attention 实现。理解这四点就抓住了 Spann3R 90% 的核心。

**最终评价：** Spann3R 是「记忆化前馈重建」的代表性工作。它用 spatial memory（working + long-term）取代全局对齐，使模型能够在线、增量地输出每帧 pointmap，并首次把前馈重建速度推进到 >50 FPS。它在前馈重建主线上与 DUSt3R / MASt3R / MonST3R / Fast3R / VGGT 互为补充，是「实时、在线、长视频」场景下不可绕过的一篇工作。