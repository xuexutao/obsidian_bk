---
type: paper-note
status: complete
domain: World-Models
paper: "Geometry-Aware Implicit Memory for Video World Models"
year: 2026
arxiv: "2606.02436"
source: https://arxiv.org/abs/2606.02436
project: https://gim-world.github.io/
code: null
tags:
  - World-Models
  - Implicit-Memory
  - Geometry-Distillation
  - VGGT
  - Kling
created: 2026-10-16
updated: 2026-10-16
---

# GIM-World：几何感知隐式记忆

## 基本信息

- **正式题名**：Geometry-Aware Implicit Memory for Video World Models
- **作者**：Zhengxuan Wei、Xu Guo、Xinghui Li、Xunzhi Xiang、Min Wei、Yiran Zhu、Qiulin Wang、Xintao Wang、Pengfei Wan、Xiangwang Hou、Qi Fan
- **机构**：南京大学、快手 Kling 团队、清华大学
- **版本**：arXiv v1，2026-06-01
- **基础模型**：Wan2.1；训练期几何教师为冻结 VGGT
- **核心关键词**：fixed-size implicit memory、camera-queryable geometry head、geometry distillation、information-guided pruning

## 一句话结论

GIM-World 不把长期记忆做成帧库或在线 3D 地图，而是压缩为固定数量 token；为了防止这些 token 只记冗余外观，它用相机射线查询记忆并要求查询结果匹配冻结 VGGT 的几何特征，把跨视角三维结构直接蒸馏进 memory state，推理时则完全移除教师，取得第一人称 Reprojection **81.70**、第三人称 **87.10**。

---

## 1. 研究背景与问题定义

### 1.1 世界模型究竟应记住什么

长程视频生成中的记忆大致分为两类。显式记忆保存历史帧或在线 3D 重建，容易受启发式检索、外观冗余、重建误差和动态物体影响；隐式记忆把历史压成紧凑状态，但如果训练目标只有下一段视频生成，压缩状态未必主动保留跨视角几何。GIM-World 的问题意识是：**固定大小只是容量约束，几何教师才决定这份容量被用来记什么。**

![](assets/GIM-World%20-%20几何感知隐式记忆/figures/01-teaser.png)

> 图 1：在同一场景多次移动与重访时，baseline 出现几何畸变和 memory drift；GIM-World 的视角序列与底层场景结构保持一致。来源：论文 Figure 1。

### 1.2 任务目标

给定历史观测与相机：

$$\mathcal H_t=\{(x_i,c_i)\}_{i=1}^{t}$$

以及未来动作块 $a_{t+1:t+K}$，生成：

$$x_{t+1:t+K}\sim p_\theta(\cdot\mid\mathcal H_t,a_{t+1:t+K})$$

无界历史被记忆算子压缩：

$$m_t=\mathcal M(\mathcal H_t),\qquad p_\theta(x_{t+1:t+K}\mid m_t,a_{t+1:t+K})$$

目标是让 $m_t$ 大小固定、可被相机查询、保留跨视角结构，并且推理时不运行显式 3D 重建或几何教师。

### 1.3 贡献

1. 两层轻量 Transformer 把可变历史压成固定 memory tokens。
2. 用 camera-queryable geometry head 从记忆恢复指定视角的 VGGT 特征，将几何监督直接作用于 memory，而非生成器中间层。
3. 用 pose-time 高斯过程互信息贪心选择历史帧，使编码成本在长历史下有界。
4. 在 MIND 第一/第三人称长程 rollout 上同时改善像素、动作和 3D reprojection 指标。

---

## 2. 核心方法总览

![](assets/GIM-World%20-%20几何感知隐式记忆/figures/02-method-pipeline.png)

> 图 2：上方是推理主链：历史帧→VAE→信息引导剪枝→固定 token 记忆→DiT→追加历史。左下为 compact-attention memory encoder；右下为只在训练出现的 VGGT 教师与相机射线查询几何头。来源：论文 Figure 2。

### 2.1 Fixed-size implicit memory

历史帧被 token 化为：

$$H=\{h_{i,p}\}_{i=1,p=1}^{T,P}$$

相机 embedding 只加到历史 token：

$$\tilde h_{i,p}=h_{i,p}+E_c(c_i)$$

随后拼接 $N_m$ 个可学习查询 $Q_0$，经过两层 encoder，取前 $N_m$ 个输出作为 memory：

$$Z_0=[Q_0;\tilde H],\qquad m=Z_{1:N_m}$$

$N_m$ 被设为与 20 个 latent frames 的 patch-token 数相同。记忆 query 自身不绑定 pose，避免退化为固定帧槽；相机信息由历史 token 提供。

### 2.2 Compact attention encoder

每层执行：

$$Z\leftarrow Z+\operatorname{Expand}(\operatorname{Attn}(\operatorname{Compact}(Z)))$$

$$Z\leftarrow Z+\operatorname{FFN}(Z)$$

`Compact` 把每个 $s\times s$ token 块经共享线性层压缩，attention 在低分辨率运行，再 `Expand` 恢复；FFN 保持全分辨率。默认 $s=2$，attention token 数减少 4×。memory query 与 history 使用分离的 rotary positional grids。默认 encoder 运行时间仅为 DiT decode 的 **0.26%**。

---

## 3. 几何教师蒸馏

### 3.1 为什么不能直接对齐 memory token

压缩后的 memory token 不再与输入帧/patch 一一对应。若强行让 memory token 对齐 VGGT token，会给记忆施加人工索引，使其退化为 frame-indexed cache。GIM-World 改为：给定历史相机，从 memory 查询该视角应看到的几何特征，再与 VGGT 对齐。

### 3.2 Camera-queryable geometry head

每步从历史均匀采样帧 $i$，冻结 VGGT 产生目标特征：

$$G_i\in\mathbb R^{H_gW_g\times d_g}$$

对每个 patch $(u,v)$，构建世界坐标射线：

$$\rho(c_i)_{u,v}=[o_i,d_{i,u,v}]\in\mathbb R^6$$

其中 $o_i$ 是相机原点，$d_{i,u,v}$ 是归一化射线方向。query 为：

$$q_{i,u,v}=\operatorname{MLP}_{\mathrm{ray}}(\rho(c_i)_{u,v})+e_{u,v}$$

几何头通过 cross-attention 从 memory 取回该视角所需信息，再经 query-grid self-attention 与 FFN：

$$\hat G_i=\operatorname{FFN}(\operatorname{SelfAttn}(\operatorname{CrossAttn}(q_i,m)))$$

逐 patch cosine loss：

$$\mathcal L_{\mathrm{geo}}=1-\frac{1}{H_gW_g}\sum_{u,v}\cos(\hat G_{i,u,v},G_{i,u,v})$$

每步随机采一帧，是全历史几何对齐目标的无偏估计，因此蒸馏成本不随历史长度线性增加。

### 3.3 与 Geometry Forcing 的区别

Geometry Forcing 把 VGGT 特征蒸馏到 diffusion 中间层，主要约束生成器；GIM-World 直接约束“记忆是否能回答一个相机视角下的几何查询”。这一区别被消融强力支持：Geometry Forcing 的 Reproj. 只有 63.40，而 queryable memory supervision 达到 81.70。

---

## 4. Information-guided pruning

### 4.1 为什么固定 memory 仍需要剪枝

输出 memory 大小虽固定，但编码器若读取全部历史 token，前处理成本仍随 $T$ 增长。模型最多选择 $K=200$ 个历史帧：

$$S^*=\arg\max_S I(S;\mathcal H\setminus S),\quad |S|\le K$$

### 4.2 Pose-time kernel

历史帧相似度由位置、朝向和时间共同定义：

$$k(c_i,c_j)=\exp\left(-\frac{\|p_i-p_j\|_2^2}{2\sigma_p^2}-\frac{\angle(\mathbf f_i,\mathbf f_j)^2}{2\sigma_r^2}-\frac{(t_i-t_j)^2}{2\sigma_t^2}\right)$$

给定集合 $A$ 后，候选帧 $h$ 的 GP 后验方差为：

$$\sigma^2(h\mid A)=k(h,h)-\mathbf k_{hA}^\top K_{AA}^{-1}\mathbf k_{Ah}$$

贪心增益：

$$s_t=\arg\max_{h\in\bar S_{t-1}}\frac12\log\frac{\sigma^2(h\mid S_{t-1})}{\sigma^2(h\mid\bar S_{t-1}\setminus\{h\})}$$

分子大意味着当前已选集不能解释该帧，分母小意味着该帧能代表其余未选历史；因此偏好“新颖且有代表性”的视角。需要注意，核函数只使用 pose、朝向与时间，不直接使用图像内容。

---

## 5. 训练与推理

### 5.1 总目标

生成器采用 flow matching：

$$\mathcal L_{\mathrm{FM}}=\mathbb E\left[\|v_\theta(z_t,t,m,a)-v^*\|_2^2\right]$$

总损失：

$$\mathcal L=\mathcal L_{\mathrm{FM}}+\lambda\mathcal L_{\mathrm{geo}},\quad\lambda=0.05$$

memory encoder、geometry head 与 diffusion backbone 单阶段联合训练；VGGT 冻结。训练 88K steps，学习率 $10^{-5}$，batch size 32。推理时删除 VGGT、geometry head 和几何损失，仅保留剪枝、两层 memory encoder 与 DiT。

### 5.2 数据与协议

MIND 基于 UE5，含 100 个第一人称和 100 个第三人称 clips，每段约 3 分钟、24 FPS、原始 1080p，按两类各 50/50 划分训练/评估。评估使用 720p；每段前约 1/4 为 context，后 3/4 用相同动作做自回归 prediction。

动作包括 WASD 平移和方向键相机旋转。像素指标为 MSE/PSNR/SSIM/LPIPS；动作使用 ViPE 恢复轨迹、Sim(3) 对齐后报告平移/旋转 RPE；几何用 Depth Anything 3 估计深度和位姿，计算跨帧 cycle reprojection score。

---

## 6. 实验结果

### 6.1 与 SOTA 比较

| 视角 | 方法 | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Trans. ↓ | Rot. ↓ | Reproj. ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 第一人称 | CaM | 0.0706 | 12.50 | 0.3917 | 0.6953 | **0.0235** | 0.5697 | 66.14 |
| 第一人称 | SSM | 0.0796 | 11.96 | 0.3953 | 0.7439 | 0.0379 | 0.6702 | 53.95 |
| 第一人称 | **GIM-World** | **0.0614** | **13.40** | **0.4135** | **0.6304** | 0.0247 | **0.4126** | **81.70** |
| 第三人称 | CaM | 0.0671 | 12.81 | 0.4051 | 0.6480 | 0.0241 | 0.4385 | 69.17 |
| 第三人称 | SSM | 0.0928 | 11.76 | 0.4037 | 0.6830 | 0.0201 | 0.2971 | 61.26 |
| 第三人称 | **GIM-World** | **0.0605** | **13.58** | **0.4303** | **0.5974** | **0.0106** | **0.1588** | **87.10** |

第一人称 Trans. 仍由 CaM 的 0.0235 略胜 GIM 的 0.0247；其余可比主要指标由 GIM-World 领先。几何收益最突出：第一/第三人称 Reproj. 达到 81.70/87.10。

![](assets/GIM-World%20-%20几何感知隐式记忆/figures/03-first-person-results.png)

> 图 3：400 帧第一人称 rollout。FramePack、CaM 和 SSM 在约 100 帧后出现布局漂移、墙体/地标变化；GIM-World 与 GT 的场景结构和重访视角更接近。来源：论文 Figure 3。

### 6.2 几何监督消融

| 变体 | MSE ↓ | LPIPS ↓ | Trans. ↓ | Rot. ↓ | Reproj. ↑ |
|---|---:|---:|---:|---:|---:|
| w/o geometry | 0.0628 | 0.6485 | 0.0252 | 0.5028 | 58.80 |
| Geometry Forcing | 0.0623 | 0.6452 | 0.0251 | 0.4823 | 63.40 |
| **Queryable memory supervision** | **0.0614** | **0.6304** | **0.0247** | **0.4126** | **81.70** |

![](assets/GIM-World%20-%20几何感知隐式记忆/figures/04-geometry-ablation.png)

> 图 6：无几何监督和 Geometry Forcing 在 400 帧内出现尺度/视角漂移；直接监督记忆的方案更接近 GT。来源：论文 Figure 6。

这组消融表明，几何特征“蒸馏到哪里”比“有没有教师”更重要。只有让 memory 能回答相机查询，才真正改变了长期状态表示。

### 6.3 剪枝消融

在同样 $K=200$ 的预算下，Uniform / Camera FPS / MI greedy 的 Reproj. 分别为 **77.05 / 77.52 / 81.70**，LPIPS 为 0.6582 / 0.6553 / **0.6304**。互信息贪心在所有报告指标上最佳。

### 6.4 Compact attention 消融

| 变体 | Time / DiT ↓ | MSE ↓ | LPIPS ↓ | Reproj. ↑ |
|---|---:|---:|---:|---:|
| Full attention | 2.80% | **0.0608** | 0.6512 | **81.85** |
| Global Compact $s=2$ | 0.20% | 0.0696 | 0.6892 | 62.45 |
| **Attention Compact $s=2$** | **0.26%** | 0.0614 | **0.6304** | 81.70 |
| Attention Compact $s=4$ | 0.10% | 0.0651 | 0.6498 | 76.18 |

只压缩 attention、保留全分辨率 FFN 是关键；若连 FFN 一起压缩，Reproj. 从 81.70 降到 62.45。默认方案几乎保留 full attention 的几何质量，却快约 10.8×。

---

## 7. 与 Astra、DecMem、AnchorWorld 的关系

Astra 仍直接条件化历史视频块，用噪声降低历史的控制支配力；DecMem 从全历史中检索细粒度块，并用局部窗口稳定注意力；GIM-World 选择完全不同的容量策略：把历史压成固定 token，再借教师决定其几何语义。相比 DecMem，GIM 推理开销更轻，但依赖训练期 VGGT 和相机信息，且没有 DecMem 式分钟级开放历史块召回。AnchorWorld 的 pose-associated anchor view 是用户指定的外部世界状态；GIM 的 memory 是自动压缩的内部状态。两者结合可让锚点提供可编辑语义，隐式记忆负责跨视角几何保持。

---

## 8. 局限与批判性评价

### 8.1 论文未单列 limitations

以下是从实验边界可直接确认的限制，而非作者正式声明：

1. 只在 MIND 第一/第三人称数据上评测，尚无驾驶、机器人、真实世界或开放域验证。
2. 需要历史相机位姿，训练还依赖冻结 VGGT；论文未报告教师的绝对训练成本。
3. pruning 核不读取图像内容，动态物体或外观突变可能无法被 pose-time 新颖性捕捉。
4. 1000 帧结果只有单个第一人称场景，缺乏 baseline 与聚合指标。
5. 只报告 encoder 相对 DiT 的 0.26%，未给绝对延迟、显存、参数量和训练硬件。
6. 没有标准差、置信区间或系统 failure-case 统计；自身仅承认个别帧有小幅 viewpoint offset。

### 8.2 个人判断

GIM-World 的最强洞见是把 teacher distillation 从“让生成器更懂几何”转成“让记忆成为可查询的几何数据库”。这种监督与 memory 的接口一致，因而消融远强于 Geometry Forcing。代价是相机和教师成为训练前提；如果输入位姿不准，或 VGGT 在动态/弱纹理场景中偏差较大，错误可能被压缩进固定 memory，论文尚未测试这一点。

---

## 9. 复现与实践建议

1. 先复现 w/o geometry、Geometry Forcing、queryable memory 三组，不要只复现完整模型；Reproj. 58.80→63.40→81.70 是核心验收链。
2. 使用 $N_m=$20 latent frames 的 patch-token 数、2 个 encoder blocks、$s=2$、$K=200$、$\lambda=0.05$、88K steps、batch 32、学习率 $10^{-5}$。
3. 确保相机射线在统一世界坐标系中构建，并为 query grid 添加共享二维位置 embedding。
4. 推理导出图中不应加载 VGGT 或 geometry head；若仍存在教师计算，就没有验证论文的轻量主张。
5. 剪枝需复现 Uniform、Camera FPS 和 MI greedy，确认收益来自互信息准则而非单纯限长。
6. 当前未给出官方代码与权重，复现难点主要是 MIND 数据预处理、相机约定、VGGT feature 层选择和 reprojection 评测实现。

---

## 10. 个人启发与后续问题

- 几何教师能否换成多教师：VGGT 约束静态几何，动态 4D 模型约束可动物体？
- 若没有可靠相机，能否联合估计 pose，并用不确定性调节几何损失？
- 固定 memory 与 DecMem 的稀疏历史块是否可形成快/慢两级记忆？
- AnchorWorld 的 anchor pose 能否直接充当 camera query，把可编辑锚点写入 GIM memory？

---

## 11. 材料来源

- 官方论文：https://arxiv.org/abs/2606.02436
- 项目页：https://gim-world.github.io/
- 代码：论文未提供明确仓库链接
- 本地 PDF 与逐图溯源：`assets/GIM-World - 几何感知隐式记忆/sources.md`
