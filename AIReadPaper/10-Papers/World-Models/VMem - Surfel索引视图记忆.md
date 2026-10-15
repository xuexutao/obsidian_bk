---
type: paper-note
status: complete
domain: World-Models
paper: "VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory"
year: 2025
arxiv: 2506.18903
version: v3
venue: ICCV 2025 Highlight
source: https://arxiv.org/abs/2506.18903
project: https://v-mem.github.io
code: null
tags: [World-Model, View-Memory, Surfel, Retrieval, Novel-View-Synthesis, SEVA, Interactive-Generation]
created: 2026-10-15
updated: 2026-10-15
---

# VMem：Surfel 索引视图记忆

## 基本信息

- **作者**：Runjia Li、Philip Torr、Andrea Vedaldi、Tomas Jakab（University of Oxford）
- **版本/发表**：arXiv v3，2025-08-14；ICCV 2025 Highlight
- **任务**：从一张输入图出发，按用户交互指定的相机轨迹，每次生成少量新视图；长时间探索并返回旧区域时保持场景一致
- **生成器**：SEVA image-set generator；高效版用 LoRA 微调为 $K=4$ 参考视图、$M=4$ 目标视图
- **几何估计**：CUT3R point map predictor；几何只建立 view index，不直接渲染为最终 RGB
- **资源**：[[assets/VMem - Surfel索引视图记忆/sources/sources|来源记录]]；[官方 PDF](assets/VMem%20-%20Surfel%E7%B4%A2%E5%BC%95%E8%A7%86%E5%9B%BE%E8%AE%B0%E5%BF%86/sources/paper.pdf)

## 一句话结论

**VMem 用粗糙 surfel 几何回答“目标视角会看到哪些表面、哪些历史 RGB 视图看过这些表面”：每个 surfel 存观察它的 view indices，目标视角渲染可见 surfel 后按索引出现频次投票取 top-$K$ 旧视图；生成器仍以原始 RGB+相机为条件，因此几何误差通常只造成检索错误而非直接污染画面，在 $K=4$ 时以约 12× 速度优势保留多数 $K=17$ 性能。**

---

## 1. 研究问题与方法定位

交互场景生成有两条传统路线：

1. **2D outpainting + 3D 重建**：不断把新图拼进几何，重建/拼接误差会积累并直接污染后续渲染；
2. **geometry-free multi-view/video generation**：只给最近几帧，避免硬几何误差，却因上下文预算丢失远期视图。

VMem 提出中间路线：几何不作为最终世界表示，也不直接生成像素，只做**历史视图的空间倒排索引**。生成器每次仍读取少量原始 RGB 视图，既保留视觉细节，又把上下文预算花在与当前目标表面最相关的旧视图上。

![](assets/VMem%20-%20Surfel%E7%B4%A2%E5%BC%95%E8%A7%86%E5%9B%BE%E8%AE%B0%E5%BF%86/figures/01-teaser.png)

> **图 1（Figure 1，PDF p.1）**：从单图沿 A→E 轨迹生成；无记忆版本在后段墙面、门窗与布局明显漂移，VMem 回访时保持结构。左侧点云只是索引辅助，不是最终 RGB scene representation。来源：<https://arxiv.org/abs/2506.18903>，官方源码 `imgs/vmem_teaser.pdf`。

---

## 2. 任务、输入与输出

给定初始 RGB 图 $\mathbf x_1\in\mathbb R^{H\times W\times3}$ 和用户相机序列 $\{\mathbf c_t\}_{t=1}^T$，输出视频 $\{\mathbf x_t\}_{t=1}^T$。每个 view 为：

$$v_t=(\mathbf x_t,\mathbf c_t).$$

第 $s$ 次迭代已有 $T=sM$ 帧：

$$\mathcal V^{(s)}=\{v_1,\ldots,v_T\},$$

系统接收未来 $M$ 个相机位姿，检索 $K$ 个历史 view，生成 $M$ 张新图并写回记忆。论文主配置 $K=M=4$；原始 SEVA 为 $K+M=21$，即 $K=17,M=4$。

输入可以在每轮根据已生成画面交互决定，因此目标不是离线给定整条轨迹，而是低延迟、可继续探索的 autoregressive image-set generation。

---

## 3. 记忆表示：RGB 视图库 + Surfel 倒排索引

### 3.1 Surfel 定义

每个 surfel 是：

$$
\mathbf s_k=(\mathbf p_k,\mathbf n_k,r_k,\mathcal I_k),
$$

其中 $\mathbf p_k\in\mathbb R^3$ 是位置，$\mathbf n_k\in\mathbb R^3$ 是法向，$r_k$ 是覆盖半径，$\mathcal I_k\subseteq\{1,\dots,T\}$ 是观察过该表面的历史 view indices。所有 surfel 构成：

$$\mathcal S^{(s)}=\{\mathbf s_k\}_{k=1}^{N^{(s)}}.$$

系统另维护 octree 加速空间检索。真正的高保真记忆仍是 $\mathcal V^{(s)}$ 中的 RGB+camera；$\mathcal S^{(s)}$ 是“表面→视图”的索引。

### 3.2 为什么用 Surfel

相对点云，surfel 有局部法向和半径，可进行有覆盖面积、深度与遮挡关系的 splatting；相对 mesh，它不要求全局拓扑正确，更适合不断追加、几何不精确的生成场景。

### 3.3 与 SPMem 的根本差别

[[SPMem - 长期空间记忆视频世界模型]] 把点云渲染直接送入 Condition DiT，几何错会成为强生成条件；VMem 只用 surfel 找 RGB 参考图。只要检索仍拿到正确视图，粗几何无需像可渲染地图那样精确。这是 VMem 的鲁棒性来源。

---

## 4. 记忆读取、写入与生成公式

### 4.1 Reading：可见表面索引投票

对下一批 $M$ 个目标相机先计算平均 pose $\bar{\mathbf c}_s\in SE(3)$。平移直接平均；旋转转四元数、对齐到同一半球后归一化：

$$
\bar{\mathbf q}=\frac{\sum_m\tilde{\mathbf q}_m}{\|\sum_m\tilde{\mathbf q}_m\|},\quad
\tilde{\mathbf q}_m=\operatorname{sign}(\mathbf q_m\cdot\mathbf q_1)\mathbf q_m.
$$

从 $\bar{\mathbf c}_s$ 渲染 surfel 及其 view-index 属性，按所有像素中索引出现频次排序，取 top-$K$：

$$
\mathcal I^*=\operatorname{TopK}(\operatorname{Freq}(\operatorname{Render}(\mathcal S^{(s)},\bar{\mathbf c}_s))),
$$

$$
\mathcal V^*=\{v_t\}_{t\in\mathcal I^*}.
$$

NMS 保留相似 pose 中频次最高的 view，避免反复访问区域垄断上下文。

### 4.2 Generator：RGB 与 Plücker camera 条件

SEVA 接收 $K$ 个 reference RGB、参考相机与 $M$ 个目标相机的 Plücker embedding：

$$
\{\mathbf x_{T+m}\}_{m=1}^{M}\sim\psi\left(
\{(\mathbf x_t,\mathbf c_t)\}_{t\in\mathcal I^*},
\{\mathbf c_{T+m}\}_{m=1}^{M}
\right).
$$

### 4.3 Writing：新视图变为 Surfel 并合并

CUT3R 将新视图与检索旧视图联合估计，保证新 point maps 对齐现有坐标系。旧 depth 与用户相机在优化中冻结，维持因果一致性。point map 先按 $\sigma$ 下采样；局部法向由邻域差分叉积：

$$
\mathbf n=\frac{(\mathbf p_{u+1,v}-\mathbf p_{u-1,v})\times(\mathbf p_{u,v+1}-\mathbf p_{u,v-1})}{\|(\cdot)\times(\cdot)\|}.
$$

半径启发式与深度正比、与焦距及斜视角有关：

$$
r=\frac{\frac12 D/f}{\alpha+(1-\alpha)|\mathbf n\cdot(\mathbf p-\mathbf O)|}.
$$

若新旧 surfel 中心距离小于 $d$ 且法向余弦相似度高于 $\theta$，把新 frame index 加到旧 $\mathcal I_k$；否则新建 surfel。

![](assets/VMem%20-%20Surfel%E7%B4%A2%E5%BC%95%E8%A7%86%E5%9B%BE%E8%AE%B0%E5%BF%86/figures/02-method.png)

> **图 2（Figure 2，PDF p.3）**：左侧 surfel index 检索 RGB references，reference/target Plücker 与噪声共同输入 image-set generator；新视图由 writing module 回写索引。几何控制的是“读哪几张图”，而不是“直接画什么”。来源：<https://arxiv.org/abs/2506.18903>，官方源码 `imgs/vmem_method.pdf`。

![](assets/VMem%20-%20Surfel%E7%B4%A2%E5%BC%95%E8%A7%86%E5%9B%BE%E8%AE%B0%E5%BF%86/figures/03-memory-index.png)

> **图 3（Figure 4，PDF p.5）**：Reading 从目标相机渲染可见 surfel 并通过索引投票取旧视图；Writing 对新图估 point map、生成/合并 surfel。该数据结构本质上是带遮挡感知的空间倒排表。来源：<https://arxiv.org/abs/2506.18903>，官方源码 `imgs/vmem_spatial_memory.pdf`。

---

## 5. 训练、推理与计算成本

### 5.1 训练

原始 SEVA 固定 $K+M=21$。作者用 RealEstate10K 对高效版做 LoRA 微调：

- $K=4,M=4$；
- LoRA rank 256；
- 600,000 iterations；
- 8×A40；
- 每 GPU batch 24；
- AdamW，learning rate $3\times10^{-6}$，weight decay $10^{-4}$；
- cosine annealing；
- 在线随机采样 context views。

这个训练规模并不“小”：每步全局 batch 192，且 600k steps。12× 是推理相对 SEVA 的速度收益，不代表训练廉价。

### 5.2 推理超参

- CFG scale 3；
- point map scaling/downsampling factor $\sigma=0.03$；
- surfel radius 参数 $\alpha=0.2$；
- RTX 4090 上约 4.16/4.2 秒生成一帧；原始 SEVA 约 50 秒。

仍远未实时，但对交互式“点一下等几秒”的探索已比 50 秒可用。

### 5.3 错误传播路径

$$
\text{生成图误差}\rightarrow\text{CUT3R point map误差}\rightarrow\text{surfel合并/索引归属错误}\rightarrow\text{检索到错误RGB}\rightarrow\text{SEVA生成偏离}\rightarrow\text{继续写入错误索引}.
$$

相对 SPMem，这条链有“软隔离”：错误几何不直接显示，只在选错 reference 时影响生成；原始 RGB 仍可覆盖粗几何误差。但错误索引若持续投票，会形成长期错误吸引子。论文没有置信度衰减、索引纠错或 loop-closure 全局优化。

---

## 6. 数据集、评测协议与指标

### 6.1 数据

- **RealEstate10K**：室内房地产视频，既用于高效版微调也用于主评测；
- **Tanks and Temples**：六个 advanced indoor/outdoor scenes，测试更大相机运动的 OOD 泛化；
- 网络/手机拍摄图：只做定性。

训练与主测试同属 RealEstate10K 域；作者承认自然景观、动态对象泛化有限。

### 6.2 三种协议

1. **标准 short/long-term NVS**：每 10 帧取样；评第 50 帧和 $\ge200$ 帧，但轨迹很少重访；
2. **cycle trajectory**：走到终点再原路返回，专门测试看过区域的重访；
3. **Tanks and Temples cycle**：前 50 帧，不做时间下采样。

### 6.3 指标

- 视觉/细节：FID↓、PSNR↑、LPIPS↓、SSIM↑；
- 相机：DUSt3R 从生成图估 pose，计算旋转距离

$$R_{dist}=\arccos\left(0.5(\operatorname{tr}(R_{gen}R_{gt}^T)-1)\right),$$

与平移距离

$$t_{dist}=\|t_{gt}-t_{gen}\|_2.$$

作者正确指出：单图出发时“真值世界”不存在，低层像素相似并不等价于真实多视图一致；cycle 也相对简单、遮挡有限。

---

## 7. 主要定量结果

### 7.1 标准 short/long-term

在 RealEstate10K：

- 短期第 50 帧，VMem $K=4$：LPIPS 0.287、PSNR 18.49、SSIM 0.406、FID 17.12；
- 长期 $\ge200$ 帧，VMem $K=17$：LPIPS 0.452、PSNR 14.09、SSIM 0.227、FID 23.56。

标准长期轨迹很少重访，因此 $K=4$ 的 VMem 长期项（PSNR 13.12）不如 $K=17$；这说明它的优势不是任意 extrapolation，而是**有限上下文下找回曾经看过的区域**。

### 7.2 RealEstate10K cycle

| 方法 | LPIPS↓ | PSNR↑ | SSIM↑ | FID↓ | $R_{dist}$↓ | $T_{dist}$↓ |
|---|---:|---:|---:|---:|---:|---:|
| SEVA $K=17$ | 0.401 | 11.82 | 0.217 | 24.72 | 0.902 | 0.492 |
| VMem $K=4$ | 0.397 | 15.72 | 0.297 | 24.97 | **0.821** | 0.392 |
| **VMem $K=17$** | **0.304** | **18.15** | **0.377** | **24.18** | 0.892 | **0.165** |

用 4 个相关视图可达到原始 17-view SEVA 的 LPIPS，并把 PSNR 提高 3.9 dB；17-view VMem 则进一步提高到 18.15 dB。

### 7.3 Tanks and Temples OOD

VMem $K=4$ 获得 LPIPS 0.472、PSNR 14.11，优于所有基线；SSIM 0.121 略低于 GenWarp/ViewCrafter 0.129，旋转/平移也非全优。这是“多数指标领先”，不是全面领先。

![](assets/VMem%20-%20Surfel%E7%B4%A2%E5%BC%95%E8%A7%86%E5%9B%BE%E8%AE%B0%E5%BF%86/figures/04-results.png)

> **图 4（Figure 6，PDF p.8）**：两条超过 400 帧的 cycle 轨迹。Lookout/GenWarp/MotionCtrl/ViewCrafter 在转回时逐渐改写门窗和走廊；VMem 在 backward 段重新取回旧视图，接近 GT。来源：<https://arxiv.org/abs/2506.18903>，官方源码 `imgs/vmem_long-term_comparison_figure.pdf`。

---

## 8. 检索消融与失败边界

### 8.1 检索策略消融

| K | 检索 | LPIPS↓ | PSNR↑ | SSIM↑ | $R_{dist}$↓ | $T_{dist}$↓ |
|---:|---|---:|---:|---:|---:|---:|
| 17 | Temporal | 0.477 | 13.92 | 0.188 | 0.976 | 0.254 |
| 17 | Camera distance | 0.397 | 15.72 | 0.297 | **0.821** | 0.392 |
| 17 | FOV | 0.374 | 15.75 | 0.292 | 0.911 | 0.382 |
| 17 | **VMem surfel** | **0.304** | **18.15** | **0.377** | 0.892 | **0.165** |
| 4 | Temporal | 0.794 | 7.52 | 0.018 | 1.942 | 0.458 |
| 4 | Camera distance | 0.422 | 13.27 | 0.187 | 1.787 | 0.319 |
| 4 | FOV | 0.424 | 13.11 | 0.192 | 1.782 | 0.285 |
| 4 | **VMem surfel** | **0.381** | **14.82** | **0.275** | **0.793** | **0.124** |

预算越小，表面可见性索引的价值越大。相机距离/FOV 只判断视角接近，无法区分隔墙、遮挡与“同位置看不同表面”；surfel 投票把共视细化到表面。

### 8.2 明示与隐含失败

论文没有单独 failure figure，但在附录给出边界：

- cycle 轨迹简单、遮挡有限，surfel 的遮挡优势尚未被强压测；
- 训练数据以室内房产为主，动态对象、自然景观会退化；
- CUT3R 与 SEVA 的错误都会限制 VMem；
- 4.16 秒/帧不适合 VR 实时交互；
- 单图 hallucinated world 没有唯一 GT，PSNR/SSIM 可能奖励“回到同样纹理”，不一定是真正三维自洽。

另外，surfels 数量、octree 内存、view database 长期存储和超长序列检索时延均未报告。

---

## 9. 局限、Future Work 与复现成本

### 9.1 局限

- 几何虽为软索引，但错误会导致错误引用投票，缺少置信度和索引清理。
- view memory 原始 RGB 线性增长；只对相似 pose 去旧保新，未形成严格容量上限。
- 平均目标 pose 代表 $M=4$ 视图；若批内相机跨度大，单一平均视角可能漏掉各目标的不同可见表面。
- 对动态场景没有时间/对象状态索引，同一 surfel 可能关联互相冲突的时刻。
- $K=4$ 高效模型的训练很重，且只在 RealEstate10K 微调。

### 9.2 Future Work

更强 image-set generator、point map predictor 和单步扩散可直接提升 VMem。更进一步可做每目标视角独立检索、动态 surfel/时间戳、基于置信度的索引衰减、回环校正，以及固定容量的 view consolidation。

### 9.3 复现成本

**高，但模块边界比 SPMem 清晰。** 需要 SEVA、CUT3R、surfel/octree renderer 与 8×A40 的 600k-step LoRA。算法定义和训练超参较完整，读取/写入可独立实现；但官方代码未在论文/arXiv 页面提供，严格指标复现仍受 SEVA 训练与数据处理细节影响。

---

## 10. 个人判断与横向比较

### 10.1 个人判断

VMem 是四篇中最漂亮的“表示—职责分离”：**几何只负责寻址，RGB 负责内容，生成器负责补全。** 它承认重建不会完美，并主动把几何从最终像素表示降级为索引，因此比硬点云条件更抗误差。消融也明确证明 surfel 可见面投票优于时间、相机距离和 FOV。

保留意见是：12× 加速主要来自把 SEVA 参考帧从 17 减到 4并额外微调，VMem 检索本身不是全部速度来源；生成仍需 4 秒/帧，且长期内存增长没有系统分析。

### 10.2 四篇方法定位

| 方法 | 内容记忆 | 寻址 | 几何强度 | 典型错误链 |
|---|---|---|---|---|
| [[Context as Memory - 历史上下文驱动的场景记忆]] | RGB 帧 | FOV 共视 | 弱 | FOV 误检→错误旧帧条件 |
| [[SPMem - 长期空间记忆视频世界模型]] | 静态点云+近帧+关键帧 | 目标轨迹渲染/未知区 | 强 | 重建错→TSDF 错→硬结构条件错 |
| **VMem** | RGB view DB；surfel 存 indices | 可见 surfel 频次投票 | 中 | point map错→索引错→reference错 |
| [[FramePack - 固定上下文预算的长视频生成]] | 多尺度 latent | 时间/特征重要性排序 | 无 | 生成误差被压缩历史反复传播 |

若应用要求“曾看过的表面要恢复，但几何估计未必足够准”，VMem 是比 SPMem 更稳妥的选择；若没有相机/几何，FramePack 才能覆盖；若已有可靠相机且不想微调专门生成器，Context as Memory 更轻。
