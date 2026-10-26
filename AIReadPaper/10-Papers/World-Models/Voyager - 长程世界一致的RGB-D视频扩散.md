---
type: paper-note
status: complete
domain: World-Models
paper: "Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation"
year: 2025
arxiv: "2506.04225"
doi: "10.48550/arXiv.2506.04225"
source: https://arxiv.org/abs/2506.04225
project: https://voyager-world.github.io
code: https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager
tags:
  - World-Models
  - Video-Generation
  - RGB-D
  - 3D-Cache
  - Camera-Control
  - Scene-Generation
created: 2026-10-26
updated: 2026-10-26
---

# Voyager：长程世界一致的 RGB-D 视频扩散

## 基本信息

- **正式标题**：*Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation*（开源名 HunyuanWorld-Voyager）
- **作者**：Tianyu Huang\*（哈工大）、Wangguandong Zheng\*（东南大学）、Tengfei Wang†（腾讯混元）、Yuhao Liu、Zhenwei Wang（港城大）、Junta Wu、Jie Jiang（腾讯混元）、Hui Li（哈工大）、Rynson W.H. Lau（港城大）、Wangmeng Zuo†（哈工大）、Chunchao Guo（腾讯混元）；\* 共同一作，† 通讯
- **版本**：arXiv v1，2025-06-04；正文 7 页 + 补充材料（附录 A–D 合并在同一 PDF，共 14 页）
- **开源**：2025-09-02 发布推理代码、权重（HuggingFace `tencent/HunyuanWorld-Voyager`）与 data engine
- **任务**：单张图像 + 文本 + 用户自定义相机轨迹 → 世界一致的 RGB-D 视频 → 直接得到点云 / 3DGS 场景，并可自回归无限扩展
- **基座**：HunyuanVideo-I2V（双流转单流 DiT + 3D-VAE，flow matching）
- **本地论文**：[paper.pdf](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/sources/paper.pdf)；来源清单见 [sources.md](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/sources.md)

## 一句话结论

Voyager 把「三维缓存渲染出来的残缺视图 → 视频扩散补全」这条 GEN3C / ViewCrafter 式路线做了一个关键升级：**条件里同时注入残缺 RGB 和残缺深度，输出端也同时生成对齐的 RGB 和深度**，再把生成帧反投影回带点剔除的世界缓存，形成「生成—重建—再生成」的闭环；它在 WorldScore 拿到当时最高的 77.62 平均分，并让生成结果无需 SfM/VGGT 即可直接初始化 3DGS。但论文对长程一致性、深度精度和误差累积几乎只有定性证据，单段 49 帧在单张 H20 上要约 32 分钟，本质上仍是离线的「可探索场景生成器」，而不是实时交互世界模型。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

游戏、VR、影视和机器人仿真都需要「用户能沿任意相机路径走进去」的三维场景。原生 3D 生成（TRELLIS、Hunyuan3D 等）擅长物体，但场景级 3D 数据极度稀缺；视频生成模型拥有海量 2D 先验，却天然缺少持久的三维结构。Voyager 想回答的问题是：**能否让视频扩散模型在给定单图和相机轨迹时，生成足够几何一致、可以直接变成 3D 的长程视频？**

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/01-teaser.png)

> 图 1：上半部分是「世界探索」，输入单图和相机轨迹，输出沿该轨迹的场景视频（第 15/30/45/60 帧）；下半部分是「世界重建」，模型同时生成 RGB 与对齐的深度，二者反投影即得右侧点云。这张图点出了论文的两个卖点：相机可控的长程漫游 + 免后处理的直接重建。来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2506.04225

### 1.2 现有方法的瓶颈

作者把 NVS / 视频式世界建模的问题归纳为三点，这三点也恰好对应论文的三个模块。

第一是**长程空间不一致**。以 CameraCtrl、MotionCtrl、SEVA 为代表的方法把相机参数作为隐式条件喂给网络，缺少显式三维锚定，一旦轨迹变长或回访旧区域，就会出现内容漂移。第二是**视觉幻觉**。GEN3C、FlexWorld、See3D、ViewCrafter 等方法改用「从点云渲染的残缺 RGB 图」做显式条件，一致性明显变好，但残缺 RGB 在复杂遮挡下会出现前后景穿插的错误像素——这些错误不仅误导推理，在训练时还会变成错误监督。第三是**事后重建**。上述方法只输出 RGB，要得到可用的 3D 还得再跑 SfM/MVS 或 VGGT，耗时且会引入几何伪影。

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/02-rgb-vs-depth-condition.png)

> 图 2：同一点云在不同新视角下渲染的残缺 RGB 与残缺深度。Novel View 2 里椅子背后的远处像素「透」了过来，RGB 上看起来像一团杂乱纹理，模型难以判断遮挡；而深度图上前景椅子和背景的数值差异一目了然。右侧 Output 对比显示，只用 RGB 条件会生成错误的遮挡关系，RGB-D 条件则正确。这是全文最核心的动机图。来源：论文 Figure 2，第 2 页。

### 1.3 本文核心贡献

1. **World-Consistent Video Diffusion**：在条件端加入残缺深度，在输出端联合生成 RGB 与深度（沿高度方向拼接，借 DiT 全注意力让两种模态逐像素交互），并用轻量 Control Block 强化对几何条件的跟随。作者声称这是第一个在给定相机轨迹下联合生成 RGB 与深度序列的视频模型。
2. **Long-Range World Exploration**：世界缓存 + 基于可见性与法线的点剔除（存储约省 40%），配合重叠段噪声初始化和平均再去噪的平滑采样，实现自回归扩展。
3. **Scalable Video Data Engine**：VGGT 估位姿 + MoGe 估深度 + 最小二乘对齐 + Metric3D 分位数定尺度，为任意视频自动标注度量尺度的相机与深度，整理出 10 万+ 训练片段。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

输入为首帧图像 $I_0\in\mathbb{R}^{3\times H\times W}$、文本提示 $y$ 与用户给定的相机轨迹 $\{c_k\}_{k=0}^{T-1}$。输出为与轨迹对齐的 RGB 帧 $\{I_k\}$ 和深度帧 $\{D_k\}$；因为相机是已知输入，二者可以直接反投影成世界坐标系下的点云序列，再作为 3DGS 的初始化。

论文隐含了几个重要假设：场景是**静态**的（世界缓存是点云，不建模运动物体）；深度是**度量尺度**的（训练数据统一到 Metric3D 尺度，因此相机平移量具有物理意义）；单次生成 49 帧，长程靠自回归拼接。

### 2.2 关键符号和目标函数

基座 HunyuanVideo 的 3D-VAE 把视频 $[I_0,\dots,I_{T-1}]$ 压成形状为 $(\frac{T}{c_t}+1)\times C\times\frac{H}{c_s}\times\frac{W}{c_s}$ 的潜变量 $\mathbf{z}_0$，其中 $c_t$、$c_s$ 是时间与空间压缩率。去噪网络是「先双流、后单流」的 DiT：前 $N_D$ 个双流块 $f_D^i$ 分别处理视频与文本 token，后 $N_S$ 个单流块 $f_S^i$ 处理拼接后的序列。训练目标是 flow matching 的速度回归：

$$\mathcal{L}=\mathbb{E}_{t,\mathbf{z}_0,\epsilon}\left\|\hat{\mathbf{u}}_t-\mathbf{u}_t\right\|^2,\quad\mathbf{u}_t=\frac{d\mathbf{z}_t}{dt}$$

推理时用一阶 Euler ODE 求解得到 $\mathbf{z}_0$，再经 3D-VAE 解码。Voyager 没有改变这一目标，所有创新都在「条件怎么构造、输出包含什么、长程怎么拼」上。

---

## 3. 核心方法

### 3.1 总体框架

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/03-method-pipeline.png)

> 图 3：Voyager 总览。左侧用户输入（图像 + 文本 + 相机）；对首帧估深度后反投影成点云，按每个目标相机渲染出残缺 RGB 与残缺深度（中上 Geometry-Injected）；这些条件经 Encoder 送入带 Control Block 的 DiT，输出 RGB-D 帧（右下）；生成帧再投影回右上角的 World Cache，作为下一轮生成的条件来源。来源：论文 Figure 3，第 4 页。

整个系统是一个闭环：**世界缓存 → 渲染残缺 RGB-D → 扩散补全成完整 RGB-D → 反投影更新缓存**。与 GEN3C 等工作的最大差别在于两个「D」：条件里有深度，输出里也有深度。前者解决遮挡幻觉，后者让「更新缓存」这一步不再依赖外部深度估计器，从而保证缓存和生成内容之间的自洽。

### 3.2 Geometry-Injected Frame Condition：把深度放进条件

已有方法用点云 $p\in\mathbb{R}^{N\times6}$（xyz + rgb）渲染出新视角的残缺图像 $\hat{I}_v$ 作为条件。Voyager 额外渲染一张与之对齐的**残缺深度图**。具体做法是：先为训练视频的每一帧估计深度 $D_k$ 和相机 $c_k$（来自第 4.1 节的数据引擎）；由于推理时只有首帧可见，训练条件也只用首帧构造——把 $D_0$ 按 $c_0$ 反投影得到点云 $p_0$，再对第 $k$ 帧计算可见性掩码：

$$M_k=\text{render}(p_0,c_k)$$

用 $M_k$ 遮挡掉 GT 帧中不可见的区域，就得到残缺图像 $\hat{I}_k$ 和残缺深度 $\hat{D}_k$。值得注意的是，训练条件是「GT 帧被掩码」而不是「点云重新渲染」，所以条件中的可见像素本身是干净的，避免了图 2 那种渲染穿透带来的错误监督——这一点论文没有明说，但从公式可以推出（个人解读）。

### 3.3 World-Consistent Video Diffusion：联合生成 RGB 与深度

#### 3.3.1 朴素做法及其问题

最常见的注入方式是把条件潜变量沿通道拼接，再用 patch embedding 投回 Transformer 维度：

$$\mathbf{z}'_{t,0}=f_{\text{emb}}(\text{concat}(\mathbf{z}_t,\mathbf{z}_{\text{rgb}},\mathbf{z}_{\text{depth}}))$$

之后依次经过双流块和单流块：

$$\mathbf{z}'_{t,i},\mathbf{z}'_{y,i}=f_D^i(\mathbf{z}'_{t,i-1},\mathbf{z}'_{y,i-1},t),\quad i=1,\dots,N_D$$

$$\mathbf{z}''_{t,i}=f_S^i(\mathbf{z}''_{t,i-1},t),\quad i=1,\dots,N_S$$

这种方式对预训练权重改动最小，但深度只是「输入侧的额外通道」，模型并不需要真正理解几何。作者指出，残缺区域可能从细小裂缝变化到大面积空白，单纯通道拼接难以应对这么大的变化范围。

#### 3.3.2 Depth-Fused Video Generation

Voyager 的关键选择是**让深度也成为输出**。具体是把 RGB 和深度沿高度方向拼成一张「上下图」：

$$\mathbf{I}_k=[I_k,\Phi,D_k]_h,\quad\hat{\mathbf{I}}_k=[\hat{I}_k,\Phi,\hat{D}_k]_h,\quad\mathbf{M}_k=[M_k,\Phi,M_k]_h$$

其中 $\Phi$ 是一行占位符，用来帮助模型区分上下两种内容。新的输入变为 $\mathbf{z}'_{t,0}=f_{\text{emb}}(\text{concat}(\mathbf{z}_t,\hat{\mathbf{z}}_0,m))$，$\hat{\mathbf{z}}_0$ 是条件视频的潜变量，$m$ 是掩码经 max-pooling 下采样后的版本（max-pooling 保证只要块内有一个像素可见就标为可见）。

这个设计的直觉在于：DiT 是全注意力的，RGB token 和深度 token 处在同一序列里，**每个 RGB patch 都能直接 attend 到同位置以及其他位置的深度 patch**，几何与外观在像素级交互，而不是通过通道混合的间接耦合。代价同样明显——视频 token 数近似翻倍，注意力计算量约为原来的 4 倍（个人估算，论文未给出）。另一个巧妙之处是：这一做法完全不需要修改 VAE，深度被当作一张灰度图像编码，复用了视频 VAE 的空间压缩能力。

#### 3.3.3 Context-Based Control Enhancement

通道拼接只在网络入口注入条件，越往深层条件信号越弱，表现为生成帧与条件图错位。作者参考 VideoPainter，复制 DiT 的**第一个双流块和第一个单流块**作为 Control Block $\hat{f}_D$、$\hat{f}_S$：

$$\mathbf{z}_D=\hat{f}_D(\mathbf{z}'_{t,0}),\quad\mathbf{z}_S=\hat{f}_S(\mathbf{z}_D)$$

$$\mathbf{z}'_{t,i}=\mathbf{z}'_{t,i}+l_D(\mathbf{z}_D),\quad\mathbf{z}''_{t,i}=\mathbf{z}''_{t,i}+l_S(\mathbf{z}_S)$$

$l_D$、$l_S$ 是零初始化的线性层，因此训练初期等价于原模型，不会破坏预训练能力。与 ControlNet 复制整个 encoder 不同，这里只复制两个块，然后把同一份控制特征加到**每一个**对应主干块上。作者的理由是「早期层特征保留更多上下文信息」，所以把它们广播到所有层可以增强像素级可控性。

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/04-diffusion-details.png)

> 图 4：扩散模型细节（附录 Figure 9）。输入由噪声潜变量、图像潜变量（时间维补零对齐）、条件潜变量和掩码四部分拼接；颜色区分三个训练阶段：Stage-1 只有 RGB 相关输入，Stage-2 加入深度（每张图的下半部分），Stage-3（粉色）只训练两个 Control Block，其输出通过 ⊕ 逐像素加回主干。来源：论文 Figure 9，第 11 页。

#### 3.3.4 三阶段训练

附录 A 给出了训练流程，这部分对理解各模块贡献很关键：

| 阶段 | 训练内容 | 数据 | 说明 |
|---|---|---|---|
| Stage 1 | 全量微调 DiT，只用 RGB 条件与 RGB 输出 | RealEstate + DL3DV + UE | 先学会「残缺 RGB → 完整视频」 |
| Stage 2 | 全量微调 DiT，加入深度条件与深度输出 | RealEstate + UE（去掉 DL3DV） | DL3DV 相机运动太快，不适合深度训练 |
| Stage 3 | **冻结 DiT**，只训练 Control Block | 仅 UE（有 GT 深度） | 用最干净的几何监督强化条件跟随 |

训练时宽高比从 $\{1,1.25,1.5,1.75\}$ 中随机采样以支持多比例输出，单次生成 49 帧。GPU 数量、迭代步数、学习率和总训练时长**论文均未披露**。

### 3.4 Long-Range World Exploration：世界缓存与平滑采样

#### 3.4.1 World Caching with Point Culling

自回归视频生成的通病是「只能看到最近几帧」，远处历史被遗忘。Voyager 的回答是把全部历史存成**点云**——它与时间长度解耦，任何历史区域只要相机再次看到，就能被重新渲染成条件。问题是点数会随帧数线性增长到百万级，而且多帧重叠区域会因为深度误差而叠出噪声。

作者设计了逐帧增量的剔除规则：给定已累积点云 $\hat{p}$ 和当前相机 $c_i$，先渲染可见性掩码 $M=\text{render}(\hat{p},c_i)$；

- 对于**不可见区域**，当前帧的新点直接加入缓存；
- 对于**可见区域**，若已有点的表面法线与当前视线方向夹角超过 $90^\circ$（即已有点是背对相机的、实际上从当前视角看不到正面），则也用新点更新；
- 其余可见区域保留旧点、丢弃新点。

这相当于一个非常轻量的「首次观测优先 + 背面替换」策略，没有做任何融合或 TSDF 式加权平均。论文报告存储点数下降约 40%，并能避免多帧聚合带来的噪声积累。

#### 3.4.2 Smooth Video Sampling

即使每段都与缓存几何一致，相邻段之间仍会出现**色彩不连续**（几何对了，但光照、色调可能跳变）。作者用两步解决：

1. 将轨迹划分为**重叠一半长度**的段；当前段的重叠区域用上一段的生成结果做噪声初始化，相当于给后一段一个「已知开头」；
2. 两段都完成后，对重叠区域取平均，再加入轻度噪声并做最后一轮去噪，消除拼接痕迹。

这是一种 SDEdit 式的拼接修复，简单有效，但代价是每段有一半的帧是重复计算的，且多了一轮额外去噪。

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/10-ablation-culling-sampling.png)

> 图 5：长程模块的定性消融（论文 Figure 8）。上排点剔除：(a) 存全部点 302,992 个，红框处出现重叠噪声；(b) 只存不可见区域的点 132,715 个，红框处出现空洞；(c) 加法线检查后 194,253 个，视觉效果接近 (a) 且无噪声，点数约为 (a) 的 64%。下排平滑采样：不做采样时第二段与第一段在房屋细节上不一致，加入平滑采样后过渡连贯。注意：该图图注与 Figure 7 完全相同，属于论文的复制错误。来源：论文 Figure 8，第 8 页。

### 3.5 Scalable Video Data Engine

#### 3.5.1 数据构成

| 来源 | 规模 | 特点 |
|---|---|---|
| RealEstate10K | 74,766 个片段 | 以室内房屋为主，兼有少量室外 |
| DL3DV-10K | 精选 3,000 个视频 → 约 18,000 个片段 | 原始数据大多相机运动快、抖动大 |
| Unreal Engine 渲染 | 1,500 个场景模型 → 10,000+ 个片段 | 提供 GT 度量深度，增加风格多样性 |
| 合计 | 100,000+ 个片段 | — |

#### 3.5.2 标注流程

RealEstate 和 DL3DV 都没有 GT 相机与深度。此前 GEN3C、FlexWorld 等工作用 DROID-SLAM 之类的稠密立体方法标注，几何一致性不足。Voyager 的流程是三步串联：

**第一步，VGGT 给出位姿与粗深度。** VGGT 的深度精度一般，但它与相机位姿是自洽的，这是后续对齐的锚点。

**第二步，MoGe 提供精细深度并对齐到 VGGT。** 把两者都转换为视差（disparity），在非天空区域 $\mathbf{M}$ 上用最小二乘求解尺度与偏移：

$$\min_{s,b}\left\|\mathbf{M}\cdot\left(\frac{s}{d_{\text{MoGe}}}+b-\frac{1}{d_{\text{VGGT}}}\right)\right\|^2$$

在视差空间对齐的好处是远处像素的权重更小，近景细节主导拟合，同时天空这种「无穷远」区域被掩掉。

**第三步，Metric3D 统一到度量尺度。** 由于 UE 数据是真实度量深度，所有数据需要对齐到同一尺度。作者用 Metric3D 的 20%–80% 分位数区间与 MoGe 的分位数区间之比作为尺度因子，同时缩放深度和相机平移：

$$s_{\text{metric}}=\frac{q(0.8,\mathbf{d}_{\text{Metric3D}})-q(0.2,\mathbf{d}_{\text{Metric3D}})}{q(0.8,\mathbf{d}_{\text{MoGe}})-q(0.2,\mathbf{d}_{\text{MoGe}})}$$

$$d_{\text{metric}}=s_{\text{metric}}\cdot d_{\text{MoGe}},\quad C^{\text{metric}}_{\text{cam}}=\begin{bmatrix}R&s_{\text{metric}}\cdot T\\0&1\end{bmatrix}$$

用分位数区间而非均值或中位数，是为了对离群点稳健；同时只缩放平移 $T$ 而保持旋转 $R$ 不变，保证了深度与位姿的几何自洽。

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/05-data-engine.png)

> 图 6：数据引擎总览（附录 Figure 10）。左侧 VGGT 提供相机参数与粗深度，右下 MoGe 提供相对深度并通过最小二乘对齐，右上 Metric3D 通过分位数尺度转换为度量深度。来源：论文 Figure 10，第 11 页。

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/06-warp-compare.png)

> 图 7：训练用 warp 条件图对比（附录 Figure 11）。FlexWorld 用 3DGS 渲染的深度做 warp，红框处出现大面积破碎和漂浮点；Voyager 数据引擎的深度更一致，warp 结果更干净。这直接决定了训练条件的质量。来源：论文 Figure 11，第 11 页。

### 3.6 推理流程与复杂度

推理时：对输入图估深度（Gradio Demo 中由深度估计器完成）→ 构造初始点云缓存 → 按用户轨迹渲染 49 帧残缺 RGB-D 条件 → 扩散生成 RGB-D → 反投影并按剔除规则更新缓存 → 用重叠半段的方式渲染下一段条件并继续生成。

论文正文没有给出任何推理时间。根据官方仓库 README：540p 单卡峰值显存 60GB（推荐 80GB）；512×768、49 帧、50 步在单张 H20 上需要 **1925 秒**，用 xDiT 序列并行在 8 张 H20 上为 288 秒（6.69× 加速）。这意味着生成一段约 2 秒的视频就需要半小时级单卡时间，Voyager 显然定位在**离线内容生产**。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

训练数据见 3.5 节。评测分为两类：一是 **RealEstate10K 测试集上随机抽取的 150 个片段**，用于视频生成和 3DGS 重建的有参考对比；由于测试集同样没有 GT 相机，评测相机和深度也是由 Voyager 自己的数据引擎估计得到的。二是 **WorldScore 静态基准**，包含 2,000 个跨室内外、写实与风格化的样例，每例给定输入图和相机轨迹，属于域外评测。

### 4.2 Baseline 与评价指标

视频生成的对比方法有四个开源相机可控模型：SEVA（直接以相机参数为条件），以及与 Voyager 同属点云条件路线的 ViewCrafter、See3D 和 FlexWorld。指标为 PSNR、SSIM 和 LPIPS。

场景重建对比中，baseline 只输出 RGB，因此统一用 VGGT 估计相机并初始化点云，再训练 3DGS；Voyager 则分别报告「用 VGGT 后处理」和「直接用自己生成的深度初始化」两种结果。

WorldScore 对比 6 个当时榜单前列的方法：两个 3D 方法 WonderJourney、WonderWorld，以及四个视频方法 EasyAnimate、Allegro、Gen-3、CogVideoX-I2V。

### 4.3 实现细节

基于 HunyuanVideo-I2V 微调，三阶段训练，单段 49 帧，多宽高比训练。官方推理默认 50 步、flow shift 7.0、embedded CFG 6.0，Control Block 可通过 `--use-context-block` 开关。训练算力、步数、batch size 均未披露。

---

## 5. 实验结果

### 5.1 主要定量结果

#### 5.1.1 RealEstate10K 视频生成

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| SEVA | 16.648 | 0.613 | 0.349 |
| ViewCrafter | 16.512 | 0.636 | 0.332 |
| See3D | 18.189 | 0.694 | 0.290 |
| FlexWorld | 18.278 | 0.693 | 0.281 |
| **Voyager** | **18.751** | **0.715** | **0.277** |

Voyager 三项全部第一，但相对最强的 FlexWorld 优势并不大（PSNR +0.47 dB，LPIPS 只差 0.004）。值得注意的是，同属点云条件路线的 See3D、FlexWorld 与 Voyager 处于同一梯队，而纯相机参数条件的 SEVA 明显落后，这说明「显式三维条件」本身是最大的收益来源，深度联合生成在此基础上带来的是增量提升。

#### 5.1.2 3DGS 场景重建

| 方法 | 后处理 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| SEVA | VGGT | 15.581 | 0.602 | 0.452 |
| ViewCrafter | VGGT | 16.161 | 0.628 | 0.440 |
| See3D | VGGT | 16.764 | 0.633 | 0.440 |
| FlexWorld | VGGT | 17.623 | 0.659 | 0.425 |
| Voyager | VGGT | 17.742 | 0.712 | 0.404 |
| **Voyager** | **无（自身深度）** | **18.035** | **0.714** | **0.381** |

这张表比上一张更有说服力。在同样经过 VGGT 后处理的条件下，Voyager 的 SSIM 领先 FlexWorld 0.053，说明生成视频本身的多视角一致性更好；而直接使用自己生成的深度初始化时，PSNR 再提升 0.29 dB、LPIPS 再降 0.023，证明生成的深度不仅「看起来像」，而且比 VGGT 从生成视频中反推的几何更适合作为 3DGS 初始化。附录 Figure 12 也从定性角度展示了 Voyager 初始化点云比 VGGT 更干净。

#### 5.1.3 WorldScore 世界生成

| 方法 | 平均 | 相机控制 | 物体控制 | 内容对齐 | 3D 一致 | 光度一致 | 风格一致 | 主观质量 |
|---|---|---|---|---|---|---|---|---|
| WonderJourney | 63.75 | 84.6 | 37.1 | 35.54 | 80.6 | 79.03 | 62.82 | 66.56 |
| WonderWorld | 72.69 | **92.98** | 51.76 | **71.25** | **86.87** | 85.56 | 70.57 | 49.81 |
| EasyAnimate | 52.85 | 26.72 | 54.5 | 50.76 | 67.29 | 47.35 | 73.05 | 50.31 |
| Allegro | 55.31 | 24.84 | 57.47 | 51.48 | 70.5 | 69.89 | 65.6 | 47.41 |
| Gen-3 | 60.71 | 29.47 | 62.92 | 50.49 | 68.31 | 87.09 | 62.82 | 63.85 |
| CogVideoX-I2V | 62.15 | 38.27 | 40.07 | 36.73 | 86.21 | **88.12** | 83.22 | 62.44 |
| **Voyager** | **77.62** | 85.95 | **66.92** | 68.92 | 81.56 | 85.99 | **84.89** | **71.09** |

Voyager 平均分 77.62 领先第二名 WonderWorld 近 5 分，但细看分项会发现它**并不是在每个几何维度上都最强**：相机控制（85.95 vs 92.98）、内容对齐和 3D 一致性（81.56 vs 86.87）都输给 3D 方法 WonderWorld，光度一致性也略逊于 CogVideoX 和 Gen-3。它真正的优势在于**均衡**——同时具备 3D 方法级别的相机控制和视频方法级别的主观质量与风格一致性，而 WonderWorld 的主观质量只有 49.81。作者还补充说明，由于条件是用度量深度构造的，Voyager 的相机实际移动幅度比其他方法更大，生成难度更高。这个解释合理，但论文没有给出量化证据。

### 5.2 定性结果

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/07-video-generation-comparison.png)

> 图 8：RealEstate10K 上的视频生成对比（论文 Figure 4）。第一行楼梯场景相机移动较大时，各 baseline 在红框处出现结构错误；最后一行浴室场景中，只有 Voyager 保留了输入图中架子上的物品细节，SEVA 甚至出现了大块崩坏。来源：论文 Figure 4，第 6 页。

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/08-3dgs-reconstruction.png)

> 图 9：3DGS 重建定性对比（论文 Figure 5）。最后一行吊灯案例最能说明问题：Voyager 保留了大部分吊灯细节，而 baseline 甚至无法重建出基本形状。正文在讨论表 2 时误写为「Figure 3」，实际应为本图。来源：论文 Figure 5，第 7 页。

### 5.3 消融实验

#### 5.3.1 世界一致视频扩散

作者把三个训练阶段的模型分别在 WorldScore 上评测：

| 设置 | 相机控制 | 内容对齐 | 3D 一致 |
|---|---|---|---|
| RGB-only（Stage 1） | 74.98 | 48.92 | 68.86 |
| RGB-D（Stage 2） | 85.04 | 65.72 | 78.58 |
| Full（Stage 3，+Control Block） | **85.95** | **68.92** | **81.56** |

**加入深度带来的收益是压倒性的**：相机控制 +10.06、内容对齐 +16.80、3D 一致性 +9.72。Control Block 则是锦上添花：相机控制只涨 0.91，但 3D 一致性再涨 2.98。这说明论文最核心的贡献确实是深度的引入，而不是 Control Block。

需要指出的是，这个消融把「条件加深度」和「输出加深度」两件事**捆在了一起**——RGB-D 模型同时改变了输入和输出，因此无法区分收益究竟来自残缺深度条件对遮挡的消歧，还是来自联合生成深度带来的几何正则。对理解方法机理而言，这是一个关键缺失（个人判断）。

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/09-ablation-rgbd-control.png)

> 图 10：三个训练阶段的定性对比（论文 Figure 7）。RGB-only 模型在相机进入未见区域时生成不一致内容（如 (d) 中浴室右侧出现黑色大块）；RGB-D 模型更贴合输入，但仍有小瑕疵（(a) 中多出吊篮、(d) 中多出毛巾）；完整模型结果最合理。来源：论文 Figure 7，第 8 页。

#### 5.3.2 长程生成

点剔除与平滑采样的消融（图 5）**只有定性结果**。点数从 302,992 降到 194,253，约节省 36%，与正文所说「约 40%」基本一致。平滑采样没有任何量化指标，例如段间色彩差异、FVD 或重访一致性。

### 5.4 泛化、效率与失败案例

![](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/figures/11-applications.png)

> 图 11：四类应用（论文 Figure 6）。(a) 三段相机轨迹完全不同的长程生成；(b) 图生 3D：在「特斯拉靠着帐篷」这种多物体组合上，Trellis 帐篷缺失、Rodin 未生成帐篷、Hunyuan3D v2.5 物体间距错误，Voyager 的侧视图中甚至能透过车窗看到帐篷；(c) 保留原深度条件、只替换参考图实现风格迁移（美式村庄 / 夜景）；(d) 作为单目视频深度估计器。来源：论文 Figure 6，第 8 页。

这些应用证明了 RGB-D 联合输出的通用性，尤其是 (c)「只换参考图、不换深度」的风格迁移非常巧妙，说明模型确实把几何和外观解耦到了不同的输入通路上。但需要清醒地看到，(b) 中 Voyager 输出的是多视角视频而非可编辑的 3D 资产，与 Trellis、Hunyuan3D 的比较并不在同一任务定义下；(d) 的深度估计也**没有任何定量评测**（如 Sintel、ScanNet 上的 AbsRel / δ1）。

效率方面，单段 49 帧单卡 H20 约 32 分钟，8 卡也需要近 5 分钟。论文没有展示失败案例，也没有讨论动态物体、强反光表面、天空等深度不可靠区域的表现。

---

## 6. 与相关工作的关系

从技术谱系上看，Voyager 位于「**显式 3D 缓存 + 视频扩散补全**」这条路线上，可以放在下面的坐标中理解：

| 维度 | 代表方法 | Voyager 的位置 |
|---|---|---|
| 相机条件方式 | CameraCtrl / SEVA（隐式参数） → ViewCrafter / GEN3C / See3D / FlexWorld（渲染 RGB） | 渲染 **RGB + 深度** |
| 输出模态 | 绝大多数只输出 RGB | **RGB + 深度**，可直接重建 |
| 长程记忆 | 滑窗帧（StreamingT2V）、帧检索（[[Context as Memory - 历史上下文驱动的场景记忆]]）、Surfel 索引（[[VMem - Surfel索引视图记忆]]） | 点云世界缓存 + 法线剔除 |
| 训练数据标注 | DROID-SLAM 等稠密 SLAM | VGGT + MoGe + Metric3D |

与 [[GEN3C - 三维感知的世界一致视频生成]] 相比，两者都使用 3D 缓存，但 GEN3C 只用 RGB 渲染作条件、只输出 RGB，缓存更新依赖外部深度估计器；Voyager 让模型自己输出深度，缓存更新不需要外部估计，闭环更自洽。

在腾讯混元的世界模型谱系中，Voyager 是 [[HunyuanWorld 1.0 - 分层可探索三维世界生成]] 的长程扩展组件——HunyuanWorld 1.0 用全景图生成初始世界，再借助 Voyager 生成 RGB-D 视频并维护点云缓存，扩展到初始全景之外的区域。后续 [[HYWorld 1.5 - 开放世界交互生成模型]] 转向动作条件的实时交互生成，[[HY-World 2.0 - 多模态三维世界生成重建与模拟]] 则进一步统一生成与重建。

数据引擎使用的三个基础模型分别对应 [[VGGT - 前馈式通用三维重建]]、[[MoGe - 开放域单目几何与仿射不变点图]] 和 [[Metric3D v2 - 规范相机空间的度量几何基础模型]]。稍晚的 [[Lyra 1.0 - 视频扩散自蒸馏三维重建]] 走的是另一条路：不让视频模型输出深度，而是把视频扩散的潜变量自蒸馏为 3DGS 解码器。之后的 [[WorldCrafter - 隐式三维记忆视频世界模型]] 则批评了显式深度记忆的在线成本，改用隐式 3D 记忆 token；[[Orbia - 视频世界模型三维一致性评测]] 提供了更严格的回访一致性评测，恰好补上了 Voyager 缺少的长程定量证据。

---

## 7. 局限与批判性评价

**论文自身承认或暗示的局限**：单次只能生成 49 帧，长程依赖自回归；DL3DV 这类快速运动视频不适合深度训练；场景是静态的。

**个人判断的主要问题**：

第一，**长程一致性缺乏定量证据**。论文标题强调 Long-Range 和 World-Consistent，但所有定量实验都是单段 49 帧，长程能力只有论文 Figure 6(a) 的三段示例和 Figure 8 的定性对比（本笔记图 11、图 5），没有回环 / 重访误差、累积漂移曲线或分钟级评测。

第二，**误差会通过缓存闭环累积**。生成的深度被反投影写入缓存，再作为后续段的条件；剔除策略只是「首次观测优先」，没有融合、置信度加权或全局 bundle adjustment。一旦某段深度有尺度或局部形变误差，就会被永久固化到世界中，并误导后续生成（个人推断）。

第三，**训练与推理条件分布不一致**。训练时条件只来自首帧点云对 GT 帧的掩码，可见区域是干净的；推理时条件来自多段生成结果累积出的缓存渲染，存在噪声、空洞和渲染穿透。论文没有做「用带噪缓存训练」或类似 Diffusion Forcing 的抗暴露偏差处理。

第四，**相机控制完全依赖缓存可见性**。Voyager 没有 Plücker 射线等显式相机编码，相机信息只通过「哪些像素可见、在哪里」间接传达。当相机大幅旋转进入完全未见区域时，条件几乎全空，相机运动的约束会变弱（个人推断，论文未讨论）。

第五，**评测存在自洽性偏差**。RealEstate 测试的「GT」相机和深度由 Voyager 自己的数据引擎估计，与训练标注同源，可能对 Voyager 有利；测试样本只有 150 个；深度质量本身从未被直接度量。

第六，**写作较仓促**。Figure 8 的图注是从 Figure 7 复制过来的；正文讨论表 2 时把 Figure 5 误写为 Figure 3；Control Block 在附录中又被称为 ControlNet blocks。这些都不影响结论，但说明 v1 版本的严谨度一般。

**适用边界**：静态、可由单目深度较好描述的场景（室内、街景、风格化场景）；离线内容生产而非实时交互；需要「生成即重建」的 3D 资产管线。

---

## 8. 复现与实践建议

| 项目 | 情况 |
|---|---|
| 推理代码与权重 | 已开源（GitHub + HuggingFace），含 Gradio Demo |
| Data Engine | 已开源（`data_engine/`） |
| 训练代码 | README 未提供训练脚本，暂未找到 |
| 训练数据 | RealEstate、DL3DV 公开；1,500 个 UE 场景渲染数据未公开 |
| 推理显存 | 540p 最低 60GB，推荐 80GB |
| 推理速度 | 512×768、49 帧、50 步：单张 H20 1925 s；8 张 H20 288 s |
| 训练算力 | 论文未披露；基座为 13B 级 HunyuanVideo，全量微调至少需要多机 80GB 卡 |

实践建议：

1. **只想用**：直接用官方 Gradio，上传图片并选择相机方向即可得到 RGB-D 视频；输出的深度可以直接喂给 3DGS 初始化，省掉 VGGT/COLMAP。
2. **想做数据**：data engine 本身就很有价值，VGGT 位姿 + MoGe 深度 + Metric3D 尺度的组合可以直接复用到其他需要度量尺度 RGB-D 视频标注的任务上。
3. **想改进**：最值得补的是长程定量评测（可用 Orbia 或 WorldScore 动态版）、缓存融合策略（置信度加权或 TSDF）、以及在带噪缓存条件下的训练。
4. **想加速**：可以考虑把 50 步蒸馏到少步（DMD / Self-Forcing 系列），或者只对深度分支使用更低分辨率以减少 token 数。

---

## 9. 个人启发与后续问题

Voyager 最值得记住的不是某个具体模块，而是一个朴素的判断：**视频生成模型想要成为「世界模型」，输出就不应该只有 RGB**。一旦深度成为输出，世界状态就可以被显式维护，生成与重建之间的边界就被打通了。后续的 [[WorldMirror - 多模态先验驱动的通用前馈三维重建]]、[[GAE - 三维一致世界生成的几何原生潜空间]]、[[AEON - 统一视频与三维世界模型]] 等工作，都可以看作是沿着「让生成模型的潜空间或输出原生具备几何」这个方向继续推进。

从我自己关注的 3DGS 与几何质量角度，有几个问题值得继续追：

- **能否直接输出 Gaussian 参数而非深度？** 深度只是 2.5D 表示，背面和遮挡区域仍需多次生成才能补全；如果输出端换成逐像素 Gaussian（类似 pixelSplat / AEON 的思路），缓存可以直接是 3DGS，并天然支持各向异性和不透明度。
- **高度拼接 vs 通道拼接 vs 独立 VAE**：把深度当灰度图塞进 RGB VAE 是否浪费了表示能力？专门的几何 VAE（如 GAE）是否会更好？
- **缓存的一致性维护**：点云缓存没有拓扑和语义，无法支持对象级编辑；如果把缓存换成分层 mesh 或分组的 3DGS，能否与 HunyuanWorld 1.0 的分层世界表示更好地结合？
- **相机控制的显式化**：Plücker 编码加缓存渲染双通路，是否能修复「大旋转进入未见区域时控制变弱」的问题？

**评级**：★★★★☆。思路清晰、工程完整、开源彻底，「RGB-D 联合生成 + 点云缓存闭环」是视频世界模型走向 3D 的一个重要范式节点；扣一星是因为长程与深度质量缺少定量证据，误差累积问题没有正面处理，且推理成本很高。

---

## 10. 材料来源

| 材料 | 链接 / 位置 |
|---|---|
| arXiv 论文页 | https://arxiv.org/abs/2506.04225 |
| 官方 PDF（含附录 A–D） | [sources/paper.pdf](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/sources/paper.pdf) |
| 项目页 | https://voyager-world.github.io |
| 官方代码 | https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager |
| 模型权重 | https://huggingface.co/tencent/HunyuanWorld-Voyager |
| 混元技术报告 PDF | https://3d-models.hunyuan.tencent.com/voyager/voyager_en/assets/HYWorld_Voyager.pdf |
| 图片与来源清单 | [sources.md](assets/Voyager%20-%20长程世界一致的RGB-D视频扩散/sources.md) |
