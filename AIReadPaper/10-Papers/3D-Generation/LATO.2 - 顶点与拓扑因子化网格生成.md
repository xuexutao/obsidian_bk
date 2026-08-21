---
type: paper-note
status: done
domain: 3D-Generation
paper: LATO.2: Factorized 3D Mesh Generation with Vertex and Topology Flow
year: 2026
arxiv: 2607.10623
doi: null
source: https://arxiv.org/abs/2607.10623
project: null
code: https://github.com/LoHhhha/LATO.2
tags:
  - 3D-Generation
  - mesh-generation
  - flow-matching
  - topology-aware
created: 2026-08-21
updated: 2026-08-21
---

# LATO.2：将顶点几何与拓扑连接解耦的 3D Mesh 生成

## 基本信息

- **论文名称**：*LATO.2: Factorized 3D Mesh Generation with Vertex and Topology Flow*
- **作者**：Hang Long、Tianhao Zhao、Junkai Lin、Youjia Zhang、Huipeng Guo、Rendong Liang、Jiale Xu、Jozef Hladký、Matthias Nießner、Yuanming Hu、Wei Yang
- **机构**：华中科技大学、Meshy AI、慕尼黑工业大学等
- **时间**：2026 年 7 月（arXiv:2607.10623v2）
- **arXiv**：[2607.10623](https://arxiv.org/abs/2607.10623)
- **代码**：[LoHhhha/LATO.2](https://github.com/LoHhhha/LATO.2)
- **重要性评估**：★★★★★（5/5）

## 一句话结论

LATO.2 把显式三角网格生成拆成"先生成连续顶点、再以顶点为条件生成离散连接"的两阶段 Flow Matching，并由共享的粗体素 scaffold 锚定几何与拓扑两条流；这一因子化结构同时解决了联合潜空间下的顶点漂移与破面问题，并自然获得可控顶点数、分部高分辨率生成、网格拼接与拓扑自适应编辑等能力。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

3D 内容创作链路上的一个长期痛点是：**生成模型能否同时输出"几何正确"和"拓扑可用"的网格**。所谓"拓扑可用"，是指顶点在曲面上排布紧凑、邻接关系连续、没有自交与重复面，更接近美术师在 Maya / Blender 里手工建模的习惯，而不是 Marching Cubes 之类等值面提取的均匀但杂乱的结果。LATO.2 关注的是显式三角网格 $\mathcal{M} = (\mathbf{V},\mathbf{E})$ 的端到端条件生成，其中 $\mathbf{V}$ 是顶点集合、$\mathbf{E}$ 是边集合，二者性质截然不同：前者是连续量，后者是离散组合量。

工业 3D 资产不仅要求表面形状正确，还要求顶点分布紧凑、面连接连贯，才能可靠地用于绑定、动画变形、着色和高效渲染。基于 SDF、Voxel 或神经场的 3D 生成模型虽然几何上可用，却要经过 Marching Cubes 等提取，得到的网格往往面数过高、三角形排列混乱，缺少接近美术师建模习惯的拓扑。

### 1.2 现有方法的瓶颈

围绕这一目标，过去两年形成了三条主流路线，但都存在明显瓶颈：

1. **体素 / 隐式场 + Marching Cubes**：以 GET3D、EG3D、Shap-E、TRELLIS 等为代表，生成隐式表示后再抽取网格。结果是面数多、连接不规整，难以直接进入生产流程。
2. **自回归 Mesh 生成**：PolyGen、MeshGPT、MeshAnythingV2、BPT、DeepMesh、MeshRipple 等把网格压成 token 序列再用 Transformer 自回归生成。优势是拓扑可控，劣势是高分辨率网格会被序列化成长达数千甚至上万的 token 流，推理慢、错误累积、长程拓扑一致性难保持。
3. **联合潜变量 + Flow Matching**：LATO 把顶点位置和连接关系同时塞进一个 64³ 结构化潜空间，再用整流流学习分布。问题是两类变量性质不同——一个是空间坐标，一个是离散邻接矩阵——被迫共享一个潜空间使流学习复杂化，表现为顶点漂移和表面断裂。

LATO.2 论文对此给出一个关键判断：**"连接的离散性是把它与几何联合建模的产物。"** 一旦先独立解出顶点，连接关系就降为"已知点之间的两两关系"，自然可以用连续的逐顶点特征表达，再解码为成对边概率。

### 1.3 本文核心贡献

LATO.2 显式做了三件事：

1. 提出**分解式 Flow Matching 框架**：把网格生成分解为顶点流（V-Flow）和顶点条件下的连接流（T-Flow），两阶段锚定于同一个粗体素 scaffold。
2. 引入两条专用 VAE：**V-VAE** 通过稀疏细分解码器+亚体素偏移头把顶点精度从体素中心提到亚体素；**T-VAE** 通过邻接掩码自注意力把离散连接压成连续的逐顶点 16 维潜变量。
3. 在分解结构上挂载两个**生产导向能力**：分部高分辨率生成（part-wise generation，可输出数万面级网格）和拓扑自适应编辑（topology-adaptive editing，用户修改顶点后重跑 T-Flow 即可重生成连接），都不需要对顶点重新优化。

![](assets/LATO.2%20-%20顶点与拓扑因子化网格生成/fig1-overview.png)

> 图 1：论文 Figure 1。上方一行展示 V-Flow 在指定顶点数（V=500、V=2000）下生成顶点云后，T-Flow 把它们连接成 F=1274、F=4504 面的不同分辨率网格；下方一行给出 LATO.2 的三大应用形态——单体生成、多部件生成与拓扑自适应编辑（部件拼接 + 部件旋转）。  
> 来源：论文 Figure 1，第 1 页，https://arxiv.org/abs/2607.10623

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

**输入条件** $c$：单张图像或点云，用于指定目标形状。视觉条件在训练时通过 DINOv2 从网格随机视角渲染图提取，再以 token 形式注入到 V-Flow 的 cross-attention 里。

**输出**：显式三角网格
$$
\mathcal{M} = (\mathbf{V},\mathbf{E}),\quad \mathbf{V}=\{\mathbf{v}_i\}_{i=1}^{N}\subset\mathbb{R}^{3},\quad \mathbf{E}=\{e_{ij}\}_{i,j=1}^{N}
$$
其中 $N$ 是用户可控的顶点数，$e_{ij}\in\{0,1\}$ 是无向邻接矩阵。面集合由边集合经环检测（loop detection）恢复。

**额外条件**：用户可显式指定目标顶点数预算 $N$，模型将其编码为
$$
\mathbf{c}_{\mathrm{vn}} = \log_2 N
$$
作为条件输入。这一步非常关键——它把"顶点数"这个离散全局超参数变成了与连续 flow 兼容的条件信号。

**假设**：(1) 目标网格可由体素 scaffold 提供粗空间支持；(2) 顶点集与连接关系分别由独立的连续/可微模型学习；(3) 训练数据足够多样，使顶点-拓扑联合分布有良好覆盖（论文采用约 450K 真实资产 + 100K 程序化合成）。

### 2.2 关键符号和目标函数

整体训练目标是 V-VAE、V-Flow、T-VAE、T-Flow 四个模块的损失之和：

$$
\mathcal{L} = \underbrace{\mathcal{L}_v}_{\text{V-VAE 重建与正则}} + \underbrace{\mathcal{L}_{V\text{-Flow}}}_{\text{顶点流匹配}} + \underbrace{\mathcal{L}_t}_{\text{T-VAE 重建与正则}} + \underbrace{\mathcal{L}_{T\text{-Flow}}}_{\text{拓扑流匹配}}
$$

各损失内部组成详见第 3.8 节。这里的"目标函数"并非单一可微表达式，而是四个独立优化目标的总和——这也是把"先顶点后拓扑"这一因子化结构落到训练上最直接的体现：每个模块只对自己的局部目标负责，不需要在统一目标里做两难权衡。

---

## 3. 核心方法

### 3.1 总体框架

LATO.2 由**两个表示模块 + 两个生成模块 + 一个共享 scaffold** 构成，四个阶段都使用同一种几何锚定（粗体素 scaffold $\hat{\mathcal{S}}$）：

| 模块 | 作用 | 关键性质 |
|---|---|---|
| **V-VAE** | 把网格顶点压成结构化潜变量；解码时恢复亚体素精度顶点 | 编码器/解码器，独立于生成任务 |
| **T-VAE** | 把邻接关系压成逐顶点 16 维潜变量 | 通过邻接掩码注意力把连接信息"挤"进 per-vertex latent |
| **V-Flow** | 在 V-VAE 潜空间上做条件整流流 | 接受图像特征 + $\log_2 N$ 条件 |
| **T-Flow** | 在 T-VAE 潜空间上做条件整流流 | 以已生成/编辑的顶点为条件 |
| **Scaffold** $\hat{\mathcal{S}}$ | 粗稀疏体素，提供活动空间 | 由结构规划器从图像/点云生成 |

推理时执行三步：(1) 由条件生成粗 scaffold $\hat{\mathcal{S}}$；(2) V-Flow 在 $\hat{\mathcal{S}}$ 的活跃体素上采样顶点潜变量 $\mathbf{z}_{\mathbf{v}}$，解码为高精度顶点 $\hat{\mathbf{V}}$；(3) T-Flow 以 $\hat{\mathbf{V}}$ 和 $\hat{\mathcal{S}}$ 为条件采样拓扑潜变量 $\mathbf{z}_{\mathbf{t}}$，解码为成对边，再经环检测得到面。

![](assets/LATO.2%20-%20顶点与拓扑因子化网格生成/fig2-pipeline.png)

> 图 2：论文 Figure 2。上半部分：V-VAE 接收 voxelize 后的顶点位移场并解码出带 offset 的顶点；T-VAE 接收顶点 + 邻接掩码自注意力，把每顶点特征压成 latent 再通过 Connection Head 预测边。下半部分：V-Flow 在稀疏体素 + DINOv2 图像 token 上做整流流；T-Flow 在顶点 token + 3D RoPE + 3D CNN scaffold 特征上做整流流；最终由 T-VAE 解码器恢复连接。  
> 来源：论文 Figure 2，第 3 页，https://arxiv.org/abs/2607.10623

### 3.2 V-VAE：亚体素精度的顶点表示

直接回归顶点坐标不适定——同一形状的顶点集无序且基数可变。LATO.2 沿用 LATO 的 **Vertex Displacement Field (VDF)** 表示：

- 在网格表面均匀采样 $K$ 个点 $\mathcal{P}=\{\mathbf{p}_k\}_{k=1}^{K}$，论文 $K=819{,}200$；
- 每个点关联三个属性：位置 $\mathbf{p}_k$、表面法线 $\mathbf{n}_k$、以及位移向量 $\mathbf{d}_k$——后者指向该点所在面的某个顶点，**将"该点属于哪个顶点"作为监督信号**。

**编码器** $\mathcal{E}_{\mathbf{v}}$：用 PointNet 嵌入点特征→在 $1024^3$ 稀疏体素上栅格化→稀疏 3D 卷积下采样→稀疏 Transformer 注意力压缩，得到 $64^3$、32 通道的稀疏结构化潜变量 $\mathbf{z}_{\mathbf{v}}$。这里关键设计是"高分辨率聚合 + 注意力下采样"——细粒度证据必须在压缩之前被捕获，否则就会丢失（消融实验验证：移除 VDF 下采样路径后 CD(L1) 立刻从 0.0003 飙到 0.0034）。

**解码器** $\mathcal{D}_{\mathbf{v}}$：从 $\mathbf{z}_{\mathbf{v}}$ 出发做粗到细的 sparse subdivision。每一步都先用当前层级的占用预测剪掉不含顶点的体素，再通过稀疏 cross-attention 注入全局上下文。**最细层不仅输出占用体素中心 $\hat{\mathbf{v}}_i$，还预测一个浮点偏移 $\boldsymbol{\delta}_i$**：
$$
\mathbf{v}_i = \hat{\mathbf{v}}_i + \boldsymbol{\delta}_i
$$
这一步把顶点从离散体素中心移到亚体素精度位置，是 V-VAE 相对前代 LATO 的最大改进（CD(L1) 从 0.0038 降到 0.0003，约 92% 降幅）。

### 3.3 V-Flow：可控顶点数与分辨率的顶点流

V-Flow 在 V-VAE 的潜空间上做条件整流流（rectified flow），沿用 TRELLIS 的稀疏流 Transformer 结构：

- **结构规划器**先从图像/点云生成粗稀疏结构 $\hat{\mathcal{S}}$，再对 $\hat{\mathcal{S}}$ 的活跃体素 token 化；
- **整流流目标**：把采样时间 $\tau\in[0,1]$ 时的中间状态定义为 $\mathbf{z}_{\mathbf{v}}^\tau = (1-\tau)\mathbf{z}_{\mathbf{v}} + \tau\epsilon$，其中 $\epsilon\sim\mathcal{N}(0,\mathbf{I})$，网络预测速度场 $v_\theta(\mathbf{z}_{\mathbf{v}}^\tau,\tau,\mathbf{c}_{\mathrm{img}},\mathbf{c}_{\mathrm{vn}})$；
- **条件注入**：图像 token $\mathbf{c}_{\mathrm{img}}$ 通过 cross-attention 作为 K/V；$\mathbf{c}_{\mathrm{vn}}=\log_2 N$ 通过 MLP 嵌入后与时间步嵌入一起通过 adaLN-zero 注入。

整流流的好处是 ODE 求解路径接近直线，在 50 步左右就能取得不错效果，对高分辨率体素流更友好。

### 3.4 T-VAE：把离散连接变成条件 latent

T-VAE 的核心问题是**把无序、基数可变的邻接矩阵装进固定维度的逐顶点特征**。论文给出的解法很巧妙：

**编码器** $\mathcal{E}_{\mathbf{t}}$ 是 8 层自注意力 Transformer，但关键在**注意力掩码**——每个顶点 token 只能关注自身和真实相邻顶点：

$$
M_{ij} = \begin{cases}0, & e_{ij}=1 \text{ 或 } i=j \\ -\infty, & \text{otherwise}\end{cases},\quad
\mathrm{Attn}(i,j)=\mathrm{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt{d}}+M_{ij}\right)
$$

由于掩码是"是否相邻"信息进入编码器的**唯一通道**，网络必须把连接信息吸收进逐顶点 16 维 latent $\mathbf{z}_{\mathbf{t}}$。这相当于把"图结构 → per-vertex 特征"的图编码问题约束到一个超窄瓶颈里。

**解码器** $\mathcal{D}_{\mathbf{t}}$ 是普通 Transformer（无掩码自注意力），对任意顶点对用对称 pairwise MLP 评估边概率：

$$
\hat{e}_{ij} = \sigma\!\left[\mathrm{MLP}(\mathbf{h}_i\oplus\mathbf{h}_j) + \mathrm{MLP}(\mathbf{h}_j\oplus\mathbf{h}_i)\right]
$$

其中 $\oplus$ 为拼接；两个方向各算一次取平均，保证预测邻接矩阵对称。推理时取阈值得到边集合 $\hat{\mathbf{E}}$，再通过 loop detection 把边集恢复为三角面。

**候选集训练**：对全部 $N^2$ 对计算损失不可行。训练时使用一个候选集 $\mathcal{C}$：(a) 正样本——每个顶点的真实邻居；(b) 难负样本——top-K 最近邻（通常是 8–16 个）；(c) 易负样本——均匀随机采样对。三类样本合在一起用非对称 focal loss 监督。这一设计让解码器既学到"该连的连"，又学到"该断的断"。

### 3.5 T-Flow：顶点条件下的拓扑流

T-Flow 的输入是已生成/编辑后的顶点集 $\hat{\mathbf{V}}$，输出是 T-VAE 潜空间中的逐顶点 latent $\mathbf{z}_{\mathbf{t}}$。为了让模型既能做无条件采样（用于编辑已有网格），也能做条件采样（用于引导新生成），论文同时训练两种条件方式：

- **顶点位置**：直接通过 token 进入 T-Flow，用 3D RoPE 编码；
- **粗几何上下文** $\mathbf{c}_{\mathbf{g}}=\text{3DCNN}(\hat{\mathcal{S}})$：把 scaffold 用 3D CNN 编码后送入 cross-attention，提供局部邻域之外的全局形状信息；
- 训练时按一定概率**随机丢弃** $\mathbf{c}_{\mathbf{g}}$，使模型既能做条件生成也能做无条件生成；
- 对注意力块的 Q、K 应用 **RMS 归一化**，稳定不同顶点数（$\sim$500 到 $\sim$4000）下的注意力。

整流流训练目标与 V-Flow 同构：
$$
\mathcal{L}_{T\text{-Flow}} = \mathbb{E}_{\mathbf{z}_{\mathbf{t}},\,\tau,\,\epsilon,\,\mathbf{c}_{\mathbf{g}}}\!\left[\left\|v_\phi(\mathbf{z}_{\mathbf{t}}^\tau,\tau,\mathbf{c}_{\mathbf{g}})-(\epsilon-\mathbf{z}_{\mathbf{t}})\right\|_2^2\right]
$$

消融实验显示，**移除粗几何条件**后 HD 从 0.0657 上升到 0.0694、$|\mathrm{NC}|$ 从 0.8341 下降到 0.8238——说明仅有顶点位置不足以决定全局一致的拓扑，必须借助 scaffold 提供的远距上下文。

### 3.6 共享体素 scaffold 与分部生成

两个阶段都锚定在同一个 $\hat{\mathcal{S}}$ 上，它的作用有二：

- **空间约束**：V-Flow 只需在 $\hat{\mathcal{S}}$ 活跃体素上采样，避免在空区域浪费容量；
- **全局条件**：T-Flow 用 3D CNN 把 $\hat{\mathcal{S}}$ 编码为 $\mathbf{c}_{\mathbf{g}}$，让局部连接预测能参考远距形状。

**分部生成（part-wise generation）** 直接源自 scaffold 的可分性：当目标网格顶点数超出单体隐变量容量时，把粗 scaffold 切成多个 part，每个 part 单独归一化到完整 latent 空间跑一次 V-Flow，再把所有 part 顶点拼回原坐标。T-Flow 既可以一次性在所有顶点上跑（global topology），也可以分部生成连接后通过 topology-adaptive stitching 拼接。

![](assets/LATO.2%20-%20顶点与拓扑因子化网格生成/fig3-multipart-pipeline.png)

> 图 3：论文 Figure 3。结构规划器先在图像/点云生成的稀疏结构上加上 part 级 3D 包围框，得到 box-aware 结构；编码后与图像 token 做 cross-attention 融合；再由结构解码器预测 part-aware 稀疏结构；最后 LATO.2 对每个 part 单独跑 V-Flow，T-Flow 既可联合也可分部生成连接。  
> 来源：论文 Figure 3，第 4 页，https://arxiv.org/html/2607.10623v2

**（个人判断）** 分部生成看起来并不"省容量"——它只是把固定隐变量容量反复分配到局部区域，以多 part 的方式拼出更高分辨率。真正的硬件开销来自 V-Flow 在多个 part 上多次调用，但 T-Flow 的 vertex token 预算随顶点数线性增加，所以总体复杂度仍是 $O(N^2)$。对工业界来说，这种"以小博大的容量复用"对生成 10K–50K 顶点级的高质量资产是务实选择。

### 3.7 拓扑自适应编辑

**核心机制**：因为 T-Flow 以"已实现顶点"为条件，对顶点的任何修改（用户编辑或程序化重组）都可以通过重跑 T-Flow 传播到连接关系，而不必重新优化顶点。这等于把"修改顶点 → 自动出新拓扑"做成了一个原子操作。

**两种应用**：
- **Mesh stitching**：把多个网格/部件对齐后取顶点并集，T-Flow 在合并后的顶点集合上生成连接，自动跨越新界面产生连贯拓扑；
- **Part transformation**：局部旋转/平移时，未变换区域内部拓扑仍然有效，仅过渡区域（连接移动和未移动顶点的面）需要重生成。

![](assets/LATO.2%20-%20顶点与拓扑因子化网格生成/fig4-topology-editing.png)

> 图 4：论文 Figure 4。(a) Mesh stitching：把不同网格顶点并集后，T-Flow 在接合区域生成无缝连接（红色为新合成面）。(b) Part transformation：直接把部件旋转会拉出面拉伸与自相交（左侧红框），重跑 T-Flow 后得到无相交、拓扑连贯的网格（右侧）。  
> 来源：论文 Figure 4，第 5 页，https://arxiv.org/html/2607.10623v2

### 3.8 训练目标与损失函数

四个模块各用一组损失，全部以加权和组合。

**(1) V-VAE 总损失**
$$
\mathcal{L}_v = \sum_{r}\mathcal{L}^r_{\mathrm{asy}} + \mathcal{L}_{\mathrm{off}} + \mathcal{L}_{\mathrm{KL}}
$$
- $\mathcal{L}^r_{\mathrm{asy}}$：每个细分层级 $r$ 的**非对称 focal loss**，用于对抗"含顶点体素 vs 空体素"的严重类别不平衡（典型比例约 1:10000）。
- $\mathcal{L}_{\mathrm{off}}$：对亚体素偏移 $\boldsymbol{\delta}_i$ 的 MSE 损失。
- $\mathcal{L}_{\mathrm{KL}}$：latent 分布的正则项。

**(2) T-VAE 总损失**
$$
\mathcal{L}_t = \frac{1}{|\mathcal{C}|}\sum_{(i,j)\in\mathcal{C}}\mathcal{L}_{\mathrm{asy}}(\hat{e}_{ij},e_{ij}) + \mathcal{L}_{\mathrm{KL}}
$$
在候选集 $\mathcal{C}$（正样本+难负样本+易负样本）上计算非对称 focal loss，并加 KL 正则。

**(3) V-Flow / T-Flow 整流流损失**
$$
\mathcal{L}_{V\text{-Flow}} = \mathbb{E}_{\mathbf{z}_{\mathbf{v}},\,\tau,\,\epsilon,\,\mathbf{c}_{\mathrm{img}},\,\mathbf{c}_{\mathrm{vn}}}\!\left[\left\|v_\theta(\mathbf{z}_{\mathbf{v}}^\tau,\tau,\mathbf{c}_{\mathrm{img}},\mathbf{c}_{\mathrm{vn}})-(\epsilon-\mathbf{z}_{\mathbf{v}})\right\|_2^2\right]
$$
$$
\mathcal{L}_{T\text{-Flow}} = \mathbb{E}_{\mathbf{z}_{\mathbf{t}},\,\tau,\,\epsilon,\,\mathbf{c}_{\mathbf{g}}}\!\left[\left\|v_\phi(\mathbf{z}_{\mathbf{t}}^\tau,\tau,\mathbf{c}_{\mathbf{g}})-(\epsilon-\mathbf{z}_{\mathbf{t}})\right\|_2^2\right]
$$
其中 $\epsilon\sim\mathcal{N}(0,\mathbf{I})$、$\mathbf{z}^\tau=(1-\tau)\mathbf{z}+\tau\epsilon$、$\tau\in[0,1]$。整流流的优势是 ODE 路径接近直线，推理步数可压缩到数十步。

### 3.9 推理流程与复杂度

完整推理路径如下：
1. **结构生成**：DINOv2 抽图像特征 + 条件结构规划器 → 粗稀疏结构 $\hat{\mathcal{S}}$。
2. **顶点生成**：V-Flow 在 $\hat{\mathcal{S}}$ 活跃体素上采样 $\mathbf{z}_{\mathbf{v}}$ → V-VAE 解码器粗到细还原 $\hat{\mathbf{V}}$；分部模式则在每 part 独立运行后拼回。
3. **拓扑生成**：T-Flow 以 $\hat{\mathbf{V}}$ 和 $\hat{\mathcal{S}}$ 为条件采样 $\mathbf{z}_{\mathbf{t}}$ → T-VAE 解码器评估所有顶点对得到边集 → 环检测得到面。

**复杂度**：(a) V-Flow 单次前向在 $64^3$ 潜空间上做 attention，复杂度与 $\hat{\mathcal{S}}$ 活跃体素数线性相关；(b) T-VAE 解码器默认评估全部 $N^2$ 对，复杂度 $O(N^2)$；(c) 顶点数越大，T-Flow 的 self-attention 也近似 $O(N^2)$。在论文 4K 顶点的设置下仍可处理，更大规模需要稀疏候选边选择。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

训练数据混合了**真实资产**和**程序化合成网格**：

| 来源 | 数量/角色 | 用途 |
|---|---|---|
| TRELLIS500K | 大规模真实 3D 资产 | 主要监督来源 |
| Objaverse / Objaverse-XL | 大规模真实 3D 资产 | 覆盖多类别 |
| 3D-FUTURE / Toys4K / ABO | 室内家具 / 玩具 / 商品 | 增加特定类别多样性 |
| **程序化合成网格** | 100K，由基本几何体（立方体、圆柱、圆锥、球体）经细分、变形、扭转、翘曲和组合而成 | **专门扩充顶点密度、曲率、连接模式覆盖** |

总计约 **450K 真实资产 + 100K 合成网格**。消融实验显示移除合成网格后稀疏占用 F1 从 0.9735 降到 0.9137，说明这部分数据对模型在不同拓扑模式上的泛化能力至关重要。

**数据预处理**：每网格均匀采样 $K=819{,}200$ 个表面点；每个点关联位置、法线、顶点位移向量。VDF 编码时先在 $1024^3$ 稀疏体素上聚合局部证据，再压缩为 $64^3$、32 通道的 sparse latent。

### 4.2 Baseline 与评价指标

**基线方法** 分为两类：
- **自回归 Mesh 生成器**：MeshAnythingV2、FastMesh、MeshSilkSong、BPT、DeepMesh、MeshRipple；
- **基于隐变量流的拓扑感知生成器**：LATO、MeshFlow。

**评价指标**：
- **CD（L1 / L2）**：预测面与参考面之间均匀采样点的平均双向几何差异（Chamfer Distance），L2 惩罚大偏差更重；
- **HD**：Hausdorff Distance，最大表面偏差；
- **|NC|**：法线一致性（Normal Consistency），衡量局部表面方向吻合度，越高越好；
- **顶点级重建**（V-VAE 评测）：在最细分辨率下评估稀疏占用预测的 Precision、Recall、F1、IoU。

### 4.3 实现细节

- **V-VAE**：PointNet 编码点特征→$1024^3$ 稀疏体素→稀疏 3D 卷积下采样→稀疏 Transformer（隐藏维度 512、8 注意力头），输出 $64^3$×32 通道；解码器粗到细上采样到 $1024^3$，包含占用剪枝细分、隐变量交叉注意力、亚体素偏移回归头。
- **V-Flow**：12 块条件流匹配 Transformer，适配自 TRELLIS；条件为 DINOv2 图像特征 + 顶点数条件。
- **拓扑建模**：顶点坐标离散化为 $K=1024$ bins；Fourier 特征嵌入，宽度 $d=768$；T-VAE 邻接掩码注意力 + 全顶点注意力，逐顶点 latent 维度 $d_z=16$；解码侧用拓扑堆栈 + 成对 MLP 分类器。
- **优化与硬件**：AdamW，学习率 $1\!\times\!10^{-4}$；8×NVIDIA H100 GPU。
- **训练时长**：V-VAE 4 天；T-VAE 1 天；V-Flow 7 天，有效批大小 256；T-Flow 2 天，按顶点 token 预算动态批处理。
- **参数量**（仅可训练参数，VAE 含编码器/解码器，不含冻结的条件编码器）：

| 模块 | 参数量 |
|---|---|
| V-VAE | ~320M |
| V-Flow | ~160M |
| T-VAE | ~180M |
| T-Flow | ~240M |

---

## 5. 实验结果

### 5.1 主要定量结果

#### 5.1.1 顶点级重建：V-VAE 对比（论文 Table 1）

| 方法 | 量化层级 | CD(L2)↓ | CD(L1)↓ | HD↓ |
|---|---|---:|---:|---:|
| MeshGPT | 128 | 0.0013 | 0.0185 | 0.0955 |
| PivotMesh | 128 | 0.0074 | 0.0395 | 0.2227 |
| MeshCraft | 256 | 0.0106 | 0.0524 | 0.2842 |
| LATO | 512 | 0.0000 | 0.0038 | 0.0083 |
| **LATO.2（V-VAE）** | **1024 + δ** | **0.0000** | **0.0003** | **0.0069** |

**说明**：从 LATO 的 $512^3$ 量化到 LATO.2 的 $1024^3$ + 浮点 offset，CD(L1) 下降约 92%、HD 下降约 17%。这一栏不参与下游比较，但它是后续 mesh 生成的上界——vertex 重建越准，CD/HD 越小。

#### 5.1.2 几何条件 Mesh 生成对比（论文 Table 2）

| 方法 | 类型 | CD(L2)↓ | CD(L1)↓ | HD↓ | \|NC\|↑ |
|---|---|---:|---:|---:|---:|
| MeshAnythingV2 | AR | 0.1083 | 0.1505 | 0.2318 | 0.6946 |
| FastMesh | AR | 0.0822 | 0.1163 | 0.1397 | 0.6939 |
| MeshSilkSong | AR | 0.0654 | 0.0937 | 0.1455 | 0.7759 |
| BPT | AR | 0.0603 | 0.0862 | 0.1067 | 0.8111 |
| DeepMesh | AR | 0.0529 | 0.0765 | 0.0946 | 0.8218 |
| MeshRipple | AR | 0.0458 | 0.0668 | 0.0941 | 0.8174 |
| MeshFlow | Flow | 0.0455 | 0.0668 | 0.0772 | 0.8227 |
| LATO | Flow | 0.0421 | 0.0617 | 0.0738 | 0.8262 |
| **LATO.2** | **Flow** | **0.0407** | **0.0596** | **0.0657** | **0.8341** |

**这张表说明了什么**：
- **整体趋势**：AR 方法在 CD 上普遍弱于 Flow 方法——这是因为 token 序列在长程上难保持几何一致性；Flow 方法在 latent 空间并行生成，CD 优势明显。
- **LATO.2 vs LATO**：CD(L2)/CD(L1)/HD 分别下降 3.3%/3.4%/11%，|NC| 提升约 1%。HD 降幅最大，说明亚体素 offset 主要消除了"几何上 OK 但点位置不对"的伪影。
- **法线一致性**：LATO.2 的 |NC| 比 LATO 略高，意味着面连接更"贴"曲面；这与因子化后 T-Flow 能更专注地学习连接模式有关。
- **绝对水平**：CD(L1)=0.0596、HD=0.0657、|NC|=0.8341 在显式 Mesh 生成里是目前的 SOTA 水平，但仍远未达到"接近 GT mesh"的水平——CD 数量级 $10^{-2}$ 表明面采样点的几何偏差仍有几个百分点。

#### 5.1.3 拓扑生成分解评估（论文 Table 3）

| 顶点来源 | 拓扑模块 | CD(L2)↓ | CD(L1)↓ | HD↓ | \|NC\|↑ |
|---|---|---:|---:|---:|---:|
| GT-Verts | T-VAE | 0.0351 | 0.0478 | 0.0859 | 0.8615 |
| GT-Verts | T-Flow | 0.0399 | 0.0543 | 0.1047 | 0.8199 |
| V-VAE | T-Flow | 0.0393 | 0.0538 | 0.0993 | 0.8201 |
| V-Flow | T-Flow | 0.0407 | 0.0570 | 0.1029 | 0.8051 |

**说明**：
- 第一行（GT-Verts + T-VAE）是上界——给定真值顶点，T-VAE 解码器本身的重建能力；
- 第二行（GT-Verts + T-Flow）显示**拓扑流模型相对确定性 T-VAE 解码器**仍有约 5–10% 的 CD 损失，但 |NC| 仅下降约 5%，说明流模型主要牺牲的是细粒度连接精度，整体拓扑结构仍在；
- 第三行（V-VAE + T-Flow）几乎与第二行持平，说明 T-Flow 对"顶点来源"不敏感——给定顶点集，无论是真值、V-VAE 解码还是 V-Flow 生成，T-Flow 都能给出可比质量的连接；
- 第四行（V-Flow + T-Flow）才是最终 LATO.2 端到端结果，与完整 Table 2 一致。

**这张表说明了什么**：LATO.2 的因子化结构确实把误差"按阶段归因"了——大部分端到端退化来自 V-Flow 的顶点分布而非 T-Flow 的连接预测。这对诊断模型问题非常友好：哪一行指标变差，就去修哪个模块。

### 5.2 定性结果

#### 5.2.1 几何条件生成的定性对比

![](assets/LATO.2%20-%20顶点与拓扑因子化网格生成/fig5-partwise.png)

> 图 5：论文 Figure 6（图中文件名沿用历史命名 fig5-partwise.png，内容为 Figure 6 定性对比）。在蝎子、人偶、双足角色、公仔等形状上，自回归方法（BPT、DeepMesh、FastMesh、MeshRipple）常出现断裂表面、缺失薄结构或不规则局部连接；隐变量流基线（MeshFlow、LATO）形状更连贯但仍有局部错误；LATO.2（最右列）在保持精细结构的同时面连接更干净。  
> 来源：论文 Figure 6，第 7 页，https://arxiv.org/html/2607.10623v2

#### 5.2.2 V-VAE / T-VAE 重建可视化

![](assets/LATO.2%20-%20顶点与拓扑因子化网格生成/fig7-vae-recon.png)

> 图 6：论文 Figure 7。(a) V-VAE 顶点重建可视化：灰色为正确重建、黄色为缺失 GT 顶点、红色为假阳性预测。(b) Offset head 效果：有 offset 时误差显著降低。(c) T-VAE 从学习到的拓扑 latent 完美重建原网格连接。  
> 来源：论文 Figure 7，第 7 页，https://arxiv.org/html/2607.10623v2

#### 5.2.3 顶点数可控生成

![](assets/LATO.2%20-%20顶点与拓扑因子化网格生成/fig9-vertex-control.png)

> 图 7：论文 Figure 9。给定相同结构体素，把目标顶点预算从 0.2K 逐步提到 4K，输出网格越来越稠密、细节更细，但物体级几何保持一致。T-Flow 自动适配连接关系到新生成的顶点集合。  
> 来源：论文 Figure 9，第 8 页，https://arxiv.org/html/2607.10623v2

### 5.3 消融实验

#### 5.3.1 V-VAE 消融（论文 Table 4）

| 配置 | CD(L1)↓ | HD↓ | ACC↑ | F1↑ | Recall↑ | IOU↑ |
|---|---:|---:|---:|---:|---:|---:|
| w/o 合成网格训练 | 0.0005 | 0.0071 | 0.9134 | 0.9137 | 0.9140 | 0.9034 |
| w/o OffsetHead | 0.0011 | 0.0082 | 0.9752 | 0.9735 | 0.9730 | 0.9603 |
| w/o VDF 下采样 | 0.0034 | 0.0145 | 0.8981 | 0.9025 | 0.9069 | 0.8873 |
| **Full V-VAE** | **0.0003** | **0.0069** | **0.9752** | **0.9735** | **0.9730** | **0.9603** |

**每个配置说明了什么**：
- **w/o OffsetHead**：CD(L1) 从 0.0003 升到 0.0011（约 3.7×），但 ACC/F1 完全不变。证明 OffsetHead **专门负责连续顶点定位精度**，对"哪一格有顶点"这种离散判定不贡献——这是非常干净的模块化设计。
- **w/o VDF 下采样**：CD(L1) 飙升到 0.0034（>10×），F1 也从 0.9735 跌到 0.9025。说明**"先在高分辨率上聚合证据，再压缩"**是 V-VAE 性能的核心——直接在低分辨率上做占用判定会丢失大量局部信息。
- **w/o 合成网格训练**：F1 从 0.9735 降到 0.9137（约 6 个百分点），CD(L1) 退化相对小。说明合成网格主要提升**拓扑模式覆盖**而非"逐点精度"，是 LATO.2 能在不同几何/连接类型上稳定工作的重要数据补充。

#### 5.3.2 T-Flow 消融（论文 Table 5）

| 配置 | CD(L2)↓ | CD(L1)↓ | HD↓ | \|NC\|↑ |
|---|---:|---:|---:|---:|
| w/o 几何条件 $\mathbf{c}_{\mathbf{g}}$ | 0.0411 | 0.0601 | 0.0694 | 0.8238 |
| **Full T-Flow** | **0.0407** | **0.0596** | **0.0657** | **0.8341** |

**说明**：
- 移除 $\mathbf{c}_{\mathbf{g}}$ 后 HD 增加 5.6%，|NC| 下降 1.2%，证明 scaffold 提供的粗几何上下文对**远距连接一致性**很关键——仅有局部顶点位置不足以决定一个"洞"该不该被填。
- 这一栏也间接说明"流模型 vs 确定性解码"的差距在缩小，T-Flow 引入的额外随机性不至于显著伤害质量。

### 5.4 泛化、效率与失败案例

**泛化能力**：
- 同一套权重在图像和点云两种条件下训练，推理时统一使用 V-Flow；
- 顶点数条件 $\log_2 N$ 是连续函数，可在 $[0.2K,\,4K]$ 区间内任意指定；
- Part-wise 生成让单次模型可输出 10K+ 顶点网格，无需重新训练。

**效率**：推理默认对全部顶点对做边概率评估，复杂度 $O(N^2)$。在论文 4K 顶点以内仍可处理；超过后需要稀疏候选选择（论文未披露具体方案）。

**失败案例**：**论文未提供具体的失败案例可视化或描述**，这是一个需要后续工作补齐的缺口。**（个人判断）** 从方法上推测可能的失败模式包括：(a) 极端细长结构（机械臂、树枝）的连接错乱；(b) 高亏格（多洞）形状在 T-Flow 上的拓扑判定混乱；(c) part-wise 拼接时跨 part 界面的接缝瑕疵。

---

## 6. 与相关工作的关系

LATO.2 的定位可从三条线索看清。

**与"3D 形状生成大背景"的关系**：2D 扩散先验的 SDS 类方法（DreamFusion、ProlificDreamer）、LRM 类前馈重建器、3D 原生隐变量（Shap-E）、结构化潜变量（TRELLIS）都能产生可用的 3D 形状，但都把网格当作后处理步骤，等值面提取得到的网格稠密不规则。LATO.2 显式以"输出 mesh"为目标，是这一大类里少数端到端输出可生产拓扑的工作。

**与"自回归 Mesh 生成"的关系**：PolyGen / MeshGPT 开创新的"mesh token 化 → 自回归"路线，后续 MeshAnythingV2、BPT、MeshSilkSong、TreeMeshGPT、MeshRipple 等通过更紧凑的序列化（相邻 token 化、EdgeBreaker 遍历、分块索引、树基/前沿感知扩展）来缩短序列。FastMesh、DeepMesh、Mesh-RFT、QuadGPT 等进一步探索确定性预测、强化微调和四边形生成。**LATO.2 的差异化在于放弃自回归的串行生成，改在结构化 latent 上做并行流匹配**，把"长序列错误累积"问题转化为"潜空间流学习问题"。

**与"基于 Flow 的显式 Mesh 生成"的关系**：DMesh/DMesh++ 用可微面存在概率 + Delaunay 候选复形；PolyDiff 在量化三角形汤上做离散 DDPM；MeshCraft 用 DiT 在连续面级 token 上做流；SpaceMesh 用连续顶点嵌入表示拓扑；MeshFlow 用边中心视图拼接顶点对特征；**LATO 用顶点位移场 + 联合潜空间**。**LATO.2 与上述所有工作的关键区别是显式把生成过程因子化为 V-Flow + T-Flow**：顶点几何通过局部稀疏细化解码到亚体素精度，拓扑建模为以生成顶点集为条件的关系潜变量分布。

**与并发工作 Nexus 的关系**：NeurIPS/ICLR 同期出现的 Nexus 同样采用"先顶点后拓扑"策略——顶点由粗到细的八叉树扩散生成，拓扑由图自编码 + SpaceTime 损失监督。LATO.2 偏结构化潜变量 + 整流流，Nexus 偏八叉树 + 离散图自编码；两者共同验证了"因子化"的趋势。

**（个人判断）** 站在 2026 年看，LATO.2 与 MeshFlow、PolyFlow、Mesh BDF 共同标记了一条**"为离散拓扑寻找连续生成空间"的转向**——以 LATO.2 为代表的方法把网格生成从"序列化"问题重新定位为"两个连续分布的条件采样"问题。这与 2D 图像生成从 DALL-E 1 的 VQ-VAE + AR 转向 Stable Diffusion 的 latent diffusion 几乎同构。

---

## 7. 局限与批判性评价

### 7.1 论文明确承认的局限

1. **顶点误差不可修正**：T-Flow 只能连接已有顶点，无法修正 V-Flow 已经生成错误的顶点位置。这意味着因子化结构的鲁棒性上限由 V-Flow 决定，T-Flow 不能"挽救"几何错误。
2. **二次复杂度**：拓扑解码默认对所有顶点对评估，复杂度 $O(N^2)$。在论文规模下仍可处理，但更大网格（数万顶点）需要稀疏候选边选择（论文未给出具体方案）。
3. **还没有完整资产**：目前只覆盖几何与连接，没有 UV、纹理、材质属性，距离"开箱即用的生产资产"还有距离。
4. **面向三角网格**：输出侧通过边环恢复三角面，没有处理高质量四边形 edge flow、UV seam、可绑定（rig-friendly）拓扑等更严格生产要求。

### 7.2 批判性评价

**论文的优势**：
- **建模分工清晰**：V-Flow 管连续几何、T-Flow 管离散连接，误差可隔离。Table 3 的四行组合直接给出"问题在哪一段"的可解释诊断。
- **生产导向能力内置**：part-wise generation 与 topology-adaptive editing 不是 paper-only demo，而是建立在同一接口（"对顶点重跑 T-Flow"）上，应用一致性强。
- **顶点数可控**：$\mathbf{c}_{\mathrm{vn}}=\log_2 N$ 让"分辨率"成为连续可调条件，比 AR 方法的固定 token 长度灵活。

**论文可以更进一步**：
- **缺乏失败案例**：这是评测论文的常见缺口。从方法推测，复杂拓扑、细长结构和跨 part 接缝是潜在风险点，但论文未给出可视化证据。
- **T-VAE 候选集训练依赖人工设计**：正/难负/易负三类样本的配比与 top-K 选取是超参数，论文未披露敏感性。
- **第二阶段仍要扫所有顶点对**：在 4K 顶点以内可处理，但 T-Flow 的 self-attention 已经是 $O(N^2)$，意味着整流流推理和拓扑解码两个环节都需要稀疏化才能扩展到 10K+ 顶点。
- **（个人判断）** "production-oriented"应谨慎理解：它确实**已具备生产导向的显式几何和编辑能力**，但**距离完整资产交付仍缺少** quad topology、UV、材质、严格流形保证和超大规模拓扑解码。

---

## 8. 复现与实践建议

**代码与权重**：官方仓库 [LoHhhha/LATO.2](https://github.com/LoHhhha/LATO.2) 公开了训练和推理脚本。复现的硬件门槛较高：

| 阶段 | 硬件 | 时长 | 备注 |
|---|---|---|---|
| V-VAE | 8×H100 | 4 天 | 819K 表面点 / 网格 |
| V-Flow | 8×H100 | 7 天 | 有效批大小 256 |
| T-VAE | 8×H100 | 1 天 | — |
| T-Flow | 8×H100 | 2 天 | 动态批处理 |

合计**约 14 天 8×H100**，对应 2688 GPU·小时。

**数据准备**：
- 需要批量从 Objaverse-XL / TRELLIS500K 等源下载并归一化网格；
- VDF 采样阶段（每个网格 819,200 点）本身是大头，I/O 与归一化建议做并行；
- 100K 程序化合成网格需要单独写生成脚本（论文未公开代码，**论文未披露**合成参数细节）。

**实践建议**：
- **推理优先分部生成 + 拓扑自适应编辑**：单 pass 的 V-Flow 容量有限，工业场景的复杂资产应走 part-wise。
- **注意 $\log_2 N$ 的离散性**：虽然论文把它当连续条件，但训练样本通常按对数分桶；推理时尽量取训练中见过的分桶中心点。
- **Offset head 必须保留**：消融显示它专攻连续精度，CD(L1) 直接 3.7× 退化，移除代价太大。
- **top-K 候选边选择**：生产部署若顶点数超过 4K，应在 T-VAE 解码器前增加几何近邻预筛选，把 $N^2$ 降到 $N\cdot K$。

---

## 9. 个人启发与后续问题

**启发 1：因子化是显式结构生成的统一钥匙**。LATO.2 把"几何 vs 拓扑"分到两阶段，每阶段都得到一个干净的生成目标。这与"图像 vs 文本"在多模态模型里被解耦为不同 tokenizer 是同一种思路：**当混合表示的各部分性质差距过大时，因子化比联合建模更易学习**。可以预期，未来的纹理网格、四边形网格、可绑定网格都可能在 LATO.2 的两阶段范式上扩展。

**启发 2：T-Flow 把"修改顶点 → 自动出新拓扑"做成了原子操作**。这是非常具有产品价值的设计——一个 3D 资产生成工具如果允许用户拖动顶点/部件，再自动重新生成连接，整个 DCC 工作流就被简化成"几何调整 + 单次重连接"。这是 LATO.2 在"原生 Mesh 生成"工作里少有的**功能化**而非**纯刷点**的能力。

**启发 3：误差隔离带来可解释评测**。Table 3 那种"GT-Verts + T-VAE"做上界、再逐步替换的评测范式，非常适合任何两阶段生成模型的调试。它告诉研究者"下一阶段是去优化 V-Flow 还是 T-Flow"。

**后续问题**：
1. **拓扑反馈能否反过来修正顶点**？论文局限中提到的"未来让拓扑反馈修正顶点"如果实现，可能形成 V-T-V-T 迭代细化范式，类似 image-to-image 中的 IP-Adapter + ControlNet 联合。
2. **能否扩展到四边形网格**？quad topology 是工业管线的硬要求，AG 元素从三角面改为四边形面需要重新设计 loop detection 和 T-VAE 的边概率定义。
3. **T-Flow 是否可以"多模态"**？比如同一组顶点上同时学习几个候选拓扑分布，让用户挑选，类似扩散模型的 guidance。
4. **（个人判断）** 真正决定 LATO 系列能否进入 DCC 流程的关键，不是 CD 再涨两个点，而是**它能否在 1–2 秒内完成高质量 4K 顶点网格的端到端生成**。目前 8×H100 的训练规模暗示推理时延并不理想。

---

## 10. 材料来源

### 10.1 论文原文

- arXiv 论文页：https://arxiv.org/abs/2607.10623
- arXiv HTML（v2）：https://arxiv.org/html/2607.10623v2
- arXiv PDF：https://arxiv.org/pdf/2607.10623v2

### 10.2 项目资源

- 官方代码：https://github.com/LoHhhha/LATO.2
- 前代 LATO 论文：[LATO: Latent Topological Optimization (arXiv 2506.03015)](https://arxiv.org/abs/2506.03015)，作为 VDF / 联合潜空间基线参照。

### 10.3 关键图源（本地）

| 本地文件 | 论文图号 | 来源 URL | 获取日期 | 用途 |
|---|---|---|---|---|
| `assets/LATO.2 - 顶点与拓扑因子化网格生成/fig1-overview.png` | Figure 1 | https://arxiv.org/html/2607.10623v2/image/teaser.png | 2026-08-20 | 整体框架与三大能力概览 |
| `assets/LATO.2 - 顶点与拓扑因子化网格生成/fig2-pipeline.png` | Figure 2 | https://arxiv.org/html/2607.10623v2/image/pipeline.png | 2026-08-20 | 核心方法 4 模块总览 |
| `assets/LATO.2 - 顶点与拓扑因子化网格生成/fig3-multipart-pipeline.png` | Figure 3 | https://arxiv.org/html/2607.10623v2/image/multi-part_gen_pipeline.png | 2026-08-21 | 分部生成 pipeline |
| `assets/LATO.2 - 顶点与拓扑因子化网格生成/fig4-topology-editing.png` | Figure 4 | https://arxiv.org/html/2607.10623v2/edit_res.png | 2026-08-21 | 拓扑自适应编辑 |
| `assets/LATO.2 - 顶点与拓扑因子化网格生成/fig5-partwise.png` | Figure 6 | https://arxiv.org/html/2607.10623v2/image/cmpv4.png | 2026-08-20 | 几何条件生成的定性对比 |
| `assets/LATO.2 - 顶点与拓扑因子化网格生成/fig7-vae-recon.png` | Figure 7 | https://arxiv.org/html/2607.10623v2/VAE-perf.png | 2026-08-21 | V-VAE / T-VAE 重建可视化 |
| `assets/LATO.2 - 顶点与拓扑因子化网格生成/fig9-vertex-control.png` | Figure 9 | https://arxiv.org/html/2607.10623v2/density-2.png | 2026-08-21 | 顶点数可控生成 |

### 10.4 检索与获取记录

- 论文原文通过 `WebFetch` 从 https://arxiv.org/html/2607.10623v2 获取正文与图注信息。
- 所有补充图片通过 `curl` 直接下载 arXiv 官方 HTML 路径；长边像素均 < 4000px，未触发 `sips -Z 2000` 缩放。
- 现有 3 张图片（fig1-overview.png、fig2-pipeline.png、fig5-partwise.png）按要求**未移动、未删除、未改名**。
