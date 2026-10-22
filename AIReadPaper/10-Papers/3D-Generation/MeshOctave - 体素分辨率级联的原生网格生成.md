# MeshOctave - 体素分辨率级联的原生网格生成

## 基本信息

- **论文**：MeshOctave: Vertex Split-and-Rewire Cascades for Native Mesh Generation（arXiv:2609.38985v1，2026-09-30）
- **作者**：Junkai Lin（林俊凯）¹²、Tianhao Zhao²、Hang Long¹²、Huipeng Guo¹、Jielei Zhang¹、Youjia Zhang¹²、Jiale Xu²、Wenbing Li¹²、Rendong Liang²、Jozef Hladký³、Matthias Nießner⁴、Yuanming Hu（胡渊鸣）²、Wei Yang（杨炜）¹†
- **机构**：¹华中科技大学、²Meshy AI、³独立研究员、⁴慕尼黑工业大学（TUM）；一作同时是 MeshRipple（CVPR 2026）的一作，工作在 Meshy AI 实习期间完成
- **项目页**：https://maymhappy.github.io/MeshOctave/
- **一句话**：把"尺度"定义为**二进制体素网格分辨率**而非网格简化步骤，用分辨率坍缩的逆操作「拆分-重连」（split-and-rewire）把 next-scale 生成范式搬进显式拓扑网格，配一个尺度条件化的 mask-uniform 离散扩散模型，在几何保真与拓扑有效性上同时超过 AR 与流匹配基线，10 个尺度约 90 秒生成 1 万面网格（快约 10 倍）
- **评级**：★★★★☆

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/01-teaser-cascading-resolution.png)

> Figure 1（p.1）：从单个 voxel 出发（Level 0），每级并行预测所有父面的"实例化子顶点 + 重连"，坐标分辨率逐级加倍，到 Level 10 输出艺术家风格网格。底部是生成样例。来源：arXiv:2609.38985。

---

## 1. 研究背景

### 1.1 原生网格生成的两条既有路线及其瓶颈

生成带显式拓扑的紧凑艺术家网格（artist-style mesh），目前两大范式各有硬伤：

- **自回归（AR）**（MeshGPT、MeshAnythingV2、BPT、DeepMesh、MeshRipple 等）：把网格序列化成顶点/面 token 逐个生成。保住了离散连通性，但推理延迟随面数线性增长、误差逐步累积，而且把人工排序强加在**置换不变**的图结构上。
- **连续扩散 / 流匹配**（MeshFlow、LATO、LATO.2、Meshy T2 等）：并行去噪网格元素，但连通性要么转成连续场再事后阈值化（预测稍有偏差就丢面/多面），要么把拓扑退化成"坐标恰好重合"（triangle-soup 离散扩散）。连通性只在**最终单一尺度**上解析——即使是 coarse-to-fine 架构，宏观几何与微观拓扑也要同时预测。

### 1.2 Next-scale 范式适配网格的三个门槛

VAR（Tian et al., 2024）的 next-scale 预测已成功用于图像、体素、八叉树、点云，但要搬到拓扑保持的网格上，粗化算子必须满足三条性质：

1. **全局并行**：粗化在整个网格上一次并行完成，尺度转换是单次 pass；
2. **规范性（canonicity）**：粗化是几何的内在确定函数，而不是启发式调度；
3. **序无关可解码**：逆细化能分解成彼此独立的局部决策，可并发解码。

已有的 coarse-to-fine 网格方法全部不合格：**VertexRegen** 反演 QEM 边坍缩（每次一个局部操作），**ARMesh** 从基点逐步扩张单纯复形——层级的形状反映的是**代价度量的平局裁决**而非几何本身，且细化严格串行，等于把 AR 延迟请回来了。

---

## 2. 核心方法一：Split-and-Rewire 级联

### 2.1 分辨率坍缩（Resolution Collapse）

把尺度直接定义在空间分辨率上。层级 $l$ 的网格 $\mathcal{M}^l=(V^l,F^l)$，顶点坐标量化到分辨率 $2^l$ 的均匀网格（$l=10$ 即 $1024^3$）。坍缩算子 $\mathcal{C}$ 把网格减半分辨率：

$$V^{l-1}\leftarrow\pi(V^l),\quad F^{l-1}\leftarrow\{(\pi(v_0^l),\pi(v_1^l),\pi(v_2^l))\mid\langle v_0^l,v_1^l,v_2^l\rangle\in F^l\}$$

其中 $\pi$ 把细顶点映射到其所属粗体素；面直接投影、去重，但**保留退化面**（投影后重合的三角形不丢弃）。三个关键性质：

- 合并共享同一体素的顶点、**继承其合并后的全部连通性**——这是网格版 vertex clustering（Rossignac & Borrel, 1993）的网格对齐形式；
- 整个网格一次并行 pass，层级链 $\mathcal{M}^L\xrightarrow{\mathcal{C}}\mathcal{M}^{L-1}\xrightarrow{\mathcal{C}}\cdots\xrightarrow{\mathcal{C}}\mathcal{M}^0$ 是**规范的**：由初始网格和空间网格完全决定，无任何启发式；
- 从单根 voxel 出发最多 $\log_2 L$ 级（$L$ 为最细网格边长），每级坐标精度每轴 +1 bit。

### 2.2 面中心的拆分-重连（Split-and-Rewire）

生成就是坍缩的逆过程 $\mathcal{S}$。分辨率加倍后，每个父顶点 $v^{l-1}$ 分裂出 $2^3=8$ 个八分体（octant）子位置。逆操作分解为每父面独立的两个决策：

- **Split（拆分）**：一个 8-bit 二值占据向量 $O(v^{l-1})\in\{0,1\}^8$，标记 8 个子体素中哪些真的实例化成子顶点；
- **Rewire（重连）**：任意两个父顶点的子顶点间连通性用 $8\times8$ 二值矩阵建模。对父面 $f^{l-1}=\langle v_0^{l-1},v_1^{l-1},v_2^{l-1}\rangle$：

$$M_{ij}(a,b)=\mathbb{1}\left[\langle\mathcal{O}_a(v_i^{l-1}),\mathcal{O}_b(v_j^{l-1})\rangle\in\mathcal{E}(\mathcal{M}^l)\right],\quad 0\le i\le j\le 2$$

$i=j$ 为**父内矩阵** $S_i$（同一父顶点的子顶点互连），$i<j$ 为**父间矩阵** $C_{ij}$（两个父顶点的子顶点互连）。

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/02-face-centric-split-and-rewire.png)

> Figure 2（p.4）：每个父面 = 3 个 8-bit 占据向量（split）+ 3 个父内 + 3 个父间 8×8 连接矩阵（rewire）。左图展示坍缩/逆坍缩的往复；中间红蓝二值占据向量区分"实例化子顶点"与"空子体素"。来源：arXiv:2609.38985。

### 2.3 九元结构 token 与三个工程细节

每个父面的逆操作完整表示为九元结构 token：

$$z_{f^{l-1}}=(O_0,O_1,O_2;\ S_0,S_1,S_2;\ C_{01},C_{02},C_{12})$$

训练时对每次坍缩记录 $z$；生成时从单 voxel 逐面执行 $\mathcal{S}$ 即可完全恢复 $\mathcal{M}^L$。附录里有三个容易被忽略但很重要的设计：

- **Octant-absolute 索引**：$S_i,C_{ij}$ 的行列按固定的 octant id $a\in\{0..7\}$ 索引，而不是按"实际存在的子顶点"压缩排序。矩阵永远是满 8×8，一个父顶点有 8 个子顶点还是 1 个子顶点，编码行为完全一致——不需要合并/重排/预算子顶点。
- **占据消歧连通性**：解码时 $S_i$ 的行列按 $O_i$ 掩码、$C_{ij}$ 的行按 $O_i\otimes O_j$ 掩码后再读出边。这保证"给空 octant 槽位预测出的连接字节"不会被误读成该槽位有顶点——占据与连通在表示上严格可分离。
- **退化面占位**：细特征占不满一个三角形时（悬边、孤立 octant），不丢弃而是解码成零面积退化面——悬边 $(A,B,B)$、孤立点 $(A,A,A)$，用与普通面相同的规则。这样 $\mathcal{S}$ 在下一级仍有祖先可细化，而不是静默抹掉当前分辨率还无法成三角的结构。

**这套表示的妙处**：尺度转换被建模为**无序集合**——每面的 token 彼此独立、无需序列化，序列长度随每级面数动态变化，天然支持动态长度网格、任意停止的 LOD（每级中间结果都是可用网格）。

---

## 3. 核心方法二：尺度自适应离散扩散

### 3.1 模型架构

给定粗网格 $\mathcal{M}^{l-1}$，生成器为每个父面预测九元结构 token $z$。$z$ 只含二值向量/矩阵，因此预测被建成**离散扩散**过程。

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/03-scale-conditioned-diffusion.png)

> Figure 3（p.5）：结构 token $z$ 与父面顶点坐标逐 slot 相加（$x_{f,k}=\text{Embed}(z_{f,k})+\text{Embed}(f[k])$）→ hourglass 编码（9→3→1 token/面）→ Diffusion Transformer 主干（Self/Cross Attention）→ hourglass 解码回 9 token/面 → 占据头（256-way 分类）+ 8 行 AR 解码器逐行出连接矩阵。$t$ 与尺度 $l$ 经 adaLN 调制，条件点云经 PC Encoder 注入。来源：arXiv:2609.38985。

输入融合：$z$ 的 9 个元素与父面坐标的 3 个顶点（每顶点 3 维）元素数一致，直接**逐 slot 相加**嵌入。主干是**对称 hourglass Transformer**（9/3/1 token/面），解码端：

- **占据头 $D_o$**：256-way 分类器——输出索引直接映射到 8×1 二值占据向量（$2^8=256$ 种组合一类）；
- **连接头 $D_c$**：轻量 MLP **自回归逐行**解码 8 行，$p_c(M\mid h_m)=\prod_{r=1}^8 p_c(m_r\mid m_{<r},h_m)$——每行条件于已解码行与隐藏态。扩散意义上整个矩阵仍是单个 slot，8 行同掩码/同揭示。

### 3.2 Mask-Uniform 双路噪声

标准 masked diffusion 的未掩码 token 永久固定，早期预测错误会**不可逆传播**到最终网格。沿用 TSSR 的思路，双路共享同一主干，用附加在点云特征上的 `[MODE]` flag 切换：

- **Mask 路**：损坏 slot 换成 `[MASK]`，只在被掩码 slot 上算损失（$\Omega=B$）——学会从部分 token 预测结构；
- **Uniform 路**：损坏 slot 换成该 slot 状态空间的**均匀随机值**，且**全序列算损失**（$\Omega=\{1,\dots,9n\}$）——损坏位置不标注，模型必须学会**自己发现并修正错误**。

噪声调度：每 slot 独立以 $r(t)=\cos(\pi t/2)$ 概率损坏（$t=0$ 全损坏、$t=1$ 干净），标准 $z_0$ 参数化直接预测干净 token：

$$L(\theta)=\mathbb{E}_{t,l,\tilde z}\Big[\sum_{(f,k)\in\Omega}-\log p_\theta(z^\star_{f,k}\mid\tilde z,F^{l-1},t,l,c)\Big]$$

两个配套技巧：

- **多类 focal loss** 替代交叉熵：连接矩阵约 92% 的行是空行，focal（$\gamma$ 指数、$\gamma=0$ 退化为 CE）自动降权已自信分类的 slot，防止空行主导梯度；
- **自生成噪声**：uniform 路训练时以概率 $p_{selfgen}$ 用"mask 路前向（无梯度）的贪心预测"替代均匀噪声——让修正路见过**真实分布的错误**而非合成噪声。

### 3.3 置信度头与 propose-and-refine 推理

Uniform 路额外训练**置信度头** $C_\phi$，预测模型自己每个 slot 的预测是否正确（$y_{f,k}=\mathbb{1}[\arg\max p_\theta=z^\star_{f,k}]$），作用在 $sg[h_{f,k}]$ 上（stop-gradient，不回传主干），少数错误类加权 $\kappa>1$：

$$L_{conf}=\frac{1}{9n}\sum_{f,k}\kappa_{f,k}\,\text{BCE}\big(C_\phi(sg[h_{f,k}]),y_{f,k}\big)$$

总损失 $L_{total}=sL_{mask}+(1-s)(L_{unif}+\lambda L_{conf})$，$s\sim\text{Bernoulli}(0.5)$。

推理是**propose-and-correct 循环**（附录 Algorithm 2）：全 `[MASK]` 画布起步 → mask 路提议全部 slot → uniform 路修正全序列并给置信度 → **不硬冻结**，按 $(1-conf_{f,k})\cdot sched(s)$ 概率随机 re-mask（$sched$ 线性退火 0.2→0.9）→ 循环 T=20 步。低置信 slot 更可能被重掩，让网络有机会**推翻早期错误**、后期逐步承诺稳定结构。这正好治了"masked diffusion 早期错误不可逆"的病。

### 3.4 顶点锚定 3D RoPE

与 TSSR 的 1D 序列索引不同，MeshOctave 用**无序的面级 token 块**保持块置换等变性，结构信息全部注入位置编码：

- 64d RoPE：30 个频率对给 3D 体素坐标（2 锚点 × 3 轴 × 5 slots）、2 个给面内角色 $\tau$（占据 / 父内 / 父间连接）；
- 每个 token 锚定到显式顶点对：顶点类 token $(v_a,v_a)$，连接类 token $(v_a,v_b)$——**顶点与边共享一半几何旋转角，使 vertex-edge 关联成为显式相位对齐**而非隐式推断（附录 Table 4）；
- 父顶点按 z-y-x 字典序排序 canonical 化共享边；token 降维（9→3→1）后粗 token 仍锚定全部三个父顶点。

### 3.5 实现细节（复现口径）

| 项 | 配置 |
|---|---|
| 主干 | 对称 hourglass Transformer，9/3/1 token/面，24 blocks（stage 深度 [2,4,6,6,4,2]），width 1024，16 heads，SwiGLU 2816，RMSNorm，**624M** |
| 调制 | adaLN-Zero，时间步 $t$ + 尺度 $l$ 嵌入求和；可变长 FlashAttention-2 |
| 条件 | Michelangelo 式点云编码器（8 层 / 768 宽 / 1024 latents，193M，联合微调），40960 点-法线对（50k 表面采样下采样），随机缩放 + $\sigma=0.01$ 高斯抖动（p=0.5） |
| 数据 | 350K 网格（Objaverse + Toys4K 过滤），≤15k 面 / 最细级 135k token |
| 训练 | 8×H800 80GB，bfloat16 + 梯度检查点，7 天 |
| 推理 | 每尺度 ~8s（20 扩散步），10 尺度共 ~90s |

---

## 4. 实验

### 4.1 定量对比

评测集：ObjaverseXL + Toys4K 选 300 网格 + 三个大规模 3D 生成模型（TRELLIS、Hunyuan3D、Direct3D 系）生成 200 个网格；每网格均匀采 4096 表面点。指标：CD-L1/L2、Hausdorff 距离（几何保真）、绝对法向一致性 |NC|（表面朝向与局部结构）。

| 指标 | 最强 AR（MeshRipple） | 最强 Flow（LATO.2） | **MeshOctave** |
|---|---|---|---|
| CD-L2 ↓ | 0.0505 | 0.0406 | **0.0392** |
| CD-L1 ↓ | 0.0729 | 0.0595 | **0.0581** |
| HD ↓ | 0.0957 | 0.0665 | **0.0645** |
| \|NC\| ↑ | 0.8187 | 0.8333 | **0.8478** |

对 12 个基线（8 AR + 2 next-scale AR + 3 flow）**全指标第一**。值得注意的是：flow 系普遍强于 AR 系（避免表面破碎、保证完整），而 MeshOctave 以离散扩散身份超过最强 flow 一个可见差距——说明"尺度级联 + 结构 token"比"单一尺度上解连通性"更有效，不是 diffusion 本身带来的。

### 4.2 推理延迟：核心卖点

- MeshOctave：每尺度 ~8s（20 步），10 尺度 ~90s，**几乎与面数无关**（token 并行，只有尺度数随分辨率对数增长）；
- ARMesh / VertexRegen：5k 面 ~15 分钟；10k 面时 33.3 / 40.3 分钟（超出训练范围，仅计时时长未验有效性）；
- 10k 面下 MeshOctave **1.6 分钟**，约 **20× 加速**（5k 面口径约 10×）。

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/11-inference-speed.png)

> Figure 11（p.17）：wall-clock 时间 vs 输出面数。两条基线曲线随面数线性爬升（且超出 5k 训练范围后网格有效性未验证），Ours 几乎水平（1.6 min @10k faces）。来源：arXiv:2609.38985。

### 4.3 定性与消融

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/04-qualitative-comparison.png)

> Figure 4（p.8）：AR 系（BPT/DeepMesh/FastMesh/MeshRipple）常见表面破碎断裂；flow 系（MeshFlow/LATO.2）倾向过平滑抹掉细节；Ours 保住精细结构与细小部件。来源：arXiv:2609.38985。

对最接近的两个 next-scale 基线（VertexRegen、ARMesh，无公开权重，作者在 200k 网格 <5k 面上按官方配置重训 5 天）：两者依赖逐元素串行细化，**误差复合导致严重几何畸变**；MeshOctave 整级并行解析，网格明显更完整、更水密。

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/05-generation-process.png)

> Figure 6（p.8）：左侧尺度间生成（Scale 2→10，从体素块到完整形状），右侧尺度内去噪（Diff. Step 2→20，从破碎碎片到连贯网格）。来源：arXiv:2609.38985。

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/07-ablation-table-qualitative.png)

> Table 3 + Figure 7（p.9）：四个变体各替换一个组件的定量结果与同一形状的定性对比。来源：arXiv:2609.38985。

消融四个变体（CD-L2 / HD / |NC|）：

| 变体 | CD-L2 ↓ | HD ↓ | \|NC\| ↑ | 结论 |
|---|---|---|---|---|
| **Full** | **0.0430** | **0.0739** | **0.8379** | — |
| Continuous 扩散 | 0.0512 | 0.0983 | 0.6685 | 放松离散约束 → **拓扑坍塌最重**，\|NC\| 掉 0.17 |
| w/o Uniform 路 | 0.0453 | 0.0884 | 0.7955 | 缺修正机制，早期承诺无法推翻 |
| w/o AR head | 0.0496 | 0.1005 | 0.7907 | 行独立预测 → 虚假连边 |
| 1D RoPE | 0.0469 | 0.0915 | 0.7911 | 丢显式 3D 坐标 → 局部几何邻近感知退化 |

**最刺眼的是 Continuous 那行**：把离散结构 token 松弛成连续量后 |NC| 从 0.838 崩到 0.669——这从反面证明了"结构 token 必须离散"这一设计判断，也顺带回答了"为什么不直接用 flow matching 出拓扑"。

### 4.4 应用：零重训的自适应网格细分

以粗网格表面采样的点云为条件，MeshOctave **不重训**即可执行 mesh subdivision。与 Loop、Neural Subdivision、SubdivAR 这类**均匀、固定拓扑**细分不同，它自适应预测子顶点占据与连通——可以改拓扑、把面集中到复杂区域。定性上比基线更接近 GT 的细节分布。

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/08-mesh-subdivision.png)

> Figure 8（p.9）：Coarse → Loop / NS / SubdivAR / Ours / GT。放大框内可见 Ours 在复杂区域（机械结构、塔尖）集中面数，简单区域保持稀疏。来源：arXiv:2609.38985。

---

## 5. 局限与评价

### 5.1 论文自述局限（Conclusion 原文承认两点）

1. **跨尺度仍是顺序的**：逐尺度生成的延迟高于单尺度 flow matching（90s vs flow 系的数秒级）；
2. **粗尺度拓扑错误会传播且无回溯**：一旦 Level 2 把拓扑定错，后面所有级别都在错误的祖先上细化——论文把"双向转换（bidirectional transitions）动态修正粗拓扑"列为未来工作。

### 5.2 我的补充评价

**强的方面**：
- "尺度 = 网格分辨率"这个重定义干净利落：粗化是确定性坍缩而非 QEM 平局裁决，规范、可逆、全局并行，三条性质一次满足。**octant-absolute 索引 + 占据/连通严格分离 + 退化面占位**这三个附录细节说明表示设计经过认真打磨，不是纸上框架；
- Table 1 的方法论定位（Raw + Masked diffusion + Next-scale 的唯一交叉点）诚实且准确；
- 实验覆盖了质量、速度、消融、应用四个维度，且延迟对比"每点都是实测且验证了网格有效"（对基线超范围部分明确标注"未验证"），口径干净。

**需要留意的**：
- **体素对齐的表示偏差**：所有顶点坐标被量化到二进制网格（每级 +1 bit），最细 1024³ 时量化误差约 1e-3 相对尺度——对毫米级 CAD 精度资产可能不够；顶点永远落在八叉树格点上，艺术家里常见的"故意偏移"顶点没法表达；
- **拓扑风格问题**：生成的连通性是"体素坍缩层级"的产物，是否真的等价于艺术家 retopology 的拓扑习语（edge loop、极点分布），论文只给了 |NC| 和目测，没有拓扑习语层面的指标；
- **条件仍是点云**：没有做 image/text 条件的端到端版本，点云来自上游生成模型时误差如何级联未讨论；
- **未开源**（截至 v1，项目页无代码），350K 数据过滤细节也待披露，复现成本高（8×H800 × 7 天 + 数据管线）。
- 跨尺度顺序性意味着生成分辨率上限受尺度数与逐尺度的错误累积双重约束；论文中间尺度可视化（Figure 10）里 Level 4-6 的形状仍明显破碎，实际可用 LOD 大概从 6-8 级起。

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/09-intra-scale-denoising.png)

> Figure 9（p.16）：尺度内去噪过程。Step0 预测碎片化、不连通；随迭代预测 + 重掩码，轮廓与局部连通逐步改善。来源：arXiv:2609.38985。

![](10-Papers/3D-Generation/assets/MeshOctave%20-%20体素分辨率级联的原生网格生成/figures/10-generation-across-resolutions.png)

> Figure 10（p.17）：跨分辨率的完整生成轨迹（Resolution 2 → 1024），每级都是可独立使用的 LOD。低分辨率下形状明显破碎，6-8 级后才结构完整。来源：arXiv:2609.38985。

---

## 6. 与既有笔记的关系

- 与 [[MeshGPT - 三角网格自回归生成]]、[[FACE - 基于面的自回归网格生成]]、[[QuadGPT - 原生四边形网格自回归生成]] 的 AR 路线相对：MeshOctave 是对"逐 token 序列化"的正面替代，可视为 [[三维网格生成表征演进 - Survey]] 里 next-scale 分支的最新节点；
- 与 [[LATO - 拓扑保持潜变量的网格流匹配]]、[[LATO.2 - 顶点与拓扑因子化网格生成]]（同厂 HUST 团队的 flow 路线）、[[TRELLIS - 结构化潜变量三维生成]]（其结构化潜变量思想与体素层级一脉相承，且测试集用其生成网格）、[[TRELLIS.2 - 原生几何与材质的紧凑三维潜变量生成]] 对照阅读：连续潜变量 vs 离散结构 token 之争，消融里 Continuous 变体的崩坏是直接证据；
- 应用方向 mesh subdivision 与 [[Quad-Mesh-文献全景梳理]] 中的细分/重拓扑议题衔接；对本大王持续关注的 **mesh wiring / topology 质量**议题，这篇给出了"生成端就带显式拓扑"的新范式参照。

---

## 7. 附件与溯源

| 文件 | 内容 | 来源 |
|---|---|---|
| `figures/01-teaser-cascading-resolution.png` | Figure 1 级联生成 teaser | arXiv:2609.38985 p.1，2026-10-22 裁剪 |
| `figures/02-face-centric-split-and-rewire.png` | Figure 2 面中心拆分-重连与 9 元 token | 同上 p.4 |
| `figures/03-scale-conditioned-diffusion.png` | Figure 3 尺度条件离散扩散模型 | 同上 p.5 |
| `figures/04-qualitative-comparison.png` | Figure 4 与 AR/flow 基线定性对比 | 同上 p.8 |
| `figures/05-generation-process.png` | Figure 6 尺度间生成 + 尺度内去噪 | 同上 p.8 |
| `figures/06-lod-baseline-comparison.png` | Figure 5 与 VertexRegen/ARMesh LOD 对比 | 同上 p.8 |
| `figures/07-ablation-table-qualitative.png` | Table 3 消融 + Figure 7 定性消融 | 同上 p.9 |
| `figures/08-mesh-subdivision.png` | Figure 8 网格细分应用 | 同上 p.9 |
| `figures/09-intra-scale-denoising.png` | Figure 9 尺度内去噪过程 | 同上 p.16 |
| `figures/10-generation-across-resolutions.png` | Figure 10 跨分辨率生成轨迹 | 同上 p.17 |
| `figures/11-inference-speed.png` | Figure 11 推理耗时曲线 | 同上 p.17 |

- 原始 PDF：`sources/meshoctave-2609.38985v1.pdf`（arXiv 官方，2026-10-22 下载）
