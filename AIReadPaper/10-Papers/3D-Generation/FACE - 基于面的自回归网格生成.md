---
type: paper-note
status: done
domain: 3D-Generation
paper: "FACE: A Face-based Autoregressive Representation for High-Fidelity and Efficient Mesh Generation"
year: 2026
arxiv: 2603.01515
doi: null
source: https://arxiv.org/abs/2603.01515
project: null
code: null
tags: [3D-Generation, Mesh, Autoregressive, ARAE, VecSet, Face-Tokenization, Image-to-Mesh, Latent-Diffusion]
rating: 4.0
created: 2026-08-25
updated: 2026-08-25
---

# FACE：基于「一面一 token」的高保真高效网格自回归表征

## 基本信息

- **论文全称**：FACE: A Face-based Autoregressive Representation for High-Fidelity and Efficient Mesh Generation
- **作者机构**：Hanxiao Wang、Weize Quan、Dong-Ming Yan（CASIA 中科院自动化所 / UCAS）、Yuan-Chen Guo、Ying-Tian Liu、Zi-Xin Zou、Ding Liang、Yan-Pei Cao（VAST）、Biao Zhang（KAUST）
- **会议/年份**：arXiv preprint（v2，2026-03-03 提交）
- **arXiv**：[2603.01515](https://arxiv.org/abs/2603.01515)
- **核心贡献**：把自回归网格生成的语义层级从「顶点坐标」提升到「三角面」，用「one-face-one-token」策略让序列长度缩短 9 倍、压缩比做到 0.11（此前最优 0.22），并用 VecSet 编码器 + 自回归 Face Decoder 的 ARAE 框架拿到重建 SOTA
- **领域**：3D-Generation（直接 mesh 生成 / 自回归自编码器 / 面级 token 化 / image-to-mesh）

## 一句话结论

FACE 直接回应了 MeshGPT 那一代「直接网格生成」的致命短板——把网格拍平成 9N 长的顶点坐标序列，自注意力的 $O(N^2)$ 代价让高分辨率网格生成不可行。它的解法是把语义层级**从「顶点」上移到「面」**：一个三角面（9 个坐标）整体压成 1 个 token，序列长度立减 9 倍、注意力开销理论下降 81×，压缩比 0.11 刷新 SOTA；再用 VecSet 编码器保证潜空间质量，配一个 latent diffusion 就能顺手做 image-to-mesh。本质是「用更聪明的 token 化，而不是更复杂的压缩算法」来破效率瓶颈。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

三角网格是工业标准的 3D 表达，但直接生成「高保真、拓扑连贯」的网格仍是图形学与视觉里的「圣杯」。近年的主流范式是**自回归网格生成**（MeshGPT 开创、MeshXL/PivotMesh 继承），把网格拍平成 1D 顶点坐标序列、逐 token 生成。这条路线避免了早期两阶段方法（PolyGen）的误差累积，却引入一个致命代价：transformer 自注意力的 $O(S^2)$ 复杂度，在 $S=9|F|$ 的长序列上直接爆炸，让「上千面」的高分辨率网格生成算力不可负担。

### 1.2 现有方法的瓶颈

后续大量工作试图「压缩序列」来缓解瓶颈，作者把它们归为三条技术路线：

1. **遍历/邻接优化**（EdgeRunner、MeshAnythingV2、MeshSilksong）：精细排序面以复用顶点。缺点：遍历算法脆弱、易破坏网格全局结构。
2. **token 化策略**（TreeMeshGPT 的「一顶点一 token」、BPT/Nautilus 的 block indexing、FreeMesh 的坐标合并）：改 token 表示。缺点：block 索引会让词汇表爆炸。
3. **架构改进**（Meshtron 的分层结构、iFlame 的混合注意力）：换架构换效率。

作者的核心判断是：这些方法都在**治标**（对付「序列长」这个症状），没触及病根——**语义层级错了**。在顶点坐标这种「低级」语义上操作，序列天然就长。此外，「面作 token」的想法在扩散模型里（PolyDiff、MeshCraft）被探索过，但扩散的非序列性质难以生成长度可变的输出、保证拓扑完整；而**在自回归框架里实现「面即 token」此前无人做到**——这正是 FACE 的定位。

### 1.3 本文核心贡献

1. **one-face-one-token**：把每个三角面（9 维坐标向量）当作一个统一 token，序列长度缩短 9 倍。
2. **压缩比 0.11**：相比此前最优（0.22）再翻倍，直接砍半序列长度。
3. **VecSet 编码器 + 自回归 Face Decoder**：端到端 ARAE，多数据集重建 SOTA（HD/CD 全面超越 TreeMeshGPT、BPT 等）。
4. **潜空间可迁移**：训练 latent diffusion（DiT）实现单图到网格生成，验证了 FACE 潜空间的通用性。
5. **可扩展性**：1.2B 大模型 + 1024 分辨率量化，进一步提精度，展示 scaling 潜力。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **任务**：建模条件概率 $p(M|P)$——给定输入形状 $P$（点云表示，含 XYZ + 法向），重建/生成高保真三角网格 $M$。
- **输入**：8192 个带法向的表面采样点（large 模型用 65536 点）。
- **输出**：三角网格 $M$，表示为有序面序列 $F = (f_1, f_2, \dots, f_N)$，每个面 $f_i = (v^0_i, v^1_i, v^2_i) \in \mathbb{R}^9$，坐标量化到 $[0,127]$（large 模型 $[0,1023]$）。
- **关键假设**：① 面序列的确定性排序（ZYX 字典序）足以支撑自回归建模，无需复杂图遍历；② 全局几何可由一个紧凑 VecSet 潜变量完整承载，供面级生成逐层交叉注意力查询。

---

## 3. 核心方法

### 3.1 Face-based 自回归表征：one-face-one-token

网格被确定为面序列 $F=(f_1,\dots,f_N)$。每个面由 3 顶点 9 坐标定义：

$$f_i = (v^0_i, v^1_i, v^2_i) \in \mathbb{R}^{9}$$

**面排序**：作者发现一个出人意料地简单而有效的确定性空间排序——按每个面「最小坐标顶点」的 ZYX 字典序排序（沿用 MeshGPT/MeshXL 的做法）。相比 BFS/DFS 图遍历（优化顶点局部性），ZYX 排序消除了大量系统复杂度却不损质量（§4.5 消融验证）。

**one-face-one-token 策略**：这是框架的基石。标准做法把每个面拆成 9 个坐标 token（序列长 $9|F|$）；FACE 把整个 9D 面向量当作**一个单元**，用轻量 MLP 投影成单个 $d_{model}$ 维 token。这一设计直接攻击 $O(S^2)$ 瓶颈——自注意力序列长度从 $9|F|$ 降到 $|F|$，理论计算量降 **81×**、flash attention 下内存约降 **9×**，而 Face Pooling / CausalMLP 只增加可忽略的线性 $O(N)$ 开销。

![FACE 端到端管线：VecSet 编码器压缩点云 → 自回归 Face Decoder 逐面生成](assets/FACE%20-%20基于面的自回归网格生成/figures/02-pipeline.png)

### 3.2 模型架构：非对称 ARAE

#### 3.2.1 Shape Encoder（VecSet）

编码器把点云 $P\in\mathbb{R}^{m\times3}$ 压成紧凑潜变量 VecSet $C\in\mathbb{R}^{k\times d_{latent}}$，采用 3DShape2VecSet 的成熟架构：

$$C' = \text{CrossAttn}(Q=\bar{Q},\ K=K_P,\ V=V_P)$$

$$C = \text{TransformerEncoder}_{L_E}(C')$$

先用 FPS 采样 $k=2048$ 个 query 点，经 cross-attention 聚合全点云信息，再用 $L_E=8$ 层标准 Transformer Encoder 精炼成最终 VecSet $C$（bottleneck 维度 64）。这个 VecSet 作为全局条件信号，逐层注入解码器。

#### 3.2.2 Autoregressive Face Decoder

解码器逐面生成 $F$，每步三步走：

1. **Face Embedding（Face Pooling）**：第 $i$ 步的前缀 ground-truth 面 $f_{i-1}$ 经轻量 MLP 投影为 token $t_{i-1} = \text{MLP}_{embed}(f_{i-1})$——这是「one-face-one-token」的架构实现。
2. **Transformer 处理**：$L_D=24$ 层，每层做两步注意力：

$$H'_l = \text{CausalSelfAttn}(H_l)$$

$$H_{l+1} = \text{CrossAttn}(Q=H'_l,\ K=C,\ V=C)$$

   因果自注意力捕捉已生成面序列的局部结构与连通性；交叉注意力以 VecSet 为 K/V，在**每一层**注入全局形状上下文，保证每个局部生成决策都被整体几何指导。第 $i$ 步输出一个潜在面向量 $h_i\in\mathbb{R}^{d_{model}}$。
3. **Face Decoding Head（CausalMLP）**：把 $h_i$ 解码回面的 9 个量化坐标。这里用 TreeMeshGPT 的 **CausalMLP** 头，在面内引入**第二级自回归**——第 $j$ 个坐标的预测不仅依赖 $h_i$，还依赖同面内已预测的前 $j-1$ 个坐标，形成「面级（transformer）→ 坐标级（CausalMLP）」的分层因果依赖，显著优于一次性并行预测 9 个坐标（消融 Tab.5）。

![FACE image-to-mesh 管线：DiT 扩散生成 VecSet → 冻结的 Face Decoder 解码为网格](assets/FACE%20-%20基于面的自回归网格生成/figures/03-image-to-mesh.png)

### 3.3 训练目标与效率分析

**端到端目标**：单一的重建损失——对每个面的 9 个量化坐标做交叉熵，全网平均：

$$L = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{9} \text{CrossEntropy}(L_{i,j}, c_{i,j})$$

其中 $L_{i,j}$ 是 CausalMLP 对第 $i$ 面第 $j$ 坐标输出的 logit 向量，$c_{i,j}$ 是 ground-truth 坐标 token index。**关键**：不像 MeshGPT 那样额外训一个 VAE 做坐标 token 化，FACE 的面表示在单一端到端循环里**直接为生成任务优化**。

**效率分析**：自注意力复杂度 $O(S^2)$，之前方法 $S=9|F|$，FACE 是 $S=|F|$，9× 序列缩减带来理论 81× 计算降低。Tab.1 定义压缩比 = 模型需处理的序列长度 /「每坐标一 token」基线，FACE 拿到 0.11，是此前最优（TreeMeshGPT / Mesh-Silksong 的 0.22）的 2 倍。

### 3.4 Image-to-Mesh（latent diffusion）

为验证潜空间质量，训练一个 **DiT**（350M 参数）在 FACE 潜空间做去噪，用预训练 **DINOv3** 提取的图像特征作条件，flow matching 目标。推理时冻结的 FACE 解码器把生成的 VecSet 逐面解码成网格（无需任何 finetune）——这强力证明 FACE 学到了通用、可迁移的 3D 形状表征。

---

## 4. 实验与结果

### 4.1 实现细节与评测设置

- **ARAE**：总 500M 参数，非对称设计（解码器大于编码器）；编码器 8 层 × 768 维，解码器 24 层 × 1024 维；输入 8192 点，VecSet 2048 token × 64 维；Objaverse 取面数 <4000 的 ~130K 网格；Muon 优化器（lr $6\times10^{-4}$、wd 0.1），100K steps，8×A100 80GB；推理 top-1 确定性采样。
- **DiT**：350M 参数，50K 网格 × 10 随机渲染，flow matching，400K steps，32×A100，Euler 100 步采样。
- **数据集**：Objaverse（500 测试样本）、Toys4K（900 样本）、Famous（100 个经典复杂形状，如 elk、Armadillo），三者均**不在训练集**内，考验泛化。
- **指标**：Hausdorff Distance（HD，最坏情况偏差）+ Chamfer Distance（CD，平均邻近度）。

### 4.2 压缩比（Tab.1）

| 方法 | MeshXL | MeshAnything | MeshGPT | PivotMesh | EdgeRunner | MeshAnything v2 | DeepMesh | Nautilus | BPT | Mesh-Silksong | TreeMeshGPT | **FACE** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 压缩比↓ | 1.00 | 1.00 | 0.67 | 0.67 | 0.47 | 0.46 | 0.28 | 0.27 | 0.26 | 0.22 | 0.22 | **0.11** |

FACE 的 0.11 相比 MeshGPT 的 0.67 是 6 倍提升，相比此前最优（0.22）再翻倍。

### 4.3 重建质量（Tab.2）

| 方法 | Objaverse HD↓/CD↓ | Toys4K HD↓/CD↓ | Famous HD↓/CD↓ |
| --- | --- | --- | --- |
| MeshAnything | 0.327 / 0.128 | 0.264 / 0.110 | 0.399 / 0.177 |
| MeshAnythingV2 | 0.265 / 0.089 | 0.238 / 0.090 | 0.488 / 0.203 |
| TreeMeshGPT | 0.183 / 0.055 | 0.133 / 0.047 | 0.226 / 0.064 |
| BPT | 0.126 / 0.043 | 0.091 / 0.037 | 0.143 / 0.061 |
| **FACE** | **0.090 / 0.041** | **0.067 / 0.033** | **0.077 / 0.049** |

FACE 在三个数据集上 HD/CD 全面领先。Toys4K 上 HD 0.067 比次优 BPT 的 0.091 低 26%；Famous 上对复杂、风格迥异的形状（与训练分布差异大）仍 SOTA，证明强泛化。定性上（Fig.4），baseline 常产生孔洞、残缺部件或过度平滑，FACE 输出干净表面、保留锐利特征，连多连通分量的复杂形状（如树）也最完整。

![FACE 在 Toys4K 上的重建定性对比](assets/FACE%20-%20基于面的自回归网格生成/figures/04-reconstruction.png)

### 4.4 Image-conditioned 生成

因 baseline 未开源权重，只与 EdgeRunner 项目页的官方结果定性对比（Fig.5）。FACE 细节更好、与输入图像对齐更准、拓扑连通性更佳；EdgeRunner 常生成断裂碎片与不完整结构。例如 Lego 人偶的手与肩（第二行）、鸟的眼部几何（第四行），FACE 都能重建出来而 baseline 失败。

![FACE 的 image-conditioned 生成（对比 EdgeRunner）](assets/FACE%20-%20基于面的自回归网格生成/figures/05-image-conditioned.png)

### 4.5 消融实验

**面排序（Tab.3）**：ZYX（HD 0.103/CD 0.047）和 ZYX-component（0.110/0.045）远优于图遍历 DFS（0.171/0.077）、BFS（0.728/0.528）。空间排序大幅碾压图遍历，且简单的 ZYX 已足够。

**VecSet query（Tab.4）**：降采样点云作 query（0.103/0.047）优于可学习 query（0.132/0.058），与 3DShape2VecSet 结论一致。

**坐标解码策略（Tab.5）**：CausalMLP（0.103/0.047）大幅优于 Parallel Decode（单 MLP 输出 9×|V|，0.426/0.239）和 Attention-based（0.132/0.064）——面内坐标的因果依赖是关键。

### 4.6 Scaling Up

训练 1.2B 大模型（Ours-large）：采样点提到 65536、坐标量化到 $[0,1023]$、内部 380K 高质量网格数据集。Fig.6 显示大模型在整体形状、细粒度细节与锐利特征上都更精准——证明 FACE 具备良好 scaling 特性。

---

## 5. 局限与评价

### 5.1 局限性（论文自述 + 个人观察）

1. **离散表征有上限**：即便 scale 到 1024 分辨率，离散坐标仍非无限，细节精度存在天花板。
2. **依赖点云输入**：极细或薄结构（如自行车辐条）可能采样不足，导致局部重建不完整。
3. **仍受自回归慢采样约束**：逐面 + 面内逐坐标的两级自回归，虽然序列短了，但 token 数（$N\times9$）与 MeshGPT 同级，采样速度提升主要来自注意力序列缩短，而非 token 总数减少。
4. **代码未开源**（截至笔记撰写）：image-conditioned 部分只能定性对比，复现门槛高。

### 5.2 个人判断与启示

FACE 的价值在于**点破了自回归网格生成的根本症结**：序列长不是压缩算法不够聪明，而是「在顶点坐标这个错误的语义层级上操作」。把 token 语义上移到「面」，用最朴素的方式（整体投影 + 字典序排序）就拿到了 6 倍于 MeshGPT 的压缩比，且不牺牲质量——这印证了「表征的语义层级」比「压缩的工程巧思」更根本。

它同时把两条技术线缝合了起来：**VecSet 编码器**（来自 3DShape2VecSet 系，本综述另一半的主角）提供高质量全局潜变量，**自回归 Face Decoder**（来自 MeshGPT 系）负责可变长度的拓扑生成——FACE 因此成为「神经场潜变量」与「直接网格生成」两条路线的交汇点。这也是它和 MeshGPT 最本质的差异：MeshGPT 的 token 化（RQ + 图卷积）是独立预训练的，而 FACE 把面表示直接融进端到端 ARAE 里联合优化。

### 5.3 复现成本与建议

- **成本**：ARAE 500M 参数、8×A100 100K steps；DiT 350M、32×A100 400K steps——规模中等，是「直接网格生成」路线里较亲民的工程实践。
- **建议**：代码未开源，复现需自建。关键实现点：① VecSet 编码器照搬 3DShape2VecSet；② Face Pooling 用单层 MLP；③ CausalMLP 头面内坐标自回归；④ Muon 优化器（对 3D 生成有稳定收益）。

---

## 相关链接

- 论文 PDF：[assets/sources/paper.pdf](assets/FACE%20-%20基于面的自回归网格生成/sources/paper.pdf)（10 页，34.4 MB）
- 溯源表：[assets/sources/sources.md](assets/FACE%20-%20基于面的自回归网格生成/sources/sources.md)
- arXiv：https://arxiv.org/abs/2603.01515
