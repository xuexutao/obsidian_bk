# LATO.2：将顶点几何与拓扑连接解耦的 3D Mesh 生成

## 基本信息

- **论文名称**：*LATO.2: Factorized 3D Mesh Generation with Vertex and Topology Flow*
- **作者**：Hang Long、Tianhao Zhao、Junkai Lin、Youjia Zhang、Huipeng Guo、Rendong Liang、Jiale Xu、Jozef Hladký、Matthias Nießner、Yuanming Hu、Wei Yang
- **机构**：华中科技大学、Meshy AI、慕尼黑工业大学等
- **时间**：2026 年 7 月
- **arXiv**：[2607.10623](https://arxiv.org/abs/2607.10623)
- **代码**：[LoHhhha/LATO.2](https://github.com/LoHhhha/LATO.2)
- **原始文件**：[本地 PDF](file:///Users/eilonxtxue/Downloads/lota2.pdf)
- **重要性评估**：★★★★★（5/5）

---

## 1. 一句话总结

LATO.2 把原生三角网格生成拆成“先生成连续顶点、再根据这些顶点生成离散连接”的两阶段 Flow Matching，从根本上缓解了顶点漂移和破面问题，并自然获得可控顶点数、分部高分辨率生成、网格拼接及拓扑自适应编辑能力。

![](assets/LATO.2%20阅读总结/fig1-overview.png)

*图 1：论文 Figure 1。LATO.2 先通过 V-Flow 生成指定数量的顶点，再用 T-Flow 逐步恢复连接；同一因子化框架还支持多部件生成和编辑后自动重建拓扑。*

---

## 2. 研究问题与动机

工业 3D 资产不仅要求表面形状正确，还要求顶点分布紧凑、面连接连贯，才能可靠地用于绑定、动画变形、着色和高效渲染。基于 SDF、Voxel 或神经场的 3D 生成模型通常要经过 Marching Cubes 等等值面提取，得到的网格虽然几何上可用，却往往面数过高、三角形排列混乱，缺少接近美术师建模习惯的拓扑。

自回归 Mesh 模型能直接生成顶点和面，但高分辨率网格会被序列化成极长 token 流，推理慢且容易积累错误。新一代 Flow 方法尝试在连续 latent 中并行生成拓扑，不过此前通常把连续顶点位置和离散连接关系塞入同一个联合 latent；两类变量性质不同，强行耦合会增加流学习难度，表现为顶点漂移、断面和局部连接错误。

LATO.2 的核心判断是：**顶点回答“几何在哪里”，拓扑回答“这些顶点怎样连接”，二者相关但不应由同一个生成变量同时决定。** 因此模型先确定顶点，再把拓扑建模成以已实现顶点为条件的概率分布。

---

## 3. 核心方法

### 3.1 总体架构：V-Flow 后接 T-Flow

完整流程由四个模块组成：V-VAE 学习高精度顶点 latent，V-Flow 在图像或点云条件下生成该 latent；T-VAE 把离散邻接关系压入逐顶点连续 latent，T-Flow 再根据已经生成或编辑后的顶点预测 topology latent。两阶段共享一个粗粒度稀疏 voxel scaffold，用于约束活动空间并提供全局几何信息。

![](assets/LATO.2%20阅读总结/fig2-pipeline.png)

*图 2：论文 Figure 2。上半部分是顶点与拓扑各自的 VAE，下半部分是生成阶段的 V-Flow 与 T-Flow。最终由连接头预测顶点对之间的边，再通过环检测恢复三角面。*

### 3.2 V-VAE 与 V-Flow：生成亚体素精度顶点

V-VAE 沿用 Vertex Displacement Field（VDF）：在网格表面采样点，并为每个采样点记录指向其所属三角形某个顶点的位移。论文对每个网格采样 819,200 个表面点，先在 $1024^3$ 稀疏体素中聚合局部证据，再压缩为 $64^3$、32 通道的稀疏 latent。

解码器采用 coarse-to-fine subdivision，并不断剪掉不含顶点的体素。最细层不仅输出占用体素中心，还预测浮点偏移 $oldsymbol\delta_i$：

$$\mathbf v_i=\hat{\mathbf v}_i+\boldsymbol\delta_i$$

这一步把顶点从离散体素中心移到亚体素精度的位置。V-Flow 使用条件 Rectified Flow 生成顶点 latent，接受图像特征和目标顶点数量条件 $c_{\mathrm{vn}}=\log_2N$，因此推理时可以直接指定顶点预算。

### 3.3 T-VAE 与 T-Flow：把离散连接变成条件 latent

T-VAE 为每个顶点学习一个 16 维 topology latent。编码时，自注意力 mask 只允许顶点关注自身及真实相邻顶点；这是编码器观察连接关系的唯一通道，因而邻接信息必须被压入逐顶点 latent。解码器对任意顶点对的隐藏特征做对称 pairwise MLP，输出无向边概率：

$$\hat e_{ij}=\sigma\left[\mathrm{MLP}(\mathbf h_i\oplus\mathbf h_j)+\mathrm{MLP}(\mathbf h_j\oplus\mathbf h_i)\right]$$

T-Flow 则在实际生成或编辑后的顶点集合上采样 topology latent。顶点位置通过 3D positional encoding 注入，同时共享的 $64^3$ scaffold 提供全局形状条件。得到边集合后，系统用 loop detection 恢复三角面。这样，当用户移动、旋转或拼接顶点时，只需重新运行 T-Flow，就能产生适配新几何的连接，而不必重新优化顶点。

### 3.4 Part-wise generation

整体 latent 的容量会限制可生成顶点数。LATO.2 先把粗 scaffold 分成多个 part，再将每个 part 单独归一化到完整 latent 空间，分别以最大容量运行 V-Flow。各 part 顶点回到全局坐标后，可联合运行 T-Flow，或先分部生成拓扑再做 stitching。

这种做法等价于把固定 latent 容量反复分配给局部区域：part 越多，总顶点和面数越高，局部细节也越丰富。同样的机制还可用于局部 refinement，只重生成需要加密的区域。

---

## 4. 主要实验结果

### 4.1 顶点重建

V-VAE 在 $1024^3$ 量化基础上加入浮点 offset，取得 CD(L1) 0.0003、HD 0.0069。相比前代 LATO 的 0.0038 和 0.0083，CD(L1) 下降约 92%，说明亚体素偏移显著降低了体素量化误差。

### 4.2 完整 Mesh 生成

| 方法 | 类型 | CD(L2) ↓ | CD(L1) ↓ | HD ↓ | $|NC|$ ↑ |
|---|---|---:|---:|---:|---:|
| BPT | AR | 0.0603 | 0.0862 | 0.1067 | 0.8111 |
| DeepMesh | AR | 0.0529 | 0.0765 | 0.0946 | 0.8218 |
| MeshRipple | AR | 0.0458 | 0.0668 | 0.0941 | 0.8174 |
| MeshFlow | Flow | 0.0455 | 0.0668 | 0.0772 | 0.8227 |
| LATO | Flow | 0.0421 | 0.0617 | 0.0738 | 0.8262 |
| **LATO.2** | **Flow** | **0.0407** | **0.0596** | **0.0657** | **0.8341** |

相对前代 LATO，LATO.2 的 HD 下降约 11%，法线一致性也进一步提高。定性结果中，它对细长结构、复杂肢体和局部表面保持更完整，减少了其他方法常见的断裂、缺失与错误连接。

![](assets/LATO.2%20阅读总结/fig5-partwise.png)

*图 3：论文 Figure 6。与 BPT、DeepMesh、FastMesh、MeshRipple、MeshFlow 和 LATO 相比，LATO.2 在复杂部件与局部结构上整体更完整。*

### 4.3 消融结论

- **Offset Head** 不改变体素 occupancy 指标，却把 CD(L1) 从 0.0011 降至 0.0003，证明它专门解决连续顶点定位误差。
- **高分辨率 VDF 聚合再下采样**最关键；移除后 CD(L1) 恶化至 0.0034、HD 至 0.0145，说明局部证据必须在压缩前捕获。
- 加入 **100K 程序化合成网格**，F1 从 0.9137 提升至 0.9735，显著扩充了顶点密度、曲率和连接模式覆盖。
- 移除 T-Flow 的全局几何条件后，HD 从 0.0657 上升至 0.0694，$|NC|$ 从 0.8341 降至 0.8238；仅有顶点位置不足以决定全局一致拓扑。

---

## 5. 局限与评价

### 5.1 论文明确承认的局限

1. **误差单向传递**：T-Flow 只能连接现有顶点，无法修正 V-Flow 已经生成错误的顶点位置。
2. **二次复杂度**：默认拓扑解码评估全部顶点对，复杂度为 $O(N^2)$；面对更大网格时需要稀疏候选边选择。
3. **还不是完整生产资产**：目前只覆盖几何与连接，尚未生成 UV、纹理和材质属性。
4. **面向三角网格**：当前输出侧通过边环恢复三角面，没有解决高质量四边形 edge flow、UV seam 和可绑定拓扑等更严格生产要求。

### 5.2 我的评价

LATO.2 最有价值的地方并非单纯刷新 CD，而是提出了一个更合理的建模分工：**用 V-Flow 学连续几何，用 T-Flow 学条件拓扑。** 这个设计让生成、编辑和拼接共享同一个接口；顶点变化后，拓扑不再是必须手工维护的静态附属物，而成为可以重新采样的条件分布。

它也说明 2026 年原生 Mesh 生成正在从“把 Mesh 压成 token 再自回归”转向“为离散拓扑寻找连续生成空间”。与 MeshFlow、PolyFlow、Mesh BDF 等工作相比，LATO.2 的特色是阶段职责清晰、误差可隔离，并把 part-wise generation 与 topology-adaptive editing 直接建立在因子化结构上。

不过，论文中的“production-oriented”仍应谨慎理解：更准确地说，它已经具备生产导向的显式几何和编辑能力，但距离完整资产交付仍缺少 quad topology、UV、材质、严格流形保证以及超大规模拓扑解码。作者提出的下一步——让拓扑反馈反过来迭代修正顶点，并扩展到完整外观资产——正是决定这条路线能否真正进入 DCC 流程的关键。
