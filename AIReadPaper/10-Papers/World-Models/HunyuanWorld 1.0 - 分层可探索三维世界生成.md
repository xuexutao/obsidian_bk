---
type: paper-note
status: complete
domain: World-Models
paper: "HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels"
year: 2025
arxiv: "2507.21809"
doi: "10.48550/arXiv.2507.21809"
source: https://arxiv.org/abs/2507.21809
project: https://3d-models.hunyuan.tencent.com/world/
code: https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
tags:
  - World-Models
  - 3D-Generation
  - Panorama
  - Layered-Mesh
  - Text-to-World
  - Image-to-World
created: 2026-10-16
updated: 2026-10-16
---

# HunyuanWorld 1.0：分层可探索三维世界生成

## 基本信息

- **正式标题**：*HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels*
- **作者/团队**：Tencent HunyuanWorld Team；项目负责人 Chunchao Guo、Tengfei Wang
- **版本**：arXiv v1 2025-07-29；v2 2025-08-13
- **任务**：从文本或单张透视图离线生成可导出、可漫游、对象可分离的三维世界
- **主要输出表示**：语义分层的纹理网格；可选分层 3DGS、对象级 Hunyuan3D 资产与天空 HDRI
- **本地论文**：[paper.pdf](assets/HunyuanWorld%201.0%20-%20分层可探索三维世界生成/sources/paper.pdf)

## 一句话结论

HunyuanWorld 1.0 的关键不是“直接用生成模型吐出整个 3D 世界”，而是用 **360° 全景图作为数据丰富的二维世界代理**，再借助 VLM 分层、跨层深度对齐与 sheet warping 把它转换为可导出的分层 mesh；它解决了离线内容资产的 360° 覆盖、图形管线兼容和对象可编辑性，但几何仍高度依赖单目深度、补全与启发式分层，论文也没有提供几何真值评测、严格消融或端到端生成时延。

---

## 1. 研究背景与问题定义

### 1.1 两条路线各缺什么

视频世界模型拥有海量二维训练数据和强生成先验，容易覆盖多样场景与动态，但长视频会积累漂移，逐帧生成昂贵，而且 RGB 帧本身不能直接进入 Unity、Unreal、物理引擎或传统 DCC 管线。显式 3D 方法有持久几何和高效渲染，却受制于世界级 3D 数据稀缺、大场景表示昂贵，以及“整个场景粘成一块”导致对象不可编辑。

HunyuanWorld 1.0 的折中是：**先在 2D 全景空间做生成，再在 3D 空间做结构化资产化**。全景图覆盖单一观察点的 $360^\circ\times180^\circ$ 视野，既可由成熟图像扩散模型学习，又比普通透视图更适合作为世界初始化。

![](assets/HunyuanWorld%201.0%20-%20分层可探索三维世界生成/figures/01-teaser.png)

> 封面 teaser：论文展示室内、自然、城市、灾难和风格化世界，说明目标是“世界级场景多样性”，而不是单类室内重建。该图是源码中的完整封面原图。来源：arXiv 源码，PDF p.1。

### 1.2 核心贡献

1. **Panoramic world proxy**：统一文本与单图输入，先生成 ERP 全景图。
2. **Agentic world layering**：VLM 决定哪些实例应独立建模，并把全景图分解为前景对象、背景/地形与天空。
3. **Layer-aligned reconstruction**：各层独立估深后，对重叠区域做跨层深度匹配，维持遮挡顺序。
4. **显式可交付资产**：以分层 mesh 为主表示，可选完整对象生成、HDRI 天空和分层 3DGS。
5. **长程扩展接口**：用 Voyager 生成 RGB-D 视频并维护点云缓存，扩展初始全景之外的区域。

---

## 2. 任务定义、输入输出与表示

### 2.1 输入

- **Text-to-World**：用户文本；中文先翻译并由 LLM 扩写，使 prompt 接近训练 caption 分布。
- **Image-to-World**：单张 pinhole 图像；先由 MoGe/UniK3D 一类模型估计相机内参与深度，再投影到 ERP 空间。图像分支还用 VLM 生成 scene-aware prompt：显著对象被写入负面约束，避免在补全区域复制雕像、车辆等已有主体。

### 2.2 中间状态与最终输出

核心中间状态是 ERP 全景图 $I^{pan}$。Agent 把它分为

$$\mathcal{L}=\{L_{fg}^{1},\ldots,L_{fg}^{n},L_{bg},L_{sky}\},$$

每层包含 RGBA 图像、语义/实例掩码与深度。最终输出主要是每层网格的并集：

$$\mathcal{W}_{mesh}=\bigcup_{L\in\mathcal{L}}\operatorname{WarpGrid}(I_L,D_L,M_L).$$

上式是对论文 pipeline 的紧凑记号：实际机制是 WorldSheet 风格的 grid sheet warping，不是一个端到端学习出的隐式场。前景还可绕过薄片投影，交给 Hunyuan3D 生成完整闭合对象，再自动放回场景；天空可输出 HDRI；论文也说明可从分层深度优化 3DGS。

### 2.3 “可交互”准确指什么

这里的交互首先是**生成完成后的资产交互**：漫游、碰撞、刚体/流体模拟、对象平移/旋转/缩放。它不是 HY-World 1.5 那种用户每次输入动作后即时生成下一视频块的在线世界模型。1.0 的生成是离线资产构建，实时的是 mesh/浏览器/引擎渲染阶段。

---

## 3. 核心方法

### 3.1 总体流程

![](assets/HunyuanWorld%201.0%20-%20分层可探索三维世界生成/figures/02-method-pipeline.png)

> Figure 2：左侧用 Panorama-DiT 把图像或文本变为全景代理；中间由 VLM、检测与分割完成语义分层和补全；右侧独立估深、跨层对齐并生成分层 mesh。可选分支用图生 3D 得到完整前景对象，并把天空输出为 HDRI。来源：论文 Figure 2，PDF p.5。

流程的因果顺序很重要：

1. 生成全景代理；
2. 识别“值得交互”的对象实例；
3. 逐层移除对象并补全被遮挡区域；
4. 各层估深并对齐；
5. sheet warping / 3D object generation；
6. 压缩、导出，必要时再用 Voyager 延伸世界。

它不是统一网络，而是多个生成、理解、重建与几何工具组成的级联系统。优点是每阶段可替换、输出可编辑；缺点是误差会逐层传递。

### 3.2 Panorama-DiT：把透视图/文本变成 360° 代理

Panorama-DiT 采用扩散 Transformer。文本模式只以增强后的 prompt 为条件；图像模式先把输入投影到 ERP，VAE 编码后与 noisy panorama latent 拼接，再加入 scene-aware 文本描述。

ERP 的两个专门问题是极区畸变和左右边界不连续。论文采用：

- **elevation-aware augmentation**：训练时以概率 $p$ 对全景图做比例为 $r$ 的垂直循环位移，增强视点高度鲁棒性；论文未披露 $p,r$ 的具体值。
- **circular denoising**：推理时在 latent 左右循环 padding，并随去噪过程渐进 blend，使 $0^\circ/360^\circ$ 接缝连续。

### 3.3 Agentic world layering

VLM 先根据场景语义和空间关系选出需要独立建模的对象，例如近处车辆而非远处整栋建筑。随后：

1. 对 ERP 图做 circular padding，避免跨左右边界的对象被切成两半；
2. Grounding DINO 类检测器给出框；
3. ZIM 类分割器生成 mask；
4. 以面积/NMS 合并重叠和跨边界碎片；
5. 自回归地移除当前对象并补全遮挡区域，得到多层 RGBA；
6. 单独训练背景移除补全模型与天空 HDRI 补全模型。

这一“agentic”主要是 VLM 驱动的语义决策和工具编排，不应理解成会在 3D 环境中自主试错的强化学习 agent。

### 3.4 跨层深度对齐

原始全景图先得到基础深度 $D_{base}$，第一前景层可直接取对应区域。后续前景和移除对象后的背景分别估深；若直接使用，各层可能处于不同尺度/偏移，遮挡顺序会错。论文在重叠区域最小化层间距离。机制可写成：

$$\theta_l^*=\arg\min_{\theta_l}\sum_{x\in\Omega_l}\left\|g(D_l(x);\theta_l)-D_{base}(x)\right\|,$$

其中 $\Omega_l$ 是有效重叠区域，$g$ 表示论文所称的 depth matching 变换。报告没有给出 $g$ 的具体参数形式、损失范数和鲁棒估计细节，因此不能进一步声称它是特定的线性或非线性标定。天空深度被设为略大于所有层最大深度的常数。

### 3.5 分层网格与完整对象

- **直接投影**：由 RGBA、深度和 mask 把网格 sheet warp 到 3D；极区做平滑，mask 边界做抗锯齿。
- **完整对象生成**：裁出实例图，送入 Hunyuan3D，生成闭合资产后按估计布局放回世界。
- **背景**：先压缩异常深度分布，再把 ERP 背景 warp 为网格。
- **天空**：常深度网格或 HDRI 环境贴图。

分层表示的价值在于对象级编辑和图形引擎兼容；它的根本边界是：普通 sheet warping 仍是从单个全景观察点展开的表面，强遮挡背面并没有真实观测，只有补全先验。

### 3.6 Voyager 长程扩展

Voyager 接收初始世界视图和用户相机轨迹，生成空间一致的 RGB-D 视频。初始 mesh 构成点云 cache；每个目标相机先渲染 cache 作为部分条件，扩散模型补全未见区域，新 RGB-D 又回写 cache。为延长轨迹，系统按 clip 自回归生成并剔除冗余点。

这部分把 1.0 从“中心全景附近的离线 mesh”扩展到更远区域，但 Voyager 是附加的视频补全组件。主论文没有给出长程扩展的单独定量、消融或失败率，官方仓库当前 README 也不足以证明 Voyager 代码/权重已与主 pipeline 同等完整发布。

### 3.7 存储与推理工程

- 离线 mesh decimation + texture baking + XAtlas UV：论文报告约 **80%** 体积下降。
- Web Draco：报告约 **90%** 体积下降，并可由 WebAssembly 加载。
- TensorRT、选择性 step cache、多 GPU 并行 CFG：用于加速扩散推理。

论文称这些优化支持“fast generation / real-time performance”，但没有端到端秒数、GPU 型号、FPS 或 time-to-first-view。因而只能确认渲染与加载被工程化，不能把 1.0 标成实时生成模型。

---

## 4. 数据与实验设置

### 4.1 数据工程

数据来自商业采购、开放数据与 UE 渲染。自动过滤包括去重、尺寸、循环位移增强、内容安全、人像、水印、清晰度、美学和畸变；之后由人工过滤接缝、极区遮挡、几何不一致、对象重复、异常人体和狭窄杂乱场景。

![](assets/HunyuanWorld%201.0%20-%20分层可探索三维世界生成/figures/03-data-pipeline.png)

> Figure 3：数据质量不是只靠一个审美分数，而是自动过滤与人工几何检查串联。论文没有披露清洗前后规模、来源占比与许可细目，这限制了可复现性。来源：论文 Figure 3，PDF pp.5–6。

Caption 采用三阶段流程：VLM 重描述 → LLM 蒸馏成长短不同的层级 caption → 专业标注员核验。论文未披露数据总量、训练轮数、优化器、学习率、GPU-hours 和模型参数量。

### 4.2 评测协议与 baseline

由于生成任务没有 GT 世界，论文主要在渲染图上评估：

- 质量：BRISQUE↓、NIQE↓、Q-Align↑；
- 条件一致性：图像输入用 CLIP-I↑，文本输入用 CLIP-T↑；
- 图生全景：Diffusion360、MVDiffusion；
- 文生全景：再加 PanFusion、LayerPano3D；
- 图生世界：WonderJourney、DimensionX；
- 文生世界：LayerPano3D、Director3D。

图生世界沿 $0^\circ$ 到 $90^\circ$ 的 7 个视点渲染；文生世界按 6 个方位渲染。所有评测渲染为 $960\times960$。这一协议能测观感与 prompt 对齐，却不能测 mesh 拓扑、尺度、碰撞正确性或物理可用性。

---

## 5. 定量、定性与消融

### 5.1 主定量结果

| 任务 | 最强对照 | 对照结果 | HunyuanWorld 1.0 | 主要增益 |
|---|---|---|---|---|
| 图生全景 | MVDiffusion | BRISQUE 47.7 / NIQE 7.0 / Q-Align 2.7 / CLIP-I 80.8 | **45.2 / 5.8 / 4.3 / 85.1** | 质量与输入一致性均提升 |
| 文生全景 | LayerPano3D | 49.6 / 6.5 / 3.7 / CLIP-T 21.5 | **40.8 / 5.8 / 4.4 / 24.3** | 全部指标领先 |
| 图生世界 | DimensionX | 45.2 / 6.3 / 3.5 / CLIP-I 83.3 | **36.2 / 4.6 / 3.9 / 84.5** | 新视图观感更好 |
| 文生世界 | LayerPano3D / Director3D | 最佳对照 BRISQUE 35.3、NIQE 4.8、Q-Align 3.9、CLIP-T 23.5 | **34.6 / 4.3 / 4.2 / 24.0** | 各项小幅领先 |

结果支持“渲染出来更好看、更贴输入”，但指标都在 2D 渲染空间，不能证明 3D 几何真实。尤其 Q-Align/CLIP 对遮挡背面、薄片结构和碰撞代理几乎不敏感。

### 5.2 定性结果

![](assets/HunyuanWorld%201.0%20-%20分层可探索三维世界生成/figures/04-image-to-world-comparison.png)

> Figure 12：同一输入下，WonderJourney 容易改变房间布局或对象；DimensionX 的视点范围和结构稳定性有限；HunyuanWorld 的三视图更连贯。但该图是作者选择的案例，不能代替全测试集几何统计。来源：论文 Figure 12，PDF p.15。

### 5.3 消融证据缺口

论文**没有给出组件级消融表**。因此以下核心设计无法从报告中独立量化：

- circular denoising 对接缝的净贡献；
- scene-aware negative prompt 是否确实减少对象复制；
- agentic layering 相比固定类别分层的收益；
- cross-layer depth alignment 对遮挡/几何指标的提升；
- 完整对象生成相对 sheet warping 的质量—成本权衡；
- Voyager cache、点剔除和 smooth sampling 的独立作用。

这不是“消融结果为负”，而是**报告未提供消融**。主表是与外部方法比较，不能替代内部组件验证。

### 5.4 失败案例与可预期故障

论文没有专门 failure-case figure 或失败率。由 pipeline 可以定位以下失效链：

1. 单图内参/深度估错 → ERP 投影扭曲 → 分层与网格均受污染；
2. VLM 漏掉可交互对象 → 对象被烘进背景 mesh，无法独立编辑；
3. 检测/分割跨接缝失败 → 对象被切开或重复；
4. inpainting 幻觉 → 被遮挡区域的纹理与几何不真实；
5. 各层估深尺度不一致 → 错误遮挡、悬浮或穿插；
6. sheet warping 对背面/厚度建模不足 → 大幅平移后暴露薄片；
7. Voyager 早期错误回写点云 cache → 后续视图持续继承错误。

这些是根据明确机制推导的风险，不是论文统计过的失败概率。

---

## 6. 与 HY-World 1.5 / WorldPlay 和 HY-World 2.0 的范式关系

| 系统 | 生成时机 | 世界状态/输出 | 用户输入 | 主要目标 | “实时”边界 |
|---|---|---|---|---|---|
| **HunyuanWorld 1.0** | 先离线生成完整资产 | 分层 mesh；可选 3DGS/HDRI/对象资产 | 文本或单图；后续相机路径 | 资产导出、360° 漫游、对象编辑 | 生成未给实时证据；生成后 mesh 渲染可实时 |
| **HY-World 1.5 / WorldPlay** | 在线按 chunk 继续生成 | 动作条件 RGB 视频 + 历史图像记忆 | 初始图/文本 + 键鼠/相机动作 | 低延迟响应、长期回访一致性 | 720p/24 FPS 依赖 8×H800 与系统优化；不是显式 3D 资产 |
| **HY-World 2.0** | 离线四阶段生成或前馈重建 | 以 3DGS 为主，附点云、深度、法线、相机与 coarse mesh | 文本、单图、多图或视频 | 统一 sparse-input 生成与 dense-input 重建 | 单世界约 712 秒（H20 系统）；实时的是 WorldLens 渲染/碰撞 |

1.0 的核心问题是“怎样交付可用 3D 资产”；1.5 把问题转为“怎样根据动作立刻生成下一段观测”；2.0 又回到离线显式 3D，但把 1.0 的单全景薄片资产升级为“规划多轨迹 → 视频模型扩展关键视图 → WorldMirror 重建 → 优化 3DGS”，并让同一重建骨干直接处理真实多视图/视频。

---

## 7. 局限、Future Work 与批判性评价

### 7.1 作者披露边界

报告结论没有单列 limitations 或 future work，也没有专门失败案例。可明确确认的边界来自正文：全景代理天然缺少遮挡区域；单次生成的探索范围有限；Voyager 用生成式 RGB-D 和 cache 弥补，但其量化未在本报告展开。

### 7.2 关键局限

1. **几何证据不足**：没有 Chamfer、F-score、深度误差、mesh 完整率、拓扑或碰撞指标。
2. **级联误差**：内参、全景、VLM、分割、补全、深度、对齐任一阶段出错都会向后传播。
3. **单中心世界代理的上限**：全景解决旋转覆盖，不等同于自由平移所需的多视图观测。
4. **动态与物理不是生成目标**：物理引擎可以消费 mesh，不代表模型学习了动力学或因果状态转移。
5. **披露不足**：数据规模、训练配置、参数量、GPU-hours、端到端耗时、硬件下限均缺失。
6. **比较协议偏视觉**：论文最强结论应表述为“渲染质量/条件对齐领先所选 baseline”，而不是“几何全面 SOTA”。

### 7.3 个人判断

这项工作的长期价值在“**全景先验 + 语义分层 + 显式图形资产**”的产品化组合。它把强二维生成能力转成了可装载、可编辑、可碰撞的资产，且分层输出比单体 NeRF/3DGS 更接近制作管线。但从世界模型定义看，它更像一个离线 3D 内容生成系统：没有可验证的动态状态、闭环动作转移或策略接口。后续 1.5 和 2.0 分别补的是“在线动作反馈”和“更完整、更统一的显式 3D”，并非简单版本号升级。

---

## 8. 复现与开源状态

### 8.1 当前可确认状态

官方仓库已提供：

- `demo_panogen.py` 与 `demo_scenegen.py` 推理入口；
- Text / Image PanoDiT 权重；
- Scene / Sky PanoInpaint 权重；
- 示例、环境文件、浏览器 viewer 与 Draco 支持。

公开版说明以 Flux 为图像生成底座。README 测试环境为 Python 3.10、PyTorch 2.5.0+cu124，可选 FP8 GEMM/attention 与 cache。仓库存在 LICENSE/NOTICE，但模型、代码和第三方组件的具体授权应逐项阅读，不能只按 README 推断。

### 8.2 不能确认的内容

- 完整训练代码与训练数据不可由报告直接复建；
- README 未给最低显存和端到端耗时；
- “Lite Version / Voyager”在 Open-Source Plan 中出现，不等同于当前全链路已发布；
- 论文中的 VLM、检测、分割、MoGe/UniK3D、Hunyuan3D、TensorRT/Draco 构成复杂依赖链。

### 8.3 最小复现建议

1. 先运行 Image-to-Panorama，检查 ERP 接缝与输入图保真；
2. 固定全景，分别验证对象 mask、移除补全和跨层遮挡；
3. 对关闭/开启深度对齐做回归测试，至少测重投影误差、层间穿插率；
4. 导出 mesh 到 Blender/Unity，验证纹理、碰撞与对象变换；
5. 若评估 Voyager，应单独测闭环轨迹的一致性和 cache 污染，而不能只看精选视频。

完整复现的主要成本不是单次推理，而是未公开的全景数据、人工质检、层移除三元组、天空 HDRI 数据及多个外部大模型。

---

## 9. 后续问题

1. 是否能用 2.0 的 WorldMirror/3DGS 替代 1.0 的薄片 mesh，同时保留对象级分层？
2. 如何把 WorldPlay 的在线观测写回 1.0/2.0 的持久显式地图，而不是只存历史图像？
3. 对象可编辑性应如何量化：实例完整率、对象—背景粘连率、碰撞正确率还是引擎导入成功率？
4. 生成式补全区域缺乏真值时，如何对几何不确定性建模并阻止其被物理模拟误用？

---

## 10. 材料来源

完整来源、图片映射与核验记录见 [sources.md](assets/HunyuanWorld%201.0%20-%20分层可探索三维世界生成/sources.md)。
