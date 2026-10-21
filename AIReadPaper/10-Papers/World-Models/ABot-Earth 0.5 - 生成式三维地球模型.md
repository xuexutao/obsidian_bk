---
type: paper-note
status: complete
domain: World-Models
paper: "ABot-Earth 0.5: Generative 3D Earth Model"
year: 2026
arxiv: 2606.09967
doi: null
source: https://arxiv.org/abs/2606.09967
project: https://abot-earth.amap.com/
code: https://github.com/amap-cvlab/ABot-Earth-0.5
tags:
  - World-Model
  - 3D-Generation
  - 3DGS
  - Satellite-Imagery
  - Digital-Earth
  - Level-of-Detail
created: 2026-10-16
updated: 2026-10-16
---

# ABot-Earth 0.5：生成式三维地球模型

## 基本信息

- **正式标题**：ABot-Earth 0.5: Generative 3D Earth Model
- **作者**：Ming Qian、Tianjian Ouyang、Mingchao Sun、Zijian Wang、Jincheng Xiong 等，AMAP CV Lab
- **来源状态**：官方 GitHub 技术报告与 arXiv:2606.09967 预印本均存在，因此本文按 `paper-note` 处理；截至本笔记写作时未确认同行评审 venue
- **输入**：地理配准卫星影像；推理不要求精确拍摄角或多视图重叠
- **输出**：地理配准、可分块流式加载、带分层 LOD 的原生 3D Gaussian Splatting（3DGS）场景
- **报告版本提醒**：官方仓库 2026-09-10 已宣布 ABot-Earth 0.7，项目页当前可能展示新版；本笔记严格分析 0.5 论文与对应报告

## 一句话结论

ABot-Earth 0.5 把大尺度三维地图生产从“多视图倾斜摄影重建”改写为“**卫星单条件 + 真实城市 3DGS 先验的生成**”，并把 200m 训练 tile 扩到 1.6km 生产 block，再经地理对齐、zoom 14–19 六级 LOD 和地图引擎调度部署；它最有价值的是把生成模型、瓦片生产与万亿 Gaussian 流式渲染连成系统，但论文没有公开生成网络、潜空间和训练损失的关键细节，也没有公平的同协议几何 benchmark 或正式消融，因此其证据更接近“工业系统可行性报告”，而不是可完整复现的算法论文。

---

## 1. 研究背景与问题转移

### 1.1 传统地球三维化的成本瓶颈

传统城市级 3D 地图依赖密集倾斜摄影、LiDAR、反复外业采集和长时间摄影测量，几何精度高，但数据获取慢、覆盖昂贵、更新滞后。生成式方案试图用 learned prior 从稀疏条件补全未观测几何，把成本从每个城市的重复采集转移到一次性模型训练。

卫星图适合作为全球条件，因为具有地理坐标、覆盖广且更新频率高；困难是它只提供俯视/斜视外观，建筑立面、树冠下方和地面视角高度欠约束。ABot-Earth 0.5 因而不是“从卫星恢复真实三维”的纯重建系统，而是**以卫星布局为条件、从真实城市重建中学习先验后生成合理 3D**。

![](assets/ABot-Earth%200.5%20-%20生成式三维地球模型/figures/01-teaser.png)

> **图 1：0.5 报告展示的全球覆盖。** 图中蓝点和十个城市样例对应作者声称的 300+ 城市、190+ 国家发布范围。它是部署覆盖与定性效果展示，不是地理或几何误差评测。图片已从 arXiv 源码提取并视觉核验。  
> 来源：论文 Figure 1，<https://arxiv.org/abs/2606.09967>

### 1.2 四个系统矛盾

论文把问题归为四项：复杂户外场景更适合非结构化 3DGS 而非规则 mesh；地球级内容必须具有多级 LOD；公里尺度不能单次整体生成且 tile 接缝要平滑；真实卫星来自不同传感器、GSD 和角度，与训练时模拟条件存在域差异。

---

## 2. 训练数据：先重建真实城市，再学习生成

### 2.1 三类数据源

1. **卫星数据**：多角度 off-nadir 影像通过 ABot-3DGS/FromOrbit2Ground 重建，2.5D Z-monotonic SDF 恢复城市结构，扩散修复网络补立面纹理；
2. **航空数据**：高分辨率倾斜摄影是主要来源，可融合 LiDAR 或摄影测量 mesh；
3. **城市/低空数据**：街景、UAV 与其他低空影像和航空数据联合配准，增强建筑立面与近地视角。

公开数据包括 DFC 2019、UrbanScene3D、UrbanBIS、CrossLoc、Mill-19、UAVD4L、DenseUAV 与 UC-GS；论文还使用未完整披露的自有数据。

### 2.2 训练目标为何是 3DGS scene

ABot-3DGS 先把多源影像重建为城市级真实 3DGS，执行跨视角配准、分层并行优化、深度/多视图约束、动态物体移除和外观解耦。生成模型学习的目标不是原始照片或人工 mesh，而是清理后的 Gaussian primitives。这使模型能够直接表示树木、立面、水面等非流形结构，但训练目标继承了重建误差与修复先验。

### 2.3 Tile 与多视角监督

训练场景以 200m×200m 重叠滑窗切块，归一化到标准坐标并聚类清理 floaters。每个 tile 从多个高度、FOV、pitch、yaw 渲染训练视图，同时对相机参数施加扰动。重叠区域给模型边界上下文，为后续滑窗大场景生成做准备。

### 2.4 三级质量控制

- **Tile 级**：PSNR、SSIM、LPIPS、几何、VLM 感知与空间完整性；
- **View 级**：累计 opacity 过滤空洞/边缘视图，VLM 检查清晰度和伪影；
- **Dataset 级**：按场景类别平衡，在 embedding 空间聚类去重，减轻单一城市形态主导。

![](assets/ABot-Earth%200.5%20-%20生成式三维地球模型/figures/02-data-pipeline.png)

> **图 2：训练数据生产链。** 卫星/航空/城市影像经 ABot-3DGS 重建、200m tile 切分、多视角渲染和三级质量筛选，最终形成城市级 3DGS tiles。图像文字与边缘已视觉核验完整。  
> 来源：论文 Figure 2，§2，<https://arxiv.org/abs/2606.09967>

---

## 3. 卫星条件的原生 3DGS 生成

### 3.1 Compression–generation 范式

论文称模型在数百万无序 Gaussian primitives 构成的真实场景上学习紧凑 latent，再从 latent 直接解码新的 3DGS。其目标是绕过 mesh 的规则拓扑与封闭表面假设，使植被、水面和复杂建筑能以 Gaussian 集合表达。

但报告没有公开编码器/解码器结构、latent 形状、Gaussian 参数化、扩散或自回归过程、损失函数、参数量、优化器和训练步数。因而“原生 3DGS generation”是明确的系统设计，尚不是可独立重建的算法配方。

### 3.2 卫星条件与跨域适配

训练时从已有 3DGS 场景渲染模拟卫星视图作为条件；推理时输入真实卫星图，并通过 VLM-based harness 按传感器、分辨率、质量和拍摄特征动态适配，以缩小模拟—真实 domain gap。论文没有给出这个 harness 的网络细节或消融，因此其独立贡献无法量化。

### 3.3 输入尺度与地理预处理

全球卫星源的 GSD 不一致，Web Mercator（EPSG:3857）在高纬度还有明显面积畸变。生产链先拼接 raster tiles，再按目标地理范围做各向同性重采样，把有效 GSD 归一到训练范围。每个 tile 的 geographic bounding box 被保留，供生成后恢复全球坐标。

---

## 4. 大尺度连续生成

### 4.1 训练 tile 到生产 block

在 A100 上，单次处理 4K 卫星图，对应约 1.6km×1.6km、2.56km²，相比 200m×200m 训练 tile 面积扩大 64 倍。内部重叠滑窗融合相邻 tile 的过渡区域，目标是减少硬拼接裂缝。

论文措辞是 block 内 **near-seamless**，并明确承认跨 1.6km 生产 block 的完全无缝推理尚未完成。因此不能把“seamless Earth”理解为全球任意跨块边界已经无缝。

### 4.2 全球生产资源估算

| 项目 | 报告数值 |
|---|---:|
| 单次生产 block | 1.6km×1.6km = 2.56km² |
| 单 block 推理 | A100，约 25 分钟 |
| 单位面积 | 约 9.77 分钟/km² |
| 全球建成区假设 | 约 800,000km² |
| 生产 block | 约 312,500；部署段另写约 320,000 |
| 集群 | 1,000 GPUs |
| 完整生产估算 | 10 天内 |

这是工程规划与作者报告的生产数据，不是第三方复现实验。论文还在一处写最终输出为“hundreds of billions” Gaussians，后文按 320,000×1 亿给出约 3.2 trillion，两处统计口径不一致。

---

## 5. LOD：从模型输出到地图引擎

### 5.1 原生多 LOD 与实际部署

方法章节强调 inherent multi-LOD decoder 直接生成分层 3DGS，避免完全依赖事后简化。部署章节给出更精确实现：

- zoom 17–19：模型原生多分辨率输出；
- zoom 14–16：从 zoom 17 用 Bhattacharyya distance 统计抽稀，在 CPU 侧并行执行。

因此准确表述是“**高精度三级原生输出，低精度三级后处理抽稀**”，而不是所有 LOD 都端到端原生生成。

### 5.2 地理对齐与重分块

生成 block 先通过生产时仿射参数恢复到 EPSG:3857，再以 tile 中心建立 ENU 局部切平面；位置、旋转四元数和尺度统一变换，同时保留 ENU 原点维持全球配准。之后所有 Gaussian 重新分配到标准 `{zoom}/{x}/{y}` 地图瓦片，跨原始推理 block 合并数据。

### 5.3 六级 LOD 与空间索引

系统构建 zoom 14–19 六级四叉树，并提供两套索引：OGC 3D Tiles `tileset.json` 供标准客户端使用；隐式 `{zoom}/{x}/{y}` 路径供地图引擎和 CDN 缓存。

![](assets/ABot-Earth%200.5%20-%20生成式三维地球模型/figures/03-global-lod-pipeline.png)

> **图 4：全球等面积切块与 LOD 四叉树。** 左侧表示并发生产的地球瓦片，右侧区分 zoom 17–19 模型原生输出和 zoom 14–16 CPU 抽稀。该本地图由论文 Figure 4 的两个官方子图并排整理，未修改图中数据；已视觉核验完整。  
> 来源：论文 Figure 4，§4，<https://arxiv.org/abs/2606.09967>

### 5.4 地图引擎实时调度

Amap Yunjing 每帧按视口选择 tile 和精度：近景加载 zoom 17–19，远景加载 zoom 14–15，并在层级间平滑淡入淡出；复用视锥剔除、异步加载和 tile scheduling。约 320,000 blocks、每块约 1 亿 Gaussians 的原始规模远超消费 GPU 容量，LOD 不是可选优化，而是系统可运行的前提。

![](assets/ABot-Earth%200.5%20-%20生成式三维地球模型/figures/04-lod-rendering.png)

> **图 5：地图引擎的 LOD 渲染。** 左半以颜色示意近、中、远层级，右半显示融合后的城市视图。它展示调度效果，不提供客户端 FPS、带宽或显存数据。图片已从官方源码原图提取并视觉核验。  
> 来源：论文 Figure 5，§4.2，<https://arxiv.org/abs/2606.09967>

---

## 6. 实验结果

### 6.1 FID/KID

| 方法 | FID↓ | KID↓ |
|---|---:|---:|
| CityDreamer | 97.3 | 0.096 |
| GaussianCity | 86.9 | 0.090 |
| EarthCrafter | 69.5 | 0.061 |
| **ABot-Earth 0.5** | **16.1** | **0.006** |

数值差距很大，但论文明确说明各 baseline 使用不同 GT 图像集、相机位姿与近远景采样，且 baseline 数字直接引自各自论文。因此这不是统一 benchmark 的 apples-to-apples 结果，只能证明 ABot 在自己的评测协议中生成渲染与真实分布接近，不能把 FID 下降 53.4 解释成同条件算法增益。

### 6.2 Google Earth 与商业系统比较

作者人评称 ABot 整体 aesthetics 高于 Google Earth，而 Google Earth 在 geometry accuracy 和 texture fidelity 上更强。论文没有充分披露参与者数量、样本量、显著性检验。这个结果与系统定位一致：生成模型擅长视觉合理与颜色和谐，传统摄影测量仍擅长测绘级结构和精确纹理。

对于著名地标，论文没有坚持纯生成，而是使用 COLMAP SfM/MVS 从众包照片重建，再转为 3DGS、地理注册并嵌入生成背景。这一混合方案实用，也间接说明 0.5 难以精确复现具有强身份约束的建筑。

### 6.3 下游 Embodied AI 证据

论文提出无人机导航、避障、控制和 closed-loop simulation 作为应用，却没有提供任务成功率、sim-to-real 迁移或策略训练实验。当前结果支持“可提供大范围视觉空间先验”，不支持“已证明缩小机器人 sim-to-real gap”。

---

## 7. 消融与证据缺口

报告没有正式 Ablation Study，也没有逐项验证：原生 3DGS vs mesh/voxel；VLM harness；模拟卫星条件；重叠滑窗；overlap 宽度；multi-LOD decoder；Bhattacharyya 抽稀；多源数据；VLM 清洗；跨视角融合。方法由多个工业模块组成，但读者无法从现有表格判断哪一项真正决定 FID、接缝或部署性能。

论文还缺少最关键的几何指标：高度误差、深度误差、Chamfer Distance、地理定位误差、tile seam error 与 LOD distortion；也没有公开浏览器 FPS、显存、网络带宽和加载延迟。这些指标比单纯 FID/KID 更能回答“是不是可用的 3D 地球”。

---

## 8. 作者声称与已验证事实

| 层级 | 表述 | 判断 |
|---|---|---|
| **论文实测/报告** | A100 单 block 25 分钟、约 9.77 分钟/km² | 作者系统数据，记录明确 |
| **部署统计** | 约 320k blocks、约 3.2 万亿 Gaussians、六级 LOD | 报告给出，但 Gaussian 总量与前文口径不一致 |
| **非统一 benchmark** | FID 16.1、KID 0.006 优于引用基线 | 不是相同 GT/相机协议，不能作严格 SOTA 证明 |
| **人类主观比较** | 美学高于 Google Earth | 缺少完整人评协议；Google 几何/纹理仍领先 |
| **作者展示** | 300+ 城市、190+ 国家 | 支持部署覆盖声明，不代表每处几何准确 |
| **应用愿景** | 缓解 UAV sim-to-real | 没有闭环任务实验，尚未证明 |

---

## 9. 局限、未来工作与个人判断

### 9.1 明确局限

几何和纹理仍弱于 Google Earth；当前主要是航空级而非街景级；只保证生产 block 内 near-seamless；推理范围受 GPU 显存限制；著名地标需混合重建；跨 block 完全无缝未实现；算法细节不足以复现；LOD“全原生”表述与实际低层抽稀存在口径差异。

### 9.2 作者未来方向

论文明确提出从 sky 到 ground、提升街景级细节与 reconstruction-grade fidelity、扩展场景多样性，并系统验证户外 3D 生成 scaling laws。部署段还把跨 block 完全无缝列为关键目标。

### 9.3 个人判断

ABot-Earth 0.5 的创新重心是系统集成：训练数据本身来自真实城市 3DGS，模型输出也是 3DGS，后端继续沿用地图瓦片和 LOD 基础设施，表示贯穿训练、生成与部署。它降低的是“快速获得视觉上合理大范围 3D 底图”的成本，不是取代测绘重建。若用于仿真，首先应明确任务对几何精度的容忍度；若需要碰撞、导航或地标识别，必须补充几何校准、高精度地标与闭环验证。

---

## 10. 材料来源

本地保存了 arXiv 官方 PDF、GitHub 官方技术报告及四张经视觉核验的关键图：

- `assets/ABot-Earth 0.5 - 生成式三维地球模型/sources/paper.pdf`
- `assets/ABot-Earth 0.5 - 生成式三维地球模型/sources/github-tech-report.pdf`
- `assets/ABot-Earth 0.5 - 生成式三维地球模型/sources.md`
