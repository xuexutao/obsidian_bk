---
type: paper-note
status: complete
domain: World-Models
paper: "Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding"
year: 2026
arxiv: "2603.19235"
doi: "10.48550/arXiv.2603.19235"
source: "https://arxiv.org/abs/2603.19235"
project: "https://h-embodvis.github.io/VEGA-3D/"
code: "https://github.com/H-EmbodVis/VEGA-3D"
tags:
  - World-Models
  - Implicit-3D-Prior
  - Video-Diffusion
  - MLLM
  - Spatial-Reasoning
  - VLA
created: 2026-10-16
updated: 2026-10-16
---

# VEGA-3D：视频生成先验驱动的空间理解

## 基本信息

- **正式题目：** *Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding*。
- **作者与机构：** Xianjin Wu、Dingkang Liang、Tianrui Feng、Kui Xia、Yumeng Zhang、Xiaofan Li、Xiao Tan、Xiang Bai；华中科技大学与百度。
- **版本与会议：** arXiv v1 于 2026-03-19 提交，本文阅读 v3（2026-07-17）；论文标注 ECCV 2026 接收，主文与补充共 32 页。
- **任务：** 不输入点云、不做显式深度/BEV重建，直接从冻结视频生成模型的中间 DiT feature 中提取隐式三维先验，增强 3D 场景理解、视频空间推理与机器人操控。
- **重要性评估：** ★★★★☆（4/5）。它为“生成模型隐式三维先验”提供了可测量、可消融、可复用的证据链；但生成分支主要是 feature encoder，而非动作条件 world rollout，因而对 planning 的贡献是表征级间接增益，不应夸大为模型预测未来后做规划。

## 一句话结论

VEGA-3D 将冻结的 Wan2.1-T2V 1.3B 当作 Latent World Simulator：给输入多视图视频 latent 加中等强度噪声，在第 20 个 DiT block 读取特征，再与 SigLIP/MLLM 语义 token 做逐 token 自适应门控融合；这种隐式生成先验显著改善定位型 3D 任务，并让 OpenVLA-OFT 在 LIBERO 平均成功率从 97.0% 增至 97.3%，但单独使用生成特征会全面崩溃，说明它是语义表征的补充而非替代。

---

## 1. 研究背景与问题演化

### 1.1 显式 3D 路线的规模瓶颈

让 MLLM 理解三维通常有两种做法：直接输入点云/深度，或用相机 pose、深度教师、BEV reconstruction 把 2D feature 抬升到 3D。前者受制于高质量 3D 数据规模，后者需要复杂几何 pipeline 与额外监督。两者都把三维能力视为需要显式注入的新模块。

VEGA-3D 反过来问：大规模视频生成模型为了生成跨帧一致的视频，已经被迫学习遮挡、视差、物体恒常性和运动规律；这些中间表征是否能直接作为 label-free 3D prior？这使技术主轴从“构造显式几何 latent”转为“开采生成模型内部的隐式空间知识”。

![](assets/VEGA-3D%20-%20视频生成先验驱动的空间理解/figures/01-paradigm.png)

> **图 1：三种空间增强范式。** (a) 显式 3D 输入受数据规模限制；(b) 几何教师/重建需要复杂监督；(c) VEGA-3D 从冻结生成器提取先验并与语义 token 融合。右侧雷达图展示主要 benchmark 提升。来源：论文 Figure 1，第 1 页，官方 TeX 原图；已用 Read 验证。

### 1.2 论文的三个可检验问题

1. 生成模型的 multi-view feature consistency 是否确实更强，并与下游三维能力相关？
2. 应从哪一个噪声阶段、哪一个网络层读取 feature？
3. 连续物理/几何 feature 如何与离散语义 token 融合，且不损伤 caption/QA 能力？

论文没有停留在“视频模型像 world model”的比喻，而是为第一个问题定义 correspondence score，为第二个问题扫描 diffusion timestep/DiT depth，为第三个问题做多种 fusion 消融。

### 1.3 核心贡献

- 定义跨视角 voxel correspondence score，量化不同 backbone 的隐式几何一致性。
- 把冻结视频生成模型通过噪声激活为 Latent World Simulator，不在训练时生成像素，也不需要几何标签。
- 用 token-level adaptive gated fusion 对齐 generative 与 semantic 两个 manifold。
- 在 ScanRefer、Multi3DRefer、Scan2Cap、ScanQA、SQA3D、VSI-Bench、LIBERO 与真机上验证迁移，并开放代码。

---

## 2. 任务定义与输入输出

### 2.1 MLLM 基线

给定文本 token $x$ 和多视图/视频输入 $V$，语义视觉分支产生

$$v=f_{proj}(f_{enc}(V)),$$

MLLM 用自回归交叉熵训练：

$$\mathcal L_{CE}(\Theta)=-\sum_{i=1}^{L}\log p_{\Theta}(y_i\mid y_{<i},x,v).$$

作者认为这种 vocabulary-space 监督把“左/右错了”仅当作普通 token mismatch，缺少度量几何约束，容易形成 spatial blindness。

### 2.2 VEGA-3D 的输入与输出

输入仍是 $T$ 帧 RGB/multi-view image 和文本问题，不需要推理时点云、深度或相机 pose。语义分支给出 $f_{sem}$；冻结视频生成器给出 $f_{gen}$；融合后送入 LLM，输出 grounding box、caption/answer、空间选择题答案，或交给 VLA action head 输出操作动作。

需要区分两类几何信息：训练/推理主路径不需要显式 3D；但作者为分析 correspondence score，使用 ScanNet 的深度与相机外参把 token 投影到同一 voxel。该显式几何只用于诊断，不是 VEGA-3D 主模型输入。

---

## 3. 核心方法

### 3.1 生成模型作为 Latent World Simulator

Wan2.1 使用 flow matching。给 clean latent $z_0$、噪声 $\epsilon$ 与连续时间 $t$，基础目标为

$$\mathcal L_{FM}(\psi)=\mathbb E\left[\|u_t-v_{\psi}(z_t,t,c)\|_2^2\right].$$

VEGA-3D 不微调 Wan2.1，而是先用其 VAE 编码 $z_0=E(V)$，再沿训练时的线性 noising path 注入噪声：

$$z_k=(1-t_k)z_0+t_k\epsilon,\qquad t_k=\frac{k}{K},\quad K=1000.$$

以空文本 prompt 输入冻结 DiT，读取第 $l$ 层中间特征：

$$f_{raw}=\Phi^{(l)}(z_k,k;c_{text}=\texttt{""}).$$

空 prompt 的目的，是避免文本语义 hallucination，让 feature 尽量由视觉证据与生成器学到的 dynamics 决定。默认选 $k=300$（噪声比 0.3）和第 20 个 DiT block，再池化/展平到每帧 196 token。

“加噪再读 feature”是核心：clean VAE latent 只编码输入，未充分调用 DiT 的恢复机制；极端噪声又摧毁几何。中等噪声迫使模型动用其 learned prior 来恢复结构，因此中间去噪阶段最有空间信息。

![](assets/VEGA-3D%20-%20视频生成先验驱动的空间理解/figures/02-method-pipeline.png)

> **图 2：VEGA-3D 双分支流程。** 右侧冻结 VAE+DiT 读取中间生成 feature，左侧语义 encoder 保留识别能力，二者在进入 LLM 前做 adaptive gated fusion。火焰表示可训练、雪花表示冻结。来源：论文 Figure 4，第 7 页，官方 TeX 原图；已用 Read 验证。

### 3.2 multi-view correspondence score

为检验 feature 是否把同一物理点在不同视角映射到相近位置，作者将 posed RGB token 通过深度与外参放入 0.1 m voxel。对 voxel $m$ 在视角 $v_i,v_j$ 的 feature，定义

$$S_{voxel}^{(m)}=\frac{h_{m,v_i}^{\top}h_{m,v_j}}{\|h_{m,v_i}\|\|h_{m,v_j}\|}.$$

补充材料说明，同一视角落入同一 voxel 的 token 先平均并 L2 normalize；只比较不同视角；最终是按跨视角 pair 加权平均，而非每 voxel 等权。dataset 先按 scene 求 score，再对有效 scene 求均值。

结果显示 DiT 型 Wan2.1-T2V/VACE correspondence 分别 96.88%/97.04%，VGGT 为 77.21%，V-JEPA v2 为 72.00%；SVD 和 SD2.1 仅 17.95%/23.83%。这表明“生成模型天然有三维”不能一概而论：架构、训练规模和长程 attention 很关键，UNet 生成器并不自动产生强跨视角一致性。

### 3.3 Adaptive Gated Fusion

两个分支先用独立 MLP 投影到 LLM hidden dimension：

$$F_{gen}=P_{gen}(f_{gen}),\qquad F_{sem}=P_{sem}(f_{sem}).$$

对第 $i$ 个 spatial token，门控为

$$g_i=\sigma\left(W_g^{\top}\operatorname{Concat}(\operatorname{LN}(F_{gen,i}),\operatorname{LN}(F_{sem,i}))+b_g\right),$$

融合表示为

$$F_i^{fused}=(1-g_i)F_{gen,i}+g_iF_{sem,i}.$$

这不是让 LLM 对两串不同长度 token 再做昂贵 cross-attention，而是在对应 token 位置做标量仲裁：需要定位时提高生成几何权重，需要类别/语言细节时保留语义权重。其隐含假设是两个分支已通过 pooling 对齐到相同 token grid；若 correspondence 不可靠，逐位置 convex combination 也会混入错位特征。

### 3.4 从 world representation 到 action

在 LIBERO 中，作者以 OpenVLA-OFT 为基线，语言 backbone 与 action head 不变，只在 policy learning 前向视觉流注入融合 token。因此 VEGA-3D 对 action 的作用路径是：

$$\text{video prior}\rightarrow\text{geometry-aware visual token}\rightarrow\text{VLA policy}\rightarrow\text{action}.$$

它没有给候选动作滚动未来、没有 action-conditioned simulation，也没有 model predictive control。将其称为“World-Action Model”表达的是生成预训练 feature 可服务 action grounding，而不是经典意义的 world-model planning。

---

## 4. 实验设置与复现成本

### 4.1 三组任务

- **3D scene understanding：** ScanRefer、Multi3DRefer、Scan2Cap、ScanQA、SQA3D；基线 Video-3D LLM。
- **视频空间推理：** VSI-Bench；基线 Qwen2.5VL-7B，训练数据沿用 VG-LLM S1。
- **机器人操控：** LIBERO 四套任务；基线 OpenVLA-OFT；另有 Agilex Piper 真机 Click Bell 20 次试验。

默认每个 scene 均匀采样 32 帧。主实验使用 8×H100，Adam，global batch 128；language model 最大学习率 $10^{-5}$，visual backbone $2\times10^{-6}$。官方代码要求 Python 3.10、PyTorch 2.4.0+cu121、FlashAttention 2.7.4.post1，并依赖 LLaVA-Video-7B-Qwen2、SigLIP、Wan2.1-T2V 1.3B、ScanNet/Video-3D-LLM 数据组织。

### 4.2 实际复现门槛

代码与部分 checkpoint 已公开，优于另外四篇；但“plug-and-play”主要指架构接口，不代表低成本。32 帧×Wan2.1 DiT 中间 feature、7B 级 MLLM 与多任务数据仍需要 H100 级显存。可离线缓存生成 feature，显著降低重复问答成本，但会带来大规模存储与预处理时间。

Figure 7 显示 832×480 输入的未缓存 latency 约 70.35 ms、缓存后约 51.94 ms；1280×720 未缓存约 325.78 ms、缓存后约 109.72 ms。高分辨率 VRAM 仍接近 29 GB。因而实时机器人若场景持续变化，不能完全享受“每 scene 缓存一次”的理想条件。

---

## 5. 定量结果与 Baseline

### 5.1 3D 场景理解

| 方法 | ScanRefer Acc@0.25 / 0.5 | Multi3DRefer F1@0.25 / 0.5 | Scan2Cap C / B-4 | ScanQA C / EM | SQA3D EM | Avg. Rank↓ |
|---|---:|---:|---:|---:|---:|---:|
| Video-3D LLM | 58.1 / 51.7 | 58.0 / 52.7 | 83.8 / 41.3 | 102.1 / 30.1 | 58.6 | 5.1 |
| 3DRS | 62.9 / 56.1 | 60.4 / 54.9 | 86.1 / 41.6 | 104.8 / 30.3 | 60.6 | 2.7 |
| Ross3D | 61.1 / 54.4 | 59.6 / 54.3 | 81.3 / 43.4 | **107.0 / 30.8** | **63.0** | 2.6 |
| **VEGA-3D** | **63.2 / 56.2** | **60.8 / 55.1** | 83.2 / 42.2 | 106.3 / 30.4 | 61.3 | **2.2** |

收益集中在 localization/grounding：ScanRefer Acc@0.5 从 51.7 提到 56.2，Multi3DRefer F1@0.5 从 52.7 提到 55.1。Scan2Cap CIDEr 反而从 83.8 降到 83.2，说明几何 prior 不保证语义描述更好。Ross3D 用显式 BEV reconstruction supervision，在 ScanQA/SQA3D 仍更强；VEGA-3D 的优势是无需显式 3D supervision 且总体 rank 最好。

### 5.2 VSI-Bench 与机器人

VSI-Bench 总平均从 Qwen2.5VL-7B 的 48.9 提到 50.5；相对方向从 39.7 提到 45.1，route planning 43.0→43.1，说明增益并不均匀，几何关系题更受益。

LIBERO 平均成功率从 OpenVLA-OFT 的 97.0% 提高到 97.3%：Spatial 97.5→97.4、Object 98.3→99.4、Goal 97.8→97.0、Long 94.4→95.2。绝对提升很小且有两个 suite 下降，但在高度饱和基线上，Object/Long 的增益支持“精细接触定位与持续空间记忆受益”的解释。真机 Click Bell 从 8/20（40%）提高到 11/20（55%），样本量较小，适合视为可行性证据而非稳定泛化结论。

### 5.3 Backbone 消融

Wan2.1-T2V 在 grounding 指标最好；Wan2.1-VACE 在部分 QA 指标更强。VGGT、DINOv3、V-JEPA v2 也能增强基线，说明改善不完全是“生成模型独有”；真正相关的变量更接近跨视角 correspondence 与 feature stage。论文的 NOS 与 correspondence 呈正相关，但 NOS 在 discriminative/generative 两组内分别 min-max normalization，不能把两组分数直接当作绝对能力比较。

---

## 6. 关键消融：什么真正起作用

### 6.1 中等噪声和中间层

![](assets/VEGA-3D%20-%20视频生成先验驱动的空间理解/figures/03-noise-layer-ablation.png)

> **图 3：噪声比例与 DiT 层消融。** ScanRefer Acc@0.5 在 $t_k=0.3$ 达 56.2，高于 clean latent 的 55.6 和高噪声阶段；第 20 层优于第 10/12/15/25/28 层。来源：论文 Figure 6，第 14 页，官方源图；已用 Read 验证。

这直接反驳“直接拿生成模型最后一层就行”。clean latent 主要是 VAE 编码，深层接近像素渲染目标，早层偏纹理；中间噪声×中间网络深度才兼顾局部位置与抽象 dynamics。选择仍是手工 grid search，跨 backbone 是否通用尚未证明。

### 6.2 融合方式

| 融合 | ScanRefer 0.25/0.5 | Scan2Cap C/B-4 | ScanQA C/EM | SQA3D EM |
|---|---:|---:|---:|---:|
| Baseline | 58.1/51.7 | 83.8/41.3 | 102.1/30.1 | 58.6 |
| Only generative | 54.9/48.3 | 25.2/30.0 | 74.0/21.1 | 52.0 |
| Add | 61.5/54.6 | 81.4/41.6 | 106.3/30.4 | **61.8** |
| Channel-gated | 61.8/54.9 | 82.2/41.9 | 105.7/30.3 | 61.2 |
| **Adaptive gated** | **63.2/56.2** | **83.2/42.2** | **106.3/30.4** | 61.3 |

最重要的负结果是 only-generative 全面低于 baseline：生成 prior 有空间结构，却缺少可靠语义。Cross-attention 也未占优，说明更复杂的融合并不自动更好；对应 token 的轻量门控更适合解决两种 feature 的 distribution gap。

---

## 7. 两条主轴与 planning 判断

### 7.1 隐式三维先验路线的成立条件

VEGA-3D 证明的是“部分大规模 DiT 生成模型的中间 feature 具有可迁移空间一致性”，不是“任何视频模型都是三维模型”。成立条件至少包括：足够强的多视角/视频预训练、全局 attention、合适的噪声阶段、合适的中间层，以及保留语义分支。

与 BEVWorld/SparseWorld 的显式 latent 相比，它不必预定义 BEV 范围、anchor 或相机标定，迁移到室内场景更灵活；但 latent 没有直接可读的米制坐标，无法轻易做碰撞检测、约束优化或可验证 rollout。

### 7.2 是否真正影响 planning

论文最接近 planning 的证据是 LIBERO policy 成功率与真机操作提升，但其机制是**当前观测表征增强**。生成器没有接受候选动作、没有模拟动作后未来，也没有让 policy 在 imagined rollout 中搜索。因而应称为“生成先验辅助 action grounding”，而非 model-based planning。

要证明 world model 真正参与 planning，下一步需要：给候选 action/trajectory 注入生成器或学习 action-conditioned latent dynamics；滚动未来状态；对成功、碰撞、接触和任务进度评分；比较同一视觉 encoder 下有无 rollout 的闭环收益。论文自己也把“扩展到更大 VLA 的闭环 embodied control”列为 future work，说明该问题尚未完成。

---

## 8. 局限、失败案例与 Future Work

### 8.1 作者明确局限

- 生成先验只补充语义 feature，caption 等语义重任务提升不稳定。
- 不同生成 backbone 差异大，DiT 明显优于部分 UNet。
- 冻结视频生成器增加显存和延迟，缓存只能部分缓解。
- 最佳 timestep/layer 依赖手工选择。
- 实验仍以室内多视图为主，动态开放世界覆盖不足。

作者提出将 prior 蒸馏到轻量 encoder、学习自适应 feature extraction、扩展动态开放环境与更大 VLA 的闭环控制。

![](assets/VEGA-3D%20-%20视频生成先验驱动的空间理解/figures/04-failure-case.png)

> **图 4：ScanRefer 失败案例。** 蓝框通常落在合理区域，却无法在拥挤场景中区分近邻同类实例。这说明隐式 prior 提供 coarse spatial anchor，但不保证实例身份与细粒度关系正确。来源：论文 Figure A2，第 23 页，官方 TeX 原图；已用 Read 验证。

### 8.2 进一步判断

1. correspondence score 使用 GT depth/pose 做诊断，主模型虽不依赖它，但指标本身仍受 voxel size、遮挡和 depth validity 处理影响。
2. feature caching 的性能数字更适合静态 scene 多问答；机器人连续控制每帧变化，缓存复用范围有限。
3. LIBERO 基线已 97%，0.3 个百分点平均提升需要更多随机种子、置信区间和非饱和长时任务确认。
4. 空 prompt 降低文本 hallucination，却也可能忽略任务相关 dynamics；未来可研究受控、但不泄漏答案的 action/instruction conditioning。

---

## 9. 复现建议

1. 先按官方仓库复现 ScanRefer 小规模设置，固定 Video-3D LLM 与训练数据，只改变 generative branch。
2. 严格记录 Wan checkpoint、VAE、timestep $k$、DiT block、pooling grid 和输入分辨率；这些参数都会改变结论。
3. 同时报告 grounding 与 caption/QA，防止只挑几何受益指标。
4. 复算 correspondence score 时遵循补充材料的 pair-weighted averaging、same-view merge 与 NaN scene 规则。
5. 若用于机器人，单独测在线 feature extraction latency，不要只引用静态 scene 缓存结果。
6. 若研究 planning，应新增 action-conditioned rollout，而不是把 current-frame feature gain 直接解释为世界模型规划。

---

## 10. 个人启发

这篇论文把“视频生成模型学到物理世界”从宣传语变成可操作方法：先定义能与下游任务关联的内部指标，再定位最有信息的去噪阶段和层，最后设计不会吞掉语义的融合接口。更深的启发是，显式空间 latent 与隐式生成先验并非竞争终局：前者适合约束、碰撞与规划，后者适合规模化预训练。最有希望的系统可能让隐式先验负责感知补全和不确定性，显式 latent 负责可验证 dynamics 与规划。

---

## 11. 材料来源

详见 [sources.md](assets/VEGA-3D%20-%20视频生成先验驱动的空间理解/sources.md)。核心依据为 arXiv v3 官方 PDF（含 Supplementary）、官方 TeX 原图、官方代码与项目页；所有插图均已逐张 Read 验证。