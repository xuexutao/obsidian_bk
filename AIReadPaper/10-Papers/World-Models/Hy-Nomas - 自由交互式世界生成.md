# Hy-Nomas：自由交互式世界生成

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文名称 | Hy-Nomas: Towards Free-Form Interactive World Generation |
| 团队 | 腾讯混元（Tencent-Hunyuan） |
| 发布时间 | 2026-10-23（HuggingFace 权重 11:18 UTC、代码 09:01 UTC、Bench 09:10 UTC 同日上传） |
| 正式论文 | **暂无**。模型卡标注 "Technical Report (Coming Soon)"，指向的 PDF 实际 404（2026-10-24 核实）；无 arXiv 条目 |
| 许可 | Apache-2.0（Qwen 文本编码器遵循其自身许可） |
| 官方渠道 | [HuggingFace](https://huggingface.co/tencent/Hy-Nomas) · [GitHub](https://github.com/Tencent-Hunyuan/Hy-Nomas) · [Demo](https://hy-nomas.hy.tencent.com/) · [Hy-Nomas-Bench](https://huggingface.co/datasets/tencent/Hy-Nomas-Bench) |
| 阅读材料 | HF 模型卡 + GitHub README + Bench README + 推理配置文件（均已归档至 `sources/`） |

> ⚠️ **这是一次"先权重、后论文"的反常发布**。截至 2026-10-24，没有任何官方博客、arXiv 论文或评测分数；第三方（OrcaRouter、AlphaSignal、TSN）的整理文章是主要的信息补充。本笔记基于**官方模型卡与代码仓库**，所有架构数字都可追溯到配置文件，但训练数据、训练 recipe、消融与 benchmark 结果**全部空缺**。技术报告发布后需要回来更新。

---

## 1. 研究背景：它和 text-to-video 的本质区别

Hy-Nomas 官方定义为"支持自由形式多轮交互（文本指令 + 相机控制）的视频世界生成模型"。它与普通文生视频的关键差异在于**世界的持续性**：

- 普通文生视频：一个 prompt 返回一段孤立的 clip，想改内容就得重开、且与上一段无关；
- Hy-Nomas：把世界当成**一串 segment 的链**逐段生成，每一段都条件在 KV cache 里的全部历史上。用户可以在生成过程中**中途**引入物体、改风格、改场景、描述动作、推动相机——而不是在开头一次性写完 prompt。

一个容易误读的数字：**24 FPS 是输出视频的播放帧率，不是生成速度**。生成延迟取决于硬件、分辨率、segment 长度和历史长度，官方未公布任何延迟数据。

### 1.1 在混元自家产品线中的位置

| 模型 | 形态 | 与 Hy-Nomas 的区别 |
|---|---|---|
| [[WorldPlay - 长期一致的实时交互世界模型]]（混元世界模型 1.5） | 实时交互视频世界模型 | 键盘/鼠标连续输入 + 重构上下文记忆 + Context Forcing 蒸馏，主打实时性；Hy-Nomas 转向 segment 自回归 + 文本自由交互 |
| [[HYWorld 1.5 - 开放世界交互生成模型]] | 交互世界模型 | 前代交互线 |
| [[HunyuanWorld 1.0 - 分层可探索三维世界生成]] / [[HY-World 2.0 - 多模态三维世界生成重建与模拟]] | 3D mesh / Gaussian splat 世界 | 输出的是**显式 3D 资产**，Hy-Nomas 输出的是**流式视频**——两者不可混淆 |

---

## 2. 核心架构

![](assets/Hy-Nomas%20-%20自由交互式世界生成/figures/01-framework-dual-stream-mmdit.png)

> Hy-Nomas 框架图。三条路径：**紫色**为文本（结构化 prompt → MLLM → Token Refiner），**蓝色**为视频（历史 video clip → 3D VAE Encoder → Patch Embed，与噪声拼接后进双流块，经 3D VAE Decoder 输出 Generated Segment k），**绿色**为相机（Camera Trajectory → Pose Encoder → Camera Proj → Camera Cross-Attn）。中间是 ×N 个 Dual-Stream MMDiT Block（推理配置 N=54），核心为 Segment-wise Causal Joint Attention（3D RoPE），顶部 KV Cache 携带全部历史 segment，Timestep Embed 通过调制注入噪声等级。来源：官方模型卡 `assets/method_framework_h.png`。

### 2.1 三路条件通道

**文本通道**：文本编码器取自 `Qwen2.5-VL-7B-Instruct` 的 language-model 组件，输出 3,584 维隐状态，经**两层 token refiner** 投影到 DiT 的 2,048 维隐空间。每段有独立的 segment-specific prompt，用户可以在生成中途改变要求的内容。Web demo 里还挂了一个独立的 **Nomas-PE** 提示词增强服务（18 分片、共 55.6 GB），把用户随意输入的指令转成结构化生成 prompt——demo 必需，离线推理可直接读准备好的 prompt 文件。

**相机通道**：这是它比"在 prompt 里用文字描述运镜"更硬核的地方——相机是**独立于语言的条件通路**。归一化的 4×4 相机外参矩阵（camera-to-world）按 4 帧一组、展平为 64 维输入，与视频 latent 的时间压缩对齐；经学习的 pose embedding 和投影后，在每个双流块的 video MLP 之后，由**视频 token 对相机特征做 cross-attention**（`per_chunk_camera_attn=true`、`camera_chunk_size=1` 时对齐到单个 latent 帧）。推理时相机轨迹是 `[N,4,4]` 的 `c2ws` 矩阵 + `[N,3,3]` 内参，配套轨迹生成工具；平移/旋转强度默认 1.0/2.5。文本与相机可以同时用且相互独立——"市场即将散场"的文字和"沿过道走过去"的轨迹是两条可组合的控制。

**视频通道与潜空间**：使用 **DCAE-V `f32t4c32`** 潜空间——32 通道、空间每维 32× 压缩、时间 4× 压缩。这段压缩编码器是开源的 DC-AE 视频版（`dc-ae-v-1.0-f32t4c32-bf16.pt`，约 4.3 GB）。

### 2.2 Segment 自回归 DiT 与终身 KV cache

主干是**稠密多模态扩散 Transformer（MMDiT）**，配置如下（摘自官方推理配置）：

| 属性 | 配置 |
|---|---|
| Transformer blocks | **54 个双流块，0 个单流块** |
| Hidden size | 2,048 |
| 注意力头 | 16 头 × 128 维 |
| FFN | 4× 扩展（8,192） |
| Latent 输入/输出通道 | 32 / 32，patch 大小 1×1×1（t×h×w） |
| 位置编码 | **3D RoPE**，时间/高/宽维度 [16, 56, 56]，theta 256 |
| 去噪目标 | **Flow velocity**（流匹配），Euler ODE 采样 |

每个双流块内部：文本流与视频流先做 **segment-wise causal joint attention**（3D RoPE 编码视频位置，本层 KV cache 提供此前所有 segment 的上下文），随后各自过独立的 MLP，视频分支再经 **Camera Cross-Attn** 查询相机特征。时间步调制（learned shifts/scales/residual gates）作用于双流。段间注意力是**因果的**——生成当前段不需要未来段；推进到下一段时缓存的 K/V 直接复用。

**记忆方案**：标准配置保留**完整历史**（`max_cache_segments=null`，不滑窗），缓存随生成持续增长；单卡配置则把 GPU 上的活跃历史限制在 **18 个 latent 帧**（时间 4× 压缩，即约 72 帧/3 秒），更早的历史归档到 CPU 内存、可继续增长。

对比一下这个领域的记忆谱系就很清楚：[[Matrix-Game 3.0 - 长时记忆实时交互世界模型]]、[[WorldCrafter - 隐式三维记忆视频世界模型]]、[[DecMem - 解耦记忆的分钟级一致世界生成]]、[[GIM-World - 几何感知隐式记忆]]、[[SPMem - 长期空间记忆视频世界模型]] 都在给 KV cache 之外造显式记忆模块或 3D 索引，Hy-Nomas 的答案是**最朴素的：什么都不加，就靠终身 KV cache + 完整历史**——代价是缓存无界增长，好处是零额外模块、对象/光照/几何一致性完全由注意力上下文承载。它是否真能撑住长程空间记忆（回访场景不漂移），官方没有给数据，这正是 Bench-4 想测但没给分数的部分。

### 2.3 解码与蒸馏推理

- **解码器**：微调版 **PixelUMM**（36 层、hidden 4,096、32 query 头 / 8 KV 头），以生成的 DCAE-V latent 为条件重建像素，RGB 按 4×16×16 tubelet 处理；**1 个解码步**，时间窗口 10 latent 帧、重叠 5 帧（与 DiT 的 8 个去噪步相互独立）。另有 FP8 量化版解码器。
- **蒸馏推理**：默认 **8 步去噪**（代码支持 4/6/8 步），Euler ODE，text/camera CFG 均为 1.0（蒸馏模型不需要强引导）。输出默认 **720×1280 @ 24 FPS**。
- **参数量**：官方从未公布。fp32 DiT 权重 34.2 GB → 按每参数 4 字节推算约 **8.5B**（9.06 GB 的 INT8 版与此一致）——这是第三方由文件大小做的算术，不是官方口径。

### 2.4 交互协议

离线推理的 prompt 是结构化 JSON：`Style`、`perspective_view`、`subject_list`（主体 id + 属性），以及按 `start_time`/`end_time`（`MM:SS.mmm`）排布的 `segments`——首段建立世界，后续段用 `description_change` 描述演化，`contained_actions` 带各自时间戳描述主体动作。segment 长度由事件时间戳推导（`inference.video_length=event`），官方示例是"发光森林里的小狐狸，第 5 秒起路旁绽开蓝色发光花"。

---

## 3. 工程与部署

标准配置用 **8 路序列并行**（`sp_size=8`）+ 分布式 PixelUMM 解码，默认一组 8 卡。更有意思的是**单卡 RTX 4090 配置**，四板斧组合：

1. **INT8 存储 + BF16 计算**：DiT 权重与 KV cache 都存 INT8，线性层与注意力计算保持 BF16；
2. **有界历史**：GPU 活跃历史截为 18 latent 帧（`temporal_context` 采样），旧历史归档 CPU；
3. **CPU offload**：文本编码器、DiT、解码器按各自处理阶段在 CPU/GPU 间搬运；
4. **单卡算子**：DiT 用 FlexAttention，解码用 CUTLASS FP8，关闭序列并行（`sp_size=1`）。

注意两点：单卡声明只覆盖**视频推理**，demo 必需的 Nomas-PE 要另算资源；CPU-GPU 搬运有延迟代价，且 18 帧历史预算不限制当前段的显存峰值。

### 3.1 微调管线（设计得很讲究）

官方 SFT 流程把**训练与部署解耦**：LoRA 在**非蒸馏 base checkpoint** 上用 segment forcing 管线训练（rank/alpha=256、零 dropout、Muon 优化器、恒定学习率 1e-4、2,000 步），训完再**合并进蒸馏 checkpoint**——这样合并后的模型仍保留 4/6/8 步的快速推理能力。数据侧提供 captioning harness（逐段打标 + DCAE-V 编码 + 打包 Arrow/Parquet，相机轨迹归一化后挂载）和离线序列 packing（示例 `max-seq-len 360000`，DP16×SP8 用 128 卡）。官方还放出两个 LoRA 适配器：**Cinematic**（视觉质量）和 **Robotics**（在 RealSource-World 真实双臂机械臂数据上训练，增强机器人仿真能力）——后者值得 VLA 方向的人注意，见 §5.3。

---

## 4. Hy-Nomas-Bench：有考卷、没成绩

配套发布的评测集含 **4 个基准、280 个用例、220 条参考相机轨迹**，专测连续世界生成中的**交互可控性与一致性**（而非画质）：

| 基准 | 用例 | 考察点 | 轮数/段数 | 平均时长 |
|---|---:|---|---:|---:|
| Bench-1 多轮文本 | 60 | prompt 跟随、物理合理性 | 2–6 轮 | 37.88 s |
| Bench-2 相机导航 | 100 | 相机控制 | 4–20 轮 | 27.05 s |
| Bench-3 文本相机交错 | 60 | 三者综合 | 2–7 轮 | 30.59 s |
| Bench-4 空间记忆 | 60 | **回访一致性** | 8 轮 | 18.00 s |

设计上有几个可圈可点之处：Bench-2 覆盖 49 个场景、8 种基础相机动作和 24 种双键组合，每个固定动作都配等时长**逆操作返回**；Bench-4 的每个第一人称场景做 4 组"出发/返回"，用相同有效相机位姿下生成帧的对比来度量物体与场景一致性——这正是 [[Orbia - 视频世界模型三维一致性评测]] 强调的"新生成内容回访持久性"思路的官方化。场景覆盖真实/游戏/CG/动画风格，prompt 以英文为主（Bench-1 含少量中文）。

**但官方没有公布任何分数**——模型卡原话："Architecture and sampling defaults are configuration details, rather than benchmark results."。这换来一个可验证的立场：别人可以自己跑 280 个用例检验它，包括与那些"自报最佳成绩"的闭源世界模型对比。截至 2026-10-24 尚无第三方跑分。

---

## 5. 局限与评价

### 5.1 信息空缺清单（技术报告前需当"待验证"处理）

- **无训练信息**：预训练数据、规模、课程、蒸馏方法（"segment forcing" 只在 LoRA 微调语境被提到）、post-training 全部未知；
- **无评测分数、无延迟数据**：连自家模型在自家 Bench 上的成绩都没有；
- **参数量为第三方推算**（~8.5B）；
- 无消融、无与 WorldPlay/Matrix-Game/Lingbot-World 的对比；
- 权重 196.4 GB 全量下载门槛不低（其中 Nomas-PE 就占 55.6 GB）。

因此这篇笔记**暂不给 ★ 评级**，待技术报告发布后回来补全。以官方已披露信息的完备程度看，它更像一个**工程开源包**而不是一篇论文。

### 5.2 我的判断

**架构上没有新东西，选择很有态度。** 54 双流 MMDiT + 3D RoPE + cross-attention 相机条件 + 流匹配，每一个组件都是现成的；它的态度在于**组合方式与工程下限**：终身 KV cache 而不是记忆模块（对比 [[Matrix-Game 3.0 - 长时记忆实时交互世界模型]]、[[WorldCrafter - 隐式三维记忆视频世界模型]] 的显式 3D 记忆路线）、独立相机通路而不是文本内嵌控制（对比 [[Yume 1.5 - 文本控制的交互式世界生成]]）、把单卡 4090 作为一等公民支持而不是事后优化。官方敢于发布"无分数 + 可复现考卷"的组合，在同行普遍自报最佳成绩（如 Odyssey-3 的 best-of-8）的环境里反而是一种表态。

**长程一致性的隐忧。** 纯 KV cache 记忆的时间成本随段数线性增长（标准配置），单卡配置的 18 潜帧预算≈3 秒——超出这个窗口的历史只能进 CPU 归档，等于"记得存了但看不清"。空间记忆（回访一致性）恰恰需要远历史，Bench-4 的 8 轮、18 s 场景设计得比较温和，真正的分钟级回访表现存疑。这与 [[Context as Memory - 历史上下文驱动的场景记忆]]、[[SPMem - 长期空间记忆视频世界模型]] 等工作要解决的问题直接相关。

### 5.3 与我研究方向的连接

- **Robotics LoRA 是个信号**：混元在交互世界模型上预置了具身适配器（真实双臂数据训练、合并进蒸馏 checkpoint 部署）。世界模型 → VLA 的路径正从"论文概念"变成"模型卡里的一个目录"。世界模型的视频先验 + 少量真实机器人 LoRA，对 [[VPP - 预测性视觉表征的通用机器人策略]] 一类工作的数据杠杆值得持续跟踪。
- **记忆与 3D 状态的张力**：Hy-Nomas 的"记忆"全是 2D 注意力上下文，没有任何显式 3D 几何状态。而相机通路的存在说明它隐式学到了 ego-motion 与场景结构的对应。如果它能在 Bench-4 上拿高分，就说明"足够大的因果注意力上下文"可以替代显式 3D 记忆——这本身是一个值得验证的科学问题；如果拿不高，则支持 WorldCrafter/GIM-World 的几何记忆路线。
- **与 [[EASI - 多模态模型空间智能统一评测]] 的互补**：EASI 测 VLM 的空间理解，Hy-Nomas-Bench 测世界生成的空间记忆，"理解侧"与"生成侧"的空间智能评测正在各自成型。

## 6. 附件与溯源

- `figures/01-framework-dual-stream-mmdit.png`：官方架构图（HF 模型卡 `assets/method_framework_h.png`，2910×1200 原图），2026-10-24 获取。
- `sources/`：HF 模型卡（HF_README.md）、GitHub README.md / README_CN.md、Hy-Nomas-Bench README（BENCH.md）原文快照，获取日期 2026-10-24。
- 技术报告 `Hy_Nomas_Tech_Report.pdf` 在 GitHub 仓库中**尚为 404**；发布后应下载归档并补全本笔记的训练/评测章节。
