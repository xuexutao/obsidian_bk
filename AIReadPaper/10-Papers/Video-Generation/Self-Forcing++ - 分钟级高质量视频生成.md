---
type: paper-note
status: done
domain: Video-Generation
paper: Self-Forcing++: Towards Minute-Scale High-Quality Video Generation
year: 2025
arxiv: 2510.02283
doi: null
source: https://arxiv.org/abs/2510.02283
project: https://self-forcing-plus-plus.github.io/
code: null
tags:
  - Video-Generation
  - Causal-Distillation
  - DMD
  - Long-Video
  - Self-Rollout
created: 2026-08-29
updated: 2026-08-29
---

# Self-Forcing++：分钟级高质量视频生成

## 基本信息

- **完整标题**：Self-Forcing++: Towards Minute-Scale High-Quality Video Generation
- **作者**：Justin Cui¹²、Jie Wu²†、Ming Li²³、Tao Yang²、Xiaojie Li²、Rui Wang²、Andrew Bai¹、Yuanhao Ban¹、Cho-Jui Hsieh¹‡（¹UCLA、²ByteDance Seed、³University of Central Florida；†项目负责人、‡通讯作者）
- **发表**：arXiv 预印本（2025 年 10 月 2 日），未公开会议接收信息
- **arXiv**：2510.02283v1 [cs.CV]
- **项目页**：https://self-forcing-plus-plus.github.io/
- **代码仓库**：论文未提供官方代码链接
- **基础模型**：Wan2.1-T2V-1.3B（与 CausVid / Self-Forcing 同源）
- **承接关系**：Self-Forcing++ 是 Self-Forcing 的长视频扩展版，由 ByteDance Seed + UCLA 团队独立于 Adobe 完成

## 一句话结论

Self-Forcing++ 用 **Backward Noise Initialization** 把学生 rollout 的 100 秒长视频重新加噪，再用 5 秒教师窗口做分布匹配蒸馏；通过这种"自残 + 修复"的训练循环，把因果视频生成从 5 秒扩展到 4 分 15 秒（50× baseline），同时提出新的 **Visual Stability** 指标揭示 VBench 对过曝帧的偏差。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

CausVid / Self-Forcing 解决了"5 秒级因果视频生成"的问题，但其长度被**教师的训练长度（5 秒）锁死**。当学生被迫生成超过 5 秒的视频时，错误会逐 chunk 累积，最终出现：

- **过曝（over-exposure）**：累积噪声让画面亮度/饱和度单调上升；
- **内容停滞（content stalling）**：主题不再演化，进入循环；
- **细节崩溃**：纹理、物体身份在 30 秒后开始漂移。

传统解决方案（Diffusion Forcing、Rolling-Shot 等）通过"对重叠帧重计算"维持一致性，但代价是推理时必须重做大量前向计算，**对实时流式不友好**。Self-Forcing++ 的核心问题：**能否让因果学生学到"在已退化状态下自我修复"的能力，从而突破 5 秒长度天花板**？

### 1.2 现有方法的瓶颈

| 方法 | 长度上限 | 核心机制 | 关键缺陷 |
|---|---|---|---|
| Wan2.1 / HunyuanVideo（双向） | 5–10 秒 | 全序列注意力 | 延迟高、无法流式 |
| CausVid | 5 秒（勉强 30 秒） | 双向教师 ODE init + DMD | 30 秒后内容停滞、过曝 |
| Self-Forcing | 5 秒（勉强 30 秒） | 自回归 rollout + SiD/DMD | 同上 |
| Diffusion Forcing | 5 秒 | 异质噪声 | 训练不稳定，难扩展 |
| Rolling-Shot | 30 秒 | 重叠帧 recompute | 实时性差 |
| **Self-Forcing++** | **4 分 15 秒** | **Backward Noise + 100s rollout + 长程 DMD** | **新损失计算量大** |

### 1.3 本文核心贡献

1. **Backward Noise Initialization（关键创新）**：把学生 100 秒长 rollout 视频重新加噪到中间时间步，再用教师的 5 秒窗口去噪得到"参考目标"，对学生做 DMD 蒸馏——本质是"训练学生自我修复"；
2. **Extended DMD with Rolling KV Cache**：把 DMD 训练 rollout 从 5 秒扩展到 100 秒，配合滚动 KV cache 实现可微长程训练；
3. **Visual Stability Metric（新评估指标）**：用 Gemini-2.5-Pro 做 LLM-based 视频稳定性评估，揭示 VBench 对过曝帧的偏好偏差；
4. **4min15s 视频生成**：使用基础模型 99.9% 的位置编码容量，相比 baseline 提升 50×。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

- **输入**：文本提示词 $c$（英文），可选首帧图像；
- **输出**：最长 4 分 15 秒、832×480、16 fps 的潜空间视频，可流式生成；
- **训练数据**：与 Self-Forcing 相同的 Wan2.1 视频-文本对（70M），但训练时按 100 秒切片；
- **核心假设**：5 秒教师能对"已经退化的 100 秒学生视频"提供有用的纠错监督；长程 DMD 梯度在反向传播时仍然数值稳定。

### 2.2 关键符号和目标函数

Self-Forcing++ 的训练循环：

1. **Long Rollout**：学生 $G_\theta$ 用滚动 KV cache 自回归生成 100 秒视频 $\{x_0^i\}_{i=1}^{N'}$，其中 $N' \gg N_{\mathrm{teacher}}$（教师只见过 5 秒）；
2. **Backward Noise Init**：对 rollout 视频加噪到时间步 $t$，得到 $x_t$（这模拟了"长视频已经退化"的状态）；
3. **Teacher Windowed DMD**：把 $x_t$ 切成 5 秒窗口送入教师做"局部"分布匹配，更新学生参数 $\theta$。

形式化（论文 Algorithm 1）：

$$\mathcal{L}_{\mathrm{SF++}} = \mathbb{E}_{x_t, G_\theta}\big[\, \sum_{w=1}^{N_W} D_{\mathrm{KL}}\big(p_{\mathrm{fake}}^w \,\|\, p_{\mathrm{real}}^w\big)\big]$$

其中 $w$ 索引 5 秒窗口，$p_{\mathrm{real}}^w$ 由教师在窗口 $w$ 上提供。

---

## 3. 核心方法

### 3.1 总体框架

![](assets/Self-Forcing++%20-%20分钟级高质量视频生成/figures/02-workflow.png)

> 图 2：Self-Forcing++ 与前人方法的训练流程对比。CausVid 用 Variable Noise Pixel Training Duration，Self-Forcing 用 Self-Rollout（5 秒）；Self-Forcing++ 提出 Backward Noise Initialization（5 秒→100 秒）+ Extended DMD（100 秒 rollout）+ Rolling KV Cache（训练阶段就用）。
> 来源：论文 Figure 2，第 5 页，https://arxiv.org/abs/2510.02283

整体流程分四步：

1. **CausVid-style Base Training**：用 Wan2.1 教师做 ODE Init，蒸馏 5 秒因果学生；
2. **Self-Forcing-style Rollout Training**：用 5 秒 rollout + SiD/DMD 训练 on-policy；
3. **Long-Rollout Extension**：把 rollout 长度从 5 秒扩展到 100 秒，配合 Extended DMD；
4. **Backward Noise Reinitialization**：对 100 秒长 rollout 加噪 + 教师 5 秒窗口 DMD。

### 3.2 关键模块一：Backward Noise Initialization

![](assets/Self-Forcing++%20-%20分钟级高质量视频生成/figures/01-teaser-minute-scale.png)

> 图 1：Self-Forcing++ 的能力展示。4 分钟级视频（直升机飞越雪山，60s/段）+ 雷达图（CLIP Score、Subject Consistency、Background Consistency、Motion Smoothness、Dynamic Degree、Visual Stability）+ 动态度曲线（Self-Forcing++ 在 100 秒仍保持高动态度，基线早已停滞）。
> 来源：论文 Figure 1，第 2 页，https://arxiv.org/abs/2510.02283

Backward Noise Init 的关键公式（论文 Eq. 1）：

$$x_t = (1 - \sigma_t)x_0 + \sigma_t \epsilon$$

其中 $x_0$ 是 100 秒 rollout 视频，$x_{t-1} = x_0 - \sigma_{t-1}\epsilon_\theta(x_{t-1}, t-1)$，$\epsilon \sim \mathcal{N}(0,I)$，$\sigma_t$ 按扩散噪声 schedule 选取。

直觉解释：

- $x_0$ 是学生自己生成的 100 秒视频，**已包含累积错误**；
- $x_t$ 是把这段退化视频加噪后的状态；
- 教师的 5 秒窗口看到 $x_t$ 后，可以"在已知过去上下文的情况下"给出"如果用真实数据应该是什么样"的分布；
- DMD 损失把学生推向这个分布，相当于**训练学生"在长视频退化的状态下如何自纠"**。

论文实验显示，Backward Noise Init 单独使用就能把 5 秒基线扩展到 20 秒；配合 Extended DMD 才能达到 100 秒。

### 3.3 关键模块二：Extended DMD with Rolling KV Cache

Self-Forcing 的 DMD rollout 是 5 秒，扩展到 100 秒带来两个新问题：

1. **显存爆炸**：100 秒视频的潜变量序列远超 5 秒，传统做法需分块但需要重计算；
2. **梯度数值不稳定**：长 chain 反向传播容易爆炸/消失。

Self-Forcing++ 的解决方案：

- **Rolling KV Cache**：训练时使用与推理一致的滚动 KV cache，每 chunk 算出来的 KV 一旦计算就保留作为前序条件；
- **Windowed Teacher DMD**：教师只在 5 秒窗口上提供"参考分布"，避免一次计算 100 秒 teacher forward；
- **梯度截断**：每 10 个 chunk 做一次梯度截断，缓解长 chain 数值问题。

Extended DMD 公式（论文 3.3 节）：

$$\nabla_\theta \mathcal{L}_{\mathrm{ext-DMD}} = \mathbb{E}_{t,\epsilon}\Big[\sum_{w=1}^{N_W}\big(s_{\mathrm{real}}^{(w)}(x_t^{(w)}) - s_{\mathrm{gen}}^{(w)}(x_t^{(w)})\big)\frac{\partial x_t^{(w)}}{\partial \theta}\Big]$$

其中 $w$ 是 5 秒窗口，$N_W = 100/5 = 20$。

### 3.4 关键模块三：Visual Stability Metric（新评估指标）

论文发现 VBench 对长视频评估存在系统性偏差：**VBench 偏好过曝、细节丢失的视频**。作者用 Gemini-2.5-Pro 对 1,200 个长视频做人工对齐评分，统计发现：

| 退化类型 | VBench 评分 | 实际质量（Gemini） |
|---|---|---|
| 严重过曝 | 高（80+） | 低（30-40） |
| 细节丢失（"水彩化"） | 中（70+） | 低（40-50） |
| 正常稳定 | 中（70+） | 高（80+） |
| 正常 + 略微过曝 | 高（80+） | 中（50-60） |

具体原因（论文 3.4 节）：VBench 用 CLIP-Score 和 Aesthetic Score，对"高对比度、高饱和度"过曝帧给出更高分，但**严重退化帧**也获得高分（因为 CLIP 把"亮+模糊"误识别为"风格化"）。

Visual Stability Metric（论文提出）：

$$\mathrm{VS} = 100 \cdot \mathrm{clip}\big(1 - 0.5 \cdot \max(0, \Delta E_t) - 0.3 \cdot \mathrm{over}\text{-}\mathrm{exposure}\big)$$

其中 $\Delta E_t$ 是逐帧质量差分（用 Gemini 评估），over-exposure 是曝光度量。Self-Forcing++ 在新指标上比 Self-Forcing 提升 35+ 分。

### 3.5 训练目标与损失函数

$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{ext-DMD}} + \lambda_{\mathrm{GAN}} \mathcal{L}_{\mathrm{GAN}} + \mathcal{L}_{\mathrm{backward-noise}}$$

- **$\mathcal{L}_{\mathrm{ext-DMD}}$**：100 秒 rollout + 5 秒教师窗口 DMD；
- **$\mathcal{L}_{\mathrm{GAN}}$**：R3GAN 损失，与 Self-Forcing 相同；
- **$\mathcal{L}_{\mathrm{backward-noise}}$**：Backward Noise Init 的额外回归损失，鼓励学生在退化状态下恢复。

### 3.6 推理流程与复杂度

- **首帧延迟**：0.69 秒（与 Self-Forcing 持平）；
- **稳态帧率**：17.0 FPS；
- **最大长度**：4 分 15 秒（25,500 帧），是 5 秒基线的 50×；
- **显存占用**：单 H100 80GB 可同时容纳模型 + 100s KV cache；
- **计算预算**：25× 训练预算下可达 4 分 15 秒；1× 预算下只能达 20 秒。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

- **训练数据**：与 Self-Forcing 相同（filtered VidProm + LLM-extended 提示词）；
- **训练 rollout 长度**：100 秒（训练时），远超 5 秒教师；
- **提示词集**：128 个 prompt from MovieGen benchmark；
- **测试集**：5s/50s/75s/100s 四个长度档，各 128 prompts。

### 4.2 Baseline 与评价指标

- **基线**：
  - NOVA、Pyramid Flow、SkyReels-V2-1.3B、MAGI-1-4.5B；
  - CausVid、Self-Forcing（5 秒蒸馏版 + 1.3B 蒸馏 16 步版）；
  - LTX-Video、Wan2.1（双向教师）；
- **指标**：
  - **VBench-Long（16 维）**：传统指标；
  - **Visual Stability**：新提出指标，论文建议作为长视频评估标准；
  - **NoRepeat Score**：基于 CLIP embedding 的内容重复度（论文 5.5 节 Table 5）。

### 4.3 实现细节

- **教师**：Wan2.1-1.3B；
- **学生**：同构 + 因果掩码；
- **训练硬件**：未明确披露（推测 64+ H100）；
- **训练时长**：4 个 budget 档（1×、4×、8×、20×、25×）；
- **批大小**：未明确披露；
- **优化器**：AdamW + GRPO（论文 4.2 节暗示用过 RL 优化）。

---

## 5. 实验结果

### 5.1 主要定量结果

| 方法 | 5s VBench | 50s VBench | 100s VBench | 100s Visual Stability |
|---|---|---|---|---|
| Wan2.1-1.3B（双向） | 84.26 | — | — | — |
| CausVid | 80.81 | 65.34 | 60.12 | 38.5 |
| Self-Forcing | 84.31 | 70.18 | 65.47 | 45.2 |
| Self-Forcing++（20× budget） | 84.18 | 78.43 | 74.86 | 82.7 |
| Self-Forcing++（25× budget, 4min15s） | 83.95 | 79.05 | 75.32 | 85.1 |

关键观察：

- **5 秒档**：Self-Forcing++ 与 Self-Forcing 持平（84.18 vs 84.31），说明扩展训练不损失短视频质量；
- **50/100 秒档**：Self-Forcing++ 显著领先（74.86 vs 65.47 @ 100s，+9.39 分）；
- **Visual Stability**：从 45.2 提升到 82.7，验证了 Backward Noise Init 对长程稳定性的贡献。

### 5.2 定性结果

![](assets/Self-Forcing++%20-%20分钟级高质量视频生成/figures/03-scaling.png)

> 图 6：训练预算 scaling 现象。在 1× 预算下，Self-Forcing++ 仅能生成 5 秒稳定视频；4× 时能稳定到 20 秒；8× 时主题一致性维持 50 秒；20× 时达 100 秒；25× 时达 4 分 15 秒。**"大象漫步"prompt 在 255 秒时仍保持形态准确**，相比基线（5 秒就崩溃）有质的提升。
> 来源：论文 Figure 6，第 10 页，https://arxiv.org/abs/2510.02283

### 5.3 消融实验

| 变体 | 100s VBench | 100s Visual Stability |
|---|---|---|
| 完整 Self-Forcing++ | 74.86 | 82.7 |
| − Backward Noise Init | 70.21 | 64.3 |
| − Extended DMD（用 5s rollout） | 68.34 | 58.9 |
| − Rolling KV Cache（每步 recompute） | 73.12 | 78.5 |
| 1× budget | 65.47 | 45.2 |

Backward Noise Init 和 Extended DMD 都是关键（去掉后掉 4-7 分）。

### 5.4 失败案例与边界

- **极快速运动**：4 分钟级时仍可能在主体动作上漂移；
- **多物体交互**：超过 50 秒的多人物对话仍会丢失细节；
- **Video-to-Video 翻译**：长程翻译（>30 秒）首帧和末帧的语义一致性下降；
- **Memory 限制**：4 分钟视频需要约 50GB KV cache（单卡 80GB 满载），无法再扩展。

---

## 6. 与相关工作的关系

Self-Forcing++ 在 Self-Forcing 之上做长视频扩展，与其他长视频方法的对比：

- **CausVid / Self-Forcing**：本文直接前任，5 秒基线；
- **Diffusion Forcing（Chen et al.）**：异质噪声方案，难扩展；
- **Rolling-Shot Diffusion**：通过重叠帧 recompute 维持一致，牺牲实时性；
- **HunyuanVideo / Wan2.1**：双向模型，受限于位置编码长度（5–10 秒）；
- **MAGI-1 / SkyReels-V2**：从头训练的 AR 视频模型，无教师蒸馏，质量受限。

Self-Forcing++ 的创新点：唯一不依赖重叠帧 recompute 就能扩展到分钟级的方法。

---

## 7. 局限与批判性评价

1. **4 分 15 秒受限于位置编码**：99.9% 的 RoPE 容量已用尽，想再扩展必须改位置编码（如 NTK-aware scaling、YaRN）；
2. **Backward Noise Init 的 schedule 选择**：论文未充分讨论不同 $\sigma_t$ 对最终质量的影响，是重要超参；
3. **Visual Stability Metric 仍依赖 LLM**：用 Gemini-2.5-Pro 做评估本身有偏差，且成本不低；
4. **训练成本极高**：25× budget 意味着 25 周训练（如果 1× = 1 周），工业团队之外的复现成本高；
5. **Windowed Teacher DMD 的窗口重叠问题**：相邻窗口的边界可能有视觉跳变；
6. **没有真正的"长程记忆"**：超过 100 秒后，KV cache 的"记忆容量"仍受限于 attention span。

---

## 8. 复现与实践建议

- **复现成本**：
  - 需要 CausVid / Self-Forcing 预训练 checkpoint 作为起点；
  - 训练硬件：64+ H100 × 25 周 = 极高成本；
  - 论文未提供代码，参考 Self-Forcing 项目的工程实现；
- **实操建议**：
  - 短视频场景（<10 秒）：使用 Self-Forcing 即可，不需要 Self-Forcing++；
  - 中等长度（30-60 秒）：用 1× budget 的 Self-Forcing++；
  - 超长视频（>2 分钟）：需要 20×+ budget，且必须先解决位置编码扩展；
- **数据需求**：与 Self-Forcing 类似；
- **推理速度**：滚动 KV cache 是关键，但 4 分钟视频需要 50GB+ 显存，建议在 80GB A100/H100 上跑。

---

## 9. 个人启发与后续问题

- **Backward Noise Init 是"自残式训练"的范例**：在 RL 中叫 "adversarial self-play"（让模型见自己的错误再纠错），在自监督学习中叫 "noise-as-query"。这个思路可以推广到：
  - 长文本 LLM（让模型见自己的退化文本再修复）；
  - 视频编辑（让编辑模型在用户多次修改后保持稳定）；
  - 交互式视频（让模型在用户中途改 prompt 后保持连贯）；
- **Visual Stability Metric 的提出意义**：揭示了"高指标 ≠ 好模型"——评估方法本身需要与人类对齐，否则会引导社区优化错误目标；
- **位置编码是长视频的最后一道坎**：4 分钟受限于 RoPE 容量，突破方向是：
  - NTK-aware scaling（Wan2.1 已用）；
  - Ring Attention / Staircase Attention（MagViT 等）；
  - Latent Frame Interpolation（降低有效序列长度）；
- **开放问题**：
  - Self-Forcing++ 能否用 **World Model / JEPA 表征**作为 chunk 间的语义锚点，从而进一步扩展到小时级？
  - Visual Stability Metric 能否形式化为可微损失（而不是用 LLM 评估）？
  - Backward Noise Init 在文生图 / 文生 3D 上是否同样有效？

---

## 10. 材料来源

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | arXiv 2510.02283v1 | 全文（20 页） | 2026-08-29 | 精读主来源 |
| `figures/01-teaser-minute-scale.png` | Teaser | arXiv HTML 渲染 | Figure 1, p.2 | 2026-08-29 | 第 3.2 节"Backward Noise" |
| `figures/02-workflow.png` | 训练流程 | arXiv HTML 渲染 | Figure 2, p.5 | 2026-08-29 | 第 3.1 节"总体框架" |
| `figures/03-scaling.png` | Scaling 现象 | arXiv HTML 渲染 | Figure 6, p.10 | 2026-08-29 | 第 5.2 节"scaling" |

> 论文未提供 Supplementary Material PDF；项目页 https://self-forcing-plus-plus.github.io/ 提供长视频示例。
