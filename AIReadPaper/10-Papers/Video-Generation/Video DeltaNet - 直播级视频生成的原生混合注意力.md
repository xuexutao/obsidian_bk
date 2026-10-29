---
type: paper-note
status: complete
domain: Video-Generation
paper: "Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation"
year: 2026
arxiv: "2609.20744"
doi: "10.48550/arXiv.2609.20744"
source: "https://arxiv.org/abs/2609.20744"
project: "https://openvdn.github.io/"
code: "https://github.com/OpenVDN/vdn-minimax-h3"
code_commit: "e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b"
weights: "https://huggingface.co/OpenVDN/vdn-minimax-h3"
tags:
  - Video-Generation
  - Efficient-Attention
  - Linear-Attention
  - Diffusion-Transformer
  - DeltaNet
  - MiniMax-H3
created: 2026-10-29
updated: 2026-10-29
---

# Video DeltaNet：直播级视频生成的原生混合注意力

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | *Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation* |
| 作者 | Haocheng Xi, Yiming Xie, Hexu Zhao, Yiwen Zhang, Michael Liu, Thomas Creavin, Kurt Keutzer, Xiuyu Li, Zhaoyang Lv, Chenfeng Xu, Haiwen Feng |
| 机构 | UC Berkeley、Impossible, Inc.、UT Austin |
| 版本 | arXiv v3，2026-10-02；共 21 页 |
| 基座 | MiniMax H3，稠密单流视频/音频 DiT |
| 代码 | [OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3)，已开源训练与推理代码；本文核对 commit `e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b` |
| 权重 | [OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3)，完整下载约 82 GB |
| 许可证 | 代码 Apache-2.0；权重遵循 MiniMax H3 Community License，不能把代码许可证直接套到权重上 |
| 核心任务 | 把视频 DiT 中占主导的全局 Dense Softmax，改造成“局部精确 Softmax + 双向线性记忆”的混合注意力 |

## 一句话结论

**Video DeltaNet 的真正贡献不是“用线性注意力替掉 Softmax”，而是把不同时间尺度的交互明确分工：局部纹理与短程运动交给精确 Softmax，首尾边界交给稠密锚点，远程上下文交给双向逐帧 VDA；再通过分阶段 teacher alignment、8 步 DMD2 蒸馏和硬件定制内核，把 14.3 秒 768p 视频的 DiT 去噪压到 8×B200 上 6.70 秒，同时质量接近 Dense H3。**

> [!important] 先给判断
> - **方法价值高**：VDA 把一帧内强相关 patch 的写入变成联合最小二乘求解，而不是彼此不知情的独立加法；这是比“换一个稀疏 mask”更有结构性的贡献。
> - **工程价值更高**：论文的速度不是单一算法带来的，而是 VDA 2.6–3.2×、50→8 步约 6.25×、8 卡并行约 7×叠加的系统结果。
> - **标题需谨慎解读**：“livestream”主要指生成速度快于播放速度，并不等于严格在线因果生成；模型使用双向扫描、末帧锚点和整段 clip，算法本身是非因果的。
> - **评级：★★★★☆**。方法、代码、权重和系统实现都较完整；扣分点是评测规模仅 103 个 prompt、硬件依赖重、理论稳定性范围有限，而且“近无损”并非所有指标都成立。

---

## 1. 研究背景与问题定义

### 1.1 为什么视频 DiT 的注意力会成为瓶颈

长视频扩散模型在每个去噪步都要重复处理整段时空 token。论文对 MiniMax H3 的 profiling 显示，**Softmax attention 占 denoiser 运行时间的 85% 以上**。若视频 token 数为 $N$，Dense Softmax 需要显式处理 $N^2$ 级别的 query-key 配对；视频更长、分辨率更高时，这一项很快压倒 QKV 投影和 FFN。

线性注意力看似自然：把过去上下文压进固定大小的状态 $S$，每个 token 只需读取该状态，复杂度可随序列长度近似线性增长。但论文指出，直接迁移语言模型中的线性注意力会遇到三个不匹配：

1. **固定容量与视频细节不匹配**：人物身份、场景布局、外观和长程运动都要进入固定状态，而 Dense Softmax 保留了全部 key/value。
2. **逐 token 更新与逐帧并行不匹配**：一帧的所有空间 patch 同时可见，强行规定 patch 顺序没有自然语义；把相关 patch 独立写入又会互相冲突。
3. **随机新分支与预训练残差流不匹配**：直接把随机初始化的线性分支插入预训练 H3，会同时改变信息流和激活尺度，可能在学会记忆前先破坏原能力。

### 1.2 论文要解决的核心问题

目标不是做一个完全线性的模型，而是在给定预训练 Dense H3 的前提下回答：

> 如何只把“昂贵但可压缩”的远程 video-to-video 交互换成线性记忆，同时保住局部细节、多模态条件和预训练能力？

因此 VDN 只重构视频 token 之间的注意力：凡涉及文本或音频的交互仍保留 Softmax。它是一种**按模态和时间距离分工的混合注意力**，不是全模型统一替换。

### 1.3 三项核心贡献

1. **混合视频注意力**：局部窗口 Softmax、首尾全局锚点、双向远程线性记忆三者互补。
2. **Video Delta Attention（VDA）**：把一帧所有空间写入联合求解，并证明固定特征和门控下的继承状态转移非扩张。
3. **完整改造链路**：逐层/端到端 teacher alignment、QKVO LoRA 协同适配、8-step DMD2 蒸馏，以及 SGLang、FP8、融合 kernel 和 8 卡分支并行。

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与默认工作负载

论文的主评测工作负载固定为：

- 输出视频：$1344\times768$，345 帧，24 fps，约 14.375 秒；
- 视频 VAE latent：$(24,102,48,84)$；
- 音频 latent：$(2,32,575)$；
- 文本 embedding：$(L,5120)$，来自 Qwen3-VL；
- 视频 patch size：$(1,2,2)$，所以每个 latent frame 有 $24\times42=1008$ 个视频 token；
- latent frame 数 $F=102$，视频 token 总数为 $102\times1008=102{,}816$。

训练与评测支持 T2VA、I2VA、L2VA 和 FL2VA 等 H3 条件模式。论文速度数字主要报告 DiT 去噪，不应与包含文本编码、VAE 解码、音频处理和封装的完整端到端延迟混淆；官方模型卡补充给出的 8×B200 预热后端到端时间约为 9.0 秒。

### 2.2 关键符号

| 符号 | 含义 |
|---|---|
| $F$ | latent 视频帧数，默认 102 |
| $U=H_\ell W_\ell$ | 每帧空间 token 数，默认 1008 |
| $q_{t,u},k_{t,u},v_{t,u}$ | 第 $t$ 帧第 $u$ 个 token 的 query、key、value |
| $S_t\in\mathbb R^{d_v\times d_k}$ | 处理完第 $t$ 帧后的循环记忆 |
| $\alpha_t$ | 按 key 通道作用的记忆衰减门 |
| $\beta_{t,u}$ | 当前 token 的非负写入门 |
| $A_t$ | 当前帧的加权 key Gram 统计 |
| $B_t$ | 当前帧的 value-key 写入统计 |
| $O^S,O^L$ | Softmax 与 Linear 两分支输出 |

---

## 3. 核心方法

### 3.1 总体框架：按时间角色拆分注意力

![](assets/Video%20DeltaNet%20-%20直播级视频生成的原生混合注意力/figures/01-attention-allocation.png)

> 图 2：Softmax 与 Linear 分支的职责划分。黄色对角带负责局部视频交互，金色首尾行列是边界锚点，蓝色区域由双向线性记忆覆盖；文本和音频相关交互保持稠密。该设计不是近似同一张 attention map，而是把互补区域分给不同算子。  
> 来源：论文 Figure 2，第 3 页，[arXiv:2609.20744](https://arxiv.org/abs/2609.20744)。

VDN 把 video-to-video attention 分成三块：

1. **局部精确区**：查询帧所属 5-frame VAE chunk，加上前后各一个 chunk，共通常 15 个 latent frame，使用 Softmax。
2. **首尾边界锚点**：每帧可访问首、末 latent frame 的全部 token；首、末帧也可访问完整序列，形成 four-way connectivity。
3. **远程互补区**：局部窗口之前和之后的帧分别由前向、反向线性状态汇总。

局部窗口与 H3 VAE 的 5 帧自然 chunk 对齐，避免窗口边界切断 tokenizer 的时间单元。首尾锚点对 I2V/FL2V 尤其有意义：条件帧不必先被压进固定大小记忆，而是保持直接可见。

### 3.2 双向线性记忆与文本初始化

对查询帧 $t$，前向状态汇总局部窗口之前的帧，反向状态汇总窗口之后的帧：

$$o_t^L=S_t^\rightarrow q_t+S_t^\leftarrow q_t$$

两个远程区域互不重叠，首尾锚点也从线性分支输入中移除，因此同一个视频帧不会被重复计算。

Linear 分支还把全部文本 token 汇总成文本状态 $S_T$，并把两个方向初始化为：

$$S_0^\rightarrow=S_0^\leftarrow=\frac12S_T$$

这样两方向读出相加后，文本状态总量约为一份。值得注意的是，文本仍通过 Softmax 全局可见，所以 Linear 分支中的文本状态不是“补漏”，而是额外条件注入。

### 3.3 两分支如何校准并融合

![](assets/Video%20DeltaNet%20-%20直播级视频生成的原生混合注意力/figures/02-vdn-architecture.png)

> 图 3：VDN 混合层与 VDA 分支。两分支共享预训练 QKV，但 Softmax 分支保留 QK Norm 与 RoPE；Linear 分支对 K/V 做短卷积、SiLU 和归一化，使用独立门控、RMSNorm 与输出投影。  
> 来源：论文 Figure 3，第 5 页，[arXiv:2609.20744](https://arxiv.org/abs/2609.20744)。

两分支都复用预训练 QKV 投影，但后处理不同：

- **Softmax 分支**：沿用 H3 的 QK normalization、RoPE、局部窗口和 anchor mask。
- **Linear 分支**：不使用 RoPE；K/V 先经过深度可分离 $5\times5$ 空间卷积和 5-tap 时间卷积，再做 SiLU；Q/K 做 L2 normalization。

由于局部 Softmax 删除了大量远程 key，概率会更集中；Linear readout 的尺度又与 Softmax 不同，因此论文不直接相加，而是分别校准：

$$\widetilde O^S=G^S\odot O^S$$

$$\widetilde O^L=G^L\odot\operatorname{RMSNorm}(O^L)$$

$$Y=\widetilde O^SW_O^S+\widetilde O^LW_O^L$$

其中 $G^S,G^L$ 是内容相关 sigmoid gate，$W_O^S,W_O^L$ 是独立输出投影。Linear recurrence 内部另有 $\alpha$ 和 $\beta$，分别控制旧记忆保留和新关联写入。

### 3.4 从逐 token Delta Rule 到逐帧联合写入

标准 Delta Rule 先衰减旧状态，再按当前 key 的预测残差写入：

$$\bar S_t=S_{t-1}\operatorname{Diag}(\alpha_t)$$

$$S_t=\bar S_t+\beta_t(v_t-\bar S_tk_t)k_t^\top$$

如果一帧的 $U$ 个 patch 都基于同一个冻结旧状态并行更新，可写成：

$$S_t^{\mathrm{batch}}=\bar S_t+\sum_{u=1}^{U}\beta_{t,u}(v_{t,u}-\bar S_tk_{t,u})k_{t,u}^\top$$

定义一帧的两个统计量：

$$A_t=K_t^\top\operatorname{Diag}(\beta_t)K_t$$

$$B_t=V_t^\top\operatorname{Diag}(\beta_t)K_t$$

则冻结状态更新为：

$$S_t^{\mathrm{batch}}=\bar S_t(I-A_t)+B_t$$

问题是：所有 patch 都根据同一个更新前状态计算残差。若多个空间位置的 key 高度相关，它们会重复强化同一方向或产生冲突，却看不到同帧其他位置正在写什么。

### 3.5 VDA：联合正则最小二乘解

VDA 不人为排序 patch，而是直接求一个同时解释整帧关联的新状态：

$$S_t=\arg\min_S\frac12\|S-\bar S_t\|_F^2+\frac12\sum_{u=1}^{U}\beta_{t,u}\|Sk_{t,u}-v_{t,u}\|_2^2$$

第一项要求不要无约束地抹掉旧状态，第二项要求同一个新状态同时拟合当前帧全部 key-value 关系。一阶条件为：

$$S_t(I+A_t)=\bar S_t+B_t$$

闭式解是：

$$S_t=(\bar S_t+B_t)(I+A_t)^{-1}$$

这里真正关键的是 $(I+A_t)^{-1}$。它在 $d_k\times d_k$ 的通道空间中求解，而不是在 $U\times U$ 的空间 token 维度中求解；默认 $d_k=128$，远小于 $U=1008$。该逆矩阵显式感知 key 的相关结构：重叠方向会被更强抑制，正交方向可独立写入。

一个直观极端是：若 $U$ 个位置都使用同一个单位 key $k$、门控均为 $\beta$，则该方向的写入权重为：

$$\frac{U\beta}{1+U\beta}$$

它随重复证据增加而饱和，不会像简单相加那样无限线性增长；若 key 相互正交，每个方向则保持 $\beta/(1+\beta)$，不会因为整帧 token 多而统一被 $1/U$ 缩弱。

### 3.6 非扩张性：理论保证到底保证了什么

把衰减项展开：

$$S_t=S_{t-1}M_t+J_t$$

$$M_t=\operatorname{Diag}(\alpha_t)(I+A_t)^{-1},\qquad J_t=B_t(I+A_t)^{-1}$$

因为 $A_t\succeq0$，$(I+A_t)^{-1}$ 的特征值都位于 $(0,1]$；若 $0\leq\alpha_{t,j}\leq1$，则：

$$\|M_t\|_2\leq1$$

因此对两个不同进入状态的扰动：

$$\|\Delta S_t\|_F\leq\|\Delta S_{t-1}\|_F$$

这比加法更新的 $I-A_t$ 更稳，因为当 $\lambda_{\max}(A_t)>2$ 时，后者可能放大旧状态。

但这个命题的边界必须说清：它只在**当前帧 prepared features 与 gates 固定**时约束继承状态，不保证新写入 $J_t$ 有界，也不证明输入依赖的完整神经网络、扩散轨迹或生成视频全局稳定。

### 3.7 窗口边界与 decay bridge

假设局部 Softmax 窗口是 $[\ell_t,h_t]$。前向状态取窗口左边界之前，反向状态取右边界之后；跨过窗口时不应重复写入局部帧，但要累计它们的衰减：

$$\widetilde S_t=P_{\ell_t-1}\operatorname{Diag}\left(\prod_{r=\ell_t}^{t}\alpha_r\right)+R_{h_t+1}\operatorname{Diag}\left(\prod_{r=t}^{h_t}\alpha_r\right)$$

$$o_{t,u}^L=\widetilde S_tq_{t,u}$$

这就是 `bridge="alpha"`：跨过局部区域只应用衰减，不重复注入局部 token。若窗口覆盖整段视频，代码直接退化回原 Dense Attention，并关闭 Linear 分支，避免重复计算。

---

## 4. 训练流程

### 4.1 A1：逐层 teacher alignment

- 每个 Linear 分支单独训练 200 steps；
- backbone 和 Softmax gate 冻结；
- Softmax gate 固定为初始化值 0.99；
- 每层梯度裁剪 0.1；
- 目标是先让单层新分支对齐冻结 Dense teacher 的局部行为。

A1 虽短，却是参数变化最大的阶段。Table 4 显示写入门 $\beta$ 相对位移达到 35.6%，而 1.93B 参数的输出投影 $W_O$ 变化仅 1.25%，说明主要是在建立新分支的门控和状态动力学，而不是大幅改写宽投影。

### 4.2 A2：组装后的端到端对齐

- 所有校准过的 Hybrid Attention 层装回完整网络；
- 端到端训练 500 steps；
- 预训练权重和 Softmax gates 继续冻结；
- 全局梯度裁剪 1.0。

这一阶段修正逐层独立校准看不到的跨层组合误差。

### 4.3 B：QKVO LoRA 协同适配

- 训练 2,000 steps；
- 对 DiT 各层及 token refiner 的 Q/K/V/O 投影加入 rank 64、alpha 64 的 LoRA；
- 联合训练 Linear 分支、Softmax gates 与 LoRA；
- FFN、embedding 和输出 head 仍冻结。

这种顺序很重要：如果一开始就允许 Softmax gate 降低，优化器最容易走的捷径是压低原分支来减少误差，而不是让新分支学会有用的远程记忆。

### 4.4 50 步到 8 步蒸馏

架构适配完成后，作者再做 DMD2 风格蒸馏：

- teacher 是 VDN-H3 自己的 50-step sampler，而不是 Dense H3；
- 不使用 GAN term；
- student 从社区 MiniMax-H3-Turbo-LoRA 初始化；
- generator、real-score、fake-score 共享 FSDP backbone，通过 adapter 区分角色；
- 每次 generator 更新配 3 次 fake-score 更新；
- 最终 checkpoint 训练 250 个 generator steps。

这把两种误差拆开了：A1/A2/B 解决“Dense→Hybrid”，DMD2 解决“50→8 步”。因此 8-step 质量下降不能简单归因于 VDA。

---

## 5. 数据集与实验设置

### 5.1 数据与训练环境

- 10,015 个预编码视频片段；
- 每段 1344×768、345 帧、24 fps；
- 训练不包含 VAE 和文本 encoder，直接读取缓存 latent/embedding；
- PyTorch 2.13、CUDA 12.9；
- H200/B200 集群，FSDP2/HSDP、activation checkpointing、pinned-memory activation offload；
- 参数 gather 使用 bf16，梯度 reduce 与 decay-$\alpha$ 模块使用 FP32。

数据量对“训练一个视频 foundation model”很小，但这里做的是冻结大部分 H3 后的结构适配，不应与从头预训练的数据规模等量比较。

### 5.2 基线与指标

50-NFE 比较 Dense H3、训练免费稀疏基线 Radial Attention 和 Stage-B VDN-H3；8-NFE 比较 Dense H3、蒸馏 VDN-H3 和 FastH3 v2。所有模型使用相同的 103 个固定第三方 prompt。

主要指标包括 EvalCrafter VQA 美学/技术质量、Q-Align、DOVER++、FIRM 的 Instruction Following / Perceptual Quality / World Coherence，以及 FL2VA 的 PSNR/SSIM。RAFT mean flow 只衡量运动幅度，不是越高越好的质量分数。

---

## 6. 实验结果

### 6.1 质量：接近 Dense，但不是全面无损

![](assets/Video%20DeltaNet%20-%20直播级视频生成的原生混合注意力/figures/03-quality-comparison.png)

> 图 5–6：50 NFE 与 8 NFE 的质量比较。50 步 VDN-H3 在多数指标上接近 Dense H3，并在相近延迟下优于 Radial；8 步 VDN-H3 在与 FastH3 v2 共享的 7 项指标上全部更高，但 FL2VA 数据并不完整。  
> 来源：论文 Figure 5–6，第 9 页，[arXiv:2609.20744](https://arxiv.org/abs/2609.20744)。

50 NFE 下，VDN-H3 在 VQAA、DOVER++ 和 FIRM Perceptual Quality 上高于 Dense H3；落后项的差距较小：Q-Align 0.08、World Coherence 0.03、FL2VA PSNR 0.02 dB、SSIM 0.0024。相对延迟匹配的 Radial Attention，VDN 在 7 个共同质量指标中的 6 个更高。

8 NFE 下，相对 FastH3 v2：

- 7 个共同质量指标全部更高；
- VQAT +10.78，DOVER++ +2.67；
- 单 B200 还快约 1.2×。

不过“near-lossless”更适合理解为总体工程结论，而不是每个指标逐项无损：VDN 对 Dense 的部分指标仍有小幅下降，Radial/FastH3 也缺少 FL2VA 完整结果。

### 6.2 效率：三个加速杠杆叠加

| 配置 | GPU | NFE | B200 延迟 | 相对单卡 Dense 累计加速 |
|---|---:|---:|---:|---:|
| Dense H3 | 1 | 50 | 799.6 s | 1.0× |
| VDN-H3 | 1 | 50 | 307.9 s | 2.6× |
| + few-step | 1 | 8 | 49.3 s | 16.2× |
| + distributed | 8 | 8 | 6.70 s | 119.3× |

H200 上对应的单次 backbone 加速为 3.2×，8 卡 8 步去噪为 12.5 秒。注意 119.3×/141.6× 是相对**单卡** Dense H3；论文摘要中的 14.5× 是“相同 8 卡数量”的口径，二者不能混用。

当 latent frame 从 42 增加到 102 时，Softmax density 从 42.08% 降到 19.98%，B200 attention speedup 从 1.7× 增至 3.0×。这说明 VDN 的优势确实随视频变长而增长，而不是只来自固定常数优化。

### 6.3 架构消融

| 方法 | VQAA | VQAT | Q-Align | FAST-VQA | DOVER++ | FIRM WC |
|---|---:|---:|---:|---:|---:|---:|
| VDN | 30.56 | 90.11 | 92.69 | 83.47 | 3.62 | 1.83 |
| 去掉 Joint VDA solve | 30.14 | 89.42 | **92.88** | 83.32 | **3.76** | 1.76 |
| 去掉 Boundary anchors | 29.81 | 88.76 | **92.93** | 83.34 | 3.50 | 1.79 |
| 仅 Window Softmax | 28.24 | 87.01 | 92.18 | 81.57 | 3.24 | 1.82 |

联合 VDA 在多数指标上更好并有更干净的稳定性性质，但并非全面获胜：加法更新的 Q-Align 和 DOVER++ 反而更高。去掉 anchor 后 8 项中 7 项下降，但 Q-Align 上升 0.24。最强证据来自“无 anchor VDN vs 仅 Window Softmax”：只移除远程 Linear 分支，就使 VQAA -1.57、VQAT -1.75、FAST-VQA -1.77，说明局部窗口确实无法独自恢复远程上下文。

### 6.4 Kernel 级消融

![](assets/Video%20DeltaNet%20-%20直播级视频生成的原生混合注意力/figures/04-kernel-ablation.png)

> 图 7：核心算子的融合优化。VDA-Prep、Gather、Epilogue 和 inverse 都显著加速；但单卡 chunk scan 反而变慢，只有 8 卡 head sharding 后才有效。  
> 来源：论文 Figure 7，第 11 页，[arXiv:2609.20744](https://arxiv.org/abs/2609.20744)。

关键结果：

- VDA-Prep：H200 18.0→1.6 ms，B200 17.1→3.4 ms；
- VDA-Gather：7.2×/7.5×；
- VDA-Epilogue：7.3×/9.1×；
- 小矩阵 inverse：H200 7.8→1.7 ms，B200 6.3→1.3 ms；
- 8 卡 chunk scan：H200 4.6→1.1 ms，B200 3.0→0.8 ms；
- 单卡 chunk scan 会增加组合开销，并非所有配置都更快。

这说明论文的速度主张高度依赖系统协同：仅有公式而没有融合 kernel、FP32 小矩阵路径、FP8 宽 GEMM、head sharding 与异步双分支，很难复现最终数字。

---

## 7. 与相关工作的关系

### 7.1 相比稀疏 Softmax

Radial Attention 等方法仍在原 QK token 对上做选择，只是跳过一部分配对；优点是无需新状态，缺点是远程信息依赖稀疏模式是否覆盖。VDN 则把远程上下文压成递归状态，计算随帧数线性增长，但引入固定状态容量和训练适配问题。

### 7.2 相比 Gated DeltaNet / Kimi Delta Attention

语言模型通常逐 token 更新，token 顺序天然存在。VDA 的创新在于把“一帧”当成更新单元，用联合解处理相关空间写入；这是视频结构本身带来的归纳偏置。

### 7.3 相比 SANA-WM

SANA-WM 的并行帧更新需要按 $1/\sqrt U$ 缩放 key，以约束加法转移。VDA 通过 $(I+A_t)^{-1}$ 自适应处理谱结构，不需要额外 frame-size scaling，并避免所有方向都被统一削弱。

### 7.4 相比纯 few-step 蒸馏

FastH3 主要减少 NFE，但单次 backbone 仍承担昂贵 Attention。VDN 同时压低“每步成本”和“步数”，因此两种收益可乘；这是其比单纯 4/8-step 模型更有系统意义的地方。

---

## 8. 局限与批判性评价

### 8.1 “Livestream”不是在线因果生成

VDN 使用双向扫描、末帧全局 anchor，并一次性处理完整视频 clip；FL2VA 还显式依赖末帧条件。因此这里的“livestream”更接近**生成速度快于视频播放速度**，不是边接收输入边因果输出的在线流式生成。若迁移到无限流或没有未来帧的场景，反向状态和末帧 anchor 都要重设计。

### 8.2 速度高度依赖顶级硬件与专用实现

Headline 结果依赖 8×B200、MXFP8、SGLang、FlashAttention varlen、定制 CUDA/Triton kernel 和 NVLink 分支并行。模型卡虽提供 24 GB 单卡 offload 路径（345 帧约 22 GB，FP8 约 20 GB），但这解决的是“能运行”，不是“直播级速度”。

### 8.3 评测覆盖偏窄

- 训练适配数据仅 10,015 段；
- 评测仅 103 个固定 prompt；
- 主要只测 768p、14.3 秒、固定宽高比和帧率；
- 没有更长视频、不同分辨率、普通消费卡、系统性人类偏好或安全评测；
- 论文没有集中展示失败案例。

因此“随长度扩展”的效率趋势有证据，但质量是否在 30 秒、1 分钟甚至更长视频保持，仍未验证。

### 8.4 固定状态的信息瓶颈仍存在

论文自己也承认纯线性替换会明显掉质量。最终保留局部 Softmax、首尾 anchor、文本/音频稠密交互和独立门控，本质上说明固定大小状态无法完全代替 Dense Attention 的细粒度可寻址记忆。

### 8.5 理论性质不等于生成稳定性

非扩张证明只约束固定特征和门控下的 inherited state。它没有约束新写入、输入依赖门控、跨层残差、扩散多步误差或最终视频运动稳定性。理论贡献应准确表述为“局部状态转移性质”，不能扩大成全模型稳定性保证。

### 8.6 结果口径容易误读

6.70 秒是 DiT denoising；119.3× 是 8 卡 VDN 相对单卡 Dense；14.5× 才是同 GPU 数量口径；模型卡最新 SGLang 数字为 6.9 秒去噪、约 9.0 秒端到端。报告时必须同时注明 GPU 数、NFE、是否包含解码和基准配置。

---

## 9. 复现与实践建议

### 9.1 最小可用路径

官方仓库和 Hugging Face 提供 Diffusers `ModularPipeline`。345 帧在 24 GB GPU 上可逐 block offload；权重总量约 82 GB，包括：

```text
ckpts/
├── h3-base/            # MiniMax H3 基座，约 72 GB
├── stage-b-step-2000/  # 50-step VDN，约 4.3 GB
└── stage-dmd-step-250/ # 8-step VDN，约 5.1 GB
```

若只验证画质和接口，优先使用 Diffusers；若复现论文速度，需要官方 tuned config 或 SGLang serving，而不是普通 eager PyTorch。

### 9.2 训练复现难点

1. 数据预编码格式必须匹配 H3 的 packed sequence 和 token tags；
2. A1 是逐层 teacher trajectory 对齐，不是普通端到端 finetune；
3. $A_t$ 必须 FP32 并显式对称化，否则真实视频 patch 的强相关性会让 Cholesky 数值失稳；
4. 50→8 步使用外部 Turbo LoRA 初始化和多 adapter DMD2，不能只从论文公式推回完整训练；
5. 论文未公开原始 10,015 段训练数据索引，配置中的 `index_file` 仍需用户准备。

### 9.3 我会如何做低成本消融

若资源有限，建议按下列顺序验证：

1. 单层 A1 alignment，先确认新分支能拟合 teacher 激活；
2. 在短视频和较小 $F,U$ 上比较 `vdn_solve` 与 `sana_scaled`；
3. 固定 50 NFE，只测 Dense→Hybrid，避免与蒸馏误差混杂；
4. 再加入 8-step adapter；
5. 最后才做 FP8、Ulysses 和 SGLang 性能优化。

---

## 10. 个人启发与后续问题

### 10.1 “帧级联合写入”可迁移到其他组结构

VDA 的本质是：当一组 token 天然同时出现且高度相关时，不应假装它们是独立序列写入，而应在组内联合求解。这个思想可迁移到：

- 多视角图像的一组相机视图；
- 3DGS 同一局部邻域的一组 Gaussian；
- mesh 中同一 face/ring 的 vertex 或 edge token；
- VLA 中同一时刻多传感器观测。

对 3D 资产建模尤其值得追问：是否可以把“同一局部拓扑邻域”视为 frame-like group，用 Gram-aware 联合更新避免相关几何 token 重复写入。

### 10.2 混合注意力的关键不是比例，而是职责可解释

VDN 的局部/边界/远程划分比固定稀疏率更有意义，因为每部分对应具体物理角色：纹理与短程运动、全局首尾约束、长程身份与布局。后续工作可以进一步学习动态窗口或内容自适应 anchor，但应保留这种职责解释，而不是只优化 FLOPs。

### 10.3 值得继续验证的问题

1. 取消末帧可见后，VDN 能否变成真正因果的在线视频生成？
2. 固定 $128\times128$ 状态在更长视频上何时出现容量拐点？
3. anchor 是否可以从固定首尾扩展为内容自适应关键帧？
4. VDA 的联合解能否与低秩状态、分块状态或多尺度状态结合？
5. 在相同 8 卡预算下，系统加速来自算法、通信布局和硬件特性的比例如何跨平台变化？

---

## 11. 代码级实现

### 11.1 仓库概览与论文模块映射

本文检查官方仓库 commit：`e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b`（2026-10-04）。核心映射如下：

| 论文模块 | 代码入口 | 作用 |
|---|---|---|
| Hybrid Attention 总控 | `src/models/hybrid_attention.py::HybridAttention` | 共享 QKV、运行 Window Softmax 与 Linear branch、独立投影后相加 |
| 双向 Linear branch | `src/models/linear_attention/branch.py::BidirectionalLinearBranch` | 特征准备、逐帧统计、双向 scan、边界 gather、RMSNorm 与 gate |
| VDA 联合解 | `src/models/linear_attention/delta_rule.py::VdnDelta` | 计算 $(I+A)^{-1}$ 对应 transition 与 injection |
| 帧统计与双向扫描 | `src/models/linear_attention/scan.py` | 构造 $A,B$，执行前后向 recurrence，应用 decay bridge |
| 局部 Softmax | `src/models/softmax_attention/` | window mask、FlexAttention / decomposed / reference backend |
| 三阶段训练 | `src/training/train_stage_a1.py`、`train_stage_a2.py`、`train_stage_b.py` | teacher alignment 与 LoRA co-adaptation |
| DMD2 蒸馏 | `src/training/train_stage_dmd.py`、`dmd.py` | 50→8 NFE 蒸馏 |
| 8 卡分支并行 | `src/inference/utils/ulysses*.py` | uneven all-to-all，Softmax/Linear 分支分配不同 rank |

### 11.2 默认全链路张量总表

以下形状来自论文默认 345 帧、1344×768 配置与代码注释；文本长度 $L$ 可变：

| 阶段 | 张量形状 | 说明 |
|---|---|---|
| 视频 VAE latent | $(24,102,48,84)$ | channel、latent frame、高、宽 |
| patch 后每帧网格 | $(24,42)$ | patch size $(1,2,2)$ |
| 每帧视频 token | $U=1008$ | $24\times42$ |
| 全视频 token | $102{,}816$ | $F\times U$ |
| residual stream | $[N,5376]$ | 代码注释约 $N\approx105k$ |
| raw Q/K/V | $[N,56,128]$ | 56 heads，head dim 128 |
| 帧内 Q | $[102,1008,56,128]$ | `query_by_frame` |
| 帧内 K/V | $[102,56,1008,128]$ | 为 batched matmul 调整布局 |
| 写入门 $\beta$ | $[102,56,1008]$ | 每 token、每 head 一个标量 |
| $A_t$ | $[102,56,128,128]$ | FP32 key Gram 统计 |
| $B_t$ | $[102,56,128,128]$ | value-key 写入；本模型 $d_v=d_k=128$ |
| $\alpha_t$ | $[102,56,128]$ | 每帧、每 head、每 key 通道衰减 |
| 前/后向状态库 | 各 $[102,56,128,128]$ | 代码注释单库约 0.35 GiB，双库约 0.7 GiB |
| Linear readout | $[102,1008,56,128]$ | 与 query 相乘后的远程读出 |
| 展平 Linear 输出 | $[102816,7168]$ | 56×128，再投影到 5376 |

这里有一个容易忽视的点：QKV 投影宽度 $56\times128=7168$ 大于 residual hidden size 5376；因此 Linear branch 的独立输出投影是 $7168\rightarrow5376$，论文 Table 4 标注该宽矩阵约 1.93B 参数并不意外。

### 11.3 逐模块拆解

#### 11.3.1 共享 QKV 与分支分流

**输入输出**：`HybridAttention._qkv()` 接收 $[N,5376]$，生成 raw 与 QK-Norm+RoPE 两套 $[N,56,128]$ 表示。

**为什么这样设计**：Linear branch 复用 QKV，避免再付一次最宽投影成本；但它读取 pre-QK-norm、pre-RoPE 特征，自行做 SiLU/L2Norm/ShortConv。

**关键超参**：`anchor_frames="both"`，`chunk=5`，`radius=1`，`linear_head_dim=128`。

**代码与论文差异**：代码对 full-cover 窗口直接调用原始 attention backend，使输出精确退化为 teacher；这是论文正文没有展开的正确性保护。

#### 11.3.2 帧统计 $A,B$

**输入输出**：K/V 从 $[F,H,U,d]$ 与 $\beta:[F,H,U]$ 归约为 $A,B:[F,H,d,d]$。

**为什么这样设计**：把 1008 个空间 token 压成两个 $128\times128$ 小矩阵，后续 scan 的成本与空间 token 数解耦。

**关键超参**：`STATS_CHUNK_FRAMES=16`；$A$ 保持 FP32，$B$ 可由 bf16 matmul 后提升。

**代码与论文差异**：代码额外执行 $A\leftarrow(A+A^\top)/2$。注释说明真实视频 patch 强相关，bf16 非对称误差会破坏 $I+A$ 的正定性；这是论文理论成立到工程稳定之间的关键补丁。

#### 11.3.3 VDA factorization 与 scan

**输入输出**：每帧的 $A,B,\alpha$ 生成 transition/injection，随后递推：

$$S_t=S_{t-1}M_t+J_t$$

**为什么这样设计**：把昂贵 factorization 批量放在 scan 外，串行部分每帧只剩一个小矩阵乘和加法。

**关键超参**：发布 checkpoint 使用 `delta_rule="vdn_solve"`，`a_fp32=true`，`bridge="alpha"`。

**代码与论文差异**：仓库通用 PyTorch 路径的 `VdnDelta` 使用 Cholesky + triangular solve 构造逆矩阵；论文部署路径描述融合 CUDA Gauss–Jordan，并称直接输出 transition/injection。也就是说，**公式一致，但 reference/training 路径与 headline serving kernel 不是同一实现**；普通仓库运行不能自动等价复现论文 6.70 秒。

#### 11.3.4 文本状态与边界 gather

**输入输出**：全部文本 token 作为一个 chunk 写成 $S_T$，两个方向各从 $0.5S_T$ 开始；窗口外状态经 decay bridge 汇合后供每个 query 读取。

**为什么这样设计**：远程状态也带 prompt 条件；只跨衰减、不跨写入，保持 Softmax/Linear 视频区域互补。

**关键超参**：`TEXT_STATE_SCALE=0.5`，首尾 anchor 在 Linear 输入中物理切片移除，而不是只 mask 输出。

**代码与论文差异**：代码注释明确承认文本会在 Softmax 和 Linear 中各出现一次，Linear 文本状态是条件增强，不是严格 complement；论文图示容易让人误以为所有区域完全不重叠。

#### 11.3.5 多卡分支并行

**输入输出**：Ulysses 将 head 分给不同 rank，再做 all-to-all 恢复序列布局。

**为什么这样设计**：Softmax 与 VDA 算子形态不同，均匀 head sharding 未必负载平衡；B200 tuned config 用 5 张卡执行 Softmax、3 张卡执行 Linear，H200 用 6+2。

**关键超参**：B200 `softmax_ranks=5`，8 NFE，FP8，decomposed Softmax backend。

**代码与论文差异**：论文正文主要写“按 attention head 切分”，代码表明最终是**分支专用的不均匀 rank 分配**，且不同 GPU 架构的最佳比例不同。

### 11.4 计算账与复现建议

对每帧每 head，VDA 的主要数学量是：

- 统计：$A=K^\top\operatorname{Diag}(\beta)K$ 与 $B=V^\top\operatorname{Diag}(\beta)K$，复杂度约 $O(Ud^2)$；
- factorization：$(I+A)^{-1}$，复杂度 $O(d^3)$；
- scan：状态乘 transition，约 $O(d^3)$；
- readout：每个 query 读取 $d\times d$ 状态，约 $O(Ud^2)$。

默认 $U=1008,d=128$，所以空间归约与 readout 仍很重，但它们对帧数是线性的，不再发生跨帧 $O((FU)^2)$ 配对。真正部署时，显存流量、布局转换和 kernel launch 与 FLOPs 同样重要，这正是论文大量系统优化存在的原因。

复现时建议明确区分三档目标：

1. **数学正确性**：使用 reference/PyTorch backend，对齐 Dense teacher；
2. **单卡可运行**：Diffusers + block offload，接受速度较慢；
3. **论文级性能**：B200/H200 tuned config、FP8、融合 kernel、Ulysses 或 SGLang。

不要拿第 1/2 档的 eager 延迟否定论文，也不要把第 3 档 8×B200 数字当作普通单卡可获得的算法速度。

---

## 12. 材料来源

- 官方论文与图表来源见同目录 `assets/Video DeltaNet - 直播级视频生成的原生混合注意力/sources.md`。
- 官方代码核对版本：`e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b`。
- 本笔记中的实验数字以 arXiv v3 PDF 为准；模型卡 6.9 秒/9.0 秒作为更新后的 serving 补充口径单独标注。
