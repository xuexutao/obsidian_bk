---
type: paper-note
status: complete
domain: VLM
paper: "InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency"
year: 2025
arxiv: "2508.18265"
doi: null
source: "https://arxiv.org/abs/2508.18265"
project: null
code: "https://github.com/OpenGVLab/InternVL"
tags:
  - VLM
  - reinforcement-learning
  - visual-token-compression
  - multimodal-deployment
  - InternVL
created: 2026-10-15
updated: 2026-10-15
---

# InternVL3.5：级联强化学习与视觉压缩

## 基本信息

- **模型族**：dense 1B/2B/4B/8B/14B/38B；MoE 20B-A4B、30B-A3B、241B-A28B。语言侧来自 Qwen3 或 GPT-OSS，视觉侧为 InternViT-300M/6B。
- **训练规模**：预训练约 116M 样本、250B token；SFT 约 56M 样本、130B token；MPO 约 200K preference pairs；在线 RL 使用约 70K query 的 MMPR-Tiny。
- **核心新增**：Cascade RL = MPO 离线强化学习 + GSPO 在线强化学习；ViCO 训练 ViR；DvD 解耦视觉和语言部署。
- **重要性判断**：★★★★★。InternVL3.5 不只追求准确率，而是把训练稳定性、视觉 token 成本和线上服务吞吐作为同一系统问题处理。

**一句话结论：**

InternVL3.5 的技术主线可以概括为：**先用低成本、可共享 rollout 的 MPO 把策略推入高质量区域，再用在线 GSPO抬高上限；同时用 ViCO 让模型适应 256/64 token 两档视觉压缩，再由 ViR 按 patch 选档，最后通过 DvD 将视觉计算与自回归语言计算拆到不同服务器并行执行。**

---

## 1. 研究背景与问题定义

### 1.1 两个同时恶化的瓶颈

InternVL3 已通过原生多模态预训练、MPO 和 VisualPRM 获得强能力，但更复杂的数学、GUI、具身和高分辨率任务暴露两类瓶颈。第一，纯离线偏好优化便宜而稳定，却受既有 rollout 质量限制；纯在线 RL 能探索新轨迹，却采样昂贵、容易不稳定。第二，高分辨率、多图和视频会让视觉 patch 数暴涨，ViT 计算、LLM prefill 和视觉 token KV cache 同时增加。

InternVL3.5 因而并行解决“能力上限”和“推理成本”：Cascade RL 负责推理，ViR/ViCO 负责视觉 token 压缩，DvD 负责服务端调度。

![](assets/InternVL3.5%20-%20级联强化学习与视觉压缩/figures/01-teaser.png)

> 图 1：InternVL3.5 在多模态综合、推理、文本和 agentic benchmark 汇总分上的位置。该图用于展示覆盖面；不同模型列的评测来源包含官方结果与复现结果，不应视为完全同条件竞赛。来源：论文 Figure 1，第 2 页。

### 1.2 任务输入输出

输入仍是文本与单图、多图或视频组成的消息。Dynamic High Resolution 先按宽高比切分 image patches；每个 patch 进入 InternViT 后原始表示有 1024 个视觉 token。普通 InternVL3.5 经 pixel shuffle 压为 256 token；Flash 版本允许进一步压到 64 token。视觉 token 经 MLP 接入 LLM，输出文本答案、推理轨迹、GUI action、SVG 或具身任务回答。

强化学习阶段的输入是问题 $x$，输出为完整响应 $y$；reward 可来自可验证答案或任务评分。在线阶段同一问题采样 $G$ 个响应，以组内归一化 reward 计算优势。部署阶段输入消息在 vision server 分离图像与文本：视觉特征通过网络传到 language server，与文本上下文融合后完成 prefill/decoding。

---

## 2. 模型架构与阶段训练

### 2.1 架构与模型规模

InternVL3.5 延续 ViT–MLP–LLM，而非原始 InternVL 的 QLLaMA。1B–14B 与 20B/30B MoE 使用 InternViT-300M；38B 与 241B-A28B 使用 InternViT-6B。最大型号的语言模型为 Qwen3-235B-A22B，加 5.5B 视觉参数后总计 240.7B、激活约 28B，因此“A28B”包含语言与视觉活跃参数。

### 2.2 原生预训练与 SFT

预训练仍只在文本 token 上计算 next-token loss：

$$\mathcal{L}_i=-\log p_\theta(x_i\mid x_{<i})$$

样本有效输出长度为 $N$ 时使用 square averaging：

$$\mathcal{L}'_i=\frac{w_i}{\sum_jw_j}\mathcal{L}_i,\quad w_i=N^{-0.5}$$

约 250B token 的预训练数据中，纯文本:多模态约为 1:2.5，最大序列 32K。SFT 约 130B token，纯文本:多模态约为 1:3.5；新增带详细过程的 Thinking 数据，并对过程清晰度、冗余和格式严格过滤，同时扩展 GUI、具身与 SVG 能力。

![](assets/InternVL3.5%20-%20级联强化学习与视觉压缩/figures/02-training-process.png)

> 图 3：InternVL3.5 先经历约 250B token 原生预训练、约 130B token SFT，再以约 270K RL 样本完成 MPO+GSPO；Flash 版本另加约 30B token 的 ViCO。来源：论文 Figure 3，第 5 页。

---

## 3. Cascade RL：离线暖启动再在线精炼

### 3.1 为什么级联

离线 RL 使用固定 rollout，样本收集与参数更新解耦，便于共享数据、稳定训练并降低 reward hacking；但上限受数据覆盖限制。在线 RL 根据当前 policy 持续采样，更能修正自身分布和探索高奖励区域，但 rollout 成本高，弱初始策略还会产生低质量样本，使 reward 方差大。

Cascade RL 先用 MPO 把模型推到更高质量、较稳定的区域，再用 GSPO 继续在线优化。它不是把两种 loss 同时相加，而是**阶段性 coarse-to-fine**：前一阶段改善后一阶段的采样分布。

### 3.2 阶段一：MPO 离线强化学习

MPO 继承 InternVL3 的三项损失：

$$\mathcal{L}_{\mathrm{MPO}}=w_p\mathcal{L}_p+w_q\mathcal{L}_q+w_g\mathcal{L}_g$$

其中 DPO 学 chosen/rejected 相对偏好，BCO 学每个响应的绝对质量，LM loss 保持 chosen response 的生成能力。MMPR-v1.2 提供约 200K preference pairs。其 rollout 可跨模型复用，所以 1B 到 241B 型号不必各自承担完整在线采样成本。

### 3.3 阶段二：GSPO 在线强化学习

对问题 $x$ 采样 $G$ 个响应 $y_i$，reward 组内标准化：

$$\widehat{A}_i=\frac{r(x,y_i)-\operatorname{mean}(\{r(x,y_j)\}_{j=1}^{G})}{\operatorname{std}(\{r(x,y_j)\}_{j=1}^{G})}$$

GSPO 的 clipped objective 为：

$$\mathcal{L}_{\mathrm{GSPO}}=\mathbb{E}\left[\frac1G\sum_{i=1}^{G}\min\left(s_i(\theta)\widehat{A}_i,\operatorname{clip}(s_i(\theta),1-\varepsilon,1+\varepsilon)\widehat{A}_i\right)\right]$$

关键是序列级 importance ratio 使用逐 token ratio 的几何平均：

$$s_i(\theta)=\exp\left(\frac1{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}\mid x,y_{i,<t})}\right)$$

几何平均减弱序列长度对 ratio 的直接放大。论文强调 GSPO 不需要 reference model，且对 dense 和 MoE 均有效。在线数据 MMPR-Tiny 从 MMPR-v1.2 中筛出模型正确率在 0.2–0.8 的问题，并加入近期多模态数据，约 70K queries；避开过易与过难题，提高 reward 区分度。

### 3.4 训练效率与收益

8B 上，SFT baseline 七项平均 53.6；MPO 用约 0.3K GPU-hours 达 56.3；纯 GSPO 1 episode 用约 5.5K 达 57.3，2 episodes 用约 11K 达 58.2；Cascade RL 用约 5.8K 达 60.3。也就是说，MPO 的小额成本不只带来直接增益，还把后续在线 RL 的起点变强，使同约 5.8K GPU-hours 超过两轮纯 GSPO。

![](assets/InternVL3.5%20-%20级联强化学习与视觉压缩/figures/04-cascade-rl-ablation.png)

> 图 5：从 InternVL3、InternVL3.5-Instruct、MPO 到完整 Cascade RL，1B–241B-A28B 各尺度的七项推理平均分基本逐阶段上升。来源：论文 Figure 5，第 22 页。

---

## 4. ViCO 与 ViR：按语义压缩视觉 token

### 4.1 ViR 的决策单位

Dynamic High Resolution 只决定按几何宽高如何切 patch；ViR 决定**每个 patch 的语义信息是否值得保留较多 token**。高信息 patch 走 $1/4$ pixel shuffle，1024→256 token；低信息 patch 走 $1/16$，1024→64 token。论文报告最终平均减少约 50% 视觉 token，而非所有 patch 一律压到 64。

### 4.2 ViCO 阶段一：一致性训练

直接把已经训练好的模型切到 64 token 会造成输出漂移。ViCO 使用冻结的原始 InternVL3.5 作为 reference，始终以 256 token/patch 推理；policy 对每个图像随机选择 256 或 64 token，最小化 token-level 输出分布 KL：

$$\mathcal{L}_{\mathrm{ViCO}}=\mathbb{E}_{\xi\sim\{1/4,1/16\}}\left[\frac1N\sum_{i=1}^{N}\operatorname{KL}\left(\pi_{\theta_{ref}}(y_i\mid y_{<i},I)\,\|\,\pi_{\theta_{policy}}(y_i\mid y_{<i},I_\xi)\right)\right]$$

这相当于把未压缩模型的行为蒸馏给可变压缩 policy。使用与 SFT 类似的数据，目的是覆盖原能力分布，而不是只针对少量 OCR task 过拟合。

### 4.3 ViCO 阶段二：router training

主 MLLM 冻结，只训练二分类 ViR。对 patch $i$，比较 64 token 与 256 token 条件下的损失比例：

$$r_i=\frac{\mathcal{L}_{\mathrm{ViCO}}(y_i\mid I_{1/16})}{\mathcal{L}_{\mathrm{ViCO}}(y_i\mid I_{1/4})}$$

若 $r_i<\tau$，压缩影响小，label 0，走 64 token；否则 label 1，走 256 token。$\tau$ 取历史滑动窗口 $r_i$ 的第 $k$ 百分位，以保持标签大致平衡。router 数据主要来自视觉信息丰富的 OCR/VQA 子集。

### 4.4 ViR 的性能代价

8B 的九项平均从 80.2 降至 Flash 的 79.8；38B 从 83.9 降至 83.4；30B MoE 从 82.6 降至 82.2；235B MoE 从 85.0 降至 84.5。DocVQA/InfoVQA 等高分辨率任务也只小幅下降，但“nearly 100% performance”是相对保留率表述，绝对分仍有 0.4–0.5 的平均损失，且评测未覆盖所有开放式任务。

---

## 5. DvD：视觉与语言解耦部署

### 5.1 计算特性不匹配

ViT 前向高度并行、无长期状态；LLM decoding 自回归、依赖 KV cache，更受内存带宽和时延影响。把二者放在同一 server 串行执行，会让视觉 batch 计算阻塞 LLM prefill/decoding，也难以分别按硬件特性扩容。

![](assets/InternVL3.5%20-%20级联强化学习与视觉压缩/figures/03-dvd-deployment.png)

> 图 4：DvD 把 ViT+MLP（Flash 还包括 ViR）放在 vision server，只把 LLM 放在 language server；视觉计算、BF16 特征传输和语言处理构成异步三级流水线。来源：论文 Figure 4，第 8 页。

### 5.2 系统实现

vision server 批量处理图像并输出紧凑 embedding，经 TCP 单向传输 BF16 特征；可选 RDMA 提速。language server 接收视觉特征，与文本融合后完成 prefill 和 decoding。三阶段异步重叠，且视觉/语言硬件可以独立配置。

### 5.3 吞吐收益

在语言模型均部署于 8 张 A100、输入负载 16 requests/s 时：

| 分辨率 | 38B baseline | +DvD | +DvD+ViR | 241B-A28B baseline | +DvD | +DvD+ViR |
|---|---:|---:|---:|---:|---:|---:|
| 448 | 12.39 | 14.69（1.19×） | 18.62（1.50×） | 11.29 | 14.05（1.24×） | 20.84（1.85×） |
| 896 | 2.71 | 5.06（1.87×） | 10.97（4.05×） | 2.54 | 4.73（1.86×） | 8.81（3.47×） |
| 1344 | 1.48 | 2.92（1.97×） | 5.14（3.47×） | 1.37 | 2.75（2.01×） | 4.27（3.12×） |

视觉越重，DvD 越有价值；4.05× 是 38B、896 分辨率、DvD+ViR 相对 baseline 的特定结果，不应概括为所有请求统一四倍。

---

## 6. 数据与完整训练流水线

### 6.1 阶段数据

| 阶段 | 数据规模 | 配比/筛选 | 目标 |
|---|---:|---|---|
| Native pre-training | 116M samples，约 250B token | text:multimodal≈1:2.5 | 全参数视觉—语言共同预训练 |
| SFT | 56M samples，约 130B token | text:multimodal≈1:3.5 | 指令、Thinking、GUI、具身、SVG |
| Offline RL/MPO | MMPR-v1.2，约 200K pairs | 固定 rollout，可跨模型共享 | 稳定暖启动 |
| Online RL/GSPO | MMPR-Tiny，约 70K queries | 基础正确率 0.2–0.8 + 新数据 | 在线探索与上限提升 |
| ViCO consistency | 约 30B token，主要复用 SFT 分布 | 256/64 token 随机压缩 | 压缩行为蒸馏 |
| ViR router | SFT 子集，偏 OCR/VQA | KL loss ratio 自动标注 | patch 级二分类路由 |

### 6.2 工程栈

训练主要基于 XTuner，使用 FSDP、data packing、FP8/DeepGEMM、fused cross-entropy、FlashAttention-3 与 MoE grouped GEMM；在线 RL 基于 verl。论文提供了比 InternVL3 更具体的 8B RL GPU-hour 对账，但仍未披露 250B+130B token 全训练的总 GPU-hour。

---

## 7. 量化结果与 baseline

### 7.1 推理能力

七项推理平均上，InternVL3.5-2B/4B/8B/14B/38B/241B-A28B 分别为 50.7/57.4/60.3/62.0/66.0/66.9；对应 InternVL3-2B/8B/38B/78B 为 32.4/44.3/52.8/54.6。8B 的 +16.0 是论文所称最大总体推理增益之一，且主要来自更强 SFT 与 Cascade RL 联合，而非 RL 单独贡献。

与 contemporaneous baseline 比，8B 的 60.3 低于 MiMo-VL-RL-8B 的 64.8 和 GLM-4.1V-9B 的 62.6；38B 的 66.0 高于 Skywork-R1V3-38B 的 59.5；241B-A28B 的 66.9 低于 GLM-4.5V 69.0、Gemini-2.5-Pro 72.8 和 GPT-5 74.1。论文的开源领先在不同规模和任务上并非无条件成立。

### 7.2 综合与 agentic 能力

241B-A28B 的 general 14 项平均 74.1，与表中 GPT-5 的 74.0 接近，但这由不同来源结果混合汇总，且单项差异大：InternVL3.5 在 OCRBench 90.7、MTVQA 39.3、MVBench 76.5 较强；GPT-5 在 VideoMME 81.8、HallusionBench 65.2、LongVideoBench 72.6 更高。Agentic 方面，241B-A28B 的 ScreenSpot/ScreenSpot-v2/OSWorld-G 为 89.8/92.9/53.2，显示从静态 grounding 到真实 OS 操作仍有明显落差。

### 7.3 Parallel Thinking

论文将 deep thinking 与 VisualPRM Best-of-N 的 parallel thinking 结合。4B/8B/241B-A28B 七项平均分别再提升 +2.6/+2.1/+1.8。规模越大，边际收益反而较小，可能是 base policy 已更强，也可能是 critic 或候选多样性成为瓶颈；论文未做完整成本—收益曲线。

---

## 8. 消融、局限与未来工作

### 8.1 消融结论

1. **SFT 已贡献大量增益**：8B 从 InternVL3 的 44.3 到 Instruct 53.6；不能把最终 60.3 全算给 Cascade RL。
2. **MPO 是便宜暖启动**：8B 仅约 0.3K GPU-hours 从 53.6 到 56.3。
3. **GSPO 提高上限但贵**：两轮约 11K GPU-hours到 58.2；Cascade 约 5.8K 到 60.3。
4. **ViR 近似保性能而非零损失**：九项平均约下降 0.4–0.5，换取约 50% 视觉 token 减少。
5. **DvD 与 ViR 正交**：前者重排硬件流水线，后者减少视觉计算/LLM token，组合收益最大。

### 8.2 明确和隐含局限

论文没有独立 limitations 章节，也没有系统讨论训练数据版权、安全或 RL reward 偏差。正文可直接观察到的限制包括：HallusionBench 仍是弱项；高分辨率吞吐实验只覆盖特定请求率和 8×A100 语言服务；ViR 仅二档压缩，router label 来自自身 KL proxy；ViCO 需要冻结 reference model，训练额外约 30B token；全量预训练/SFT 算力未披露。

Cascade RL 的 reward 定义和 verifier 细节未在主文充分展开，难以判断 70K MMPR-Tiny 上的提升是否迁移到开放式、不可验证任务。在线 RL 只筛选初始正确率 0.2–0.8 的题，提升可能集中于“恰好可学”的 benchmark 分布。

### 8.3 未来方向

论文结论主要概括贡献，没有正式 future-work 清单。合理且直接由结果导出的方向包括：把 ViR 从二分类扩到多档或连续预算；将路由目标从 teacher KL 扩展到端到端延迟/显存；评估 DvD 在多图与长视频真实流量上的排队与容错；研究 Cascade RL 在 GUI/具身的长 horizon 环境奖励；公开全量训练成本、reward pipeline 和数据去重细节。

---

## 9. 复现成本与个人判断

### 9.1 复现成本

- **权重推理**：1B–8B 可在常见推理硬件上尝试；38B 与 MoE 需要多卡。动态高分辨率的真实成本取决于 patch 数，不能只看参数量。
- **Cascade RL**：8B 论文对账约 5.8K GPU-hours，且还不含 SFT 与 rollout 数据构造；普通团队可先复现 MPO，不宜直接从在线 GSPO起步。
- **ViCO/ViR**：需双模型 teacher-policy 前向与约 30B token consistency training，训练成本并不“轻”；router 本身很小，但其可靠性来自前一阶段蒸馏。
- **DvD**：需要两个服务池、特征协议、异步 pipeline、网络带宽与调度系统。单机 demo 无法复现吞吐收益，必须在有并发的在线系统测量。

### 9.2 我的判断

Cascade RL 的价值不在发明一个新 policy loss，而在训练顺序：用廉价离线偏好优化改善在线采样分布。它像 curriculum，又像 policy initialization；最有说服力的证据是 8B 在约一半的纯 GSPO 两轮成本下获得更高平均分。

ViCO/ViR 则把视觉 token 压缩从固定规则变成了“语义条件下的预算分配”。但当前 router 只根据 teacher-student KL 间接判断重要性，尚未显式学习“这个 patch 对最终任务是否必要”。DvD 是最偏系统、也最容易被忽略的贡献：只减少 FLOPs 不一定提升线上吞吐，只有把 ViT 和自回归 LLM 的不同硬件节奏解耦，压缩才真正转化为服务收益。

### 9.3 最终结论

InternVL3.5 是一篇训练算法与推理系统共同设计的技术报告。它把 InternVL3 的“原生预训练 + MPO + VisualPRM”升级为“原生预训练 + 大规模 Thinking SFT + MPO→GSPO + ViCO/ViR + DvD”。最可信的贡献是 8B RL 成本消融与 DvD/ViR 吞吐表；最需要进一步验证的是大规模综合榜单的同条件可比性、reward 泛化和 ViR 在开放域长视频/agent 任务中的失真边界。

---

## 10. 材料来源

- 官方论文：<https://arxiv.org/abs/2508.18265>
- 本地 PDF：`assets/InternVL3.5 - 级联强化学习与视觉压缩/sources/paper.pdf`
- 来源与 Figure/Page 记录：`assets/InternVL3.5 - 级联强化学习与视觉压缩/sources.md`
- 官方代码：<https://github.com/OpenGVLab/InternVL>
