---
type: paper-note
status: done
domain: World-Models
paper: "Yume-1.5: A Text-Controlled Interactive World Generation Model"
year: 2025
arxiv: "2512.22096"
source: https://arxiv.org/abs/2512.22096
project: https://stdstu12.github.io/YUME-Project/
code: https://github.com/stdstu12/YUME
tags:
  - World-Models
  - Interactive-Video
  - Long-Context
  - Text-Control
created: 2026-10-15
updated: 2026-10-15
---

# Yume 1.5：文本控制的交互式世界生成

## 基本信息

- **作者**：Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, Kaipeng Zhang
- **机构**：Shanghai AI Laboratory；Fudan University；Shanghai Innovation Institute
- **基础模型**：Wan2.2-TI2V-5B，704×1280/16 FPS 训练；评测为 544×960、96 帧
- **交互形式**：WASD 人物/平移控制、方向键相机旋转、文本事件编辑；支持 text-to-world 与 image-to-world
- **官方披露核验（2026-10-15）**：论文给出的 GitHub、项目页与 Sekai 数据仓库可定位；官方代码和模型权重已公开。论文未披露训练 GPU 数量、总 wall-clock、模型完整参数分解与事件数据许可细节。

## 一句话结论

Yume 1.5 以 TSCM 将久远历史同时在时空与通道维压缩，再用线性注意力融合，并以类似 Self-Forcing 的分布匹配蒸馏把采样压到 4 步；结果是单张 A100 上约 12 FPS 的 540p 可交互世界，但其“无限”指固定单步开销而非已验证的无限一致性。

## 1. 问题、动机与贡献

既有交互视频世界模型面临三重矛盾：完整历史导致计算持续增长；截断历史损害回访一致性；few-step 蒸馏又会放大自回归误差。此外，许多工作仅支持键鼠控制，不支持文本触发世界事件。

![](assets/Yume%201.5%20-%20文本控制的交互式世界生成/figures/01-teaser.png)

> **图解**：统一展示文本生世界、图像生世界和文本事件编辑，三种模式都可连续接受人物与相机键盘控制。来源：论文 **Figure 1, PDF p.1**，https://arxiv.org/pdf/2512.22096（获取：2026-10-15）。

贡献包括：
1. Joint Temporal-Spatial-Channel Modeling（TSCM）压缩全历史，使后期块的推理耗时趋于稳定。
2. 将 self-generated history 与 TSCM 结合做分布匹配蒸馏，缓解 few-step rollout 的误差累积。
3. 将 caption 拆为事件与动作两部分，支持低开销键盘控制和文本事件生成。

## 2. 输入、输出与任务定义

- **输入**：单张图像或文本场景描述；当前动作描述；可选事件描述。
- **输出**：按 chunk 自回归生成的视频世界；训练分辨率 704×1280、16 FPS，Yume-Bench 评测为 544×960、96 帧。
- **控制词表**：人物/相机平移为 W/A/S/D 及组合；相机旋转为上下左右及组合。
- **边界**：模型预测像素视频，不维护显式 mesh、occupancy 或可查询物理状态；“世界事件”是视频语义编辑，不等同于可验证状态变化。

## 3. 整体架构与公式

I2V 时，将条件 latent $z_c$ 零填充到预测长度，用二值 mask $M_c$ 与噪声 $z$ 融合：

$$z_{in}=M_c\odot z_c+(1-M_c)\odot z.$$

事件描述和动作描述分别经 T5 编码后拼接。有限动作词表可预编码缓存；事件 embedding 只在初始/事件变化时重算，减少持续推理中的文本编码开销。主干使用 Wan DiT 和 Rectified Flow loss。

![](assets/Yume%201.5%20-%20文本控制的交互式世界生成/figures/02-tscm-architecture.png)

> **图解**：(a) DiT 中加入线性注意力；(b) 事件/动作文本分解；(c) 越久远的历史采用更高压缩率；(d) chunk 推理保留双路压缩记忆。来源：论文 **Figure 3, PDF p.4**，https://arxiv.org/pdf/2512.22096（获取：2026-10-15）。

## 4. TSCM：长历史的时空-通道联合记忆

### 4.1 时空压缩

历史帧按距离分层 patchify：最近 $t-1,t-2$ 用 $(1,2,2)$；$t-3$ 到 $t-6$ 用 $(1,4,4)$；$t-7$ 到 $t-23$ 用 $(1,8,8)$，更久历史继续提高压缩，并以约 1/32 随机采样时间帧。初始帧始终以 $(1,2,2)$ 保留。

### 4.2 通道压缩与线性注意力

另一支把历史以 $(8,4,4)$ patchify，并把通道压到 96。预测 token 与压缩历史经 ReLU feature map 的线性注意力：

$$o_l=\frac{\left(\sum_i v_i\phi(k_i)^\top\right)\phi(q_l)}{\left(\sum_j\phi(k_j)^\top\right)\phi(q_l)},\quad \phi=\mathrm{ReLU}.$$

标准注意力对 token 数敏感，故用时空压缩；线性注意力对通道维敏感，故用通道压缩。两条路径互补，但远历史仍是有损摘要，不是精确回访缓存。

## 5. Self-Forcing with TSCM 与蒸馏

![](assets/Yume%201.5%20-%20文本控制的交互式世界生成/figures/03-self-forcing-distillation.png)

> **图解**：生成器把自己的输出经 TSCM 作为下一块历史；real/fake score 网络在完整 rollout 上提供分布匹配梯度，避免训练时只看真实历史。来源：论文 **Figure 4, PDF p.6**，https://arxiv.org/pdf/2512.22096（获取：2026-10-15）。

训练分两段：先在 T2V/I2V 数据间交替训练 foundation model，再把 generator、real model、fake model 从 foundation 权重初始化。DMD 梯度为：

$$\nabla L_{DMD}=-\mathbb E_t\left[(s_{real}(F(G_\theta(z),t),t)-s_{fake}(F(G_\theta(z),t),t))\frac{dG_\theta}{d\theta}\frac{dz}{ }\right].$$

与原始 DMD 的关键区别是历史条件来自模型预测，并经 TSCM 而非 KV cache 表示。论文并未给出 fake/real 网络每项优化器参数、CFG、EMA 或损失权重，因此无法从论文独立精确复刻蒸馏阶段。

## 6. 数据与训练设置

- **Real-world**：Sekai-Real-HQ；由轨迹导出离散键鼠动作。原 caption 用于 T2V，InternVL3-78B 重标注动态事件用于 I2V。
- **Synthetic**：OpenVid 去重抽样 80K captions；Wan2.1-14B 生成 80K 个 720p 视频，VBench 过滤后留 50K。
- **Event**：志愿者覆盖城市日常、科幻、幻想、天气四类；10K 第一人称图像经 Wan2.2-14B-I2V 生成 10K 视频，人工筛到 4K。
- **训练**：Wan2.2-5B；Adam，lr $10^{-5}$，batch 40；foundation 10K iter，Self-Forcing+TSCM 600 iter；均使用 A100，但 **GPU 数量与 wall-clock 未披露**。
- **评测**：Yume-Bench 的 IF、SC、BC、MS、AQ、IQ；4 个推理步。

## 7. 主要定量、消融与定性结果

| 模型 | 生成时间 s ↓ | IF ↑ | SC ↑ | BC ↑ | MS ↑ | AQ ↑ | IQ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wan2.1 | 611 | 0.057 | 0.859 | 0.899 | 0.961 | 0.494 | 0.695 |
| MatrixGame | 971 | 0.271 | 0.911 | 0.932 | 0.983 | 0.435 | 0.750 |
| Yume | 572 | 0.657 | 0.932 | 0.941 | 0.986 | 0.518 | 0.739 |
| **Yume 1.5** | **8** | **0.836** | **0.932** | **0.945** | **0.985** | **0.506** | **0.728** |

Yume 1.5 的优势主要是控制与速度；AQ/IQ 并非全表最佳。论文另报告单张 A100、540p 平均约 12 FPS。

30 秒 T2V 分成 6 个 5 秒片段：第 6 段中，Self-Forcing+TSCM 的 aesthetic 0.523 vs baseline 0.442；image quality 0.601 vs 0.542。

![](assets/Yume%201.5%20-%20文本控制的交互式世界生成/figures/04-speed-ablation.png)

> **图解**：全上下文计算迅速越过 8 秒截断线；TSCM 在约 8 个 block 后趋于每块约 4 秒，说明其核心收益是长度无关的稳定步耗。来源：论文 **Figure 7, PDF p.8**，https://arxiv.org/pdf/2512.22096（获取：2026-10-15）。

TSCM 与普通 spatial compression 的消融：IF 0.836 vs 0.767；其余 SC/BC/MS/AQ/IQ 接近，说明 TSCM 的主要收益是降低历史运动方向对当前控制的干扰，而非普遍提升单帧画质。

![](assets/Yume%201.5%20-%20文本控制的交互式世界生成/figures/05-qualitative-results.png)

> **图解**：同一夜街场景下，Yume 1.5 用 4 步保持较明确的转向响应；其他方法用 50 步。图为精选案例，不能独立证明所有场景的控制成功率。来源：论文 **Figure 8, PDF p.9**，https://arxiv.org/pdf/2512.22096（获取：2026-10-15）。

## 8. 失败案例、局限与 Future Work

1. 作者明确报告车辆倒行、人物反向行走；高人群密度下退化，720p 只能缓解。
2. 5B 模型容量与低延迟存在直接权衡；论文提出 MoE 作为 future work，但未实验。
3. 30 秒实验只能支持中短程稳定性，不能证明分钟/小时级语义、几何或物理一致性。
4. TSCM 对远历史强压缩，精细物体状态、文字和遮挡后重现可能丢失。
5. 事件数据只有 4K 人工筛选视频，类别窄且由视频模型合成，容易继承教师偏差。
6. 未披露动作到真实位移/尺度的标定、训练随机种子、GPU 数与训练时长。

## 9. 复现成本与个人判断

- 公开 5B 权重降低了推理门槛，但论文性能依赖 A100；720p 训练需 Wan2.2-5B、T5、长视频 latent 与蒸馏双网络，单卡难以复现训练。
- 数据侧至少涉及 Sekai-Real-HQ、50K 筛选合成视频与 4K event 视频，并需要 InternVL3-78B/Wan14B 标注生成管线。
- 因 GPU 数和 wall-clock 未披露，不能给出可信 GPU-hour；只能确认 10K+600 iterations、batch 40。
- **个人判断**：Yume 1.5 是很强的长上下文工程折中，尤其适合“持续向前探索”；但它没有精确空间索引，回访一致性和可交互物理仍弱于显式/检索式记忆方案。

## 10. 材料来源与披露状态

- 本地 PDF、5 张官方源码图与来源清单：`../../assets/Yume 1.5 - 文本控制的交互式世界生成/`。
- 详细清单：`../../assets/Yume 1.5 - 文本控制的交互式世界生成/sources/sources.md`。
- **已披露**：主干、TSCM 压缩率、线性注意力、数据数量、迭代数、batch/lr、主要指标、代码和权重。
- **未披露**：训练 GPU 数/时长、蒸馏完整超参数、模型各模块参数量、事件数据许可与更长时域评测。
