---
type: paper-note
status: complete
domain: World-Models
paper: "Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval"
year: 2025
arxiv: 2506.03141
version: v2
venue: SIGGRAPH Asia 2025
source: https://arxiv.org/abs/2506.03141
project: https://context-as-memory.github.io/
code: null
tags: [World-Model, Interactive-Video, Long-Term-Memory, Retrieval, Camera-Control, Diffusion-Transformer]
created: 2026-10-15
updated: 2026-10-15
---

# Context as Memory：历史上下文驱动的场景记忆

## 基本信息

- **作者**：Jiwen Yu、Jianhong Bai、Yiran Qin、Quande Liu、Xintao Wang、Pengfei Wan、Di Zhang、Xihui Liu
- **机构**：香港大学、浙江大学、快手 Kling Team
- **版本**：arXiv v2，2025-08-12；SIGGRAPH Asia 2025
- **任务**：给定文本、用户相机轨迹和历史生成帧，流式生成可持续探索、重访时仍保持场景一致的长视频
- **核心资源**：[[assets/Context as Memory - 历史上下文驱动的场景记忆/sources/sources|来源记录]]；[官方 PDF](assets/Context%20as%20Memory%20-%20%E5%8E%86%E5%8F%B2%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A9%B1%E5%8A%A8%E7%9A%84%E5%9C%BA%E6%99%AF%E8%AE%B0%E5%BF%86/sources/paper.pdf)

## 一句话结论

**Context as Memory 不学习独立记忆向量，也不重建 3D 地图，而是直接把历史 RGB 帧当记忆；它用已知相机位姿的 FOV 共视关系检索少量相关旧帧，再与当前待生成 latent 沿时间维拼接进同一 DiT，在固定 20 帧上下文下显著提高“离开后再回来”的像素与场景一致性，但其二维平面运动、规则检索和 1B 内部模型使结论仍主要适用于受控相机探索。**

---

## 1. 研究背景与核心问题

长视频“能一直生成”不等于世界“持续存在”。DFoT、Oasis、GameFactory 一类方法通常只保留最近几十帧；相机转身后，旧区域退出滑动窗口，模型再次看到该区域时只能重新想象，导致建筑、物体与空间关系变化。

论文把问题归因为**可访问历史受限**：若所有历史帧都可见，模型原则上能复制旧内容；但全历史注意力计算昂贵，而且大量无关帧会制造干扰。因此真正的问题不是“记忆要不要存在”，而是：

1. 记忆保存什么表示；
2. 依据什么从无限历史中取回少量有用信息；
3. 如何把可变数量记忆注入预训练视频 DiT，而不增加复杂外部模块。

![](assets/Context%20as%20Memory%20-%20%E5%8E%86%E5%8F%B2%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A9%B1%E5%8A%A8%E7%9A%84%E5%9C%BA%E6%99%AF%E8%AE%B0%E5%BF%86/figures/01-teaser.png)

> **图 1（Figure 1，PDF p.1）**：四条长相机轨迹分别在仓库、神庙、海滨村庄和日式街区中绕行并重访；跨越数百到上千帧后，同一空间仍保留主要布局与外观。它验证的是“空间重访”，不是一般动作长期建模。来源：<https://arxiv.org/abs/2506.03141>，官方源码 `figures/teaser.pdf`。

---

## 2. 任务定义、输入与输出

### 2.1 输入

每个自回归段接收：

- 文本提示 $\mathbf p$；
- 用户指定的逐帧相机外参 $\mathbf{cam}=[R,t]\in\mathbb R^{f\times 3\times4}$；
- 最近一帧，保证段间运动连续；
- 从全部历史中检索的至多 $k-1$ 张 RGB 帧；
- 当前 77 帧目标段对应的噪声 latent。

实验设置 $k=20$。上下文 RGB 帧彼此不连续，逐张通过 causal 3D-VAE 编码，各自形成一个 latent；77 个输出帧以时间压缩率 4 编为 20 个 latent。

### 2.2 输出

输出为 $640\times352$、77 帧的视频段；不断重复检索—生成—写入历史即可形成长视频。任务要求同时满足：

- 相机轨迹可控；
- 短期运动连续；
- 重访旧区域时恢复过去的场景内容；
- 开放域首图上仍能使用记忆机制。

### 2.3 评价定义

论文区分两类重访比较：

- **Ground Truth Comparison**：以真实历史帧作上下文，生成结果与 GT 对齐；
- **History Context Comparison**：把模型生成历史作为上下文，比较回访帧与先前生成帧，更直接暴露自回归误差传播。

---

## 3. 记忆表示、检索依据与几何使用

### 3.1 记忆表示：原始帧，而非地图或特征库

所有历史生成帧直接存为 RGB/frame latent，不做 embedding 汇聚、3D 重建或状态压缩。优点是信息保真、接口简单，视觉细节可直接被 DiT 注意力读取；代价是存储线性增长，且每次只能取有限子集。

这是一种**图像记忆（view memory）**，不是显式世界状态。它记住“过去看见什么”，不保证“场景实际三维结构是什么”。

### 3.2 检索依据：相机视场共视

因为每一段都由用户相机轨迹控制，历史帧天然带相机位姿。方法把相机运动限制在 XY 平面、旋转限制为 z 轴 yaw，用每台相机 FOV 的左右边界射线判断两个扇形是否相交，并以交点距目标相机的距离排除过远伪交叠。

FOV 过滤后若候选仍过多：

1. 连续候选段只随机保留一帧（Non-adj），去除相邻冗余；
2. 可选地偏向时空距离较远的帧，补充长期覆盖。

### 3.3 是否使用几何

它使用**弱几何/相机几何**：检索依赖位姿和 FOV，但不估深度、不建点云、不做遮挡推理。与 SPMem/VMem 的关键区别是：几何只负责“候选路由”，不成为记忆内容或生成条件。因此速度与鲁棒性较好，但墙体遮挡、楼层关系、非平面运动都可能让 FOV 共视产生误检。

![](assets/Context%20as%20Memory%20-%20%E5%8E%86%E5%8F%B2%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A9%B1%E5%8A%A8%E7%9A%84%E5%9C%BA%E6%99%AF%E8%AE%B0%E5%BF%86/figures/02-method.png)

> **图 2（Figure 3，PDF p.4）**：左侧把无限历史经过 Memory Retrieval 压成固定数量上下文，与当前噪声帧共同进入自回归 DiT；右侧用相机视锥重叠选出高共视旧帧。图中没有 3D scene state，说明该方法的几何只服务检索。来源：<https://arxiv.org/abs/2506.03141>，官方源码 `figures/framework.pdf`。

---

## 4. 核心机制与公式

### 4.1 基础视频扩散

3D-VAE 把视频 $\mathbf x$ 编码为 $\mathbf z=Encoder(\mathbf x)$。模型训练为噪声预测：

$$
\mathcal L(\phi)=\mathbb E\left[\|\epsilon_\phi(\mathbf z_t,\mathbf p,t)-\epsilon\|\right].
$$

相机条件经单层 MLP 编码后，在空间注意力与 3D 注意力之间相加：

$$
\mathbf F_i=\mathbf F_o+\mathcal E_c(\mathbf{cam}),
$$

对应损失为：

$$
\mathcal L_{\mathbf{cam}}=\mathbb E\left[\|\epsilon_{\phi,\phi_{MLP}}(\mathbf z_t,\mathbf p,\mathbf{cam},t)-\epsilon\|\right].
$$

### 4.2 Context-as-Memory 注入

设干净历史上下文 latent 为 $\mathbf z^c$，当前待预测 noisy latent 为 $\mathbf z_t$。二者沿 frame/time 维拼接：

$$
\epsilon_\phi(\{\mathbf z_t,\mathbf z^c\},\mathbf p,t).
$$

上下文与目标 token 在每个 DiT block 中平等参加注意力，但输出时只更新 $\mathbf z_t$，保持 $\mathbf z^c$ 不变。相比额外 cross-attention/adapter，这种做法直接复用全序列注意力，天然支持可变上下文长度。

为尽量保留预训练能力，目标 latent 沿用预训练时间位置；新加上下文分配独立 RoPE 位置。论文没有给出防止不同历史帧时间顺序混淆的专门机制，模型主要依靠内容与相机条件理解它们。

### 4.3 训练与推理算法

- **训练**：在 7,601 帧长视频中随机选一个 77 帧目标段；在其余帧中预计算 FOV 共视并取 $k-1$ 张；目标首帧作为第 $k$ 张上下文。10% 概率只给最近帧，模拟视频开头。
- **推理**：每段从全部已生成历史检索 $k-1$ 张，再加最近帧；生成后把新帧追加到历史池。

这种训练对齐了检索结构，但上下文主要来自真实视频；推理上下文来自模型输出，仍存在 train–test history gap。

---

## 5. 训练、推理与错误传播路径

### 5.1 训练配置

- 内部 1B 参数 text-to-video DiT；
- causal 3D-VAE，时间压缩率 4；
- 77 帧、$640\times352$；
- 上下文 20 RGB 帧/20 latent；
- 训练超过 10,000 iterations，batch size 64；
- 8×NVIDIA A100；
- 推理 50 sampling steps，文本使用 CFG。

### 5.2 错误如何传播

错误链条是：

$$
\text{早期生成误差}\rightarrow\text{写入历史帧}\rightarrow\text{FOV 检索重新取回}\rightarrow\text{DiT 复制/强化误差}\rightarrow\text{后续历史继续污染}.
$$

作者的正面解释是，检索到较早、误差更少的帧可给后段更强约束，因此 FID/FVD 也改善。反面是：方法没有置信度、冲突消解或旧记忆修复；一旦错误旧帧与新观察冲突，注意力自行决定相信谁。FOV 误检还会把空间邻近但被遮挡的内容注入当前段。

### 5.3 推理复杂度

历史存储随时长线性增长，但 DiT 输入保持 $k=20$。检索是规则搜索；论文没有报告历史池达到数千/数万帧时检索索引与内存开销，也未报告端到端交互延迟。

---

## 6. 数据、基准与评价设置

### 6.1 数据

作者在 Unreal Engine 5 中收集 100 条长视频，每条 7,601 帧、30 fps，覆盖 12 类室内外/写实/风格化场景。轨迹由平滑多段线/B-spline 构成：每 77 帧移动 3–6 m、yaw 变化小于 60°；相机 24 mm 焦距、FOV 52.67°。每 77 帧用多模态 LLM 标注 caption。

局限是数据总量不大且运动自由度受控：位置仅 XY，旋转仅 yaw。它非常适合验证 FOV 检索，却不能证明六自由度、动态物体丰富的真实探索中仍成立。

### 6.2 比较方法

所有方法在同一 1B backbone、同一数据和训练迭代上重实现：首帧上下文、首帧+随机帧、DFoT 最近窗口、FramePack 分层压缩、Context-as-Memory。该控制变量设计比直接比较不同商业模型更可信。

### 6.3 指标

- 记忆/像素一致性：PSNR↑、LPIPS↓；
- 生成分布质量：FID↓、FVD↓；
- 轨迹：旋转离开再返回，通过对应帧直接比较。

PSNR/LPIPS 偏向像素复现，不能完整衡量三维一致性；论文也没有遮挡、动态实体状态或交互动作成功率。

---

## 7. 主要定量结果

### 7.1 主结果

| 方法 | GT PSNR↑ | GT LPIPS↓ | GT FID↓ | GT FVD↓ | History PSNR↑ | History LPIPS↓ | History FID↓ | History FVD↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 首帧 | 15.72 | 0.5282 | 127.55 | 937.51 | 14.53 | 0.5456 | 157.44 | 1029.71 |
| 首帧+随机 | 17.70 | 0.4847 | 115.94 | 853.13 | 17.07 | 0.3985 | 119.31 | 882.36 |
| DFoT | 17.63 | 0.4528 | 112.96 | 897.87 | 15.70 | 0.5102 | 121.18 | 919.75 |
| FramePack | 17.20 | 0.4757 | 121.87 | 901.58 | 15.65 | 0.4947 | 131.59 | 974.52 |
| **Context-as-Memory** | **20.22** | **0.3003** | **107.18** | **821.37** | **18.11** | **0.3414** | **113.22** | **859.42** |

最关键的现象是随机历史帧在 History Comparison 上优于 DFoT/FramePack：对重访任务，“时间近”不是“空间相关”；哪怕随机抽样，也有机会拿到旧区域，而最近窗口必然遗忘。

![](assets/Context%20as%20Memory%20-%20%E5%8E%86%E5%8F%B2%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A9%B1%E5%8A%A8%E7%9A%84%E5%9C%BA%E6%99%AF%E8%AE%B0%E5%BF%86/figures/03-results.png)

> **图 3（Figure 5，PDF p.6）**：左侧与 GT 比较，右侧测试“转出去再转回来”。Context-as-Memory 能恢复书架、货架等旧内容；DFoT/FramePack 因只保留近邻历史，返回时生成了另一套场景。来源：<https://arxiv.org/abs/2506.03141>，官方源码 `figures/comparison.pdf`。

### 7.2 如何解读

相对 FramePack，History PSNR 提升 2.46 dB，LPIPS 从 0.4947 降到 0.3414；但这不意味着通用质量全面领先，因为 FramePack 在本文中按作者实现、使用其指数时间压缩，并未使用面向重访的特征相似度版本。

---

## 8. 消融、失败案例与证据边界

### 8.1 上下文大小

| 上下文数 | GT PSNR | History PSNR | History LPIPS | FPS |
|---:|---:|---:|---:|---:|
| 1 | 15.72 | 14.53 | 0.5456 | 1.60 |
| 10 | 19.14 | 17.75 | 0.3985 | 1.20 |
| 20 | 20.22 | 18.11 | 0.3414 | 0.97 |
| 30 | 20.31 | 18.19 | 0.3319 | 0.79 |

20→30 的收益很小而速度继续下降，$k=20$ 是合理折中。即使如此，0.97 fps 与“交互式”实时仍有明显距离。

### 8.2 检索策略

| 策略 | GT PSNR↑ | History PSNR↑ | History LPIPS↓ |
|---|---:|---:|---:|
| Random | 17.70 | 17.07 | 0.3985 |
| FOV+Random | 19.17 | 17.47 | 0.3896 |
| FOV+Non-adj | 20.11 | **18.19** | 0.3571 |
| FOV+Non-adj+Far-space-time | **20.22** | 18.11 | **0.3414** |

FOV 与去相邻冗余是主要贡献；Far-space-time 的增益不稳定，History PSNR 甚至略降，说明复杂启发式并非必需。

### 8.3 失败与缺失证据

作者在附录承认：1B base model 对复杂开放域轨迹会迅速积累误差；当前不能支持从任意图像进行复杂、动态、长时间自由探索。正文没有独立失败图，也没有六自由度、强遮挡、多层建筑、动态物体重访、长期状态改变测试；这些应视为信息缺口，而不是默认成功。

---

## 9. 局限、Future Work 与复现成本

### 9.1 局限与未来方向

- FOV 相交忽略深度与遮挡；二维平面假设限制真实导航。
- 相机位姿必须由控制信号已知；自由视频需额外位姿估计器。
- 记忆保存所有 RGB 历史，长期存储与搜索未量化。
- 规则检索不可学习，无法利用语义、动态状态与不确定度。
- 训练上下文偏向 GT，未专门解决生成历史分布偏移。
- 作者建议探索可学习/self-supervised context compression，并在更大 backbone 上验证。

### 9.2 复现成本

**高且无法完全复现。** 训练主干是未公开的内部 1B DiT；数据依赖 12 个 UE5 资产、轨迹与渲染流水线。论文给出 8×A100、10k+ steps、batch 64，足以估算实验级复现，但没有代码、权重、训练时长、优化器、学习率与精确检索阈值。可复现的部分主要是 FOV 检索逻辑和对公开视频 DiT 的接口设计。

---

## 10. 个人判断与记忆路线比较

### 10.1 个人判断

这篇论文最有价值的不是“把帧塞回注意力”本身，而是明确指出：**长时记忆的关键排序轴可以是空间相关性，而不是时间新近性。** 同一 backbone 的公平比较支持这一点。它也是四篇中最小改动、最易迁移的方案；但视觉帧本身就是世界状态这一假设较弱，复杂遮挡与动态环境中会暴露上限。

### 10.2 与 SPMem、VMem、FramePack 的对照

| 方法 | 记忆表示 | 检索依据 | 几何角色 | 主要错误传播路径 |
|---|---|---|---|---|
| **Context as Memory** | 原始历史帧 | 已知位姿的 FOV 共视 | 弱几何，仅筛帧 | 错帧被重新检索并复制；FOV 遮挡误检 |
| [[SPMem - 长期空间记忆视频世界模型]] | 静态点云 + 最近帧 +关键帧 | 目标轨迹渲染点云；未知区阈值选关键帧 | 强几何，直接条件生成 | 重建/配准错→TSDF 误删→点云稀疏→控制失败 |
| [[VMem - Surfel索引视图记忆]] | RGB 视图库 + surfel→视图索引 | 目标视角可见 surfel 的索引投票 | 中等几何，只做索引 | 点图错→surfel 关联错→检索错；生成图再污染索引 |
| [[FramePack - 固定上下文预算的长视频生成]] | 所有历史的多尺度 latent token | 时间邻近、特征相似或混合排序 | 不使用显式几何 | 历史误差被压缩并传播；纯因果链持续漂移 |

如果目标是受控相机重访且已有可靠位姿，Context as Memory 是最低工程门槛；若需要明确静态结构，SPMem 更强但脆弱；若只想用粗几何找回原始视图，VMem 的折中更合理；若任务是一般人物/动作长视频而非空间重访，FramePack 的固定预算更通用。
