---
type: paper-note
status: complete
domain: Video-Generation
paper: "Frame Context Packing and Drift Prevention in Next-Frame-Prediction Video Diffusion Models"
year: 2025
arxiv: 2504.12626
version: v3
source: https://arxiv.org/abs/2504.12626
project: https://github.com/lllyasviel/FramePack
code: https://github.com/lllyasviel/FramePack
tags: [Video-Generation, Long-Video, Context-Compression, Autoregressive-Diffusion, Drift-Prevention, HunyuanVideo, Wan]
created: 2026-10-15
updated: 2026-10-15
---

# FramePack：固定上下文预算的长视频生成

## 基本信息

- **作者**：Lvmin Zhang、Shengqu Cai、Muyang Li、Gordon Wetzstein、Maneesh Agrawala
- **机构**：Stanford University、MIT
- **版本**：arXiv v3，2025-10-14（本文按最新版本精读）
- **任务**：把预训练视频扩散模型改成 next-frame/next-frame-section 生成器，在固定 token 预算内读取成百上千历史帧，并抑制自回归 drift
- **基础模型**：主实验 HunyuanVideo 13B；另实现 Wan；支持 T2V/I2V
- **资源**：[[assets/FramePack - 固定上下文预算的长视频生成/sources/sources|来源记录]]；[官方 PDF](assets/FramePack%20-%20%E5%9B%BA%E5%AE%9A%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A2%84%E7%AE%97%E7%9A%84%E9%95%BF%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90/sources/paper.pdf)

## 一句话结论

**FramePack 用“按重要性排序 + 指数级不同 patchify kernel”把无限历史压成收敛到常数的 latent token 预算，再用预生成端点/反向填充打断纯因果误差链，或用 K-means 离散历史缩小训练真值与推理生成历史的分布差；它不依赖显式几何，因而最通用，但旧帧会被极度降采样，空间重访记忆弱于基于几何的精确检索。**

---

## 1. 问题：遗忘与漂移的矛盾

FramePack 把长视频的两种失败分开：

- **Forgetting**：上下文窗口有限，人物、场景与早期事件逐渐不可见；
- **Drifting**：单段小误差成为下一段条件，持续累积后离开训练分布，画质、结构和语义恶化。

两者存在张力：增强历史依赖可减轻遗忘，却会把更多错误传下去；给历史加噪、mask 或切断依赖可抑制漂移，却会更快遗忘。论文因此不是只提出 packing，而是“固定预算记忆 + 防漂移采样/训练”的组合。

输入历史帧具有巨大时间冗余，所有帧用全分辨率 token 不划算。核心假设是：**不同历史帧对下一段的重要性不同，应分配不同 token 数，而不是简单选/删。**

---

## 2. 任务定义、输入与输出

设每次生成 $S$ 帧 latent section：

$$\mathbf X\in\mathbb R^{S\times h\times w\times c},$$

历史为 $T$ 帧：

$$\mathbf F\in\mathbb R^{T\times h\times w\times c},\qquad T\gg S.$$

若每帧 token 长度为 $L_f$（480p 的 Hunyuan/Wan/Flux 约 1560），普通 DiT 总长度：

$$L=L_f(T+S),$$

随历史线性增长，注意力计算近似二次增长。FramePack 的输出仍是下一帧/下一小段，但历史输入被多分辨率 patchify 后拼成固定上限 token 序列。

推理支持：

- 正向流式 T2V/I2V；
- 先生成终点再填中间；
- 从终点反向朝高质量用户首图填充；
- 多 prompt 的多端点规划；
- 千帧级历史条件。

---

## 3. 记忆表示、重要性与是否使用几何

### 3.1 记忆表示

FramePack 不建立外部 memory bank，而把所有历史视频 latent 变成**不同压缩率的 token 块**。重要帧保留更多空间/时间 token，低重要帧只剩极粗摘要；尾部可删除、每帧追加一个 token，或全局平均池化为一个共享尾 token。

因此它是“连续多尺度历史表示”，不是 [[Context as Memory - 历史上下文驱动的场景记忆]]/[[VMem - Surfel索引视图记忆]] 的离散视图检索，也不是 [[SPMem - 长期空间记忆视频世界模型]] 的 3D 地图。

### 3.2 重要性排序

论文支持三种依据：

1. **时间邻近**：最近帧最重要；
2. **特征相似度**：与估计目标 $\hat{\mathbf X}$ 余弦相似的旧帧更重要；
3. **混合**：内容相似度 + 平滑时间先验。

余弦相似度：

$$
\operatorname{sim}_{cos}(\mathbf F_i,\hat{\mathbf X})
=\sum_p\frac{(\mathbf F_i)_p\hat{\mathbf X}_p^\top}{\|(\mathbf F_i)_p\|\|\hat{\mathbf X}_p\|}.
$$

时间项：

$$
\operatorname{sim}_{time}=e^{-(time(\mathbf F_i)-time(\hat{\mathbf X}))^2},
$$

混合项：

$$
\operatorname{sim}_{hybrid}=\operatorname{sim}_{cos}+\lambda_{time}\operatorname{sim}_{time}.
$$

世界模型返回旧场景时，feature/hybrid 排序理论上可把久远但相似的帧提到高分辨率层；主论文详细定量却以多种 packing/sampling 配置为主，feature-hybrid 的单独数字被放到“supplementary”但 arXiv 源码未给出对应表，这是信息缺口。

### 3.3 是否使用几何

**不使用显式几何。** 没有相机位姿、FOV、深度、点云或 surfel。优点是适用于人物、舞蹈、电影与任意普通视频；缺点是特征相似不等于空间共视，隔墙/相反视角/同一角色不同状态可能被混淆。

---

## 4. Frame Context Packing 机制

### 4.1 指数压缩与固定上限

按重要性排序后，$\mathbf F_0$ 最重要、$\mathbf F_{T-1}$ 最不重要。第 $i$ 帧的 token 长度：

$$
\phi(\mathbf F_i)=\frac{L_f}{\lambda^i},\qquad \lambda>1.
$$

总长度：

$$
L=S L_f+L_f\sum_{i=0}^{T-1}\frac1{\lambda^i}
=S L_f+L_f\frac{1-1/\lambda^T}{1-1/\lambda}.
$$

当 $T\to\infty$：

$$
\lim L=L_f\left(S+\frac{\lambda}{\lambda-1}\right).
$$

论文正文漏写最后公式中的 $L_f$ 乘子，但从上下文和前式可知它应存在。取 $\lambda=2$ 时，所有历史帧总 token 量上限约等于 2 个完整帧，再加 $S$ 个目标帧。

### 4.2 如何压缩

不是先把图片缩小再统一 patchify，而是直接改变输入 patchify 3D kernel $(p_f,p_h,p_w)$。同一压缩率 64 可由 $(1,8,8)$、$(4,4,4)$、$(16,2,2)$ 等实现，分别偏空间或时间压缩。

常用 kernel $(2,4,4)$、$(4,8,8)$、$(8,16,16)$ 有独立输入投影参数，并由预训练 projection 插值初始化；更高压缩先做 $2\times2\times2$ 下采样再走最大 kernel。不同 token 网格的 RoPE phase 用 average pooling 对齐。

![](assets/FramePack%20-%20%E5%9B%BA%E5%AE%9A%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A2%84%E7%AE%97%E7%9A%84%E9%95%BF%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90/figures/01-packing-method.png)

> **图 1（Figure 1，PDF p.4）**：左侧先按时间或特征相似度重排历史；右侧给出几何级数、层级重复、时间 kernel、对称级数等 GPU token 布局。FramePack 的本质不是拼接缩略图，而是为每组历史帧选择不同 3D patchify 粒度。来源：<https://arxiv.org/abs/2504.12626>，官方源码 `imgs/pack2.pdf`。

---

## 5. 防漂移机制与错误传播

### 5.1 端点规划与采样顺序

纯因果链近似：

$$P(\mathbf X_t\mid\mathbf X_{t-1}),$$

前一步误差必然进入后一步。若先建立未来锚点，则中间段近似：

$$P(\mathbf X_t\mid\mathbf X_{t_1},\mathbf X_{t_2}),\quad t_1<t<t_2,
$$

获得双向约束。

- **Endpoint planning**：第一轮同时生成开头与结尾，后续填空；适合周期/小范围动作。
- **Inverted sampling**：I2V 中从生成终点反向填到用户高质量首图，误差传播方向朝可靠输入。
- **Multiple endpoints**：不同 prompt 先生成多个远距离端点，再分段填充，适合故事规划。
- **Random-access RoPE**：跳过尚未生成位置的 phase，使非连续时间索引合法。

### 5.2 History discretization

训练看到真实连续 latent，推理看到模型生成 latent；分布可被网络识别，形成 observation bias。作者对大规模 latent pixels 做 K-means codebook $\Omega\in\mathbb R^{K\times C}$：

$$
Q(\mathbf F)_p=\arg\min_k\|\mathbf F_p-\Omega_k\|_2,
$$

训练时用 $\Omega_{Q(\mathbf F)}$ 替代连续历史。适当量化让真实/生成 latent 落到相同 codeword，降低差异敏感性。$K=1$ 完全切断记忆；$K\to\infty$ 等价于不量化。正文称 $K=128$ 是良好折中，Table 2 代表配置使用 $K=256$。

![](assets/FramePack%20-%20%E5%9B%BA%E5%AE%9A%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A2%84%E7%AE%97%E7%9A%84%E9%95%BF%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90/figures/02-drift-prevention.png)

> **图 2（Figure 2，PDF p.5）**：阴影块为本轮生成。端点/反向/多端点采样从时间拓扑上截断误差链；右侧离散历史从表示空间缩小 train–test gap。前者是推理调度，后者需要训练适配。来源：<https://arxiv.org/abs/2504.12626>，官方源码 `imgs/sampling_v2.pdf`。

### 5.3 错误传播路径

$$
\text{当前段连续latent误差}\rightarrow\text{作为高权重近历史}\rightarrow\text{下一段复制}\rightarrow\text{逐段漂移}.
$$

FramePack 记住更多历史时，早期错误也可能被压缩后长期存在；feature similarity 还可能主动把相似错误重新提升为高分辨率上下文。端点法通过改变依赖图阻断传播，离散化通过让小误差映射到同一 codeword 抑制传播，但都不能纠正语义上稳定、量化后仍一致的错误。

---

## 6. 训练、推理、数据与评测

### 6.1 训练与系统信息

- 基础：HunyuanVideo 13B、Wan；
- 分辨率：主文 480p；
- 训练：H100 clusters，具体 GPU 数/steps/优化器未披露；
- 容量示例：单个 8×A100-80G 节点做 13B LoRA，window 2/3 时 batch 约 64，window 4/5 时 batch 32；
- 数据：遵循 LTXVideo 数据收集流程，覆盖多分辨率与质量等级，但数据规模、来源组成、过滤阈值未披露；
- 论文声称 13B 模型千帧推理可在 6GB/8GB laptop GPU 上运行，但未在正文给吞吐、时长与完整硬件表。

### 6.2 评测集

- T2V：512 个真实用户 prompts；
- I2V：512 个真实用户 image-prompt pairs；
- 默认长视频 30 秒，短视频 5 秒；
- 每种 ablation 100 个结果，A/B 测试保证至少 100 次评价。

### 6.3 指标

全局：MUSIQ clarity、LAION aesthetic、VBench motion/dynamic、ViCLIP semantic、VBench anatomy、ArcFace identity。

漂移指标比较视频开头/结尾各 15%：

$$
\Delta_{drift}^{M}(V)=|M(V_{start})-M(V_{end})|.
$$

绝对值使正向/反向生成都可评，但无法区分“结尾变差”与“结尾意外变好”，也可能把有意剧情变化误判为 drift。

---

## 7. 主要定量与消融结果

### 7.1 核心结论

- inverted anti-drifting 在 7 个全局指标中 4 项最佳，并在全部 drift 指标上最优/并列最优；
- 代价是 dynamic range 较小，说明它倾向稳定、保守运动；
- vanilla sampling 的 dynamic 高可能来自失控漂移，而非真实动态质量；
- discrete history 保留较大动态范围，同时 ELO 达到第一档。

代表行：

| 方案 | Clarity↑ | Motion↑ | Dynamic↑ | Semantic↑ | $\Delta$Clarity↓ | $\Delta$Motion↓ | $\Delta$Semantic↓ | $\Delta$Anatomy↓ | ELO↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla `...g9` | 67.15% | 94.08% | 92.97% | 20.73% | 3.18% | 3.42% | 7.45% | 18.05% | 1074 |
| Endpoint anti-drift `...g9_x_f1k1` | 69.51% | 96.97% | 77.41% | 23.03% | 2.72% | 2.58% | 4.75% | 13.73% | 1142 |
| Inverted anti-drift 代表配置 | 70.48% | 98.16% | 89.18% | 27.01% | **2.18%** | 1.85% | **2.54%** | 11.71% | 1230 |
| Vanilla + discrete history 代表配置 | 69.78% | 96.91% | 91.39% | 24.40% | 2.30% | 2.49% | 4.12% | 14.11% | 1222 |

![](assets/FramePack%20-%20%E5%9B%BA%E5%AE%9A%E4%B8%8A%E4%B8%8B%E6%96%87%E9%A2%84%E7%AE%97%E7%9A%84%E9%95%BF%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90/figures/03-ablation-results.png)

> **图 3（Table 1，PDF p.8）**：四组配置覆盖 vanilla、endpoint anti-drifting、inverted anti-drifting、discrete history。各组内 packing 尾部和 kernel 变化影响较小，真正拉开差距的是采样依赖图和历史表示。来源：<https://arxiv.org/pdf/2504.12626>，官方 PDF 完整表格裁图。

### 7.2 与替代方案

Table 2 中两个代表 FramePack 候选 ELO 1220/1224、Rank 1；Anchor frames 与部分 DiffusionForcing 约 1170、Rank 2；causal attention 1087、Rank 4；重复 I2V 1015、Rank 5。Inverted 方案四项漂移差分别约 2.25/1.85/2.68/8.58%，均优于比较方法；discrete history 的 Dynamic 91.74%，比 inverted 89.29% 更活跃。

### 7.3 证据边界

实验强力支持“防漂移采样/离散历史有效”，但对论文另一半“packing 本身如何优于固定滑窗、FAR 等长上下文结构”的独立受控证据较弱。很多表格是在 FramePack 内比较调度，不是只改 packing 与否。

---

## 8. 失败案例与适用边界

论文没有专门 failure figure，失败需从结果与机制推断：

- **反向/端点法运动保守**：dynamic score 下降，复杂非周期动作可能被锚点夹住；
- **端点本身会漂**：多端点只能让间隔足够远时降低传播，不能保证每个 endpoint 正确；
- **极旧帧信息消失**：指数压缩最终小于一个 latent pixel，只能删除/单 token/全局池化；这不是无损记忆；
- **相似度循环依赖**：排序需估计 $\hat{\mathbf X}$，估计错会提升错误历史；连续帧的相似度排序还可能突变，故需时间平滑；
- **离散历史过粗**：小 $K$ 减漂移但段间失去细节/连续性，大 $K$ 又回到 train–test gap；
- **生成顺序受约束**：需要自由实时交互时，未来 endpoint 未知，inverted/planned 方法并不总可用。

论文没有给具体视觉失败例、显存/速度随千帧变化曲线、长期语义事件记忆或 3D 重访一致性测量，应明确记为信息缺口。

---

## 9. 局限、Future Work 与复现成本

### 9.1 局限

- 无显式几何或对象状态，对空间重访只能靠特征相似与压缩外观。
- 固定 token 上限以信息损失换可扩展性，“处理所有历史”不等于“保留所有历史”。
- 论文主表非常大但缺置信区间；ELO 每配置至少 100 次，统计稳定性有限。
- 数据规模、完整训练 recipe、H100 集群规模未披露。
- feature/hybrid packing 声称在补充材料评估，当前 arXiv v3 源码中缺少对应详细结果。
- 端点规划需要预知未来，不适合所有实时交互式 world model。

### 9.2 Future Work

可把重要性从 frame-level 扩展为 region/token-level，使用学习式路由；将几何/对象/身份相似度与时间混合；把离散历史改为可学习 tokenizer；结合 on-policy rollout 训练，直接让模型处理自身误差；在世界模型中与 VMem/SPMem 的空间寻址结合。

### 9.3 复现成本

**中高。** 开源仓库降低了推理门槛，6–8GB GPU 可跑优化版；但论文级 13B LoRA 训练需要 8×A100-80G 或 H100 集群。代码层需要多 kernel 输入投影、RoPE pooling、非顺序采样和可选 K-means codebook。由于训练数据和完整 recipe 不透明，复现 demo 容易，复现表格较难。

---

## 10. 个人判断与记忆路线比较

### 10.1 个人判断

FramePack 的理论亮点是一个非常干净的预算证明：历史 token 的几何级数收敛，使序列长度与视频时长解耦。更重要的是它没有把 forgetting 和 drifting 混为同一问题，而是分别改记忆结构和误差传播拓扑。

但“无限历史”容易被宣传性误读：很老的帧只剩极少 token，无法承担精确场景复现。它更适合角色身份、整体风格、动作与一般长视频，而不是需要像素级回环一致的空间世界。

### 10.2 四篇横向定位

| 方法 | 记忆表示 | 检索/分配依据 | 几何 | 错误传播控制 |
|---|---|---|---|---|
| [[Context as Memory - 历史上下文驱动的场景记忆]] | 取回的完整 RGB 帧 | 相机 FOV 共视 | 弱 | 早期帧可减误差，但无专门抗 drift |
| [[SPMem - 长期空间记忆视频世界模型]] | 静态点云+近帧+关键帧 | 点云渲染/未知区 | 强 | 未解决 drift；几何闭环可放大错误 |
| [[VMem - Surfel索引视图记忆]] | RGB views + surfel indices | 可见表面投票 | 中 | 几何只影响检索，错误隔离较好 |
| **FramePack** | 全历史多尺度 latent tokens | 时间、特征或混合重要性 | 无 | 端点/反向依赖图 + 离散历史 |

如果没有相机位姿、深度或重建器，FramePack 是唯一自然适用的方案；若目标是返回同一房间时恢复门窗，VMem/SPMem 的空间索引更合适；最有潜力的组合是用 VMem 找到少数关键旧视图，再用 FramePack 对其余普通历史做固定预算压缩，并用离散历史/on-policy 训练抑制 drift。
