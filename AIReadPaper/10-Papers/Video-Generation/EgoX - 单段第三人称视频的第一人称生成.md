---
title: "EgoX - 单段第三人称视频的第一人称生成"
aliases:
  - "EgoX"
  - "Egocentric Video Generation from a Single Exocentric Video"
authors:
  - Taewoong Kang
  - Kinam Kim
  - Dohyeon Kim
  - Minho Park
  - Junha Hyung
  - Jaegul Choo
year: 2025
venue: CVPR 2026
arxiv: "2512.08269"
tags:
  - paper
  - video-generation
  - camera-control
  - exo-to-ego
  - novel-view-synthesis
status: detailed-read
created: 2026-10-23
source: "https://arxiv.org/abs/2512.08269"
---

# EgoX：单段第三人称视频的第一人称生成

## 1. 基本信息

**论文全名**：*EgoX: Egocentric Video Generation from a Single Exocentric Video*。KAIST AI 与首尔国立大学合作（三人共同一作，Jaegul Choo 通讯），CVPR 2026 接收。官方全文见 [arXiv:2512.08269](https://arxiv.org/abs/2512.08269)，项目页 [keh0t0.github.io/EgoX](https://keh0t0.github.io/EgoX/)，本地 PDF 为 `assets/EgoX - 单段第三人称视频的第一人称生成/sources/paper.pdf`。

> **一句话结论**：EgoX 把「单段第三人称视频 → 第一人称视频」做成了一个大模型条件生成问题：用单目+视频深度对齐后点云渲染出像素对齐的 ego prior，把「像素对齐的 ego prior 走通道拼接、需隐式 warp 的 exo 全景走宽度拼接」当作一套条件注入策略，再配合以 3D 方向余弦为逐对偏置的 Geometry-Guided Self-Attention 抑制无关区域——冻结 Wan2.1 14B 主干、只训 LoRA，就在 Ego-Exo4D 上把 PSNR 从 14.5 拉到 16.05、FVD 从 508 拉到 184，并泛化到《黑暗骑士》这类 in-the-wild 电影片段。

![EgoX teaser](assets/EgoX%20-%20%E5%8D%95%E6%AE%B5%E7%AC%AC%E4%B8%89%E4%BA%BA%E7%A7%B0%E8%A7%86%E9%A2%91%E7%9A%84%E7%AC%AC%E4%B8%80%E4%BA%BA%E7%A7%B0%E7%94%9F%E6%88%90/figures/01-teaser.png)

*图 1 解读：输入《黑暗骑士》中 Joker 的第三人称片段，EgoX 直接生成他眼中的第一人称画面——手部动作、远处爆炸与烟雾都在正确位置出现。这类 in-the-wild 输入没有 GT、也没做任何微调，展示的是纯预训练知识的迁移。来源：论文 Figure 1，Page 1。*

## 2. 研究背景与问题定义

**任务定义**：给定第三人称（exocentric）视频 $X=\{X_i\}_{i=0}^F$ 和第一人称相机轨迹 $\phi=\{\phi_i\}_{i=0}^F$，生成同场景的第一人称（egocentric）视频 $Y=\{Y_i\}_{i=0}^F$。要求同时做到两件事：忠实重建 exo 视角下可见的内容，并以几何一致的方式合理合成 ego 视角下不可见的区域。

这个任务比普通相机控制难一个量级。现有相机控制模型（CameraCtrl、ReCamMaster、TrajectoryCrafter 等）针对的是**适度视角变化**——新视角与原视角有大量重叠；而 exo→ego 需要极端的相机位姿平移，视野几乎完全更换。论文把难点拆成两条：

1. 极端视角偏移带来大片**未知区域**（uninformed regions），必须靠场景理解而非直接观测去合成；
2. exo 视频中只有一小部分与 ego 视角相关，模型必须**区分该用的相关信息与该抑制的无关内容**。

![挑战示意](assets/EgoX%20-%20%E5%8D%95%E6%AE%B5%E7%AC%AC%E4%B8%89%E4%BA%BA%E7%A7%B0%E8%A7%86%E9%A2%91%E7%9A%84%E7%AC%AC%E4%B8%80%E4%BA%BA%E7%A7%B0%E7%94%9F%E6%88%90/figures/02-challenge-concept.png)

*图 2 解读：以 ego 相机为中心把 exo 画面分成三类——不相关（unrelated）、部分相关（partially related）、未知（uninformed）。有效生成要求只保留相关内容、丢弃无关背景、并在未知区域做几何一致的补全。这也是全文方法设计（条件注入 + GGA）的问题原型。来源：论文 Figure 2，Page 2。*

**前人的回避策略**说明了任务难度：EgoExo-Gen 需要额外输入第一帧 ego 图像；Exo2Ego-V 需要**四个**同步 exo 相机来减少未知区域，且空间/时间模块分开训练。EgoX 的卖点就是把这些附加条件全部砍掉：单一 exo 视频 + 一条 ego 轨迹。

## 3. 核心方法

总体流水线分三步：先把 exo 视频抬升为 3D 点云并从目标 ego 轨迹渲染出 **ego prior 视频**（Sec 3.1）；再把 ego prior 与 exo 视频以不同方式拼进视频扩散模型（Sec 3.2）；最后在 self-attention 里加入几何引导偏置（Sec 3.3）。

![方法总览](assets/EgoX%20-%20%E5%8D%95%E6%AE%B5%E7%AC%AC%E4%B8%89%E4%BA%BA%E7%A7%B0%E8%A7%86%E9%A2%91%E7%9A%84%E7%AC%AC%E4%B8%80%E4%BA%BA%E7%A7%B0%E7%94%9F%E6%88%90/figures/03-method-pipeline.png)

*图 3 解读：exo 视频经 3D lifting（单目+视频深度对齐）得到点云，从 ego 轨迹栅格化出 ego prior 视频；两者分别过冻结 VAE encoder 后，$z_t$ 与 $p_0$ 通道拼接（C）、与 $x_0$ 宽度拼接（W），连同二值 mask 一起送入 LoRA 微调的 DiT，DiT 内部用 Geometry-Guided Self-Attention 代替普通自注意力。采样完成后只解码 ego 部分。来源：论文 Figure 3，Page 4。*

### 3.1 Ego prior 的点云渲染

先给每帧估计两种深度：单图深度估计器输出的 $D^m$（逐帧独立、细节好但时间上抖动）和视频深度估计器（Video Depth Anything）输出的 $D^v$（时间平滑但只有仿射不变的相对尺度）。融合方式是：只取静态背景区域（动态物体被 mask 掉），用动量更新策略逐帧优化仿射参数 $\hat\alpha,\hat\beta$，把 $D^v$ 对齐到 $D^m$ 的尺度：

$$D^f=\frac{1}{\hat\alpha/D^v+\hat\beta}$$

得到的对齐深度 $D^f$ 配合相机内参转成 3D 点云，用点云渲染器从 ego 轨迹渲染：

$$P=\text{render}(X, D^f, \phi)$$

这个 ego prior 视频 $P$ 同时携带两类信息：显式的逐像素 RGB（被点云覆盖的区域）和隐式的相机轨迹线索（空洞的分布暗示视角走向）。附录 Figure 9 显示深度对齐这一步不可省略——不对齐时即使静态背景也会逐帧漂移，等价于给模型灌入虚假的相机运动。

### 3.2 双条件注入策略

这是全文第一处关键设计。底座是 **Wan2.1 (14B) Image-to-Video 的 inpainting 变体**——选 inpainting 版正是为了原生支持「噪声潜变量 + 条件潜变量」的通道拼接。两种条件的拼接方式刻意不同：

- **Ego prior $p_0$ 走通道拼接**：$p_0$ 与目标视频视角完全一致、逐像素对齐，通道拼接可以把这种细粒度对应关系直接注入每个 token，等价于给模型一张「粗糙但位置正确的答案纸」。
- **Exo 潜变量 $x_0$ 走宽度拼接**：$x_0$ 与 $z_t$ 视角不同、无像素对应关系，硬做通道拼接反而破坏空间结构；宽度拼接让两者在 token 序列里保持各自的 2D 布局，把「跨视角对应 + 空间 warp」留给模型在 attention 中隐式学出来。

整体关系写作：

$$z_{t-1}=f_\theta(x_0, z_t \mid x_0, p_0, m^1, m^0)$$

其中 $m^1, m^0$ 是二值 mask，标记每个空间区域是条件还是待合成区域。

第二处关键设计是 **clean latent**：与 SDEdit 式的做法（条件潜变量也带噪）不同，$x_0$ 在全部去噪步中始终保持干净潜变量形态，只有 $z_t$ 被更新。这样模型在任何 timestep 都能从 $x_0$ 引用细粒度细节，消融显示去掉后生成画面会丢失勺子、案板上的小配料这类小物体（FVD 从 184 恶化到 343）。采样完成后把潜变量中 exo 部分裁掉，只解码 ego 部分。

### 3.3 Geometry-Guided Self-Attention（GGA）

宽度拼接解决「怎么把 exo 塞进来」，GGA 解决「该看 exo 的哪里」。当 ego query token $q^{ego}$ 对 exo key token $k^{exo}$ 计算注意力时，理想的相关性应该同时考虑语义相似（外观）与 3D 空间对齐。EgoX 用 3.1 的点云给出几何先验：从 ego 相机中心 $c_i$ 到每个 query/key token 位置 $\tilde q,\tilde k$ 计算单位方向向量 $\hat q,\hat k$，再以方向余弦构造偏置加进注意力 logits。**注意两个向量共用同一个起点 $c_i$（当帧 ego 相机中心），而不是各自从自己的相机出发**——这正是 cos 可比的前提；并且 $\hat q,\hat k$ 只依赖相机中心**位置**，不依赖其朝向，朝向信息已由 query 自身的视线方向编码。

$$s'_{m,n}=s_{m,n}+\log\left(g(\hat q_m,\hat k_n)\cdot\lambda_g\right),\qquad g(\hat a,\hat b)=\cos\text{sim}(\hat a,\hat b)+1$$

![GGA 示意](assets/EgoX%20-%20%E5%8D%95%E6%AE%B5%E7%AC%AC%E4%B8%89%E4%BA%BA%E7%A7%B0%E8%A7%86%E9%A2%91%E7%9A%84%E7%AC%AC%E4%B8%80%E4%BA%BA%E7%A7%B0%E7%94%9F%E6%88%90/figures/04-gga.png)

*图 4 解读：同一个 key token（红色）因相机中心不同，相对不同帧 ego 相机的方向向量完全不同（红 vs 橙）。方向相近的 blue–red 对获得高分，方向相反的 green–orange 对被压低。这说明几何偏置必须逐帧逐对计算，不能复用静态旋转矩阵编码（如 4Diff 那样的图像 RoPE 方案）。来源：论文 Figure 4，Page 5。*

由于视频里 ego 相机中心逐帧变化，方向偏置必须对每一帧的每对 query–key 单独计算。实现上（附录 F.1）：在像素空间计算 3D 方向后按 VAE 下采样因子（4×16×16 patch）取平均降到潜空间，偏置项在推理前**一次性预计算**成加性 attention mask，从而复用优化的 attention kernel；ego-to-exo 与 exo-to-ego 两个方向的 attention 分开用不同 kernel，避免全层加偏置爆显存。最终注意力权重为：

$$a_{m,n}=\frac{\exp(s_{m,n})\,g(\hat q_m,\hat k_n)^{\lambda_g}}{\sum_{j=1}^{l}\exp(s_{m,j})\,g(\hat q_m,\hat k_j)^{\lambda_g}}$$

附录 H.1 强调 GGA **必须在训练时就启用**：只推理时加偏置，模型面对从未见过的注意力分布，质量反而明显下降。

## 4. 实验

### 4.1 实验设置

- **数据**：从 Ego-Exo4D 精选 4000 段（3600 训练 / 400 测试），另收集 100 段未见过场景测泛化。文本 prompt 用 GPT-4o 按固定 system prompt 生成（同时描述 exo 与 ego 视角，附录 Table 6 给出完整 prompt，要求穷举可见物体、聚焦手部交互、禁止主观形容词）。
- **训练**：Wan2.1 14B I2V inpainting 变体 + LoRA（rank=256），batch size 1，**8×H200 训练一天**。
- **基线**：Exo2Ego-V（四相机输入的专用模型）、TrajectoryCrafter（SOTA 相机控制）、Wan Fun Control（通道拼接条件）、Wan VACE（辅助条件网络）——所有基线用同一数据微调以保证公平。
- **指标**：三层——图像级（PSNR/SSIM/LPIPS/CLIP-I）、物体级（SAM2 分割 + DINOv3 匹配，匹配阈值余弦相似度 ≥0.9，再算中心位置误差 / bbox IoU / 轮廓精度）、视频级（FVD + VBench 时序闪烁 / 运动平滑 / 动态程度）。

### 4.2 主要结果

| 场景 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP-I↑ | 位置误差↓ | IoU↑ | 轮廓精度↑ | FVD↓ | 动态程度↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Seen | Exo2Ego-V | 14.53 | 0.384 | 0.569 | 0.774 | 156.66 | 0.074 | 0.364 | 622.47 | 0.985 |
| Seen | TrajectoryCrafter | 13.05 | 0.375 | 0.606 | 0.780 | 100.74 | 0.128 | 0.427 | 546.09 | 0.947 |
| Seen | Wan Fun Control | 12.25 | 0.463 | 0.617 | 0.810 | 112.57 | 0.076 | 0.417 | 595.07 | 0.901 |
| Seen | Wan VACE | 12.95 | 0.413 | 0.626 | 0.829 | 109.62 | 0.114 | 0.376 | 508.69 | 0.673 |
| Seen | **EgoX** | **16.05** | **0.556** | **0.498** | **0.896** | **61.81** | **0.363** | **0.546** | **184.47** | 0.974 |
| Unseen | Exo2Ego-V | 12.70 | 0.439 | 0.597 | 0.679 | 214.32 | 0.003 | 0.296 | 1283.50 | 0.978 |
| Unseen | **EgoX** | **14.38** | **0.457** | **0.552** | **0.877** | **149.93** | **0.092** | **0.481** | **440.64** | 0.989 |

（节选自论文 Table 1，只保留首尾方法；完整表含 TrajectoryCrafter / Wan Fun Control / Wan VACE 的 unseen 数字。）

三个观察值得注意：

1. **物体级指标拉开最大差距**：seen 场景 IoU 0.363 对基线最高 0.128，接近 3 倍——说明 EgoX 保住的不是像素统计相似，而是场景几何与物体一致性。这正是「ego prior 给对齐、GGA 抑制无关」两件事的直接收益。
2. **Wan VACE 的时序指标陷阱**：Wan VACE 在时序闪烁/运动平滑上得分最高，但论文明确指出这是因为它生成了几乎静止的视频（动态程度只有 0.673）——「平滑」与「正确」不是一回事，读这类对比表要盯住动态程度列。
3. **unseen 场景衰减可控**：PSNR 16.05→14.38、IoU 0.363→0.092 有下降，但相对基线的优势结构不变；Exo2Ego-V 在 unseen 上 IoU 直接掉到 0.003，基本失效。

![定性对比](assets/EgoX%20-%20%E5%8D%95%E6%AE%B5%E7%AC%AC%E4%B8%89%E4%BA%BA%E7%A7%B0%E8%A7%86%E9%A2%91%E7%9A%84%E7%AC%AC%E4%B8%80%E4%BA%BA%E7%A7%B0%E7%94%9F%E6%88%90/figures/05-qualitative.png)

*图 5 解读：seen / unseen / in-the-wild 三档定性对比。in-the-wild 一档里 Exo2Ego-V 与 ego GT 均为 N/A（前者需要四相机、后者无真值），而 EgoX 从单段 exo 视频生成了烟花、X 光检查等场景的合理第一人称画面。TrajectoryCrafter 在大位移下出现空间扭曲，Wan VACE / Fun Control 则把 exo 视角的内容错误地「印」进了 ego 视角。来源：论文 Figure 5，Page 6。*

用户研究（19 人，20 段 unseen + 10 段 in-the-wild）在重建准确性、运动/相机一致性、整体质量三个问题上 EgoX 均显著领先全部四个基线。

### 4.3 消融研究

| 变体（seen 场景） | PSNR↑ | SSIM↑ | LPIPS↓ | 位置误差↓ | IoU↑ | FVD↓ | 动态程度↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EgoX 完整版 | **16.05** | **0.556** | **0.498** | **61.81** | 0.363 | **184.47** | 0.974 |
| w/o GGA | 14.77 | 0.539 | 0.530 | 64.30 | 0.326 | 254.08 | 0.877 |
| w/o Ego prior | 13.67 | 0.479 | 0.573 | 90.70 | **0.417** | 211.50 | 0.802 |
| w/o clean latent | 15.07 | 0.528 | 0.540 | 70.17 | 0.376 | 343.33 | 0.864 |
| Prior 走宽度、Exo 走通道 | 13.83 | 0.471 | 0.594 | 83.08 | 0.213 | 274.14 | 0.915 |
| 两者都走宽度 | 14.85 | 0.499 | 0.545 | 71.93 | 0.261 | 242.83 | 0.910 |
| GGA 仅推理时启用 | 15.23 | 0.540 | 0.521 | 64.34 | 0.324 | 193.82 | 0.899 |

（综合论文 Table 2 与 Table 4。）

三个消融结论都有信息量：

- **条件布局不是可换的实现细节，而是本质设计**。把 ego prior 与 exo 的拼接方向互换（prior 宽度、exo 通道），PSNR 掉 2.2、IoU 掉 44%；都走宽度（需 resize 压回原尺寸）同样全面变差。原因正如方法节所料：像素对齐的信息就该走通道，需要 warp 的信息就该保留空间结构走宽度。Exo 走通道时 GGA 甚至无法运行（空间结构已破坏）。
- **ego prior 与 GGA 分工不同**：去掉 ego prior 后 IoU 反而微升（0.417），但位置误差暴涨（90.7）、动态程度暴跌（0.802）——没有点云先验模型更「自由发挥」，物体外观可能不错但视角/轨迹跟随崩坏；去掉 GGA 则是几何对齐变差 + 注意力涣散。
- **clean latent 主要保护小物体细节**：定性上表现为勺子、案板小配料这类物件的消失，FVD 恶化幅度（184→343）比 PSNR 更明显。

![消融对比](assets/EgoX%20-%20%E5%8D%95%E6%AE%B5%E7%AC%AC%E4%B8%89%E4%BA%BA%E7%A7%B0%E8%A7%86%E9%A2%91%E7%9A%84%E7%AC%AC%E4%B8%80%E4%BA%BA%E7%A7%B0%E7%94%9F%E6%88%90/figures/06-ablation.png)

*图 6 解读：三个消融变体与完整版的定性对比（足球场/自行车/厨房三例）。w/o GGA 出现视野偏移与无关内容混入；w/o Ego Prior 视角跟随错误；w/o clean latent 丢失小物体。来源：论文 Figure 6，Page 8。*

![注意力可视化](assets/EgoX%20-%20%E5%8D%95%E6%AE%B5%E7%AC%AC%E4%B8%89%E4%BA%BA%E7%A7%B0%E8%A7%86%E9%A2%91%E7%9A%84%E7%AC%AC%E4%B8%80%E4%BA%BA%E7%A7%B0%E7%94%9F%E6%88%90/figures/07-attention.png)

*图 7 解读：以 ego 视角中心 token 为 query 的注意力图。w/o GGA 时注意力大面积洒在无关区域；w/ GGA 后集中到几何上对应的可见区域。附录 Figure 10 还展示了一个更微妙的效应：没有 GGA 时模型会attend到 exo 画面中「视野外发生的事件」，从而在 ego 视角凭空生成不该出现的事件——GGA 的价值不只是对齐，还有「不该看见的就不要生成」。来源：论文 Figure 7，Page 8。*

![深度对齐](assets/EgoX%20-%20%E5%8D%95%E6%AE%B5%E7%AC%AC%E4%B8%89%E4%BA%BA%E7%A7%B0%E8%A7%86%E9%A2%91%E7%9A%84%E7%AC%AC%E4%B8%80%E4%BA%BA%E7%A7%B0%E7%94%9F%E6%88%90/figures/08-depth-align.png)

*图 8 解读：点云渲染的深度对齐消融。上排（无对齐）逐帧深度尺度不一致，静态背景在帧间漂移，等价于虚假相机运动；下排（有对齐）点云随时间稳定。这一步借用 ViPE 的动量仿射对齐，成本极低但决定了 ego prior 是否可用。来源：论文 Figure 9，Page 12（Supplementary）。*

### 4.4 运行开销

附录 Table 5：完整版去噪阶段约 10.5 分钟（H200 单卡），去掉 GGA 约 6.5 分钟——GGA 的逐对偏置带来约 60% 额外开销；ego prior 本身的点云渲染 <10 秒，clean latent 零额外开销。对非交互场景可接受，但离「可用」还有距离。

## 5. 局限与评价

### 5.1 论文自己承认的局限

1. **ego 相机轨迹 $\phi$ 是「给定输入」而非模型估计量**。这一点在任务定义（Sec 3 开头 `Given X and egocentric camera pose ϕ`）里就定死了：EgoX 解决的是「已知轨迹下的视角转换」，不是「估计轨迹」。轨迹来源有三档：**Ego-Exo4D 提供真值**（Aria 眼镜 SLAM 轨迹 + 多 exo 相机联合标定，两者在同一场景世界坐标系下）；**in-the-wild 例子由作者用 Viser 在 lift 出的 3D 点云里手动摆放相机外参**（附录 F.2，Figure 8）——因为是直接在点云里摆，坐标系天然与 exo 点云一致；**未来方向**是接视频头姿跟踪器或 SMPL 系位姿估计器。坐标系一致性是硬要求：ego 位姿必须与 exo 点云同处一个世界系，否则方向余弦无意义。没有自动头部位姿估计，这条流水线就不闭环，这是全文最大的工程缺口。
2. **任务歧义导致的失败**（附录 H.5 / Figure 12）：exo 视角下单臂弯曲、另一臂被遮挡时，模型可能生成双臂都伸直的 ego 画面。作者认为这更多是任务本身信息不足（人类观察者也难判断），而非模型缺陷——这个辩护成立，但也意味着该任务存在无法从单段 exo 视频消除的固有不确定性。

### 5.2 我的评价

**优点**：

- **把一个「专用模型」问题改写成「预训练大模型 + 条件设计」问题**，是典型的 GenAI 时代打法：Wan2.1 inpainting 变体天然提供了通道拼接接口，LoRA r=256、8 卡一天的成本说明这个方向对普通实验室也可复现。
- **条件拼接布局的消融（Table 4）是全文最有教学价值的部分**：「像素对齐→通道、需 warp→宽度」不是玄学，而是把信息结构（是否有逐像素对应）映射到注入方式的设计原则，可推广到参考图驱动生成、多视角条件等一整类问题。
- GGA 的「预计算加性偏置 mask + 分方向 attention kernel」是工程上成熟的几何先验注入方式，与 Ctrl-X / RoPE 位置编码一族思路相通，不增加可学习参数。

**不足与疑问**：

- **评测协议偏宽松**：测试与训练同源（Ego-Exo4D 精选子集），unseen 也只是「未见过场景」而非域外；in-the-wild 完全没有量化指标（无 GT），只有定性 + 用户研究。指标绝对值（PSNR 16）也说明像素级重建远未解决，这个任务本质上以合成补全为主，PSNR/SSIM 的解释力有限。
- **深度估计是单点依赖**：单目深度 + 视频深度对齐的质量直接决定 ego prior 的可用性（论文 Figure 9 只展示了正面案例），在反光、透明、快速运动物体密集的场景下点云先验可能崩坏，但论文没有给出 prior 失败时的行为分析。
- **推理 10.5 分钟/段**且 14B 底座，距离实时或交互式应用（AR/VR 的实际卖点场景）还很远。
- 动态物体被 mask 掉后只渲染静态背景，ego 视角中占画面很大比例的**自己的手/身体**实际全靠生成先验补全——论文没有专门分析手部动作的保真度（而这对模仿学习恰恰最重要），只在用户研究里间接覆盖。

**与已有笔记的关联**：EgoX 是 [[GEN3C - 三维感知的世界一致视频生成]] 中「几何负责约束、扩散负责补全」思想的近亲，但它走了相反的路线——GEN3C 用显式 3D cache 渲染强条件、要求相机可重投影区域占多数；EgoX 面对几乎零重叠的极端视角，把 3D 降级为「部分先验 + 注意力偏置」，把跨视角对应交还模型。两者合起来给出了一个谱系：**相机可控性越强，显式几何比重越高；视角差越大，生成先验比重越高**。此外它用的 Wan2.1 条件接口与 [[Wan2.1 - 开源大规模视频生成基础模型]] 的 inpainting 变体直接相关。

**复现成本**：数据公开（Ego-Exo4D）、底座开源（Wan2.1）、深度估计器与渲染器均为开源组件；主要成本是 8×H200 一天的 LoRA 训练与 GGA 的逐对偏置实现。论文未宣布开源代码（截至阅读时项目页未见 release），但方法描述足够详细，属于「可以复现但需自己补工程」的级别。

**总评**：★★★★☆。任务定义清晰、消融扎实、设计原则（条件布局映射信息结构 + 几何先验做注意力偏置）有超出本文的迁移价值；扣分在 ego 轨迹依赖手动指定、域外评测缺失与推理开销。
