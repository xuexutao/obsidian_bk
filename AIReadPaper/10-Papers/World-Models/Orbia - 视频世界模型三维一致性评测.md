---
type: paper-note
status: complete
domain: World-Models
paper: "Orbia: Do Generated Videos Form a Consistent 3D World?"
year: 2026
arxiv: "2610.20693"
doi: "10.48550/arXiv.2610.20693"
source: https://arxiv.org/abs/2610.20693
project: https://orbia-bench.github.io/
code: https://github.com/worldbench/Orbia
dataset: https://huggingface.co/datasets/Orbia/Orbia
tags:
  - World-Models
  - Benchmark
  - 3D-Consistency
  - Video-Generation
  - Revisit
created: 2026-10-21
updated: 2026-10-21
---

# Orbia：视频世界模型三维一致性评测

## 基本信息

- **正式题名**：Orbia: Do Generated Videos Form a Consistent 3D World?
- **版本**：arXiv v2，2026-10-18；70 页技术报告，43 图、12 表
- **团队**：NUS、Lightspeed 等，作者包括 Dongyue Lu、Lingdong Kong、Mike Zheng Shou、Shuicheng Yan 等
- **类型**：Benchmark / evaluation protocol
- **规模**：720 个实例；480 个约 22 秒短序列、240 个约 60 秒分钟级扩展；6 种探索—重访协议
- **评测对象**：27 个模型，其中 19 个 camera/action-conditioned world models、8 个 text-image-to-video 模型
- **资源**：[[assets/Orbia - 视频世界模型三维一致性评测/sources|来源记录]]；[官方 PDF](assets/Orbia%20-%20视频世界模型三维一致性评测/sources/paper.pdf)

## 一句话结论

Orbia 观察到当前视频世界模型常用画质、单项几何或 VLM 主观判断来证明“一致性”，但逼真的连续帧并不保证它们描述同一个三维世界；它因此把 3D consistency 拆成**输入内容是否被保留、模型新生成的区域能否在重访时保持、生成视图能否被同一个三维场景解释**三个维度，并用注册 RGB-D 参考、跨视图锚点、6 种可控探索—重访轨迹、跨视图深度/特征/尺度一致性和 held-out Gaussian 重建测试，统一评测 27 个 text、action 和 camera 条件模型。其关键结论是：SolarWM-H3 以 **76.32** 综合第一但没有模型六项全优，world models 的控制均分 57.29 远高于 TI2V 的 32.48，而视觉质量与三项一致性的模型级相关仅 0.33/0.21；分钟级探索会暴露控制下降、内容遗忘和几何退化，因此世界模型必须用“持续探索 + 显式回访 + 三维重建”而非画面好看来评估。

---

## 1. 研究背景与问题定义

### 1.1 世界模型为什么需要专门的 3D consistency benchmark

视频生成模型正在被当作可交互世界使用：用户沿相机轨迹探索，模型持续生成新观测。若模型真正在模拟世界，至少应满足三点：

1. 初始图像中已经观察到的物体和布局不能被随意改写；
2. 模型自己生成的新区域，在离开后回来时仍应存在且外观、深度一致；
3. 不同时间、不同视角生成的帧应能共同解释同一个三维场景。

已有 benchmark 往往只检查相机误差、少量几何属性，或用 VLM 判断“是否一致”。这些信号无法区分模型究竟是记住了输入，还是保留了自己创造的内容，也不能可靠说明多视角是否几何兼容。

![](assets/Orbia%20-%20视频世界模型三维一致性评测/figures/01-teaser.png)

> **图 1（Teaser）**：Orbia 将评测分为输入保留、生成内容持久性、3D 自一致性和长时一致性，并用六种重访协议、三类数据源和统一适配器评估 text/action/camera 模型。来源：<https://arxiv.org/abs/2610.20693>。

### 1.2 统一任务形式

交互式生成模型写作：

$$\hat V=\mathcal M(I_0,s,\mathcal U)$$

其中 $I_0$ 是初始图像，$s$ 是场景描述，$\mathcal U$ 可以是文本指令、离散动作或相机位姿序列。Orbia 要求所有模型从同一初始图像出发执行等价探索—重访任务，再评估输出视频 $\hat V$ 是否构成稳定 3D 世界。

### 1.3 核心贡献

1. 提出首个系统、全面评估交互式世界模型 3D consistency 的 benchmark。
2. 将一致性拆成 input preservation、generated-content persistence、3D self-consistency 三个可诊断维度。
3. 构建覆盖公开数据、自采场景和 UE 模拟的 720 个实例及分钟级重访扩展。
4. 设计六种控制探索—重访轨迹，揭示简单前后运动看不出的失败。
5. 通过 native-interface adapters 横向评估 27 个模型，并验证自动分数与人类排序高度一致。

---

## 2. Benchmark 构建

### 2.1 数据来源

| 来源 | 实例数 | 作用 |
|---|---:|---|
| 公开数据集 | 240 | ScanNet++、SpatialVID、Sekai、MiraData 等真实分布 |
| 自采真实场景 | 60 | 补充可控真实室内外场景 |
| Unreal Engine 模拟 | 180 | 精确几何、可控轨迹和复杂环境 |
| **合计** | **720** | 480 短序列 + 240 分钟级扩展 |

短序列约 22 秒、包含 1 次重访；分钟级扩展约 60 秒、包含 3 次重访。

### 2.2 四阶段构建流程

![](assets/Orbia%20-%20视频世界模型三维一致性评测/figures/02-benchmark-construction.png)

> **图 2（Figure 2）**：从源视频恢复 pose/depth，按模板选择输入—查询视图，再用 VLM+SAM3 发现并验证静态锚点，合成无碰撞轨迹和重访，最后自动打分与人工筛选。来源：<https://arxiv.org/abs/2610.20693>。

**阶段 1：几何引导参考视图选择**

保留已有 3D 几何；缺失时用 ViPE 估计 pose 和 depth。根据模板要求的旋转、平移和可见性选择输入视图 $P_0$ 与查询视图 $P_q$。

**阶段 2：锚点发现与验证**

锚点是两端都可见、独特且静态的地标。流程为：

1. VLM 提出候选物体；
2. SAM3 在输入和查询视图分别分割；
3. 按面积、连通性和完整性过滤；
4. tracking 验证跨视图身份；
5. 用深度反投影检查可见性和深度一致性。

**阶段 3：轨迹合成与重访设计**

把轨迹模板参数拟合到 $P_0,P_q$，最后返回 $P_r=P_0$；检查与场景几何的碰撞和间隙。重访区域按照位姿接近、相对输入的新颖性以及两次观察的共享可见表面选择。

**阶段 4：质量评估与人工策展**

按图像质量、锚点可见性、mask 完整性、相机运动和重访共享表面排序，再进行人工检查，并由 VLM 生成事实性场景描述。论文报告人工标注与策展超过 300 小时。

---

## 3. 六种探索—重访协议

![](assets/Orbia%20-%20视频世界模型三维一致性评测/figures/03-trajectory-templates.png)

> **图 3（轨迹模板）**：六种协议从简单平移返回到环绕锚点、进入遮挡后返回，逐步暴露深度、视差、旋转和新生成内容持久性问题。来源：<https://arxiv.org/abs/2610.20693>。

| 模板 | 运动 | 主要压力 |
|---|---|---|
| T1 Forward-back | 沿视线前进再后退 | 深度和尺度一致 |
| T2 Lateral return | 横移后返回 | 视差与侧向结构 |
| T3 Yaw return | 原地转向后回正 | 旋转稳定性与视野外内容 |
| T4 Arc return | 弧线移动后返回 | 平移旋转耦合 |
| T5 Orbit around anchor | 绕锚点环绕 | 物体身份和多视角对应 |
| T6 Peek-and-return | 探查遮挡区再返回 | 新生成内容是否被记住 |

六种轨迹共享起点 $P_0$ 与终点 $P_r$，并包含交叉视图查询 $P_q$ 或 first visit / revisit 对。

---

## 4. 三大一致性维度

### 4.1 Input Preservation：是否保留初始输入

该维度测试模型是否忠实保留输入图像中已经可见的场景。

- **Scene appearance fidelity**：在输入可见表面比较返回视图和输入、查询视图和 RGB-D 参考，指标为 DINO similarity、PSNR、SSIM、LPIPS；
- **Anchor appearance fidelity**：只在锚点 mask 内计算，逐锚点平均，防止大物体掩盖小物体失败；
- **Scene geometry fidelity**：深度 AbsRel，使用共享场景级尺度；
- **Anchor geometry fidelity**：比较锚点相对周围场景的深度关系；
- **Anchor identity consistency**：SAM3.1 跟踪锚点后计算 DINO 余弦相似度。

深度误差可写为：

$$\mathrm{AbsRel}=\frac1N\sum_i\frac{|d_i^{pred}-d_i^{ref}|}{d_i^{ref}}$$

### 4.2 Generated-content Persistence：模型是否记住自己创造的内容

输入图像没有展示的区域不存在真值，模型可以自由生成。但一旦生成，就应在回访时保持。Orbia 比较 first visit 与 revisit：

- 用深度和相机投影将两次观测配准；
- 在共同可见区域比较 DINO 特征；
- 计算重访深度 AbsRel；
- 用双向最近点 F-score 衡量表面一致：

$$F=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$$

这一维度最能区分“只会复刻输入”和“真正有世界记忆”的模型。

### 4.3 3D Self-consistency：不同视图能否描述同一场景

该维度不依赖请求轨迹，而检查生成帧之间是否几何兼容：

1. **Cross-view depth**：估计相机后双向深度重投影，在有效、未遮挡对应上计算 AbsRel；
2. **Cross-view feature**：在相同对应上计算 DINO 余弦相似度；
3. **Regional scale agreement**：图像分为 $4\times4$ 区域，衡量观测/重投影深度比在区域间是否离散；
4. **Held-out reconstruction**：用 DA3 从 64 帧构建 Gaussian 场景，渲染 16 个未参与重建的生成视角，比较 PSNR、SSIM、LPIPS。

最后一项非常关键：局部相邻帧可以看似一致，但整段视频可能无法由一个三维场景解释。

### 4.4 Long-horizon stability

在 5 秒窗口内追踪视觉质量、控制准确度和局部几何。稳定性同时考虑平均水平与相对初始窗口的保留率；短视频和分钟级视频按 30%/70% 加权。

### 4.5 综合分数

| Tier | 内容 | 权重 |
|---|---|---:|
| T1 | Control alignment | 25% |
| T2 | Visual quality | 25% |
| T3 | Input preservation | 10% |
| T4 | Generated-content persistence | 10% |
| T5 | 3D self-consistency | 10% |
| T6 | Temporal stability | 20% |

三项 3D consistency 合计只占 30%，因此综合排名仍很依赖控制和画质。这是读总榜时必须注意的口径。

---

## 5. 模型适配与实验设置

Orbia 支持三类控制接口：

- **Camera-conditioned**：直接输入请求的外参和内参；
- **Action-conditioned**：把连续轨迹转换为离散动作；
- **TI2V**：把轨迹和场景转成文本描述，以初始图像和 prompt 生成视频。

27 个模型包括 SolarWM-H3、Lyra2、EVOKE、Echo-WM、Gen3C、HY-WorldPlay、Genie3、Seedance2.5、Kling3.0、Cosmos3-Super 等。

TI2V 的控制差距部分来自文本接口天然无法精确表达连续相机轨迹，因此“TI2V 控制差”既是模型能力问题，也是接口问题。

---

## 6. 主要实验结果

### 6.1 World models 总榜

| 排名 | 模型 | 控制 | 画质 | 输入保留 | 生成持久 | 3D 自一致 | 时间稳定 | 总分 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | SolarWM-H3 | 79.93 | 74.76 | 69.26 | **70.28** | 68.84 | **84.05** | **76.32** |
| 2 | Lyra2 | **81.64** | 71.37 | 68.79 | 70.25 | 61.51 | 82.31 | 74.77 |
| 3 | EVOKE | 66.47 | **76.48** | 68.81 | 66.43 | 68.02 | 83.18 | 72.70 |
| 4 | Echo-WM | 70.74 | 75.02 | 65.91 | 61.42 | 65.26 | 82.69 | 72.24 |
| 5 | Gen3C | 75.51 | 67.55 | **75.23** | 68.58 | 60.16 | 78.96 | 71.95 |

HY-WorldPlay 领先 3D self-consistency，但不在综合前五。没有任何模型同时领先六项。

### 6.2 TI2V 模型

| 排名 | 模型 | 控制 | 画质 | 输入保留 | 生成持久 | 3D 自一致 | 时间稳定 | 总分 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Seedance2.5 | **43.23** | 76.97 | **72.84** | 67.70 | 72.54 | **80.39** | **67.44** |
| 2 | Kling3.0 | 38.58 | 77.38 | 69.14 | 63.32 | 71.30 | 79.30 | 65.23 |
| 3 | Cosmos3-Super | 27.92 | **79.04** | 66.04 | **69.75** | **74.55** | 79.73 | 63.72 |

Cosmos3-Super 在 TI2V 内画质、生成持久和 3D 自一致最好，但控制很弱，因此总分第三。

### 6.3 核心发现

1. **控制是 world models 与 TI2V 的最大差距**：均值分别为 57.29 和 32.48，相差 24.80。
2. **显式 3D memory 有优势**：前五 world models 中 Lyra2、EVOKE、Gen3C 都使用不同形式的显式几何/3D cache。
3. **画质不等于一致性**：视觉质量与三项一致性均值的相关，world models 仅 0.33，TI2V 仅 0.21。
4. **输入保留不预测生成内容持久性**：模型能恢复初始图像，不代表能记住探索中新造的区域。
5. **长时探索弱化控制与一致性的关系**：world models 中控制与输入保留的相关从短视频 0.63 降至长视频 0.32；与生成持久性的相关从 0.53 降至 0.22。
6. **局部一致不等于全局可重建**：跨视图深度或特征看似稳定的视频，held-out Gaussian 渲染仍可能失败。

![](assets/Orbia%20-%20视频世界模型三维一致性评测/figures/05-long-horizon.png)

> **图 4（长时稳定性）**：随着时间窗口推进，许多模型的质量变化不大，但控制和几何持续下降；视觉稳定会掩盖遗忘与几何退化。来源：<https://arxiv.org/abs/2610.20693>。

### 6.4 数据源和轨迹差异

![](assets/Orbia%20-%20视频世界模型三维一致性评测/figures/04-source-scores.png)

> **图 5（Figure 9）**：公开真实视频通常得分更高，可能因为其分布更接近视频生成器训练数据；T1 forward-back 对 TI2V 尤其困难，T4 在长视频中明显下降。来源：<https://arxiv.org/abs/2610.20693>。

TI2V 在全部六种模板上控制都弱于 world models。短 forward-back 中，TI2V 控制均值为 44.05，world models 为 62.41。

### 6.5 人类评测一致性

作者从 90 个 case、6 个模型中组织 20 名评测者，分别按六个 tier 排名。自动评分与人工模型级分数的 Spearman 相关在 **0.859–0.921**，其中 temporal stability 最高。这说明自动指标至少能较可靠地复现模型排序。

---

## 7. 与相关工作的关系

Orbia 不是新世界模型，而是对 [[WorldCrafter - 隐式三维记忆视频世界模型]]、Lyra 2.0、EVOKE、HY-WorldPlay、DecMem 等记忆机制的统一检验框架。它把这些模型的口号“长期一致”“可重访”“空间记忆”拆成可测量对象。

与 WorldLens 等世界模型 benchmark 相比，Orbia 更聚焦 3D 一致性；与只用 VLM 判断记忆的协议相比，它引入深度、锚点、点云和 held-out reconstruction。与 GST-Bench 等空间理解评测不同，Orbia 评估生成内容本身，而不是 VLM 对视频的理解。

---

## 8. 局限与批判性评价

### 8.1 测量依赖外部估计器

深度、相机和重建依赖 ViPE、DA3、SAM3、DINO 等模型。若某种视频风格更容易被这些估计器处理，分数可能存在 evaluator bias。尤其是生成视频没有真实相机和深度时，几何指标本质上衡量“外部估计器是否能把视频解释成稳定 3D”。

### 8.2 综合权重偏向控制与画质

控制和画质共占 50%，三项一致性合计仅 30%。如果研究重点是空间记忆，应优先看 T3/T4/T5 分项，而不是只看总分。

### 8.3 主要是静态场景

锚点要求静态地标，动态主体和非刚性变化的持久性没有被充分评估。人物离开再出现、物体被交互改变后的状态记忆仍是空白。

### 8.4 接口不公平问题

TI2V 必须通过文本表达轨迹，天然难以控制精确相机；action 模型也存在连续轨迹离散化误差。因此跨接口比较能反映“当前可用系统”的能力，但不能完全归因于生成 backbone 本身。

### 8.5 个人判断

Orbia 的最大价值是把“世界模型一致性”从感觉描述变成多维诊断。尤其是将**输入保留**与**生成内容持久性**分开，非常重要：前者可以靠 I2V 条件复制，后者才真正考验模型是否形成了内部世界状态。held-out Gaussian reconstruction 则补上了局部帧对齐无法发现的全局冲突。

---

## 9. 复现与使用建议

1. 研究空间记忆时重点报告 T4 generated persistence 和 T5 self-consistency；
2. 至少同时运行 T1、T3、T6：分别检查尺度、视野外记忆和遮挡新内容；
3. 分开报告 short 与 minute-long，不要只给加权总分；
4. 对 TI2V 与 camera-conditioned 模型分组比较；
5. 同时检查 held-out reconstruction，避免被相邻帧特征相似欺骗；
6. 若模型生成动态场景，应额外设计动态主体回访指标。

官方已开放项目页、代码和 Hugging Face 数据集，适合作为世界模型实验的标准外部评测。

---

## 10. 最终评价

**评级：★★★★★**

Orbia 是视频世界模型领域非常关键的评测基础设施。它没有直接提出更强生成模型，却清楚揭示了当前模型的真实短板：画面越来越好，并不意味着世界在三维中保持一致；模型能复原输入，也不意味着能记住自己生成的新区域；分钟级探索中，控制、内容和几何会同时退化。

它的局限主要在评估器依赖、静态场景假设与总分权重。但作为第一套系统区分“输入保留、生成记忆、三维自洽”的 benchmark，Orbia 很可能成为后续 world-model memory 论文必须报告的标准之一。

---

## 11. 材料来源

- [arXiv](https://arxiv.org/abs/2610.20693)
- [项目页](https://orbia-bench.github.io/)
- [代码](https://github.com/worldbench/Orbia)
- [数据集](https://huggingface.co/datasets/Orbia/Orbia)
- [[assets/Orbia - 视频世界模型三维一致性评测/sources|本地来源记录]]
