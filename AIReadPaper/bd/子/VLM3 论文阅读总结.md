## 1. 背景

这篇小红书笔记的正文因站点风控未能完整提取，我根据标题线索 **“VLM 是天生的 3D 学习者”** 追溯到了对应的官方论文 **[VLM³: Vision Language Models Are Native 3D Learners](https://arxiv.org/abs/2605.30561)** 与官方代码仓库 **[facebookresearch/VLM3](https://github.com/facebookresearch/VLM3)**，并以官方论文 PDF 与 arXiv source 为主完成本次沉淀。

- **归档日期：** 2026-06-05
- **主领域：** VLM
- **重要性评级：** ★★★★☆
- **原始线索：** [小红书笔记标题线索](http://xhslink.com/o/2GFNszlIdR2)
- **官方论文：** [arXiv 2605.30561](https://arxiv.org/abs/2605.30561)
- **官方代码：** [GitHub - facebookresearch/VLM3](https://github.com/facebookresearch/VLM3)

**一句话判断：**

这篇工作最重要的价值，不是又做了一个 3D 专用头，而是反过来证明：**标准 VLM 在输入预处理、文本引用方式和数据配比合适时，本身就可以学出强 3D 能力**。

这篇内容围绕 VLM³ 展开，核心观点是：很多 3D 任务里被默认“必须”的复杂设计——额外编码器、任务专用 head、多损失回归、重数据增强——未必是必要条件。作者试图证明，只要解决 **相机歧义**、**像素/区域引用方式** 和 **数据混合与扩展** 这三个关键点，标准视觉语言模型就能覆盖深度估计、像素对应、相机位姿估计、物体级 3D 理解等多类任务。

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=Nzk2Mjk0M2YxNjY3ZDg0YzQzYTM5MzU1ODJlZmUzYzFfbGZvNWV5NGVoeHlpMnZuMndrdkFtWHYyUjFWQXhWb2dfVG9rZW46SVZRcGJXRzY4b2laRzV4OWxKYmNiVWNibkRjXzE3ODI5ODAxMjE6MTc4Mjk4MzcyMV9WNA&add_watermark=true&scene_type=CCM)

---

## 2. 文章主线 / 论文线索

|项目|名称|要点|
|---|---|---|
|外部线索|VLM3：VLM是天生的3D学习者|小红书笔记标题直接对应论文核心论断，即 “VLMs are native 3D learners”。|
|核心论文|VLM³: Vision Language Models Are Native 3D Learners|Meta 与 Princeton 的工作，主张标准 VLM 经过最小改造即可学习多种细粒度 3D 任务。|
|核心前作|DepthLM|作者前作，证明标准 VLM 可以做像素级 metric depth；VLM³ 在此基础上进一步扩展到更广泛的 3D 任务，并把 marker-based pixel reference 改成了纯文本 reference。|
|官方实现|facebookresearch/VLM3|README 给出了方法总览、主要结果、推理用法与模型入口，验证论文并非纯概念性描述。|

**论文主线可以概括为三句话：**

1. **标准 VLM 并不天然缺 3D 能力，缺的是合适的表述方式和训练组织。**
2. **把 3D 任务的输入输出统一成文本空间之后，很多任务专用结构都可以去掉。**
3. **当相机归一化与文本引用问题解决后，真正决定上限的是数据混合和训练规模，而不是更复杂的结构。**

---

## 3. Pipeline / Architecture + I/O 数据流

**核心方法论：** VLM³ 基本保持标准 VLM 架构不变，以 Qwen3-VL-4B 为基础模型，重点改的是 **输入对齐方式**、**坐标表达方式** 与 **训练样本组织方式**，而不是网络结构本身。

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ODE0MmE0MzhkMDhiOGRkM2Q4YjYwMzU5NmY3MmRhYmVfWDVxbkxvdlJqMDJSVEcyWmtlcng5WFFzdHhqV3NVZ1RfVG9rZW46T3JDamJ3cWxub3JsWGd4ZGpJU2NSM3pqbldoXzE3ODI5ODAxMjE6MTc4Mjk4MzcyMV9WNA&add_watermark=true&scene_type=CCM)

### 3.1 核心 Pipeline

|阶段|输入 / 输出|说明|
|---|---|---|
|输入组织|输入：单张或多张图像 + 文本问题|不同任务可接单视图或双视图输入。深度估计、物体级 3D 理解偏单图；像素对应与相机位姿偏双图。|
|相机归一化|输入：原始图像、相机内参或估计内参；输出：统一焦距后的图像|把图像 resize 到等效焦距 1000 像素，解决 camera ambiguity，使多数据集混训成为可能。若无内参，则先用单图 calibration 模型估计。|
|文本化引用|输入：像素 / 区域位置；输出：文本坐标描述|把横纵坐标统一归一到 [0, 2000) 区间，在文本里直接引用像素点或 bbox，不再需要在图片上渲染 marker，也不需要额外 object encoder。|
|统一训练|输入：图像 + 文本问题；输出：文本答案|使用标准 text-based SFT。作者强调很多任务连回归 formulation 都不是必要条件，直接 next-token 预测文本答案即可。|
|任务输出|输出：文本形式的深度、坐标、方向、姿态参数等|例如深度输出米数、像素对应输出目标坐标、位姿输出平移距离 / 单位方向向量 / yaw-pitch-roll。|

### 3.2 三个关键 ingredient

#### A. Focal Length Unification

- **要解决的问题：** 不同数据源相机参数不同，直接混训会让 VLM 面对同样的视觉几何却映射到不同尺度。
- **做法：** 统一到等效焦距 1000 pixels。
- **意义：** 这是把不同数据集拉到同一个几何尺度坐标系的关键预处理，也是后续大规模混训的基础。

#### B. Text-based Pixel / Region Reference

- **旧方案：** DepthLM 依赖 visual prompting，在图像上打 marker 来指定像素。
- **新方案：** 直接在文本中用归一化坐标表示像素或区域。
- **关键洞察：** 不是“VLM 完全不懂文本坐标”，而是 **原来的 prompt 设计不对**。当像素空间被规范化后，文本坐标可以工作得很好。
- **直接收益：**
    - 同一张图可打包多个 QA，减少重复图像计算；
    - 可以天然扩展到输出也需要坐标的任务，如 pixel correspondence；
    - 同时兼容 object-level 与 pixel-level 两类问题。

#### C. Data Mixture and Scaling

- **作者结论：** 解决相机歧义和文本引用后，真正重要的是数据混合权重与规模扩展。
- **关键现象：** 简单地把更多数据堆进去，如果权重不对，性能甚至会比小规模训练更差。
- **解释：** 小数据集或简单数据集更容易被大模型过拟合，因此需要按数据规模和难度重新配比。

### 3.3 四类任务的 I/O 设计

|任务|输入|输出|
|---|---|---|
|Metric Depth|单张图像 + 多个归一化像素坐标问题|每个 query pixel 到相机的距离，单位为米|
|Object-level 3D|单张图像 + 文本中的 bbox 坐标 / 原始问题模板|上下左右前后关系、尺寸、高度、距离、方向等文本答案|
|Pixel Correspondence|两张图 + 第一张图中的查询像素|第二张图上的对应像素坐标 (x2, y2)|
|Camera Pose|两张图 + 位姿问题模板|平移距离、平移单位向量、旋转 yaw / pitch / roll 文本结果|

### 3.4 代表性 Prompt 设计

```Plain
Depth: How far is the pixel at (x, y) from the camera? Both x and y are normalized to between [0, 2000).
```

Pixel correspondence: Given these two images, what pixel in the second image corresponds to pixel (x1, y1) in the first image? Report the answer as (x2, y2).

Pose / translation distance: Estimate the magnitude of the camera translation between the two viewpoints.

Pose / translation direction: Using the first camera as the reference frame, describe the displacement of the camera between the two views, and give the unit direction vector (x, y, z).

Pose / rotation: Describe the camera reorientation from the first viewpoint to the second as yaw, pitch and roll.

### 3.5 我对 I/O 逻辑的理解

**这篇论文最值得注意的地方，是它把很多典型 3D 任务都重新表述成了 “图像 + 文本 query → 文本 answer” 的统一接口。** 这意味着：

- 上游输入统一成视觉 token + 文本 token；
- 下游输出统一成可解析的数值文本；
- 中间不再强依赖任务专用 decoder 或回归 head；
- 训练目标统一成 next-token prediction。

---

## 4. 实验与关键信息

### 4.1 训练设置与超参数

|任务|学习率|Batch Size|训练样本数|训练资源|
|---|---|---|---|---|
|Depth Estimation|5.5e-5|1344|32M samples（每样本 10 pixels）|32 GPUs × 3 天|
|Object-level 3D|3.5e-4|640|1M images|32 GPUs × 3 小时|
|Pixel Correspondence|2e-5|2816|80M samples（每样本 10 QA）|64 GPUs × 7 天|
|Camera Pose|5e-5|448|10M samples|32 GPUs × 4 天|

**统一训练细节：** cosine learning rate schedule、warmup ratio 0.1、AdamW、FSDP hybrid shard、gradient clipping 0.02、gradient checkpointing、bfloat16、Flash Attention 2。

### 4.2 数据组成

#### Depth 训练数据（26M images，其中论文给出了主要配比）

|数据集|图像数量|mixture weight|
|---|---|---|
|Argoverse2|1M|0.21|
|Waymo|700K|0.04|
|NuScenes|200K|0.01|
|ScanNet++|1M|0.01|
|Taskonomy|4M|0.51|
|HM3D|9M|0.78|
|Matterport3D|190K|0.006|
|Internal street-view data|10M|1.0|

#### Pixel Correspondence / Camera Pose 训练数据（约 9.9M image pairs）

|数据集|图像对数量|
|---|---|
|BlendedMVS|450K|
|dynamicreplica|1M|
|sailvos3d|350K|
|ScanNet++|1M|
|MPSD|13K|
|RealEstate-10K|880K|
|DL3dv-10k|2.6M|
|MegaDepth|190K|
|Aria Synthetic Environment|2M|
|GTA-SFM|90K|
|Tartanairv2|850K|
|UnrealStereo4K|270K|
|MVS Synth|190K|
|Spring|20K|
|总计|9.9M|

**补充实现细节：** 多视图数据构造时，作者随机采样 covisibility > 25% 的图像对；在 ScanNet++ 中 hold out 30 个 scene 以确保评测场景未见。

### 4.3 主结果：对 VLM 基线的比较

#### A. Metric Depth（δ1，越高越好）

|模型|Argoverse2|DDAD|NuScenes|ETH3D|ScanNet++|SUNRGBD|iBims1|NYUv2|平均|
|---|---|---|---|---|---|---|---|---|---|
|Qwen2.5-VL-72B|0.119|0.140|0.186|0.220|0.272|0.276|0.212|0.324|0.219|
|Qwen3-VL-4B|0.004|0.073|0.070|0.071|0.147|0.176|0.080|0.158|0.101|
|Qwen3-VL-32B|0.017|0.099|0.029|0.167|0.373|0.463|0.122|0.393|0.208|
|Gemini-2.5-Pro|0.280|0.252|0.365|0.328|0.380|0.270|0.466|0.394|0.342|
|GPT-5|0.218|0.302|0.382|0.313|0.428|0.471|0.307|0.540|0.370|
|SpaceLLaVA-13B|0.100|0.067|0.083|0.090|0.269|0.233|0.208|0.178|0.154|
|SpatialRGPT-8B|0.055|0.046|0.100|0.220|0.346|0.369|0.240|0.265|0.205|
|Seed1.5-VL|0.040|0.074|0.028|0.309|0.593|0.689|0.627|0.841|0.400|
|DepthLM-3B|0.808|0.724|0.870|0.745|0.838|0.850|0.890|0.868|0.824|
|DepthLM-7B|0.833|0.747|0.865|0.718|0.850|0.859|0.920|0.915|0.838|
|**VLM³-4B**|**0.896**|**0.818**|**0.970**|**0.810**|**0.976**|**0.867**|**0.960**|**0.935**|**0.904**|

#### B. Object-level 3D Understanding

**Qualitative（Acc，越高越好）**

|模型|Below/Above|Left/Right|Big/Small|Tall/Short|Wide/Thin|Behind/Front|Overall|
|---|---|---|---|---|---|---|---|
|Qwen3-VL-4B|63.33|85.71|85.85|70.54|83.65|60.91|75.00|
|Qwen3-VL-32B|72.50|83.81|82.08|68.75|87.50|67.27|76.98|
|SpatialRGPT-8B|99.17|99.04|79.24|89.28|83.65|87.27|89.80|
|**VLM³-4B**|97.84|**99.37**|**90.78**|**90.82**|**92.86**|75.54|**91.35**|

**Quantitative（Acc / AbsRel）**

|模型|Direct Distance|Horizontal Distance|Vertical Distance|Width|Height|Direction|Overall|
|---|---|---|---|---|---|---|---|
|Qwen3-VL-4B|18.24 / 51.94|18.03 / 80.64|14.15 / 160.62|3.76 / 552.54|12.78 / 874.07|0.0 / 180.0°|11.16 / 343.96|
|Qwen3-VL-32B|22.30 / 97.14|17.21 / 67.60|22.64 / 44.93|0.00 / 582.34|2.26 / 647.74|0.0 / 180.0°|10.73 / 287.95|
|SpatialRGPT-8B|35.1 / 0.35|59.0 / 0.27|53.8 / 0.27|51.9 / 0.31|54.9 / 0.63|95.3 / 17.1°|58.33 / 0.37|
|**VLM³-4B**|34.09 / 0.37|53.38 / 0.29|**58.41** / 0.27|44.11 / 0.39|**65.64** / 0.41|**95.42** / **10.5°**|**58.51 / 0.35**|

#### C. Pixel Correspondence（EPE，越低越好）

|模型|ETH3D|DTU|TA-WB|Average|
|---|---|---|---|---|
|Qwen3-VL-4B|179.80|89.64|190.40|153.28|
|Qwen3-VL-32B|196.53|88.76|195.53|160.27|
|**VLM³-4B**|**15.18**|**10.71**|**20.21**|**15.37**|

#### D. Camera Pose（AUC@30°，越高越好）

|模型|ETH3D|ScanNet++|Average|
|---|---|---|---|
|Qwen3-VL-4B|10.0|0.7|5.4|
|Qwen3-VL-32B|11.7|3.9|7.8|
|**VLM³-4B**|**93.3**|**94.7**|**94.0**|

### 4.4 与专家视觉模型的比较

|任务|对比方法|关键数字|VLM³|结论|备注|解读|
|---|---|---|---|---|---|---|
|Depth|UnidepthV2 / MoGe-2|UnidepthV2: DDAD 0.882, NuScenes 0.870, ETH3D 0.852, SUNRGBD 0.964, iBims1 0.945；MoGe-2: DDAD 0.856, ETH3D 0.908, iBims1 0.924|0.818 / 0.970 / 0.810 / 0.867 / 0.960|总体接近，部分数据集更优|指标是 δ1|VLM³ 在 NuScenes、iBims1 上很强，但在 DDAD、ETH3D、SUNRGBD 仍非全面碾压。|
|Pixel Correspondence|DKM / RoMa / UFM|DKM 41.30，RoMa 21.88，UFM 7.89|15.37|优于 DKM、RoMa，落后 UFM|指标是平均 EPE|说明文本化统一范式已经能追平一部分专家模型，但在 dense matching 上仍未到 SOTA 极限。|
|Camera Pose|DUSt3R / MapAnything / VGGT / DA3-Giant|30.6 / 80.8 / 88.0 / 94.7|94.0|超过 VGGT，接近 DA3-Giant|指标是 AUC@30°|这是最“反直觉”的结果：不用专门回归头，仅靠文本输出就做到接近 SOTA。|

### 4.5 分析实验

|分析维度|设置|结果|
|---|---|---|
|像素引用方式|Visual Prompting，8M samples + 1QA|δ1 = 0.849|
|像素引用方式|Text-based，8M samples + 1QA|δ1 = 0.853|
|数据混合|Uniform Weight，32M samples + 10QA|δ1 = 0.842|
|数据混合|Dataset-size Weight，32M samples + 10QA|δ1 = 0.884|
|数据混合|VLM³ Weight，32M samples + 10QA|δ1 = 0.904|
|模型 / 数据规模|32B，32M samples + 10QA|δ1 = 0.873|
|模型 / 数据规模|8B，32M samples + 10QA|δ1 = 0.880|
|模型 / 数据规模|4B，64M samples + 10QA|δ1 = 0.880|
|模型 / 数据规模|4B，32M samples + 10QA|**δ1 = 0.904**|

**作者给出的核心分析结论：**

- 文本像素引用并没有比 visual prompting 差，反而更简洁、更高效、更易扩展；
- 数据混合权重比“单纯多训一些样本”更重要；
- 在当前数据量级下，小模型并非弱点，反而更不容易过拟合；
- 扩大模型到 8B / 32B 或扩大到 64M samples，并没有继续提升深度结果。

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=N2M1MzUwYWM1NjYxZWFiZjQ3MTEzYTI0YWU1Yjk0OTNfaFNvM2NKRzZHbXNlekVvbU9RcHJNMUQ0RndjTndYRXdfVG9rZW46QkZycGJsdTJ1b2dscjZ4Z3E2dmNmaDhobnhoXzE3ODI5ODAxMjE6MTc4Mjk4MzcyMV9WNA&add_watermark=true&scene_type=CCM)

### 4.6 值得警惕的限制 / 假设

- **依赖数据规模与数据配比。** 论文的核心收益很大程度来自混合数据与 re-weighting，不只是 prompt 技巧。
- **含有内部 street-view 数据。** Depth 结果使用了额外 10M internal outdoor street-view images，这会抬高复现门槛。
- **并非所有任务都绝对 SOTA。** 在 pixel correspondence 上仍落后 UFM。
- **对象级任务借助了单图相机标定。** 对无内参互联网图片，需要额外 calibration 模型估计内参。
- **更大模型并未更强。** 至少在这套数据规模下，32B/8B 反而过拟合或训练性价比更差。

---

## 5. 个人评注 / 下一步

### 5.1 这条内容为什么值得进“技术视野”

**我认为这篇工作最值得记住的不是一个 benchmark number，而是一种建模态度：**

对很多 3D 问题，作者在反证“复杂任务专用设计是否真的必要”。这和当前多模态大模型不断追求统一接口、统一输出域、统一训练目标的趋势高度一致。

它对“旭涛的技术视野”有三层价值：

1. **VLM 范式价值：** 证明 VLM 可以从语义理解继续外延到几何理解，而不一定要分叉出一套专用 3D 模型族。
2. **I/O 设计价值：** 对算法策略很有启发——很多任务难点不一定在 backbone，而在输入输出空间如何统一表达。
3. **训练策略价值：** 这篇论文再次强调，数据 mixture 与 sample packing 的设计，可能比模型结构微创新更影响最终能力上限。

### 5.2 与既有关注方向的关系

- 与 **VLM / MLLM 能力边界扩展** 强相关；
- 与 **3D / 空间智能是否能被统一多模态模型吸收** 强相关；
- 与 **具身前置能力** 也有潜在线索，因为像素对应、位姿估计、尺度深度本身就是很多下游空间决策模块的前置能力。

### 5.3 建议后续跟进

- **优先补读前作 DepthLM**：对比它为何从 visual prompting 过渡到 text-based pixel reference。
- **追踪复现门槛**：尤其是 internal 10M street-view data 是否是 depth 结果跃升的关键因子。
- **关注统一空间表述接口**：可继续对照 SpatialRGPT、VGGT、DA3、UFM，看哪些任务真正还能继续被“文本统一输出”吞并。
- **观察是否能迁移到 VLA / 具身任务**：如果几何理解可统一为文本 token 预测，那么动作规划里的空间状态表征也许能复用类似接口。

### 5.4 我的结论

如果只看一句话，我会把这篇工作定义为：

**“它不是在证明 VLM 也能做一点 3D，而是在证明：3D 任务中大量被默认必要的专用建模套路，可能在 foundation model 时代并不是必要条件。”**
