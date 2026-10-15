---
type: paper-note
status: complete
domain: World-Models
paper: "Video World Models with Long-term Spatial Memory"
year: 2025
arxiv: 2506.05284
version: v1
source: https://arxiv.org/abs/2506.05284
project: https://spmem.github.io/
code: null
tags: [World-Model, Spatial-Memory, Point-Cloud, TSDF, Episodic-Memory, CogVideoX, Camera-Control]
created: 2026-10-15
updated: 2026-10-15
---

# SPMem：长期空间记忆视频世界模型

## 基本信息

- **作者**：Tong Wu、Shuai Yang、Ryan Po、Yinghao Xu、Ziwei Liu、Dahua Lin、Gordon Wetzstein
- **机构**：Stanford、上海交通大学、上海人工智能实验室、南洋理工、香港中文大学
- **版本**：arXiv v1，2025-06-05
- **基础模型**：CogVideoX-5B-I2V，基于 Diffusion as Shader（DaS）初始化
- **任务**：在文本/动作和目标相机轨迹控制下，自回归生成动态视频，同时长期保持静态世界结构与历史实体细节
- **资源**：[[assets/SPMem - 长期空间记忆视频世界模型/sources/sources|来源记录]]；[官方 PDF](assets/SPMem%20-%20%E9%95%BF%E6%9C%9F%E7%A9%BA%E9%97%B4%E8%AE%B0%E5%BF%86%E8%A7%86%E9%A2%91%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/sources/paper.pdf)

## 一句话结论

**SPMem 把长时世界记忆拆成“TSDF 融合后的静态 3D 点云、最近 5 帧 working memory、稀疏历史关键帧 episodic memory”三条通道：点云渲染通过 ControlNet 式 Condition DiT约束结构，近帧维持运动，关键帧通过 cross-attention 补回身份与纹理；它显著提高重访 PSNR/SSIM/LPIPS，但重建—融合—再条件生成形成一条脆弱的几何误差链，快速大幅相机运动会直接让空间记忆坍缩。**

---

## 1. 背景、动机与核心贡献

短上下文视频世界模型只“记得最近”，返回旧区域时会改写场景。单纯扩大视频上下文又受到 DiT 二次注意力成本限制；按时间压缩旧帧虽能延长上下文，却没有持久的三维空间结构。

SPMem 借用认知记忆的分工：

1. **spatial memory**：保存长期稳定的静态空间；
2. **working memory**：保存最近动态状态；
3. **episodic memory**：稀疏保存过去的高细节事件/视图。

论文贡献是把三者做成可在线写入、可联合读取的生成系统，并建立 90K geometry-grounded 训练对与重访评测。

![](assets/SPMem%20-%20%E9%95%BF%E6%9C%9F%E7%A9%BA%E9%97%B4%E8%AE%B0%E5%BF%86%E8%A7%86%E9%A2%91%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/figures/01-teaser.png)

> **图 1（Figure 1，PDF p.1）**：左侧全局静态点云提供可持续空间骨架；右侧 episodic/working/spatial 三类记忆共同生成带动作的动态视频。关键思想是“静态结构长期化、动态状态短期化、视觉细节关键帧化”。来源：<https://arxiv.org/abs/2506.05284>，官方源码 `figures/teaser.pdf`。

---

## 2. 任务定义、输入与输出

### 2.1 输入

每次自回归迭代接收：

- 目标相机轨迹；
- 动作文本/未来动作描述；
- 由全局静态点云沿目标轨迹渲染的 condition video，未知区为黑色；
- 最近 5 帧 RGB/video latent；
- 稀疏历史参考帧；
- 当前目标帧噪声 latent。

### 2.2 输出

模型一次生成 49 帧、$480\times720$ 视频。推理后对新帧在线预测 point map、与全局坐标对齐、TSDF 过滤动态区域并融合进静态点云，然后进入下一轮。输出不只是视频，也隐含更新后的全局空间记忆与关键帧集合。

### 2.3 任务难点

- 点云只能约束已观察静态区域，未知区仍要生成；
- 人物/动物等动态对象不应被“焊死”到静态地图；
- 只靠稀疏几何难保留纹理与身份；
- 只靠最近帧又会遗忘远期视图；
- 生成结果反过来用于重建，错误可能闭环放大。

---

## 3. 三类记忆表示与检索/写入依据

### 3.1 长期空间记忆：静态点云

新视频由 CUT3R 在线预测公制 point map 和相机参数；TSDF fusion 融合多帧 RGB-D，只保留跨帧一致的静态表面。体素 $v$ 的更新为：

$$
D'(v)=\frac{W(v)D(v)+w_i d_i(v)}{W(v)+w_i},\qquad W'(v)=W(v)+w_i.
$$

$D(v)$ 是 TSDF，$W(v)$ 是累积置信权，$d_i(v)$ 是第 $i$ 帧对体素的截断符号距离。动态物体在不同帧深度不一致，理论上形成低置信噪声并被抑制。

**表示**是融合后的静态 3D point cloud；**检索**不是选若干旧帧，而是沿目标相机轨迹直接渲染当前可见点云，得到与目标帧一一对齐的结构条件。

### 3.2 Working memory：最近 5 帧

最近 5 帧 token 与目标 token 沿时间维拼接，提供动作速度、姿态与局部时间连续性。这条通道同时包含静态和动态内容，但生命周期很短。

### 3.3 Episodic memory：稀疏历史关键帧

点云太稀疏，无法保存脸、服装、小物体与纹理。系统监控新揭示未知区域的 mask 面积；超过阈值时把对应帧加入历史集合。生成时将关键帧编码为 reference tokens，以 historical cross-attention 注入：目标 video tokens 作 query，历史 tokens 作 key/value。

论文没有充分说明推理时如何从不断增长的 episodic set 中选择最终参考帧、阈值和容量上限；这是实现信息缺口。

### 3.4 几何角色

三篇记忆世界模型中，SPMem 对几何依赖最强：几何既是长期记忆本体，又被渲染后直接条件生成。它比 [[Context as Memory - 历史上下文驱动的场景记忆]] 的 FOV 规则和 [[VMem - Surfel索引视图记忆]] 的几何索引更“硬”，因而结构控制更直接，失败也更灾难性。

---

## 4. 网络机制与公式

### 4.1 扩散目标

标准前向加噪：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
$$

训练目标：

$$
\mathcal L(\theta)=\mathbb E_{t,x_0,\epsilon}\|\epsilon_\theta(x_t,t)-\epsilon\|_2^2.
$$

### 4.2 静态点云条件分支

点云渲染序列先通过 CogVideoX 3D-VAE 编码。作者复制预训练 CogVideoX 前 18 个 DiT blocks 构成 Condition DiT；每层输出经 zero-initialized linear 投影后加到主 Denoising DiT 对应特征，结构近似 ControlNet。零初始化保证开始训练时不破坏主干。

### 4.3 三通道融合

- 点云渲染 condition：结构与相机约束；
- recent context：与目标 token 拼接，保证帧级连续；
- historical reference：cross-attention，恢复远期细节；
- action text：T5 编码后条件扩散。

![](assets/SPMem%20-%20%E9%95%BF%E6%9C%9F%E7%A9%BA%E9%97%B4%E8%AE%B0%E5%BF%86%E8%A7%86%E9%A2%91%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/figures/02-method.png)

> **图 2（Figure 2，PDF p.4）**：上方点云渲染进入 18-block Condition DiT；最近帧与目标噪声沿时间拼接；历史关键帧跨层 cross-attention；生成结果再回写点云。它清楚展示了 SPMem 的闭环，也暴露了错误可从视频传到重建再传回生成。来源：<https://arxiv.org/abs/2506.05284>，官方源码 `figures/pipeline.pdf`。

---

## 5. 训练、推理与数据构建

### 5.1 训练数据

原始视频来自 MiraData。每条切成 97 帧：前 49 帧为 source，后 48 帧为 target，并共享过渡帧。离线流程：

1. Mega-SAM 估计内参、外参、逐帧深度/动态 point maps；
2. source 通过 TSDF fusion 得到静态体；
3. 用 target 相机从静态体渲染 visibility mask 与静态 reconstruction；
4. target 原始 RGB 作为未来监督，保留动态对象；
5. Qwen 生成未来动作文本。

最终 90K 结构化样本。Mega-SAM 输入缩放到 $384\times672$，并用光流与 covariance-based decomposition 优化相机运动；TSDF 网格最大维限制 1200，超出则增大 voxel size 防止内存溢出。

### 5.2 训练配置

- CogVideoX-5B-I2V / DaS 初始化；
- 49 帧，$480\times720$；
- 6,000 iterations；
- learning rate $2\times10^{-5}$；
- batch size 8；
- 8×A100。

论文未给采样步数、优化器、训练时长、关键帧阈值、TSDF 截断距离和 voxel size，复现链仍不完整。

### 5.3 在线推理

离线数据构建用 Mega-SAM 求精度；在线生成因全局对齐与显存限制改用 CUT3R。系统保存 CUT3R recurrent state 与 pose retriever 参数，下一段继续初始化，使各段 point map 尽量处于同一坐标系。新静态点经 TSDF 后直接与全局记忆合并。

### 5.4 错误传播路径

$$
\text{视频生成误差}\rightarrow\text{CUT3R 深度/位姿误差}\rightarrow\text{TSDF 错删或鬼影}\rightarrow\text{点云渲染错误}\rightarrow\text{Condition DiT 强约束下一段}\rightarrow\text{新生成误差}.
$$

另一路是 episodic 关键帧：错误人物/物体一旦被选为参考，会通过 cross-attention 长期维持。论文明确说 drift（画质随自回归累积下降）不在解决范围内。

---

## 6. 评测设置与基线

- 测试集：MiraData 未见的 500 条视频；
- 基线：TrajectoryCrafter、DaS、Wan2.1-Inpainting；
- 重访协议：相机沿轨迹前进再反向返回，同一位姿两次观察；
- 静态一致性：PSNR、SSIM、LPIPS；
- 一般质量：VBench 的 aesthetic、imaging、temporal flickering、motion smoothness、subject/background consistency；
- 用户研究：14 个代表案例，20 名有 3D/4D 经验参与者，1–4 分 Average Human Ranking，评 camera accuracy、static consistency、dynamic plausibility。

定量评测把动态内容与静态结构混在整帧指标中，且反向轨迹较规则；因此 PSNR 提升支持重访复现，却不足以证明任意路径下有一致的 3D 世界。

---

## 7. 主要结果

### 7.1 重访一致性与用户研究

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Cam-Acc↑ | Stat-Cons↑ | Dyn-Plaus↑ |
|---|---:|---:|---:|---:|---:|---:|
| TrajectoryCrafter | 11.71 | 0.4380 | 0.5996 | 1.6320 | 1.7802 | 1.6255 |
| DaS | 12.01 | 0.4512 | 0.5874 | 2.5660 | 2.4396 | 2.7033 |
| Wan2.1-Inpainting | 12.16 | 0.4506 | 0.5875 | 2.1760 | 2.3956 | 2.2701 |
| **SPMem** | **19.10** | **0.6471** | **0.3069** | **3.6260** | **3.3846** | **3.4011** |

相对最强 PSNR 基线提升约 6.94 dB，LPIPS 近乎减半，是明确的大幅增益。但作者也承认 19.10 dB 仍离像素级复现很远。

### 7.2 VBench

SPMem 在 aesthetic 0.5835、temporal 0.7580、motion 0.9886、subject 0.9359 上最好；Wan2.1 的 imaging 0.6788 和 background 0.9513 略高。论文解释 Wan2.1 常生成较静态画面，背景指标高不等于遵循几何。

![](assets/SPMem%20-%20%E9%95%BF%E6%9C%9F%E7%A9%BA%E9%97%B4%E8%AE%B0%E5%BF%86%E8%A7%86%E9%A2%91%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/figures/03-results.png)

> **图 3（Figure 4，PDF p.7）**：三组分别测试大相机旋转、同位姿重访、文本动作。SPMem 不仅复现雕像/建筑，也能让角色按动作继续演化；基线常静止、变形或丢失人物。来源：<https://arxiv.org/abs/2506.05284>，官方源码 `figures/qualitative.pdf`。

---

## 8. 消融与失败案例

### 8.1 三类记忆消融

| 配置 | Aesthetic | Imaging | Temporal | Motion | Subject | Background |
|---|---:|---:|---:|---:|---:|---:|
| 无 episodic memory | 0.5603 | 0.6485 | 0.7260 | 0.9870 | 0.9326 | 0.9489 |
| 无 working memory | 0.5551 | 0.6384 | 0.6740 | 0.9862 | 0.9331 | 0.9453 |
| **完整模型** | **0.5835** | **0.6701** | **0.7580** | **0.9886** | **0.9359** | **0.9506** |

working memory 对 temporal flickering 贡献更大；episodic memory 对视觉细节与远期实体更关键。遗憾的是表中没有“无 spatial memory”，因此无法量化论文标题所强调的长期空间记忆本体相对两种帧记忆的独立贡献。

![](assets/SPMem%20-%20%E9%95%BF%E6%9C%9F%E7%A9%BA%E9%97%B4%E8%AE%B0%E5%BF%86%E8%A7%86%E9%A2%91%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/figures/04-ablation.png)

> **图 4（Figure 5，PDF p.8）**：去掉 working memory 后人物运动/位置不连贯；去掉 episodic memory 后角色身份与细节偏离；完整模型最接近 GT。点云只提供结构，不能替代两类图像记忆。来源：<https://arxiv.org/abs/2506.05284>，官方源码 `figures/ablation.pdf`。

### 8.2 明确失败案例

![](assets/SPMem%20-%20%E9%95%BF%E6%9C%9F%E7%A9%BA%E9%97%B4%E8%AE%B0%E5%BF%86%E8%A7%86%E9%A2%91%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B/figures/05-failure.png)

> **图 5（Figure 6，PDF p.9）**：蜘蛛侠在楼宇间快速摆荡，连续相机位姿距离过大、角度突变，4D 重建产生鬼影；TSDF 将本应静态的表面当成不一致观测过滤，点云越来越稀，最终相机控制与场景一致性同时失效。来源：<https://arxiv.org/abs/2506.05284>，官方源码 `figures/failurecase.pdf`。

失败图非常重要：SPMem 不是“几何自动纠错”，而是依赖几何质量。TSDF 对动态干扰的抑制与对重建误差的误删是同一机制的两面。

---

## 9. 局限、Future Work 与复现成本

### 9.1 局限

- TSDF fusion 并不能可靠区分动态与重建误差；大视角变化会产生鬼影和过度过滤。
- 长期记忆偏向静态结构；动态实体只由短期/稀疏帧承担，不能保存持续变化的对象状态。
- 没有解决自回归 drift，只解决 forgetting/spatial consistency。
- spatial memory 独立消融缺失；关键帧容量、检索策略和阈值披露不足。
- 90K 数据来自 MiraData 派生，几何由 Mega-SAM 估计，不是真值；训练监督本身可能带系统性偏差。
- 推理速度、点云规模随时长增长、CUT3R 与 TSDF 的在线成本未报告。

### 9.2 Future Work

作者建议与 FramePack 式时间压缩结合，同时改善空间与人物一致性；更根本的方向包括可更新的动态 4D memory、带不确定度的融合、地图冲突检测/回环校正，以及显式针对 drift 的训练。

### 9.3 复现成本

**非常高。** 需要 CogVideoX-5B/DaS、Mega-SAM、CUT3R、TSDF、点云渲染、T5/Qwen 标注和 8×A100。训练本身仅 6k steps，但前处理 90K 视频的 4D 重建与在线闭环工程更重。论文未开源代码/权重，多个关键超参缺失；外部团队更现实的目标是复现模块原型，而非复现完整指标。

---

## 10. 个人判断与四篇路线比较

### 10.1 个人判断

SPMem 的贡献是把“世界应该记住什么”做了清晰分工：静态空间、近期动态、远期细节不能靠一种表示包办。它的主结果也最直观地支持显式几何对重访的价值。但系统把未经真值校验的生成帧重建成强条件点云，闭环脆弱；缺少无 spatial memory 消融更削弱了对核心模块的归因。

### 10.2 记忆、检索、几何、错误传播

| 方法 | 记忆表示 | 检索/读取依据 | 几何 | 主要错误传播 |
|---|---|---|---|---|
| [[Context as Memory - 历史上下文驱动的场景记忆]] | 历史 RGB 帧 | 相机 FOV 共视 | 弱，仅筛选 | 误检/错误旧帧被复制 |
| **SPMem** | 静态点云 + 近帧 + 关键帧 | 目标轨迹点云渲染；未知区触发关键帧 | **强，直接约束生成** | 生成→重建→TSDF→条件生成闭环 |
| [[VMem - Surfel索引视图记忆]] | RGB 视图库 + surfel 索引 | 可见 surfel 的 view-index 投票 | 中，只索引不渲染为最终条件 | 几何错主要导致检索错，破坏较局部 |
| [[FramePack - 固定上下文预算的长视频生成]] | 多尺度历史 latent | 时间/特征重要性排序 | 无显式几何 | 历史误差持续压缩传播，靠端点/离散化截断 |

工程取舍上，SPMem 适合“相机轨迹与静态布局是第一优先级”的世界模型；若几何估计不稳，VMem 只用粗几何做视图索引更安全；若没有相机/几何，FramePack 更通用；若已有精确位姿且追求低改造成本，Context as Memory 更直接。
