---
type: paper-note
status: complete
domain: World-Models
paper: "ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment"
year: 2026
arxiv: 2603.23376
doi: null
source: https://arxiv.org/abs/2603.23376
project: https://github.com/amap-cvlab/ABot-PhysWorld
code: https://github.com/amap-cvlab/ABot-PhysWorld
tags:
  - World-Model
  - Robotic-Manipulation
  - Physics-Alignment
  - Diffusion-DPO
  - Action-to-Video
  - Embodied-AI
created: 2026-10-16
updated: 2026-10-16
---

# ABot-PhysWorld：物理对齐机器人操控世界模型

## 基本信息

- **正式标题**：ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment
- **署名**：AMAP CV Lab；完整贡献者见论文 Contributions
- **来源状态**：官方 GitHub 技术报告已公开，同时存在 arXiv:2603.23376 预印本；截至本笔记写作时未确认正式同行评审录用
- **模型骨干**：Wan2.1-I2V-14B，480×832、81 帧训练设置
- **任务**：根据首帧与文本指令预测机器人操控未来；或根据首帧与结构化动作序列生成动作可控视频
- **核心贡献**：三百万级具身视频整理、VLM 物理偏好构造、LoRA Diffusion-DPO、空间 action map + 并行 context block、EZSbench 零样本评测

## 一句话结论

ABot-PhysWorld 的核心进展是把“物理合理”从泛化的视频似然目标改造成**任务相关检查表上的偏好排序问题**：用两个不同 VLM 分别出题和评分，再以显存可承受的 Diffusion-DPO 提高物理 Domain Score，同时用空间 action map 保留动作控制；论文最强证据是 PAI-Bench 中 DPO 把 Domain Score 从 0.8785 提到 0.9306 且视觉质量几乎不变，但评测仍高度依赖 MLLM-as-Judge、缺少闭环机器人成功率，因而“更像物理正确的视频”尚不能等同于“可替代真实动力学的机器人世界模型”。

---

## 1. 研究问题

### 1.1 通用视频模型为何会在操控中违反物理

通用视频数据可以让模型生成逼真的机械臂和物体，却很少提供接触、摩擦、夹持、碰撞和质量变化的密集监督。标准 SFT 又对所有训练样本一视同仁，无法显式惩罚“画面漂亮但物体穿透”“夹爪未接触却吸起物体”“刚体突然变形”等错误。

论文把问题拆成三层：首先要从多来源机器人数据中筛出视觉—动作同步且运动有效的片段；其次要让优化目标区分物理优劣；最后要把低维动作转换为视频网络容易理解的空间条件，同时避免全量微调破坏预训练动态先验。

### 1.2 模型整体结构

系统先在清洗具身视频上全参数微调 Wan2.1-I2V-14B，得到 text/image-to-video 基础模型；随后为同一条件生成多个候选，用解耦 VLM 判别器构造 winner/loser 偏好对；再用 LoRA Diffusion-DPO 做物理对齐。动作控制则是独立的 A2V 阶段：把末端位姿和夹爪状态绘制为空间 action map，由复制的 context blocks 处理并残差注入冻结主干。

![](assets/ABot-PhysWorld%20-%20物理对齐机器人操控世界模型/figures/02-physics-alignment.png)

> **图 2：SFT 到物理偏好对齐的两阶段流程。** 上半部训练未来视频预测，下半部由 question setter 生成物理检查表、critic 评分多个候选，再把最优/最差视频送入 LoRA DPO。图中关键是“出题”和“评分”解耦，而不是同一 VLM 自评。图片已从 arXiv 源码提取并视觉核验。  
> 来源：论文 Figure 2，§3.2，<https://arxiv.org/abs/2603.23376>

---

## 2. 数据整理：从三百万片段到物理监督

### 2.1 数据来源与过滤

原始池接近 300 万条真实操控视频，来自 AgiBot、RoboCoin、RoboMind、Galaxea 与 Open X-Embodiment。过滤链包括：

1. **视频质量门控**：分辨率、视角、80–500 帧长度，去除移动相机或异常片段；
2. **运动过滤**：2 FPS 灰度帧上的 Farnebäck 光流，剔除近静止与异常振荡；
3. **时间一致性**：八帧 768D CLIP 特征，删除黑屏、跳切和错误拼接；
4. **视觉—动作对齐**：把关节/末端/夹爪动作叠加成 action map，由 Qwen3-VL 检查夹爪轨迹、开合与画面是否一致。

### 2.2 分层重平衡

作者按机器人形态和任务频率分层采样：头部任务只保留 8–15%，中部任务保留 40–50%，长尾任务全部保留，再对大数据集设上限、小数据集设最低配额。它针对的是“大量重复 pick-and-place 淹没稀有操作”的问题。

### 2.3 物理感知 caption

Qwen3-VL-32B 先抽取机器人形态、物体材料/尺寸、布局、接触和状态变化；Qwen3-32B-FP8 再按 Scene Setup、Action Detail、State Transition、Camera Summary 写成四阶段叙述。这里的 caption 不只是语义描述，而是在文本条件中显式暴露“接触后发生了什么”。

![](assets/ABot-PhysWorld%20-%20物理对齐机器人操控世界模型/figures/01-data-pipeline.png)

> **图 1：数据过滤、任务配额与物理 caption。** 左侧展示五个数据集经过质量、运动、时间一致性和动作对齐过滤；中部显示头部降采样与长尾保留；右侧是两阶段 caption。该图支撑“数据系统完整”，但论文未报告过滤后的准确样本数。  
> 来源：论文 Figure 1，§2，<https://arxiv.org/abs/2603.23376>

---

## 3. 物理偏好对齐

### 3.1 为什么使用解耦 VLM 判别器

对一个首帧—指令条件，基础模型生成 $N$ 个候选视频。若同一 VLM 同时生成问题并回答，容易出现自洽偏差。作者将角色拆分：

- **Proposer：Qwen3-VL-32B-Thinking**，根据具体任务生成分层检查表；
- **Scorer：Gemini 3 Pro**，全局扫描视频、标记可疑帧并回溯确认。

Tier 1 是穿透、反重力等致命错误，使用单票否决；Tier 2 检查接触质量、无接触移动、放置关系和微观动态。检查表混合正负问题，减少一律回答“无违规”的迎合偏差。多轮锦标赛先选 winner $y_w$，再从败者组选 loser $y_l$，复杂度为 $O(N)$，避免全排列比较。

### 3.2 Diffusion-DPO 目标

对 winner/loser latent 分别加噪，比较策略模型与 SFT reference 的单步去噪误差。记 $L_\theta$ 为当前策略误差，$L_{\mathrm{ref}}$ 为参考模型误差：

$$\mathcal L_{\mathrm{DPO}}=-\mathbb E\left[\log\sigma\left(-\frac\beta2\left[(L_\theta(z_w)-L_\theta(z_l))-(L_{\mathrm{ref}}(z_w)-L_{\mathrm{ref}}(z_l))\right]\right)\right].$$

目标相对降低物理优选视频 $z_w$ 的预测误差、提高违规视频 $z_l$ 的误差，同时用 reference 差分限制模型偏离 SFT 分布。

### 3.3 显存可行的 LoRA 实现

14B 策略与 reference 若同时保留完整计算图会 OOM。论文冻结主干，在 self-attention 的 q/k/v/o 与 FFN 插入 rank-64、scale-64 LoRA；计算 reference loss 时临时关闭 LoRA，因此无需另一套 14B 参数。DPO 使用 AdamW、学习率 $10^{-6}$、10-step warmup、$\beta=5000$、BF16、gradient checkpointing、每设备 batch 1，并报告每 epoch 500 steps、100 epochs。

### 3.4 方法的风险边界

偏好对本身来自闭源/大模型判断器，并非力/触觉传感器或已知动力学。DPO 学到的是“在该检查协议下更容易被判为物理合理”的视觉分布。解耦能降低同模自评偏差，却不能消除两类 VLM 共享的视觉盲点和提示词偏差。

---

## 4. 动作条件生成

### 4.1 从低维动作到空间 action map

单臂动作为 7D：3D 位置、3D 姿态与夹爪开合；双臂扩展到 14D。系统利用相机内外参把 3D 位置投影为二维中心，把旋转矩阵三个主轴绘制为彩色箭头，箭头长度编码深度；夹爪状态绘制为圆形 mask，透明度编码开合，双臂用红/蓝通道区分。

这一步把抽象控制向量转成与像素空间对齐的结构条件，使网络能把“手要去哪里”直接对应到图像位置。代价是依赖准确相机标定，而且 action map 仍不能表达力、关节动力学与接触刚度。

### 4.2 并行 context block

直接 AdaLN 注入低维动作可能缺少空间结构；把 action map 与 noisy latent 拼接后全量微调又可能遗忘预训练先验。论文从主 DiT 选择性复制每五个 block，实际为层 0、5、10、15、20、25、30、35，形成可训练 context branch，冻结主干。每个分支输出经零初始化卷积残差注入：

$$\boldsymbol x_i=\operatorname{DiT}_i(\boldsymbol x_{i-1})+\alpha W_{\mathrm{zero}}^{(i)}\boldsymbol h_i.$$

零初始化保证训练开始时控制分支不扰动原模型，之后逐步学习动作响应。

![](assets/ABot-PhysWorld%20-%20物理对齐机器人操控世界模型/figures/03-action-injection.png)

> **图 4：空间 action map 与并行 context blocks。** 上支路只处理动作图，下支路保留冻结的 Wan/DiT 视频主干，控制信号经 Zero Conv 分层残差注入。这是一种“保留生成先验、局部增加控制”的结构。图像边缘和文字已视觉核验完整。  
> 来源：论文 Figure 4，§3.3，<https://arxiv.org/abs/2603.23376>

---

## 5. 训练设置与成本

TI2V 在 128 张 NVIDIA H20 上训练 6,000 steps，global batch 128，学习率 $10^{-5}$；DPO 使用前述 LoRA 设置；A2V 训练 20,000 steps、batch 16、学习率 $5\times10^{-5}$，随机帧采样与可变 stride 增强。论文披露了主要优化参数，这是优点；但未给每阶段墙钟时间、能耗、候选视频总数和 VLM 标注调用成本，完整复现仍昂贵。

---

## 6. 评测设计

### 6.1 PAI-Bench / PBench 机器人子集

174 个复杂操控视频来自 BridgeData V2、AgiBot 与 OXE。Qwen2.5-VL-72B-Instruct 回答 886 个二元问题，Domain Score 覆盖空间/接触 36.3%、时间/因果 28.6%、物理属性/状态变化 34.1%。Quality Score 汇总审美、成像、主体/背景一致性、运动平滑和 I2V 一致性。

### 6.2 EZSbench

EZSbench 组合训练中未见的机器人形态、场景、任务与视角。初始图像一部分由 Nano Banana 合成，另一部分在真实机械臂图上用 VLM 改背景。Qwen3-VL-32B-Thinking 出题，Qwen2.5-VL-72B-Instruct 回答；30–50% 问题强制为负向问题。分数为：

$$S_v=\frac1{|Q_v|}\sum_{q\in Q_v}\mathbb I(\operatorname{VQA}(v,q)=\operatorname{GT}(q)).$$

“训练独立”主要指组合式 OOD 场景，而不是评测完全独立于生成模型生态；合成图、VLM 出题与 VLM 判分都可能带来分布偏差。论文也未给出 benchmark 总样本数。

### 6.3 A2V 评测

从动作视频数据集中均匀采样 200 例，用 PSNR/SSIM 比较帧级重建；微调 YOLO 检测夹爪轨迹，再用 nDTW 评价轨迹一致性。它验证动作跟随，但测试集来自内部 A2V 数据分布，不是严格跨机器人闭环任务。

---

## 7. 结果与消融

### 7.1 DPO 的核心受控证据

| 指标 | Base | +DPO | 变化 |
|---|---:|---:|---:|
| Quality Score | 0.7678 | 0.7676 | -0.0002 |
| Domain Score | 0.8785 | **0.9306** | **+0.0521** |
| Avg. | 0.8232 | **0.8491** | **+0.0259** |
| Subject Consistency | 0.9406 | 0.9355 | -0.0051 |
| Motion Smoothness | 0.9916 | 0.9908 | -0.0008 |

这张表支持：DPO 主要提升物理/领域评分，传统视觉质量几乎不变；但部分子指标略降，因此“完全无损”仍应谨慎。

### 7.2 PBench 横向比较

DPO 模型的 Domain Score 0.9306 和平均分 0.8491 为表中最高；Veo 3.1 的 Quality Score 0.7740 高于 ABot 的 0.7676。更准确的结论是 ABot 在该 MLLM 判分协议下物理项领先，而非整体视觉质量全面领先。

![](assets/ABot-PhysWorld%20-%20物理对齐机器人操控世界模型/figures/04-pbench-results.png)

> **图 5：PBench 定性案例。** 图中标注了基线的夹爪/物体变形、穿透、无接触抓取和目标错误，作者方法完成了目标操作。它是有解释性的案例证据，但单个场景不能替代表格的总体统计。图片标签与边界已视觉核验完整。  
> 来源：论文 Figure 5，§4，<https://arxiv.org/abs/2603.23376>

### 7.3 EZSbench 与 A2V

EZSbench 上 ABot 的 Quality 0.7694、Domain 0.8366、Avg 0.8030，高于表中的 WoW-wan、GigaWorld-0、Cosmos-Predict 2.5 与 UnifoLM-WMA-0。该表没有 Veo 3.1 或 Sora v2 Pro，故不能用它证明在 EZSbench 上胜过二者。

A2V 结果为 PSNR 21.09、SSIM 0.8126、Trajectory 0.8522，高于 Enerverse-AC 的 20.42/0.7542/0.8157 和 Gen-Sim 的 18.05/0.7413/0.6195。该表也没有 Veo/Sora；论文结论中“trajectory consistency compared to Veo 3.1 and Sora v2 Pro”的措辞超出了 Table 3 的直接证据范围。

### 7.4 缺失的组件消融

论文没有分别消融 proposer/scorer 解耦、Tier 1 veto、正负问题比例、锦标赛采样、LoRA rank、$\beta$、action map 编码元素、context block 数量、数据重平衡与物理 caption。现有唯一清晰的因果证据是 base 与 +DPO 的对比。

---

## 8. 作者声称与统一证据的严格区分

| 类别 | 结论 | 证据状态 |
|---|---|---|
| **受控消融** | DPO 提升 PBench Domain Score 且 Quality 基本不变 | Table 1 直接支持 |
| **横向 benchmark** | PBench 平均分、EZSbench 表中平均分领先列出基线 | 对应表直接支持，但依赖 MLLM-as-Judge |
| **A2V 比较** | 优于 Enerverse-AC 与 Gen-Sim | Table 3 支持 |
| **更强宣传** | A2V 轨迹一致性超过 Veo/Sora | Table 3 未列二者，不能直接支持 |
| **物理世界模型** | 视频更少出现穿透/反重力等视觉违规 | benchmark 与案例支持趋势 |
| **可用于真实闭环控制** | 提高真实机器人任务成功率 | 论文没有闭环实验，不成立 |

---

## 9. 局限、未来工作与个人判断

### 9.1 作者明确承认

论文结论明确写出两点：模型依赖固定视角数据；缺少 closed-loop evaluation。未来工作是 multi-view generation 和 real-world deployment。

### 9.2 进一步可见的边界

固定视角过滤限制动态相机与多视角一致性；VLM-as-Judge 可能把视觉可辨识性当作物理真实性；EZSbench 总规模和去重细节不透明；偏好构造依赖 Gemini 3 Pro、Nano Banana 等外部模型；训练需要 128 张 H20；动作图依赖相机标定；没有力、触觉、关节动力学或真实任务成功率。

### 9.3 个人判断

物理偏好对齐的方向是合理的：最大似然确实不会主动区分“看似合理”和“物理正确”。但真正的物理验证必须跨过两个层次：第一层是独立、人类或程序化物理评测，降低 VLM judge 循环；第二层是把生成模型放入闭环规划，观察它是否提高动作选择和真实机器人成功率。当前 ABot-PhysWorld 完成了第一层的一个可扩展近似，还没有完成第二层。

---

## 10. 材料来源

本地保存了 arXiv 扩展预印本、GitHub 官方技术报告和四张源码原图：

- `assets/ABot-PhysWorld - 物理对齐机器人操控世界模型/sources/paper.pdf`
- `assets/ABot-PhysWorld - 物理对齐机器人操控世界模型/sources/github-tech-report.pdf`
- `assets/ABot-PhysWorld - 物理对齐机器人操控世界模型/sources.md`
