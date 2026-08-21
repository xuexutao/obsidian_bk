---
type: paper-note
status: done
domain: VLA
paper: World Action Models are Zero-shot Policies
year: 2026
arxiv: 2602.15922
doi: null
source: https://arxiv.org/abs/2602.15922
project: https://dreamzero0.github.io/
code: https://github.com/dreamzero0/dreamzero
tags:
  - VLA
  - World-Models
created: 2026-08-20
updated: 2026-08-21
---

# DreamZero：视频世界动作模型的零样本机器人策略

## 基本信息

- **论文标题**：World Action Models are Zero-shot Policies
- **作者**：Seonghyeon Ye†、Yunhao Ge*、Kaiyuan Zheng*、Shenyuan Gao*、Sihyun Yu*、George Kurian*、Suneel Indupuru*、You Liang Tan*、Chunning Zhu、Jiannan Xiang、Ayaan Malik、Kyungmin Lee、William Liang、Nadun Ranawaka、Jiasheng Gu、Yinzhen Xu、Guanzhi Wang、Fengyuan Hu、Avnish Narayan、Johan Bjorck、Jing Wang、Gwanghyun Kim、Dantong Niu、Ruijie Zheng、Yuqi Xie、Jimmy Wu、Qi Wang、Ryan Julian、Danfei Xu、Yilun Du、Yevgen Chebotar、Scott Reed、Jan Kautz、Yuke Zhu†、Linxi "Jim" Fan†、Joel Jang†
  - †项目负责人，*核心贡献者
- **机构**：NVIDIA
- **arXiv**：2602.15922v1 [cs.RO]，2026 年 2 月 17 日
- **项目页**：https://dreamzero0.github.io/
- **代码仓库**：https://github.com/dreamzero0/dreamzero
- **许可**：CC BY 4.0
- **OpenReview**：cd33uUB609

## 一句话结论

DreamZero 提出 **World Action Model (WAM)**：以预训练视频扩散模型（Wan2.1-I2V-14B-480P）为主干，仅新增极少量机器人参数，联合预测**未来视频潜变量与动作**；在 AgiBot G1 与 DROID 上对未见任务/未见环境取得超过 2 倍于最强 VLA 基线的零样本泛化，借助 38× 推理加速使 14B 自回归视频 DiT 跑到 7 Hz 闭环控制，并仅用 10–20 分钟跨 embodiment 视频或 30 分钟 play data 即可完成新机器人迁移。

---

## 1. 研究背景与问题定义

### 1.1 研究问题

机器人基础模型正在经历从"专用模仿策略"向"通用预训练 VLA"的范式转变。这类模型把视觉-语言模型的语义先验迁移到机器人控制，已经能够听懂新指令、识别新物体。然而 VLA 在两类分布外场景中仍然脆弱：**未见过的物理动作**（如解鞋带这类训练数据里没有的精细操作）和**全新的环境布局**（光线、家具、桌面高度都变了）。一旦任务或环境超出训练演示的覆盖，VLA 的成功率往往断崖式下跌。

DreamZero 把这一困难上溯到一个更根本的归因：VLA 预训练于"静态图像-文本"对，缺乏对物理动力学和运动控制的稠密先验；并且其训练目标天然是"观测→动作"的直接映射，没有一个"未来世界应该长什么样"的中间表示。这导致模型即便识别对了任务，也难以想象合理的运动轨迹。

### 1.2 现有方法的瓶颈

论文将现有路线系统归纳为三组瓶颈：

1. **VLA 的"语义强、动力学弱"**。OpenVLA、π₀、GR00T 等模型在 VLM 初始化下对语言与物体有出色泛化，但只能从静态图文预训练继承先验，缺乏对时空动力学和精细灵巧运动的表示。
2. **数据收集代价昂贵**。要让 VLA 真正适应新环境，研究者往往要跨数百个环境采集人类遥操作数据，且任务泛化受限于语言条件动作原语的固定库。
3. **VLA 难以从"非重复、多样化"演示中学习**。VLA 在高度重复的专家演示上表现尚可，但遇到差异巨大的混合数据集时，常常只能学会占主导的"拾-放"行为而忽略其他语义。

### 1.3 本文核心贡献

论文明确把"**未来视觉状态**"作为动作生成的核心中间变量，训练一个联合视频-动作的世界动作模型。其贡献可总结为四点：

1. **方法**：以 Wan2.1-I2V-14B 为 backbone、加上最小参数（state/action 编/解码器），用 flow matching 联合去噪视频潜变量与动作；只在视频侧自回归，并通过 KV cache 回写真实观测来避免误差累积。
2. **泛化**：在 AgiBot G1 与 DROID 上对未见任务/未见环境取得 ≥2× 最强 VLA 基线的零样本任务进度；其中 DROID 上成功率 22.5%，约为 GR00T N1.6（12.5%）与 π₀.₅（7.5%）的 1.8–3 倍。
3. **实时性**：CFG Parallelism + DiT Caching + torch.compile/CUDA Graphs + Kernel/Scheduler 优化 + NVFP4 量化 + DreamZero-Flash 累计 ~38× 加速，将 5.7 s 延迟压到 150 ms，实现 7 Hz 闭环。
4. **跨 embodiment 迁移**：YAM 20 分钟视频或人类 12 分钟视频即可让 AgiBot G1 在未见任务上取得 42% 以上的相对提升；30 分钟 play data 即可将策略适配到 YAM 新机器人并保留零样本泛化。

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设

DreamZero 是一个**条件生成模型**，在每个动作块（action chunk）上联合预测未来视觉帧和未来动作。它的输入、输出与隐含假设如下：

- **输入**：
  - 历史视觉观测 $\mathbf{o}_{0:l}$：来自多个相机的图像序列（多视角会拼成 composite frame，**论文未披露**具体拼接分辨率）
  - 语言指令 $\mathbf{c}$：自然语言任务描述
  - 本体感觉状态 $\mathbf{q}_l$：当前步机器人关节/夹爪/底盘状态
- **输出**：
  - 未来视频 $\mathbf{o}_{l:l+H}$：长度 $H$ 的未来视觉帧
  - 未来动作 $\mathbf{a}_{l:l+H}$：与未来视频对齐的连续动作块
- **假设**：
  - $H$ 是固定的预测时间跨度（AgiBot 上 48 步 × 30 Hz = 1.6 s，DROID 上 24 步 × 15 Hz = 1.6 s）
  - $l$ 是从轨迹中随机采样的索引；同一 chunk 内的所有帧共享同一个去噪时间步 $t_k$
  - 视频采样率为 5 FPS，动作采样率 30 Hz（AgiBot）/15 Hz（DROID）

### 2.2 关键符号和目标函数

DreamZero 的核心是把"观测→动作"这一直接映射改写为"观测 + 指令 + 本体 → 未来视频 + 未来动作"的联合预测。其联合分布可以分解为视频预测项与逆动力学（IDM）项的乘积：

$$\underbrace{\pi_0(\mathbf{o}_{l:l+H}, \mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)}_{\text{DreamZero}} = \underbrace{\pi_0(\mathbf{o}_{l:l+H} \mid \mathbf{o}_{0:l}, \mathbf{c}, \mathbf{q}_l)}_{\text{video prediction}} \cdot \underbrace{\pi_0(\mathbf{a}_{l:l+H} \mid \mathbf{o}_{0:l+H}, \mathbf{q}_l)}_{\text{IDM}}$$

直觉上：**先在脑内"想象"未来长什么样，再从这个想象的未来反推出应该执行什么动作**。这种分解的好处是显式地让动作受"视觉上是否合理"约束，而不会被卡在某个专家演示里。

但论文没有把分解当成两个独立模型去实现，而是用**单一端到端模型**对 $[\mathbf{z}_{t_k}^k, \mathbf{a}_{t_k}^k]$ 做联合 flow-matching 去噪，避免模态不对齐。具体损失函数为：

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z},\mathbf{a},\{t_k\}}\Bigg[\frac{1}{K}\sum_{k=1}^{K} w(t_k) \big\lVert \mathbf{u}_{\theta}\big([\mathbf{z}_{t_k}^{k}, \mathbf{a}_{t_k}^{k}]; \mathcal{C}_k, \mathbf{c}, \mathbf{q}_k, t_k\big) - \mathbf{v}^{k} \big\rVert^2 \Bigg]$$

其中符号含义：
- $\mathbf{z}_0^k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，$\mathbf{z}_1^k$ 为干净视频潜变量；$\mathbf{a}_0^k,\mathbf{a}_1^k$ 同理作用于归一化动作
- $\mathbf{z}_{t_k}^{k} = t_k \mathbf{z}_1^{k} + (1-t_k) \mathbf{z}_0^{k}$（对动作亦然）是噪声插值
- $\mathcal{C}_k = \{(\mathbf{z}_1^{j}, \mathbf{a}_1^{j})\}_{j=1}^{k-1}$：前 $k-1$ 个干净 chunks 的上下文
- $\mathbf{v}^{k} = [\mathbf{z}_1^{k}, \mathbf{a}_1^{k}] - [\mathbf{z}_0^{k}, \mathbf{a}_0^{k}]$：速度目标
- $w(t_k) > 0$：权重函数（**论文未披露**具体形式）
- 训练时用 teacher forcing：当前 chunk 在所有前序干净 chunk 的条件下被去噪

## 3. 核心方法

### 3.1 总体框架

DreamZero 是一种 World Action Model。它沿用 Wan2.1-I2V-14B-480P 这个已在大规模图文/视频对上预训练好的 14B 视频扩散 Transformer，并在它上面**只增加三组小参数**：state encoder、action encoder、action decoder；文本编码器、图像编码器、VAE 全部冻结。这种"轻触式"扩展的动机是尽量保留视频 backbone 已经学到的世界动力学先验，不让少量机器人数据把通用先验冲掉。

![](assets/DreamZero%20-%20视频世界动作模型的零样本机器人策略/fig4-model.png)

> 图 1：DreamZero 的训练（左）与推理（右）流程。训练时三路输入（视频帧/语言指令/本体状态）经 VAE、文本/状态编码器送入 Causal DiT 块，对噪声 video latent 与 noise action 做联合 flow matching；推理时用 KV cache 实现自回归生成，每执行一个 action chunk 立即用真实观测回写 cache，避免长期依赖自己生成的视频。  
> 来源：论文 Figure 4，https://arxiv.org/html/2602.15922

DreamZero 的关键架构选择有四：

1. **只在视频模态上做自回归**。这样既可利用 KV cache 加速，又让"过去真实视频"始终作为下一帧生成的依据，避免闭环动作预测中的误差传播。
2. **Chunk-wise 自回归生成**。每 chunk 含 $K=2$ 个潜帧，默认 $M=4$ 个 chunk（最大上下文 8 个潜帧 ≈ 33 原始帧 ≈ 6.6 s），与 LLM 的可变长度 token 训练相容。
3. **多视角直接拼接**。多相机画面被拼成一帧再送入视频 backbone，**不修改** backbone 结构（避免了为多视角重写位置编码）。
4. **joint flow matching over video & action**。video latent 与 action 共享同一个去噪时间步、共享同一个 transformer 输出头，但用各自解码器解码出像素空间和关节空间。

### 3.2 关键模块一：联合视频-动作 Flow Matching

与经典 DDPM 相比，flow matching 是一条从纯噪声 $\mathbf{x}_0$ 到目标 $\mathbf{x}_1$ 的常速度直线，训练更稳、采样更快。DreamZero 的关键变体是**把 video latent 和 action 当作"一个长 token 序列"一起加噪、一起去噪**：

$$[\mathbf{z}_{t_k}^{k}; \mathbf{a}_{t_k}^{k}] = t_k [\mathbf{z}_1^k; \mathbf{a}_1^k] + (1-t_k)[\mathbf{z}_0^k; \mathbf{a}_0^k]$$

直觉解释：模型每一步都被要求去噪一个混合 token（视频+动作），于是它**必须同时理解两者的对应关系**——视频预测偏了，动作也跟着偏，反之亦然。这正是把"未来视觉"作为动作生成强约束的核心机制。论文指出共享时间步会让 action head 学会"在带噪声的视频上下文下也能给出可执行动作"，这为后面的 DreamZero-Flash 提供了训练-测试一致性。

### 3.3 关键模块二：闭环推理与 KV Cache 回写

DreamZero 在推理时有两步关键设计：

**第一步：异步自回归去噪**。机器人不再"等"模型把一个 chunk 全部去完才开始执行：控制器持续执行最新 chunk，模型在最新真实观测上并发去噪下一个 chunk。论文给出目标延迟为 $\lesssim 200$ ms（相对 1.6 s 的 chunk horizon 是 8× 余量），但**单卡 H100/GB200 朴素实现要 5.7 s**。

**第二步：用真实观测回写 KV cache**。这是 WAM 相对于"纯视频生成"的关键结构性优势：每执行完一个动作 chunk，模型都丢弃自己预测的 video latent，转而用真实观测帧重新编码并更新 cache，从而**切断自回归视频生成固有的误差累积**。这一点既缓解了"想到错的就越想越错"，又让模型作为"有状态策略"可以利用视觉历史处理需要记忆的任务（比如"把它放回刚才那个位置"）。

**算法骨架**（论文 Algorithm 2）：

1. **Prefill**：将初始多视角图像编码成潜变量，前向通过 DiT 更新 KV cache
2. **采样循环**：采样噪声 → 联合去噪（flow matching $t:0 \to 1$）→ 解码动作 → 异步执行 → 用真实观测回写 KV cache

### 3.4 训练目标与损失函数

损失函数即 §2.2 中的 flow-matching MSE。**教师强制（teacher forcing）** 是训练期的核心安排：每个 chunk 在前序干净 chunk 真实潜变量的条件下被去噪；通过注意力掩码限制当前 chunk 只能看到 $k-1$ 个干净 chunk 的 token。这种"过去干净、未来噪声"的安排，使训练时的去噪难度与推理时一致（推理时上一次预测的视频块也会立即被真实观测替换）。

### 3.5 推理流程与复杂度

DreamZero 的延迟主要来自三块：16 步迭代去噪、14B DiT 骨干、推理阻塞机器人动作。论文没有公开各步的绝对延迟，但给出了**累加 38× 加速**的明细（详见 §5.4）。最终的部署配置是 **2× GB200 GPU、150 ms/chunk、7 Hz**。这与消费级 GPU 上 VLA 跑 20 Hz 以上相比仍有数量级差距，但已经是 14B 自回归视频 DiT 第一次在真实机器人上跑成闭环。

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理

**AgiBot G1 预训练集**（约 500 小时）：

- **规模**：7 200 个 episode，每个平均 4.4 分钟、约 42 个子任务
- **场景**：22 个独特环境（家庭、餐厅、超市、咖啡店、办公室、仓库、实验室、酒店、零售店等）
- **embodiment**：双机械臂+移动底盘（bi-manual mobile manipulation）
- **数据哲学**：**任务多样性优先于演示重复**。每个任务收集 50 个 episode 之后弃用，迫使操作员持续提出新任务——这正是论文反复强调"非重复、多样化"训练的关键
- **任务粒度**：每个 episode 标注 3 个粗粒度任务（如"整理物品""地面垃圾清理""购物篮归还"）

**DROID 预训练集**（公开数据）：

- **embodiment**：Franka 单臂
- **性质**：异构性最强的公开机器人数据集之一
- **作用**：验证 WAM 在**开源、跨机构、跨环境**数据上是否仍然有效

**下游后训练数据**（三个真实任务）：

| 任务 | 时长 | 阶段数 | 随机化 |
|---|---|---|---|
| 折衬衫 | 33 h | 5 个连续阶段 | 2 种衬衫、初始位置随机 |
| 水果打包 | 12 h | 1 步 | 10 个水果，组合和位置随机 |
| 清理桌面 | 40 h | 5 件垃圾 + 5 件餐具 | 物体类型、组合、位置随机 |

### 4.2 Baseline 与评价指标

**基线模型**：两个 SOTA VLA：

1. **GR00T N1.6**（NVIDIA 2025）
2. **π₀.₅**（Physical Intelligence 2025）

每个基线用两种初始化：

- **From-scratch**：仅加载 VLM 预训练权重，无机器人数据
- **From-pretrained**：用官方 checkpoint（已在大规模跨 embodiment 数据上预训练）

两个基线都在**与 DreamZero 完全相同**的数据上继续训练，并且**计算预算匹配**（相同 batch size × 训练步数）。这是一种相当"用力"的公平设置。

**评价指标**：

- **平均任务进度（average task progress, 0–1.0）**：每个任务被手工拆成若干阶段，每阶段计 1/N 分；越接近 1 表示完成度越高。这是论文最核心的数值指标。
- **成功率（success rate）**：DROID 任务上额外报告，用于兼容既有 VLA 评测习惯
- **未见任务/未见环境**：评估时**故意**在训练集未覆盖的场景中执行；每个 checkpoint 跑 80–100 rollouts 取平均

### 4.3 实现细节

- **AgiBot 训练**：100K 步，全局 batch size 128
- **DROID 训练**：100K 步，全局 batch size 128
- **动作表示**：相对关节位置（论文报告 LoRA 效果不佳，故全参数微调）
- **数据过滤**：剔除空闲动作（idle action）
- **消融基线**：在 §5.3 的消融里用 Wan2.1-I2V-5B-480P 起步以对比模型尺寸
- **后训练**：每任务 50K 步，更新除文本/图像编码器与 VAE 之外的全部参数
- **动作平滑**：执行前用 Savitzky-Golay 滤波（窗口 21、多项式阶 3）抑制高频抖动

## 5. 实验结果

### 5.1 主要定量结果

DreamZero 用 4 个研究问题组织实验（Q1–Q6，详见 §5.3 与 §5.4），核心定量结论如下。

**Q1：能否从多样化、非重复数据中学习？**  
在 AgiBot G1 的 **seen-task 新环境**评测中，DreamZero 达到 **62.2%** 平均任务进度，最强 VLA baseline（π₀.₅-pretrained 或 GR00T N1.6-pretrained）为 **27.4%**——超过 2 倍。From-scratch VLA 几乎为 0。在 DROID 上 DreamZero 也优于所有多 embodiment 预训练 VLA baseline。

**Q2：能否零样本泛化到未见任务？**  
在 AgiBot G1 的 **unseen-task** 评测中，DreamZero 达到 **39.5%**，而最强预训练 VLA 为 **16.3%**、From-scratch VLA < 1%。DROID 上 DreamZero **49%** 任务进度 / **22.5%** 成功率，分别约为 GR00T N1.6（31% / 12.5%）的 1.6/1.8 倍、π₀.₅（33% / 7.5%）的 1.5/3 倍。

**Q3：后训练能否匹配 VLA？**  
在 3 个下游任务上，DreamZero 在折衬衫、清理桌面上与 VLA 基线相当，在水果打包上显著超越。值得注意的是，From-scratch VLA 在这些任务上**过拟合训练数据**，在新环境/新距离/不同桌高时全部失败——DreamZero 即使没有跨 embodiment 预训练也能匹配或超越。

**Q6：DreamZero-Flash 实时性**  
| 方法 | 去噪步数 | 任务进度 | 推理速度 | 加速 |
|---|---|---|---|---|
| DreamZero | 4 | **83% ± 6.1%** | 350 ms | 1× |
| DreamZero | 1 | 52% ± 10.2% | 150 ms | 2.33× |
| DreamZero-Flash | 1 | **74% ± 10.1%** | 150 ms | 2.33× |

这说明把视频/动作噪声调度解耦后，**单步推理也能恢复 4 步性能的大部分**（仅低 9%），同时推理快 2 倍。论文把 DreamZero-Flash 训练安排在主模型训练的最终阶段，**论文未披露**具体 epoch/步数。

这些数据"说明什么"：第一，**联合视频-动作建模在零样本泛化上确实具备结构性优势**——仅仅"换种预测目标"就能让任务进度翻倍，提示 VLA 的瓶颈可能不在模型容量，而在训练信号。第二，**未来视觉中间表示是有效的归纳偏置**——尤其在 AgiBot 这种"非重复、跨环境"的数据上，VLA 几乎无法学习，DreamZero 仍能保留 60%+ 进度。第三，**视频预测质量对动作有直接传导**——从 §5.4 的失败分析看，DreamZero 的失败几乎都源自视频生成错误而非动作预测错误。

### 5.2 定性结果

![](assets/DreamZero%20-%20视频世界动作模型的零样本机器人策略/fig1-overview.png)

> 图 2：DreamZero 的总体能力拼图。上方为多样化预训练数据（双机械臂+单臂+任务多样性>重复）；中部为 World Action Model 同时输出未来视频与连续动作；下方为"未见图任务/未见环境"的零样本泛化样例（解盒子、扇汉堡、按电梯、压下烤面包机杠杆）与"新 embodiment 少样本适配"（30 分钟 play data 后抓泰迪熊、橙子放进南瓜、泡面放进纸袋、香蕉上架）。  
> 来源：论文 Figure 1，https://arxiv.org/html/2602.15922

论文在正文中展示了三类定性证据：

1. **视频-动作对齐**：预测的未来视频与生成动作紧密对齐（Figure 2），即使在完全未见过的任务上也是如此。这从直觉上验证了"先想象未来，再反推动作"这一范式确实被模型学到了。
2. **自由形式指令**（Figure 3）：DreamZero 能在自然语言指令下完成"操作物体""使用工具""人机交互"等多样化任务，包括未见过的物体组合和工具使用。
3. **失败案例分析**：DreamZero 的失败**几乎全部**源于视频生成错误，而非动作预测错误——策略会忠实地执行一个视觉上看起来"合理"但物理上不可达的轨迹。这既说明视频-动作耦合足够紧，也说明**改进视频骨干会直接转化为更好的 WAM 性能**（论文明确把这一点列为"未来工作"）。

相比之下，预训练 VLA baseline 在失败时通常表现为"伸手去抓物体"（无论指令是什么），表明它们过拟合到了占主导的"拾-放"行为，而不是真正理解了任务语义。

### 5.3 消融实验

论文 Table 4 用 50K 步、batch size 32、PnP Easy 任务，给出三组核心消融：

| 消融问题 | 配置 | 模型 | 数据 | 任务进度 |
|---|---|---|---|---|
| **Q1：数据多样性** | DreamZero (AR) | 14B | Repetitive | 33% ± 4.2% |
| | DreamZero (AR) | 14B | **Diverse** | **50% ± 6.3%** |
| **Q2：模型规模** | DreamZero (AR) | 5B | Diverse | 21% ± 4.2% |
| | DreamZero (AR) | 14B | Diverse | **50% ± 6.3%** |
| | VLA | 5B | Diverse | 0.0% |
| | VLA | 14B | Diverse | 0.0% |
| **Q3：自回归 vs 双向** | DreamZero (BD) | 14B | Diverse | 50% ± 14.4% |
| | DreamZero (AR) | 14B | Diverse | **50% ± 6.3%** |

**关键发现**：

- **Q1**：数据多样性比重复演示更值钱（50% vs 33%），即使在简单拾放任务上也成立。这与 VLA 范式下的"反复演示"经验相反——WAM 能从"见过很多种任务"中获益。
- **Q2**：WAM 有清晰规模效应（14B 50% vs 5B 21%）；而 VLA **无论 5B 还是 14B 都学不会多样化数据**（0%）。这强烈提示训练目标比模型规模更重要。
- **Q3**：自回归与双向 DiT 在 PnP Easy 上任务进度相当（50% vs 50%），但**自回归版本方差小 2.3×**（±6.3% vs ±14.4%），动作更平滑（论文 Figure 5 给出时序一致性证据），并且**推理快 3–4×**（KV cache + 可变长度的功劳）。这是 DreamZero 选用 AR 而非 BD 的实证依据。

### 5.4 泛化、效率与失败案例

**Q4：跨 embodiment 迁移**：

| 方法 | 任务进度 |
|---|---|
| DreamZero（基线） | 38.3% ± 7.6% |
| DreamZero + Human2Robot Transfer | **54.3% ± 10.4%** |
| DreamZero + Robot2Robot Transfer | **55.4% ± 9.5%** |

数据量：YAM 20 分钟（双机械臂），人类 12 分钟（每任务 8 个演示，72 条多视角轨迹，覆盖 9 个未见任务）。两种迁移的相对提升均 **>42%**。YAM 形态与 AgiBot 接近（都是双机械臂平行夹爪）所以 Robot2Robot 略胜；Human 形态差异更大但仍能跑通，说明视觉未来这一中间变量对 embodiment 不那么敏感。

![](assets/DreamZero%20-%20视频世界动作模型的零样本机器人策略/fig11-embodiment-transfer.png)

> 图 3：跨 embodiment 迁移的数据源。左为 YAM 双机械臂真实操作视频（Robot2Robot），右为人类第一人称演示（Human2Robot），中间为 AgiBot G1 真实部署画面。形态差异巨大但都能成为有效训练信号。  
> 来源：论文 Figure 11，https://arxiv.org/html/2602.15922

**Q5：少样本新 embodiment 适应**：用 YAM 机器人上 55 条轨迹 / 11 个任务（约 30 分钟 play data）后训练 DreamZero-AgiBot checkpoint。模型既能保持零样本泛化，又能泛化到训练时完全没见过的物体（南瓜、泰迪熊、笔、杯面、纸袋等）。这说明"世界动力学"知识确实可以独立于 embodiment 单独迁移。

**效率与加速（Q6 补充）**：

| 优化 | H100 | GB200 |
|---|---|---|
| Baseline | 1× | 1.1× |
| + CFG Parallelism | 1.9× | 1.8× |
| + DiT Caching | 5.5× | 5.4× |
| + Torch Compile + CUDA Graphs | 8.9× | 10.9× |
| + Kernel & Scheduler Opts. | 9.6× | 14.8× |
| + Quantization (NVFP4) | — | 16.6× |
| + DreamZero-Flash | — | **38×** |

最终在 GB200 上把 5.7 s 延迟压到 150 ms，**实现 7 Hz 实时闭环控制**（使用 2 块 GB200）。除 DiT caching 和量化外，其他优化在数学上与 baseline 等价，**无可测量的性能退化**。

**实验协议与数据图谱**：

![](assets/DreamZero%20-%20视频世界动作模型的零样本机器人策略/fig5-main-results.png)

> 图 4：AgiBot G1 上的训练与评估任务图谱。预训练（左上）覆盖打包、备货、洗碗、叠衣、收拾五类多样化任务；Seen Tasks Evals（中间）包含整理短袖、碗堆叠、笔入杯、短裤折叠、水果拾放等已知任务但新环境；Unseen Task Evals（左下）含解鞋带、堆方块、按电梯、摘帽、熨衣服等训练中完全缺席的任务；Post-training 任务（右上）含折衬衫、水果打包、清理桌面。  
> 来源：论文 Figure 5，https://arxiv.org/html/2602.15922

### 5.5 关于"为什么是 WAM 而不是 VAM"

论文在相关工作与正文中花了不少篇幅解释命名："**World** Action Model" 而非 "Video Action Model"——视频只是世界的一种表示方式，未来 WAM 还可能把动作与触觉、力反馈、学习的潜变量等其他模态对齐。这一命名选择本身就是对"未来视觉是必要中间层而非任意一种辅助信号"的一种表态。（个人判断：这一表态对后续研究者是有方向性的，避免大家把 WAM 范式窄化为"用视频做 loss"）。

---

## 6. 与相关工作的关系

### 6.1 与 VLA 的关系

DreamZero 的论战对手是 VLA 这一族。论文明确指出 VLA 在两类分布外场景（未见过的新物理动作 + 全新环境布局）上泛化受限，并系统地把 VLA 范式的瓶颈归因于三件事：（1）从静态图文预训练，缺乏时空先验；（2）数据采集代价昂贵；（3）难以从"非重复、多样化"演示中学习。DreamZero 用 WAM 同时回应了三个瓶颈：视频预训练带时空先验；跨 embodiment 视频几乎免费；多样化数据反而是优势（消融 Q1）。

### 6.2 与 Video Model-based Robot Policies 的关系

这条线又有三支：

1. **"先合成视频、再提取动作"**（如 UniSim、Genie）：用独立 IDM 从生成视频里抠动作，缺点是视频与动作两套模型模态不对齐。
2. **"用人类视频预训练"**（如 HAT、GR-1）：用人类视频训练单独的视频生成器，再迁移到机器人。
3. **"联合视频-动作生成"**（iVideoGPT、3D-VLA 等）：与 DreamZero 同属一支，但多从零或从 VLA 起步。

DreamZero 的相对位置：基于**预训练视频扩散 backbone**（Wan2.1-14B）、**显式系统地**探索数据多样性和规模、**用自回归架构**（更适合长程世界-动作建模）、并在**新任务新环境**上取得 SOTA 泛化与 SOTA 跨 embodiment 迁移。换言之，它在数据规模、模型规模、架构选择、工程化上同时打满，是这条主线目前最完整的方案。

### 6.3 与其他世界模型架构的关系

附录 A 用一张表对比了三类世界模型架构：

- **潜空间世界模型**（JEPA、Dreamer）：在抽象潜空间预测未来状态，效率高但需要目标条件规划或测试时搜索，建模的是 $p(s_{t+1}|s_t, a_t)$
- **3D 点云世界模型**（PointWorld）：统一状态与动作在共享 3D 空间，但需要 MPPI 等显式优化来生成动作轨迹
- **WAM（DreamZero）**：联合建模 $p(\mathbf{o}_{t:t+H}, \mathbf{a}_{t:t+H} | \mathbf{o}_{0:t}, \mathbf{c})$，无需测试时优化，直接产生与预测视觉未来对齐的动作

DreamZero 的关键差异在于"**动作作为条件生成的一部分**"——它不把动作当成 RL 优化变量，也不当作独立 IDM 的输出，而是和未来视觉在同一个去噪过程里生成。这是它能去掉测试时优化的根本原因。

### 6.4 与 Fast-WAM 的关系

后续工作 Fast-WAM（arXiv:2603.16666）指出 DreamZero 真正的价值在训练期而非推理期：它在测试时跳过未来视频想象，但用一次前向编码得到的世界表征直接生成动作即可达到接近 DreamZero 的精度、且推理快 4× 以上。这反过来佐证了 DreamZero 论文中"未来视觉"作为**训练期归纳偏置**的核心论点。（个人判断：Fast-WAM 补全的是 DreamZero 推理时"想象未来"是否必要的实证；DreamZero 论文仍以"想象未来+反推动作"作为方法骨架，两者并不冲突。）

---

## 7. 局限与批判性评价

论文自己列出了五条局限，我们结合个人判断一并讨论。

1. **WAM 的 Scaling Laws 尚未建立**。论文说"尚未确立模型大小、数据集大小与训练计算之间的缩放规律"，并预期 WAM 的缩放曲线会与 VLA 不同、可能呈现更直接的动作缩放规律。这是一项**坦率的开放问题**——也是 WAM 范式能否在大模型时代复现 LLM 式"大力出奇迹"的关键不确定性。（个人判断：如果缩放曲线真的比 VLA 陡，那 WAM 在数据-算力维度上的天花板会比当前显示的更高，反之亦然。）

2. **人类数据利用非常有限**。仅 12 分钟人类视频。论文明确把"利用互联网规模第一人称视频"列为未来方向——这条线一旦打通，将是 WAM 相对 VLA 的**结构性优势**（VLA 也无法直接消费第一人称视频）。

3. **推理成本仍然很高**。7 Hz 需要 2× GB200，而 VLA 在消费级 GPU 上就能跑 20 Hz。论文把"更快推理"列为未来方向，并期待更小的视频骨干也能泛化。（个人判断：在 NVIDIA 自己的产品线内 7 Hz 是可接受的演示阈值，但若想进入 ROS 2 + 边缘 Jetson 的工业部署栈，仍有数量级差距。）

4. **长程推理有限**。当前架构只保留约 6 秒视觉记忆。论文把双系统架构（System 2 规划器）或扩上下文窗口列为未来方向——这恰好对应近期 LLM 路线上的"上下文工程"和"o1 式慢思考"两大主题。

5. **高精度任务受限**。继承行为克隆的局限，在钥匙插入、精细装配这类亚厘米级任务上仍可能失败。论文指出**多样化策略优先广度可能低估高精度所需的稠密演示**，这是一个**取舍**而非简单的 bug。

**个人批判性评价**：

- **方法层面**，"未来视频 + 逆动力学"这一分解本身并不是新想法（IDM 在 RL 里有数十年历史）。DreamZero 的真正贡献在于**用 video diffusion backbone 把分解从"潜变量预测 + 简单回归"升级为"高保真视频生成 + 共享去噪"**，让两个模态在训练目标层面就被紧密耦合。这是一种"用更好的表示 + 更好的 backbone 实现老想法"的范式升级。
- **评测层面**，论文用 80–100 rollouts/checkpoint 的规模做主要结论；这个数字在 VLA 论文中算偏多，但与商业级产品测试仍有距离（个人判断：建议读者关注未公开的"100+ free-form 任务"细节）。
- **数据层面**，AgiBot 数据集是内部 500 小时数据，外部研究者无法复现"完全一致"的实验，这是论文最重要的复现门槛。开源代码 + 部分数据 + 仿真器（PolaRiS / Genie Sim 3.0）的组合是部分缓解而非完全解决。
- **价值判断**：DreamZero 不是 VLA 的替代品，而是一条**与 VLA 互补**的路线——VLA 擅长语言语义广度，WAM 擅长物理动作泛化；用未来的视角看，二者很可能在 System 1/System 2 双系统架构里各自占据一层。

## 8. 复现与实践建议

基于论文与公开材料，复现 DreamZero 需要做好以下心理和资源准备：

1. **算力门槛高**。AgiBot 主实验 14B 模型、100K 步、batch size 128；2× GB200 才能跑到 7 Hz 实时控制。对绝大多数学术实验室来说这是**单实验数十万美元**的成本。
2. **数据门槛高**。AgiBot 500 小时是 NVIDIA 内部采集；外部可用的 DROID 数据已经足够复现"Q2 零样本泛化"的核心数字，但无法复现 AgiBot Q1 数字。
3. **代码可用性**。GitHub 仓库已开源（**论文未披露**具体开源范围——README 写了"模型权重、推理代码、RoboArena 真实世界基准与 PolaRiS/Genie Sim 3.0 仿真基准代码"，但训练脚本与全量数据未明确开源）。建议从**推理与下游评估代码**入手，而非重训。
4. **关键复现步骤**：
   - 准备好 2× GB200 或等效硬件（H100 × 2 也能跑，但延迟稍高）
   - 加载 Wan2.1-I2V-14B-480P 预训练权重作为初始化
   - 添加 state encoder / action encoder / action decoder 三组小参数
   - 用 flow matching loss 在自己机器人的轨迹数据上 fine-tune
   - 推理时务必启用 KV cache 回写真实观测，并异步执行 action chunk
5. **低资源复现路径**（个人判断）：用 5B 模型 + DROID 子集 + 4 步去噪 + 单步 DreamZero-Flash 思路，可以在 1× H100 上完成"机制验证"级别的实验；这是论文里 §5.3 消融的 5B baseline 已覆盖的范围。
6. **推理时易踩的坑**：
   - 关闭 KV cache 回写真实观测会立即引发误差累积（论文 Figure 9 给了对照）
   - 多视角拼接分辨率**论文未披露**，需要从代码反推
   - Savitzky-Golay 滤波是关键，不要忽略

## 9. 个人启发与后续问题

读完 DreamZero 后，几个值得持续追踪的问题：

1. **WAM 是否真能成为"具身基础模型"？** 如果 WAM 的缩放曲线在未来 1–2 年被证明显著优于 VLA，那么整个 VLA 主线（OpenVLA、π₀、GR00T）都面临范式重估风险。**需要重点跟踪**：更大 backbone（如 Wan2.2 / Cosmos / Veo）上 WAM 的零样本数字。
2. **"想象未来"vs"编码世界"之争**。Fast-WAM 已经从"推理时不需要想象未来"方向提出挑战。如果这条线成立，DreamZero 的核心叙事（"想象未来→反推动作"）会被改写为"训练时想象未来、推理时只看世界表征"。这会改变整个赛道的优化目标——更多研究将流向"如何学到一个好的世界表征"。
3. **System 2 与 WAM 的结合**。DreamZero 当前只有约 6 秒记忆，论文明确把"双系统架构"列为未来方向。这与 LLM 主线上"o1 式慢思考 + 快速策略"的趋势同构。WAM 是否会演化出"先用 VLM 做高层任务分解、再用 WAM 跑局部闭环"的双层结构？大概率会。
4. **跨 embodiment 数据飞轮**。如果 WAM 真的能从人类视频吸收 50%+ 性能，那像 Ego4D 这种 4 000+ 小时第一人称视频将成为新时代的"ImageNet"。**待观察**：是否有团队在 Ego4D 上 pretrain WAM，并报告零样本数字。
5. **与 3D 表征的融合**。GaussianDream 已经在 VLA 训练阶段引入 3D Gaussian 重建与短时未来 3D 演化预测；DreamZero 仍以 2D 视频为主。两类世界模型可能在中长期融合为"2D 视频 + 3D 几何 + 动作"三模态联合生成。

（个人判断：DreamZero 最值得我们这些 VLA/具身研究从业者借鉴的不是"用视频扩散做策略"这一具体技巧，而是**重新审视训练目标**的勇气——它把"观测→动作"这一隐含的 BC 范式替换成"观测+指令+本体→未来视觉+动作"，从而解锁了"非重复、多样化"数据下的学习。这与 VLM 时代从"判别式预训练"切换到"生成式预训练"的范式转换在精神上同源。）

## 10. 材料来源

### 主要参考

| 类别 | 名称 | 链接 | 用途 |
|---|---|---|---|
| 论文 PDF | arXiv:2602.15922v1 | https://arxiv.org/abs/2602.15922 | 精读主来源 |
| 论文 HTML | arXiv HTML 版 | https://arxiv.org/html/2602.15922 | 提取方法图与定量结果 |
| 项目页 | DreamZero Project | https://dreamzero0.github.io/ | 视频与定性结果 |
| 代码仓库 | dreamzero/dreamzero | https://github.com/dreamzero0/dreamzero | 复现与推理代码 |
| OpenReview | cd33uUB609 | https://openreview.net/forum?id=cd33uUB609 | 同行评议记录 |

### 本地材料

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `assets/.../fig1-overview.png` | 总览图 | arXiv HTML | Figure 1, p.1 | §5.2 定性结果 |
| `assets/.../fig4-model.png` | 方法图 | arXiv HTML | Figure 4, p.4 | §3.1 总体框架 |
| `assets/.../fig5-main-results.png` | 评估任务图谱 | arXiv HTML | Figure 5, p.5 | §5.4 评测协议 |
| `assets/.../fig11-embodiment-transfer.png` | 跨 embodiment 迁移 | arXiv HTML | Figure 11, p.10 | §5.4 跨 embodiment 迁移 |

> 注：论文 PDF、Supplementary Material 与官方仿真器（PolaRiS / Genie Sim 3.0）暂未本地化保存，已在材料来源中保留官方链接；如需复现 AgiBot 主实验，建议从 GitHub 仓库下载。

### 相关延伸阅读

- **Fast-WAM**（arXiv:2603.16666）："Do World Action Models Need Test-time Future Imagination?"——DreamZero 的姊妹工作，验证 WAM 价值主要来自训练期。
- **GR00T N1.6**（Bjorck et al. 2025）：最强 VLA baseline 之一。
- **π₀.₅**（Physical Intelligence 2025）：另一个强 VLA baseline。
- **Wan2.1-I2V-14B-480P**：DreamZero 的视频 backbone。
- **DROID**、**AgiBot G1**：分别对应单臂公开数据与双机械臂+底盘内部数据。
