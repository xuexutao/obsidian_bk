---
type: paper-note
status: done
domain: World-Models
paper: Cosmos 3: Omnimodal World Models for Physical AI
year: 2026
arxiv: 2606.02800
doi: 10.48550/arXiv.2606.02800
source: https://arxiv.org/abs/2606.02800
project: https://research.nvidia.com/labs/cosmos-lab/cosmos3/
code: https://github.com/NVIDIA/Cosmos
tags:
  - World-Models
  - Physical-AI
  - Omnimodal
  - Mixture-of-Transformers
  - Video-Generation
  - World-Action-Model
  - Robot-Policy
created: 2026-10-24
updated: 2026-10-31
---

# Cosmos 3：物理 AI 的全模态统一世界模型

## 基本信息

- **正式名称**：Cosmos 3: Omnimodal World Models for Physical AI
- **机构 / 作者**：NVIDIA，集体署名（arXiv 提交人 Yin Cui，完整作者列表含 195+ 人）
- **版本**：arXiv 2606.02800，v1 提交于 2026-06-01，v4 修订于 2026-06-23（正文 80 页 + 附录约 59 页，合计 139 页）
- **发布模型**：报告首发 `Cosmos3-Super`（64B）与 `Cosmos3-Nano`（16B）；截至 2026-10-31，官方集合还已发布 `Cosmos3-Edge`（4B）及 T2I/I2V、Transfer/Sim、Nano/Edge Policy-DROID 等后训练权重。必须区分“报告首发状态”与“当前仓库状态”。
- **开源边界**：Cosmos 3 权重、推理与后训练框架已经开放，许可为 **OpenMDW-1.1**；但百亿级原始媒体池、十万亿级训练 token、完整预训练运行环境和内部 checkpoint 不可复刻。`NVIDIA/Cosmos` 当前只是 landing repo；**Cosmos 3 专属实现位于 `NVIDIA/cosmos-framework`，不能拿旧 `Cosmos-Predict2`/`Cosmos-Transfer2` 源码冒充 Cosmos 3**。
- **家族定位**：Cosmos 平台的第三代，把前代分离的 Predict（世界生成）、Transfer（结构控制）、Reason（物理推理）三条产品线 **合并进同一个 Mixture-of-Transformers 骨干**
- **重要性评估**：★★★★★。它第一次把一个"能读、能想、能生成视频音频、还能输出机器人动作"的全模态世界模型完整开源，并且给了 48 项推理基准 + 五类生成基准 + 真机策略榜首的系统证据。扣分点是复现门槛（Super 用 2048 张 GB200 训 17.86T token）、自动指标已明显饱和、以及若干子能力（中文文字渲染、音频保真度、通用推理）仍落后闭源前沿。

## 源码、权重与 revision 锁定（2026-10-31）

### 可复现入口表

| 对象 | 固定版本 / 入口 | 本次实际核验内容 | 边界 |
|---|---|---|---|
| 报告 | arXiv `2606.02800v4`，本地 `sources/paper.pdf` | 架构、训练目标、数据与消融 | 报告不是源码 |
| Landing repo | `NVIDIA/Cosmos`，页面核到提交 `8e2f3c7…`（2026-10-30 可见提交） | README、COOKBOOK、模型入口 | 当前根目录无模型实现；历史 cookbook/evaluation 已迁出 |
| **Cosmos 3 框架** | `NVIDIA/cosmos-framework@50e883466491abacdcb69843e1e026479d927a95` | `inference/model.py`、`sequence_packing/`、action transforms、训练/推理 examples | 这是本节 shape 与代码行为的主依据，明确是 Cosmos 3，不是 2.x |
| Nano 权重 | `nvidia/Cosmos3-Nano@e59a53c25979a090fa8706c9acc0c254a6e89b92` | `config.json`、`checkpoint.json`、`model_index.json`、`transformer/`、`vae/`、`sound_tokenizer/`、`vision_encoder/` | 权重约 16B；下载和运行成本高 |
| 关键 config | 上述 revision 的 `config.json` | `state_ch=48`、VAE 时/空压缩 `4/16`、`patch_spatial=2`、`hidden=4096`、`max_action_dim=64`、sound `64ch@25Hz`、上下文 `74000` | 配置是当前发布权重，不应倒灌解释报告中未公开的训练细节 |
| 2.x 对照 | 旧 Predict/Transfer 仓库与本文 §1.2 | 只用于代际差异 | **不得作为 Cosmos 3 的实现证据** |

### 端到端链路与动态 shape

统一记像素视频为 $x\in\mathbb{R}^{B\times3\times T\times H\times W}$，文本长度为 $L$，隐藏维为 $d$（Nano 为 4096），动作步数为 $T_a$、原始动作维为 $D_a$。

| 阶段 | 输入 shape | 变换与原因 | 输出 shape / 官方例子 |
|---|---|---|---|
| text | `[B,L]` token id | Qwen3-VL tokenizer + AR embedding；保留语言 next-token 能力 | `[B,L,d]`；Nano `d=4096` |
| image/video（理解） | `[B,3,T,H,W]` | ViT `patch=16`、`temporal_patch=2`、`spatial_merge=2`；局部 patch 保留空间语义 | 近似 `[B,N_u,d]`，$N_u=\lceil T/2\rceil\lceil H/16\rceil\lceil W/16\rceil/4$（整除时）；走 AR/Reasoner 路径 |
| image/video（生成） | `[B,3,T,H,W]` | Wan2.2 VAE 把像素变成连续 latent，避免主干直接处理像素 | `[B,48,T_z,H/16,W/16]`，$T_z=1+\lfloor(T-1)/4\rfloor$ |
| latent patchify | 上一行 | 空间再按 `2×2` patch 化并线性投影到 $d$；总空间压缩 `16×2=32` | `[B,N_v,d]`，$N_v=T_z\lceil H/32\rceil\lceil W/32\rceil$ |
| audio | `[B,2,S]`，48 kHz stereo | AVAE `hop=1920`、`io_channels=64`；每个 hop 对齐一个物理时间 token | `[B,N_s,64]→[B,N_s,d]`，$N_s=\lceil S/1920\rceil\approx25\cdot duration$ |
| action / control | `[B,T_a,D_a]` | 按域归一化，pad 到 `max_action_dim=64`，domain-aware 投影；每个动作时刻一个 token | `[B,T_a,64]→[B,T_a,d]`；未 padding 的真实输出再裁回 `[B,T_a,D_a]` |
| unified backbone | AR `[B,N_{AR},d]` + DM `[B,N_{DM},d]` | AR 因果；DM 对 AR+DM 全注意力；两路独立 LN/MLP，two-way joint attention | shape 不变；$N_{DM}=N_v+N_s+T_a$（按启用模态求和） |
| understanding head | AR hidden | LM head 自回归；原因是离散文本适合 next-token | logits `[B,N_{AR},151936]`，再解码文本 |
| generation heads | DM hidden | `llm2vae/llm2sound/llm2action` 预测 rectified-flow velocity；原因是连续、多解输出适合去噪 | video velocity 与 patchified latent 同形，sound `[B,N_s,64]`，action `[B,T_a,64]` |
| decode | 去噪后的连续 token | VAE/AVAE/action output projection | RGB `[B,3,T,H,W]`、audio `[B,2,S]`、action `[B,T_a,D_a]` |

**代入官方默认 T2V 示例**：框架 benchmark 使用 `B=1,T=189,H=480,W=848,fps=24`。因此 $T_z=1+188/4=48$，空间 token 网格为 $\lceil480/32\rceil\times\lceil848/32\rceil=15\times27$，生成视觉序列为 `[1, 48×15×27, 4096] = [1,19440,4096]`，VAE 解码回 `[1,3,189,480,848]`。若同时生成约 $189/24=7.875$ 秒音频，则 $N_s\approx197$；实际样本数需按 1920 hop 对齐。该计算来自 Cosmos 3 当前 config 与 token counter，不是对 Cosmos 2.x 的类比。

**代入源码最小 action-FD 示例**：`examples/integration/trainer_level_training.py` 构造 video `[1,3,5,128,128]` 与 raw action `[4,7]`，后者 pad 成 `[4,64]`。视频经 VAE 得 $T_z=2$，patch 网格 `4×4`，即 32 个视觉 token；动作是 4 个 token。`SequencePlan` 将首帧设为 clean vision、全部动作设为 clean action，余下视觉 token 为去噪目标。FD / ID / WAM / policy 的结构差异只来自 `condition_frame_indexes_vision/action`，不换 backbone。

### 逐模块超参、选择原因与 2.x 差异

| 模块 | Cosmos 3 当前公开超参 | 为什么这样做 | 相比 Cosmos 2.x |
|---|---|---|---|
| VAE/tokenizer | `state_ch=48`，时压缩 4、空压缩 16、patch 2 | 把 480p/720p 视频降到可承受 token 数 | 继承 Wan 系 VAE 思路，但 3 的统一序列还接音频/动作 |
| MoT | Nano 36 层、`d=4096`、32 heads/8 KV、FFN 12288 | 理解与生成共享注意力语义，同时隔离各自 LN/MLP | 2.x Predict/Transfer 是分离产品；3 是同一 backbone/config 模式切换 |
| 时间位置 | unified 3D MRoPE，`base_fps=24`，margin `15000` | 对齐不同 FPS、25Hz 音频和动作采样率，防 AR/DM 边界位置拥挤 | 2.x 主要服务视觉生成，未统一动作/音频时间轴 |
| flow matching | train steps 1000；shift `256/480/720 = 3/5/10`；action loss ×10 | 分辨率越高越偏高噪声训练；动作元素 MSE 数值小需增权 | 3 把视觉、音频、动作放入同一连续生成目标 |
| action | `max_action_dim=64`、32 embodiment domains | 用 padding + 域投影吸收不同具身维数 | 2.x 没有一等 action modality |
| packing | `max_num_tokens_after_packing=74000` | 可变分辨率/时长下按 token 而非样本数控显存 | 从视频专用 batch 演进为全模态统一打包 |

### 复现成本分级

- **接口复现**：固定 HF revision，用官方 Diffusers/vLLM-Omni；至少需下载数十 GB 权重，BF16 为官方验证精度。单卡只能依赖 offload/量化或较小输出，不能据此复现论文吞吐。
- **后训练复现**：框架内置 8×H100 80GB 示例 recipe；可复现 SFT、DROID policy、Transfer/Sim 等公开路径，但数据预处理和 DCP 转换仍重。
- **基础训练复现**：Nano 31.05T token/1024×GB200、Super 17.86T/2048×GB200，外部团队基本不可复现；公开框架与权重不等于公开全部原始训练语料和生产基础设施。

## 一句话结论

Cosmos 3 用**一条 token 序列 + 两套参数**把理解与生成缝在一起：序列前段是自回归的 AR 子序列（语言 + ViT 视觉，走 Reasoner 塔），后段是扩散的 DM 子序列（VAE 视觉 + 音频 + 动作，走 Generator 塔），两塔只在共享注意力处相遇（DM 可看 AR，AR 不看 DM）。同一个权重靠"哪些 token 是干净的、哪些是加噪的"这一件事切换出 VLM / 文生图 / 文生视频 / 图生视频 / 音视频 / 视频迁移 / 前向动力学 / 逆动力学 / 策略九种模式，从而在机器人、驾驶、智慧基建三类 Physical AI 任务上同时拿到开源第一。

---

## 1. 研究背景与问题定义

### 1.1 从"模型编排"到"单一骨干"

论文开篇给出的动机非常直白：现有 Physical AI 系统是被"拼"出来的。一个收拾餐桌的家庭机器人需要串起四五个互不相干的模型——VLM 做定位与规划、VLA 或 WAM 出动作序列、前向动力学模型或世界模型来模拟与评估未来。作者认为这种分离在原理上就站不住：**理解需要推理世界的未来演化和动作后果，而生成又依赖对世界与智能体行为的紧凑结构化表征**，两者是互相需要的，却被硬拆成判别式与生成式两条技术栈。

因此 Cosmos 3 的目标不是"再做一个更好的视频生成模型"，而是回答一个架构问题：**能不能用一套权重、不加任何结构改动，同时承担 VLM、图像生成器、音视频生成器、世界动作模型和正/逆动力学模型？** 论文的图 1 把这六种"模型类别"画成同一个六边形骨干的六个面，是全文的立论图。

![](assets/Cosmos%203%20-%20物理AI全模态统一世界模型/figures/01-unified-backbone.png)

> 图 1（Figure 1, p.5）：Cosmos 3 作为 Physical AI 通用骨干。语言、图像、视频、音频、动作五类模态在同一套网络里被联合建模，向下收束了 VLM、图像生成、音视频生成、策略 / 世界动作模型、前向动力学、逆动力学六类原本分离的模型。注意"动作"在这里被提到与视觉音频并列的一等模态，这是它区别于普通全模态模型的关键。

### 1.2 它在 Cosmos 家族里的位置

要理解 Cosmos 3，必须先厘清 Cosmos 不是模型而是一个**平台 + 三族模型**。结合本库已有的 [[Cosmos WFM - 物理AI世界基础模型平台]] 与 [[Cosmos-Predict 2.5 - 视频基础模型驱动的物理世界模拟]] 两篇笔记，三代的演进可以这样串：

| 代际 | 组织方式 | 代表产物 | 关系 |
|---|---|---|---|
| Cosmos 1.0 | **三个独立模型族**，各训各的 | Predict1 / Transfer1 / Reason1 | 定义平台：预测、控制转换、物理推理三件事 |
| Cosmos 2.5 | 每条线各自打磨成产品 | Predict2.5（流匹配统一 T2W/I2W/V2W）、Transfer2.5、Reason1 | 单线做到极致；Transfer2.5 直接长在 Predict2.5 主干上，已出现"合并"苗头 |
| **Cosmos 3** | **一个 MoT 双塔骨干，模式即配置** | Cosmos3-Super / Nano / Edge | 三族降维：Transfer 变成一种生成模式，Predict 的任务变成 Generator 的 T2V/I2V/V2V，Policy 变成动作的第三种模式 |

具体地，三者之间的技术继承关系非常清晰：

- **Tokenizer**：Predict2.5 用 WAN2.1 VAE，Cosmos 3 换成 **Wan2.2-TI2V-5B 的 VAE**（时间 4×、空间 16×16 + 2×2 patch merge，合计 32×32）；
- **位置编码**：Predict2.5 删掉绝对位置编码只留 3D RoPE，Cosmos 3 在此之上做 **3D MRoPE + 绝对时间调制（FPS modulation）**；
- **文本条件**：Predict2.5 用 Cosmos-Reason1 的多层激活拼接替代 T5，Cosmos 3 更彻底——Reasoner 塔本身就是文本编码器，外加一个 **结构化 JSON prompt upsampler**（Claude Opus 4.6 或 Reasoner 自己）；
- **控制生成**：Transfer2.5 为每种控制信号挂一个独立 ControlNet 分支，Cosmos 3 **把控制视频当作 clean DM token 直接拼进序列**，一个骨干原生支持 edge / blur / depth / segmentation / world-scenario-map 五种控制的任意组合；
- **数据基础设施**：1.0/2.5 是"每管线一张 Postgres 表 + CDC 同步到 Databricks"，Cosmos 3 换成统一的 **Lance 列式数据集（SILA）**，启动延迟 30–60 min → ~5 min，吞吐 10×。

难得的是论文给了一处**同表的前代对照**（Tab. 12）：PAIBench-G T2V 上 Cosmos-Predict2.5-2B 为 76.5，Cosmos3-Nano 为 79.4、Super 为 80.0；I2V 上 Predict2.5-2B 为 81.2、Cosmos3-Super 为 82.8；人评 Cosmos-HUE T2V 从 Predict2.5-14B 的 82.1 涨到 Cosmos3-Super 的 89.3；以自我为中心的 Human World Bench 更是从 38.7 跳到 71.9。也就是说这一代的提升在**人评与物理合理性**上远大于自动指标——这一点值得单独记一笔。

### 1.3 核心贡献清单

1. **MoT（Mixture-of-Transformers）双塔架构**：AR 与扩散两条通路共享注意力、不共享 LayerNorm/MLP，双双从预训练 VLM 初始化；
2. **把 action 提升为一等模态**：设计跨具身的统一动作表示（ego pose / effector pose / grasp state）与 domain-aware 投影层；
3. **三种动作模式统一于噪声分配**：前向动力学、逆动力学、策略只靠"哪些 token 干净、哪些加噪"区分；
4. **3D MRoPE + 绝对时间调制**：用 FPS modulation 把视频、音频（25 tok/s）、动作（采样率）对齐到同一物理时间轴；
5. **数据与工程全栈开源**：结构化 JSON captioner、五套 SDG 合成数据、SILA 数据平台、JointDataLoader、two-way flat attention 内核。

---

## 2. 模型架构

### 2.1 总体结构：一条序列、两个子序列、两套参数

Cosmos 3 的主干是一个**单栈 Transformer**，但每一层有两套参数。输入序列被切成前后两段：

- **AR 子序列（Reasoner）**：语言 token + ViT 编码的视觉 token，末尾以 `<EOS>`、`<BOG>` 收尾。走**因果自注意力**。
- **DM 子序列（Generator）**：VAE 编码的视觉 token + 音频 token + 动作 token。走**全双向注意力**，但 KV 是 `[K_AR; K_DM]` 的拼接。

推理时语言用 next-token prediction 吐出，其余模态用迭代去噪（flow matching）吐出。

### 2.2 模态编码器

| 模态 | 编码器 | 关键细节 |
|---|---|---|
| 视觉（理解） | ViT（16×16 patch）+ 两层 MLP 做 2×2 token merge | 与主干联合训练；用 DeepStack 聚合多层级特征，并插入文本形式的时间戳 |
| 视觉（生成） | **Wan2.2-TI2V-5B 的 VAE** | 时间压缩 4×、空间 32×32；**训练时冻结** |
| 音频 | 48 kHz 立体声，hop size 1920 → **25 token/s** | 冻结；线性投影进隐层 |
| 动作 | 线性投影（domain-aware，见 2.3） | 从零初始化，与主干联合优化 |

一个容易忽略的细节：每个非语言模态在进入 MoT 之前都会加上一个**可学习的模态专属 embedding**，让共享参数能区分 token 来源。

### 2.3 统一动作表示：跨具身的几何公分母

这是我认为全文最有设计感的一节。不同具身的控制空间天差地别（关节轨迹、方向盘命令、人体姿态、相机变换），作者不做"每种机器人一个头"，而是把任何动作拆成至多三个共享几何分量：

- **ego pose（9D）**：智能体主观测系的相对位姿 = 3D 平移 + 6D 旋转（Zhou et al. 2019 的过参数化表示，用 SVD 还原成 SO(3)）；
- **effector pose（9D）**：末端执行器位姿，同样 3D + 6D；
- **grasp state**：机械爪开合 1D，或人手五指指尖 3D 位置共 15D。

位姿一律用**相对伪动作** $\Delta T_t = T_{t-1}^{-1} T_t$，刻意绕开 PID 参数、底层驱动接口这类具身专属细节；grasp state 则相反，直接编码当前状态而不取时间差分。于是：

- 相机 / 自动驾驶车 = 9D（只有 ego）；
- 以自我为中心的人 = 57D（头 9 + 左右腕各 9 + 双手指尖各 15）；
- 单臂机器人 = 10D，双臂 = 20D，人形 = 29D。

![](assets/Cosmos%203%20-%20物理AI全模态统一世界模型/figures/02-unified-action.png)

> 图 3（Figure 3, p.8）：统一动作表示。异质的具身控制信号被压成由共享几何分量拼出的紧凑向量；长度差异由 **domain-aware 输入/输出投影**吸收——每个域有独立的 $W^{(k)}_{in}$、$W^{(k)}_{out}$，但共享同一套 MoT 主干。这样"换一个机器人"只需换一对投影矩阵，主干语义空间保持连续。

编码与解码就是两行线性映射，其中 $k$ 是域标识符：

$$z = W^{(k)}_{in} x + b^{(k)}_{in}$$

$$\hat{x} = W^{(k)}_{out} z + b^{(k)}_{out}$$

所有投影参数从零初始化并与主干联合优化；预测的 6D 旋转经 SVD 还原为 $3\times3$ 的 SO(3) 矩阵。训练前按维度统计归一化到约 $[-1, 1]$，以平衡不同具身的动作量级。

### 2.4 Token 排布与生成模式

所有任务共用同一套排布规则，只有三条：

1. AR token 一律排在 DM token 之前；
2. DM 内部，每个模态的**干净条件 token 排在加噪 token 之前**；
3. 条件段与扩散段内部都按 **视觉 → 音频 → 动作** 排序。

在这个约定下，各模式的序列就是"哪些 token 带波浪号"的差别：

$$S_{T2I} = [S_{AR}, \tilde{v}_1]$$

$$S_{T2V+Audio} = [S_{AR}, \tilde{v}_{1:N}, \tilde{s}]$$

$$S_{V2V} = [S_{AR}, v_{1:P}, \tilde{v}_{P+1:N}]$$

$$S_{Transfer} = [S_{AR}, v^{ctrl}_{1:N}, \tilde{v}_{1:N}]$$

其中 $S_{AR} \triangleq [l_1, \dots, l_n, \langle EOS\rangle, \langle BOG\rangle]$ 是所有模式共享的前缀。$P=1$ 即图生视频，$P>1$ 即视频续写。这个设计最漂亮的地方在于：**视频迁移和 V2V 是同一种排布，只是条件帧换成了控制视频**——Transfer 因此不需要任何 ControlNet 分支。

动作部分则用同一套"干净/加噪"开关定义三种模式：

![](assets/Cosmos%203%20-%20物理AI全模态统一世界模型/figures/03-action-modes.png)

> 图 4（Figure 4, p.10）：三种动作模式。动作 token $a_t$ 夹在相邻视频 token 之间，代表 $v_{t-1} \to v_t$ 的转移。**前向动力学**：动作干净、视频加噪（给定动作预测未来观测）；**逆动力学**：视频干净、动作加噪（从观测反推动作）；**策略**：视频与动作同时加噪（联合生成干预及其视觉后果）。三种模式共用同一套权重，切换成本为零。

在后文 §5.6 的消融里，作者进一步证明这三种模式**不是互斥的多任务而是互相增强的**（PushT 上联合训练让 ID 的 MSE 降 72%）。

### 2.5 MoT 双塔与双流联合注意力

每层 Transformer 有两条通路（图 5），都从同一个预训练 VLM 初始化，各自拥有 LayerNorm、注意力投影、FFN：

$$O_{AR} = \mathrm{Attn}_{causal}(Q_{AR}, K_{AR}, V_{AR})$$

$$O_{DM} = \mathrm{Attn}_{full}(Q_{DM}, [K_{AR}; K_{DM}], [V_{AR}; V_{DM}])$$

这个**非对称**是刻意的：DM token 可以看到全部文本提示与所有条件/扩散 token，从而保持时空一致性；而 AR token **永远不接收来自 DM 的更新**，保证预训练 VLM 的自回归语言能力不被扩散信号污染。

![](assets/Cosmos%203%20-%20物理AI全模态统一世界模型/figures/04-mot-architecture.png)

> 图 5（Figure 5, p.11）：MoT 架构与注意力掩码。左：单栈 Transformer，AR 子序列在前、DM 子序列在后，各自走独立 LayerNorm/MLP，只在共享自注意力处汇合。右：掩码矩阵——AR 区是下三角（因果），DM 区是全连接（双向），DM 对 AR 是"全 attend"，AR 对 DM 是零。这一张图基本就是 Cosmos 3 相对所有"统一理解+生成"方案的架构主张。

工程上作者为此专门写了 **two-way flat attention**：把两条通路的 KV 按样本粒度交错拼成 $[R_0, G_0, R_1, G_1, \dots]$，每个 DM query 只在自己的 $[R_i, G_i]$ 块内做双向注意力，从而两次变长 kernel launch 完成一层的异质掩码。相比 FlexAttention 基线，端到端训练吞吐提升 **22%**。

### 2.6 3D MRoPE 与绝对时间调制

这是另一个我认为比架构更"卡脖子"的设计。原版 3D MRoPE（Qwen3-VL）把隐层切成 (t, h, w) 三段，但时间分量只记录**离散 token 序号**。当视频、音频、动作以不同帧率/采样率同时生成时，序号相同并不代表物理时刻相同——这会直接导致音画不同步、动作与观测错位。

Cosmos 3 的解法是引入 **TPS（temporal steps per second）**，并定义单位时间步长：

$$\delta t = \frac{TPS_{base}}{TPS}$$

其中视频 $TPS = FPS / 4$（4 是 VAE 时间压缩比），音频 $TPS = 48000/1920 \approx 25$，动作 $TPS$ 就是其采样频率；基准取 $TPS_{base} = 24/4 = 6$（24 FPS 是最常见帧率）。于是 16 FPS 的片段每帧时间步是 1.5，30 FPS 是 0.8，**相同物理时长占据相同的位置编码区间**（图 6 右）。论文在附录 E.2 给出消融：相对无控制基线，纯文本注入 FPS 提升 +0.77，纯 MRoPE 调制提升 +1.12，两者叠加 +1.30（Composite = DOVER 画质 × 运动保真度）。

另一个工程细节值得一提：AR 与 DM 子序列之间插入了 **15000 的固定时间间隙**。原因是让 DM 从 AR 末尾的时间偏移直接续下去，会让最后一个语言 token 和第一帧视觉 token 的时间嵌入几乎相同，导致大尺寸模型出现**过饱和与棋盘格伪影**。这个"位置空间缓冲区"是纯工程试错产物，但很典型。

### 2.7 三个档位

| 档位 | 总参数 | Dense 主干 | 初始化 | 层数 / 隐层 / 头数(KV) / FFN |
|---|---|---|---|---|
| Cosmos3-Edge | 4B | 2B（从零训） | 自研，架构近似 Qwen3-1.7B（去 QK-Norm，FFN 用 ReLU²） | 28 / 2048 / 16 (8) / 9216 |
| Cosmos3-Nano | 16B | 8B | Qwen3-VL-8B | 36 / 4096 / 32 (8) / 12288 |
| Cosmos3-Super | 64B | 32B | Qwen3-VL-32B | 64 / 5120 / 64 (8) / 25600 |

**读参数时要小心**：64B 是**两塔合计**，单塔等价于一个 32B 的 VLM。跨论文比较时这个口径差 2×。Edge 的 LLM 用 Megatron 从零训了 15T token（Nemotron 语料，8K 上下文），再单独做 128K 长上下文扩展（90B token，RoPE base 1e8），最后用 Nemotron-Cascade-2 的 26M SFT 样本微调；它在 HMMT25 Feb 上拿到 76.3（Qwen3.5-2B 仅 22.9），但 MMLU-Pro 62.6 反而低于 Qwen3.5-2B 的 66.5——一个 2B 小模型被极端偏向数理训过后的典型画像。

---

## 3. 训练数据

### 3.1 Reasoner 数据：24.2M 样本

- **预训练 22.0M**：19.7M 来自 Nemotron Nano 2 数据集合 + 2.3M 自建（数学、视频、空间接地、指令跟随）。两阶段策展：语义去重（Qwen3-VL-Embedding-8B / PE-Core-G14-448 嵌入 → K-means 分簇 → 簇内余弦相似度 > 0.95 判重，去掉 4.23%）+ **AI judge 质量过滤**（Gemma-4-31B-it 按 Faithfulness / Completeness / Correctness 三维 1–5 打分）。
- **SFT 2.2M**：视频-文本占 50%，聚焦机器人、驾驶、智慧基建三域。此处 judge 阈值从预训练的 2 提到 5（保留率 78% → 46%），因为此时精度比覆盖更重要。

SFT 数据的构造方式值得单独看，它基本是"用仿真器和外部 VLM 造监督"：

- **空间**：2D/3D 接地（LocateAnything 为主）、真实场景空间 QA（图像指代表达、机器人放置点、跨视角对应）、**仿真器接地的具身空间推理**（答案直接由相机、物体位姿、深度、mask、可见性、可行域计算出来，再用程序化 + VLM 双 critic 校验）；
- **时间**：人工密集时间描述（平均 1.8 秒 / 14.2 词）、FoundationMotion 生成的四选一运动问答、相机运动模式标注；
- **物理合理性**：把本文 Cosmos-HUE 的 13.5K 人工标注 (视频, 问题, 答案) 三元组**回收成训练数据**，加上 VideoPhy-2 的 3.4K 物理评分视频；
- **驾驶**：1 万条人工 CoT + 约 110 万条自动标注决策链（用 SOTA VLM + 原始视频 + ego 轨迹 + 元动作生成）、Nexar 行车记录仪 24K 视频的时序事件定位、MADS 的度量级 3D 车辆接地；
- **机器人**：Action-CoT（Qwen3-VL-72B 出接地理由、Molmo-7B 定位、MolmoAct/DROID 出运动目标，串成"识别物体 → 定位 → 2D 路径点"的可检查推理链）、MimicGen 子任务切分、BEHAVIOR-1K 的 83K 长程规划、39.8 万轮手术机器人 VQA；
- **智慧基建**：仓库空间智能（Omniverse 合成，93K RGB-D + 873K QA）、密集行人检测（20.8 万图、560 万人工框，发布前做模糊脱敏）、CARLA 碰撞预测 + TAR 交通异常推理。

### 3.2 Generator 数据：图像 / 视频 / 音频 / 动作

**图像与视频**：从 78 亿原始图、30 亿原始视频中筛出 **7.67 亿图像 + 3.477 亿视频片段**。策展五步：原始采集（TransNetV2 场景切分、ffmpeg cropdetect 去黑边）→ 嵌入去重（图用 Qwen3-VL-Embedding-8B，视频用 Cosmos-Embed1-448p，各 2 万簇 cuML KMeans）→ 47 类分层语义标注与质量过滤（图像看美学分、去 collage/水印/白底/NSFW；视频看 DOVER 美学 + DOVER 技术 + VTSS 训练适宜度三项 0–9 分与约 100 个二值伪影标签）→ 标注 → 按分辨率时长分片。分辨率以 720p（图 26.8% / 视频 36.4%）和 480p（26.0% / 30.8%）为主。

**中训**再收紧一次：图像池 1560 万（60% 真实 + 36% 合成 + 4% 文字渲染），视频池 7470 万（46.0% 精选真实 + 43.9% 域专属 + 10.1% 困难样例如高速运动与精细操作）。

**音频**：预训练 1.389 亿条带可用音轨的片段（<30 秒的 6250 万条用 Qwen3-Omni-Captioner 补描述）；中训池 1880 万，策展原则极其实用——**只有嘴型同步的语音才保留，画外音要删，非乐器演奏的 BGM 要删**。管线是 SAM-Audio 人声分离 → SyncNet 唇同步打分（has_face 且 confidence ≥ 3.0）→ FireRedASR2S 估 speech/music 比例 → Qwen3-VL 判 is_music_instrument（保护乐器演奏视频）。

**动作**（中训引入）：**840 万条 episode、6.13 万小时**，四根支柱占比为以自我为中心的人类运动 67.4%（4.13 万小时，自有双手动操作数据集，头部 RGB + 双手 21 关键点 3D 姿态）、自动驾驶 16.3%（1 万小时，NVIDIA Hyperion 内部日志，刻意挖掘弯道与横纵向机动而非直线巡航）、机器人 8.7%（5.4 千小时，AgiBot / Franka Panda / Google Robot / WidowX-250 / UMI / UR 六个来源，**成功与失败 episode 都要**）、相机运动 7.5%（4.6 千小时，用 **ViPE + DepthAnything3** 从预训练视频里估位姿，再严格过滤抖动与异常内参）。

> 这里有一个可以直接连回本库其他笔记的点：Cosmos 3 的相机动作数据是用 **DepthAnything3**（我们前面对比过 VGGT / DA3 / WorldMirror 2.0）标注的，且在 §5.6 的自动驾驶逆动力学表里，Cosmos 3 正是以 **VGGT 和 DA3 为 baseline** 并大幅超越。详见后文 §5.6。

### 3.3 结构化 JSON caption：一个被低估的贡献

作者明确说：自由形式的 caption "往往精确但不完整——能把可见内容描述准确，却在复杂场景漏掉重要细节"。因此全阶段统一改用**预定义结构的 JSON schema**（主体、背景、光照、美学、摄影，视频再加物理变化、物体交互、人体动作、相机运动等时间字段），并用两个微调过的 Qwen3-VL-8B 分别充当图像 captioner 与视频 captioner。

配套还做了一个 caption 质量基准：precision 用 VLM 把生成 caption 拆成原子断言，逐条对着源媒体验证（惩罚幻觉）；recall 对着人工标注的原子断言清单比对（衡量完整性）。结论是结构化标注**在保持高 precision 的同时显著提高了 recall**。

这个 JSON schema 后来演变成了推理时的 "prompt contract"（§5.7），是 Cosmos 3 推理接口的核心。

### 3.4 五套开源合成数据（SDG）

针对长尾的 Physical AI 场景，作者用 Omniverse 造了五套数据并全部开源：SDG-PhyxSim（刚体碰撞、铰接物体、可变形材料、流体、光学效应）、SDG-RobotSim（6–8 种机器人具身的操控与移动）、SDG-DriveSim（常规 + 极端交通场景）、SDG-SynHuman（人体动力学、相机运动先验、多角色交互）、SDG-Warehouse（人-叉车安全交互）。附录 C.7 给了加入 SDG 的消融。

---

## 4. 训练方法

### 4.1 Reasoner 训练

- **初始化**：Edge 用自研 2B，Nano 用 Qwen3-VL-8B，Super 用 Qwen3-VL-32B。
- **一个反直觉的发现**：作者发现 Qwen 系列常用的"先冻住 VLM 只训 projector 的对齐阶段"**并不必要**，从头就全参数联合训练更好。
- **目标与配置**：next-token prediction；两 epoch，no-replacement 采样器均匀拼接所有数据集；序列上限 16k token（单样本 2048 图 token / 8192 视频 token）；**平方根归一化的 per-token 损失加权**（作者称显著提升下游分数与训练稳定性）；AdamW，LLM/projector 峰值 lr $5\times10^{-5}$、ViT $5\times10^{-6}$，cosine 衰减到峰值的 0.1 倍，10% warmup；weight decay 0.05，梯度裁剪 1.0。
- **SFT**：8200 步、全局 batch 512、lr $1\times10^{-5}$/$10^{-6}$、1000 步 warmup、$(\beta_1,\beta_2)=(0.9, 0.95)$、weight decay 0.1。两个关键手法：**重要性感知采样**（每个数据集按重要性/质量/规模分配固定采样预算）和 **1:4 的预训练:SFT 混合**（防止下游特化侵蚀通用能力），外加 80 万条轻量指令跟随数据。

### 4.2 Generator 预训练

**目标**：全模态统一的 rectified flow matching。噪声潜变量 $x_\sigma = \sigma\epsilon + (1-\sigma)x_0$，单个去噪器 $v_\theta(x_\sigma, \sigma, c)$ 预测恒定速度 $v^* = \epsilon - x_0$，用 masked MSE 训练（条件 token 不计入损失）。三个细节：

- **逐模态独立采样噪声水平 $\sigma$**：图像/音频/动作走 logit-normal，视频走 **mode sampling**（作者说画质更好）；
- **rectified-flow shift 重参数化** $\sigma = s\bar{t}/(1 + (s-1)\bar{t})$，其中 $\bar{t} = 1-t$，$s \ge 1$ 把边际推向高噪声；
- **分辨率自适应 shift**：256p → $s=1$，480p → $s=3$，720p → $s=5$（中训提高到 3/5/10）。

**多分辨率 + 序列打包**：三档分辨率 × 五种宽高比 × 可变帧数（256p/480p 最多 400 帧，720p 300 帧），不同分辨率的序列打包进**固定的 74,000 token 上下文**以避免重编译与 padding。训练模式配比为 T2I 20% / T2V 56% / I2V 16% / V2V 8%（文档另一处写 video 内部 T2V 70% / I2V 20% / V2V 10%，两者是外层与内层的口径差异）。

**优化**：**只更新生成侧参数，Reasoner 塔冻结**（这是 Cosmos 3 与 Predict2.5 一致的关键决策——保住语言与理解能力）。FusedAdamW，lr $1\times10^{-4}$，$(\beta_1,\beta_2)=(0.9, 0.99)$，weight decay 0.05，线性衰减到 0.30×，文本 dropout 10% 用于 CFG。

**算力**：Nano 训 **31.05T token / 1024 张 GB200**；Super 训 **17.86T token / 2048 张 GB200**。注意 Super 的 token 数反而更少——大模型的 token 预算被算力摊薄了。

### 4.3 中训：动作与迁移在这里进场

中训有两个目标：**域特化**（补上罕见但关键的 Physical AI 动态）和**多模态整合**（从图/视频/音频扩展到动作与控制）。混合配比（Tab. 6）：

| 训练流 | 模式 | 占比 |
|---|---|---|
| 图像 | T2I | 10% |
| 视频 | T2V / I2V / V2V | 32% |
| 视频+音频 | T2(V+A) / I2(V+A) / V2(V+A) | 8% |
| **动作** | FD / ID / policy | **25%** |
| 通用迁移 | edge / blur / depth / seg | 20% |
| 驾驶迁移 | world-scenario map | 5% |

动作损失被**放大 10×**（归一化动作向量的 per-element MSE 天然偏小）。动作沿用视觉的噪声 schedule。优化 lr $1\times10^{-4}$，LambdaLinear（起始因子 0.4，周期 100,000）。算力：Nano 2.4T token / 1024 GB200，Super 1.9T token / 2048 GB200。

### 4.4 后训练三条线

**(1) Cosmos3-Super-Text2Image**：两阶段 SFT。Stage 1 用 45% 真实 / 40% 合成 / 15% 文字渲染训 20k 步（lr $1\times10^{-4}$，2k warmup）；Stage 2 用 47 万对超高质图文再训 2k 步。两阶段都固定 70k 上下文、只用 >720p 的图。

**(2) Cosmos3-Super-Image2Video**：480p / 189 帧（24 fps 约 8 秒）；数据是"过滤后的预训练数据 + agentic 工作流按模型弱点定向检索的样本 + 1000 条人工精选视频 + 约 2 万条合成视频"（合成占约 6% token）；**混入 20% T2I 图像 token** 以保住语义对齐；10k 步，lr $1\times10^{-5}$，约 500 亿 token。

**(3) Cosmos3-Nano-Policy-DROID**（真机策略，最有说服力的一条）：从 Nano 中训权重恢复，**动作编码器/解码 MLP/动作 embedding 全部重新初始化**，动作相关参数给 5× 学习率。输入 = 当前本体状态 + 三视图（腕视 360×640 在上，两个外视 180×320 左右并排在下，拼成 540×640 canvas）；输出 = **32 步未来绝对关节位置**（15 Hz），同时把 RGB 视频帧当辅助输出一起预测。推理时只用 **4 步扩散**（shift 5）、CFG 3 + CFG 并行、**跳过视频 latent 解码**，从而能部署在 **2 张 RTX Pro 6000** 上；关节位置控制器用 Franky 执行。

### 4.5 基础设施：三处值得记的工程创新

**SILA 数据平台**：用统一的 Lance 列式表取代"每管线一张 Postgres 表"，每个策展信号（caption、tag、质量分、嵌入）是一个带类型的列。分布式协调下沉到 **Lance fragment 级别**（worker 租约 + 心跳，心跳停止即自动回收），恢复单元与可见性单元解耦；Ray 分阶段执行并施加反压；节点本地 vLLM 端点避免中心推理瓶颈。结果：作业启动 30–60 min → ~5 min，吞吐 **10×**，峰值每天处理十亿级行标注。嵌入直接在 Lance 上建 LanceDB IVF_PQ 索引（64K 分区 + PQ 压缩，4096 维，覆盖数百亿行），检索与元数据不再需要双系统同步。

**JointDataLoader**：多模态训练里单样本 token 数跨越两个数量级（一段 720p 两秒视频比几十条短文本 caption 还多），传统"每 rank 相同样本数"会带来 padding 浪费、负载不均、NCCL 超时三件套。四个解法：① **token 预算打包**（$T_{max}$ 而非样本数定界）；② 多流 joint loader；③ **rank 同步的流选择**（全局种子按迭代下标选流，保证所有 rank 同一步处理同模态同分辨率，吞吐 +54%）；④ **look-ahead 打包**（装不下的样本挪进 lookaside buffer，继续往下扫更小的样本填缝，迭代结束按原顺序放回队首，有效序列长度 +8%）。另外还有冷启动预热：构造期先从每个流取一个 batch 再放分布式 barrier，避免第一步与 NCCL 竞争触发 watchdog 超时。

---

## 5. 实验结果

### 5.1 Reasoner：48 个基准，四组平均

| 能力组 | 基准数 | Super | Nano | Qwen3-VL-32B | Gemini 3.1 Pro† |
|---|---|---|---|---|---|
| 通用 | 19 | **73.7** | 69.6 | 72.8 | 77.5 |
| 机器人 | 17 | **57.8** | 55.1 | 52.6 | 58.2 |
| 智慧基建 | 9 | **62.6** | 61.0 | 56.1 | 58.6 |
| 驾驶 | 3 | **79.3** | 76.0 | 40.7 | 47.2 |

读这张表要注意三点：

1. **通用能力仍明显落后 Gemini 3.1 Pro**（73.7 vs 77.5），且不是平均落后——MMMU-Pro 48.1 vs 70.6、HallusionBench 49.0 vs 64.2、LogicVista 55.9 vs 81.9、IFBench 37.0 vs 42.5，差距集中在专家级多模态推理、抗幻觉与指令跟随；
2. **但在 Physical AI 三个域反超闭源**：驾驶 79.3 vs 47.2（AVSpecialStopBehavior 81.6 vs 16.3，几乎碾压），智慧基建 62.6 vs 58.6；
3. **VideoPhy2 物理合理性 47.4 是所有模型最高**（Gemini 仅 28.7），说明域数据确实转化成了物理常识。

另外一个值得注意的现象：Nano 在 HealthSurgiBench（44.5 → 56.1）和 VSI-Bench（60.9 → 63.0）上**反超 Super**，作者未解释。这类尺寸倒挂在文中还有几处，通常是小模型在特定分布上过拟合或大模型数据配比被稀释的信号。

### 5.2 图像生成

| 模型 | UniGenBench All | CVTG-500L GNED/PNED | CVTG-102ch GNED/PNED | Aesv2 | HPSv3 |
|---|---|---|---|---|---|
| Cosmos3-Super-Text2Image | **91.36** | **80.88 / 89.08** | 32.02 / 41.22 | 5.91 | 11.60 |
| Gemini 3 Pro Image† | 90.69 | 59.24 / 71.79 | **46.00 / 76.40** | 5.70 | 11.78 |
| Qwen-Image-2512 | 84.25 | 79.68 / **90.86** | 46.33 / 71.26 | 5.92 | 11.03 |

英文场景文字渲染（CVTG-500L）拿下第一，UniGenBench 总分第一，Artificial Analysis 众投榜开源第一 / 总榜第四。但**中文文字渲染是明显短板**（32.02 vs Gemini 46.00、Qwen-Image 46.33），这是训练数据以英文为主的直接后果。另外注意美学分（5.91）和 HPSv3（11.60）与各家差距很小——这类指标同样接近饱和。

### 5.3 视频生成

| 基准 | Cosmos3-Super | 最强开源对照 | 最强闭源 |
|---|---|---|---|
| PAIBench-G T2V | **80.0**（Domain 86.8 / Quality 73.1） | Wan2.2-A14B 78.0 | Veo-3.1 79.1 |
| PAIBench-G I2V | **82.8**（87.3 / 78.2） | HunyuanVideo-1.5 81.7 | Veo-3.1 82.6 |
| RBench I2V | 58.1% | Nano 58.4% | Wan 2.6 60.7% |
| Physics-IQ I2V | **43.8**（+WMReward BoN 48.9） | Wan2.2-A14B 38.3 | Sora2 42.3（BoN 46.4） |
| Physics-IQ V2V | **59.7**（BoN 63.4） | Magi-1 56.0（BoN 62.6） | — |
| Cosmos-HUE T2V（人评） | **89.3** | Wan2.2-A14B 88.2 | Veo-3.1 91.3 |
| Cosmos-HUE I2V（人评） | **89.6** | Wan2.2-A14B 88.4 | Veo-3.1 89.7 |
| Human World Bench | **71.9** | Wan2.2-A14B 60.7 | Veo-3.1 67.8 |

三个必须点出的观察：

- **自动指标已饱和**：PAIBench-G 上 Super 与 Nano 只差 0.6 分（80.0 vs 79.4），所有模型的 Quality 分项挤在 72.0–73.2 的 1.2 分带里。作者在附录里自己承认"同一批模型人评跨约 10 分，自动指标只跨约 4 分"。这与我们在 [[Cosmos-Predict 2.5 - 视频基础模型驱动的物理世界模拟]] 里看到的"14B 与 2B 自动指标完全相同"是同一个病。
- **Physics-IQ 拉开差距要付出推理时算力**：I2V 直出 43.8，加 WMReward best-of-N 重排才到 48.9；V2V 直出 59.7 → BoN 63.4。也就是说"物理合理性 SOTA"里有一部分是 inference-time alignment 的贡献，不是纯权重能力。
- **HWB 的跃升最惊人**：Predict2.5-14B 只有 38.7，Cosmos3-Super 到 71.9，超过 Veo-3.1 的 67.8。这是以自我为中心人手生成这一细分能力上的质变，大概率来自占动作数据 67.4% 的 egocentric 支柱。

**Cosmos-HUE 人评协议**本身也值得记：它把主观 Likert 打分换成**原子二值验证**——三层 VLM 流水线（Domain Strategist 分域 → Scene Parser 出场景清单 → Auditor 出原子问题，全部跑 GPT 5.2）为每个视频生成最多 20 个 Yes/No/Unclear 问题，两名标注员独立评、分歧升级给 QC。关键设计是 **"Unclear 记为 No"**（不进分子但留在分母），所以模型不能靠"不生成被问到的动作"来骗分。100 个 prompt × 5 seed = 每模型 500 视频，真实视频作为上界参考是 93.6/94.4。

### 5.4 音视频生成

Cosmos-SoundBench（144 条 FoleyBench prompt，只测非语音）：

| 模型 | AVQ | SAV | SA | AVAlign | Visual Sup. | PQ |
|---|---|---|---|---|---|---|
| Cosmos3-Nano | **7.34** | **8.35** | **8.33** | **8.16** | 9.10 | 6.32 |
| Cosmos3-Super | 7.31 | 8.34 | 8.30 | 8.14 | **9.18** | 6.28 |
| Seedance-1.5-Pro† | **7.64** | 8.21 | 8.22 | 8.06 | 8.61 | **7.06** |
| Veo-3.1† | 7.45 | 8.21 | 8.21 | 8.01 | 8.85 | 6.68 |

结论很清晰：**语义与音画对齐（SAV/SA/AVAlign）Cosmos 3 全项第一，但感知音质（PQ）明显垫底**（6.28–6.32 vs Seedance 7.06）。AVQ 是 $0.5\times SAV + 0.5\times PQ$，所以总分被 PQ 拖到闭源之后。作者的定性证据（图 20）是锤击瞬间的频谱瞬变与视觉接触帧严格对齐。

### 5.5 视频迁移：一个骨干打四个 ControlNet

PAIBench-C 单控制结果：Cosmos3-Nano 拿到 DOVER 10.39（vs Transfer2.5 的 9.49）与 Seg mIoU 0.72（vs 0.68）第一；Cosmos3-Super 拿到 Edge F1 0.50（vs 0.45）与 Depth si-RMSE 0.58（vs 0.68）第一。Blur SSIM 三家都在 0.90–0.91，指标已饱和。驾驶域 AVBench-C 上 ego drift 从 0.008 降到 0.003，人工画质从 2.59 提到 2.86，车道线保真度基本持平（2.45–2.50）。

**这是全文最有产品意义的一条结论**：四个专用 ControlNet 分支塌缩进一个统一骨干后，不仅没退化，反而在每个模态上都由某一个档位超过了专用适配器基线。也就是说"控制条件"可以被统一建模成"序列里一段干净 token"。

### 5.6 动作生成：与 VGGT / DA3 的直接对照

| 任务 | 指标 | Cosmos3 (MT-init) | 专用/通用 baseline |
|---|---|---|---|
| 自动驾驶（ID） | RRE / RTE / ATE | **0.211° / 0.014 m / 0.98 m**（Nano） | DA3: 0.312° / 0.354 m / 9.29 m；VGGT: 0.596° / 0.768 m / 23.46 m |
| 相机运动（FD） | RRE / RTE / ATE | **0.142° / 0.026 m / 0.99 m**（Super） | Lingbot-World 0.299°/0.057 m/2.88 m；HY-World1.5 0.377°/0.042 m/1.39 m |
| 机器人（FD） | PSNR | **26.04 dB**（Super） | Ctrl-World 22.99 dB |
| 自我中心（FD） | PSNR | **16.19 dB**（Super） | LOME 9.36 dB |
| 机器人（Policy） | RoboLab 成功率 | **39.7%**（specific） | π0.5 28.1%；DreamZero 23.9% |

这张表有几个信息量极大的点：

1. **VGGT 和 DA3 在度量尺度轨迹上被碾压**（ATE 23.46 m / 9.29 m vs 0.98 m）。原因不是架构劣势，而是**两者输出归一化尺度、且没有针对驾驶域做后训练**；Cosmos 3 用内部驾驶日志 + 域专属数据直接预测度量尺度位姿。这印证了我们在对比 VGGT/DA3/WorldMirror 时的判断——前馈几何模型是通用基座，一旦任务被限定到具体域且有数据，专用模型会反超。
2. **PT-init vs MT-init 是全文最重要的一条因果证据**（设计得很干净：同尺寸、同数据、同算力，只换初始化）：机器人 FD 从 Nano PT 的 23.24 dB 提到 MT 的 25.52 dB；LIBERO-10 新具身适配在 500 步时 MT 24.6% vs PT **0.0%**，2000 步时 97.4% vs 95.2%。也就是说**统一动作中训确实学到了可迁移的动作先验**，而不只是多看了点数据。
3. **动作域协同矩阵**给了"哪些域该一起训"的实证（图 28）：相机运动（单域最弱之一）从 AV 数据拿到 +0.86 FD PSNR，从 Google Robot / Franka Single / Franka Dual 也分别拿到 +0.79 / +0.73 / +0.48——**受益于"泛化的动作条件视觉先验"而非同类具身**；WidowX-250 从 Google Robot 获得 +1.39 FD PSNR 与 +2.29 policy PSNR；而 Franka Panda Single/Dual 在本文里只有 23 / 4 小时数据，单域训就快速饱和，混合收益混杂（作者自己点明了这点）。

![](assets/Cosmos%203%20-%20物理AI全模态统一世界模型/figures/05-action-synergy.png)

> 图 28（Figure 28, p.71）：动作域协同矩阵。行是评测域、列是加入的共训域，对角线是单域基线（2000 步），非对角线是 50/50 混合（4000 步，保证每域训练预算相同）。绿色为正迁移。四块分别是 FD PSNR、ID MSE、policy PSNR、policy MSE。这张图的方法论价值大于数值价值——它把"语言域迁移矩阵"的思路搬到了动作域，是可复用的混合设计工具。

4. **人类自我中心数据是机器人的有效预热**：先在 egocentric 上 warmup 再适配 AgiBot，AgiBot FD PSNR 在每个测量步都更高（5K 步 +0.94，后续稳定 +1.3~1.6），而反过来 AgiBot 给 egocentric 只有 +0.03。这是一个很实用的"用海量人类视频给机器人打底"的正例。
5. **策略模式下的视频-动作一致性**：在 RoboLab（held-out 环境）执行预测的动作块，再与模型自己预测的视频比 PSNR——左视 23.19 dB，腕视 17.33 dB（腕视更难，因为相机随末端移动、要推断新暴露的内容）。

### 5.7 推理接口：结构化 JSON 就是控制面

Cosmos 3 的推理不是"给一句话"，而是先由一个 upsampler（Claude Opus 4.6 或 Cosmos 3 Reasoner 自己）把用户意图翻译成**schema 约束的 JSON 场景程序**：先描述场景布局与世界状态，再指定事件的时间推进，最后补音频描述，并显式带上分辨率、宽高比、时长、FPS 等控制字段。prompt contract 分四块：instructions（任务级行为，如"只输出一个 fenced JSON"）、description（原始语义请求 + 条件图/视频）、task_constraints（字段顺序、时序格式、时长边界、I2V 首帧在 $t=0$ 必须匹配等）、output_json_template（schema 形状）。

默认采样配置（Tab. 21）：音视频 50 步 / guidance 6 / shift 10；T2I 特化 50 步 / guidance 4 / shift 3；I2V 特化 shift 5；FD/ID 50 步 / guidance 1 / shift 5；**policy 只需 4 步 / guidance 3**；迁移用双权重 CFG（文本 guidance 3 + 控制 guidance 1.5，每步跑三次去噪器：全条件、仅提示、控制+负提示，分别外推）。负提示逐模型逐模式用自动基准迭代搜索，action 模式与 T2I 特化版发现**空字符串最好**。

---

## 6. 局限与评价

### 6.1 论文自己承认或数据本身暴露的短板

1. **自动指标饱和，量化证据弱于定性结论**。PAIBench-G T2V 上 Super 与 Nano 只差 0.6 分；所有模型 Quality 分项挤在 1.2 分带内；PAIBench-C 的 Blur SSIM 三家都在上界附近。作者在附录明说人评跨 10 分、自动指标只跨 4 分。这意味着"SOTA"更多由人评与榜单支撑。
2. **通用推理仍落后闭源一档**。MMMU-Pro 48.1 vs 70.6、HallusionBench 49.0 vs 64.2、LogicVista 55.9 vs 81.9。Cosmos 3 的定位是 Physical AI 专用骨干，不是通用 VLM 竞争者，但若指望它做开放域多模态推理会失望。
3. **中文文字渲染是硬伤**（CVTG-102ch GNED 32.02，不到 Gemini/Qwen-Image 的七成）。
4. **音频保真度垫底**（PQ 6.28 vs Seedance 7.06），只赢了语义对齐维度。
5. **动作数据严重不平衡**：egocentric 占 67.4% 且是**专有数据集**（不可复现），机器人只占 8.7%（5.4 千小时）。协同矩阵里 Franka Panda 单/双臂只有 23 / 4 小时，作者自陈"小域单域训很快饱和，加更大域收益混杂"。
6. **评测管线依赖闭源模型**。prompt upsampler 用 Claude Opus 4.6/4.7，UniGenBench 的 judge 是 Gemini 3.1 Pro，Cosmos-HUE 的问题生成跑 GPT 5.2，AVQ 的 judge 是 Claude + Gemini + GPT 集成。这让报告的分数难以被第三方按同样口径复现。
7. **复现门槛**：Nano 31.05T token / 1024 张 GB200（预训练）+ 2.4T（中训）；Super 17.86T / 2048 张 + 1.9T。论文未给完整 wall-clock，无法换算 GPU-hours 或成本。对学术界来说这是一个"只能读、不能跑"的模型。
8. **机器人策略的绝对成功率仍然很低**：RoboLab 在 specific 指令下 39.7%，复杂任务只有 29.4%（vague 指令下 4.1%）。榜首是相对榜首，不是可用级。
9. **报告首发时 Edge 缺席、当前已补发**：v4 报告写的是 Nano/Super 首发、Edge 后续发布；截至 2026-10-31，官方已提供 Cosmos3-Edge 与 Edge-Policy-DROID 权重。报告中的 Edge 消融仍显示小档位在部分 Physical AI 基准明显弱于 Nano/Super（如 VANTAGE-SOT 4.8 vs 62.2），因此“已发布”不等于能力差距消失。
10. **Nano 在若干基准上反超 Super**（HealthSurgiBench 56.1 vs 44.5、VSI-Bench 63.0 vs 60.9、BlinkDepth 92.7 vs 91.9），文中未讨论。可能是 Super 的数据配比被更多域稀释，也可能是评测噪声，但作者没给解释。

### 6.2 我的判断

**最值得借鉴的不是架构，是三件事的方法论。**

第一，**"模式 = 噪声分配"这个抽象**。把 T2I/T2V/I2V/V2V/Transfer/FD/ID/Policy 八种任务压成"哪些 token 干净、哪些加噪"这一件事，是从 Predict2.5 的"mask token + frame replacement"演化出来的更彻底的形式化。它带来的直接收益是：**加一个新任务不需要加参数，只需要定义一种 token 排布**。对做多任务统一模型的人来说，这是比 MoT 双塔本身更可迁移的思想。

第二，**"用可控的初始化对照证明中训有效"**。PT-init vs MT-init 这套设计（固定模型尺寸、数据、算力，只换是否经过动作中训）把"多域动作数据到底有没有用"从玄学变成了可测量的问题；LIBERO-10 上 500 步 24.6% vs 0.0% 的差距尤其有说服力。再叠加协同矩阵，等于给"多域混合训练"提供了一套可复用的配比设计工具。

第三，**把工程细节写进论文**。AR-DM 之间 15000 的时间间隙、rank 同步流选择（+54% 吞吐）、look-ahead 打包（+8%）、two-way flat attention（+22%）、SILA 的 fragment 租约（启动 30–60 min → 5 min，吞吐 10×）——这些在一般论文里会被压缩成一句话，Cosmos 3 给了完整机制与量化收益。对做大规模多模态训练的人来说，第 5 章的工程价值不低于第 2 章。

**放在我们的研究语境里**，Cosmos 3 有两点直接相关的启示：一是它的相机运动数据标注依赖 ViPE + DepthAnything3，而自动驾驶逆动力学的 baseline 就是 VGGT 与 DA3——说明**前馈几何模型正在成为世界模型数据管线的标准零件**（生成数据 → 标注 → 再训练生成模型），这是一个正在固化的分工；二是它仍然在像素空间做预测，没有显式状态与物理求解器，"物理合理性"完全来自数据分布与人工/模型评判，因此长程一致性、可验证的物理约束、以及与仿真器的闭环对齐都还是开放问题。

---

## 7. 与本库其他笔记的关系

- **上游平台与二代**：[[Cosmos WFM - 物理AI世界基础模型平台]]（Predict/Transfer/Reason 三族的平台定义）、[[Cosmos-Predict 2.5 - 视频基础模型驱动的物理世界模拟]]（本代直接的架构与数据前身，PAIBench-G 表里有难得的前后同表对照）
- **数据管线的零件**：相机位姿标注用 DepthAnything3 与 ViPE；自动驾驶逆动力学以 VGGT、DepthAnything3 为 baseline（属 3D-Reconstruction / Foundations 域，本库尚未建对应笔记）
- **对照的世界动作模型**：[[Dyna-2 - 世界动作模型百万小时缩放定律]]、[[Fast-WAM - 无测试时未来想象的世界动作模型]]（可对照看动作中训与策略模式的差异）
- **评测**：Cosmos-HUE 的原子二值人评协议可与 [[Orbia - 视频世界模型三维一致性评测]] 对照阅读——一个用人类事实核验，一个用几何一致性，是互补的两条评测路线

---

## 附：核验说明

- 全部数据、公式与实验数值均取自本地官方 PDF（`assets/.../sources/paper.pdf`，arXiv 2606.02800v4，139 页）正文与附录，未使用二手解读。
- 五张配图均从该 PDF 按原始 Figure 页面完整裁剪并保留图注，路径为相对本笔记的 `assets/Cosmos 3 - 物理AI全模态统一世界模型/figures/`。
- 版本与许可信息以 arXiv abs 页（v1 2026-06-01、v4 2026-06-23，DOI 10.48550/arXiv.2606.02800）与摘要中的 OpenMDW-1.1 声明为准。
- 仓库现状备注（2026-10-31 核）：`NVIDIA/Cosmos` 是 landing repo；Cosmos 3 专属模型、训练、推理、sequence packing 与动作数据代码已迁入并公开在 `NVIDIA/cosmos-framework@50e883466491abacdcb69843e1e026479d927a95`。本笔记的源码结论不再引用旧 Cosmos 2.x 实现。Nano 权重固定为 `nvidia/Cosmos3-Nano@e59a53c25979a090fa8706c9acc0c254a6e89b92`。
