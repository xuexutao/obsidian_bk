---
paper: "GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives"
arxiv: "https://arxiv.org/abs/2609.39601"
source: "https://groundingpi.github.io/"
code: "https://github.com/groundingpi/GroundingPI"
code_commit: "090932430bbbeea004cfa461767ede7f563c0f3a"
code_revision: "08d397c8769f7f10b96ff6d56ee6f56acb04181f"
created: 2026-10-27
updated: 2026-10-27
tags:
  - paper
  - VLA
  - visual-grounding
  - physical-intelligence
  - robot-learning
  - autonomous-driving
  - multimodal
rating: 9/10
license: "Apache-2.0 + Kimi K3 派生条件"
code_status: "官方代码、推理、SFT、权重和34项评测已开源，但多阶段预训练、数据引擎、GroundingPI专属GRPO与下游GroundingPI policy未完整开源"
---
# GroundingPI：以视觉基元构建面向物理智能的感知底座——论文与代码精读
## 基本信息
| 项目 | 内容 |
|---|---|
| 论文 | GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives |
| arXiv | 2609.39601v2，2026-10-16 |
| 作者 | Qize Yu、Lianrui Fan、Boyu Chen、Jiaqi Liang 等 |
| 核心任务 | 统一 box、point、OCR、GUI、layout、visual prompting 等空间感知任务，并评估其向机器人操作和自动驾驶的迁移 |
| 模型组成 | MoonViT-V2 + MLP projector + Qwen3-4B decoder |
| 完整参数量 | 4,843,664,384，约 4.844B；“4B”是语言骨干家族名称与近似口径 |
| 训练暴露量 | $500.41\mathrm{B}+221.17\mathrm{B}=721.58\mathrm{B}$ token |
| 感知评测 | 34 个 benchmark entry、11 类 capability、论文正文称 44 个 baseline |
| 下游评测 | nuScenes open-loop、RoboTwin 2.0、RoboCasa-GR1，后两者均为仿真 |
| 官方仓库版本 | `090932430bbbeea004cfa461767ede7f563c0f3a` |
| HF revision | `08d397c8769f7f10b96ff6d56ee6f56acb04181f` |
| 许可证 | **Apache-2.0 + Kimi K3 派生条件**，不能简写为纯 Apache-2.0 |
| 开源状态 | **官方代码、推理、SFT、权重和34项评测已开源，但多阶段预训练、数据引擎、GroundingPI专属GRPO与下游GroundingPI policy未完整开源** |
本文的核心判断是：GroundingPI 最有价值的地方，不只是把 grounding 榜单再推高一截，而是把“先学会精确地看见哪里，再用稀缺动作数据学习怎么做”变成了可检验的物理智能研究路线。
但这条路线当前仍主要由相关性和受控迁移实验支持，尚未形成“grounding 监督必然导致更强闭环控制”的严格因果结论。
全文没有实际运行模型、训练或复现实验；所有数字来自论文、官方仓库、官方配置与权重索引的静态核对。
## 1. 一页结论与证据分级
### 1.1 论文试图回答什么
GroundingPI 讨论的不是“VLM 能否描述场景”，而是“有限动作监督之前，基础模型是否已经具备足够精细、可调用的空间感知”。
通用 VLM 往往能回答“图里有什么”，却可能在密集实例、微小目标、严格边界、文字区域和 GUI 点击位置上失败。
对物理智能而言，这类失败会把两种学习负担同时压给稀缺动作数据：
- 先从动作示范中重新学“目标在哪里”；
- 再学习“如何到达、抓取、放置或跟踪该目标”。
GroundingPI 的回答是把 box 和 point 作为视觉基元，把语义标签、协议标记和量化坐标放入同一词表，以自回归序列统一表达多种感知任务。
然后，它把这套基础模型接到统一动作专家上，观察更强 grounding 是否会转化为更好的机器人操作和驾驶轨迹预测。
### 1.2 四级证据结论
#### 1.2.1 强结论
**强结论 1：GroundingPI 在论文定义的 11 类感知能力等权平均上取得 73.68。**
这一数字高于 GPT-6 Astra 的 71.54，也显著高于同尺寸通用 VLM。
73.68 不是 34 个 benchmark 的样本级 micro average，也不是简单把主表全部列相加后除以 34。
论文附录明确说明，headline Avg 是 11 个独立 capability score 的算术平均。
**强结论 2：量化坐标协议显著降低结构化框输出的 token 成本。**
GroundingPI 在 COCO/Dense200 上分别为 7.6/5.1 tokens per box。
论文引用的 SEED1.5-VL 外部统计为 148.8/74.5 tokens per box。
该结论只涉及序列化效率，不等价于端到端推理延迟提升 19.6 倍或 14.6 倍。
**强结论 3：在论文固定的下游动作架构与训练预算内，GroundingPI 的表征迁移很强。**
它在 RoboTwin Full 达到 66.20%，Clean2Random 达到 17.60%。
它在 RoboCasa-GR1 达到 ID 37.75%、Type OOD 28.94%、Appearance OOD 37.78%、Container OOD 37.86%。
它在 nuScenes open-loop 上达到平均 L2 0.296 m。
这些是“给定论文协议下”的强结论，而不是现实机器人部署结论。
#### 1.2.2 中等结论
**中等结论 1：精确空间感知有助于动作学习和 OOD 泛化。**
依据是固定 ActionDiT、动作数据、优化预算、动作表示和推理流程后，只改变预训练基础模型，GroundingPI 在五个六个操作设置中第一，并在四个 OOD 设置中第一。
但不同 backbone 仍同时改变了模型规模、编码器、连接器、预训练目标和特征可访问性。
因此实验控制了下游变量，却没有把“grounding 监督”从所有上游差异中单独隔离。
**中等结论 2：grounding 预训练规模与 OOD 操作能力共同上升。**
四个暴露点从 88.4B 增至 221B 时，Grounding Avg 从 61.69 增至 73.68，OOD Avg 从 26.59 增至 33.38。
不过只有四个点，且没有重复种子不确定性；最多能说“在这四个配置上呈上升趋势”。
#### 1.2.3 弱结论
**弱结论 1：OCR 可能是空间感知学习的催化剂。**
去掉含 OCR/layout/GUI 的 F 组后，RoboCasa ID/OOD 分别下降 5.33/3.63 个百分点。
但 F 单独训练表现很差，且 F 同时包含三个任务族，无法把增益归因于 OCR transcription。
论文自己也把该机制称为 hypothesis。
**弱结论 2：感知原生基础模型可能适合 System 1 执行层。**
论文提出“强推理模型负责规划、感知原生模型负责快速执行”的双系统设想。
现有实验只比较基础表征迁移，没有真正构建并测试该双系统。
#### 1.2.4 尚未证明
尚未证明 grounding 是动作迁移收益的唯一因果来源。
尚未证明模型在真实机器人、真实车辆或闭环驾驶中有效。
尚未证明量化 token 协议带来实时闭环延迟优势。
尚未证明 OCR transcription 本身而非 OCR 几何框、数据量或联合训练带来迁移收益。
尚未证明 73.68 的 11 类等权聚合是物理智能最合理的能力权重。
### 1.3 论文全景图
![](assets/GroundingPI%20-%20视觉基元物理智能感知底座/figures/01-teaser.png)
**图解：** 上半部分把 grounding、OCR、dense point、affordance pointing、layout、GUI、robot referring 等任务放在统一空间接口下；下半部分把 11 类感知评测与驾驶、机器人动作迁移连接起来。作者的叙事顺序是“Basic Perception first，Action Learning next”，即先用大规模感知监督建立可复用的 where，再让下游学习 how。
**图号与页码：** Figure 1，PDF Page 1。
**官方来源：** 论文 PDF；官方仓库 `docs/assets/teaser.png`；项目页。
## 2. 研究问题、定位与贡献
### 2.1 从语义理解到可执行空间接口
传统 VLM 的输出中心通常是自然语言。
机器人和驾驶真正需要的是可落到图像位置、物体实例和局部区域上的表示。
GroundingPI 不直接声称自己是完整 policy，而是把自身定位为 perceptual foundation。
这一定位很重要，因为它避免把感知、规划和控制都塞进同一指标中。
其输出 primitive 主要有两类：
- box：表达目标的二维范围；
- point：表达点击点、物体点、affordance 或 ordered trajectory coordinate。
同一个协议还允许：
- 标签到多个实例的映射；
- referring expression 到唯一或多个区域；
- OCR transcription 到文字框；
- `None` 表示显式询问但不存在的目标；
- visual prompt 用输入参考框指定“找类似对象”。
### 2.2 相比检测器和通用 VLM 的差异
闭集检测器依赖固定类别表，难以统一 referring、OCR、GUI 和视觉提示。
开放词汇检测器能接收类别词，但通常不是自然语言自回归模型，也不天然覆盖点、文字转录和轨迹。
通用 VLM 有语言灵活性，却未必为密集、小目标和严格几何定位做过足够训练。
GroundingPI 的设计选择是保留语言模型的条件生成能力，同时用原子坐标 token 把几何变成词表元素。
因此它不是把 detector head 接到 LLM 外面，而是让语义和几何都经同一个 causal LM head 生成。
### 2.3 三项主要贡献
第一，提出一个覆盖 11 类感知能力的统一 grounding foundation model。
第二，通过规模、数据组成、OCR mixing、坐标表示和视觉编码器消融分析什么样的感知预训练更可能迁移到动作。
第三，在统一下游动作专家下比较 VLM、WAM、grounding model 和 embodied foundation，尝试把“基础表征质量”从“动作头差异”中分离出来。
### 2.4 贡献边界
模型不是直接从图像输出电机命令的 VLA checkpoint。
HF 权重是 grounding VLM。
仓库中的 StarVLA/OpenWAM 内容是 backbone comparison workflow，需要各自环境、数据和 policy checkpoint。
官方 README 明确写明，release 不提供一个可把 GroundingPI 权重直接加载成最终 action policy 的适配器，见 `README.md:350-357`。
## 3. 方法：统一的自回归空间感知模型
### 3.1 总体架构
![](assets/GroundingPI%20-%20视觉基元物理智能感知底座/figures/02-architecture.png)
**图解：** 图像先经 MoonViT-V2 得到 patch feature，2×2 相邻空间 patch 被分组，projector 把每组四个 1024 维向量展平为 4096 维，再映射到 Qwen3-4B 的 2560 维语言空间。语言查询和视觉 embedding 进入同一序列，decoder 自回归生成标签、分隔符、box/point 坐标和结束标记。
**容易误读之处：** 图中“2×2 merge”不是四个空间 patch 取平均。四个空间 patch 的信息全部保留并 flatten；只有视频/时间聚合路径涉及时间维 pooling。
**图号与页码：** Figure 2，PDF Page 3。
**官方来源：** 论文 PDF；官方 HF 资源 `assets/fig2-architecture.png`；代码 `models/vlm/modeling_groundingpi_vision.py:791-823`。
给定图像 $I$ 与查询 $P$，视觉编码器和连接器产生：
$$V=C_\phi(E_\psi(I))$$
语言模型按因果顺序生成可变长输出 $Y$：
$$p_\theta(Y\mid I,P)=\prod_{i=1}^{|Y|}p_\theta(y_i\mid V,P,y_{<i})$$
这里的 $y_i$ 既可能是普通文字 token，也可能是协议 token 或坐标 token。
### 3.2 MoonViT-V2 视觉编码器
#### 3.2.1 输入输出
MoonViT-V2 接收 NaViT 风格的动态分辨率 patch 序列。
官方配置为 27 个视觉层、hidden width 1024、FFN width 4096、12 个 attention head、patch size 14。
输入 patch 上限为 4096。
空间 2×2 分组后，最多输出 $4096/4=1024$ 个视觉 token。
#### 3.2.2 为什么这样设计
动态分辨率避免把长图、文档、GUI 截图和普通照片都压到固定方形尺寸。
patch budget 又给计算量设置硬上限，防止高分辨率输入让视觉序列无限增长。
MoonViT-V2 相比普通 MoonViT 和 Qwen3-ViT 的消融分数更高，但该比较同时改变预训练视觉表示，不能只归因于某个局部结构。
#### 3.2.3 关键超参
- 层数：27；
- hidden width：1024；
- FFN width：4096；
- patch：14×14；
- 最大输入 patch：4096；
- 2×2 merge；
- 最大 projector 输出 token：1024；
- 输入均值和标准差均为 $(0.5,0.5,0.5)$。
上述配置来自官方 `config.json`、`preprocessor_config.json` 与论文 Table 8。
#### 3.2.4 代码与论文差异
论文主文只说 projector “aggregates” 2×2 feature，容易让读者联想到 average pooling。
代码里 projector 的输入宽度显式计算为 $1024\times2\times2=4096$，说明空间通道被拼接保留，见 `models/vlm/modeling_groundingpi_vision.py:791-823`。
`merge_type="sd2_tpool"` 中的 `tpool` 处理时间维；空间维仍经过重排和 flatten，见 `models/vlm/modeling_groundingpi_vision.py:717-721`。
当前公开 image processor 只接受 image；传入其他 media type 会抛错，见 `models/vlm/image_processing_groundingpi.py:75-95`、`models/vlm/image_processing_groundingpi.py:152-159`。
### 3.3 2×2 merge 的真实张量语义
#### 3.3.1 空间维不做平均
设四个相邻 patch 为 $p_{00},p_{01},p_{10},p_{11}\in\mathbb{R}^{1024}$。
空间聚合不是：
$$\bar p=\frac{p_{00}+p_{01}+p_{10}+p_{11}}{4}$$
实际是：
$$p_{\mathrm{merge}}=[p_{00};p_{01};p_{10};p_{11}]\in\mathbb{R}^{4096}$$
因此 token 数下降四倍，但单 token 的 projector 输入维度扩大四倍。
这是一种“序列长度换通道宽度”的信息重排，而不是先做不可逆均值压缩。
#### 3.3.2 为什么这一点重要
对 dense grounding 和 tiny object，2×2 内部的局部差异可能决定边界和实例分离。
平均池化会更早抹掉四个位置的差异。
flatten 允许后续 MLP 学习如何组合四个子位置。
代价是第一层线性映射参数较多：$4096\times4096$。
#### 3.3.3 时间维 pooling
视觉编码器路径中的 `tpool_patch_merger` 可以沿时间分组处理视频帧。
但当前 GroundingPI processor 把视频路径限制掉，且主模型注释说明视频当前主要作为多图/文字时间戳处理。
所以公开 release 不应被描述成完整时空视频 encoder。
代码证据见 `models/vlm/modeling_groundingpi.py:1471-1481`。
### 3.4 Projector
#### 3.4.1 输入输出
projector 对每个 2×2 组执行：
$$\mathbb{R}^{4096}\rightarrow\mathbb{R}^{4096}\rightarrow\mathbb{R}^{2560}$$
中间激活是 GELU，输出使用 RMSNorm。
完整顺序还包括视觉输出的 LayerNorm、空间分组、两层 bias-free linear 和最终 RMSNorm。
#### 3.4.2 为什么这样设计
第一层保持 4096 宽度，让模型先在四 patch 的联合空间内重新混合。
第二层压到语言 hidden width 2560，才能作为视觉 token 写入 Qwen3 输入 embedding。
输出归一化缓解视觉 feature 与文本 embedding 数值尺度不一致。
#### 3.4.3 关键超参
- 输入宽：4096；
- 隐层宽：4096；
- 输出宽：2560；
- linear bias：False；
- activation：GELU；
- post norm：RMSNorm；
- normalization epsilon：$10^{-5}$。
projector 参数量为 27,267,584。
#### 3.4.4 代码与论文差异
论文图没有画出 projector 后的 RMSNorm。
代码明确使用 `self.post_norm(self.proj(x))`，见 `models/vlm/modeling_groundingpi_vision.py:791-823`。
论文主文也没有展开 bias-free 细节；代码两层 `nn.Linear(...,bias=False)`。
### 3.5 Qwen3-4B 语言骨干
#### 3.5.1 输入输出
Qwen3 decoder 接收文本 embedding 与被替换进去的视觉 embedding。
它有 36 层，hidden width 2560，FFN width 9728。
注意力使用 GQA：32 个 query head、8 个 KV head，head dimension 128。
所有 36 层均为 full attention，没有 sliding window。
#### 3.5.2 为什么这样设计
普通 decoder-only LM 能直接复用 causal LM 训练目标。
共享词表让类别词、OCR 字符串、协议和坐标处于同一个概率空间。
GQA 减少 KV cache 体积，适合自回归生成长结构化序列。
#### 3.5.3 关键超参
- 层数：36；
- hidden：2560；
- FFN：9728；
- 32 Q heads；
- 8 KV heads；
- head dim：128；
- RoPE base：5,000,000；
- 最大位置容量：262,144；
- attention dropout：0；
- RMSNorm epsilon：$10^{-6}$；
- 词表：152,670。
#### 3.5.4 代码与论文差异
模型位置容量 262,144 不等于论文训练上下文长度。
Pretrain Stage 1/2/4 用 8192，Stage 3 用 32768；公开 SFT 默认只用 2048。
公开 vLLM service 默认 `max_model_len=16384`，也没有使用完整位置容量，见 `infer/serve_vllm.py:39-50`。
### 3.6 448×448 的具体张量账
输入一张 $448\times448$ RGB 图像。
patch size 为 14，因此网格为：
$$H_p=W_p=448/14=32$$
输入 patch 数为：
$$N_{\mathrm{patch}}=32\times32=1024$$
`navit_patchify` 后的 `pixel_values` 形状是 `[1024,3,14,14]`，`image_grid_thw=[1,32,32]`。
该形状定义见 `models/vlm/media_utils.py:319-341`。
MoonViT-V2 输出每个 patch 1024 维，概念形状为 `[1024,1024]`。
2×2 分组后空间网格为 $16\times16$。
输出 token 数为：
$$N_{\mathrm{visual}}=(32/2)\times(32/2)=256$$
每组四个 feature flatten 后为 `[256,4096]`。
projector 第一层后仍为 `[256,4096]`。
第二层与 RMSNorm 后为 `[256,2560]`。
最终 256 个视觉 embedding 替换文本序列中的 256 个 image placeholder embedding。
这个例子也说明：模型上限 4096 input patches 并不意味着每张 448 图都用满上限。
### 3.7 动态分辨率
image processor 读取原始宽高，根据 patch size、merge size、总 patch 上限和单边上限动态 resize。
如果计算出的输出 token 超过 `max_output_tokens`，它会按比例缩小 input patch limit 并重新计算，见 `models/vlm/image_processing_groundingpi.py:75-92`。
默认配置：
- `in_patch_limit=4096`；
- `patch_size=14`；
- `merge_kernel_size=2`；
- `patch_limit_on_one_side=512`。
因此理论最大视觉输出 token 为：
$$N_{\max}=\left\lfloor\frac{4096}{2^2}\right\rfloor=1024$$
环境变量 `IMAGE_MAX_TOKEN_NUM` 还可进一步收紧预算，见 `models/vlm/image_processing_groundingpi.py:34-44`。
公开 native serving 配置使用 512，而训练配置使用 1024；不能把模型上限、训练值和服务默认值混为一谈。
### 3.8 视觉 embedding 如何进入语言序列
模型先根据 `input_ids` 得到文本 embedding。
视觉分支返回各图的 projected feature。
随后代码拼接所有 image embedding，构建 image placeholder mask，并用 `masked_scatter` 覆盖对应文本位置。
关键实现位于 `models/vlm/modeling_groundingpi.py:1310-1318`。
语言 decoder 不需要另一个 cross-attention tower；它把视觉 token 当作序列中的 embedding。
position ID 是普通一维 Qwen3 position，而非 Qwen2-VL 三轴 mRoPE。
SFT plugin 专门把 position tensor 保持为 `[batch,sequence]`，见 `train/vlm/plugin.py:100-153`。
### 3.9 共享词表上的自回归生成
输出头是一个从 2560 到 152,670 的无 bias linear：
$$\mathrm{logits}_t=W_{\mathrm{LM}}h_t\in\mathbb{R}^{152670}$$
代码见 `models/vlm/modeling_groundingpi.py:1369-1378`、`models/vlm/modeling_groundingpi.py:1537-1547`。
输入 embedding 与输出 head 不共享权重。
二者各自包含 $152670\times2560=390,835,200$ 个参数。
普通词、结构 token 和坐标 token 都由同一个 head 预测。
prefill 完成后，generation code 会丢弃 image tensor，避免每个 decode step 重跑 vision tower，见 `models/vlm/modeling_groundingpi.py:1607-1616`。
## 4. GAM：把几何变成语言 token
### 4.1 词表扩展
基础 tokenizer 有 151,669 个可寻址 token。
GroundingPI 新增：
- 1000 个坐标 token：`<0>` 到 `<999>`；
- 1 个类别分隔 token：`</c>`。
因此最终词表为：
$$151669+1000+1=152670$$
坐标 token ID 连续分布在 151669–152668。
`</c>` 的 ID 为 152669。
扩展脚本在 `train/tokenizer/extend_tokens.py:31-50` 固定这些常量。
它将 1001 个 token 作为 additional special tokens 添加并检查最终长度，见 `train/tokenizer/extend_tokens.py:147-166`。
输入 embedding 和 LM head 都独立扩容，坐标初值来自对应数字字符串 token row 的均值，见 `train/tokenizer/extend_tokens.py:220-233`。
### 4.2 坐标量化
论文先把图像坐标归一化到中间 $[0,1000]$ 网格，再映射到原子 token 的 $[0,999]$ 网格。
对中间整数坐标 $v$：
$$q(v)=\left\lfloor\frac{999v+500}{1000}\right\rfloor$$
代码用纯整数实现 round-half-up：
$$q(v)=\frac{999v+500}{1000}\text{ 的整数下取整}$$
实现见 `grounding_pi/protocol.py:804-843`。
从 token 恢复图像坐标时：
$$x_{\mathrm{pixel}}=\frac{q_x}{999}W,\quad y_{\mathrm{pixel}}=\frac{q_y}{999}H$$
源 `xywh` 先转成 `xyxy`。
polygon 转为 axis-aligned enclosing box。
非法、反向、退化或非有限几何会被拒绝，而不是静默修复。
### 4.3 输出协议
一个 box entry 的规范形式是：
```text
<|object_ref_start|>car<|object_ref_end|><|box_start|><100><200><500><650><|box_end|>
```
point 只使用两个坐标：
```text
<|object_ref_start|>car center<|object_ref_end|><|box_start|><300><425><|box_end|>
```
显式不存在的目标使用普通文字 `None`：
```text
<|object_ref_start|>car<|object_ref_end|><|box_start|>None<|box_end|>
```
同一标签的多个实例共享 label wrapper，坐标 tuple 之间用逗号连接且不插空格。
不同 label entry 之间用逗号加一个空格连接。
box 采用 $(x_1,y_1,x_2,y_2)$，point 采用 $(x,y)$。
完整 grammar、常量和 task arity 定义在 `grounding_pi/protocol.py:1-18`、`grounding_pi/protocol.py:30-42`、`grounding_pi/protocol.py:121-136`。
序列化器确保坐标 token 相邻，见 `grounding_pi/protocol.py:370-400`。
严格 parser 会消费完整字符串并拒绝重复 label、重复 tuple、非法顺序和尾部分隔符，见 `grounding_pi/protocol.py:475-545`。
### 4.4 为什么不用文本数字
文本数字可能被 tokenizer 拆成多个 token。
不同数值的 token 长度不稳定。
括号、逗号、空格和小数格式又会引入额外生成错误。
原子坐标 token 让每个轴值固定占一个 token。
一个 box 的几何主体恒为四个 token。
同一 label 下多实例共享 wrapper，可进一步摊薄结构开销。
### 4.5 GAM 的工程约束
vLLM/OpenAI-compatible 请求必须保留 special token：
- `skip_special_tokens=false`；
- `spaces_between_special_tokens=false`。
否则解码器可能删掉 wrapper，或在相邻坐标 token 间插空格，破坏 canonical parser。
官方 client 强制设置这两个参数，见 `grounding_pi/client.py:87-91`。
评测 runner 在 GAM mode 也强制覆盖，见 `eval/eval_runner.py:163-166`。
### 4.6 协议的优势与局限
优势是统一、紧凑、可严格验证。
局限是仍为自回归输出，预测实例越多，输出序列越长。
量化引入有限空间误差；分辨率越高或目标越小，1/999 网格误差越值得关注。
axis-aligned box 无法表达旋转框和精细 mask。
point/box 只能表达“在哪里”，不能直接表达接触力、动力学、可达性和闭环反馈。
## 5. 数据引擎与监督构造
### 5.1 五步数据引擎
![](assets/GroundingPI%20-%20视觉基元物理智能感知底座/figures/03-data-engine.png)
**图解：** 数据引擎不是简单让单一教师生成伪标签，而是先产生候选，再按字段融合，进行任务依赖验证，对 pending case 调用补充教师或局部视图，最后形成有版本的监督并迭代训练统一 grounding expert。
**图号与页码：** Figure 3，PDF Page 3。
**官方来源：** 论文 PDF，Figure 3；论文 Appendix A.3。
五步可以概括为：
1. **Candidate generation**：教师 ensemble 与已有 grounding expert 产生对象、box、mask、属性、文字和布局候选；
2. **Field fusion**：语义字段和几何字段分别融合，不把整条 annotation 当成不可分割单元；
3. **Task validation**：依据任务检查几何、语法、语义、coverage 和 required fields；
4. **On-demand evidence**：pending 项调用 complementary teacher、crop/local view 或 rollout；
5. **Versioned supervision**：接受的标签进入训练集，更新统一 expert；错误标签可 revise 或 retire。
### 5.2 字段级融合为何重要
两个教师可能分别擅长类别识别和几何边界。
整条标注二选一会丢失互补信息。
字段级融合允许保留已验证的语义，同时只为不确定 box 请求额外证据。
对象—部件包含关系和 word—line 关系也不被简单当作重复实例合并。
### 5.3 任务依赖验证
一个 supervision item 只有在所有 required field 都 accepted 时才被接受。
需要“找出全部实例”的查询还要通过 coverage 检查。
`None` 不是“教师没有输出”的同义词。
缺失预测不能证明目标不存在；absence 本身也需要验证。
这对 dense grounding、counting 和 negative query 特别关键。
### 5.4 一份记录如何派生多个任务
category–box 可生成 grounding。
属性、关系和描述可生成 referring。
有效实例区域可生成 pointing。
同类 exemplar correspondence 可生成 visual prompting。
文字区域加 transcription 可生成 OCR。
这种复用提高监督利用率，但每个派生任务仍保留自己的 required fields 和 coverage condition。
### 5.5 数据透明度问题
论文没有完整列出所有 teacher、版本、调用比例和成本。
没有公布每一步 accepted/rejected/unresolved 的数量。
没有公布人工审核规模和抽检错误率。
没有公开完整训练数据清单与样本级 provenance。
没有给出跨训练集和 34 benchmark 的完整去重、近重复和泄漏审计。
没有公开 data engine 的生产代码。
因此读者可以理解机制，却无法从 release 重建 721.58B token 的训练语料。
## 6. 训练：从基础 VLM 到空间 SFT 与 GRPO
### 6.1 三个总阶段与四个监督 stage
训练包含三个总阶段：
- Pretrain 1：Base VLM training，内部含 Stage 1–3；
- Pretrain 2：Coordinate alignment/SFT，对应 Stage 4；
- RL：从 Stage 4 checkpoint 开始进行 GRPO。
总 token 暴露量为：
$$500.41\mathrm{B}+221.17\mathrm{B}=721.58\mathrm{B}$$
其中 500.41B 对应基础 VLM 训练暴露，221.17B 对应空间专门化暴露；221B 也是架构图中的 spatial pretraining 近似口径。
这里是 token exposure，不是独立图文 pair 数，也不是去重样本量。
### 6.2 Pretrain 1 Stage 1：projector alignment
冻结视觉 encoder 和语言模型，只训练 projector。
监督数据为 image-caption。
目标是 assistant response 的 causal cross-entropy。
没有额外 feature-distance 或 contrastive objective。
context/packing 均为 8192。
128 GPU，per-device batch 6，global batch 768。
projector peak LR 为 $5\times10^{-5}$。
weight decay 为 0。
### 6.3 Pretrain 1 Stage 2：joint multimodal pretraining
解冻视觉 encoder、projector 和语言模型，更新全部模块。
同时使用 text-only 与 general image–text 数据。
文本样本对所有有效 causal target 监督。
多模态样本只监督 assistant response。
语言 peak LR 为 $10^{-5}$。
视觉和 projector peak LR 均为 $10^{-6}$。
context/packing 均为 8192。
128 GPU，per-device batch 2，global batch 256。
### 6.4 Pretrain 1 Stage 3：general visual/video understanding
继续更新全部模块。
数据覆盖视觉问答、指令跟随、image caption 和 video。
context/packing 扩大到 32768，以容纳更长多模态序列。
该阶段没有单独混入 spatial specialization 数据，也没有独立 text-only quota。
image-caption 仍保留在 general visual mixture 中。
128 GPU，per-device batch 2，global batch 256。
语言、视觉、projector LR 与 Stage 2 相同。
### 6.5 Pretrain 2 / Stage 4：coordinate alignment SFT
Stage 4 从 Stage 3 checkpoint 开始。
八个 spatial task group 为：
- detection；
- GUI grounding；
- referring-expression grounding；
- referring-expression pointing；
- OCR；
- document layout；
- dense pointing and counting；
- visual prompting。
所有模块更新。
context/packing 回到 8192。
128 GPU，per-device batch 2，global batch 256。
语言 LR 为 $10^{-5}$，视觉和 projector 为 $10^{-6}$。
坐标 token、语义 label 和 protocol marker 使用同一个 next-token cross-entropy。
没有额外 IoU regression loss。
### 6.6 监督损失
令 $S$ 为 assistant response 中有效 token 位置集合：
$$\mathcal{L}_{\mathrm{SFT}}=-\frac{1}{|S|}\sum_{t\in S}\log p_\theta(y_t\mid I,P,y_{<t})$$
prompt、visual token、padding 和空 thinking-prefix 不参与多模态监督。
Stage 2 对 text 与 vlm domain 分别按有效 target 归一化，再相加。
Stage 1、3、4 只使用 VLM domain loss。
packing 保留每个样本的 causal boundary 和 supervision mask。
### 6.7 共同优化配置
Pretrain 1 和 Pretrain 2 都使用 128 GPU。
精度为 BF16。
分片为 ZeRO-1。
使用 FlashAttention 和 activation recomputation。
optimizer 为 AdamW，betas $(0.9,0.95)$，epsilon $10^{-8}$。
gradient clipping 为 1.0。
Stage 2–4 weight decay 0.1。
cosine warmup 3%。
每个 stage 配置 budget 为 1 epoch。
seed 与 data seed 都为 42。
### 6.8 两阶段后训练图
![](assets/GroundingPI%20-%20视觉基元物理智能感知底座/figures/04-training-pipeline.png)
**图解：** 左侧 SFT 让模型在 teacher forcing 下学习完整结构化输出；右侧每个 image-query 从旧策略采样 $G=8$ 个 response，用 grounding 或 OCR reward 计算组内标准化 advantage，再执行 clipped GRPO。图中 missing、duplicate 和 aligned rollout 展示 reward 不只评价单框 IoU，还处理集合完整性和格式可靠性。
**图号与页码：** Figure 4，PDF Page 6。
**官方来源：** 论文 PDF，Figure 4；Appendix A.6。
### 6.9 GRPO 的 rollout 与冻结范围
每个 prompt 采样 $G=8$ 个完整自回归回答。
策略初始化自 Stage 4 SFT checkpoint。
reference model 是被冻结的同一 SFT checkpoint 副本。
vision encoder 和 projector 在 RL 全程冻结。
只更新语言参数，包括 input embedding 和 untied output head。
使用 256 GPU、BF16、ZeRO-2。
每卡 response batch 1，accumulation 1。
distinct prompt batch 32，response batch 256。
每次 rollout 只做 1 次 update。
LR 为常数 $5\times10^{-7}$，无 warmup。
clip epsilon 为 0.2，KL beta 为 0.02。
### 6.10 Grounding 奖励
$R_{set}$ 逐 reference 寻找最大 IoU prediction，再检查类别一致性。
它强调 reference coverage。
由于每个 reference 独立匹配，prediction 可以被复用，所以论文提醒该 surrogate 甚至可能超过 1。
$R_{strict}$ 综合：
- structured format：0.10；
- count agreement：0.10；
- IoU 0.50/0.75/0.95 的平均 detection F1：0.50；
- matched-box IoU：0.25；
- ordering compliance：0.05；
- oversized penalty：−0.07；
- duplicate penalty：−0.03。
两个 reward 先在同 prompt 的八个响应内独立标准化：
$$\widetilde A_i=0.7Z(R_{set,i})+0.3Z(R_{strict,i})$$
然后组合 advantage 再在整个 response batch 标准化。
### 6.11 OCR Hungarian text-geometry reward
OCR prediction 和 reference 都表示为 box–text pair 集合。
模型先计算 box IoU 和规范化文字 edit similarity。
文字规范化包括 Unicode NFKC、case folding 和 alphanumeric filtering。
所有 affinity 都通过 Hungarian maximum-weight one-to-one assignment。
hard affinity 要求文字精确匹配且 IoU 超过阈值。
soft affinity 使用：
$$A^{soft}_{ij}=\sqrt{I_{ij}E_{ij}}\,\mathbb{I}[I_{ij}\ge0.10]\mathbb{I}[E_{ij}\ge0.20]$$
reward 同时支持 word-level、line-level、local granularity group 和复杂文字的 complementary reference view。
最终还加入格式、count、duplicate、oversized、invalid 等项。
OCR 所有子项先合成一个 scalar，再做 prompt 内标准化；不是每个 OCR 子项单独标准化。
训练 reward 与公开 benchmark metric 是不同概念。
### 6.12 公开 SFT recipe 与论文训练不能混淆
官方开源 `configs/train/vlm.yaml` 是 release 后的可运行 SFT recipe，不是 721.58B token 的论文全流程。
公开默认值包括：
- full fine-tuning；
- BF16；
- `max_length=2048`；
- packing length 2048；
- per-device batch 2；
- gradient accumulation 1；
- LM LR $10^{-5}$；
- vision LR $10^{-6}$；
- aligner LR $10^{-6}$；
- 1 epoch；
- weight decay 0.1；
- ZeRO-1；
- expected world size 1。
因此默认 global batch 为：
$$2\times1\times1=2$$
论文 Stage 4 的 global batch 是 256，context 8192，运行于 128 GPU。
公开 recipe 只能证明官方提供了 SFT 基础设施，不能被当成论文训练配置的完整复现。
## 7. 感知实验：34 benchmark 与 11 类能力
### 7.1 评测套件构成
34 个 benchmark entry 分为八组：
- Detection 4：COCO、LVIS、Dense200、VisDrone；
- Referring boxes 6：RefCOCOg val/test、RefCOCO、RefCOCOg、RefCOCO+ family entry、HumanRef；
- Object pointing 7：RefCOCOg val/test、COCO、LVIS、Dense200、VisDrone、HumanRef；
- Spatial pointing 4：RefSpatial Location/Placement/Unseen、RoboSpatial Context；
- GUI 3：ScreenSpot-Pro、ScreenSpot-V2、OSWorld-G；
- OCR 4：HierText、ICDAR2015、TotalText、SROIE；
- Layout 2：DocLayNet、M6Doc；
- Visual prompting 4：FSC147、Dense200、COCO、LVIS。
仓库 registry 实际有 42 个 recipe。
论文 suite 排除四个 `*_Labelless` 和四个 `*_Boxonly` variant，保留 `groundingpi34`。
证据见 `eval/README.md:78-97` 与 `configs/eval/suites.json:34-69`。
### 7.2 73.68 的统计口径
headline Avg 是 11 个 capability score 的等权平均。
可以写成：
$$\mathrm{Avg}_{11}=\frac{1}{11}\sum_{c=1}^{11}s_c$$
每个 $s_c$ 又按该 capability 自己的 benchmark 与聚合规则计算。
它不是：
$$\frac{1}{34}\sum_{b=1}^{34}s_b$$
更不是把 34 项所有样本预测合并后算 micro metric。
这种等权方案避免数据量大的 benchmark 支配总分。
但它也人为赋予每类能力相同权重，未必符合真实机器人或驾驶的风险权重。
### 7.3 44 与 45 个 baseline 的计数矛盾
摘要、Figure 1 caption 和正文都写 44 baselines。
Figure 5 图中文字却写 “Across 45 Baselines Including GPT-6 Astra”。
这不是读者计算错误，而是论文图文内部计数不一致。
可能原因包括是否把 Astra、GroundingPI、自有模型变体或 external row 计入 baseline，但论文没有给出明确 reconciliation。
严谨引用时应写“正文称 44，Figure 5 标成 45”。
### 7.4 capability wheel
![](assets/GroundingPI%20-%20视觉基元物理智能感知底座/figures/05-capability-wheel.png)
**图解：** 雷达/轮图展示 11 类能力上 GroundingPI 与四个最强 baseline 的覆盖范围。它强调的是跨任务宽度，不意味着每个子 benchmark 都第一；例如 GPT-6 Astra 在 frontier GUI、Dense200 pointing、TotalText 等项目仍强。
**图号与页码：** Figure 5，PDF Page 8。
**官方来源：** 论文 PDF；官方 HF 资源 `assets/fig5-grounding-performance.png`。
### 7.5 Table 1：检测与 referring grounding
下表完整摘录三种关键模型的八列 F1mIoU：
| 模型 | COCO | LVIS | Dense200 | VisDrone | HumanRef | RefCOCOg val | RefCOCOg test | RefCOCO avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B | 46.53 | 49.86 | 14.02 | 31.14 | 70.06 | 75.27 | 75.88 | 75.24 |
| GroundingPI | 62.98 | 56.02 | 74.53 | 40.44 | 88.56 | 85.62 | 84.36 | 83.92 |
| GPT-6 Astra | 62.75 | 54.97 | 65.04 | 37.12 | 83.01 | 74.98 | 78.91 | 77.81 |
最显著的是 Dense200：74.53 对 14.02。
这支持 GroundingPI 对密集实例的专门优势。
但 VisDrone 40.44 仍低于 SenseNova-Vision 42.35，tiny-object 并未被彻底解决。
GroundingPI 的 COCO F1@IoU 0.95 仍是明显薄弱项，不能把 mIoU aggregate 解读为所有边界阈值全面领先。
### 7.6 Table 2：robot、spatial 与 GUI grounding
| 模型 | RefSpatial avg | RefSpatial Unseen | RoboSpatial Context | ScreenSpot-Pro | ScreenSpot-V2 | OSWorld-G |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B | 49.00 | 27.27 | 64.75 | 56.74 | 92.30 | 56.91 |
| GroundingPI | 75.50 | 75.32 | 73.77 | 65.78 | 96.15 | 74.82 |
| GPT-6 Astra | 85.93 | 81.93 | 65.69 | 93.17 | 97.88 | 86.70 |
GroundingPI 相对同尺寸 Qwen3-VL-4B 提升大。
但 Astra 在 RefSpatial 与 GUI 多列显著领先。
这正是“专门化 4.844B 模型接近 frontier、但没有全面超越 frontier”的更准确说法。
### 7.7 Table 3：OCR 与 document layout
| 模型 | HierText | ICDAR2015 | TotalText | SROIE | DocLayNet | M6Doc |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B | 23.48 | 28.41 | 38.35 | 40.41 | 40.81 | 24.73 |
| GroundingPI | 41.70 | 55.68 | 49.32 | 72.47 | 85.08 | 74.82 |
| GPT-6 Astra | 39.58 | 48.87 | 53.55 | 53.57 | 77.54 | 60.59 |
GroundingPI 在 SROIE 和 M6Doc 的优势很大。
TotalText 上 Astra 53.55 高于 GroundingPI 49.32。
DocLayNet 上 SenseNova-Vision 85.53 略高于 GroundingPI 85.08。
所以 OCR/layout 的总体优势也不是全列第一。
### 7.8 Table 4：object pointing
| 模型 | HumanRef | RefCOCOg val | RefCOCOg test | COCO | LVIS | Dense200 | VisDrone |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B | 66.89 | 76.43 | 77.64 | 65.33 | 55.08 | 21.72 | 23.50 |
| GroundingPI | 88.79 | 90.26 | 90.06 | 84.79 | 79.51 | 81.27 | 67.47 |
| GPT-6 Astra | 83.83 | 87.80 | 84.90 | 82.17 | 77.14 | 86.57 | 65.62 |
GroundingPI 在七列中六列第一。
Astra 在 Dense200 pointing 上以 86.57 领先 GroundingPI 的 81.27。
有趣的是 GroundingPI 在 Dense200 box grounding 更强，但 point selection 略弱。
这表明“找一个合适点”和“估计完整边界”不是同一能力。
### 7.9 baseline 协议异质性
有些结果来自本地评测。
有些带星号的是外部论文公布结果。
SenseNova-Vision 的部分 OCR 分数混合 published 和 locally evaluated。
某些 GUI 模型协议不兼容，被标为 N/A。
GroundingPI 自身数字取五个随机种子、每种两次，共十次均值。
本地运行的其他 baseline 因成本只做三种 seed、每种一次。
GPT-6 Astra 还设置 thinking effort=High。
因此表格是尽量统一后的大规模比较，但不是所有模型、所有任务完全同质的单一实验协议。
## 8. 迁移到物理智能
### 8.1 统一下游接口的设计
每个 foundation backbone 对观察和语言指令只做一次 forward。
从 backbone 的 $N_B$ 个 block 中按归一化深度抽取八层：
$$\ell_j=\operatorname{round}\left(\frac{j(N_B-1)}{7}\right),\quad j\in\{0,\ldots,7\}$$
每个 hidden state 先 norm，再用 backbone-specific linear projector 映射到 width 1024。
加入 learned depth embedding 后，交给共享 learned-query resampler 压成 64 token。
每个 depth 条件的形状为 `[64,1024]`。
八个 depth 总条件预算为 512 token，但分别注入八个对应 cross-attention block。
### 8.2 固定 ActionDiT
统一 action expert 是 16 个 atomic block 的 π-style ActionDiT。
偶数 block 0、2、…、14 是八个 cross-attention block。
奇数 block 1、3、…、15 是八个 action self-attention block。
不是“16 个每个都含 self+cross 的复合层”。
共享配置：
- 8 个 backbone depth；
- 每 depth 64 condition token；
- width 1024；
- 16 attention head；
- head dim 64；
- FFN width 4096；
- 32 planning token；
- action horizon 16；
- 无 raw text/proprioception bypass；
- 推理 4 次 Euler step；
- 每个 observation 只做一次并缓存 backbone forward。
论文出处为 Appendix B.2.1–B.2.2。
公开 OpenWAM 实现也描述同一 topology，见 `vla/openwam/README.md:1-13`。
### 8.3 动作训练预算
每个 foundation 使用相同动作示范、采样与增强。
训练 40,000 update。
使用 40 个 accelerator、BF16、ZeRO-2。
per-device batch 32，global batch 1280。
Action expert/interface LR 为 $10^{-4}$。
适配 backbone 参数 LR 为 $10^{-5}$。
cosine decay，2000 warmup step，最小 LR $5\times10^{-7}$。
每个 observation 采两个 flow noise sample。
### 8.4 物理迁移结果图
![](assets/GroundingPI%20-%20视觉基元物理智能感知底座/figures/06-physical-transfer.png)
**图解：** 左侧 RoboTwin 比较场景内训练和 Clean→Random 的环境变化；中间 RoboCasa 比较 ID 与对象类型、外观、容器组合 OOD；右侧把 nuScenes L2 取倒数绘图以统一“越高越好”的方向。图中 GroundingPI 在五个六个操作设置中第一，四个 OOD 设置全部第一。
**关键数字：** RoboTwin Full/ID 66.20，Clean2Random/OOD 17.60；RoboCasa ID 37.75，Type OOD 28.94，Appearance OOD 37.78，Container OOD 37.86；nuScenes 平均 L2 0.296 m。
**图号与页码：** Figure 6，PDF Page 11。
**官方来源：** 论文 PDF；官方 HF 资源 `assets/fig6-physical-intelligence.png`。
### 8.5 RoboTwin 2.0
RoboTwin Full 同时训练和评测 Clean 与 Randomized data。
GroundingPI 成功率为 66.20%。
Clean2Random 只在 Clean demonstration 上训练，在 Randomized scene 评测。
变化包括 clutter、lighting、table/background texture、tabletop height 和 language instruction。
GroundingPI 为 17.60%，第二名 Rex-Omni 为 14.10%。
该 OOD 更接近 scene robustness，而不是新任务或新动力学泛化。
### 8.6 RoboCasa-GR1
RoboCasa 使用 humanoid dexterous hand，并从 head camera 观察，不使用 wrist camera。
GroundingPI ID 为 37.75%，低于 RynnBrain 的 39.00%。
三类 OOD 分别考察：
- Type：未见对象类别，28.94%；
- Appearance：熟悉对象/容器的新纹理，37.78%；
- Container：见过对象的新容器组合，37.86%。
论文 OOD Avg 不是三列简单平均。
它按 suite 大小加权：
$$SR_{OOD}=\frac{14SR_{Container}+18SR_{Appearance}+32SR_{Type}}{64}$$
GroundingPI 的加权 OOD Avg 为 33.38%。
### 8.7 nuScenes open-loop
输入是当前前视相机图像与最多七条 ego state history。
history 间隔 0.5 秒，最长覆盖 3 秒。
输出未来 3 秒六个 cumulative position：
$$Y=((x_1,y_1),\ldots,(x_6,y_6))$$
$x$ 范围为 $[-60,60]$ m，$y$ 范围为 $[-20,20]$ m。
轨迹也复用 0–999 token，但这里是 metric range quantization，不是图像坐标归一化。
GroundingPI 平均 L2 为 0.296 m。
Qwen3-VL-4B 为 0.301 m，RynnBrain 为 0.308 m。
绝对增益只有 0.005 m 和 0.012 m。
该结果一致但幅度小，且只测 open-loop recorded trajectory agreement。
它不证明 closed-loop 安全性、规划稳定性或实时车辆控制。
### 8.8 action-data efficiency
RoboCasa ID 使用 25%、50%、75%、100% demonstration。
GroundingPI 在三个 reduced-data 设置都领先。
50% 数据时达到 28.75%。
该分数高于所有 baseline 在 75% 数据时的结果；75% 最强 baseline RynnBrain 为 27.75%。
但 full data 时 RynnBrain 39.00 高于 GroundingPI 37.75。
因此强项是有限动作示范下的适配，不是 full-data ID 绝对第一。
更重要的是，action-data efficiency 不等于 total training efficiency。
GroundingPI 之前已经消耗 721.58B token 的感知与多模态训练。
论文没有比较达到同一成功率所需的总 FLOPs、能耗或总数据成本。
### 8.9 迁移证据的正确解读
统一 action head 消除了原生 policy head 和训练预算的主要干扰。
但“只换 backbone”并不等于“只换 grounding 能力”。
不同 backbone 同时改变：
- 模型规模；
- 视觉 encoder；
- 语言 architecture；
- connector；
- tokenization；
- 预训练数据；
- 预训练目标；
- intermediate feature accessibility。
最稳妥结论是“GroundingPI 这个完整预训练 foundation 更适合该统一动作 recipe”。
不能据此写成“grounding supervision 已被严格证明是唯一原因”。
## 9. Scaling、数据组成与架构消融
### 9.1 88.4B 到 221B 的 scaling
![](assets/GroundingPI%20-%20视觉基元物理智能感知底座/figures/07-scaling.png)
**图解：** 四个 checkpoint 使用相同初始 backbone、相同下游 ActionDiT、相同动作示范和动作训练 recipe，仅改变 grounding-training exposure。Grounding score 与 OOD Avg 单调上升；ID 在第三点达到 38.75 后，完整 221B 点回落到 37.75。
**图号与页码：** Figure 9，PDF Page 14。
**官方来源：** 论文 PDF，Figure 9；Appendix Table 16。
四个点为：
| Grounding token exposure | Grounding Avg | RoboCasa ID | RoboCasa OOD Avg |
|---:|---:|---:|---:|
| 88.4B | 61.69 | 33.92 | 26.59 |
| 132.6B | 69.71 | 35.92 | 30.00 |
| 176.8B | 72.22 | 38.75 | 32.06 |
| 221B | 73.68 | 37.75 | 33.38 |
从端点看，Grounding Avg 增 11.99 pp，ID 增 3.83 pp，OOD 增 6.78 pp。
但 ID 并非单调，最后一步下降 1.00 pp。
四点且没有误差条，不足以建立 scaling law。
### 9.2 六类监督组成
![](assets/GroundingPI%20-%20视觉基元物理智能感知底座/figures/08-data-composition.png)
**图解：** A–F 六组分别是 Basic Grounding、Dense Grounding、Referring、Basic Pointing、Robo Pointing 和 Else/OCR。右侧 leave-one-out 以 full six 为基准，显示移除 Dense 与 F 组对操作迁移影响最大；但实验矩阵并不完整，不能把各组 drop 当成独立可加的因果贡献。
**图号与页码：** Figure 10，PDF Page 15。
**官方来源：** 论文 PDF，Figure 10；Appendix Table 17。
论文主图把 F 简写为 OCR，附录更准确地写成 Else/OCR、Layout、GUI。
这一区别很重要：去掉 F 的效果不能只归因于 OCR。
### 9.3 组合实验数字
| 训练组 | RoboCasa ID | OOD Avg | nuScenes L2 |
|---|---:|---:|---:|
| A+D+F | 30.92 | 29.28 | 0.310 |
| A+B+C+F | 34.92 | 31.28 | 0.299 |
| A+B+C+D+F | 35.83 | 31.84 | 0.296 |
| A+B+D+E+F | 35.67 | 32.28 | 0.298 |
| A+C+D+E+F | 33.08 | 29.91 | 0.311 |
| A+B+C+D+E | 32.42 | 29.75 | 0.308 |
| F only | 24.42 | 7.28 | 0.391 |
| All six | 37.75 | 33.38 | 0.296 |
去掉 B，ID/OOD 分别下降 4.67/3.47 pp，L2 增加 0.015 m。
去掉 C，ID/OOD 分别下降 2.08/1.09 pp，L2 增加 0.002 m。
去掉 E，ID/OOD 分别下降 1.92/1.53 pp，L2 在显示精度下不变。
去掉 F，ID/OOD 分别下降 5.33/3.63 pp，L2 增加 0.012 m。
### 9.4 为什么 dense grounding 可能重要
密集场景要求区分相邻实例。
小目标要求保留很细的边界变化。
对两个同尺寸 box，若只沿水平方向偏移 $\delta$：
$$\operatorname{IoU}(\delta)=\frac{w-|\delta|}{w+|\delta|}$$
要满足阈值 $\tau$：
$$|\delta|\le w\frac{1-\tau}{1+\tau}$$
容许的绝对偏移与目标宽度 $w$ 成正比。
目标越小，同样 IoU 阈值允许的像素误差越小。
这给 dense/tiny supervision 向操作与驾驶迁移提供了合理机制解释。
但实验没有把 object size、crowding 和 annotation coverage 分开控制。
### 9.5 OCR mixing
三种 OCR schedule：
| OCR schedule | RoboCasa ID | OOD Avg | nuScenes L2 |
|---|---:|---:|---:|
| Cold-start mixing | 35.17 | 32.38 | 0.296 |
| Batch mixing | 34.92 | 32.53 | 0.296 |
| Cold-start + batch mixing | 37.75 | 33.38 | 0.296 |
联合 schedule 的操作结果最好。
驾驶 L2 在报告精度下完全相同。
这支持“早期接触 OCR 与持续联合训练可能互补”。
但它仍没有 text-loss-off、geometry-only 的完整对照结果。
论文提出的 transcription-gradient alignment 只是可检验机制，不是已证实机制。
### 9.6 架构与训练消融
完整配置 Grounding Avg 为 73.68。
文本坐标为 71.21，量化坐标增益 2.47 pp。
文本坐标速度只有量化配置的 0.25×。
普通 MoonViT 为 73.17，比 MoonViT-V2 低 0.51 pp。
Qwen3-ViT 为 71.65，比 MoonViT-V2 低 2.03 pp。
去掉 Stage III RL 为 72.86，RL 增益为 0.82 pp。
“Stage III”指总训练流程的第三阶段 RL，不是 Pretrain 1 内部的 Stage 3。
这种命名很容易混淆。
视觉 encoder 消融同时改变 pretrained representation，不能隔离某一个 layer fusion 机制。
### 9.7 token efficiency
| 模型 | COCO boxes/img | tokens/img | tokens/box | Dense200 boxes/img | tokens/img | tokens/box |
|---|---:|---:|---:|---:|---:|---:|
| SEED1.5-VL | 4.2 | 631.0 | 148.8 | 73.1 | 5446.3 | 74.5 |
| GroundingPI | 6.0 | 45.6 | 7.6 | 87.2 | 444.7 | 5.1 |
COCO 每框约减少 $148.8/7.6\approx19.6$ 倍 token。
Dense200 每框约减少 $74.5/5.1\approx14.6$ 倍 token。
这些比值是 serialization ratio。
它们没有控制模型大小、kernel、KV cache、采样实现、硬件和 batch。
论文没有报告 observation-to-action 端到端延迟。
## 10. 代码级实现
### 10.1 仓库概览与许可证
官方仓库 commit 为 `090932430bbbeea004cfa461767ede7f563c0f3a`。
主要目录：
- `grounding_pi/`：轻量 HTTP client、GAM protocol；
- `models/vlm/`：模型、视觉塔、processor 和 image processor；
- `infer/`：native 与 vLLM service；
- `train/`：tokenizer migration、SFT plugin 与 validation；
- `eval/`：34 项任务、prompt、parser、metric；
- `configs/`：训练、服务、评测 recipe；
- `vla/`：StarVLA/OpenWAM backbone comparison integration。
原创项目贡献使用 Apache License 2.0，见 `README.md:403-411`。
MoonViT/Kimi 派生实现与适用 derivative work 保留 Kimi K3 条件，见 `THIRD_PARTY_NOTICES.md:25-31`。
所以许可证必须写成：**Apache-2.0 + Kimi K3 派生条件**。
### 10.2 论文模块到代码入口
| 论文模块 | 代码入口 | 关键行 |
|---|---|---|
| 动态分辨率 | `models/vlm/image_processing_groundingpi.py` | 34-44，75-95，148-184 |
| NaViT patchify | `models/vlm/media_utils.py` | 319-341 |
| MoonViT-V2 forward/merge | `models/vlm/modeling_groundingpi_vision.py` | 717-721 |
| 2×2 projector | `models/vlm/modeling_groundingpi_vision.py` | 791-823 |
| 视觉 embedding 注入 | `models/vlm/modeling_groundingpi.py` | 1310-1318 |
| Qwen3 decoder | `models/vlm/modeling_groundingpi.py` | 1330-1357 |
| LM head | `models/vlm/modeling_groundingpi.py` | 1369-1378，1537-1547 |
| generation cache | `models/vlm/modeling_groundingpi.py` | 1607-1616 |
| processor glue | `models/vlm/processing_groundingpi.py` | 16-84 |
| GAM grammar | `grounding_pi/protocol.py` | 1-18，30-42，370-545 |
| tokenizer 扩展 | `train/tokenizer/extend_tokens.py` | 31-50，147-166，220-233 |
| SFT 注册 | `train/vlm/plugin.py` | 76-111，156-210 |
| native image placeholder | `infer/serve_vlm.py` | 163-169 |
| vLLM launcher | `infer/serve_vllm.py` | 13-36，39-90 |
| 34 项 suite | `configs/eval/suites.json` | 34-69 |
| eval contract | `eval/eval_runner.py` | 163-166 |
| OCR parser/metric | `eval/OCR/utils.py` | 511-517，586-734，1238-1240，1340-1346 |
| VLA comparison | `vla/openwam/README.md` | 1-13，154-157 |
### 10.3 全链路具体张量表
以单张 448×448 图、文本 batch 1 为例：
| 阶段 | 典型张量 | 形状 | 说明 |
|---|---|---|---|
| RGB 图 | image | `[448,448,3]` | 原始示例 |
| Patchify | `pixel_values` | `[1024,3,14,14]` | 32×32 patch |
| 网格 | `image_grid_thw` | `[1,3]`，值 `[1,32,32]` | T/H/W patch grid |
| ViT 输出 | patch hidden | `[1024,1024]` | 每 patch 1024 维 |
| 2×2 分组 | grouped hidden | `[256,4,1024]` | 概念视图 |
| flatten | projector input | `[256,4096]` | 四 patch 不平均 |
| MLP hidden | hidden | `[256,4096]` | GELU 前后宽度不变 |
| projector output | image embeds | `[256,2560]` | RMSNorm 后 |
| 文本 embedding | inputs embeds | `[1,L,2560]` | 含 image placeholder |
| masked scatter | multimodal embeds | `[1,L,2560]` | 256 个位置被视觉 embedding 替换 |
| Qwen3 hidden | decoder hidden | `[1,L,2560]` | 36 层 full attention |
| LM logits | logits | `[1,L,152670]` | 共享词表 |
| decode step | next logits | `[1,152670]` | AR 逐 token 生成 |
### 10.4 核心模块一：image processor
#### 10.4.1 输入输出
输入 PIL image 或 image list。
输出 `pixel_values` 与 `image_grid_thw`。
代码将 resize、pad、normalize 和 patchify 串起来，见 `models/vlm/image_processing_groundingpi.py:148-184`。
#### 10.4.2 为什么这样设计
不同任务输入宽高差异大，固定 resize 会破坏文档、GUI 和长宽比极端图像的局部结构。
NaViT patch 序列允许保持更合适的长宽比，同时用 patch budget 控制成本。
#### 10.4.3 关键超参
patch 14、merge 2、input patch limit 4096、单边 patch limit 512、默认最大 output token 1024。
`IMAGE_MAX_TOKEN_NUM` 能把服务预算降到更小值。
#### 10.4.4 代码与论文差异
论文给出能力上限 1024 projected visual token。
公开 native serve profile 默认设为 512，而训练 profile 为 1024。
当前 processor 不接视频，和论文 Stage 3 的 video training 叙述之间存在 release 范围差异。
### 10.5 核心模块二：MoonViT-V2 与 merge
#### 10.5.1 输入输出
输入 `[N,3,14,14]` patch 与 grid metadata。
输出 1024 维 patch hidden，再按 grid 分组。
#### 10.5.2 为什么这样设计
视觉塔负责高分辨率局部编码。
2×2 group 把序列缩短四倍，使 Qwen3 的 self-attention 成本可控。
#### 10.5.3 关键超参
27 层、1024 hidden、4096 FFN、12 head、最多 4096 patch。
#### 10.5.4 代码与论文差异
“merge”不是 spatial average。
代码的 `.view(item.shape[0],-1)` 保留四个 patch channel，见 `models/vlm/modeling_groundingpi_vision.py:815-822`。
只有 temporal pooling 路径对时间维聚合。
### 10.6 核心模块三：projector
#### 10.6.1 输入输出
输入每组 4096 维，输出 2560 维。
#### 10.6.2 为什么这样设计
同时完成局部空间融合与视觉—语言宽度对齐。
#### 10.6.3 关键超参
`4096→4096→2560`，GELU，两个 bias-free linear，RMSNorm epsilon $10^{-5}$。
#### 10.6.4 代码与论文差异
主图没有呈现 LayerNorm/RMSNorm 和 bias 细节。
源码和 Appendix Table 8 才给出完整结构。
### 10.7 核心模块四：多模态 Qwen3
#### 10.7.1 输入输出
输入是已替换 image placeholder 的 `[B,L,2560]` embedding。
输出 `[B,L,2560]` hidden 与 `[B,L,152670]` logits。
#### 10.7.2 为什么这样设计
把几何预测转为标准 next-token prediction，复用 LM 训练、generation、KV cache 和服务栈。
#### 10.7.3 关键超参
36 层、2560 hidden、9728 FFN、32Q/8KV、full attention。
#### 10.7.4 代码与论文差异
模型使用普通一维 RoPE，而非 Qwen-VL mRoPE。
SFT plugin 必须覆盖 Qwen2VL template 的默认 position 行为，否则会生成错误维度，见 `train/vlm/plugin.py:100-153`。
### 10.8 核心模块五：GAM protocol
#### 10.8.1 输入输出
输入结构化 `GroundingEntry` 或文本 prompt。
输出 canonical string，或严格 parser 的结构化 tuple。
#### 10.8.2 为什么这样设计
避免每个数据清洗、评测和 client 各写一套正则，降低协议漂移。
#### 10.8.3 关键超参
1000 网格、bbox arity 4、point/trajectory arity 2、entry separator `, `、tuple separator `,`。
#### 10.8.4 代码与论文差异
release parser 比论文示意图严格得多。
它检查 Unicode control character、reserved-token injection、重复 phrase、重复 tuple、canonical sort 和退化 box，见 `grounding_pi/protocol.py:220-327`。
### 10.9 核心模块六：tokenizer migration
#### 10.9.1 输入输出
输入基础 GroundingPI/Qwen3 checkpoint。
输出扩展到 152,670 行的 tokenizer、input embedding 和 LM head。
#### 10.9.2 为什么这样设计
输入输出 embedding 未绑定，所以必须分别扩容和初始化。
只改 tokenizer 而不改两张矩阵会导致 shape mismatch 或随机缺行。
#### 10.9.3 关键超参
基础 tokenizer 151669，基础 embedding row 151936，目标 row 152670，hidden 2560。
#### 10.9.4 代码与论文差异
基础 embedding 原先有不可寻址 padding row。
migration 会重用/重排其中一部分并严格验证固定 control token ID，这些工程细节论文未述。
### 10.10 核心模块七：SFT
#### 10.10.1 输入输出
输入已预处理 cache、完整模型、tokenizer manifest 与 YAML。
输出 full-finetuned checkpoint。
#### 10.10.2 为什么这样设计
基于 ms-swift 注册 model/template，复用 packing、distributed training 和 checkpoint 管理。
#### 10.10.3 关键超参
公开默认 `max_length=2048`、batch 2、world size 1、BF16、ZeRO-1。
#### 10.10.4 代码与论文差异
公开 plugin 只支持 image+text GAM training；video/audio 会报错，见 `train/vlm/plugin.py:84-98`。
公开 recipe 不含论文 Pretrain 1 三阶段，也不含 GRPO。
### 10.11 核心模块八：vLLM 服务
#### 10.11.1 输入输出
输入模型目录和 serving overlay。
输出 OpenAI-compatible API，默认模型名 `groundingpi`。
#### 10.11.2 为什么这样设计
overlay 复用原权重，只改兼容 vLLM Transformers backend 的代码文件，并校验 manifest 与 source model。
#### 10.11.3 关键超参
BF16、eager、TP=1、max model length 16384、max sequences 1、memory utilization 0.7、每请求一图、禁视频。
见 `infer/serve_vllm.py:39-80`。
#### 10.11.4 代码与论文差异
服务要求 bundled Transformers 5.7.0 fork 与 vLLM 0.18.x，见 `infer/serve_vllm.py:53-67`。
这不是“任意 pip 安装最新版 vLLM 即可”。
launcher 还会写 `engine_runtime.json`，但本文没有运行该流程。
### 10.12 核心模块九：34 项评测
#### 10.12.1 输入输出
输入 suite、数据路径、已有服务和 task recipe。
输出 raw response、cache、per-task metric 与 `summary.json`。
#### 10.12.2 为什么这样设计
统一服务协议和任务 registry，允许同一评测框架比较 GAM、VLM、Rex-Omni 等不同解析模式。
#### 10.12.3 关键超参
完整 suite 名为 `groundingpi34`。
release 默认模板只跑每 task 8 sample smoke test；完整评测要显式取消 limit。
#### 10.12.4 代码与论文差异
runner 只报告 per-task 结果，不计算论文 11 类 headline aggregate，也不会自动复现十次统计平均，见 `eval/README.md:157-170`。
registry 有 42 recipe，但论文只选 34。
### 10.13 核心模块十：VLA 下游
#### 10.13.1 输入输出
输入各类 pretrained foundation 的八层 hidden state。
输出统一 ActionDiT 的 flow velocity，再经四步 Euler 得到 action chunk。
#### 10.13.2 为什么这样设计
去掉各模型原生 action head，把 backbone feature quality 放到相同控制容量下比较。
#### 10.13.3 关键超参
8 depth×64 token×1024 width；16 atomic block；32 planning token；action horizon 16；4 Euler step。
#### 10.13.4 代码与论文差异
公开 `vla/` 是 StarVLA/OpenWAM comparison workflow。
官方明确说它不提供 released GroundingPI checkpoint 的直接 action-policy adapter，见 `README.md:350-357`。
因此论文中的 GroundingPI policy 不能从 HF grounding checkpoint 一条命令恢复。
### 10.14 至少十二条论文/代码差异与实现补充
1. 论文主文“2×2 aggregate”语义模糊；代码证明空间四 patch flatten，不平均。
2. 论文称 4B；权重索引和 Table 9 显示完整模型实际 4,843,664,384 参数。
3. 主文 SFT 描述“先全量更新再冻结视觉侧细化”；Appendix Table 10 的 Stage 4 列为 all modules，明确冻结发生在 RL。
4. 论文多阶段训练用 128 GPU、8192/32768 context；公开默认 SFT 是单卡、2048 context、global batch 2。
5. 论文包含 500.41B 基础训练和 221.17B 空间训练；仓库未提供完整多阶段数据与 launcher。
6. 论文详细定义 GroundingPI GRPO；仓库 release 没有该专属 RL recipe。
7. 论文 data engine 是核心贡献；仓库没有对应生产数据引擎。
8. 模型位置容量 262,144；公开 vLLM 默认只开放 16,384。
9. 模型视觉上限 1024 token；native serving profile 默认 `IMAGE_MAX_TOKEN_NUM=512`。
10. 论文 Stage 3 包含 video；公开 processor 与 SFT template 只支持 image。
11. 论文报告 34 项 headline aggregate；公开 evaluator 不自动计算 11 类总分。
12. 论文 GroundingPI 数字是十次均值；一次 full-suite 命令只产生一次 run。
13. 论文展示最终物理智能迁移；HF release 只给 grounding VLM，不给最终 GroundingPI policy checkpoint。
14. 仓库根 LICENSE 是 Apache-2.0；完整适用范围还受 Kimi K3 派生条件约束。
15. 论文 OCR RL reward 使用 Hungarian text-geometry 组合；公开 OCR benchmark parser/metric 不等同于训练 reward。
16. Figure 5 写 45 baselines；摘要和正文写 44。
17. 论文把 F 组视觉上简写成 OCR；Appendix 明确 F 同时含 OCR、layout、GUI。
18. 论文把 quantized speed 写成相对优势；代码/实验没有提供闭环 observation-to-action latency。
### 10.15 参数与显存计算账
模块参数：
| 模块 | 参数量 |
|---|---:|
| Vision encoder | 401,214,464 |
| Projector | 27,267,584 |
| Decoder excluding vocab matrices | 3,633,511,936 |
| Input embedding | 390,835,200 |
| Output head | 390,835,200 |
| Total | 4,843,664,384 |
BF16 单参数 2 byte，纯权重大小为：
$$4,843,664,384\times2=9,687,328,768\ \mathrm{bytes}$$
这与官方 `model.safetensors.index.json` metadata 完全一致。
纯权重约 9.02 GiB。
实际推理还需 KV cache、activation、workspace、视觉 tensor 和框架开销。
全量 AdamW 训练若保存 FP32 master weight、gradient 和两阶 moment，理论状态内存远高于纯 BF16 权重，必须使用 ZeRO 与 activation recomputation。
视觉 token 从 4096 patch 降到最多 1024 后，语言 self-attention 的视觉序列部分理论二次项相对未 merge 序列约降为 $1/16$，但 projector 和文本 token 成本仍存在。
### 10.16 最小复现路径
以下是基于官方文档整理的“建议路径”，本文未执行：
1. checkout 官方 commit `090932430bbbeea004cfa461767ede7f563c0f3a`；
2. 下载 HF revision `08d397c8769f7f10b96ff6d56ee6f56acb04181f` 到 `weights/vlm`；
3. 核对 tokenizer、custom model code、processor、weight index 和 manifest；
4. 按 `README.md:84-109` 准备对应 Linux/Python/accelerator runtime；
5. 用 `configs/release/vlm_vllm_gpu.yaml` 启动单图 vLLM 服务；
6. 请求中设置 `skip_special_tokens=false` 与 `spaces_between_special_tokens=false`；
7. 先用单图 GAM prompt 验证 canonical output；
8. 下载 Grounding-EvalData 并执行 deep data check；
9. 用 `groundingpi34` 和 `limit:null` 跑完整 suite；
10. 自行按论文 11 capability 规则聚合，而不是把 34 项直接平均；
11. 若做 SFT，明确这是 release recipe，不是论文全流程；
12. 若做 policy transfer，需要独立准备 StarVLA/OpenWAM 环境、动作数据和 policy checkpoint。
## 11. 局限与评价
### 11.1 评测协议并非完全同质
表格混合 external published score、本地三次 baseline run、自有模型十次 run 和 frontier API 设置。
任务 parser 兼容性也导致某些模型 N/A。
这不使结果失效，但降低了“所有模型严格同场竞技”的强度。
### 11.2 44/45 baseline 计数不一致
Figure 5 与摘要/正文有明确数字冲突。
论文应给出 baseline 名单和计数规则，避免宣传口径不一致。
### 11.3 backbone comparison 不能隔离 grounding 因果
固定下游 ActionDiT 是有价值的控制。
但 backbone 是一个包含 architecture、scale、data、objective 的整体 treatment。
真正隔离 grounding 需要在同一个 base model、同一 token budget、同一数据来源下，只改变 grounding supervision。
### 11.4 驾驶证据较弱
nuScenes 仅 open-loop。
0.296 对 0.301 的绝对提升为 0.005 m。
没有 closed-loop collision、route completion、comfort、latency 和 intervention 指标。
论文也没有隔离 OCR 读路牌对驾驶结果的贡献。
### 11.5 仿真操作缺少真机
RoboTwin 与 RoboCasa 都是 simulation。
随机纹理和对象组合不能覆盖真实视觉噪声、摩擦、标定漂移、遮挡和执行误差。
论文 Ethics Statement 也明确没有部署真实车辆或机器人。
### 11.6 73.68 的权重选择具有主观性
11 类等权强调能力广度。
但机器人抓取、GUI、OCR、layout 和 visual prompting 对物理系统的价值不可能天然相等。
更好的报告方式应同时给出 macro、任务族权重敏感性和下游相关加权分数。
### 11.7 数据与泄漏透明度不足
没有完整训练数据表。
没有公开 data engine。
没有完整去重、近重复、benchmark contamination 和 teacher memorization 审计。
在 721.58B token 规模下，这一缺口会直接影响 benchmark 解释。
### 11.8 scaling 只有四个点
Grounding 与 OOD 单调上升，但 ID 最后回落。
没有 error bar、重复 seed 或 compute-optimal 控制。
所以只能描述 trajectory，不能声称普适 scaling law。
### 11.9 composition 不是完整 factorial design
六组全 factorial 需要 $2^6=64$ 种组合，论文只评估少数组合和部分 leave-one-out。
缺少 A-only、B-only 和完整 pairwise interaction。
F 还把 OCR/layout/GUI 合成一组。
因此不能从 5.33 pp 直接得出 OCR 独立贡献 5.33 pp。
### 11.10 action-data efficiency 不等于总效率
论文减少的是 downstream demonstration fraction。
没有把 721.58B token、128/256 GPU 训练和数据引擎成本纳入效率账。
这一结果适合表述为“动作示范效率”，不适合表述为“总体数据/计算效率”。
### 11.11 没有端到端延迟
文本坐标 0.25× speed 和 tokens/box 都是局部 generation 指标。
物理闭环还包括图像传输、preprocess、ViT、prefill、decode、policy denoise 和执行接口。
论文没有给 observation-to-action latency 或控制频率。
### 11.12 “4B”实际约 4.844B
4B 容易让读者只按 Qwen3 language family 理解规模。
完整模型还含 401M vision、27M projector 和两张 391M vocab matrix。
做参数效率比较时应使用 4,843,664,384，而不是简单写 4.0B。
### 11.13 总体评价
**方法价值：高。** 把空间协议、数据引擎、SFT、RL 和下游 transfer 连接成完整研究问题。
**实验覆盖：很高。** 34 entry、11 capability、多个下游和多组消融，远超只报几张表的 grounding 工作。
**因果强度：中等。** 下游控制充分，但上游 backbone 与数据因素没有完全解耦。
**开源完整度：中等偏高。** 权重、推理、SFT 与评测很扎实；真正昂贵和新颖的 pretraining/data engine/GRPO/policy 没有完整开放。
**复现难度：高。** 单模型推理可复现，论文训练与 policy 结果难以端到端复现。
## 12. 与旭涛研究方向的联系/可借鉴思路
### 12.1 把“哪里”作为动作模型的预训练接口
如果旭涛关注 VLA、机器人感知或 action representation，最直接的借鉴不是复制 721B token 训练，而是验证：
- 在同一 VLM 上加入 box/point supervision；
- 保持动作头与动作数据不变；
- 比较 ID、scene OOD、entity OOD 与 limited-demo adaptation。
这种设计比直接换整个 foundation 更容易做因果归因。
### 12.2 建立视觉基元到 action token 的中间层
GroundingPI 的 point/box 是稀疏、可解释、语言条件化的中间变量。
可以研究将其用于：
- target selection；
- affordance proposal；
- keypoint-conditioned diffusion policy；
- object-centric memory；
- failure localization；
- human correction interface。
关键是不要把 box 当作完整物理状态。
接触、深度、姿态、可达性和动力学仍需额外表征。
### 12.3 借鉴统一 backbone-to-action 控制协议
八层归一化深度抽样、共享 projector、64-token resampler 和固定 ActionDiT 是很好的 backbone comparison template。
旭涛可以进一步增强实验：
- 同时报告 frozen-backbone 与 end-to-end adaptation；
- 比较单层、八层和 learnable layer selection；
- 控制 projector 参数量；
- 测量 condition cache 的延迟和显存；
- 用 representation probing 检查空间信息在哪些层最可读。
### 12.4 把 dense/tiny 作为物理场景压力测试
真实操作中的小按钮、细柄、遮挡边缘和密集零件与 Dense200/VisDrone 有共同难点。
可以构建 action-centric dense benchmark：
- 多个相似实例中选择目标；
- 像素级小幅扰动导致抓取点改变；
- 遮挡和背景纹理随机化；
- object identity 与 location 分别变化；
- 分离 perception failure 与 control failure。
### 12.5 验证 OCR 催化假说
最关键的对照不是“有 OCR vs 无 OCR”，而是：
1. geometry-only OCR box；
2. transcription-only 或 masked transcription loss；
3. full text+geometry；
4. 相同 token budget 的非 OCR local-caption control；
5. 相同字符边界但打乱文字语义。
再测 non-text dense grounding、ViT local feature、gradient alignment 和下游操作。
这能区分“局部高频视觉监督”“区域语义绑定”和“额外数据量”三个机制。
### 12.6 从 open-loop 走向 closed-loop
驾驶方向应加入：
- closed-loop route completion；
- collision 与 off-road；
- comfort；
- sign/light-specific subgroup；
- observation-to-action latency；
- perception perturbation robustness；
- uncertainty-triggered fallback。
机器人方向应加入真机、相机标定漂移、动态遮挡与恢复行为。
### 12.7 数据工程可复现性
若资源有限，可实现小规模、透明版本的数据引擎：
- 每个字段记录 teacher provenance；
- 保留 accepted/rejected/unresolved reason；
- 对局部 crop 保存坐标映射；
- 做 benchmark hash 和 embedding near-duplicate audit；
- 报告人工抽检置信区间；
- 对每个版本记录标签增删与依赖传播。
透明度可能比单纯扩充伪标签量更有研究价值。
## 13. 关键术语
### 13.1 Grounding foundation model
以语言条件化空间定位为核心能力的基础模型，输出 box、point、文字区域或其他视觉基元。
### 13.2 Visual primitive
可被动作和推理模块复用的基础空间单元；本文主要是 box 与 point。
### 13.3 GAM
GroundingPI 使用的原生 spatial-token protocol，将标签 wrapper、box wrapper、`None` 和 0–999 坐标 token 组织为 canonical sequence。
### 13.4 Coordinate alignment
把语言词表中的离散坐标 token 与图像位置对齐的 Stage 4 SFT。
### 13.5 Field fusion
不在整条 annotation 级二选一，而是分别融合语义、box、mask、属性等字段。
### 13.6 Coverage validation
对“找全所有实例”的任务检查是否覆盖完整目标集合，而不只检查已有预测是否合法。
### 13.7 R_set
面向集合完整性的 grounding reward，通过 reference-wise 最大 IoU 与类别验证构造 soft precision/recall/F1。
### 13.8 R_strict
结合多阈值 F1、matched IoU、格式、数量、顺序和惩罚项的严格 grounding reward。
### 13.9 Hungarian matching
在 OCR prediction 与 reference 之间做最大权一对一分配，防止一个 prediction 重复匹配多个文字实例。
### 13.10 ActionDiT
基于 diffusion/flow matching 的 Transformer action expert；本文统一使用 16 atomic block、八 cross、八 self。
### 13.11 Normalized-depth tap
按相对深度而非绝对层号从不同 backbone 抽取八层 hidden state。
### 13.12 Action-data efficiency
固定预训练 foundation 与动作 recipe，只减少下游 demonstration fraction 后的性能；不等于总体训练效率。
### 13.13 Open-loop L2
预测轨迹与记录轨迹的离线欧氏误差，不包含预测对环境状态的反馈影响。
### 13.14 Macro capability average
对能力类别先独立聚合，再对 11 类等权平均；与 benchmark micro average 不同。
## 14. 图表索引
### 14.1 本笔记引用的八张图
| 本地文件 | 论文图号 | PDF 页 | 内容 |
|---|---|---:|---|
| `figures/01-teaser.png` | Figure 1 | 1 | 模型定位、11 类能力与物理迁移总览 |
| `figures/02-architecture.png` | Figure 2 | 3 | MoonViT-V2、2×2 merge、projector、Qwen3 与 GAM |
| `figures/03-data-engine.png` | Figure 3 | 3 | 多教师、字段融合、验证、补证与迭代 |
| `figures/04-training-pipeline.png` | Figure 4 | 6 | SFT 与 GRPO 两阶段后训练 |
| `figures/05-capability-wheel.png` | Figure 5 | 8 | 11 类 capability 与强 baseline 对比 |
| `figures/06-physical-transfer.png` | Figure 6 | 11 | RoboTwin、RoboCasa、nuScenes |
| `figures/07-scaling.png` | Figure 9 | 14 | 88.4B–221B 感知和动作 scaling |
| `figures/08-data-composition.png` | Figure 10 | 15 | A–F 数据组成与 leave-one-out |
### 14.2 关键论文表
Table 1：detection 与 referring grounding。
Table 2：robot/spatial pointing 与 GUI grounding。
Table 3：OCR 与 document layout。
Table 4：object pointing。
Table 5：output token efficiency。
Table 8：architecture 与 visual processing。
Table 9：完整参数量。
Table 10：Pretrain 1/2 训练配置。
Table 11：RL 配置。
Table 12–13：grounding/OCR reward 组成。
Table 14–15：统一 ActionDiT 与动作训练预算。
Table 16：四点 scaling 数字。
Table 17：数据组消融。
Table 18：OCR mixing。
## 15. 材料来源
### 15.1 一手材料
论文 arXiv 页面：<https://arxiv.org/abs/2609.39601>。
官方 PDF v2：<https://arxiv.org/pdf/2609.39601v2>。
项目页：<https://groundingpi.github.io/>。
官方 GitHub：<https://github.com/groundingpi/GroundingPI>。
官方 Hugging Face：<https://huggingface.co/GroundingPI/GroundingPI>。
官方 ModelScope：<https://www.modelscope.ai/models/GroundingPI/GroundingPI>。
代码固定版本：`090932430bbbeea004cfa461767ede7f563c0f3a`。
HF 固定 revision：`08d397c8769f7f10b96ff6d56ee6f56acb04181f`。
获取与核对日期：2026-10-27。
### 15.2 本地材料
论文抽取：`groundingpi-staging/source.md`。
官方 PDF：`groundingpi-staging/source.pdf`。
官方代码快照：`/tmp/groundingpi-code-20261027`。
HF 配置与 API 快照：`assets/GroundingPI - 视觉基元物理智能感知底座/sources/`。
详细来源、原始 URL、图像 provenance、实际阅读源码和开源边界见同资产目录的 `sources.md`。
### 15.3 开源覆盖结论
已公开：官方代码、GroundingPI 权重、tokenizer、processor、custom model code、native Transformers 推理、vLLM 服务、SFT recipe、34 项评测配置与指标、StarVLA/OpenWAM backbone comparison workflow。
未完整公开：500.41B+221.17B 多阶段预训练数据与训练流程、生产 data engine、GroundingPI 专属 GRPO、完整训练语料与泄漏审计、可直接加载 GroundingPI checkpoint 的最终 action policy、论文 policy checkpoint。
因此本笔记将其评为“推理与评测复现较强，论文训练与物理策略端到端复现不足”。
