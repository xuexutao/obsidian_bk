# GigaBrain-0：世界模型驱动的 VLA 数据引擎

## 1. 背景

**一句话判断：** 这篇论文的核心价值，不是单纯再做一个 VLA backbone，而是把 **world model 前移为 VLA 的数据引擎**，用可规模化生成的数据去缓解真实机器人数据昂贵、单一、难扩展的问题。

**重要性评估：** ★★★★☆（4/5）

**主领域归档建议：** VLA

**为什么值得关注：**

- 它把真实机器人数据瓶颈，转写成“真实数据 + 多种 world-model 生成数据”的混合训练问题。
- 它不是只追求单任务高分，而是同时关注 **长程任务、灵巧操作、移动操作、外观泛化、摆放泛化、视角泛化、端侧部署**。
- 论文还给出一个轻量版 `GigaBrain-0-Small`，显示这条路线不只是研究 demo，也在尝试落到端侧机器人控制。

**原始出处：**

- [arXiv 论文页](https://arxiv.org/abs/2510.19430)
- [官方 PDF](https://arxiv.org/pdf/2510.19430)
- [项目页](https://gigabrain0.github.io)

这篇内容讨论的是一套名为 **GigaBrain-0** 的 Vision-Language-Action 模型。作者认为，现有通用机器人 VLA 的主要瓶颈并不完全在模型结构，而在 **训练数据规模、场景多样性与采集成本**。因此论文的目标是：在不无限增加真实机器人采集量的前提下，**<span style="color: red">用 world model 生成的大规模、多样化、具有物理可行性的训练数据</span>**，去提升 VLA 的跨任务与跨环境泛化能力。

---

## 2. 文章主线 / 论文线索

**主论文：**

- **GigaBrain-0: A World Model-Powered Vision-Language-Action Model**

**主线可以拆成三层：**

1. **数据层：** 用 GigaWorld 生成多种增强数据，而不是只依赖真实机器人轨迹。
2. **模型层：** 用 `预训练 VLM + 动作生成专家` 的混合架构，把语义理解和连续动作生成解耦。
3. **监督层：** 通过 **RGBD 输入** 和 **Embodied CoT**，强化空间理解与长程推理能力。

**论文里真正的新意，不只是“用了生成数据”，而是把生成数据做成一个系统性的 recipe：**

- Real2Real transfer：把真实轨迹重渲染成不同纹理、材质、光照与颜色。
- View transfer：在保任务语义的前提下，重建新相机视角。
- Sim2Real transfer：从仿真场景生成轨迹，再做真实化迁移。
- Human video transfer：把第一视角人类操作视频转成机器人可执行演示。
- Video generation + IDM：从单帧与文本提示生成未来操作视频，再反推动作序列。
- Multiview generation：生成跨视角一致的视频，用于提升 3D-aware 训练效果。

**论文最强主张**

- world model 可以作为 **VLA 数据扩展器**
- RGBD 能补足空间几何感知
- Embodied CoT 能提升长程任务执行
- 小模型也能做到端侧实时控制

**论文仍未完全展开的部分**

- GigaWorld 本身的模型与训练细节留待后续 technical report
- 多种生成数据源各自的独立贡献没有做充分 ablation
- 与更多开源/闭源 VLA 的系统对比还不够充分

---

## 3. Pipeline / Architecture + I/O 数据流

![](assets/GigaBrain-0%20-%20世界模型驱动的VLA数据引擎/framework.png)

### 3.1 总体数据流

**一句话 Pipeline：** `真实/生成混合数据` → `VLM 编码 RGBD + 指令 + 状态` → `Embodied CoT 中间推理` → `DiT 预测连续 action chunks` → `机器人执行动作`

### 3.2 输入 / 中间表示 / 输出

|阶段|内容|技术要点|
|---|---|---|
|输入|论文完整模型：RGBD、指令、状态、可选轨迹监督；公开 base：RGB、指令、状态|代码接口是 $K$ 路 `[B,C,224,224]` 图像列表。完整模型可令 `C=4`；公开 checkpoint 固定 `C=3`，且关闭轨迹 token。状态经离散化后进入最长 200 的语言 prefix。|
|视觉编码|SigLIP / PaliGemma2|SigLIP 每图输出 256×1152，再投影为 256×2304；3 路相机默认得到 768 个视觉 token。4 通道模式扩展 patch conv，新增深度核零初始化。|
|中间推理 1|Manipulation Trajectory|代码支持 10 个 learnable trajectory tokens，经 GRU 回归 `[B,10,4]`，每点是左右手 `(x_l,y_l,x_r,y_r)`；公开 base 的 `enable_learnable_traj_token=false`。|
|中间推理 2|Subgoal Language|作为 prefix 中受 loss mask 监督的 next-token prediction；可单独自回归生成当前子目标。|
|中间推理 3|Discrete Action Tokens|FAST action token 可与 task/subgoal 一起进入最长 200 的 prefix，并提供 CE 监督；连续动作仍由 suffix flow expert 生成。|
|动作生成|Continuous action chunks|`[B,50,32]` noisy action 经 1024-D action expert；PaliGemma2 prefix 与 action suffix 每层共享 attention。KI 开启时 flow 只读 detach 后的 prefix KV。|
|输出|连续动作、可选语言/轨迹|flow head 输出 `[B,50,32]`，反归一化并裁回本体 $D_a$；可选 language logits 与 10×4 trajectory。policy 不在线生成未来视频。|

### 3.3 模型结构拆解

**结构上是一个“语义专家 + 动作专家”的混合体：**

- **Vision-Language Expert：** 负责理解 RGBD 场景、语言指令、状态与 Embodied CoT。
- **Action Expert：** 负责连续动作 chunk 的扩散式生成。
- **Knowledge Insulation：** 用于隔离离散语义推理流与连续动作优化流，减少互相污染。

### 3.4 监督目标

论文联合优化三类目标：

1. **CoT token 的 next-token prediction**：监督 subgoal language 与 discrete action token。
2. **动作 flow matching loss**：监督连续动作 chunk 的生成。
3. **轨迹回归损失**：监督 10 个 2D 轨迹关键点。

### 3.5 数据引擎是怎么工作的

**真实数据部分：**

- 公共数据：AgiBotWorld、RoboMind、Open X-Embodiment。
- 自采数据：来自 Agilex Cobot Magic 与 AgiBot G1，合计 **1182 小时**，覆盖 **3100 平方米**、**14 个真实场景**，包含工业、商业、办公、家庭、实验室等环境。

**标注与处理：**

- 无深度时，用 `MoGe` 估计 metrically scaled depth。
- 长任务的 subgoal 不是纯手工切，而是按 gripper state transition 自动切分，再用 `Qwen-VL-2.5` 生成 subgoal language。
- 2D manipulation trajectory 通过 3D 末端轨迹投影到头戴相机图像平面获得。
- 同一任务最多保留 **50 条**具有多样性的 demonstration，用于去重与提效。

**生成数据部分：**

- **Real2Real**：固定动作语义，改变纹理/材质/颜色/光照。
- **View Transfer**：基于深度重投影到新视角，再用视频补全模型修补遮挡区域。
- **Sim2Real**：在 Isaac Sim 中构造场景，基于 EmbodiedGen / ArtVIP 等资产生成仿真轨迹，再做真实化迁移。
- **Human Video Transfer**：把 EgoDex 第一视角人类手部操作，转成机器人手臂执行版本。
- **Video Generation + IDM**：从单帧和文本提示合成操作视频，再用 Inverse Dynamics Model 反推机器人动作。
- **Multiview Video Generation**：生成多视角一致视频，提升跨视角训练效果。

![](assets/GigaBrain-0%20-%20世界模型驱动的VLA数据引擎/real2real.png)

![](assets/GigaBrain-0%20-%20世界模型驱动的VLA数据引擎/viewtransfer.png)

![](assets/GigaBrain-0%20-%20世界模型驱动的VLA数据引擎/sim2real.png)

![](assets/GigaBrain-0%20-%20世界模型驱动的VLA数据引擎/human_transfer.png)

![](assets/GigaBrain-0%20-%20世界模型驱动的VLA数据引擎/videogen.png)

![](assets/GigaBrain-0%20-%20世界模型驱动的VLA数据引擎/multiview.png)

> 补注：上图为 Multiview Video Generation（论文图 9），生成多视角一致视频，用于提升 3D-aware 训练效果。

### 3.6 官方代码级实现：发布版策略的真实 forward

> **版本边界与证据标记**：初代 GigaBrain-0 固定到 `open-gigaai/giga-brain-0@19328ff7d16e5047b7401a00aa6677d08dc160c7`（2025-11-28 首次发布附近），其实际模型实现来自官方依赖 `open-gigaai/giga-models@b5e69a53e4b6efa965b615cc648d229a9f4c7413`；发布权重固定到 `open-gigaai/GigaBrain-0-3.5B-Base@6b169fe5d01353a91974a0cf418de30bb04c9394`。当前主仓库 HEAD 已演进为 GigaBrain-0.7，不能拿 0.7 三系统代码冒充本文初代实现。以下使用【源码事实】、【配置推导】、【论文描述】、【无法确认】标记证据层级。

#### 3.6.1 模块到代码入口表

| 模块 | 官方入口 | 作用 |
|---|---|---|
| 训练入口 | `giga_brain_0/giga_brain_0_trainer.py::GigaBrain0Trainer.forward_step` | 取 batch、加 flow 噪声、调用 policy、汇总三类 loss |
| 数据变换 | `giga_brain_0/giga_brain_0_transforms.py::GigaBrain0Transform` | delta action、分位数归一化、32-D padding、图像/文本/轨迹变换 |
| 策略主入口 | `giga_models/models/vla/giga_brain_0/modeling_giga_brain_0.py::GigaBrain0Policy.forward` | prefix/suffix 构造、attention mask、双专家前向、输出动作/语言/轨迹 |
| 视觉与双专家主干 | `paligemma2_with_expert.py::PaliGemma2WithExpertModel` | 27 层 SigLIP；26 层 PaliGemma2 + action expert 联合 attention |
| Flow loss | `giga_brain_0/giga_brain_0_loss.py::GigaBrain0Loss` | Beta timestep、线性 flow path、动作 MSE、文本 CE、轨迹 MSE |
| 推理入口 | `pipeline_giga_brain_0.py::GigaBrain0Pipeline.__call__` | 预处理、10-step Euler 去噪、反归一化、delta→absolute、裁回原动作维 |
| 默认配置 | `configs/giga_brain_0_from_scratch.py` | 3 相机、224²、50×32 action、L≤200；深度与轨迹默认关闭 |

#### 3.6.2 原始输入、预处理与默认数字

【源码事实】默认发布链路接收最多 3 路相机 `cam_high/cam_left_wrist/cam_right_wrist`，每路原始 `[3,H_0,W_0]` 经等比缩放和 padding 变成 `[3,224,224]`，再归一化到 `[-1,1]`。若打开 RGBD，深度会与 RGB 沿 channel 拼成 `[4,224,224]`，缺失/随机丢弃深度填 0；训练器会把 SigLIP patch conv 从 3 通道扩到 4 通道，新通道权重为 0。

【源码事实】机器人 state/action 先按 embodiment 做 delta 变换与 1%/99% 分位数归一化，再从原始 $D_s,D_a$ 补零到 32-D：`state=[B,32]`、`action=[B,50,32]`。默认 `discrete_state_input=true`，所以 state 不作为独立连续 suffix token，而是由 `PromptTokenizerTransform` 编进最长 200 的语言/FAST token 序列。`embodiment_id∈{0,1,2}` 选择动作输入、输出两端各自的 embodiment-specific linear。

【源码事实】发布权重的官方 `config.json` 是：`vlm_type=paligemma2`、`vlm_hidden_size=2304`、`proj_width=1024`、`n_action_steps=50`、`max_action_dim=max_state_dim=32`、`num_steps=10`、`vision_in_channels=3`、`enable_knowledge_insulation=true`、`enable_next_token_prediction=true`、`enable_learnable_traj_token=false`。因此论文所述 RGBD 与 10 点轨迹是**代码支持/论文训练配方**，不是公开 base checkpoint 的默认输入输出。

#### 3.6.3 按真实 forward 顺序的全链路张量表

设相机数为 $K≤3$，文本/离散 state/可选 FAST 或 subgoal 合并后的有效长度为 $L≤200$；发布 checkpoint 取 $K=3$、RGB、无 trajectory token。

| 顺序 | 张量与 shape | 变化原因与证据 |
|---:|---|---|
| 1 | 每路 RGB `[B,3,H_0,W_0] → [B,3,224,224]` | 【源码事实】`ImageTransform` 等比 resize+pad、训练增强、归一化到 `[-1,1]` |
| 2 | 可选 RGBD `[B,4,224,224]` | 【源码事实】`enable_depth_img=true` 时拼深度；【配置事实】公开 base 为 3 通道，故此步默认不发生 |
| 3 | raw state/action → `[B,32]`、`[B,50,32]` | 【源码事实】delta/quantile normalize 后 `PadStatesAndActions(32)`；时间 padding 另由 `[B,50]` mask 控制 |
| 4 | prompt/state/FAST/subgoal → token `[B,L]`，pad 到 `[B,200]` | 【源码事实】不同 sample recipe 决定哪些位置产生 next-token loss |
| 5 | SigLIP patch embedding `[BK,3,224,224] → [BK,256,1152]` | 【源码事实】patch 14，16×16=256 token，vision hidden 1152，27 层 |
| 6 | multimodal projector `[BK,256,1152] → [BK,256,2304]` | 【源码事实】把视觉宽度对齐 PaliGemma2 hidden 2304 |
| 7 | reshape/concat → image prefix `[B,K·256,2304]` | 【源码事实】先批量编码全部相机，再恢复相机轴并沿 token 轴拼接 |
| 8 | language embedding `[B,200,2304]` | 【源码事实】token embedding 乘 `sqrt(2304)`；state 若离散编码已包含在这 200 个位置中 |
| 9 | 可选 trajectory token `[B,10,2304]` | 【源码事实】仅 `enable_learnable_traj_token=true` 时插在图像和语言之间；发布 base 为 false |
| 10 | prefix `[B,K·256(+10)+200,2304]` | 【源码事实】图像、可选轨迹、语言顺序拼接；有效位由 pad mask 决定 |
| 11 | clean action 加噪：$x_t=t\epsilon+(1-t)a∈R^{B×50×32}$ | 【源码事实】$t∼0.999·Beta(1.5,1)+0.001$，target $u_t=\epsilon-a$ |
| 12 | embodiment action projection `[B,50,32] → [B,50,1024]` | 【源码事实】不同 embodiment 选不同输入权重；flow time 经两层 MLP 得 AdaRMS 条件 `[B,1024]` |
| 13 | suffix `[B,50,1024]` | 【源码事实】50 个动作步各一个 token；动作块内部共享同一 attention block |
| 14 | prefix/suffix QKV → 各自 8 heads×256 | 【源码事实】prefix residual width 2304、suffix width 1024，但二者独立投影到相同 attention head 几何 |
| 15 | Q/K/V 沿 sequence concat，在 26 层做联合 attention | 【源码事实】attention 空间融合；输出再按 prefix/suffix 长度切开，各走 2304/1024 宽度的独立 MLP |
| 16 | mask：prefix 内按图文/语言块规则；prefix 不看 suffix；suffix 看 prefix 和整个 action block | 【源码事实】suffix 的 block mask 为 `[1,0,…,0]`，故 50 个 action token 同组双向可见，而非逐动作自回归 |
| 17 | Knowledge Insulation：prefix KV prefill 后 detach | 【源码事实】公开 base 开启 KI；flow suffix 可读 prefix KV，但 flow 梯度不回传到 VLM prefix |
| 18 | action head `[B,50,1024] → v_t=[B,50,32]` | 【源码事实】embodiment-specific output linear，监督目标为 `ε-a` |
| 19 | language head `prefix[-200:] → [B,200,V]` | 【源码事实】仅 loss mask 标记的 subgoal/FAST/普通语言位置算 next-token CE |
| 20 | 可选 traj：10 token → GRU 256 → `[B,10,4]` | 【源码事实】每点 4-D 是左右手 `(x_l,y_l,x_r,y_r)`；发布 base 未启用 |
| 21 | 推理从 `N(0,I) [B,50,32]` 做 10 次 Euler，`dt=-0.1` | 【源码事实】从 $t=1$ 噪声走到 $t=0$ 数据；prefix KV 只算一次 |
| 22 | 反归一化、delta→absolute、slice → `[B,50,D_a]` | 【源码事实】最后裁回机器人原始动作维；32-D 只是跨本体模型槽位 |

#### 3.6.4 代码—论文差异、开源边界与复现建议

- **World model 不在 policy forward 内**：【论文描述】GigaWorld 负责生成训练数据；【源码事实】`GigaBrain0Policy.forward` 只接图像、语言/离散状态和 noisy action，没有视频 latent，也不输出未来视频。不能把它写成在线 world-model planner。
- **公开 base 没有 RGBD/轨迹头**：【源码事实】HF revision 明确 `vision_in_channels=3`、`enable_learnable_traj_token=false`；README 也说明发布权重为了易用排除了 depth 与 2D trajectory。笔记前文的 RGBD/轨迹应标成论文完整模型，而非公开 checkpoint 默认行为。
- **初代仓库不是单仓自包含**：【源码事实】训练器只 import `GigaBrain0Policy`，真正主干在官方 `giga-models`；数据和 trainer 还依赖 `giga-datasets/giga-train`。复现时必须固定四个仓库/包版本，不能只 pin 主仓库。
- **`from_scratch` 名称误导**：【源码事实】该配置实际写有 `pretrained='open-gigaai/GigaBrain-0-3.5B-Base'`，默认是从发布 base 继续训练；若真从 PaliGemma2 初始化，需走 `pretrained_paligemma_path` 分支。
- **tokenizer 配置有历史不一致**：【源码事实】初版 config 写 `google/paligemma-3b-pt-224`，但 HF 模型配置是 `vlm_type=paligemma2`；后续官方配置已改为 PaliGemma2。复现初代权重应优先服从 checkpoint config 并显式校验 vocab/token IDs。
- **动作 padding 需要 mask 审计**：【源码事实】初版 flow loss 只有时间 mask，没有逐维 action mask；32-D 尾部零 padding 也会进入 MSE。若新增本体的 $D_a<32$，应确认该行为是否符合预期，或采用后续官方实现的 `[B,T,32]` dimension mask。
- **无法确认项**：【无法确认】论文 GigaWorld 的完整生成 backbone、质量筛选、多视角一致性训练和 IDM 细节仍未随初代 policy 代码完整开放；公开 repo 能复现 policy/post-training，不能据此宣称已经复现论文的数据引擎。

---

## 4. 实验与关键信息

![](assets/GigaBrain-0%20-%20世界模型驱动的VLA数据引擎/main_exp.png)

### 4.1 主实验设置

论文在两类机器人平台上做实机验证：

- **PiPER 双臂平台**
- **AgiBot G1 平台**

覆盖三类任务：

- **灵巧操作**：laundry folding、paper towel preparation
- **长程任务**：juice preparation、table bussing
- **移动操作**：boxes moving、laundry baskets moving

**各任务微调数据与训练配置：**

- Laundry Folding：300 条 demonstration，batch size 128，40K steps
- Paper Towel Preparation：100 条 demonstration，batch size 128，20K steps
- Table Bussing：100 条 demonstration，batch size 128，20K steps
- Juice Preparation：489 条 demonstration，batch size 128，35K steps
- Boxes Moving：300 条 demonstration，batch size 128，30K steps
- Laundry Baskets Moving：378 条 demonstration，batch size 192，30K steps

对比基线主要是：

- **π0**（使用官方开源代码，并采用相同微调配置）

### 4.2 六个实机任务结果

> 下表数值为根据论文主结果图直接读取的 **近似值**；论文正文只明确写出相对提升幅度，没有单独给出完整数字表。

|任务|π0|GigaBrain-0|解读|
|---|---|---|---|
|Laundry Folding（G1）|约 50%|约 60%|复杂可变形物体操作中有明显提升，说明 RGBD + 生成数据有助于空间理解与接触动作稳定性。|
|Paper Towel Preparation（PiPER）|约 20%|约 50%|提升最明显之一，说明在精细且步骤受约束的灵巧任务中，Embodied CoT 与深度信息的帮助较大。|
|Juice Preparation（G1）|约 90%|约 90%|两者都已较高，说明该任务可能接近当前数据规模下的饱和区。|
|Table Bussing（PiPER）|约 65%|约 90%|长程任务提升非常显著，契合论文关于 subgoal language 与 structured reasoning 的主张。|
|Boxes Moving（G1）|约 80%|约 90%|移动与操作结合场景中继续领先，说明统一 VLA 对全局导航 + 局部交互衔接有效。|
|Laundry Baskets Moving（PiPER）|约 20%|约 30%|有提升，但绝对成功率仍不高，说明这类重物/大物体搬运仍然困难。|

### 4.3 泛化实验：生成数据混合比例 α 的影响

![](assets/GigaBrain-0%20-%20世界模型驱动的VLA数据引擎/generalization_exp.png)

论文单独考察了 **world model 生成数据在训练 batch 中的采样概率 α**。这部分很关键，因为它直接回答了“生成数据到底有没有帮助，以及混多少最合适”。

|实验|0%|25%|50%|75%|90%|结论|
|---|---|---|---|---|---|---|
|Laundry Folding 新外观|约 24%|约 43%|约 67%|约 75%|约 83%|仅用白色衣物真实数据几乎不够；加入 Real2Real 生成数据后，颜色/纹理泛化明显提升。|
|Table Bussing 新摆放|约 25%|约 45%|约 71%|约 91%|约 89%|Sim2Real 补充的摆放多样性非常有效，最佳点大约出现在 75% 左右。|
|Table Bussing 新视角|约 33%|约 57%|约 75%|约 81%|约 87%|View transfer 对跨视角鲁棒性帮助很大，且整体呈现随 α 上升而持续收益的趋势。|

### 4.4 端侧部署结果

`GigaBrain-0-Small` 使用更小的 `SmolVLM2` 和约 **100M action expert**，论文给出的整机对比如下：

|模型|FLOPs|参数量|推理显存|推理时延|成功率|
|---|---|---|---|---|---|
|π0|4400 GFLOPs|3.2B|17.5 GB|1.28 s|80%|
|GigaBrain-0-Small|840 GFLOPs|402M|1.9 GB|0.13 s|80%|

这个结果很有代表性：

- 计算量从 **4400 GFLOPs** 降到 **840 GFLOPs**。
- 参数量从 **3.2B** 降到 **402M**。
- 显存从 **17.5GB** 降到 **1.9GB**。
- 时延从 **1.28s** 降到 **0.13s**。
- 成功率仍保持 **80%**。

也就是说，作者证明了这条路线不只服务大模型训练，也有机会进入边缘机器人部署场景。

### 4.5 我认为最重要的结论

1. **world model 生成数据确实能显著提高 VLA 泛化能力。**
2. **收益并不只在外观泛化，也覆盖摆放变化、相机视角变化和长程操作。**
3. **RGBD + Embodied CoT** 让模型不只是“看懂”，而是更像在做具身任务分解与空间推理。
4. **小模型版结果非常关键**，说明未来这条路线可能走向“训练时重、部署时轻”的工程范式。

---

## 5. 个人评注 / 下一步

**我对这篇论文的定位：** 它不是最纯粹的 world model 论文，也不是最纯粹的 VLA 架构论文；它更像是一篇 **“world model 作为 VLA 数据基础设施”** 的系统型论文。

### 5.1 对当前技术视野的价值

这篇论文最值得你跟踪的地方有三点：

- **它把数据合成问题系统化了。** 不是简单做 augmentation，而是把真实采集、仿真、视角迁移、外观迁移、人类视频迁移、视频生成、IDM 反推动作串成一个统一数据引擎。
- **它把具身推理显式化了。** Embodied CoT 不是只让模型“想一想”，而是把轨迹、子目标语言、离散动作 token 都纳入中间监督。
- **它兼顾端侧部署。** 这对于真正落地机器人系统很重要，不然很多方法只能停留在云端推理或实验室环境。

### 5.2 我觉得仍然需要警惕的问题

- **GigaWorld 细节披露不够完整。** 很多关键模块只在文中概述，作者明确说完整细节会放到后续 technical report。
- **baseline 还不够丰富。** 主实验主要与 `π0` 对比，虽然直观，但不足以完全说明相对所有强 VLA 的优势边界。
- **缺少细粒度 ablation。** 例如 RGBD、Embodied CoT、Knowledge Insulation、不同生成数据源各自贡献多少，论文没有充分拆开。
- **部分任务绝对成功率仍不高。** 例如 laundry baskets moving 依然只有约 30%，说明重物搬运、复杂移动操作仍远未解决。

### 5.3 建议后续跟进

- 继续追踪 **GigaWorld technical report**，重点看：
    - 视频生成 backbone 是什么
    - 质量筛选分数如何定义
    - 多视角一致性是怎么训练出来的
    - IDM 如何保证动作可执行性
- 横向对比这篇论文与以下路线：
    - **DreamGen / GR00T**：同样重视 world-model-generated data
    - **Qwen-VLA / PrimitiveVLA / ReconVLA**：更偏 VLA 范式、监督组织与策略泛化
    - **视频世界模型到机器人策略** 这条线：看它和 action-conditioned world model 的关系
- 如果后续你要做内部追踪，我建议把它同时挂在：
    - **主分类：VLA**
    - **关联观察：世界模型**

**一句话总结：**GigaBrain-0 值得进入你的主线视野，因为它提出了一种很现实的判断：**通用机器人 VLA 的下一步，不一定先靠更大的 policy，而可能先靠更强的数据引擎。**
