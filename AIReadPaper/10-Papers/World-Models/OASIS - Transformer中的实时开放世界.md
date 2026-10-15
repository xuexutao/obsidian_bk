---
type: project-note
status: complete
domain: World-Models
paper: "Oasis: A Universe in a Transformer"
year: 2024
arxiv: null
doi: null
source: https://oasis-model.github.io/
project: https://oasis-model.github.io/
code: https://github.com/etched-ai/open-oasis
tags:
  - World-Model
  - Interactive-Video
  - Diffusion-Forcing
  - Real-Time-Generation
  - Minecraft
created: 2026-09-04
updated: 2026-09-04
---

# OASIS：Transformer 中的实时开放世界

## 基本信息

- **正式名称**：Oasis: A Universe in a Transformer
- **团队**：Decart × Etched
- **发布日期**：2024-10-31
- **性质**：**官方研究项目与技术展示，不是正式论文或完整技术报告**
- **开源内容**：500M 下缩版权重、ViT-VAE 权重与动作条件推理代码
- **在线演示**：更大的未完整公开 checkpoint + Decart 推理栈
- **官方性能主张**：实时约 20 FPS
- **项目页**：<https://oasis-model.github.io/>
- **代码**：<https://github.com/etched-ai/open-oasis>

## 一句话结论

OASIS 用“ViT 空间自编码器 + 动作条件 latent DiT + 逐帧自回归 + Diffusion Forcing + 推理期动态加噪”把 Minecraft 式世界直接变成一个可实时操作的视频生成过程，并以 20 FPS 演示证明神经网络可以在没有传统游戏引擎的情况下即时生成画面、规则与交互；它的历史意义大于实验完备性——系统非常早地展示了可玩的生成世界，但没有正式论文、完整训练配方和严格 benchmark，长时记忆、精确物体控制与域外泛化都明显不足。

---

## 1. 研究背景与问题定义

### 1.1 从视频生成到可操作的体验

传统视频模型接收文本或图像，一次性生成固定长度片段；用户无法在生成过程中改变方向、拾取物品或破坏方块。OASIS 把任务改写为：给定当前历史画面与用户动作，实时预测下一帧，使视频模型本身承担“游戏引擎”的角色。

这种设定有三个硬约束：

1. **因果性**：未来帧只能依赖过去帧和当前动作；
2. **低延迟**：每次输入必须在数十毫秒级得到可见反馈；
3. **长程稳定**：自回归错误不能迅速滚成画面崩溃。

OASIS 选择 Minecraft 式开放世界作为测试场，是因为它同时包含第一人称运动、光照、方块规则、动物、物品栏与破坏/建造行为，能直观检验模型是否仅“生成视频”，还是已经学到部分可交互规律。

### 1.2 与 GameNGen、DIAMOND 的关系

官方页面把 OASIS 与 GameNGen、DIAMOND 并列。三者都用生成模型模拟游戏，但 OASIS 强调端到端 Transformer：空间自编码器与扩散骨干都采用 Transformer，以便稳定缩放并适配 Etched 的 Sohu ASIC。它不追求显式状态或传统物理引擎，而是从带动作的游戏视频中隐式学习世界转移。

### 1.3 贡献与证据边界

OASIS 的主要贡献是工程演示：

- 将用户键鼠输入直接条件化到逐帧 latent diffusion；
- 使用 Diffusion Forcing 训练每帧可处于独立噪声级别的序列模型；
- 用动态加噪抑制长序列误差累积；
- 在专用推理栈上实现约 20 FPS；
- 开源 500M 下缩版与推理代码。

但官方页面没有给出训练数据规模、损失曲线、消融表、统一 benchmark 或完整硬件配置。因此下面的“实验结论”必须理解为官方展示与代码证据，而非同行评审论文级验证。

---

## 2. 任务定义与输入输出

### 2.1 输入

- 一张初始图像或已有视频帧；
- 历史帧对应的 latent/KV context；
- 当前用户动作，至少包括 W/A/S/D 与鼠标控制；
- 扩散采样的随机噪声。

开源仓库的 `generate.py` 可以从样例视频取 prompt frame，也可以通过 `--prompt-path` 指定 PNG/JPG/JPEG。动作条件与历史帧共同驱动后续生成。

### 2.2 输出

输出是逐帧生成的 Minecraft 式游戏视频。官方大型演示可进行移动、跳跃、拾取、破坏、放置方块、打开箱子、与动物互动等操作。项目页声称输出达到 20 FPS；代码默认使用 10 个 DDIM steps，但开源 500M 的实际吞吐取决于 GPU，不能直接套用在线演示数字。

### 2.3 隐含假设

OASIS 没有显式的世界状态、碰撞系统、对象数据库或规则脚本。其核心假设是：只要动作条件视频模型规模足够大，画面、物理、规则和可交互性可以共同压缩进生成模型参数与短期历史中。

---

## 3. 核心方法

### 3.1 ViT-VAE + Diffusion Transformer

模型由两部分组成：

1. **空间自编码器**：以 Vision Transformer 为基础的 VAE，将 RGB 帧压缩为 latent，并把生成 latent 解码回图像；
2. **latent diffusion backbone**：DiT 在潜空间根据历史帧和用户动作预测下一帧。

![](assets/OASIS%20-%20Transformer中的实时开放世界/figures/01-architecture.png)

> **官方架构图：** 历史帧先经 ViT-VAE 编码，latent DiT 同时接收 W/A/S/D 与鼠标输入，预测下一帧 latent，再由 ViT-VAE 解码。循环执行后，模型本身构成逐帧运行的神经游戏引擎。
> 来源：官方项目页 Architecture，<https://oasis-model.github.io/>

与一次性生成整段视频的双向模型不同，OASIS 必须采用因果自回归：第 $t+1$ 帧由历史 $x_{\le t}$ 与动作 $a_t$ 条件化。可抽象为：

$$p(x_{1:T}\mid a_{1:T-1})=\prod_{t=1}^{T-1}p_\theta(x_{t+1}\mid x_{\le t},a_t).$$

真正困难在于每个 $x_t$ 都是模型自己生成的；任何误差都会进入下一步条件。

### 3.2 Diffusion Forcing

OASIS 使用 Diffusion Forcing 训练。它不要求整段序列共享同一个扩散时刻，而是给每个 token/帧独立采样噪声级别。训练目标可概括为：

$$\mathcal L=\mathbb E_{x,\epsilon,\mathbf t}\left[\sum_i\left\|\epsilon_i-\epsilon_\theta(x_i^{t_i},x_{<i}^{\mathbf t_{<i}},a_{\le i},t_i)\right\|_2^2\right],$$

其中每个位置 $i$ 的噪声时间 $t_i$ 可以不同。模型由此学会在“部分历史干净、部分历史带噪、当前帧高噪”的多种状态下去噪，适合自回归生成和长上下文修正。

这个训练范式的重要性在于：推理中的历史从来不完美。让模型在训练时看到不同程度受污染的历史，比纯 teacher forcing 更能抵抗误差累积。

### 3.3 Dynamic Noising

仅靠 Diffusion Forcing 仍不足以稳定无限 rollout。OASIS 在推理时对上下文采用**动态加噪调度**：扩散过程早期给历史帧注入更多噪声，降低模型对错误高频细节的过度依赖；后期逐渐减少噪声，让模型重新锁定可持续的细节。

![](assets/OASIS%20-%20Transformer中的实时开放世界/figures/02-dynamic-noising.png)

> **Dynamic Noising：** 横轴是时间/KV cache frames，纵轴是扩散步骤。早期步骤把历史和当前帧置于更强噪声下，后期逐步恢复细节。其直觉类似给模型一个“忘掉局部错误、保留整体结构”的修复通道。
> 来源：官方项目页 Architecture，<https://oasis-model.github.io/>

这是一种实用但不完美的稳定化：它可以抑制错误滚雪球，却也可能抹掉细粒度状态，导致物体身份、库存和远期场景无法精确保持。

### 3.4 动作条件与推理循环

开源代码提供动作条件 frame generation。每一步：

1. 编码初始/上一帧；
2. 将历史 latent 与动作送入 DiT；
3. 用 DDIM 少步去噪得到下一帧 latent；
4. VAE 解码并写入上下文；
5. 接收下一次用户输入，继续循环。

开源仓库的价值是证明系统并非只有视频演示：500M DiT 与 ViT-L/20 VAE 权重可以下载，本地可从自定义图像继续生成。但仓库主要是推理代码，没有完整训练代码与数据流水线。

---

## 4. 数据、训练与复现信息

### 4.1 已公开信息

- 模型以动作条件游戏视频训练；
- 使用 Diffusion Forcing；
- 开源 500M 下缩版 DiT checkpoint `oasis500m.safetensors`；
- 开源 ViT-VAE checkpoint `vit-l-20.safetensors`；
- 依赖 PyTorch、torchvision、einops、diffusers、timm、PyAV；
- 推理脚本默认生成 `video.mp4`，可接受自定义 prompt image。

### 4.2 未公开信息

官方没有披露：

- 训练视频小时数、场景和动作覆盖；
- 完整数据清洗与同步方式；
- 大型在线 checkpoint 参数规模；
- 总训练算力、batch size、学习率与训练步数；
- 动态加噪具体调度与系统级 latency 分解；
- 传统 GPU 上复现 20 FPS 的完整配置。

因此复现只能做到“运行开源 500M 推理”，不能严格复现官方大型实时演示。

---

## 5. 结果与实验证据

### 5.1 实时性能

官方页面声称 OASIS 实时输出约 20 FPS，并将其与当时的 Sora、Mochi-1、Runway Gen-3 做数量级对比：后者生成 1 秒视频需要数秒到数十秒，而交互系统必须约每 0.04–0.05 秒返回一帧。

![](assets/OASIS%20-%20Transformer中的实时开放世界/figures/03-performance.png)

> **官方吞吐对比：** OASIS 柱约为 20+ FPS，其余离线视频模型远低于实时。该图说明系统目标差异，但不是统一硬件、参数量、分辨率和质量下的严格 benchmark，不能据此直接宣称画质或模型效率全面领先。
> 来源：官方项目页 Performance，<https://oasis-model.github.io/>

官方还强调 OASIS 为 Etched 的 Sohu Transformer ASIC 优化，并预计专用硬件可把大模型推到更高分辨率和更低成本。这一部分属于硬件路线展望，不是当前开源 500M 的可复现结果。

### 5.2 交互行为

项目页展示模型能够：

- 放置非立方方块；
- 根据火把表现光照变化；
- 与动物互动；
- 吃东西恢复生命；
- 使用铲子比徒手更快；
- 打开箱子、切换物品与在昼夜环境移动。

这些样例说明模型捕获了部分视觉—动作—规则相关性。但官方没有统计每种行为的成功率，也没有区分模型是真正维护了游戏状态，还是依赖短期画面模式完成局部预测。

### 5.3 泛化

页面展示夜晚、类似太空的暗色地貌、多种对象和角色，认为模型可用少量额外训练扩展到新地图与体验。这个结论主要来自精选样例；没有跨域测试集、零样本成功率或受控对照。

---

## 6. 失败案例与局限

官方 Future Explorations 明确列出五类失败：

1. 远处画面模糊；
2. 不确定对象的时序一致性差；
3. 域外场景泛化不足；
4. 库存与手持物品难以精确控制；
5. 长上下文中的记忆有限。

![](assets/OASIS%20-%20Transformer中的实时开放世界/figures/04-limited-memory.webp)

> **官方长程记忆失败案例：** 经过数秒交互后，场景细节和状态难以稳定维持。OASIS 的上下文更像短期视觉缓存，而不是可查询、可写入的长期世界状态。
> 来源：官方项目页 Future Explorations，<https://oasis-model.github.io/>

### 6.1 根本性局限

- **没有显式状态**：库存、对象身份、位置与物理变量都隐含在像素/latent 中，容易漂移；
- **动态加噪的两难**：噪声有助于修复错误，也会抹除精确细节；
- **没有正式评测**：缺少动作成功率、长程一致性、物体永久性、速度—质量曲线；
- **开源版本与演示版本不一致**：500M 可运行，但 20 FPS 大型演示的完整模型与推理栈未公开；
- **域单一**：Minecraft 式视觉和规则离真实世界交互仍很远；
- **成本依赖专用硬件愿景**：项目把进一步规模化与 Sohu ASIC 绑定，通用硬件上的经济性未证实。

### 6.2 官方未来方向

团队认为扩大模型与数据规模能改善大多数失败，并提出更高效训练、扩大模型到 100B+、4K 生成、文本/音频等新控制模态以及更强推理硬件。这里需要保持谨慎：规模化可能提升视觉先验，但“精确持久状态”未必能靠参数量自然解决，可能仍需要显式记忆、对象状态或几何表示。

---

## 7. 与后续工作的关系

OASIS 是交互视频世界模型的重要早期节点。后续 GameFactory、Matrix-Game、Solaris、HYWorldPlay、LingbotWorld 等工作分别补它的短板：

- GameFactory 系统化动作注入与游戏数据构建；
- Matrix-Game 2.0 / Self-Forcing 把少步蒸馏与训练—推理对齐推进到实时流式；
- Solaris 扩展多人共享世界；
- HYWorld 1.5 与 Matrix-Game 3.0 引入空间/相机感知长期记忆；
- LingbotWorld 扩展规模与上下文，并重新设计动作表示。

从技术脉络看，OASIS 的价值是把问题暴露出来：一旦模型真的被用户持续操作，短视频评测掩盖的暴露偏差、记忆漂移、动作延迟和服务成本都会成为一等问题。

---

## 8. 复现与实践建议

### 8.1 可复现范围

公开仓库允许下载 500M DiT 与 ViT-VAE，运行 `generate.py` 从视频帧或自定义图片生成动作条件后续帧。建议将其视为**推理 demo 与架构参考**，不要视为完整训练复现。

### 8.2 建议测量

若本地运行，至少记录：

- GPU 型号、分辨率、DDIM steps、显存和实际 FPS；
- 每种离散动作的视觉响应延迟与成功率；
- 100/500/1000 帧后的画面稳定性；
- 折返重访时的 LPIPS/特征相似度；
- 动态加噪开关与不同调度下的清晰度—稳定性曲线；
- 库存、持有物和场景物体的状态保持率。

### 8.3 复现成本判断

- **运行 500M**：中等，单张现代 NVIDIA GPU 可尝试；
- **训练同类模型**：高，缺少训练代码与数据；
- **复现官方 20 FPS 大型版本**：很高或当前不可完整复现，依赖未公开 checkpoint 与专用推理栈。

---

## 9. 个人评价

OASIS 是一项“原型价值高、论文证据弱”的工作。它最重要的贡献不是宣称学会了完整 Minecraft，而是让交互世界模型第一次以可玩的形式进入公众视野，并清楚证明：**下一帧预测速度、动作条件和长期稳定必须一起设计**。

同时，它也提醒我们不要把“看起来像游戏”误判为“拥有世界状态”。在没有显式记忆与状态评测的情况下，OASIS 更准确的定位是实时动作条件视频生成器，而不是已经可靠掌握规则、对象永久性和因果动力学的完整世界模型。

---

## 10. 材料来源

- 官方研究项目页：<https://oasis-model.github.io/>
- 官方开源仓库：<https://github.com/etched-ai/open-oasis>
- 500M 权重：<https://huggingface.co/Etched/oasis-500m>
- Decart 工程说明：<https://www.decart.ai/articles/oasis-interactive-ai-video-game-model>
- 本地来源清单：[[assets/OASIS - Transformer中的实时开放世界/sources]]

> 注：本项目无正式论文 PDF，所有定量陈述均按官方项目页和开源仓库的公开范围标注。
