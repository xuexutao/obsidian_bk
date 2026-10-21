---
type: paper-note
status: complete
domain: World-Models
paper: "Yume: An Interactive World Generation Model"
year: 2025
arxiv: "2507.17744"
doi: 10.48550/arXiv.2507.17744
source: https://arxiv.org/abs/2507.17744
project: https://stdstu12.github.io/YUME-Project/
code: https://github.com/Trilec/YUME
tags:
  - World-Models
  - Interactive-Video
  - Open-Domain
  - Camera-Control
  - Long-Video
created: 2026-10-16
updated: 2026-10-16
---

# Yume 1.0：开放域交互世界生成

## 基本信息

- **作者**：Xiaofeng Mao、Shaoheng Lin、Zhen Li、Chuanhao Li、Wenshuo Peng、Tong He、Jiangmiao Pang、Mingmin Chi、Yu Qiao、Kaipeng Zhang
- **机构**：Shanghai AI Laboratory、Fudan University、Shanghai Innovation Institute
- **版本**：arXiv:2507.17744v1，2025-07-23；论文自称 preview version
- **基础模型**：SkyReels-V2-14B-540P / Wan 架构，544×960、16 FPS
- **任务**：从任意图像创建可用连续键盘输入探索的开放域视频世界
- **官方资源**：[论文](https://arxiv.org/abs/2507.17744) · [项目页](https://stdstu12.github.io/YUME-Project/) · [代码](https://github.com/Trilec/YUME) · [权重](https://huggingface.co/stdstu123/Yume-I2V-540P) · [Sekai 数据](https://github.com/Lixsp11/sekai-codebase)

## 一句话结论

Yume 1.0 把城市漫游视频的连续相机轨迹量化为可用 WASD/方向键触发的文本动作，以 MVDT、分层历史压缩、AAM 与 TTS-SDE 构成开放域交互生成管线，并在 Yume-Bench 上把动作跟随从 MatrixGame 的 0.271 提升到 0.657；但它只实证到 18 秒、动作切换有惯性，14 步蒸馏又把控制分数降到 0.557，尚无真正实时 FPS 与对象交互。Yume 1.5 因而把主要问题从“基础开放域交互”转向“固定开销的全历史压缩 + 文本事件”，并用 Self-Forcing 式蒸馏解决少步 rollout 的历史错配。

---

## 1. 研究问题与贡献

### 1.1 为什么从 Minecraft 转向开放域真实世界

早期游戏世界模型可从引擎直接获得动作标签，场景和物理规则也相对封闭；现实世界视频没有同步键盘输入，相机轨迹噪声更大、内容更多样。Yume 的问题是：能否把任意街景、照片、动漫或生成图像变成可导航世界，而不为每种控制另训一个动作模块？

![](assets/Yume%201.0%20-%20开放域交互世界生成/figures/01-teaser.png)

> **Figure 1：交互式开放域生成。** 单张真实图像作为入口，用户持续按 WASD 与方向键改变人物移动和相机旋转，模型按 chunk 自回归扩展世界。图中多类输入说明其目标不是复现一个固定游戏，而是泛化到任意视觉域。  
> 来源：论文 Figure 1，PDF p.1，arXiv LaTeX 源码 `figures/yume1.pdf`。

### 1.2 四条技术主线

1. **Camera Motion Quantization（QCM）**：把 noisy c2w 轨迹映射到离散动作，再写成自然语言条件。
2. **MVDT + 历史记忆**：随机 mask 降低训练开销并改善结构；按时间远近压缩历史，分块生成长视频。
3. **高级采样器**：AAM 用低/高频重组减轻伪影，TTS-SDE 用“向未来旅行再回写”增强动作跟随。
4. **加速**：14 步对抗蒸馏加 DiT residual cache，避免直接压到一步带来的质量崩溃。

---

## 2. 输入、输出与任务边界

### 2.1 输入输出

输入为单张图像或历史视频、固定场景前缀、离散动作描述和速度描述；输出为 544×960、16 FPS 的 33 帧视频块。每生成一块，系统把新帧加入历史并按时间距离进行不同强度的 patchify，再预测下一块。

相机动作包含人物/相机平移 forward、backward、left、right，以及 turn left/right、tilt up/down，可组合为键盘操作。当前版本主要控制相机导航；论文愿景中的对象交互、神经信号控制以及统一 text/image/video-to-world 尚未实现。

### 2.2 视频世界而非显式三维世界

Yume 不输出深度、mesh、occupancy 或对象状态。所谓记忆是有损历史视频 token，物理与几何一致性只能从生成像素判断。“无限生成”表示自回归过程可以继续调用，而不是论文已经证明任意时长都保持几何、语义和对象永久性。

---

## 3. 数据与相机动作量化

### 3.1 Sekai 数据

原始收集 10,471 小时步行视频与 628 小时无人机视频，实际下载后为 8,409/214 小时；经切镜、亮度、质量和轨迹过滤得到 Sekai-Real 6,620 小时，再按内容、地点、类别和轨迹多样性筛成 Sekai-Real-HQ 400 小时。MegaSaM 估计 camera-to-world 轨迹，量化并截取动作一致、长度超过 33 帧的片段，得到 139,019 clips。

附录另说动作分布图统计 21,526 clips，但未解释它与 139,019 的子集关系。数据规模大不等于动作标签精确：轨迹估计、量化和分段都会引入误差，这也是论文采用人工 Instruction Following 评测的原因。

### 3.2 相机运动量化

给定位姿序列 $\mathcal C=\{C_0,\ldots,C_{N-1}\}$ 与动作字典 $\mathbb A_{set}$，先计算相对变换

$$T_{rel,actual}=C_{curr}^{-1}C_{next},$$

再选最接近的 canonical $SE(3)$ 动作：

$$A^*_{current}=\arg\min_{A^{(j)}\in\mathbb A_{set}}\operatorname{Distance}(T_{rel,actual},T_{canonical}^{(j)}).$$

动作通过预定义字典转为文字，和固定场景前缀一起进入 text encoder。因此 Yume 可以复用预训练 I2V 的文本接口，无需新增控制网络。代价是连续轨迹被离散化，细粒度位移幅度与绝对相机位姿不再被精确约束。

论文还从连续位置/朝向计算平移 $\mathcal V$、方向变化 $\mathcal D$ 与旋转动态 $\mathcal R$。例如相邻位移向量夹角为

$$\phi=\arccos\frac{v_1\cdot v_2}{\lVert v_1\rVert\lVert v_2\rVert},$$

相邻相机朝向夹角为

$$\theta=\arccos\frac{z_i\cdot z_{i+1}}{\lVert z_i\rVert\lVert z_{i+1}\rVert}.$$

速度也被转写成文本，并在推理时固定以避免波动。论文没有披露完整 canonical transforms、Distance 权重和速度文本档位，限制了 QCM 的精确复现。

---

## 4. 整体架构与 MVDT

### 4.1 四组件总览

![](assets/Yume%201.0%20-%20开放域交互世界生成/figures/02-overview.png)

> **Figure 2：Yume 总体框架。** 左上将相机轨迹量化为离散动作及文本；右上是 Encoder–Side-Interpolator–Decoder 组成的 MVDT；左下用分层压缩历史训练；右下按块自回归生成。  
> 来源：论文 Figure 2，PDF p.9，arXiv LaTeX 源码 `figures/yume2.pdf`。

基础训练采用 rectified flow。干净视频 latent 为 $z_0$、噪声为 $z_1$ 时，路径与损失为

$$z_t=tz_1+(1-t)z_0,$$

$$\theta^*=\arg\min_\theta\mathbb E_{t,z_0,z_1}\left[\lVert z_1-z_0-v_\theta(z_t,c,t)\rVert_2^2\right].$$

### 4.2 Masked Video Diffusion Transformer

对 $N$ 个视频 token 随机 mask 比率 $\rho=0.3$，Encoder 只处理未遮挡集合 $z_{t,u}$；Side-Interpolator 以 learnable latent token 补出被遮挡区域：

$$z_I=SA(\operatorname{Concat}[z_l,Encoder(z_{t,u})]),$$

$$\hat z_{t,u}=(1-MASK)\odot z_{t,u}+MASK\odot z_I.$$

Decoder 再处理完整插值表示。训练同时提供完整特征与 masked 特征，避免网络只学补洞；推理直接处理完整 token，不再插值。

MVDT 消融中，启用后 Aesthetic 0.517 vs 0.492、Imaging 0.702 vs 0.694、Background 0.929 vs 0.902，但 Motion Smoothness 0.985 vs 0.991。它主要改善结构与背景一致性，并非所有指标均提升；论文也未报告训练显存/FLOPs 实际下降多少。

### 4.3 I2V/V2V 条件

条件图像/视频经 Wan-VAE 得到 $z_c$，mask $M_c$ 标记保留与待生成区域，DiT 输入为通道拼接 $[z_t,z_c,M]$；I2V 额外使用 CLIP 图像特征，文本由 umT5 编码。V2V 只取条件视频最后一帧供 CLIP 编码，本身没有永久记忆能力。

训练中 30% iterations 将单张图复制 16 次构造静态条件视频，防止模型过度复制条件视频的同质运动。这直接对应论文观察到的“动作惯性”问题：即使有动作文字，历史运动仍可能压过当前命令。

---

## 5. 长视频记忆：分层压缩而非无限上下文

![](assets/Yume%201.0%20-%20开放域交互世界生成/figures/03-long-video-memory.png)

> **Figure 3：自回归长视频。** 当前 33 帧预测同时读取首帧、按时间距离压缩的历史帧、动作文字与速度；生成结果写回历史后继续下一块。近期帧分辨率较高，远期帧被更强压缩。  
> 来源：论文 Figure 3，PDF p.10，arXiv LaTeX 源码 `figures/yume3.pdf`。

最终历史压缩策略为：

$$\begin{aligned}
t-1\text{ 到 }t-2 &: (1,2,2),\\
t-3\text{ 到 }t-6 &: (1,4,4),\\
t-7\text{ 到 }t-23 &: (1,8,8),\\
\text{首帧} &: (1,2,2).
\end{aligned}$$

每次预测未来 33 帧；训练时以 0.3 概率从 33–400 帧视频采历史，以 0.7 概率从 400–800 帧长视频采历史。采样时将加噪历史与当前 latent 按帧拼接：

$$\hat z_{in}=((1-t_n)I_{in}+t_nz_{noise})\oplus z_{in},$$

$$z_{t_{n-1}}=z_{in}+(t_{i-1}-t_i)v_\theta(\hat z_{in},c,t_n).$$

这个设计比只保留最近几帧更接近“记忆”，但历史范围仍有限且有损；没有按空间位置检索，也没有对象级状态。它擅长持续向前漫游，不保证离开很久后回到原地仍重建同一场景。

---

## 6. 高级采样：画质与控制的权衡

### 6.1 AAM：保留结构、重做高频

AAM 先做一次完整去噪得到 $z_{orig}$，第二次去噪的前 $K$ 步，把第一次结果的低频与当前 refinement 的高频重组：

$$z_{orig,t_i}=(1-t_i)z_{orig}+t_iz_{t_{N-1}},$$

$$z'_{high}=z'_{t_i}-B(z'_{t_i}),$$

$$z'_{t_i}\leftarrow B(z_{orig,t_i})+z'_{high}.$$

标准 50 步 CFG 需 100 NFE；AAM 用 30 步无 CFG 初次生成 + 30 步 CFG refinement，共 90 NFE。AAM 消融把 Aesthetic/Imaging/Dynamic 从 0.517/0.702/0.929 提升至 0.529/0.737/0.937。

![](assets/Yume%201.0%20-%20开放域交互世界生成/figures/05-aam-results.png)

> **Figure 7：AAM 定性对比。** 下排在栏杆、雪人轮廓和建筑装饰中保留更多结构细节；红框同时提醒该图是精选案例，应与小规模消融结合解读。  
> 来源：论文 Figure 7，PDF p.20，arXiv LaTeX 源码 `figures/vis2.pdf`。

关键限制是 AAM 会让当前片段与历史断裂，作者认为原因在于其预训练模型仍偏 I2V、未充分 V2V 微调。因此它适合单段高质量合成，却未真正解决长视频质量。

### 6.2 TTS-SDE：用随机“时间旅行”换动作跟随

每隔 $s=5$ 步向更干净方向旅行 $l=5$ 步，使用未来状态更准确的 velocity 重建当前更新；SDE 引入噪声

$$\Delta z=\eta\sqrt{|t_{i-1}-t_i|}\mathcal N(0,I),$$

以增强文本动作条件的影响。结果为：

| Sampler | IF↑ | Subject↑ | Background↑ | Smooth↑ | Aesthetic↑ | Imaging↑ |
|---|---:|---:|---:|---:|---:|---:|
| ODE | 0.657 | **0.932** | **0.941** | **0.986** | 0.518 | **0.739** |
| SDE | 0.629 | 0.927 | 0.938 | 0.985 | 0.516 | 0.737 |
| TTS-ODE | 0.671 | 0.923 | 0.936 | 0.985 | **0.521** | 0.737 |
| TTS-SDE | **0.743** | 0.921 | 0.933 | 0.985 | 0.507 | 0.732 |

TTS-SDE 把控制提升 0.086，但牺牲其余指标。正文称其 improved aesthetic，与表中 0.518→0.507 相矛盾；可信结论应是“控制更强、质量略降”，不是全面改善。论文还未披露 $\eta$ 和额外 NFE，算法索引也有 $t_{i+1}/t_{i-1}$ 疑点。

---

## 7. 加速：14 步蒸馏与残差缓存

### 7.1 对抗蒸馏

作者拒绝直接一步生成，因为复杂场景适应和长视频稳定性会恶化；最终从 50 步压到 14 步。总损失为

$$\mathcal L_{total}=\mathcal L_{diffusion}+\lambda_{adv}\mathcal L_{adv},$$

预测 clean sample：

$$\hat x_0=z_t-t\,v_\theta(z_t,t,c),$$

判别器与生成器损失：

$$\mathcal L_D=\mathbb E[\operatorname{ReLU}(1-D(x_{real}))]+\mathbb E[\operatorname{ReLU}(1+D(\hat x_0))],$$

$$\mathcal L_{adv}=-\mathbb E[D(\hat x_0)].$$

| 模型 | 时间 s↓ | IF↑ | Subject↑ | Background↑ | Smooth↑ | Aesthetic↑ | Imaging↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50-step | 583.1 | **0.657** | **0.932** | **0.941** | **0.986** | 0.518 | **0.739** |
| 14-step Distil | **158.8** | 0.557 | 0.927 | 0.940 | 0.984 | **0.519** | **0.739** |

约 3.67× 加速几乎不损害常规画质，却使动作跟随下降 0.100。这个反直觉结果非常重要：少步模型首先丢失的不是单帧画质，而是弱文本控制信号。

### 7.2 DiT residual cache

模型在选定 blocks 上缓存某一步的 residual：

$$\mathcal C^l_{t_n}=(Block^l(x^{l-1}_{t_n})-x^{l-1}_{t_n})_{bf16},$$

后续步直接复用：

$$x^l_{t_{n-1}}=x^{l-1}_{t_{n-1}}+\mathcal C^l_{t_n}.$$

40 个 DiT blocks 中，作者按移除 block 后的 MSE 选择 10 个影响最小的中间层缓存；首尾层更重要。论文没有披露 cache interval、具体层号、显存和蒸馏+缓存最终速度，因此不能据此声称实时。

---

## 8. 实验与消融

### 8.1 设置

训练使用 SkyReels-V2-14B-540P，544×960、16 FPS、Adam、lr $10^{-5}$、batch 40、7,000 iterations，硬件为 A800，但 GPU 数量和总时长未披露。Yume-Bench 有 70 个样本，每个 96 帧；所有比较方法使用 50 steps 和相同 random seed。Instruction Following 为人工评估，但 annotator 数、协议和一致性未披露。

### 8.2 主要结果

| 模型 | IF↑ | Subject↑ | Background↑ | Smooth↑ | Aesthetic↑ | Imaging↑ |
|---|---:|---:|---:|---:|---:|---:|
| Wan-2.1 | 0.057 | 0.859 | 0.899 | 0.961 | 0.494 | 0.695 |
| MatrixGame | 0.271 | 0.911 | 0.932 | 0.983 | 0.435 | **0.750** |
| **Yume** | **0.657** | **0.932** | **0.941** | **0.986** | **0.518** | 0.739 |

Yume 的核心提升是开放域控制，而非所有画质指标：Imaging 仍低于 MatrixGame。由于 MatrixGame 原生使用键鼠模块、Yume 使用文本量化控制，比较既反映模型能力，也混合了不同控制接口和数据域。

![](assets/Yume%201.0%20-%20开放域交互世界生成/figures/04-qualitative-results.png)

> **Figure 6：开放域与编辑结果。** 左侧展示真实森林、海岸与城市漫游；右侧的昼夜、天气和风格变化是 Yume 与外部图像编辑模型组合的结果，不是 Yume 1.0 原生的 chunk-level 文本事件控制。  
> 来源：论文 Figure 6，PDF p.20，arXiv LaTeX 源码 `figures/yume4.pdf`。

### 8.3 长视频实验

论文只测试 18 秒：每次生成 2 秒，共 9 次外推。前 8 秒沿测试动作，后 10 秒切换为持续 W。Subject 从 0.934 降到 0.930，Background 从 0.947 降到 0.941；8–12 秒动作切换受历史惯性影响，12 秒后 IF 从 0.636 恢复到 0.819。

原文把 0.947→0.941 称为下降 8.6%，把 0.636→0.819 称为恢复 22.3%，均与直接计算不完全一致，本文保留原始数值而不采用百分比结论。18 秒不能支撑“无限一致性”，只能证明中短程可连续扩展。

### 8.4 控制注入消融

在只训练 1,000 iterations、20 个视频的附录实验中：AdaLN-Zero/Cross-Attention/Text injection 的 IF 为 0.35/0.45/0.45。文本注入并非独占最高，Aesthetic 与 Background 反而最低；作者选择它主要因为兼容预训练模型且零新增可学习参数。这是工程权衡，不是质量全面领先。

---

## 9. 失败案例、局限与 Future Work

### 9.1 明确失败

- 历史运动造成动作切换惯性，不能立即反向；
- V2V 会复制条件视频的同质运动，削弱新命令；
- AAM 用于长视频时可能造成 chunk 边界断裂；
- TTS-SDE 提升控制但降低 subject/background/aesthetic/imaging；
- 14 步蒸馏保持画质却明显损害动作控制；
- 复杂场景仍出现模糊细节、异常纹理、闪烁与几何扭曲；
- 无对象交互、无显式世界状态，也未验证回访一致性。

### 9.2 作者承认的总体限制

结论明确列出 visual quality、runtime efficiency 与 control accuracy；未来还需对象交互、统一 text/image/video 输入和更丰富控制。AAM 的下一步是 V2V 微调并接入长视频，蒸馏则需要在加速和文本/动作控制之间找到更好平衡。

### 9.3 证据边界

“开放数据/代码/权重”是重要贡献，但论文没有端到端 FPS、输入到画面的延迟、单卡配置或长于 18 秒的定量；评测又依赖未充分披露协议的人工 IF。因此 Yume 1.0 证明了开放域交互生成的可行性，却没有证明达到实时、长期稳定或可作为物理模拟器。

---

## 10. 与 Yume 1.5 的问题演化

[[Yume 1.5 - 文本控制的交互式世界生成|Yume 1.5]] 不是简单提高分辨率，而是针对 1.0 暴露的三个结构性问题：

| 1.0 的瓶颈 | 1.5 的回应 | 实质变化 |
|---|---|---|
| FramePack-like 历史随长度增加，远历史高度有损 | TSCM 同时做时空压缩与通道压缩，并用线性注意力汇总 | 从有限层级历史变成近似固定单步成本的全历史摘要 |
| 动作只是完整文本条件，事件变化依赖外部图像编辑 | 将 caption 拆成 event 与 action，两者 embedding 分开缓存/更新 | 从“导航”扩展到 chunk-level 文本事件改变世界 |
| 14 步 GAN 蒸馏损害控制，训练仍多看真实历史 | Self-Forcing with TSCM + DMD，历史来自学生自身 rollout，压到 4 步 | 从离线少步蒸馏转向训练—推理历史分布对齐 |

1.0 已有“把动作写成文本”的接口，但 Figure 6 的 Night→Day/Sunny→Snowy 是与外部图像编辑模型组合，不等于 1.5 的原生事件 prompt。1.5 真正新增的是**事件与动作解耦**，使动作可以高频变化、事件低频变化，同时避免每帧重算全部文本条件。

1.5 的 TSCM 也并未解决精确空间回访：它是压缩历史而不是相机几何索引。系列路线更适合开放域持续探索；若目标是分钟级回到原地并恢复相同对象，仍需显式/检索式空间记忆。

---

## 11. 开源与复现

### 11.1 已公开

当前官方仓库可定位 MVDT、FramePack-like 训练、长视频 DDP/FSDP 推理、TTS-SDE、对抗蒸馏、相机轨迹量化和模型权重。README 声称 Yume 1.0 开放数据、训练/推理代码与权重；训练 MVDT 至少需要 16 张 A100。

### 11.2 复现障碍

- 论文训练用 A800，但数量、GPU-hour 和 wall-clock 未披露；
- QCM canonical transforms/距离权重、速度文本映射不完整；
- TTS-SDE 的 $\eta$、AAM 最终核参数、蒸馏 $\lambda_{adv}$ 和 cache interval 未披露；
- 仓库已同时包含 1.0/1.5/5B 相关代码，复现时必须锁定 14B/540p 配置和对应 commit；
- 训练 clips 的精确 manifest 与过滤版本不足以逐样本复建。

### 11.3 推荐复现顺序

1. 先用公开权重复现 2 秒 I2V 与 QCM 动作，不启用 TTS/AAM/cache；
2. 分别启用 TTS-SDE 和 AAM，核对控制—画质权衡；
3. 再做 9 段/18 秒 rollout，记录动作切换后的恢复时间；
4. 测 50→14 steps 时 IF 的下降是否可重现；
5. 最后尝试 MVDT/FramePack 训练，因为该阶段至少 16×A100，成本远高于推理验证。

---

## 12. 材料来源

本地官方 PDF、5 张经预览核验的 arXiv 源码关键图和开源状态记录见 [sources.md](assets/Yume%201.0%20-%20开放域交互世界生成/sources.md)。
