---
type: survey
status: done
domain: Foundations
title: LDM 多条件控制的三种注入方式：拼接、残差注入与 Score 组合
source: https://www.zhihu.com/question/2087917372912964602/answer/2097704075991184349
tags:
  - Foundations
  - Diffusion
  - 可控生成
  - ControlNet
  - CFG
created: 2026-10-26
updated: 2026-10-26
---

# LDM 多条件控制的三种注入方式：拼接、残差注入与 Score 组合

## 基本信息

**报告主题**：当 Latent Diffusion Model 同时接收多个条件时，「拼接 / 残差注入 / Score 组合」分别改动了生成过程中的哪一个环节，以及各自在多条件冲突时会怎样失效

**材料来源**：知乎问题「LDM接收多个条件时，拼接、残差注入与score组合分别改变了什么？」下的回答，作者「叫我Alonzo就好了」（同名公众号）

**涉及论文**：LDM（2112.10752）、ControlNet（2302.05543）、T2I-Adapter（2302.08453）、Uni-ControlNet（2305.16322）、IP-Adapter（2308.06721）、Composable Diffusion（2206.01714）

**撰写日期**：2026-10-26

**一句话结论**：三种方式不是同一层面的三种 trick，而是分别作用在**去噪网络的输入端、中间层、输出端**。拼接让网络端到端去学联合条件分布 $p(z|c_1,c_2)$；残差注入在冻结基座上叠加特征修正量 $\Delta h$；Score 组合在推理时把各条件的梯度场线性相加，隐式构造一个 Product-of-Experts 分布。三者对冲突条件的处理方式与失效模式也因此各不相同。

---

## 1. 问题定义：条件到底在哪一步改变了生成

### 1.1 统一框架：条件 score 与反向漂移

把三种做法放进同一个条件 score-based diffusion 框架里看最清楚。LDM 先用 VAE 编码器把图像压到潜空间 $z_0=\mathcal{E}(x)$，前向过程加噪得到 $z_t=\alpha_tz_0+\sigma_t\epsilon$，反向过程由去噪网络预测噪声 $\epsilon_\theta(z_t,t,c)$。在均方误差最优的意义下，这个噪声预测等价于条件 score：

$$s_\theta(z_t,t,c)\approx\nabla_{z_t}\log p_t(z_t\mid c)$$

对应的训练目标为：

$$\mathcal{L}=\mathbb{E}\left[\left\|\epsilon-\epsilon_\theta\left(z_t,t,\tau_\theta(y)\right)\right\|_2^2\right]$$

其中 $\tau_\theta$ 是针对不同模态的条件编码器。无论条件是文本、分割图还是深度图，它的最终作用都是改变每一步反向转移 $p(z_{t-1}|z_t,c)$ 的漂移项，把采样轨迹推向满足约束的流形区域。**三种注入方式的本质区别，就在于条件 $c$ 在计算图的哪一个节点介入这个漂移项**。

![](assets/LDM多条件控制的三种注入方式%20-%20Survey/figures/02-ldm-architecture.png)

> **图 1｜LDM 整体架构**（LDM, Figure 3, Page 4；来源 [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)）。左侧粉色为像素空间的编码器 $\mathcal{E}$ 与解码器 $\mathcal{D}$，中间绿色为潜空间扩散与去噪 U-Net，右侧为条件分支：语义图、文本、表征、图像等条件经 $\tau_\theta$ 编码后，要么通过右下角的 concat 开关拼接到输入，要么以 $K,V$ 的形式进入 U-Net 内部的 cross-attention。LDM 原文就已经同时给出了「拼接」与「cross-attention」两条接入通路。

### 1.2 一个典型的多条件任务长什么样

![](assets/LDM多条件控制的三种注入方式%20-%20Survey/figures/01-unicontrolnet-teaser.png)

> **图 2｜单条件与多条件可控生成示例**（Uni-ControlNet, Figure 1, Page 3；来源 [arXiv:2305.16322](https://arxiv.org/abs/2305.16322)）。上两行是 Canny、MLSD、HED、Sketch、OpenPose、Depth、Segmentation、Content 八种单条件控制；下两行是多条件组合，例如「边缘 + 深度」「骨骼 + 分割」「边缘 + 参考图」。

这张图揭示了多条件生成的基本分工：边缘、深度、骨骼等**结构条件**负责几何布局，参考图或文本等**语义/外观条件**负责内容与风格。真正的难点在于多路信号同时进入、甚至局部互相矛盾时，模型该听谁的——而这恰恰取决于条件是在哪一层被注入的。

---

## 2. 拼接：改变网络的输入观测空间

### 2.1 Spatial concatenation：通道维拼接

假设有两个空间条件 $c_1\in\mathbb{R}^{H\times W\times d_1}$ 与 $c_2\in\mathbb{R}^{H\times W\times d_2}$（如深度图与分割图）。常规做法是先用轻量卷积下采样，投影到与潜变量相同的空间分辨率，再在通道维上与带噪潜变量拼接：

$$\tilde{z}_t=\mathrm{concat}\left[z_t,\phi_1(c_1),\phi_2(c_2)\right]$$

去噪网络学习的映射由 $f_\theta:z_t\to\hat{\epsilon}$ 扩展为 $f_\theta:(z_t,c_1,c_2)\to\hat{\epsilon}$。关键在于：**拼接本身不预设任何「多条件应该如何组合」的规则**。只要训练数据充分覆盖联合分布 $(z_0,c_1,c_2)\sim p_{\mathrm{data}}$，最优去噪器学到的就是联合条件下的 score：

$$\epsilon^*(z_t,c_1,c_2)\approx\nabla\log p_t(z_t\mid c_1,c_2)$$

由于完全依赖网络的非线性表达能力去捕捉条件之间的统计相关性，拼接在表达能力上的理论上限高于任何后验线性叠加——它不需要条件独立假设。

### 2.2 拼接在多条件冲突时的两个短板

**其一，缺少显式的冲突仲裁机制，强依赖训练分布覆盖度。** 所有条件在首层卷积处平起平坐，网络内部没有「谁优先」的归纳偏置。例如深度图说人物在画面左侧、分割图却把人物轮廓画在右侧，这种左右矛盾的成对标注几乎不会出现在训练集里，推理时拼接张量 $[z_t,c_{\mathrm{depth}},c_{\mathrm{seg}}]$ 直接落入 OOD 区域，网络无所适从，结果往往是结构崩坏或伪影。

**其二，容易被各条件的初始特征尺度左右。** 首层卷积计算 $h=W_zz_t+W_ec_e+W_dc_d$，如果高频边缘特征的范数明显大于平滑深度特征，激活就会被数值更大的模态主导。缺少归一化、独立投影头或门控时，各条件的「话语权」由幅值而非语义决定。

### 2.3 Token concatenation：序列维拼接与 softmax 零和博弈

另一类拼接发生在序列维：把文本特征 $C_{\mathrm{text}}\in\mathbb{R}^{N_t\times d}$ 与图像提示特征 $C_{\mathrm{image}}\in\mathbb{R}^{N_i\times d}$ 拼成 $C=[C_{\mathrm{text}};C_{\mathrm{image}}]$，一同作为 cross-attention 的 key/value。此时被改变的是注意力内部的 softmax 分布：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d}}\right)V$$

每个视觉 query 都要在文本 key 与图像 key 上做**同一次**归一化。softmax 权重和恒为 1，于是变成零和博弈：一方注意力变大，另一方就被挤压。典型症状是图像参考太强，文本里的修改指令完全失效。

![](assets/LDM多条件控制的三种注入方式%20-%20Survey/figures/03-ipadapter-decoupled-attn.png)

> **图 3｜IP-Adapter 的解耦交叉注意力**（IP-Adapter, Figure 2, Page 4；来源 [arXiv:2308.06721](https://arxiv.org/abs/2308.06721)）。图像经冻结的 Image Encoder、可训练的 Linear + LN 投影为 Image Features，文本经冻结的 Text Encoder 得到 Text Features；两者分别进入**各自独立**的 Cross Attention（右上虚线框），只有红色的图像分支 cross-attention 是新训练的。

IP-Adapter 的做法正是对上述零和问题的直接回应：文本分支和图像分支各自做 softmax，再在输出端把两个 context 向量线性相加，

$$Z_{\mathrm{new}}=\mathrm{Attn}(Q,K_t,V_t)+\lambda\cdot\mathrm{Attn}(Q,K_i,V_i)$$

从机制上避免了跨模态 token 在同一个概率单纯形上争夺权重，同时用 $\lambda$ 显式控制图像提示的强度。值得注意的是，这一步其实已经把「token 拼接」改造成了「两路注意力输出的残差叠加」，从输入层面的融合滑向了下一节的特征层面融合。

---

## 3. 残差注入：在中间层修正去噪轨迹

### 3.1 一阶分解：基座先验 + 条件修正量

残差注入保持输入端 $z_t$ 不变，在 backbone 中间层叠加外部特征增量。设预训练模型第 $l$ 层原始特征为 $h_l$，条件分支输出同尺度增量 $r_l(c)$，单条件时 $\tilde{h}_l=h_l+\lambda r_l(c)$，多条件时自然扩展为多路线性叠加：

$$\tilde{h}_l=h_l+\sum_{i=1}^{K}\lambda_ir_{i,l}(c_i)$$

它本质上是对条件去噪函数做一阶近似分解 $f(z,c)\approx f_0(z)+\Delta f(z,c)$：冻结的基座已经学会生成自然图像，新条件只需学「如何在各尺度上推一把 hidden states」，不必从头学整个图像分布。这也解释了为什么这类方法在小样本微调下仍然稳定。

### 3.2 ControlNet：可训练副本 + 零卷积

![](assets/LDM多条件控制的三种注入方式%20-%20Survey/figures/04-controlnet-arch.png)

> **图 4｜Stable Diffusion U-Net 与 ControlNet 的连接方式**（ControlNet, Figure 3, Page 4；来源 [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)）。左侧带锁的灰色块是冻结的 SD 编码器、中间块与解码器；右侧蓝色块是编码器与中间块的可训练副本，条件 $c_f$ 经零卷积加到输入 $z_t$ 上，副本每一层的输出再经零卷积加回 SD 解码器对应的 skip connection。

ControlNet 冻结 SD 主干，复制一份 U-Net 编码器作为旁路，用权重和偏置都初始化为零的 $1\times1$ 卷积（zero convolution）连回主干。训练第 0 步时旁路输出 $\Delta h_l^{\mathrm{control}}=0$，整个网络与原版 SD 完全一致，基座先验被完整保留；随着训练推进，旁路逐步学会施加偏移。零初始化的意义在于：有害噪声不会在训练初期破坏预训练特征，同时梯度仍能流入零卷积（因为输入非零），参数可以从零长出来。

多条件场景下，Multi-ControlNet 直接把多路增量加权求和：

$$h_l'=h_l+\lambda_er_l^e+\lambda_dr_l^d+\lambda_pr_l^p$$

在 diffusers 中，每个分支的 $\lambda_i$ 就是 `controlnet_conditioning_scale`，缩放后的残差汇入对应 skip connection。

### 3.3 T2I-Adapter 与 Uni-ControlNet：更轻、更分工

![](assets/LDM多条件控制的三种注入方式%20-%20Survey/figures/05-t2iadapter-arch.png)

> **图 5｜T2I-Adapter 的多条件加权注入**（T2I-Adapter, Figure 3, Page 4；来源 [arXiv:2302.08453](https://arxiv.org/abs/2302.08453)）。上方是整体冻结的 Stable Diffusion；左下多个条件 $c_1,\dots,c_k$ 各自通过可训练 Adapter，乘权重 $\omega_i$ 后在 $\oplus$ 处相加，再注入 U-Net 编码器各尺度。右下展示单个 Adapter 内部：Pixel Unshuffle 后经四个尺度（$64\times64$ 到 $8\times8$）的 Conv + ResBlock，输出 $F_c^1\dots F_c^4$。

T2I-Adapter 用轻量串行卷积适配器替代沉重的 U-Net 编码器副本，提取多尺度特征 $\mathbf{F}_c$ 后直接加到编码器特征上：$\hat{\mathbf{F}}_{enc}^i=\mathbf{F}_{enc}^i+\mathbf{F}_c^i$。参数量与推理开销远低于 ControlNet，组合多个条件也只是在加法节点处乘权重求和。

![](assets/LDM多条件控制的三种注入方式%20-%20Survey/figures/06-unicontrolnet-framework.png)

> **图 6｜Uni-ControlNet 的 local / global 分路设计**（Uni-ControlNet, Figure 2, Page 4；来源 [arXiv:2305.16322](https://arxiv.org/abs/2305.16322)）。橙色 Local Control Adapter 接收所有局部空间条件，以多尺度残差的方式 Add 到主干；绿色 Global Control Adapter 把全局条件（如参考图的 CLIP embedding）转成扩展的 condition token，与 Text Tokens 一起形成 Extended Prompt，经 cross-attention 进入网络。

Uni-ControlNet 的关键洞察是**不同性质的条件归纳偏置不同**：空间几何条件天然适合按位置对齐的残差注入，全局风格/内容条件天然适合序列 token。它把两类条件分别放进两条通路，并且无论条件数量多少都只需训练 2 个 adapter，是「拼接（token 维）+ 残差注入（空间维）」在同一框架中的结合。

### 3.4 残差注入在多条件冲突时的问题

需要强调的是，多路残差注入是**在同一个网络内部叠加多个特征修正向量**，并不是在组合概率分布。它的冲突表现在两个层面：

**方向抵消与幅值压制。** 若两个分支在同一位置的残差方向相反（$r_1^{\top}r_2<0$），相加后互相抵消，两边控制都变弱；若一路残差范数远大于另一路，数值占优者会直接盖过其他分支。

**局部空间错位。** 例如同一区域里 Canny 边缘轮廓在横坐标 100，深度跳变边界却在 140，两个 ControlNet 都在改写附近的 $h_l(x)$，不加区分地相加就会在交界处出现重影、结构扭曲或边缘模糊。工程上常用空间 mask 把作用区域隔开：

$$h'=h+m_1\odot r_1+m_2\odot r_2,\quad m_1\odot m_2\approx0$$

---

## 4. Score 组合：在采样阶段构造目标分布

### 4.1 从贝叶斯公式推出引导梯度

Score 组合完全不碰单次前向的内部特征，而是在推理时对多次前向得到的 score 做线性组合。由 $p_t(z_t|c_i)\propto p_t(z_t)p_t(c_i|z_t)$ 取对数求梯度：

$$s_i(z_t)=s_0(z_t)+\nabla_{z_t}\log p_t(c_i\mid z_t)$$

即单条件 score 与无条件 score 之差 $g_i=s_i-s_0$ 恰好是条件似然提供的引导梯度。多条件时把各路引导梯度加权累加到无条件先验上：

$$s_{\mathrm{comp}}=s_0+\sum_{i=1}^{K}w_i(s_i-s_0)=\Big(1-\sum_{i=1}^{K}w_i\Big)s_0+\sum_{i=1}^{K}w_is_i$$

当 $w_i=1$ 时简化为 $s_{\mathrm{comp}}=\sum_is_i-(K-1)s_0$（两条件即 $s_1+s_2-s_0$）。之所以要减去 $(K-1)s_0$，是因为每个 $s_i=s_0+g_i$ 都自带一份先验，直接相加会把先验重复计算 $K$ 次。

### 4.2 隐式的 Product-of-Experts 分布

把 $s_{\mathrm{comp}}$ 积分回密度空间，它隐式定义的目标分布是乘积专家形式：

$$p_{\mathrm{comp}}(z_t)\propto p_t(z_t)\prod_{i=1}^{K}p_t(c_i\mid z_t)^{w_i}\propto p_t(z_t)^{1-\sum_iw_i}\prod_{i=1}^{K}p_t(z_t\mid c_i)^{w_i}$$

也就是说，score 组合不关心网络内部如何表征，只是用 $K+1$ 次独立前向算出各梯度，再在输出端做向量合成，从而在概率密度层面造出一个新的目标分布。

![](assets/LDM多条件控制的三种注入方式%20-%20Survey/figures/07-composable-diffusion.png)

> **图 7｜Composable Diffusion 的推理期组合流程**（Composable Diffusion, Fig. 2, Page 5；来源 [arXiv:2206.01714](https://arxiv.org/abs/2206.01714)）。每个去噪步中，同一个带噪图像 $x_t$ 分别以句子 $c_1,\dots,c_n$ 为条件送入扩散模型，得到各自的 score $\epsilon_\theta(x_t,t|c_i)$，在「Concept Conjunction」处合成后进入下一步迭代。

Composable Diffusion 把扩散模型看作隐式 EBM，以 training-free 的方式通过 score 的加减实现概念的逻辑 AND（相加）与 NOT（相减），改善了复杂概念组合与属性绑定。

### 4.3 CFG 是最简单的一元 Score 组合

日常使用的 Classifier-Free Guidance 在形式上就是一元 score 组合：

$$s_{\mathrm{CFG}}=s_{\mathrm{uncond}}+w(s_{\mathrm{cond}}-s_{\mathrm{uncond}})$$

对应的目标分布是 $p(z)p(c|z)^w$，即把条件似然锐化到 $w$ 次幂。guidance scale 过大时出现过饱和与高频伪影，原因正是密度被过度锐化，过大的梯度步长把 latents 推离真实数据流形。

### 4.4 Score 组合的三个理论边界

**梯度方向相消。** 若 $g_1\approx-g_2$，合成引导力 $g_1+g_2\approx0$，采样器在这一步失去明确的条件推进方向，多条件控制直接失效。

**高密度支持集交集为空。** PoE 要求样本在所有专家下都有高密度。若两个条件语义或几何互斥，$\mathrm{supp}(p_1)\cap\mathrm{supp}(p_2)\approx\varnothing$，乘积分布几乎处处接近零，采样器在低密度区域盲目迭代，数值不稳定、图像崩坏。

**隐含的条件独立假设。** 严格的联合后验是 $p(z|c_1,c_2)\propto p(z)p(c_1,c_2|z)$，而 $s_1+s_2-s_0$ 偷偷假设了 $p(c_1,c_2|z)=p(c_1|z)p(c_2|z)$。但同一物体的深度、边缘、分割之间，或人体姿态与场景布局之间，往往存在强耦合。独立假设不成立时，线性叠加与真实联合后验梯度之间存在系统性偏差——这也是后续工作转向按时间步动态调权重、或把输入端融合与输出端引导结合使用的动机。

---

## 5. 总结与个人评价

### 5.1 三种方式横向对比

| 维度 | 拼接 | 残差注入 | Score 组合 |
|---|---|---|---|
| 作用位置 | 去噪网络输入端（通道维 / token 维） | backbone 中间层 hidden states | 推理时网络输出端（score / $\epsilon$） |
| 学到 / 构造的是什么 | 联合条件 score $\nabla\log p(z\|c_1,c_2)$ | 基座先验 + 条件修正量 $f_0+\Delta f$ | PoE 分布 $p(z)\prod_ip(c_i\|z)^{w_i}$ |
| 是否需要训练 | 需要，且需联合数据 | 需训练旁路，基座冻结 | training-free（需单条件模型） |
| 推理开销 | 1 次前向 | 1 次前向 + 旁路开销 | $K+1$ 次前向 |
| 冲突处理机制 | 隐式，靠数据覆盖 | 权重 $\lambda_i$ + 空间 mask | 权重 $w_i$ + 正负号 |
| 典型失效 | 矛盾组合 OOD、尺度主导、softmax 零和 | 方向抵消、幅值压制、空间错位重影 | 梯度相消、支持集无交集、独立性假设偏差 |
| 代表工作 | LDM concat、IP-Adapter（反例式改进） | ControlNet、T2I-Adapter、Uni-ControlNet | Composable Diffusion、CFG |

### 5.2 个人判断

这篇回答最有价值的地方是**换了一个切分视角**：不再把三种方法当作平行的工程 trick，而是按「改的是观测空间、特征空间还是分布本身」来分层，每一层对应一种冲突失效模式。这个框架对理解后续工作很有用，比如 IP-Adapter 本质是把 token 维拼接退回成残差叠加，Uni-ControlNet 是按条件的归纳偏置在两层之间做分工，而 CFG 只是 score 组合的一元特例。

我认为有几处可以补充或修正。第一，**「拼接表达能力上限更高」是有前提的**：它只在训练数据覆盖联合分布时成立，而多条件联合标注恰恰是最稀缺的，这正是残差注入能在实践中胜出的原因——用架构上的先验（冻结基座 + 一阶分解）换数据效率。第二，**softmax 的零和竞争并非纯粹的缺点**，它也提供了一种按内容自适应分配权重的机制；解耦注意力换来可控性，代价是失去了跨模态的相互抑制。第三，原文以 U-Net 时代的工作为主，而在 SD3 / FLUX 这类 MM-DiT 架构中，文本 token 与图像 token 在联合自注意力里拼接、双向交互，「token 拼接」重新成为主流，条件与生成内容之间的界线进一步模糊——这一趋势值得单独讨论。第四，原文没有给出任何定量证据（如多条件冲突下各方法的 FID / 条件一致性对比），所有失效模式均停留在机制分析层面，属于概念梳理而非实证综述。

对我自己的研究而言，这个分层框架可以直接迁移到 3D 生成的多条件控制：例如多视图图像 + 法线 + 拓扑约束同时作为条件时，几何类条件更适合残差注入或空间拼接，而风格、材质类全局条件更适合 token 通路；若想 training-free 地组合多个已训练好的 3D 先验，则要警惕 PoE 的支持集交集问题。

**评级**：★★★★☆（框架清晰、推导干净，适合作为可控生成的入门地图；缺少定量对比与 DiT 时代的延伸）

---

## 6. 材料来源

- 主来源：知乎回答 https://www.zhihu.com/question/2087917372912964602/answer/2097704075991184349
- 图片均从各论文 arXiv 官方 PDF 裁剪，详见 `assets/LDM多条件控制的三种注入方式 - Survey/sources.md`
- 相关笔记：[[自编码器与变分自编码器家族 - Survey]]、[[DMD - 分布匹配蒸馏一步图像生成]]
