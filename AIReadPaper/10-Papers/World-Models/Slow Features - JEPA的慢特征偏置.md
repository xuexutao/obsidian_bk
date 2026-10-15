---
type: paper-note
status: complete
domain: World-Models
paper: Joint Embedding Predictive Architectures Focus on Slow Features
year: 2022
arxiv: 2211.10831
doi: 10.48550/arXiv.2211.10831
source: https://arxiv.org/abs/2211.10831
project: null
code: https://github.com/vladisai/JEPA_SSL_NeurIPS_2022
tags:
  - World-Models
  - JEPA
  - Slow-Features
  - VICReg
  - SimCLR
  - Shortcut-Learning
created: 2026-10-15
updated: 2026-10-15
---

# Slow Features：JEPA 的慢特征偏置

## 基本信息

- **正式标题**：Joint Embedding Predictive Architectures Focus on Slow Features
- **作者**：Vlad Sobal、Jyothir S V、Siddhartha Jalagam、Nicolas Carion、Kyunghyun Cho、Yann LeCun
- **arXiv**：2211.10831，v1 提交于 2022-11-20
- **论文说明**：arXiv comments 标注为 “4 pages (3 figures) short paper for SSL Theory and Practice workshop at NeurIPS 2022”
- **代码**：作者公开仓库见 frontmatter

## 一句话结论

JEPA 并不天然选择“有用状态”：在动作条件时序预测中，VICReg 和 SimCLR 都会优先编码跨时间最稳定、最容易预测的特征；当固定背景比运动目标更慢时，训练目标存在只复制背景而忽略目标的平凡最优解，实验在移动圆点和三圆点玩具环境中验证了这一捷径。

---

## 1. 研究背景与问题定义

### 1.1 为什么研究 reconstruction-free 世界模型

**论文事实。** 像素世界模型常以重建未来帧为目标，但像素损失不会区分任务关键对象和背景，可能产生 object-vanishing 等问题。JEPA 提供另一条路：不重建像素，而是让预测器在表征空间预测未来 encoder 输出。直觉上，encoder 可以主动忽略不可预测噪声，把容量留给环境动力学。

本文问了一个更尖锐的问题：如果无奖励、无下游标签的离线预训练只要求“未来表征可预测”，模型会选择什么信息？作者比较 VICReg-JEPA、SimCLR-JEPA、像素重建、Inverse Dynamics Modeling（IDM）、监督下界和随机网络，并用冻结表征上的线性 probe 检查圆点位置是否保留。

### 1.2 任务、输入输出与假设

环境被写成 MDP $\mathcal M=(\mathcal O,\mathcal A,P,R)$，但预训练阶段只能访问固定离线序列：

$$
(o_1,a_1,o_2,a_2,\ldots,o_{T+1}),
$$

不能访问奖励，也不知道最终评估任务。encoder $g_\phi:\mathcal O\to\mathbb R^D$ 把图像变为状态 $s_t=g_\phi(o_t)$；前向模型 $f_\theta:\mathbb R^D\times\mathcal A\to\mathbb R^D$ 根据状态和动作预测下一表征：

$$
\tilde s_{t+1}=f_\theta(\tilde s_t,a_t),\qquad \tilde s_1=g_\phi(o_1).
$$

测试时冻结 encoder 与 predictor，只训练单层线性函数 $q(\tilde s_t)$ 恢复圆点二维坐标。因而测到的不是图像重建质量，而是多步预测表征中是否线性保留位置。

关键假设是：圆点位置是下游关心的状态，背景是 distractor；changing 背景每帧重采样，难以预测，fixed 背景在单条序列内不变但跨序列重采样，是最慢且容易复制的特征。如果 JEPA 确实更关注可预测因素，那么这两种噪声会诱发相反行为。

---

## 2. 方法与训练机制

### 2.1 两种 JEPA 训练路径

![](assets/Slow%20Features%20-%20JEPA的慢特征偏置/figures/01-jepa-training-diagrams.png)

> 图 1：左侧 VICReg-JEPA 在各时间步对 encoder 表征施加 variance/covariance 正则，并让预测表征接近目标表征；右侧 SimCLR-JEPA 用同一时间步的预测输出与 encoder 输出构成 InfoNCE 正样本。来源：论文 Figure 1，第 2 页，https://arxiv.org/abs/2211.10831

两种方法都从第一帧编码开始，再用动作自回归展开，且梯度穿过时间传播。论文明确说明所有方法使用相同 encoder/predictor 架构，不使用图像增强，以便隔离训练目标差异。

VICReg 目标为：

$$
\mathcal L_{\text{VICReg}}
=\alpha\mathcal L_{\text{prediction}}
+\beta\mathcal L_{\text{variance}}
+\mathcal L_{\text{covariance}}.
$$

预测项比较展开预测与逐帧编码：

$$
\mathcal L_{\text{prediction}}
=\frac{1}{NT}\sum_{t=2}^{T+1}\sum_{i=1}^{N}
\lVert \tilde S_{t,i}-S_{t,i}\rVert_2^2.
$$

variance 项要求每个表征维度在 batch 中具有至少阈值 $\gamma$ 的标准差，防止所有样本输出常数；covariance 项压低不同表征维度之间的非对角协方差，减少维度冗余。注意，这两个正则是在同一时间步跨 batch 计算，不要求单条序列中的表征随时间变化。

SimCLR 版本使用余弦相似度温度缩放：

$$
\operatorname{expsim}(u,v)=
\exp\left(\frac{u^\top v}{\tau\lVert u\rVert\lVert v\rVert}\right),
$$

把 $S_{t,i}$ 与 $\tilde S_{t,i}$ 视为正对，batch 中其他样本作为负对，并对两个方向的 InfoNCE 取平均。它鼓励正对 alignment 和跨样本 uniformity，却同样没有规定“应该对齐圆点而非背景”。

### 2.2 对照方法

- **Reconstruction**：增加 decoder $d_\xi$，最小化 $\frac1T\sum_t\lVert d_\xi(\tilde s_t)-o_t\rVert_2^2$。像素目标迫使状态保留足以恢复背景与圆点的信息。
- **IDM**：根据相邻编码 $g_\phi(o_t),g_\phi(o_{t+1})$ 预测动作 $a_t$，同时训练前向预测。它把注意力偏向被动作影响的对象。
- **Supervised**：把 probe 误差端到端传回 encoder 和 predictor，是位置任务的理想下界参考。
- **Random**：固定随机 encoder/predictor 后训练 probe。
- **Center**：始终预测中心位置，是任务的高误差参考。

### 2.3 慢特征平凡解

这是全文最重要的理论说明。设每条序列的固定背景噪声为：

$$
s\sim\mathcal N(0,\sigma^2I),\qquad s\in\mathbb R^D.
$$

令 encoder 完全忽略圆点，直接从所有帧复制该背景：

$$
g_\phi(o_t)=g_\phi(o_{t+1})=s,
$$

并令前向模型为恒等映射 $f_\theta(s,a)=s$，即忽略动作。因为同一序列内背景不变：

$$
\mathcal L_{\text{prediction}}
=\frac{1}{TN}\sum_{t,i}\lVert S_{t,i}-S_{t+1,i}\rVert_2^2=0.
$$

跨不同序列时背景会重新采样，所以 batch 方差仍为 $\sigma^2$。只要 $\sigma$ 足够大，VICReg variance hinge 为零；若噪声维度独立，非对角协方差也为零。于是三项损失都能同时为零，却没有编码任何圆点位置或动作动力学。

对 SimCLR，作者援引 alignment/uniformity 结果：固定背景使同序列正对完美对齐，跨序列高维随机背景可近似均匀分布在单位球面，因此也能达到目标偏好的几何结构。结论不是“VICReg 正则无效”或“SimCLR 会坍塌”，而是它们成功防止了常数坍塌，却允许语义错误的 shortcut solution。

**个人判断。** 这一区分非常关键：表征统计量健康不等于状态语义正确。高方差、低协方差、优良对比均匀性都可能来自背景指纹。

---

## 3. 数据集与实验设置

### 3.1 单圆点环境

![](assets/Slow%20Features%20-%20JEPA的慢特征偏置/figures/02-dataset-examples.png)

> 图 2：左侧显示不同亮度系数 $\alpha$ 下的 uniform/structured 噪声；右侧对比每帧变化的 structured 背景与整段固定的 structured 背景。fixed 仍会在不同序列之间重新采样。来源：论文 Figure 2，第 3 页，https://arxiv.org/abs/2211.10831

圆点在 $1\times1$ 正方形中移动，坐标 $c_t=(c_t^x,c_t^y)\in[0,1]^2$，动作 $a_t\in\mathbb R^2$，满足 $\lVert a_t\rVert_2\le D$，实验中 $D=0.14$。位置按

$$
c_{t+1}=\max(0,\min(c_t+a_t,1))
$$

更新。每个 episode 有 16 个动作和 17 个状态；图像分辨率为 $28\times28$，圆点以 $\sigma=0.05$ 的 Gaussian blur 渲染。动作方向来自 Von Mises 分布，步长在 $[0,D]$ 均匀采样。

背景有两类：uniform 噪声逐像素采样；structured 噪声由 CIFAR-10 图像叠加。两类都分 changing 与 fixed。系数 $\alpha$ 控制噪声最亮像素相对圆点最亮像素的强度。

### 3.2 数据规模、评测与公平性

预训练使用 100 万条序列，probe 训练使用 30 万条，评估使用 1 万条。主图对每个方法、噪声类型和强度分别调超参数；每个点报告 17 个时间步上的平均位置 RMSE，并以 3 个随机种子的标准差作为阴影。附录另给出只在无噪声设置选超参数、迁移到其他噪声条件的结果。

模型架构为三层 convolution encoder，末端线性层输出 512 维；predictor 是单层 GRU，hidden size 512、动作输入维度 2；重建 decoder 与 encoder 对称。论文未使用 latent variable。

计算资源方面，附录明确写明实验运行在 AMD MI50 或 Nvidia RTX 8000 GPU 上；每种噪声类型和强度采样 100 组随机超参数，最佳配置再跑 3 个种子，每个单独实验少于 1 GPU 小时。不能据此直接推出总算力，因为组合数量与并行方式没有汇总。

---

## 4. 实验结果

### 4.1 主结果：changing 与 fixed 产生相反结论

![](assets/Slow%20Features%20-%20JEPA的慢特征偏置/figures/03-noise-performance.png)

> 图 3：四个子图分别是 changing uniform、changing structured、fixed uniform、fixed structured。JEPA 在逐帧变化噪声下保持较低误差，却在固定噪声出现后快速恶化；重建在主图所调范围内更稳健，IDM 与监督基线主要跟踪可控圆点。来源：论文 Figure 3，第 4 页，https://arxiv.org/abs/2211.10831

**论文事实。** 无 distractor 时所有学习方法表现良好。噪声每帧变化时，VICReg-JEPA 和 SimCLR-JEPA 可以忽略不可预测背景；作者还发现它们适应更高 changing 噪声时不依赖重新调参，而 reconstruction 在附录的未调参设置中对 changing uniform 噪声明显退化。

当背景在序列内固定时，VICReg 与 SimCLR 的位置 probe 误差迅速接近弱基线，structured 和 uniform 均如此。作者将其解释为模型编码固定背景而忽略圆点。主文同时指出 reconstruction 在 $\alpha\le1.5$ 的所有设置中表现良好。由于图中主实验对每个方法和条件单独调参，这个结论不能归因于只给 reconstruction 更合适的超参数。

IDM 在单圆点四类噪声中表现很好，因为唯一受动作控制的变化就是圆点。不过这不是通用胜利：IDM 目标只保证保留有助于动作反推的信息，可能丢弃与动作无关但对未来任务有用的状态。

### 4.2 三圆点实验揭示不同目标的选择偏置

附录构造三个通道：action-controlled dot、动作未知的随机运动 dot、每个 episode 内静止但跨 episode 换位置的 stationary dot。目标是保留可控点和静止点，忽略随机点。

结果中，VICReg 对 stationary dot 的 RMSE 为 $0.066\pm0.026$，但 action dot 为 $0.277\pm0.041$；SimCLR 对 stationary dot 为 $0.025\pm0.001$，action dot 为 $0.193\pm0.001$。两者明显优先最慢的静止点。IDM 对 action dot 为 $0.035\pm0.000$，却对 stationary dot 为 $0.272\pm0.000$，说明它只保留动作相关因素。Reconstruction 同时较好保留 action dot（$0.021\pm0.001$）和 stationary dot（$0.026\pm0.002$），而随机点误差较高。

**个人判断。** 三圆点结果比单一噪声实验更有解释力：它表明每种自监督目标都在定义“状态是什么”。JEPA 选择时间最稳定因素，IDM 选择可控因素，reconstruction 选择可恢复因素；没有外部任务约束时，不存在自动等同于“有用”的目标。

### 4.3 证据是否足够

实验、平凡解和三圆点结果相互吻合，足以证明这些具体目标在该构造环境中存在慢特征捷径。但它不足以证明所有 JEPA、真实视频或更大模型必然失败。作者也只测试 VICReg 与 SimCLR 两个目标，没有图像增强，没有分层 JEPA，没有现代 masking 策略，环境维度和语义都很简单。因此最稳妥的结论是“已识别出可复现失败模式”，而不是“JEPA 路线无效”。

### 4.4 调参与附录结果应怎样解读

附录 Figure 5 固定无噪声条件下选出的超参数，再直接迁移到其他噪声条件。此时 VICReg 和 SimCLR 会在更低噪声强度上提前失败，reconstruction 在 changing uniform 噪声中也显著变差。这个补充实验说明主图展示的是“每个条件充分调参后的能力”，而不是默认配置的鲁棒性。另一方面，主图已经允许每种方法单独调参，JEPA 在 fixed 背景下仍失败，因此慢特征现象不能简单归咎于损失权重没调好。

需要注意，作者对每种噪声类型和强度随机尝试 100 组超参数，再用最佳配置跑 3 个种子。主图阴影只反映最佳配置的种子波动，不包含超参数搜索不确定性，也不反映挑选最佳配置产生的选择偏差。复现时既要报告与论文相同的 best-of-search 曲线，也应补充固定配置曲线、中位配置表现和搜索预算，才能区分目标本身、调参敏感性与偶然性。

---

## 5. 机制解释与更深含义

### 5.1 “慢”为什么比“可控”更容易

在表征预测损失中，最便宜的方案是寻找时间上变化最小的量。固定背景不需要理解动作，也不需要预测移动；只需复制序列身份。VICReg 的跨 batch 方差正好奖励背景在不同 episode 间变化，SimCLR 的负样本区分也奖励其作为 episode fingerprint。圆点虽然是真正动力学对象，却因持续移动而带来更高预测难度。

### 5.2 这不是传统 collapse

常数 collapse 会让所有输入映射到相同向量，VICReg variance 和 SimCLR negatives 可以阻止它。本文发现的是 shortcut representation：不同 episode 的表示丰富、分散、可区分，同一 episode 内又高度稳定，但它编码的是错误因素。只用表征方差、协方差谱或对比损失收敛判断训练健康，会漏掉此类失败。

### 5.3 与 JEPA 纲领的张力

JEPA 纲领希望 encoder 忽略“不可预测且不重要”的细节。本论文说明，仅用可预测性无法判断重要性：云层、相机污点、墙纹或 episode 固定水印可能比交通参与者更可预测。也就是说，“可预测抽象”需要额外偏置来区分稳定背景、可控对象、任务状态和因果结构。

### 5.4 从目标函数看三种归纳偏置

Reconstruction 最小化观测空间误差，倾向保留“能重建什么”；IDM 最小化动作反推误差，倾向保留“什么受自身动作影响”；JEPA 的时序一致与预测目标倾向保留“什么跨时间最容易持续”。这三类偏置都不是错误，只是覆盖的充分统计量不同。静态地标对导航重要，却与即时动作无关；快速运动目标对避碰重要，却不是最慢因素；随机纹理能被重建，却可能毫无控制价值。真正的世界状态通常是这些集合的交集与并集，因此单一代理目标很难无监督地恢复所有下游所需变量。

**个人判断。** 比“选择最佳损失”更合理的路线，是让不同分支承担不同充分性：一个 head 保持观测可辨识性，一个 head 学动作条件动力学，一个 head 学长期不变量，再通过容量约束或任务条件门控组合。评价时也应分别测静态环境、可控主体、不可控主体和随机细节，而不是用一个平均 probe 分数掩盖选择偏置。

---

## 6. 局限与失败模式

### 6.1 作者明确承认的局限

实验只在玩具数据上完成，真实视频和更大模型是否同样表现尚不清楚；JEPA 目标只测试 VICReg 与 SimCLR，不能推广到 BYOL、Barlow Twins 或其他约束；论文提出 H-JEPA 或额外时间变化约束可能解决问题，但没有实验验证。

### 6.2 实验设计的边界

1. 圆点始终可见，动力学简单，probe 只测二维位置；
2. fixed 噪声被刻意设为 episode 内完全静止，是强捷径；
3. structured 噪声来自 CIFAR-10 叠加，与真实相机背景的几何和运动不同；
4. 主图逐条件调参，有利于研究目标能力上限，但不代表部署时稳健性；
5. linear probe 只能测试线性可读信息，不能证明表征完全不含圆点；
6. 自回归 17 步会混合 encoder 与 predictor 误差，未分别给出单步诊断。

### 6.3 真实系统中的对应失败

**个人判断。** 固定噪声可对应相机序列水印、场景 ID、车内饰、天空与建筑纹理、数据采集设备特征。更隐蔽的情况是“慢但有部分相关性”的变量：模型在训练分布中靠道路外观预测行为，跨城市或跨天气即失效。另一个失败是多尺度竞争：短期控制需要快速特征，长期地点识别需要慢特征；简单惩罚时间不变会从一个极端走向另一个极端。

### 6.4 外部有效性与可能的反例

真实视频并不总有完全固定背景，摄像机运动、光照和遮挡会削弱论文构造的捷径；大型模型和数据增强也可能同时编码背景与主体，而非被迫二选一。因此从玩具环境外推时，应把“特征变化速度”视为连续竞争关系，而不是 fixed/changing 二分类。反过来，真实数据通常包含更多可作为序列身份的稳定线索，模型甚至可能组合多个弱线索，形成比单一噪声更难发现的捷径。

如果下游任务本来就是地点识别、场景检索或地图定位，慢背景并非错误特征；只有当任务需要运动预测、控制或主体交互时，它才构成失败。这说明论文的“有用”定义依赖圆点位置 probe。更广泛的评估应准备多个互相冲突的任务，并测同一表征是否支持重新组合，而不是预先把所有慢信息标记为噪声。

---

## 7. 复现与实践建议

### 7.1 最小复现步骤

按论文生成 $28\times28$ 单圆点序列，固定 $T=16$、$D=0.14$，分别构造 uniform/structured 与 changing/fixed 四种背景。实现共享三层 CNN encoder、512 维表征和单层 GRU predictor，训练 VICReg、SimCLR、reconstruction、IDM。预训练后冻结模型，用 17 个展开状态训练线性位置 probe，报告时间平均 RMSE 与 3 seed 标准差。

如果资源有限，可以先缩小论文的 100 万预训练序列，但必须明确这不再是精确复现；关键现象应通过“fixed 与 changing 差值”而非绝对 RMSE 判断。作者代码已公开，优先核对数据生成、损失权重搜索和 probe 展开逻辑。

### 7.2 必加诊断

除了位置 probe，还应训练 background-ID probe、动作 probe、单步与多步位置 probe；比较每个维度的时间方差和跨 episode 方差；将同一圆点轨迹换不同背景，测表征距离是否主要随背景变化；打乱动作测试 predictor 是否真正使用 $a_t$；对背景做跨时间随机化、crop 或 color jitter，观察捷径是否转移而非消失。

### 7.3 改进方向及风险

论文建议输入 image difference/optical flow、使用 H-JEPA、或增加防止表征跨时间恒定的约束。实践中可进一步加入 action-conditioned contrast、对象级 masking、可控性辅助损失、多时间尺度 head、少量任务相关 probe 或互信息上限。但不能粗暴最大化时间变化：静态物体、地图和目标位置也可能重要。更合理的目标是同时保留 **slow-but-relevant** 与 **fast-and-controllable** 因素，并让不同 head 或层级承担不同时间尺度。

### 7.4 复现实验的验收标准

最小验收不应是“训练能收敛”，而应同时满足四点。第一，无噪声条件下各学习方法都能让位置 probe 显著优于 Random 与 Center。第二，changing 背景下 JEPA 对噪声增强具有相对稳定性。第三，fixed 背景下 JEPA 的位置 probe 明显恶化，并且 background probe 变强，从而直接证明信息转移到背景。第四，三圆点实验应重现 VICReg/SimCLR 偏向 stationary dot、IDM 偏向 action dot、reconstruction 同时保留两者的排序关系。

还应做两个反事实检查：固定背景但在训练时随机交换不同序列的背景，若失败缓解，说明 episode fingerprint 是因果因素；保持背景固定但给圆点位置增加轻量监督或可控性损失，若位置恢复而背景 probe 下降，说明问题来自目标选择而非模型容量。这样的干预比单纯重复曲线更能验证论文机制。

---

## 8. 个人判断

### 8.1 论文最重要的贡献

它没有提出更复杂模型，而是给出一个非常清晰的反例：防坍塌正则解决的是“有没有信息”，不是“信息是否正确”；预测损失解决的是“能否预测”，不是“是否对任务有用”。这一点对所有 latent dynamics、视频自监督和世界模型都适用。

### 8.2 对结果的保留判断

玩具问题的优势是变量可控、机制可证明，缺点是与真实视频差距大。论文标题“JEPA Focus on Slow Features”应理解为对所测 JEPA 目标的经验和理论结论，不宜扩大为所有联合嵌入预测架构的定理。此外，reconstruction 的相对稳健不表示像素重建在复杂场景更优；当背景维度巨大时，它也可能浪费容量，只是这个小环境允许同时重建圆点和背景。

### 8.3 研究启发

最值得延伸的是把“表示选择”变成显式多目标：predictability、controllability、task relevance、information sufficiency 与 invariance 之间需要可观测权衡。层级模型也不应只按网络深度命名，而应验证每一层分别承载何种时间尺度和哪些变量。任何世界模型论文若只报告预测损失或下游单一分数，都应补充 shortcut probes 与反事实背景测试。

---

## 9. 阅读结论

本文给 JEPA 路线提供了必要的负面证据。它没有否定表征空间预测，而是指出“忽略不可预测细节”必须搭配更强的选择原则：否则系统会把动态主体当作麻烦，把固定背景当作世界状态。对工程实践而言，最直接的教训是主动构造 fixed/changing、controllable/uncontrollable、static/dynamic 的正交压力测试，并同时探测模型保留了什么、丢掉了什么。

---

## 10. 材料来源

- arXiv：https://arxiv.org/abs/2211.10831
- 官方 PDF：https://arxiv.org/pdf/2211.10831
- 代码：https://github.com/vladisai/JEPA_SSL_NeurIPS_2022
- 本地论文：`assets/Slow Features - JEPA的慢特征偏置/sources/paper.pdf`
- 来源清单：`assets/Slow Features - JEPA的慢特征偏置/sources.md`
