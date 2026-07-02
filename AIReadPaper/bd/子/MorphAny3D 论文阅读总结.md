**结论先看：** 这是一篇把 **Trellis / SLAT 这类原生 3D 生成先验** 真正用到 **跨类别 3D morphing** 上的工作。作者没有重新训练 3D 生成器，而是直接在注意力机制里改“特征如何融合”，再补一个姿态纠正策略，就把 **语义合理性、时序平滑性、审美质量** 同时拉了起来。

**重要性评估：★★★★☆（4/5）**

- **为什么值得看：** 它不是再做一个新的 3D backbone，而是证明了 **Structured Latent / SLAT 可以作为 3D 形变与编辑的统一操作空间**。
    
- **为什么不是 5 星：** 任务场景仍偏垂直，生成一段 morphing 序列的成本也较高（单帧 30s），而且上限仍受 Trellis 本身的细粒度表达能力约束。
    

## 1. 背景

这篇工作对应的官方论文是 [MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing](https://arxiv.org/abs/2601.00204)，来自南京大学与北京大学，arXiv v3 显示已被 **CVPR 2026 接收**。外部线索来自公众号文章 [CVPR 2026 | 解锁3D变形新境界！南大&北大提出MorphAny3D，让3D生成大模型秒变“变形魔法师”](https://mp.weixin.qq.com/s/-Dqms6q7UHt-DowMGZe_aA)。

传统 3D morphing 大多依赖 **源形体与目标形体之间的对应关系**。这类方法在同类物体内还能工作，但一旦遇到 **跨类别形变**，例如“大象 → 挖掘机”“蜜蜂 → 双翼飞机”，稠密对应就很容易失真，最终要么结构崩，要么过渡不自然。

论文把现有路线分成三类：

1. **基于匹配的 3D morphing**：平滑，但跨类别时很难保持语义合理。
    
2. **先做 2D morphing，再逐帧升到 3D**：单帧可能看起来合理，但因为每一帧单独升维，时序一致性差。
    
3. **直接插值 Trellis 的噪声或条件特征**：看起来最省事，但在 SLAT 空间里并不能自动得到结构合理的过渡。
    

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ZjJhMWNmYTdjNmM5NDQyNjM0ZGY2OGM4ZDJjODZhYjhfR0thS2Z4RUFyVU9ZeEZFa2RzVDkxMFpuN0wwbEw3WWRfVG9rZW46RUx2TWJMQ2Rwb2lnNVl4VlhuOGNhTG9NbkFkXzE3ODI5ODA0MjU6MTc4Mjk4NDAyNV9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

## 2. 文章主线 / 论文线索

### 2.1 主论文

- **论文名：** [MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing](https://arxiv.org/abs/2601.00204)
    
- **作者：** Xiaokun Sun, Zeyu Cai, Hao Tang, Ying Tai, Jian Yang, Zhenyu Zhang
    
- **机构：** 南京大学、北京大学
    
- **状态：** arXiv v3，论文页 comments 标注为 **Accepted by CVPR 2026**
    
- **项目页：** [MorphAny3D Project Page](https://xiaokunsun.github.io/MorphAny3D.github.io/)
    
- **代码：** [GitHub - XiaokunSun/MorphAny3D](https://github.com/XiaokunSun/MorphAny3D)
    

### 2.2 依赖与技术主线

这篇工作不是从零训练一个新模型，而是建立在 **Trellis** 的 **Structured LATent (SLAT)** 表示之上。

- **关键底座论文：** [Structured 3D Latents for Scalable and Versatile 3D Generation (Trellis)](https://arxiv.org/abs/2412.01506)（论文中引用为 CVPR 2025）
    
- **核心前提：** Trellis 的 SLAT 表示有显式、规则、可操作的结构，既能承载几何，也能承载外观，因此适合作为 **训练后无需再训练** 的下游编辑 / 形变空间。
    

### 2.3 一句话主线概括

如果用一句话概括本文：

**MorphAny3D 不是学习“从 A 到 B 怎么变”，而是借助预训练 3D 生成器的结构化 latent prior，在生成过程中把 source / target / previous-frame 的信息以更合理的方式注入注意力层，从而无训练地生成语义合理、时序平滑的 3D morphing 序列。**

## 3. Pipeline / Architecture + I/O 数据流

### 3.1 任务定义

输入与输出非常明确：

- **输入：** 源对象 `x_src` 与目标对象 `x_tgt`
    
    - 可以是真实 3D 资产
        
    - 也可以是 Trellis 已经生成过的 3D 资产
        
- **输出：** 一个长度为 50 帧的 morphing 序列 `{x^n}_{n=0}^{N}`
    
    - `N = 49`
        
    - `α^n = n / N`
        
    - `α=0` 对应 source，`α=1` 对应 target
        

### 3.2 底层表示：Trellis 的两阶段 SLAT 生成

论文完全建立在 Trellis 的两阶段流程上：

1. **SS Stage（Sparse Structure）**
    
    1. 在 `64^3` 体素网格上预测物体的稀疏结构 `P = {p_i}`
        
    2. 作用：确定全局结构 / 空间骨架
        
2. **SLAT Stage（Structured Latent）**
    
    1. 在 SS 阶段确定的活跃 sparse voxel 上，预测局部 latent `Z = {z_i}`
        
    2. 每个局部 latent 同时编码细粒度几何与外观
        
3. **Decoder**
    
    1. 将生成得到的 SLAT 解码为常规 3D 表示
        
    2. 论文明确写到可解码为 **mesh、NeRF、3DGS** 等
        

### 3.3 初始化 I/O：source / target 如何进入模型

这里分两种情况：

- **真实 3D 资产**：
    
    - 先做 3D inversion，得到
        
        - `f_init^src`, `f_init^tgt`：初始带噪 latent
            
        - `c_src`, `c_tgt`：图像条件特征
            
- **Trellis 已生成资产**：
    
    - 直接复用缓存的初始 latent 与条件
        

作者把每个对象的 latent 记为：

- `f = (f_ss, f_slat)`
    

也就是说，SS 与 SLAT 两阶段各自都有自己的初始 latent。

### 3.4 初始帧特征如何构造

第 `n` 帧不是直接从 source 或 target 单独生成，而是先通过 **球面插值（slerp）** 得到初始 noisy feature。

对 SS 阶段，论文给出的公式是：

```Plain
f_init_ss^n = sin((1-α^n)θ) / sin(θ) * f_init_ss^src
            + sin(α^n θ) / sin(θ) * f_init_ss^tgt
```

其中：

- `θ` 是 source / target 初始特征的夹角
    
- SLAT 阶段也做类似插值，但需要先基于欧氏距离找到对应 sparse voxel
    

**意义：** 这一步只是在 latent 初始状态上设置“这帧当前应该更像 source 还是更像 target”，真正决定生成质量的关键，仍然是后面注意力层里的信息融合方式。

### 3.5 论文的关键观察：不要直接在 attention 里混 K / V

作者先复现并分析了先前工作常见的做法：

- 在 cross-attention 或 self-attention 中，直接对 source / target 的 `K`、`V` 做线性混合
    
- 也就是典型的 KV-Fused Attention
    

论文发现：

- **KV-Fused CA**：有助于语义合理性，但会引入局部结构伪影
    
- **KV-Fused SA**：有助于时序平滑性
    
- **两者同时用**：反而破坏 plausible-smooth trade-off
    

原因在于：

- 在 **cross-attention** 中，`K/V` 来自 **patch-wise DINOv2 特征**
    
- source 与 target 的空间对齐 patch **并不保证语义对齐**
    
- 直接混合 `K/V`，会让“头部 query 去 attend 背景 patch”，从而产生局部错误结构
    

### 3.6 模块一：Morphing Cross-Attention（MCA）

MCA 的做法很干净：

- **不是先把 source / target 的 K、V 混起来**
    
- 而是分别算两次 attention 输出，再按 `α` 对两个输出做加权
    

写成公式就是：

```Plain
MCA(Q^n, K_src/tgt, V_src/tgt)
= (1-α^n) * Attn(Q^n, K_src, V_src)
+ α^n * Attn(Q^n, K_tgt, V_tgt)
```

### 3.7 为什么 MCA 比 KV-Fused CA 更合理

这一步的本质是把“特征融合的位置”后移：

- **KV-Fused CA**：在 attention 之前混条件特征
    
- **MCA**：在 attention 之后混注意力输出
    

这样做的好处是：

- 每次 attention 仍然面对一个 **语义自洽** 的 source 或 target 条件
    
- query 不会被“半 source 半 target 的 patch 特征”误导
    
- 最终融合发生在更高层的语义结果上，而不是底层 patch token 上
    

论文用两种证据证明 MCA 有效：

- **Attention map 可视化**：MCA 会继续对准语义一致区域，KV-Fused CA 会偏到背景
    
- **t-SNE 特征轨迹**：MCA 的轨迹更稳定、更连续，KV-Fused CA 会出现断裂和抖动
    

### 3.8 模块二：Temporal-Fused Self-Attention（TFSA）

MCA 解决的是“这一帧像不像样”，但 **标准 3D 生成器逐帧独立生成**，所以时序平滑还不够。

TFSA 的思路是：

- 第 `n` 帧生成时，不只看当前帧 latent 的 self-attention
    
- 还把 **上一帧已经生成出的特征** 引入进来
    

公式为：

```Plain
TFSA(Q^n, K^n, V^n, K^(n-1), V^(n-1))
= (1-β) * Attn(Q^n, K^n, V^n)
+ β * Attn(Q^n, K^(n-1), V^(n-1))
```

其中：

- `β ∈ [0,1]`
    
- 论文经验设置 `β = 0.2`
    

### 3.9 TFSA 的直观理解

它其实不是用 source / target 的 3D latent 去硬混，而是把 **前一帧已经 plausible 的 3D 结果** 当作 temporal anchor。

所以它的收益是：

- 强化帧间连续性
    
- 保持局部几何和纹理演化更稳定
    
- 比“直接把 source / target latent 混进 self-attention”更不容易伤害语义合理性
    

### 3.10 模块三：Orientation Correction（OC）

前两个模块解决了“结构合理”和“时序平滑”，但作者还发现一个特殊问题：

- 中间 morphing 阶段常发生 **姿态跳变（orientation jump）**
    
- 尤其集中在 `α ≈ 0.5` 的中间阶段
    
- 主要表现为 **yaw 方向 90° / 180° / 270° 的突然跳转**
    

作者进一步统计了 1000 个 Trellis 生成物体的姿态分布，发现：

- 大多数是 canonical pose
    
- 但非 canonical pose 会聚集在若干固定 yaw 角度
    
- 这说明问题更像是 **Trellis 学到的 pose prior 偏置**，而不是随机噪声
    

因此他们提出了一个非常工程化但有效的策略：

1. 在 SS 阶段得到当前帧稀疏结构 `P^n`
    
2. 生成四个候选：
    
    1. `P^n`
        
    2. `P^n_90°`
        
    3. `P^n_180°`
        
    4. `P^n_270°`
        
3. 与上一帧结构 `P^(n-1)` 计算 Chamfer Distance
    
4. 选择 CD 最小的那个候选作为纠正后的 `P^n_hat`
    
5. 再送入后续 SLAT 阶段
    

这一步只在“姿态突然跳了”的时候发挥作用；如果没有 jump，原始 `P^n` 仍会因为 CD 最小而被保留。

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MDU3YWRjZmM1NDI0NDA2NWJmNDk0MzM3NzA5ZjYwYzJfZEVNWWdDNWJrNFBLMGlHeUdybXFmTlQzSnVQYjlsZm9fVG9rZW46QWlKeWJIWTE2bzlDMzd4REhZUWM2cmQybnFjXzE3ODI5ODA0MjU6MTc4Mjk4NDAyNV9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

### 3.11 完整 I/O 数据流总结

|阶段|输入|核心操作|输出|
|---|---|---|---|
|资产准备|源对象 `x_src`、目标对象 `x_tgt`|若是真实资产则做 3D inversion；若是 Trellis 生成资产则复用缓存 latent 与条件|`f_init^src`、`f_init^tgt`、`c_src`、`c_tgt`|
|帧初始化|source / target 初始特征与 morph weight `α^n`|对 SS / SLAT 初始特征做 slerp，得到第 `n` 帧的初始 noisy feature|`f_init^n`|
|SS Stage|`f_init_ss^n`、`c_src`、`c_tgt`|cross-attention 中使用 MCA，生成当前帧 sparse structure|`P^n`|
|姿态纠正|`P^n` 与上一帧 `P^(n-1)`|构造 0/90/180/270 度四个 yaw 候选，用 CD 选最接近上一帧的结构|`P^n_hat`|
|SLAT Stage|`f_init_slat^n`、`P^n_hat`、上一帧特征|cross-attention 继续使用 MCA；self-attention 中使用 TFSA 引入上一帧信息|`Z^n`（当前帧结构化 latent）|
|解码|`Z^n`|通过 Trellis decoder 解码为标准 3D 表示|mesh / NeRF / 3DGS 以及最终渲染帧 `x^n`|

## 4. 实验与关键信息

### 4.1 实验设置

论文正文与补充材料里给出的实现细节比较完整：

- **底座模型：** Image-to-3D Trellis
    
- **训练方式：** **完全不重新训练**，只替换 attention 模块并加入 orientation correction
    
- **硬件：** 单张 **A6000 GPU**
    
- **单帧耗时：** 约 **30 秒**
    
- **显存占用：** 约 **24GB**
    
- **采样步数：** SS / SLAT 两阶段都用 **25 steps**
    
- **CFG scale：** **5.0**
    
- **序列长度：** **50 帧**
    

评测数据：

- 从真实 3D 数据集和 Trellis 生成资产中取 **50 对 source-target pair**
    

评测指标：

- **FID**：衡量视觉合理性 / plausibility
    
- **PPL**：衡量帧间平均感知变化，越低越平滑
    
- **PDV**：衡量相邻帧变化方差，越低越均匀
    
- **AS**：用 Gemini-2.5 和 ChatGPT-5 选择最有审美吸引力的结果，占比越高越好
    
- **UP**：49 位有 CV / 图形背景的参与者做偏好投票
    

### 4.2 对比基线

论文覆盖了四类对比：

1. **匹配式 3D morphing**
    
    1. 3DInterp
        
    2. SLATInterp
        
    3. 对应关系由 DenseMatcher 提供
        
2. **2D morphing → 再升维到 3D**
    
    1. DiffMorpher
        
    2. FreeMorph
        
3. **直接插值底层噪声 / 条件特征**
    
    1. DirectInterp
        
4. **已有现代 3D morphing 方法**
    
    1. MorphFlow
        

另外，**3DMorpher** 因为没有完整开源代码，只做了补充材料里的定性对比。

### 4.3 定量结果

这是论文最有说服力的一组结果：

|方法|FID ↓|PPL ↓|PDV ↓|AS ↑|UP ↑|
|---|---|---|---|---|---|
|3DInterp|409.14|2.55|0.0006|1.00%|0.61%|
|SLATInterp|348.31|6.53|0.0010|0.00%|1.43%|
|DiffMorpher|208.08|6.65|0.0021|5.00%|0.82%|
|FreeMorph|164.68|5.89|0.0027|11.00%|3.27%|
|DirectInterp|150.94|3.72|0.0039|2.00%|5.51%|
|MorphFlow|284.96|2.41|0.0009|0.00%|1.63%|
|**MorphAny3D**|**111.95**|**2.47**|**0.0006**|**81.00%**|**86.73%**|

### 4.4 这些数字说明了什么

我觉得这张表的关键信息不是“全面第一”这么简单，而是说明了一个更重要的现象：

- **匹配式方法**（3DInterp / MorphFlow）在 PPL、PDV 上往往不差，说明“插值平滑”不难做到；但 FID 非常差，说明 **平滑不等于合理**。
    
- **2D-first 方法**（DiffMorpher / FreeMorph）虽然借到了强 2D 先验，但升到 3D 后时序会散，PPL / PDV 都不理想。
    
- **DirectInterp** 已经说明“把生成先验拉进来”是对的，但因为融合方式太粗暴，PDV 甚至最差。
    
- **MorphAny3D** 最大的贡献是把 **plausibility 与 smoothness 的平衡点** 往前推了一大截：
    
    - FID 最低，说明结构与语义最合理
        
    - PPL 只比最优值高 0.06，说明几乎没牺牲平滑性
        
    - AS / UP 极高，说明视觉主观效果上的优势很明显
        

### 4.5 消融实验

作者对三部分都做了消融：

|配置|FID ↓|PPL ↓|PDV ↓|
|---|---|---|---|
|KV-Fused CA|125.47|3.82|0.0013|
|MCA|112.18|3.66|0.0010|
|MCA + TFSA|113.22|2.87|0.0007|
|**MCA + TFSA + OC**|**111.95**|**2.47**|**0.0006**|

可以看到：

- **MCA** 主要拉低 FID，说明它首先解决的是“局部结构不合理”的问题。
    
- **TFSA** 主要拉低 PPL / PDV，说明它确实主要贡献于时序一致性。
    
- **OC** 继续把 PPL / PDV 往下拉，说明姿态跳变的确是一个独立且真实存在的问题，而不是个别 case。
    

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NWNmYTk0YmU0N2NmZWRkYjdlNTZmYTlmMzJmMDQwYmJfZDh6eFEyZ3ZoWkwwaTlURkkyNXAwWWFCOXN1NmI2U1pfVG9rZW46RFFBS2JsWmVHb0RNOTB4T2ZScGNneDR3bm9kXzE3ODI5ODA0MjU6MTc4Mjk4NDAyNV9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

### 4.6 定性结果与典型案例

论文给出的可视化结果非常有代表性。作者特别强调一个例子：

- **elephant → excavator**
    
    - MorphAny3D 会把 **象鼻** 与 **挖掘机吊臂** 隐式对齐
        
    - 生成中间态时，会出现同时带有两种概念特征、但仍然“像一个合理物体”的过渡体
        

这其实说明模型不是在做低层几何插值，而是在利用 3D 生成先验完成更高层的 **语义部件映射**。

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=Mjc2ZmZhY2RmMDljYTk4NjJkZmYzMjZmNGVjMWQ3MWNfNzlFNmlqemJkemR2TmhINnFtRDBUc2tkaVNDb3ZiaGxfVG9rZW46Q1NMQWJFQnJRbzJpMjZ4eUxXTWNjS2d1bkFjXzE3ODI5ODA0MjU6MTc4Mjk4NDAyNV9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

### 4.7 额外能力

这篇论文不只做了单一路径 morphing，还展示了三个延展能力：

1. **Disentangled 3D Morphing**
    
    1. 只在 SS stage 开启模块：主要改全局结构，保留局部细节
        
    2. 只在 SLAT stage 开启模块：主要改局部纹理 / 细节，不改整体轮廓
        
2. **Dual-Target 3D Morphing**
    
    1. SS stage 指向一个 target
        
    2. SLAT stage 指向另一个 target
        
    3. 等价于“一个目标管形体，一个目标管细节”
        
3. **3D Style Transfer**
    
    1. 用 style image 作为 SLAT 目标
        
    2. 保持源结构，把风格与局部几何特征迁移过去
        

### 4.8 泛化能力

作者还把同一套方法迁移到其他 SLAT-based 模型：

- **Hi3DGen**
    
- **Text-to-3D Trellis**
    

结论是：

- 如果底层仍是 SLAT 家族，MorphAny3D 的思路基本能直接复用
    
- 对 text-to-3D 版本，只需要把 MCA 里的图像条件换成文本条件
    
- 但论文也明确指出：**文本条件天然比图像条件更抽象**，所以 morphing 质量与时序一致性会下降
    

### 4.9 局限性

这部分也很重要，作者没有回避：

- **细粒度结构仍会出伪影**
    
    - 因为方法仍受 Trellis 本身的表示能力上限约束
        
- **对 yaw 对称物体的姿态纠正仍可能失效**
    
    - 因为 90° / 180° 旋转后，CD 可能不足以区分真正的姿态连续性
        
- **推理成本偏高**
    
    - 单帧 30s，整段 50 帧序列并不轻量
        

## 5. 个人评注 / 下一步

### 5.1 我觉得这篇论文真正重要的点

这篇工作最值得记住的，不是“又做出一个更好的 morphing demo”，而是它把下面这件事说清楚了：

**SLAT 不只是一个 3D 生成表征，它也是一个可以直接做编辑、控制、形变与组合的操作空间。**

换句话说，作者证明了：

- 一旦 3D foundation model 的 latent 足够结构化
    
- 下游很多任务未必要重新训练一个 task-specific 模型
    
- 很多时候只需要在 **attention 路由、条件注入、时序约束** 这些层面做“轻量但懂结构”的改造，就能把能力释放出来
    

这对 3D 生成方向很有启发，因为它提示了一条路线：

- **先把 3D 基础表征做对、做强、做统一**
    
- 再通过训练后模块化操作去扩展到 morphing / editing / stylization / composition / controllable generation
    

### 5.2 与当前技术视野的关系

从领域归类看，我认为它最适合放在 **3D生成**，而不是“三维重建”：

- 论文的核心不是恢复真实世界几何
    
- 而是借助 3D 生成先验做 **跨类别形变与资产生成式过渡**
    
- 它的价值更接近 **native 3D generative model 的可操作性扩展**
    

同时它和近期几条主线是连着的：

- **Trellis / SLAT 类结构化 3D latent**
    
- **training-free 3D editing / stylization / world synthesis**
    
- **将 3D 生成基础模型从“出结果”扩展到“可编辑资产工作流”**
    

### 5.3 可继续跟进的问题

我建议后续重点跟三件事：

1. **是否能嫁接到更强的 3D backbone**
    
    1. 如果把同样的注意力融合思路放到更新的 native 3D generator 上，是否还能稳定提升？
        
2. **是否可以做更长序列、更强控制**
    
    1. 例如加入显式轨迹约束、关键帧约束、部件级 morphing 约束
        
3. **是否能从 morphing 走向通用 3D edit program**
    
    1. 现在这篇已经展示了 disentangled morphing、dual-target morphing、3D style transfer
        
    2. 下一步很自然就是统一成更通用的 **3D asset editing / composition interface**
        

**一句话收尾：** MorphAny3D 的价值在于，它把“3D 生成模型能不能做高质量 morphing”这个问题，转成了“如何在结构化 latent 的 attention 中正确融合 source / target / temporal 信息”。这不是一次单点技巧优化，而是在说明 **结构化 3D latent 将成为后续 3D 编辑与可控生成的重要公共接口**。

