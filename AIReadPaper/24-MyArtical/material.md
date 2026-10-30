### 1、EgoX论文

egox是一个 video gen 模型，输入一段第三人称操作视频（exo video），生成这个人的第一人称操作视频。
首先会把 exo video 升维到 3d点云，然后给定一个轨迹，利用这个轨迹在点云中渲染一个 ego prior video，将 exo video 经过冻结的VAE后变为 $x_0$ latent，将 exo prior video经过 vae变为 $p_0$ ，待生成的第一人称latent是 $z_t$
- ego prior video 本来就是渲染的第一人称点云，所以很可能有破洞或者其他问题，但是他跟目标视角是齐平的，所以逐点对齐才有意义，ego prior video 的latent和 $z_t$ 在通道维度拼接
- ego video 和 要生成的第一人称视角就不是逐点对齐了，但是可以说 要生成的第一人称视角的很多信息需要从 ego video中查找，所以 $z_t$ 最好的做法是和 ego video 的latent 在宽度层面拼接，当作条件注入策略
修改Attention，改为 GGA 模式，利用第一人称视角下某一个 latent 的方向向量和第三人称下的方向做cos，让 cos 值大的 attention 更大

### JEPA

JEPA，首先看JE，联合嵌入，基本上就是对xy都encoder到latent中再去做预测。LeCun认为，如果直接在pixel层面的生成模型的话，必须把一些没用的像素也都画出来，容量和算力就被迫花在这些细节上，另外，对于下一状态应该是有很多种不同的表示，导致预测是容易生成平均的模糊的画面。
 
 $\hat{s}_y = P(s_x, z)$ 

因为 y 也需要经过 encoder，所以就容易造成，x和y的编码器都偷懒，都输出同一个常数。所以需要针对这个优化。
第一种的话就是，对比学习，通过提高负样本的数量（能量）。想 CLIP，SigLIP  其实不太支持，因为高维空间负样本量是巨大的

从结构入手，1、s_x 尽可能携带x的信息，s_y 尽可能携带y的信息，s_y必须是可以从 s_x预测的，z内容尽可能少 （z作为 predictor 的条件输入）

像是IJEPA还是VJEPA基本上都是用自监督的方法训的编码器（z是mask token），而后面的DINO-WM就是吧jepa的思想变成可交互的世界模型的一篇工作。

masked video作为 x ，使用 encoder，unmask video 作为 y，由于stop gradient，使用 ema方式从 x 的梯度中更新参数，作为 ema encoder。

VJEPA2 就是比较重要的，它分为两阶段训练了，第一个阶段也是训练一个 encoder，通过挖空的方式
- vjepa2 和 vjepa2-AC 使用的 z 不同，vjepa2 的z只是一个mask 的位置编码，而vjepa2AC的z是机器人动作（末端位姿、旋转、夹爪开合）和当前位姿【action-conditioned predictor】
![](assets/Material/file-20261030221331674.png)

### Cosmos

![](assets/Material/file-20261030221331674%201.png)
 genie给出的贡献可以理解为，他只输入视频和文本，通过将latent 的action 足够小，只有 8个32维的动作，让视频生成模型在预测下一帧的时候学会这些简单的动作
![](assets/Material/file-20261030221331674%202.png)

### WAM

dreamZero · LingBot-VA · Cosmos policy

### VGGT 和 DA3

VGGT 的关键确实是 over-complete multi-task supervision，VGGT 各个 head 之间主要是通过共享 backbone 产生“软耦合”，不一定存在严格的跨 head 几何一致性约束。

### Trellis 理解

Trellis的流程是首先对一个资产归一化到 -0.5到0.5 之间（最长边归一），进行体素化到 64^3 的 0-1 网格，接下来有两个去处，
- 第一个去encoder_ss_latent.py 的 **SparseStructure-VAE** 的 encoder 下采样**4倍**变成 16^3 x 8 的latent 【  
$1×64^3→32×64^3→128×32^3→512×16^3→8×16^3$ , 两次 stride-2 下采样 】。这就是第一个flow model学习的分布。推理时再还原回 $64^3$。
- 第二个多视图dino特征，渲染时渲染资产的150 views ，512x 512 的RGBA，再resize到 518x518 。经过DinoV2 Vit-L/14-reg【518/14=37】，输出的 x-prenorm包含【1个cls token， 4 个register token， 37^2 = 1369 个 patch token，patch token是 1024 维度】丢掉 cls 和 register-token之后，reshape 成  1024x37x37 的特征图，然后把 active voxel 的格心坐标用相机内外参投影到每个视角，然后取平均，存成npz文件，包含两个 key：indices（Nx3，uint8，取值0-63）、patchtokens（Nx1024）
```python
# 丢弃cls、register token
feature["x_prenorem"][:,dinov2_model.num_register_tokens + 1, :]
```
接下来把dino特征，encoder-latent.py 了 1024 -> 8。 把(coords, tokens) 组成稀疏张量，经过SLAT VAE encoder（稀疏卷积和稀疏注意力）压到每体素8维，这个 （Nx8）的稀疏张量就是 SLAT。空间分辨率始终为 64^3 

SLAT理解：叫做结构化潜在表示，主要是在稀疏的3D网格上定义局部潜在变量，同时表征几何和纹理。
