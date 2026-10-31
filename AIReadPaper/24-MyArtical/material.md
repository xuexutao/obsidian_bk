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

输入也是 518x518 大小的，patch 网格维 37x37 的，每帧图像token为37x37=1369，每个patch对应一个token。每个token的维度是1024，共1369x1024个标量。然后每一帧要加1个 camera token 和 4 个 VGGT register token。这样每帧对应的就是【（B，1374，1024）】，也是先丢弃DINO内部的cls 和 register token。
Alternating Attention运行24轮。交替之后在通道维度拼接，多头注意力一直是，16head，每个head 64 dim。 16x64=1024。
例如，对于一个3帧的图片集来说，(1,3,1374,1024) --reshape--> (3,1374,1024) --frameAttn--> (3,1374,1024) --globalAttn--> (1,4122,1024) --reshape--> (3,1374,1024) --...-->

### Trellis 理解

Trellis的流程是首先对一个资产归一化到 -0.5到0.5 之间（最长边归一），进行体素化到 64^3 的 0-1 网格，接下来有两个去处：
- 第一个去encoder_ss_latent.py 的 **SparseStructure-VAE** 的 encoder 下采样**4倍**变成 16^3 x 8 的latent 【  
$1×64^3→32×64^3→128×32^3→512×16^3→8×16^3$ , 两次 stride-2 下采样 】。这就是第一个flow model学习的分布。推理时再还原回 $64^3$。
- 第二个多视图dino特征，渲染时渲染资产的150 views ，512x 512 的RGBA，再resize到 518x518 。经过DinoV2 Vit-L/14-reg【518/14=37】，输出的 x-prenorm包含【1个cls token， 4 个register token， 37^2 = 1369 个 patch token，patch token是 1024 维度】丢掉 cls 和 register-token之后，reshape 成  1024x37x37 的特征体。这37x37展开后还是一个图片的patch呢～
- 要澄清一下，对于前期的体素化过程，trellis用了比较粗暴的方法，64级量化之后，他会判断有哪些体素和物体表面有交集，有的话，就认定为是一个active voxel，他存的时候只用了 1、这个active voxel的整数索引坐标（x,y,z） 还有2、体素格心坐标 $p_i$ 。直接忽略了原始的三角形、顶点、与体素的精确交点，后续trellis2似乎补上了。所以后面要做的就是，拿这个格心坐标，利用150个视图的相机位置，转换一下，看看某个格心坐标都落在每个视角的哪里，例如落在视角view1 的某个位置，转换的话大概率是个小数，他周围邻近的有4个 1024 维的向量【37x37的特征图】，然后对这4个1024维向量做双线形差值【4个token加权】得到一个 1024维的向量，然后又是一个简化，直接把150个视角的1024维向量做平均，不考虑可见性，最后得到这个voxel的dino特征，就存在npz文件的 patchtokens里面。这就是DINO特征聚合，每个active voxel有 1、一个三维整数坐标 $p_i$  2、一个1024维的特征 $f_i$  

做完特征融合之后其实还有一个步骤，使用sparse swin transformer 降维，不能让 64^3 个原始voxel全做注意力，改成只在 active voxel 上做 attention，且不仅如此还是用 swin8。也就是将 64^3 的全量体素，划分成 8x8x8 的局部窗口，然后再这个局部窗口内，只对active voxel 做attention，例如一个 窗口中有 M 个 active voxel，attention输入就是 $X^{M\times C}$，然后分别产生 Q、K、V，计算Attention(Q,K,V);这意味着，一个voxel更新自己的特征时，会根据内容的特性，从同一窗口中的其他 active voxel 中提取信息。 这引入一个问题，如果相邻的两个局部窗口彼此之间是没有交互的，所以 swin transformer 就会交替执行两种模型，第一种是 0、8、16、24。。 这样，第二种是偏移一半长度，改成 4、12、20、28. 这样就有了交互，所以Transformer这样交替下去就好，信息可以跨窗口边界传递。【Sparse Swin Transformer blocks】

接下来 Nx1024 维度的聚合 DINO 特征经过 Sparse Linear 到 Nx768 维度，经过 12 个Sparse Swin Transformer。还是 Nx768。之后，再经过 VAE 压缩到 $N \times 16 = N \times (\mu_8, \log \sigma_8^2)$ 这就是 SLat 里保存的，这16维度，经过VAE的 $z_i=\mu_i+\sigma_i \odot \epsilon_i$ 得到真正的 $z_i \in R^8$

【768只是trellis内部采用的一个维度，trellis默认是每个head 64维，所以一般trellis只需要12个heads就好】

```python
# 丢弃cls、register token
# vitl14-reg，每个图像块大小是 14*14*3=588 pixel，经过一个 588*1024 的MLP到1024维
###
#Conv2d(
#    in_channels=3,
#    out_channels=1024,
#    kernel_size=14,
#    stride=14,
#)
### 从588到1024，可以理解为1024个可学习的patch检测器：有些维度可能响应边缘、有些响应颜色组合等。他不增加信息，只是把像素投影到一个更方便Transformer处理的特征空间中，且经过Vit的多层self attention之后，第（i，j）个token已经不再只看第（i，j）个14*14patch了，他会与整张图的其他token交换信息。所以最终的1024维度更接近：这个局部图像块是什么，在整个物体中扮演什么角色。
feature["x_prenorem"][:,dinov2_model.num_register_tokens + 1, :]
```

SLAT理解：叫做结构化潜在表示，主要是在稀疏的3D网格上定义局部潜在变量，同时表征几何和纹理。

### hunyuan3D 理解

最核心的就是把几何和纹理生成分成两阶段生成

都经过重要性采样（5120点）、表面均匀采样之后（5120点），分别执行最远点采样（）
