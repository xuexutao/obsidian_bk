### 1、EgoX论文

egox是一个 video gen 模型，输入一段第三人称操作视频（exo video），生成这个人的第一人称操作视频。
首先会把 exo video 升维到 3d点云，然后给定一个轨迹，利用这个轨迹在点云中渲染一个 ego prior video，将 exo video 经过冻结的VAE后变为 $x_0$ latent，将 exo prior video经过 vae变为 $p_0$ ，待生成的第一人称latent是 $z_t$
- ego prior video 本来就是渲染的第一人称点云，所以很可能有破洞或者其他问题，但是他跟目标视角是齐平的，所以逐点对齐才有意义，ego prior video 的latent和 $z_t$ 在通道维度拼接
- ego video 和 要生成的第一人称视角就不是逐点对齐了，但是可以说 要生成的第一人称视角的很多信息需要从 ego video中查找，所以 $z_t$ 最好的做法是和 ego video 的latent 在宽度层面拼接，当作条件注入策略
修改Attention，改为 GGA 模式，利用第一人称视角下某一个 latent 的方向向量和第三人称下的方向做cos，让 cos 值大的 attention 更大

### JEPA

JEPA，首先看JE，联合嵌入，基本上就是对xy都encoder到latent中再去做预测。LeCun认为，如果直接在pixel层面的生成模型的话，必须把一些没用的像素也都画出来，容量和算力就被迫花在这些细节上，另外，对于下一状态应该是有很多种不同的表示，导致预测是容易生成平均的模糊的画面。

因为 y 也需要经过 encoder，所以就容易造成，x和y的编码器都偷懒，都输出同一个常数。所以需要针对这个优化。
第一种的话就是，对比学习，通过提高负样本的数量（能量）。想 CLIP，SigLIP  其实不太支持，因为高维空间负样本量是巨大的

从结构入手，1、z_x 尽可能携带x的信息，z_y 尽可能携带y的信息，z_y必须是可以从 z_x预测的

