### 1、EgoX论文

egox是一个 video gen 模型，输入一段第三人称操作视频（exo video），生成这个人的第一人称操作视频。
首先会把 exo video 升维到 3d点云，然后给定一个轨迹，利用这个轨迹在点云中渲染一个 ego prior video，将 exo video 经过冻结的VAE后变为 $x_0$ latent，将 exo prior video经过 vae变为 $p_0$ ，待生成的第一人称latent是 $z_t$
- ego prior video 本来就是渲染的第一人称点云，所以很可能有破洞或者其他问题，但是他跟目标视角是齐平的，所以逐点对齐才有意义，ego prior video 的latent和 $z_t$ 在通道维度拼接
- ego video 和 要生成的第一人称视角就不是逐点对齐了，但是可以说 要生成的第一人称视角的很多信息需要从 ego video中查找，所以 $z_t$ 最好的做法是和 ego video 的latent 在宽度层面拼接，当作条件注入策略
修改Attention，改为 GGA 模式，利用第一人称视角下某一个 latent 的方向向量和第三人称下的方向做cos，让 cos 值大的 attention 更大
