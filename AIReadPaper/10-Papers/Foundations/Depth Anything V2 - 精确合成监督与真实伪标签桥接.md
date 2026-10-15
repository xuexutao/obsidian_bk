---
type: paper-note
status: complete
domain: Foundations
paper: "Depth Anything V2"
year: 2024
arxiv: "2406.09414"
source: https://arxiv.org/abs/2406.09414
project: https://depth-anything-v2.github.io/
code: https://github.com/DepthAnything/Depth-Anything-V2
tags:
  - monocular-depth
  - synthetic-data
  - knowledge-distillation
  - DINOv2
created: 2026-10-15
updated: 2026-10-15
---

# 2. Depth Anything V2：精确合成监督与真实伪标签桥接

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | **Depth Anything V2** |
| 作者 | Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao |
| 会议 | NeurIPS 2024 |
| 模型 | DINOv2-S/B/L/G + DPT，约 25M–1.3B 参数 |
| 核心数据 | 595K 精确合成图像 + 62M 真实无标签图像 |
| 输出 | 仿射不变 inverse depth；另提供 metric fine-tuned 版本 |
| 重要性 | ★★★★★：把“真实标签越多越好”改写为“精确合成 teacher + 真实伪标签 student” |

> **一句话结论：** V2 发现 V1 的主要上限不在模型，而在真实深度标签：传感器、双目与 SfM 标签常在透明面、反射、动态区域和细边界上错误。它先用 595K 精确合成图像训练 DINOv2-G teacher，再用 teacher 给 62M 真实图像打细粒度伪标签，最后只用真实伪标签训练各尺寸 student，从而同时获得合成监督的锐利边界与真实分布的开放域鲁棒性。

---

## 1. 研究背景与问题转移

### 1.1 V1 已经强，但深度仍“粗”

V1 的 DINOv2 + DPT 在开放域相对深度上很稳，却容易丢失篮球网、栏杆、椅腿等薄结构，也会被镜子、玻璃、屏幕反射误导。同期 Marigold 等扩散模型细节更丰富，但慢、重，并且复杂布局鲁棒性不足。

V2 不把问题归因于判别式网络能力，而是追查监督源：真实 RGB-D、双目和 SfM 深度本身就在细结构、透明面、动态物体上有系统噪声。模型忠实拟合这些“GT”，自然学出模糊边界。

![](assets/Depth%20Anything%20V2%20-%20精确合成监督与真实伪标签桥接/figures/01-teaser.png)

> 图 1：V2 在鲁棒性、细节、速度和参数规模上的总览。Small 模型约 25M 参数、V100 约 60 ms；Large 约 335M、约 213 ms，并在 DA-2K 上达到 97.1%。来源：论文 Figure 1，第 1 页。

### 1.2 三个问题与答案

1. **判别式模型是否必然比扩散模型粗？** 否。用精确合成标签即可训练出锐利预测。
2. **为什么不能直接只用合成图？** 域差和场景覆盖有限，普通尺寸模型泛化失败。
3. **怎样同时保留精度与泛化？** 用最大的 DINOv2-G 合成 teacher，为真实开放域图像生成精确伪标签，再蒸馏到任意尺寸 student。

---

## 2. 输入、输出与架构

与 V1 相同：单张 RGB 输入，DINOv2 encoder + DPT decoder，训练分辨率 $518\times518$。基础模型输出仿射不变 inverse depth；metric 模型通过 ZoeDepth 风格微调或合成 metric 数据微调得到。

V2 的创新几乎全部在训练数据设计，而非架构。这使“数据结论”更可信，也让四种模型尺寸共享同一套配方。

---

## 3. 精确合成监督

### 3.1 真实标签为什么有害

- 深度传感器看不准玻璃和镜面；
- 双目在重复纹理、弱纹理处匹配错误；
- SfM 在动态对象和离群点上失败；
- 细杆、小孔和边界常被置为无效或被平滑。

论文消融甚至发现，只占 5% 的 HRWSI 真实训练数据就能明显破坏原本锐利的合成训练结果。

### 3.2 合成标签的优势与不足

5 个精确合成数据集共 595K：BlendedMVS、HyperSim、IRS、TartanAir、Virtual KITTI 2。它们可提供完整、稠密、无传感器噪声的深度，包含透明和反射表面的真实几何。

不足是颜色、布局和物体类别与真实世界存在域差；固定引擎场景也限制多样性。纯合成训练时，只有 DINOv2-G 能较好迁移到真实图，且仍会把云和人头预测错。

---

## 4. 三阶段 Teacher–Student 路线

![](assets/Depth%20Anything%20V2%20-%20精确合成监督与真实伪标签桥接/figures/02-training-pipeline.png)

> 图 7：V2 核心流程。精确但域窄的合成图训练 giant teacher；teacher 标注真实无标签图；student 只学习多样且精确的真实伪标签。来源：论文 Figure 7，第 6 页。

### 4.1 阶段一：合成 Teacher

用 DINOv2-G 在 595K 合成图上训练 160K iterations，batch 64。DINOv2-G 的预训练泛化用于跨越 synthetic-to-real 的第一段域差。

### 4.2 阶段二：真实图伪标签

Teacher 为与 V1 相同的 8 个真实来源、62M 图像生成 pseudo depth。伪标签兼具：

- 真实图像分布；
- 合成 teacher 的细边界；
- 比原始传感器/SfM 标签更完整的透明与反射区域。

### 4.3 阶段三：Student

ViT-S/B/L/G student 在伪标签真实图上训练 480K iterations，batch 192。对每张伪标签忽略损失最大的 10% 区域，降低 teacher 错误影响。关键消融表明，小模型**只用真实伪标签**通常比“合成 + 真实伪标签”更好，因此最终 student 不再混入合成图。

这是一种预测级蒸馏：不是在原始有标签数据上对齐 feature/logit，而是在大量额外真实图上学习 teacher 的几何输出，更适合 1.3B → 25M 的巨大容量差。

---

## 5. 损失函数

### 5.1 Scale-and-shift invariant loss

沿用 MiDaS 的仿射不变深度损失 $\mathcal L_{ssi}$，允许跨数据集全局尺度和偏移不一致。

### 5.2 Gradient matching

合成标签边界准确，因此梯度损失终于能发挥作用：

$$\mathcal L=\mathcal L_{ssi}+2\mathcal L_{gm}$$

权重从 0.5 提升到 4.0 时边界持续锐化，论文最终取 2.0，在标准指标和视觉锐度间折中。对粗糙真实标签使用同样损失没有明显收益，说明“损失设计必须和标签质量匹配”。

### 5.3 语义保持

在真实伪标签训练中保留 V1 的 DINOv2 feature alignment，以抑制伪标签噪声并保持语义迁移性。

---

## 6. 新评测：DA-2K

传统 NYUv2/KITTI 等 benchmark 有三类问题：标签噪声、场景单一、分辨率低。V2 构建 DA-2K：1K 高分辨率图像、2K 对人工复核的像素相对深度关系，覆盖室内、室外、非真实图像、透明/反射、恶劣成像、航拍、水下、对象八类场景。

它不替代稠密深度 benchmark，而是作为开放域鲁棒性的补充。需要警惕的是，DA-2K 由作者提出，结论仍需独立评测验证。

---

## 7. 实验结果

### 7.1 传统相对深度 benchmark

| 模型 | KITTI AbsRel/$\delta_1$ | NYU | Sintel | ETH3D | DIODE |
|---|---|---|---|---|---|
| V1 ViT-L | 0.076/0.947 | **0.043/0.981** | **0.458/0.760** | **0.127/0.882** | 0.066/0.952 |
| V2 ViT-L | **0.074/0.946** | 0.045/0.979 | 0.487/0.752 | 0.131/0.865 | 0.066/0.952 |

V2 在旧指标上仅与 V1 相当，部分数据还略差；作者认为旧 GT 对新模型的细边界与透明面改进不敏感。这个解释有道理，但也意味着不能只凭作者定性图就断言“全面更准”。

### 7.2 DA-2K

- 社区模型：Marigold 86.8%、GeoWizard 88.1%、DepthFM 85.8%、V1 88.5%；
- V2：Small 95.3%、Base 97.0%、Large 97.1%、Giant 97.4%。

### 7.3 透明表面

NTIRE 2024 Transparent Surface Challenge zero-shot $\delta_1$：MiDaS 0.259、V1 0.535、V2 0.836；简单微调 V2 达 0.912，接近第一名 0.917。

### 7.4 度量深度迁移

- NYUv2，V2-L：$\delta_1=0.984$、AbsRel=0.056、RMSE=0.206；
- KITTI，V2-L：$\delta_1=0.983$、AbsRel=0.045、RMSE=1.861。

这些数字与 V1 相近，V2 的优势主要是细节和开放域鲁棒性，而非标准 benchmark 大幅跃升。

![](assets/Depth%20Anything%20V2%20-%20精确合成监督与真实伪标签桥接/figures/03-v1-v2-comparison.png)

> 图 13：V1 与 V2 的开放域对比。V2 对薄结构、透明物体、镜面和复杂布局更稳；这类差异在低分辨率稠密 benchmark 上往往不充分反映。来源：论文 Figure 13，附录。

---

## 8. 关键消融

### 8.1 真实伪标签是否必要

对 ViT-S：纯合成 DA-2K 89.8%，加入/改用真实伪标签后 94.1–95.3%；NYU $\delta_1$ 从 0.928 升到 0.971–0.973。说明真实图是跨域桥梁，而不是可选增益。

### 8.2 手工真实标签 vs teacher 伪标签

同一 DIML 真实图：

- manual label：DA-2K 80.2%；
- pseudo label：89.7%。

六个深度数据集也几乎全面改善，直接证明“标签来源比图像来源更关键”。

### 8.3 数据多样性不能由重复训练替代

只用 SA-1B 11M、保持 480K iterations，不如 8 个真实数据集的 62M。模型需要更多场景分布，而不是在单一分布上多跑 epochs。

### 8.4 Test-time resolution scaling

训练在短边 518，推理提高到 2×、4× 分辨率仍能继续锐化边界。这是实用优点，但显存和延迟近似随 token 数迅速增长，不能视为“免费”。

---

## 9. 与 V1、DA3 的关系

| 版本 | 主要问题 | 核心答案 |
|---|---|---|
| V1 | 开放域覆盖不足 | 62M 无标签数据 + 难化 student + DINOv2 语义保持 |
| V2 | 真实深度标签粗糙、透明/细边界错误 | 合成 giant teacher + 真实伪标签桥接 + student 蒸馏 |
| DA3 | 单目深度无法统一相机与多视图几何 | plain ViT + depth-ray 最小目标 + 任意视图训练 |

V2 仍是单图深度基础模型；DA3 才把“Depth Anything”从深度估计扩展到视觉空间恢复。

---

## 10. 下游角色：GEN3C 的单目深度前端与误差传播

### 10.1 DA2 在 GEN3C 中不是终点，而是 3D condition 的入口

GEN3C 的核心数据流是 **RGB → DA2 depth → colored point cloud → target-view render → diffusion condition**。DA2 对每个源像素给出深度，GEN3C 再结合相机内外参反投影为彩色点云，并从目标相机渲染 RGB 与 coverage/disocclusion mask；视频扩散模型消费的不是裸深度，而是已经经过几何变换、与目标视角对齐的渲染条件。因此 DA2 在这里扮演的是**单目几何前端**：它不直接决定最终像素，却决定哪些历史外观被投到哪里、哪些区域被标记为已观测或需要生成。

尺度处理也属于接口的一部分。RE10K、DL3DV 中，DA2 相对深度需对齐 DROID-SLAM 的场景尺度；Waymo 中则刚性对齐 LiDAR 尺度。也就是说，DA2 擅长的细边界与开放域排序解决“局部形状”，SLAM/LiDAR 对齐解决“全局尺度”；若跳过后者，单目仿射不确定性会直接与相机平移量冲突。

### 10.2 depth → point cloud → render condition 的误差如何放大

这条链路不是把同一个深度误差原样传下去，而是会把标量误差变成目标视角中的结构化伪影：

1. **全局 scale/shift 偏差**：反投影后的场景被整体拉伸或压缩；目标相机一旦平移，渲染光流幅度就错误，表现为镜头执行不足/过冲与表面错位。
2. **低频表面形状偏差**：墙面、地面等大区域的深度倾斜会变成弯曲或漂浮点云；重投影后覆盖范围错误，进一步把本应补全的区域误标为“已有观测”，或反之。
3. **遮挡边界与前景膨胀**：边界深度混合会把前景颜色投到背景，在新视角中形成撕裂、双边和 foreground leakage；coverage mask 也随之错误，扩散模型容易把伪影当作可信条件复制。
4. **薄结构、透明/反射区域错误**：栏杆、玻璃、镜面若被放到错误深度，会产生 floaters、孔洞或不可能的重复结构。DA2 相比 V1 的锐利边界能减轻此问题，但不能消除单目歧义。
5. **逐帧不一致与反馈环**：动态视频或长视频更新中，每帧深度的小幅抖动会堆成多层点云；GEN3C 又对新生成帧估深度并追加 cache，于是可能形成“生成误差 → 深度误差 → 点云误差 → 条件误差 → 下一段生成误差”的闭环。

GEN3C 的噪声消融说明视频先验能修复**中等**深度扰动，但噪声增大后 PSNR/LPIPS 明显恶化；所以更准确的表述不是“生成模型不需要准深度”，而是“DA2 提供可容错的几何支架，扩散模型负责吸收有限误差与填补反遮挡”。

### 10.3 与 Lyra 2.0“几何只做路由”的对照

GEN3C 把点云渲染出的 **RGB/mask 直接作为生成条件**，几何既决定对应关系，也把历史外观硬投到目标视角；优点是局部相机控制强，缺点是深度边界、尺度和 cache 累积误差会被显式写进像素条件。

Lyra 2.0 对**长期空间记忆**采取更保守的策略：每帧 DA3 深度与点云独立保存，不融合成全局点云；下采样点云只负责检索覆盖目标区域的历史帧，全分辨率深度只负责把 canonical coordinates / depth warp 成对应提示并注入 attention 的 Q/K，历史 RGB 本身不做几何 warp。于是几何回答的是“该召回哪一帧、目标 token 应看哪个历史 token”，而不是“目标像素应该长什么样”。需要保留一个口径边界：Lyra 2.0 仍保留最近帧的 GEN3C 式 depth warp 以加强局部相机控制；“geometry as routing, not rendering”主要指它新增的**长期空间记忆通道**。

两者构成清晰演进：DA2/GEN3C 证明“不完美深度渲染也比纯 pose embedding 强”，Lyra 2.0 则进一步承认长程运行时硬渲染会把误差固化，因此把几何从外观提供者降为检索与 attention 路由器。

---

## 11. 局限与评价

### 11.1 论文明确局限

- 62M 图像伪标签和训练计算成本很高；
- 合成数据场景仍不够多样；
- DA-2K 只有稀疏像素对，不能替代稠密几何评测；
- 基础模型仍非统一 metric depth。

### 11.2 个人判断

V2 最强的结论是：**真实图不等于真实标签，synthetic 不等于不真实的几何。** 对深度这种传感器标签常有系统缺陷的任务，精确合成监督可能比大规模真实“GT”更可靠；真实图则最适合承担分布桥接。它为 DA3、Depth Pro、MoGe 等后续工作确立了一个共同趋势：架构变化越来越小，监督表示与数据质量越来越重要。

---

## 12. 复现成本与建议

- 完整 teacher 需要 DINOv2-G，student 还需 62M 伪标签，个人从零复现不现实；
- 实践建议直接用官方 Small/Base/Large 权重；
- 需要极细边界时可提高测试分辨率，但应先评估显存；
- 需要绝对尺度时，优先考虑 UniDepth、Metric3D v2、Depth Pro 或 DA3 metric，而不是把 V2 relative 输出直接当米制深度。

---

## 13. 材料来源

见 [sources.md](assets/Depth%20Anything%20V2%20-%20精确合成监督与真实伪标签桥接/sources.md)。
