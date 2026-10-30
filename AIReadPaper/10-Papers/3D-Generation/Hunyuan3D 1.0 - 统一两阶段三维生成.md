---
type: paper-note
status: complete
domain: 3D-Generation
paper: "Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation"
year: 2024
arxiv: "2411.02293"
doi: null
source: "https://arxiv.org/abs/2411.02293"
project: "https://huggingface.co/tencent/Hunyuan3D-1"
code: "https://github.com/Tencent-Hunyuan/Hunyuan3D-1"
code_commit: "454284503670312d4e06f6251c9be2f9f6d0fae7"
tags:
  - 3D-Generation
  - Image-to-3D
  - Text-to-3D
  - Multi-view-Diffusion
  - Sparse-view-Reconstruction
  - Triplane
created: 2026-10-30
updated: 2026-10-30
---

# Hunyuan3D 1.0：统一两阶段三维生成

## 基本信息

| 项目 | 内容 |
|---|---|
| 论文 | *Tencent Hunyuan3D-1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation* |
| 作者 | Xianghui Yang、Huiwen Shi、Bowen Zhang、Fan Yang、Jiacheng Wang 等，Tencent Hunyuan3D Team |
| 版本 | arXiv:2411.02293v5，2025-01-23 |
| 任务 | Text-to-3D 与 Image-to-3D；输出可纹理化、可编辑的显式 mesh |
| 核心路线 | 单图 → 六视图扩散 → 稀疏视图前馈重建 → triplane/SDF → Marching Cubes mesh |
| 两个版本 | Lite：SD 2.1 多视图骨干；Standard：SDXL 多视图骨干，论文称参数量约为 Lite 与既有同类模型的 3 倍 |
| 官方代码 | 已开源推理、权重下载说明与 Baking；截至 2026-10-30 **未开源训练代码** |
| 精读 commit | `454284503670312d4e06f6251c9be2f9f6d0fae7` |
| 评分 | ★★★★☆：两阶段工程闭环清晰、速度与质量平衡优秀；但训练细节、硬件、数据规模和定量消融披露不足 |

## 1. 一句话结论

Hunyuan3D 1.0 的关键不是直接在 3D latent 中“从零生成”，而是先借助强 2D diffusion 把单图歧义展开为六个固定水平视角，再用“六张有标定生成图 + 一张无标定原条件图”的混合输入恢复高分辨率 triplane、NeuS 风格 SDF 与显式 mesh；它把单图到 mesh 压缩到 Lite 约 10 秒、Standard 约 25 秒，但论文的可复现信息明显弱于其工程完整度。

![](assets/Hunyuan3D%201.0%20-%20统一两阶段三维生成/figures/01-teaser.png)

> 图 1：Hunyuan3D 1.0 的 3D 资产总览。论文用同一框架接收文本或单图，最终输出带几何与外观的 3D 资产。需要注意，文本入口并非文本直接驱动 3D 网络，而是先由 Hunyuan-DiT 生成条件图，再进入 Image-to-3D 主链路。  
> 来源：论文 Figure 1，PDF 第 1 页，https://arxiv.org/abs/2411.02293

---

## 2. 研究背景与问题定义

### 2.1 为什么 3D 生成不能简单复制 2D scaling law

语言、图像和视频模型的快速扩展依赖海量训练数据，而高质量 3D 资产同时包含几何、拓扑、纹理、材质和相机可见性，数据规模与标准化程度远弱于 2D。论文引用的 Objaverse-XL 虽然已达到千万级资产，仍远小于互联网图文数据，而且资产质量、类别、纹理和拓扑差异极大。

因此，Hunyuan3D 1.0 选择的不是“完全摆脱 2D 先验”，而是把成熟的 2D diffusion 当作**单视图歧义展开器**：先从条件图合成多个新视角，再把问题从不适定的单图重建，转换为信息更充分的稀疏多视图重建。

### 2.2 两类既有路线的矛盾

| 路线 | 优点 | 主要瓶颈 |
|---|---|---|
| SDS / 逐对象优化 | 可复用强 2D diffusion prior，质量上限高 | 每个对象通常需 5 分钟到 1 小时，容易过饱和，难以成为交互式资产工具 |
| 单次前馈重建 | 数秒输出 3D | 对未见物体、薄结构和单图歧义的泛化不足 |
| 多视图扩散 + 重建 | 将语义生成与几何恢复解耦 | 多视图不一致、采样成本、固定视角覆盖不足；重建器通常只接受全部有位姿或全部无位姿输入 |

Hunyuan3D 1.0 的问题可以概括为：**能否把 2D diffusion 的泛化能力和前馈 3D 重建的速度接成一个系统，同时把多视图扩散产生的噪声、不一致和不可见区域交给下游重建器吸收？**

### 2.3 核心贡献

1. **统一文生/图生 3D 的两阶段框架。** 文本先经 Hunyuan-DiT 得到条件图，之后与图生 3D 共用同一主链路。
2. **0° elevation 的六视图环轨。** 六张目标图位于同一水平圆环，方位角为 $0^\circ,60^\circ,120^\circ,180^\circ,240^\circ,300^\circ$，便于相邻视角共享可见区域。
3. **视角与时间共同自适应的 CFG。** 正面视图保持更强条件控制，背面逐步降低 CFG，减少阴影和“把正面 logo 复制到背面”的问题。
4. **Hybrid Inputs。** 六张生成图带预定义相机，原始条件图位姿未知；模型用全零相机 embedding 标记原图，使其补充水平环轨看不到的顶部、底部和遮挡区域。
5. **轻量 triplane 超分辨率。** 仅在 $64\times64$ token 上做 Transformer，再用线性 unpatchify 展开到 $256\times256$，避免在高分辨率 token 上继续做二次复杂度的 self-attention。

---

## 3. 任务定义与完整数据流

### 3.1 输入、输出与假设

图生 3D 输入一张前景对象图像；文生 3D 输入中英文文本，但代码先把文本变成一张 $1024\times1024$ 条件图。两条路线最终都输出显式 mesh，可选顶点颜色、UV texture、GLB 和 DUSt3R 辅助 Baking。

系统隐含三个重要假设：

- 对象可被归一化到以原点为中心的有限包围域；
- 六张水平环轨视图能够覆盖主体的大部分侧表面；
- 即使多视图扩散存在不一致，下游网络也能通过训练学会“容错重建”，而不是逐像素忠实复现每张生成图。

### 3.2 图生 3D 主链路

![](assets/Hunyuan3D%201.0%20-%20统一两阶段三维生成/figures/02-method-pipeline.png)

> 图 2：完整两阶段框架。上半部分是单图经多视图 diffusion 生成六视图，再由 sparse-view reconstruction 恢复 mesh；下半部分分别展开 Reference Attention、Adaptive CFG，以及 triplane token 经 Transformer 和线性 upsampler 得到 $256\times256$ 三平面。  
> 来源：论文 Figure 2，PDF 第 3 页，https://arxiv.org/abs/2411.02293

主链路可写成：

$$I_{cond}\xrightarrow{\text{MVD}}\{I_v\}_{v=1}^{6}\xrightarrow{\text{SVRM}(I_{cond},\{I_v,K_v,R_v,t_v\})}T\xrightarrow{f_{\theta}}\operatorname{SDF}(\mathbf{x})\xrightarrow{\text{MC}}\mathcal{M}$$

其中 $I_{cond}$ 是原始条件图，$I_v$ 是六张生成视图，$K_v,R_v,t_v$ 是预定义相机，$T\in\mathbb{R}^{3\times120\times256\times256}$ 是三平面特征，$f_{\theta}$ 将任意 3D 查询点映射到 SDF 与 RGB，$\mathrm{MC}$ 表示 Marching Cubes，$\mathcal{M}$ 为显式网格。

实际执行顺序是：

1. U²-Net/rembg 去背景并把主体居中到 $512\times512$ RGBA；
2. Lite 或 Standard MVD 一次生成 $3\times2$ 六视图拼图；
3. 把六视图重排为固定方位角顺序，并在末尾加入原条件图；
4. DINOv2 编码七张图，相机向量通过 AdaNorm 调制图像 encoder；
5. 12288 个可学习 triplane query 对图像 token 做 16 层 cross/self-attention；
6. $64\times64\times1024$ 低分辨率 token 线性展开成 $256\times256\times120$ 三平面；
7. 在规则 $384^3$ 空间网格上查询 SDF，Marching Cubes 提取表面；
8. 可选 xatlas 展 UV、nvdiffrast 生成纹理，或再用 DUSt3R 对齐生成图和 mesh 渲染图做 Baking。

### 3.3 文生 3D 的真实入口

论文只明确说系统“involves Hunyuan-DiT”，没有证据表明文本 embedding 直接进入多视图 UNet。官方代码对应的是：

$$p\xrightarrow{\text{Hunyuan-DiT}}I_{cond}\xrightarrow{\text{RemoveBG}}I_{rgba}\xrightarrow{\text{Image-to-3D pipeline}}\mathcal{M}$$

代码默认给 prompt 追加“白色背景、3D 风格、最佳质量”，以 25 步 Hunyuan-DiT 生成 $1024\times1024$ 图像。因此，1.0 的“统一”主要是**共享后半段资产生成管线**，而不是文本和图像在同一 3D 模型中原生统一编码。

---

## 4. 核心方法

### 4.1 多视图扩散：把六张图作为一个大画布联合生成

<span style="color: red">Lite 以 SD 2.1 为骨干，目标拼图为 $960\times640$，每个视图约 $320\times320$；Standard 以 SDXL 为骨干，拼图为 $1536\times1024$，每个视图约 $512\times512$。</span>六张图不是六次独立采样，而是作为一个 $3\times2$ grid 联合去噪，因此同一个 UNet receptive field 与 attention 能跨视图交换信息。

相机固定在 0° elevation，原因不是它能观察整个物体，而是相邻方位角之间具有更稳定、更大的重叠区域。代价也很明确：顶部与底部天然缺失，所以论文必须在第二阶段引入原始条件图补偿。

### 4.2 Reference Attention：条件图不是普通 cross-attention

条件图会额外通过同一个 denoising UNet。在每个 self-attention 层，代码先保存条件图分支产生的 K/V，再在生成六视图分支读取时沿 token 维拼接：

$$K'=[K_{gen};K_{ref}],\qquad V'=[V_{gen};V_{ref}]$$

这意味着条件图直接成为每层 self-attention 的可检索记忆，作用比只在输入端拼接图像 embedding 更强：六张目标图在每个尺度都能查询条件图的语义、纹理和局部外观。

### 4.3 View-aware Adaptive CFG

论文把正面视图的 CFG 写为：

$$w_t=2+16\left(\frac{t}{1000}\right)^5$$

其他视角再乘视角系数：

$$w_{t,v}=w_t\tau_v,\qquad \tau_v\in[0.5,1],\qquad \tau_{front}=1,\quad \tau_{back}=0.5$$

若按常见反向扩散约定，早期 $t\approx1000$ 时正面 CFG 约为 18，后期降到 2。高 CFG 先锁定正面轮廓与语义，随后降低以恢复纹理细节；视角离正面越远，条件图可见区域越少，CFG 越低，避免把正面图案强行复制到不可见背面。

Lite 代码把六个 grid slot 的视角权重写成 $[1,5/6,2/3,1/2,2/3,5/6]$，与论文的前后视图定义一致。值得注意的是，当前 Standard 代码默认采用恒定 CFG=2.0，只有 Lite 显式实现逐视角 adaptive CFG；这与论文把 Adaptive CFG 描述成一般方法存在差异，说明公开推理分支未必完全等同于论文内部实验版本。

### 4.4 Hybrid Inputs：让已知位姿与未知位姿共存

六张 MVD 输出具有固定相机，因此重建器知道它们的相对方位；原始条件图可能是俯视、仰视或任意相机，无法可靠标定。论文没有粗暴估计原图位姿，而是给它一个全零 camera embedding，形成 view-agnostic branch：

$$e_{cam}(I_v)=[\operatorname{vec}(T_v),f_x,f_y,c_x,c_y]\in\mathbb{R}^{20},\qquad e_{cam}(I_{cond})=\mathbf{0}$$

这样网络能区分“六张有几何语义的固定视图”和“只提供补充视觉证据的原图”。相机向量在代码中不是先转 Plücker ray，而是通过 MLP 调制 DINOv2 每层 AdaNorm。这个设计更接近“告诉图像编码器当前视图属于什么相机角色”，而非显式把像素投射到 3D 射线。

### 4.5 Triplane Transformer 与线性超分辨率

七张 $504\times504$ 图像进入 DINOv2 ViT-B/14。每张图被切成 $36\times36=1296$ 个 patch，加一个 CLS token，共 1297 token；七视图合并后为 $9079\times768$ context。

三平面初始 query 数为：

$$N_q=3\times64\times64=12{,}288$$

16 层 Transformer 让三平面 query 对图像 context 做 cross-attention，并在 query 内做 self-attention。输出形状为 $[1,12288,1024]$，随后每个低分辨率 token 用线性层映射到 $120\times4\times4=1920$ 维，再重排为：

$$[1,3,120,256,256]$$

这个操作与 PixelShuffle 思想相似，但不在高分辨率 token 上继续做 attention。若直接把三平面 query 从 $64^2$ 提升到 $256^2$，token 数会放大 16 倍，self-attention 的两两交互规模会放大约 256 倍；线性 unpatchify 把主要计算留在低分辨率阶段。

![](assets/Hunyuan3D%201.0%20-%20统一两阶段三维生成/figures/03-triplane-super-resolution.png)

> 图 3：低分辨率 triplane 与线性超分辨率后的对比。高分辨率版本改善了钟面数字、边缘和局部凹凸，但论文只给出定性结果，没有报告 CD/F-score 的单模块增益。  
> 来源：论文 Figure 3，PDF 第 3 页，https://arxiv.org/abs/2411.02293

### 4.6 从三平面到 SDF、RGB 与显式 mesh

任意查询点 $\mathbf{x}\in\mathbb{R}^3$ 分别投影到 XY、XZ、YZ 三个平面，每个平面双线性采样 120 维特征，拼接得到 360 维：

$$\phi(\mathbf{x})=[T_{xy}(\pi_{xy}(\mathbf{x}));T_{xz}(\pi_{xz}(\mathbf{x}));T_{yz}(\pi_{yz}(\mathbf{x}))]\in\mathbb{R}^{360}$$

两层 MLP 执行 $360\rightarrow64\rightarrow4$，输出一个 raw SDF 与三个 sigmoid RGB 通道：

$$(s(\mathbf{x}),\mathbf{c}(\mathbf{x}))=f_\theta(\phi(\mathbf{x}))$$

仓库保留 NeuS 风格 ray marcher，可把 SDF 通过 CDF 差转为 alpha 做体渲染；但最终生产链路不是输出 NeRF，而是在 $[-0.6,0.6]^3$ 的 $384^3$ 规则网格上查询 SDF，再以零等值面运行 Marching Cubes。之后用 Open3D 清理、平滑和简化，并可用 xatlas 展 UV、nvdiffrast 生成 $1024^2$ texture。

---

## 5. 数据集与训练设置

### 5.1 内部数据与渲染

两阶段都在“类似 Objaverse”的内部数据集上训练。作者只说明过滤了复杂场景、无有效纹理和严重畸变资产，并把对象缩放到 unit sphere；**资产数量、来源许可、类别分布、训练/验证划分和清洗保留率均未披露**。

条件图随机采样：

- elevation：$[-20^\circ,60^\circ]$；
- azimuth：$[0^\circ,360^\circ]$；
- 随机 HDR；
- FOV：论文写作 $U(47,0.01)$；
- camera distance：论文写作 $U(1.5,0.1)$。

后两项不符合常见“上下界”记法，论文也没有解释第二个参数是范围还是扰动，因此不能擅自改写为高斯分布或 $\pm$ 区间。

每个对象渲染 24 张目标图：方位角从 $0^\circ$ 到 $345^\circ$，步长 $15^\circ$；elevation 固定 0°，FOV 为 47.9°，相机距离 1.5，Blender 以 $1024\times1024$ 渲染。推理六视图相当于从 24 个训练方位中每隔 60°取一个。

### 5.2 两阶段分开训练

| 阶段 | 论文披露的设置 | 未披露内容 |
|---|---|---|
| 多视图扩散 | Lite=SD2.1，Standard=SDXL；$3\times2$ grid；条件图训练时随机 resize 到 `[256,512]`，推理固定 512；目标图训练为 $320^2$ | diffusion objective、sampler、optimizer、学习率、batch size、训练步数、硬件 |
| Sparse-view reconstruction | DINO encoder；triplane latent；先用 $256^2$ 多视图训练，再用 $512^2$ 微调 | RGB/mask/SDF/Eikonal/normal 等损失及权重、渲染采样策略、训练周期与硬件 |

论文方法部分称 Standard 单视图格子为 $512^2$，Implementation 又说 target images 均 resize 到 $320^2$。作者未说明这是推理/训练差异、额外高分辨率微调，还是文本表述不完整，笔记不强行统一。

---

## 6. 实验结果与消融

### 6.1 评测协议

作者在 GSO 与 OmniObject3D 上随机抽取“约 70 个对象”，但没有明确是每个数据集各 70 个还是合计 70 个。每个预测 mesh 表面采样 10,000 点，计算 Chamfer Distance 与不同阈值下的 F-score；对生成姿态未知的方法先用 ICP 对齐。

ICP 能剥离全局位姿误差，使评测更聚焦形状本身，但也会弱化“模型是否生成在正确坐标系”这一资产可用性问题。

### 6.2 GSO 定量结果

| Method | CD ↓ | F-score@0.1 ↑ | F-score@0.2 ↑ | F-score@0.5 ↑ |
|---|---:|---:|---:|---:|
| OpenLRM | 0.214 | 0.605 | 0.840 | **0.997** |
| InstantMesh | 0.216 | 0.670 | 0.862 | 0.977 |
| Hunyuan3D-Lite | 0.199 | 0.661 | 0.877 | 0.986 |
| **Hunyuan3D-Std** | **0.175** | **0.735** | **0.910** | 0.987 |

Standard 在 CD、F@0.1 与 F@0.2 上最佳，说明其对精细几何和严格阈值更有优势；但 F@0.5 的最优值属于 OpenLRM，而 Lite 的 F@0.1 也略低于 InstantMesh。因此论文“所有 F-score 均优于全部 baseline”的文字概括并不完全符合表格。

### 6.3 OmniObject3D 定量结果

| Method | CD ↓ | F-score@0.1 ↑ | F-score@0.2 ↑ | F-score@0.5 ↑ |
|---|---:|---:|---:|---:|
| TripoSR | 0.157 | 0.776 | 0.915 | **0.999** |
| OpenLRM | 0.158 | 0.754 | 0.940 | 0.992 |
| Hunyuan3D-Lite | 0.150 | 0.786 | 0.938 | 0.997 |
| **Hunyuan3D-Std** | **0.136** | **0.814** | **0.948** | 0.998 |

Standard 仍在更严格的 CD、F@0.1、F@0.2 上领先；F@0.5 由 TripoSR 以 0.999 略胜。我的判断是，Standard 的优势主要体现在**局部表面位置更精确**，而在宽松阈值下，各方法都已接近饱和。

### 6.4 定性结果

![](assets/Hunyuan3D%201.0%20-%20统一两阶段三维生成/figures/04-qualitative-comparison.png)

> 图 4：单图生成定性对比。论文认为 OpenLRM/TripoSR 在箱体、肥皂等几何上失真且纹理模糊，InstantMesh 有更多表面细节但仍存在椅座、杯子 logo 和方盒边角伪影；Standard 在轮廓、纹理清晰度和局部细节上最稳定。  
> 来源：论文 Figure 4，PDF 第 6 页，https://arxiv.org/abs/2411.02293

这组图对“Standard 质量更高”有直观支持，但它是作者选择的少量案例，没有报告随机样本的失败率或按类别统计，不能据此推断所有类别都稳定领先。

### 6.5 Adaptive CFG 消融

![](assets/Hunyuan3D%201.0%20-%20统一两阶段三维生成/figures/05-adaptive-cfg-ablation.png)

> 图 5：固定 CFG=4、只随时间变化的 CFG 与视角+时间联合自适应 CFG。固定 CFG 容易让背面变暗；只做时间自适应仍会把正面仙人掌图案复制到背面；视角自适应降低背面条件强度后更自然。  
> 来源：论文 Figure 7，PDF 第 7 页，https://arxiv.org/abs/2411.02293

该消融验证了机制直觉，但只有一只杯子的定性结果，没有跨类别的 CLIP 相似度、多视图一致性、LPIPS 或人工偏好统计。

### 6.6 Hybrid Input 消融

![](assets/Hunyuan3D%201.0%20-%20统一两阶段三维生成/figures/06-hybrid-input-ablation.png)

> 图 6：仅使用六张水平生成视图时，大蒜顶部近似平面；加入无标定原图后，模型恢复了蒜根附近的凹陷。这说明原始条件图确实能补充 0° elevation 环轨看不到的顶部信息。  
> 来源：论文 Figure 8，PDF 第 7 页，https://arxiv.org/abs/2411.02293

Hybrid Input 的有效性依赖原条件图真的看到了缺失区域。如果原图也是严格水平正视，或者顶部被遮挡，全零 camera embedding 并不能凭空恢复可靠几何。

### 6.7 速度、用户研究与口径问题

| 口径 | 报告时间 |
|---|---:|
| 摘要：多视图生成 | 约 4 秒 |
| 摘要：重建 | 约 7 秒 |
| 方法部分：重建网络前向 | 2 秒以内 |
| Lite：单图到 mesh | 约 10 秒 |
| Standard：单图到 mesh | 约 25 秒 |
| UV unwrapping + texture baking | 额外约 15 秒 |

论文没有报告 GPU 型号，且“2 秒重建”与摘要“7 秒重建”的阶段边界未解释。最稳妥的使用结论是：Lite 约 10 秒、Standard 约 25 秒生成 mesh，不包含约 15 秒的 UV/texture 后处理。

作者还报告 Hunyuan3D 在 3D plausibility、image-asset alignment、geometry details、texture details、texture-geometry coherency 五个用户偏好维度上均最高，但没有公开参与者人数、样本量、盲测方式、置信区间或显著性检验，因此只能视为支持性证据，而不是严格统计结论。

---

## 7. 与相关工作的关系

### 7.1 与 Zero123++ 的关系

Hunyuan3D 1.0 直接沿用了“把多视图 tile 到一张大图里联合扩散”和 Reference Attention 的思路。它的主要扩展是更大 backbone、水平环轨、view-aware CFG，以及一个为生成视图噪声专门训练的下游重建器。

### 7.2 与 LRM / OpenLRM / InstantMesh 的关系

LRM 系列倾向于从单图或少量图直接回归三平面；Hunyuan3D 把“补全多视角”明确前置，并让 SVRM 同时看到六张有标定生成图与一张无标定原图。代价是系统更长、显存更高，而且误差可能从 T2I/MVD 级联到几何。

### 7.3 与后续 Hunyuan3D 2.x 的位置

1.0 的代表性是**2D 多视图扩散 + 前馈重建**路线：它把几何生成建立在生成视图上。后续 2.x 更强调 3D shape latent、VecSet diffusion、高分辨率几何和纹理生成。理解 1.0 很重要，因为它解释了混元 3D 系列最初如何用强 2D prior 快速建立可用资产管线，也为后续为什么要减少多视图中间表示的信息瓶颈提供了参照。

---

## 8. 局限与批判性评价

### 8.1 论文明确暴露的问题

- 水平环轨缺失顶部和底部信息；
- 多视图扩散会产生噪声与跨视图不一致；
- 固定 CFG 会导致背面阴影或正面纹理复制；
- 低分辨率 triplane 容易出现 aliasing 与细节损失。

### 8.2 我认为更关键的局限

1. **文本模式存在单图信息瓶颈。** 文本先被压成一张 T2I 图，后续 3D 只能忠实恢复这张图表达出来的对象；文本中不可见的背面、内部结构和材质属性没有独立通道进入 3D。
2. **误差是级联的。** Hunyuan-DiT 的构图错误、RemoveBG 的 alpha 错误、MVD 的跨视图冲突都会传入 SVRM。重建器能容错，但不能区分“应被修正的扩散幻觉”和“真实的罕见结构”。
3. **数据与训练不可复现。** 内部数据规模、许可证、损失函数、优化器、batch size、训练步数和硬件都没有公开；官方仓库也只有推理代码。
4. **实验规模偏小。** “约 70 个对象”的口径不清，且定量评测经 ICP 对齐；三个核心模块的消融均只有定性案例。
5. **资产可用性仍有限。** 1.0 输出的是从 SDF 提取并后处理的三角网格，没有原生保证干净拓扑、四边面边流、部件结构、可绑定性和 PBR 材质。这正是后续 [[QuadGPT - 原生四边形网格自回归生成]] 等工作继续处理的问题。
6. **Baking 的商业边界。** 官方 Baking 依赖 DUSt3R，README 明确提示其 CC BY-NC-SA 4.0 许可不适合商业使用，工程集成时需要单独评估。

### 8.3 适用边界

更适合：单主体、白底或可稳定抠图、外表面为主、相对封闭的消费品/角色/装饰资产；不适合：复杂场景、多物体遮挡、透明/反射材质、内部结构、严格拓扑或工业 CAD 精度需求。

---

## 9. 复现成本与实践建议

### 9.1 官方显存门槛

| 版本 | 常规模式 | `--save_memory` |
|---|---:|---:|
| Standard | 30 GB VRAM | 24 GB VRAM |
| Lite | 22 GB VRAM | 18 GB VRAM |

分阶段运行时，官方给出的最低参考是：图生 Standard 约 16 GB、图生 Lite 约 10 GB；但速度口径没有绑定具体 GPU。

### 9.2 最小可跑路径

1. Linux + CUDA 12.1 环境；
2. 安装仓库依赖、PyTorch3D 与 nvdiffrast；
3. 下载 `tencent/Hunyuan3D-1` 权重；文本模式额外下载 Hunyuan-DiT；
4. 先跑 Lite image-to-3D，关闭 texture mapping 与 Baking，确认 `views.jpg` 和基础 OBJ；
5. 再打开 Standard、UV texture、GIF 或 DUSt3R Baking；
6. 若显存不足，按官方 separately scripts 拆成 T2I、MVD、SVRM 三段。

### 9.3 调试建议

- 先检查 `img_nobg.png`：错误 alpha 会污染所有后续阶段；
- 保存并逐张检查六视图：若背面出现正面 logo，优先检查 Lite adaptive CFG 与视图重排；
- Standard 与 Lite 的 grid 顺序不同，不能共用无重排的拆图逻辑；
- 检查 SVRM 输入必须是六视图 + 原图共 7 张，实际 resize 到 504 而不是 512；
- 分别验证“顶点颜色 OBJ”和“UV texture OBJ”，避免把 Baking 问题误判为几何问题。

---

## 10. 个人启发与后续问题

### 10.1 关键启发

Hunyuan3D 1.0 展示了一个很实用的系统原则：**不要要求一个模型同时解决所有不确定性。** MVD 负责利用 2D prior 补全外观和视角，SVRM 负责把不完全一致的观测投影回 3D，一层线性 upsampler 再把昂贵 attention 与高分辨率几何解耦。这种模块化设计在 2024 年非常适合快速做出生产原型。

### 10.2 对 Asset-JEPA / 3D 表征研究的启发

1. 六视图与原图的“已标定/未标定混合输入”可以视为多源证据融合问题；后续可研究让模型显式估计每个视图的可靠度，而不是只用全零 camera embedding 区分角色。
2. 当前 triplane 学的是外表面场，没有显式拓扑和部件 token。若要走向 Asset-JEPA，可把 geometry、topology、UV、material 分成可对齐但非完全耦合的表征子空间。
3. MVD 的不一致可以被当作训练时的 corruption：与其只追求更一致的生成图，也可以训练 3D encoder 对视图冲突保持鲁棒，并输出不确定性。

### 10.3 建议继续追踪的问题

- Standard 的论文实验是否真的使用 view-aware adaptive CFG？
- Hybrid Input 能否扩展为多张任意位姿、带置信度的未标定图？
- 能否用原生 3D latent 替代六视图 grid，同时保留 2D diffusion 的语义泛化？
- 如何把 SDF mesh 升级为部件化、四边面拓扑、可绑定和 PBR-ready 资产？

---

## 11. 代码级实现

### 11.1 开源范围与仓库概览

官方仓库开源推理、检查点下载方式、UV texture 与 Baking；README 的 Training 仍未勾选。仓库中没有 Dataset/DataLoader、optimizer、训练循环、loss 汇总和 checkpoint 保存入口。因此本节能精确拆解的是**推理张量流与资产导出**，不能反推完整训练目标。

| 论文模块 | 代码入口 | 关键产物 |
|---|---|---|
| 统一管线 | `main.py` | 串联 T2I、去背景、MVD、SVRM、Baking、GIF |
| 文本转条件图 | `infer/text_to_image.py::Text2Image` | $1024^2$ RGB 条件图 |
| 去背景 | `infer/removebg.py::Removebg` | $512^2$ RGBA 主体图 |
| 多视图生成 | `infer/image_to_views.py::Image2Views` | Lite $960\times640$ / Std $1536\times1024$ grid |
| Lite MVD | `mvd/hunyuan3d_mvd_lite_pipeline.py` | SD2.1、Reference Attention、Adaptive CFG |
| Standard MVD | `mvd/hunyuan3d_mvd_std_pipeline.py` | SDXL、双 vision encoder、Reference Attention |
| 六视图转 mesh | `infer/views_to_mesh.py::Views2Mesh` | 重排视图并调用 SVRM |
| Sparse-view 重建 | `svrm/predictor.py`、`svrm/ldm/models/svrm.py` | triplane、SDF、mesh、UV texture |
| 三平面生成 | `svrm/ldm/modules/translator/img_to_triplane.py` | $[B,3,120,256,256]$ |
| SDF/RGB decoder | `svrm/ldm/modules/rendering_neus/synthesizer.py` | 每查询点 1 SDF + 3 RGB |
| Baking | `third_party/mesh_baker.py`、`third_party/gen_baking.py` | DUSt3R 对齐与纹理回投 |

### 11.2 全链路张量总表

| # | 模块 | 输入 | 输出 | 关键配置 |
|---:|---|---|---|---|
| 1 | RemoveBG | PIL RGB/RGBA | `[512,512,4]` | 主体框外扩 1.15 倍 |
| 2 | Lite MVD VAE latent | `[1,3,640,640]` condition / 目标 grid | condition `[1,4,80,80]`；目标 `[1,4,120,80]` | SD2.1，VAE 8× 下采样 |
| 3 | Std MVD VAE latent | `[1,3,512,512]` condition / 目标 grid | condition `[1,4,64,64]`；目标 `[1,4,192,128]` | SDXL，VAE 8× 下采样 |
| 4 | 六视图拆分 | Lite/Std grid | 6×`[320,320,3]` 或 6×`[512,512,3]` | Std 重排 `[0,2,4,5,3,1]` |
| 5 | SVRM 输入 | 6 views + condition | image `[1,7,3,504,504]`；camera `[1,7,20]` | 条件图 camera 全零 |
| 6 | DINOv2-B/14 | `[7,3,504,504]` | `[7,1297,768]` → `[1,9079,768]` | 36×36 patch + CLS |
| 7 | Triplane query Transformer | context `[1,9079,768]`；query `[1,12288,1024]` | `[1,12288,1024]` | 16 层，16 heads×64 |
| 8 | Linear unpatchify | 3×`[64,64,1024]` | `[1,3,120,256,256]` | 每 token → 4×4 token |
| 9 | SDF/RGB decoder | point `[1,P,3]` | SDF `[1,P,1]`，RGB `[1,P,3]` | 三平面采样后 360→64→4 |
| 10 | Mesh extraction | 规则点 $384^3=56{,}623{,}104$ | 顶点/三角面 mesh | AABB $[-0.6,0.6]^3$，iso=0 |
| 11 | UV texture | mesh + triplane | OBJ/MTL/PNG/GLB | xatlas，texture $1024^2$ |

### 11.3 逐模块拆解

#### 11.3.1 文本、去背景与条件图标准化

**输入输出与内部张量流：** 文本模式先由 Hunyuan-DiT 以 25 步生成 $1024^2$ 图像；RemoveBG 使用 U²-Net/rembg 获取 alpha，按 alpha>0.1 计算主体框，外扩 1.15 倍后缩放到 $512^2$ RGBA。

**为什么这样设计：** 多视图生成训练在中心化对象上，若直接接受复杂背景，模型会把地面、阴影或背景边界当成对象几何。统一尺度也能减少相机距离与主体大小的耦合。

**关键超参：** `t2i_steps=25`、`t2i_seed=0`、alpha 背景阈值 16/255、主体框阈值 0.1、padding ratio 1.15。

**代码与论文差异：** 论文只说使用 Hunyuan-DiT，没有披露 prompt 后缀、negative prompt 和 PAG；这些属于开源推理实现，不应反写成论文训练方法。

#### 11.3.2 Lite / Standard 多视图扩散

**输入输出与内部张量流：** Lite 将条件图处理到 $640^2$，VAE latent 为 `[1,4,80,80]`，目标 grid latent 为 `[1,4,120,80]`，解码到 $960\times640$；Standard 条件 latent 为 `[1,4,64,64]`，目标 grid latent 为 `[1,4,192,128]`，解码到 $1536\times1024$。两者都先跑带同级噪声的 reference latent，缓存 self-attention hidden states，再拼入目标 grid 分支。

**为什么这样设计：** grid 联合生成让视图间共享去噪上下文；Reference Attention 保持输入对象身份与纹理；Standard 用更大的 SDXL cross-attention 维度 2048，提高容量，Lite 用 SD2.1 的 1024 维获得更低显存。

**关键超参：** 主入口统一传 `gen_steps=50`。Lite UNet 通道 `[320,640,1280,1280]`；Standard `[320,640,1280]`，transformer layers/block 为 `[1,2,10]`。Lite 使用 Euler Ancestral scheduler 与显式视角/时间 adaptive CFG。

**代码与论文差异：** Lite pipeline 内部默认 75 步，但主入口覆盖为 50；Standard 默认恒定 CFG=2.0，而论文把 adaptive CFG 作为一般方法。`Image2Views` 还把 generator 参数误写成 `generat`，显式 generator 没有真正传入 pipeline，主要依赖全局 seed。

#### 11.3.3 Camera-modulated DINOv2 与 Hybrid Inputs

**输入输出与内部张量流：** 六张固定视图加原图共 7 张，统一 resize 到 $504^2$；每个相机由 $4\times4$ c2w 展平 16 维，加 $(f_x,f_y,c_x,c_y)$ 4 维组成 20 维。原图 20 维全零。DINOv2-B/14 输出每图 1297 个 768 维 token，最终合并为 9079 token。

**为什么这样设计：** 504 可被 patch size 14 整除，得到 $36\times36$ patch；全零 camera 不是伪造原图位姿，而是让网络学习“这是无标定补充证据”。

**关键超参：** DINOv2 ViT-B/14，12 blocks、12 heads、hidden 768、MLP ratio 4；camera MLP 为 $20\rightarrow768\rightarrow768$，用于每层 AdaNorm。

**代码与论文差异：** 论文只写 camera embedding，没有 Plücker ray；主 mesh 链路也没有把相机先转成 ray。源码注释里仍有过期的 4 views / 512 size 描述，实际 assert 和 transform 是 7 views / 504 size。

#### 11.3.4 Triplane Transformer 与 linear unpatchify

**输入输出与内部张量流：** 12288 个 1024 维 query 经过 16 层 cross-attention+self-attention，与 `[1,9079,768]` 图像 context 融合；再 reshape 为三张 $64\times64$ 平面，每 token 线性映射到 1920 维并重排到 `[1,3,120,256,256]`。

**为什么这样设计：** 低分辨率 attention 保留全局建模能力，高分辨率只做局部通道重排，显著降低显存和计算量，同时提高 SDF 查询的空间带宽。

**关键超参：** depth=16，heads=16，head dim=64，query dim=1024，context dim=768，FFN dim=4096，dropout=0，triplane channels=120。

**代码与论文差异：** `svrm.yaml` 有 `upsample_time: 2`，但当前 linear path 不读取它，实际倍率硬编码为 4；位置 embedding 以全零初始化，并留有 TODO，依赖 checkpoint 学习。

#### 11.3.5 SDF、Marching Cubes 与纹理

**输入输出与内部张量流：** 每个 3D 点从三个平面各采 120 维，拼成 360 维，经 MLP 输出 SDF+RGB。主链路按 chunk 查询 56,623,104 个规则网格点，得到 `[1,384,384,384,1]` SDF，Marching Cubes 提取零等值面；再以 0.007 voxel clustering、2 次 simple smooth 和 quadric decimation 清理 mesh。

**为什么这样设计：** SDF 适合稳定提取封闭表面；显式 mesh 比 NeRF/3DGS 更易被 DCC 工具接收。把颜色也放在同一 triplane decoder 中，使顶点颜色与 UV texture 都能复用同一外观场。

**关键超参：** AABB $[-0.6,0.6]^3$、grid=384、query chunk=$2^{20}$、iso-level=0、texture=$1024^2$、hole padding radius=2。

**代码与论文差异：** 代码保留 ray sampler 与 NeuS ray marcher，但 `export_mesh_with_uv()` 的生产链路直接查规则网格，不经过相机 ray；代码注释中的“nerf render”不代表最终表示是 NeRF。`OSGDecoder` 定义了 sphere SDF bias，但当前 forward 中相关调用被注释，不生效。

### 11.4 推理计算账

| 阶段 | 主要序列/空间规模 | 主干 | 前向次数/显存压力 |
|---|---|---|---|
| T2I | $1024^2$ 图像扩散 | Hunyuan-DiT | 默认 25 步；约 14 GB 参考显存 |
| Lite MVD | latent `[1,4,120,80]` | SD2.1 UNet | 主入口 50 步；阶段参考约 8 GB |
| Standard MVD | latent `[1,4,192,128]` | SDXL UNet | 50 步；阶段参考约 16 GB |
| DINO | 7×1297 token | ViT-B/14×12 blocks | 一次编码，camera AdaNorm |
| Triplane | 12288 query × 1024 | 16 层 Transformer | query self-attention 是重建主计算瓶颈 |
| SDF extraction | $384^3$ 点 | 360→64→4 MLP | 约 54 个 $2^{20}$ chunk |
| Texture/Baking | $1024^2$ UV | xatlas+nvdiffrast；可选 DUSt3R | 额外约 15 秒；Baking 还需非商业 DUSt3R 依赖 |

### 11.5 代码中的工程陷阱

1. `main.py` 默认 `max_faces_num=120000`，README 表格写 90000，`Views2Mesh.call()` 默认又是 10000；应以实际调用命令为准。
2. `MV23DPredictor` 虽接收 `ckpt_path`，加载时却硬编码 `./weights/svrm/svrm.safetensors`，命令行重定向 checkpoint 可能无效。
3. Baking 的可选导入逻辑不完全健壮；未安装 DUSt3R 时可能在进入容错分支前失败。
4. `main.py --do_render` 使用 `glob` 但文件中未显式导入，并且 GIF renderer 的初始化与 Baking 可用性耦合。
5. Standard 与 Lite 的 grid 槽位顺序不同；若绕过 `Views2Mesh.order`，会把相机和图像错误配对。

---

## 12. 材料来源

- 论文：https://arxiv.org/abs/2411.02293
- 官方 PDF：https://arxiv.org/pdf/2411.02293
- 官方代码：https://github.com/Tencent-Hunyuan/Hunyuan3D-1
- 官方权重与配置：https://huggingface.co/tencent/Hunyuan3D-1
- 本地来源清单：[assets/Hunyuan3D 1.0 - 统一两阶段三维生成/sources.md](assets/Hunyuan3D%201.0%20-%20统一两阶段三维生成/sources.md)
- 相关笔记：[[TRELLIS - 结构化潜变量三维生成]]、[[3DShape2VecSet - 神经场与扩散生成的三维形状表征]]、[[QuadGPT - 原生四边形网格自回归生成]]
