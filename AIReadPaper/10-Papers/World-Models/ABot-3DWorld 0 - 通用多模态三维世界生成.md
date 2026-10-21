---
type: paper-note
status: complete
domain: World-Models
paper: "ABot-3DWorld 0: A Universal World Model to Explore Any 3D Space"
year: 2026
arxiv: 2607.11673
doi: null
source: https://arxiv.org/abs/2607.11673
project: https://abot-world.amap.com/plaza
code: null
tags:
  - World-Model
  - 3D-Generation
  - 3DGS
  - Panorama-Video
  - Spatial-Generative-Primitive
  - Multimodal-Generation
created: 2026-10-16
updated: 2026-10-16
---

# ABot-3DWorld 0：通用多模态三维世界生成

## 基本信息

- **正式标题**：ABot-3DWorld 0: A Universal World Model to Explore Any 3D Space
- **作者**：Mingchao Sun、Luyang Tang、Yu Liu、Xu Yan、Zhan Li 等，AMAP CV Lab
- **来源**：arXiv:2607.11673，2026 年 7 月公开预印本；截至本笔记写作时未确认同行评审 venue
- **输入**：文本、单图、多视图照片或随手拍视频
- **输出**：可自由视点浏览的照片级 3D Gaussian Splatting（3DGS）世界
- **核心中间表示**：Spatial Generative Primitive（SGP）=`360° 全景图 + 空间点云`
- **关键底座**：ABot-Pano、MoGe-2 类单目几何、VACE 14B 全景视频模型、3DRL、ABot-3DGS 与 FLUX repair-and-refit

## 一句话结论

ABot-3DWorld 0 的关键不是把“文本转 3D、单图转 3D、视频重建”简单并列，而是用 SGP 把不同输入先统一到**外观全景 + 度量点云**，再以全景视频高覆盖探索空间，最后用逐场景 3DGS 优化和两轮 FLUX 修复收敛为可渲染资产；论文对 3DRL、修复器和端到端感知质量给出部分定量证据，但 SGP、轨迹规划、scene memory、全景接缝等关键模块缺少独立消融，“通用”和“优于 Marble”仍有相当部分依赖作者定性展示。

---

## 1. 问题定义：统一生成与重建

### 1.1 为什么不能直接从所有输入回归 3DGS

文本和单图只提供稀疏证据，需要生成未观测区域；多视图与视频则包含真实几何和外观，首要目标是忠实恢复。如果统一模型忽略这种差异，富输入容易被“重新想象”，而极简输入又会因为几何欠约束而产生空洞。论文的解决办法不是让所有输入共享同一个前端，而是让它们通过不同 lifting 路径汇合到同一 SGP，再共享后续探索与重建引擎。

![](assets/ABot-3DWorld%200%20-%20通用多模态三维世界生成/figures/01-teaser.png)

> **图 1：统一系统覆盖的输入与场景尺度。** 文本、单图、多视图和视频最终都输出室内、街景或航拍 3DGS；该图证明系统定位和样例覆盖，不等于几何精度的受控比较。图片已从 arXiv 源码提取并视觉核验。  
> 来源：论文 Figure 1，<https://arxiv.org/abs/2607.11673>

### 1.2 两种工作模式

- **生成式补全**：文本或单图先生成完整 ERP 全景，再预测点云，允许模型创造未观测空间；
- **几何恢复**：多视图或视频先做相机注册和稀疏到稠密 3DGS 重建，在代表性 POI 渲染多个全景与点图，再修补空洞，尽量保留真实证据。

因此“一个统一模型”更准确地说是**统一后半段世界生成管线**，不是所有输入共享完全相同的前端算法。

---

## 2. SGP：空间生成基元

### 2.1 定义与角色分工

单个 SGP 定义为：

$$\mathcal S\triangleq(\mathbf P,\mathbf G),$$

其中 $oldsymbol P$ 是完整 $360^\circ$ ERP 全景图，承担高质量外观；$oldsymbol G$ 是空间点云，承担稠密、度量化几何。每个 SGP 另有关联相机位姿 $oldsymbol T$，用于把多个基元注册到共享坐标系，但位姿和探索轨迹都不属于基元本体。大场景写成：

$$\mathcal W=\{\mathcal S_1,\ldots,\mathcal S_K\}.$$

这种分解的直觉是：单张全景拥有完整角域和高外观密度，点云提供全景生成模型缺失的空间约束。它比直接保存密集 mesh 更紧凑，也比纯图像条件更适合规划轨迹。

### 2.2 极简输入到 SGP

ABot-Pano 是基于 DiT 的多模态 ERP 生成器。透视输入先投影到 ERP，经 VAE 编码后由 Control 分支逐层注入；训练时独立 dropout 文本与图像条件，使同一模型支持 text-to-pano、image-to-pano 与 text+image-to-pano。随后把 ERP 转换为 cubemap，使用 MoGe-2 类单目几何估计各面深度并重投影为全景点图。

全景训练还加入 circular padding、Distortion-aware Cube Loss 与 Rotation-consistent Yaw Loss，分别处理左右接缝、极区畸变和水平旋转一致性。论文定性显示其条件忠实度优于 HY-Pano 2.0，但也承认极区纹理略弱、开放场景多样性受到强条件约束。

### 2.3 富输入到多个 SGP

多视图或视频通过关键帧采样、相机注册和 ABot-3DGS 稀疏到稠密重建恢复同一度量空间；系统在多个代表性 POI 渲染 $oldsymbol P$ 与 $oldsymbol G$，并对未覆盖区域引导修补。该路径保留真实图像证据，不把输入先摘要成文本再重新生成，因此理论上更少幻觉；但论文没有给出专门的几何误差表来量化“更忠实”多少。

![](assets/ABot-3DWorld%200%20-%20通用多模态三维世界生成/figures/02-sgp-pipeline.png)

> **图 2：多模态输入—SGP—全景视频—3DGS 的完整流程。** 极简输入经 PanoGen + MonoGeom 得到单个 SGP，富输入经重建与修复形成多个 SGP；规划器沿蓝/绿轨迹探索后再重建最终世界。该图清楚区分“生成式补全”和“几何恢复”两条前端。  
> 来源：论文 Figure 2，§1/§3.1，<https://arxiv.org/abs/2607.11673>

---

## 3. 数据引擎

### 3.1 为什么先重建再渲染训练视频

真实全景视频不可避免包含拍摄者、车辆或设备，轨迹难以精确控制，也缺少与 RGB 对齐的 3D 真值。论文采用 `Capture → Reconstruct → Render`：先把真实采集、多视图、航拍和许可互联网内容重建成 3DGS 资产，再使用虚拟全景相机渲染严格对齐的训练样本。

每个 81 帧样本同时包含目标 RGB、轨迹、点云 RGB、深度、可见性 mask 与最多四张带位姿参考全景。控制流共享同一投影与遮挡计算，减少不同监督之间的几何矛盾。数据经有效深度、碰撞轨迹、VLM 感知质量和类别平衡过滤。

### 3.2 数据开放性边界

资产库同时含私有地面、室内、城市与航拍 3DGS，部分自举风格化数据也来自系统自身生成。论文没有公开资产总量、具体许可证构成和完整训练成本，因此外部复现最现实的目标是复现模块思想，而不是同规模数据分布。

---

## 4. 几何约束的轨迹规划

### 4.1 从全景几何到可导航图

系统先从全景预测稠密距离与法线，把 ERP 网格反投影为点云/三角网格，再用 RecastNavigation 构建 NavMesh。语义目标通过 `Qwen3-VL-8B → Grounding DINO → SAM3 → 深度反投影` 注册到 3D 导航图。

### 4.2 三类互补轨迹

1. **Spatial Coverage**：九条轨迹覆盖可导航区域并增加视差；
2. **Autonomous Roaming**：五条轨迹处理狭窄和杂乱空间；
3. **Semantic Engagement**：五条轨迹主动接近书架、展柜等高信息目标。

候选路径还要经过 KD-tree 碰撞检测、完整 pose 序列余弦相似度去重、三次 B 样条平滑，并重采样为 21 个等距关键帧。这里体现一个重要设计：生成视点不是任意相机路径，而是为后续 3DGS 可重建性服务。

---

## 5. 全景视频作为空间探索载体

### 5.1 为什么选择全景视频

窄 FOV 多视图需要大量相机才能覆盖完整环境，并容易在视锥之间留下盲区。ERP 全景在一个相机中心观察完整 $360^\circ$，提高每帧覆盖率与跨视图重叠。系统给定 SGP 点云 $oldsymbol G$ 与轨迹 $oldsymbol\tau$，生成：

$$\hat V\in\mathbb R^{T\times H\times W\times3}.$$

代价是 ERP 极区畸变、左右拓扑接缝和局部有效分辨率下降；后续 3DRL 与 FLUX 修复正是为这些代价补偿。

### 5.2 轨迹对齐控制

14B VACE 控制视频模型接收四类逐帧条件：

- 点云 RGB 渲染：锚定已观测外观；
- 深度视频：提供布局、遮挡与时间几何；
- 点云可见性 mask：区分有证据和需补全区域，但不作为硬编辑 mask；
- Plücker ray map：把每个 ERP 像素绑定到世界空间射线：

$$p_t(u,v)=\left[d_{w,t}(u,v),\ o_{w,t}\times d_{w,t}(u,v)\right]\in\mathbb R^6.$$

相较单一外参 token，ray map 显式表达每个像素从哪里、朝什么方向观察。

### 5.3 Scene memory 与全景接缝

系统保存此前生成的带位姿全景，并按相机中心距离检索最多四张：

$$\mathcal M=\{(I_j^{\mathrm{ref}},T_j^{\mathrm{ref}})\}_{j=1}^{N_{\mathrm{ref}}},\qquad N_{\mathrm{ref}}\le4.$$

记忆图作为外观和布局锚点。ERP 左右边界连续性则在三处处理：VAE 编码前 circular padding；每次去噪把 latent 与条件共同水平滚动；解码前再做 latent circular padding。LightX2V 一致性蒸馏把采样从 50 步压缩到 4 步；在 4×RTX 4090 上，全景视频生成约 12 分钟/场景。

![](assets/ABot-3DWorld%200%20-%20通用多模态三维世界生成/figures/03-panoramic-video.png)

> **图 6：几何与记忆共同控制的全景视频生成。** Plücker、mask、depth、RGB 与邻近 reference panorama 进入控制分支，主视频 DiT 仍保持生成先验。图像文字、边缘和输出帧已视觉核验完整。  
> 来源：论文 Figure 6，§3.4，<https://arxiv.org/abs/2607.11673>

---

## 6. 3DRL：优化可重建性而非只看单帧

### 6.1 光流循环一致性奖励

普通 flow matching 优化每帧真实感，不直接保证视点变化符合稳定 3D。论文把 ERP 转为沿赤道采样的 $K$ 个透视视频，在帧对上用 RAFT 估计前向/后向光流。像素循环误差为：

$$e_i^k(x)=\left\|F_{t_i\to t_i+s}^k(x)+\mathcal W(F_{t_i+s\to t_i}^k,F_{t_i\to t_i+s}^k)(x)\right\|_2.$$

对视图、帧对和有效像素求平均后：

$$r_{\mathrm{cycle}}=\operatorname{clip}\left(1-\frac{\bar e}{\tau_{\mathrm{cycle}}},0,1\right).$$

### 6.2 防止静止视频奖励作弊

近静止视频也可能有很低循环误差，因此加入与冻结 RL 前模型输出的 LPIPS fidelity reward：

$$r_{\mathrm{fid}}=1-\operatorname{LPIPS}(\hat V,\hat V_{\mathrm{ref}}).$$

这项设计很重要：3DRL 优化的是光流代理，而非真实几何；fidelity reward 只能约束分布漂移，仍不能把代理奖励等同于严格 3D 正确性。

### 6.3 受控消融

在 100 个未见场景上，w/ 3DRL 相比 w/o 3DRL：PSNR 34.11→35.30，SSIM 0.942→0.951，LPIPS 0.089→0.082，Confidence 0.396→0.412；审美、成像、运动平滑、主体与背景一致性也小幅上升。这是论文最完整的模块级消融，支持 3DRL 提升视频可重建性与感知质量。

---

## 7. 全景视频到 3DGS

### 7.1 粗重建

系统使用 SfM 恢复相机轨迹和稀疏点云，并把已规划轨迹作为强先验，然后逐场景优化 vanilla 3DGS。作者选择优化式重建而非一次前馈回归，是为了更贴合生成视频并兼容大型真实采集，但代价是每场景仍需优化。

### 7.2 两轮 FLUX repair-and-refit

全景视频的局部透视裁剪分辨率低，不同探索轨迹还会产生光度和几何漂移。修复器把当前 3DGS 渲染视图与原始全景的高质量 reference crop 经共享 VAE/DiT 联合处理，从低质量 latent 做单次确定性修复，而不是从噪声重新生成。修复图再监督 3DGS 重拟合，共执行两轮。

![](assets/ABot-3DWorld%200%20-%20通用多模态三维世界生成/figures/04-repair-refit.png)

> **图 13：两轮 FLUX 修复的定性变化。** 第一轮主要消除模糊和严重不一致，第二轮继续强化纹理和稳定细节。该图是定性组件验证，没有给出每一轮独立数值。图像三列与底部标签已视觉核验完整。  
> 来源：论文 Figure 13，§4.4，<https://arxiv.org/abs/2607.11673>

### 7.3 野外场景鲁棒性

ABot-3DGS 还包含天空球壳/隐式天空、GLO + bilateral grid 外观校准、MCMC-style densification 和动态物体 transient suppression。它们分别处理天空深度不确定、曝光漂移、反射/弱纹理和静态场景假设被破坏的问题。

4×RTX 4090 上，FLUX repair 每场景约 3 分钟、每轮 3DGS 约 1 分钟；三轮 3DGS + 两轮 FLUX 的重建阶段总计约 10 分钟（含额外开销）。这不包含前述全景视频生成约 12 分钟，应分别报告。

---

## 8. 实验结果与证据边界

### 8.1 3DRL 与修复器

| 实验 | 关键结果 | 能支持的结论 |
|---|---|---|
| 3DRL 消融 | PSNR +1.19 dB；SSIM +0.009；LPIPS -0.007 | 光流循环奖励在给定协议下改善可重建性 |
| FLUX vs Difix3D+ | 21.6118/0.7926/0.2314 vs 20.2968/0.7184/0.2667 | 欠约束侧视角修复优于该基线 |
| 两轮修复 | Figure 13 定性逐轮变清晰 | 不能量化第二轮独立收益 |

### 8.2 与 HY-World 2.0

在 All split 上，ABot 的 Q-Align 4.0426 低于 HY-World 2.0 的 4.0755，但 CLIP-IQA+ 0.5469、Laion-Aes 5.4903、CLIP-I 0.9174 分别高于 0.5348、5.3137、0.9105。更准确的结论是：**审美和输入内容忠实度占优，低层感知质量接近，Q-Align 并非领先。**

### 8.3 Marble 比较

论文声称富输入下场景保真度明显强于 Marble，但证据主要来自作者定性图，没有同设置几何误差或完整数值表。因此应写成“论文定性展示更忠实”，而不是“统一 benchmark 已证明全面领先”。

### 8.4 缺失的消融

论文未单独量化 circular padding、ray map、visibility mask、scene memory、三类轨迹、LightX2V 蒸馏、天空/GLO/bilateral grid/transient suppression 的贡献。整个系统组件很多，现有实验不能建立所有模块的因果必要性。

---

## 9. 局限、未来工作与个人判断

### 9.1 已确认的局限

ERP 极区与局部透视有效分辨率有限；独立轨迹仍会发生光度/几何漂移；视频 DiT 并非原生 3D 模型；逐场景修复与 3DGS 优化增加时延；天空、反射、透明和动态物体依旧困难；富输入的几何忠实度缺少标准几何指标；多个核心组件未做独立消融。

### 9.2 作者明确的未来方向

结论明确提出三项：把多 SGP 组合扩展到城市级并维持跨基元几何一致；更深地接入行星级 3D 地图服务，实现 globe-to-room 连续探索；把当前碰撞网格和 portal 扩展到 relighting、动态元素与 Agent 交互。

### 9.3 个人判断

SGP 的真正价值是把“场景是什么样”与“场景在哪里”拆成全景外观和点云几何，使生成前端与重建前端能共享后半段。系统的代价也同样清晰：全景视频并不是最终 3D，必须经过 3DRL、修复器和优化式 3DGS 才能得到稳定资产。它更像一条工程化世界生产线，而非单个端到端网络。未来最关键的验证不是再加更多漂亮样例，而是公开同协议几何、导航和跨 SGP 接缝 benchmark。

---

## 10. 材料来源

本地官方 PDF、四张源码原图及来源清单见：

- `assets/ABot-3DWorld 0 - 通用多模态三维世界生成/sources/paper.pdf`
- `assets/ABot-3DWorld 0 - 通用多模态三维世界生成/sources.md`
