---
type: paper-note
status: complete
domain: World-Models
paper: "DecMem: Towards Minute-Long Consistent World Generation with Decoupled Memory"
year: 2026
arxiv: "2605.31336"
source: https://arxiv.org/abs/2605.31336
project: https://jeffreyyzh.github.io/DecMem-Page
code: null
tags:
  - World-Models
  - Long-Horizon-Generation
  - Memory
  - Sparse-Attention
  - Kling
created: 2026-10-16
updated: 2026-10-16
---

# DecMem：解耦记忆的分钟级一致世界生成

## 基本信息

- **正式题名**：DecMem: Towards Minute-Long Consistent World Generation with Decoupled Memory
- **作者**：Zhenhao Yang、Xiaoshi Wu、Zhengyao Lv、Xiaoyu Shi、Xintao Wang、Pengfei Wan、Kun Gai、Kwan-Yee K. Wong
- **机构**：香港大学、快手 Kling 团队
- **版本**：arXiv v1，2026-05-29
- **核心关键词**：attention dispersion、Sparse Global Memory（SGM）、Anchored Local Memory（ALM）、block retrieval、minute-long rollout

## 一句话结论

DecMem 认为长程世界生成失败不只是“历史太长、算不动”，更是大量弱相关 token 稀释关键记忆的**注意力分散**；它用可学习的块级 Sparse Global Memory 检索远程细节，再用最近 8 个 latent 的 Anchored Local Memory 稳住局部外推，并通过门控把长期记忆作为局部锚点上的增量，最终在 3.6496 FPS 下实现约一分钟的可控一致生成。

---

## 1. 研究背景与问题定义

### 1.1 长视频的“重访”测试

世界模型不仅要生成下一帧，还要在离开一个区域后再次返回时恢复相同布局、物体和纹理。滑窗方法一旦把旧帧移出窗口就失去该区域；FOV/位姿检索能取回历史帧，却是帧级启发式选择；在线 3D 重建提供几何先验，但引入重建误差、动态物体脆弱性与额外成本。最直接的全历史 dense attention 虽然细粒度、可学习，却同时遭遇计算量增长和注意力分散。

![](assets/DecMem%20-%20解耦记忆的分钟级一致世界生成/figures/01-motivation.png)

> 图 1：上半部分显示 Dense Attention 在 810 帧附近发生结构崩溃，而 DecMem 保持目标；右侧显示其单 latent 耗时近似稳定。下半部分对比可学习 block retrieval 与 WorldMem 的 FOV 帧检索，前者能从不同帧局部取回细粒度对应。来源：论文 Figure 1。

### 1.2 Attention dispersion

随着历史增长，dense attention 给越来越多弱相关 token 分配“小但非零”的长尾权重。附录用两个阈值统计：权重小于 0.02 视为可忽略，大于 0.05 视为关键。rollout 继续时，可忽略权重总量上升、关键权重总量下降。结果不是仅仅变慢，而是高频细节消失、场景身份漂移和几何崩溃。

简单 temporal decay 也不能解决：它无差别衰减远程 token，既去掉冗余，也可能切断真正有用的远程记忆。DecMem 因而将“全局检索”和“局部稳定”拆开建模。

### 1.3 贡献

1. 把长上下文失败明确归因于计算增长与注意力分散两种机制。
2. SGM 先按块检索、再在候选块内部做 token-level attention，实现细粒度且有界的全局访问。
3. ALM 以最近局部窗口提供高置信锚点，防止远程长尾噪声主导输出。
4. 在 WorldMem 数据上显著提高训练窗内与外推指标，并展示一分钟重访一致性。

---

## 2. 任务定义与输入输出

### 2.1 输入与输出

历史视频经 VAE 编码为：

$$\mathbf z_0^{1:T}\in\mathbb R^{C\times T\times H\times W}$$

模型接收历史 latent、当前噪声 latent $	extbf z_t^{T+1}$、动作 $	extbf a$、扩散时间 $t$、相机内外参和位置编码，输出下一 latent 的 rectified-flow 速度，去噪后得到 $	extbf z_0^{T+1}$。实现采用 chunk-wise autoregression：每块包含 4 个 latent，块内可以相互注意，块间保持因果。

### 2.2 基础目标与动作注入

teacher forcing 下的目标为：

$$\mathcal L=\left\|\mathbf v_\theta(\mathbf z_0^{1:T},\mathbf z_t^{T+1},t)-\mathbf v^{T+1}\right\|_2^2$$

其中：

$$\mathbf v^{T+1}=\epsilon^{T+1}-\mathbf z_0^{T+1},\quad\epsilon^{T+1}\sim\mathcal N(0,I)$$

动作经轻量模块 $\psi$ 映射后与 patch token 相加：

$$\mathbf x=\operatorname{Patchify}(\mathbf z)\oplus\psi(\mathbf a)$$

---

## 3. 核心方法：Sparse Global + Anchored Local

![](assets/DecMem%20-%20解耦记忆的分钟级一致世界生成/figures/02-method-pipeline.png)

> 图 4：SGM 从全部历史中检索 top-$k$ 块，再在候选块 token 上做 context-aware attention；ALM 只关注最近窗口；最终把全局输出作为受门控增量加入局部输出。图中为简化示意，实际每帧 6 块、$k=80$、ALM 窗口为 8。来源：论文 Figure 4。

### 3.1 Sparse Global Memory（SGM）

每个 frame latent 划分为 6 个不重叠块。当前 query block 内先做 pooling，再与全部历史块的 pooled features 计算相关性，为每个 query 选 top-$k$ 历史块。可概括为：

$$\mathcal C_{t,i}=\operatorname{TopK}\left(\operatorname{AttnScore}(\bar q_{t,i},\{\bar k_{\tau,j}\}_{\tau<t,j}),k\right)$$

这一步是块级粗筛；随后 query block 内每个 token 与候选块的全部 token 做细粒度 dense attention，得到 $o_t^{\mathrm{sgm}}$。因此检索成本受固定 $k$ 约束，同时保留“同一历史帧只有局部区域相关”的表达能力。默认 $k=80$。

### 3.2 Anchored Local Memory（ALM）

ALM 对最近 $w=8$ 个 frame latents 做 sliding-window attention，输出 $o_t^{\mathrm{alm}}$。它不是低配 SGM，而是注意力稳定器：相邻帧通常最相关，可以提供平滑过渡和可靠细节基线，避免全局候选中的长尾弱相关项直接污染生成。

### 3.3 门控融合

最终输出为：

$$o_t=o_t^{\mathrm{alm}}+G_t\odot o_t^{\mathrm{sgm}}$$

$G_t$ 由当前帧特征产生。这个不对称设计很关键：局部记忆是基础路径，远程记忆是受控增量。仅靠 SGM，在约 600 帧后仍会因为注意力分散而崩溃；仅靠 ALM，则一出窗口就失去场景身份。

### 3.4 多模态位置编码

每个 72 维 attention head 被均分为相机、空间、时间三组，各 24 维：

$$\mathbf R_{\mathrm{full}}^{(i)}=\operatorname{diag}(\mathbf R_{\mathrm{cam}},\mathbf R_{\mathrm{sp}},\mathbf R_{\mathrm{tem}})$$

相机内参 $K_t$ 与 world-to-camera 外参 $T_t^{cw}$ 组成投影矩阵：

$$P_t=[K_t\ \mathbf 0]T_t^{cw}$$

扩展为可逆齐次矩阵 $\tilde P_t$ 后，通过 Kronecker product 构造 camera-channel 变换：

$$\mathbf R_{\mathrm{cam}}(P_t)=\mathbf I_{d_{\mathrm{cam}}/4}\otimes\tilde P_t$$

空间和时间分别使用 2D/1D RoPE。对 value/output 只施加 camera 与 spatial 变换，不作用于 temporal 通道。该设计让远程检索不仅依赖外观，还携带相机、空间 patch 和时间关系。

### 3.5 Action CFG

动作 guidance 为：

$$\hat{\mathbf v}_\theta=\mathbf v_\theta(\mathbf z_t,\varnothing,t)+s[\mathbf v_\theta(\mathbf z_t,a_t,t)-\mathbf v_\theta(\mathbf z_t,\varnothing,t)]$$

使用 $s=7.5$。它牺牲少量短程 PSNR，但显著改善长程 LPIPS/FID。

---

## 4. 数据与训练设置

### 4.1 数据

WorldMem 数据约 11K 视频，每段 1500 帧、360×640、10 FPS；训练随机采 237 帧并 resize 到 352×640。评估使用 300 个与训练不重叠的视频。另在 Context-as-Memory 的 island、city、chemical plant 环境中测试左右往返重访。

### 4.2 两阶段训练

- Stage 1：从 1B full-sequence latent diffusion transformer 初始化，改造成 causal generator，训练 25K steps。
- Stage 2：加入 DecMem，再联合训练 25K steps。
- 资源：64×NVIDIA H200，global batch 64，AdamW，学习率 $2\times10^{-5}$，总计约 7 天。

主评测先用 221 个 GT frames 初始化 memory bank，再生成 120 帧；训练窗内取 222–229 帧，外推取 334–341 帧。所有 diffusion 方法使用 20 个 denoising steps。

### 4.3 Baseline 与指标

MineWorld 使用 32 帧滑窗，Oasis 使用 8 帧滑窗，WorldMem 使用 8 帧局部窗口加 FOV-based frame retrieval。另与 Matrix-Game 2.0、WorldPlay 做 30 秒单图初始化用户研究。指标包括 PSNR、LPIPS、FID、视觉质量 VQ、动作控制 AC、时空一致 STC 与 FPS。

---

## 5. 实验结果

### 5.1 主结果

| 方法 | 窗内 PSNR ↑ | LPIPS ↓ | FID ↓ | 外推 PSNR ↑ | LPIPS ↓ | FID ↓ | VQ ↑ | AC ↑ | STC ↑ | FPS ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MineWorld | 20.2989 | 0.1789 | 41.9661 | 14.6109 | 0.3736 | 74.2075 | 15.06% | 14.49% | 14.29% | 0.1588 |
| Oasis | 24.1293 | 0.1196 | 15.9163 | 13.4232 | 0.4475 | 63.8851 | 25.87% | 22.26% | 19.41% | 1.9806 |
| WorldMem | 26.5414 | 0.0797 | 11.7379 | 19.1401 | 0.2673 | 38.4689 | 19.31% | 25.33% | 24.16% | 0.5424 |
| **DecMem** | **30.0785** | **0.0494** | **9.8904** | **25.2294** | **0.1006** | **16.2667** | **39.77%** | **37.81%** | **42.12%** | **3.6496** |

DecMem 对 WorldMem 的外推改善非常大：PSNR 19.1401→25.2294、LPIPS 0.2673→0.1006、FID 38.4689→16.2667。它也比最快 baseline Oasis 快约 1.84×，但 3.6496 FPS 仍低于数据的 10 FPS 实时速度。

### 5.2 分钟级结果

![](assets/DecMem%20-%20解耦记忆的分钟级一致世界生成/figures/03-minute-long-rollout.png)

> 图 6：0–60 秒的往返探索中，DecMem 在多次离开和重访后维持场景结构。红框标记预测帧，其余为记忆帧。来源：论文 Figure 6。

### 5.3 Top-$k$ 消融

| $k$ | 中程 PSNR ↑ | LPIPS ↓ | FID ↓ | 长程 PSNR ↑ | LPIPS ↓ | FID ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 24.3749 | 0.1105 | 16.9071 | 19.6497 | 0.2217 | 27.7425 |
| 50 | 25.0605 | 0.1023 | 16.2642 | 19.9425 | 0.2100 | 27.0902 |
| **80** | 25.2294 | 0.1006 | 16.2667 | **20.5896** | 0.2019 | **25.2748** |
| 100 | **25.4616** | **0.0962** | **15.5984** | 20.5535 | **0.1994** | 25.8790 |

更多候选并非总是更好。$k=100$ 的中程指标较优，但长程 PSNR/FID 比 $k=80$ 变差，支持“候选过多会再次稀释上下文”的解释。

### 5.4 SGM/ALM 组件消融

![](assets/DecMem%20-%20解耦记忆的分钟级一致世界生成/figures/04-module-ablation.png)

> 图 7：Dense 的每块耗时随历史近线性上升；w/o SGM 的 LPIPS 从早期就迅速恶化；w/o ALM 在后期 FID 爆升；完整模型保持近似恒定耗时并取得最低后期误差。来源：论文 Figure 7。

- **w/o SGM**：退化为局部模型，越出 8-latent 窗口后迅速丢失历史场景，整体质量最差。
- **w/o ALM**：早期尚可，约 600 帧后高频细节和几何崩溃，甚至差于 vanilla Dense。
- **Dense + decay**：后期略有缓解，但 300–700 帧可能比 Dense 更差，因为远程有用信息也被衰减。
- **完整模型**：早期接近 Dense，后期持续领先，说明 SGM 与 ALM 不是可替代的两个版本，而是分别处理记忆召回和注意力稳定。

### 5.5 CFG 消融

关闭 CFG 时长程 PSNR 为 21.4909，高于开启后的 20.5896；但 LPIPS 从 0.2019 变为 0.2380，FID 从 25.2748 恶化到 42.5515。CFG 以像素对齐为代价，换取更稳定的长程感知分布。

---

## 6. 与 Astra、GIM-World、AnchorWorld 的关系

Astra 通过给历史加噪处理“历史压制动作”，DecMem 则假定历史必须长期保存，转而解决“全历史如何高效且不分散地被读取”。GIM-World 更进一步把历史压成固定大小 memory token，并用 VGGT 几何教师决定记忆应编码什么；DecMem 不依赖外部几何教师，保留细粒度历史块，因此更接近可学习检索系统。AnchorWorld 的 anchor views 是用户指定的局部世界状态，而 DecMem 的 block memory 是模型从生成历史中自动积累的经验；两者未来可组合成“外部锚点 + 内部长期记忆”。

---

## 7. 局限与批判性评价

### 7.1 作者明确局限

作者明确表示尚未研究蒸馏加速，3.6496 FPS 未达到实时。未来计划结合 compressed global memory 与 fine-grained object-level memory，构建实时混合记忆模型。

### 7.2 证据边界

1. 主量化集中在 Minecraft/WorldMem，开放真实世界、动态物体和遮挡下的结论尚不充分。
2. 训练成本高：64×H200、约 7 天。
3. 依赖相机参数与专门位置编码，未报告相机噪声鲁棒性。
4. 分钟级展示很强，但超过一分钟的聚合指标和 full DecMem failure cases 未系统报告。
5. SGM 检索错误是否会通过门控污染生成，论文没有专门诊断。

### 7.3 个人判断

DecMem 的核心贡献不是“又一个 sparse attention”，而是把局部记忆设为不可缺少的**统计锚点**。SGM 提供召回能力，ALM 控制注意力分布；门控结构承认远程记忆应当是选择性修正，而非取代局部连续性。这个设计比单纯扩大 context window 更符合长程生成的误差传播规律。

---

## 8. 复现与实践建议

1. 先在 221-frame context + 120-frame prediction 协议上复现 Dense、ALM-only、SGM-only 和完整模型，重点画出随帧位置变化的 LPIPS/FID，而非只报告终点。
2. 先固定每帧 6 blocks、$k=80$、ALM $w=8$；确认稳定后再搜索 $k$，因为更多候选可能恶化长程结果。
3. 检查 chunk 内双向、chunk 间因果的 attention mask，错误 mask 会造成训练泄漏。
4. 相机/空间/时间通道各 24 维，需严格实现 value/output 的 camera-spatial 变换；这部分比常规 RoPE 更易出错。
5. 使用 20 denoising steps 与 action CFG $s=7.5$；评测同时报告 PSNR 与 LPIPS/FID，避免被短程 PSNR 误导。
6. 代码仓库截至记录日未明确公开，完整复现需自行实现 SGM kernel、检索缓存和多模态位置编码；计算资源是主要门槛。

---

## 9. 个人启发与后续问题

- SGM 能否从固定 top-$k$ 改为基于不确定性的动态预算？
- ALM 是否可采用对象级局部锚点，而非固定最近 8 个 latent？
- 与 GIM-World 组合时，几何隐式 token 应作为 SGM 候选、ALM 锚点，还是第三条分支？
- 与 AnchorWorld 组合时，用户锚点是否应具有不可被生成历史覆盖的高优先级？

---

## 10. 材料来源

- 官方论文：https://arxiv.org/abs/2605.31336
- 项目页：https://jeffreyyzh.github.io/DecMem-Page
- 代码：论文未提供明确仓库链接
- 本地 PDF 与逐图溯源：`assets/DecMem - 解耦记忆的分钟级一致世界生成/sources.md`
