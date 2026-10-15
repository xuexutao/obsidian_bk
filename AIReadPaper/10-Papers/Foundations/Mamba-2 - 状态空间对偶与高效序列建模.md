---
title: "Mamba-2 - 状态空间对偶与高效序列建模"
aliases:
  - "Transformers are SSMs"
  - "Structured State Space Duality"
  - "Mamba-2"
authors:
  - Tri Dao
  - Albert Gu
year: 2024
venue: ICML 2024
arxiv: "2405.21060"
tags:
  - paper
  - SSM
  - Mamba
  - sequence-modeling
  - efficient-attention
status: 精读完成
---

# Mamba-2：状态空间对偶与高效序列建模

> **论文**：*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*  
> **作者**：Tri Dao, Albert Gu　|　**会议**：ICML 2024　|　**arXiv**：[2405.21060](https://arxiv.org/abs/2405.21060)  
> **本地材料**：[[assets/Mamba-2 - 状态空间对偶与高效序列建模/sources.md|sources.md]]；官方 PDF、TeX 源码与官方实现快照均已离线保存。

## 1. 一句话结论与阅读地图

**一句话结论**：论文把选择性状态空间模型写成半可分（semiseparable）结构矩阵，又把一类带结构掩码的注意力写成同一对象的另一种张量收缩顺序；由此得到兼具 SSM 线性递推与 Attention 矩阵乘效率的 SSD 算法，并据此构造出训练核心比 Mamba-1 selective scan 快 **2–8×** 的 Mamba-2。

必须先澄清标题里的“Transformers are SSMs”：论文证明的不是“任意标准 softmax Transformer 都等于 SSM”，而是 **SSM 与特定的 kernel/structured masked attention 之间存在大交集**。真正的桥梁是序列变换矩阵的结构，而不是把 softmax Attention 原样改写成任意 SSM。

![](assets/Mamba-2%20-%20状态空间对偶与高效序列建模/figures/fig01_duality_roadmap.png)

*图 1：论文路线图。SSM 经 semiseparable matrix 进入结构矩阵视角，Attention 经 SMA 进入相同视角；二者交集形成 SSD，再落到高效算法与 Mamba-2。*

全文可按四层理解：

1. **表示层**：SSM recurrence 等价于一个下三角 semiseparable matrix；
2. **对偶层**：同一序列变换既能用线性递推计算，也能用类 Attention 的二次形式计算；
3. **算法层**：把矩阵分块，块内走 matmul，块间只传递压缩状态；
4. **架构层**：重排投影、增加归一化、引入 head/group 结构，使 Mamba-2 适合 Tensor Parallel 与大状态维度。

## 2. 问题背景、核心贡献与边界

### 2.1 为什么需要另一条序列建模路线

标准因果 Attention 在训练时要处理长度为 $T$ 的两两交互，核心代价为 $O(T^2)$，自回归推理还需保留随上下文增长的 KV cache。SSM 则把历史压入固定大小状态：训练可线性随序列长度增长，生成时状态大小不随 $T$ 增长。但 Mamba-1 的 selective scan 以标量递推/扫描为主，不能充分利用 GPU Tensor Core 擅长的大矩阵乘，且原 block 的串行投影会增加张量并行通信。

论文的核心问题不是“Attention 还是 SSM 谁更强”，而是：**能否找到一个共同的数学对象，使模型同时拥有 SSM 的线性状态压缩、Attention 的并行矩阵乘，以及 Transformer 已成熟的系统优化方式？**

### 2.2 核心贡献

1. **SSM = semiseparable matrix transformation**：状态维度 $N$ 对应下三角子块的秩上界，递推只是结构矩阵乘的一种算法。
2. **Structured Masked Attention（SMA）**：将 causal mask 推广为任意可快速乘法的结构矩阵 $L$，把 linear attention 的“换收缩顺序”推广成统一框架。
3. **Structured State Space Duality（SSD）**：证明 scalar-identity SSM 与 1-semiseparable SMA 具有完全相同的线性形式和二次形式；任意有界阶、可高效自回归的 SMA，其掩码必须是 semiseparable。
4. **SSD 分块算法**：对 semiseparable 矩阵做块分解；对角块用 Attention 风格 matmul，非对角块通过低秩状态因子传递，在 $T$ 上线性扩展且主要由矩阵乘构成。
5. **Mamba-2 与系统设计**：并行产生 $A,X,B,C$，增加输出前归一化，引入 Transformer 式 head vocabulary，做到每个 block 只需一次 all-reduce，并支持 sequence/context parallelism。

### 2.3 适用边界

SSD 仍是**选择性、时变 SSM**，但将 $A_t$ 从 Mamba-1 的对角结构进一步限制为每个 head 内的 scalar-identity：$A_t=a_t I$。这是以部分表达力换取硬件友好性。论文结论主要由 Pile 上最高 2.7B 参数模型、合成 MQAR 与 A100 kernel benchmark 支撑，不能直接外推到任意模态、任意超大规模训练。

## 3. 从 SSM recurrence 到序列变换矩阵

### 3.1 基本递推

线性时不变 SSM（LTI）为

$$
h_t = Ah_{t-1}+Bx_t,\qquad y_t=C^\top h_t,
$$

其中 $x_t,y_t\in\mathbb R$，隐状态 $h_t\in\mathbb R^N$。连续时间参数 $\mathring A,\mathring B$ 与步长 $\Delta$ 经离散化规则变成 $A,B$。当 $A,B,C$ 固定时，系统既能用 recurrence 计算，也等价于全局卷积。

Mamba 引入时变、输入依赖的 selective SSM：

$$
h_t=A_t h_{t-1}+B_t x_t,\qquad y_t=C_t^\top h_t.
$$

$A_t,B_t,C_t$ 随 token 改变，使模型能按内容选择保留或遗忘信息；代价是失去 LTI 卷积形式，只能依赖 scan/recurrence。对一个 head 内的 $P$ 个通道，可将同一序列变换广播到 $X,Y\in\mathbb R^{T\times P}$。

### 3.2 展开 recurrence

从零初始状态展开：

$$
h_t=\sum_{s=0}^{t}A_tA_{t-1}\cdots A_{s+1}B_sx_s,
$$

因此

$$
y_t=\sum_{s=0}^{t}C_t^\top A_tA_{t-1}\cdots A_{s+1}B_sx_s.
$$

定义下三角矩阵

$$
M_{ts}=\begin{cases}
C_t^\top A_tA_{t-1}\cdots A_{s+1}B_s,&t\ge s,\\
0,&t<s,
\end{cases}
$$

便有 $y=Mx$，或对多通道 $Y=MX$。这一步把“递归网络”变成了“结构矩阵乘法”：**模型是 $M$，recurrence 只是计算 $MX$ 的一个实现。**

## 4. Semiseparable matrix：为什么状态等价于低秩历史

### 4.1 定义与直觉

若下三角矩阵 $M\in\mathbb R^{T\times T}$ 中，任意完全位于下三角区域的子矩阵秩都不超过 $N$，则称其为 $N$-semiseparable。其 sequentially semiseparable（SSS）表示正是

$$
M_{ji}=C_j^\top A_jA_{j-1}\cdots A_{i+1}B_i.
$$

SSM 的状态维度 $N$ 因而有了矩阵意义：跨越任意时间切分，“过去输入对未来输出的影响”都必须通过至多 $N$ 维接口传递，所以离开对角线的块天然低秩。反过来，每个 $N$-semiseparable 矩阵也存在 $N$ 维 SSS 表示。因此论文得到精确等价：

$$
\mathrm{SSM}(A,B,C)(x)=\mathrm{SSS}(A,B,C)\,x.
$$

![](assets/Mamba-2%20-%20状态空间对偶与高效序列建模/figures/fig02_semiseparable_matrix.png)

*图 2：SSM 的序列变换矩阵。蓝色下三角子块的秩上界等于状态维度 $N$；状态就是跨时间块传递的低秩瓶颈。*

### 4.2 标量递推是 1-semiseparable 矩阵乘

当 $N=1$ 且忽略输入/输出缩放时，

$$
M_{ji}=\prod_{k=i+1}^{j}a_k,
$$

矩阵乘 $y=Mx$ 等价于

$$
y_t=a_ty_{t-1}+x_t.
$$

论文称这一原语为 **cumprodsum**。它既是最小 SSM，也是后续块间状态传播的核心。一般 $N$-SS 矩阵只需 $O(NT)$ 参数，并可在 $O(NT)$ 时间/空间完成矩阵向量乘；这说明高效 SSM 算法本质上是高效 semiseparable matrix multiplication。

### 4.3 这一步的重要性

将网络表示与具体 kernel 解耦后，可以选择不同矩阵分解：纯 recurrence、parallel scan、分块低秩分解乃至其它 HSS/SSS 算法。Mamba-2 的创新因此不只是“换一个 CUDA kernel”，而是先证明可计算对象的结构，再从结构推出更适合硬件的算法。

## 5. SMA 与 SSD：Attention 和 SSM 在哪里真正相遇

### 5.1 从 masked attention 到 SMA

去掉 softmax 的 kernel attention 可写成

$$
Y=(L\circ QK^\top)V,
$$

其中 $L$ 是因果掩码。把它看成四项张量收缩后，有两种顺序：

- **二次形式**：先算 $QK^\top$，再乘 mask 和 $V$；
- **线性形式**：先算每步的 $K^\top V$，再通过 $L$ 聚合，最后由 $Q$ 收缩。

传统 linear attention 的 $L$ 是下三角全 1 矩阵，其乘法就是 cumulative sum。论文指出：要线性化并不要求 $L$ 全为 1，只要求 $L$ 是可快速乘法的结构矩阵，于是定义

$$
\mathrm{SMA}(Q,K,V,L)=\operatorname{contract}(Q,K,V,L).
$$

不同 $L$ 可产生 causal、decay、Toeplitz、Fourier 或 semiseparable structured attention。

### 5.2 SSD 的精确交集

令 SSM 的转移为 scalar-identity：$A_t=a_t I$，则

$$
M_{ji}=\left(\prod_{k=i+1}^{j}a_k\right)(C_j^\top B_i).
$$

令

$$
L=1\mathrm{SS}(a),\qquad Q=C,\qquad K=B,\qquad V=X,
$$

便得到

$$
Y=\bigl(L\circ CB^\top\bigr)X.
$$

它一方面是 SSM recurrence，另一方面又是带 1-semiseparable mask 的 attention。这里 $a_t\in[0,1]$ 形成输入依赖的时间衰减/遗忘掩码，取代纯 causal mask 的“过去一律等权可见”。这就是 **state space dual model**：

- SSM 视角：线性时间、固定状态、适合自回归；
- Attention 视角：显式块内两两交互、适合矩阵乘并行；
- 矩阵视角：两者只是同一个 $M$ 的不同分解与收缩顺序。

进一步，论文证明：若一种 SMA 是有界阶的高效自回归过程，则其结构 mask 必须是 semiseparable。也就是说，semiseparable 不是随意选的技巧，而刻画了“能被固定阶 recurrence 实现的注意力掩码”。

## 6. SSD 分块算法、并行路径与复杂度

### 6.1 分块核心

将长度 $T$ 按 chunk size $Q$ 划为 $T/Q$ 块。semiseparable 矩阵的：

- **对角块**描述同一 chunk 内 input→output；
- **非对角块**描述跨 chunk 的影响，并可分解为 input→state、state→state、state→output 三个低秩因子。

![](assets/Mamba-2%20-%20状态空间对偶与高效序列建模/figures/fig03_ssd_block_algorithm.png)

*图 3：SSD 分块算法。橙色对角块用块内二次形式并行计算，绿色/黄色/蓝色低秩因子把跨块历史压缩为状态。*

完整计算分四步：

1. **块内输出（对角块）**：假设 chunk 初始状态为 0，用 quadratic SMA 计算该块输入对该块输出的贡献；各 chunk 完全并行，主要是 matmul。
2. **块内输入→状态（右因子）**：每块通过 $(N,Q)\times(Q,P)$ matmul 得到其“零初始状态下的末状态”，形状为 $(N,P)$。
3. **块间状态传播（中心因子）**：只在长度 $T/Q$ 的 chunk 序列上做 1-SS scan/cumprodsum，求每块真实初始状态。相比逐 token scan，递推长度缩短 $Q$ 倍。
4. **状态→块内输出（左因子）**：用 $(Q,N)\times(N,P)$ matmul 把历史状态转成当前块输出，与步骤 1 相加。

这种“块内 Attention、块间 SSM”的说法只是计算解释；二者计算的是同一个精确 SSD，不是近似或两个独立网络的拼接。

### 6.2 复杂度

论文在 $N=P=Q$ 时给出最整齐的结论：

| 计算方式 | 训练 FLOPs | 训练中间内存 | 自回归状态/推理特征 | 硬件特点 |
|---|---:|---:|---:|---|
| 标准 Attention | $O(T^2N)$ | 通常含 $O(T^2)$ score | KV cache 随 $T$ 增长 | matmul 友好 |
| 纯 selective scan | $O(TN^2)$ | 可做到 $O(TN)$ | 固定 $O(N^2)$ 状态 | 标量 scan 较难吃满 Tensor Core |
| SSD block algorithm | $O(TN^2)$ | $O(TN)$ | 总推理 FLOPs $O(TN)$、状态 $O(N^2)$ | 主要由 $N\times N$ batched matmul 构成 |

块间 scan 的工作量为 $O((T/Q)NP)$；令 $P=N=Q$ 后为 $O(TN)$，相对主 matmul 的 $O(TN^2)$ 可忽略。关键收益不是改变 SSM 的渐近 FLOPs，而是把相同量级的工作重排成 GPU 擅长的矩阵乘。

### 6.3 为什么大状态不再明显拖慢

Mamba fused scan 的时间会随 state dimension $N$ 近线性上升；SSD 把 $N$ 放入较大的矩阵乘维度，能提高 Tensor Core 利用率，所以从 $N=16$ 增至 $64/128/256$ 时延增长较缓。更大的状态能缓解固定状态模型的信息瓶颈，这在 MQAR 上尤其重要。

## 7. Mamba-2 block、head 与系统并行

### 7.1 从串行 Mamba block 到并行 Mamba block

Mamba-1 先把输入投影成 $X$，再从卷积后的 $X$ 生成 $\Delta,B,C$，投影链存在先后依赖。Mamba-2 把 SSD 看作

$$
(A,X,B,C)\mapsto Y,
$$

因此在 block 开头用一次并行投影同时产生这些分支，类似 Attention 同时产生 $Q,K,V$。同时在 gate 之后、输出投影之前增加 GroupNorm/RMSNorm/LayerNorm，缓解大模型训练不稳定。

![](assets/Mamba-2%20-%20状态空间对偶与高效序列建模/figures/fig04_mamba2_architecture.png)

*图 4：Mamba-2 将 $A,X,B,C$ 的生成移到 block 起点并行完成，减少串行投影，增加输出前归一化。*

### 7.2 Head 不是装饰，而是状态共享模式

对总维度 $D=HP$：

- $N$：每个序列变换的状态扩展维；
- $P$：head 内 value/input 维；
- $H$：head 数；模型变宽时通常固定 $N,P\in\{64,128\}$，增加 $H$。

论文把 Attention 的 head vocabulary 迁移到 SSM：

| SSM 模式 | Attention 类比 | 共享关系 |
|---|---|---|
| Multi-head SSM (MHS) | MHA | $A,B,C,X$ 均按 head 独立 |
| Multi-contract SSM (MCS) | MQA | $X,B$ 共享，$A,C$ 独立 |
| Multi-expand SSM (MES) | MKA | $X,C$ 共享，$A,B$ 独立 |
| Multi-input SSM (MIS) | MVA | $B,C$ 共享，$A,X$ 按 head 独立 |

Mamba/Mamba-2 默认的关键模式是 **MIS/MVA**。Mamba-1 可理解为 $P=1$ 的 MIS；Mamba-2 把 $P$ 提到 64/128，并可把 MIS 推广成 grouped-input SSM / grouped-value attention（GIS/GVA），以便分片。

### 7.3 Tensor Parallel

Mamba-1 若先得到局部 $X$ 再生成全局 $B,C,\Delta$，中间需要一次额外 all-reduce，连同输出端共两次通信。Mamba-2 直接从 block 输入并行产生各 GPU 自己的 $A,B,C,X$ group，每个 SSM head 完整留在单设备；GroupNorm 的 group 数设为 TP degree 的倍数，block 内无需额外同步，最终输出只做一次 all-reduce，和 Transformer Attention/MLP 相同。

![](assets/Mamba-2%20-%20状态空间对偶与高效序列建模/figures/fig05_tensor_parallelism.png)

*图 5：两路张量并行示意。输入/输出投影分片，每张 GPU 独立持有完整 SSM heads，末端仅一次 all-reduce。*

### 7.4 Sequence / Context Parallel

长序列可按时间切到不同 GPU。每个设备接收前一设备的初始状态，处理本地 chunk，再把末状态传给下一设备；通信随设备数线性增长。这与 SSD 的块分解完全同构。对 variable-length packing，只需在样本边界令 $A_t=0$，切断跨样本状态传播，无需构造复杂二维 Attention mask。

## 8. 实验、2–8× 加速与消融证据

### 8.1 Kernel 速度与 scaling

![](assets/Mamba-2%20-%20状态空间对偶与高效序列建模/figures/fig06_speed_benchmark.png)

*图 6：A100 80GB PCIe 上的核心算子时间。$N=64$ 时 SSD 比 Mamba fused scan 快 2–8×；约从长度 2K 起快于 FlashAttention-2。*

论文报告：

- SSD 核心比 Mamba 优化 selective scan 快 **2–8×**；
- 与 FlashAttention-2 在长度约 **2K** 处交叉，长度 **16K** 时约快 **6×**；
- 速度曲线在超长序列上保持近线性，但这些是核心 kernel benchmark，不等于完整语言模型端到端速度；
- 短序列（如 2K）时，纯 Mamba-2 每层都是 SSD，而同参数 Transformer 常是一半 Attention、一半高效 MLP，因此整模训练未必更快。

![](assets/Mamba-2%20-%20状态空间对偶与高效序列建模/figures/fig07_state_size_scaling.png)

*图 7：固定长度 4K 时，Mamba scan 随状态维度明显变慢，而 SSD 到 $N=256$ 仍变化较小。*

![](assets/Mamba-2%20-%20状态空间对偶与高效序列建模/figures/fig08_scaling_laws.png)

*图 8：Pile、长度 8192 的 Chinchilla scaling。125M–1.3B 范围内，Mamba-2 在困惑度—FLOPs 前沿上匹配或优于 Mamba 与 Transformer++。*

### 8.2 MQAR 与语言建模

MQAR 要模型记住多个 key-value 对，是固定状态模型的压力测试。Mamba-2 在 $N=16$ 时已显著优于 Mamba-1；把 $N$ 增至 64、256 又持续提升，说明更大状态确实增加可记忆信息容量。但作者明确承认：即使控制 $N=16$，究竟是哪项架构变化主导 Mamba-2 的提升仍未确定。

Pile 300B token 结果中：

- Mamba-2-780M：Pile ppl 7.26，零样本平均 53.5；
- Mamba-2-1.3B：Pile ppl 6.66，零样本平均 56.4；
- Mamba-2-2.7B：Pile ppl 6.09，零样本平均 60.2；
- 后者超过同数据上的 Mamba（2.8B，Pile ppl 6.22、平均 59.9）以及 Pythia-2.8B，并总体匹配/超过更大 Pythia。

### 8.3 关键消融

**Block 设计（约 125M）**：

| 投影 | 额外归一化 | 参数量 | Pile ppl ↓ |
|---|---:|---:|---:|
| Sequential | 否 | 129.3M | 11.76 |
| Sequential | 是 | 129.3M | 11.54 |
| Parallel | 否 | 126.5M | 11.66 |
| Parallel（Mamba-2） | 是 | 126.5M | **11.49** |

并行投影既少参数又略好，归一化继续改善困惑度，并在大规模预实验中提高稳定性。

**Head pattern**：在 $N=P=64$、总状态规模受控时，MIS/MVA 最好。125M 设置下 MIS 为 11.66，MCS/MQA 为 12.62，MES/MKA 为 12.59；360M 设置下分别为 8.73、9.33、9.36。这表明“共享哪一个投影”会实质影响能力，不能只按总状态量判断。

**Kernel approximation 的负结果**：cosFormer、Performer、Based/ReBased 等 softmax 近似没有稳定超过简单 pointwise activation；加入 softmax 分母式归一化还常导致不稳定。默认保留 Swish 主要为延续 Mamba-1，作者甚至指出完全移除 activation 可能更简单，但未充分测试。

**混合 Attention**：350M、48 层模型中，纯 Mamba-2 ppl 为 8.60；加入 4–6 个分散的 Attention 层降至 8.29–8.26，约 **10% Attention 层**最好。2.7B 上，58 SSD + 6 Attention 的 Mamba-2-Attention 获得 Pile ppl 5.95、零样本平均 61.0，优于纯 Mamba-2（6.09、60.2）和 Transformer++（6.13、60.2）。合理解释是：SSD 擅长持续压缩与序列映射，Attention 提供直接检索历史 token 的通道，两者互补。

## 9. Lyra 1.0：为何 3DGS decoder 混用 Transformer + Mamba-2

Lyra 1.0（*Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation*, arXiv:2509.19296）的目标是把视频扩散模型的多轨迹 latent 解码成显式 3D Gaussian Splatting。它采用 Mamba-2 的原因是**高效融合超长多视角序列**，不是让 Mamba-2 本身承担 3D 几何建模。

### 9.1 数据规模决定了不能全用全局 Attention

Lyra 每个样本使用 $V=6$ 条轨迹、每条 $L=121$ 帧，共 726 个 $704\times1280$ 视图。视频 VAE 以时间/空间各 8× 压缩，得到视频 latent $Z$；相机内外参转成 Plücker ray embedding $E$。两者经 $2\times2$ patchification 后成为长 token 序列。若每层都对这些 token 做全局 Attention，显存与计算随 token 数平方增长。

### 9.2 混合结构的明确分工

Lyra 的 reconstruction backbone 为

$$
[1\times\mathrm{Transformer}+7\times\mathrm{Mamba\text{-}2}]\times2,
$$

共 16 层、hidden size 512：

- **Transformer（2 层）**：提供显式全局 token-to-token 交互，尤其用于跨相机轨迹、跨时间和跨空间位置的信息对齐；
- **Mamba-2（14 层）**：以线性序列扩展和高硬件效率反复传播、混合长序列信息，使绝大多数层不承担二次 Attention 成本；
- **latent-space decoding**：真正避免 OOM 的第一层关键，不应把全部节省都归因于 Mamba-2。

Lyra 的消融也支持这种解释：混合 backbone 前向约 3213 ms，纯 Transformer 约 20922 ms，约 **6.5×** 加速；同时 PSNR 24.77 对 24.58，质量没有因高效化下降。去掉可学习的 multi-view fusion 后 PSNR 则从 24.77 跌到 17.73，说明跨轨迹融合本身不可省。

### 9.3 为什么它不是 3D 几何模型

Mamba-2 的输入和输出都是 token/hidden feature，它既不定义 3D Gaussian，也不直接执行投影或光栅化。Lyra 中真正产生几何的是后续预测头与监督：

1. 转置 3D 卷积把融合特征映射为每个候选 Gaussian 的 14 个参数；
2. 14 维包括位置 $(x,y,z)$、尺度 $(s_x,s_y,s_z)$、旋转四元数、opacity 与 RGB；
3. 3DGS renderer、相机 Plücker 条件、depth loss 和多视图重投影约束这些参数形成一致几何。

因此更准确的表述是：**Mamba-2 是 Lyra 3DGS decoder 内的高效序列融合骨干；显式 3D 表示是最终 Gaussian 参数集合，几何一致性来自相机条件、预测头、渲染与深度监督。**把 Lyra 使用 Mamba-2 解读成“Mamba-2 是 3D 生成模型”属于层级混淆。

## 10. 局限、复现清单与结论

### 10.1 局限与批判性判断

1. **理论范围有限**：SSD 对偶覆盖的是 scalar-identity SSM 与 1-semiseparable SMA，不是任意 softmax Attention；标题有意采用宽泛说法。
2. **表达力—效率交换**：$A_t=a_tI$ 比一般对角 $A_t$ 更受限。论文假设更大 $N$ 与架构可弥补，但没有证明在所有任务上都成立。
3. **固定状态仍是瓶颈**：混合少量 Attention 的结果优于纯 SSD，反而证明需要精确历史检索时，压缩状态并不总够用。
4. **速度结论依赖硬件与形状**：2–8× 是 A100 80GB PCIe 上核心算子对 Mamba scan 的结果；端到端还受投影、卷积、归一化、通信和 MLP 配比影响。
5. **规模与数据边界**：语言实验最高 2.7B、主要基于 Pile；尚不能据此确认数十/数百 B、不同数据配方或多模态训练的 scaling 排序。
6. **归因仍不完整**：MQAR 在相同 $N$ 下也大幅变好，作者未能确定是 block、head、归一化还是优化共同作用。
7. **最小实现不是生产 kernel**：Listing 1 要求长度可被 block size 整除，并显式构造块内 mask；复现吞吐需使用官方 fused kernels，而非只跑参考 PyTorch。

### 10.2 可操作复现路线

1. **数学单测**：在短序列上随机生成稳定的 $A,B,C,X$，比较逐 token recurrence、显式 $M=(L\circ CB^\top)$ 与 `ssd_minimal.py` 输出及梯度。
2. **算法单测**：遍历 $T,Q,N,P$，特别检查非整除长度、初始/末状态、causal 边界、packed sequence 中 $A_t=0$ 的隔离效果。
3. **数值稳定性**：分别测试 fp32、bf16；监测 `exp(segsum(A))`、长序列累计衰减与 GroupNorm 前后的幅值。
4. **性能复现**：固定 A100/H100 型号、dtype、batch、heads、$T/N/P/Q$，预热后同时报告 kernel time、峰值显存和端到端 tokens/s；不可只比较理论 FLOPs。
5. **能力复现**：先做 MQAR 的 $N=16/64/256$ 对照，再做 Pile 小模型；将 SSD、block 重排、extra norm、head pattern 分开消融。
6. **混合模型**：对固定参数/FLOPs预算，比较纯 SSD、纯 Transformer、约 10% Attention 混合，Attention 层应分散且避开首尾。

官方实现推荐阅读顺序：

1. `mamba_ssm/modules/ssd_minimal.py`：论文 Listing 1 对应的最小 SSD；
2. `mamba_ssm/modules/mamba2_simple.py`：简化 Mamba-2 block；
3. `mamba_ssm/modules/mamba2.py`：完整工程实现；
4. `mamba_ssm/models/mixer_seq_simple.py`：组装语言模型；
5. 再对照本地 TeX 的 `structure/efficient.tex`、`architecture.tex`、`systems.tex` 与 `experiments.tex`。

### 10.3 最终判断

Mamba-2 最重要的贡献不是宣告 Transformer 失效，而是给出一个更可迁移的抽象：**序列模型可以先被视为结构矩阵，再根据训练、生成、硬件和并行目标选择收缩顺序与分解算法。**SSD 把“线性 recurrence”和“二次 Attention”从两类模型改写为同一变换的两种计算视角；Mamba-2 则证明这套数学统一能落实为可扩展 block、较大状态和真实的 2–8× 核心加速。Lyra 等混合架构进一步表明，更现实的方向不是单一模块取代一切，而是让少量 Attention 负责稀疏的全局检索，让 Mamba-2 负责大多数长序列融合，再由领域专用头完成几何、视觉或其它结构化预测。

### 10.4 图表索引

| 图 | 文件 | 用途 | 验证状态 |
|---|---|---|---|
| 1 | `fig01_duality_roadmap.png` | SSD 理论路线图 | 已逐图 Read，完整清晰 |
| 2 | `fig02_semiseparable_matrix.png` | SSM 与半可分矩阵 | 已逐图 Read，完整清晰 |
| 3 | `fig03_ssd_block_algorithm.png` | SSD 分块算法 | 已逐图 Read，完整清晰 |
| 4 | `fig04_mamba2_architecture.png` | Mamba-1/2 block 对比 | 已逐图 Read，完整清晰 |
| 5 | `fig05_tensor_parallelism.png` | Tensor Parallel | 已逐图 Read，完整清晰 |
| 6 | `fig06_speed_benchmark.png` | 序列长度—速度 | 已逐图 Read，完整清晰 |
| 7 | `fig07_state_size_scaling.png` | 状态维度—速度 | 已逐图 Read，完整清晰 |
| 8 | `fig08_scaling_laws.png` | Pile scaling laws | 已逐图 Read，完整清晰 |
