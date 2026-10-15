# Sources — Mamba-2

生成日期：2026-10-15

## 1. 论文主来源

- 论文：Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. ICML 2024.
- arXiv 摘要页：https://arxiv.org/abs/2405.21060
- arXiv PDF：https://arxiv.org/pdf/2405.21060
- arXiv TeX 源码：https://arxiv.org/e-print/2405.21060
- arXiv ID：2405.21060；本地保存版本：v1（2024-05-31）。

### 本地文件

| 内容 | 相对路径 | SHA-256 |
|---|---|---|
| 官方 PDF | `sources/arxiv-2405.21060v1.pdf` | `6fe16414fa471a5ab150cddcba5840d3a94e347e96992c50c12c1fb8b919479f` |
| 官方 TeX 源码包 | `sources/arxiv-2405.21060-source.tar` | `ad88400a9f02116d8386524a1a220c880ef1a367f6cfc5ca03b508dd0be60f27` |
| 解包后的 TeX | `sources/tex/` | 来源同上 |
| PDF 抽取文本与逐页渲染 | `sources/fetched/` | 由官方 PDF 本地生成 |

`file` 核验结果：PDF 为 PDF 1.5；arXiv 源码包为 gzip-compressed tar 数据。

## 2. 官方实现

- 官方仓库：https://github.com/state-spaces/mamba
- 本地源码快照：`sources/state-spaces-mamba-main.zip`
- 解包目录：`sources/code/mamba-main/`
- SHA-256：`9ae06061ff50f6f6e2a442570837d911acdf49141482cf3bddb2011669944197`

建议核对的实现文件：

| 用途 | 本地相对路径 |
|---|---|
| SSD 最小参考实现（论文 Listing 1） | `sources/code/mamba-main/mamba_ssm/modules/ssd_minimal.py` |
| Mamba-2 简化实现 | `sources/code/mamba-main/mamba_ssm/modules/mamba2_simple.py` |
| Mamba-2 完整实现 | `sources/code/mamba-main/mamba_ssm/modules/mamba2.py` |
| 语言模型组装 | `sources/code/mamba-main/mamba_ssm/models/mixer_seq_simple.py` |

说明：仓库快照下载自 `main` 分支，属于抓取当日快照，不等同于带固定 tag 的论文发布版本；复现实验时应额外记录实际 commit。

## 3. 原图来源与逐图验证

以下 PNG 均直接由 arXiv 官方 TeX 包中的矢量 PDF 以 4× matrix（约 288 DPI）本地栅格化；未使用网页截图或二手重绘。每张图均已使用 Read 单独打开核验，确认画布完整、边缘未裁切、文字与曲线可读。

| 笔记图 | TeX 原始文件 | 论文含义 | Read 核验 |
|---|---|---|---|
| `figures/fig01_duality_roadmap.png` | `sources/tex/fig/ssd_roadmap.pdf` | Figure 1：SSD 路线图 | 完整、清晰 |
| `figures/fig02_semiseparable_matrix.png` | `sources/tex/fig/semiseparable.pdf` | Figure 2：SSM = semiseparable matrix | 完整、清晰 |
| `figures/fig03_ssd_block_algorithm.png` | `sources/tex/fig/ssd_algorithm.pdf` | Figure 5：SSD block decomposition | 完整、清晰 |
| `figures/fig04_mamba2_architecture.png` | `sources/tex/fig/architecture_2.pdf` | Figure 6：Mamba-1/2 block 对比 | 完整、清晰 |
| `figures/fig05_tensor_parallelism.png` | `sources/tex/fig/mamba_tp.pdf` | Figure 7 左：Tensor Parallel | 完整、清晰 |
| `figures/fig06_speed_benchmark.png` | `sources/tex/fig/ssm_ssd.pdf` | Figure 10 左：长度—速度 | 完整、清晰 |
| `figures/fig07_state_size_scaling.png` | `sources/tex/fig/ssm_ssd_dstate.pdf` | Figure 10 右：状态维度—速度 | 完整、清晰 |
| `figures/fig08_scaling_laws.png` | `sources/tex/fig/pile_8k_mamba2.pdf` | Figure 9：Pile scaling laws | 完整、清晰 |

对应图注可在以下 TeX 文件复核：

- `sources/tex/structure/intro.tex`
- `sources/tex/structure/ssm.tex`
- `sources/tex/structure/efficient.tex`
- `sources/tex/structure/architecture.tex`
- `sources/tex/structure/systems.tex`
- `sources/tex/structure/experiments.tex`

## 4. Lyra 1.0 专节来源

- 论文：*Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation*
- arXiv：https://arxiv.org/abs/2509.19296
- arXiv HTML：https://arxiv.org/html/2509.19296
- 使用信息：3DGS decoder 输入、`[1 Transformer + 7 Mamba-2] × 2` 架构、16 层/hidden size 512、14 维 Gaussian 输出、multi-view fusion 与纯 Transformer 消融。

Lyra 信息只用于笔记第 9 节的应用解释；Mamba-2 的理论、算法、速度和语言实验数字均以 arXiv:2405.21060 原文为准。

## 5. 可追溯性说明

- 主笔记：`../../Mamba-2 - 状态空间对偶与高效序列建模.md`
- 图片引用均指向本目录下 `figures/`，可离线浏览。
- `sources/fetched/source.md` 是从官方 PDF 提取的全文文本，便于全文检索；公式排版与最终数字应以本地官方 PDF/TeX 为准。
- `sources/fetched/images/pages/` 保留 52 页整页渲染用于视觉复核；未清理个人 Vault 中的取材文件。
- 本次任务未编辑 `Paper-Index.md`。
