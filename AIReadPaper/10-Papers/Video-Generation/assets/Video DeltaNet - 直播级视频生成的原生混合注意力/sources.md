# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 / 版本 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2609.20744 | arXiv v3，全文，21 页 | 2026-10-29 | 精读主来源、公式与实验核对 |
| `figures/01-attention-allocation.png` | 论文图 | https://arxiv.org/pdf/2609.20744 | Figure 2，p.3 | 2026-10-29 | 解释 Softmax / Linear 职责分配 |
| `figures/02-vdn-architecture.png` | 论文图 | https://arxiv.org/pdf/2609.20744 | Figure 3，p.5 | 2026-10-29 | 解释混合层和 VDA 分支架构 |
| `figures/03-quality-comparison.png` | 论文图 | https://arxiv.org/pdf/2609.20744 | Figure 5–6，p.9 | 2026-10-29 | 50/8 NFE 质量比较 |
| `figures/04-kernel-ablation.png` | 论文图 | https://arxiv.org/pdf/2609.20744 | Figure 7，p.11 | 2026-10-29 | Kernel 级效率消融 |

## 官方外部材料

| 材料 | URL | 核对版本 | 获取日期 | 用途 |
|---|---|---|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2609.20744 | v3，2026-10-02 | 2026-10-29 | 标题、作者、摘要、版本和 DOI |
| arXiv HTML | https://arxiv.org/html/2609.20744v3 | v3 | 2026-10-29 | 章节、公式、表格和附录交叉核对 |
| 项目页 | https://openvdn.github.io/ | 2026-10-29 页面 | 2026-10-29 | 方法演示、serving 说明与定性样例 |
| 官方代码仓库 | https://github.com/OpenVDN/vdn-minimax-h3 | commit `e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b` | 2026-10-29 | 代码级实现、配置、张量形状和训练流程 |
| 官方权重/模型卡 | https://huggingface.co/OpenVDN/vdn-minimax-h3 | 2026-10-29 页面 | 2026-10-29 | 权重结构、Diffusers/SGLang 用法、显存和最新性能 |
| MiniMax H3 基座 | https://huggingface.co/MiniMaxAI/MiniMax-H3 | 官方模型页 | 2026-10-29 | 基座溯源 |

## 代码核对清单

- `src/models/hybrid_attention.py`：共享 QKV、Window Softmax、Linear branch 与独立输出投影。
- `src/models/linear_attention/branch.py`：短卷积、逐帧统计、文本状态、双向 scan、RMSNorm 和输出门控。
- `src/models/linear_attention/delta_rule.py`：`vdn_solve`、`sana_scaled` 与 `vdn_scaled` 三种更新后端。
- `src/models/linear_attention/scan.py`：FP32 Gram 统计、显式对称化、前后向状态库和 decay bridge。
- `configs/training/stage_a1_c1_vdn_anchor.yaml`：15 帧局部窗口、首尾 anchor、K/V short conv、text state。
- `configs/training/stage_b_c1_vdn_anchor.yaml`：rank-64 QKVO LoRA 与 2,000-step 协同适配。
- `configs/training/stage_dmd_vdn.yaml`：8-step DMD2、3 次 fake-score 更新、250 generator steps。
- `configs/inference/8nfe_tuned_fp8_ulysses_b200.yaml`：8×B200 的 5+3 分支并行与 FP8 配置。

## 获取与使用说明

- 论文没有单独发布 supplementary PDF；附录已并入 arXiv 主 PDF。
- 关键图均从官方 PDF 的完整页面以高清渲染后裁剪，保留完整图和图注；未使用 PDF 内嵌碎片。
- 未下载约 82 GB 的完整模型权重；代码级分析基于官方仓库、论文和 Hugging Face 模型卡中公开的 resolved 结构信息。
- 代码采用 Apache-2.0；权重单独受 MiniMax H3 Community License 约束。
