# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（v1，22 页，含附录） | <https://arxiv.org/pdf/2601.16163> | 全文 | 2026-10-24 | 精读主来源 |
| `figures/01-teaser.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 1，p.1；源码 `fig/cosmos_policy_figure1.jpeg` | 2026-10-24 | 总览：输入输出与三种预测目标 |
| `figures/02-latent-frame-injection.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 2，p.4；源码 `fig/cosmos_policy_diffusion_sequence_v2_main_version.001.jpeg` | 2026-10-24 | 核心方法：latent frame injection |
| `figures/03-latent-sequence-detailed.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 8，p.16（附录 A.1）；源码 `fig/cosmos_policy_diffusion_sequence_v2_detailed_for_appendix.001.jpeg` | 2026-10-24 | 空白首帧、4 张相同图的实现细节 |
| `figures/04-balanced-batches.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 12，p.21（附录 A.4.1）；源码 `fig/balanced_batches_figure.001.jpeg` | 2026-10-24 | 策略 / 世界模型 / 价值三种条件掩码 |
| `figures/05-noise-distribution.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 9，p.16（附录 A.2.1）；源码 `fig/noise_schedules_figure.001.jpeg` | 2026-10-24 | 噪声分布改造 |
| `figures/06-aloha-rollouts.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 3，p.6；源码 `fig/cosmos_policy_aloha_rollouts.001.jpeg` | 2026-10-24 | ALOHA 四个真机任务示例 |
| `figures/07-aloha-results.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 4，p.8；源码 `fig/aloha_task_performance_results_v2.001.jpeg` | 2026-10-24 | 真机主结果 |
| `figures/08-baseline-failures.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 5，p.9；源码 `fig/pi05_openvlaoft_aloha_rollouts.001.jpeg` | 2026-10-24 | π0.5 / OpenVLA-OFT+ 失败模式 |
| `figures/09-world-model-refinement.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 6，p.9；源码 `fig/cosmos_policy_planning_rollouts.001.jpeg` | 2026-10-24 | rollout 精修前后的世界模型预测 |
| `figures/10-planning-results.jpg` | 官方源码图 | <https://arxiv.org/abs/2601.16163> | Figure 7，p.10；源码 `fig/planning_results.001.jpeg` | 2026-10-24 | model-based vs model-free 规划结果 |

## 在线材料（未下载，仅记录）

| 材料 | URL | 获取日期 | 用途 |
|---|---|---|---|
| arXiv LaTeX 源码包 | <https://arxiv.org/e-print/2601.16163> | 2026-10-24 | 全文与附录精读、提取官方原图 |
| 项目页 | <https://research.nvidia.com/labs/dir/cosmos-policy/> | 2026-10-24 | 确认 ICLR 2026、rollout 视频 |
| 官方代码 | <https://github.com/nvlabs/cosmos-policy> | 2026-10-24 | 复现资源 |
| HF 模型集合 | <https://huggingface.co/collections/nvidia/cosmos-policy> | 2026-10-24 | 4 个 checkpoint、224×224 输入分辨率、NSCLv1 非商用许可 |

## 提取与核验记录

- 使用 `research-field-survey/scripts/extract_source_figures.py --list` 下载 arXiv 源码包，图片均为源码 `fig/` 目录中的官方 JPEG 原图（非 PDF 内嵌碎片），用 `sips` 等比缩放到最长边 ≤2400 px、JPEG 质量 90。
- 图号与页码已用 PyMuPDF 扫描 `paper.pdf` 的图注逐一核对。
- Figure 7 的规划数值（49.0 / 52.0 / 60.0；70.0 / 70.7 / 84.0；59.5 / 61.3 / 72.0）和 Figure 4 的真机数值从图中读取，并与附录 Table 3 交叉核对。
