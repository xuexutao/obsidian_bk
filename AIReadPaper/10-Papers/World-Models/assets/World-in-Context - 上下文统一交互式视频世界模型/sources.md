# Sources：World-in-Context

## 1. 核心结论

- 获取与核对日期：**2026-10-27**。
- 论文：*World-in-Context: Interactive Video World Modeling with In-Context Conditioning*。
- arXiv：`2610.33169v1`，发布日期 2026-10-23。
- 官方仓库固定 commit：`a985fd003458f07de5c8a537d750efbadd989aa9`。
- 截至核对日期，官方 GitHub **仅为占位展示仓库**。
- 推理代码、训练代码、配置、WiC 权重、数据和 examples 均未发布。
- README 第 33 行之后明确称 code/weights “under review”，其后命令属于 **planned release**。
- 因此代码判定为：**未开源 / coming soon**。
- 本来源记录只做静态核对，没有运行模型、推理、训练或 benchmark。

## 2. 官方入口

| 材料 | 官方 URL | 获取日期 | 用途 |
|---|---|---|---|
| arXiv 页面 | https://arxiv.org/abs/2610.33169 | 2026-10-27 | 论文元信息与版本 |
| 官方 PDF | https://arxiv.org/pdf/2610.33169 | 2026-10-27 | 公式、实验、附录与图表主来源 |
| 项目页 | https://haroldchen19.github.io/WiC-site/ | 2026-10-27 | 官方展示入口 |
| 官方 GitHub | https://github.com/aigc3d/World-in-Context | 2026-10-27 | 仓库状态与高分辨率资产 |
| 固定 commit | https://github.com/aigc3d/World-in-Context/commit/a985fd003458f07de5c8a537d750efbadd989aa9 | 2026-10-27 | 可重复仓库审计 |
| Wan backbone | https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B | 2026-10-27 | README 唯一链接的模型权重 |

## 3. 论文全文与本地抽取

| 本地材料 | 类型 | 原始来源 | 范围 | 用途 |
|---|---|---|---|---|
| `sources/paper-v1.pdf` | 官方 PDF v1 | https://arxiv.org/pdf/2610.33169 | 全文 22 页 | Obsidian 内长期留档；页面、图表、公式与附录核对 |
| `sources/paper-extracted.md` | 去 NUL 的全文抽取副本 | 官方 PDF | 全文、参考文献与附录 | 搜索、数字转录与公式核对 |

说明：PDF 是最终事实来源；文本抽取可能打乱公式、表格与分页，因此关键数字同时结合已准备的 PDF 裁图核对。

## 4. 十张图的 provenance

| 本地文件 | 材料类型 | 论文位置/官方资产 | 原始 URL | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `figures/01-teaser.png` | 官方仓库高分辨率 teaser | Figure 1，PDF Page 1；GitHub `assets/teaser.png` | https://github.com/aigc3d/World-in-Context/blob/a985fd003458f07de5c8a537d750efbadd989aa9/assets/teaser.png | 2026-10-27 | FREE/FPP/TPP 与双向切换总览 |
| `figures/02-icc-it-rope.png` | 官方仓库高分辨率方法图 | Figure 2，PDF Page 3；GitHub `assets/method.png` | https://github.com/aigc3d/World-in-Context/blob/a985fd003458f07de5c8a537d750efbadd989aa9/assets/method.png | 2026-10-27 | ICC 与 IT-RoPE 总体结构 |
| `figures/03-emergent-view-switching.png` | 官方 PDF 裁图 | Figure 3，PDF Page 4 | https://arxiv.org/pdf/2610.33169 | 2026-10-27 | 6.25%→37.5%→93.8% switching |
| `figures/04-view-motion-dynamics.png` | 官方 PDF 裁图 | Figure 4，PDF Page 5 | https://arxiv.org/pdf/2610.33169 | 2026-10-27 | FREE/FPP 振荡差异 |
| `figures/05-action-camera-control.png` | 官方 PDF 裁图 | Figure 5，PDF Page 5 | https://arxiv.org/pdf/2610.33169 | 2026-10-27 | action-only/camera-only/joint 四指标 |
| `figures/06-main-results.png` | 官方仓库高分辨率表格 | Table 1，PDF Page 6；GitHub `assets/table.png` | https://github.com/aigc3d/World-in-Context/blob/a985fd003458f07de5c8a537d750efbadd989aa9/assets/table.png | 2026-10-27 | WBench 主结果完整转录 |
| `figures/07-icc-conditioning-ablation.png` | 官方 PDF 裁图 | Figure 11，PDF Page 12 | https://arxiv.org/pdf/2610.33169 | 2026-10-27 | ICC/Additive/AdaLN 条件丰富度消融 |
| `figures/08-it-rope-ablation.png` | 官方 PDF 裁图 | Figure 12，PDF Page 12 | https://arxiv.org/pdf/2610.33169 | 2026-10-27 | 双分支、单分支与 endpoint 消融 |
| `figures/09-data-efficiency.png` | 官方 PDF 裁图 | Figure 13，PDF Page 12 | https://arxiv.org/pdf/2610.33169 | 2026-10-27 | 50K+250、100K+500、125K+750 数据效率 |
| `figures/10-wic-causal.png` | 官方 PDF 裁图 | Figure 15，PDF Page 15 | https://arxiv.org/pdf/2610.33169 | 2026-10-27 | WiC-Causal 长序列定性结果 |

图源分类核对：

- `01-teaser.png`、`02-icc-it-rope.png`、`06-main-results.png` 来自固定 commit 的官方 GitHub 高分辨率资产。
- `03`、`04`、`05`、`07`、`08`、`09`、`10` 来自官方 arXiv PDF 裁图。
- 正文全部使用相对笔记目录的 URL-encoded 路径。
- 没有使用二手媒体截图或第三方重绘图。

## 5. 官方仓库固定版本

- 本地只读快照：`/tmp/world-in-context-code-20261027`。
- 固定 commit：`a985fd003458f07de5c8a537d750efbadd989aa9`。
- commit 时间：2026-10-27T02:14:08Z。
- commit message：`update link`。
- commit 修改范围：只改 README 中论文标题、Report 链接与 arXiv ID。
- commit 未签名；这里只记录 GitHub API 的 `verified: false / reason: unsigned`，不据此判断内容真实性。

### 5.1 仓库树

GitHub tree API 在该 commit 下返回 **9 个 tracked files**，`truncated: false`：

1. `.gitignore`
2. `LICENSE`
3. `README.md`
4. `assets/badge-paper.svg`
5. `assets/badge-website.svg`
6. `assets/logo.png`
7. `assets/method.png`
8. `assets/table.png`
9. `assets/teaser.png`

缺失项：

- 无 Python 源码；
- 无 `wic/` package；
- 无推理入口；
- 无训练入口；
- 无 `requirements.txt`；
- 无 `pyproject.toml` / `setup.py`；
- 无 `configs/`；
- 无 `scripts/`；
- 无 `examples/`；
- 无数据格式文档；
- 无 WiC checkpoint、adapter 或权重索引。

### 5.2 API 与文本快照

| 本地文件 | 内容 | 核对日期 |
|---|---|---|
| `sources/github-repo-api.json` | 仓库元数据、homepage、license、默认分支 | 2026-10-27 |
| `sources/github-tree-a985fd0.json` | 固定 commit 的完整 tree；9 files | 2026-10-27 |
| `sources/github-commit-a985fd0.json` | commit 元数据、diff 与父 commit | 2026-10-27 |
| `sources/github-tags.json` | GitHub tags API 快照；结果 `[]` | 2026-10-27 |
| `sources/github-releases.json` | GitHub releases API 快照；结果 `[]` | 2026-10-27 |
| `sources/github-readme-a985fd0.md` | 固定 commit 的 README 快照 | 2026-10-27 |
| `sources/github-license-a985fd0.txt` | 固定 commit 的 LICENSE 快照 | 2026-10-27 |

## 6. README 已发布内容与 planned release 的边界

README 当前可直接验证的内容只有论文/项目链接、文字说明、图片和未来使用文档。

以下均为 README 的 **planned release interface**，当前仓库不存在对应源码/config，不能验证：

- Linux；
- Python 3.10；
- NVIDIA GPU with BF16；
- PyTorch 2.4；
- CUDA 12.4；
- FlashAttention 2.7.4；
- 推理 832×480；
- 81 frames；
- 16 FPS；
- CFG 5；
- 50 sampling steps；
- 11 examples：3 FREE、3 FPP、3 TPP、2 switching；
- 只优化 LoRA、condition encoder、IT-RoPE；
- per-GPU batch 1；
- Stage 1/Stage 2 JSON config；
- `wic.infer`、`wic.train` 等命令；
- `docs/data_format.md`；
- 输出 `video.mp4` 与 `ui.mp4`。

这些计划值不得写成论文实际训练配置，也不得声称命令已经运行。

## 7. 权重、tag 与 release 审计

- GitHub tags API：无 tag。
- GitHub releases API：无 release。
- 未发现官方 Hugging Face 上的 WiC 权重、WiC adapter 或完整模型卡。
- 未发现官方 ModelScope 上的 WiC 权重、WiC adapter 或完整模型卡。
- README 只链接公开的 `Wan-AI/Wan2.2-TI2V-5B` backbone。
- 可下载 Wan backbone 不等于可恢复 World-in-Context。
- 缺少 condition encoder、IT-RoPE、view/action/camera encoder 与训练 adapter 参数。

## 8. LICENSE 记录

- GitHub API 将仓库 license 识别为 `Apache-2.0`。
- 根目录保存 Apache License 2.0 全文。
- README 称未来实现计划按 Apache-2.0 发布，第三方代码与模型权重遵循各自许可证。
- 当前仓库没有实际实现代码，因而无法审计哪些未来文件受何种第三方条款约束。
- LICENSE 模板末尾保留：`Copyright [2023] [Zhongjie Duan]`。
- 该姓名与论文作者及 Alibaba Token Hub 标识不一致，疑似模板归属残留。
- 本记录仅陈述文本异常与 provenance，不作法律判断。

## 9. 最终开源覆盖判断

**已公开：**

- arXiv PDF；
- 项目页；
- 展示型 GitHub 仓库；
- README；
- LICENSE；
- teaser/method/table 高分辨率资产；
- Wan2.2-TI2V-5B 外部 backbone 链接。

**未公开：**

- WiC 推理代码；
- WiC 训练代码；
- 真实 config；
- WiC 权重/adapter；
- condition encoder 与 IT-RoPE 实现；
- 数据清单与样本；
- 500 switching sequences；
- 8K Stage 2 curriculum 构造脚本；
- examples；
- 评测脚本；
- 日志、seed、硬件与性能 profiling。

结论：截至 2026-10-27，World-in-Context 应按 **未开源 / coming soon** 处理；不能撰写或声称存在官方“代码级实现”。
