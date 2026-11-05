# Sources

获取日期：2026-11-05

## 本地材料

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | SHA-256 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF，arXiv v2 | https://arxiv.org/pdf/2610.06805v2 | 全文，v2（2026-11-01） | `fc6b8b1c04d7f2809bcb44815bc9188a29488ebbc2f0f8e1c175d1e3f8f5e0eb` | 精读主来源；用 `workbuddy-pdf` CLI 的 `md` 模式读取，成功提取 1,150 行正文；CLI 提示 71 个图片对象未写出，关键图改用作者项目页高清原图 |
| `figures/overview.png` | 官方项目页高清图 | https://h-jepa.com/assets/figures/overview.png | 论文 Figure 1，PDF p.1；项目页 Fig.1 | `8eb9b0432ac88f7ea5ef64686a470bd208ab6ba9fb49bccd605c45096a8c98aa` | 总体层级学习结果与 coarse-to-fine 规划概览 |
| `figures/depth-probes-wide.png` | 官方项目页高清图 | https://h-jepa.com/assets/figures/depth_probes_wide.png | 论文 Figure 4 的 probe 面板，PDF p.5；项目页 Fig.2 | `fdcf51adb35a704031f3d8b0ab3b272147c84e2c130b6f873e0502b2035f6f11` | 选择性抽象：快变量随深度丢失、慢变量保留 |
| `figures/planning-results-pareto.png` | 官方项目页高清图 | https://h-jepa.com/assets/figures/planning-results-pareto.png | 论文 Figure 6 上半部，PDF p.6；项目页 Fig.4 | `9113f5afd2c06b5a939d21f1727b49950b724574e0acea38a3f051a585fab5dd` | 四个模拟环境的成功率—规划计算量 Pareto |
| `figures/goal-costs.png` | 官方项目页高清图 | https://h-jepa.com/assets/figures/goal-costs.png | 论文 Figure 7，PDF p.7；项目页 Fig.3 | `3f83e755594dc2e15a01d618b737e97e084817c407f4a1fa8256e84dabf7bf73` | AntMaze 各层 latent goal-cost geometry |
| `figures/droid-ablation.png` | 官方项目页高清图 | https://h-jepa.com/assets/figures/droid-ablation.png | 论文 Figure 9(b)，PDF p.9；项目页 Fig.6 | `e3ebd9e1eda327c9a4f0936e0a0fcc5439c97a7a1b67fead502c032e734accb6` | DROID slow-feature collapse、IDM、HWM 与 H-JEPA 消融 |

所有图片均于 2026-11-05 逐张预览核验：文件完整、清晰、坐标轴/图例/边缘未被裁切。`depth-probes-wide.png` 是项目页单独发布的 probe 宽图，不含论文 Figure 4 右侧的 decoder 可视化；正文据实只把它用于 probe 结果。

## 在线官方来源

| 材料 | URL | 获取日期 | 用途与备注 |
|---|---|---|---|
| arXiv 摘要页 v2 | https://arxiv.org/abs/2610.06805v2 | 2026-11-05 | 标题、作者、版本历史、DOI、摘要；v2 最后修订于 2026-11-01 |
| arXiv HTML v2 | https://arxiv.org/html/2610.06805v2 | 2026-11-05 | 交叉核对公式、图注、表格、章节和附录；与本地 PDF 同为 v2 |
| 作者项目页 | https://h-jepa.com/ | 2026-11-05 | 官方摘要、关键图高清资源、视频与实验说明 |
| 官方代码 | https://github.com/kevinghst/H-JEPA | 2026-11-05 | 只读克隆到临时目录分析；固定 commit `97054ecaff83e13839195ec4b92d74c08a0127c5`，`git cat-file -t` 验证为有效 commit，工作树保持 clean |
| 官方数据集 | https://huggingface.co/datasets/jepa-world-models/h-jepa | 2026-11-05 | FourRoom、AntMaze、Push-T 验证集、DROID 数据与固定评测任务 |
| 官方权重 | https://huggingface.co/jepa-world-models/h-jepa | 2026-11-05 | 模拟环境 84 个模型权重与 DROID 9 个模型权重；仓库 README 给出目录布局 |

## 代码核查范围

固定提交：`97054ecaff83e13839195ec4b92d74c08a0127c5`

重点核查文件：

- `h_jepa/main_hjepa.py`：训练入口、优化器参数组、数据归一化、Lightning 运行配置。
- `h_jepa/hjepa_utils.py`：逐层建模、teacher forcing、SIGReg/IDM 损失和端到端总损失。
- `h_jepa/models/hjepa.py`：层级编码、stride/window chunk、逐层裁剪与稀疏 level-1 编码。
- `h_jepa/models/jepa.py`：state/action window 对齐、JEPA encode/predict/rollout 与 latent cost。
- `h_jepa/models/encoders/seq_encoder.py`：高层 macro-action sequence encoder。
- `h_jepa/models/predictors/predictors.py`：causal attention 与 AdaLN action conditioning。
- `h_jepa/loss.py`：SIGReg 与 inverse-dynamics 实现。
- `h_jepa/hierarchical_solver.py`：top-down planner、跨层投影、子目标传递与闭环重规划。
- `h_jepa/config/train/base/ant.yaml`、`h_jepa/config/train/ant_hjepa_l3.yaml`：Visual AntMaze 三层训练张量与超参。
- `h_jepa/config/eval/ant_l3.yaml`：三层规划 horizon、candidate、优化步数和执行节奏。
- `h_jepa/config/train/base/droid.yaml`、`h_jepa/config/train/droid_hjepa_l2.yaml`、`h_jepa/config/eval/droid_l2.yaml`：DROID 的 IDM、5 fps、stride 3 与开放环评测。
- `docs/RESULTS.md`：论文结果的官方仓库转录值。

## 读取限制与版本边界

- `workbuddy-pdf` CLI 成功读取本地官方 v2 PDF，因此正文结论以 PDF 为主，不需要以 HTML 替代；PDF 图像对象未由文本模式导出，不影响文字、公式和表格读取，图片使用作者项目页原图。
- 论文 PDF 报告 DROID 重编码训练语料为 74,530 episodes、220,084 views；当前代码仓库 README 的下载/训练说明写 74,896 episodes（排除 16 个评测 clips）。笔记将前者作为论文事实，并把后者标为发布仓库口径差异，不强行合并。
- 论文只披露使用 H200 GPU；未披露模拟实验的 wall-clock 训练时长。仓库对 DROID 给出 2 GPU、总 batch 256、100 epochs、每 epoch 292 iterations，但没有可靠总时长，因此笔记不补造时间。
