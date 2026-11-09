# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2609.01551 | 全文，v1 | 2026-11-09 | 精读主来源、公式与实验核对 |
| `figures/01-temporal-pca-trajectories.png` | 论文图完整裁剪 | https://arxiv.org/pdf/2609.01551 | Figure 1，PDF p.4 | 2026-11-09 | 说明逐层时间表征几何 |
| `figures/02-layerwise-probing.png` | 论文图完整裁剪 | https://arxiv.org/pdf/2609.01551 | Figure 3，PDF p.6 | 2026-11-09 | 说明三类属性的逐层可读出性 |
| `figures/03-oscillatory-geometry.png` | 论文图完整裁剪 | https://arxiv.org/pdf/2609.01551 | Figure 4，PDF p.7 | 2026-11-09 | 说明帧数、层深与振荡几何 |
| `figures/04-spline-steering.png` | 论文图完整裁剪 | https://arxiv.org/pdf/2609.01551 | Figure 5，PDF p.8 | 2026-11-09 | 比较线性与 spline/tangent steering |
| `figures/05-nearest-neighbor-distance.png` | 论文图完整裁剪 | https://arxiv.org/pdf/2609.01551 | Figure 6，PDF p.9 | 2026-11-09 | 定量比较两条 steering 路径到经验轨迹的距离 |
| `figures/06-class-pca-separation.png` | 论文图完整裁剪 | https://arxiv.org/pdf/2609.01551 | Figure 7，PDF p.15 | 2026-11-09 | 对比相机运动与直觉物理的 PCA 类别分离 |
| — | arXiv 摘要页 | https://arxiv.org/abs/2609.01551 | 元信息、提交历史、DOI | 2026-11-09 | 标题、作者、版本与链接核验 |
| — | 官方交互可视化 | https://vid-rep-pca.netlify.app/ | 论文脚注 1、Figure 5 图注 | 2026-11-09 | V-JEPA 2 PCA Explorer、3D 轨迹与 steering playback |

## 代码与补充材料核查

- 截至 2026-11-09，论文 PDF、arXiv 摘要页、arXiv HTML 与官方交互可视化页面均未提供本论文的 GitHub、Hugging Face、训练/评测代码或配置文件。
- Web 检索未发现可确认属于作者团队的官方仓库；CatalyzeX 仅显示“Request Code”，不能视为开源。
- 未发现独立 Supplementary Material。论文附录已包含数据统计、层索引、线性 steering 公式、逐任务表格与额外曲线。
- 因无官方代码，本笔记不编造“代码级实现”章节；复现分析仅依据论文中公开的方法与超参。

## 已核对的一手文本

- 论文正文与附录：PDF pp.1–18。
- 主要公式：Eq. (1)（PDF p.4）、Eq. (2)–(4)（PDF p.5）、Eq. (5)（PDF p.12）。
- 主要结果：Figures 2–6（PDF pp.5–9），Tables 1–5（PDF pp.12–14），Figures 7–10（PDF pp.15–18）。
