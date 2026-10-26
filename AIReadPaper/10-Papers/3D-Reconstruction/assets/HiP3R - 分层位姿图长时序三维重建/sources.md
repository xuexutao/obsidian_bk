# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2610.31635 | 全文，arXiv v1 | 2026-10-26 | 精读主来源、公式与实验数字核对 |
| `sources/project-framework.pdf` | 项目页方法框架矢量图 | https://kkakkkka.github.io/HiP3R/assets/method/framework.pdf | 项目页 Method | 2026-10-26 | 方法结构溯源与高清查看 |
| `figures/01-motivation.png` | 论文关键图 | https://arxiv.org/pdf/2610.31635 | Figure 1，第 2 页 | 2026-10-26 | 解释 chunk 对齐与 frame 冲突的区别 |
| `figures/02-method-pipeline.png` | 论文方法图 | https://arxiv.org/pdf/2610.31635 | Figure 2，第 3 页 | 2026-10-26 | 解释两阶段层次位姿图完整流程 |
| `figures/03-qualitative-reconstruction.png` | 定性对比图 | https://arxiv.org/pdf/2610.31635 | Figure 3，第 7 页 | 2026-10-26 | 展示未闭环、重复表面与 ghosting 的修复 |
| `figures/04-long-sequence-trajectories.png` | 长序列结果图 | https://arxiv.org/pdf/2610.31635 | Figure 4，第 9 页 | 2026-10-26 | 展示 KITTI、VBR、Oxford 的轨迹改进 |

## 网页与版本信息

- arXiv 摘要页：https://arxiv.org/abs/2610.31635
- arXiv HTML：https://arxiv.org/html/2610.31635v1
- 项目页：https://kkakkkka.github.io/HiP3R/
- 提交版本：v1，2026-10-22
- DOI：https://doi.org/10.48550/arXiv.2610.31635

## 开源代码核查

核查日期：2026-10-26。

1. 论文摘要写明“codes are available at project page”，链接指向项目页。
2. 项目页当前提供论文展示、交互重建、视频、轨迹比较和 `framework.pdf`，未发现 GitHub、Hugging Face、Papers with Code 或源码压缩包入口。
3. 以论文全名、HiP3R、作者 Zunnan Xu 搜索，未找到可确认归属于作者团队的官方仓库。
4. 因此本笔记按“尚未公开可用代码”记录，frontmatter `code: null`；后续若项目页更新，应补查并追加代码级实现章节。

## 补充材料

截至核查日期，arXiv 与项目页未发现独立 Supplementary Material。论文正文共 21 页，后半部分包含实现细节、额外消融与数据集说明。
