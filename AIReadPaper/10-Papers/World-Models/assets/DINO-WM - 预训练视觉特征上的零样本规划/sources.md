# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v2） | https://arxiv.org/pdf/2411.04983 | 全文（含附录） | 2026-10-15 | 精读主来源、公式与实验数据核对 |
| `figures/01-teaser.png` | arXiv TeX 源图 | https://arxiv.org/e-print/2411.04983v2 | Figure 1，正文第 3 页 | 2026-10-15 | 训练、测试时规划与性能动机总览 |
| `figures/02-method-pipeline.png` | arXiv TeX 矢量源图渲染 | https://arxiv.org/e-print/2411.04983v2 | Figure 2，正文第 5 页；源文件 `figures/arch.pdf` | 2026-10-15 | 完整方法架构与潜空间 MPC 数据流 |
| `figures/03-key-results.png` | arXiv TeX 源图 | https://arxiv.org/e-print/2411.04983v2 | Figure 8，附录第 16 页 | 2026-10-15 | PointMaze、Push-T、Granular 规划定性对比 |
| `figures/04-open-loop-rollouts.png` | arXiv TeX 源图 | https://arxiv.org/e-print/2411.04983v2 | Figure 4，正文第 6 页 | 2026-10-15 | Push-T 与 Granular 长期开环预测对比 |

## 官方页面与实现

| 材料 | 原始来源 | 获取日期 | 说明 |
|---|---|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2411.04983 | 2026-10-15 | arXiv ID、作者、版本历史与摘要；v2 更新于 2025-02-01 |
| ICML 2025 / PMLR 论文页 | https://proceedings.mlr.press/v267/zhou25t.html | 2026-10-15 | 正式发表信息：ICML 2025，PMLR 267:79115–79135 |
| ICML 2025 虚拟会场 | https://icml.cc/virtual/2025/poster/46026 | 2026-10-15 | 官方会议展示页与 lay summary |
| 项目页 | http://dino-wm.github.io | 2026-10-15 | 方法简介、六环境视频、泛化与失败案例可视化 |
| 官方代码 | https://github.com/gaoyuezhou/dino_wm | 2026-10-15 | 训练、CEM 规划、环境配置、数据与 checkpoint 说明 |
| 官方数据与 checkpoint | https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28 | 2026-10-15 | 公开数据；PointMaze、PushT、Wall 预训练模型 |

## 版本与材料说明

- 本地 `paper.pdf` 保存 arXiv v2 官方版本；该版本已经使用 ICML 2025 accepted 模板，并含完整附录。
- PMLR 页面确认论文正式发表于第 42 届 International Conference on Machine Learning（ICML 2025），PMLR 第 267 卷，第 79115–79135 页。
- PMLR 页面未列出独立 supplementary PDF；实验细节、消融、额外可视化和超参数已包含在 arXiv PDF 附录中，因此未伪造或另存不存在的补充材料。
- 四张本地图均来自 arXiv v2 TeX 源包；其中 Figure 2 原图为矢量 PDF，已按白底高分辨率完整渲染为 PNG。所有图片均经本地预览复核，未使用 PDF 内嵌碎片。
