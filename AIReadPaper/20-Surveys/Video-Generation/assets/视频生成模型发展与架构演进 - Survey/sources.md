# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/videogpt.pdf` | 论文 PDF | https://arxiv.org/abs/2104.10157 | 全文 | 2026-08-25 | 图 1（Figure 2, p.3）：VQ-VAE + 自回归两阶段训练 |
| `sources/makeavideo.pdf` | 论文 PDF | https://arxiv.org/abs/2209.14792 | 全文 | 2026-08-25 | 图 2（Figure 2, p.4）：Make-A-Video 高层架构 |
| `sources/videoldm.pdf` | 论文 PDF | https://arxiv.org/abs/2304.08818 | 全文 | 2026-08-25 | 图 3（Figure 2, p.2）：Temporal Video Fine-Tuning 对比 |
| `sources/dit.pdf` | 论文 PDF | https://arxiv.org/abs/2212.09748 | 全文 | 2026-08-25 | 图 4（Figure 3, p.3）：DiT 架构与三种 conditioning 变体 |
| `sources/hunyuanvideo.pdf` | 论文 PDF | https://arxiv.org/abs/2412.03603 | 全文 | 2026-08-25 | 图 5（Figure 5, p.5）：总体架构；图 6（Figure 3, p.3）：训练系统 |
| `sources/cogvideox.pdf` | 论文 PDF | https://arxiv.org/abs/2408.06072 | 全文 | 2026-08-25 | 文中引用（3D VAE + expert transformer），未单独裁图 |
| `sources/animediff.pdf` | 论文 PDF | https://arxiv.org/abs/2307.04725 | 全文 | 2026-08-25 | 文中引用（motion module），未单独裁图 |
| `sources/moviegen.pdf` | 论文 PDF | https://arxiv.org/abs/2410.13720 | 全文 | 2026-08-25 | 文中引用（流匹配 + 多任务），未单独裁图 |

## 文中提及但未单独下载的材料

- **Sora 技术报告**：https://openai.com/index/video-generation-models-as-world-simulators/ —— OpenAI 官方博客，spacetime patches 原图见此页 "Turning visual data into patches" 一节。由于 OpenAI 页面为 React 动态渲染（img 标签通过 CDN 在客户端加载），WebFetch 无法抓取静态 URL，本综述未本地化原图，统一以 HunyuanVideo Figure 5 作为「3D VAE + DiT 时空统一表示」的可下载代表。

## 图片裁剪说明

所有 `figures/` 下图片均使用 `pymupdf`（zoom=2.4，约 172 DPI）从对应 PDF 页面裁剪，按「相对比例坐标 (x0, y0, x1, y1)」定位，保留完整图表与图注。

裁剪工具：`/Users/eilonxtxue/.workbuddy/skills/research-field-survey/scripts/extract_figure.py`

## 未引用的辅助材料（已下载但未在正文裁剪使用）

- `sources/videogpt.pdf` 中的样本结果图、qualitative examples
- `sources/makeavideo.pdf` 中的超分辨率与帧插值示意
- `sources/dit.pdf` 中的 patchify 与 GFlops 对比图（Figure 4）
- `sources/cogvideox.pdf`、 `sources/animediff.pdf`、 `sources/moviegen.pdf` 中的 teaser / 表 1 等

以上材料仅供后续细读时参考，本综述正文未引用。