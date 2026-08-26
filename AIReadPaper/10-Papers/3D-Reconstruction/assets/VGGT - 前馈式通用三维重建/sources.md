# 来源记录 — VGGT

## 官方论文

| 材料 | 说明 |
| --- | --- |
| `sources/paper.pdf` | VGGT 官方论文 PDF（arXiv 2503.11651） |
| 原始 URL | https://arxiv.org/abs/2503.11651 |
| 获取日期 | 2026-08-26 |
| 用途 | 全文精读、关键数据与图源 |

## 延伸精读：DPT

为展开正文 §2.4.1 / 第 5 节（dense head 与 DPT 谱系）而补充精读的论文。

| 材料 | 说明 |
| --- | --- |
| `sources/dpt_2103.13413.pdf` | DPT 官方论文 PDF《Vision Transformers for Dense Prediction》（Ranftl, Bochkovskiy, Koltun；ICCV 2021） |
| 原始 URL | https://arxiv.org/abs/2103.13413 |
| 获取日期 | 2026-08-26 |
| 用途 | Reassemble / Read 三变体 / Fusion 结构与消融数据（Table 1–9）的一手依据 |

## 关键图来源

| 图片 | 对应论文图 | 页码 | 用途 |
| --- | --- | --- | --- |
| `figures/fig1_teaser.png` | Figure 1（teaser） | Page 1 | 展示 VGGT 单次前馈同时输出相机/点图/深度/点轨迹的能力 |
| `figures/fig2_architecture.png` | Figure 2（Architecture Overview） | Page 3 | 展示 DINO patchify → 特殊 token → Alternating-Attention → 多头输出架构 |
| `figures/fig3_qualitative.png` | Figure 3（定性对比） | Page 4 | 多视图深度与点云重建的定性对比 |
| `figures/dpt_fig1_architecture.png` | DPT 论文 Figure 1（三联图） | Page 3（DPT PDF） | 展示 DPT 整体架构 + Reassemble 展开 + Fusion block 内部，支撑正文 §2.4.1 |

> VGGT 三张图均由官方 PDF 按图区坐标完整裁剪，非内嵌碎片。DPT 图同样由 DPT 官方 PDF 第 3 页按图区坐标（含 Left/Center/Right 三联）整体裁剪，400 dpi。
