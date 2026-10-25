# Sources

本笔记对应的材料来源清单。主来源为知乎回答（二手解读，无官方 PDF）；配图全部从原回答所引用论文的 arXiv 官方 PDF 按页裁剪（PyMuPDF，2.4× 渲染），未使用知乎转存图。论文 PDF 仅用于裁图，未在本地保留。

## 主来源

| 本地文件 | 材料类型 | 原始来源 | 原文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| （正文） | 知乎回答 | https://www.zhihu.com/question/2087917372912964602/answer/2097704075991184349 | 全文 | 2026-10-26 | 综述主来源 |

## 配图来源

| 本地文件 | 材料类型 | 原始来源 | 原文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `figures/01-unicontrolnet-teaser.png` | 定性结果图 | https://arxiv.org/pdf/2305.16322 | Figure 1, Page 3 | 2026-10-26 | 单/多条件可控生成示例 |
| `figures/02-ldm-architecture.png` | 架构图 | https://arxiv.org/pdf/2112.10752v2 | Figure 3, Page 4 | 2026-10-26 | LDM 条件接入框架 |
| `figures/03-ipadapter-decoupled-attn.png` | 架构图 | https://arxiv.org/pdf/2308.06721 | Figure 2, Page 4 | 2026-10-26 | 解耦交叉注意力 |
| `figures/04-controlnet-arch.png` | 架构图 | https://arxiv.org/pdf/2302.05543v3 | Figure 3, Page 4 | 2026-10-26 | ControlNet 零卷积连接 |
| `figures/05-t2iadapter-arch.png` | 架构图 | https://arxiv.org/pdf/2302.08453v2 | Figure 3, Page 4 | 2026-10-26 | T2I-Adapter 多条件加权注入 |
| `figures/06-unicontrolnet-framework.png` | 架构图 | https://arxiv.org/pdf/2305.16322 | Figure 2, Page 4 | 2026-10-26 | local / global 分路 |
| `figures/07-composable-diffusion.png` | 流程图 | https://arxiv.org/pdf/2206.01714v6 | Fig. 2, Page 5 | 2026-10-26 | 推理期 score 组合 |

## 涉及论文

- LDM：https://arxiv.org/abs/2112.10752
- ControlNet：https://arxiv.org/abs/2302.05543
- T2I-Adapter：https://arxiv.org/abs/2302.08453
- Uni-ControlNet：https://arxiv.org/abs/2305.16322
- IP-Adapter：https://arxiv.org/abs/2308.06721
- Composable Diffusion：https://arxiv.org/abs/2206.01714
