# Sources

## 本地材料

| 本地文件 | 材料类型 | 原始来源 | 论文位置 / TeX 原文件 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF（arXiv v4） | https://arxiv.org/pdf/2408.08872v4 | 全文，12 页 | 2026-10-16 | 精读主来源、表格与页码核验 |
| `figures/01-framework-and-stages.png` | arXiv TeX 完整原图 | https://arxiv.org/e-print/2408.08872 | Figure 1，`figures/blip3-figure1-v1.pdf`，PDF p.2 | 2026-10-16 | BLIP-2/BLIP-3 对照与四阶段训练流程 |
| `figures/02-architecture.png` | arXiv TeX 完整原图 | https://arxiv.org/e-print/2408.08872 | Figure 2，`figures/figure1.pdf`，PDF p.2 | 2026-10-16 | 视觉塔、token sampler、LLM 与交错序列架构 |
| `figures/03-stage1-data-recipe.png` | arXiv TeX 完整原图 | https://arxiv.org/e-print/2408.08872 | Figure 3，`figures/figure2.pdf`，PDF p.5 | 2026-10-16 | Stage-1 数据采样比例与数据形态 |
| `figures/04-dataset-examples.png` | arXiv TeX 完整原图 | https://arxiv.org/e-print/2408.08872 | Figure 4，`figures/blip3-datasets-visualization-v4.pdf`，PDF p.6 | 2026-10-16 | OCR、高清 OCR、grounding 数据格式 |
| `figures/05-multi-image-examples.png` | arXiv TeX 完整原图 | https://arxiv.org/e-print/2408.08872 | Figure 6，`figures/sft-examples/sft-example.pdf`，PDF p.8 | 2026-10-16 | 多图与交错图文能力示例 |
| `figures/06a-anyres-ablation.png` | arXiv TeX 完整原图 | https://arxiv.org/e-print/2408.08872 | Figure 7(a)，`figures/sft-ablation/ocr-ablation.pdf`，PDF p.8 | 2026-10-16 | any-resolution 逐 patch 采样消融 |
| `figures/06b-instruction-aware-ablation.png` | arXiv TeX 完整原图 | https://arxiv.org/e-print/2408.08872 | Figure 7(b)，`figures/sft-ablation/instruct-ablation.pdf`，PDF p.8 | 2026-10-16 | instruction-aware sampler 消融 |

以上 7 张图片均直接从 arXiv v4 TeX 源码包中的 PDF 原图渲染，未从论文页面截图，也未抽取零散内嵌位图；已逐张使用 `Read` 复核边界、图例、坐标轴和文字完整性。

## 官方页面与发布资源

| 来源 | URL | 核验内容 |
|---|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2408.08872 | v1 提交于 2024-08-16；v4 更新于 2025-09-16；标题、作者、摘要、版本历史 |
| ICCV 2025 Workshops / Findings（CVF） | https://openaccess.thecvf.com/content/ICCV2025W/Findings/html/Xue_BLIP-3_A_Family_of_Open_Large_Multimodal_Models_ICCVW_2025_paper.html | 正式收录信息：ICCV Workshops 2025，pp. 6183–6194 |
| Salesforce AI Research 项目页 | https://www.salesforceairesearch.com/opensource/xGen-MM/index.html | 模型、数据、代码发布入口与 2024-05/07/08 发布记录 |
| 官方代码分支 | https://github.com/salesforce/LAVIS/tree/xgen-mm | xGen-MM 微调与推理代码 |
| 官方模型集合 | https://huggingface.co/collections/Salesforce/xgen-mm-1-models-662971d6cecbf3a7f80ecc2e | 当前可见的模型与数据条目 |
| 4B base 模型卡 | https://huggingface.co/Salesforce/xgen-mm-phi3-mini-base-r-v1.5 | Phi-3-mini 实际发布、few-shot 结果、Apache-2.0、风险说明 |
| 4B 多图指令模型卡 | https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5 | 单图/多图结果、实际发布状态与许可证 |
| 4B DPO 模型卡 | https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-dpo-r-v1.5 | 项目级 DPO 模型结果；注意 arXiv v4 正文已将 DPO 训练段落注释掉 |
| BLIP3-OCR-200M | https://huggingface.co/datasets/Salesforce/blip3-ocr-200m | 已发布数据集入口 |
| BLIP3-GROUNDING-50M | https://huggingface.co/datasets/Salesforce/blip3-grounding-50m | 已发布数据集入口 |

## 上游论文

| 工作 | URL | 在本笔记中的作用 |
|---|---|---|
| BLIP-2 | https://arxiv.org/abs/2301.12597 | Q-Former、ITC/ITM/ITG 两阶段桥接范式 |
| InstructBLIP | https://arxiv.org/abs/2305.06500 | instruction-aware Q-Former 与 BLIP-3 对应消融 |
| CLIP | https://arxiv.org/abs/2103.00020 | softmax 双向对比视觉语言预训练基线概念 |
| SigLIP | https://arxiv.org/abs/2303.15343 | BLIP-3 最终冻结视觉塔的上游预训练目标 |
| Phi-3 Technical Report | https://arxiv.org/abs/2404.14219 | BLIP-3 语言底座家族；3.8B mini 与 14B medium 的参数背景 |

## 版本与开放性说明

- arXiv v1/v2 与早期项目页使用标题 **xGen-MM (BLIP-3): A Family of Open Large Multimodal Models**；v4 PDF 与 ICCV Workshops 正式版本标题为 **BLIP-3: A Family of Open Large Multimodal Models**。
- v4 论文声称发布 4B 和 14B 的 base / instruction 模型，但正文只写 “Phi-3 series”，没有把两个规模明确映射到具体 Phi-3 checkpoint。当前官方 Hugging Face collection 可核验的主线是 `phi3-mini` 4B 系列；未在该集合中看到 14B 条目。因此笔记区分“论文报告/承诺”与“项目页实际可见发布”。
- 论文声称开放全部训练数据和预处理版本；当前官方入口可直接核验 BLIP3-KALE、BLIP3-OCR-200M、BLIP3-GROUNDING-50M 等，但未在官方 collection 中看到 BLIP3-OCR-HD-30M。不能把论文中的开放承诺等同于所有资源均已完整上线。
- 项目页与模型卡存在 DPO 安全模型；但 arXiv v4 正文没有 DPO 实验，TeX 中相关方法、数据与安全微调段落均被注释。笔记将其作为“项目发布补充”，不作为 v4 论文的正式实验结论。
- 模型卡明确说明代码与权重采用 Apache-2.0；训练数据来自多种上游数据集，许可证、隐私、版权与下架机制不能由模型许可证代替。
