# iBOT 来源与图像追溯

- 核验日期：2026-10-15
- 论文：iBOT: Image BERT Pre-Training with Online Tokenizer
- arXiv：2111.07832
- 会议：ICLR 2022（Poster）

## 1. 主要来源

1. 官方 arXiv PDF  
   https://arxiv.org/pdf/2111.07832  
   本地存档：`sources/paper.pdf`  
   用途：论文正文、公式、主实验、附录、图表与复现成本；本笔记中的论文页码均按该 29 页 PDF 计数。

2. ICLR 2022 / OpenReview 论文页  
   https://openreview.net/forum?id=ydopy-e6Dg  
   用途：会议版本、录用状态、评审与作者回复；OpenReview 页面标记为 Accept (Poster)。训练时间和显存数据也已收入论文 Appendix E, Table 19，因此数值以本地 PDF 为最终核验依据。

3. ICLR 2022 官方虚拟会场  
   https://iclr.cc/virtual/2022/poster/6156  
   用途：确认 ICLR 2022 Poster 元信息与官方摘要。

4. ByteDance 官方实现  
   https://github.com/bytedance/ibot  
   用途：训练命令、模型权重、日志、更新后的随机遮挡配置、下游评估与可视化脚本。仓库说明 ViT-S/16 的 800-epoch 复现配置，并报告随机遮挡 ViT-B/16 的 84.1% ImageNet 微调准确率和 51.5 COCO box AP。

## 2. 延伸比较来源

5. BEiT  
   https://arxiv.org/abs/2106.08254  
   用途：比较离线 DALL-E dVAE tokenizer、离散 token id 和两阶段预训练。

6. DINO  
   https://arxiv.org/abs/2104.14294  
   用途：比较跨视图 CLS 自蒸馏、EMA teacher、centering 与 sharpening；iBOT 的全局损失和防坍塌机制直接继承自 DINO。

7. MAE  
   https://arxiv.org/abs/2111.06377  
   用途：比较像素回归、75% 高遮挡率、非对称 encoder-decoder 与无需 teacher/tokenizer 的生成式 MIM 路线。

8. DINOv2  
   https://arxiv.org/abs/2304.07193  
   用途：确认 iBOT patch-level objective 被纳入 DINOv2 主配方；DINOv2 改为 DINO/iBOT 独立 projection heads，并加入 Sinkhorn-Knopp centering 与 KoLeo regularizer。

9. DINOv3  
   https://arxiv.org/abs/2508.10104  
   用途：确认 iBOT loss 继续出现在 DINOv3 的主训练与 refinement objective 中；DINOv3 进一步用 Gram anchoring 修复超长训练下 patch 局部一致性退化。

## 3. 本地图像追溯

所有图片均从官方 arXiv PDF 以 300 DPI 裁取。原因是论文关键框架和表格包含矢量元素，PDF 裁图比抽取内嵌位图更完整。每张图均已逐张视觉核验，未使用碎片图。

| 本地文件 | 原论文标识 | PDF 页 | 裁剪范围（pt） | 原始 URL | 内容 |
|---|---|---:|---|---|---|
| `figures/01-online-tokenizer-framework.png` | Figure 3 | 4 | x=72–540, y=55–285 | https://arxiv.org/pdf/2111.07832 | iBOT 在线 tokenizer、student/teacher、CLS 跨视图损失和 patch 同视图 MIM 损失 |
| `figures/02-local-semantic-patterns.png` | Figure 4（连同 Table 7） | 7 | x=72–540, y=78–325 | https://arxiv.org/pdf/2111.07832 | patch token 的车灯、狗耳等高层局部语义，以及条纹、卷曲表面等纹理模式 |
| `figures/03-main-results-lowshot-dense.png` | Tables 4–6 | 6 | x=72–540, y=70–332 | https://arxiv.org/pdf/2111.07832 | 低样本、无监督聚类、COCO 检测/实例分割与 ADE20K 语义分割主结果 |
| `figures/04-tokenizer-ablation.png` | Table 9 | 9 | x=70–540, y=55–305 | https://arxiv.org/pdf/2111.07832 | MIM、CLS 自蒸馏、共享 head、DINO/BEiT tokenizer 的关键消融 |

## 4. 版本与数值说明

- 本地 PDF 是 arXiv v3，日期 2022-01-27，并标注“Published as a conference paper at ICLR 2022”。
- 论文 Table 1–3 的 ViT-B/16 默认结果为 77.1% k-NN、79.5% linear、84.0% fine-tuning；ViT-L/16 + ImageNet-22K 在 512 分辨率微调为 87.8%。
- GitHub 后续更新的随机遮挡模型与论文默认 block-wise/random-MIM 调度并非完全相同：ViT-B/16 随机遮挡 [0.65, 0.75] 报告 84.1% fine-tuning、51.5 box AP。笔记分别标注，避免混用。
- 论文没有报告 VOC/COCO CorLoc 等标准“无监督对象发现”指标。笔记明确区分：Table 5 是无监督图像聚类；Figure 4/6 是局部模式与注意力的定性对象发现证据，不能替代标准 object discovery benchmark。
