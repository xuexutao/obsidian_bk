# DINO 精读来源与图像溯源

- **论文**：Emerging Properties in Self-Supervised Vision Transformers
- **arXiv ID**：2104.14294v2
- **会议**：ICCV 2021，pp. 9650–9660
- **溯源日期**：2026-10-15
- **对应笔记**：[[10-Papers/Foundations/DINO - 自监督视觉Transformer的涌现性质]]

## 1. 一手来源

### 1.1 arXiv 官方记录

- 摘要页：https://arxiv.org/abs/2104.14294
- 官方 PDF：https://arxiv.org/pdf/2104.14294
- 版本：v2，2021-05-24
- 作者：Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin
- arXiv 页面标注：21 pages
- 用途：全文、公式、表格、附录消融、图像提取的首要依据。

### 1.2 ICCV / CVF Open Access 官方页面

- 论文页：https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html
- 会议版 PDF：https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf
- Supplementary Material：https://openaccess.thecvf.com/content/ICCV2021/supplemental/Caron_Emerging_Properties_in_ICCV_2021_supplemental.pdf
- 书目信息：Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 9650–9660
- 用途：核对正式发表信息、页码、摘要、论文与补充材料入口。

### 1.3 官方代码仓库

- 仓库：https://github.com/facebookresearch/dino
- README：https://github.com/facebookresearch/dino/blob/main/README.md
- 训练入口：https://github.com/facebookresearch/dino/blob/main/main_dino.py
- ViT 实现：https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
- k-NN 评测：https://github.com/facebookresearch/dino/blob/main/eval_knn.py
- Linear 评测：https://github.com/facebookresearch/dino/blob/main/eval_linear.py
- DAVIS 评测：https://github.com/facebookresearch/dino/blob/main/eval_video_segmentation.py
- Copydays 评测：https://github.com/facebookresearch/dino/blob/main/eval_copy_detection.py
- Attention 可视化：https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
- 许可证：Apache 2.0
- 用途：核对官方预训练权重结果、运行命令、软件版本、训练时间与下游评测流程。

## 2. 本地原文文件

- 相对路径：`sources/paper.pdf`
- 下载源：https://arxiv.org/pdf/2104.14294
- 下载日期：2026-10-15
- 文件大小：30,930,829 bytes
- PDF 版本：1.5
- 页数：21
- SHA-256：`aa464cfd59a428890190bea1065823a491a853478b4fb2a25f5eb44d442ce296`
- 完整性检查：文件头被系统识别为 PDF；PyMuPDF 可打开并读取 21 页。

## 3. 关键图表提取记录

所有图表均直接从 `sources/paper.pdf` 以 PyMuPDF 300 DPI 裁剪。
未使用 PDF 内嵌位图碎片，也未使用第三方重绘图。
页码按 PDF 文件从 1 开始计数。
坐标为 PDF point，顺序是 `(left, top, right, bottom)`。

| 本地文件 | 原论文编号 | PDF 页 | 裁剪坐标 | 像素尺寸 | 原始 URL | Read 验证 |
|---|---|---:|---|---:|---|---|
| `figures/figure-1-attention-emergence.png` | Figure 1 | 1 | `(45, 202, 550, 396)` | 2105×809 | https://arxiv.org/pdf/2104.14294 | 2026-10-15：完整，原图与图注均可见 |
| `figures/figure-2-dino-teacher-student.png` | Figure 2 | 2 | `(304, 68, 550, 361)` | 1026×1222 | https://arxiv.org/pdf/2104.14294 | 2026-10-15：完整，teacher/student、EMA、sg、centering、loss 与图注均可见 |
| `figures/table-2-imagenet-results.png` | Table 2 | 5 | `(304, 68, 548, 490)` | 1018×1759 | https://arxiv.org/pdf/2104.14294 | 2026-10-15：完整，表头、同架构比较与跨架构结果均可见 |
| `figures/figure-4-segmentation-comparison.png` | Figure 4 | 8 | `(47, 66, 290, 331)` | 1014×1105 | https://arxiv.org/pdf/2104.14294 | 2026-10-15：完整，监督/DINO mask、Jaccard 表与图注均可见 |
| `figures/table-7-component-ablation.png` | Table 7 | 8 | `(305, 68, 548, 329)` | 1014×1088 | https://arxiv.org/pdf/2104.14294 | 2026-10-15：完整，全部 9 行消融、缩写说明与图注均可见 |

## 4. 图表与笔记论点对应

### Figure 1

- 支撑论点：DINO-ViT 最后一层 `[CLS]` attention 在无标签训练下自动对齐前景对象。
- 限定：可视化不是因果解释，也不是专用分割模型输出。

### Figure 2

- 支撑论点：DINO 是 cross-view self-distillation。
- 图中可核对：不同视图、student、teacher、teacher centering、temperature softmax、stop-gradient、cross-entropy、EMA。
- 限定：图为单对视图的简化示意；完整训练使用两个 global views 与若干 local views。

### Table 2

- 支撑论点：DINO ViT-S/8 达到 78.3% k-NN；DINO ViT-B/8 达到 80.1% linear。
- 支撑论点：相同 ViT-S 下，DINO 的 k-NN/linear 明显高于 BYOL、MoCo-v2 与 SwAV。
- 限定：跨架构区块不是严格同计算量比较，patch size 对吞吐影响显著。

### Figure 4

- 支撑论点：PASCAL VOC12 上，DINO attention mask 的 Jaccard 明显高于监督 ViT。
- 数值：ViT-S/16 为 45.9，ViT-S/8 为 44.7。
- 限定：主文保留 60% attention mass，并展示最佳 head。

### Table 7

- 支撑论点：移除 momentum 后结果为 0.1/0.1，发生坍塌。
- 支撑论点：移除 multi-crop，k-NN/linear 从 72.8/76.1 降到 67.9/72.5。
- 支撑论点：CE 换 MSE 降到 52.6/62.4。
- 支撑论点：加入 predictor 或 Sinkhorn–Knopp 对默认 DINO 没有实质收益。

## 5. 关键事实页码索引

| 主题 | PDF 页码 | 原文位置 |
|---|---:|---|
| 两项涌现性质与贡献概述 | 1–2 | Abstract / Introduction |
| Figure 2 teacher–student 框架 | 2 | Figure 2 |
| Cross-view loss 与 multi-crop | 3 | Eq. (3), Section 3.1 |
| Teacher EMA | 4 | Teacher network |
| Centering + sharpening 防坍塌 | 4 | Avoiding collapse, Eq. (4) |
| ViT / projection head / BN-free | 4–5 | Network architecture / Vision Transformer |
| 训练超参数与 temperature | 5 | Implementation details |
| k-NN 与 linear 评测协议 | 5, 18–19 | Evaluation protocols / Appendix F |
| ImageNet 主结果 | 5–6 | Table 2 / Section 4.1 |
| 地标检索 | 6 | Table 3 |
| 复制检测 | 6 | Table 4 |
| DAVIS 视频对象分割 | 7 | Table 5 |
| Attention/object discovery | 7–8 | Figures 3–4 |
| 迁移学习 | 8 | Table 6 |
| 关键组件消融 | 8 | Table 7 |
| Teacher 构造消融 | 9 | Figure 6 |
| Center/sharpen collapse 分析 | 9 | Figure 7 / Eq. (5) |
| 计算与显存 | 9–10 | Table 8 |
| 小 batch | 10 | Table 9 |
| BYOL/MoCo/SwAV 方法对比 | 14–15 | Tables 13–15 |
| Projection head 细节 | 16 | Appendix C |
| Temperature / center 附加消融 | 17 | Appendix D |
| Multi-crop 附加消融 | 18 | Appendix E |

## 6. 官方仓库复现信息摘录

以下信息来自官方 README，核对日期为 2026-10-15：

- 开发环境：Python 3.6、PyTorch 1.7.1、CUDA 11.0、torchvision 0.8.2。
- Vanilla ViT-S：8 GPU、100 epochs、约 1.75 天；约 69.3% k-NN、74.0% linear。
- 增强配置：16 GPU、300 epochs、约 2.6 天；约 73.3% k-NN、76.0% linear。
- 官方权重结果：ViT-S/16 74.5/77.0，ViT-S/8 78.3/79.7，ViT-B/16 76.1/78.2，ViT-B/8 77.4/80.1（顺序为 k-NN/linear）。
- Copydays strong 示例：官方 README 给出 `mAP=0.858` 的评测输出示例；论文 Table 4 报告 ViT-B/8 为 85.5 mAP，两者受具体权重/运行配置影响，应以对应实验记录为准。
- DAVIS：README 明确提示使用 PyTorch 1.7.1；其说明称在多数 PyTorch 1.8.1 环境中无法复现相同结果。

## 7. 引用格式

```bibtex
@InProceedings{Caron_2021_ICCV,
  author    = {Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J\'egou, Herv\'e and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  title     = {Emerging Properties in Self-Supervised Vision Transformers},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {9650--9660}
}
```

## 8. 来源使用原则

- 方法、公式、实验数字：以本地 arXiv v2 PDF 为准。
- 正式出版信息：以 ICCV / CVF Open Access 页面为准。
- 运行命令、环境、权重与墙钟时间：以 `facebookresearch/dino` 官方 README 为准。
- 未使用博客、媒体解读或第三方重绘作为事实依据。
- 所有本地关键图均已逐张通过 Read 进行视觉完整性核验。
