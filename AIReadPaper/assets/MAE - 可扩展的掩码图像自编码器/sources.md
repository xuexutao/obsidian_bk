# MAE 来源记录

- 记录日期：2026-10-15
- 论文：Masked Autoencoders Are Scalable Vision Learners
- 作者：Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick
- arXiv ID：2111.06377
- 会议：CVPR 2022，pp. 16000–16009

## 1. 主要官方来源

1. arXiv 摘要页：<https://arxiv.org/abs/2111.06377>
   - v1：2021-11-11；v3：2021-12-19。
   - v2 增加迁移学习结果，v3 增加鲁棒性评测。
2. arXiv 官方 PDF：<https://arxiv.org/pdf/2111.06377>
   - 本地文件：`sources/paper.pdf`
   - 页数：14
   - 文件大小：7,445,748 bytes
   - SHA-256：`1b490443925c72a2b7c770f90dd797e248729ae34a57e1abfe9ed36751c4cc5b`
3. CVPR 2022 Open Access 页面：<https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html>
   - 用于核对会议、页码、作者、摘要、正式 PDF 与 supplemental。
4. CVPR 2022 正式 PDF：<https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf>
5. CVPR 2022 Supplemental：<https://openaccess.thecvf.com/content/CVPR2022/supplemental/He_Masked_Autoencoders_Are_CVPR_2022_supplemental.pdf>
6. 官方 PyTorch/GPU 复现仓库：<https://github.com/facebookresearch/mae>
   - README 明确说明：论文原始实现为 TensorFlow+TPU；仓库是 PyTorch+GPU 重实现。
   - 仓库包含预训练、微调、线性探测、可视化 demo 与 ViT-B/L/H checkpoint。
   - 许可证：CC-BY-NC 4.0。
7. arXiv 官方 TeX 源码入口：<https://export.arxiv.org/e-print/2111.06377>
   - 仅在临时目录解包，用于优先提取原始矢量图；未复制进 Vault。

## 2. 本地图片来源与验证

所有图片均来自 arXiv v3 官方 TeX 源码包的 `fig/*.pdf`，以 300 DPI 渲染为 PNG；没有使用网页转载图或 PDF 碎片图。2026-10-15 已逐张用 Read 进行视觉检查，确认边界完整、文字可读、无裁切。

### Figure 1：非对称 encoder-decoder

- 本地文件：`figures/fig1-asymmetric-architecture.png`
- 原始源码图：`fig/arch.pdf`
- 论文位置：Figure 1，PDF 第 1 页
- 官方链接：<https://arxiv.org/pdf/2111.06377#page=1>
- 分辨率：2438 × 1350
- SHA-256：`dfafd089a79f211e9c134132c07ffdd4ecaf923e6e807ac28b4b8b587d874e92`
- 验证结论：输入掩码、仅可见 patch 进入 encoder、decoder 接收完整 token 序列、像素 target 四部分均完整。

### Figure 2：ImageNet 80% 掩码重建

- 本地文件：`figures/fig2-imagenet-reconstruction-80-mask.png`
- 原始源码图：`fig/samples.pdf`
- 论文位置：Figure 2，PDF 第 2 页
- 官方链接：<https://arxiv.org/pdf/2111.06377#page=2>
- 分辨率：3713 × 1275
- SHA-256：`3a9dc6cf695df8a855849725dad983b571cd704532c128fdb051f5e68d78cd01`
- 验证结论：12 组三联图均完整；每组依次为 masked input、MAE reconstruction、ground truth；80% mask ratio 只保留 39/196 patches。

### Figure 8：模型 scaling 主结果

- 本地文件：`figures/fig8-scaling-main-result.png`
- 原始源码图：`fig/model_size.pdf`
- 论文位置：Figure 8，PDF 第 7 页
- 官方链接：<https://arxiv.org/pdf/2111.06377#page=7>
- 分辨率：2000 × 1000
- SHA-256：`ec92e5746a178dfbbe84f2dec4263c32089e94bb2299487e980c863d8340cc52`
- 验证结论：MAE、IN1K 监督训练、原始 ViT 的 IN1K/JFT-300M 曲线及 ViT-B/L/H 标注完整。

## 3. 数字核对范围

- 方法与消融：论文 §3、§4.1、Tables 1–2、Figures 5–7。
- ImageNet 主结果与 scaling：§4.2、Table 3、Figure 8。
- 线性探测与部分微调：§4.3、Figure 9、Appendix Table 12。
- 迁移：§5、Tables 4–7。
- 鲁棒性与实现细节：Appendix A–C、Tables 8–13。
- 复现入口与 checkpoint：facebookresearch/mae README、PRETRAIN.md、FINETUNE.md、`main_pretrain.py`、`main_finetune.py`、`main_linprobe.py`。
