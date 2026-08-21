# Sources

本目录收录论文 *An Empirical Study of Training Pixel-Space Text-to-Image Diffusion Models* 的官方与衍生材料,用于支撑 [[Pixel-Space T2I - 潜空间到像素空间的文生图训练]] 笔记的精读。

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | https://arxiv.org/pdf/2608.16887 | 全文 (9.6 MB) | 2026-08-21 | 精读主来源,所有公式/表格/消融的原始依据 |
| `figures/00-motivation-curves.png` | 动机定量曲线 | https://arxiv.org/html/2608.16887v1/pic/latent-vs-pixel-bench.png | Figure 1, 第 3 页 (Section 3) | 2026-08-21 | 第 1 节"像素空间预训练收敛更慢"的定量证据 |
| `figures/01-overview.png` | 训练过程定性样本 | 论文 PDF (arXiv:2608.16887) | Figure 2, 第 3 页 (Section 3) | 2026-08-21 | 第 1 节动机:同一 prompt 下 pixel/latent 在 2K/10K/50K/150K 步的视觉演化 |
| `figures/02-method.png` | 解码头对比 (JiT vs DiP) | 论文 PDF (arXiv:2608.16887) | Figure 8, 第 7 页 (Section 4.4) | 2026-08-21 | 第 3 节解码器:卷积 DiP 抹平线性头网格伪影 |
| `figures/03-results.png` | 端到端推理延迟分解 | 论文 PDF (arXiv:2608.16887) | Figure 12, 第 9 页 (Section 5.2) | 2026-08-21 | 第 5 节结果:20.12s → 0.20s 各级加速的来源分解 |
| `figures/04-patch.png` | 渐进式 patch 尺寸对比 | 论文 PDF (arXiv:2608.16887) | Figure 11, 第 9 页 (Section 5.1) | 2026-08-21 | 第 3/5 节:ps16/ps32/ps32-adapt16/ps64-adapt32 的局部细节与伪影 |

## 论文元数据

- **arXiv**: 2608.16887v1 (Submitted 17 Aug 2026)
- **DOI (arXiv 颁发, DataCite pending)**: https://doi.org/10.48550/arXiv.2608.16887
- **HTML 版**: https://arxiv.org/html/2608.16887v1
- **作者**: Dengyang Jiang, Ruoyi Du, Zhennan Chen, Dongyang Liu, Zanyi Wang, Mingzhe Zheng, Xiangpeng Yang, Huanqia Cai, Aiming Hao, Yuming Jiang, Peng Gao, Harry Yang, Steven Hoi
- **单位**: Alibaba Token Hub / Alibaba Group / HKUST / Nanjing University / UCSD
- **分类**: cs.CV
- **代码/项目页**: 论文未披露 (v1 截止 2026-08-17 暂未公开)

## 引用方式

```bibtex
@article{jiang2026empirical,
  title={An Empirical Study of Training Pixel-Space Text-to-Image Diffusion Models},
  author={Jiang, Dengyang and Du, Ruoyi and Chen, Zhennan and Liu, Dongyang and Wang, Zanyi and Zheng, Mingzhe and Yang, Xiangpeng and Cai, Huanqia and Hao, Aiming and Jiang, Yuming and Gao, Peng and Yang, Harry and Hoi, Steven},
  journal={arXiv preprint arXiv:2608.16887},
  year={2026}
}
```

## 补充说明

- 4 张原图 (`01-overview` ~ `04-patch`) 在笔记初次编写时已从论文 PDF 裁剪保存,本轮精读予以复用,未做移动或删除。
- 新增的 `00-motivation-curves.png` 为本轮从 arXiv HTML 官方原图直链下载,2347×910 PNG,长边 < 4000 px,无需 sips 压缩。
- 所有图片均能在论文 PDF 中定位到对应 Figure 编号,论文 PDF 已落盘到 `sources/paper.pdf`,便于在没有网络的本地环境回溯。
- 论文未提供 Supplementary Material PDF,部分超参数与组件消融表 (Table 4/5) 位于主论文附录。
