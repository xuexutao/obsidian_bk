# Sources

## 1. 主论文

- Nichol, Alex; Dhariwal, Prafulla. **Improved Denoising Diffusion Probabilistic Models**. ICML 2021 / arXiv:2102.09672v1.
- arXiv 摘要页：https://arxiv.org/abs/2102.09672
- arXiv 官方 PDF：https://arxiv.org/pdf/2102.09672
- 本地官方 PDF：`Improved_DDPM_arXiv_2102.09672.pdf`
- PDF 大小：12,442,086 bytes
- SHA-256：`1f041c12d72020bb40cc69c644694027fa89131ab05349ec07dc48a929f75adf`
- 全文抽取：`source.md`
- 抓取元数据：`meta.json`

## 2. 官方实现

- OpenAI 官方代码：https://github.com/openai/improved-diffusion
- 仓库对应论文中的 learned sigma、cosine schedule、loss-second-moment importance sampling、timestep respacing 与 DDIM sampling。

## 3. 前后关系论文

- Ho, Jain, Abbeel. **Denoising Diffusion Probabilistic Models**. arXiv:2006.11239：https://arxiv.org/abs/2006.11239
  - 本文的直接基线：预测噪声的 `L_simple`、固定反向方差、线性噪声日程与长链 ancestral sampling。
- Dhariwal, Nichol. **Diffusion Models Beat GANs on Image Synthesis**. arXiv:2105.05233：https://arxiv.org/abs/2105.05233
  - 后续常称 ADM；在 Improved DDPM 的可扩展扩散骨架上继续做架构消融，并引入 classifier guidance，在 fidelity–diversity 间提供可控权衡。

## 4. 已核验原文图表

以下图片均从本地官方 PDF 以 300 DPI 裁剪，并保留原文图号、坐标轴、图例或表格标题：

1. `fig01_variance_and_vlb_terms.png`：原文 Figure 1–2，方差上下界接近程度与 VLB 各时间步贡献。
2. `fig02_cosine_schedule.png`：原文 Figure 3–5，linear/cosine 加噪过程与累计信号曲线。
3. `fig03_objectives_and_ablation.png`：原文 Figure 6–7、Table 1–2，目标函数、梯度噪声及消融结果。
4. `fig04_fewer_sampling_steps.png`：原文 Figure 8，采样步数与 FID。
5. `fig05_precision_recall_and_samples.png`：原文 Table 4、Figure 9，GAN 对比、precision/recall 与样本多样性。
6. `fig06_scaling.png`：原文 Figure 10，模型规模、训练算力与 FID/NLL scaling。
7. `fig07_objective_tradeoff.png`：原文 Figure 13–14，`L_hybrid` 与 `L_vlb` 在时间步上的互补及样本差异。
8. `fig08_overfitting.png`：原文 Figure 16，cosine schedule 的更快优化与更早过拟合。

## 5. 核验说明

- 主论文 PDF 共 17 页；全文抽取 54,495 字符；抓取器报告 `need_agent_fetch=false`、`warnings=[]`。
- 笔记中的公式、数值和实验结论以本地官方 PDF 为主，arXiv HTML 仅用于交叉核对可搜索文本。
- `images/pages/` 为逐页渲染核验材料；`images/img_*` 为 PDF 内嵌位图抽取，不作为方法图的唯一依据。