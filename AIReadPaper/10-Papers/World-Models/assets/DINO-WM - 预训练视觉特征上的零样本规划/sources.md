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

## 2026-10-31 源码、config 与权重 revision

| 对象 | 固定 revision | 已读文件 | 核验用途 |
|---|---|---|---|
| 官方 GitHub | `gaoyuezhou/dino_wm@0a9492fa12044b852ae9e001cc74604b79c8bb0c` | `train.py` | 模型实例化、优化器、DINO/动作/本体/decoder 维度 |
| 同上 | 同一 commit | `models/dino.py` | `facebookresearch/dinov2`、`dinov2_vits14`、`x_norm_patchtokens` |
| 同上 | 同一 commit | `models/visual_world_model.py` | `[B,T,3,S,S]→[B,T,N,E]`、动作/本体 tile+concat、rollout 与 loss detach 边界 |
| 同上 | 同一 commit | `models/vit.py` | frame-causal mask、6-layer predictor 输入输出保持 shape |
| 同上 | 同一 commit | `models/vqvae.py` | 默认辅助 decoder；`quantize=False`，不参与 predictor 梯度 |
| 同上 | 同一 commit | `planning/cem.py`、`planning/mpc.py`、`planning/objectives.py` | `[N_c,T,K]` 候选、elite 更新、终点 latent 代价、闭环重规划 |
| 同上 | 同一 commit | `conf/train.yaml`、`conf/predictor/vit.yaml`、`conf/plan_pusht.yaml` | 224→196/196 patches/384D；`H=3`；CEM 300/30/30；源码 predictor lr `5e-4` |
| HF 数据/权重镜像 | `gaoyuezhou/dino_wm@d5c69e8fe60c3ed6b01d2f6e88fbfd66be078e57`（dataset repo） | `checkpoints/outputs.zip`、dataset archives | PointMaze/PushT/Wall checkpoint 与四类数据固定快照 |

### 关键边界与差异记录

- GitHub 当前 HEAD 与论文接收版不是同一类 revision：前者锁定“可运行公开代码”，后者锁定“论文实验描述”。数值不一致时两者分列，不用当前 config 篡改论文历史。
- 当前 `conf/train.yaml` 的 `predictor_lr=5e-4`；旧笔记误写为 `5e-5`，已修正并注明来源。
- 论文延迟段落使用 100 CEM samples / 10 iterations；当前 PushT 规划 config 是 `num_samples=300,opt_steps=30,topk=30`。两者不能合并成一个“官方默认”。
- HF `outputs.zip` 仅声明 PointMaze、PushT、Wall 模型；Rope/Granular 只有数据，没有公开 checkpoint。
- DINOv2 权重由源码通过 `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")` 获取，但上游 DINOv2 commit 未被 DINO-WM 仓锁定；严格离线复现还应自行固定上游 revision/权重哈希。

## 版本与材料说明

- 本地 `paper.pdf` 保存 arXiv v2 官方版本；该版本已经使用 ICML 2025 accepted 模板，并含完整附录。
- PMLR 页面确认论文正式发表于第 42 届 International Conference on Machine Learning（ICML 2025），PMLR 第 267 卷，第 79115–79135 页。
- PMLR 页面未列出独立 supplementary PDF；实验细节、消融、额外可视化和超参数已包含在 arXiv PDF 附录中，因此未伪造或另存不存在的补充材料。
- 四张本地图均来自 arXiv v2 TeX 源包；其中 Figure 2 原图为矢量 PDF，已按白底高分辨率完整渲染为 PNG。所有图片均经本地预览复核，未使用 PDF 内嵌碎片。
