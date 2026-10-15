# Sources

获取日期：2026-09-04

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 用途 |
|---|---|---|---|---|
| `sources/paper.pdf` | 官方技术报告 PDF | <https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf> | 全文，20 页 | 精读主来源、公式与实验核验 |
| `figures/01-teaser.png` | PDF 完整裁图 | 同上 | Figure 1, p.4 | 动机、能力与应用概览 |
| `figures/02-system-overview.png` | PDF 完整裁图 | 同上 | Figure 2, p.5 | 数据、四阶段训练与流式推理总览 |
| `figures/03-worldplay-architecture.png` | PDF 完整裁图 | 同上 | Figure 4, p.7 | Chunk-wise AR、双动作与记忆重构 |
| `figures/04-dual-action-dit.png` | PDF 完整裁图 | 同上 | Figure 5, p.7 | Dual Action / PRoPE / DiT 注入结构 |
| `figures/05-context-forcing.png` | PDF 完整裁图 | 同上 | Figure 7, p.10 | Context Forcing 与 memory-aligned DMD |
| `figures/06-key-results.png` | PDF 完整裁图 | 同上 | Table 2 + Figure 9, p.13 | 短长期指标、VBench、人类偏好与消融 |

## 官方在线材料

| 材料 | URL | 获取日期 | 用途与核验结果 |
|---|---|---|---|
| HY-World / HunyuanWorld 官方项目页 | <https://3d-models.hunyuan.tencent.com/world/> | 2026-09-04 | 项目入口；当前页面已以 HY-World 2.0 为主，1.5 细节以报告与仓库为准 |
| HY-WorldPlay 官方 GitHub | <https://github.com/Tencent-Hunyuan/HY-WorldPlay> | 2026-09-04 | 模型列表、训练/推理代码、24 FPS 说明、显存需求与发布状态 |
| HY-WorldPlay 官方模型 | <https://huggingface.co/tencent/HY-WorldPlay> | 2026-09-04 | Checkpoints 下载入口 |
| 在线 Demo | <https://3d.hunyuan.tencent.com/sceneTo3D> | 2026-09-04 | 交互体验入口 |

## 资料缺口

- 未找到与该技术报告分离发布的 Supplementary Material；详细 Action/Memory/Context Forcing 消融被报告指向 WorldPlay 独立论文，WorldCompass 细节被指向其独立论文。
- 报告未披露完整四阶段训练的 GPU 型号/总卡数、总 GPU-hours、学习率、batch size、RL reward 权重和总训练时长。
- 报告没有专门的失败案例图、失败率、物理约束违反率或 agent task success 指标。
- `24 FPS` 是 8-GPU 并行、4-step distillation、量化、KV cache、流式 VAE/Triton 的系统结果；未披露统一硬件上的逐项延迟消融。
