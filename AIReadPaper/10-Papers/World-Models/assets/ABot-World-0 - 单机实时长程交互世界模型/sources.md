# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2607.19191 | 全文 | 2026-10-16 | 精读主来源 |
| `figures/01-teaser.png` | arXiv 源码原图 | https://arxiv.org/abs/2607.19191 | Figure 1 | 2026-10-16 | 动作控制场景与系统定位 |
| `figures/02-training-pipeline.png` | arXiv 源码原图 | https://arxiv.org/abs/2607.19191 | Figure 3 | 2026-10-16 | 双向教师、因果蒸馏与 LongForcing |
| `figures/03-model-architecture.png` | arXiv 源码原图 | https://arxiv.org/abs/2607.19191 | Figure 4 | 2026-10-16 | 动作 adapter 与角色身份记忆 |
| `figures/04-longforcing-ablation.png` | arXiv 源码原图 | https://arxiv.org/abs/2607.19191 | Figure 10 | 2026-10-16 | 60 秒 LongForcing 消融 |

## 官方链接

- arXiv：https://arxiv.org/abs/2607.19191
- arXiv HTML：https://arxiv.org/html/2607.19191
- 项目页：https://amap-cvlab.github.io/ABot-World/
- GitHub：https://github.com/amap-cvlab/ABot-World
- 在线 Studio：https://abot-world.amap.com/
- 模型权重：https://huggingface.co/acvlab/ABot-World-0-5B-LF

## 核验说明

- 四张图片均从 arXiv 官方源码包提取，已使用 `Read` 逐张视觉核验；未发现边缘、坐标轴、图例或文字被裁切。
- `paper.pdf` 可正常解析，共 33 页并具有文本层。
- “最高 16 FPS”对应 Table 2 的 MXFP4 配置 15.831 FPS；质量导向 FP8 配置约 12.405–13.269 FPS。
- 小时级与天级稳定性主要是作者定性展示；LongForcing 的受控定量消融长度为 60 秒。
