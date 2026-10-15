# Sources

> OASIS 没有正式论文或技术报告 PDF；本笔记依据 Decart × Etched 官方研究项目页与官方开源仓库撰写，因此不创建 `sources/paper.pdf`。

| 本地文件 | 材料类型 | 原始来源 | 页面位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `figures/01-architecture.png` | 官方架构图 | https://oasis-model.github.io/arch_new.png | Architecture | 2026-09-04 | 说明 ViT-VAE + DiT 与动作条件 |
| `figures/02-dynamic-noising.png` | 官方方法图 | https://oasis-model.github.io/dyno.png | Architecture / Dynamic Noising | 2026-09-04 | 说明长序列动态加噪 |
| `figures/03-performance.png` | 官方性能图 | https://oasis-model.github.io/speed.png | Performance | 2026-09-04 | 展示官方 20 FPS 主张 |
| `figures/04-limited-memory.webp` | 官方失败案例 | https://oasis-model.github.io/3_second_memory.webp | Future Explorations | 2026-09-04 | 展示长时记忆局限 |

## 官方链接

- 研究项目页：https://oasis-model.github.io/
- 开源仓库：https://github.com/etched-ai/open-oasis
- 500M 权重：https://huggingface.co/Etched/oasis-500m
- Decart 工程说明：https://www.decart.ai/articles/oasis-interactive-ai-video-game-model

## 证据边界

- 官方项目页公开的是研究展示而非同行评审论文，没有完整训练配方、训练数据规模、正式 benchmark 表或消融实验。
- 开源仓库提供 500M 下缩版模型权重与推理代码；官方 20 FPS 演示使用更大的未完整公开 checkpoint 与 Decart 推理栈，不能直接等同于开源 500M 本地性能。
- `01-architecture.png` 原图带透明背景，已铺成白底以保证 Obsidian 浅色主题可读，图中文字与内容未修改。
