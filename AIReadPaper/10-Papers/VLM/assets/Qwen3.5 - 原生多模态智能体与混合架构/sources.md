# Qwen3.5 - 原生多模态智能体与混合架构 · Sources

> 官方正式名称为 Qwen3.5；截至 2026-10-20 没有独立的 `Qwen3.5-VL` 型号或 arXiv 技术报告。Qwen3.5 本身即原生视觉语言模型。

| 本地文件 | 材料类型 | 原始来源 | 位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/config-9b.json` | 官方模型配置 | <https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json> | 完整配置 | 2026-10-20 | 核验 GDN/full attention、视觉塔、MRoPE、上下文与 MTP |
| `sources/preprocessor-config.json` | 官方视觉预处理配置 | <https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/preprocessor_config.json> | 完整配置 | 2026-10-20 | 核验 patch、temporal patch、merge 与像素预算 |
| `sources/model-card-397b.md` | 官方旗舰模型卡 | <https://huggingface.co/Qwen/Qwen3.5-397B-A17B> | README | 2026-10-20 | 旗舰架构、基准、部署与引用 |
| `sources/model-card-9b.md` | 官方 9B 模型卡 | <https://huggingface.co/Qwen/Qwen3.5-9B> | README | 2026-10-20 | 中型模型架构、视觉基准与推理配置 |
| `figures/00-official-release.png` | 官方发布页截图 | <https://qwen.ai/blog?id=qwen3.5> | 页面顶部 | 2026-10-20 | 正式命名和产品定位 |
| `figures/01-grounding-demo.png` | 官方视觉输入 | <https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/cotground/vstar_180.png> | Visual grounding Demo | 2026-10-20 | 视觉定位示例输入 |
| `figures/02-visual-reasoning-process.png` | 官方视觉推理中间图 | <https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/Thinking_with_image/0e5fa726-72e2-4b96-9709-6aa7270a9983.png> | Thinking with image Demo | 2026-10-20 | 图像 mask 与中间结构化 |
| `figures/03-visual-reasoning-result.png` | 官方视觉推理结果 | <https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/Thinking_with_image/a2529aad-b91c-4992-bfe5-0903d2b4ef33.png> | Thinking with image Demo | 2026-10-20 | 代码工具求解迷宫 |
| `figures/04-mathvision-demo.jpg` | 官方视觉数学输入 | <https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg> | 模型卡 image input 示例 | 2026-10-20 | 几何视觉推理 |
| `figures/05-vision-benchmark.png` | 官方视觉 benchmark 截图 | <https://qwen.ai/blog?id=qwen3.5> | “视觉语言”章节 | 2026-10-20 | 旗舰视觉评测 |
| `figures/06-infrastructure.png` | 官方基础设施截图 | <https://qwen.ai/blog?id=qwen3.5> | “基础设施”章节 | 2026-10-20 | FP8、多模态训练与异步 RL |

## 其它官方来源

- Qwen3.5 官方发布：<https://qwen.ai/blog?id=qwen3.5>
- Qwen3.5 官方模型集合：<https://huggingface.co/collections/Qwen/qwen35>
- Qwen3.5-27B：<https://huggingface.co/Qwen/Qwen3.5-27B>
- Qwen3.5-35B-A3B：<https://huggingface.co/Qwen/Qwen3.5-35B-A3B>
- Transformers 架构文档：<https://huggingface.co/docs/transformers/main/model_doc/qwen3_5>
- 官方 GitHub：<https://github.com/QwenLM/Qwen3.5>

## 取材边界

- 无独立论文 PDF，因此没有 `sources/paper.pdf`。
- 官方博客未披露完整训练数据、训练超参数、RL 算法与受控消融；正文中均明确标注这一限制。
- 7 张本地图均已在正文中引用并经预览核验。
