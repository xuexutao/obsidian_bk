# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 / 版本 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 官方论文 PDF | https://arxiv.org/pdf/2610.09450 | arXiv:2610.09450v1，19 页，含附录 | 2026-10-29 | 精读主来源 |
| `figures/01-teaser.png` | 文生图样例 | https://arxiv.org/pdf/2610.09450 | Figure 1，p.1 | 2026-10-29 | 展示约一百万像素、多宽高比的像素空间生成效果 |
| `figures/02-method-architecture.png` | 方法架构图 | https://arxiv.org/pdf/2610.09450 | Figure 4，p.5 | 2026-10-29 | 解释 8 个双流块、16 个单流块和 4 层 PiT 像素头 |
| `figures/03-depth-results.png` | 深度定性结果 | https://arxiv.org/pdf/2610.09450 | Figure 7，p.8 | 2026-10-29 | 比较 latent Klein、pixel Klein 与 Iris-3B 的深度迁移表现 |
| `figures/04-restoration-results.png` | 图像恢复结果 | https://arxiv.org/pdf/2610.09450 | Figure 8，p.10 | 2026-10-29 | 比较 4× 恢复的局部纹理与保真度 |
| `sources/config.yaml` | 官方基础模型配置 | https://huggingface.co/speridlabs/iris-3b/resolve/main/config.yaml | Hugging Face release config | 2026-10-29 | 核对层数、宽度、patch、文本编码器和 flow 配置 |
| `sources/turbo-distill.yaml` | 官方 Turbo 采样配置 | https://huggingface.co/speridlabs/iris-3b/resolve/main/turbo/distill.yaml | 4-step release schedule | 2026-10-29 | 核对 Turbo 的固定时间步 |

## 官方在线材料

| 材料 | URL | 获取日期 | 用途 |
|---|---|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2610.09450 | 2026-10-29 | 论文元数据 |
| arXiv HTML | https://arxiv.org/html/2610.09450 | 2026-10-29 | 章节、公式与表格交叉核对 |
| 官方项目页 | https://speridlabs.com/research/iris | 2026-10-29 | 生成样例、深度、恢复与 Turbo 更新 |
| 官方代码仓库 | https://github.com/speridlabs/iris-3b | 2026-10-29 | 代码级实现分析 |
| 官方模型仓库 | https://huggingface.co/speridlabs/iris-3b | 2026-10-29 | 权重、配置、推理命令与模型卡 |
| 官方在线演示 | https://huggingface.co/spaces/speridlabs/iris-3b | 2026-10-29 | 基础版生成、深度与超分演示 |

## 代码版本与阅读范围

- 官方仓库 commit：`13fcf6a4f7fcb36561eac3daa5ad30d0b0a1822e`（提交时间：2026-10-15T09:33:19+02:00）。
- 许可证：Apache-2.0；冻结文本编码器 `Qwen/Qwen3-VL-4B-Instruct` 也采用 Apache-2.0。
- 主要核查文件：
  - `src/iris3b/models/dit.py`
  - `src/iris3b/models/blocks/mmdit.py`
  - `src/iris3b/models/blocks/single_stream.py`
  - `src/iris3b/models/blocks/pit.py`
  - `src/iris3b/nn/attention.py`
  - `src/iris3b/nn/embeddings.py`
  - `src/iris3b/text/qwen3_vl.py`
  - `src/iris3b/flow/{transport,schedule,solver}.py`
  - `src/iris3b/repa.py`
  - `src/iris3b/downstream/{depth,restoration}.py`
  - `configs/iris3b/{stage1_256,stage2_512,stage3_1024,sft_1024,turbo_stage1_6step,turbo_stage2_4step}.yaml`
  - `scripts/{sample,train,depth,upscale,distill}.py`
- 代码分析由只读仓库检索完成，关键形状、参数量、训练阶段与论文差异已回读源码行号；没有修改仓库。
- 未发现单独 Supplementary PDF；19 页官方 PDF 已包含附录 A–C。
