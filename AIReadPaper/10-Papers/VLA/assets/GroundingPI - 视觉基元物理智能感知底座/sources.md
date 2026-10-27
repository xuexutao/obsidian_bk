# GroundingPI 材料来源与开源边界

- 获取与核对日期：2026-10-27
- 论文：GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives
- arXiv：2609.39601v2
- 官方代码 commit：`090932430bbbeea004cfa461767ede7f563c0f3a`
- Hugging Face revision：`08d397c8769f7f10b96ff6d56ee6f56acb04181f`
- 许可证口径：**Apache-2.0 + Kimi K3 派生条件**
- 本文件只记录静态阅读与核对来源；未运行模型、训练或完整评测。

## 1. 官方入口

| 类型 | 官方地址 | 用途 |
|---|---|---|
| 论文页面 | https://arxiv.org/abs/2609.39601 | 摘要、版本、作者、论文元信息 |
| 官方 PDF v2 | https://arxiv.org/pdf/2609.39601v2 | 方法、实验、附录、图表和页码的一手来源 |
| 项目页 | https://groundingpi.github.io/ | 项目总览、演示和官方资源入口 |
| GitHub | https://github.com/groundingpi/GroundingPI | 代码、配置、文档、许可证和第三方声明 |
| Hugging Face | https://huggingface.co/GroundingPI/GroundingPI | 模型权重、processor、tokenizer、custom code 与模型卡 |
| ModelScope | https://www.modelscope.ai/models/GroundingPI/GroundingPI | 官方模型镜像 |
| HF Demo | https://huggingface.co/spaces/GroundingPI/GroundingPI | 官方在线演示入口 |
| 34 项评测数据 | https://huggingface.co/datasets/Skywalker0410/Grounding-EvalData | Grounding-EvalData bundle |

GitHub 固定版本：

```text
090932430bbbeea004cfa461767ede7f563c0f3a
```

Hugging Face 固定 revision：

```text
08d397c8769f7f10b96ff6d56ee6f56acb04181f
```

## 2. 本地论文材料

| 本地文件 | 来源 | 说明 |
|---|---|---|
| `/Users/eilonxtxue/WorkBuddy/2026-10-27-19-36-47/groundingpi-staging/source.pdf` | arXiv PDF v2 | 官方 PDF，一手版面与页码依据 |
| `/Users/eilonxtxue/WorkBuddy/2026-10-27-19-36-47/groundingpi-staging/source.md` | 官方 PDF 文本抽取 | 用于定位方法、训练、表格、附录和精确数字 |
| `/tmp/groundingpi-code-20261027` | 官方 GitHub | 固定在上述 commit 的代码快照 |

论文抽取首页记录的 PDF 来源：

```text
https://arxiv.org/pdf/2609.39601v2
```

## 3. 八张论文图的来源映射

所有本地图均从官方论文 PDF 对应 Figure 裁取，用于 Obsidian 离线引用。

| 本地文件 | 论文图号 | PDF 文件页 | 官方来源 | 内容 |
|---|---:|---:|---|---|
| `figures/01-teaser.png` | Figure 1 | 1 | arXiv PDF v2；仓库 `docs/assets/teaser.png` | 感知能力、34 项评测与物理智能迁移总览 |
| `figures/02-architecture.png` | Figure 2 | 3 | arXiv PDF v2；HF `assets/fig2-architecture.png` | MoonViT-V2、2×2 merge、projector、Qwen3 与 GAM |
| `figures/03-data-engine.png` | Figure 3 | 3 | arXiv PDF v2 | candidate、field fusion、validation、on-demand evidence 与 iteration |
| `figures/04-training-pipeline.png` | Figure 4 | 6 | arXiv PDF v2 | token CE SFT 与 $G=8$ GRPO |
| `figures/05-capability-wheel.png` | Figure 5 | 8 | arXiv PDF v2；HF `assets/fig5-grounding-performance.png` | 11 类感知能力与领先 baseline |
| `figures/06-physical-transfer.png` | Figure 6 | 11 | arXiv PDF v2；HF `assets/fig6-physical-intelligence.png` | RoboTwin、RoboCasa 与 nuScenes |
| `figures/07-scaling.png` | Figure 9 | 14 | arXiv PDF v2 | 88.4B、132.6B、176.8B、221B scaling |
| `figures/08-data-composition.png` | Figure 10 | 15 | arXiv PDF v2 | A–F task group、组合结果和 leave-one-out |

仓库 README 中三张 HF 图片使用的官方固定资源 revision 为：

```text
afeca16451e4ad4aec9ebbe91fc3f63f8bfa5c49
```

对应官方资源 URL：

```text
https://huggingface.co/GroundingPI/GroundingPI/resolve/afeca16451e4ad4aec9ebbe91fc3f63f8bfa5c49/assets/fig2-architecture.png
https://huggingface.co/GroundingPI/GroundingPI/resolve/afeca16451e4ad4aec9ebbe91fc3f63f8bfa5c49/assets/fig5-grounding-performance.png
https://huggingface.co/GroundingPI/GroundingPI/resolve/afeca16451e4ad4aec9ebbe91fc3f63f8bfa5c49/assets/fig6-physical-intelligence.png
```

Figure 1 的仓库来源为固定代码 commit 下的：

```text
docs/assets/teaser.png
```

Figure 3、4、9、10 以官方 PDF 对应页面作为 provenance；未将第三方转绘图作为来源。

## 4. 已保存的六个 Hugging Face 配置/API 文件

本地目录：

```text
/Users/eilonxtxue/Documents/hexo/obsidian_bk/AIReadPaper/10-Papers/VLA/assets/GroundingPI - 视觉基元物理智能感知底座/sources/
```

### 4.1 `config.json`

本地文件：`sources/config.json`

原始 URL：

```text
https://huggingface.co/GroundingPI/GroundingPI/resolve/08d397c8769f7f10b96ff6d56ee6f56acb04181f/config.json
```

已核对字段：

- 视觉层数 27；
- 视觉 hidden 1024；
- patch size 14；
- `merge_kernel_size=[2,2]`；
- `merge_type="sd2_tpool"`；
- projector hidden 4096；
- projector output 2560；
- Qwen3 层数 36；
- Qwen3 hidden 2560；
- 32 query heads、8 KV heads；
- vocab size 152670。

### 4.2 `generation_config.json`

本地文件：`sources/generation_config.json`

原始 URL：

```text
https://huggingface.co/GroundingPI/GroundingPI/resolve/08d397c8769f7f10b96ff6d56ee6f56acb04181f/generation_config.json
```

已核对字段：

- `use_cache=true`；
- EOS token 包含 151645 与 151643。

### 4.3 `preprocessor_config.json`

本地文件：`sources/preprocessor_config.json`

原始 URL：

```text
https://huggingface.co/GroundingPI/GroundingPI/resolve/08d397c8769f7f10b96ff6d56ee6f56acb04181f/preprocessor_config.json
```

已核对字段：

- `in_patch_limit=4096`；
- `patch_size=14`；
- `merge_kernel_size=2`；
- `patch_limit_on_one_side=512`。

### 4.4 `model.safetensors.index.json`

本地文件：`sources/model.safetensors.index.json`

原始 URL：

```text
https://huggingface.co/GroundingPI/GroundingPI/resolve/08d397c8769f7f10b96ff6d56ee6f56acb04181f/model.safetensors.index.json
```

已核对 metadata：

```json
{
  "total_parameters": 4843664384,
  "total_size": 9687328768
}
```

该文件是完整参数量 4,843,664,384 与 BF16 权重字节数 9,687,328,768 的直接依据。

### 4.5 `hf-model-api.json`

本地文件：`sources/hf-model-api.json`

原始 API：

```text
https://huggingface.co/api/models/GroundingPI/GroundingPI/revision/08d397c8769f7f10b96ff6d56ee6f56acb04181f
```

已核对：

- model ID `GroundingPI/GroundingPI`；
- revision `08d397c8769f7f10b96ff6d56ee6f56acb04181f`；
- pipeline tag `image-text-to-text`；
- safetensors 参数量；
- 模型包许可证标签与文件列表摘要。

### 4.6 `hf-model-tree.json`

本地文件：`sources/hf-model-tree.json`

原始 API：

```text
https://huggingface.co/api/models/GroundingPI/GroundingPI/tree/08d397c8769f7f10b96ff6d56ee6f56acb04181f?recursive=true&expand=true
```

已核对：

- HF revision 下的文件树；
- 文件大小与 object ID；
- 资源图片与权重文件；
- 各文件最近提交记录。

## 5. 实际阅读过的论文区段

| 论文位置 | 主要内容 |
|---|---|
| Main §3.1 | MoonViT-V2、projector、Qwen3、共享词表 AR |
| Main §3.2 | data engine 总述 |
| Main §3.3 | Base/SFT/GRPO 总述 |
| Main §4 | driving 与 manipulation 接口 |
| Main §5.1 | 34 benchmark、baseline 与 metric |
| Main Table 1–4 | 感知主结果 |
| Main §5.2 | nuScenes、RoboTwin、RoboCasa、data efficiency |
| Main §5.3 | scaling、composition、讨论与消融 |
| Appendix A.1 | GAM prompt/output grammar、坐标约定 |
| Appendix A.2 | architecture、tokenizer、参数量、visual processing |
| Appendix A.3 | data engine 字段融合、验证、迭代 |
| Appendix A.4–A.5 | Pretrain 1 Stage 1–3、Pretrain 2/Stage 4 |
| Appendix A.6 | GRPO、$R_{set}$、$R_{strict}$、OCR Hungarian reward |
| Appendix B.1 | nuScenes trajectory serialization 与 open-loop metric |
| Appendix B.2 | 8-depth interface、Fixed-16 ActionDiT、训练预算 |
| Appendix C.1–C.5 | data efficiency、scaling、composition、OCR、token cost |
| Appendix D | architecture mechanism discussion |
| Appendix E | limitations |
| Appendix F | 34 项完整结果与 11 类 aggregate 口径 |

## 6. 实际阅读过的核心源码

代码根目录：`/tmp/groundingpi-code-20261027`

### 6.1 模型与视觉处理

- `models/vlm/modeling_groundingpi_vision.py:717-721`
  - `sd2_tpool` merge 路径。
- `models/vlm/modeling_groundingpi_vision.py:791-823`
  - `PatchMergerMLPV2`；四 patch flatten；`4096→4096→2560`；GELU；RMSNorm。
- `models/vlm/modeling_groundingpi_vision.py:910-927`
  - vision tower、projector、language model 的构造。
- `models/vlm/modeling_groundingpi.py:1310-1318`
  - `masked_scatter` 将 image embedding 写入文本序列。
- `models/vlm/modeling_groundingpi.py:1330-1357`
  - 一维 position ID 与 Qwen3 language model forward。
- `models/vlm/modeling_groundingpi.py:1369-1378`
  - `2560→152670` untied LM head。
- `models/vlm/modeling_groundingpi.py:1537-1547`
  - logits 与 causal LM loss。
- `models/vlm/modeling_groundingpi.py:1607-1616`
  - prefill 后丢弃 image tensor，避免重复视觉 forward。
- `models/vlm/modeling_groundingpi.py:1471-1481`
  - 公开视频处理路径的限制说明。
- `models/vlm/image_processing_groundingpi.py:34-44`
  - patch、merge 与 `IMAGE_MAX_TOKEN_NUM`。
- `models/vlm/image_processing_groundingpi.py:75-95`
  - 动态 resize 与 output token 上限。
- `models/vlm/image_processing_groundingpi.py:148-184`
  - resize、pad、normalize、patchify 与返回 tensor。
- `models/vlm/media_utils.py:319-341`
  - NaViT patchify 的精确形状。
- `models/vlm/processing_groundingpi.py:16-84`
  - tokenizer 与 image processor glue。

### 6.2 GAM 与 tokenizer

- `grounding_pi/protocol.py:1-18`
  - canonical grammar。
- `grounding_pi/protocol.py:30-42`
  - 0–999 网格与 special token 常量。
- `grounding_pi/protocol.py:121-179`
  - task/prompt/data route 枚举。
- `grounding_pi/protocol.py:213-327`
  - token、phrase safety、几何合法性与排序。
- `grounding_pi/protocol.py:370-400`
  - canonical serializer。
- `grounding_pi/protocol.py:475-545`
  - strict parser 与 validator。
- `grounding_pi/protocol.py:804-843`
  - 1000→999 坐标 remap。
- `train/tokenizer/extend_tokens.py:31-50`
  - 词表大小、矩阵形状与固定 token ID。
- `train/tokenizer/extend_tokens.py:147-166`
  - 新增 1000 坐标 token 与 `</c>`。
- `train/tokenizer/extend_tokens.py:220-233`
  - input/output embedding 独立扩容与初始化。

### 6.3 SFT 与服务

- `train/vlm/plugin.py:76-111`
  - GroundingPI model/template 注册、image-only media 与一维 position ID。
- `train/vlm/plugin.py:113-153`
  - packing 与 collator 的 Qwen3 position 修正。
- `train/vlm/plugin.py:156-210`
  - language/aligner/vision 模块路径和保存文件。
- `configs/train/vlm.yaml`
  - 公开默认 SFT recipe：BF16、2048、batch 2、world size 1。
- `configs/release/vlm_train.yaml`
  - 单节点单进程 launcher 与 `IMAGE_MAX_TOKEN_NUM=1024`。
- `infer/serve_vlm.py:163-169`
  - 根据 grid 与 merge 倍数扩展 image placeholder。
- `infer/serve_vllm.py:13-36`
  - serving overlay 的 source/weight/manifest 校验。
- `infer/serve_vllm.py:39-90`
  - vLLM 0.18、Transformers 5.7 fork、BF16、eager、TP=1、16K context。
- `grounding_pi/client.py:87-91`
  - 强制保留 special token 与相邻坐标 token。

### 6.4 评测与物理智能

- `configs/eval/suites.json:34-69`
  - `groundingpi34` 的 34 个 task entry。
- `eval/README.md:78-110`
  - 42 recipe 与论文 34 entry 的区别、full-suite 配置。
- `eval/README.md:157-170`
  - per-task output；不自动计算论文 aggregate；单次运行不复现十次均值。
- `eval/eval_runner.py:163-166`
  - GAM mode 的 special-token decode contract。
- `eval/OCR/utils.py:511-517`
  - OCR GAM entry regex。
- `eval/OCR/utils.py:586-734`
  - OCR 恢复与严格解析。
- `eval/OCR/utils.py:1238-1240`
  - parse error 的判定范围。
- `eval/OCR/utils.py:1340-1346`
  - F1mIoU 与 parse error 聚合。
- `vla/openwam/README.md:1-13`
  - Fixed-16 π-style ActionDiT、8-depth normalized tap。
- `vla/openwam/README.md:154-157`
  - action expert、conditioner 和 architecture 入口。
- `README.md:350-357`
  - StarVLA/OpenWAM workflow 与“没有 GroundingPI 直接 policy adapter”的边界。

### 6.5 仓库、许可证与第三方来源

- `README.md:15-20`
  - arXiv、HF、ModelScope、项目页、GitHub 官方入口。
- `README.md:76-80`
  - GroundingPI checkpoint 与 GAM mode。
- `README.md:214-216`
  - vLLM 默认与 decode 参数。
- `README.md:260-268`
  - GAM 示例。
- `README.md:287-308`
  - 34 benchmark 与 full evaluation。
- `README.md:312-344`
  - 公开 SFT workflow。
- `README.md:403-411`
  - Apache-2.0 与 Kimi K3 派生条件。
- `THIRD_PARTY_NOTICES.md:7-13`
  - bundled ms-swift、Transformers、lmms-eval revision 与许可证。
- `THIRD_PARTY_NOTICES.md:25-31`
  - Kimi K3 派生代码与许可证边界。
- `LICENSE`
  - 仓库原创贡献的 Apache License 2.0 正文。

## 7. 开源覆盖边界

### 7.1 已公开

- 官方代码仓库；
- GroundingPI grounding VLM 权重；
- tokenizer 与 1000 个坐标 token；
- image processor 与 custom model code；
- 原生 Transformers 推理；
- vLLM OpenAI-compatible 单图服务；
- batch annotation client；
- release SFT recipe 与 ms-swift plugin；
- 34 项评测 task list、parser、metric 与执行流程；
- Grounding-EvalData 下载入口；
- StarVLA/OpenWAM backbone comparison workflow；
- 许可证和第三方来源声明。

### 7.2 未完整公开

- 500.41B token 的 Pretrain 1 数据与完整训练编排；
- 221.17B token 的空间训练数据；
- 论文生产级 data engine；
- teacher ensemble、字段验证和人工审核的完整实现；
- GroundingPI 专属 GRPO 训练代码与 reward pipeline；
- 完整训练语料、样本 provenance、去重与 benchmark 泄漏审计；
- 论文最终 GroundingPI action-policy checkpoint；
- 可把 HF GroundingPI checkpoint 直接加载成最终 policy 的 adapter；
- 论文 10-run 结果的一键统计聚合。

### 7.3 最终开源状态表述

**官方代码、推理、SFT、权重和34项评测已开源，但多阶段预训练、数据引擎、GroundingPI专属GRPO与下游GroundingPI policy未完整开源。**

## 8. 许可证说明

仓库原创贡献由根目录 Apache License 2.0 覆盖。

但 `models/vlm/` 含 Kimi K3 派生实现，适用 derivative work 继续受上游 Kimi K3 License 条件约束。

因此不能只写“Apache-2.0”。

本笔记采用完整表述：

```text
Apache-2.0 + Kimi K3 派生条件
```

具体依据：

- `README.md:403-411`；
- `THIRD_PARTY_NOTICES.md:25-31`；
- `models/vlm/LICENSE`；
- HF 模型包 `LICENSE`。

## 9. 核对限制

- 本次为论文、图片、配置和源码的静态阅读。
- 未运行模型推理。
- 未执行 SFT、预训练或 GRPO。
- 未下载并运行 34 项评测数据。
- 未复现 RoboTwin、RoboCasa 或 nuScenes 结果。
- 页面号按官方 PDF 文件页记录。
- 表格数字以论文主表与附录为准。
- 对 44/45 baseline 的图文冲突保持原样记录，不自行猜测计数修正。
