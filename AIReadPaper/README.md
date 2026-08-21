# AIReadPaper 写入规范

> 本目录用于沉淀单篇论文的详细精读笔记与多篇论文领域综述。任何人、Skill 或 Agent 在这里写入论文内容时，都必须遵守本文件。

## 最高优先级规则

**先在 `Paper-Index.md` 注册，再从官方论文源取材，最后创建或更新论文笔记。**

禁止先生成笔记、最后再补索引。若索引尚未登记，正文写作不得开始。单篇论文笔记必须基于论文原文，而不是只根据二手文章、摘要或项目宣传材料撰写。

## 1. 强制执行流程

1. 打开 `Paper-Index.md`。
2. 使用论文正式名称、论文短名、arXiv ID、DOI、原始链接和已有 `[[Obsidian 内链]]` 查重。
3. 判断所属领域：VLM、视频生成、3D生成、三维重建、世界模型、VLA、基础模块；无法判断时放入“待分类”。
4. 在对应领域表格先新增一行：
   - 时间：当前日期，格式为 `YYYY-MM-DD`；
   - 文章题目：外部文章标题；若直接阅读论文，可填写论文短名或中文标题；
   - 论文名称：论文正式英文名称；
   - 文章简要总结：写作前允许填写“待完成”；
   - 原始链接：优先填写 arXiv、出版社论文页或作者项目页；
   - 阅读总结链接：预先填写 `[[计划使用的笔记名]]`。
5. 注册成功后，从官方或作者授权的论文源获取原文和支持材料：
   - 下载论文 PDF；
   - 下载可公开获取的 Supplementary Material；
   - 查阅项目页、官方代码仓库和作者公开资料；
   - 提取并保存方法图、总体架构图、关键实验图和消融图。
6. 阅读论文正文与补充材料后，按照本文件的详细精读结构创建或更新论文笔记。
7. 将关键图片插入对应章节，并为每张图片写明图号、内容解读和来源。
8. 完成笔记后回到 `Paper-Index.md`：
   - 将“待完成”替换为最终简要总结；
   - 在总结末尾加入 `★` 评级；
   - 确认阅读总结内链能够解析到实际笔记。
9. 最后检查正文、公式、图片、来源记录、相对附件路径和索引链接。

如果索引中已经存在同一论文，应优先更新原笔记，不得通过增加“精读”“新版”“Reading Note”或相近文件名重复创建。

## 2. 目标目录结构

```text
AIReadPaper/
├── Paper-Index.md
├── README.md
├── 10-Papers/
│   ├── VLM/
│   ├── Video-Generation/
│   ├── 3D-Generation/
│   ├── 3D-Reconstruction/
│   ├── World-Models/
│   ├── VLA/
│   ├── Foundations/
│   └── Inbox/
├── 20-Surveys/
│   ├── VLM/
│   ├── Video-Generation/
│   ├── 3D-Generation/
│   ├── 3D-Reconstruction/
│   ├── World-Models/
│   ├── VLA/
│   └── Foundations/
├── 90-Archive/
└── 99-Templates/
```

当前目录已按上述目标结构完成迁移。后续不得重新创建 `bd/子/`、`AI/`、`3DGeneration/` 等旧分类，也不得自行建立平行目录；`Paper-Index.md` 始终是唯一登记入口。

## 3. 目录与领域映射

| 索引领域 | 单篇论文目录 | 领域综述目录 |
|---|---|---|
| VLM | `10-Papers/VLM/` | `20-Surveys/VLM/` |
| 视频生成 | `10-Papers/Video-Generation/` | `20-Surveys/Video-Generation/` |
| 3D生成 | `10-Papers/3D-Generation/` | `20-Surveys/3D-Generation/` |
| 三维重建 | `10-Papers/3D-Reconstruction/` | `20-Surveys/3D-Reconstruction/` |
| 世界模型 | `10-Papers/World-Models/` | `20-Surveys/World-Models/` |
| VLA | `10-Papers/VLA/` | `20-Surveys/VLA/` |
| 基础模块 | `10-Papers/Foundations/` | `20-Surveys/Foundations/` |
| 待分类 | `10-Papers/Inbox/` | 暂不创建综述 |

## 4. 文件命名

### 4.1 单篇论文

```text
{论文短名} - {中文主题}.md
```

示例：

```text
Pixal3D - 像素对齐三维生成.md
TRELLIS - 结构化潜变量3D生成.md
VGGT-Ω - 通用三维几何建模.md
DreamZero - 零样本机器人策略.md
```

文件名中不追加以下状态词：

- 阅读总结
- 论文阅读总结
- 精读
- Reading Note
- Reading Notes
- Final
- 最新版

写作状态应记录在 frontmatter 的 `status` 中，而不是反复改文件名。

### 4.2 领域综述

```text
{主题} - Survey.md
```

示例：

```text
Feed-Forward 3D Reconstruction - Survey.md
3D Asset Generation - Survey.md
```

领域综述涉及的论文也必须先在 `Paper-Index.md` 逐篇登记。若某篇论文没有独立阅读笔记，可以暂时链接到该综述；以后补写独立笔记时再更新链接。

## 5. 论文源与支持材料规范

### 5.1 来源优先级

取材优先使用以下来源，顺序从高到低：

1. arXiv 官方论文页与官方 PDF；
2. 出版社或会议官方论文页与 PDF；
3. 作者项目页与 Supplementary Material；
4. 作者或研究机构的官方代码仓库；
5. 作者公开的演讲、海报、视频和技术说明；
6. 可信的二手解读，仅用于补充背景，不得替代论文原文。

不得绕过付费墙、登录限制或其他访问控制。遇到无法合法下载的材料时，在 `sources.md` 中保留原始链接和无法获取的说明，不得伪造本地材料。

### 5.2 每篇论文必须收集的材料

对于公开可获取的论文，至少保存：

- 官方论文 PDF：`sources/paper.pdf`；
- 来源清单：`sources.md`；
- 关键论文图 3 张，优先顺序为：
  1. Fig.1 或 teaser / motivation 图，用于解释“为什么做”；
  2. method pipeline / architecture 图，用于解释“怎么做”；
  3. 关键定性结果、定量对比或 ablation 图，用于解释“是否有效”。

若论文没有 3 张合适图片，应尽可能保存现有关键图，并在笔记中说明缺失原因。不得为了满足数量使用无关图片。

有公开材料时，还应按需保存：

- `sources/supplementary.pdf`：补充材料；
- 项目页中的高分辨率结果图、交互示意或视频封面；
- 官方代码仓库中有助于理解方法的模型结构图、配置说明或数据流程图；
- Benchmark、数据集、训练配置和复现说明；
- 对理解论文结论有直接帮助的作者演讲或海报链接。

不要复制整个项目网站或整个代码仓库到附件目录。只保留与笔记直接相关、来源明确的支持材料，其余内容在 `sources.md` 中记录链接即可。

### 5.3 图片提取原则

论文中的方法图通常由矢量元素和多张小位图组成。**禁止直接使用 PDF 中抽出的零散图片碎片充当完整方法图。**

正确流程：

1. 下载官方 PDF 到 `sources/paper.pdf`；
2. 将目标页临时渲染为高清整页图，用于定位 Fig.1、Fig.2 和关键实验图；
3. 根据图注位置从 PDF 精确裁剪完整图表，推荐使用 300 DPI；
4. 裁剪时尽量保留完整图像和图注，不能截掉标题、坐标轴、图例或边缘；
5. 用图像预览复核清晰度和完整性，不合格则重新裁剪；
6. 只把最终采用的图片放入 `figures/`，删除整页渲染图和未引用的碎片图。

已有工具可用于定位和裁剪论文图：

```text
research-field-survey/scripts/scan_figures.py
research-field-survey/scripts/extract_figure.py
```

若 arXiv HTML 或项目页提供官方高分辨率 SVG、PNG 或 WebP，可以优先下载官方原图；仍需在 `sources.md` 记录来源 URL。

## 6. 详细论文阅读笔记规范

单篇论文笔记必须是“详细精读”，不能只复述摘要或列出若干 bullet points。方法与实验分析原则上应占正文的 60% 以上。

建议结构如下：

```markdown
# 论文短名：中文主题

## 基本信息

## 一句话结论

---

## 1. 研究背景与问题定义

### 1.1 研究问题
### 1.2 现有方法的瓶颈
### 1.3 本文核心贡献

---

## 2. 任务定义与输入输出

### 2.1 输入、输出与假设
### 2.2 关键符号和目标函数

---

## 3. 核心方法

### 3.1 总体框架
### 3.2 关键模块一
### 3.3 关键模块二
### 3.4 训练目标与损失函数
### 3.5 推理流程与复杂度

---

## 4. 数据集与实验设置

### 4.1 数据集与数据处理
### 4.2 Baseline 与评价指标
### 4.3 实现细节

---

## 5. 实验结果

### 5.1 主要定量结果
### 5.2 定性结果
### 5.3 消融实验
### 5.4 泛化、效率与失败案例

---

## 6. 与相关工作的关系

---

## 7. 局限与批判性评价

---

## 8. 复现与实践建议

---

## 9. 个人启发与后续问题

---

## 10. 材料来源
```

写作要求：

- “基本信息”可不编号，其余章节使用层级编号；
- 标题不得跳级：`## 1.` → `### 1.1` → `#### 1.1.1`；
- 中文散文式行文为主，避免把全文写成碎片化短句列表；
- 清楚解释任务的输入、输出、假设、核心表示和完整 pipeline；
- 关键公式要说明变量含义、优化目标和直觉，不能只复制公式；
- 实验部分必须解释主要表格说明了什么，不能只罗列数字；
- 区分论文明确结论、作者解释和个人判断；个人推断应明确标注；
- 数学公式使用 Obsidian 语法：行内 `$内容$`，块级 `$$内容$$`；
- 至少插入 3 张关键图；图片应放在相关章节附近，而不是集中堆到文末；
- 每张图片下方必须写一句以上的中文解读，并注明来源；
- 必须讨论方法价值、适用边界、失败案例、局限和个人判断；
- 对复现成本、数据需求、训练资源、推理速度和代码可用性进行评估；
- 信息缺失时如实标注“论文未披露”或“暂未找到”，不得补造事实。

### 6.1 图片引用示例

```markdown
![](assets/TRELLIS%20-%20结构化潜变量3D生成/figures/02-method-pipeline.png)

> 图 2：TRELLIS 的总体生成流程。该图说明……  
> 来源：论文 Figure 2，第 4 页，https://arxiv.org/abs/xxxx.xxxxx
```

图号应尽量对应论文原始 Figure 编号；本地文件名前添加顺序编号只是为了排序，不应改变论文原图号的含义。

## 7. Frontmatter 建议

```yaml
type: paper-note
status: draft
domain: 3D-Generation
paper: 论文正式英文名称
year: 2026
arxiv: xxxx.xxxxx
doi: null
source: https://arxiv.org/abs/xxxx.xxxxx
project: null
code: null
tags:
  - 3D-Generation
created: YYYY-MM-DD
updated: YYYY-MM-DD
```

领域综述将 `type` 改为 `survey`。没有 DOI、项目页或代码仓库时保留 `null`，不要编造链接。

## 8. 附件与溯源规范

### 8.1 目录结构

单篇论文的本地材料统一放在同领域的附件目录：

```text
10-Papers/<Domain>/
├── <笔记名>.md
└── assets/
    └── <笔记名>/
        ├── figures/
        │   ├── 01-teaser.png
        │   ├── 02-method-pipeline.png
        │   └── 03-key-results.png
        ├── sources/
        │   ├── paper.pdf
        │   └── supplementary.pdf
        └── sources.md
```

所有新增论文材料都应直接写入新结构下的 `assets/<笔记名>/`，内部使用 `figures/`、`sources/` 和 `sources.md`。禁止再向旧目录写入内容。

### 8.2 来源清单

每篇论文的 `sources.md` 至少记录：

```markdown
# Sources

| 本地文件 | 材料类型 | 原始来源 | 论文位置 | 获取日期 | 用途 |
|---|---|---|---|---|---|
| `sources/paper.pdf` | 论文 PDF | arXiv URL | 全文 | YYYY-MM-DD | 精读主来源 |
| `figures/02-method-pipeline.png` | 方法图 | arXiv URL | Figure 2, p.4 | YYYY-MM-DD | 核心方法章节 |
```

所有本地图片都必须能追溯到原始 URL、论文 Figure 编号或页面位置。项目页、代码仓库和演讲材料也应记录标题、URL、获取日期和用途。

### 8.3 文件命名

```text
01-teaser.png
02-method-pipeline.png
03-key-results.png
04-ablation.png
05-failure-cases.png
```

文件名使用小写英文和连字符。不得使用 `image1.png`、`截图.png`、`未命名.png` 等无法判断内容的名称。

### 8.4 清理原则

完成笔记后，仅保留：

- 正文实际引用的关键图片；
- 官方论文 PDF 和有价值的补充材料；
- `sources.md`；
- 对理解或复现确有帮助的少量支持文件。

整页临时渲染图、重复下载文件、未使用的碎片图和缓存不应进入最终附件目录。涉及删除个人目录中的临时文件时，仍需先列出并确认，使用废纸篓机制，不得永久删除。

## 9. 写完后的检查清单

- [ ] 已在 `Paper-Index.md` 先行注册
- [ ] 已按论文名称、arXiv ID、DOI、链接和内链查重
- [ ] 主要内容基于官方论文 PDF，而不是只依据二手解读
- [ ] 已阅读可获取的 Supplementary Material
- [ ] 领域与目录对应
- [ ] 文件名符合“论文短名 - 中文主题”
- [ ] 标题层级连续且带编号
- [ ] 已说明任务输入、输出、核心表示和完整 pipeline
- [ ] 已解释关键公式、实验结果和消融实验
- [ ] 至少包含 3 张来源明确的关键图；不足时已说明原因
- [ ] 方法图为完整图表，不是 PDF 内嵌图片碎片
- [ ] 每张图均有中文解读、Figure/Page 信息和来源链接
- [ ] 官方 PDF、补充材料和关键图片已保存到对应 assets 目录
- [ ] `sources.md` 能追溯每项本地材料
- [ ] 公式符合 Obsidian 语法
- [ ] 附件目录与笔记名一致，正文使用相对路径
- [ ] 已讨论局限、失败案例、复现成本和个人判断
- [ ] 索引摘要已回填并包含 ★ 评级
- [ ] `[[阅读总结链接]]` 可以正常打开
- [ ] 已清理未使用的整页渲染图、碎片图和重复文件

不满足以上检查项时，任务不视为完成。
