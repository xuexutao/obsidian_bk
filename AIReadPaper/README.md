# AIReadPaper 写入规范

> 本目录用于沉淀单篇论文阅读笔记与多篇论文领域综述。任何人、Skill 或 Agent 在这里写入论文内容时，都必须遵守本文件。

## 最高优先级规则

**先在 `Paper-Index.md` 注册，再创建或更新论文笔记。**

禁止先生成笔记、最后再补索引。若索引尚未登记，正文写作不得开始。

## 1. 强制执行流程

1. 打开 `Paper-Index.md`。
2. 使用论文正式名称、论文短名、arXiv 链接和已有 `[[Obsidian 内链]]` 查重。
3. 判断所属领域：VLM、视频生成、3D生成、三维重建、世界模型、VLA、基础模块；无法判断时放入“待分类”。
4. 在对应领域表格先新增一行：
   - 时间：当前日期，格式为 `YYYY-MM-DD`；
   - 文章题目：外部文章标题；若直接阅读论文，可填写论文短名或中文标题；
   - 论文名称：论文正式英文名称；
   - 文章简要总结：写作前允许填写“待完成”；
   - 原始链接：优先使用论文页、PDF、项目页或原始文章来源；
   - 阅读总结链接：预先填写 `[[计划使用的笔记名]]`。
5. 注册成功后，才可以创建或更新论文笔记。
6. 完成笔记后回到 `Paper-Index.md`：
   - 将“待完成”替换为最终简要总结；
   - 在总结末尾加入 `★` 评级；
   - 确认阅读总结内链能够解析到实际笔记。
7. 最后检查正文图片、相对附件路径和索引链接。

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

当前目录迁移尚未全部完成时，不得同时自行创建另一套平行分类。若目标目录尚不存在，先向旭涛确认迁移状态；无论处于新旧目录，`Paper-Index.md` 始终是唯一登记入口。

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

## 5. 单篇论文正文规范

一级标题使用“论文短名 + 论文主题”，并与文件名主体对应。建议结构如下：

```markdown
# 论文短名：中文主题

## 基本信息

---

## 1. 研究背景

### 1.1 问题定义

---

## 2. 核心方法

### 2.1 总体框架

---

## 3. 实验与结果

---

## 4. 局限与评价
```

要求：

- “基本信息”可不编号，其余章节使用层级编号；
- 标题不得跳级：`## 1.` → `### 1.1` → `#### 1.1.1`；
- 中文散文式行文为主，避免把全文写成碎片化短句列表；
- 数学公式使用 Obsidian 语法：行内 `$内容$`，块级 `$$内容$$`；
- 每篇单篇论文笔记至少包含 1 张从论文 PDF 原文提取的关键图；
- 结尾应包含方法价值、局限和个人判断，而不只是翻译摘要。

## 6. Frontmatter 建议

```yaml
type: paper-note
status: draft
domain: 3D-Generation
paper: 论文正式英文名称
year: 2026
source: https://arxiv.org/abs/xxxx.xxxxx
tags:
  - 3D-Generation
created: YYYY-MM-DD
updated: YYYY-MM-DD
```

领域综述将 `type` 改为 `survey`。

## 7. 附件规范

单篇论文专属图片放在同领域的附件目录：

```text
10-Papers/<Domain>/assets/<笔记名>/
```

正文使用相对路径：

```markdown
![](assets/<笔记名>/01-method-overview.png)
```

附件命名采用：

```text
01-teaser.png
02-method-overview.png
03-experiment.png
```

不得把论文图片散落在 `AIReadPaper/` 根目录，也不得让附件目录与笔记名称不一致。

## 8. 写完后的检查清单

- [ ] 已在 `Paper-Index.md` 先行注册
- [ ] 已查重，没有创建近似重复笔记
- [ ] 领域与目录对应
- [ ] 文件名符合“论文短名 - 中文主题”
- [ ] 标题层级连续且带编号
- [ ] 至少包含一张论文原图
- [ ] 公式符合 Obsidian 语法
- [ ] 附件放在与笔记同名的目录中
- [ ] 索引摘要已回填并包含 ★ 评级
- [ ] `[[阅读总结链接]]` 可以正常打开

不满足以上检查项时，任务不视为完成。
