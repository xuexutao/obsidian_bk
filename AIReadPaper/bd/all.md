## 文档定位

这是一个面向**技术情报持续沉淀**的工作文档，服务于“旭涛的技术视野”长期积累。每条新任务原则上对应一行记录：先读源链接，再生成阅读总结飞书文档，最后归档到所属领域表格。

## 使用约定

1. **一条任务 = 一行记录**。
2. **优先按技术主题归档**，而不是按文章来源归档。
3. 每行固定保留以下字段：时间、文章题目、论文名称、文章简要总结、原始链接、阅读总结飞书文档。
4. 若一篇内容涉及多篇论文，优先选择**主论文**入主表；其余论文在阅读总结文档中展开。
5. 若暂时无法判断领域，先放入“待分类”表，后续再迁移。

---

## 领域总览

**当前主领域**

- VLM
- 视频生成
- 3D生成
- 三维重建
- 世界模型
- VLA
- 基础模块

**推荐记录流程**

1. 接收链接
2. 提取正文与潜在论文
3. 生成阅读总结文档
4. 识别技术领域
5. 将一行刷入对应表格

---

## VLM

| 时间         | 文章题目              | 论文名称                                                | 文章简要总结                                                                                                                              | 原始链接                                     | 阅读总结飞书文档                                                                  |
| ---------- | ----------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------- |
| 2026-06-05 | VLM3：VLM是天生的3D学习者 | VLM3: Vision Language Models Are Native 3D Learners | 文章围绕 VLM³ 展开，核心创新是用焦距统一、归一化文本坐标与数据重配比，把标准 VLM 扩展到深度估计、像素对应、相机位姿和物体级 3D 理解等多类任务。它代表 3D 学习正从任务专用结构走向统一 VLM 接口，是值得重点跟踪的多模态几何主线工作。★★★★☆ | [原文链接](http://xhslink.com/o/2GFNszlIdR2) | [阅读总结](https://bytedance.larkoffice.com/docx/PwOCdto79oed9RxsSlqcPUoSnYd) |

## 视频生成

|时间|文章题目|论文名称|文章简要总结|原始链接|阅读总结飞书文档|
|---|---|---|---|---|---|
|2026-05-28|英伟达 LongLive-2.0 论文精读：NVFP4 并行架构如何让长视频生成训练快 2.15 倍|LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation|文章围绕 LongLive-2.0 展开，重点解释其如何用 Balanced SP 与 NVFP4 打通长视频生成的训练和推理基础设施，并以 2.15x 训练加速、1.84x 推理加速与 45.7 FPS 证明实时交互长视频生成开始具备工程可行性。该工作代表低精度从推理走向训练、视频生成从模型竞争走向系统协同优化，值得持续跟踪。 ★★★★☆|[原文链接](https://mp.weixin.qq.com/s/4XDyEEOKsXY3ox32FPlBpQ)|[LongLive-2.0 阅读总结](https://bytedance.larkoffice.com/docx/WgRsdAXeUoF8CmxmM1FcVOoAnTc)|

## 3D生成

|时间|文章题目|论文名称|文章简要总结|原始链接|阅读总结飞书文档|
|---|---|---|---|---|---|
|2026-05-28|开源 \| 单图生成带物理约束的3D模型，支持刚/柔/关节体、可拆分、无需人工绑定骨骼与蒙皮|PhysX-Omni|文章围绕 PhysX-Omni 展开，核心创新是把单图 3D 生成提升为仿真就绪资产生成：统一支持刚体、柔体和关节体，提出 Template-based RLE 几何表示，并配套构建 PhysXVerse 数据集与 PhysX-Bench 评测。重要性评估：★★★★☆（4/5），因为它直接命中机器人仿真资产自动化这一高价值问题，但真实工业鲁棒性仍需继续观察。|[原文链接](https://mp.weixin.qq.com/s/ihGHuVWO-XkXupE-dKCtIQ)|[阅读总结](https://bytedance.larkoffice.com/docx/GcKpdtYx5o8rJSxsSsEcsPbknng)|
|2026-05-29|Roblox 悄悄做了件大事：开源3D 资产生成，终于可以“按零件清单”定制了！|CubePart: An Open-Vocabulary Part-Controllable 3D Generator|文章围绕 CubePart 展开，核心价值是把开放词汇 schema 直接变成 3D 多部件生成的控制接口，使输出资产能按零件名与引擎脚本对齐并直接驱动行为。它同时补齐了大规模开放词汇部件数据流水线，代表 3D 生成从“生成形状”走向“生成可用资产”的重要一步。重要性评估：★★★★☆（4/5）。|[原文链接](https://mp.weixin.qq.com/s/YvscqEVimQEWDp1_p1N2Sg)|[阅读总结](https://bytedance.larkoffice.com/docx/XZWXdQ899ouKaXxWFZVcfywtn4g)|
|2026-06-02|【顶会精读】新稀疏体素搞定高质量3D生成|Native and Compact Structured Latents for 3D Generation|文章围绕 TRELLIS.2 / Native and Compact Structured Latents for 3D Generation 展开，核心价值是提出统一编码几何与 PBR 材质的 O-Voxel 表示，并结合 SC-VAE 与 4B flow-matching 在紧凑潜空间中实现高质量 3D 资产生成。它代表 3D 生成进一步走向“可直接使用资产”的重要路线，重要性评估：★★★★★（5/5）。|[原文链接](https://mp.weixin.qq.com/s/8eORJ_6CLZKVBWGoka3_jA)|[阅读总结](https://bytedance.larkoffice.com/docx/F6i6da1hLouNvcxhhUdcGufMnkc)|
|2026-06-05|Hunyuan3D-PolyGen：腾讯推出的美术级3D生成新突破|Hunyuan3D-PolyGen（论文待确认）|文章围绕 Hunyuan3D-PolyGen 展开，重点解释其如何通过网格自回归建模、BPT 压缩表示与质量优化，把 3D 生成从“可看”推进到更接近游戏美术生产可用的资产生成。它代表 3D 生成从几何结果竞争走向工业可交付标准竞争，是值得持续跟踪的一条工程主线。★★★★☆|[原文链接](https://zhuanlan.zhihu.com/p/1926224806765897119?share_code=oGdkG205k5sT&utm_psn=2037608389782532657)|[阅读总结](https://bytedance.larkoffice.com/docx/CGO1dqnU0ofGcbx3ZIpcrvjzn8c)|
|2026-06-05|微软开源TRELLIS：拍张照片就能生成3D模型|Structured 3D Latents for Scalable and Versatile 3D Generation|文章围绕 TRELLIS 展开，重点说明其如何用统一的 SLAT 表示把图像/文本生成 3D 资产、局部编辑和多格式解码整合进一套框架。它代表 3D 生成正从单一结果展示走向可编辑、可导出、可接入资产工作流的统一基础模型路线，值得重点跟踪。★★★★★|[原文链接](https://mp.weixin.qq.com/s/sqKf4gt46DQbVZgp1OQagA)|[阅读总结](https://bytedance.larkoffice.com/docx/HtkbdJmcBoW9MZxnN7LcXIhUnRb)|
|2026-06-05|SAM-3D|SAM 3D: 3Dfy Anything in Images|文章围绕 Meta 的 SAM 3D 展开，核心价值是把单张自然图像中的目标恢复为带几何、纹理与布局信息的 3D 资产，并通过大规模真实世界数据引擎与两阶段架构提升 in-the-wild 场景可用性。它代表 image-to-3D 正从 demo 走向 foundation model 化与资产工作流化，是 3D 生成主线里非常值得重点跟踪的一篇工作。★★★★★|[原文链接](https://mp.weixin.qq.com/s/HW5lCkIQLdR9M5jTsUFRig)|[阅读总结](https://bytedance.larkoffice.com/docx/W0Tpd4dUuo1bk2xqdBwc9ummnph)|
|2026-06-05|3D 塞进 MLLM，理解+生成+编辑全开！🧊|ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding|文章围绕 ShapeLLM-Omni 展开，核心价值是把 3D mesh 压成离散 token 并原生接入 MLLM，在一套自回归框架中统一实现 3D 理解、生成与编辑。它代表 3D 多模态系统从“外挂式单任务模型”走向“统一原生 3D MLLM”的重要拐点，值得持续跟踪。重要性评估：★★★★☆（4/5）。|[原文链接](http://xhslink.com/o/1ln7xuHZ1MT)|[阅读总结](https://bytedance.larkoffice.com/docx/Ut9qdb5kroB7nCxICeAcPhswnAe)|
|2026-06-05|CVPR 2026 \| 解锁3D变形新境界！南大&北大提出MorphAny3D，让3D生成大模型秒变“变形魔法师”|MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing|文章围绕 MorphAny3D 展开，核心价值是证明 Structured Latent / SLAT 不仅能做 3D 生成，还能在无需训练的前提下支持跨类别 3D morphing、时序平滑约束与姿态纠正。它代表 3D 生成基础模型正从“出结果”走向“可编辑资产工作流”，值得持续跟踪。重要性评估：★★★★☆（4/5）。|[原文链接](https://mp.weixin.qq.com/s/-Dqms6q7UHt-DowMGZe_aA)|[阅读总结](https://bytedance.larkoffice.com/docx/C3D6dVn6CoV453xJGTuco6SQng0)|
|2026-06-06|SymTRELLIS：把对称性直接写进 3D 生成过程|SymTRELLIS: Symmetry-Enforced Voxel Latents for 3D Generation|论文围绕 SymTRELLIS 展开，核心价值是以 latent-space transform mapper + velocity symmetrization 的方式，在不重训 TRELLIS.2 主模型的前提下把对称性直接注入 image-to-3D 采样过程。它代表 3D 生成开始从“视觉上像”进一步走向“结构上对、功能上可用”的约束生成路线，值得持续跟踪。重要性评估：★★★★☆（4/5）。|[原文链接](https://arxiv.org/abs/2606.04108)|[阅读总结](https://bytedance.larkoffice.com/docx/DzGGdWnKOoZ43lxJ9VrcKBUXnoe)|
|2026-06-10|ICML: 3D生成新范式! 从高维几何到基元几何|Rethinking 3D Shape Generation: Diffusion over Superquadrics|文章围绕 DoSs 展开，核心价值是把 3D diffusion 的生成空间从体素/点云/网格等高维稠密几何改写为 superquadric primitive tokens，在约 7KB 扩散状态和 0.6s 采样速度下实现可竞争质量，并原生支持部件级编辑与几何约束设计。它代表 3D 生成开始从“在密集几何上堆算力”走向“在结构化几何基元上做生成”的新范式，值得持续跟踪。重要性评估：★★★★☆（4/5）。|[原文链接](http://xhslink.com/o/9YOkzTN3pdz)|[阅读总结](https://bytedance.larkoffice.com/docx/DhaGdLuqkoQOyNx34ZOckH5VnAB)|
|2026-06-14|SIGGRAPH’26 \| 仅需30% Token！SimArt：打破3D铰链物体生成瓶颈|SIMART: Decomposing Monolithic Meshes into Sim-ready Articulated Assets via MLLM|文章围绕 SIMART 展开，重点说明其如何把静态 3D mesh 直接转成带部件分解、关节参数与 URDF 逻辑的 sim-ready articulated asset，并通过 Sparse 3D VQ-VAE 将 3D token 冗余压缩约 70%。它代表 3D 生成正从“生成静态外观”走向“生成可仿真、可交互的功能资产”，对具身仿真与资产流水线都很有价值。★★★★★|[原文链接](https://mp.weixin.qq.com/s/naNumcwGoAVuLu6nKQj8fA)|[阅读总结](https://bytedance.larkoffice.com/docx/Nz5bdDwlaoPD41xHXc8cl4d5nQb)|
|2026-08-20|LATO.2：将顶点与拓扑解耦的原生 Mesh 生成|LATO.2: Factorized 3D Mesh Generation with Vertex and Topology Flow|论文把显式网格生成拆成 V-Flow 与 T-Flow：先生成可控数量、亚体素精度的顶点，再以这些顶点为条件生成连接拓扑。它在 CD、HD 与法线一致性上超过 LATO、MeshFlow 等方法，并自然支持分部高分辨率生成、拼接和编辑后重拓扑；但仍受 $O(N^2)$ 解码及缺少 UV/材质限制。★★★★★|[arXiv](https://arxiv.org/abs/2607.10623)|[[LATO.2 阅读总结]]|

## 三维重建

|时间|文章题目|论文名称|文章简要总结|原始链接|阅读总结飞书文档|
|---|---|---|---|---|---|
|2026-06-06|D4RT：统一、高效的动态4D场景重建新范式|Efficiently Reconstructing Dynamic Scenes One D4RT at a Time|文章围绕 D4RT 展开，核心创新是把动态 4D 重建统一成对全局场景表示的查询接口，用一个轻量解码器同时完成深度、点云、3D 轨迹与相机参数估计，并在多个基准上实现 SOTA 精度与数量级效率提升。它代表动态三维重建正从多头、多模块管线走向统一几何表示与按需解码，是非常值得重点跟踪的主线工作。★★★★★|[知乎专栏原文](https://zhuanlan.zhihu.com/p/1982152380603712066?share_code=1k9Vwc6h4yl45&utm_psn=2046581540344292113)|[D4RT 论文阅读总结](https://bytedance.larkoffice.com/docx/YyE0dQfAJoIGSQxRvlFcxwcOnNm)|

## 世界模型

|时间|文章题目|论文名称|文章简要总结|原始链接|阅读总结飞书文档|
|---|---|---|---|---|---|
|2026-06-05|图文速览\|VWM综述解读：视觉世界模型|From Seeing to Knowing the World: A Survey of Vision World Models|文章围绕视觉世界模型综述展开，系统梳理 VWM 的统一定义、四大架构家族与评测体系，并明确区分“好看的视频生成”和“真正可用于理解与决策的世界建模”。它非常适合作为后续持续跟踪 world model、VLA 与 4D 世界仿真论文的坐标系。★★★★☆|[原文链接](https://mp.weixin.qq.com/s/y5FQApHEMvhTpjAWa8gXtw)|[阅读总结](https://bytedance.larkoffice.com/docx/GVBsdKBgvoEBcYxZ5XMcWSu3n2b)|
|2026-05-28|DreamDojo：基于大规模人类视频的通用机器人世界模型|DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos|文章围绕 DreamDojo 展开，重点说明如何用 4.47 万小时第一人称人类视频与连续隐式动作预训练通用机器人世界模型，并通过后训练与蒸馏实现强 OOD 泛化和实时交互。该工作同时打通策略评估、模型规划与遥操作闭环，是世界模型方向的高价值主线论文。 ★★★★★|[原文链接](https://mp.weixin.qq.com/s/ok-WWqM80ckW8HDDqEFmjw)|[DreamDojo 阅读总结](https://bytedance.larkoffice.com/docx/GUVpdRIJlo6pCIxAZH2cKwOKnze)|
|2026-06-10|LeWorldModel：从像素稳定端到端训练 JEPA 世界模型|LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels|论文提出 LeWorldModel，用 next-embedding prediction + SIGReg 两项损失就能从像素端到端稳定训练 JEPA 世界模型，在 15M 参数、单卡数小时训练下实现与强基线接近或更优的控制效果，并把规划速度提升到最高 48×。它值得纳入世界模型主线，因为它把“稳定训练、低成本、可规划”三件事用极简 recipe 统一起来。重要性评估：★★★★☆（4/5）。|[arXiv](https://arxiv.org/abs/2603.19312)|[阅读总结](https://bytedance.larkoffice.com/docx/ZHNedx8I6om161xkxVcc9BONnZf)|
|2026-06-11|LeCun新证明：世界是高斯的|When Does LeJEPA Learn a World Model?|文章追溯到 LeJEPA 理论论文，证明在高斯潜变量与平稳加噪转移条件下，LeJEPA 可线性恢复真实 latent，并使 latent-space planning 与真实状态规划等价。它非常值得纳入世界模型主线，因为它第一次把 JEPA 表征几何、线性 probe 与控制规划连成了可证明的理论闭环。重要性评估：★★★★☆（4.5/5）。|[原文链接](https://mp.weixin.qq.com/s/Fm45QXc6vTnLJOyXScbkHQ)|[阅读总结](https://bytedance.larkoffice.com/docx/LKpDdxFlAozjIixK6aJczB8anyg)|
|2026-08-20|Fast-WAM：世界动作模型真需要在测试时「想象未来」吗？|Fast-WAM: Do World Action Models Need Test-time Future Imagination?|论文提出 Fast-WAM，训练时保留视频联合训练、推理时跳过未来预测：用 Wan2.2-5B 视频 DiT 做一次前向编码得到隐式世界表征后直接生成动作（视频 DiT + 1B action expert 的 MoT 混合体）。受控消融表明 WAM 的价值主要来自训练期视频建模而非推理期未来想象——RoboTwin 91.8%、LIBERO 97.6%，推理延迟仅 190ms、比 imagine-then-execute 类 WAM 快 4 倍以上。★★★★☆|[arXiv](https://arxiv.org/abs/2603.16666)|[[Fast-WAM 阅读总结]]|

## VLA

|时间|文章题目|论文名称|文章简要总结|原始链接|阅读总结飞书文档|
|---|---|---|---|---|---|
|2026-05-28|AAAI 2026杰出论文奖 \| ReconVLA：具身智能研究首次获得AI顶级会议最佳论文奖|ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver|文章围绕 ReconVLA 这一 VLA 新范式展开，用目标区域重建替代显式 grounding，解释其如何提升机器人对任务目标的隐式对齐与长时序操作成功率。该工作兼具方法启发性与方向标志性，值得纳入具身智能主线持续跟踪。 ★★★★★|[原文链接](https://mp.weixin.qq.com/s/ybCbRhy3GqoLHGtV7rokng)|[阅读总结](https://bytedance.larkoffice.com/docx/IeUNdjEdYoj8RQxbIMccDZDDnjd)|
|2026-05-28|利用大量真实人类活动视频来进行可扩展的 VLA 预训练|Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos|文章围绕 VITRA 展开，重点说明如何把无标注真实人类活动视频自动转成与机器人操控对齐的 VLA 预训练数据，并验证其能显著提升真实机器人在 seen / unseen 场景中的泛化能力。该工作兼具数据范式与模型方法价值，值得作为 VLA 预训练主线重点跟踪。 ★★★★★|[原文链接](https://mp.weixin.qq.com/s/mA2FDEyl-aj7UWzgpyFjlg)|[阅读总结](https://bytedance.larkoffice.com/docx/ZPtBdaOk2o7Vptxy6eYc8OmRneg)|
|2026-05-28|DreamZero 如何把视频世界模型变成零样本机器人策略|World Action Models are Zero-shot Policies|文章围绕 DreamZero 展开，重点说明如何将预训练视频扩散世界模型改造成联合预测未来视频与动作的机器人策略，并在零样本泛化、跨 embodiment 迁移和实时闭环控制上取得显著提升。该工作连接了世界模型与 VLA 两条主线，值得作为具身基础模型的重要路线持续跟踪。 ★★★★★|[原文链接](https://mp.weixin.qq.com/s/Pwg9u51zVYnkXejnzpEhyw)|[阅读总结](https://bytedance.larkoffice.com/docx/LJtPdEdXHokqZmxOHXScQ7UpnUe)|
|2026-05-29|PrimitiveVLA：把 VLA 从整轨迹记忆改造成可复用动作原语学习|PrimitiveVLA: Learning Reusable Motion Primitives for Efficient and Generalizable Robotic Manipulation|论文提出 Primitive-Centric Disassemble & Assemble 范式，通过自动 primitive 拆解、MCR 统一表示与 planner-switch 闭环组装，显著提升 VLA 的数据效率、未见任务泛化与长程执行能力。它不是单纯换 backbone，而是在训练监督组织层面重构 VLA recipe，是具身主线里非常值得重点跟踪的一篇工作。 ★★★★★|[arXiv](https://arxiv.org/abs/2605.28634v1)|[阅读总结](https://bytedance.larkoffice.com/docx/Ph4Cdy5puoPwrpx9Mw9c7wTInAh)|
|2026-05-31|Qwen 也开始做机器人大脑了：Qwen-VLA 如何把操作、导航和动作生成统一起来？|Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments|文章围绕 Qwen-VLA 展开，核心价值是把机器人操作、视觉导航、人类动作与轨迹预测统一为 action-and-trajectory prediction，并通过 Qwen3.5 + DiT、embodiment-aware prompt 和统一 action tensor 验证 generalist VLA 的可行性。它兼具基础模型意义与工程启发性，是具身主线里非常值得重点跟踪的一篇工作。 ★★★★★|[原文链接](https://mp.weixin.qq.com/s/ZiY7YwQOYAoz56dcKt3W0w)|[阅读总结](https://bytedance.larkoffice.com/docx/RfXhdaMLgomAKbxMEDeci5R1nHe)|
|2026-06-05|GigaBrain-0：World Model 驱动的 VLA 数据引擎|GigaBrain-0: A World Model-Powered Vision-Language-Action Model|论文提出 GigaBrain-0，用 Real2Real、View Transfer、Sim2Real、Human Transfer 与 Video-Gen 等 world-model 生成数据扩充 VLA 训练，并结合 RGBD + Embodied CoT 提升长程与跨场景泛化。它代表 VLA 正把 world model 从“辅助模块”前移为“数据引擎”，对降低真实机器人采集成本很有启发。重要性评估：★★★★☆（4/5）。|[arXiv](https://arxiv.org/abs/2510.19430)|[阅读总结](https://bytedance.larkoffice.com/docx/G9YndAZVIov650xHRA5cE6Vonve)|
|2026-06-06|GaussianDream：面向机器人操控的前馈式 3D Gaussian 世界模型|GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation|论文提出 GaussianDream，在 VLA 训练阶段引入当前 3D Gaussian 重建与短时未来 3D 演化预测，并在推理时仅保留 prefix，因此兼顾了显式几何建模与可控时延。它把 3D 表征、world model 和 VLA 训练型增强三条线较好地接到了一起，值得持续跟踪。★★★★☆|[arXiv](https://arxiv.org/abs/2605.20752)|[阅读总结](https://bytedance.larkoffice.com/docx/C5ADdaPOMoqVM0xDV54cuJ0inw9)|

## 基础模块

|时间|文章题目|论文名称|文章简要总结|原始链接|阅读总结飞书文档|
|---|---|---|---|---|---|
|2026-05-28|CVPR2026 最佳论文候选！清华阿里提出ViT³ 本文第一...|ViT³: Unlocking Test-Time Training in Vision|系统梳理视觉 TTT 的设计空间，并提出线性复杂度的 ViT³ 基线，在分类、检测、分割、生成等任务上验证有效，适合用来建立视觉 TTT 方法版图。⭐⭐⭐⭐☆|[原始链接](http://xhslink.com/o/5YCCXMdtTwB)|[阅读总结](https://bytedance.larkoffice.com/docx/NUp0dfsBHobNrXx18LTc4RTfnu7)|
|2026-06-06|CoSMo3D：用规范空间建模开放世界 3D 部件分割|CoSMo3D: Open-World Promptable 3D Semantic Part Segmentation through LLM-Guided Canonical Spatial Modeling|论文提出 CoSMo3D，用 LLM 引导的跨类别 canonical dataset 与训练期 canonical branch，把“部件语义 = 几何 + 规范空间位置”显式注入 3D 表征，在多套开放世界 3D 部件分割基准上稳定优于 Find3D 等方法。对构建更稳健的 3D 语义理解底座很有启发。重要性评估：★★★★☆（4/5）。|[arXiv](https://arxiv.org/pdf/2603.01205)|[阅读总结](https://bytedance.larkoffice.com/docx/UH5odhZ32oi96wxwxW4cdvzvnvd)|
|2026-06-06|CVPR 2026 Oral！NVIDIA王炸新作PixelDiT：图像生成的"终极答案"来了？|PixelDiT: Pixel Diffusion Transformers for Image Generation|论文提出 PixelDiT，用 dual-level DiT、pixel-wise AdaLN 与 token compaction 让端到端像素空间扩散在 ImageNet 256/512 与 1024² T2I 上接近强 latent diffusion 水平。它把“无 VAE 的生成底座”从概念验证推进到可竞争范式，对扩散模型基础架构演进很有参考价值。★★★★★|[原始链接](https://mp.weixin.qq.com/s/HkDKkl5BkRI7YdwDksizsA)|[阅读总结](https://bytedance.larkoffice.com/docx/Yc8ydvsITo5u2exz2lhc8rnCnWc)|
|2026-06-08|CVPR 2026 最佳学生论文提名奖! ChordEdit：基于最优传输理论的免训练单步图像编辑|ChordEdit: One-Step Low-Energy Transport for Image Editing|文章围绕 ChordEdit 展开，核心创新是把 one-step 图像编辑建模为动态最优传输，并用时间窗平滑得到低能量 Chord Control Field，在无需训练、无需反演的前提下实现高保真实时编辑。它把“快模型不可控编辑”转成可解释的控制问题，对快速扩散/流模型的稳定控制很有启发。★★★★☆|[原始链接](https://mp.weixin.qq.com/s/4XePzgnExCzhhgXJz89YOA)|[阅读总结](https://bytedance.larkoffice.com/docx/DowzdWFyzoFE9fxkMXXcVEMNnVc)|
|2026-06-08|硬刚潜空间！英伟达&罗切斯特大学发布PixelDiT，用1.61的FID证明：像素级生成才是未来！|PixelDiT: Pixel Diffusion Transformers for Image Generation|文章围绕 PixelDiT 展开，重点解释其如何用 dual-level Transformer、pixel-wise AdaLN 与 token compaction 实现端到端像素空间扩散，并在 ImageNet 256/512 与 1024² 文生图上逼近强 latent diffusion。它把“无 VAE 的像素生成”从概念验证推进到可竞争架构范式，对扩散基础模型和高保真图像编辑都很有参考价值。★★★★★|[原始链接](https://mp.weixin.qq.com/s/jGKNJcQAp7-BOUpT87-MNg)|[阅读总结](https://bytedance.larkoffice.com/docx/S3kLdWIzdoHJcRxwf6DcmPrqnJb)|
|2026-06-09|Vision Banana：图像生成器正在变成通用视觉学习器|Image Generators are Generalist Vision Learners|论文提出 Vision Banana，通过低比例视觉任务 instruction tuning 把图像生成模型统一改写为 segmentation、depth 与 surface normal 的 RGB 输出接口，并在多项 2D/3D benchmark 上超过或逼近 specialist，同时基本保持生成能力。它对“生成式预训练正在成为通用视觉底座”的判断很强，是基础模块方向非常值得重点跟踪的一篇工作。★★★★★|[arXiv](https://arxiv.org/pdf/2604.20329)|[阅读总结](https://bytedance.larkoffice.com/docx/QLE0dMAdro9YbmxMxN1ct6kJnWf)|
|2026-08-20|【文生图】先在 latent 学，再把 VAE 扔掉：阿里把 1024² 文生图压到 0.20 秒，端到端快 4.75 倍|An Empirical Study of Training Pixel-Space Text-to-Image Diffusion Models|文章围绕 pixel-space 文生图展开，核心是 latent-to-pixel 训练配方：预训练借 latent space 学语义、后训练切到 pixel space 直接生成 RGB，配合同源合成数据+真实图 1:1 混合、DiP 轻量解码头、渐进式大 patch 与 4 步蒸馏，把 1024² 端到端延迟从 0.95s 压到 0.20s（快 4.75×）。★★★★☆|[原始链接](https://mp.weixin.qq.com/s/D2DRtwmUqT-ygKVxY2N-Xg)|[[Pixel-Space文生图 latent-to-pixel 阅读总结]]|
|2026-08-20|四边形网格的组合式建模：保留动画拓扑的局部 Boolean 管线|Modeling Quadrilateral Meshes by Composition|论文提出一套面向动画资产的四边形网格组合流程：先在三角网格上执行稳健 Boolean，再只重建交界线附近的 patch，并用整数优化协调边界细分数，从而输出封闭纯四边形网格并尽量保留原始 edge flow。原型在普通案例中达到约 0.25–3.27 秒，但对尖锐特征、输入分辨率和初始 patch layout 较敏感。★★★★☆|[本地 PDF](file:///Users/eilonxtxue/Downloads/2971.pdf)|[[Modeling Quadrilateral Meshes by Composition 阅读总结]]|

---

## 待分类

|时间|文章题目|论文名称|文章简要总结|原始链接|阅读总结飞书文档|
|---|---|---|---|---|---|
|待补充|待补充|待补充|待补充|待补充|待补充|

---

## 后续自动追加流程

后续 Skill 的目标流程：收到一个微信公众号 / 小红书 / X 链接后，先抽取正文与候选论文，再产出一篇阅读总结飞书文档，最后把该任务作为一行插入到所属领域表格。

**建议执行顺序：**

1. 输入一个外部链接。
2. 识别文章来源与正文可提取方式。
3. 提取文章中的论文名称、项目名称、代码仓库或方法线索。
4. 生成一篇新的阅读总结飞书文档，沉淀核心方法、Pipeline、I/O、实验发现与评注。
5. 识别所属领域。
6. 将结果追加到对应表格。
