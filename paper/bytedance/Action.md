# 代码逻辑分析：ActionFilter 与 ActionLoc

这份文档只保留两大章：第一章讲 `ActionFilter`，第二章讲 `ActionLoc`。两条路线本质上都建立在同一个基础上：**先用冻结的 V-JEPA2 视频编码器提取时空表征，再在这个冻结表征上训练一个更轻量的下游头**。区别在于：

- `ActionFilter` 做的是**视频级二分类**；

- `ActionLoc` 做的是**时间段定位**，也就是 temporal action localization。

---

## 第一章：ActionFilter

你现在这条训练命令是：

```bash

python -m evals.main --fname configs/eval/vitl/binary.yaml \

--devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

```

先说结论：代码里并没有一个显式叫 `ActionFilter` 的类，这个名字是你当前这套**配置实例**对应的业务名。它在代码里的真实身份是一个 `video_classification_frozen` 任务，配置定义在 `configs/eval/vitl/binary.yaml:1`，入口任务名是 `video_classification_frozen`，见 `configs/eval/vitl/binary.yaml:2`。

### 1. ActionFilter 现在到底是什么原理

当前 ActionFilter 的工作原理可以概括成一句话：

> 用冻结的 V-JEPA2 ViT-L 编码器把视频变成时空 token，再训练一个 attentive classifier 做二分类，最后用多头 ensemble 给出“这个视频/窗口是否有有效动作”的概率。

这个结论对应的关键代码有几处：

- 本地多卡训练入口在 `evals/main.py:16-113`；

- 根据配置动态导入任务的逻辑在 `evals/scaffold.py:13-19`；

- 真正进入视频分类训练循环的是 `evals/video_classification_frozen/eval.py:49-332`；

- 冻结 encoder 的逻辑在 `evals/video_classification_frozen/models.py:42-45`；

- 加载 V-JEPA2 ViT-L 并做多 clip 聚合的逻辑在 `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py:41-149`；

- 分类头定义在 `src/models/attentive_pooler.py:103-137`。

### 2. 这条命令是怎么运行起来的

运行路径比较清楚：

1. `python -m evals.main` 先进入 `evals/main.py:93-113`；

2. 它会读取 `configs/eval/vitl/binary.yaml`，见 `evals/main.py:61-79`；

3. 你传了 8 张 GPU，所以它会直接起 8 个本地进程，见 `evals/main.py:109-113`；

4. 每个进程会把 `CUDA_VISIBLE_DEVICES` 设成对应的单卡，见 `evals/main.py:50`；

5. 然后通过 `src/utils/distributed.py:17-72` 初始化分布式；

6. 最后根据 `eval_name: video_classification_frozen` 进入 `evals/video_classification_frozen/eval.py:49-332`。

这意味着你的命令不是“只验证”，而是真正进入训练循环；因为 `--val_only` 默认为 `False`，见 `evals/main.py:17`，而配置里也没有开启 `val_only`，见 `configs/eval/vitl/binary.yaml:1-165`。

### 3. ActionFilter 的模型结构

这套模型可以拆成两部分：

#### 3.1 冻结骨干：V-JEPA2 ViT-L

配置里指定的预训练 checkpoint 和 encoder 构造方式在 `configs/eval/vitl/binary.yaml:150-165`。真正加载逻辑在 `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py:41-78`。

这里会：

- 从 `pretrained_vjepa2_checkpoints/vitl.pt` 读取 `target_encoder` 权重，见 `configs/eval/vitl/binary.yaml:151` 和 `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py:49-68`；

- 构造 `vit_large`，其定义在 `src/models/vision_transformer.py:275-286`；

- `vit_large` 的核心规模是：`embed_dim=1024, depth=24, num_heads=16`，见 `src/models/vision_transformer.py:275-286`。

更关键的是，这个 encoder 会被完全冻结：

- `evals/video_classification_frozen/models.py:42-45` 里显式把所有参数 `requires_grad=False`。

所以当前 ActionFilter 不是 full finetune，而是 **frozen feature extractor**。

#### 3.2 可训练头：AttentiveClassifier

分类头初始化在 `evals/video_classification_frozen/eval.py:168-178`，类定义在 `src/models/attentive_pooler.py:103-137`。

这个头不是简单 linear probe，而是：

- 有可学习 query token，见 `src/models/attentive_pooler.py:34`；

- 有若干 transformer block，见 `src/models/attentive_pooler.py:45-58`；

- 最后用 cross-attention 从视频 token 中“读出”一个全局判别向量，见 `src/models/attentive_pooler.py:37-42, 98-100`；

- 再接线性层 `Linear(embed_dim, num_classes)` 做分类，见 `src/models/attentive_pooler.py:132-137`。

配置里这部分的超参是：

- `num_heads: 16`，见 `configs/eval/vitl/binary.yaml:12`；

- `num_probe_blocks: 4`，见 `configs/eval/vitl/binary.yaml:13`；

- `num_classes: 2`，见 `configs/eval/vitl/binary.yaml:20`。

### 4. ActionFilter 的输入数据和标签逻辑

训练数据来自两个 CSV：

- 训练集 `configs/eval/vitl/binary.yaml:16`；

- 验证集 `configs/eval/vitl/binary.yaml:17`。

CSV 解析逻辑在 `src/datasets/video_dataset.py:25-57`。它要求：

- 第一列是视频路径；

- 第二列是整数 label。

例如：

```text

/path/to/a.mp4 0

/path/to/b.mp4 1

```

另外，代码还特地兼容了一种带表头的二分类 CSV，如果首行 label 不是数字、后面全是数字，就会跳过表头，见 `src/datasets/video_dataset.py:39-47`。

如果从数据生成逻辑反推，仓库里的 `scripts/jsonl_to_csv.py:17-68` 说明：

- `result` 为空 -> `label=0`，见 `scripts/jsonl_to_csv.py:46-47`；

- `result` 非空 -> `label=1`，见 `scripts/jsonl_to_csv.py:46-47`。

因此，当前 ActionFilter 的监督目标本质上就是：

> 给定一个视频，判断里面是否存在有效动作/有效结果。

### 5. ActionFilter 的视频采样和张量流

视频采样逻辑在 `src/datasets/video_dataset.py:304-401`。当前配置为：

- `frames_per_clip: 16`，见 `configs/eval/vitl/binary.yaml:19`；

- `frame_step: 4`，见 `configs/eval/vitl/binary.yaml:18`；

- `num_segments: 2`，见 `configs/eval/vitl/binary.yaml:21`；

- `num_views_per_segment: 3`，见 `configs/eval/vitl/binary.yaml:22`。

这意味着一个样本会被切成 **2 段**，每段取 **16 帧**，采样间隔为 **4 帧**。单个 clip 覆盖约 `16 * 4 = 64` 帧的原始跨度。

训练时，代码实际上把训练视角硬编码为 1 个 view：

- `evals/video_classification_frozen/eval.py:184-200`。

验证时才使用 3 个视角：

- `evals/video_classification_frozen/eval.py:201-217`；

- 多视角裁剪逻辑在 `evals/video_classification_frozen/utils.py:192-232`。

从张量角度看：

- 每个 view 的输入大致是 `[B, 3, 16, 256, 256]`；

- encoder 输出 token 后，由 `ClipAggregation` 沿时间把两个 segment 拼起来，见 `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py:119-149`；

- 单个 clip 的 token 数大约是 `8 * 16 * 16 = 2048`，其中 `8` 来自 `16 / tubelet_size(2)`；

- 两段拼接后，每个 view 送进分类头的大致张量是 `[B, 4096, 1024]`；

- 分类头最终输出 `[B, 2]` 的 logits，见 `src/models/attentive_pooler.py:134-137`。

### 6. ActionFilter 是怎么训练出来的

训练主循环在 `evals/video_classification_frozen/eval.py:262-330`，单个 epoch 的核心逻辑在 `evals/video_classification_frozen/eval.py:334-449`。

流程是：

1. DataLoader 读取 clips、labels、clip_indices，见 `evals/video_classification_frozen/eval.py:360-378`；

2. encoder 在 `torch.no_grad()` 下前向，见 `evals/video_classification_frozen/eval.py:381-383`；

3. 每个分类头对 encoder 输出做前向，见 `evals/video_classification_frozen/eval.py:384-387`；

4. 用交叉熵训练，见 `evals/video_classification_frozen/eval.py:357,389`；

5. 只对分类头反向传播，不更新 backbone，见 `evals/video_classification_frozen/eval.py:402-410`。

这里还有一个非常重要的特点：**你现在训练的不是一个头，而是 20 个头**。

原因是 `configs/eval/vitl/binary.yaml:26-146` 里定义了 20 组 `multihead_kwargs`。代码会按这个列表长度创建 20 个 `AttentiveClassifier`，见 `evals/video_classification_frozen/eval.py:169-178`，并分别给它们各自的 optimizer / scheduler，见 `evals/video_classification_frozen/eval.py:221-228, 561-580`。

所以当前 ActionFilter 更准确的定义是：

> 一个冻结的 V-JEPA2 encoder，上面并行训练 20 个 attentive probe head，最后再做概率平均的集成模型。

### 7. ActionFilter 训练产物和推理方式

保存目录规则在 `evals/video_classification_frozen/eval.py:127-135`。按当前配置，理论上的输出目录是：

```text

checkpoints/evals/vitl/binary/video_classification_frozen/binary-vitl16-16x2x3-16f-20w/

```

里面应该主要有：

- `latest.pt`

- `log_r0.csv`

checkpoint 保存逻辑在 `evals/video_classification_frozen/eval.py:245-260`，实际保存的是：

- 20 个分类头的状态；

- optimizer 状态；

- scaler 状态；

- epoch 等元信息。

推理相关脚本主要是：

- `app/infer_csv.py:169-462`；

- `app/infer_action_by_dir.py:432-1453`。

默认会从：

```text

{folder}/video_classification_frozen/{tag}/latest.pt

```

读取 probe checkpoint，见 `app/infer_csv.py:60-68` 和 `app/infer_action_by_dir.py:24-32`。

推理时的组合方式是：

1. 先用冻结 encoder 提取特征；

2. 对每个 head 输出 logits；

3. 多视角平均；

4. 多 head 再平均。

对应逻辑在 `app/infer_csv.py:382-389`。

如果是对长视频按窗口做过滤，脚本会再把每个窗口的正类概率和阈值比较，默认阈值是 `0.5`，见 `app/infer_action_by_dir.py:789-801`。

### 8. 你现在这种 ActionFilter 做法的优点

当前做法有几个很明显的优点：

1. **训练稳定**。冻结 backbone 后，只训练 classifier，参数量小，收敛通常更稳，见 `evals/video_classification_frozen/models.py:42-45`。

2. **对小数据更友好**。如果业务标注不算很大，冻结大模型通常比 full finetune 更安全。

3. **工程简单**。数据只需要视频路径 + 0/1 标签，接入成本低，见 `src/datasets/video_dataset.py:25-57`。

4. **推理一致性较好**。训练和推理都复用同一个 frozen encoder，不容易出现 backbone 漂移。

5. **多头 ensemble 自带鲁棒性**。20 个头的平均结果通常比单头更稳，尤其在 noisy label 场景下有价值。

6. **天然适合做第一道筛选器**。对于海量视频，先做有无动作的粗筛非常合理。

### 9. 你现在这种 ActionFilter 做法的缺点

缺点也比较明确：

1. **上限可能受 frozen encoder 限制**。如果你的业务视频分布和预训练数据差异很大，完全冻结会限制性能上限。

2. **20 头训练和部署成本不低**。虽然每个头不大，但 20 个头一起训练、保存、加载和推理，都会增加复杂度。

3. **准确率不一定等于可用性**。当前日志主看 acc，但业务更可能关心 precision / recall / PR-AUC。

4. **阈值 0.5 是硬编码，不一定最优**，见 `app/infer_action_by_dir.py:789-801`。

5. **训练是视频级监督，长视频推理是窗口级判断**，训练-推理粒度并不完全一致。

6. **验证稳定性存在隐患**。验证时的时序采样可能仍带随机性，模型选择会抖动。

### 10. ActionFilter 最值得改进的点

如果后续只挑最重要的事做，我建议优先看这些：

1. **先修验证链路和评估稳定性**。当前 `run_one_epoch` 的函数签名与验证调用看起来存在明显不匹配风险，见 `evals/video_classification_frozen/eval.py:289-300` 和 `evals/video_classification_frozen/eval.py:334-352`。这类问题应先解决，否则实验结论不稳。

2. **验证集去随机化**。验证最好固定时序采样，避免每次分数波动。

3. **补二分类核心指标**。至少补充 precision / recall / F1 / ROC-AUC / PR-AUC，不要只看 acc。

4. **阈值校准**。把阈值从硬编码改成验证集扫描出来的最优值，而不是固定 `0.5`。

5. **处理类别不均衡**。如果 0/1 分布不均，可以尝试 class weight、focal loss、label smoothing。

6. **把 20-head 方案拆成两阶段**。先做超参搜索，再决定是选单个最好头，还是保留少量 top-k 做 ensemble。

7. **试试部分解冻或 LoRA/Adapter**。如果你怀疑 frozen backbone 表达不够贴业务，可以先小步试最后几层或轻量适配器。

8. **让训练目标更贴近窗口级使用场景**。比如把视频级二分类逐步升级到窗口级弱监督或 MIL。

---

## 第二章：ActionLoc

你现在这条 ActionLoc 路线的训练命令是：

```bash

python -m evals.main --fname configs/eval/vitl/actionloc_robotics.yaml \

--devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

```

这里你说它叫 `ActionLoc`，从代码角度看，它对应的是 `video_actionloc_frozen` 任务，配置在 `configs/eval/vitl/actionloc_robotics.yaml:1`，任务名在 `configs/eval/vitl/actionloc_robotics.yaml:6`。

它和 ActionFilter 的最大区别是：

- ActionFilter 输出的是**整个视频/窗口一个类别概率**；

- ActionLoc 输出的是**一组时间段 `[start_time, end_time]` + 分数 + 类别**。

### 1. ActionLoc 现在到底是什么原理

当前 ActionLoc 的原理可以概括成一句话：

> 先用冻结的 V-JEPA2 提取窗口级视频特征，再把这些特征压成纯时间序列，最后交给 ActionFormer 做单阶段时序动作定位。

这条路线的关键代码是：

- 主训练逻辑在 `evals/video_actionloc_frozen/eval.py:176-549`；

- 数据集与窗口采样逻辑在 `evals/video_actionloc_frozen/dataloader.py:315-788`；

- ActionFormer 头的初始化在 `evals/video_actionloc_frozen/models.py:29-113`；

- 你 vendored 的 ActionFormer 核心 meta arch 在 `actionformer_release/libs/modeling/meta_archs.py:162-409`；

- README 里对这条集成路线的说明在 `README.md:353-411`。

### 2. 这条 ActionLoc 命令是怎么运行起来的

命令入口和 ActionFilter 相同，仍然从 `evals/main.py:16-113` 进入，再通过 `evals/scaffold.py:13-19` 跳到 `evals/video_actionloc_frozen/eval.py:176-549`。

最大的区别是具体任务逻辑不同：

- ActionFilter 在 `evals/video_classification_frozen/eval.py:49-332`；

- ActionLoc 在 `evals/video_actionloc_frozen/eval.py:176-549`。

配置文件 `configs/eval/vitl/actionloc_robotics.yaml:1-76` 的关键信息是：

- `eval_name: video_actionloc_frozen`，见 `configs/eval/vitl/actionloc_robotics.yaml:6`；

- `tag: actionloc-robotics`，见 `configs/eval/vitl/actionloc_robotics.yaml:5`；

- train jsonl：`configs/eval/vitl/actionloc_robotics.yaml:18`；

- val jsonl：`configs/eval/vitl/actionloc_robotics.yaml:19`；

- `single_class: true`，见 `configs/eval/vitl/actionloc_robotics.yaml:24`；

- `frames_per_clip: 64`，见 `configs/eval/vitl/actionloc_robotics.yaml:26`；

- `frame_step: 4`，见 `configs/eval/vitl/actionloc_robotics.yaml:27`；

- `num_segments: 3`，见 `configs/eval/vitl/actionloc_robotics.yaml:28`；

- `num_epochs: 20`，见 `configs/eval/vitl/actionloc_robotics.yaml:36`；

- `batch_size: 16`，见 `configs/eval/vitl/actionloc_robotics.yaml:37`；

- 3 组 `multihead_kwargs`，见 `configs/eval/vitl/actionloc_robotics.yaml:40-58`。

所以当前 ActionLoc 不是 20 头，而是 **3 个 ActionFormer 头并行训练**。

### 3. ActionLoc 的模型结构

这条路线依然可以拆成两部分。

#### 3.1 冻结的 V-JEPA2 编码器

ActionLoc 的 encoder 初始化在 `evals/video_actionloc_frozen/models.py:8-26`，它内部直接复用了 `video_classification_frozen` 的 encoder 初始化逻辑，也就是还是：

- 读 `pretrained_vjepa2_checkpoints/vitl.pt`；

- 构造 `vit_large`；

- 加载 `target_encoder`；

- 冻结参数。

所以 ActionLoc 与 ActionFilter 共用同一个 frozen backbone 思路。

#### 3.2 ActionFormer 定位头

真正的定位头初始化在 `evals/video_actionloc_frozen/models.py:29-113`。这里默认创建的是：

- `meta_arch = LocPointTransformer`，见 `evals/video_actionloc_frozen/models.py:39`；

- 对应类定义在 `actionformer_release/libs/modeling/meta_archs.py:162-409`。

它的大体结构是：

1. 一个时序 backbone（默认 convTransformer），见 `actionformer_release/libs/modeling/meta_archs.py:245-279`；

2. 一个 FPN，把不同尺度的时间特征组织起来，见 `actionformer_release/libs/modeling/meta_archs.py:283-294`；

3. 一个 point generator，为不同特征层生成候选点，见 `actionformer_release/libs/modeling/meta_archs.py:296-304`；

4. 一个分类头 `cls_head` 和一个回归头 `reg_head`，分别负责“这里是不是动作”和“动作边界偏移多少”，见 `actionformer_release/libs/modeling/meta_archs.py:306-320`。

这和传统 proposal-based 时序检测不同，它更接近**单阶段、按时间点直接分类和回归边界**。

### 4. ActionLoc 的输入标注格式和数据逻辑

ActionLoc 训练不是 CSV，而是 JSONL。README 里写得很清楚，见 `README.md:367-388`。每一行至少要有：

```json

{

"video_path": "/abs/path/to/video.mp4",

"original_data": {

"fps": 30.0,

"duration": 12.34,

"local_video_path": "/abs/path/to/video.mp4"

},

"result": [

{"start_time": 1.23, "end_time": 2.34, "action_en": "action_name"}

]

}

```

实际代码对应在 `evals/video_actionloc_frozen/dataloader.py:272-300` 和 `README.md:369-388`。

当前你的 robotics 配置里：

- `label_field: action_en`，见 `configs/eval/vitl/actionloc_robotics.yaml:22`；

- `single_class: true`，见 `configs/eval/vitl/actionloc_robotics.yaml:24`。

也就是说，虽然 jsonl 里可以有动作名字，但当前训练时会把所有动作都折叠成**单一动作类**，见 `evals/video_actionloc_frozen/dataloader.py:283-300` 和 `evals/video_actionloc_frozen/dataloader.py:566-574`。

这个设定非常重要，它说明你现在的 ActionLoc 目标不是“分很多动作类别”，而是：

> 只要把动作时间段找出来就行，先不细分类别。

### 5. ActionLoc 的窗口采样和特征构造

这部分是当前 ActionLoc 最关键的设计。

数据集定义在 `evals/video_actionloc_frozen/dataloader.py:315-618`。当前配置是：

- `frames_per_clip: 64`，见 `configs/eval/vitl/actionloc_robotics.yaml:26`；

- `frame_step: 4`，见 `configs/eval/vitl/actionloc_robotics.yaml:27`；

- `num_segments: 3`，见 `configs/eval/vitl/actionloc_robotics.yaml:28`；

- `window_sampling: random`，见 `configs/eval/vitl/actionloc_robotics.yaml:31`。

这意味着一个训练样本不是整视频，而是一个**随机窗口**：

- 每个 clip 有 64 帧；

- 一共 3 个 clip；

- 相邻帧间隔 4；

- 所以一个训练窗口覆盖的原始帧跨度大约是 `64 * 3 * 4 = 768` 帧，见 `evals/video_actionloc_frozen/dataloader.py:504-547`。

随后数据集会把原视频中的标注时间段裁到这个窗口里：

- 把绝对时间减去 `window_start_time`；

- 与窗口边界相交后保留；

- 超出窗口的部分截断；

- 完全不相交的丢弃。

对应逻辑在 `evals/video_actionloc_frozen/dataloader.py:546-580`。

这一步非常关键，因为它决定了你其实是在做：

> 从长视频中采一个随机窗口，让模型学会在这个窗口内找动作起止边界。

### 6. ActionLoc 中 encoder 特征是怎么变成 ActionFormer 输入的

这是这条路线最有意思的地方。

ActionLoc 并不是直接把 V-JEPA2 的全部时空 token 喂给 ActionFormer，而是先把 token 压成纯时间序列。核心逻辑在 `evals/video_actionloc_frozen/eval.py:81-95` 和 `evals/video_actionloc_frozen/eval.py:583-599`。

函数 `_tokens_to_temporal(...)` 做了这件事：

- 先把 token reshape 成 `[B, T_total, S, D]`；

- 再对空间维 `S` 做均值池化；

- 得到 `[B, T_total, D]` 的纯时序特征。

对当前配置来说：

- 每个 clip 有 64 帧；

- `tubelet_size=2`，所以每个 clip 对应 `64 / 2 = 32` 个时间 token；

- 一共有 3 个 clip，所以 `T_total = 32 * 3 = 96`；

- encoder 的 `D = 1024`。

因此 `_tokens_to_temporal(...)` 之后的大致张量是：

- `[B, 96, 1024]`

接着它会被 permute 成：

- `[B, 1024, 96]`

再交给 ActionFormer，见 `evals/video_actionloc_frozen/eval.py:597-598`。

这可以理解为：

> V-JEPA2 负责提“视频语义特征”，ActionFormer 负责在时间维上做“边界建模和定位”。

### 7. ActionLoc 的监督与训练方式

训练循环在 `evals/video_actionloc_frozen/eval.py:458-549`，单个训练 epoch 在 `evals/video_actionloc_frozen/eval.py:552-671`。

一个 step 的逻辑是：

1. 取出窗口样本 `(clips, targets, clip_indices)`，见 `evals/video_actionloc_frozen/eval.py:579-581`；

2. 用 frozen encoder 提特征，见 `evals/video_actionloc_frozen/eval.py:583-599`；

3. 把 GT 的秒级 segment 转成 ActionFormer 的时间网格坐标，见 `evals/video_actionloc_frozen/eval.py:98-103` 和 `evals/video_actionloc_frozen/eval.py:600-614`；

4. 把 `feats + segments + labels + fps + duration + feat_stride` 组成 `video_list`；

5. 调 ActionFormer 前向，训练时返回 losses，见 `evals/video_actionloc_frozen/eval.py:622-624` 和 `actionformer_release/libs/modeling/meta_archs.py:360-379`；

6. 对 ActionFormer 头反向传播，encoder 仍然冻结，见 `evals/video_actionloc_frozen/eval.py:626-639`。

和 ActionFilter 一样，这里 backbone 还是 frozen 的；但和 ActionFilter 不同的是，监督不再是一个视频级类别，而是一组时间段边界。

### 8. ActionLoc 的评估指标和推理方式

验证逻辑在 `evals/video_actionloc_frozen/eval.py:674-773`。它会：

- 先跑 loss；

- 再切到 eval 模式跑解码；

- 最后基于 tIoU 计算 mAP。

关键指标函数在 `evals/video_actionloc_frozen/eval.py:106-163`，调用了 `compute_average_precision_detection`，见 `evals/video_actionloc_frozen/eval.py:18`。

也就是说，ActionLoc 的核心指标不是 accuracy，而是：

> 不同 tIoU 阈值下的 detection mAP。

推理脚本主要有两个：

- 对 jsonl 输入做定位：`app/infer_actionloc.py:162-470`；

- 对目录里的长视频做滑窗定位：`app/infer_actionloc_by_dir.py:427-1228`。

长视频推理时的逻辑是：

1. 先把长视频按固定窗口滑动，见 `app/infer_actionloc_by_dir.py:686-694`；

2. 每个窗口跑 frozen encoder + ActionFormer，见 `app/infer_actionloc_by_dir.py:700-745`；

3. 把窗口内预测的相对时间转回绝对时间，见 `app/infer_actionloc_by_dir.py:765-787`；

4. 最后按 IoU 做合并/NMS，见 `app/infer_actionloc_by_dir.py:309-332` 和 `app/infer_actionloc_by_dir.py:898-921`。

### 9. 你现在这种 ActionLoc 做法的优点

当前做法的优点非常明确：

1. **结构拆分很合理**。V-JEPA2 负责强表征，ActionFormer 负责时间定位，这是经典的“特征提取 + 检测头”解耦设计。

2. **相较直接端到端训练更稳**。定位任务难、标注也贵，冻结大 backbone 能减少训练不稳定。

3. **窗口裁剪 + GT 裁切的训练方式比较贴近实际长视频使用场景**，见 `evals/video_actionloc_frozen/dataloader.py:546-580`。

4. **single_class 设定对 robotics 场景很实用**。如果当前主要目标是先把“动作片段”找出来，而不是细分类别，这样更容易学。

5. **ActionFormer 本身适合 temporal localization**。它天然是单阶段边界回归，不需要额外 proposal 系统。

6. **mAP 评估比 acc 更贴任务本质**。这点比 ActionFilter 更接近真实使用价值。

7. **已经具备长视频推理链路**。`app/infer_actionloc_by_dir.py:427-1228` 基本把工程闭环补全了。

### 10. 你现在这种 ActionLoc 做法的缺点

缺点同样很现实：

1. **空间信息被均值池化丢掉了**。`_tokens_to_temporal(...)` 直接对空间 token 求均值，见 `evals/video_actionloc_frozen/eval.py:81-95`。这能简化问题，但会损失动作发生位置、主体显著性等空间线索。

2. **当前是单类定位，不能区分动作类型**。`single_class: true` 让训练更简单，但也限制了模型表达能力。

3. **窗口随机采样可能不总能覆盖完整动作**。尤其长动作或跨窗口动作，训练时可能经常被截断。

4. **验证集 batch_size=1，整体吞吐会偏低**，见 `configs/eval/vitl/actionloc_robotics.yaml:38`。

5. **依赖视频解码，数据瓶颈明显**。相比直接读预提特征，在线解码会重很多。

6. **当前配置只给 ActionFormer 三组超参，搜索空间还比较窄**。

7. **encoder 仍然完全冻结**。如果 robotics 视频和预训练域差异大，时间定位上限可能被特征瓶颈卡住。

### 11. ActionLoc 最值得改进的点

如果要提升 ActionLoc，我会建议优先看这些方向：

1. **优先解决训练数据与窗口采样策略的匹配问题**。当前 `window_sampling: random`，见 `configs/eval/vitl/actionloc_robotics.yaml:31`。这对泛化有帮助，但也可能让长动作被切碎。可以尝试增加 `center` / `start0` / curriculum sampling，或混合长短窗口。

2. **从单类定位升级到多类定位**。如果业务后面确实需要动作类型，这一步一定要做。当前 `single_class: true` 是把问题先做简单了，但长期会限制能力上限。

3. **不要只做空间均值池化**。`_tokens_to_temporal(...)` 现在把空间全部平均，后续可以尝试：

- attention pooling；

- ROI/主体感知池化；

- 保留少量空间 token 再做时空联合建模。

4. **尝试部分解冻 encoder 或加 Adapter/LoRA**。特别是 robotics 场景，如果动作形态和公开视频差异明显，这一步很可能有收益。

5. **增大 ActionFormer 配置搜索空间**。当前默认 `actionformer: {}`，具体超参全靠 `evals/video_actionloc_frozen/models.py:39-111` 的默认值。可以系统地扫：

- `embd_dim`

- `fpn_dim`

- `backbone_arch`

- `iou_threshold`

- `pre_nms_topk`

- `max_seg_num`

6. **训练和推理都可以增强长时建模**。现在 `actionloc_robotics_long` 配置已经把 `num_segments` 从 3 提到 6，见 `configs/eval/vitl/actionloc_robotics_long.yaml:28`，这说明你已经意识到长上下文的重要性。这条方向是正确的。

7. **做更完善的 error analysis**。建议看这些分桶：

- 短动作 vs 长动作

- 单动作视频 vs 多动作视频

- 动作边界模糊 vs 边界清晰

- 解码失败/帧率异常视频

8. **推理阶段的 NMS / merge 需要标定**。当前目录推理里会做 `_merge_segments(...)`，见 `app/infer_actionloc_by_dir.py:309-332`。这里的 `merge_iou` 和 `min_score` 都值得在验证集上系统调。

9. **把数据瓶颈和模型瓶颈分开看**。当前视频在线解码很重，训练慢并不一定是模型慢。可以考虑缓存中间特征，或预提 temporal features 做二阶段训练。

### 12. ActionLoc 相比 ActionFilter 的优劣对比

如果把两条路线放一起看：

- `ActionFilter` 更像**粗筛器**。它简单、便宜、稳定，适合先判断“这个视频值不值得继续处理”。

- `ActionLoc` 更像**精筛器/精定位器**。它更复杂、训练更重，但能给出具体动作起止时间。

因此，从系统设计角度，你现在这两条路线其实是很互补的：

1. 先用 ActionFilter 快速过滤掉明显无动作的视频；

2. 再对剩余视频使用 ActionLoc 做时间定位。

这个搭配的优点是整体算力更省、流程更工程化；缺点是如果 ActionFilter 漏掉真正有动作的视频，后面的 ActionLoc 就永远看不到它们，所以 ActionFilter 阈值一定要偏保守并认真校准。

### 13. 总结：你现在这两条路线的整体判断

你的当前方案其实已经很像一个完整的视频动作处理系统雏形：

- `ActionFilter` 负责做视频级粗筛；

- `ActionLoc` 负责做时间段级精定位；

- 两者都建立在冻结的 V-JEPA2 表征上，因此工程复用度高。

从优点看，这套做法稳定、清晰、易扩展，也很适合从“先做出来、先跑起来”的角度推进；从缺点看，它的主要问题集中在：

- backbone 全冻结带来的上限问题；

- 训练/验证与真实推理之间的一致性问题；

- 指标体系和阈值校准还不够业务化；

- ActionLoc 还在单类定位、空间信息也被压得太狠。

如果后续只选最重要的三件事，我建议优先顺序是：

1. **先把 ActionFilter / ActionLoc 的验证与指标体系做扎实**；

2. **再做阈值和推理参数校准，让系统真的可上线**；

3. **最后再尝试 partial unfreeze / Adapter / 更强时空头，去追模型上限**。