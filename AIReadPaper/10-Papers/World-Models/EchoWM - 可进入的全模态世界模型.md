---
type: paper-note
status: complete
domain: World-Models
paper: "EchoWM: Open and Enterable Omnimodal World Models"
year: 2026
arxiv: 2608.23189
doi: 10.48550/arXiv.2608.23189
source: https://arxiv.org/abs/2608.23189
project: https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/wm
code: https://github.com/jd-opensource/JoyAI-Echo
tags:
  - World-Models
  - Omnimodal
  - 6DoF
  - Audio-Visual-Generation
created: 2026-10-16
updated: 2026-10-16
---

# EchoWM：可进入的全模态世界模型

## 基本信息

- **作者/机构**：Songchun Zhang、Yaowei Li、Junhao Zhuang 等；JD Joy Future Academy 联合 HKUST、PKU、HKU、THU、Stanford 等
- **版本**：arXiv v1，2026-08-24；42 页、24 图
- **底座**：论文明确采用 LTX-2.3 系列联合音视频 diffusion transformer，并增加 UCPE 相机分支与自回归后训练。
- **输出**：720p 视频，以及同步的环境声、音乐、画内/画外语音；支持第一/第三人称、连续导航和多 turn 延展。

## 一句话结论

EchoWM 的核心不是把键盘动作直接编码进视频模型，而是把“用户希望如何观察世界”抽象成统一的度量相对 6DoF 相机意图，再让第一/第三人称的相机—角色耦合从数据中学习；该设计在 WBench 和 SANA-WM-Bench 上提供强控制与视觉持续性，但音频只通过联合生成间接响应轨迹、动作语义局限于相机运动，且仍缺少显式持久 3D 世界记忆。

---

## 1. 研究背景与问题定义

### 1.1 “可进入”比“可控视频”多了什么

普通可控视频在单个 clip 内执行 prompt 或相机条件；可进入世界要求用户在生成过程中连续改变导航，模型还要保持环境、角色、声场和历史状态。现有世界模型多为无声视觉 rollout，或绑定游戏特定动作词表。EchoWM 试图用一个 camera-intent 接口同时覆盖开放域第一人称观察、第三人称角色跟随、原生音视频生成与多 turn continuation。

### 1.2 任务形式

给定媒体上下文 $\mathcal M$（空、参考图或同步音视频前缀）、结构化文本 $\mathcal C$ 和控制 $U_{1:T}$，先构造校准后的轨迹 $P_{1:T}$，联合建模

$$p_{\theta,\phi}(V_{1:T},A_{1:T}\mid\mathcal M,\mathcal C,P_{1:T}).$$

其中 $\theta$ 是音视频主干，$\phi$ 是轨迹条件分支。静态文本定义场景、风格、主体和初始 viewpoint；$P_{1:T}$ 只定义随时间变化的相机运动。

### 1.3 关键设计选择

- 用**已实现的相机运动**而不是 raw action 训练，因为动作日志稀缺，且同一输入遇到碰撞或控制器平滑后可产生不同位移。
- 第一/第三人称共用 camera-intent；第三人称角色怎么跟随相机不是手写控制器，而是数据先验。
- 音频保留在联合世界生成中，不建立显式 action-to-sound 分支。
- 将 A/V prior、轨迹学习和联合巩固分成三个训练阶段，减少数据分布冲突。

---

## 2. 总体架构

![](assets/EchoWM%20-%20可进入的全模态世界模型/figures/01-teaser.png)

> **图 1（论文 Figure 1）**：一套模型同时展示开放域第一人称导航、第三人称相机—主体控制、环境声/音乐/语音联合生成和多 turn 继续/切换。来源：官方 arXiv LaTeX 图源，论文第 2 页。

![](assets/EchoWM%20-%20可进入的全模态世界模型/figures/02-method-overview.png)

> **图 2（论文 Figure 3）**：输入经共享 camera-intent 接口变成相对 6DoF 轨迹并做全局尺度校准；UCPE 分支只注入视频流，A/V cross-attention 联合生成音视频；训练依次为 AV-CPT、Action-SFT、Joint-FT 和 AR post-training。来源：官方 arXiv LaTeX 图源，论文方法章节。

### 2.1 统一相机意图

相机到世界位姿 $T_t\in SE(3)$ 相对首帧表示为

$$\Delta T_t=T_0^{-1}T_t.$$

键盘状态 $\mathbf u_t$ 经固定映射 $M$ 变成相机局部 6DoF twist，并累计为

$$\boldsymbol\xi_t=M\mathbf u_t,\qquad T_t=T_{t-1}\operatorname{Exp}(\widehat{\boldsymbol\xi}_t),\qquad \Delta T_t=T_0^{-1}T_t.$$

连续 metric pose 绕过 $M$，但走同一坐标转换和相对化。训练因此不依赖 WASD 的具体布局，推理端仍可把任何控制设备适配到相机增量。

第一人称时轨迹近似观察者 ego-motion；第三人称时轨迹约束相机，角色运动与跟随关系由数据决定。这里的“controller-agnostic”只对导航/视点成立，不应扩张为通用动作世界模型。

### 2.2 全局尺度校准

相对位姿仍有平移尺度问题。逐 clip 归一化会把快/慢、近/远运动压成同一幅度，逐 chunk 归一化还会在续写边界产生速度跳变。EchoWM 取每条训练 clip 最大位移 $m_i$ 的 90% 分位数：

$$m_i=\max_k\|\Delta\mathbf t_{i,k}\|_2,\qquad s_{\mathrm{global}}=Q_{0.9}(\{m_i\}_{i\in\mathcal D_{\mathrm{train}}}),$$

$$\widehat{\mathbf t}_{i,k}=\Delta\mathbf t_{i,k}/s_{\mathrm{global}}.$$

超出范围的轨迹被过滤而不是截断，旋转保持不变；训练和推理复用同一个 $s_{\mathrm{global}}$。这保留了速度控制的数值语义，但并不等于模型获得绝对米制世界坐标。

### 2.3 UCPE 几何注入

Plücker ray 的 moment 项显式依赖世界原点。EchoWM 采用 ray-local UCPE：每个时空 token 根据校准 pose 和内参构造局部 ray frame，用相对 ray transform 修改并行 camera attention 的 Q/K/V；分支输出经 zero-init 线性层加回原视频 self-attention。几何分支共享于第一/第三人称，viewpoint 文本提供初始观察者—主体关系。

轨迹只直接作用于视频 token；音频对动作的响应来自已有 A2V/V2A cross-attention 和 Joint-FT，而非单独监督的 action-to-audio 控制器。论文对此表述谨慎：它声称“交互中的声学一致性”，不声称确定性的动作声效模拟。

---

## 3. 世界数据引擎

### 3.1 四类互补数据

1. **内部 gameplay**：程序化控制、action log、高分辨率 RGB、原生立体声和元数据；用 ReShade 插件去 HUD，但需从视频恢复 pose。
2. **人类互联网 gameplay**：无 action log，但有更自然的控制节奏、玩家反应、对白和评论音频。
3. **UE simulation**：一人称头戴相机加四个角色相对第三人称相机，具真值 metric pose、内参和键盘状态；缺少自然音频。
4. **通用互联网视频**：扩展真实/电影画面、声学和复杂 camera-subject 关系；主要提供 A/V prior，少量通过 pose 筛选后进入控制训练。

论文对 control-clean mixture 抽样 28,605 条轨迹做覆盖分析，包含直线、斜线、回环和环绕，但没有披露总视频小时数和各数据源比例。

### 3.2 长窗位姿恢复

非 UE 视频先保留约一分钟连续窗口，再用 ViPE 联合 VGGT-Omega 的时序几何与 MoGe-2 的 metric depth 约束优化 pose/内参。稀疏轨迹用 SLERP 插值旋转、线性插值平移，然后再切成短训练 clip。先长窗重建、后切 clip，可让邻近 clip 共享几何坐标并提供更长 baseline。

过滤检查重建可靠性、内参稳定性、帧间抖动、突变、退化运动和尺度异常。即使如此，互联网视频的 pose 仍可能被 motion blur 或场景内容系统性影响。

### 3.3 防止文本运动泄漏

结构化标注包含 scene、style、viewpoint、subject、narrative、speech、sound。Action-SFT 删除 narrative、speech、sound，只保留静态字段，使轨迹成为唯一运动来源；Joint-FT 恢复 speech 和 sound，但继续删除 narrative。这个设计是论文中少见且重要的“反 shortcut”措施：若 caption 已写明“向左移动/镜头环绕”，模型无需读取轨迹也能生成看似正确的运动。

---

## 4. 训练与长程生成

### 4.1 三阶段主训练

- **AV-CPT**：关闭轨迹分支，在 AV-rich mixture 上更新完整主干；混合空条件、参考帧和同步 A/V prefix，学习文本生成、首帧生成和连续音视频续写。
- **Action-SFT**：冻结主干，仅训练 camera branch；使用 control-clean mixture 和视觉 denoising loss，并强采样参考帧以固定场景内容。
- **Joint-FT**：在 pose 与音视频都可靠的小型 balanced mixture 上，以低学习率联合更新主干和轨迹分支。

这种课程把“学会世界长什么/听起来怎样”与“学会按轨迹移动”分开，再小步合并，避免控制数据的弱音频和 A/V 数据的弱几何互相污染。

### 4.2 音视频同步机制

视频与音频 latent 划分为覆盖相同物理时间的 macro-chunk，每对 chunk 共享 diffusion noise level。teacher forcing 的因果可见性为

$$\mathcal S(\mathcal B_{i,\mathrm{noisy}})=\{\mathcal B_{j,\mathrm{clean}}:j<i\}\cup\{\mathcal B_{i,\mathrm{noisy}}\}.$$

相同 mask 同时作用于视频自注意力、音频自注意力、A2V、V2A 和 UCPE，避免任何一路获得未来信息。同步来自联合时钟和跨模态注意，而不是后置音轨拼接。

![](assets/EchoWM%20-%20可进入的全模态世界模型/figures/03-audio-visual-results.png)

> **图 3（论文 Figure 16）**：四组样例同时展示时间戳帧、波形和频谱，覆盖黑洞环境声/语音、室内音乐与环境声、角色低语和宫殿风声/鸟鸣。图可证明模型确实原生输出多类音频，但仅凭可视化不能量化声画事件延迟。来源：官方 arXiv LaTeX 图源，论文定性实验章节。

### 4.3 SGF 与四步 EchoWM-Flash

teacher forcing 只消费真值历史。SGF 的 Forward 1 以部署 sampler 无梯度生成完整 chunk 序列；Forward 2 把已生成 clean latent 当 stop-gradient 历史，重算 K/V 与上下文注意力，使后续 loss 能监督早期 chunk 如何写入因果记忆。DMD 的 real-score 冻结、fake-score 跟踪学生分布，把多步生成蒸馏成四步 EchoWM-Flash。

长程 SGF 串联多个短 segment，使用 sink + FIFO bounded cache；每个 segment 都计算 DMD，而不是只监督终点。它优化的是对累积错误历史的鲁棒性，不是显式 3D 地图记忆。

### 4.4 两种多 turn 模式

- **双向 continuation**：重编码上一 segment 的同步音视频尾窗，作为下一 turn 的 clean prefix。
- **因果 streaming**：持续维护视频、音频和 camera-attention KV cache；sink 保存早期前缀，local window 保存最近历史，淘汰旧 token 后重基 temporal RoPE 与 camera encoding。

二者共享 6DoF 接口，但历史管理方式不同。sink 不是完整持久 memory：离开 FIFO 且不在 sink 的地点仍会丢失。

---

## 5. 实验结果

### 5.1 WBench

在 158 个 Navigation case 上，EchoWM 平均 81.7、排名第一；Quality 81.5、Setting 79.4、Interaction 87.2、Consistency 89.8、Physical 70.6。四步 EchoWM-Flash 平均 81.0，Interaction 87.9 为最高，但 Consistency 77.5，显示蒸馏后即时控制保留得比世界状态更好。

值得注意的是 Physical 70.6 明显低于其 Interaction/Consistency，且并非榜首，说明“轨迹跟随好”不代表物理规律可靠。

### 5.2 SANA-WM-Bench

短程 241 帧时，EchoWM 在 Simple/Hard 的 VBench 为 83.91/83.96。未蒸馏模型在 961 帧 Simple split 的 VBench 81.36、回访 PSNR 15.10 dB；旋转误差从 241 帧的 0.523° 增至 961 帧的 3.223°，Hard split 达 12.05°。

Flash 在 961 帧 Simple/Hard 的 VBench 为 80.13/81.06，旋转误差 5.009°/14.221°，在对比因果模型中整体领先。其 $\Delta IQ$ 为 2.09/1.30，并非最小；论文正确指出，低起点模型也可能因为“一直差”而具有较小质量下降，因此应结合绝对 IQ 解读。

### 5.3 人评

200 个 case 对 LingBot-World-v2，EchoWM overall preferred 46.50%，对方 21.57%；对 HappyOyster，EchoWM 63.13%，对方 27.06%。但对 HappyOyster 的 semantic following，EchoWM 13.63% 对 29.44%；motion 18.94% 对 24.56%。论文并非全维度领先，优势主要集中于时空一致性与视觉美学。

### 5.4 消融

论文称消融覆盖 scale、camera-intent semantics、inference speed 和 A/V task mixture，但正式结果主要为**定性曲线/案例**：固定全局尺度比 baseline 更平滑地接近主体；不同轨迹幅度能控制生成速度；显式 viewpoint 语义有助于第三人称跟随。没有给出移除全局校准、UCPE、Action-SFT 或 long-SGF 后的完整数值表，且 AV-CPT task-mixture 对照仍以注释形式保留，未形成最终实验。

因此主 benchmark 证明“完整系统强”，却不足以精确归因每个模块。最需要补的是 global vs per-clip scale 的数值误差、UCPE vs Plücker 的同数据同参数比较，以及 TF/short-SGF/long-SGF 的长程递进表。

---

## 6. 真实空间泛化证据

论文展示对未见 2D/2.5D 游戏和机器人环境的定性泛化，并在 WBench/SANA-WM 上评估回访、轨迹恢复与第三人称交互。证据强度可分三层：

1. **较强**：公开 benchmark 上从生成视频重估 pose，与目标轨迹比较；961 帧回访 PSNR/SSIM/LPIPS 检验同位姿再现。
2. **中等**：200 例人评覆盖游戏、机器人、自然室外、家庭室内和驾驶场景。
3. **较弱**：未见域主要是精挑 qualitative 样例，没有独立的真实机器人闭环或真实相机位姿 benchmark。

所以它证明了“跨视觉域的可控生成”，尚未证明具身系统里的真实闭环空间泛化或可用于规划的状态准确性。

---

## 7. 局限与批判性评价

### 7.1 论文明确局限

- 仅建模可化为相机运动的导航/视点动作；不支持攻击、跳跃、操控或机器人命令。
- 不具备游戏引擎的显式规则、碰撞状态和确定性转移。
- 没有显式持久 3D memory，长程继续时几何、身份、世界状态和音频仍会漂移。
- 交互范围受全局轨迹过滤器限制，超大平移、高速或异常旋转的泛化未建立。
- 恢复 pose 仍有噪声与内容相关偏差。

### 7.2 我的判断

camera intent 是一个很有效的“最小公共控制层”：它比动作 token 更容易从海量视频获得，又比文本 motion caption 更可度量。但它也把世界交互压缩成观察者运动；第三人称中看似“角色受控”的效果本质是模型根据相机轨迹猜测合理的角色运动，无法保证动作因果唯一性。

音视频联合生成是明显优势，但 trajectory 不直接进入 audio branch，意味着脚步/碰撞声与控制的关系靠相关性学习。若目标是仿真或机器人训练，应增加事件级 action-to-sound 评测与接触状态，而不能只依赖音频美学和定性频谱。

---

## 8. 复现建议

### 8.1 数据侧

优先复现“约一分钟长窗 pose recovery 后再切短 clip”，并保留 action intent 与 realized trajectory 两条信号。严格删除 motion narrative，做 text-only/control-only 诊断，避免模型通过 caption 猜运动。

### 8.2 模型侧

1. 从联合 A/V backbone 开始，先验证 synchronized-prefix continuation。
2. 仅新增 zero-init UCPE branch，在冻结主干条件下学习 trajectory residual。
3. 固定全局尺度并保存训练统计，推理禁止重新归一化。
4. 再做 Joint-FT 和 TF→short SGF→long SGF；每阶段保存相同 benchmark checkpoint。
5. 最低评测集应包含硬旋转、回环、第三人称 orbit、静态场景高速移动和有/无音频事件。

### 8.3 复现成本

官方代码已公开，但训练数据主要依赖内部 gameplay、UE 管线和大规模互联网视频。论文未给出数据总量、GPU 数、batch size、学习率或完整 sampling ratio，难以精确复刻。可行路线是先复现公开数据上的 UCPE + global scale + anti-leak curriculum，再单独验证 SGF。

---

## 9. 后续问题

1. 能否加入可检索的 3D/object memory，让 revisit 不依赖 sink 恰好保留首段？
2. camera intent 与 actor intent 能否分层建模，避免第三人称角色动作完全由生成先验猜测？
3. A/V 同步能否用事件检测器计算 action onset 到 sound onset 的误差？
4. 6DoF scale 能否由 metric depth 或真实传感器锚定为绝对米制，而不只是数据集归一化单位？

---

## 10. 材料来源

- 官方论文：<https://arxiv.org/abs/2608.23189>
- 项目页：<https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/wm>
- 代码：<https://github.com/jd-opensource/JoyAI-Echo>
- 本地来源与图号映射：[[assets/EchoWM - 可进入的全模态世界模型/sources]]
