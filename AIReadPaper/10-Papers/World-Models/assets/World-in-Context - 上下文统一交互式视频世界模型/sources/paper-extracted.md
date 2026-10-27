# World-in-Context: Interactive Video World Modeling with In-Context Conditioning

> 来源: https://arxiv.org/pdf/2610.33169

**作者**: Harold Haodong Chen, Rui Peng, Rongjin Guo, Lingteng Qiu, Kejie Qiu, Chao Xu, Zilong Dong

## 摘要（arXiv 官方）

Interactive video world models require coordinated control over viewing perspectives, character actions, and camera motion. Existing methods typically leave view modes implicit in text captions, conflate action with camera motion, and inject controls through fragmented architectural pathways that capture interdependencies only indirectly. In this work, we present World-in-Context, an interactive world modeling framework powered by In-Context Conditioning (ICC). Rather than relying on specialized conditioning pathways, World-in-Context unifies view, action, and camera controls into explicit tokens contextualized directly within a shared self-attention space. Our formulation reveals that (i) explicit view tokens disentangle conflicting dynamics and unlock emergent view switching from single-view data, while (ii) action and camera inputs play complementary roles in governing local responsiveness and global stability. To preserve temporal continuity across execution intervals, we further introduce Interaction-Transport RoPE (IT-RoPE), parameterizing temporal attention with accumulated control transitions. On WBench, World-in-Context-5B achieves navigation performance on par with 22B EchoWM ($81.5$ vs. $81.6$) and attains a $93.8\%$ perspective-switching success rate, substantially outperforming existing 5B baselines ($18.8\%$). Controlled analyses further demonstrate compounding advantages of World-in-Context under richer conditioning mixtures, favorable data efficiency, and natural applicability to causal rollouts.

World-in-Context: Interactive Video World Modeling with
In-Context Conditioning
Alibaba Token Hub, Alibaba Group
 https://haroldchen19.github.io/WiC-site
§ https://github.com/aigc3d/World-in-Context
Free-Camera
First-Person
Third-Person
TPP->FPP
FPP->TPP
Figure 1. World-in-Context supports versatile interactive video generation across FREE, FPP, and TPP regimes, supporting
dynamic in-video view switching under joint action and camera control.
Abstract
Interactive video world models require coordinated control
over viewing perspectives, character actions, and camera
motion. Existing methods typically leave view modes im-
plicit in text captions, conflate action with camera motion,
and inject controls through fragmented architectural path-
ways that capture interdependencies only indirectly.
In
this work, we present World-in-Context, an interactive
world modeling framework powered by In-Context Condi-
tioning (ICC). Rather than relying on specialized condi-
tioning pathways, World-in-Context unifies view, ac-
tion, and camera controls into explicit tokens contextual-
ized directly within a shared self-attention space. Our for-
mulation reveals that (i) explicit view tokens disentangle
conflicting dynamics and unlock emergent view switch-
ing from single-view data, while (ii) action and camera
inputs play complementary roles in governing local re-
sponsiveness and global stability.
To preserve temporal
continuity across execution intervals, we further intro-
duce Interaction-Transport RoPE (IT-RoPE), parameter-
izing temporal attention with accumulated control tran-
sitions.
On WBench, World-in-Context-5B achieves
navigation performance on par with 22B EchoWM (81.5
vs. 81.6) and attains a 93.8% perspective-switching suc-
cess rate, substantially outperforming existing 5B baselines
(18.8%).
Controlled analyses further demonstrate com-
pounding advantages of World-in-Context under richer
conditioning mixtures, favorable data efficiency, and natu-
ral applicability to causal rollouts.
1. Introduction
Video generation has rapidly advanced from pas-
sive video generation [10, 22, 41] toward interactive
world models [9, 38, 40, 45], empowering users to
actively steer dynamic visual environments. In this
paradigm, the model acts as an interactive simulator
where users provide control inputs to navigate and
explore the environment, while the model dynami-
1
arXiv:2610.33169v1  [cs.CV]  23 Oct 2026

cally predicts the evolving visual observations in re-
sponse. At its core, effectively simulating such an in-
teractive world hinges on two fundamental dimen-
sions [28, 43]: ➀how the world is viewed across distinct
perspectives, and ➁how user inputs concurrently govern
character dynamics and camera viewports. Together, these
dimensions define how an agent actively perceives
and intervenes in an unfolding generative world.
Despite recent progress [15, 39, 40, 55, 59], current in-
teractive pipelines often overlook the systematic for-
mulation of these two essential dimensions: ❶view-
mode ambiguity: first- (FPP) and third-person (TPP)
perspectives increasingly appear in interactive data,
yet view mode is often specified only implicitly
through text captions [9, 38, 39, 55], while free-camera
(FREE) roaming is frequently conflated with FPP con-
trol despite their different motion semantics; and
❷action-camera entanglement: most pipelines rely
primarily on continuous camera trajectories alone for
control [13, 31, 39, 55], overlooking the distinct role of
discrete actions, especially in third-person interaction
where character and camera motion form separate de-
grees of freedom.
A naive remedy would be simply attaching auxiliary
condition channels for each missing signal.
How-
ever, existing paradigms [9, 19, 33, 40, 45, 56] typi-
cally accommodate interactive controls through sep-
arate modules (e.g., AdaLN modulation), which tends
to isolate different control signals and leaves their
cross-dependencies to be captured only indirectly, as
shown in Figure 2 (Left). As interaction becomes richer
and more compositional, coordinating these modu-
lar branches grows increasingly intricate, especially
when the interpretation of one signal inherently de-
pends on concurrent inputs and the active viewport.
In this work, we propose World-in-Context, which
unifies multi-dimensional interaction through an In-
Context Conditioning (ICC) paradigm.
Rather than
routing controls into isolated architectural branches,
World-in-Context casts diverse interactive signals
into token sequences and prepends them directly into
the shared self-attention space alongside visual la-
tents.
Within this unified interface, we revisit the
aforementioned limitations along two fronts: ♣Ex-
plicit view-mode modeling: Rather than leaving per-
spectives implicit in text captions [9, 38, 40, 59], we ex-
plicitly incorporate discrete view tokens (FREE, FPP,
and TPP) into the context. Joint training across these
regimes distinguishes their motion semantics, while
enabling in-video view switching to emerge naturally.
♠Joint action-camera modeling: We jointly incorpo-
rate discrete actions and continuous camera trajecto-
ries into the context. Crucially, beyond their neces-
sary separation in TPP interaction, we further study
the less-examined FPP and FREE regimes, where cam-
era trajectories often stand in for both forms of control
[15, 31, 39]. Our analysis reveals complementary roles:
action governs local command fidelity, whereas camera
anchors global trajectory stability.
Built upon ICC, World-in-Context natively captures
cross-condition dependencies within a unified atten-
tion space.
While in-context tokens specify what
controls are present at individual frames, naive to-
ken conditioning can still leave the dynamic evolu-
tion between distant moments to be modeled im-
plicitly.
To this end, we further introduce IT-RoPE
(Interaction-Transport RoPE), a lightweight positional
encoding that incorporates pairwise control transi-
tions directly into temporal attention. By jointly mod-
eling control conditions and their temporal evolu-
tion, World-in-Context supports coherent interaction
across viewing regimes and makes effective use of in-
teractive supervision, with increasing benefits as con-
trols become richer. More broadly, it reframes multi-
condition interaction as context modeling, offering a
unified interface for interactive world modeling.
To summarize, this work contributes threefold:
➢We
introduce
In-Context
Conditioning
(ICC)
for
interactive
world
modeling
and
present
World-in-Context, which represents view, action,
and camera controls as tokens within a shared
attention space, providing a unified interface for
diverse interaction signals.
➢We investigate the often-overlooked roles of view-
ing regimes and joint action-camera control, reveal-
ing distinct view-dependent dynamics, emergent
view switching, and complementary action-camera
roles in command fidelity and trajectory stability.
➢Extensive experiments on WBench demonstrate
that
World-in-Context-5B
matches
the
22B
EchoWM in navigation (81.5 vs.
81.6) and sub-
stantially outperforms existing 5B baselines in
perspective switching (93.8% vs.
18.8%).
Con-
trolled studies further confirm compounding gains
of World-in-Context under richer controls, favor-
able sample efficiency, and causal applicability.
2. Related Work
Interactive Video World Models.
Interactive world
models extend video generation toward closed-loop
interaction, where future observations evolve accord-
ing to user-steered conditions [3, 6, 32, 36, 45]. Fol-
lowing recent formulations [28, 43], existing mod-
els mainly navigate two core dimensions: (i) How
the world is viewed. Interactive environments span
FREE, FPP, and TPP regimes. Prior works [9, 38, 39, 55]
typically subsume them into a shared camera formu-
lation, leaving perspective implicit in text captions
and often conflating FREE roaming with FPP mo-
tion. We instead explicitly condition on discrete view
modes, distinguishing their physical motion seman-
tics and naturally enabling dynamic view switching.
(ii) How user inputs affect its evolution.
Existing
2

Figure 2. (Left) Prior methods use implicit view descriptions and separate control pathways. (Right) World-in-Context
unifies controls through ICC and models their temporal evolution with IT-RoPE.
systems rely almost exclusively on continuous camera
trajectories for navigation [9, 13, 31, 39, 55, 59] for all
view modes, which inevitably entangles character and
camera dynamics in TPP. To incorporate richer sig-
nals, recent approaches either formulate a unified ac-
tion interface across different regimes [19], or inject ac-
tions and camera poses through distinct architectural
pathways [36]. Differing from these paradigms, we
explicitly maintain view modes and represent charac-
ter actuation alongside camera motion as jointly con-
textualized controls within the same attention space,
effectively treating them as coexisting yet indepen-
dent degrees of freedom.
In-Context Conditioning.
In-context learning, orig-
inally developed for language models [7], has increas-
ingly been explored in visual generation.
Prompt
Diffusion [44] first enables diffusion models to infer
visual tasks from contextual examples, followed by
works that exploit shared attention contexts for task-
agnostic or unified image generation [16, 17, 37, 46,
50].
This paradigm has recently extended to video
through spatial or temporal in-context generation [8,
21], unified multimodal conditioning with full atten-
tion [12, 20], and reference videos as semantic prompts
[2]. IC-World [47] further explores in-context genera-
tion for shared world modeling, focusing on consis-
tency across multiple observations of the same world.
In this work, we extend in-context conditioning to in-
teractive video world modeling.
3. Preliminary: Problem Formulation
Let x ∈R𝑇×𝐻×𝑊×3 denote a video clip of 𝑇frames, and
y the initial contextual specification (e.g., a text prompt
or a starting frame). In interactive video world mod-
eling, the model generates future visual observations
conditioned on temporally aligned user interaction se-
quences u = {u𝑡}𝑇
𝑡=1, parameterizing the conditional
distribution:
𝑝𝜃(x | y, u).
(1)
Modern video diffusion and flow-matching back-
bones typically instantiate this conditional process
within a latent space. Specifically, let z𝜏denote the
latent video representations at diffusion timestep 𝜏∈
[0, 1]. Existing models typically inject interactive con-
ditions u via modular, condition-specific pathways:
ˆv𝜏= F𝜃
z𝜏, 𝑓cond(u); y, 𝜏,
(2)
where 𝑓cond(·) commonly instantiates modulation for
camera trajectories [9, 18], additive token injection for
discrete actions [5, 19], cross-attention for text-form
controls [30], or geometry-aware positional attention
and warping for camera control [39, 53, 55]. While
effective for isolated signals, such architectural sepa-
ration isolates diverse controls into disparate parame-
ter spaces, hindering the model from capturing cross-
condition dependencies directly. In this work, instead
of introducing dedicated condition branches, we re-
frame multi-modal interactive conditioning as a uni-
fied in-context sequence modeling problem:
ˆv𝜏= F𝜃
[c ∥z𝜏]; y, 𝜏,
(3)
where diverse controls are recast as unified condition
tokens c and natively contextualized with visual la-
tents within the shared attention manifold.
4. World-in-Context
4.1. Overview
To avoid the scalability bottleneck of fragmented side
channels (e.g., AdaLN or separate cross-attention lay-
ers), World-in-Context casts diverse interactive sig-
3

w/o view modeling (text)
w/ view modeling
w/ view modeling (500ft)
Figure 3. Emergent view switching. (Left) Explicit view modeling enables switching with residual artifacts; fine-tuning on
500 switching clips (“500ft") improves transitions. (Right) Success rates on WBench FPP↔TPP set [54].
nals directly into sequence tokens within a unified
attention space, as shown in Figure 2 (Right).
The
world modeling space is organized around two com-
plementary pillars: (i) how the world is viewed across
perspectives, and (ii) how user inputs steer dynamics
through concurrent character action and camera view-
port movement.
Unified In-Context Conditioning.
Instead of intro-
ducing specialized parameter pathways for each con-
trol modality, all interactive signals are projected into
a unified sequence of condition tokens c ∈R𝐿𝑐×𝐷,
matching the backbone hidden dimension 𝐷:
c =

cview ; caction ; ccamera
,
(4)
where [ · ; · ] denotes concatenation along the se-
quence dimension.
These condition tokens are
prepended directly to the visual latents z𝜏at diffu-
sion timestep 𝜏∈[0, 1]. The flow-matching generative
backbone then predicts the velocity field ˆv𝜏over this
shared sequence:
ˆv𝜏= F𝜃
[c ; z𝜏] ; y, 𝜏.
(5)
Ingesting all controls into the shared self-attention al-
lows cross-condition dependencies to be captured na-
tively without ad-hoc modules, reframing interactive
expansion into an in-context modeling problem.
Decomposing Interactive Context.
The condition
sequence c of World-in-Context explicitly decouples
the interactive degrees of freedom:
❏View Mode (cview): Disentangles FREE, FPP, and
TPP regimes via learned categorical tokens, mit-
igating view ambiguity and naturally supporting
dynamic switching.
❏Action (caction): Encodes hybrid keyboard-action
inputs to govern local character actuation, provid-
ing responsive command dynamics.
❏Camera (ccamera): Embeds relative 6-DoF camera
poses and Plücker coordinates to ensure continu-
ous trajectory tracking and viewpoint stability.
To complement in-context conditioning, we further
incorporates geometric transport into the temporal
attention mechanism to preserve continuous physi-
cal transitions across frames.
The subsequent sec-
tions elaborate on the explicit view-mode formulation
(§4.2), examine joint action-camera decoupling (§4.3),
and introduce IT-RoPE for parameterized temporal at-
tention (§4.4).
4.2. Explicit View-Mode Modeling
Most existing models [9, 31, 38, 40, 56] convey view-
ing regimes either through implicit text descriptions y
or directly via camera poses. However, identical ge-
ometric trajectories inherently entail different motion
semantics across regimes, for example, distinguish-
ing unconstrained FREE navigation from embodied
FPP motion. Leaving such distinctions to implicit text
conditioning makes it ambiguous to resolve regime-
dependent motion characteristics from identical geo-
metric inputs.
We therefore restrict y strictly to world-state specifica-
tions and introduce explicit, time-aligned view tokens
cview = {cview
𝑡
}𝑇z−1
𝑡=0 into the shared context:
cview
𝑡
= eview(𝑣𝑡) + etype(view), 𝑣𝑡∈{FREE, FPP, TPP}.
(6)
By attending directly to visual latents, cview
𝑡
acts as
a contextual switch that modulates the shared atten-
tion space toward regime-appropriate physical mo-
tion fields.
To validate the efficacy of explicit view condition-
ing, we compare our formulation against a baseline
that specifies view modes through text alone (“w/o
view modeling") Figure 4 evaluates trajectory oscil-
lations under FPP and FREE conditioning across 100
scenes, while Figure 3 presents qualitative transitions
alongside quantitative evaluations on the WBench
FPP↔TPP suite. Detailed setups are provided in Sec-
tion B.2; key observations are summarized:
Obs.❶Explicit view modeling distinguishes regime-
dependent motion dynamics.
As shown in Figure 4,
text-based conditioning yields closely overlapping os-
cillation profiles for FPP and FREE, conflating their
respective physical semantics.
In contrast, explicit
4

Figure 4. View-dependent motion dynamics. Oscillation
RMS across 100 scenes.
Explicit view modeling distin-
guishes FPP oscillations from smoother FREE motion.
view modeling encourages a distinct separation: FPP
exhibits pronounced periodic oscillations character-
istic of embodied locomotion, while FREE preserves
smoother camera trajectories. This indicates that ex-
plicit view tokens effectively mitigate motion ambigu-
ity across viewing regimes.
Obs.❷In-video view switching emerges naturally
from explicit view tokens.
Because view tokens act
as modular contextual switches within the shared
attention space, training exclusively on single-view
clips already gives rise to perspective transitions sim-
ply by varying the view token sequence during roll-
out, as shown in Figure 3. On the WBench FPP↔TPP
set, our base model without switching supervision
achieves a 37.5% success rate, substantially outper-
forming the text-based baseline (6.25%). Incorporat-
ing a minimal set of 500 transition sequences fur-
ther consolidates this emergent behavior to 93.8%,
demonstrating that explicit view conditioning estab-
lishes a receptive interface that is readily steerable and
sample-efficient to tune.
4.3. Joint Action-Camera Modeling
While existing interactive models typically rely on
camera trajectories as the primary control signal [9, 38,
49] for all view modes, character actuation and cam-
era viewpoint constitute separate degrees of freedom,
particularly in TPP. Rather than mapping them to a
single control interface [19] or routing them into dis-
parate conditioning branches [36], we model character
actions and camera trajectories as coexisting, decou-
pled conditions, contextualizing both directly within
the shared transformer sequence.
Given an action sequence and a camera trajectory,
World-in-Context encodes both modalities into time-
aligned context tokens.
For action conditioning,
each input a𝑡is decomposed into semantic subslots
S = {move, turn, Δtrans, Δrot} separating discrete key
states from continuous motion increments. The corre-
sponding action tokens are given by:
caction
𝑡,𝑝,𝑠
=LayerNorm

ℎ𝑠(a𝑡) + 𝑟𝑠(a𝑡) + e𝑡(𝑡)
+ e𝑝(𝑝) + e𝑠(𝑠)

+ etype(act),
(7)
where 𝑠∈S, 𝑝∈Pact indexes the within-step tokens,
ℎ𝑠denotes a slot-specific encoder (e.g., MLP), and 𝑟𝑠is
a zero-initialized residual branch for continuous mag-
nitude cues.
For camera conditioning, World-in-Context converts
each normalized relative pose bE𝑡into a tokenized
Plücker ray field, following [11, 20]. For a sampled
ray (𝑢, 𝑣) with camera center bo𝑡and unit direction
d𝑡,𝑢,𝑣, its Plücker coordinate is denoted by 𝝅𝑡,𝑢,𝑣=
[bo𝑡× d𝑡,𝑢,𝑣; d𝑡,𝑢,𝑣]. The corresponding camera token is
encoded as:
ccamera
𝑡,𝑢,𝑣
=LayerNorm

𝜙ray(𝝅𝑡,𝑢,𝑣)+
𝜙pose(vec(bE𝑡)) + 𝜙Δ(𝝃𝑡)

+ etype(cam),
(8)
where 𝝃𝑡∈R6 denotes the local rigid-body increment,
and the three projections encode ray geometry, global
relative pose, and local camera motion, respectively.
Beyond their essential decoupling in TPP, we further
investigate the less-examined FPP and FREE regimes,
where camera paths alone are typically assumed to
suffice. Figure 5 compares action-only, camera-only,
and joint training under our setting on the WBench
NAVIGATION set (excluding TPP scenes). Our key ob-
servations are:
action-only
camera-only
joint training
Figure 5.
Complementary action-camera control.
(Top)
Action-only training exhibits artifacts; camera-only training
yields insufficient displacement. (Bottom) Joint training bal-
ances motion accuracy and consistency on WBench NAVI-
GATION (excluding TPP).
Obs.❸Action and camera signals play complemen-
tary roles.
As shown in Figure 5, action-only condi-
tioning yields higher translation and rotation accuracy
(+6.5 and +3.9 points over camera-only), but can in-
troduce visual artifacts. Conversely, camera-only con-
ditioning improves geometric (+2.1) and photometric
(+1.9) consistency, though often at the cost of insuffi-
cient travel displacement. Joint training naturally har-
monizes these properties, achieving the highest rota-
tion accuracy (90.6) and geometric consistency (90.8)
while maintaining strong translation fidelity (89.3). In
effect, action commands govern local responsiveness,
whereas camera trajectories anchor global stability.
5

Table 1. Evaluation on WBench [54] NAVIGATION and FPP↔TPP sets. We highlight the best and second best.
Model
Size
Supported
Conditioning
Navigation
Perspective
Controls
Strategy
Quality
Setting
Interaction
Consistency
Physical
Average
Switching
EchoWM [55]
22B
Positional Attn.
81.5
79.4
86.6
89.8
70.6
81.6
32.0
LingBot-World v2 [9]
14B
AdaLN
81.8
76.8
82.8
86.5
69.1
79.4
21.9
Alaya-EVOKE [53]
14B
Warp Condition
82.8
83.8
78.6
86.9
72.1
80.8
18.8
HY-World 1.5 [18]
8.3B
Modulat.+PRoPE
78.1
72.2
86.8
86.9
66.3
78.1
12.5
Wan 2.2 [41]
5B
Text
72.2
69.3
65.4
86.4
52.3
69.1
6.3
YUME 1.5 [30]
5B
Text
Cross-Attn.
77.6
72.4
71.4
80.1
65.2
73.3
12.5
ABot-World [19]
5B
Additive Inject.
76.8
71.4
84.0
79.5
61.7
74.7
9.4
DreamX-World [39]
5B
Positional Attn.
77.5
80.8
78.6
74.9
63.3
75.0
18.8
Zing-0.5 [5]
5B
Additive Inject.
80.6
77.8
84.2
88.5
73.8
81.0
12.5
World-in-Context (Ours)
5B
In-Context
80.9
81.0
86.0
87.2
72.3
81.5
93.8
4.4. Interaction-Transport Geometric Encoding
While in-context tokens expose control inputs to
the shared attention space, standard 3D-RoPE en-
codes only relative spatiotemporal coordinates, leav-
ing the intervening control progression between dis-
tant frames implicit. To this end, World-in-Context
further introduces Interaction-Transport RoPE (IT-
RoPE), which complements spatiotemporal posi-
tional encoding by parameterizing temporal atten-
tion with control-induced transport operators accu-
mulated along the execution sequence.
Ordered
Interval
Transport.
While
input-
dependent transforms [52] and geometric encoding
[23] demonstrate the promise of data-driven atten-
tion biases, our interactive setting differs in that
cross-temporal transitions are governed by tempo-
rally ordered user inputs.
We thus maintain dual,
order-preserving transport paths for character action
and camera motion 𝑏
∈
{𝑎, 𝑐}.
Specifically, each
control input induces a step transformation U𝑏
𝑡, which
accumulates along continuous valid segments to
define the relative transport across interval (𝑖, 𝑗]:
T𝑏
𝑖→𝑗=
𝑗
Ö
𝑘=𝑖+1
U𝑏
𝑘,
𝑖< 𝑗,
(9)
where T𝑏
𝑗→𝑖= (T𝑏
𝑖→𝑗)⊤, and T𝑏
𝑖→𝑗T𝑏
𝑗→𝑘= T𝑏
𝑖→𝑘. This non-
commutative composition natively preserves control
order, allowing the model to distinguish trajectories
that share identical endpoints but differ in execution
sequences.
View-Gated Centered Attention.
To incorporate
these accumulated transitions into cross-frame rea-
soning, we augment temporal attention with a cen-
tered, transport-corrected score:
𝑠𝑖𝑗= 𝑠base
𝑖𝑗
+
∑︁
𝑏∈{𝑎,𝑐}
𝜆𝑏
√𝑟
𝑚𝑏
𝑖𝑗𝑔𝑏
𝑖𝑔𝑏
𝑗· (q𝑏
𝑖)⊤
T𝑏
𝑡𝑖→𝑡𝑗−I

k𝑏
𝑗,
(10)
where 𝑠base
𝑖𝑗
=
¯q⊤
𝑖¯k𝑗
√
𝑑
denotes standard spatiotempo-
ral attention with 3D-RoPE, q𝑏
𝑖, k𝑏
𝑗∈R𝑟are control-
specific sub-channels, and 𝜆𝑏is a learnable scale ini-
tialized to zero. The gate 𝑔𝑏
𝑖∈(0, 1) dynamically mod-
ulates control influence across viewing regimes and
token types, while 𝑚𝑏
𝑖𝑗∈{0, 1} restricts the correction
to contiguous valid segments. Crucially, subtracting
I centers the correction, ensuring it vanishes for static
intervals or 𝑡𝑖= 𝑡𝑗, while zero-initializing 𝜆𝑏preserves
native backbone behavior at the start of training. Fur-
ther details of transport accumulation and attention
computation are provided in Section A.3.
4.5. Training of World-in-Context
Training Data.
We train World-in-Context on a
compact set of 100K video clips covering FREE, FPP,
and TPP viewing regimes, accompanied by tempo-
rally aligned view, action, and camera annotations
when available.
While joint training across these
single-regime clips naturally facilitates view-mode
transitions (as analyzed in Section 4.2), we addition-
ally incorporate a small fraction of cross-view switch-
ing clips (500 sequences) with dynamic view sched-
ules to reinforce transition stability.
Training Protocol.
Following the conditional flow-
matching objective,
World-in-Context undergoes
bidirectional finetuning where video and control to-
kens jointly interact in the shared attention mani-
fold.
This bidirectional setup serves as our default
configuration to benchmark the core in-context con-
ditioning paradigm. In Section 6, we further inves-
tigate World-in-Context under broader operational
settings, including causal rollout with in-context history
and expanded game-control signals.
5. Experiments
5.1. Experimental Settings
Baselines.
World-in-Context is built upon the
widely adopted Wan2.2-TI2V-5B [41] backbone. For
direct comparisons, our primary baselines comprise
models sharing this foundation: YUME 1.5 [30], ABot-
6

World-in-Context
Zing-0.5
World-in-Context
Zing-0.5
World-in-Context
Zing-0.5
World-in-Context
Zing-0.5
World-in-Context
Zing-0.5
Free-Camera
First-Person
Third-Person
TPP->FPP
FPP->TPP
Figure 6. Qualitative comparisons of World-in-Context.
World [19], DreamX-World [39], and Zing-0.5 [5]. We
also include larger models as reference: EchoWM [55],
LingBot-World v2 [9], Alaya-EVOKE [38], and HY-
World 1.5 [18]. Since training data scales are rarely
disclosed across works, we also implement public ar-
chitectures and conduct controlled comparisons un-
der our identical data for analyses.
Evaluations.
We evaluate World-in-Context on the
widely adopted WBench [54].
Alongside the stan-
dard NAVIGATION split for trajectory tracking, we fur-
ther adopt its PERSPECTIVE SWITCHING - FPP↔TPP
suite to systematically assess dynamic view transi-
tions. Our framework natively supports both action-
only and camera-only inference; we default to action-
only control for FREE and FPP, while utilizing joint ac-
tion and camera signals for character and viewpoint
in TPP. More details are in Section B.
5.2. Main Results
Quantitative
Results.
Table
1
summarizes
performance on WBench.
Among 5B models,
World-in-Context achieves the highest NAVIGATION
7

Figure 7. Qualitative results of World-in-Context on FREE mode.
8

Figure 8. Qualitative results of World-in-Context on FPP mode.
9

Figure 9. Qualitative results of World-in-Context on TPP mode.
10

Figure 10. Qualitative results of World-in-Context on FPP ↔TPP.
11

average (81.5), performing on par with Zing-0.5
(81.0) while improving the INTERACTION score from
84.2 to 86.0 via unified in-context conditioning.
Its
navigation capability also rivals larger models like
22B EchoWM (81.6).
Crucially, World-in-Context
reaches a 93.8% success rate on FPP↔TPP, signifi-
cantly outperforming the best 5B baseline (18.8%)
and confirming its capacity for dynamic viewpoint
transitions within a single unified model.
Qualitative Results.
Figure 6 further compares
our World-in-Context with Zing-0.5.
While both
generate plausible environments, Zing-0.5 tends to
incur exaggerated motion scale, whereas ours main-
tains more measured, balanced trajectories aligned
with its higher INTERACTION score.
Furthermore,
World-in-Context
smoothly
executes
FPP↔TPP
switches where the baseline fails to transition.
Ex-
tended rollouts across diverse scene layouts and
artistic styles under FREE, FPP, and TPP modes are
detailed in Figures 7 to 9.
Notably, Figure 10 also
demonstrates coherent multi-stage switching (e.g.,
TPP→FPP→TPP) while preserving consistent scene
structure throughout.
5.3. Analysis of In-Context Conditioning
To further examine how conditioning strategies scale
with input complexity, we compare ICC with additive
injection [19] and AdaLN modulation [40] under pro-
gressively richer inputs with our same data: view (V),
action (A), and camera (C). We give the following ob-
servations:
Figure 11. Conditioning strategy analysis. ICC increasingly
benefits the coordination of diverse controls as conditioning
becomes richer. C/A/V denote camera/action/view.
Obs.❹ICC’s advantage scales with conditioning
richness.
As shown in Figure 11, with camera con-
trol alone, ICC performs on par with alternative strate-
gies (76.3 vs. 76.3-77.0). Its gains emerge under joint
action-camera inputs (+0.6 in NAVIGATION, +1.1 in
INTERACTION) and expand further when adding ex-
plicit view tokens (+1.0 and +1.4, reaching 81.2 and
86.8). These widening margins indicate that a shared
attention space coordinates diverse controls more ef-
fectively than separate injection pathways as interac-
tion becomes compositional.
5.4. Analysis of IT-RoPE
To evaluate how trajectory-aware attention affects
temporal consistency, we ablate IT-RoPE here against
standard 3D-RoPE, single-branch variants, and an
endpoint-only alternative under the fixed V+A+C
setup, as shown in Figure 12.
Figure 12.
IT-RoPE ablation.
Accumulating intervening
control transitions improves navigation and control fidelity
beyond endpoint-only conditioning.
Obs.❺Tracking intervening control evolution im-
proves
interactive
attention.
Compared
to
3D-
RoPE, full IT-RoPE enhances both NAVIGATION
(80.5
→
81.2) and INTERACTION (86.2
→
86.8).
While action transport sharpens command respon-
siveness and camera transport stabilizes spatial track-
ing, their combination delivers the most balanced per-
formance. Moreover, full IT-RoPE clearly outperforms
the endpoint-only variant (80.5/86.6), confirming that
accumulating the intervening control sequence pro-
vides critical dynamics missed by endpoint offsets
alone.
5.5. Analysis of Data Efficiency
Our default interaction-training set comprises a mod-
est 100K navigation clips and 500 switching se-
quences.
Since training-data scales are not con-
sistently disclosed by existing methods, we assess
data efficiency through controlled comparisons in Fig-
ure 13: 50K+250, 100K+500, and 125K+750. We sum-
marize our observations as follows:
Figure 13. Data scaling analysis. ICC consistently outper-
forms additive injection across training budgets.
Obs.❻ICC demonstrates data efficiency across scal-
ing regimes.
ICC consistently outperforms additive
injection across all data regimes, especially in perspec-
tive switching. With only 50K+250 clips, ICC achieves
a 75.0% switching rate, already surpassing additive in-
jection trained on twice the data (71.9% at 100K+500).
At 100K+500, ICC exceeds the largest-budget addi-
tive baseline (125K+750) in both NAVIGATION (81.5
vs. 80.3) and SWITCH (93.8% vs. 78.1%), showing that
shared in-context tokens make substantially more ef-
ficient use of interactive supervision.
12

In-domain
Out-of-domain
Real-world
Figure 14. Qualitative results of WiC-Game.
6. Applicability of World-in-Context
6.1. Causal Modeling with In-Context History
To examine the architectural extensibility of in-
context conditioning beyond bidirectional generation,
World-in-Context is naturally adapted to streaming
causal rollouts.
Following common causal training
paradigms in interactive generation [60], the model
is trained autoregressively to predict future frame
latents conditioned on historical context.
Rather
than relying on external KV cache retrieval mecha-
nisms, WiC-Causal directly incorporates historical la-
tents into the shared self-attention sequence. Specifi-
cally, the condition sequence c in Eq. (4) is augmented
with historical latent tokens:
ccausal =

cview ; caction ; ccamera ; csink ; crecent
,
(11)
where csink preserves initial context tokens as global
semantic anchors, and crecent maintains a sliding win-
dow of recent visual tokens to enforce continuity.
Table 2. Evaluation of WiC-Causal on WBench.
Method
Qual.
Set.
Inter.
Consi.
Phys.
Avg.
Switch.
Wan 2.2 [41]
72.2
69.3
65.4
86.4
52.3
69.1
6.3
YUME 1.5 [30]
77.6
72.4
71.4
80.1
65.2
73.3
12.5
ABot-World [19]
76.8
71.4
84.0
79.5
61.7
74.7
9.4
DreamX-World [39]
77.5
80.8
78.6
74.9
63.3
75.0
18.8
Zing-0.5 [5]
80.6
77.8
84.2
88.5
73.8
81.0
12.5
World-in-Context
80.9
81.0
86.0
87.2
72.3
81.5
93.8
WiC-Causal
80.5
80.4
85.7
87.6
72.0
81.2
90.6
13

As shown in Table 2, WiC-Causal exhibits competitive
interactive performance, attaining an average NAVI-
GATION score of 81.2 and a 90.6% FPP↔TPP success
rate on WBench. Compared to the bidirectional coun-
terpart (81.5 and 93.8%), causal modeling incurs neg-
ligible degradation while retaining a better perfor-
mance margin over existing 5B baselines.
Further-
more, qualitative rollouts in Figure 15 demonstrate
that WiC-Causal preserves long-horizon spatial coher-
ence across diverse viewing regimes. Notably, in loop-
closure and viewpoint reappearance scenarios (e.g., re-
turning to previously explored landscapes or room
corners), WiC-Causal consistently reproduces previ-
ously observed visual landmarks and layouts.
6.2. Scaling to Rich Game Controls
To further demonstrate the extensibility and token-
level composability beyond canonical navigation
commands, we extend World-in-Context to accom-
modate richer, high-frequency gameplay actions. Ex-
periments are conducted on Counter-Strike 2 game-
play sequences from EWM [29]. To assess sample ef-
ficiency and generalizability, WiC-Game is trained on
a compact subset of a single map (i.e., DUST II), and
tested on unseen layouts as well as real-world scenar-
ios. Specifically, the condition sequence is augmented
with discrete game-control tokens:

cfire ; cduck ; cwalk
,
(12)
corresponding to firing (weapon dynamics and muz-
zle flashes), crouching (viewport descent), and slow
walking (motion attenuation), respectively.
As illustrated in Figure 14, WiC-Game demonstrates
sensitive, decoupled control execution across three
complementary regimes: (i) In-domain: new control
trajectories within the training map (DUST II) exhibit
instant actuation responses; (ii) Out-of-distribution:
when applied to unseen game maps (e.g., MIRAGE,
NUKE), the model preserves scene consistency while
faithfully following concurrent actions; (iii) Real-world
transfer: using synthetic starting images, the learned
actuation dynamics generalize naturally to realistic vi-
sual environments.
7. Conclusion and Positioning
Conclusion.
In
this
work,
we
present
World-in-Context, a video world modeling frame-
work based on In-Context Conditioning (ICC) that
unifies view mode, character action, and camera
trajectory as tokens within a shared attention space.
We systematically investigate core interactive dimen-
sions, revealing that explicit view tokens disentangle
conflicting motion dynamics while enabling emergent
in-video switching, and that action and camera sig-
nals complement each other in local responsiveness
and global stability.
Augmented with IT-RoPE to
encode accumulated control progression in temporal
attention, World-in-Context achieves competitive
navigation on WBench (81.5) matching the 22B
EchoWM,
while
establishing
strong
perspective-
switching capability (93.8%).
Controlled analyses
further demonstrate compounding advantages under
richer conditioning, favorable sample efficiency, and
natural applicability to causal rollouts, highlighting
shared context modeling as a scalable interface for
interactive world simulation.
Positioning
of
This
Report.
This
work
fo-
cuses
primarily
on
the
algorithmic
formulation
of
interactive
conditioning,
rather
than
train-
ing
an
all-encompassing
system
artifact,
with
World-in-Context serving as a concrete instantiation
of this design. The overall capacity of an interactive
world model naturally depends on data quality,
model size, and training scale.
We use a modest
interaction-training set of 100K navigation clips
and 500 switching sequences.
Since comparable
training-data statistics are not consistently disclosed
by existing methods, we omit cross-method data-
scale comparisons from the main table. Instead, we
prioritize controlled comparisons that isolate the
architectural advantages of shared context modeling
and control-aware temporal attention, including their
effectiveness across training-data budgets.
Specif-
ically, this work centers around three foundational
control modalities, i.e., view mode, character action,
and camera motion, as they directly address the core
dimensions governing interactive navigation:
how
the environment is viewed across perspectives and
how user inputs steer character actuation alongside
viewport evolution.
These algorithmic findings
complement concurrent advances in scaling, present-
ing an extensible interface for coordinating diverse
interactive controls.
Contributions
Contributors.
Harold Haodong Chen∗, Rui Peng†,
Rongjin Guo, Lingteng Qiu, Kejie Qiu, Chao Xu,
Zilong Dong
†Corresponding Author
∗Work done during Harold H. Chen’s internship at
ATH@Alibaba.
For questions or feedback, feel free to contact harold-
chen328@gmail.com or open a GitHub issue.
14

Figure 15. Qualitative results of WiC-Causal.
15

References
[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen,
Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei
Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl tech-
nical report. arXiv preprint arXiv:2511.21631, 2025. 1
[2] Yuxuan Bian, Xin Chen, Zenan Li, Tiancheng Zhi, Shen
Sang, Linjie Luo, and Qiang Xu.
Video-as-prompt:
Unified semantic control for video generation. In In-
ternational Conference on Learning Representations, pages
114558–114583, 2026. 3
[3] Harold Haodong Chen, Rongjin Guo, Disen Lan, Wen-
Jie Shu, Hongfei Zhang, Hanzhe Hu, Shengtao Yao,
Zixin Zhang, Guibin Zhang, Zhefan Rao, et al. The past
frames the future: Memory for autoregressive video
generation. arXiv preprint arXiv:2609.28466, 2026. 2
[4] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang
Chu,
Junyu Chen,
Shuai Yang,
Xianbang Wang,
Yicheng Pan, Zhou Daquan, Huan Ling, et al. Sana-
video: Efficient video generation with block linear dif-
fusion transformer. In International Conference on Learn-
ing Representations, pages 38973–38997, 2026. 1
[5] Mingyang Chen,
Shengdong Chen,
Xiaoxiao Fu,
Bosheng Gong, Haoyuan Guo, Bowen Li, Jiawen Li,
Kejun Li, Tianpeng Li, Yin Liu, et al.
Zing-0.5: To-
ward playable worlds with real-time joint action and
text control. arXiv preprint arXiv:2609.17909, 2026. 3, 6,
7, 13
[6] Etched Decart, Quinn McIntyre, Spruce Campbell, Xin-
lei Chen, and Robert Wachen. Oasis: A universe in a
transformer. URL: https://oasis-model. github. io, 2(3):6,
2024. 2
[7] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan
Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu,
Baobao Chang, et al. A survey on in-context learning.
In Proceedings of the 2024 conference on empirical methods
in natural language processing, pages 1107–1128, 2024. 3
[8] Zhengcong Fei, Di Qiu, Debang Li, Changqian Yu, and
Mingyuan Fan.
Video diffusion transformers are in-
context learners. arXiv preprint arXiv:2412.10783, 2024.
3
[9] Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen,
Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan,
Hanlin Wang, Yichong Lu, et al. Infinite worlds with
versatile interactions. arXiv preprint arXiv:2607.07534,
2026. 1, 2, 3, 4, 5, 6, 7
[10] Yoav HaCohen, Nisan Chiprut, Benny Brazowski,
Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran
Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-
video: Realtime video latent diffusion. arXiv preprint
arXiv:2501.00103, 2024. 1
[11] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein,
Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl:
Enabling camera control for video diffusion models. In
The Thirteenth International Conference on Learning Rep-
resentations, 2025. 5
[12] Xuanhua He, Quande Liu, Zixuan Ye, Weicai Ye, Qi-
ulin Wang, Xintao Wang, Qifeng Chen, Pengfei Wan,
Di Zhang, and Kun Gai. Fulldit2: Efficient in-context
conditioning for video diffusion transformers.
arXiv
preprint arXiv:2506.04213, 2025. 3
[13] Hongrui Huang, Junke Wang, Quanhao Li, Yu-Gang
Jiang,
and Zuxuan Wu.
Disco:
World models
with discrete camera motion control.
arXiv preprint
arXiv:2606.07967, 2026. 2, 3
[14] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr
Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen,
Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al.
Vipe: Video pose engine for 3d geometric perception.
arXiv preprint arXiv:2508.10934, 2025. 1, 3
[15] Junchao Huang, Guian Fang, Shengju Qian, Xiang-
hao Kong, Zhuoran Zhao, Wei Huang, Yihua Du,
Zixin Zhang, Justin Cui, Yuchao Gu, et al. Solarwm:
Open data and scalable training for long-horizon video
world models. arXiv preprint arXiv:2609.02886, 2026. 2
[16] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Huanzhang
Dou, Yupeng Shi, Yutong Feng, Chen Liang, Yu Liu,
and Jingren Zhou. Group diffusion transformers are
unsupervised multitask learners. 2024. 3
[17] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi,
Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu,
and Jingren Zhou. In-context lora for diffusion trans-
formers. arXiv preprint arXiv:2410.23775, 2024. 3
[18] Team HunyuanWorld.
Hy-world 1.5: A systematic
framework for interactive world modeling with real-
time latency and geometric consistency. arXiv preprint,
2025. 3, 6, 7
[19] Fan Jiang, Zhaoxu Sun, Mengchao Wang, Ziyu Zhu,
Chiyu Wang, Yunpeng Zhang, Wenlin Liu, Yun Wang,
Xue Zheng, Rui Sun, et al. Abot-world-0: Infinite in-
teractive world rollout on a single desktop gpu. arXiv
preprint arXiv:2607.19191, 2026. 2, 3, 5, 6, 7, 12, 13, 1, 4
[20] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao
Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu.
Fulldit: Multi-task video generative foundation model
with full attention.
arXiv preprint arXiv:2503.19907,
2025. 3, 5
[21] Kinam Kim, Junha Hyung, and Jaegul Choo. Temporal
in-context fine-tuning with temporal reasoning for ver-
satile control of video diffusion models. arXiv preprint
arXiv:2506.00996, 2025. 3
[22] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo
Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jian-
wei Zhang, et al. Hunyuanvideo: A systematic frame-
work for large video generative models. arXiv preprint
arXiv:2412.03603, 2024. 1
[23] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma,
and Angjoo Kanazawa. Cameras as relative positional
encoding. In The Thirty-ninth Annual Conference on Neu-
ral Information Processing Systems, 2025. 6
[24] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian
Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Alek-
sander Holynski, and Noah Snavely. Megasam: Ac-
curate, fast, and robust structure and motion from ca-
sual dynamic videos. In 2025 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
10486–10496. IEEE, 2025. 3
[25] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin,
Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang
Feng, Jianwen Sun, et al. Sekai: A video dataset to-
wards world exploration. Advances in Neural Informa-
tion Processing Systems, 38, 2026. 1
[26] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen,
Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang.
Depth anything 3: Recovering the visual space from
any views. arXiv preprint arXiv:2511.10647, 2025. 3
16

[27] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng
Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu,
Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset
for deep learning-based 3d vision. In 2024 IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 22160–22169. IEEE, 2024. 1
[28] Jiuming Liu, Chaojun Ni, Mengmeng Liu, Chensheng
Peng, Fangjinhua Wang, Sitian Shen, Marc Pollefeys,
Masayoshi Tomizuka, Ayush Tewari, and Per Ola Kris-
tensson.
Towards interactive video world modeling:
Frontiers, challenges, benchmarks, and future trends.
arXiv preprint arXiv:2606.01164, 2026. 2
[29] Weiyu Ma, Zeyuan Gong, Xuhui Liu, Wenxuan Zhang,
Jian Ding, Xinyu Cui, Jian Zhao, and Mohamed El-
hoseiny. Esports world model dataset. arXiv preprint
arXiv:2610.16166, 2026. 14
[30] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kain-
ing Ying, and Kaipeng Zhang.
Yume1. 5:
A text-
controlled interactive world generation model. In Pro-
ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 7752–7761, 2026. 3, 6, 13
[31] Jisu Nam, Yicong Hong, Chun-Hao Paul Huang, Feng
Liu, JoungBin Lee, Jiyoung Kim, Siyoon Jin, Yunsung
Lee, Jaeyoon Jung, Suhwan Choi, et al. Worldcam: In-
teractive autoregressive 3d gaming worlds with cam-
era pose as a unifying geometric representation. arXiv
preprint arXiv:2603.16871, 2026. 2, 3, 4
[32] Jack Parker-Holder,
Philip Ball,
Jake Bruce,
Vib-
havari Dasagi, Kristian Holsheimer, Christos Kaplanis,
Alexandre Moufarek, Guy Scully, Jeremy Shar, Jimmy
Shi, Stephen Spencer, Jessica Yung, Michael Dennis,
Sultan Kenjeyev, Shangbang Long, Vlad Mnih, Harris
Chan, Maxime Gazeau, Bonnie Li, Fabio Pardo, Luyu
Wang, Lei Zhang, Frederic Besse, Tim Harley, Anna
Mitenkova, Jane Wang, Jeff Clune, Demis Hassabis,
Raia Hadsell, Adrian Bolton, Satinder Singh, and Tim
Rocktäschel. Genie 2: A large-scale foundation world
model. 2024. 2
[33] Runjia Qian, Zile Wang, Jihai Zhang, Kai Zou, Wei Yu,
Jiaxing Li, Zexiang Liu, Yaokun Li, Fei Kang, Kaichen
Huang, et al.
Matrix-game 3.5: Enhancing real-time
streaming interactive world models with patch mem-
ory. arXiv preprint arXiv:2608.29910, 2026. 2
[34] Quanjian Song, Yiren Song, Kelly Peng, Yuan Gao, and
Mike Zheng Shou. Worldwander: Bridging egocentric
and exocentric worlds in video generation. In European
Conference on Computer Vision, pages 612–630. Springer,
2026. 1
[35] Tomás Soucek and Jakub Lokoc. Transnet v2: An effec-
tive deep network architecture for fast shot transition
detection. In Proceedings of the 32nd ACM international
conference on multimedia, pages 11218–11221, 2024. 1
[36] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta
Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun
Zhang, Tengfei Wang, and Chunchao Guo.
World-
play:
Towards long-term geometric consistency for
real-time interactive world modeling.
arXiv preprint
arXiv:2512.14614, 2025. 2, 3, 5
[37] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu
Xue, and Xinchao Wang. Ominicontrol: Minimal and
universal control for diffusion transformer.
In 2025
IEEE/CVF International Conference on Computer Vision
(ICCV), pages 14940–14950. IEEE, 2025. 3
[38] AlayaWorld Team, Kaipeng Zhang, Chuanhao Li, Yi-
fan Zhan, Yongtao Ge, Yuanyang Yin, Jiaming Tan,
Kang He, Liaoyuan Fan, Mingliang Zhai, et al. Alaya-
world: Interactive long-horizon world modeling–full
technical report. arXiv preprint arXiv:2607.18367, 2026.
1, 2, 4, 5, 7
[39] DreamX Team, Yancheng Bai, Rui Chen, Xiangxiang
Chu, Rujing Dang, Hao Dou, Bingjie Gao, Qiwen Gu,
Siyu Hong, Jiachen Lei, et al.
Dreamx-world 1.0:
A general-purpose interactive world model.
arXiv
preprint arXiv:2606.16993, 2026. 2, 3, 6, 7, 13
[40] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong
Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Han-
lin Wang, Yinghao Xu, Shuailei Ma, et al.
Ad-
vancing open-source world models.
arXiv preprint
arXiv:2601.20540, 2026. 1, 2, 4, 12
[41] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie
Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming
Zhao, Jianxiao Yang, et al. Wan: Open and advanced
large-scale video generative models.
arXiv preprint
arXiv:2503.20314, 2025. 1, 6, 13
[42] Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin,
Jian Gao, Lin-Zhuo Chen, Yajie Bao, Chang Zeng, Yanxi
Zhou, Xiao-Xiao Long, et al. Spatialvid: A large-scale
video dataset with spatial annotations. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 42592–42603, 2026. 1
[43] Luozhou Wang, Zhifei Chen, Yihua Du, Dongyu Yan,
Wenhang Ge, Guibao Shen, Xinli Xu, Leyi Wu, Man
Chen, Tianshuo Xu, et al. A mechanistic view on video
generation as world models: State and dynamics. arXiv
preprint arXiv:2601.17067, 2026. 2
[44] Zhendong Wang, Yifan Jiang, Yadong Lu, Pengcheng
He, Weizhu Chen, Zhangyang Wang, Mingyuan Zhou,
et al. In-context learning unlocked for diffusion mod-
els. Advances in Neural Information Processing Systems,
36:8542–8562, 2023. 3
[45] Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang,
Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao
Jiang, Yichen Wei, et al. Matrix-game 3.0: Real-time and
streaming interactive world model with long-horizon
memory. arXiv preprint arXiv:2604.08995, 2026. 1, 2
[46] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao
Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang,
Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to
advanced multimodal generation. arXiv e-prints, pages
arXiv–2506, 2025. 3
[47] Fan Wu, Jiacheng Wei, Ruibo Li, Yi Xu, Junyou Li, De-
heng Ye, and Guosheng Lin. Ic-world: In-context gen-
eration for shared world modeling. In European Confer-
ence on Computer Vision, pages 483–501. Springer, 2026.
3
[48] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen,
Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan,
and Weisi Lin. Exploring video quality assessment on
user generated contents from aesthetic and technical
perspectives. In 2023 IEEE/CVF International Conference
on Computer Vision (ICCV), pages 20087–20097. IEEE,
2023. 1
[49] Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang,
Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaom-
ing Wei, Chunle Guo, Chongyi Li, et al. Infinite-world:
17

Scaling interactive world models to 1000-frame hori-
zons via pose-free hierarchical memory. arXiv preprint
arXiv:2602.02393, 2026. 5
[50] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan,
Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang,
Tiejun Huang, and Zheng Liu.
Omnigen:
Unified
image generation.
In 2025 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
13294–13304. IEEE, 2025. 3
[51] Lihe Yang, Zhen Zhao, and Hengshuang Zhao. Uni-
match v2: Pushing the limit of semi-supervised seman-
tic segmentation. IEEE Transactions on Pattern Analysis
and Machine Intelligence, 47(4):3031–3048, 2025. 1
[52] Songlin Yang, Yikang Shen, Kaiyue Wen, Shawn Tan,
Mayank Mishra, Liliang Ren, Rameswar Panda, and
Yoon Kim. PaTH attention: Position encoding via accu-
mulating householder transformations. In The Thirty-
ninth Annual Conference on Neural Information Processing
Systems, 2025. 6
[53] Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuan-
hao Li, Kaipeng Zhang, and Feng Zhao. Alaya-evoke:
From linear-scaling supervision to endless world. arXiv
preprint arXiv:2608.13546, 2026. 3, 6
[54] Kaining Ying,
Hengrui Hu,
Siyu Ren,
Jiamu Li,
Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang
Cai, and Henghui Ding.
Wbench:
A comprehen-
sive multi-turn benchmark for interactive video world
model evaluation.
arXiv preprint arXiv:2605.25874,
2026. 4, 6, 7, 2
[55] Songchun Zhang, Yaowei Li, Junhao Zhuang, Weiyang
Jin, Haoyu Wang, Xin Lu, Yilang Sun, Shiyi Zhang,
Haoran Li, Xiaoxiao Ma, et al.
Echowm: Open and
enterable omnimodal world models.
arXiv preprint
arXiv:2608.23189, 2026. 2, 3, 6, 7
[56] Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou,
Yimin Chen, Wenqiang Sun, Kaiwen Zheng, Guande
He, Xiao Yang, Chongxuan Li, et al. minwm: A full-
stack open-source framework for real-time interactive
video world models. arXiv preprint arXiv:2605.30263,
2026. 2, 4
[57] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chen-
hui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou,
Tianyi Li, and Yang You.
Open-sora: Democratiz-
ing efficient video production for all.
arXiv preprint
arXiv:2412.20404, 2024. 1
[58] Tinghui Zhou, Richard Tucker, John Flynn, Graham
Fyffe, and Noah Snavely. Stereo magnification: Learn-
ing view synthesis using multiplane images.
arXiv
preprint arXiv:1805.09817, 2018. 1
[59] Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Jun-
song Chen, Jincheng Yu, Tong He, Song Han, and Enze
Xie. Sana-wm: Efficient minute-scale world modeling
with hybrid linear diffusion transformer. arXiv preprint
arXiv:2605.15178, 2026. 2, 3
[60] Hongzhou Zhu, Min Zhao, Guande He, Hang Su,
Chongxuan Li, and Jun Zhu.
Causal forcing: Au-
toregressive diffusion distillation done right for high-
quality real-time interactive video generation.
arXiv
preprint arXiv:2602.02214, 2026. 13
18

A. Training Details of World-in-Context
A.1. Training Data
Data Sources.
The training mixture comprises a de-
fault collection of 100K single-regime navigation clips
alongside a subset of 500 FPP↔TPP switching se-
quences.
All qualitative evaluation scenes presented in
this work are strictly held out from the training set.
❏Navigation Data (100K): Clips are collected across
widely-adapted open-source video datasets, in-
cluding DL3DV [27], RealEstate [58], Sekai [25],
SpatialViD [42], EgoExo [34], and ABot-World-
Explorer [19].
❏Switching Data (500):
Viewpoint-transition se-
quences are collected from EgoExo [34] and inter-
nal data.
Quality Pipeline.
The raw video corpus is filtered
through a standard three-stage automated filtering
pipeline:
❏Scene/Shot Continuity:
Following LingBot-World
[40], TransNetV2 [35] is deployed to detect abrupt
scene transitions and shot boundaries; segmented
multi-shot videos are removed to retain strictly
continuous navigation segments.
❏Motion Filtering:
Following Open-Sora [57] and
SANA-Video [4], UniMatch [51] optical flow and
VMAF metrics are evaluated to quantify dynamic
motion magnitude. Clips displaying static stagna-
tion or excessive motion blur/discontinuities are
discarded.
❏Aesthetic Assessment: Dover [48] is utilized to score
perceptual aesthetics, eliminating clips compro-
mised by severe compression artifacts, overexpo-
sure, or low visual fidelity.
Control Annotation.
The control signals mainly
used in World-in-Context, i.e., view, camera, and ac-
tion, are annotated as follows:
❏View Mode: A vision-language foundation model
(i.e., Qwen3-VL-32B [1]) classifies clips into FREE,
FPP, and TPP regimes. FREE is characterized by
unconstrained, smooth aerial or camera roaming
(e.g., drone cinematography); FPP captures embod-
ied locomotion or in-cockpit perspectives; TPP en-
compasses external character-following camera ar-
rangements. Ambiguous instances at boundaries
are cross-verified and resolved using reconstructed
camera trajectory statistics.
❏Camera Trajectories: Following common practice,
continuous 6-DoF camera trajectories and Plücker
ray parameters are reconstructed using ViPE [14].
❏Action Inversion: While authentic character actua-
tion natively originates from game-engine teleme-
try, obtaining massive synchronized game record-
ings remains challenging.
Therefore, discrete
character actions are mainly inversely resolved
from ViPE-estimated continuous camera trajecto-
ries. Crucially, for FPP sequences, body-induced
periodic oscillations and head-bobbing motions
are decoupled and filtered out before discrete ac-
tion discretization.
World-State
Captioning.
Unlike
existing
world-
modeling
frameworks
(e.g.,
LingBot-World
[40],
Matrix-Game
3.0
[45])
that
synthesize
verbose
temporal narratives or dense temporal captions,
World-in-Context adopts a simplified world-state
captioning formulation.
Under the premise that
generative world models should decouple scene con-
stitution from action-driven exploration, captions in
this pipeline exclusively summarize the static visual
constitution of the environment (e.g., atmospheric
conditions, spatial geometry, and layout entities).
All temporal progression, dynamic steering, and
viewport navigation are delegated entirely to the
interactive control tokens.
Table 3. Navigation data composition distribution.
DL3DV
SpatialViD
ABot
EgoExo
Sekai
RealEstate
29.22%
27.5%
25.24%
8.76%
5.45%
3.83%
Data
Distribution.
The
distribution
of
finally
adopted navigation data sources is illustrated in
Table 3.
A.2. Conditioning Schema
To facilitate versatile interactive control while ac-
commodating
practical
annotation
constraints,
World-in-Context adopts a two-stage training cur-
riculum and a mixed-control schema across viewing
regimes.
Figure 16. View distribution across two stages.
Two-Stage Training Curriculum.
As outlined in
Section 4.5, training proceeds in two sequential phases
(Figure 16):
➀Stage-1 (General Navigation): World-in-Context is
trained on 100K single-regime navigation clips to
learn fundamental physical dynamics and inter-
active responsiveness across FREE, FPP, and TPP
modes.
1

➁Stage-2
(Switching
Consolidation):
Rather
than
fine-tuning
solely
on
switching
data,
World-in-Context
constructs
an
8K
curricu-
lum by mixing regular navigation clips with the
500 switching sequences, where switching clips
are oversampled to reinforce transition stability
while preserving base navigation capabilities.
Figure 17. Action/camera distribution across two stages.
Mixed Action-Camera Conditioning.
In Section 4.3,
“joint training” refers precisely to the composite con-
ditioning mixture illustrated in Figure 17, rather than
strictly enforcing concurrent dual signals on every
training instance. In real-world data, simultaneously
obtaining robust annotations for both modalities is of-
ten impractical; for instance, loose camera-following
in TPP sequences complicates clean action discretiza-
tion, and synthetic scenes can introduce camera pose
estimation artifacts.
Therefore, World-in-Context
trains across a composite conditioning mixture of
action-only, camera-only, joint, and unconditioned sam-
ples.
Distribution Transparency.
The mixture schedules
in Figure 16 and Figure 17 are reported for empiri-
cal transparency rather than claimed as an optimal
configuration, reflecting practical trade-offs against
data availability. Crucially, all comparative baselines
(e.g., additive injection and AdaLN modulation in
Section 5) follow this identical conditioning schema
to guarantee strictly controlled architectural compar-
isons.
A.3. More Details of IT-RoPE
In this section, we further detail additional mathemat-
ical formulations, implementation properties, and ef-
ficient computation of Interaction-Transport RoPE (IT-
RoPE).
Transport Accumulation and Prefix Formulation.
IT-RoPE maintains dual, decoupled transport paths
for character action and camera motion (𝑏∈{𝑎, 𝑐}),
aligned with the latent temporal grid.
Within each
continuous valid segment anchored at initial time 𝑠,
the orthogonal step transformations 𝑈𝑏
𝑡accumulate
into prefix operators:
𝑃𝑏
𝑠= 𝐼,
𝑃𝑏
𝑡= 𝑃𝑏
𝑡−1𝑈𝑏
𝑡,
𝑇𝑏
𝑖→𝑗= (𝑃𝑏
𝑖)⊤𝑃𝑏
𝑗,
(13)
which recovers the ordered interval transport 𝑇𝑏
𝑖→𝑗=
Î𝑗
𝑘=𝑖+1 𝑈𝑏
𝑘(𝑖< 𝑗) defined in Eq. (9) and supports both
forward and backward temporal relations. Because
the step transformations need not commute, the accu-
mulated transport retains order-dependent informa-
tion about the execution sequence rather than reflect-
ing only endpoint states.
Gating,
Centering,
and Initialization.
The two
transport corrections act additively on the base 3D-
RoPE attention scores.
Specifically, gates 𝑔𝑏
𝑖condi-
tioned on view mode and token type dynamically
adjust the relative influence of each control branch,
while interval masks 𝑚𝑏
𝑖𝑗∈{0, 1} restrict corrections
to token pairs within contiguous valid execution seg-
ments. Centering the transport operator by subtract-
ing the identity matrix 𝐼ensures that the correction
vanishes whenever 𝑇𝑏
𝑡𝑖→𝑡𝑗= 𝐼. This holds for same-
timestep pairs (𝑡𝑖= 𝑡𝑗) via the prefix formulation, as
well as for static intervals where the control parame-
terization yields identity step transformations (𝑈𝑏
𝑡= 𝐼).
Furthermore, zero-initializing the branch scale factors
𝜆𝑏preserves the base attention scores at initialization.
Factorized Attention Computation.
By exploiting
the orthogonality of the prefix operators 𝑃𝑏
𝑡, the cen-
tered inner product expands into:
(𝑞𝑏
𝑖)⊤(𝑇𝑏
𝑡𝑖→𝑡𝑗−𝐼)𝑘𝑏
𝑗= (𝑃𝑏
𝑡𝑖𝑞𝑏
𝑖)⊤(𝑃𝑏
𝑡𝑗𝑘𝑏
𝑗) −(𝑞𝑏
𝑖)⊤𝑘𝑏
𝑗.
(14)
Within each continuous valid segment, the correc-
tion decomposes into the difference between prefix-
transformed and untransformed query-key dot prod-
ucts.
This factorization avoids explicitly construct-
ing pairwise transformation matrices for all frame in-
tervals, allowing the prefix accumulation and corre-
sponding query-key transforms to scale linearly with
the temporal length and token count.
B. Experimental Details
B.1. Evaluation Benchmark
We evaluate World-in-Context on WBench [54], a
benchmark for interactive video world models com-
prising 289 cases and 1, 058 interaction turns. Each
case provides an initial image, a world description,
and multi-turn control sequences. We use its NAVIGA-
TION split and FPP↔TPP perspective-switching sub-
set.
Navigation Suite.
WBench benchmarks trajectory-
following navigation across five core dimensions:
• Quality: Assesses perceptual aesthetics, visual fi-
delity, motion dynamics, temporal flickering, and
inter-frame motion smoothness.
2

• Setting: Measures contextual adherence to the spec-
ified prompt, including scene appearance preserva-
tion, environmental style consistency, and subject
persistence.
• Interaction: Evaluates motion responsiveness and
command fidelity by comparing estimated camera
trajectories with action-derived reference trajecto-
ries under translational, rotational, and symmetric
steering commands.
• Consistency: Quantifies structural and geometric co-
herence, capturing character identity stability, ab-
sence of shot cuts, and photometric/geometric con-
sistency across viewpoint transitions.
• Physical: Assesses adherence to real-world physics,
commonsense dynamics,
and causal coherence
throughout interactive rollouts.
In addition to individual dimension scores, the overall
NAVIGATION average is reported.
Perspective
Switching.
For
each
requested
FPP↔TPP transition, WBench evaluates the visual
rollout to determine whether viewpoint transition
smoothness, terminal perspective, and framing geom-
etry satisfy the target specifications. Binary success
decisions are aggregated across all evaluation turns
and reported as an independent metric to isolate
dynamic
viewpoint
adaptability
from
trajectory
tracking.
All evaluations follow the official WBench protocol,
employing its standardized scoring functions, deci-
sion thresholds, and aggregation rules without modi-
fication.
B.2. Setup of Analysis in Section 4
This section details the experimental configurations,
evaluation datasets, and diagnostic metrics employed
in the analyses of Section 4.
View-Dependent Motion Dynamics (Figure 4).
To
isolate regime-dependent motion characteristics, 100
first-person navigation scenarios are randomly sam-
pled from WBench.
For each scene, the world-
state caption is held static, and the action command
sequence is fixed uniformly to continuous forward
movement (W). The model is then prompted under
contrasting view modes (FREE vs. FPP) to observe the
resulting physical trajectories.
Camera trajectories 𝑇𝑡= [𝑅𝑡, 𝑝𝑡; 0, 1] are reconstructed
using ViPE [14]. Positions are expressed relative to
the initial frame coordinates 𝑞𝑡= 𝑅⊤
0 (𝑝𝑡−𝑝0), with
𝑢𝑡= (𝑞𝑡,𝑥, −𝑞𝑡,𝑦) representing transverse (lateral and
vertical) coordinates. The smooth trajectory trend is
isolated via a second-order Savitzky-Golay filter with
a 33-frame window, yielding the transverse residual:
𝑟𝑡= 𝑢𝑡−SG33,2(𝑢)𝑡.
(15)
The oscillation amplitude is quantified via normalized
Root Mean Square error (RMSosc):
RMSosc = 100
𝐿
√︄
1
|I|
∑︁
𝑡∈I

𝑟2
𝑡,𝑥+ 𝑟2
𝑡,𝑦

,
(16)
where 𝐿= |𝑞𝑇−1,𝑧| denotes the net forward displace-
ment, and I is the interior interval excluding 0.5 sec-
onds at sequence boundaries.
This metric captures
transverse body-sway and vertical bobbing while re-
maining invariant to global forward drift.
Emergent View Switching (Figure 3).
The quan-
titative validation evaluates viewpoint transition fi-
delity on the official WBench FPP↔TPP split, adher-
ing strictly to the VLM-based transition criteria and
aggregation protocol described in Section B.1.
Action vs. Camera Control Disentanglement (Fig-
ure 5).
The comparative analysis of action and cam-
era control is conducted on the WBench NAVIGATION
suite, with third-person (TPP) instances excluded to
isolate egocentric/free dynamics.
Evaluation spans
four metrics: two local command-fidelity diagnostics
and two global consistency measures.
• Translation Accuracy: Turn sequences are segmented
into non-overlapping 0.5-second windows 𝑤
=
[𝑎, 𝑏]. Camera poses are tracked via MegaSaM [24].
For each window, the generated local displacement
𝑣𝑤= 𝑅⊤
𝑎(𝑝𝑏−𝑝𝑎) is compared against the action-
integrated reference 𝑣∗
𝑤= (𝑅∗
𝑎)⊤(𝑝∗
𝑏−𝑝∗
𝑎). A win-
dow is classified as accurate if the directional de-
viation satisfies arccos

𝑣⊤
𝑤𝑣∗
𝑤
∥𝑣𝑤∥2∥𝑣∗
𝑤∥2

≤30◦and the
displacement exceeds a minimum motion threshold
∥𝑣𝑤∥2 ≥0.004𝑑0Δ𝑡𝑤, where 𝑑0 is the median initial
positive depth and Δ𝑡𝑤is the window duration.
• Rotation Accuracy: Using identical temporal win-
dows, relative rotation vectors 𝜔𝑤= log(𝑅⊤
𝑎𝑅𝑏)∨
and references 𝜔∗
𝑤= log((𝑅∗
𝑎)⊤𝑅∗
𝑏)∨are extracted
in 𝔰𝔬(3). A window is correct if the angular ori-
entation deviation is within 30◦and the rotational
magnitude satisfies ∥𝜔𝑤∥2 ≥(𝜋/180)Δ𝑡𝑤. Accuracy
scores are averaged over valid active windows and
scaled to a percentage.
• Geometric Consistency: Following WBench, Depth
Anything 3 [26] provides predicted depths 𝐷𝑖, in-
trinsics 𝐾𝑖, and poses 𝑇𝑖.
Source pixels are back-
projected and reprojected into target frame 𝑗: 𝑋𝑗=
𝑇−1
𝑗
𝑇𝑖(𝐷𝑖(𝑢)𝐾−1
𝑖
𝑢), measuring relative depth dispar-
ity 𝑒𝑗
= median[|𝑧′ −𝐷𝑗(𝑢′)|/𝐷𝑗(𝑢′)].
The con-
sistency score is computed as GC
=
100/(1 +
mean𝑗(𝑒𝑗)).
• Photometric Consistency: Following WBench, colors
from source frames are splatted into target frame
𝑗via depth ordering. Denoting ˆ𝐼𝑗as the splatted
image and Ω𝑗as valid pixels, the average PSNR is
evaluated across frames with |Ω𝑗| ≥100. Photo-
3

metric consistency is aggregated as PC = 100 · (1 −
10−PSNR/20).
B.3. Setup of Analysis in Section 5
Additive injection and AdaLN modulation are re-
implemented following the conditioning strategies
of ABot-World [19] and LingBot-World [9], respec-
tively. For explicit view conditioning in these base-
lines, the categorical view mode is embedded at each
latent timestep, spatially broadcast, and fused with
the available action and camera representations. The
resulting composite features are added directly to the
visual patch tokens for additive injection, or projected
to predict affine modulation parameters for AdaLN.
In §5.3 and §5.4, the default World-in-Context is
evaluated prior to stage-2 switching consolidation to
demonstrate the core architectural advantages of in-
context conditioning and transport modeling.
4