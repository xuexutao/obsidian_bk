# Continuous 3D Perception Model with Persistent State

> 来源: https://arxiv.org/abs/2501.12387

**作者**: Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa

## 摘要（arXiv 官方）

We present a unified framework capable of solving a broad range of 3D tasks. Our approach features a stateful recurrent model that continuously updates its state representation with each new observation. Given a stream of images, this evolving state can be used to generate metric-scale pointmaps (per-pixel 3D points) for each new input in an online fashion. These pointmaps reside within a common coordinate system, and can be accumulated into a coherent, dense scene reconstruction that updates as new images arrive. Our model, called CUT3R (Continuous Updating Transformer for 3D Reconstruction), captures rich priors of real-world scenes: not only can it predict accurate pointmaps from image observations, but it can also infer unseen regions of the scene by probing at virtual, unobserved views. Our method is simple yet highly flexible, naturally accepting varying lengths of images that may be either video streams or unordered photo collections, containing both static and dynamic content. We evaluate our method on various 3D/4D tasks and demonstrate competitive or state-of-the-art performance in each. Project Page: https://cut3r.github.io/

Continuous 3D Perception Model with Persistent State
Qianqian Wang1,2∗, Yifei Zhang1∗, Aleksander Holynski1,2, Alexei A. Efros1, Angjoo Kanazawa1
1University of California, Berkeley
2Google DeepMind
Static Scene
Dynamic Scene
Sparse Photo Collection
Figure 1. Continuous 3D Perception. Given a stream of RGB images as input, our approach enables dense 3D reconstruction in an online,
continuous manner, estimating both camera parameters and dense 3D geometry with each incoming frame. Our framework supports various
3D tasks, processes inputs from video sequences and sparse photo collections, and can handle both static and dynamic scenes.
Abstract
We present a unified framework capable of solving a broad
range of 3D tasks. Our approach features a stateful recurrent
model that continuously updates its state representation with
each new observation. Given a stream of images, this evolv-
ing state can be used to generate metric-scale pointmaps
(per-pixel 3D points) for each new input in an online fashion.
These pointmaps reside within a common coordinate system,
and can be accumulated into a coherent, dense scene re-
construction that updates as new images arrive. Our model,
called CUT3R (Continuous Updating Transformer for 3D
Reconstruction), captures rich priors of real-world scenes:
not only can it predict accurate pointmaps from image ob-
servations, but it can also infer unseen regions of the scene
by probing at virtual, unobserved views. Our method is sim-
ple yet highly flexible, naturally accepting varying length of
images that may be either video streams or unordered photo
collections, containing both static and dynamic content. We
evaluate our method on various 3D/4D tasks and demon-
strate competitive or state-of-the-art performance in each.
Project page: https://cut3r.github.io/.
1. Introduction
Humans are online visual learners. We continuously process
streams of visual input, building on what we have learned in
the past while learning in the present. Our prior knowledge
enables us to interpret the world from minimal information;
e.g., upon entering a new restaurant, it only takes a glance
to start inferring its layout and atmosphere. But it doesn’t
stop there—as we accumulate more observations, we con-
tinuously refine our mental model of the 3D environment.
This ability to reconcile our prior knowledge of the world
with a continuous stream of new observations is crucial for
functioning effectively in an ever-changing visual world.
Building on these insights, we introduce an online 3D
perception framework that unifies three key capabilities: 1)
reconstructing 3D scenes from few observations, 2) contin-
uously refining the reconstruction with more observations,
and 3) inferring 3D properties of unobserved scene regions.
We achieve these capabilities by integrating data-driven pri-
ors with a recurrent update mechanism. The learned prior
enables our method to address challenges encountered by tra-
ditional methods (e.g., dynamic objects, sparse observations,
degenerate camera motion), while the ability to continuously
update allows it to process new observations online, and
improve the reconstruction continuously over time.
Specifically, given an image stream, our recurrent model
1
arXiv:2501.12387v1  [cs.CV]  21 Jan 2025

Reconstruction from Image Observations
+ Virtual View
Figure 2. Querying Unseen Regions. In addition to reconstructing
a scene from images, our method can also infer structure for unseen
parts of the scene, given a virtual camera query (shown in blue).
maintains and incrementally updates a persistent internal
state that encodes the scene content. With each new observa-
tion, the model simultaneously updates this state and reads
from it to predict the current view’s 3D properties, including
an estimate of that view’s dense 3D geometry (as a pointmap;
a 3D point per-pixel in a world coordinate frame) and camera
parameters (both intrinsics and extrinsics). Accumulating
these pointmaps enables online dense scene reconstruction,
as illustrated in Fig. 1. Additionally, our framework sup-
ports inferring unobserved parts of the scene: by querying
the internal state with a virtual (unseen) view, parameterized
as a raymap, we can extract the corresponding pointmap and
color for the query view, as depicted in Fig. 2.
Our framework is designed to be general and flexible,
making it well-suited for training on an extensive collec-
tion of datasets and adaptable to diverse inference scenarios.
During training, we leverage a wide variety of 3D data, in-
cluding single images, videos, and photo collections with
partial or full 3D annotations. These datasets span a broad
spectrum of scene types and contexts—static and dynamic,
indoor and outdoor, real and synthetic—enabling the model
to acquire robust and generalizable priors. During inference,
our recurrent framework naturally accepts varying numbers
of images, and supports a wide range of input data settings:
from streaming video to unstructured image collections, in-
cluding wide-baseline or even non-overlapping images. Be-
yond static scenes, it seamlessly handles videos of dynamic
scenes, estimating accurate camera parameters and dense
point clouds for moving parts of the scene.
We evaluate our method on various 3D tasks: monoc-
ular and consistent video depth estimation, camera pose
estimation, and 3D reconstruction, achieving competitive or
state-of-the-art performance in each. We also show that our
method can infer previously unseen structures and continu-
ously refine the reconstruction as new observations arrive.
2. Related Work
Tabula rasa 3D Reconstruction. Many 3D reconstruction
pipelines operate as tabula rasa1, starting from scratch for
1tabula rasa (lat.) – blank slate.
each new scene, and relying solely on the observations
currently available. Notable examples include traditional
methods such as Structure from Motion (SfM) [1, 2, 33,
59, 80, 81, 87, 88, 100] and Simultaneous Localization and
Mapping (SLAM) [14, 20, 23, 26, 46, 64, 66], as well as
more recent approaches such as Neural Radiance Fields
(NeRF) [16, 28, 62, 63, 104] and 3D Gaussian Splatting [45].
The tabula rasa nature of these approaches poses challenges
in handling scenarios with sparse observations or ill-posed
conditions. In contrast, our method leverages data-driven
priors to enable dense 3D reconstruction directly from video
sequences or photo collections, eliminating the need for
known camera extrinsics or intrinsics. The data-driven na-
ture of our method enables it to address challenging cases of
degeneracy, make predictions from as few as a single image,
and infer structures that are unobserved in the input.
Learning-Based 3D Reconstruction. Unlike tabula rasa
reconstruction, many methods integrate data-driven priors
into 3D reconstruction. One prominent direction focuses
on improving traditional reconstruction pipelines, including
replacing hand-crafted components with learning-based alter-
natives (e.g., substituting conventional feature descriptors [7,
59] with learned ones [21, 24, 78, 91]), integrating data-
driven priors [17, 95, 96, 98, 118, 124, 133] into the systems,
or optimizing entire systems end-to-end [94, 96, 103, 119].
Another substantial body of work aims to predict dense 3D
geometry directly from single or paired images. Extensive
research [8, 31, 53, 69, 72, 73, 106, 116] focuses on estimat-
ing monocular depth. While depth maps provide valuable
3D information, converting them into 3D point clouds typ-
ically requires camera intrinsics [10, 38, 69], with camera
extrinsics often left unmodeled. Notably, DUSt3R [107]
predicts two pointmaps from an image pair within the same
coordinate frame, inherently accounting for both camera
intrinsics and extrinsics. However, DUSt3R is tailored for
image pairs and does not inherently support multiple views.
While it can be extended to handle multi-view tasks via an
additional global alignment process, this process operates
offline, is time-consuming, and cannot dynamically update
the reconstruction as new observations are added. In contrast,
our method flexibly handles a varying number of images,
predicting pointmaps for each image in a shared coordinate
frame as they are received.
Continuous Reconstruction Methods. Many traditional
and learning-based methods share the ability to continu-
ously predict 3D structure in an online manner. Monocular
SLAM pipelines [26, 27, 96, 133] recover ego-motion and
3D point clouds in real-time from video sequences but typ-
ically require known camera intrinsics. Our approach is
related to learning based methods [18, 42, 92, 127], such
as 3D-R2N2 [18], which utilize recurrent neural network
architectures [25, 36, 41] for online 3D reconstruction. How-
2

ever, these methods are either object-centric [18, 42, 122]
or require posed images as input [11, 42, 79, 92, 127].
Spann3R [101], developed concurrently with our work, also
demonstrates continuous reconstruction capabilities using
a spatial memory mechanism. However, while Spann3R’s
memory serves primarily as a cache for observed scenes,
our compressed state representation not only captures ob-
served scene content but also enables inferring unobserved
structures. Finally, unlike the methods discussed in this para-
graph that only operate on static scenes, our method also
seamlessly reconstructs dynamic scenes.
Reconstructing Dynamic Scenes from Monocular Videos.
Recovering camera parameters and consistent dense geom-
etry from monocular videos of dynamic scenes presents
significant challenges for traditional SLAM pipelines. Re-
cent approaches address this by leveraging learned depth
priors. Robust-CVD [47] uses deformation splines to align
the depth maps, while CasualSAM [128] finetunes a monoc-
ular depth network on a single video. These methods in-
volve time-consuming per-video optimization. Concurrent
to our work, MonST3R [125] extends DUSt3R to predict
pointmaps for dynamic scenes by finetuning it on dynamic
datasets. However, it still follows DUSt3R’s pairwise for-
mulation and requires global alignment as post-processing.
Another concurrent work, MegaSaM [54], achieves highly
accurate and robust estimation of camera poses and scene
structure for casually captured dynamic videos. Unlike our
approach, MegaSaM is optimization-based (i.e., not feedfor-
ward), uses an explicit 3D state and does not make online
predictions.
3D Scene Priors. Predicting 3D content beyond observed
views has long been studied, but it has recently gained sig-
nificant attention due to advancements in generative mod-
eling. Regression-based few-shot novel view synthesis ap-
proaches [49, 85, 86, 122] typically only generalize across a
class of 3D scenes. With rapid advances in image and video
generative models [71, 76, 77], much of the current research
in 3D generation focuses on transferring priors from images
and videos to 3D [56, 61, 70, 113]. However, most image-
based 3D generation methods require camera parameters as
input, focus on limited data domains (e.g., object-centric), or
require additional 3D distillation processes to extract 3D con-
tent. In contrast to these view-centric approaches that gener-
ate novel views as a proxy for 3D, our method is geometry-
centric: It directly generates metric-scale pointmaps for a
virtual camera query in a scene reconstructed from a set of
images, without requiring their camera intrinsics or poses.
3. Method
Our approach takes as input a stream of images without any
camera information. The image streams can come from ei-
ther video or image collections. As a new image comes in
through the model, it interacts with the latent state represen-
tation, which encodes the understanding of the current 3D
scene. Specifically, the image simultaneously updates the
state with new information and retrieves information stored
in the state. Following the state-image interaction, explicit
3D pointmaps and camera poses are extracted for each view.
The state can also be queried with a virtual view to predict
its corresponding pointmap, capturing unseen parts of the
scene. See Fig. 3 for our method overview.
3.1. State-Input Interaction Mechanism
Our method takes a stream of images as input. For each
current image, It, it is first encoded into token representation
by a ViT encoder [22]:
Ft = Encoderi(It).
(1)
We represent the state also as a set of tokens. Prior to seeing
any image input, the state tokens are initialized as a set of
learnable tokens shared by all scenes. The image tokens
interact with the state in two ways: they update the state with
information from the current image and read the context
from the state, incorporating stored past information. We
refer to these interactions as state-update and state-readout,
respectively. This bidirectional interaction is implemented
using two interconnected transformer decoders [107, 110],
which jointly operate on both image and state tokens:
[z′
t, F ′
t], st = Decoders([z, Ft], st−1).
(2)
Here st−1 and st represent the state tokens before and after
interaction with the image tokens. F ′
t denotes the image
tokens enriched with state information. z is a learnable
“pose token” prepended to the image tokens, whose output z′
t
captures image-level information related to the scene, such
as ego motion. Within the decoders, the outputs from both
sides cross-attend to each other at each decoder block to
ensure effective information transfer.
After this interaction, explicit 3D representation can
be extracted from F ′
t and z′
t.
Specifically, we pre-
dict two pointmaps with corresponding confidence maps,
( ˆ
Xself
t
, Cself
t
) and ( ˆ
Xworld
t
, Cworld
t
). These maps are defined
in two coordinate frames: the input image’s own coordinate
frame and the world frame, respectively, where the world
frame is defined as the coordinate frame of the initial image.
Additionally, we predict the relative transformations between
the two coordinate frames, or, the ego motion, ˆPt:
ˆ
Xself
t
, Cself
t
= Headself(F ′
t)
(3)
ˆ
Xworld
t
, Cworld
t
= Headworld(F ′
t, z′
t)
(4)
ˆPt = Headpose(z′
t),
(5)
where Headself and Headworld are implemented as DPT [72],
and Headpose is implemented as an MLP network, respec-
tively. Please see the supplement for details. We extract
3

…
𝐸!
𝐸!
𝐸!
State 0
State 1
State 2
State 3
Image 1
Image 2
Image 3
…
Pointmaps 
& Cameras
Scene reconstruction
at each time
Input images
Ray map
State 
Readout
𝐸"
Online Dense 3D Reconstruction
Structure Inference via State Readout
State Update
State Readout
State Update
State Readout
State Update
State Readout
Figure 3. Method Overview. Our method performs online dense 3D reconstruction from a stream of images (video frames or a photo
collection) by using a persistent state. Each input image is encoded into visual tokens via a shared-weight ViT encoder. These tokens interact
with state tokens, where state update integrates the current image into the state, and state readout retrieves the past context stored in the state
for predictions. Both processes occur simultaneously through two interconnected ViT decoders. Outputs include pointmaps in world and
camera frames (only world pointmaps are shown) and the camera-to-world transformation. On the right, we demonstrate our method’s
ability to predict unseen views: given a query camera (as a raymap), it reads information from the state to predict its corresponding pointmap,
even for unobserved regions. For these readouts, we do not update the state. The hallucinated pointmap is highlighted with a blue border.
6-DoF pose ˆPt from the pose token z′
t, the rigid transforma-
tion from the current frame to the world. All pointmaps and
poses are in metric scale (i.e., meters).
Although predicting ˆ
Xself
t
, ˆ
Xworld
t
, and ˆPt may seem
redundant, we found this redundancy simplifies training.
It enables each output to receive direct supervision, and
importantly, it facilitates training on datasets with partial
annotations, such as those containing only pose or single-
view depth, thereby broadening the range of usable data.
3.2. Querying the State with Unseen Views
Leveraging past 3D experiences, humans can envision parts
of a scene beyond what is directly observed. We emulate this
ability by extending the state-readout operation to predict
unseen portions of the scene from a virtual camera view.
Specifically, we use a virtual camera as a query to extract
information from the state. The virtual camera’s intrinsics
and extrinsics are represented as a raymap R, a 6-channel
image encoding the origin and direction of rays at each
pixel [29, 112, 126].
Given a query raymap R, we first encode it into token
representations Fr using a separate transformer Encoderr:
Fr = Encoderr(R).
(6)
Then, the rest of the process aligns largely with what is de-
scribed in Sec 3.1. Specifically, we have Fr interact with the
current state through the same decoder module (i.e., shared
weights) as in Eq. 2 to read from the state into F ′
r. Note
that, unlike in the state-image interaction, the state is not
updated here, as the raymap serves solely as a query without
introducing new scene content. Finally, we apply the same
head networks to parse F ′
r into explicit representations, as in
Eq. 3-5. Additionally, we introduce another head Headcolor
to decode color information: ˆIr = Headcolor(F ′
r) which
corresponds to the color of each ray in the raymap R.
Querying the scene with raymaps has an interesting anal-
ogy to Masked Autoencoders (MAE) [35]. In MAE, comple-
tion occurs at the patch level, using the global context of the
entire image. Here, completion is performed at the image
level, leveraging the global context of the 3D scene captured
in the state.
3.3. Training Objective
During training, we provide the model with a sequence of
N images (either from a video sequence or an image collec-
tion). The raymap mode is enabled only for training data
with metric-scale 3D annotations. In these cases, we ran-
domly replace each image with its corresponding raymap
at a certain probability, excluding the first view. When the
scale of the 3D annotation is unknown, raymap querying
is disabled to avoid inconsistencies between the annotation
scale and the scene scale represented in the state. For clarity,
we do not differentiate between predictions generated from
a raymap or an image in this section to simplify the notation.
We denote the pointmap predictions of the current training se-
quence as X = { ˆ
X self, ˆ
X world}, where ˆ
X self = { ˆ
Xself
t
}N
t=1,
ˆ
X world = { ˆ
Xworld
t
}N
t=1 and their corresponding confidence
scores as C.
4

3D regression loss.
Following MASt3R [51], we apply a
confidence-aware regression loss to the pointmaps:
Lconf =
X
(ˆx,c)∈( ˆ
X,C)

c ·

ˆx
ˆs −x
s

2
−α log c

,
(7)
where ˆs and s are scale normalization factors for ˆ
X and X,
respectively. Similar to MASt3R [51], when the groundtruth
pointmaps are metric, we set ˆs := s to enable the model to
learn metric-scale pointmaps.
Pose loss. We parameterize the pose ˆPt as quaternion ˆqt
and translation ˆτt, and minimize the L2 norm between the
prediction and groundtruth:
Lpose =
N
X
t=1

∥ˆqt −qt∥2 +

ˆτt
ˆs −τt
s

2

.
(8)
RGB loss. When the input is raymap, besides the 3D regres-
sion loss, we also apply an MSE loss to enforce the predicted
pixel colors ˆIr to match the groundtruth: Lrgb = || ˆIr−Ir||2
2.
3.4. Training Strategy
Training Datasets.
We train our method on a diverse
set of 32 datasets, covering synthetic and real-world
data, static and dynamic scenes, scene-level and object-
centric configurations, as well as both indoor and outdoor
scenes. Examples of our datasets include CO3Dv2 [74],
ARKitScenes [5], ScanNet++ [121], TartanAir [108],
Waymo [93], MegaDepth [52], MapFree [3], DL3DV [55],
and DynamicStereo [43]. Our flexible formulation allows
training on datasets with partial annotations (i.e. only camera
parameters like RealEstate10K [131], or only single views,
like Synscapes [111]). See the supplement for the full list.
Curriculum Training. Our model is trained with a curricu-
lum. The first stage trains the model on 4-view sequences
from mainly static datasets. The second stage incorporates
dynamic scene datasets, improving the model’s ability to
handle moving objects such as humans, along with datasets
with partial annotations, further enhancing its generalization.
These two stages are trained on 224×224 images to reduce
computational costs, following DUSt3R [107]. In the third
stage, we train with higher resolution, using varied aspect
ratios and setting the maximum side to 512 pixels. Finally,
we freeze the encoder, training only the decoder and heads
on longer sequences spanning 4 to 64 views. This stage
focuses on enhancing inter-scene reasoning and effectively
handling long contexts.
Implementation Detail. We use a ViT-Large model [22] for
the image encoder Encoderi, initialized with DUSt3R en-
coder pretrained weights, and ViT-Base for the decoders.
Both the encoder and decoders operate on 16×16 pixel
patches. The state consists of 768 tokens, each with a di-
mensionality of 768. The raymap encoder Encoderr is a
lightweight encoder with 2 blocks. We use Adam-W opti-
mizer [58] with an initial learning rate of 1e−4, applying lin-
ear warmup followed by cosine decay. We train our model on
eight A100 NVIDIA GPUs each with 80G memory. Please
refer to the supplement for more details.
4. Experiments
We evaluate our method across a range of 3D tasks, including
single and video depth estimation (Sec. 4.1), camera pose
estimation (Sec. 4.2), and 3D reconstruction (Sec. 4.3).
Baselines. Our primary set of baselines are DUSt3R [107],
MASt3R [51], Spann3R [101], and MonST3R [125], where
the latter two are concurrent works. MonST3R finetunes
DUSt3R on dynamic datasets to handle dynamic scenes,
while Spann3R extends DUSt3R to support varying number
of images via additional spatial memory and operates online,
similar to our method. DUSt3R, MASt3R, and MonST3R
can only take a pair of views as input, and require an extra
global alignment (GA) stage to consolidate the pairwise
predictions. Both MASt3R and our method predict metric
pointmaps, whereas others predict relative pointmaps.
4.1. Monocular and Video Depth Estimation
Mono-Depth Estimation.
Following MonST3R [125], we
evaluate monocular depth estimation on KITTI [30], Sin-
tel [12], Bonn [68] and NYU-v2 [65] datasets covering dy-
namic and static, indoor and outdoor scenes. These datasets
are excluded from training, enabling zero-shot performance
evaluation across domains. We use absolute relative error
(Abs Rel) and δ < 1.25 (percentage of predicted depths
within a 1.25-factor of true depth) as metrics, with per-frame
median scaling per DUSt3R [107]. Results in Tab. 1 show
our method achieves state-of-the-art or competitive perfor-
mance, leading on Bonn and and NYU-v2 and ranking sec-
ond on KITTI.
Sintel
Bonn
KITTI
NYU-v2
Method
Abs Rel ↓δ<1.25 ↑Abs Rel ↓δ<1.25 ↑Abs Rel ↓δ<1.25 ↑Abs Rel ↓δ<1.25 ↑
DUSt3R
0.424
58.7
0.141
82.5
0.112
86.3
0.080
90.7
MASt3R
0.340
60.4
0.142
82.0
0.079
94.7
0.129
84.9
MonST3R
0.358
54.8
0.076
93.9
0.100
89.3
0.102
88.0
Spann3R
0.470
53.9
0.118
85.9
0.128
84.6
0.122
84.9
Ours
0.428
55.4
0.063
96.2
0.092
91.3
0.086
90.9
Table 1. Single-frame Depth Evaluation. We report the perfor-
mance on Sintel, Bonn, KITTI, and NYU-v2 (static) datasets.
Video Depth Estimation. Video depth estimation evaluates
per-frame depth quality and inter-frame depth consistency
by aligning predicted depth maps to ground truth using a per-
sequence scale. For metric pointmap methods like ours and
MASt3R, we also report results without alignment. Compar-
isons for both methods are presented in Tab. 2.
Under per-sequence scale alignment, our method consis-
tently outperforms DUSt3R [107] and MASt3R [51]. The
5

Sintel
BONN
KITTI
Alignment
Method
Optim. Onl. Abs Rel ↓δ<1.25 ↑Abs Rel ↓δ<1.25 ↑Abs Rel ↓δ <1.25 ↑
FPS
Per-sequence scale
DUSt3R-GA [107]
✓
0.656
45.2
0.155
83.3
0.144
81.3
0.76
MASt3R-GA [51]
✓
0.641
43.9
0.252
70.1
0.183
74.5
0.31
MonST3R-GA [125]
✓
0.378
55.8
0.067
96.3
0.168
74.4
0.35
Spann3R [101]
✓
0.622
42.6
0.144
81.3
0.198
73.7
13.55
Ours
✓
0.421
47.9
0.078
93.7
0.118
88.1
16.58
Metric scale
MASt3R-GA [51]
✓
1.022
14.3
0.272
70.6
0.467
15.2
0.31
Ours
✓
1.029
23.8
0.103
88.5
0.122
85.5
16.58
Table 2. Video Depth Evaluation. We report scale-invariant depth and metric depth accuracy on Sintel, Bonn, and KITTI datasets. Methods
requiring global alignment are marked “GA”, while “Optim.” and “Onl.” indicate optimization-based and online methods, respectively.
We also report the FPS on KITTI dataset using 512× 144 image resolution for all methods on an A100 GPU, except Spann3R which only
supports 224×224 inputs. We present a subset of baselines here; please refer to the supplementary material for full comparisons.
Sintel
TUM-dynamics
ScanNet
Method
Optim. Onl. ATE ↓RPE trans ↓RPE rot ↓ATE ↓RPE trans ↓RPE rot ↓ATE ↓RPE trans ↓RPE rot ↓
Particle-SfM [129]
✓
0.129
0.031
0.535
-
-
-
0.136
0.023
0.836
Robust-CVD [47]
✓
0.360
0.154
3.443
0.153
0.026
3.528
0.227
0.064
7.374
CasualSAM [128]
✓
0.141
0.035
0.615
0.071
0.010
1.712
0.158
0.034
1.618
DUSt3R-GA [107]
✓
0.417
0.250
5.796
0.083
0.017
3.567
0.081
0.028
0.784
MASt3R-GA [51]
✓
0.185
0.060
1.496
0.038
0.012
0.448
0.078
0.020
0.475
MonST3R-GA [125]
✓
0.111
0.044
0.869
0.098
0.019
0.935
0.077
0.018
0.529
DUSt3R [107]
✓
0.290
0.132
7.869
0.140
0.106
3.286
0.246
0.108
8.210
Spann3R [101]
✓
0.329
0.110
4.471
0.056
0.021
0.591
0.096
0.023
0.661
Ours
✓
0.213
0.066
0.621
0.046
0.015
0.473
0.099
0.022
0.600
Table 3. Evaluation on Camera Pose Estimation on Sintel [12], TUM-dynamic [89], and ScanNet [19] datasets. Our method achieves the
best overall performance among all online methods.
global alignment they use assumes that the scene is static,
and enforcing multi-view consistency can only improve the
reconstruction of static regions but may impair the recon-
struction of moving objects. In contrast, our method lever-
ages implicit alignment through the state, making it adapt-
able to both static and dynamic scenes while remaining fully
online. Our method also significantly outperforms the other
online method, Spann3R [101], which is based on spatial
memory designed with a static scene assumption and trained
only on static 3D scenes. MonST3R [125] achieves state-
of-the-art performance, but it relies on an additional global
alignment stage that incorporates extra input like optical flow
in its optimization. In comparison, our method performs
comparably, or even better on the KITTI dataset, while re-
maining online and achieving nearly 50× speedup. In the
metric-scale setting, our method also significantly outper-
forms MASt3R for most metrics.
4.2. Camera Pose Estimation
Following MonST3R [125], we evaluate camera pose esti-
mation accuracy on Sintel [12], TUM dynamics [89], and
ScanNet [19] datasets. Note that both Sintel and TUM-
dynamics contain major dynamic objects, making it chal-
lenging for traditional SfM and SLAM systems. We report
Absolute Translation Error (ATE), Relative Translation Er-
ror (RPE trans), and Relative Rotation Error (RPE rot) after
Sim(3) alignment with the ground truth, as in [17, 125, 129].
Unlike most visual odometry methods [17, 34, 96], our
method does not require any camera calibration. We compare
to baselines that share this feature. Most prior approaches
do so through test-time optimization, as seen in Robust-
CVD [47] and CasualSAM [128], which jointly estimate
camera parameters and dense depth maps per sequence.
The results are presented in Tab. 3. We separately high-
light the leading approaches for methods that require addi-
tional optimization and those that do not (i.e., online). For
the online category, we additionally include DUSt3R [107]
where we align all video frames with first frame, without
using GA. Although a gap persists between optimization-
based and online methods, our approach achieves the best
overall performance among online methods, particularly in
dynamic scenes.
4.3. 3D Reconstruction
We evaluate scene-level reconstruction on the 7-scenes [83]
and NRGBD [4] datasets using accuracy (Acc), completion
(Comp), and normal consistency (NC) metrics, as in prior
works [4, 101, 102, 107, 132]. To assess performance on
image collections with minimal or no overlap, we evaluate
using sparsely sampled images: 3 to 5 frames per scene for
6

Input Video
Spann3R
MonST3R
Ours
Figure 4. Qualitative Results on In-the-wild Internet Videos. We compare our method with concurrent works Spann3R [101] and
MonST3R [125]. Our method achieves the best qualitative results.
7 scenes [83]
NRGBD [4]
Acc↓
Comp↓
NC↑
Acc↓
Comp↓
NC↑
Method
Optim. Onl.
Mean
Med.
Mean
Med.
Mean
Med.
Mean
Med.
Mean
Med.
Mean
Med.
FPS
DUSt3R-GA [107]
✓
0.146
0.077
0.181
0.067
0.736
0.839
0.144
0.019
0.154
0.018
0.870
0.982
0.68
MASt3R-GA [51]
✓
0.185
0.081
0.180
0.069
0.701
0.792
0.085
0.033
0.063
0.028
0.794
0.928
0.34
MonST3R-GA [51]
✓
0.248
0.185
0.266
0.167
0.672
0.759
0.272
0.114
0.287
0.110
0.758
0.843
0.39
Spann3R [101]
✓
0.298
0.226
0.205
0.112
0.650
0.730
0.416
0.323
0.417
0.285
0.684
0.789
12.97
Ours
✓
0.126
0.047
0.154
0.031
0.727
0.834
0.099
0.031
0.076
0.026
0.837
0.971
17.00
Table 4. 3D reconstruction comparison on 7-Scenes [83] and NRGBD [4] datasets. While operating online, our method achieves
competitive performance, on par with and even surpassing offline methods that employ global alignment.
the 7-Scenes dataset and 2 to 4 frames per scene for the
NRGBD dataset. The results are presented in Tab. 4. Our
method significantly outperforms the other online approach
Spann3R [101], and achieves comparable or sometimes bet-
ter results than the top optimization-based method, DUSt3R-
GA, while operating online at 25× the speed. This highlights
our method’s effectiveness with sparse image collections.
Qualitative Results. We compare the reconstruction quality
of our method with Spann3R [101] and MonST3R [125]
on in the wild Internet videos in Fig. 4. Spann3R [101] is
neither designed nor trained on dynamic scenes, making it
less effective at handling moving objects, such as humans.
MonST3R [125] is finetuned on dynamic scenes, potentially
overfitting and degrading performance on static 3D scenes.
In contrast, our method operates online and achieves state-of-
the-art performance across both static and dynamic scenes.
4.4. Analysis
State Update Analysis. Our model continuously updates
its state representation as new data arrives, relying solely on
past and current observations without knowledge of future
inputs. As more observations accumulate, the state should be
able to refine its understanding of the 3D world, leading to
improved predictions. We demonstrate this capability of our
method in Tab. 5. Using the same experimental setup as in
Sec. 4.3, we introduce an additional version of our approach
called “revisiting”: we first run our method online to obtain
the final state that has seen all images, then we freeze this
7

7-Scenes
NRGBD
Method
Acc↓
Comp↓
NC↑
Acc↓
Comp↓
NC↑
DUSt3R-GA [107]
0.146
0.181
0.736
0.144
0.154
0.870
Ours
0.126
0.154
0.727
0.099
0.076
0.837
Ours Revisit
0.113
0.107
0.732
0.094
0.076
0.844
Table 5. State Update Analysis on 7-Scenes [83] and NRGBD [4]
datasets.
online
revisiting
…
…
Figure 5. State Update Analysis. Compared to online, revisiting
incorporates global context which improves overall reconstruction
results, especially in the highlighted regions.
state and use it to process the same set of images again to
generate predictions.
This setup differs from the online setup by allowing the
state to see the full context of the scene during the first
run. As shown in Tab. 5, revisiting improves performance
compared to the online version, especially for accuracy. This
verifies that the state representation effectively updates with
additional observations. See Fig. 5 for a qualitative example.
Inferring Unseen Regions via State Readout. To the best
of our knowledge, our method is the first to enable the infer-
ence of unseen structures in metric scale for general scenes,
supporting both single and multiple views, without requiring
camera intrinsics or poses for the input images. We show
qualitative results of the generated structures in Fig. 6. For
this experiment, we use the validation set of the MapFree [3]
and ARKitScenes datasets, both with metric camera pose
annotations. Importantly, these scenes are not seen by our
model during training. In each example, we input a single
image to our model and then query the state using a raymap
of the ground truth image (unseen by the state). The model
is expected to generate pointmaps that align with the ground
truth. While the predictions may lack some high-frequency
details – owing to the deterministic nature of our approach –
Input Image
GT Image
Pred. Depth
Before Query
After Query
Figure 6. Inferring New Structure via. State Readout. From top
to bottom: the input image; the ground truth (GT) image, used to
query the state via its camera parameters (note: GT image is not
given to the model); the depth map from the predicted pointmap;
the pointmap prediction of the input image alone; and the pointmap
combined with the predicted pointmap in a shared coordinate frame.
they accurately follows the viewpoint transformation, even
with significant viewpoint changes between the input and
ground truth. In addition, our method generates new struc-
tures beyond what what is observed in the input, such as the
bushes in the first example, the ground in the second, the
oven in the third, and the stool in the last. This demonstrates
that our method captures generalized 3D scene priors.
5. Conclusion
In this paper, we propose an online 3D perception model
with a continuously updating, persistent state. Given an im-
age stream, our model simultaneously performs state-update
(which updates the state) and state-readout (which retrieves
information from the state) for each observation. The out-
put at each step includes camera parameters and pointmaps
in the world frame, which accumulate into a dense recon-
struction of the scene over time. This simple formulation is
general yet powerful enough to solve a number of 3D/4D
tasks, handling both videos and photo collections, and pro-
cessing both static and dynamic scenes. The generalized 3D
scene priors captured by our method enable the inference
of new structures unobserved in the input views by probing
the state with a raymap. Experimental results on extensive
3D/4D tasks verify the effectiveness of our method.
8

Limitations.
As with many online methods, our approach
may eventually drift over very long sequences due to the ab-
sence of global alignment. Extending our work with explicit
or implicit global alignment is an interesting future direction.
Additionally, since our structure generation is performed
via a deterministic rather than a generative approach, it can
produce blurry results – especially when extrapolating view-
points too far from the provided views – a common issue
with regression-based methods. Incorporating a generative
formulation could address this limitation. Finally, training
recurrent networks can be time-consuming. We leave these
fundamental directions for future work.
Acknowledgements.
We would like to thank Noah
Snavely, Haiwen Feng, Chung Min Kim, Justin Kerr, Song-
wei Ge, Chenfeng Xu, Letian Fu, and Ren Wang for help-
ful discussions.
We especially thank Noah Snavely for
his guidance and support.
This project is supported in
part by DARPA No. HR001123C0021, IARPA DOI/IBC
No. 140D0423C0035, NSF:CNS-2235013, Bakar Fellows,
ONR, MURI, TRI and BAIR Sponsors. The views and con-
clusions contained herein are those of the authors and do
not represent the official policies or endorsements of these
institutions.
References
[1] Sameer Agarwal, Noah Snavely, Steven M Seitz, and
Richard Szeliski. Bundle adjustment in the large. In Com-
puter Vision–ECCV 2010: 11th European Conference on
Computer Vision, Heraklion, Crete, Greece, September 5-11,
2010, Proceedings, Part II 11, pages 29–42. Springer, 2010.
2
[2] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian
Simon, Brian Curless, Steven M Seitz, and Richard Szeliski.
Building rome in a day. Communications of the ACM, 54
(10):105–112, 2011. 2
[3] Eduardo Arnold, Jamie Wynn, Sara Vicente, Guillermo
Garcia-Hernando,
´Aron
Monszpart,
Victor
Adrian
Prisacariu, Daniyar Turmukhambetov, and Eric Brachmann.
Map-free visual relocalization: Metric pose relative to a
single image. In ECCV, 2022. 5, 8, 15, 16
[4] Dejan Azinovi´c, Ricardo Martin-Brualla, Dan B Goldman,
Matthias Nießner, and Justus Thies. Neural rgb-d surface
reconstruction. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 6290–
6301, 2022. 6, 7, 8
[5] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry,
Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe,
Daniel Kurz, Arik Schwartz, and Elad Shulman. ARK-
itscenes - a diverse real-world dataset for 3d indoor scene
understanding using mobile RGB-d data. In Thirty-fifth Con-
ference on Neural Information Processing Systems Datasets
and Benchmarks Track (Round 1), 2021. 5, 16
[6] Zuria Bauer, Francisco Gomez-Donoso, Edmanuel Cruz,
Sergio Orts-Escolano, and Miguel Cazorla. Uasol, a large-
scale high-resolution outdoor stereo dataset. Scientific data,
6(1):162, 2019. 16
[7] Herbert Bay, Andreas Ess, Tinne Tuytelaars, and Luc
Van Gool. Speeded-up robust features (surf). Computer
vision and image understanding, 110(3):346–359, 2008. 2
[8] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter
Wonka, and Matthias M¨uller. Zoedepth: Zero-shot trans-
fer by combining relative and metric depth. arXiv preprint
arXiv:2302.12288, 2023. 2
[9] Michael J Black, Priyanka Patel, Joachim Tesch, and Jin-
long Yang. Bedlam: A synthetic dataset of bodies exhibit-
ing detailed lifelike animated motion. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 8726–8737, 2023. 15, 16
[10] Aleksei Bochkovskii, Ama¨el Delaunoy, Hugo Germain, Mar-
cel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen
Koltun. Depth pro: Sharp monocular metric depth in less
than a second. arXiv, 2024. 2
[11] Aljaz Bozic, Pablo Palafox, Justus Thies, Angela Dai, and
Matthias Nießner. Transformerfusion: Monocular rgb scene
reconstruction using transformers. Advances in Neural In-
formation Processing Systems, 34:1403–1414, 2021. 3
[12] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black. A
naturalistic open source movie for optical flow evaluation.
In European Conf. on Computer Vision (ECCV), pages 611–
625. Springer-Verlag, 2012. 5, 6, 17
[13] Yohann Cabon, Naila Murray, and Martin Humenberger.
Virtual kitti 2, 2020. 16
[14] Cesar Cadena, Luca Carlone, Henry Carrillo, Yasir Latif, Da-
vide Scaramuzza, Jos´e Neira, Ian Reid, and John J Leonard.
Past, present, and future of simultaneous localization and
mapping: Toward the robust-perception age. IEEE Transac-
tions on robotics, 32(6):1309–1332, 2016. 2
[15] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej
Halber, Matthias Niessner, Manolis Savva, Shuran Song,
Andy Zeng, and Yinda Zhang. Matterport3d: Learning
from rgb-d data in indoor environments. arXiv preprint
arXiv:1709.06158, 2017. 16
[16] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and
Hao Su. Tensorf: Tensorial radiance fields. In European
conference on computer vision, pages 333–350. Springer,
2022. 2
[17] Weirong Chen, Le Chen, Rui Wang, and Marc Pollefeys.
Leap-vo: Long-term effective any point tracking for visual
odometry. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 19844–
19853, 2024. 2, 6, 16, 17
[18] Christopher B Choy, Danfei Xu, JunYoung Gwak, Kevin
Chen, and Silvio Savarese. 3d-r2n2: A unified approach for
single and multi-view 3d object reconstruction. In Computer
Vision–ECCV 2016: 14th European Conference, Amsterdam,
The Netherlands, October 11-14, 2016, Proceedings, Part
VIII 14, pages 628–644. Springer, 2016. 2, 3
[19] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber,
Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-
annotated 3d reconstructions of indoor scenes. In Proc.
Computer Vision and Pattern Recognition (CVPR), IEEE,
2017. 6, 16, 17
9

[20] Andrew J Davison, Ian D Reid, Nicholas D Molton, and
Olivier Stasse. Monoslam: Real-time single camera slam.
IEEE transactions on pattern analysis and machine intelli-
gence, 29(6):1052–1067, 2007. 2
[21] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabi-
novich. Superpoint: Self-supervised interest point detection
and description. In Proceedings of the IEEE conference on
computer vision and pattern recognition workshops, pages
224–236, 2018. 2
[22] Alexey Dosovitskiy. An image is worth 16x16 words: Trans-
formers for image recognition at scale.
arXiv preprint
arXiv:2010.11929, 2020. 3, 5
[23] Hugh Durrant-Whyte and Tim Bailey. Simultaneous local-
ization and mapping: part i. IEEE robotics & automation
magazine, 13(2):99–110, 2006. 2
[24] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Polle-
feys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-net:
A trainable cnn for joint description and detection of lo-
cal features. In Proceedings of the ieee/cvf conference on
computer vision and pattern recognition, pages 8092–8101,
2019. 2
[25] Jeffrey L Elman. Finding structure in time. Cognitive sci-
ence, 14(2):179–211, 1990. 2
[26] Jakob Engel, Thomas Sch¨ops, and Daniel Cremers. Lsd-
slam: Large-scale direct monocular slam.
In European
conference on computer vision, pages 834–849. Springer,
2014. 2
[27] Christian Forster, Zichao Zhang, Michael Gassner, Manuel
Werlberger, and Davide Scaramuzza. Svo: Semidirect visual
odometry for monocular and multicamera systems. IEEE
Transactions on Robotics, 33(2):249–265, 2016. 2
[28] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong
Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels:
Radiance fields without neural networks. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 5501–5510, 2022. 2
[29] Ruiqi Gao*, Aleksander Holynski*, Philipp Henzler,
Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan,
Jonathan T Barron, and Ben Poole*. Cat3d: Create any-
thing in 3d with multi-view diffusion models. arXiv preprint
arXiv:2405.10314, 2024. 4
[30] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel
Urtasun. Vision meets robotics: The kitti dataset. The
International Journal of Robotics Research, 32(11):1231–
1237, 2013. 5
[31] Cl´ement Godard, Oisin Mac Aodha, Michael Firman, and
Gabriel J Brostow. Digging into self-supervised monocular
depth estimation. In Proceedings of the IEEE/CVF inter-
national conference on computer vision, pages 3828–3838,
2019. 2
[32] Jose L. G´omez, Manuel Silva, Antonio Seoane, Agn`es
Borr´as, Mario Noriega, Germ´an Ros, Jose A. Iglesias-
Guitian, and Antonio M. L´opez. All for one, and one for all:
Urbansyn dataset, the third musketeer of synthetic driving
scenes, 2023. 16
[33] Richard Hartley and Andrew Zisserman.
Multiple view
geometry in computer vision. Cambridge university press,
2003. 2
[34] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
Deep residual learning for image recognition. In Proceed-
ings of the IEEE conference on computer vision and pattern
recognition, pages 770–778, 2016. 6
[35] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr
Doll´ar, and Ross Girshick. Masked autoencoders are scal-
able vision learners. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
16000–16009, 2022. 4
[36] S Hochreiter. Long short-term memory. Neural Computation
MIT-Press, 1997. 2
[37] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou,
Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and
Hao Tan. Lrm: Large reconstruction model for single image
to 3d. arXiv preprint arXiv:2311.04400, 2023. 15
[38] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long,
Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and
Shaojie Shen. Metric3d v2: A versatile monocular geometric
foundation model for zero-shot metric depth and surface
normal estimation. arXiv preprint arXiv:2404.15506, 2024.
2
[39] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong
Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter:
Generating consistent long depth sequences for open-world
videos. arXiv preprint arXiv:2409.02095, 2024. 16, 17
[40] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra
Ahuja, and Jia-Bin Huang. Deepmvs: Learning multi-view
stereopsis. In IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2018. 16
[41] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals,
Andrew Zisserman, and Joao Carreira. Perceiver: General
perception with iterative attention. In International confer-
ence on machine learning, pages 4651–4664. PMLR, 2021.
2
[42] Abhishek Kar, Christian H¨ane, and Jitendra Malik. Learning
a multi-view stereo machine. Advances in neural informa-
tion processing systems, 30, 2017. 2, 3
[43] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia
Neverova, Andrea Vedaldi, and Christian Rupprecht. Dy-
namicstereo: Consistent dynamic depth from stereo videos.
CVPR, 2023. 5, 16
[44] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Met-
zger, Rodrigo Caye Daudt, and Konrad Schindler. Repurpos-
ing diffusion-based image generators for monocular depth
estimation. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2024.
15, 17
[45] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler,
and George Drettakis. 3d gaussian splatting for real-time
radiance field rendering. ACM Trans. Graph., 42(4):139–1,
2023. 2
[46] Georg Klein and David Murray. Parallel tracking and map-
ping for small ar workspaces. In 2007 6th IEEE and ACM
international symposium on mixed and augmented reality,
pages 225–234. IEEE, 2007. 2
[47] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Ro-
bust consistent video depth estimation. In Proceedings of
10

the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 1611–1621, 2021. 3, 6, 16, 17
[48] Anastasiia Kornilova, Marsel Faizullin, Konstantin Pakulev,
Andrey Sadkov, Denis Kukushkin, Azat Akhmetyanov,
Timur Akhtyamov, Hekmat Taherinejad, and Gonzalo Ferrer.
Smartportraits: Depth powered handheld smartphone dataset
of human portraits for state estimation, reconstruction and
synthesis. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 21318–
21329, 2022. 15, 16
[49] Zihang Lai, Sifei Liu, Alexei A Efros, and Xiaolong Wang.
Video autoencoder: self-supervised disentanglement of
static 3d structure and motion.
In Proceedings of the
IEEE/CVF International Conference on Computer Vision,
pages 9730–9740, 2021. 3
[50] Hoang-An Le, Thomas Mensink, Partha Das, Sezer
Karaoglu, and Theo Gevers. Eden: Multimodal synthetic
dataset of enclosed garden scenes. In Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer
Vision, pages 1579–1589, 2021. 15, 16
[51] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Ground-
ing image matching in 3d with mast3r.
arXiv preprint
arXiv:2406.09756, 2024. 5, 6, 7, 16, 17
[52] Zhengqi Li and Noah Snavely. Megadepth: Learning single-
view depth prediction from internet photos. In Computer
Vision and Pattern Recognition (CVPR), 2018. 5
[53] Zhengqi Li and Noah Snavely. Megadepth: Learning single-
view depth prediction from internet photos. In Proceedings
of the IEEE conference on computer vision and pattern
recognition, pages 2041–2050, 2018. 2, 16
[54] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang,
Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holyn-
ski, and Noah Snavely. Megasam: Accurate, fast, and robust
structure and motion from casual dynamic videos. arXiv
preprint arXiv:2412.04463, 2024. 3
[55] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin,
Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu,
et al.
Dl3dv-10k: A large-scale scene dataset for deep
learning-based 3d vision. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 22160–22169, 2024. 5, 15, 16
[56] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tok-
makov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3:
Zero-shot one image to 3d object, 2023. 3
[57] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan,
Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi.
Hoi4d: A 4d egocentric dataset for category-level human-
object interaction. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
21013–21022, 2022. 15, 16
[58] I Loshchilov. Decoupled weight decay regularization. arXiv
preprint arXiv:1711.05101, 2017. 5
[59] David G Lowe.
Distinctive image features from scale-
invariant keypoints. International journal of computer vision,
60:91–110, 2004. 2
[60] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nali-
vayko, and Andr´es Bruhn. Spring: A high-resolution high-
detail dataset and benchmark for scene flow, optical flow
and stereo. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 4981–4991,
2023. 16
[61] Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, and
Andrea Vedaldi. Realfusion: 360 reconstruction of any
object from a single image. In CVPR, 2023. 3
[62] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik,
Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf:
Representing scenes as neural radiance fields for view syn-
thesis. Communications of the ACM, 65(1):99–106, 2021.
2
[63] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexan-
der Keller. Instant neural graphics primitives with a mul-
tiresolution hash encoding. ACM transactions on graphics
(TOG), 41(4):1–15, 2022. 2
[64] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D
Tardos. Orb-slam: a versatile and accurate monocular slam
system. IEEE transactions on robotics, 31(5):1147–1163,
2015. 2
[65] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob
Fergus. Indoor segmentation and support inference from
rgbd images. In ECCV, 2012. 5
[66] Richard A Newcombe, Steven J Lovegrove, and Andrew J
Davison. Dtam: Dense tracking and mapping in real-time.
In 2011 international conference on computer vision, pages
2320–2327. IEEE, 2011. 2
[67] Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 3d
ken burns effect from a single image. ACM Transactions on
Graphics, 38(6):184:1–184:15, 2019. 16
[68] E. Palazzolo, J. Behley, P. Lottes, P. Gigu`ere, and C. Stach-
niss. ReFusion: 3D Reconstruction in Dynamic Environ-
ments for RGB-D Cameras Exploiting Residuals. 2019.
5
[69] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia
Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth:
Universal monocular metric depth estimation. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 10106–10116, 2024. 2
[70] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Milden-
hall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv
preprint arXiv:2209.14988, 2022. 3
[71] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray,
Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever.
Zero-shot text-to-image generation. In International confer-
ence on machine learning, pages 8821–8831. Pmlr, 2021.
3
[72] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-
sion transformers for dense prediction. ICCV, 2021. 2, 3,
15
[73] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad
Schindler, and Vladlen Koltun. Towards robust monocu-
lar depth estimation: Mixing datasets for zero-shot cross-
dataset transfer. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 44(3), 2022. 2
[74] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler,
Luca Sbordone, Patrick Labatut, and David Novotny. Com-
mon objects in 3d: Large-scale learning and evaluation of
11

real-life 3d category reconstruction. In International Con-
ference on Computer Vision, 2021. 5, 16
[75] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit
Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb,
and Joshua M. Susskind. Hypersim: A photorealistic syn-
thetic dataset for holistic indoor scene understanding. In
International Conference on Computer Vision (ICCV) 2021,
2021. 16
[76] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Patrick Esser, and Bj¨orn Ommer. High-resolution image
synthesis with latent diffusion models. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 10684–10695, 2022. 3
[77] Chitwan Saharia, William Chan, Saurabh Saxena, Lala
Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour,
Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans,
et al. Photorealistic text-to-image diffusion models with
deep language understanding. Advances in neural informa-
tion processing systems, 35:36479–36494, 2022. 3
[78] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz,
and Andrew Rabinovich.
SuperGlue: Learning feature
matching with graph neural networks. In CVPR, 2020. 2
[79] Mohamed Sayed, John Gibson, Jamie Watson, Victor
Prisacariu, Michael Firman, and Cl´ement Godard. Simplere-
con: 3d reconstruction without 3d convolutions. In European
Conference on Computer Vision, pages 1–19. Springer, 2022.
3
[80] Johannes Lutz Sch¨onberger and Jan-Michael Frahm.
Structure-from-motion revisited. In Conference on Com-
puter Vision and Pattern Recognition (CVPR), 2016. 2, 15
[81] Steven M Seitz, Brian Curless, James Diebel, Daniel
Scharstein, and Richard Szeliski. A comparison and eval-
uation of multi-view stereo reconstruction algorithms. In
2006 IEEE computer society conference on computer vision
and pattern recognition (CVPR’06), pages 519–528. IEEE,
2006. 2
[82] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang,
Yujun Shen, Matteo Poggi, and Yiyi Liao. Learning tem-
porally consistent video depth from video diffusion priors.
arXiv preprint arXiv:2406.01493, 2024. 16, 17
[83] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram
Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene
coordinate regression forests for camera relocalization in
rgb-d images. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 2930–2937,
2013. 6, 7, 8
[84] Samarth Sinha, Roman Shapovalov, Jeremy Reizenstein, Ig-
nacio Rocco, Natalia Neverova, Andrea Vedaldi, and David
Novotny. Common pets in 3d: Dynamic new-view synthesis
of real-life deformable categories. CVPR, 2023. 15, 16
[85] Vincent Sitzmann, Michael Zollh¨ofer, and Gordon Wetzstein.
Scene representation networks: Continuous 3d-structure-
aware neural scene representations. Advances in Neural
Information Processing Systems, 32, 2019. 3
[86] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh
Tenenbaum, and Fredo Durand. Light field networks: Neu-
ral scene representations with single-evaluation rendering.
Advances in Neural Information Processing Systems, 34:
19313–19325, 2021. 3
[87] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo
tourism: exploring photo collections in 3d. In ACM siggraph
2006 papers, pages 835–846. 2006. 2
[88] Noah Snavely, Steven M Seitz, and Richard Szeliski. Model-
ing the world from internet photo collections. International
journal of computer vision, 80:189–210, 2008. 2
[89] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram
Burgard, and Daniel Cremers. A benchmark for the evalua-
tion of rgb-d slam systems. In 2012 IEEE/RSJ international
conference on intelligent robots and systems, pages 573–580.
IEEE, 2012. 6, 17
[90] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen
Bo, and Yunfeng Liu. Roformer: Enhanced transformer with
rotary position embedding. Neurocomputing, 568:127063,
2024. 15
[91] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and
Xiaowei Zhou. Loftr: Detector-free local feature matching
with transformers. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
8922–8931, 2021. 2
[92] Jiaming Sun, Yiming Xie, Linghao Chen, Xiaowei Zhou,
and Hujun Bao. Neuralrecon: Real-time coherent 3d re-
construction from monocular video. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recog-
nition, pages 15598–15607, 2021. 2, 3
[93] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien
Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou,
Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han,
Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Et-
tinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang,
Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov.
Scalability in perception for autonomous driving: Waymo
open dataset. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2020.
5, 16
[94] Chengzhou Tang and Ping Tan. Ba-net: Dense bundle ad-
justment network. arXiv preprint arXiv:1806.04807, 2018.
2
[95] Keisuke Tateno, Federico Tombari, Iro Laina, and Nassir
Navab. Cnn-slam: Real-time dense monocular slam with
learned depth prediction. In Proceedings of the IEEE con-
ference on computer vision and pattern recognition, pages
6243–6252, 2017. 2
[96] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam
for monocular, stereo, and rgb-d cameras. Advances in
neural information processing systems, 34:16558–16569,
2021. 2, 6, 16, 17
[97] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch vi-
sual odometry. Advances in Neural Information Processing
Systems, 2023. 16, 17
[98] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch vi-
sual odometry. Advances in Neural Information Processing
Systems, 36, 2024. 2
[99] Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger.
Smd-nets: Stereo mixture density networks. In Proceedings
12

of the IEEE/CVF conference on computer vision and pattern
recognition, pages 8942–8952, 2021. 16
[100] Bill Triggs, Philip F McLauchlan, Richard I Hartley,
and Andrew W Fitzgibbon. Bundle adjustment—a mod-
ern synthesis.
In Vision Algorithms: Theory and Prac-
tice: International Workshop on Vision Algorithms Corfu,
Greece, September 21–22, 1999 Proceedings, pages 298–
372. Springer, 2000. 2
[101] Hengyi Wang and Lourdes Agapito. 3d reconstruction with
spatial memory. arXiv preprint arXiv:2408.16061, 2024. 3,
5, 6, 7, 16, 17
[102] Hengyi Wang, Jingwen Wang, and Lourdes Agapito. Co-
slam: Joint coordinate and sparse parametric encodings for
neural real-time slam. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 13293–13302, 2023. 6
[103] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and
David Novotny. Vggsfm: Visual geometry grounded deep
structure from motion. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 21686–21697, 2024. 2
[104] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku
Komura, and Wenping Wang. Neus: Learning neural im-
plicit surfaces by volume rendering for multi-view recon-
struction. arXiv preprint arXiv:2106.10689, 2021. 2
[105] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng,
Kaiyong Zhao, and Xiaowen Chu. Irs: A large natural-
istic indoor robotics stereo dataset to train deep models for
disparity and surface normal estimation. In 2021 IEEE In-
ternational Conference on Multimedia and Expo (ICME),
pages 1–6. IEEE, 2021. 15, 16
[106] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang,
Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking
accurate monocular geometry estimation for open-domain
images with optimal training supervision, 2024. 2
[107] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris
Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d
vision made easy. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition, pages
20697–20709, 2024. 2, 3, 5, 6, 7, 8, 15, 16, 17
[108] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu,
Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and
Sebastian Scherer. Tartanair: A dataset to push the limits of
visual slam. 2020. 5, 16
[109] Yiran Wang, Min Shi, Jiaqi Li, Zihao Huang, Zhiguo Cao,
Jianming Zhang, Ke Xian, and Guosheng Lin. Neural video
depth stabilizer. In Proceedings of the IEEE/CVF Inter-
national Conference on Computer Vision (ICCV), pages
9466–9476, 2023. 16, 17
[110] Philippe Weinzaepfel, Vincent Leroy, Thomas Lucas, Ro-
main Br´egier, Yohann Cabon, Vaibhav Arora, Leonid Ants-
feld, Boris Chidlovskii, Gabriela Csurka, and J´erˆome Re-
vaud. Croco: Self-supervised pre-training for 3d vision tasks
by cross-view completion. Advances in Neural Information
Processing Systems, 35:3502–3516, 2022. 3
[111] Magnus Wrenninge and Jonas Unger. Synscapes: A pho-
torealistic synthetic dataset for street scene parsing. arXiv
preprint arXiv:1810.08705, 2018. 5, 15, 16
[112] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi
Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d:
Create anything in 4d with multi-view video diffusion mod-
els. arXiv preprint arXiv:2411.18613, 2024. 4
[113] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong
Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor
Verbin, Jonathan T Barron, Ben Poole, et al. Reconfusion:
3d reconstruction with diffusion priors. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 21551–21561, 2024. 3
[114] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Liang Pan
Jiawei Ren, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian,
Dahua Lin, and Ziwei Liu. Omniobject3d: Large-vocabulary
3d object dataset for realistic perception, reconstruction and
generation. In IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), 2023. 16
[115] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. Rgbd
objects in the wild: Scaling real-world 3d object learning
from rgb-d videos. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition, pages
22378–22389, 2024. 16
[116] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi
Feng, and Hengshuang Zhao. Depth anything: Unleashing
the power of large-scale unlabeled data. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 10371–10381, 2024. 2
[117] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xi-
aogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth
anything v2. arXiv:2406.09414, 2024. 15, 17
[118] Nan Yang, Lukas von Stumberg, Rui Wang, and Daniel
Cremers. D3vo: Deep depth, deep pose and deep uncer-
tainty for monocular visual odometry. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 1281–1292, 2020. 2
[119] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan.
Mvsnet: Depth inference for unstructured multi-view stereo.
In Proceedings of the European conference on computer
vision (ECCV), pages 767–783, 2018. 2
[120] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren,
Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A large-
scale dataset for generalized multi-view stereo networks.
Computer Vision and Pattern Recognition (CVPR), 2020. 16
[121] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner,
and Angela Dai. Scannet++: A high-fidelity dataset of 3d in-
door scenes. In Proceedings of the IEEE/CVF International
Conference on Computer Vision, pages 12–22, 2023. 5, 16
[122] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa.
pixelnerf: Neural radiance fields from one or few images.
In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition, pages 4578–4587, 2021. 3
[123] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu,
Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu,
Zhangyang Xiong, Tianyou Liang, et al. Mvimgnet: A
large-scale dataset of multi-view images. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 9150–9161, 2023. 15, 16
[124] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and
Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splat-
13

ting. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition, pages 19447–19456,
2024. 2
[125] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jam-
pani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-
Hsuan Yang.
Monst3r: A simple approach for estimat-
ing geometry in the presence of motion. arXiv preprint
arxiv:2410.03825, 2024. 3, 5, 6, 7, 16, 17
[126] Jason Y Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang,
Deva Ramanan, and Shubham Tulsiani. Cameras as rays:
Pose estimation via ray diffusion. In International Confer-
ence on Learning Representations (ICLR), 2024. 4
[127] Xiaoshuai Zhang, Sai Bi, Kalyan Sunkavalli, Hao Su, and
Zexiang Xu. Nerfusion: Fusing radiance fields for large-
scale scene reconstruction. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages 5449–5458, 2022. 2, 3
[128] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Ru-
binstein, Noah Snavely, and William T Freeman. Structure
and motion from casual videos. In European Conference on
Computer Vision, pages 20–37. Springer, 2022. 3, 6, 16, 17
[129] Wang Zhao, Shaohui Liu, Hengkai Guo, Wenping Wang, and
Yong-Jin Liu. Particlesfm: Exploiting dense point trajecto-
ries for localizing moving cameras in the wild. In European
Conference on Computer Vision, pages 523–542. Springer,
2022. 6, 16, 17
[130] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wet-
zstein, and Leonidas J Guibas. Pointodyssey: A large-scale
synthetic dataset for long-term point tracking. In Proceed-
ings of the IEEE/CVF International Conference on Com-
puter Vision, pages 19855–19865, 2023. 15, 16
[131] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe,
and Noah Snavely. Stereo magnification: Learning view
synthesis using multiplane images. ACM Trans. Graph.
(Proc. SIGGRAPH), 37, 2018. 5, 15, 16
[132] Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu,
Hujun Bao, Zhaopeng Cui, Martin R Oswald, and Marc
Pollefeys. Nice-slam: Neural implicit scalable encoding for
slam. In Proceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 12786–12796,
2022. 6
[133] Zihan Zhu, Songyou Peng, Viktor Larsson, Zhaopeng Cui,
Martin R Oswald, Andreas Geiger, and Marc Pollefeys.
Nicer-slam: Neural implicit scene encoding for rgb slam. In
2024 International Conference on 3D Vision (3DV), pages
42–52. IEEE, 2024. 2
14

Appendix
A. Training Datasets
We trained our model on 32 datasets that covers a diverse
range of scene types, including static and dynamic envi-
ronments, as well as indoor, outdoor, and object-centric
scenarios. A complete list of these datasets is provided in
Tab. 6.
The original MapFree [3] and DL3DV [55] datasets do
not include dense depth maps. We performed multi-view
stereo (MVS) reconstruction [80] using the provided cam-
era parameters to generate dense depth maps.
This re-
sults in complete annotations for these datasets for training.
RealEstate10K [131], CoP3D [84], and MVImgNet [123]
also do not provide dense depth maps. For these three
datasets, we only use the provided camera parameters to
supervise the camera prediction. For RealEstate10K, we
only include a subset of 2325 training scenes for training.
EDEN [50], IRS [105], Synscapes [111], SmartPor-
traits [48], and HOI4D [57] are treated as single views. To
train on single-view datasets with a specified context length,
we construct sequences by stacking independent views to
the desired context length, and importantly always reset the
state to s0 after each view. This allows us to jointly train
using both multi-view and single-view data within the same
batch. Although both EDEN [50] and SmartPortraits [48]
provide camera poses, EDEN [50] lacks clear documenta-
tion of camera conventions, and SmartPortraits [48] offers
camera poses that are not synchronized with RGBD frames.
Therefore, we treat both as single-view datasets.
For PointOdyssey [130], we filter scenes with incor-
rect depth annotations (mostly scenes with fogs, like
cab h bench ego2) and scenes with unrealistic motion
and material (like Ani). For BEDLAM [9], we remove
scenes with panorama backgrounds.
B. More Implementation Details
Sequence Sampling Details.
Our training dataset com-
prises a combination of video sequences and unordered
photo collections.
For video sequences, we subsample
frames at intervals randomly selected between 1 and k,
where k is set for each dataset based on its frame rate and
camera motion. Within each sequence, either variable or
fixed intervals are used, each accounting for approximately
half of the samples. For photo collections, we use simi-
lar methods as in DUSt3R [107] and compute the overlap
ratios between images to guide the frame sampling. Addi-
tionally, when the scene from a video is largely static, we
shuffle the frames and treat them as a photo collection to
increase data diversity. When the sequences contain major
dynamic objects (like sequences from BEDLAM [9] and
PointOdyssey [130] datasets), we only treat them as videos
and feed frames into the model in temporal order using a
fixed interval.
When the data is metric scale, frames (excluding the first
frame) in a sequence are randomly masked with a 20% prob-
ability and replaced by their corresponding raymap inputs,
using ground truth intrinsics and poses. Note that raymap
mode is activated only when data are in metric scale, as
our model learns metric-scale 3D scene priors. When the
3D annotation is at an unknown scale, raymap querying is
disabled to avoid scale inconsistency with the scene content
captured in the state.
More Architecture Details. Similar to DUSt3R [107], we
reduce training costs by first training the model on 224×224
image resolution with linear heads, and then increasing
the resolution and setting the longer side of the images to
512 pixels. Specifically, in the first two stages of training,
Headself and Headworld are implemented as linear layers. In
the final two stages, Headself and Headworld are switched
to DPT [72] architecture. Compared to Headself, Headworld
incorporates an additional modulation function, which modu-
lates F ′
t using the pose token z′
t within the Layer Normaliza-
tion layers. This modulation design is inspired by LRM [37]
and aims to integrate pose information to achieve implicit
rigid transformations. Specifically, within Headworld, we
first use two self-attention blocks modulated by the pose
token z′
t to generate the pose-modulated tokens, which is
then fed as input to either the linear or DPT architecture to
generate the final pointmap output ˆ
Xworld
t
. The dimension of
z′
t is 768, and Headpose is a 2-layer MLP whose hidden size
is 768. We apply Rotary Positional Embedding (ROPE) [90]
to the query and key feature before each attention operation.
More Training Details.
In the first stage of training, we
use the following datasets: ARKit, ARKit-HighRes, Scan-
Net, ScanNet++, TartanAir, Waymo, MapFree, Blended-
MVS, HyperSim, MegaDepth, Unreal4K, DL3DV, CO3Dv2,
WildRGBD, and VirtualKITTI2. In the second stage, we
incorporate the rest of datasets. In the final stage (long con-
text training), we exclude single-view datasets (EDEN, IRS,
Synscapes, 3D Ken Burns, SmartPortraits, UrbanSyn, and
HOI4D) and train only on multi-view datasets, as the goal of
the final stage training is to enhance scene-level reasoning
within a sequence. Unlike DUSt3R, which applies color
jittering to each image independently, we perform sequence-
level color jittering by applying the same color jitter across
all frames in a sequence.
C. More Comparisons
Video Depth Estimation.
We expand the video depth com-
parison in the main paper and compare with a wider range
of baseline methods, including single-frame depth tech-
niques (Marigold [44] and Depth-Anything-V2 [117]), video
15

Dataset Name
Scene Type
Metric?
Real?
Dynamic?
Camera only?
Single View?
ARKitScenes [5]
Indoor
Yes
Real
Static
No
No
ARKitScenes-HighRes [5]
Indoor
Yes
Real
Static
No
No
ScanNet [19]
Indoor
Yes
Real
Static
No
No
ScanNet++ [121]
Indoor
Yes
Real
Static
No
No
TartanAir [108]
Mixed
Yes
Synthetic
Dynamic
No
No
Waymo [93]
Outdoor
Yes
Real
Dynamic
No
No
MapFree [3]
Outdoor
Yes
Real
Static
No
No
BlendedMVS [120]
Mixed
No
Synthetic
Static
No
No
HyperSim [75]
Indoor
Yes
Synthetic
Static
No
No
MegaDepth [53]
Outdoor
No
Real
Static
No
No
Unreal4K [99]
Mixed
Yes
Synthetic
Static
No
No
DL3DV [55]
Mixed
No
Real
Static
No
No
CO3Dv2 [74]
Object-Centric
No
Real
Static
No
No
WildRGBD [115]
Object-Centric
Yes
Real
Static
No
No
VirtualKITTI2 [13]
Outdoor
Yes
Synthetic
Dynamic
No
No
Matterport3D [15]
Indoor
Yes
Real
Static
No
No
BEDLAM [9]
Mixed
Yes
Synthetic
Dynamic
No
No
Dynamic Replica [43]
Indoor
Yes
Synthetic
Dynamic
No
No
PointOdyssey [130]
Mixed
Yes
Synthetic
Dynamic
No
No
Spring [60]
Mixed
Yes
Synthetic
Dynamic
No
No
MVS-Synth [40]
Outdoor
Yes
Synthetic
Dynamic
No
No
UASOL [6]
Outdoor
Yes
Real
Static
No
No
OmniObject3D [114]
Object-Centric
Yes
Synthetic
Static
No
No
RealEstate10K [131]
Indoor
No
Real
Static
Yes
No
MVImgNet [123]
Object-Centric
No
Real
Static
Yes
No
CoP3D [84]
Object-Centric
No
Real
Dynamic
Yes
No
EDEN [50]
Outdoor
Yes
Synthetic
Static
No
Yes
IRS [105]
Indoor
Yes
Synthetic
Static
No
Yes
Synscapes [111]
Outdoor
Yes
Synthetic
Dynamic
No
Yes
3D Ken Burns [67]
Mixed
No
Synthetic
Static
No
Yes
SmartPortraits [48]
Indoor
Yes
Real
Dynamic
No
Yes
UrbanSyn [32]
Outdoor
Yes
Synthetic
Dynamic
No
Yes
HOI4D [57]
Indoor
Yes
Real
Dynamic
No
Yes
Table 6. Training Datasets. We provide more details of our training datasets. We classify a dataset as dynamic if annotations exist for
moving objects like humans. If there is only camera parameters (intrinsics and extrinsics) available, we mark them as “camera only”. If the
dataset only contains depth and intrinsics for single views, we mark them as “single view”.
depth approaches (NVDS [109], ChronoDepth [82], and
DepthCrafter [39]), and joint depth-and-pose methods such
as Robust-CVD [47], CasualSAM [128], DUSt3R [107],
MASt3R [51], MonST3R [125], and Spann3R [101]. The
results are shown in Tab. 7.
Camera Pose Estimation Similar to video depth estima-
tion, we include a diverse set of baselines for camera pose
estimation. Learning-based visual odometry methods, such
as DROID-SLAM [96], DPVO [97], and LEAP-VO [17],
require ground truth camera intrinsics as input. Optimization-
based methods, including Particle-SfM [129], Robust-CVD
[47], CasualSAM [128], DUSt3R-GA [107], MASt3R-GA
[51], and MonST3R-GA [125], generally operate more
slowly compared to online methods like Spann3R [101] and
our proposed approach. To assess performance in an online
setting, we also evaluate DUSt3R without global alignment.
The results are presented in Tab. 8.
16

Sintel
BONN
KITTI
Alignment
Method
Optim. Onl. Abs Rel ↓δ<1.25 ↑Abs Rel ↓δ<1.25 ↑Abs Rel ↓δ <1.25 ↑
FPS
Per-sequence
scale & shift
Marigold [44]
✓
0.532
51.5
0.091
93.1
0.149
79.6
<0.1
Depth-Anything-V2 [117]
✓
0.367
55.4
0.106
92.1
0.140
80.4
3.13
NVDS [109]
✓
0.408
48.3
0.167
76.6
0.253
58.8
-
ChronoDepth [82]
✓
0.687
48.6
0.100
91.1
0.167
75.9
1.89
DepthCrafter [39]
✓
0.292
69.7
0.075
97.1
0.110
88.1
0.97
Robust-CVD [47]
✓
0.703
47.8
-
-
-
-
-
CasualSAM [128]
✓
0.387
54.7
0.169
73.7
0.246
62.2
-
DUSt3R-GA [107]
✓
0.531
51.2
0.156
83.1
0.135
81.8
0.76
MASt3R-GA [51]
✓
0.327
59.4
0.167
78.5
0.137
83.6
0.31
MonST3R-GA [125]
✓
0.333
59.0
0.066
96.4
0.157
73.8
0.35
Spann3R [101]
✓
0.508
50.8
0.157
82.1
0.207
73.0
13.55
Ours
✓
0.454
55.7
0.074
94.5
0.106
88.7
16.58
Per-sequence scale
DUSt3R-GA [107]
✓
0.656
45.2
0.155
83.3
0.144
81.3
0.76
MASt3R-GA [51]
✓
0.641
43.9
0.252
70.1
0.183
74.5
0.31
MonST3R-GA [125]
✓
0.378
55.8
0.067
96.3
0.168
74.4
0.35
Spann3R [101]
✓
0.622
42.6
0.144
81.3
0.198
73.7
13.55
Ours
✓
0.421
47.9
0.078
93.7
0.118
88.1
16.58
Metric scale
MASt3R-GA [51]
✓
1.022
14.3
0.272
70.6
0.467
15.2
0.31
Ours
✓
1.029
23.8
0.103
88.5
0.122
85.5
16.58
Table 7. Video Depth Evaluation. We report scale&shift-invariant depth, scale-invariant depth and metric depth accuracy on Sintel, Bonn,
and KITTI datasets. Methods requiring global alignment are marked “GA”, while “Optim.” and “Onl.” indicate optimization-based and
online methods, respectively. We also report the FPS on KITTI dataset using 512× 144 image resolution for all methods, except Spann3R
which only supports 224×224 inputs.
Sintel
TUM-dynamics
ScanNet
Method
Optim. Onl. ATE ↓RPE trans ↓RPE rot ↓ATE ↓RPE trans ↓RPE rot ↓ATE ↓RPE trans ↓RPE rot ↓
DROID-SLAM [96]
✓
0.175
0.084
1.912
-
-
-
-
-
-
DPVO [97]
✓
0.115
0.072
1.975
-
-
-
-
-
-
LEAP-VO [17]
✓
0.089
0.066
1.250
0.068
0.008
1.686
0.070
0.018
0.535
Particle-SfM [129]
✓
0.129
0.031
0.535
-
-
-
0.136
0.023
0.836
Robust-CVD [47]
✓
0.360
0.154
3.443
0.153
0.026
3.528
0.227
0.064
7.374
CasualSAM [128]
✓
0.141
0.035
0.615
0.071
0.010
1.712
0.158
0.034
1.618
DUSt3R-GA [107]
✓
0.417
0.250
5.796
0.083
0.017
3.567
0.081
0.028
0.784
MASt3R-GA [51]
✓
0.185
0.060
1.496
0.038
0.012
0.448
0.078
0.020
0.475
MonST3R-GA [125]
✓
0.111
0.044
0.869
0.098
0.019
0.935
0.077
0.018
0.529
DUSt3R [107]
✓
0.290
0.132
7.869
0.140
0.106
3.286
0.246
0.108
8.210
Spann3R [101]
✓
0.329
0.110
4.471
0.056
0.021
0.591
0.096
0.023
0.661
Ours
✓
0.213
0.066
0.621
0.046
0.015
0.473
0.099
0.022
0.600
Table 8. Evaluation on Camera Pose Estimation on Sintel [12], TUM-dynamic [89], and ScanNet [19] datasets. Note that unlike the the
rest of the methods, the three methods in the first section require ground truth camera intrinsics as input.
17