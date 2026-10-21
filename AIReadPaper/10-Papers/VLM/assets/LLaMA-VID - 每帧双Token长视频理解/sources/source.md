# LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models

> 来源: https://arxiv.org/abs/2311.17043

**作者**: Yanwei Li, Chengyao Wang, Jiaya Jia

## 摘要（arXiv 官方）

In this work, we present a novel method to tackle the token generation challenge in Vision Language Models (VLMs) for video and image understanding, called LLaMA-VID. Current VLMs, while proficient in tasks like image captioning and visual question answering, face computational burdens when processing long videos due to the excessive visual tokens. LLaMA-VID addresses this issue by representing each frame with two distinct tokens, namely context token and content token. The context token encodes the overall image context based on user input, whereas the content token encapsulates visual cues in each frame. This dual-token strategy significantly reduces the overload of long videos while preserving critical information. Generally, LLaMA-VID empowers existing frameworks to support hour-long videos and pushes their upper limit with an extra context token. It is proved to surpass previous methods on most of video- or image-based benchmarks. Code is available  https://github.com/dvlab-research/LLaMA-VID}{https://github.com/dvlab-research/LLaMA-VID

LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models
Yanwei Li1*
Chengyao Wang1∗
Jiaya Jia1,2
CUHK1
SmartMore2
Abstract
In this work, we present a novel method to tackle the
token generation challenge in Vision Language Models
(VLMs) for video and image understanding, called LLaMA-
VID. Current VLMs, while proficient in tasks like image
captioning and visual question answering, face computa-
tional burdens when processing long videos due to the ex-
cessive visual tokens. LLaMA-VID addresses this issue by
representing each frame with two distinct tokens, namely
context token and content token. The context token encodes
the overall image context based on user input, whereas
the content token encapsulates visual cues in each frame.
This dual-token strategy significantly reduces the overload
of long videos while preserving critical information. Gener-
ally, LLaMA-VID empowers existing frameworks to support
hour-long videos and pushes their upper limit with an extra
context token. It is proved to surpass previous methods on
most of video- or image-based benchmarks. Code is avail-
able at https://github.com/dvlab-research/LLaMA-VID.
1. Introduction
Large Language Models (LLMs) [43, 53, 64], through their
capacity to generate contextually accurate responses, have
significantly advanced the field of AI. Drawing from the
strengths of LLMs, Vision Language Models (VLMs) [14,
33, 44] have been developed to extend these capabilities to
visual data, demonstrating their adeptness in tasks like im-
age captioning and visual question answering. However, a
substantial challenge emerges in the context of long video,
where an excessive number of tokens are required to repre-
sent consecutive frames. The computational demands esca-
late with the video length, thereby constraining the practical
application of VLMs for extensive videos.
Recently, several approaches have been proposed to han-
dle videos, moving beyond image-only VLMs. These meth-
ods aim to alleviate the token issue by utilizing repre-
sentative queries [30, 62] or applying temporal compres-
sion [38, 39]. Despite these efforts, the challenge of long
*equal contribution
IDEFICS-9B
InstructBLIP
Qwen-VL-Chat
LLaVA-1.5
VideoLLaMA
VideoChat 
Video-ChatGPT
BT-Adapter
LLaMA-VID
Consistency
VizWiz
GQA
MMB
SQA-Image
SEED
VQA-V2
MME
POPE
ActivityNet
MSVD
MSRVTT
Correctness
Detail
Context
Temporal
2.5
3.0
3.6
2.5
3.0
70%
58%
50%
66%
66%
56%
70%
1522
60%
80%
86%
2.0
2.0
2.8
2.5
2.3
39%
50% 35%
83%
1461
55%
45%
50%
48%
48%
43%
Figure 1. The proposed LLaMA-VID achieves leading perfor-
mance on most of benchmarks with 7B LLMs. The video-based
and image-based benchmarks are noted in blue and purple color,
respectively. Please refer to Tables 1, 2, and 3 for more details.
videos remains unresolved.
The primary obstacle stems
from the excessive number of tokens required for each
video frame. For instance, models like BLIP [14, 29] and
LLaVA [33] require 32 and over 256 tokens respectively
for a single image. A video containing 10K frames would
thus necessitate over 320K tokens, exceeding the capacity
of current VLMs. Furthermore, simple temporal compres-
sion can significantly damage the representation over long-
term intervals. This drawback hampers their performance,
thereby underscoring the need for a robust solution.
In this work, we present LLaMA-VID, a novel approach
to effectively manage the token generation issue in long
videos. Our core idea is to represent each video frame with
two distinct tokens: context token and content token. The
context token is designed to encode the overall context of
the image based on user input, which efficiently condenses
the broader picture into a single token. Simultaneously, the
content token captures finer aspects of each frame. Accord-
ing to computational constraints, the length of content token
can be extended to include more details, e.g., 1 token/frame
1
arXiv:2311.17043v1  [cs.CV]  28 Nov 2023

for video input and beyond 256 token/frame for single im-
age. In this way, the overload of long videos can be signifi-
cantly reduced without sacrificing critical information.
In particular, our method employs a dual-token genera-
tion strategy that is both efficient and effective. For each
frame, we first extract image features using a pre-trained vi-
sion transformer [16], akin to other VLMs [14, 33]. The
key question is how to generate the context-related to-
ken according to user instructions. We provide the solu-
tion by leveraging the cross-modality design [15, 29] for
instruction-guided queries, which carry the interactive in-
tention from users. For context token, these queries interact
with previously generated image features in the designed
attention module, termed as context attention. To generate
content token, the image features are average pooled to for-
mulate tokens that adapt to different settings. For instance,
global pooling is adopted to maintain efficiency for video
input while details are preserved with more tokens for sin-
gle image input. The context and content tokens are subse-
quently projected to the space of LLMs with simple linear
layers for final prediction. Furthermore, to better support
hour-long videos in VLMs, we construct an instruction-
based dataset that contains 9K movie-level conversations
for plot reasoning and detail understanding.
Generally, LLaMA-VID can be distinguished from two
aspects. On one hand, with the dual-token paradigm, each
frame can be efficiently encoded with only two tokens,
which empowers existing LLMs to support long videos. On
the other hand, the context token aggregates the most in-
formative feature of each image, which further extends the
upper limit of VLMs with an extra token.
The overall framework, dubbed LLaMA-VID, can be
easily instantiated with various decoders and LLMs, as
elaborated in Section 3.
Extensive empirical studies are
conducted in Section 4 to reveal the effectiveness of each
component. Remarkably, our model can complete training
within 2 days on a single machine with 8×A100 GPUs, and
it outperforms previous leading methods on most of video-
and image-based benchmarks, as shown in Figure 1.
2. Related Work
In this section, we first review large language models and
delve into recent advances in vision language models.
2.1. Large Language Models
The field of Natural Language Processing (NLP) has wit-
nessed tremendous advancements with the evolution of
LLMs. Transformer [55] marked a pivotal milestone, with
subsequent language models [15, 35, 64] demonstrating re-
markable capabilities.
GPT [6] revolutionized this field
by utilizing generative pre-trained transformers for auto-
regressive prediction, which is proved to be a potent lan-
guage modeling paradigm. Recent groundbreaking works,
such as ChatGPT [43], GPT-4 [44], and LLaMA [53],
have pushed the boundaries even further. Trained on vast
amounts of text data, these models exhibit exceptional ca-
pabilities in complex linguistic tasks. To leverage the po-
tential of pre-trained LLMs, instruction tuning [45, 56] is
a crucial component for high-quality output. This strategy
is widely adopted in open-source models like Alpaca [52]
and Vicuna [13], which improve over LLaMA [53] using
specially designed instruction pairs.
There are also re-
searches [57, 60] that utilize the reasoning ability of LLMs
and invoke pre-defined tools for visual applications. Differ-
ent from them, we collect multi-modality instruction data
that contains text, images, and videos in this work, which is
employed to empower LLMs for long video processing.
2.2. Vision Language Models
The advancements in computer vision and NLP have led
to the emergence of vision-language models (VLMs) that
integrate vision models with language models for cross-
modality understanding [11, 58] and reasoning [19, 27,
37].
Pioneering large-scale VLMs like CLIP [46] and
ALIGN [24] have extended language models to vision-
language tasks. The recent progress has seen an increas-
ing focus on leveraging the power of LLMs.
Notably,
Flamingo [2] and BLIP-2 [29] utilize web-scale image-
text pairs for cross-modality alignment, thereby enhanc-
ing learning performance.
To further exploit the poten-
tial of such pre-trained models, InstructBLIP [14] and
MiniGPT-4 [65] construct high-quality instruction pairs
based on BLIP-2 and achieve superior results. Simultane-
ously, LLaVA [33] employs a simple linear projector with a
few learnable parameters to align the image and text space
of LLaMA. Given the tailored instruction data, this straight-
forward approach demonstrates strong capabilities. To sup-
port video understanding in LLMs, Video-LLaMA [62] and
VideoChat [30] attempt to utilize BLIP-2 for video embed-
ding extraction, while Video-ChatGPT [39] proposes spatial
and temporal pooling for video features. However, given the
substantial number of tokens required for each frame, LLMs
encounter significant challenges when processing extensive
video sequences. It prevents previous work from represent-
ing long video sequences that exceed a duration of one hour
in LLMs. To solve the issue, we propose to efficiently en-
code each frame with only 2 tokens, which supports long
video understanding in existing LLMs.
3. LLaMA-VID
The framework of LLaMA-VID is conceptually simple:
encoder and decoder are adopted to produce visual embed-
ding and text-guided features, respectively; context token
and content token are transformed with the tailored token
generation strategy; instruction tuning is designed to un-
leash the potential of LLMs for image and video.
2

Text Decoder
Visual Encoder
Video/Image
time: &
User: What are
the couple doing?
Context A(en)on
Projector
Projector
Content
Context
……
LLM
User: What are
the couple doing?
Pool
&
& + 1
……
❄
Assistant: They are
hugging in a flying 
gesture on the bow.
Visual Embedding
Text Query
)!
)!
*!
+!
,!
,!"
,!#
❄Frozen in training
Scalable token
Figure 2. The framework of LLaMA-VID. With user directive, LLaMA-VID operates by taking either a single image or video frames
as input, and generates responses from LLM. The process initiates with a visual encoder that transforms input frames into the visual
embedding. Then, the text decoder produces text queries based on the user input. In context attention, the text query aggregates text-related
visual cues from the visual embedding. For efficiency, an option is provided to downsample the visual embedding to various token sizes, or
even to a single token. The text-guided context token and the visually-enriched content token are then formulated using a linear projector
to represent each frame at time t. Finally, the LLM takes the user directive and all visual tokens as input and gives responses.
3.1. Encoder and Decoder
The proposed LLaMA-VID can be utilized to interact with
single image or long videos. For clarity, we assume the in-
put image is captured from a video sequence, as presented
in Figure 2. Given a video frame Vt ∈RH×W ×3 at time
t, a transformer-based visual encoder is first employed to
produce the visual embedding Xt ∈RN×C. Here, N =
H/p × W/p and C indicate the number of image patches
and embedding channels, respectively. The patch size p is
typically set to 14 for ViT-based backbones [16, 17, 46].
Meanwhile, we take the user instruction as input and gen-
erate the text-guided query Qt ∈RM×C with the produced
Xt, where M denotes the number of queries. As depicted
in Figure 2, this cross-modality interaction predominantly
occurs in the text decoder, which can be easily instantiated
with BERT [15] or QFormer [14], as compared in Table 6.
In this way, the text query Qt contains highlighted visual
cues that are most related to the user instruction.
3.2. Token Generation
With the text query Qt and visual embedding Xt, we can
easily generate representative tokens for LLMs. Specifi-
cally, context attention is designed to aggregate text-related
visual features and condense them to a single context to-
ken. As shown in Figure 2, it takes Qt and Xt as input and
formulates the context-related embedding Et ∈R1×C as
Et = Mean(Softmax(Qt × XT
t ) × Xt),
(1)
where the Softmax function and Mean operation are con-
ducted along the N and M dimensions, respectively. Un-
like QFormer [14] that adopts 32 visual queries as LLMs
tokens, we only utilize the text query Qt to aggregate the
Algorithm 1 Pseudo Code for Token Generation.
# B: batch size; C: channel size; n: content shape
# M: query length; N: shape of flatten image pacthes;
# text_q: text query in shape (B, M, C)
# vis_embed: visual embedding in shape (B, N, C)
# Key part 1: calculate context-related embedding
ctx_embed = text_q @ vis_embed.transpose(-1,-2)
ctx_embed = ctx_embed / (vis_embed.shape[-1]**0.5)
ctx_embed = (ctx_embed.softmax(-1)@vis_embed).mean(1)
ctx_embed = self.ctxproj(ctx_embed[:,None])
# Key part 2: calculate visual embedding
cur_shape = int(vis_embed.shape[1]**0.5)
vis_embed = vis_embed.reshape(B, cur_shape, -1, C)
vis_embed = F.avg_pool2d(vis_embed.permute(0,3,1,2),
kernel_size=cur_shape//n, stride=cur_shape//n)
vis_embed = vis_embed.permute(0,2,3,1).flatten(1,2)
vis_embed = self.visproj(vis_embed)
# concat token in shape (B, n+1, C), n in [1,N]
final_token = torch.cat([ctx_embed, vis_embed], dim=1)
F: torch.nn.functional; ctxproj, visproj: predefined linear projectors.
visual features with high-response scores to input instruc-
tions. As a result, the most crucial visual cues related to user
input are efficiently preserved in the condensed embedding
Et. The effectiveness of this context-related token genera-
tion is demonstrated in Table 4 and Figure 6. Subsequently,
a linear projector is utilized to transform the embedding Et
into the context token ET
t ∈R1×C, which aligns with the
language space of LLMs. Meanwhile, we employ an adap-
tive pooling strategy for the visual embedding according
to computational constraints to produce the content token
EV
t ∈Rn×C, where n ∈[1, N]. For instance, we maintain
the original resolution of visual embedding Xt when input
single image, while we downsample Xt to 1 token for long
videos. This approach significantly reduces the overload of
LLMs for each frame, thereby supporting hour-long videos
3

User: <prompt>\n<image-0><subtitle-0>,…,<image-i><subtitle-i>, 
Assistant: <answer>
Stage 3: Long Video Tuning (Op4onal)
9K
6K
User: <prompt>, Assistant: <answer>
User: <image-0>,…,<image-i>, Assistant: <caption>
Stage 1: Modality Alignment
232K
558K
User: <image>, Assistant: <caption>
User: <prompt>\n<image-0>,…,<image-i>, Assistant: <answer>
Stage 2: Instruc4on Tuning
98K
User: <prompt>\n<image>, Assistant: <answer>,……(multi-turn)
40K
User: <prompt>, Assistant: <answer>,……(multi-turn)
625K
Figure 3. Multimodal data distribution and instruction format for
model training in each stage. <image> and <image-i> denote
the token for single image and the i-th video frame, respectively.
effectively. Finally, the generated context token ET
t and the
content token EV
t are concatenated to represent the frame
at time t. Along with frames at other timestamps, the entire
video sequence is translated into the language space in to-
ken format, which is then used to generate responses from
LLMs. The whole process is summarized in Algorithm 1.
3.3. Training Strategy
Training strategy, particularly instruction tuning, has
proven to be crucial in LLMs [13, 52, 53] and VLMs [14,
32, 33]. Considering training efficiency, in this work, we di-
vide the training procedure into three stages, i.e., modality
alignment, instruction tuning, and long video tuning.
Modality Alignment. As shown in Figure 2, each video
frame is projected into the space of LLMs in each forward
pass. Therefore, it is crucial to ensure visual features are
well aligned with the language space. To this end, we con-
struct a compact dataset that contains 790K high-quality
image- and video-caption pairs. As illustrated in Figure 3,
it mainly consists of 558K image-caption pairs from the
LLaVA-filtered CC3M dataset [48] and 232K video-caption
pairs sampled from the WebVid 2.5M dataset [5]. The in-
struction format for each modality is presented in Figure 3.
In this stage, we primarily optimize the context attention
and projectors in Figure 2, while freezing the pre-trained
modules like the visual encoder and text decoder.
Instruction Tuning. To enhance the multi-modality under-
standing of LLMs, we build the instruction pairs from [32]
and [39]. In particular, the constructed dataset mainly in-
volves content from three sources, i.e., 40K text conversa-
Video Frame: Only for illustra8on here, not used to produce instruc8on data. 
Script: Rose runs along the B deck promenade. She is dishevelled, her hair fl
ying. She is crying, her cheeks streaked with tears. But also angry, furious! S
haking with emotions she doesn‘t understand... hatred, self-hatred……
Script for the whole movie
Synopsis: Young Rose, angry and distraught that her mother has apparently a
rranged the marriage, considers committing suicide by jumping from the ster
n; Jack manages to pull her back over the rail after she loses her footing……
Synopsis for the whole movie
User: <prompt> Create 5 questions about the movie plot, including plot und
erstanding, plot description, plot analysis, etc. Create 5 questions about char
acters, including relationship, personality, behavior……
Instruction pairs for movie plot and characters
GPT-4: Question: What ultimately happens to the Heart of the Ocean ne
cklace?. Answer:……
User: <prompt> Create 5 complex questions about plot reasoning rather th
an simply describe the plot. Create 5 complex questions about detail scene an
d activity description……
Instruc4on pairs for movie reasoning and details
Claude-2: Question:Why doesn’t Rose get in the lifeboat with her mother
when she has a chance. Answer: ……
User: <prompt> Create 2 plot summary of this movie. The first one should
be a brief summary written in one paragraph. The second one should be a det
ail summary written in multiple paragraphs.……
Instruc4on pairs for movie summary
GPT-4: Brief Summary:……, Detail Summary:……
Figure 4. An example to construct instruction pairs for the movie
Titanic. Given the movie synopsis and script, we utilize the devel-
oped LLMs like GPT-4 [44] and Claude-2 [3] to generate movie
summaries, plot-related QA pairs, and general reasoning QA pairs.
tions from ShareGPT [1], 625K single- or multi-turn visual
QA pairs from [19, 22, 25, 26, 33, 40–42, 47, 49], and 98K
video QA pairs from [7]. For the instruction, we adopt dif-
ferent formats for text, image, and video input, as illustrated
in Figure 3. And the input prompt <prompt> and answer
<answer> vary with datasets. Please refer to [32] and [39]
for more details. Meanwhile, the image token <image-i>
is randomly inserted at the beginning or end of the user in-
put during our training. In instruction tuning, all the mod-
ules are optimized except the frozen visual encoder.
Long Video Tuning. To further unleash the potential for
hour-long videos, we construct 15K long QA pairs, includ-
ing 9K conversions in movie scenes and 6K data sampled
from LongLoRA [12] for token expanding. Specifically,
we utilize more than 400 long movies and corresponding
scripts in MovieNet [21] to build the training set.
The
key components for instruction generation are visualized
in Figure 4. Generally, the generated dataset includes QA
pairs from three aspects: video summary, movie plot, and
detail reasoning. For video summaries, we collect movie
4

Method
LLM
Res.
MSVD-QA
MSRVTT-QA
ActivityNet-QA
Acc
Score
Acc
Score
Acc
Score
FrozenBiLM [59]
DeBERTa-V2
224
32.2
–
16.8
–
24.7
–
VideoLLaMA [62]
Vicuna-7B
224
51.6
2.5
29.6
1.8
12.4
1.1
LLaMA-Adapter [63]
LLaMA-7B
224
54.9
3.1
43.8
2.7
34.2
2.7
VideoChat [30]
Vicuna-7B
224
56.3
2.8
45.0
2.5
26.5
2.2
Video-ChatGPT [39]
Vicuna-7B
224
64.9
3.3
49.3
2.8
35.2
2.7
BT-Adapter [34]
Vicuna-7B
–
67.5
3.7
57.0
3.2
45.7
3.2
LLaMA-VID
Vicuna-7B
224
69.7
3.7
57.7
3.2
47.4
3.3
LLaMA-VID
Vicuna-13B
224
70.0
3.7
58.9
3.3
47.5
3.3
Table 1. Comparison with leading methods on 4 zero-shot video QA datasets. We report results with 2 tokens for each frame. For fair
comparisons, our model is trained with data of stage 1 and stage 2 without long video tuning in Figure 3. Res indicates image resolution.
synopses to produce brief and detailed summaries for each
movie using developed LLMs like GPT-4 [44]. It brings
about 1K summary-level instruction pairs in total. For plot-
level data, we take the entire movie synopsis as input and
leverage GPT-4 [44] to generate plot-related and character-
related QA pairs. These include plot understanding, de-
scription, analysis, character relationship, personality, and
behavior. In particular, we generate 5 plot-related pairs and
5 character-related pairs for each movie, resulting in 4K
plot-level QA data. As for detail-level data, we feed the long
movie script into Claude-2 [3] and generate 5 plot-related
reasoning pairs and 5 detail-related descriptions for each
movie, which brings 4K pairs in total. With long videos and
the generated pairs, we perform instruction tuning by con-
catenating visual tokens and subtitle tokens for each frame,
as depicted in Figure 3. In this way, LLaMA-VID can well
support 64K tokens with more than 3-hour video as input.
Please refer to supplementary material for more details.
4. Experiments
In this section, we provide the experimental setup and
comparisons with leading methods on several benchmarks.
More details are attached in supplementary material.
4.1. Experimental Setup
Implementation Details. In this work, we instantiate the
model with the pre-trained EVA-G [17] for visual encoder
and QFormer [14] for text decoder by default. During train-
ing, we keep the visual encoder fixed in all stages and freeze
the text decoder, as well as the LLM, in the modality align-
ment stage, except for the BERT module in Table 6 that is
not pre-trained. Following the strategy in [32], we optimize
trainable parameters with the designed data and instructions
in Figure 3, running for 1 epoch in each stage. For video in-
put, we extract frames at a speed of 1 FPS. All models are
trained using 8×NVIDIA A100 GPUs. Additional hyper-
parameters are provided in the supplementary material.
Datasets. In this study, we construct the training set mainly
from [5, 21, 32, 39], as illustrated in Section 3.3. Moreover,
we report results on several video- and image-based bench-
marks. In particular, for video input, we evaluate the zero-
shot performance on the open-ended QA benchmarks like
MSVD [8], MSRVTT [58], ActivityNet [7], and the newly-
proposed generative performance benchmark [39]. As for
image-based evaluation, we conduct experiments on several
widely-adopted benchmarks, including GQA [22], MMB
(MMBench) [36], MME [18], POPE [31], SEED [28],
SQAI (Image-based setting in ScienceQA) [37], VQAT
(TextVQA) [50], VizWiz [20], and VQAv2 (VQA V2) [19].
4.2. Main Results
Results on Video-based Benchmarks. In Table 1, we pro-
vide a comparative evaluation of LLaMA-VID against var-
ious state-of-the-art methods across three zero-shot video
QA benchmarks: MSVD-QA [8], MSRVTT-QA [58], and
ActivityNet-QA [7].
Notably, the results are reported
with only two tokens for each frame.
It is evident that
LLaMA-VID, employing Vicuna-7B and Vicuna-13B as the
LLMs, consistently delivers superior performance across all
datasets. On the MSVD-QA and MSRVTT-QA datasets, it
achieves the accuracy of 69.7% and 57.7% with Vicuna-
7B, surpassing the previous leading approach [34] with ab-
solute gains of 2.2% and 0.7%, respectively. As for the
ActivityNet-QA dataset, LLaMA-VID attains top perfor-
mance in accuracy and the highest score of 3.3.
In Ta-
ble 2, we also carry out experiments on the newly proposed
video-based generative performance benchmark [39]. Our
LLaMA-VID is validated to achieve the best performance
across all the evaluation metrics, surpassing previous ap-
proaches by a large margin.
And we find that the per-
formance can be further improved as the LLMs scale up.
In general, LLaMA-VID is demonstrated to bring robust
performance on all benchmarks, validating its effectiveness
and efficiency for video processing.
5

Method
LLM
Res.
Correctness
Detail
Context
Temporal
Consistency
VideoLLaMA [62]
Vicuna-7B
224
1.96
2.18
2.16
1.82
1.79
LLaMA-Adapter [63]
LLaMA-7B
224
2.03
2.32
2.30
1.98
2.15
VideoChat [30]
Vicuna-7B
224
2.23
2.50
2.53
1.94
2.24
Video-ChatGPT [39]
Vicuna-7B
224
2.40
2.52
2.62
1.98
2.37
BT-Adapter [34]
Vicuna-7B
–
2.68
2.69
3.27
2.34
2.46
LLaMA-VID
Vicuna-7B
224
2.96
3.00
3.53
2.46
2.51
LLaMA-VID
Vicuna-13B
224
3.07
3.05
3.60
2.58
2.63
Table 2. Comparison with leading methods on the video-based generative performance benchmark [39]. We report results with 2 tokens for
each frame. For fair comparisons, our model is trained with data of stage 1 and stage 2 without long video tuning in Figure 3. Res indicates
image resolution. Correctness, Detail, Context, Temporal, and Consistency indicate the evaluation metric of Correctness of Information,
Detail Orientation, Contextual Understanding, Temporal Understanding, and Consistency, respectively.
Method
LLM
Res.
GQA
MMB
MME
POPE
SEED
SQAI
VizWiz VQAv2
InstructBLIP [14]
Vicuna-7B
224
49.2
36.0
–
–
53.4
60.5
34.5
–
IDEFICS-9B [23]
LLaMA-7B
224
38.4
48.2
–
–
–
–
35.5
50.9
Qwen-VL† [4]
Qwen-7B
448
59.3*
38.2
–
–
56.3
67.1
35.2
78.8*
Qwen-VL-Chat† [4]
Qwen-7B
448
57.5*
60.6
1487.5
–
58.2
68.2
38.9
78.2*
LLaVA-1.5 [32]
Vicuna-7B
336
62.0*
64.3
1510.7
85.9
58.6
66.8
50.0
78.5*
LLaMA-VID
Vicuna-7B
336
64.3*
65.1
1521.4
86.0
59.9
68.3
54.2
79.3*
BLIP-2 [29]
Vicuna-13B
224
41.0
–
1293.8
85.3
46.4
61.0
19.6
41.0
InstructBLIP [14]
Vicuna-13B
224
49.5
–
1212.8
78.9
–
63.1
33.4
–
Shikra [9]
Vicuna-13B
224
–
58.8
–
–
–
–
–
77.4*
IDEFICS-80B [23]
LLaMA-65B
224
45.2
54.5
–
–
–
–
36.0
60.0
LLaVA-1.5 [32]
Vicuna-13B
336
63.3*
67.7
1531.3
85.9
61.6
71.6
53.6
80.0*
LLaMA-VID
Vicuna-13B
336
65.0*
66.6
1542.3
86.0
62.3
70.0
54.3
80.0*
Table 3. Comparison with leading methods on 8 benchmarks. Here, we use the same training and instruction finetuning data as that
in LLaVA-1.5. We report results with 1 context token and n content tokens, where n is kept the same with that in LLaVA-1.5, i.e.,
n = (336/14)2 = 576. For fair comparisons, our model is trained without video data of stage 1 and stage 2 in Figure 3. Res indicates
input image resolution. ∗and † denote the train subset is included for training and the data is not publicly available, respectively.
Results on Image-based Benchmarks. As illustrated in
Section 3.2, LLaMA-VID not only efficiently represents
long video, but also expands the upper limit of VLMs with
an additional context token. In Table 3, we perform compar-
isons with leading VLMs on 8 widely-adopted benchmarks.
It should be noted that we maintain the same training data
and image resolution as in LLaVA-1.5 [32] for fair com-
parisons. It is evident that LLaMA-VID outperforms other
leading methods across most of benchmarks with different
LLMs. In particular, with Vicuna-7B as the LLM, LLaMA-
VID attains the best results across all the datasets and sur-
passes LLaVA-1.5 with significant gains in GQA, MME,
and VizWiz, where the improvement reaches up to 2.3%,
10.7, and 4.2%, respectively. With a larger Vicuna-13B as
the LLM, LLaMA-VID also outperforms other methods in
6 benchmarks and achieves top-2 in the other datasets. This
demonstrates the generality of the proposed LLaMA-VID,
which can be scaled up with a stronger foundation model. In
summary, LLaMA-VID is proven to push the upper bound
of VLMs, especially in efficient settings.
Qualitative Results. In Figure 5, we apply LLaMA-VID
to different types of data, including single images, short
videos, and long movies. We represent each image with 577
tokens for single images and 2 tokens for videos. LLaMA-
VID demonstrates various capabilities with different inputs.
Specifically, for single images, it focuses on details and
accurately recognizes the character without any text clues.
Moreover, it can also connect the image content to the plot
of the game in multi-turn conversations.
Given a short
trailer video, LLaMA-VID summarizes the overall plot and
infers the movie name, producer, and the type of movie. As
for a 3-hour movie, the proposed model adeptly describes
the storyline and demonstrates plot-related reasoning and
detailed understanding.
6

User: Please describe this image in detail.
Assistant: The image features a young ma
n in a blue outfit, likely from the game seri
es "The Legend of Zelda," standing on top 
of a large rock formation.
He appears to be looking out over the land
scape, possibly contemplating his next mov
e or enjoying the view. The rock formation 
is surrounded by a beautiful blue sky, with 
clouds scattered throughout the scene……
User: What's the name of this young man 
and game, and where he comes from?
Assistant: The young man in the image is n
amed Link, and he comes from the land of 
Hyrule, which is the setting for the popula
r video game series "The Legend of Zelda."
User: Try to write a short poem on it.
Assistant: A man in blue, On a cliff high a
bove, With a sword in his hand, He looks o
ut to the land.
The sky is blue and clear, The clouds are w
hite and fluffy, The man stands tall and st
rong, A warrior in his own way…...
Single Image
User: Please describe this video in detail.
Assistant: The video starts with a carto
on character standing in front of a castl
e, which is surrounded by a beautiful lan
dscape. The character is holding a bucke
t and appears to be observing the surrou
ndings. The scene then transitions to a t
rain, where the character is seen sitting 
inside the train. The train is surrounded 
by lush greenery and appears to be movin
g through a scenic route……
User: What's the name of the movie desc
ribed in this video?
Assistant: The movie described in this vi
deo is "Zootopia."
User: Who produce the movie?
Assistant: The movie is produced by Wa
lt Disney Pictures.
3-min Video
Zootopia Trailer
User: What's the type of this movie?
Assistant: The movie is a Disney anima
ted film.
3-hour Video
Avatar
User: Summarize the plot of this movie.
Assistant: Avatar follows Jake Sully, a 
disabled former Marine who is recruited 
by the RDA to work on Pandora, a moon r
ich in a valuable mineral called unobtaniu
m. Jake's twin brother, Tommy, died in a 
helmet-cam accident, and Jake is offere
d his place in the Avatar Program, which 
allows humans to link with the native Na'
vi, a humanoid species. Jake is assigned t
o the avatar of his deceased brother……
User: How human work on Pandora?
Assistant: Humans work on Pandora by u
sing avatars, which are grown from huma
n DNA mixed with the DNA of the nativ
e Na‘vi, to interact with the environment 
and the Na‘vi.
User: Why Jake Sully decide to fight agai
nst humanity with the Na'vi?
Assistant: Jake Sully decides to fight a
gainst humanity with the Na'vi after he 
falls in love with the Na'vi and their way 
of life, and after he realizes that the hu
man military is destroying their home an
d culture.
Figure 5. Examples of LLaMA-VID with single image, short video, and hour-long video using Vicuna-7B based models.
4.3. Component-wise Analysis
In this subsection, we conduct ablation studies with input
resolution 224 and 2 tokens for each image by default. Here,
we mainly perform experiments on image-based settings to
investigate the effectiveness of each component.
Generated Token Types. As illustrated in Figure 2, each
image is represented with a context token and a content to-
ken in LLMs. To validate the effectiveness of each part, we
conduct experiments with different types of tokens in Ta-
ble 4. Without the context token, the compressed content to-
ken, which encodes each image with 1 token, cannot adjust
to input instructions, leading to subpar performance. Com-
pared with a single content token, the instruction-guided
context token results in significant gains across all datasets
with only 1 token. With both tokens for each image, the
model achieves the best performance across all benchmarks.
It shows that both instruction cues in the context token and
the image content itself in the content token are important.
Generated Token Numbers. In Table 5, we conduct ex-
periments with different numbers of tokens for further in-
vestigation. With an image size 224×224, we set up exper-
context
content
GQA
POPE
SQAI
VQAT
✗
✓
53.3
80.9
66.1
46.5
✓
✗
54.3
82.4
67.7
48.3
✓
✓
55.5
83.1
68.8
49.0
Table 4. Comparison with different token types. We report results
with 1 context token (if exists) and 1 content token.
iments with n content tokens, where n = (224/14)2 = 256
for uncompressed settings in the first two rows. The results
clearly show that the context token consistently improves
performance across different benchmarks with only 1 extra
token. When we compress the content token to 1/4 with
n = 64, the performance drops about 1% to 2% but in-
creases 1% in SQAI. Considering the extra efficient set-
ting for hour-long videos, we compress the content token
to 1/256 with n = 1 by default. Compared to the original
setting without context token, we can reduce the computa-
tional cost to 1/128 with about 2%-6% performance drop,
which is generally acceptable. The linear increase in perfor-
mance presents significant potential for token compression.
7

context
content
GQA
POPE
SQAI
VQAT
0
256
61.9
85.5
67.5
53.0
1
256
63.0
86.6
67.7
53.8
1
64
60.8
85.1
68.7
52.3
1
16
58.2
83.1
67.4
50.8
1
4
56.2
83.5
68.7
49.1
1
1
55.5
83.1
68.8
49.0
Table 5. Comparison with different token numbers. We report
results with various numbers of context token and content token.
text
GQA
POPE
SQAI
VQAT
–
53.3
80.9
66.1
46.5
BERT
54.1
80.8
67.9
48.1
QFormer
55.5
83.1
68.8
49.0
Table 6. Comparison with different text decoders. We report re-
sults with 1 context token (if exists) and 1 content token.
For instance, we can dynamically compress the content to-
ken to different numbers according to resource budget and
content importance. Interestingly, the model achieves peak
performance in SQAI with only 2 tokens. This could be
attributed to the fact that problems in ScienceQA [37] fo-
cus more on visual-based reasoning rather than image de-
tails. As demonstrated in Tables 1 and 2, with only 2 tokens
for each image, LLaMA-VID still outperforms all previous
work in different video-based benchmarks. This makes it
feasible to enable LLMs for hour-long video processing.
Text Decoder. As depicted in Figure 2, the text decoder
plays an essential role in producing instruction-guided con-
text cues. Here, we further perform comparisons with dif-
ferent text decoders in Table 6. We mainly instantiate the
text decoder with two types of modules, namely BERT [15]
and QFormer [14]. For BERT, we randomly initialize it as
a cross-modality decoder and only retain the first two lay-
ers. As for QFormer, we utilize the pre-trained modules
and fix them for modality alignment. Even with a simple 2-
layer BERT, as shown in Table 6, the generated context to-
ken achieves significant gains in most of benchmarks. This
proves the effectiveness of the paradigm for context token
generation. With a pre-trained text decoder like QFormer,
the model can be further enhanced and attains peak perfor-
mance in all datasets with 2.2% to 2.7% significant gain.
Response in Context Attention. To more vividly explore
the context attention, we visualize the high response areas
with the top 20 scores in Figure 6. Specifically, we draw
the normalized heatmap for the first two queries in Qt be-
fore applying the Softmax function, which is used to formu-
Question: Are the elephants fighting?
Input
Query 1
Query 2
Question: Is this a fishing village?
Input
Query 1
Query 2
Question: What tall animals are displayed in the background?
Input
Query 1
Query 2
Figure 6. High response areas with top scores to input question
in Equation 1. We present the response of first two queries in Qt.
Images are randomly sampled from VQA V2 [19] test-dev set.
late context token in Equation 1. As shown in Figure 6, the
text-guided query Qt effectively focuses on important areas
relevant to the input questions. For example, in the second
row, when inquiring whether the image depicts a fishing vil-
lage, the query Qt focuses more on buildings along the river
and a seagull. These are all typical characteristics to distin-
guish a fishing village in common sense. Other examples
also confirm that the designed context attention successfully
achieves its goal of formulating the context token under in-
struction guidance. More visualizations can be found in the
attached supplementary material.
5. Conclusion
We have introduced LLaMA-VID, a simple yet effective to-
ken generation approach for VLMs. The central concept
behind LLaMA-VID is to represent an image with the con-
text token and the content token. In particular, the context
token is generated according to input instructions, and the
content token is produced based on the image content. De-
pending on the budget, the content token can be compressed
to one token or expressed without compression. It allows
us to represent a single image with preserved details and
efficiently encode each video frame with only two tokens.
Moreover, we have constructed an instruction dataset for
hour-long video understanding. Our experiments on several
video- and image-based benchmarks prove the superiority
of our method. We hope that LLaMA-VID can serve as a
strong benchmark for efficient visual representation.
8

A. Experimental Details
In this section, we delve into the experimental details of the
proposed LLaMA-VID framework. Generally, we adopt a
similar training strategy with that in [32], except that we
freeze the proposed text decoder during pretraining and sub-
sequently optimize it during the finetuning phase. As out-
lined in Table 7, we employ customized training settings for
the distinct stages illustrated in Figure 3 of the main paper.
Specifically, during the modality alignment phase (stage 1),
we keep the text decoder fixed and limit the maximum to-
ken length to 2K. Moving on to instruction tuning (stage 2),
we unfreeze and optimize the text decoder, accommodat-
ing a larger maximum token length of 2K to facilitate video
tuning. If stage 3 is adopted for long video tuning, we once
again freeze the text decoder to conserve memory resources
while increasing the maximum token length to a substantial
64K, catering to the demands of hour-long movie content.
Detailed training settings are further explicated in Table 7.
Settings
Stage 1
Stage 2
Stage 3
Batch size
256
128
8
Learning rate
1e-3
2e-5
2e-5
Learning schedule
Cosine decay
Warmup ratio
0.03
Weight decay
0
Epoch
1
Optimizer
AdamW
DeepSpeed stage
2
Vision encoder
Freeze
Text decoder
Freeze
Open
Freeze
Max token
2048
2048
65536
Table 7. Training settings of LLaMA-VID.
B. Instruction Tuning Details
Context Extension.
To support hour-long video under-
standing, we conduct context extension of the language
model in LLaMA-VID to accommodate inputs of up to
64K tokens. In line with prior research on long context
LLMs [10, 12], we employ position interpolation tech-
niques [10] to scale the rotary position encoding [51]
from 4K to 64K, thereby enabling the processing of ex-
tended sequences. Subsequent to this adaptation, we per-
form supervised fine-tuning using Long-VideoQA, a dataset
specifically designed for long-duration video instruction-
following tasks.
Long-VideoQA Dataset. To improve the ability for un-
derstanding hour-long videos, we develop a specialized
dataset for supervised fine-tuning purposes, named Long-
VideoQA. The Long-VideoQA dataset comprises 15K
question-and-answer (QA) pairs, with 9K of these pairs
derived from movie scenes and the remaining 6K pairs
sourced from LongAlpaca [12]. This dataset is designed to
improve the performance of models on long-duration videos
by providing them with relevant and diverse instruction-
following scenarios. Inspired by previous works on image
and short video instruction-following data collection [30,
33, 39], we leverage advanced language models like GPT-
4 [44] and Claude-2 [3] as the strong teacher, to create
instruction-following data that incorporates video content.
To effectively prompt these language models, we utilized
two types of symbolic representations to encode the video
content:
• Synopses. It offers comprehensive narratives detailing the
plot of a movie. These synopses give a broad overview of
the storyline, encompassing major events and character
developments.
• Scripts. It provides a more granular representation, in-
cluding the storyline, character dialogues, and specific ac-
tions. Scripts convey the content in a format that closely
follows the actual sequence of scenes and dialogues.
By utilizing movies from MovieNet [21] along with their
corresponding synopses and scripts, we were able to cre-
ate a rich dataset.
It serves as a foundation for finetun-
ing models to understand and respond to instructions in the
context of long-form video content. As illustrated in Fig-
ure 7, we prompt GPT-4/Claude2 to generate three types of
instruction-following data:
• Video summary. Utilizing synopses as prompts, we en-
gage GPT-4 to generate two types of summaries for each
movie: one brief summary and one detailed summary.
This approach is designed to equip the model with a com-
prehensive understanding of the video content, provid-
ing both a concise encapsulation and an in-depth expo-
sition of the narrative. The brief summary offers a quick
snapshot of the movie’s overarching plot, while the de-
tailed summary delves into the nuances, including char-
acter arcs, thematic elements, and key plot points.
• Plot-level understanding. Leveraging the synopses as in-
put prompts, we employ GPT-4 to generate both plot-
related and character-related QA pairs. This process is
tailored to yield a deeper insight into the movie’s nar-
rative structure and the intricacies of character develop-
ment. The plot-related QA pairs focus on the events, set-
tings, and conflicts within the movie, enabling the model
to grasp the story progression and the causal relationships
between different events. Simultaneously, the character-
related QA pairs delve into the personalities, motivations,
and relationships of the characters, offering a nuanced
perspective on how each character influences and is in-
fluenced by the unfolding story. In this way, we ensure
that the model acquires a well-rounded comprehension of
both the plot dynamics and the character profiles, which
9

is crucial for accurately understanding and responding to
complex instructions related to hour-long video content.
• Detail description and reasoning. By inputting the ex-
tensive movie scripts into Claude-2, we generate two dis-
tinct sets of data for each film: 5 plot-related reasoning
pairs and 5 detail-related descriptions. This methodol-
ogy serves to enhance the model’s capacity for learning
and understanding fine-grained details. The plot-related
reasoning pairs are crafted to challenge the model ability
to make inferences and understand the logical flow of the
narrative. These pairs require the model to connect differ-
ent plot points, reason about cause and effect, and predict
outcomes based on earlier events in the story. On the other
hand, the detail-related descriptions focus on capturing
the subtleties within the movie, such as specific descrip-
tions of scenes, nuanced character expressions, or particu-
lar dialogue exchanges. These descriptions are crucial for
the model to learn about intricacies that are often pivotal
for a comprehensive understanding of video content.
C. Additional Analyses
We provide more examples for long videos and images, as
well as the visualization for response in context attention.
Examples of Long Videos. In Figure 8, we further demon-
strate the effectiveness of LLaMA-VID by interacting with
a broader range of hour-long movies, including genres like
Romance, Adventure, and Sci-Fi. This diverse selection of
films allows us to evaluate the performance across differ-
ent narrative structures and thematic elements. It demon-
strates a proficient ability to summarize storylines and en-
gage in plot-related reasoning by synthesizing information
from both video frames and subtitles. In particular, we com-
pare with LLaMA 2 [54] and LongLoRA [12] by feeding
the movie name and all subtitles, respectively. As depicted
in Figure 9, the result reveals that LLaMA-VID outperforms
its counterparts in tasks involving character understanding
and plot-related reasoning. The nuanced understanding of
complex narratives and the integration of multi-modal data
underscore its potential for advanced applications in video
analysis and interaction.
Examples of Images. In Figure 10, we present additional
interactions with LLaMA-VID, showcasing the model’s
adeptness in knowledge-based perception and reasoning.
The examples illustrate the model’s capability to discern
and interpret object details within a given environment and
to utilize environmental cues to effectively respond to user
inquiries. For instance, when provided with a description or
an image of the surrounding environment, LLaMA-VID is
able to identify specific objects and their attributes. It can
then leverage this information to answer questions posed
by the user that may relate to object functions, spatial rela-
tionships, or contextual relevance. This performance is in-
dicative of the sophisticated understanding. It is not limited
to mere recognition but extends to a deeper cognitive level
where it can process and integrate environmental informa-
tion to engage in informed and accurate dialogue. Such
demonstrations reinforce the potential of LLaMA-VID as
a tool for complex interaction involving visual data and nat-
ural language processing.
Response in Context Attention. In Figure 11, we extend
the visualization of the attention mechanism by presenting
additional results of the response in context attention. This
visualization is in alignment with that in Figure 6 of the
main paper, where we highlight the areas with the highest
attention scores. Specifically, we focus on the top 20 scores
while retaining the first five queries in Qt. These visualiza-
tions serve to underline the effectiveness of the text-guided
query Qt on areas that are most pertinent to input questions.
It is dynamically adjusted based on the content of input
questions, demonstrating its ability to discern and prioritize
different regions depending on the context. For example,
in the last two rows of Figure 11, where different questions
are posed regarding the same image, LLaMA-VID adjusts
its focus accordingly, directing its attention to specific re-
gions that are relevant to each question. This capability
signifies a sophisticated level of contextual understanding
and adaptability, showcasing the potential of LLaMA-VID
in processing complex visual and textual inputs.
10

Video Frame: Only for illustra>on here, not used to produce instruc>on data.
Script: Rose runs along the B deck promenade. She is dishevelled, her hair fl
ying. She is crying, her cheeks streaked with tears. But also angry, furious! S
haking with emotions she doesn‘t understand... hatred, self-hatred……
Script for the whole movie
Synopsis: Young Rose, angry and distraught that her mother has apparently a
rranged the marriage, considers committing suicide by jumping from the ster
n; Jack manages to pull her back over the rail after she loses her footing……
Synopsis for the whole movie
User: You are an artificial intelligence visual assistant that can understan
d and analyze long videos. You will receive the movie script, which includes ou
tlines of the story, setting, dialogue, and actions of characters in a movie. 
The task is to use the provided movie script, create 5 plausible question abou
t the movie, and provide answer for these questions in detail.
Create 5 complex questions about plot reasoning rather than simply describe 
the plot.
To answer such a question, you first need to fully understand the movie, and 
then explain why a certain plot or scene in the movie happened the way it did 
based on the plot. To make the question challenging, the answer needs to be 
from some plot of the movie and different from the episode mentioned in th
e question. The user needs to fully understand the movie to answer the quest
ion.
Only include questions that have definite answers. Do not ask any question th
at cannot be answered confidently with the information from the provided sc
ript.
When using the information from the script, directly explain the movie conte
nt, and do not mention that the information source is the script. Always ask 
and answer the question as if you are directly watching the movie.
Here is an example for question and answer. Generate question and answer p
airs in this format:
Question: Why doesn't Rose get in the lifeboat with her mother when she ha
s the chance?
Answer: Even though Rose has the opportunity to safely get on Lifeboat with 
her mother Ruth and other women, she decides not to ……
InstrucDon pairs for movie reasoning
Claude-2: Question:…… Answer: ……
User: You are an artificial intelligence visual assistant that can understan
d and analyze long videos. You will receive the synopsis of a movie, which prov
ide a detailed description of the plot of the movie.
The task is to use the provided movie plot synopsis, create the plot summary 
of the given movie.
Create 2 plot summary of the movie.
The first one should be a brief summary written in one paragraph.
The second one should be a detail summary written in multiple paragraphs.
Please write the summary strictly according to the provided synopsis. Do not 
add additional information. Do not modify the storyline of the movie.
Here is an example for plot summary. Write summary in this format:
Brief Summary: Dorothy Gale, a young girl living on a Kansas farm, dreams of 
escape before a tornado whisks her away to the magical land of Oz ……
Detail Summary: Dorothy Gale is an orphaned teenager living with her Aunt E
m and Uncle Henry on a Kansas farm. Her life takes a turn when ……
Instruction pairs for movie summary
GPT-4: Brief Summary:……, Detail Summary:……
User: You are an artificial intelligence visual assistant that can understan
d and analyze long videos. You will receive the synopsis of a movie, which prov
ide a detailed description of the plot of the movie.
The task is to use the provided movie plot synopsis, create some plausible qu
estions about the movie, and provide answer for each questions.
Create 5 questions about the movie plot, including plot understanding, plot de
scription, plot analysis, etc. 
Create 5 questions about characters, including character relationship, charac
ter personality, character behavior, etc. 
Only include questions that have definite answers. Do not ask any question th
at cannot be answered confidently with the information from the provided sy
nopsis.
Understand and memorize the plot of the movie from plot synopsis first, the
n answer the question.
When using the information from the synopsis, directly explain the movie con
tent, and do not mention that the information source is the synopsis. Always 
ask and answer the question as if you are directly watching the movie.
Instruction pairs for plot level understanding
GPT-4: Question:…… Answer:……
User: You are an artificial intelligence visual assistant that can understan
d and analyze long videos. You will receive the movie script, which includes ou
tlines of the story, setting, dialogue, and actions of characters in a movie. 
The task is to use the provided movie script, create some plausible questions 
about the movie, and provide answer for each questions in detail.
Create 5 complex questions about detail scene and activity description.
To answer such questions, you first need to fully understand the movie, and t
hen describe the scene and person activity for a specific scene in detail. To 
make the question challenging, make sure these questions can only be answer
ed correctly using the visual information in the movie, rather than reading lin
es from the script only. The user needs to fully understand the movie to ans
wer the question.
Only include questions that have definite answers. Do not ask any question th
at cannot be answered confidently with the information from the provided sc
ript.
When using the information from the script, directly explain the movie conte
nt, and do not mention that the information source is the script. Always ask 
and answer the question as if you are directly watching the movie.
Here is an example for question and answer. Generate question and answer p
airs in this format:
Question: Describe how Rose returns to the cabin to save Jack after she fou
nd out Jack was innocent in detail.
Answer: After realizing that Cal and Lovejoy framed Jack by putting the dia
mond necklace in his pocket, Rose runs back into the sinking Titanic ……
Instruction pairs for movie details
Claude-2: Question:…… Answer: ……
Figure 7. Details in constructing instruction pairs for the movie Titanic. It is detailed version of that in Figure 4 of the main paper.
11

3-hour Video
Titanic
User: Describe the storyline of this movie.
Assistant: In 1912, a young artist named Jac
k Dawson wins a ticket to the first class of t
he RMS Titanic on a game show. He is accom
panied by his best friend, Fabrizio De Rossi, 
and they both board the ship with their mot
her, Ruth. Jack is immediately attracted to 
Rose DeWitt Bukater, the daughter of a wea
lthy steel magnate. Rose is engaged to Cal H
ockley, but she is not happy with her life and 
is having an affair with Jack. Jack and Rose 
spend the first night on the ship together, a
nd Jack saves Rose from falling off the ship. 
They become close friends, and Jack is invit
ed to dinner with Rose and her family. Rose'
s mother and Cal are not happy about Jack a
nd Rose's relationship, and they try to keep 
them apart. Jack and Rose continue to see e
ach other, and Jack even asks Rose to marry 
him. Rose is attracted to Jack, but she is en
gaged, and she is not ready to give up her lif
e of luxury. Jack and Rose spend the rest of 
the voyage together, and they share a roman
tic moment on the ship's bow. Jack and Rose 
are the last two passengers to board the shi
p when it departs from Southampton. They a
re the last two passengers to leave the ship, 
and they share a passionate kiss before the 
ship strikes the iceberg and sinks.
User: Who is the hero and heroine of this mov
ie?
Assistant: The hero and heroine of this movi
e are Jack Dawson and Rose DeWitt Bukater.
User: How was Jack and Rose meet?
Assistant: Jack and Rose meet each other w
hen Jack saves Rose from falling off the shi
p after she slips on the deck.
User: Why did Rose's mother hope Rose to m
arry Cal Hockley?
Assistant: Rose's mother hoped Rose would 
marry Cal Hockley because he was a wealthy 
steel tycoon who could provide for Rose and 
help her family, who were financially ruined 
after losing their fortune.
User: How does Jack get the ticket to the Tit
anic?
Assistant: Jack won the ticket to the Titani
c in a poker game.
User: Why did Rose board the Titanic?
Assistant: Rose boarded the Titanic to trav
el to America to marry her wealthy fiancé, C
aledon Hockley.
User: What is Cal's engagement gift to Rose?
Assistant: Cal's engagement gift to Rose is 
a rare diamond known as the Heart of the O
cean.
3-hour Video
Interstellar
User: Write a brief summary of the plot of 
this movie.
Assistant: In a post-apocalyptic future, Co
oper, a former astronaut and farmer, is tas
ked with piloting a mission to find a new ho
me for humanity after Earth becomes unin
habitable. He and his daughter Murphy unc
over anomalies that lead them to a secret f
acility housing an advanced space station……
User: What is the function of the wormhole? 
And where is it located?
Assistant: The wormhole is a shortcut thro
ugh space-time that leads to a distant gala
xy, and it is located near Saturn.
User: Who might have placed the wormhole 
near Saturn, and what was their purpose?
Assistant: It is possible that an advanced 
civilization from the future placed the wor
mhole near Saturn as a way to guide human
ity to new planets for survival.
User: How does Cooper transmit data from 
the black hole to Murphy?
Assistant: Cooper transmits data from the 
black hole to Murphy by using a timer watc
h with the data encoded into the second h
and as a Morse code.
User: Why does time run faster on Miller pl
anet than on Earth? And how fast it is.
Assistant: Time runs faster on Miller plane
t due to the proximity of the black hole Ga
rgantua, which has a gravitational pull that 
slows time relative to Earth. Every hour sp
ent on Miller planet is equivalent to 7 years 
on Earth. This is because the closer one is 
to a black hole, the slower time runs due to 
the extreme gravitational forces pulling ti
me into the future……
Assistant: Interstellar is a thought-provok
ing and visually stunning film that explores 
the mysteries of space and time. The story 
follows Cooper, a former astronaut, as he s
earches for a new home for humanity after 
Earth becomes uninhabitable due to blight. 
His journey leads him through a wormhole a
nd into a black hole, where he encounters a 
mysterious force that communicates throu
gh gravity. The film effectively uses the co
ncepts of time dilation and the effects of g
ravity on time to drive the plot forward. Th
e visual effects and the performances of t
he cast, particularly Anne Hathaway and Je
ssica Chastain, are outstanding. The film co
ncludes with a sense of hope and optimism, 
as Cooper finds a new home for humanity, l
eaving behind a legacy for his daughter. Ov
erall, Interstellar is an engaging and though
t-provoking film that explores complex scie
ntific concepts in a compelling way, while al
so telling a poignant and emotional story.
User: Write a review of this movie.
2-hour Video
Forrest Gump
User: Briefly describe storyline of this movie.
Assistant: "Forrest Gump" follows the life of a 
man with a low IQ, played by Tom Hanks, who e
mbarks on a journey of self-discovery and love. 
The film begins with Forrest's childhood in Alab
ama, where he forms a close friendship with Jen
ny, and his life is shaped by his mother's wisdo
m and the lessons he learns from his experience
s. Forrest's life takes unexpected turns, includi
ng his time in the Vietnam War, where he becom
es a hero and is awarded the Medal of Honor. A
fter returning home, he finds success in the shr
imping business and becomes a ping-pong champi
on, eventually meeting President Nixon. His life 
takes another turn when he embarks on a cross-
country running spree, which inspires the nation 
and leads him to reconnect with Jenny, who has 
a son named Forrest Jr. The film concludes with 
Forrest reflecting on his life and the lessons he 
has learned, as he continues to live a simple but 
fulfilling life.
User: Who is Jenny, and what is her significance 
in Forrest's life?
Assistant: Jenny is Forrest's childhood friend w
ho becomes his girlfriend and later his wife. She 
is significant in Forrest's life as she is the perso
n he loves and cares for the most, and their relat
ionship is a source of happiness and comfort.
User: Why Forrest Gump Was Awarded the Med
al of Honor After the Vietnam War?
Assistant: Forrest Gump was awarded the Meda
l of Honor for his actions during the Vietnam W
ar, specifically for his heroism in rescuing his pl
atoon mates from an ambush.
User: Why did Forrest Gump get into shrimping 
after he retired?
Assistant: Forrest Gump got into shrimping aft
er he retired because he had promised his frien
d Bubba that he would become a shrimping boat 
captain, and he wanted to honor Bubba's memor
y by keeping his promise.
User: How many American presidents did Forre
st Gump meet in the movie? What is the reason 
for each meeting?
Assistant: Forrest Gump met three American p
residents: President Kennedy, President Johnso
n, and President Nixon. He met President Kenne
dy at the White House after being named an All
-American football player, President Johnson at 
the White House after being awarded the Meda
l of Honor, and President Nixon at the White H
ouse after being named to the All-American Pin
g-Pong team.
User: Why did Lieutenant Dan Taylor blame For
rest for rescuing him?
Assistant: Lieutenant Dan Taylor blamed Forres
t for rescuing him because he felt that Forrest 
had cheated him out of his destiny to die in batt
le.
Figure 8. More examples of LLaMA-VID with hour-long movies using Vicuna-7B based models.
12

3-hour Video
Titanic
User: How did Jack Dawson and Rose DeWitt 
Bukater meet each other in this movie?
Assistant: Jack Dawson and Rose DeWitt Bu
kater met each other when Rose was about t
o fall off the Titanic, and Jack saved her.
3-hour Video
Interstellar
LLaMA-VID
User: Does Jack Dawson and Rose DeWitt Bu
kater both survive the sinking of the Titanic?
Assistant: No, only Rose survives the sinking. 
Jack is not shown to survive the sinking, but 
he is later seen in a dream that Rose has.
User: How did Jack Dawson and Rose DeWitt Bu
kater meet each other in this movie?
LLaMA 2
Assistant: In the movie Titanic, Jack Dawson a
nd Rose DeWitt Bukater met each other on the 
ship during its ill-fated maiden voyage. Rose, a 
high-society passenger, was being pursued by h
er fiancé, Cal Hockley, but she felt suffocated 
by their relationship and longed for something 
more authentic. One night, Jack, a penniless ar
tist, won passage on the ship in a poker game a
nd was able to enter the first-class section, wh
ere he met Rose on the deck. They struck up a 
conversation and quickly formed a connection, 
despite their different social backgrounds.
User: How did Jack Dawson and Rose DeWitt Bu
kater meet each other in this movie?
LongLoRA
Assistant: Jack Dawson and Rose DeWitt Buka
ter met each other on the RMS Titanic during i
ts maiden voyage in 197. Rose was traveling firs
t class and Jack was traveling third class. They 
were drawn together after Jack saved Rose fr
om drowning in the ship's swimming pool. Their 
chance meeting and subsequent romance was cu
t short when the Titanic sank, killing Jack and s
eparating Rose from her fiancé, Cal Hockley. H
owever, Rose never forgot Jack and the time t
hey spent together, carrying a flame of love fo
r him for the rest of her life.
User: Does Jack Dawson and Rose DeWitt Bukat
er both survive the sinking of the Titanic?
Assistant: No, in the movie Titanic, Jack Daws
on (played by Leonardo DiCaprio) does not survi
ve the sinking of the Titanic. He dies in the icy 
waters of the North Atlantic after helping Ros
e DeWitt Bukater (played by Kate Winslet) esc
ape on a liferaft. Rose, on the other hand, does 
survive and is rescued by the crew of the RMS 
Carpathia.
User: Does Jack Dawson and Rose DeWitt Bukat
er both survive the sinking of the Titanic?
Assistant: According to the story, Jack Dawso
n and Rose DeWitt Bukater both survive the sin
king of the Titanic. After the ship hits the iceb
erg and begins to sink, Jack saves Rose from th
e freezing water by pulling her onto the ship's 
rail. They are then able to escape the sinking s
hip together, with Jack helping Rose onto a life
boat. Though the ship is destroyed, Jack and R
ose both live on through their love.
User: How did Cooper transmit data from the 
black hole to Murphy?
Assistant: Cooper transmitted data from th
e black hole to Murphy by using a watch that 
had been given to Murphy by Cooper.
LLaMA-VID
User: How did Cooper transmit data from the bl
ack hole to Murphy?
LLaMA 2
Assistant: There is no character named Murph
y in the movie Interstellar, and there is no sce
ne where a character named Cooper transmits 
data from a black hole. The movie Interstellar 
is a work of science fiction and does not depict 
any real events or technology. Therefore, it is 
not possible for Cooper to transmit data from 
a black hole to Murphy, as there is no Murphy c
haracter in the movie and no such technology e
xists.
User: How did Cooper transmit data from the bl
ack hole to Murphy?
LongLoRA
Assistant: Cooper realizes that TARS, the rob
ot, is the only one standing between him and bei
ng able to transmit data from the black hole to 
Murphy. TARS has the ability to transmit data 
through radio waves, so Cooper comes up with a 
plan to have TARS transmit the quantum data h
e has collected while inside the black hole. Coop
er programs TARS to transmit the data in Mor
se code, which TARS is able to understand and 
decode. Through this method, Cooper is able to 
successfully transmit the data from the black 
hole to Murphy, who is able to receive and unde
rstand it.
User: Why does time pass slower on Miller's 
planet than on Earth? How slowly does it go?
Assistant: Time passes slower on Miller's pla
net because it is in orbit around a black hole, 
which has a strong gravitational pull. The str
onger the gravitational pull, the slower time 
moves. On Miller's planet, time moves at abo
ut 7 years per hour, compared to Earth's 1 h
our per hour. This is because the black hole'
s gravity is so strong that it distorts space-t
ime, causing time to run more slowly.
User: Why does time pass slower on Miller's pla
net than on Earth? How slowly does it go?
Assistant: In the movie Interstellar, time runs 
slower on Miller planet than on Earth due to th
e planet's strong gravitational pull. The stronge
r the gravitational force, the slower time passe
s, according to Einstein's theory of general rel
ativity. On Miller planet, the gravitational force 
is much stronger than on Earth, causing time to 
slow down significantly. The exact slowing down 
of time on Miller planet is not explicitly stated 
in the movie, but it is implied to be around 1.5 t
o 2 times slower than on Earth. This means tha
t for. This time dilation effect become every h
our that passes on Earth, only around 45 minut
es would pass on Miller planets more pronounce
d as you approach the planet's surface, making 
time appear to slow down even more.
User: Why does time pass slower on Miller's pla
net than on Earth? How slowly does it go?
Assistant: Miller planet orbits a black hole, whi
ch has a huge gravitational pull. This causes tim
e to run slower on Miller planet compared to Ea
rth. Specifically, time runs 7 years slower on M
iller planet for every 1 year spent on Earth. Th
e effect is more pronounced the closer one get
s to the black hole. So time runs significantly sl
ower on Miller planet, especially in the vicinity 
of the black hole.
Figure 9. Comparisons with LLaMA 2 [54] and LongLoRA [12] given the same questions.
13

User: Suppose you are a teacher, please use 
the figure to explain the distinction betwee
n evaporation and evapotranspiration.
A: In the image, there is a diagram illustrat
ing the water cycle, which includes precipita
tion, surface water, ground water, and evap
otranspiration. The diagram shows the proc
ess of evapotranspiration, which is the com
bination of evaporation and transpiration.
Evaporation is the process by which water v
apor is released from the surface of water, 
such as a pond or a lake. This occurs when t
he water surface is exposed to the atmosph
ere, and the temperature difference betwe
en the water and the air is sufficient to cre
ate a vapor pressure difference. The water 
vapor then turns into a gas and rises into th
e air.
Transpiration, on the other hand, is the pro
cess by which water vapor is released from 
the leaves of plants. This occurs when the p
lant's stomata open, allowing water vapor to 
escape into the atmosphere. The water vapo
r then cools and condenses, forming droplet
s of water on the surface of the leaf or the 
surrounding area.
In the image, the diagram shows the proces
s of evapotranspiration, which is the combin
ation of these two processes. It is the proc
ess by which water is released from the sur
face of the water and the leaves of plants, 
contributing to the overall water cycle.
Single Image
User: After seeing the below image, how pe
ople might emotionally feel and react?
Assistant: In the image, a man and a woman 
are embracing each other in front of a stag
e, with a crowd of people in the background. 
The couple appears to be enjoying a special 
moment together, possibly celebrating a sig
nificant event or sharing a heartfelt embra
ce. The presence of the stage suggests tha
t there might be a performance or a celebr
ation taking place. People in the crowd migh
t feel a mix of emotions, such as happiness, 
excitement, or nostalgia, as they witness th
e couple's affectionate gesture. Some may 
also feel a sense of connection or camarade
rie with the couple, as they share a similar 
experience in the same environment. Overal
l, the scene evokes a sense of warmth, love, 
and togetherness among the people present.
Single Image
User: Describe both the image and logo in det
ails.
Assistant: The image features two cans of 
Coca-Cola placed on a surface, which appear
s to be a frozen surface, possibly ice or a c
old surface. The cans are sitting next to ea
ch other, with one can on the left and the o
ther on the right. 
The Coca-Cola logo is prominently displayed 
on both cans, indicating that they are part o
f the same brand. The logo is a well-known a
nd recognizable symbol for the Coca-Cola br
and, featuring a red and white color scheme 
with a white letter "C" and a red letter "O" 
on the front.
User: What's the difference between two bo
ttles?
Single Image
Assistant: The difference between the two 
bottles is that one is a can and the other is 
a bottle. Both bottles are of the same bran
d, Coca-Cola, and are placed next to each ot
her on a frozen surface. However, the can i
s a more common and convenient packaging 
option for the brand, while the bottle is a t
raditional and iconic packaging option that h
as been used for many years. The can is gen
erally lighter and more portable than the bo
ttle, making it a popular choice for on-the-g
o consumption.
User: After seeing the below image, how 
people might emotionally feel and react?
Assistant: In the image, a group of people is 
gathered around a man who is being held by 
police officers. The man appears to be resis
ting the officers, and there is a sense of te
nsion and conflict in the scene. People might 
feel a mix of emotions, such as concern, fea
r, or sympathy for the man being held by th
e police. Some individuals might also feel a s
ense of injustice or outrage, depending on t
heir personal beliefs and experiences. The s
ituation could lead to various reactions, inclu
ding protests, calls for justice, or discussion
s about the use of force by law enforcemen
t. The emotional response to this image woul
d depend on the viewer's perspective and th
eir understanding of the context and circum
stances surrounding the event.
Single Image
User: What is unusual about this image?
Assistant: The unusual aspect of this image
is that a man is standing on the back of a ye
llow taxi cab while holding a clothes iron. Th
is is not a typical sight, as people usually do
not use clothes irons while traveling in a veh
icle. The man's actions suggest that he mig
ht be trying to iron clothes on the go, which
is an unconventional and potentially dangero
us way to handle such a task.
Assistant: The meme features a man standi
ng on the back of a yellow taxi cab while hol
ding a clothes iron. This unusual scene is me
ant to be humorous, as it is not a typical sig
ht to see someone using a clothes iron while
traveling in a vehicle. The meme likely aims
to entertain viewers by presenting an unexp
ected and absurd situation.
User: Can you explain this meme in detai
l?
Single Image
User: Suppose you are a detective, what can 
you infer from the visual clues in the image?
Assistant: As a detective, I can infer that t
he person living in this space has a preferen
ce for a cluttered environment. The room fe
atures a bed, a desk with a computer, a chai
r, and a door with coats hanging on it. The pr
esence of multiple coats hanging on the door 
suggests that the person might have a busy l
ifestyle or frequently goes out. The cluttere
d nature of the room might indicate that the 
person has limited time to organize or tidy u
p the space. Additionally, the presence of a 
computer and a keyboard on the desk sugges
ts that the person might be involved in work, 
study, or other computer-related activities.
Single Image
User: What is wrong with the object in the 
image?
Assistant: The object in the image, which ap
pears to be a piece of fabric, has a hole in it.
Single Image
Figure 10. More examples of LLaMA-VID with single image using Vicuna-7B based models. Images are sampled from [61].
14

Question: Are all of the cats the same color?
Input
Query 1
Query 2
Query 3
Query 4
Query 5
Question: Are all men wearing baseball caps?
Input
Query 1
Query 2
Query 3
Query 4
Query 5
Question: Did they just get married?
Input
Query 1
Query 2
Query 3
Query 4
Query 5
Question: How many kites are on the string?
Input
Query 1
Query 2
Query 3
Query 4
Query 5
Question: Is the bear looking at the camera?
Input
Query 1
Query 2
Query 3
Query 4
Query 5
Question: What are flying in the sky?
Input
Query 1
Query 2
Query 3
Query 4
Query 5
Question: Are they near the shoreline?
Input
Query 1
Query 2
Query 3
Query 4
Query 5
Figure 11. High response areas with top scores to input question in Equation 1 of the main paper. We present the response of first five
queries in Qt. Images are randomly sampled from VQA V2 [19] test-dev set.
15

References
[1] Sharegpt. https://sharegpt.com/, 2023.
[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine
Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch,
Katherine Millican, Malcolm Reynolds, et al. Flamingo: a
visual language model for few-shot learning. In NeurIPS,
2022.
[3] Anthropic.
Claude 2.
https://www.anthropic.
com/index/claude-2, 2023.
[4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan
Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren
Zhou. Qwen-vl: A frontier large vision-language model with
versatile abilities. arXiv:2308.12966, 2023.
[5] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisser-
man. Frozen in time: A joint video and image encoder for
end-to-end retrieval. In ICCV, 2021.
[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Sub-
biah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakan-
tan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Lan-
guage models are few-shot learners. In NeurIPS, 2020.
[7] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem,
and Juan Carlos Niebles. Activitynet: A large-scale video
benchmark for human activity understanding.
In CVPR,
2015.
[8] David Chen and William B Dolan. Collecting highly parallel
data for paraphrase evaluation. In ACL, 2011.
[9] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng
Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s
referential dialogue magic. arXiv:2306.15195, 2023.
[10] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuan-
dong Tian.
Extending context window of large language
models via positional interpolation.
arXiv:2306.15595,
2023.
[11] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedan-
tam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick.
Microsoft coco captions:
Data collection and evaluation
server. arXiv:1504.00325, 2015.
[12] Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian
Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning
of long-context large language models. arXiv:2309.12307,
2023.
[13] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhang-
hao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yong-
hao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P.
Xing.
Vicuna: An open-source chatbot impressing gpt-
4 with 90%* chatgpt quality.
https://lmsys.org/
blog/2023-03-30-vicuna/, 2023.
[14] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat
Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale
Fung, and Steven Hoi.
Instructblip:
Towards general-
purpose vision-language models with instruction tuning.
arXiv:2305.06500, 2023.
[15] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. Bert: Pre-training of deep bidirectional trans-
formers for language understanding.
arXiv:1810.04805,
2018.
[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov,
Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
Mostafa Dehghani, Matthias Minderer, Georg Heigold, Syl-
vain Gelly, et al. An image is worth 16x16 words: Trans-
formers for image recognition at scale. In ICLR, 2021.
[17] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu,
Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue
Cao. Eva: Exploring the limits of masked visual representa-
tion learning at scale. In CVPR, 2023.
[18] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin,
Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui
Yang, Xiawu Zheng, et al. Mme: A comprehensive eval-
uation benchmark for multimodal large language models.
arXiv:2306.13394, 2023.
[19] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Ba-
tra, and Devi Parikh. Making the v in vqa matter: Elevating
the role of image understanding in visual question answer-
ing. In CVPR, 2017.
[20] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi
Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham.
Vizwiz grand challenge: Answering visual questions from
blind people. In CVPR, 2018.
[21] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and
Dahua Lin. Movienet: A holistic dataset for movie under-
standing. In ECCV, 2020.
[22] Drew A Hudson and Christopher D Manning. Gqa: A new
dataset for real-world visual reasoning and compositional
question answering. In CVPR, 2019.
[23] IDEFICS.
Introducing idefics:
An open reproduction
of state-of-the-art visual language model.
https://
huggingface.co/blog/idefics, 2023.
[24] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh,
Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom
Duerig. Scaling up visual and vision-language representation
learning with noisy text supervision. In ICML, 2021.
[25] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and
Tamara Berg.
Referitgame: Referring to objects in pho-
tographs of natural scenes. In EMNLP, 2014.
[26] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson,
Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalan-
tidis, Li-Jia Li, David A Shamma, et al.
Visual genome:
Connecting language and vision using crowdsourced dense
image annotations. IJCV, 2017.
[27] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui
Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmenta-
tion via large language model. arXiv:2308.00692, 2023.
[28] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao
Ge, and Ying Shan. Seed-bench: Benchmarking multimodal
llms with generative comprehension.
arXiv:2307.16125,
2023.
[29] Junnan Li,
Dongxu Li,
Silvio Savarese,
and Steven
Hoi.
Blip-2: Bootstrapping language-image pre-training
with frozen image encoders and large language models.
arXiv:2301.12597, 2023.
[30] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wen-
hai Wang,
Ping Luo,
Yali Wang,
Limin Wang,
and
Yu Qiao.
Videochat: Chat-centric video understanding.
arXiv:2305.06355, 2023.
16

[31] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin
Zhao, and Ji-Rong Wen. Evaluating object hallucination in
large vision-language models. arXiv:2305.10355, 2023.
[32] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae
Lee.
Improved baselines with visual instruction tuning.
arXiv:2310.03744, 2023.
[33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning. In NeruIPS, 2023.
[34] Ruyang Liu, Chen Li, Yixiao Ge, Ying Shan, Thomas H Li,
and Ge Li. One for all: Video conversation is feasible with-
out video instruction tuning. arXiv:2309.15785, 2023.
[35] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar
Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettle-
moyer, and Veselin Stoyanov. Roberta: A robustly optimized
bert pretraining approach. arXiv:1907.11692, 2019.
[36] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang
Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He,
Ziwei Liu, et al. Mmbench: Is your multi-modal model an
all-around player? arXiv:2307.06281, 2023.
[37] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei
Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and
Ashwin Kalyan. Learn to explain: Multimodal reasoning via
thought chains for science question answering. In NeurIPS,
2022.
[38] Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui
Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley:
Video assistant with large language model enhanced ability.
arXiv:2306.07207, 2023.
[39] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and
Fahad Shahbaz Khan.
Video-chatgpt: Towards detailed
video understanding via large vision and language models.
arXiv:2306.05424, 2023.
[40] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana
Camburu, Alan L Yuille, and Kevin Murphy.
Generation
and comprehension of unambiguous object descriptions. In
CVPR, 2016.
[41] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana
Camburu, Alan L Yuille, and Kevin Murphy.
Generation
and comprehension of unambiguous object descriptions. In
CVPR, 2016.
[42] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and
Anirban Chakraborty. Ocr-vqa: Visual question answering
by reading text in images. In ICDAR, 2019.
[43] OpenAI.
Chatgpt.
https://openai.com/blog/
chatgpt/, 2023.
[44] OpenAI. Gpt-4 technical report. arXiv:2303.08774, 2023.
[45] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Car-
roll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini
Agarwal, Katarina Slama, Alex Ray, et al.
Training lan-
guage models to follow instructions with human feedback.
In NeurIPS, 2022.
[46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learn-
ing transferable visual models from natural language super-
vision. In ICML, 2021.
[47] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark,
Kenneth Marino, and Roozbeh Mottaghi.
A-okvqa:
A
benchmark for visual question answering using world knowl-
edge. In ECCV, 2022.
[48] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu
Soricut. Conceptual captions: A cleaned, hypernymed, im-
age alt-text dataset for automatic image captioning. In ACL,
2018.
[49] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and
Amanpreet Singh. Textcaps: a dataset for image captioning
with reading comprehension. In ECCV, 2020.
[50] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang,
Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus
Rohrbach. Towards vqa models that can read. In CVPR,
2019.
[51] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo
Wen, and Yunfeng Liu. Roformer: Enhanced transformer
with rotary position embedding. arXiv:2104.09864, 2021.
[52] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois,
Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B.
Hashimoto.
Stanford alpaca:
An instruction-following
llama model. https://github.com/tatsu-lab/
stanford_alpaca, 2023.
[53] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste
Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aure-
lien Rodriguez, Armand Joulin, Edouard Grave, and Guil-
laume Lample. Llama: Open and efficient foundation lan-
guage models. arXiv:2302.13971, 2023.
[54] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert,
Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov,
Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
Llama 2:
Open foundation and fine-tuned chat models.
arXiv:2307.09288, 2023.
[55] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attention is all you need. In NeurIPS, 2017.
[56] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu,
Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and
Quoc V Le. Finetuned language models are zero-shot learn-
ers. arXiv:2109.01652, 2021.
[57] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang,
Zecheng Tang, and Nan Duan.
Visual chatgpt:
Talk-
ing, drawing and editing with visual foundation models.
arXiv:2303.04671, 2023.
[58] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large
video description dataset for bridging video and language. In
CVPR, 2016.
[59] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and
Cordelia Schmid. Zero-shot video question answering via
frozen bidirectional language models. In NeurIPS, 2022.
[60] Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu
Li, and Ying Shan.
Gpt4tools: Teaching large language
model to use tools via self-instruction. arXiv:2305.18752,
2023.
[61] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang,
Chung-Ching Lin, Zicheng Liu, and Lijuan Wang.
The
17

dawn of lmms: Preliminary explorations with gpt-4v (ision).
arXiv:2309.17421, 2023.
[62] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An
instruction-tuned audio-visual language model for video un-
derstanding. arXiv:2306.02858, 2023.
[63] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu,
Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao.
Llama-adapter:
Efficient fine-tuning of language models
with zero-init attention. arXiv:2303.16199, 2023.
[64] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe,
Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab,
Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained trans-
former language models. arXiv:2205.01068, 2022.
[65] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and
Mohamed Elhoseiny.
Minigpt-4:
Enhancing vision-
language understanding with advanced large language mod-
els. arXiv:2304.10592, 2023.
18