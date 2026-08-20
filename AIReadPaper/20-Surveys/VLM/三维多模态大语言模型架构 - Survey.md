# 3d LLM模型架构分类

|架构类型|说明|示例|
|---|---|---|
|Wo 3D encoder|基于视觉VLM：多视角2d图输入，结合3d的特点。<br><br>- 优点：<br>    <br>    - 摆脱对3d高质量资产缺失的限制<br>        <br>    - 由于2dVLM的发展，模型会对3D的语义理解更好<br>        <br>- 缺点：<br>    <br>    - 对三维物体/场景的几何感知受制于2D的|![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=OTM5NDZkMzU5NjU2MDU2OTE1NWM2NzQyM2ZlZTI3OGNfOTB2REhlREo5YXBKR3ZMNWlGYWlnWklpZHYyQWxHMFJfVG9rZW46S1h6emJCMVFUb0hvb2R4a2gySmNQMjR1bm1FXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)|
|Wi 3D encoder|**基于点云、基于mesh**的encoder<br><br>- 优点：<br>    <br>    - 能直接对3d资产编码，可以有更强的几何感知能力<br>        <br>- 缺点：<br>    <br>    - 受制于3d资产的质量<br>        <br>    - 无法充分利用已有的多视角数据集|![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NWVmOTdlMWYxZmI0YTcxYzg2ZTJmNWYwOGViMmFhZjBfUU5SajRKZGFXOFVMVm1BZzc2SkJmQXdjNEUyOWlScXBfVG9rZW46WkFTTmJBbGRJb0xNUnV4eFM3U2NPUzB5bmJjXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MmQzM2FhZjVmMjU4ZTBhNjQyNjZjMTEwZDM0NjUwYTFfNGMxU2h3dWczanBSWDlMcU5NanM2eVRSZ3ZUMkJDVHdfVG9rZW46VjRPRmJDa0Mwb0tpV054QlRtZmM5aTlpbnViXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)|

# Wo 3D encoder

## LLaVA-3D

主要利用深度信息和相机位姿，直接编码到LLM中

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MGQ1Yjc4YmRlZmNlNjQ1ZGI5MDA5Y2I5ZTVmMzYyYThfZUlrZDRKWktnTTIycGlTUGYzdGx0S1c5Wjh5MndLc2NfVG9rZW46SEtIb2JiMGJMb2FaTHh4T3d2M2N6Y2NVbkV3XzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

## 3D LLM

利用多视角几何恢复出来的场景，编码为3DFeature再输入到LLM中，相比于上一个只利用深度编码，对3d的理解会更好一点

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NzA4ZWVjMTVlMzYwYWZiOWRlZjQ4YmQ4OTU0Mjg3NDlfRGZMaE9mMnlTWHp6NUdKRnlmU2RLTXBHY1o1U2VoSmpfVG9rZW46WDdwb2JOV241b0NKaWN4aUJxRWNHVFNwbktiXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

# Wi 3D encoder

## 3D表征形式

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NDI1YjQyMzAzYjU0ZTRmZjRkMzIxNDI1Mzk3MTM2NmNfZkRJSmxpaDA3dzFjTXE3VU9CdUVHaTk0dzYzTllBN2NfVG9rZW46Q2NQdWJVMExzbzBFNzZ4MVgxWWN5cnFGbkhnXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

## 点云模型

|   |   |   |   |
|---|---|---|---|
||关于点云的3DLLM总体设计|点云Encoder设计||
||**PointLLM**|||
||**Point3D LLM**|||
||**3D-LLaVA**|||
||**ULIP**|||
|||PointCLIP、PointMLP、PointBERT、Sonata||

### 点云 3DLLM 架构设计

#### 1、PointLLM ECCV 2024 Best Paper Candidate & TPAMI 2025

理解带**有颜色的对象点云**，并根据人类指令生成上下文合适的响应，展示了其对点云和常识的理解

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NjVmY2QxMjExYjg1YTFmYTRhZDdhNjhiMTRlNTFlOGNfck9LT1RmZXdDQ2VxajRiVld0bnRpTkJ0eU1ZTHpxbGVfVG9rZW46SUlhTWIydUQzb01XRGJ4TnMwUGNFaUtRbmJoXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

架构分为三部分：预训练的点云编码器 $$f_{pe}$$ ，投影器 $$f_{proj}$$ 和预训练的大语言模型（LLM）主干 $$f_{llm}$$

$$f_{pe}$$编码器将点云 $$P ∈ n×d$$ 作为输入【n = 8192 个点和 d = 6】输出点特征序列$$X = (x_1,x_2,…,x_m) ∈ R^{m×c}$$【m=513，c= 384】

- 使用 ULIP-2 的多模态预训练方法，在 Objaverse 大规模 3D 数据集 上训练 **Point-BERT** 模型，当作系统的 点云特征提取器（点编码器）。点编码器输出m = 513 个点特征，每个具有c = 384个维度。
- 投影器包含三个线性层，激活：GeLU，这将点特征映射到具有 c′ = 5120 (7B 模型) 或 c′ = 5120 (13B 模型) 维度的词元
- LLaMA 模型作为LLM 骨干，采用7B 和13B Vicuna 检查点

两阶段训练，第一阶段**特征对齐**，冻结点云编码器和大语言模型，仅训练MLP投影器。第二阶段**指令调优**阶段，冻结点云编码器，联合训练LLM和MLP投影器。

#### 2、Point-3D LLM 2025 Apple

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=N2Y3YWM2OTJmYWE0ZGNkNjNjYTBmMTc1ODlkNjQ5M2Rfb3NYcmhKdmFxTU1uek9tMVZ5T3czeVVTMEpPQUVkOVBfVG9rZW46T0xRMmIzM0s3b3k3NFJ4NHpXRmNjdlZwbldkXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

#### 3、3D-LLaVA CVPR 2025

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NGU5ZGIyMzYzZWI4ZTJmY2MyZTg4ZWM3M2I5OTVmZDlfS3ZVcUk5VjFGNG92QThXN0RXZXNSQ0lMVkhDQVZSUFpfVG9rZW46RDVVYmJCbHF4bzM0ZGp4RlM5bWNHTEh4bnRHXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

将**prompt指令和点云对齐**嵌入LLM

#### 4、统一描述：ULIP CVPR 2023

学习语言、图像和点云的统一表示

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MDllYjgwM2JkODc5OTZjY2E1NTM2ZmNkMzU0ODE5MGJfMkZuVUVLZ2lZRWpobjlVTW9HSGE1UUZpY0JmMFpNQ1lfVG9rZW46VWZlYmJZbndob2tONHl4VDU5cGNBSlJ2bnVjXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

数据：ShapeNet55构建三元数据集 $$T_i : (I_i, S_i, P_i)$$，包含大约52.5K 个CAD模型，每个模型都关联有**文本描述**的语义信息元数据。

- 从原始点云中均匀采样 $$N_p $$个点，编码器采用：PointNet++、PointBERT、PointMLP。实验
- 对单个资产使用虚拟相机渲染多视角图片【渲染30个RGB、30个深度图，每次迭代选择其中一个】
- 64 种文字模板经过编码器后做一个平均 $$h^S_i = Avg(f_S (S_i))$$

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=Y2QwYjNiYzk1NmZiMDRjZjRmZjRkNzI5Mzk2YTJjNjVfSEgybmlmM3RBWGJyOVNKYm1CejIzNDJPMVJqVnhQeVpfVG9rZW46UDhhZWJ3dnBHb0FwSkl4cUdrdWM3ZDNFbmZoXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

三个模态对齐比两个模态对齐效果好

||3DLLM 范式|||
|---|---|---|---|
|**pointLLM**|直接将**点云和QA**编码。嵌入到LLM中|||
|**Point3DLLM**|将**点云和多视图统一编码**，嵌入到LLM中|||
|**3dLLaVA**|将**prompt指令和点云对齐**嵌入LLM|||
|**ULIP**|文字、点云、图片三元组对齐，泛化性很好|||

### Point Cloud Encoder调研

#### 1、基于多视图：PointCLIP CVPR 2022

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NzNhYWUzOTIzNjRjMWFmMTlmMTlhNmYwOGEwNzhmNGJfaWlYNXBVV2VLZzR4eEJMcWQyOXRZaWk1TnJXa3EyZ05fVG9rZW46UFBEZmJiYXRMb3p1QUd4cmRlNWMzRUVSblVjXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

类CLIP流程，需要两个Encoder

1. 点云Encoder处理：

将点云投影到**多视角深度图**上，共 $$M $$ 个视角，并通过在2D 中预训练的CLIP进行3D 识别

【SimpleView：原始点云简单投影到图像平面上，并根据垂直距离设置其像素值】

2. 文本Encoder处理：

将K 类别名称放置到预定义模板的class token 位置**“Point Cloud Depth Map of a [CLASS]”**

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=YTAzYjFkZWViZGM2NGE1YjMxZmQyNWI1MDI3MWJjNTFfMjNVWVBTUnBnYzI1WDBhR0hWMXRjUTJyazhGRk1JbUFfVG9rZW46UXczUWJFUWlrb200eGp4b0plamMzalFybjVFXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

给定CLIP 编码的 M 视角特征，沿通道维度将它们连结为 $$Concate(f_{1∼M}) ∈ R^{1×MC}$$

上述的concat后的特征，经过一个三层的多层感知机（MLP），得到所有视图汇总的全局特征。

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=YWZiMzU3MmY2YjBiN2JlYmQ4NTU1ZjU2OGYwY2FjZWZfM1hLcG4wd2kyRFljVFFockVUS2I1NlB1cFF5emZKZ2xfVG9rZW46R0hEVGJxMm1Nb3owdXl4ZUp0UGNPSkF4blNnXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

然后全局特征再经过MLP生成**视图自适应特征**，残差连接原始输入，得到最终的**Adapted Features**

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NWU2NWRkNmM0OTM0MjgxYTcwODI4MDg0NGM4NGQ4M2ZfWU9JY3lWUVNpUXNnMFJGNmNZWFdMa2xSRHBkOVFrRXZfVG9rZW46R1M1cGJKMG1Nb0dibzh4SFRpcGN6OE5UbmJZXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

#### 2、基于多视图：PointCLIP V2 ICCV 2023

**PointCLIP有下面两个缺点：**

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=YWIxMGZkZGU4YmVmM2EwN2ZjZDFiNjU5YzRlYjFiYjBfeFZ6bTMza3drVDdRQkk0OUE0VnZibDlYbVdScE04ZXVfVG9rZW46RTRVdGJRa0drb2xlQ1J4S2d1WGNveVNNbjZkXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NzRlNjhjYmU4MmNlZjk0OWEwMWE2NTI4YTRhNDBhYThfcDFMWkJjQkJOVjBYcnBLYmN6SXJmRHlWY3BISklHWDJfVG9rZW46S2JuUmJOUUkybzRTcmp4ZXlmVGNYQ1VabkZlXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

V2的改进：

将点云转为体素，然后经过模糊化、高斯过滤等操作，让渲染图不再是点图，而是渐变的**深度图**

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ZDVkN2RiYjUwM2U0NDExNGIwNmJjMjAxNDNlNWY4YmJfOVoyUGU3dnZ2NlVxMThWWnU4cXN5dUNJMU9IR1Z6WjFfVG9rZW46U3M5TWJWY3pCb0RnR0V4eWx4WGNTVGxCbmU4XzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=OWIxYjFmY2YzZTk0ZGY0M2NmYTI1MWE4ZTcwMmUwZDZfbFh6WHlrSWtYaDM4SmxMZk05c1JCeEJkUnhsb3RwdjJfVG9rZW46Q2p0TGJFTkFlb0pRSXB4M2ZsRmNoZjZqbmpiXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

V2的改进：

用GPT3生成丰富3D语意，会使得图文右更好的相似度

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=YmQ4OGQxZjllYWQxMWQ2ZjIyMWIyN2Y4YzE4NTBhZGNfY2Nram5WVk5Pam1Uem41b3lpMXhIdGRaZjh3bXVYUTRfVG9rZW46VW83ZWJ4aGJGb3VYWnR4NmRUNGNYdmtSbk9kXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

|   |   |
|---|---|
|基于**多视图的点云**编码器||
|PointCLIP|将点云渲染成多视图，然后使用**视觉模型**去理解多视图的相关性，从而让模型理解点云的结构|
|PointCLIP V2|

#### 3、基于点 PointNet

推荐一点基础的点云神经网络知识：**最远点采样、pointnet、pointnet++**。在点云模型中比较基础

https://www.bilibili.com/video/BV1oT411x7TH?spm_id_from=333.788.videopod.sections&vd_source=9bf58f120c6e047e7484edd46f1f1340

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MTY1ZGUyNjFhZjg1ZjY2NTJiYTMzOTdiMDIzNDliNWNfQlYzN0pkTWg5QTg0eDhZcmZwMVk5dTAya1dSeVYwbXVfVG9rZW46SmxqcGI1SUh3b0xaMnd4c2c4NGNBd3c1bjZnXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MzUzYjk5ZjcyZTZkODgwYTVmMTRiMWQ4MjEzZDcwOTRfTkZzbEZmSFdCaEtaM3dUcERGNXlmRVVWNFkwV3g2ajZfVG9rZW46V0U3MmJqY2kwb1F3V2Z4T0JPVWNha0cybkViXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NjU5YTdiMTdkMjAzOWY4ZTVmNDE4ZTA3ZWJjYTdhMTJfaDlOMmdtNHROQnJManRWbnBYaVlxRFE0emJod2o4dElfVG9rZW46T1lnbGJPY0NQb0xRRjl4ZjhpMGNiYUNJbkFkXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ZDYxYjAzZDJkOWMyZjZhODBjM2Y1N2VlMjNiYzYwMDhfamdzdmh0QllINDYwU2twT0RZcnBVcVpwTVVpa3V1RmpfVG9rZW46Tk5uSGI2eGc5b3NoaWN4SWdDUGNPN2YwbkFkXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

改进版本，层次化点云特征提取：**PointNet++**

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=OTgyZWNjZGI3N2Q4NWM5YmRiMTQwMWZjMjEwNmRiN2VfdlZNdlJEbWNTVExEMkVtRmVNcGwyMG41VDZoUkVMbWhfVG9rZW46RjIzOWJmZ0xyb0tZT054NThDd2NUMFhpbnplXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

#### 4、基于点：PointMLP ICLR 2022 poster

pointNet++ 后，开始探索好的区域点表示，基于卷积的、基于图和基于注意力的

产生的原因，认为点云网络已经可以学习到局部特征了，接下来需要用更深层次的MLP，所以采用了ResNet的思想

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=YjQwZTNiY2YxYWJjNDFmNGEzMmRkOTliYWFjZDA0ZDhfMlhDWTNwYTdWNU00U3RJNFVwMkc2RU03SE0yQUFHTjJfVG9rZW46QmE4d2JHQUU0bzNwek14ajdZWWNXOVRIbndkXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

使用K=24的KNN选择中心点的邻居。密度不均匀

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=M2VkZjRhNmFiYjMwM2E4NGIwODI5ZGEwNjBhMGJiYjNfd0V5Q3JnRWhMQ2llVmhkOFBhMld3c09PODYxRFNPSU5fVG9rZW46QmdOTWJMOFB4b3g2cnJ4NUlueGNRYnEybmxjXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

#### 5、基于点：Point-BERT CVPR 2022

BERT -> BEiT -> point-BERT

这个就是将BERT中的 **“掩码语言建模MLM”** 拓展到点云中，变为 **“掩码点建模MLP”**

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MDlkNWViZmI2ODA0Zjg2YzY1ZmU5NWJiNmJjMmZjYzBfZWRMN21oZWVxc3V5NTRlMm1VU2UzTlhYbDBxZU9nMWtfVG9rZW46VWpqN2JJdXROb2poeld4VFBTWGNQdTVPbnpiXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

给定输入点 $$p\in R^{N\times3}$$，首先通过最远点采样（FPS）从整体点云 p 中采样 $$g$$个中心点，使用k-近邻法为每个中心点选择 n 个最近邻点，生成局部patch $$\{ p_i\}^g_{i=1}$$，通过减去中心坐标，使得patch变为**无偏**，这些无偏patch就可以类比为NLP或者vit的 patch 了，然后使用**mini-pointnet**得到点云嵌入 --》 **点嵌入序列：** $$\{f_i\}^g_{i=1}$$

对每个块的中心点 $$\{c_i\}$$ 应用一个MLP 来获得其位置嵌入 $$\{pos_i\} $$

Tokenizer

把点云中的每个点的连续特征向量 量化（quantize） 成一个来自“点词表”的 离散 token ID，这样整个点云就变成了一串“离散的点 token 序列”，类似于文本的词序列。$$\{f_i\}^g_{i=1} \to \{z_i\}^g_{i=1}$$

|   |   |   |   |   |
|---|---|---|---|---|
|3d点云编码器|||||
|**PointCLIP**|利用 CLIP 的跨模态对齐||||
|**PointNet/PointNet++**|MLP 直接作用于点/或者局部点||||
|**PointMLP**|||||
|**PointBERT**|||||

## Mesh模型

### Mesh 3DLLM 架构设计

#### 1、LLaMA-Mesh Nvidia & Tsinghua

通过将网格数据转为纯文本表示，暴力方式直接使LLM能够处理和生成3Dmesh

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MjA0NzFjM2I1NzM5Njc0NGEzNWJiMGFkNjczZTZkMGNfU3I3NmZ3dEsyME1QbmNlbWUzNHlTOTBvWEwxckJSMWFfVG9rZW46UElmV2JUTzhDb0ppQ0d4SXlJT2M1dmtobllnXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NTc2ZjI2ODZmNjIzNmM3NTcwN2Y3MTM5OTE4ZjA0ZTBfNXdpVFUybXkyUldFeFpJQ25SY1NhckhMTktVNmxQM3FfVG9rZW46UHdUU2JLSFNZb1dQUVV4R09Fa2NIZ0IxbmxjXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NjQ3MzAwMzY0MGUwY2Y0MGVkZDk2YWJiMTYwNzc4MmJfZW12QjRRZ0lod25aakx2T3VTRVJtdVVPTGxpM0RJVk5fVG9rZW46SmIxTGJCcmdOb0RWRm54dHBRM2NVZHZ3bktkXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ZTg3MjM5NWFhMzYzZmY0N2QyMGRhM2VlZGIzODIzNjlfQlVQbDAyNG14Z3ROUzNIMVFvUGc1Ylp4TmFTU1RPN2NfVG9rZW46WHhIc2J0dDFlb211emJ4cEROWmNjZVBKblFnXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

#### 2、MeshLLM

和LLaMA-Mesh一样，把Mesh当作纯文本序列，

利用3DSAMPart对Mesh进行部件级的语义分割

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MGYyMDhlODcyNDkzYTAzNmFmZjc3YzA5ZmE1NTA2ZDdfZWREY2h1Y1o5VTV1Z2dPZmllZElJVjE5UGFLdVI5cm9fVG9rZW46VjVRRGJpVGdhb2l5djF4WE43UmN0Y2xhbnZiXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=N2RhOTYzMTY4NTdkOTM3ZGJjNDUzZWU5ZWZjYjU3NWZfelpCYW50VEt2Q3NlU3NSSXFqb2JwRTFDSGRnQnBhcWZfVG9rZW46UW1JZmI5V0Zrb21oTkh4YUtsOGNtTVBTbmZRXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

1. 先通过KNN，聚类原始网格，执行：从顶点预测面以及从原始网格组装完整网格
2. 在语义分割后的网格上执行1
3. 在特定任务上训练

#### 3、MeshCoder Neurips Poster 2025

利用Blender python对一些基础模型的代码，利用代码和结构的关系使得LLM理解Mesh的几何关系

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MjAxNGQ1ZTEwYjliZmI2YWM1YjJmMDBkMTRjMGNkYjlfVVE4Q00zbWM2dHlOcnZTVDR5bVM5cm1SYU1XMHlJeFdfVG9rZW46SFpnR2JkV1cwb2U4NEh4QzNyVGNWd2t4bndjXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=YmUzZmI0YzQwNWU5Njg4ZWVlZDUzOTgxYWIzYzFiNGNfRGtidXc4WktLdmZEOERiMnF0RlVxcFhIV1lXd3YzV0tfVG9rZW46QUtSUGI3U1Rnb3dyQ1p4Q2V1amNPV3VYblBoXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=N2U0MmU1MGEwOTBmZTYyMWIwYjE1OGU1YTFlYmViMDFfaUhlTXpDeG1DY0taUHVYd0c2QUhyRThDVFljbndLbDNfVG9rZW46R3FMa2JZZzc0bzAxQTZ4WVVNY2NadEtibktlXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

|   |   |   |   |
|---|---|---|---|
||Mesh 3D LLM 范式|||
|**LLaMA-Mesh**|将Mesh解析为**纯文本**的格式，嵌入到LLM中|||
|**MeshLLM**|||
|**MeshCoder**|将mesh结构为代码类的**格式化语言**，将其嵌入到LLM中|||

### Mesh Encoder 调研

#### 1、基于图卷积的编码：MeshGPT CVPR 2024 Highlight

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=MDIxZTIzN2NmNjhiZDgyNDZiODdiMjFkYTk3NWU0OWJfcndRcW1ydlBaWFFENDhOd0JUUW9yRzQ3QXhDaHlWZ0pfVG9rZW46RFduMGJBeWZib1RaQXR4REtmRGNZN0U1bnJmXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

#### 2、MeshAnything ICLR 2025

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=N2I2YzIxZWM4MmIzMjRlYmI3MTJhMjllYTk4MWRkYWNfRWhRQUI3WklDNlFQWnFQTzVwVXNBSmdQMHVkVDFlSlpfVG9rZW46RWtRMWJuMkJNb3BjeGV4dG85ZmNzMnVrbkliXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

#### 3、基于多视角：InstantMesh github：4.1k

【图生3D Mesh】

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=NTc4ZTBhZTZlNzM4NzdjZWRlNjI1ZDlhYzI5ZDFkZmJfSzk1ajlPN25PT0JTc0pYd0hxbHh0Z25LZWpJdVZHRlRfVG9rZW46WW5GYWJBU29lb0JFV3F4ZXhRSmNhaU02blVkXzE3ODI5ODI1OTg6MTc4Mjk4NjE5OF9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

mesh生成阶段。使用Flexicubes 从三平面中提取mesh

|   |   |   |   |
|---|---|---|---|
|MeshEncoder||||
|**MeshGPT**|基于图网络对mesh编码|||
|**MeshAnything**|基于自回归方式对mesh编码|||
|**InstantMesh**|多视角方式|||
