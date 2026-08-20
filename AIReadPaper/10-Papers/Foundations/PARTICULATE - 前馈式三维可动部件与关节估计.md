## PARTICULATE: Feed-Forward 3D Object Articulation

https://arxiv.org/pdf/2512.11798 & https://ruiningli.com/particulate

这是一篇对常见的 3d 资产预测其铰链结构的论文，号称可以在几秒内预测出3d资产的铰链结构。因为其中对铰链结构的设计与编码比较清晰，所以就着重分享**如何在Transformer中整合铰链结构以及铰链参数如何表示？**

Particulate 在一次前向传播中即可预测完整的可动属性，包括可动3D部件分割、运动学结构以及可动运动的参数，耗时仅几秒。 他的主要工作并非直接合成，而是与合成任务正交的**可动性估计任务**。通过输入一个3d资产，能够将资产中的铰链部分的参数估计出来。

### 任务描述

- 给定一个三维网格 $$M=(V,F)$$，其顶点为 V ，面为 F，目标是预测其部分部件的的关节结构 $$A$$
    
- A是一个四元组： $$(P,S,K,M)$$
    
    - $$P∈N$$ 是一个**整数**，表示部件的数量
        
    - $$S:[|F|]→[P]$$ : 表示的是把每一个面 F 分配给某一个部件，值为0或1的NxP 矩阵
        
    - $$K∈[P]×[P]$$ : 称为**运动链**，是一组边的集合，值为0或1的 PxP 矩阵，值为1时证明两个部件相连
        
        - 论文中假设一个3d资产只有一个基节点（父节点），这样就形成了以父节点为根的有向树结构，称为运动树
            
    - M: 被称为**运动约束** ，是由 $$(M_{tp}、M_{pd}、M_{ra}、M_{pr}、M_{rr})$$组成
        
        - 其中每一个 $$M∗$$ 都表示一个部件相对于其父部件 刚性运动 的一个方面
            
        - (下面几个的下标，我把缩写展开，方便理解是什么意思)
            
        - $$M_{type}$$ ：表明父子部件之间运动的类型：**固定fixed、线性滑动pri、单一轴旋转rev、两者都有both**
            
        - $$M_{PrismaticDirections}$$ ：表明部件P的移动方向 → $$S^2$$→ 用一个三维的单位向量表示
            
        - $$M_{RevoluteAxes}$$ ：表明每个部件转动运动的旋转轴 → $$S^2 \times R^3$$→ 一个三维的单位向量和一个三维点表示
            
        - $$M_{PrismaticRanges}$$ ：表明每个部件移动的范围 → $$R^2$$→ 指明值域 $$[−l_{min},l_{max}]$$
            
        - $$M_{RevoluteRanges}$$ ：表明每个部件转动的范围 → $$R^2$$→ 指明值域 $$ [−θ_{min},θ_{max}]$$
            
        - 上面五个M也比较好理解，就是（运动类别、平移、转动、平移范围、转动范围），后面四个两两一组，是否有值根据第一个运动类别去判定
            

### 网络结构

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=ZGI2MDAzOTRjNWE3MzJjYmY4ZDVhNzcxNzUyMDcxOWJfRXVEQnZqeUdPdEJFMU5LbnVtUmlNR2UxaGVjckp1WGpfVG9rZW46VWJiWWJYa05Ub2ZuUWl4YkNCeWNsN2lIbkJmXzE3ODI5ODI1NDU6MTc4Mjk4NjE0NV9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

- 两部分输入：
    

从输入网格采样得到的点云，输入为：1、 $$xyz \in R^{N*3}$$ 2、法向量 $$N \in R^{N*3}$$ 3、 Partifield特征 $$F \in R^{N*d}$$,分别经过MLP和Linear 提升到D维，然后相加得到 **点云部分的输入** $$\{ P_i \}^N_{i=1}$$

> PartField 是一个训练用于匹配 2D 语义部分的 3D 特征场，它捕捉全局形状上下文并提供语义部分先验

因为部件数量未知，初始化一组可学习的查询向量 $$Q \in R^{P_{max} \times D}$$，得到 **查询部分的输入** $$\{ q_j\}_{j=1}^{P_{max}}$$。$$P_{max}$$远大于真实的部件数量，作为超参数设置。

- Transformer块
    

输入经过一个大的Transformer层（PAT），有B个注意力块。得到一系列的token。其中包括：

- 查询的自注意力
    
- 查询到点云的交叉注意力
    
- 点云到查询到交叉注意力
    

- decoder
    

后续接上不同的decoder head，来解码出A的四元组$$(P,K,M)$$

- loss
    
    - 多任务联合损失
        

$$L = L_S + L_K + L_M$$

$$L_M = L_{M_{tp}} + L_{M_{pr}} + 0.1L_{M_{rr}} + L_{M_{pd}} + L_{d_{ra}} + L_{x_{ra}} $$

### 训练

在Partnet-Mobility 和 GRScenes 数据集上训练，共有3800个物体，50个类别

在PM的测试集上进行测试，通过评估泛化交并比（gIoU）、部件感知的Chamfer距离（PC）以及平均交并比（mIoU）。Particulate 在未见过的实例上表现出良好的泛化能力。

![](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=Zjk1NjQ0YjMxMzNiMjZjMjAyNGEzMzY1NzhjYzNlNjdfMlZoMVhJaFJRU1QyS0l5YzlxWEZVc1pldUVjTUwzUXZfVG9rZW46VWhGWWJGcnBBb2RkTGV4NTNmMmNDVFJIbmJiXzE3ODI5ODI1NDU6MTc4Mjk4NjE0NV9WNA&add_watermark=true&scene_type=CCM&add_watermark=true&scene_type=CCM)

  

  

  

  

  

  

  

# 代码demo

  

  

#### URDF

一个 URDF 文件的路径（字符串），该文件描述了一个机器人或可变形物体的结构，包含：

- `link`：刚体部件（通常关联一个或多个 3D 网格文件 `.obj`/`.stl`/`.dae` 等）
    
- `joint`：连接两个 `link` 的关节，类型包括 `revolute`（旋转）、`prismatic`（平移）、`fixed`（固定）、`continuous`（无限旋转）等