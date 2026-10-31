# 来源记录 — DA3

## 官方论文

| 材料 | 说明 |
| --- | --- |
| `sources/paper.pdf` | Depth Anything 3 官方论文 PDF（arXiv 2511.10647） |
| 原始 URL | https://arxiv.org/abs/2511.10647 |
| 获取日期 | 2026-08-26 |
| 用途 | 全文精读、关键数据与图源 |

## 关键图来源

| 图片 | 对应论文图 | 页码 | 用途 |
| --- | --- | --- | --- |
| `figures/fig1_teaser.png` | Figure 1（teaser） | Page 3 | 展示 DA3 从任意数量图像 + 可选位姿恢复视觉空间的能力 |
| `figures/fig2_pipeline.png` | Figure 2（Pipeline） | Page 5 | 展示单 Transformer + input-adaptive 注意力 + Dual-DPT head 的完整管线 |
| `figures/fig6_pointcloud.png` | Figure 6（点云定性对比） | Page 16 | DA3 vs VGGT / π³ 的点云重建质量对比 |

> 三张图均由官方 PDF 按图区坐标完整裁剪，非内嵌碎片。

## 官方代码、commit 与发布历史

- 官方仓库：https://github.com/ByteDance-Seed/Depth-Anything-3
- 本轮源码锁定 commit：`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- commit 时间：2026-07-27T14:03:55+08:00
- commit 标题：`Update backend dashboard rendering`
- 初始公开 commit：`6c58ec93436c903ec72d37ae123951c732e0ca00`（2025-11-14，`init`）
- 取证日期：2026-10-31
- 说明：四个 main-series YAML 自初始公开 commit 加入；笔记中的 API、ray-pose、reference-view 与输出行为以 2026-07-27 固定 commit 为准，不把未来 `main` 变化倒写为论文事实。

## Config 与逐模块源码取证

| 路径 | symbol / 已核验事实 |
|---|---|
| `src/depth_anything_3/configs/da3-small.yaml` | ViT-S；`out_layers=[5,7,9,11]`；`alt/qknorm/rope_start=4`；Dual-DPT `dim_in=768` |
| `src/depth_anything_3/configs/da3-base.yaml` | ViT-B；同层位；Dual-DPT `dim_in=1536` |
| `src/depth_anything_3/configs/da3-large.yaml` | ViT-L；`out_layers=[11,15,19,23]`；start=8；Dual-DPT / camera head `dim_in=2048` |
| `src/depth_anything_3/configs/da3-giant.yaml` | ViT-g；`out_layers=[19,27,33,39]`；start=13；Dual-DPT / camera head `dim_in=3072`；可选 GS head |
| `src/depth_anything_3/api.py` | `DepthAnything3.forward/inference`；输入 `[B,V,3,H,W]`；HF 加载；默认 `process_res=504`；高层 `Prediction` 字段 |
| `src/depth_anything_3/utils/io/input_processor.py` | 保长宽比 resize/crop、统一多视图尺寸、14 整除、ImageNet normalize |
| `src/depth_anything_3/model/da3.py` | 总控数据流；camera condition；Dual-DPT；camera-head/ray-head 选择；默认删除 raw ray；可选 3DGS |
| `src/depth_anything_3/model/dinov2/dinov2.py` | 外部 DINOv2 S/B/L/g patch-14 包装；`img_size=518`；Giant 使用 SwiGLU |
| `src/depth_anything_3/model/dinov2/vision_transformer.py` | CLS+patch；learned position interpolation；local/global token 重排；QK norm；RoPE；`cat_token` 双流拼接 |
| `src/depth_anything_3/model/dualdpt.py` | 四层 `[B,V,P,2D]` reassembly；独立 depth/ray fusion；depth/conf 与 6D ray/conf shape |
| `src/depth_anything_3/model/cam_enc.py` | 已知相机 `[t,q,fov]` 9D→camera token `[B,V,D]` |
| `src/depth_anything_3/model/cam_dec.py` | camera feature `[B,V,2D]`→`t3+q4+fov2` |
| `src/depth_anything_3/model/utils/transform.py` | pose encoding 与 `[B,V,3,4]` extrinsics / `[B,V,3,3]` intrinsics 互转 |
| `src/depth_anything_3/utils/ray_utils.py` | `camray[:,:,:3]` 是 direction；`camray[:,:,3:]` 是 translation/origin；RANSAC homography + QL 分解 |
| `src/depth_anything_3/utils/geometry.py` | `unproject_depth`：pixel→camera→world point map |
| `src/depth_anything_3/utils/export/glb.py` | 实际 exporter 用 $K^{-1}$、depth、w2c inverse 生成世界点；默认无独立 point head |
| `src/depth_anything_3/utils/io/output_processor.py` | batch=1 去维；`depth/conf:[V,H,W]`、`extrinsics:[V,3,4]`、`intrinsics:[V,3,3]` |

## 外部视觉 backbone 边界

- 代码中的 `model/dinov2/` 是保留 Meta 版权头的 vendored DINOv2 家族实现，patch size 14，S/B/L/g 宽度为 384/768/1024/1536。
- DA3 复用 DINOv2 架构与视觉预训练起点，但发布实现从 `alt_start` 起加入 alternating local/global attention、QK norm 与 2D RoPE；“vanilla / plain”不能解释为逐行不改上游 forward。
- main-series `num_register_tokens=0`；序列前缀仅一个 CLS，后段由 camera token 覆写。
- `cat_token=true` 将最新 local stream 与当前 fused stream 拼成 $2D$，因此几何 head 输入是 768/1536/2048/3072，而不是原始 DINOv2 的 $D$。
- 发布 HF checkpoint 已含完整 DA3 backbone + heads，常规 inference 不需要单独下载 DINOv2 artifact。

## 官方权重、revision 与完整性

以下 revision 来自 Hugging Face 官方模型 API；LFS SHA-256 来自 `siblings[].lfs.sha256`，不是只靠文件名推测。

| 模型 | 官方模型页 | HF revision | `model.safetensors` 参数数 | 字节数 | LFS SHA-256 |
|---|---|---|---:|---:|---|
| DA3-SMALL | https://huggingface.co/depth-anything/DA3-SMALL | `e08cab65ca0ec38e7826075418411ab90cab4da3` | 34,299,463 | 137,248,940 | `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf` |
| DA3-BASE | https://huggingface.co/depth-anything/DA3-BASE | `f4a6c9b3c95e41c82048423d3493a81ec3fa810e` | 135,366,599 | 541,518,028 | `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5` |
| DA3-LARGE-1.1 | https://huggingface.co/depth-anything/DA3-LARGE-1.1 | `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c` | 410,941,767 | 1,643,843,860 | `739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64` |
| DA3-GIANT-1.1 | https://huggingface.co/depth-anything/DA3-GIANT-1.1 | `72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19` | 1,355,674,125 | 5,422,814,644 | `1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001` |

官方 `config.json` 在上述 revision 中分别锁定 `model_name=da3-small/base/large/giant`，其网络参数与仓库 YAML 一致。README 明确把 `-1.1` 说明为修复训练 bug 后重训的版本，并把旧 `DA3-LARGE` / `DA3-GIANT` 标为 deprecated。实际部署还应复核**具体模型卡**的许可；不能仅从代码仓库 Apache-2.0 许可证推断所有权重可商用。

## 关键代码事实与更正

1. ray map 的发布代码通道顺序为 `[direction(3), origin/translation(3)]`，不是 `[origin,direction]`。
2. camera condition 覆写 CLS slot；没有额外 append camera token，因此 token 数始终 $1+P$。
3. local attention 输入形状为 `[BV,1+P,D]`，global attention 为 `[B,V(1+P),D]`。
4. 四个读取层的 patch 输出为 `[B,V,P,2D]`；camera feature 为 `[B,V,2D]`。
5. `DualDPT` raw 输出为 `depth/depth_conf:[B,V,H,W]`、`ray:[B,V,H,W,6]`、`ray_conf:[B,V,H,W]`。
6. 默认 camera-head 路径与 `use_ray_pose=true` 路径都会在相机求出后删除 raw ray；公共 `Prediction` 不直接暴露 ray。
7. 公共输出的 extrinsics 是 OpenCV/COLMAP w2c `[V,3,4]`；生成世界点前必须求 inverse 得 c2w。
8. point map 是由 depth+camera 或 depth+ray 解析恢复的 `[B,V,H,W,3]`，不是独立学习 head / 默认返回字段。
9. 单图 $V=1$ 时 global attention 自然退化为同一视图；多图 $V>1$ 时全局序列长度为 $V(1+P)$，显存按其平方主导。

## 未覆盖边界

- 官方开源仓库以推理、评测和导出为主，未提供可逐项复刻论文全部训练数据混合与损失实现的完整训练工程；笔记中的训练损失/teacher 细节仍以论文 PDF 为主。
- 本轮没有再次下载四个大权重文件；完整性值来自官方 HF LFS 元数据。正式离线复现应下载后本地计算 SHA-256，与上表逐字节比对。
- HF revision、代码 commit 和论文版本是三个独立坐标，实验日志必须全部记录。
