- Cli
这个主要是做一些鉴权，github用gh鉴权：gh，之后可以下载github的private库
``` bash
brew install gh
gh auth login
git clone https://github.com/xuexutao/obsidian_bk.git
```
https://github.com/cli/cli/blob/trunk/docs/install_macos.md#homebrew



- pointCloudUtils
https://github.com/fwilliams/point-cloud-utils
**Point Cloud Utils** 是一个易于使用的 Python 库，用于处理和操作 3D 点云和网格
这个是纯CPU的，相对的还有 `cubvh` 是 GPU-base 的，但是在处理 large number of faces 时容易出现 GPU memory leakage issue，导致速度变慢，一般还是采用这个好



- NKSR 网格重建
https://github.com/nv-tlabs/NKSR

- 3D AffordanceNet
https://andlollipopde.github.io/3D-AffordanceNet/#/introduction

`3D AffordanceNet` 为 **_22949_** 个形状、**_23_** 个形状类别提供定义清晰的视觉曝光率评分图注释，每个类别最多定义 **_5_** 种适用性类型。从赋能类别的角度来看，`3D AfffodanceNet` 包含来自 **_18_** 个赋能类别的 **_56307_** 条赋性注释。`3D AffordanceNet` 数据集和注释为**多类**多**标签** ，意味着这些标签不互斥，每个点都可以对多种赋性标记为正。`3D AffordanceNet` 中的所有形状均来自 PartNet。标签 1 和标签 2 显示了可供性数据集的统计数据。`3D AffordanceNet` 数据集根据形状语义类别分为训练集、验证集和测试集，比例分别为 **_70%_**、**_10%_** 和 20%。



# bash Cli

```bash

find ./temp -type f | wc -l  # 查找./temp路径下有多少文件数 file  不包括文件夹


```
