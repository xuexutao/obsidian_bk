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





# bash Cli

```bash

find ./temp -type f | wc -l  # 查找./temp路径下有多少文件数 file  不包括文件夹


```
