内容基本来自这本书的第三章：[Programming in Parallel with CUDA (cambridge.org)](https://link.zhihu.com/?target=https%3A//www.cambridge.org/core/books/programming-in-parallel-with-cuda/C43652A69033C25AD6933368CDBE084C)，书是 22 年 5 月出版的，已经算比较新的了。

> 区别于其他 CUDA 书籍的一个特点是，这本书里的 CUDA 示例基于有趣的实际问题，并且还使用现代 C++ 的特性来编写出简单、优雅、紧凑的代码。目前在网上关于 CUDA 的教程或教科书中，大多数示例都太为了写而写，并且基于冗长、过时的 C 语言风格。

## **Warps and Waves**

> The GPU architecture is reflected in the way a CUDA kernel is designed and launched by host software.

为特定的问题设计优质的核函数是本书的全部。首先，你需要决定开多少个线程，当然 `Nthreads` 的选择取决于你的问题。一个线程处理一个元素是不错的选择，关键在于 `Nthreads` 应该足够大，越大越好。初学者可能会想把 `Nthreads` 设置成和 `Ncores`，设备的计算核心数量一致。RTX 2070，有 36 个 SM(`Nsm=36`)，并且每个 SM 可以处理两个 warp(`Nwarp=2`)，可以算出来 `Ncores = Nsm * Nwarp * 32 = 2304`。

这错的离谱辣，NVIDIA GPU 的一大特征就是**进程间的快速切换来掩盖内存访问**。在核函数的执行期间，每个 SM 有大量的常驻线程(resident threads)，对于 RTX 2070，`Nres = 1024`，等于 32 个线程束。而每个 SM 只能处理 2 个束，这说明剩下 30 个束都是被挂起的状态。这就是 NVIDIA 硬件掩盖内存访问的实现。

当我们启动了使用 10^9 个线程的核函数，我们说这些线程是 run in waves 的，`Nwave = Nres * Nsm = 36848`，那么就有 `10^9 / Nwaves = 27127` 个 wave，最后一波不是完整的。理想情况下，每次核函数启动最小的线程数应该是 `Nwave`，如果有更多线程，那最好是 `Nwave` 的倍数。最近几代的 GPU 通常含有 `Nres=2048`。

## **Blocks and Grids**

在 CUDA 里线程块是很重要的概念，他是一组运行在同一个 SM 上的线程。线程块的尺寸应当是 warp size 的整数倍（现在所有的 NVIDIA GPUs 这个值都是 32），最多不超过硬件支持的最大值 1024。同一个线程块里的核函数可以互相通信；不同线程块里的做不到，系统也无法同步不同的线程块。但即使在同一个 SM 的线程块也不能互相通信。

启动核函数时，我们要显式指定线程块的尺寸，以及线程块的数量。CUDA 的文档把这个称作： launching a grid of thread blocks，线程块的数量就是 grid size。因此，总线程数 `Nthreads = threads * blocks`，如果你想要很多的 `N` 个线程，那么 `blocks` 通常是很大的数字，使得 `Nthreads >= N`，这也是上面的例子里，`blocks` 是向上取整的原因（别忘记做边界检查）。

CUDA 文档里并没有说太多和 wave 有关的事情，但有暗示线程是以完整的 wave 被分配到 SM 上执行的，这反过来说明 `blocks` 最好是 SM 数量的整数倍。

## **Occupancy**

占用率(occupancy)被定义为 SM 上实际存在的线程与最大值 `Nres` 的比率，通常用百分比来表示。百分百的占用率表示完整的 waves 运行在 SMs 上。如果我们的线程块大小为256，那么只有当四个线程块在 SM 上驻留时（如果`Nres=1024`），才能达到全占用，这将减少每个线程块的可用资源。

占用率不足并不一定对性能造成影响（特别是核函数计算瓶颈而非内存瓶颈时），如果你的核函数需要大量的共享内存，那占用率确实应该较低。在这些情况下可能需要实验，比如使用全局内存代替共享内存，并依赖 L1 缓存来提高速度。

核函数里的代码可以使用下表所示的内置变量（只显示了 1D 的），来确定线程在其线程块中以及在整个网格中的秩次。

## **Warps and Cooperative Groups**

32 个线程组成的 warp 运行在同一个 SM 上。早期的 GPU，一个[线程束]共用一个程序计数器，还有一个 32-bit 的掩码来指示哪些线程对于当前的指令应该是激活的。这意味着，同一个 warp 里的所有线程是严格的步调一致(lock step)，即**隐式同步**的。由于这种设计，显式地调用 `__syncthreads()` 只对同步不同线程束管用。把不该有的 `__syncthreads()` 干掉是有意义的，同步所有线程的代价非常昂贵。**CUDA 9.0 对于 CC >= 7.0 的机器**，打破了这一规则，每个线程有了自己单独的程序计数器，这意味着同一个束里的线程不一定是同步的了。因此，老旧代码的运行结果可能不再正确！

CUDA 9.0 还提供了[协作组]（cooperative groups），它是一个强有力的 **warp 内同步方式**，并且泛化了这一思路到其他尺寸的线程组。协作组的引入，**使得线程束和线程组可以被当作成 C++ 里的对象使用，程序员可以编写更清晰的代码**。

下面的例子是 `reduce4` 的改进版。使用 `__syncwarp()` 来同步线程束内部。

```cpp
template <int blockSize> 
__global__ void reduce5(r_Ptr<int> sums,cr_Ptr<int> data, int n) {
    // ...
    if (id < 32) {
        // __syncwarps() required for devices of CC >= 7.0
        // __syncwarps() 对 CC < 7.0 的机器不会执行任何命令, 因为那些机器线程束内本身就是同步的.
        s[id] += s[id + 32]; __syncwarp();
        if(id < 16) s[id] += s[id + 16]; __syncwarp();
        if(id < 8) s[id] += s[id + 8]; __syncwarp();
        if(id < 4) s[id] += s[id + 4]; __syncwarp();
        if(id < 2) s[id] += s[id + 2]; __syncwarp();
        if(id < 1) s[id] += s[id + 1]; __syncwarp();
        if(id == 0) sums[blockIdx.x] = s[0]; // store block sum
    }
}
```

对于早期的教程，上面 `if (id < 32)` 里面的代码可能没有 `if` 和 `__syncwarp()`，这是对的，因为确实不需要同步，并且不提前退出也不会增加程序执行时间。所以，反正结果是对的（`s[1-31]` 会被污染） ，移除 `if` 还能更快。

但对于新的机器，`__syncwarp` 就是必须的了。还有一点是说 `if` 也是必须的，这是为了避免 read after write 错误：比如 `s[id] += s[id+8]`，如果在 `if(id < 8)` 里面，`s[0–7]` 会加上 `s[8–15]`；如果把 `if` 去掉了，那么可能有若干个 `s[8-15]` 里的元素还去做加上 `s[16-23]` 的操作了，再去加 `s[0-7]` 就导致了错误。

## **CUDA Objects in Cooperative Groups**

使用协作组需要导入头文件，第二行是可选的：

```cpp
#include “cooperative_groups.h”
namespace cg = cooperative_groups;
​
auto grid = cg::this_grid(); 
auto block = cg::this_thread_block(); 
auto warp = cg::tiled_partition<32>(block); 
​
// If you don’t like auto, the explicit types of these objects are:
//      cg::grid_group              grid 
//      cg::thread_block            block 
//      cg::thread_block_tile<32>   warp
```

`this_grid()` 和 `this_thread_block()` 都是内置的对象，**但对于线程束，需要显式的指定数字 32**。这些对象都可以作为参数传递给设备函数，他们没有默认构造函数，`grid` 和 `block` 只能以示例的方式构造。这些变量可以通过复制传递。在核函数内部，这些对象其实都是轻量级的 handle 指针。

`block` 和 `grid` 对象包装了这次核函数启动的属性，并且提供了另一种访问诸如 `threadIdx` 变量的方式。

瓦片(Tiled partitions)更复杂一些，上面的示例**将先前定义的线程块 `block` 分成了 32 个连续线程为一组的子线程块**，表示每个线程束。这是瓦片最常见的用处。瓦片可以有任意的尺寸，只要是 32 以下 2 的整数幂，并且可以基于线程块或者子线程块进行分区：

```cpp
auto warp = cg::tiled_partition<32>(block);     // 标准的分区: 线程块分成线程束(含有 32 个线程)
auto warp8 = cg::tiled_partition<8>(block);     // 把线程块分成每个含有 8 线程的子线程块
auto warp4A = cg::tiled_partition<4>(warp);     // 把 warp 分成 8 个 4 线程的子束
auto warp4B = cg::tiled_partition<4>(warp8);    // 把 warp8 分成 2 个 4 线程的子束
// 对于 warp4A 和 warp4B, meta_group_size() 和 meta_group_rank() 的值不同.
```

如果你打算使用 `sub-warps`，请记住**协作组只是提供了软件的支持**，GPU 硬件还是以 32 个线程为一束来执行的。因此，如果同一个线程束里两个子束里的代码发生了分歧，对性能仍然是有影响的。

![](https://pic2.zhimg.com/v2-6cddf1d0261f84047bcb3500f993b1f1_1440w.jpg)

上表展示了一些协作组的成员函数。CUDA 11.1 还加入了多束(multiwarp)的支持，比如 `cg::tiled_partition<64>(block)`，但瓦片的大小必须是束尺寸(32)的整数倍。他会隐式的使用共享内存用于束间的通信。

注意 `sync, size, thread_rank` 函数所有协作组都有。`grid.sync()` 方法对于核函数有一些限制，将会在后续提到，其他网格的方法都没有限制；`grid.is_valid()` 方法就是用来测试 `grid.sync` 能不能用的。`block.thread_rank` 返回的秩就是之前的公式，三维换一维的。`grid.thread_rank` 在前面的基础上，加上 `block_rank * block_size`。

```cpp
#include "cooperative_groups.h"
namespace cg = cooperative_groups;
​
__global__ void coop3D(int nx,int ny,int nz,int id) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    int x = block.group_index().x * block.group_dim().x+ block.thread_index().x;
    int y = block.group_index().y * block.group_dim().y+ block.thread_index().y;
    int z = block.group_index().z * block.group_dim().z+ block.thread_index().z;
    if(x >=nx || y >=ny || z >=nz) return; // in range?
    int array_size = nx*ny*nz;
    // threads in one block
    int block_size = block.size();
    // 没有网格里线程块数量的成员函数, 通过 总线程数/线程块的线程束 计算
    int grid_size = grid.size() / block.size();
    // threads in whole grid
    int total_threads = grid.size();
    // 注释里是之前的写法
    int thread_rank_in_block = block.thread_rank();
    // int thread_rank_in_block =(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
    int block_rank_in_grid = grid.thread_rank() / block.size();
    // int block_rank_in_grid = (blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
    int thread_rank_in_grid = grid.thread_rank();
    // int thread_rank_in_grid = block_rank_in_grid*block_size+thread_rank_in_block;
    // ...
}
```

上面的示例用成员函数替换了复杂的计算公式，减少出错的概率。但协作组能做的还有更多！

```cpp
template <int T> 
__device__ void show_tile(const char *tag, cg::thread_block_tile<T> p) {
    int rank = p.thread_rank(); // thread rank in tile
    int size = p.size(); // number of threads in tile
    int mrank = p.meta_group_rank(); // rank of tile in parent
    int msize = p.meta_group_size(); // the number of tiles in the parent partition
    // total number of threads in the partition = size*msize
    printf("%s rank in tile %2d size %2d rank %3d num %3d net size %d\n", 
           tag, rank, size, mrank, msize, msize*size);
}
​
__global__ void cgwarp(int id) {
    auto grid = cg::this_grid(); // standard cg
    auto block = cg::this_thread_block(); // definitions
    
    auto warp32 = cg::tiled_partition<32>(block); // 32 thread warps
    auto warp16 = cg::tiled_partition<16>(block); // 16 thread tiles
    auto warp8 = cg::tiled_partition<8>(block); // 8 thread tiles
    
    auto tile8 = cg::tiled_partition<8>(warp32); // 8 thread sub-warps
    auto tile4 = cg::tiled_partition<4>(tile8); // 4 thread sub-sub warps
​
    if(grid.thread_rank() == id) {
        printf("warps and subwarps for thread %d:\n",id);
        show_tile<32>("warp32",warp32);
        show_tile<16>("warp16",warp16);
        show_tile< 8>("warp8 ",warp8);
        show_tile< 8>("tile8 ",tile8);
        show_tile< 4>("tile4 ",tile4);
    }
}
​
int main(int argc, char *argv[]) {
    int id = (argc >1) ? atoi(argv[1]) : 12345;
    int blocks = (argc >2) ? atoi(argv[2]) : 28800;
    int threads = (argc >3) ? atoi(argv[3]) : 256;
    cgwarp<<<blocks,threads>>>(id);
    return 0;
}
​
D:\ >cgwarp.exe 1234567 28800 256
warps and subwarps for thread 1234567:
warp32 rank in tile 7 size 32 rank 4 num 8 net size 256
warp16 rank in tile 7 size 16 rank 8 num 16 net size 256
warp8 rank in tile 7 size 8 rank 16 num 32 net size 256
tile8 rank in tile 7 size 8 rank 0 num 4 net size 32
tile4 rank in tile 3 size 4 rank 1 num 2 net size 8
```

`warp8` 和 `tile8` 的区别只有他们的 "meta" 属性不同，却决于各自的父节点。

## **Tiled Partitions**

瓦片分区有更多的成员函数。对于 `shuffe` 函数，模板 `T` 可以是 `int, long, long long, float, double`，如果包含了 `cuda_fp16.h`，则还可以是 `__half` 或者 `__half2`。这些函数都隐式的执行了必要的同步操作，不会有 read after write 错误。`shulfle` 函数会返回 0 代表对应线程已经更早的退出了。

![](https://pic3.zhimg.com/v2-cba9d9aa4a4f9ad79265c6a2a723d2b0_1440w.jpg)

这些函数使能了一个瓦片里线程们的集体操作(collective operations)。`shfl` 函数允许同一个束内的线程交换局部寄存器、共享内存或者全局内存里的值！

```cpp
template <int blockSize> 
__global__ void reduce6(r_Ptr<float> sums,cr_Ptr<float> data,int n) {
    __shared__ float s[blockSize];
​
    auto grid = cg::this_grid(); // cg definitions
    auto block = cg::this_thread_block(); // for launch
    auto warp = cg::tiled_partition<32>(block); // config
    int id = block.thread_rank(); // rank in block, 等同于 threadIdx.x
    s[id] = 0.0f; // NB simplified thread linear addressing loop
​
    for(int tid=grid.thread_rank(); tid < n; tid+=grid.size()) s[id] += data[tid];
    block.sync(); // 代替原先的 __syncthreads()
​
    if(blockSize>512 && id<512 && id+512<blockSize) s[id] += s[id + 512];
    block.sync();
​
    if(blockSize>256 && id<256 && id+256<blockSize) s[id] += s[id + 256];
    block.sync();
​
    if(blockSize>128 && id<128 && id+128<blockSize) s[id] += s[id + 128];
    block.sync();
​
    if(blockSize>64 && id<64 && id+64 < blockSize) s[id] += s[id + 64];
    block.sync();
​
    // just warp zero from here
    if(warp.meta_group_rank()==0) { // 可以用 warp.size() 代替 32, 然后写 for 循环, 但常数展开更高效
        s[id] += s[id + 32]; warp.sync(); // 代替 __syncwarp(), 这里还不能 shfl_xxxx, 可能在不同束
        s[id] += warp.shfl_down(s[id],16); // shfl_down 代替了原来的 if 和 __syncwarp
        s[id] += warp.shfl_down(s[id], 8); // 一定不能加 if, shfl_xxxx 要求所有束内的线程都执行到这一指令
        s[id] += warp.shfl_down(s[id], 4);
        s[id] += warp.shfl_down(s[id], 2);
        s[id] += warp.shfl_down(s[id], 1);
        if(id == 0) sums[blockIdx.x] = s[0]; // store sum
    }
}
```

`shfl_xxxx` 函数可以交换共享内存里的值，局部寄存器也可以做到，并且会更快一些。这给我们一个灵感：

```cpp
// 不再需要 template<int blockSize> 了, 因为只要 blockSize % 32 == 0, 他就能算对
__global__ void reduce7(r_Ptr<float> sums, cr_Ptr<float> data,int n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // accumulate thread sums in register variable v
    float v = 0.0f;
    for(int tid=grid.thread_rank(); tid<n; tid+=grid.size()) v += data[tid];
    warp.sync(); // 不再需要 __syncthreads(), 更快
    v += warp.shfl_down(v,16); // |
    v += warp.shfl_down(v, 8); // | warp level
    v += warp.shfl_down(v, 4); // | reduce here
    v += warp.shfl_down(v, 2); // |
    v += warp.shfl_down(v, 1); // |
    // use atomicAdd to sum over warps, 原子操作可能比较昂贵
    // 对于新的GPU(CC>6), 可以用 atomicAdd_block 函数, 他要求原子操作在同一个线程块内
    // 比 atomicAdd (检查所有线程) 要快
    if(warp.thread_rank()==0 ) atomicAdd(&sums[block.group_index().x], v);
    // 用 warp.thread_rank()==0 代替 id%32==0 使得程序的意图更清晰
}
```

这可能是整本书最屌的一个例子了——十分简洁，和标准的求和函数完全不同。这个核函数里，所有的线程束都被同等对待，并且一直都是激活的。他要求线程块的尺寸为 32 的倍数。不用共享内存，简化的核函数可以用尽 L1 缓存（共享内存和 L1 缓存在一个池子里）。这是所谓的 "warp-only" 核函数的一个很棒的展示！

还有一点是，这里用了 `shfl_down`，但我们也可以用 `shfl_xor` 来达到一样的效果，甚至不用动传参。`shlf_xor` 的第二个参数会被当作掩码来用，XOR 的作用是交换对应的值，而不仅仅是返回值。用 `shfl_xor` 可以再五步之后，让所有 32 个线程的 `v` 保存同样且正确的求和结果（这并不会花费额外的时间）。

```cpp
v += warp.shfl_down(v, 1);
v += warp.shfl_xor(v, 1);   // 比如 0 号线程, 0 xor 1 = 1
v += warp.shfl_down(v, 16);
v += warp.shfl_xor(v, 16);  // 比如 1 号线程, 1 xor 16 = 17
```

数组求和的例子通常用来展示共享内存的作用，但现代的 GPU 配合 warp-only 技术，甚至可以超过用共享内存的性能。最后，CUDA 11 加上了更强的功能：`[cg::reduce](https://zhida.zhihu.com/search?content_id=241994661&content_type=Article&match_order=1&q=cg%3A%3Areduce&zhida_source=entity)`。

```cpp
#include "cooperative_groups/reduce.h"
​
__global__ void reduce8(r_Ptr<float> sums, cr_Ptr<float> data, int n) {
    // ... 同上, 假定 blockSize % 32 == 0
    for(int tid=grid.thread_rank(); tid<n; tid+=grid.size()) v += data[tid];
    warp.sync();
​
    // 三个参数分别是 warp, 要被缩并的数据, 缩并操作
    v = cg::reduce(warp, v, cg::plus<float>());
    //atomic add to sum over block
    if(warp.thread_rank()==0)
        atomicAdd(&sums[block.group_index().x],v);
}
```

## **Vector Loading**

我们观察到缩并包含两个步骤：每个线程累加整个数组的部分和；缩并线程的部分和，到每个线程块的一个数字。第一步的耗时和数组尺寸强相关，但第二步需要的时间和数组尺寸是无关的。因此，对于大尺寸的数组，步骤一需要的时间占多数，但我们一直在努力优化步骤二。

观察到步骤一，每个线程每轮循环都只读取一个 `int32`，这个读操作可以被折叠，因为相邻的线程读取相邻的元素。但我们可以更进一步，每次读 128 位，这回最大化 L1 缓存的效率，并且允许编译器使用 128 位的读写指令。这个技术就叫做[向量加载](https://zhida.zhihu.com/search?content_id=242003209&content_type=Article&match_order=1&q=%E5%90%91%E9%87%8F%E5%8A%A0%E8%BD%BD&zhida_source=entity)（vector-loading）

```cpp
__global__ void reduce7_vl(r_Ptr<float> sums, cr_Ptr<float> data, int n) {
    // ... 同上, 定义 cg::grid/block/warp
​
    // use v4 to read global memory
    float4 v4 = {0.0f,0.0f,0.0f,0.0f};
    // 要求 n 是 4 的倍数
    for(int tid = grid.thread_rank(); tid < n/4; tid += grid.size()) 
        // 用 reinterpret_cast 让编译器把 data 当成 float4 类型的指针, tid 会被当成 float4 数组的下标
        // 编译器会用 128-bit 的读取指令, 访问一个完整的 128-bit L1 缓存行
        // += 是对 float4 重载过的
        v4 += reinterpret_cast<const float4 *>(data)[tid];
    // accumulate thread sums in v
    float v = v4.x + v4.y + v4.z + v4.w;
    warp.sync();
​
    // ... 同上, warp 内求和
}
```

另外提一嘴，对于这类内存瓶颈函数的计算时间的测量其实也有讲究，如果你只是单纯的循环，那可能测出来的是 L2 的速度，为了测得准，你可能需要把更多的输入拿来做轮转。

![](https://pic2.zhimg.com/v2-06c9d79ad45b22e9cb29a720cd97095d_1440w.jpg)

从图里可以看出，使用向量加载是最关键的优化。并且随着数量级越来越大，性能趋近，可能是因为内存访问主导了时间。2070 上 v7/8 差别不大，但 CC 8 的新一代 GPU 添加了对束级别缩并的硬件支持，可能会更强。

## **Warp-Level Intrinsic Functions and Sub-warps**

虽然 `cg::tiled_partition` 对象常被用于表示 32 个线程组成的束，但它也可以被用来表示 2 的指数幂尺寸的子束，这时 `size`/`thread_rank` 等成员函数会返回对应子束的属性。如果你的问题可以很自然的用子集切分，这将会带来方便。当然这只是软件层面的支持，并没有额外的硬件来支持同一个[线程束](https://zhida.zhihu.com/search?content_id=242003209&content_type=Article&match_order=1&q=%E7%BA%BF%E7%A8%8B%E6%9D%9F&zhida_source=entity)里的子集做同步操作。因此线程分歧还是会导致性能变差的，但就算存在分歧，任意尺寸的 `tiled_partition` 的 `sync` 函数都不会导致死锁。

`cg` 的成员函数，好多都有前身，都是带有双下划线的内部方法（感觉没啥屌用，不展示了）。现在都化简了，并且隐式的帮忙执行了束级别的同步（以前束内是同步的，所以那些写法已经不太安全了）。不过老旧的教程里可能还会有，另外，`cg` 里面的实现，可能也偷偷帮你调用了这些函数。

## **Thread Divergence and Synchronisation**

除了分支，还有一个差不多的事情是有些线程可能会更早退出，留下没有调用 `return` 的线程们继续干活。如果一个核函数没有分支也没有早退，那线程肯定是激活的，反之亦然。区分暂停(inactive)和退出(exited)的线程很重要，大多数的 CUDA 函数都可以优雅的处理退出的线程，但对于暂停的线程不好。比如 `__syncwarp()` 出现暂停的线程可能会陷入死锁，但已经退出的线程就没问题（还有诸如 `__syncthreads(), warp.shfl_xxx` 都是危险的源头）。`shuffle` 函数还有一个问题是线程超界，比如 `w.shfl_down(v, 16)` 对于 16~31 号线程。

![](https://pic1.zhimg.com/v2-194999f5d92290b1944a56674d6652c2_1440w.jpg)

如上表所示，我们之前的缩并求和，暂停的线程返回 0 没什么问题（？我记得加了 `if` 呀），但乘法就坏了。如果知道哪个线程是激活的，那么 `bitmask` 可以用来排除暂停线程带来的未定义行为。特别的，只有被 `bitmask` 包含的线程才会被算在 `shlf_up/down`，比如 7-10 号线程被排除了，那么 6 号线程的 `shlf_down(v, 1)` 拿到的是 11 号线程的值。这个 `bitmask` 要用前面提到的双下划线函数，很麻烦。（看文章最后的 Coalesced Groups）

## **Avoiding Deadlock**

有时候确实要在线程块级别做 `__syncthreads()`，这个函数一直存在，文档里要求块里所有还没退出的线程，都要到达调用的地方，否则核函数就会永远停滞。用共享内存来搞数据的话，通常都会至少用一次 `__syncthreads()`，如果叠满 BUFF，我是说在分支里面调用包含 `__syncthreads()` 的 `__device__` 函数，那就有意思咯。

下面的例子教你怎么用线程分歧制造死锁。

```cpp
int main(int argc, char* argv[]) {
    int warps = (argc > 1)? atoi(argv[1]) : 3; // 3 warps
    int blocks = (argc > 2)? atoi(argv[2]) : 1; // 1 block, block 其实没什么意义在这个例子里
    int gsync = (argc > 3)? atoi(argv[3]) : 32; // one warp
    int dolock = (argc > 4)? atoi(argv[4]) : 1; // use lock?
    printf("about to call\n");
    deadlock<<<blocks,warps*32>>>(gsync,dolock);
    printf("done\n");
    return 0;
}
​
__global__ void deadlock(int gsync, int dolock) {
    __shared__ int lock;
​
    if (threadIdx.x == 0) lock = 0;
    __syncthreads(); // 这个同步是很安全的
​
    if (threadIdx.x < gsync) { // group A
        __syncthreads(); // sync A, 如果 blockDim.x≥gsync, 后面的线程就有机会死锁啦
        if (threadIdx.x == 0) lock = 1; // 前面这行同步成功了, 才会解开后面的 while 循环
    }
​
    else if (threadIdx.x < 2*gsync) { // group B
        __syncthreads(); // 根据文档, 早期的机器可能这样就死锁了, 但新设备到这里还是好的
    }
​
    // 如果 dolock=0, 那 C 组就是早退的, 可能没法死锁了
    if(dolock) while (lock != 1);
​
    // see message only if NO deadlock
    if(threadIdx.x == 0 && blockIdx.x==0)
        printf("deadlock OK\n");
}
```

这个核函数把线程块分成了三组：`[0, gsync-1]` 为 A 组，`[gsync, gsync*2-1]` 为 B 组，他们在不同的地方调用了 `__syncthreads()`，这还不足以死锁，我们再来一组，既不调用 `__syncthreads()`，也不退出的。看看结果

![](https://pica.zhimg.com/v2-b2f6bc66e6ea89cc1f8e2e8ca03ee2ca_1440w.jpg)

- 第一行：只有两组，只是在不一样的地方执行了同步，没发生死锁。
- 第二行：如果不打开 `while` 循环，正常；如果打开了，第三组的线程都所在循环里。
- 第三行：这个配置共 64 个线程，前 16 个是 A 组，最后 32 个是 C 组。对于高级卡，不开启 `while` 也死锁了！！因为硬件确实检查了所有非退出线程要调用同一处 `__syncthreads()`
- 第四行：共 96 个线程，前一半是 A 组，后一半是 B 组。也就是说，1 号束被分割了。没有 C 组。对于老旧的设备，正常执行；但对于新卡，都锁住了。因为线程束 1 包含了不同的调用。
- 第五行：共 128 个线程，最后一个线程束是 C 组，其余同上。这里旧设备在 `while` 打开的情况也死锁了。

结论：新设备，同一个束里还未退出的线程要调用同一处同步，但不同的束可以执行不同的同步。对于旧设备，如果剩下的线程都退出了，貌似就不会死锁。（建议别学旧设备了）

> For devices of CC≥7 For all warps in each thread-block all non-exited threads in a particular warp must execute the same `__syncthreads()` call but different warps can execute different `__syncthreads()` calls.  
> For devices of CC<7 For all warps in each thread-block, at least one thread from each warp having non-exited threads must execute a `__syncthreads()` call.

最好的办法就是避免在可能出现线程分歧的地方做 `__syncthreads()` 啦，可以尝试用线程束级别的同步，[协作组](https://zhida.zhihu.com/search?content_id=242003209&content_type=Article&match_order=1&q=%E5%8D%8F%E4%BD%9C%E7%BB%84&zhida_source=entity)会帮你。对于大型的项目，你可能没有 `__device__` 函数的源码，不知道里面是否调用了 `__syncthreads()`，优秀的作者会用一个弱一点的同步，只对当前激活的线程有效，这就下面要讲的东西。

## **Coalesced Groups**

协作组里的 `coalesced_group` 对象（合并组？）类似 `tiled_partition` 但只包含了当前线程组里激活状态的线程。和 `tiled_partition` 不同的是，不需要手动指定他的尺寸为 2 的次幂，他的尺寸是自动设置的，值是这个对象创建时的激活状态线程数量。`coalesced_group` 的打开方式和 `thread_block` 很像：

```cpp
auto a = cg::coalesced_threads(); // a for active
cg::coalesced_group a = cg::coalesced_threads();
```

`a` 包含了当前线程束所有激活状态的线程，这也是一个 warp-level 的对象。成员函数 `a.size()` 返回激活状态的线程数量，`shuffle` 函数会自动加上只包含激活线程的掩码。`a.rank()` 的取值范围是 `[0, a.size() - 1]`， 最重要的 `a.sync()` 会执行只有这些激活状态线程的同步，这可以避免死锁并且使代码的意图更清晰。注意，在合并组对象实例化后，不应该再有进一步的线程分歧出现。

下面的示例无论用什么配置启动，都不会死锁

```cpp
__global__ void deadlock_coalesced(int gsync, int dolock) {
    __shared__ int lock;
    if (threadIdx.x == 0) lock = 0;
    __syncthreads(); // normal syncthreads
​
    if (threadIdx.x < gsync) { // group A
        // 这个对象有所有 tile_partition<32> 表示的线程束的功能, 但只作用在激活的线程上
        auto a = cg::coalesced_threads(); 
        a.sync(); // sync A
        if (threadIdx.x == 0) lock = 1;
    }
    else if (threadIdx.x < 2 * gsync) { // group B
        auto a = cg::coalesced_threads();
        a.sync(); // sync B
    }
​
    if (dolock) while (lock != 1);
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("deadlock_coalesced OK\n");
}
```

前面两个例子都是讲怎么死锁和避免的，下面看看合并组能真正做什么。这个例子还是数组求和，但是先分别计算奇数线程号和偶数线程号各自的和。

```cpp
__device__ void reduce7_vl_coal(r_Ptr<float>sums, cr_Ptr<float>data, int n) {
    // 假设 a.size 是 2 的次幂(否则 shlf 有可能越界), n 是 4 的倍数
    auto g = cg::this_grid();
    auto b = cg::this_thread_block();
    auto a = cg::coalesced_threads(); // active threads in warp
​
    float4 v4 ={0,0,0,0};
    for(int tid = g.thread_rank(); tid < n/4; tid += g.size())
        v4 += reinterpret_cast<const float4 *>(data)[tid];
    float v = v4.x + v4.y + v4.z + v4.w;
    a.sync();
​
    if(a.size() > 16) v += a.shfl_down(v,16); // NB no new
    if(a.size() > 8) v += a.shfl_down(v,8); // thread
    if(a.size() > 4) v += a.shfl_down(v,4); // divergence
    if(a.size() > 2) v += a.shfl_down(v,2); // allowed
    if(a.size() > 1) v += a.shfl_down(v,1); // here
    if(a.thread_rank() == 0)
        atomicAdd(&sums[b.group_index().x],v);
}
​
__global__ void reduce_warp_even_odd(
    r_Ptr<float>sumeven, r_Ptr<float>sumodd, cr_Ptr<float>data, int n) {
    // divergent code here
    // CC < 7 的机器, 这两步是顺序执行的; >= 7 的机器, 有一些并行(比前面慢了 1.8 倍)
    if (threadIdx.x%2==0) reduce_coal_vl(sumeven, data, n);
    else reduce_coal_vl(sumodd, data, n);
}
```

下面最后一个示例，它能够计算出任何大小的合并组的求和，前提是每个线程束至少有一个激活的线程。在这个版本中，每个线程束不再需要有相同数量的激活线程。

```cpp
 // 我们把数据分成连续的块, 每个块由网格中的一个线程束来处理. 
// 如果每个束里激活线程的数量不同, 这个方法性能会差一些, 因为给每个束的任务总量是一样的.
__device__ void reduce_coal_any_vl(r_Ptr<float>sums, cr_Ptr<float>data,int n) {
    // a.size [1, 32] 都可以算
    auto g = cg::this_grid();
    auto b = cg::this_thread_block();
    auto w = cg::tiled_partition<32>(b); // whole warp
    auto a = cg::coalesced_threads();
​
    // number of warps in grid
    // g.group_dim().x = 线程块数量, w.meta_group_size() = 线程块里有多少线程束
    // a.meta_group_size() 永远等于 1, 因为合并组被视为线程束的分割, 每个束只分一个出来
    int warps = g.group_dim().x * w.meta_group_size(); 
​
    // divide data into contiguous parts, with one part per warp
    int part_size =((n/4)+warps-1)/warps; // 向上取整, /4 
    // 当前 warp 在 grid 里的编号 * part_size
    int part_start =(b.group_index().x*w.meta_group_size()+w.meta_group_rank() )*part_size;
    int part_end =min(part_start+part_size,n/4);
    
    // get part sub-sums into threads of a
    float4 v4 ={0,0,0,0};
    int id = a.thread_rank();
    // adjacent adds within the warp
    for(int k=part_start+id; k<part_end; k+=a.size())
        v4 += reinterpret_cast<const float4 *>(data)[k];
    float v = v4.x + v4.y + v4.z + v4.w;
    a.sync();

    // 这几行代码解决激活线程数量 a.size() 不是 2 整数幂的问题
    // kstart 是比 a.size 小的最大 2 的整数幂, clz 返回无符号数前导 0 的数量
    int kstart = 1 << (31 - __clz(a.size()));
    // 把后面线程的值加到前面线程
    if(a.size() > kstart) {
        float w = a.shfl_down(v,kstart);
        // only update v for valid low ranking threads
        if(a.thread_rank() < a.size()-kstart) v += w;
        a.sync();
    }
​
    // now do power of 2 reduction
    // 由于 kstart 不知道多少, 就不好 unroll 了
    for(int k = kstart/2; k>0; k /= 2)v+= a.shfl_down(v,k);
    if(a.thread_rank() == 0)
        AtomicAdd(&sums[b.group_index().x],v);
}
​
__global__ void reduce_any(r_Ptr<float>sums, cr_Ptr<float>data, int n) {
    if(threadIdx.x%3 ==0) reduce_coal_any_vl(sums,data,n);
}
```

`reduce_coal_any_vl` 的表现还不错，下图是数据大小为 2^24，<<<288, 256>>> 的表现。16 以后基本能打满内存带宽。

![](https://pic4.zhimg.com/v2-7100607831a8ceff58d150f5ce15e083_1440w.jpg)

## **HPC Features**

对于 Linux 集群上的高性能计算，网格级别的同步是可能需要的（？这不就是宿主上调用 `syncDevice` 吗），使用 `grip.sync` 函数需要特别的编译以及特殊的 API。细节就不说了。现在网格级别的同步是有限制的，主要来源于：所有的被同步的线程块，都需要同时存留在 GPU，这意味着线程块的数量不能超过 SM 的数量。

对于高性能计算的应用，核函数要在多个 GPU 上启动的，就有 `multi-grid` 的说法了。

```cpp
auto mg = this_multi_grid();
multi_grid_group mg = this_multi_grid();
​
mg.sync(); // Synchronization across multiple GPUs is possible with
```

CUDA 11 提供了许多新功能，除了前面说的 `reduce` 函数。

1 子合并组，`label` 是个自己设置的正整数，可能每个线程都不一样，用来区分。

```text
cg::coalesced_group a = cg::coalesced_threads();
cg::coalesced_group lg = cg::labeled_partition(a,label);
​
auto a = cg::coalesced_threads();
auto lg = cg::labeled_partition(a,label);
```

2 binary_partitition 是上面的一个特例，他接受布尔的 label 作为参数，最多分两组。

3 新的 memcpy_async 函数，用来加速从 GPU 全局内存到共享内存的拷贝速度。这是通过硬件支持的，绕过 L1 和 L2。引入他的主要目的是搞后面要讲的 Tensor Core