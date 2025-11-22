内容基本来自这本书的第三章：[Programming in Parallel with CUDA (cambridge.org)](https://link.zhihu.com/?target=https%3A//www.cambridge.org/core/books/programming-in-parallel-with-cuda/C43652A69033C25AD6933368CDBE084C)，书是 22 年 5 月出版的，已经算比较新的了。

> 区别于其他 CUDA 书籍的一个特点是，这本书里的 CUDA 示例基于有趣的实际问题，并且还使用现代 C++ 的特性来编写出简单、优雅、紧凑的代码。目前在网上关于 CUDA 的教程或教科书中，大多数示例都太为了写而写，并且基于冗长、过时的 C 语言风格。

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