CUDA C++编程指南

# 1.[一览表](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overview "这个标题的永久链接")

CUDA是由NVIDIA开发的平行计算平台和编程模型，通过利用GPU的强大功能，可以大幅提高计算性能。它允许开发人员使用C、C++和Fortran加速计算密集型应用程序，并被广泛应用于深度学习、科学计算和高性能计算（HPC）等领域。

# 2.什么是CUDA C编程指南？

CUDA C编程指南是官方的综合资源，解释了如何使用CUDA平台编写程序。它提供了CUDA架构、编程模型、语言扩展和性能指南的详细文档。无论您是刚刚开始还是优化复杂的GPU内核，本指南都是有效利用CUDA的全部功能的重要参考。

# 3.[介绍](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#introduction "这个标题的永久链接")

## 3.1.使用GPU的好处[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-benefits-of-using-gpus "这个标题的永久链接")

图形处理单元（GPU）[1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn1)在类似价格和功率包络范围内提供比CPU更高的指令吞吐量和内存带宽。许多应用程序利用这些更高的功能，在GPU上运行速度比在CPU上运行得更快（请参阅[GPU应用程序](https://www.nvidia.com/object/gpu-applications.html)）。其他计算设备，如FPGA，也非常节能，但编程灵活性比GPU低得多。

GPU和CPU之间的这种能力差异是因为它们的设计考虑到了不同的目标。虽然CPU的设计可以尽可能快地执行一系列操作，称为_线程_，并且可以并行执行几十个这些线程，但GPU的设计可以并行执行数千个线程（摊差较慢的单线程性能以实现更高的吞吐量）。

GPU专门用于高度并行计算，因此设计成更多的晶体管用于数据处理，而不是数据缓存和流量控制。示意图[1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing)显示了CPU与GPU的芯片资源分布示例。

[![GPU将更多的晶体管用于数据处理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)

图1 GPU将更多的晶体管用于数据处理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing "此图像的永久链接")

将更多的晶体管用于数据处理，例如浮点计算，有利于高度并行计算；GPU可以通过计算隐藏内存访问延迟，而不是依赖大型数据缓存和复杂的流控制来避免长时间的内存访问延迟，这两者都在晶体管方面都很昂贵。

一般来说，应用程序混合了并行部分和顺序部分，因此系统设计了GPU和CPU的组合，以最大限度地提高整体性能。具有高度并行性的应用程序可以利用GPU的这种大规模并行性质来实现比CPU更高的性能。

[1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id2) _图形_限定符源于这样一个事实，即20年前GPU最初创建时，它被设计为一个专门的处理器来加速图形渲染。在市场对实时、高清、3D图形的不满足需求的驱使下，它已经发展成为一种通用处理器，用于更多工作负载，而不仅仅是图形渲染。

## 3.2.CUDA®：通用并行计算平台和编程模型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-a-general-purpose-parallel-computing-platform-and-programming-model "这个标题的永久链接")

2006年11月，NVIDIA®推出了CUDA®，这是一个通用的并行计算平台和编程模型，它利用NVIDIA GPU中的并行计算引擎，以比CPU更有效的方式解决许多复杂的计算问题。

CUDA附带一个软件环境，允许开发人员使用C++作为高级编程语言。如[图2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-general-purpose-parallel-computing-architecture-cuda-is-designed-to-support-various-languages-and-application-programming-interfaces)所示，支持其他语言、应用程序编程接口或基于指令的方法，如FORTRAN、DirectCompute、OpenACC。

[![GPU计算应用程序。CUDA旨在支持各种语言和应用程序编程接口。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-computing-applications.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-computing-applications.png)

图2 GPU计算应用。CUDA旨在支持各种语言和应用程序编程接口。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-general-purpose-parallel-computing-architecture-cuda-is-designed-to-support-various-languages-and-application-programming-interfaces "此图像的永久链接")

## 3.3.可扩展的编程模型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#a-scalable-programming-model "这个标题的永久链接")

多核CPU和多核GPU的出現意味著主流處理器晶片現在是並行系統。挑战是开发透明地扩展其并行性的应用软件，以利用越来越多的处理器内核，就像3D图形应用程序透明地将其并行性扩展到核心数量大不相同的多核GPU一样。

CUDA平行编程模型旨在克服这一挑战，同时为熟悉C等标准编程语言的程序员保持低学习曲线。

其核心是三个关键的抽象——线程组的层次结构、共享内存和屏障同步——这些抽象只是作为一组最小的语言扩展暴露给程序员。

这些抽象提供了精细的数据并行性和线程并行性，嵌套在粗粒数据并行性和任务并行性中。他们引导程序员将问题划分为粗的子问题，这些问题可以通过线程块并行独立解决，并将每个子问题划分为更精细的部分，这些子问题可以通过块内的所有线程并行合作解决。

这种分解通过允许线程在解决每个子问题时进行合作来保持语言表现力，同时实现自动可扩展性。事实上，每个线程块都可以按任何顺序、并发或顺序安排在GPU中的任何可用的多处理器上，因此编译的CUDA程序可以在任何数量的多处理器上执行，如[图3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#scalable-programming-model-automatic-scalability)所示，只有运行时系统需要知道物理多处理器计数。

这种可扩展的编程模型允许GPU架构通过简单地扩展多处理器和内存分区的数量来跨越广泛的市场范围：从高性能爱好者的GeForce GPU和专业的Quadro和Tesla计算产品到各种廉价的主流GeForce GPU（请参阅[支持CUDA的GPU，](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-enabled-gpus)以获取所有支持CUDA的GPU的列表）。

![自动可扩展性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/automatic-scalability.png)

图3 自动可扩展性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#scalable-programming-model-automatic-scalability "此图像的永久链接")

笔记

GPU是围绕一系列流式多处理器（SM）构建的（有关更多详细信息，请参阅[硬件实现](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-implementation)）。多线程程序被划分为线程块，这些线程块相互独立执行，因此具有更多多处理器的GPU将比具有较少多处理器的GPU在更短的时间内自动执行程序。

# 4.更新日志[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#changelog "这个标题的永久链接")

表1变更日志[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id452 "此表的永久链接")
|版本|变化|
|---|---|
|13.0|将指令吞吐量表从CUDA C++编程指南的“_性能指南”_部分移至CUDA C++最佳实践指南的[指令优化](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#instruction-optimization)部分。删除了不受支持的架构，并更正了整数算术和类型转换的条目。|
|12.9|在[CUDA环境变量中](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#env-vars)添加了[错误日志](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#error-log-management)管理和CUDA_LOG_FILE部分|
|12.8|添加了“[TMA Swizzle”](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tma-swizzle)部分|

# 5.编程模型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model "这个标题的永久链接")

本章通过概述如何在C++中展示CUDA编程模型背后的主要概念，介绍它们。

[编程界面](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface)中对CUDA C++进行了广泛的描述。

本章中使用的向量加法示例和下一章的完整代码可以在[vectorAdd CUDA样本](https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition)中找到。

## 5.1.内核[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernels "这个标题的永久链接")

CUDA C++扩展了C++，允许程序员定义称为_内核的_C++函数，当调用时，由N不同的_CUDA线程_并行执行N次，而不是像常规C++函数那样只执行一次。

核心使用`__global__`声明指定符定义，并使用新的`<<<...>>>`_执行配置_语法（请参阅[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)）指定为给定内核调用执行该内核的CUDA线程数量。每个执行内核的线程都有一个唯一的_线程ID_，可以通过内置变量在内核内访问。

举例来说，以下示例代码使用内置变量`threadIdx`，添加两个大小为_N_的向量_A_和_B，_并将结果存储在向量_C中_。

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}

在这里，执行`VecAdd()`的_N个_线程中的每一个都会执行一个成对的添加。

## 5.2.线程层次结构[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy "这个标题的永久链接")

为了方便起见，`threadIdx`是一个3组分向量，因此可以使用一维、二维或三维线_程索引来_识别线程，形成一个一维、二维或三维线程块，称为线_程块_。这提供了一种自然的方式来调用跨域中元素的计算，如向量、矩阵或体积。

线程的索引及其线程ID以直截了当的方式相互关联：对于一维块，它们是相同的；对于尺寸的二维块_（Dx，Dy），_索引线程的线程ID（x_，y）_是_（x + y Dx）；_对于尺寸的三维块_（Dx，Dy，Dz），_索引线程_（x，y，z）_的线程ID是_（x + y Dx + z Dx Dx Dy）。_

例如，以下代码添加两个大小为_NxN_的矩阵A和_B_，并将结果存储在矩阵_C中_。

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}

每个块的线程数量是有限制的，因为一个块的所有线程都应该位于同一个流式多处理器内核上，并且必须共享该内核的有限内存资源。在当前的GPU上，一个线程块可能包含多达1024个线程。

然而，一个内核可以通过多个相同形状的线程块执行，因此线程总数等于每个块的线程数乘以块数。

如[图4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy-grid-of-thread-blocks)所示，块被组织成一维、二维或三维的线程块_网格_。网格中的线程块数量通常由正在处理的数据大小决定，这通常超过系统中的处理器数量。

[![线程块网格](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)

图4螺纹块网格[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy-grid-of-thread-blocks "此图像的永久链接")

`<<<...>>>`语法中指定的每个块的线程数和每个网格的块数可以是`int`或`dim3`类型。可以像上面的例子一样指定二维块或网格。

网格中的每个块都可以由一维、二維或三维唯一索引识别，该索引可以通过内置的`blockIdx`变量在核心内访问。线程块的维度可以通过内置的`blockDim`变量在内核内访问。

扩展之前的`MatAdd()`示例来处理多个块，代码如下所示。

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}

线程块大小为16x16（256线程），虽然在这种情况下是任意的，但这是一个常见的选择。网格创建时有足够的块，每个矩阵元素都有一个线程，就像以前一样。为了简单起见，本示例假设每个维度中每个网格的线程数被该维度中每个块的线程数均匀整除，尽管情况并非如此。

线程块需要独立执行。必须能够以任何顺序，并行或串联执行块。如[图3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#scalable-programming-model-automatic-scalability)所示，这种独立性要求允许以任何顺序和跨任意数量的内核进行调度，使程序员能够编写与内核数量一起扩展的代码。

块中的线程可以通过一些_共享内存_共享数据，并通过同步其执行来协调内存访问来进行合作。更准确地说，可以通过调用`__syncthreads()`内在函数来指定内核中的同步点；`__syncthreads()`充当障碍，在允许任何线程继续之前，块中的所有线程都必须等待。[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory)给出了一个使用共享内存的示例。除了`__syncthreads()`之外，[合作组API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups)还提供了一组丰富的线程同步原语。

为了高效合作，共享内存预计将在每个处理器内核附近（非常像L1缓存）的低延迟内存，`__syncthreads()`预计将是轻量级的。

### 5.2.1.线程块集群[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters "这个标题的永久链接")

随着NVIDIA[计算能力9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0)的引入，CUDA编程模型引入了一个可选的层次结构，称为线程块集群，由线程块组成。类似于线程块中的线程保证在流式多处理器上共同调度，集群中的线程块也保证在GPU中的GPU处理集群（GPC）上共同调度。

Similar to thread blocks, clusters are also organized into a one-dimension, two-dimension, or three-dimension grid of thread block clusters as illustrated by [Figure 5](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters-grid-of-clusters). The number of thread blocks in a cluster can be user-defined, and a maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA. Note that on GPU hardware or MIG configurations which are too small to support 8 multiprocessors the maximum cluster size will be reduced accordingly. Identification of these smaller configurations, as well as of larger configurations supporting a thread block cluster size beyond 8, is architecture-specific and can be queried using the `cudaOccupancyMaxPotentialClusterSize` API.

[![线程块集群网格](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png)

图5 线程块集群的网格[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters-grid-of-clusters "此图像的永久链接")

笔记

在使用集群支持启动的内核中，为了兼容性，gridDim变量仍然以线程块数量表示大小。集群中块的排名可以使用[集群组](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-group-cg)API找到。

线程块集群可以在内核中使用`__cluster_dims__(X,Y,Z)`的编译时内核属性或使用CUDA内核启动API `cudaLaunchKernelEx`。下面的示例展示了如何使用编译时内核属性启动集群。使用内核属性的集群大小在编译时是固定的，然后可以使用经典的`<<<,>>>`启动内核。如果内核使用编译时的集群大小，则在启动内核时无法修改集群大小。

// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}

线程块集群大小也可以在运行时设置，并且可以使用CUDA内核启动API `cudaLaunchKernelEx`启动内核。下面的代码示例展示了如何使用可扩展API启动集群内核。

// Kernel definition
// No compile time attribute attached to the kernel
__global__ void cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Kernel invocation with runtime cluster size
    {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}

In GPUs with compute capability 9.0, all the thread blocks in the cluster are guaranteed to be co-scheduled on a single GPU Processing Cluster (GPC) and allow thread blocks in the cluster to perform hardware-supported synchronization using the [Cluster Group](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-group-cg) API `cluster.sync()`. Cluster group also provides member functions to query cluster group size in terms of number of threads or number of blocks using `num_threads()` and `num_blocks()` API respectively. The rank of a thread or block in the cluster group can be queried using `dim_threads()` and `dim_blocks()` API respectively.

属于集群的线程块可以访问分布式共享内存。集群中的线程块能够读取、写入和执行分布式共享内存中的任何地址的原子。[分布式共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#distributed-shared-memory)给出了一个在分布式共享内存中执行直方图的示例。

### 5.2.2.块作为集群[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#blocks-as-clusters "这个标题的永久链接")

使用`__cluster_dims__`，启动的集群数量保持隐式，只能手动计算。

__cluster_dims__((2, 2, 2)) __global__ void foo();

// 8x8x8 clusters each with 2x2x2 thread blocks.
foo<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();

在上述示例中，内核以16x16x16线程块的网格或实际上8x8x8集群的网格启动。或者，使用另一个编译时内核属性`__block_size__`，允许启动一个显式配置为线程块集群数量的网格。

// Implementation detail of how many threads per block and blocks per cluster
// is handled as an attribute of the kernel.
__block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();

// 8x8x8 clusters.
foo<<<dim3(8, 8, 8)>>>();

`__block_size__`需要两个字段，每个字段都是3个元素的元组。第一个元组表示块维度和第二个集群大小。如果第二个元组没有通过，则假定为`(1,1,1)`）。要指定流，必须传递`1`和`0`作为`<<<>>>`中的第二个和第三个参数，最后传递流。传递其他值会导致未定义的行为。

请注意，同时指定`__block_size__`和`__cluster_dims__`的第二个元组是非法的。当指定`__block_size__`的第二个元组时，它意味着启用了“块作为集群”，编译器将识别`<<<>>>`中的第一个参数为集群的数量，而不是线程块。

## 5.3.记忆层次结构[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-hierarchy "这个标题的永久链接")

如[图6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-hierarchy-memory-hierarchy-figure)所示，CUDA线程在执行过程中可能会访问来自多个内存空间的数据。每个线程都有私有本地内存。每个线程块都有共享内存，对块的所有线程可见，并且与块的寿命相同。线程块集群中的线程块可以在彼此的共享内存上执行读取、写入和原子操作。所有线程都可以访问相同的全局内存。

还有两个额外的只读内存空间可供所有线程访问：常量和纹理内存空间。全局、常量和纹理内存空间针对不同的内存用途进行了优化（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)）。纹理存储器还为某些特定数据格式提供不同的寻址模式以及数据过滤（请参阅[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)）。

全局、常量和纹理内存空间在同一应用程序的内核启动中是持久的。

![记忆层次结构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png)

图6 记忆层次结构[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-hierarchy-memory-hierarchy-figure "此图像的永久链接")

## 5.4.异构编程[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming "这个标题的永久链接")

如[图7](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming-heterogeneous-programming)所示，CUDA编程模型假设CUDA线程在物理上独立的_设备上_执行，该_设备_作为运行C++程序的_主机_的协处理器运行。例如，当内核在GPU上执行，而C++程序的其余部分在CPU上执行时，就是这种情况。

CUDA寫程式模型還假設主機和裝置在DRAM中保持自己的獨立記憶體空間，分別稱為_主機記憶體_和_裝置記憶體_。因此，程序通过调用CUDA运行时（在[编程接口中](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface)描述）来管理内核可见的全局、常量和纹理内存空间。这包括设备内存分配和去分配，以及主机和设备内存之间的数据传输。

统一内存提供_托管内存_，以连接主机和设备内存空间。托管内存可以从系统中的所有CPU和GPU访问，作为具有共同地址空间的单个、一致的内存映像。此功能允许设备内存的过度订阅，并且可以通过消除在主机和设备上显式镜像数据的需要来大大简化应用程序的移植任务。请参阅[统一内存编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-unified-memory-programming-hd)，了解统一内存的介绍。

![异构编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/heterogeneous-programming.png)

图7异构编程[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming-heterogeneous-programming "此图像的永久链接")

笔记

串行代码在主机上执行，而并行代码在设备上执行。

## 5.5.异步SIMT编程模型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-simt-programming-model "这个标题的永久链接")

在CUDA编程模型中，线程是进行计算或内存操作的最低抽象级别。从基于**NVIDIA Ampere GPU架构**的设备开始，CUDA编程模型通过异步编程模型加速内存操作。异步编程模型定义了非同步操作相对于CUDA线程的行为。

异步编程模型定义了CUDA线程之间同步的[异步屏障](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier)的行为。该模型还解释和定义了如何在GPU中计算时使用[cuda::memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies)从全局内存异步移动数据。

### 5.5.1.异步操作[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-operations "这个标题的永久链接")

异步操作被定义为由CUDA线程发起的操作，并由另一个线程异步执行。在一个结构良好的程序中，一个或多个CUDA线程与异步操作同步。启动异步操作的CUDA线程不需要在同步线程中。

这样的异步线程（as-if线程）总是与启动异步操作的CUDA线程相关联。非同步操作使用同步对象来同步操作的完成。此类同步对象可以由用户显式管理（例如，`cuda::memcpy_async`）或在库中隐式管理（例如，`cooperative_groups::memcpy_async`）。

同步对象可以是`cuda::barrier`或`cuda::pipeline`。[使用cuda::pipeline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies)的[异步屏障](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier)和[异步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies)中对这些对象进行了详细解释。这些同步对象可以在不同的线程范围内使用。范围定义了可以使用同步对象与异步操作同步的线程集。下表定义了CUDA C++中可用的线程范围以及可以与每个线程同步的线程。

|线程范围|描述|
|---|---|
|`cuda::thread_scope::thread_scope_thread`|只有启动异步操作的CUDA线程才会同步。|
|`cuda::thread_scope::thread_scope_block`|与启动线程同步的同一线程块内的所有或任何CUDA线程。|
|`cuda::thread_scope::thread_scope_device`|同一GPU设备中的所有或任何CUDA线程与启动线程同步。|
|`cuda::thread_scope::thread_scope_system`|同一系统中的所有或任何CUDA或CPU线程与启动线程同步。|

这些线程范围是作为[CUDA标准C++库](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes)中标准C++的扩展实现的。

## 5.6.计算能力[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability "这个标题的永久链接")

设备的_计算能力_由版本号表示，有时也称为其“SM版本”。此版本号标识GPU硬件支持的功能，并在运行时由应用程序使用，以确定当前GPU上可用的硬件功能和/或指令。

计算能力包括一个主要修订号X和一个小修订号Y，用_X.Y_表示。

主要修订号表示设备的核心GPU架构。具有相同主要修订号的设备共享相同的基本架构。下表列出了与每个NVIDIA GPU架构相对应的主要修订号。

表2 GPU架构和主要修订编号[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id453 "此表的永久链接")
|主要修订号|NVIDIA GPU架构|
|---|---|
|9|NVIDIA Hopper GPU架构|
|8|英伟达安培GPU架构|
|7|NVIDIA Volta GPU架构|
|6|NVIDIA Pascal GPU架构|
|5|NVIDIA Maxwell GPU架构|
|3|NVIDIA Kepler GPU架构|

次要修订号对应于对核心架构的增量改进，可能包括新功能。

表3 GPU架构的增量更新[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id454 "此表的永久链接")
|计算能力|NVIDIA GPU架构|基于|
|---|---|---|
|7.5|NVIDIA图灵GPU架构|NVIDIA Volta GPU架构|

[支持CUDA的GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-enabled-gpus)列出了所有支持CUDA的设备及其计算能力。[计算能力](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities)提供了每种计算能力的技术规格。

笔记

特定GPU的计算能力版本不应与CUDA版本（例如，CUDA 7.5、CUDA 8、CUDA 9）混淆，后者是CUDA_软件平台_的版本。应用程序开发人员使用CUDA平台创建在多代GPU架构上运行的应用程序，包括尚未发明的未来GPU架构。虽然新版本的CUDA平台通常通过支持该架构的计算能力版本来增加对新GPU架构的原生支持，但新版本的CUDA平台通常还包括独立于硬件生成的软件功能。

从CUDA 7.0和CUDA 9.0开始，_特斯拉_和_Fermi_架构已不再受支持。

# 6.编程接口[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface "这个标题的永久链接")

CUDA C++为熟悉C++编程语言的用户提供了一个简单的路径，以便轻松编写程序供设备执行。

它由一组C++语言的最小扩展和一个运行时库组成。

核心语言扩展已在[编程模型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model)中引入。它们允许程序员将内核定义为C++函数，并在每次调用函数时使用一些新的语法来指定网格和块维度。所有扩展的完整描述可以在[C++语言扩展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions)中找到。任何包含其中一些扩展的源文件都必须使用`nvcc`编译，如[使用NVCC编译](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-with-nvcc)中所述。

运行时在[CUDA运行时](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-runtime)中引入。它提供了在主机上执行的C和C++函数，用于分配和去分配设备内存，在主机内存和设备内存之间传输数据，管理具有多个设备的系统等。运行时的完整描述可以在CUDA参考手册中找到。

运行时建立在较低级别的C API（CUDA驱动程序API）之上，该API也可以通过应用程序访问。驱动程序API通过暴露较低级别的概念，如CUDA上下文（设备主机进程的类似物）和CUDA模块（设备动态加载库的类似物）等低级别的控制。大多数应用程序不使用驱动程序API，因为它们不需要这种额外的控制级别，在使用运行时，上下文和模块管理是隐含的，因此代码更加简洁。由于运行时与驱动程序API可互操作，大多数需要一些驱动程序API功能的应用程序都可以默认使用运行时API，并且仅在需要时使用驱动程序API。驱动程序API在[驱动程序API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-api)中介绍，并在参考手册中进行了全面描述。

## 6.1.与NVCC的汇编[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-with-nvcc "这个标题的永久链接")

内核可以使用称为_PTX_的CUDA指令集架构编写，该架构在PTX参考手册中有描述。然而，使用C++等高级编程语言通常更有效。在这两种情况下，核心必须由`nvcc`编译成二进制代码才能在设备上执行。

`nvcc` is a compiler driver that simplifies the process of compiling _C++_ or _PTX_ code: It provides simple and familiar command line options and executes them by invoking the collection of tools that implement the different compilation stages. This section gives an overview of `nvcc`workflow and command options. A complete description can be found in the `nvcc` user manual.

### 6.1.1.编译工作流程[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-workflow "这个标题的永久链接")

#### 6.1.1.1.离线汇编[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#offline-compilation "这个标题的永久链接")

使用`nvcc`编译的源文件可以包括主机代码（即在主机上执行的代码）和设备代码（即在设备上执行的代码）的混合。`nvcc`的基本工作流程包括将设备代码与主机代码分开，然后：

- 将设备代码编译成汇编形式（_PTX_代码）和/或二进制形式（_cubin_对象），
    
- 并通过将[内核](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernels)中引入的`<<<...>>>`语法（并在[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)中进行更详细地描述）替换为必要的CUDA运行时函数调用来修改主机代码，以从_PTX_代码和/或_cubin_对象中加载和启动每个编译的内核。
    

修改后的主机代码要么输出为使用其他工具编译的C++代码，要么直接输出为对象代码，让`nvcc`在最后一个编译阶段调用主机编译器。

然后应用程序可以：

- 要么链接到编译的主机代码（这是最常见的情况），
    
- 或者忽略修改后的主机代码（如果有的话），并使用CUDA驱动程序API（请参阅[驱动程序API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-api)）来加载和执行_PTX_代码或_cubin_对象。
    

#### 6.1.1.2.及时汇编[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#just-in-time-compilation "这个标题的永久链接")

应用程序在运行时加载的任何_PTX_代码都由设备驱动程序进一步编译为二进制代码。这被称为_及时编译_。即时编译会增加应用程序加载时间，但允许应用程序从每个新设备驱动程序带来的任何新编译器改进中受益。这也是应用程序在编译时不存在的设备上运行的唯一方法，详见[应用程序兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility)。

当设备驱动程序及时为某些应用程序编译一些_PTX_代码时，它会自动缓存生成的二进制代码的副本，以避免在应用程序的后续调用中重复编译。当设备驱动程序升级时，缓存（称为_计算缓存_）会自动失效，因此应用程序可以从设备驱动程序中内置的新及时编译器的改进中受益。

环境变量可用于控制[CUDA环境变量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#env-vars)中描述的及时编译

作为使用`nvcc`编译CUDA C++设备代码的替代方案，NVRTC可用于在运行时将CUDA C++设备代码编译为PTX。NVRTC是CUDA C++的运行时编译库；更多信息可以在NVRTC用户指南中找到。

### 6.1.2.二进制兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#binary-compatibility "这个标题的永久链接")

二进制代码是特定于架构的。使用指定目标架构的编译器选项`-code`生成_cubin_对象：例如，使用`-code=sm_80`编译为[计算能力](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability)8.0的设备生成二进制代码。保证从一个小修订到下一个小修订的二进制兼容性，但不能保证从一个小修订到上一个修订或跨主要修订。换句话说，为计算能力_X.y_生成的_立方体_对象只会在_z≥y_的计算能力_X.z_的设备上执行。

笔记

二进制兼容性仅支持桌面。Tegra不支持它。此外，不支持桌面和Tegra之间的二进制兼容性。

### 6.1.3.PTX兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ptx-compatibility "这个标题的永久链接")

一些_PTX_指令仅在具有更高计算能力的设备上支持。例如，[Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)仅支持具有5.0及以上计算能力的设备。`-arch`编译器选项指定了编译C++到_PTX_代码时假设的计算能力。因此，例如，包含warp shuffle的代码必须使用`-arch=compute_50`（或更高版本）进行编译。

为某些特定计算能力生成的_PTX_代码总是可以编译成计算能力更大或等于的二进制代码。请注意，从早期PTX版本编译的二进制文件可能无法使用一些硬件功能。例如，从为计算能力6.0（Pascal）生成的PTX编译的计算能力7.0（Volta）的二进制目标设备不会使用Tensor Core指令，因为这些指令在Pascal上不可用。因此，最终二进制文件的性能可能比使用最新版本的PTX生成的二进制文件更差。

为目标[架构特定功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-specific-features)而编译的_PTX_代码仅在完全相同的物理架构上运行，而不是在其他地方运行。特定于架构的_PTX_代码不向前和向后兼容。使用`sm_90a`或`compute_90a`编译的示例代码仅在具有计算能力9.0的设备上运行，并且不向后或向前兼容。

为目标[家族特定功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-features)而编译的_PTX_代码仅在完全相同的物理架构和同一家族中的其他架构上运行。特定于家族的_PTX_代码与同一家族中的其他设备向前兼容，并且不向后兼容。使用`sm_100f`或`compute_100f`编译的示例代码仅在具有计算能力10.0和10.3的设备上运行。[表25](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-compatibility)显示了特定于家族的目标与计算能力的兼容性。

### 6.1.4.应用程序兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility "这个标题的永久链接")

要在具有特定计算能力的设备上执行代码，应用程序必须加载与该计算能力兼容的二进制或_PTX_代码，如[二进制兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#binary-compatibility)和[PTX兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ptx-compatibility)中所述。特别是，为了能够在具有更高计算能力的未来架构上执行代码（尚未生成二进制代码），应用程序必须加载_PTX_代码，这些代码将为这些设备及时编译（请参阅[及时编译](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#just-in-time-compilation)）。

嵌入CUDA C++应用程序中的_PTX_和二进制代码由`-arch`和`-code`编译器选项或`-gencode`编译器选项控制，如`nvcc`用户手册中所述。例如，

nvcc x.cu
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_70,code=\"compute_70,sm_70\"

嵌入与计算能力5.0和6.0（第一和第二`-gencode`选项）兼容的二进制代码以及与计算能力7.0（第三`-gencode`选项）兼容的_PTX_和二进制代码。

生成主机代码是为了在运行时自动选择最合适的代码来加载和执行，在上述示例中，这将是：

- 具有计算能力5.0和5.2的设备的5.0二进制代码，
    
- 具有计算能力6.0和6.1的设备的6.0二进制代码，
    
- 具有7.0和7.5计算能力的设备的7.0二进制代码，
    
- 对于计算能力低于7.5的设备，在运行时编译为二进制代码的_PTX_代码
    

`x.cu`可以有一个优化的代码路径，使用扭曲减少操作，例如，这些操作仅在计算能力8.0及更高版本的设备中支持。`__CUDA_ARCH__`宏可用于根据计算能力区分各种代码路径。它仅针对设备代码进行定义。例如，使用`-arch=compute_80`编译时，`__CUDA_ARCH__`等于`800`。

如果`x.cu`是使用`sm_100f`或`compute_100f`为[特定](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-features)于[家族的功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-features)编译的，则代码只能在该特定家族的设备上运行，这些设备具有计算能力10.0和10.3。对于特定于家族的代码目标，定义了一个额外的宏`__CUDA_ARCH_FAMILY_SPECIFIC__`。在本例中，`__CUDA_ARCH_FAMILY_SPECIFIC__`等于`1000`。

如果`x.cu`使用`sm_100a`或`compute_100a`为[特定](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-specific-features)于[架构的功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-specific-features)编译，则代码只能在具有计算能力10.0的设备上运行。对于特定于架构的代码目标，定义了一个额外的宏`__CUDA_ARCH_SPECIFIC__`。在本例中，`__CUDA_ARCH_SPECIFIC__`等于`1000`。由于特定于架构的特征是特定于家族的特征的超集，因此也定义了特定于家族的macro`__CUDA_ARCH_FAMILY_SPECIFIC__`，并且等于`1000`。

使用驱动程序API的应用程序必须编译代码以分离文件，并在运行时显式加载和执行最合适的文件。

The Volta architecture introduces _Independent Thread Scheduling_ which changes the way threads are scheduled on the GPU. For code relying on specific behavior of [SIMT scheduling](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture) in previous architectures, Independent Thread Scheduling may alter the set of participating threads, leading to incorrect results. To aid migration while implementing the corrective actions detailed in [Independent Thread Scheduling](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#independent-thread-scheduling-7-x), Volta developers can opt-in to Pascal’s thread scheduling with the compiler option combination `-arch=compute_60 -code=sm_70`.

The `nvcc` user manual lists various shorthands for the `-arch`, `-code`, and `-gencode` compiler options. For example, `-arch=sm_70` is a shorthand for `-arch=compute_70 -code=compute_70,sm_70` (which is the same as `-gencode arch=compute_70,code=\"compute_70,sm_70\"`).

### 6.1.5.C++兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-compatibility "这个标题的永久链接")

编译器的前端根据C++语法规则处理CUDA源文件。主机代码支持完整的C++。然而，如[C++语言支持](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-cplusplus-language-support)中所述，只有C++的子集完全支持设备代码。

### 6.1.6.64位兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#bit-compatibility "这个标题的永久链接")

64位版本的`nvcc`以64位模式编译设备代码（即指针为64位）。仅支持以64位模式编译的主机代码以64位模式编译的设备代码。

## 6.2.CUDA运行时[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-runtime "这个标题的永久链接")

运行时在`cudart`中实现，该库链接到应用程序，可以通过`cudart.lib`或`libcudart.a`静态，或通过`cudart.dll`或`libcudart.so`动态化。需要`cudart.dll`和/或`cudart.so`进行动态链接的应用程序通常将它们作为应用程序安装包的一部分。只有在链接到CUDA运行时同一实例的组件之间传递CUDA运行时符号的地址才安全。

它的所有入口点都以`cuda`前缀。

正如[异构编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming)中提到的，CUDA编程模型假设一个由主机和设备组成的系统，每个系统都有自己的独立内存。[裝置記憶體](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory)概述了用於管理裝置記憶體的執行時功能。

[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory)说明了[线程层次结构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)中介绍的共享内存的使用，以最大限度地提高性能。

[页面锁定主机内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#page-locked-host-memory)引入了页面锁定主机内存，该内存需要将内核执行与主机和设备内存之间的数据传输重叠。

[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)描述了用于在系统中各个级别实现异步并发执行的概念和API。

[多设备系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-device-system)展示了编程模型如何扩展到多个设备连接到同一主机的系统。

[错误检查](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#error-checking)描述如何正确检查运行时产生的错误。

[呼叫堆栈](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#call-stack)提到了用于管理CUDA C++调用堆栈的运行时函数。

[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)展示了纹理和表面内存空间，为访问设备内存提供了另一种方式；它们还公开了GPU纹理硬件的子集。

[图形互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graphics-interoperability)引入了运行时提供的各种功能，以便与OpenGL和Direct3D这两个主要图形API互操作。

### 6.2.1.初始化[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#initialization "这个标题的永久链接")

从CUDA 12.0开始，`cudaInitDevice()`和`cudaSetDevice()`调用初始化运行时和与指定设备关联的主要上下文。在没有这些调用的情况下，运行时将隐式使用设备0，并根据需要自行初始化以处理其他运行时API请求。在计时运行时函数调用时以及解释从第一次调用到运行时的错误代码时，需要牢记这一点。在12.0之前，`cudaSetDevice()`不会初始化运行时，应用程序通常会使用无操作运行时调用`cudaFree(0)`将运行时初始化与其他api活动隔离（为了定时和错误处理）。

运行时为系统中的每个设备创建CUDA上下文（有关CUDA上下文的更多详细信息，请参阅[上下文](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#context)）。此上下文是此设备_的主要上下文_，并在第一个运行时函数中初始化，该函数需要此设备上的活动上下文。它在应用程序的所有主机线程之间共享。作为此上下文创建的一部分，必要时会及时编译设备代码（请参阅[及时编译](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#just-in-time-compilation)），并加载到设备内存中。这一切都是透明的。例如，如果需要，对于驱动程序API互操作性，可以从驱动程序API访问设备的主要上下文，如[运行时和驱动程序API之间的互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interoperability-between-runtime-and-driver-apis)所述。

当主机线程调用`cudaDeviceReset()`这会破坏主机线程当前运行的设备的主要上下文（即[设备选择](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-selection)中定义的当前设备）。任何将此设备设为当前的主机线程所做的下一次运行时函数调用将为该设备创建新的主上下文。

笔记

CUDA接口使用在主机程序启动期间初始化并在主机程序终止期间销毁的全局状态。CUDA运行时和驱动程序无法检测此状态是否无效，因此在程序启动或主程序终止期间（隐式或显式）使用这些接口中的任何一切都会导致未定义的行为。

从CUDA 12.0开始，`cudaSetDevice()`现在将在更改主机线程的当前设备后显式初始化运行时。以前版本的CUDA延迟了新设备上的运行时初始化，直到在`cudaSetDevice()`之后进行第一次运行时调用。这一变化意味着现在检查`cudaSetDevice()`的返回值是否存在初始化错误非常重要。

参考手册错误处理和版本管理部分的运行时函数不会初始化运行时。

### 6.2.2.设备内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory "这个标题的永久链接")

正如[异构编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming)中提到的，CUDA编程模型假设一个由主机和设备组成的系统，每个系统都有自己的独立内存。内核在设备内存中运行，因此运行时提供了分配、去分配和复制设备内存的功能，以及在主机内存和设备内存之间传输数据。

设备内存可以作为_线性内存_或_CUDA阵列_分配。

CUDA数组是为纹理获取而优化的不透明内存布局。它们在[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)中进行了描述。

线性内存被分配到单个统一地址空间中，这意味着单独分配的实体可以通过指针相互引用，例如在二叉树或链接列表中。地址空间的大小取决于主机系统（CPU）和所用GPU的计算能力：

表4线性内存地址空间[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id455 "此表的永久链接")
||x86_64（AMD64）|电源（ppc64le）|手臂64|
|---|---|---|---|
|高达计算能力5.3（Maxwell）|40位|40位|40位|
|计算能力6.0（Pascal）或更新|高达47位|高达49位|高达48位|

笔记

在具有计算能力5.3（Maxwell）及更早版本的设备上，CUDA驱动程序创建了一个未提交的40位虚拟地址保留，以确保内存分配（指标）属于支持的范围。此保留显示为保留的虚拟内存，但在程序实际分配内存之前不会占用任何物理内存。

线性内存通常使用`cudaMalloc()`分配，并使用`cudaFree()`释放，主机内存和设备内存之间的数据传输通常使用`cudaMemcpy()`完成。在[内核](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernels)的向量加法代码示例中，需要将向量从主机内存复制到设备内存：

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    ...
}

线性内存也可以通过`cudaMallocPitch()`和`cudaMalloc3D()`分配。建议使用这些功能进行2D或3D阵列的分配，因为它确保分配得到适当的填充，以满足[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)中描述的对齐要求，因此确保在访问行地址或在2D数组和设备内存的其他区域之间执行副本时（使用`cudaMemcpy2D()`和`cudaMemcpy3D()`函数）时获得最佳性能。返回的音高（或步长）必须用于访问数组元素。以下代码示例分配了一个`width`x`height`的浮点值的2D数组，并展示了如何在设备代码中循环数组元素：

// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}

以下代码示例分配了浮点值的`width`x`height`x`depth`3D数组，并展示了如何在设备代码中循环数组元素：

// Host code
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                    height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}

笔记

为了避免分配过多的内存从而影响整个系统的性能，请根据问题大小向用户请求分配参数。如果分配失败，您可以回退到其他较慢的内存类型（`cudaMallocHost()``cudaHostRegister()`等），或返回一个错误，告诉用户需要多少内存被拒绝。如果您的应用程序因某种原因无法请求分配参数，我们建议将`cudaMallocManaged()`用于支持它的平台。

参考手册列出了用于在`cudaMalloc()`分配的线性内存、使用`cudaMallocPitch()`或`cudaMalloc3D()`分配的线性内存、CUDA数组以及为全局或常量内存空间中声明的变量分配的内存之间复制内存的所有各种功能。

以下代码示例说明了通过运行时API访问全局变量的各种方法：

__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr, 256 * sizeof(float));
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));

`cudaGetSymbolAddress()`用于检索指向分配给全局内存空间中声明的变量的内存的地址。分配的内存大小是通过`cudaGetSymbolSize()`获得的。

### 6.2.3.设备内存L2访问管理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-l2-access-management "这个标题的永久链接")

当CUDA内核反复访问全局内存中的数据区域时，此类数据访问可以被视为_持久_。另一方面，如果数据只访问一次，则此类数据访问可以被视为_流式传输_。

从CUDA 11.0开始，计算能力8.0及以上的设备能够影响L2缓存中数据的持久性，有可能为全局内存提供更高的带宽和更低的延迟访问。

#### 6.2.3.1.L2缓存为持久访问而留置[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-cache-set-aside-for-persisting-accesses "这个标题的永久链接")

L2缓存的一部分可以预留，用于对全局内存的持久数据访问。持久訪問優先使用L2快取的保留部分，而正常或流式，對全域性記憶體的訪問只能在持久訪問未使用L2的這一部分時使用它。

持久访问的L2缓存保留大小可以在限制范围内进行调整：

cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/

当GPU配置为多Instance GPU（MIG）模式时，L2缓存设置功能将被禁用。

使用多进程服务（MPS）时，`cudaDeviceSetLimit`无法更改L2缓存设置大小。相反，仅在MPS服务器启动时，可以通过环境变量`CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`指定留置大小。

#### 6.2.3.2.持续访问的L2政策[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-policy-for-persisting-accesses "这个标题的永久链接")

访问策略窗口指定了全局内存的连续区域和L2缓存中的持久性属性，用于该区域内的访问。

下面的代码示例展示了如何使用CUDA流设置L2持久访问窗口。

**CUDA流示例**

cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

当内核随后在CUDA`stream`执行时，全局内存范围`[ptr..ptr+num_bytes)`内的内存访问比访问其他全局内存位置更有可能保留在L2缓存中。

L2持久性也可以为CUDA图形内核节点设置，如下例所示：

**CUDA GraphKernelNode示例**

cudaKernelNodeAttrValue node_attribute;                                     // Kernel level attributes data structure
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
node_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
node_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);

`hitRatio`参数可用于指定接收`hitProp`属性的访问分数。在上述两个示例中，全局内存区域`[ptr..ptr+num_bytes)`中60%的内存访问具有持久属性，40%的内存访问具有流属性。哪些特定的内存访问被归类为持久（`hitProp`）是随机的，概率约为`hitRatio`；概率分布取决于硬件架构和内存范围。

例如，如果L2设置缓存大小为16KB，`accessPolicyWindow`中的`num_bytes`为32KB：

- 当命`hitRatio`为0.5时，硬件将随机选择32KB窗口中的16KB，以指定为持久并缓存在保留的L2缓存区域中。
    
- 当命`hitRatio`为1.0时，硬件将尝试在留置的L2缓存区域缓存整个32KB窗口。由于留置区域比窗口小，缓存行将被驱逐，以将最近使用的16KB的32KB数据保留在L2缓存的留置部分。
    

因此，可以使用`hitRatio`来避免缓存行的崩溃，并总体减少进出L2缓存的数据量。

低于1.0的`hitRatio`可用于手动控制来自并发CUDA流的不同`accessPolicyWindow`可以在L2中缓存的数据量。例如，让L2设置的缓存大小为16KB；两个不同CUDA流中的两个并发内核，每个内核都有16KBaccessPolicyWindow，并且两个核心的`hitRatio`为1.0，在争夺共享L2资源时可能会驱逐彼此的缓存行。然而，如果两个`accessPolicyWindows`的hitRatio值为0.5，它们就不太可能驱逐自己或彼此的持久缓存行。

#### 6.2.3.3.L2访问属性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-access-properties "这个标题的永久链接")

为不同的全局内存数据访问定义了三种类型的访问属性：

1. `cudaAccessPropertyStreaming`：流属性发生的内存访问不太可能在L2缓存中持续存在，因为这些访问被优先驱逐。
    
2. `cudaAccessPropertyPersisting`：与持久属性一起发生的内存访问更有可能在L2缓存中持续存在，因为这些访问优先保留在L2缓存的保留部分。
    
3. `cudaAccessPropertyNormal`：此访问属性将之前应用的持久访问属性强制重置为正常状态。具有先前CUDA内核持久属性的内存访问可能在预期使用后很长时间保留在L2缓存中。这种使用后持久性减少了不使用持久属性的后续内核可用的L2缓存量。使用`cudaAccessPropertyNormal`属性重置访问属性窗口会删除先前访问的持久（首选保留）状态，就好像之前的访问没有访问属性一样。
    

#### 6.2.3.4.L2 持久性示例[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-persistence-example "这个标题的永久链接")

以下示例展示了如何为持久访问设置L2缓存，通过CUDA流在CUDA内核中使用设置L2缓存，然后重置L2缓存。

cudaStream_t stream;
cudaStreamCreate(&stream);                                                                  // Create CUDA stream

cudaDeviceProp prop;                                                                        // CUDA device properties variable
cudaGetDeviceProperties( &prop, device_id);                                                 // Query GPU properties
size_t size = min( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size);                                  // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);                        // Select minimum of user defined num_bytes and max window size.

cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);               // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                                        // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // Persistence Property
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // Type of access property on cache miss

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Set the attributes to a CUDA Stream

for(int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);                                 // This data1 is used by a kernel multiple times
}                                                                                           // [data1 + num_bytes) benefits from L2 persistence
cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);                                     // A different kernel in the same stream can also benefit
                                                                                            // from the persistence of data1

stream_attribute.accessPolicyWindow.num_bytes = 0;                                          // Setting the window size to 0 disable it
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);   // Overwrite the access policy attribute to a CUDA Stream
cudaCtxResetPersistingL2Cache();                                                            // Remove any persistent lines in L2

cuda_kernelC<<<grid_size,block_size,0,stream>>>(data2);                                     // data2 can now benefit from full L2 in normal mode

#### 6.2.3.5.将 L2 访问权限重置为正常[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#reset-l2-access-to-normal "这个标题的永久链接")

來自之前CUDA核心的持久L2快取行可能會在使用L2中持續很長時間。因此，对于流式传输或正常内存访问以正常优先级使用L2缓存来说，将L2缓存重置为正常状态很重要。有三种方法可以将持久访问重置为正常状态。

1. 使用访问属性`cudaAccessPropertyNormal`重置之前的持久内存区域。
    
2. 通过调用`cudaCtxResetPersistingL2Cache()`将所有持久的L2缓存行重置为正常。
    
3. **最终**，未触及的线路将自动重置为正常。强烈建议不要依赖自动重置，因为发生自动重置所需的时间长度不确定。
    

#### 6.2.3.6.管理L2留置缓存的利用率[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#manage-utilization-of-l2-set-aside-cache "这个标题的永久链接")

在不同CUDA流中同时执行的多个CUDA内核可能为其流分配了不同的访问策略窗口。然而，L2留置缓存部分在所有这些并发CUDA内核之间共享。因此，这个留置缓存部分的净利用率是所有并发内核个人使用的总和。将内存访问指定为持久性，随着持久访问量超过保留的L2缓存容量，其好处就会减弱。

要管理留置L2缓存部分的利用，应用程序必须考虑以下几点：

- L2留置缓存的大小。
    
- 可能并发执行的CUDA内核。
    
- 可能同时执行的所有CUDA内核的访问策略窗口。
    
- 何时以及如何重置L2，以允许正常或流式访问以同等优先级使用之前设置的L2缓存。
    

#### 6.2.3.7.查询L2缓存属性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#query-l2-cache-properties "这个标题的永久链接")

与L2缓存相关的属性是`cudaDeviceProp`结构的一部分，可以使用CUDA运行时API进行查询`cudaGetDeviceProperties`

CUDA设备属性包括：

- `l2CacheSize`：GPU上可用的L2缓存量。
    
- `persistingL2CacheMaxSize`：可以为持久内存访问保留的最大L2缓存量。
    
- `accessPolicyMaxWindowSize`：访问策略窗口的最大大小。
    

#### 6.2.3.8.控制L2缓存设置的大小，以持续内存访问[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#control-l2-cache-set-aside-size-for-persisting-memory-access "这个标题的永久链接")

使用CUDA运行时API `cudaDeviceGetLimit`查询持久内存访问的L2保留缓存大小，并使用CUDA运行时API `cudaDeviceSetLimit`作为`cudaLimit`设置。设置此限制的最大值是`cudaDeviceProp::persistingL2CacheMaxSize`。

enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
};

### 6.2.4.共享内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory "这个标题的永久链接")

如[可变内存空间指定符中](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#variable-memory-space-specifiers)所述，共享内存使用`__shared__`内存空间指定符进行分配。

共享内存预计将比[线程层次结构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)中提到的和[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory)中详述的全域性内存快得多。它可以用作刮板内存（或软件管理缓存），以尽量减少来自CUDA块的全局内存访问，如以下矩阵乘法示例所示。

以下代码示例是矩阵乘法的直接实现，不利用共享内存。每个线程读取一行A和一列B，并计算相应的_C元素_，如[图8](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-no-shared-memory)所示。因此，_A_从全局内存中读取B_.宽度_时间，_B_读取_A.高度_时间。

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

![_图像/矩阵-乘法-无-共享-内存.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-without-shared-memory.png)

图8 没有共享内存的矩阵乘法[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-no-shared-memory "此图像的永久链接")

以下代码示例是矩阵乘法的实现，它确实利用了共享内存。在此实现中，每个线程块负责计算_C_的一个平方子矩阵_Csub_，块中的每个线程负责计算_Csub_的一个元素。如[图9](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-shared-memory)所示，_Csub_等于两个矩形矩阵的乘积：维度_A_的子矩阵（_A.width，block_size_）的行指数与_Csub_相同，维度_B_的子矩阵（_block_size，A.width_）的子矩阵与_Csub_具有相同的列索引。为了适应设备的资源，这两个矩形矩阵被分成尽可能多的维度_块_大小的_平方矩阵，_Csub_被计算为这些平方矩阵的乘积之和。这些产品中的每一个都是通过先将两个相应的平方矩阵从全局内存加载到共享内存来执行的，一个线程加载每个矩阵的一个元素，然后让每个线程计算一个产品的元素。每个线程将每个产品的结果积累到一个寄存器中，完成后将结果写入全局内存。

通过这种方式阻止计算，我们利用了快速共享内存，并节省了大量全局内存带宽，因为_A_只能从全局内存读取（_B.width / block_size_）次，_B_被读取（_A.height / block_size_）次。

上一个代码示例中的_矩阵_类型用_步长_字段增强，因此子矩阵可以用相同的类型有效地表示。[__device__](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-function-specifier)函数用于获取和设置元素，并从矩阵中构建任何子矩阵。

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;
// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}
// Thread block size
#define BLOCK_SIZE 16
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

![_图像/矩阵-乘法-共享-记忆.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-with-shared-memory.png)

图9 共享内存的矩阵乘法[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-shared-memory "此图像的永久链接")

### 6.2.5.分布式共享内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#distributed-shared-memory "这个标题的永久链接")

计算功能9.0中引入的线程块集群为线程块集群中的线程提供了访问集群中所有参与线程块的共享内存的能力。这种分区共享内存称为_分布式共享内存_，相应的地址空间称为分布式共享内存地址空间。属于线程块集群的线程可以在分布式地址空间中读取、写入或执行原子，无论该地址是属于本地线程块还是远程线程块。无论内核是否使用分布式共享内存，共享内存大小规格，静态或动态，仍然是每个线程块。分布式共享内存的大小只是每个集群的线程块数乘以每个线程块的共享内存大小。

访问分布式共享内存中的数据需要所有线程块的存在。用户可以保证所有线程块都已开始使用[集群组](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-group-cg)API中的`cluster.sync()`执行。用户还需要确保所有分布式共享内存操作都在线程块退出之前发生，例如，如果远程线程块正在尝试读取给定线程块的共享内存，用户需要确保在退出之前完成远程线程块读取的共享内存。

CUDA提供了一个访问分布式共享内存的机制，应用程序可以从利用其功能中受益。让我们来看看一个简单的直方图计算，以及如何使用线程块集群在GPU上对其进行优化。计算直方图的标准方法是在每个线程块的共享内存中进行计算，然后执行全局内存原子。这种方法的一个限制是共享内存容量。一旦直方图仓不再适合共享内存，用户就需要直接计算直方图，从而计算全局内存中的原子。借助分布式共享内存，CUDA提供了一个中间步骤，其中根据直方图仓的大小，直方图可以直接在共享内存、分布式共享内存或全局内存中计算。

下面的CUDA内核示例展示了如何在共享内存或分布式共享内存中计算直方图，具体取决于直方图仓的数量。

#include <cooperative_groups.h>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
                                   size_t array_size)
{
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; //Initialize shared memory histogram to zeros
  }

  // cluster synchronization ensures that shared memory is initialized to zero in
  // all thread blocks in the cluster. It also ensures that all thread blocks
  // have started executing and they exist concurrently.
  cluster.sync();

  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int ldata = input[i];

    //Find the right histogram bin.
    int binid = ldata;
    if (ldata < 0)
      binid = 0;
    else if (ldata >= nbins)
      binid = nbins - 1;

    //Find destination block rank and offset for computing
    //distributed shared memory histogram
    int dst_block_rank = (int)(binid / bins_per_block);
    int dst_offset = binid % bins_per_block;

    //Pointer to target block shared memory
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    //Perform atomic update of the histogram bin
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // cluster synchronization is required to ensure all distributed shared
  // memory operations are completed and no thread block exits while
  // other thread blocks are still accessing distributed shared memory
  cluster.sync();

  // Perform global memory histogram, using the local distributed memory histogram
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}

上述内核可以在运行时启动，集群大小取决于所需的分布式共享内存量。如果直方图足够小，仅适合一个块的共享内存，用户可以启动集群大小为1的内核。下面的代码片段展示了如何根据共享内存要求动态启动集群内核。

// Launch via extensible launch
{
  cudaLaunchConfig_t config = {0};
  config.gridDim = array_size / threads_per_block;
  config.blockDim = threads_per_block;

  // cluster_size depends on the histogram size.
  // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
  int cluster_size = 2; // size 2 is an example here
  int nbins_per_block = nbins / cluster_size;

  //dynamic shared memory size is per block.
  //Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
  config.dynamicSmemBytes = nbins_per_block * sizeof(int);

  CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
}

### 6.2.6.页面锁定主机内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#page-locked-host-memory "这个标题的永久链接")

运行时提供的功能允许使用_页面锁定_（也称为_固定_）主机内存（而不是`malloc()`分配的常规分页主机内存）：

- `cudaHostAlloc()`和`cudaFreeHost()`分配和释放页面锁定主机内存；
    
- `cudaHostRegister()`页面锁定由`malloc()`分配的内存范围（有关限制，请参阅参考手册）。
    

使用页面锁定的主机内存有几个好处：

- 如非[同步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)中所述，页面锁定主机内存和设备内存之间的副本可以与某些设备的内核执行同时执行。
    
- 在某些设备上，页面锁定的主机内存可以映射到设备的地址空间中，无需将其复制到或从设备内存中复制，如[“映射内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory)”中所述。
    
- 在具有前端总线的系统上，如果主机内存被分配为页面锁定，则主机内存和设备内存之间的带宽会更高，如果如[写入组合内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#write-combining-memory)中所述，则分配为写入组合，则更高。
    

笔记

页面锁定的主机内存不会缓存在非I/O连贯的Tegra设备上。此外，非I/O连贯Tegra设备不支持`cudaHostRegister()`）。

简单的零拷贝CUDA示例附带了关于页面锁定内存API的详细文档。

#### 6.2.6.1.便携式内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#portable-memory "这个标题的永久链接")

页面锁定内存块可以与系统中的任何设备一起使用（有关多设备系统的更多详细信息，请参阅[多设备系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-device-system)），但默认情况下，使用上述页面锁定内存的好处只能与分配块时当前的设备一起使用（以及所有设备共享相同的统一地址空间（如果有的话），如[统一虚拟地址空间](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space)中所述）。为了使所有设备都能获得这些优势，需要通过将标志`cudaHostAllocPortable`传递给`cudaHostAlloc()`来分配块，或者通过将标志`cudaHostRegisterPortable`传递给`cudaHostRegister()`分配页面锁定。

#### 6.2.6.2.写入组合内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#write-combining-memory "这个标题的永久链接")

默认情况下，页面锁定的主机内存被分配为可缓存。通过将flagcudaHostAllocWriteCombined传递给`cudaHostAlloc()`，可以选择将其分配为_写入组合。_写入组合内存释放了主机的L1和L2缓存资源，使应用程序的其余部分可以使用更多缓存。此外，在PCI Express总线传输期间，写入组合内存不会被窥探，这可以将传输性能提高高达40%。

从主机的写入组合内存中读取速度慢得令人望而却步，因此写入组合内存通常应用于主机仅写入的内存。

应避免在WC内存上使用CPU原子指令，因为并非所有CPU实现都能保证该功能。

#### 6.2.6.3.映射的内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory "这个标题的永久链接")

A block of page-locked host memory can also be mapped into the address space of the device by passing flag `cudaHostAllocMapped` to `cudaHostAlloc()` or by passing flag `cudaHostRegisterMapped` to `cudaHostRegister()`. Such a block has therefore in general two addresses: one in host memory that is returned by `cudaHostAlloc()` or `malloc()`, and one in device memory that can be retrieved using `cudaHostGetDevicePointer()` and then used to access the block from within a kernel. The only exception is for pointers allocated with `cudaHostAlloc()` and when a unified address space is used for the host and the device as mentioned in [Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space).

直接从内核内部访问主机内存不会提供与设备内存相同的带宽，但确实有一些优点：

- 无需在设备内存中分配一个块，并在该块和主机内存中的块之间复制数据；数据传输根据内核的需要隐式执行；
    
- 无需使用流（请参阅[并发数据传输](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-data-transfers)）将数据传输与内核执行重叠；内核源的数据传输自动与内核执行重叠。
    

然而，由于映射的页面锁定内存在主机和设备之间共享，应用程序必须使用流或事件（请参阅[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)）同步内存访问，以避免任何潜在的写后读、读后写或写入后写危险。

To be able to retrieve the device pointer to any mapped page-locked memory, page-locked memory mapping must be enabled by calling `cudaSetDeviceFlags()` with the `cudaDeviceMapHost` flag before any other CUDA call is performed. Otherwise, `cudaHostGetDevicePointer()` will return an error.

`cudaHostGetDevicePointer()`如果设备不支持映射的页面锁定主机内存，也会返回错误。应用程序可以通过检查`canMapHostMemory`设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)）来查询此功能，对于支持映射页面锁定主机内存的设备，该属性等于1。

请注意，从主机或其他设备的角度来看，在映射页面锁定内存上运行的原子函数（见[原子函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)）不是原子函数。

另请注意，CUDA运行时要求从主机和其他设备的角度来看，从主机和其他设备的角度，将1字节、2字节、4字节、8字节和16字节自然对齐的负载和存储作为单一访问保留到主机内存。在某些平台上，原子到内存可能会被硬件分解为单独的加载和存储操作。这些组件加载和存储操作对维护自然对齐的访问有相同的要求。CUDA运行时不支持PCI Express总线拓扑结构，其中PCI Express桥接器分割8字节自然对齐操作，NVIDIA不知道任何分割16字节自然对齐操作的拓扑结构。

### 6.2.7.内存同步域[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-synchronization-domains "这个标题的永久链接")

#### 6.2.7.1.记忆围栏干扰[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-interference "这个标题的永久链接")

由于内存围栏/刷新操作等待比CUDA内存一致性模型需要的更多事务，一些CUDA应用程序可能会看到性能下降。

|   |   |   |
|---|---|---|
|__managed__ int x = 0;<br>__device__  cuda::atomic<int, cuda::thread_scope_device> a(0);<br>__managed__ cuda::atomic<int, cuda::thread_scope_system> b(0);|||
|螺纹1（SM）<br><br>x = 1;<br>a = 1;|线程2（SM）<br><br>while (a != 1) ;<br>assert(x == 1);<br>b = 1;|线程3（中央处理器）<br><br>while (b != 1) ;<br>assert(x == 1);|

考虑上面的例子。CUDA内存一致性模型保证断言的条件为真，因此在从线程2写入b之前，从线程1写入`x`必须对线程3可见。

释放和获取`a`提供的内存排序仅足以使`x`对线程2可见，而不是线程3，因为它是一种设备范围操作。因此，`b`的发布和获取提供的系统范围排序不仅需要确保线程2本身发出的写入对线程3可见，还需要确保线程2可见的其他线程的写入。这被称为累积性。由于GPU在执行时无法知道哪些写入在源级别保证是可见的，哪些是偶然的，因此它必须为飞行内存操作投下保守的宽网。

这有时会导致干扰：由于GPU正在等待内存操作，它不需要在源级别，围栏/刷新可能需要比必要的更长的时间。

请注意，栅栏可能作为代码中的内在或原子明确出现，如示例中所示，或隐含地实现_与_任务边界_的同步_关系。

一个常见的例子是，当内核在本地GPU内存中执行计算时，并行内核（例如来自NCCL）正在与对等进行通信。完成后，本地内核将隐式刷新其写入，以满足与下游工作的任何_同步_关系。这可能会在通信内核的慢速nvlink或PCIe写入上完全或部分不必要地等待。

#### 6.2.7.2.用域隔离流量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isolating-traffic-with-domains "这个标题的永久链接")

从Hopper架构GPU和CUDA 12.0开始，内存同步域功能提供了一种缓解此类干扰的方法。为了换取代码的明确帮助，GPU可以通过围栏操作来减少净投射。每次内核启动都会给出一个域ID。写入和栅栏用ID标记，栅栏只会订购与栅栏域匹配的写入。在并发计算与通信的例子中，通信内核可以放置在不同的域中。

使用域时，代码必须遵守**同一GPU上不同域之间的排序或同步需要系统范围围栏**的规则。在一个领域内，设备范围围栏仍然足够。这对于累积性是必要的，因为一个内核的写入不会被另一个域内核发出的栅栏所包围。从本质上讲，通过确保跨域流量提前冲入系统范围来满足累积性。

请注意，这修改了`thread_scope_device`的定义。然而，由于内核将默认为域0，如下所述，因此保持向后兼容。

#### 6.2.7.3.在CUDA中使用域[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-domains-in-cuda "这个标题的永久链接")

可以通过新的启动属性`cudaLaunchAttributeMemSyncDomain`和`cudaLaunchAttributeMemSyncDomainMap`访问域。前者在逻辑域`cudaLaunchMemSyncDomainDefault`和`cudaLaunchMemSyncDomainRemote`之间进行选择，后者提供了从逻辑域到物理域的映射。远程域用于执行远程内存访问的内核，以便将其内存流量与本地内核隔离。然而，请注意，特定域的选择并不影响内核可以合法执行的内存访问。

可以通过设备属性`cudaDevAttrMemSyncDomainCount`查询域计数。Hopper有4个领域。为了方便便携式代码，域功能可以在所有设备上使用，CUDA将在Hopper之前报告1的计数。

拥有逻辑域可以简化应用程序的组成。在堆栈中低级别的单个核心启动，例如NCCL，可以选择语义逻辑域，而无需考虑周围的应用程序架构。更高级别可以使用映射来引导逻辑域。逻辑域的默认值（如果未设置）为默认域，默认映射是将默认域映射到0，远程域映射到1（在具有1个以上域的GPU上）。特定库可能会在CUDA 12.0及更高版本中用远程域标记启动；例如，NCCL 2.16将这样做。总体而言，这为开箱即用的常见应用程序提供了有益的使用模式，无需在其他组件、框架或应用程序级别更改代码。另一种使用模式，例如在使用nvshmem或没有明确分离内核类型的应用程序中，可能是分区并行流。流A可以将两个逻辑域映射到物理域0，流B映射到1，以此以此为。

// Example of launching a kernel with the remote logical domain
cudaLaunchAttribute domainAttr;
domainAttr.id = cudaLaunchAttrMemSyncDomain;
domainAttr.val = cudaLaunchMemSyncDomainRemote;
cudaLaunchConfig_t config;
// Fill out other config fields
config.attrs = &domainAttr;
config.numAttrs = 1;
cudaLaunchKernelEx(&config, myKernel, kernelArg1, kernelArg2...);

// Example of setting a mapping for a stream
// (This mapping is the default for streams starting on Hopper if not
// explicitly set, and provided for illustration)
cudaLaunchAttributeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(stream, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);

// Example of mapping different streams to different physical domains, ignoring
// logical domain settings
cudaLaunchAttributeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 0;
cudaStreamSetAttribute(streamA, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);
mapAttr.memSyncDomainMap.default_ = 1;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(streamB, cudaLaunchAttributeMemSyncDomainMap, &mapAttr);

与其他启动属性一样，这些属性在CUDA流、使用`cudaLaunchKernelEx`的单个启动和CUDA图中的内核节点上均匀地暴露。如上所述，典型的用法是在流级别设置映射，在启动级别设置逻辑域（或括号为流使用的一部分）。

在流捕获期间，这两个属性都被复制到图形节点。图形从节点本身中取两个属性，本质上是指定物理域的间接方式。在图形启动的流上设置的域相关属性不用于图形的执行。

### 6.2.8.异步并发执行[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution "这个标题的永久链接")

CUDA将以下操作作为独立任务公开，可以相互并行操作：

- 在主机上进行计算；
    
- 设备上的计算；
    
- 内存从主机传输到设备；
    
- 内存从设备传输到主机；
    
- 给定设备内存中的内存传输；
    
- 设备之间的内存传输。
    

这些操作之间实现的并发级别将取决于设备的功能集和计算能力，如下所述。

#### 6.2.8.1.主机和设备之间的并发执行[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-execution-between-host-and-device "这个标题的永久链接")

通过异步库函数促进并发主机执行，这些库函数在设备完成请求的任务之前将控制权返回给主机线程。使用异步调用，当有适当的设备资源可用时，CUDA驱动程序可以将许多设备操作排队在一起执行。这免除了主机线程管理设备的大部分责任，使其可以自由地执行其他任务。以下设备操作与主机是异步的：

- 内核发射；
    
- 单个设备内存中的内存副本；
    
- 64 KB或更小的内存块从主机复制到设备；
    
- 由带有`Async`后缀的函数执行的内存副本；
    
- 内存集功能调用。
    

程序员可以通过将`CUDA_LAUNCH_BLOCKING`环境变量设置为1，在全球范围内禁用在系统上运行的所有CUDA应用程序的内核启动的异步性。此功能仅用于调试目的，不应用作使生产软件可靠运行的一种方式。

如果通过分析器（Nsight Compute）收集硬件计数器，除非启用并发内核分析，否则内核启动是同步的。如果涉及非页面锁定的主机内存，则`Async`内存副本也可能是同步的。

#### 6.2.8.2.并发内核执行[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-kernel-execution "这个标题的永久链接")

一些具有2.x及更高计算能力的设备可以同时执行多个内核。应用程序可以通过检查concurrentKernels设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)）来查询此功能，对于支持它的设备，该属性等于1。

设备可以同时执行的最大内核启动次数取决于其计算能力，并列在[表27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability)中。

来自一个CUDA上下文的内核不能与来自另一个CUDA上下文的内核同时执行。GPU可以进行时间切片，为每个上下文提供前进进度。如果使用者想在SM上同時從多個程序中執行核心，則必須啟用MPS。

使用大量纹理或大量本地内存的内核不太可能与其他内核同时执行。

#### 6.2.8.3.数据传输和内核执行的重叠[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlap-of-data-transfer-and-kernel-execution "这个标题的永久链接")

一些设备可以在内核执行的同时执行到或从GPU执行异步内存复制。应用程序可以通过检查`asyncEngineCount`设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)）来查询此功能，对于支持它的设备，该属性大于零。如果主机内存涉及副本，则必须进行页面锁定。

也可以与内核执行（在支持`concurrentKernels`设备属性的设备上）和/或从设备到或从设备（对于支持`asyncEngineCount`属性的设备）同时执行设备内复制。设备内复制使用标准内存复制函数启动，目标地址和源地址位于同一设备上。

#### 6.2.8.4.并发数据传输[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-data-transfers "这个标题的永久链接")

一些具有2.x及更高计算能力的设备可以重叠复制到设备。应用程序可以通过检查`asyncEngineCount`设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)）来查询此功能，对于支持它的设备，该属性等于2。为了重叠，传输中涉及的任何主机内存都必须是页面锁定的。

#### 6.2.8.5.溪流[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams "这个标题的永久链接")

应用程序通过_流_管理上述并发操作。流是按顺序执行的命令序列（可能由不同的主机线程发出）。另一方面，不同的流可能会相互或同时执行命令；这种行为不能保证，因此不应依赖其正确性（例如，内核间通信是未定义的）。当满足命令的所有依赖项时，流上发出的命令可能会执行。依赖项可能是之前在同一流上启动的命令或来自其他流的依赖项。同步调用的成功完成保证了所有启动的命令都已完成。

##### 6.2.8.5.1.溪流的创建和破坏[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-of-streams "这个标题的永久链接")

通过创建流对象并将其指定为内核启动序列和主机`<->`设备内存副本的流参数来定义流。以下代码示例创建了两个流，并在页面锁定内存中分配一个`float`的数组`hostPtr`。

cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);

以下代码示例将这些流中的每一个定义为从主机到设备的内存副本、一个内核启动和一个从设备到主机的内存副本的序列：

for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}

每个流将其输入数组`hostPtr`的一部分复制到设备内存中的数组`inputDevPtr`，通过调用`MyKernel()`处理设备上的`inputDevPtr`并将结果`outputDevPtr`复制回`hostPtr`的同一部分。[重叠行为](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlapping-behavior)描述了本示例中流如何根据设备的功能重叠。请注意，`hostPtr`必须指向页面锁定的主机内存，才能发生任何重叠。

流通过调用`cudaStreamDestroy()`来释放。

for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);

如果调用`cudaStreamDestroy()`时，设备仍在流中工作，该函数将立即返回，一旦设备完成流中的所有工作，与流相关的资源将自动释放。

##### 6.2.8.5.2.默认流[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#default-stream "这个标题的永久链接")

内核启动和主机`<->`设备内存副本不指定任何流参数，或等效将流参数设置为零，发布到默认流。因此，它们是按顺序执行的。

对于使用`--default-streamper-thread`译标志（或在包含CUDA标头（`cuda.h`和`cuda_runtime.h`）之前定义`CUDA_API_PER_THREAD_DEFAULT_STREAM`）编译的代码，默认流是常规流，每个主机线程都有自己的默认流。

笔记

`#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1` cannot be used to enable this behavior when the code is compiled by `nvcc` as `nvcc` implicitly includes `cuda_runtime.h` at the top of the translation unit. In this case the `--default-stream per-thread` compilation flag needs to be used or the `CUDA_API_PER_THREAD_DEFAULT_STREAM` macro needs to be defined with the `-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1` compiler flag.

对于使用`--default-streamlegacy`编译标志编译的代码，默认流是一个称为_NULL流_的特殊流，每个设备都有一个用于所有主机线程的NULL流。NULL流是特殊的，因为它会导致隐式[同步](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#implicit-synchronization)中描述的隐式同步。

对于在没有指定`--default-stream`编译标志的情况下编译的代码，`--default-streamlegacy`假定为默认。

##### 6.2.8.5.3.显式同步[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#explicit-synchronization "这个标题的永久链接")

有多种方法可以明确地将流相互同步。

`cudaDeviceSynchronize()`等待所有主机线程的所有流中的所有前面命令都完成。

`cudaStreamSynchronize()`将流作为参数，并等待给定流中的所有前述命令都完成。它可用于将主机与特定流同步，允许其他流在设备上继续执行。

`cudaStreamWaitEvent()`将流和事件作为参数（请参阅[事件](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events)的描述），并在调用`cudaStreamWaitEvent()`后将添加到给定流中的所有命令延迟执行，直到给定事件完成。

`cudaStreamQuery()`为应用程序提供了一种方法来了解流中所有前面的命令是否已完成。

##### 6.2.8.5.4.隐式同步[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#implicit-synchronization "这个标题的永久链接")

如果中间提交了NULL流上的任何CUDA操作，则来自不同流的两个操作不能同时运行，除非流是非阻塞流（使用`cudaStreamNonBlocking`标志创建）。

应用程序应遵循以下准则，以提高其并发内核执行的潜力：

- 所有独立操作都应在依赖操作之前发布，
    
- 任何类型的同步都应尽可能长时间地延迟。
    

##### 6.2.8.5.5.重叠行为[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlapping-behavior "这个标题的永久链接")

两个流之间的执行重叠量取决于向每个流发出命令的顺序，以及设备是否支持数据传输和内核执行的重叠（请参阅[数据传输和内核执行的重叠](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlap-of-data-transfer-and-kernel-execution)）、并发内核执行（请参阅[并发内核执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-kernel-execution)）和/或并发数据传输（请参阅[并发数据传输](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-data-transfers)）。

例如，在不支持并发数据传输的设备上，[创建和销毁流代](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-streams)码样本的两个流根本不重叠，因为从主机到设备的内存副本在发布到流[0]后，从主机到设备的内存副本被发送到流[1]，因此只有在从设备到主机的内存副本发布到流[0]完成后才能启动。如果代码以以下方式重写（并假设设备支持数据传输和内核执行的重叠）

for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);

然后，从主机到设备发布到流[1]的内存副本与发布到流的内核启动重叠[0]。

在确实支持并发数据传输的设备上，[创建和销毁流代](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-streams)码样本的两个流确实重叠：从主机到设备发布到流[1]的内存副本与从设备到主机发布到流[0]的内存副本重叠，甚至与内核启动发布到流[0]重叠（假设设备支持数据传输和内核执行的重叠）。

##### 6.2.8.5.6.主机函数（回调）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#host-functions-callbacks "这个标题的永久链接")

运行时提供了一种通过`cudaLaunchHostFunc()`在任何一点将CPU函数调用插入流的方法。一旦在回调完成之前向流发出的所有命令，所提供的函数就会在主机上执行。

以下代码示例在向每个流中发布主机到设备内存副本、内核启动和设备到主机内存副本后，将主机函数`MyCallback`添加到两个流中的每个流中。在每个设备到主机内存副本完成后，该函数将开始在主机上执行。

void CUDART_CB MyCallback(void *data){
    printf("Inside callback %d\n", (size_t)data);
}
...
for (size_t i = 0; i < 2; ++i) {
    cudaMemcpyAsync(devPtrIn[i], hostPtr[i], size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(devPtrOut[i], devPtrIn[i], size);
    cudaMemcpyAsync(hostPtr[i], devPtrOut[i], size, cudaMemcpyDeviceToHost, stream[i]);
    cudaLaunchHostFunc(stream[i], MyCallback, (void*)i);
}

主机函数后在流中发出的命令在函数完成之前不会开始执行。

排队到流中的主机函数不得（直接或间接）进行CUDA API调用，因为如果它发出此类调用导致死锁，它最终可能会自行等待。

##### 6.2.8.5.7.流优先级[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-priorities "这个标题的永久链接")

流的相对优先级可以在创建时使用`cudaStreamCreateWithPriority()`指定。可以使用`cudaDeviceGetStreamPriorityRange()`函数获得允许的优先级范围，按[最大优先级，最小优先级]排序。在运行时，GPU调度器利用流优先级来确定任务执行顺序，但这些优先级作为提示而不是保证。在选择要启动的工作时，优先级较高的流中的待定任务优先于优先级流中的待定任务。优先级较高的任务不会优先级优先级较低的任务。GPU在任务执行期间不会重新评估工作队列，提高流的优先级不会中断正在进行的工作。流优先级影响任务执行，而不强制执行严格的排序，因此用户可以利用流优先级来影响任务执行，而无需依赖严格的排序保证。

以下代码示例获取当前设备的允许优先级范围，并创建具有最高和最低可用优先级的流。

// get the range of stream priorities for this device
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
// create streams with highest and lowest available priorities
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, greatestPriority));
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, leastPriority);

#### 6.2.8.6.程序化依赖启动和同步[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization "这个标题的永久链接")

_程序化依赖启动_机制允许依赖_次级_内核在同一CUDA流中依赖_的主内_核完成执行之前启动。从计算能力9.0的设备开始，当_辅助_内核能够完成不依赖于_主内_核结果的重要工作时，该技术可以提供性能优势。

##### 6.2.8.6.1.背景[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#background "这个标题的永久链接")

CUDA应用程序通过启动和执行多个内核来利用GPU。典型的GPU活动时间表如[图10](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#gpu-activity)所示。

[![GPU活动时间表](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-activity.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-activity.png)

图10 GPU活动时间表[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#gpu-activity "此图像的永久链接")

在这里，`secondary_kernel`在`primary_kernel`完成执行后启动。序列化执行通常是必要的，因为`secondary_kernel`依赖于`primary_kernel`生成的结果数据。如果`secondary_kernel`对`primary_kernel`没有依赖性，则可以使用[Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams)同时启动两者。即使`secondary_kernel`依赖于`primary_kernel`，也存在一些并发执行的可能性。例如，几乎所有的内核都有某种_前言_部分，在此期间执行缓冲区归零或加载常量值等任务。

[![“secondary_kernel”的序言部分](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/secondary-kernel-preamble.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/secondary-kernel-preamble.png)

图11序言部分`secondary_kernel`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#secondary-kernel-preamble "此图像的永久链接")

[图11](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#secondary-kernel-preamble)展示了可以在不影响应用程序的情况下同时执行的`secondary_kernel`部分。请注意，并发启动还允许我们在执行`primary_kernel`后隐藏`secondary_kernel`的启动延迟。

[![同时执行“primary_kernel”和“secondary_kernel”](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/preamble-overlap.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/preamble-overlap.png)

图12 `primary_kernel`的并发执行和`secondary_kernel`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#preamble-overlap "此图像的永久链接")

使用_程序化依赖启动_可以实现[图12](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#preamble-overlap)所示的`secondary_kernel`的并发启动和执行。

_程序化依赖启动_引入了对CUDA内核启动API的更改，如下一节所述。这些API至少需要计算能力9.0来提供重叠执行。

##### 6.2.8.6.2.API描述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-description "这个标题的永久链接")

在程序化依赖启动中，主内核和次要内核在同一CUDA流中启动。当主内核准备好启动时，主内核应该使用所有线程块执行`cudaTriggerProgrammaticLaunchCompletion`。二级内核必须使用可扩展启动API启动，如图所示。

__global__ void primary_kernel() {
   // Initial work that should finish before starting secondary kernel

   // Trigger the secondary kernel
   cudaTriggerProgrammaticLaunchCompletion();

   // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
   // Independent work

   // Will block until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory
   cudaGridDependencySynchronize();

   // Dependent work
}

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticStreamSerializationAllowed = 1;
configSecondary.attrs = attribute;
configSecondary.numAttrs = 1;

primary_kernel<<<grid_dim, block_dim, 0, stream>>>();
cudaLaunchKernelEx(&configSecondary, secondary_kernel);

当使用`cudaLaunchAttributeProgrammaticStreamSerialization`属性启动辅助内核时，CUDA驱动程序可以安全地提前启动辅助内核，而不是在启动辅助内核之前等待主内核的完成和内存刷新。

当所有主线程块都启动并执行了`cudaTriggerProgrammaticLaunchCompletion`，CUDA驱动程序可以启动辅助内核。如果主内核没有执行触发器，则在主内核退出中的所有线程块后隐式发生。

无论哪种情况，辅助线程块都可能在主内核写入的数据可见之前启动。因此，当二级内核配置了_Programmatic Dependent Launch_时，它必须始终使用`cudaGridDependencySynchronize`或其他手段来验证来自主内核的结果数据是否可用。

请注意，这些方法为主要和次要核心提供了同時執行的機會，但是这种行为是機會主義的，不能保證會導致並行核心執行。以这种方式依赖并发执行是不安全的，并可能导致僵局。

##### 6.2.8.6.3.在CUDA图表中使用[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#use-in-cuda-graphs "这个标题的永久链接")

程序化依赖启动可以通过[流捕获](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture)或直接通过[边缘数据](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#edge-data)在[CUDA图中](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs)使用。要在带有边缘数据的CUDA图中编程此功能，请在连接两个内核节点的边缘上使用`cudaGraphDependencyTypeProgrammatic`的`cudaGraphDependencyType`。这种边缘类型使上游内核对下游内核中的`cudaGridDependencySynchronize()`可见。此类型必须与`cudaGraphKernelNodePortLaunchCompletion`或`cudaGraphKernelNodePortProgrammatic`的出站端口一起使用。

流捕获的图形等价物如下：

|流代码（缩写）|结果图形边缘|
|---|---|
|cudaLaunchAttribute attribute;<br>attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;<br>attribute.val.programmaticStreamSerializationAllowed = 1;|cudaGraphEdgeData edgeData;<br>edgeData.type = cudaGraphDependencyTypeProgrammatic;<br>edgeData.from_port = cudaGraphKernelNodePortProgrammatic;|
|cudaLaunchAttribute attribute;<br>attribute.id = cudaLaunchAttributeProgrammaticEvent;<br>attribute.val.programmaticEvent.triggerAtBlockStart = 0;|cudaGraphEdgeData edgeData;<br>edgeData.type = cudaGraphDependencyTypeProgrammatic;<br>edgeData.from_port = cudaGraphKernelNodePortProgrammatic;|
|cudaLaunchAttribute attribute;<br>attribute.id = cudaLaunchAttributeProgrammaticEvent;<br>attribute.val.programmaticEvent.triggerAtBlockStart = 1;|cudaGraphEdgeData edgeData;<br>edgeData.type = cudaGraphDependencyTypeProgrammatic;<br>edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion;|

#### 6.2.8.7.CUDA图表[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs "这个标题的永久链接")

CUDA图形为在CUDA中提交工作提供了一种新的模式。图形是一系列操作，如内核启动，由依赖关系连接，其定义与其执行分开。这允许定义一次图形，然后反复启动。将图形的定义与其执行分开，可以实现一些优化：首先，与流相比，CPU启动成本降低，因为大部分设置都是提前完成的；其次，向CUDA呈现整个工作流程可以实现优化，而流的分段工作提交机制可能无法实现。

要查看使用图形可能的优化，请考虑流中发生的事情：当您将内核放入流中时，主机驱动程序会执行一系列操作，为在GPU上执行内核做准备。这些操作是设置和启动内核所必需的，是必须为每个发布的内核支付的间接费用。对于执行时间短的GPU内核，这个开销成本可能是整个端到端执行时间的很大一部分。

使用图表提交工作分为三个不同的阶段：定义、实例化和执行。

- 在定义阶段，程序创建图形中操作的描述以及它们之间的依赖关系。
    
- 实例化对图形模板进行快照，进行验证，并执行大部分工作的设置和初始化，目的是尽量减少启动时需要完成的工作。生成的实例被称为_可执行图。_
    
- 可执行图可以启动到流中，类似于任何其他CUDA工作。它可以在不重复实例化的情况下任意启动次数。
    

##### 6.2.8.7.1.图形结构[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-structure "这个标题的永久链接")

操作在图形中形成一个节点。操作之间的依赖关系是边缘。这些依赖关系限制了操作的执行顺序。

一旦它所依赖的节点完成，操作可以随时安排。日程安排由CUDA系统来安排。

###### 6.2.8.7.1.1.节点类型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#node-types "这个标题的永久链接")

图形节点可以是：

- 内核
    
- CPU功能调用
    
- 记忆副本
    
- 梅姆塞特
    
- 空节点
    
- 等待一个[事件](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events)
    
- 记录一个[事件](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events)
    
- 发出[外部信号信号](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#external-resource-interoperability)
    
- 等待[外部訊號燈](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#external-resource-interoperability)
    
- [条件节点](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-graph-nodes)
    
- [图形内存节点](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-memory-nodes)
    
- 子图：执行单独的嵌套图，如下图所示。
    

[![子图形示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/child-graph.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/child-graph.png)

图13子图示例[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#node-types-fig-child-graph "此图像的永久链接")

###### 6.2.8.7.1.2.边缘数据[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#edge-data "这个标题的永久链接")

CUDA 12.3在CUDA图形上引入了边缘数据。边缘数据修改边缘指定的依赖关系，由三个部分组成：出站端口、入站端口和类型。出站端口指定何时触发关联的边缘。传入端口指定节点的哪个部分依赖于关联的边缘。类型修改端点之间的关系。

端口值特定于节点类型和方向，边缘类型可能仅限于特定的节点类型。在所有情况下，零初始化边缘数据都代表默认行为。出站端口0等待整个任务，入站端口0阻止整个任务，边缘类型0与内存同步行为完全依赖相关联。

边缘数据可选择通过平行数组在各种图形API中指定到相关节点。如果省略它作为输入参数，则使用零初始化数据。如果省略它作为输出（查询）参数，如果被忽略的边缘数据都是零初始化的，API将接受这个参数，如果调用会丢弃信息，则返回`cudaErrorLossyQuery`。

边缘数据也可用于一些流捕获API：`cudaStreamBeginCaptureToGraph()``cudaStreamGetCaptureInfo()`和`cudaStreamUpdateCaptureDependencies()`在这些情况下，还没有下游节点。数据与悬垂边缘（半边缘）相关联，该边缘要么连接到未来捕获的节点，要么在流捕获结束时丢弃。请注意，一些边缘类型不会等待上游节点的完全完成。在考虑流捕获是否已完全重新加入源流时，这些边缘会被忽略，并且不能在捕获结束时丢弃。请参阅[使用流捕获创建图形](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture)。

目前，没有节点类型定义额外的传入端口，只有内核节点定义了额外的传出端口。有一个非默认依赖类型，`cudaGraphDependencyTypeProgrammatic`，它允许在两个内核节点之间实现[程序化依赖启动](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization)。

##### 6.2.8.7.2.使用Graph API创建图表[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis "这个标题的永久链接")

图形可以通过两种机制创建：显式API和流捕获。以下是创建和执行以下图表的示例。

[![使用Graph API创建图形示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/create-a-graph.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/create-a-graph.png)

图14使用Graph API创建图形示例[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-api-fig-creating-using-graph-apis "此图像的永久链接")

// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);

// For the purpose of this example, we'll create
// the nodes separately from the dependencies to
// demonstrate that it can be done in two stages.
// Note that dependencies can also be specified
// at node creation.
cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&b, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&c, graph, NULL, 0, &nodeParams);
cudaGraphAddKernelNode(&d, graph, NULL, 0, &nodeParams);

// Now set up dependencies on each node
cudaGraphAddDependencies(graph, &a, &b, NULL, 1);     // A->B
cudaGraphAddDependencies(graph, &a, &c, NULL, 1);     // A->C
cudaGraphAddDependencies(graph, &b, &d, NULL, 1);     // B->D
cudaGraphAddDependencies(graph, &c, &d, NULL, 1);     // C->D

##### 6.2.8.7.3.使用流捕获创建图形[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture "这个标题的永久链接")

流捕获提供了一种从现有基于流的API创建图形的机制。将工作启动到流中的代码部分，包括现有代码，可以与对`cudaStreamBeginCapture()`和`cudaStreamEndCapture()`的调用括起。见下文。

cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);

调用`cudaStreamBeginCapture()`将流置于捕获模式。当捕获流时，启动到流中的工作不会被列入执行。相反，它被附加到正在逐步构建的内部图上。然后通过调用`cudaStreamEndCapture()`返回此图，这也结束了流的捕获模式。通过流捕获积极构建的图形被称为_捕获图。_

流捕获可用于除`cudaStreamLegacy`（“NULL流”）以外的任何CUDA流。请注意，它可以在`cudaStreamPerThread`上使用。如果程序正在使用遗留流，则有可能将流0重新定义为每线程流，而没有功能更改。请参阅[默认流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#default-stream)。

可以用`cudaStreamIsCapturing()`查询流是否被捕獲。

可以使用`cudaStreamBeginCaptureToGraph()`将工作捕获到现有图中。工作不是捕获到内部图表，而是捕获到用户提供的图表中。

###### 6.2.8.7.3.1.跨流依赖关系和事件[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cross-stream-dependencies-and-events "这个标题的永久链接")

流捕获可以处理用`cudaEventRecord()`和`cudaStreamWaitEvent()`表达的跨流依赖关系，前提是等待的事件被记录在同一捕获图中。

当事件被记录在处于捕获模式的流中时，它会导致_捕获事件。_捕获的事件表示捕获图中的一组节点。

当捕获的事件被流等待时，如果流尚未处于捕获模式，它将将流置于捕获模式，并且流中的下一个项目将对捕获事件中的节点有额外的依赖关系。然后将两个流捕获到相同的捕获图中。

当流捕获中存在跨流依赖性时，`cudaStreamEndCapture()`仍然必须在调用`cudaStreamBeginCapture()`的同一流中调用；这是_源流_。由于基于事件的依赖关系，捕获到同一捕获图的任何其他流也必须重新加入到源流中。这如下所示。在`cudaStreamEndCapture()`所有捕获到同一捕获图的流都从捕获模式中取出。未能重新加入源流将导致整体捕获操作失败。

// stream1 is the origin stream
cudaStreamBeginCapture(stream1);

kernel_A<<< ..., stream1 >>>(...);

// Fork into stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernel_B<<< ..., stream1 >>>(...);
kernel_C<<< ..., stream2 >>>(...);

// Join stream2 back to origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

kernel_D<<< ..., stream1 >>>(...);

// End capture in the origin stream
cudaStreamEndCapture(stream1, &graph);

// stream1 and stream2 no longer in capture mode

上述代码返回的图形如[图14](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-api-fig-creating-using-graph-apis)所示。

笔记

当流从捕获模式中取出时，流中的下一个未捕获项目（如果有的话）仍将依赖于最近的未捕获项目，尽管中间项目已被删除。

###### 6.2.8.7.3.2.禁止和未处理的操作[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#prohibited-and-unhandled-operations "这个标题的永久链接")

同步或查询正在捕获的流或捕获事件的执行状态是无效的，因为它们不代表计划执行的项目。查询包含活动流捕获的更广泛的句柄的执行状态或同步，当任何关联流处于捕获模式时，例如设备或上下文句柄，也是无效的。

当捕获同一上下文中的任何流，并且不是使用`cudaStreamNonBlocking`创建的时，任何尝试使用遗留流都是无效的。这是因为传统流句柄始终包含这些其他流；将队列到传统流将对正在捕获的流产生依赖，查询或同步它将查询或同步正在捕获的流。

因此，在这种情况下，调用同步API也是无效的。同步API，如`cudaMemcpy()`将工作串到遗留流，并在返回之前对其进行同步。

笔记

一般来说，当依赖关系将捕获的东西与未捕获的东西连接起来，而是排队执行时，CUDA更喜欢返回错误，而不是忽略依赖性。将流置于或移出捕获模式时会进行例外处理；这会在模式转换前后立即切换到流中添加的项目之间的依赖关系。

通过等待来自正在捕获的流中捕获的事件，并与事件不同的捕获图相关联，合并两个单独的捕获图是无效的。在没有指定cudaEventWaitExternal标志的情况下，等待正在捕获的流中未捕获的事件是无效的。

图中目前不支持将非同步操作队列到流中的少数API，如果与正在捕获的流（如`cudaStreamAttachMemAsync()`一起调用，将返回错误。

###### 6.2.8.7.3.3.无效[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#invalidation "这个标题的永久链接")

当在流捕获期间尝试无效操作时，任何关联的捕获图都将_失效_。当捕获图无效时，进一步使用任何正在捕获的流或与图相关的捕获事件都是无效的，并将返回错误，直到流捕获使用`cudaStreamEndCapture()`结束。此调用将使相关流退出捕获模式，但也将返回错误值和空图。

##### 6.2.8.7.4.CUDA用户对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-user-objects "这个标题的永久链接")

CUDA用户对象可用于帮助管理CUDA中异步工作使用的资源的生命周期。特别是，此功能对[CUDA图形](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs)和[流捕获](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture)很有用。

各种资源管理方案与CUDA图表不兼容。例如，考虑基于事件的池或同步创建、异步销毁方案。

// Library API with pool allocation
void libraryWork(cudaStream_t stream) {
    auto &resource = pool.claimTemporaryResource();
    resource.waitOnReadyEventInStream(stream);
    launchWork(stream, resource);
    resource.recordReadyEvent(stream);
}

// Library API with asynchronous resource deletion
void libraryWork(cudaStream_t stream) {
    Resource *resource = new Resource(...);
    launchWork(stream, resource);
    cudaLaunchHostFunc(
        stream,
        [](void *resource) {
            delete static_cast<Resource *>(resource);
        },
        resource,
        0);
    // Error handling considerations not shown
}

这些方案很难使用CUDA图，因为资源的非固定指针或句柄需要间接或图形更新，以及每次提交工作时所需的同步CPU代码。如果这些考虑因素对库的调用者隐藏，并且由于在捕获期间使用不允许的API，它们也不适用于流捕获。存在各种解决方案，例如将资源暴露给呼叫者。CUDA用户对象呈现另一种方法。

CUDA用户对象将用户指定的解构函数回调与内部参考计数相关联，类似于C++ `shared_ptr`。引用可能由CPU上的用户代码和CUDA图形拥有。请注意，对于用户拥有的引用，与C++智能指针不同，没有代表引用的对象；用户必须手动跟踪用户拥有的引用。一个典型的用例是在创建用户对象后，立即将唯一用户拥有的引用移动到CUDA图。

当引用与CUDA图相关联时，CUDA将自动管理图操作。克隆的`cudaGraph_t`保留源`cudaGraph_t`拥有的每个引用的副本，具有相同的多重性。实例化的`cudaGraphExec_t`保留源`cudaGraph_t`中每个引用的副本。当`cudaGraphExec_t`在没有同步的情况下被销毁时，引用将被保留，直到执行完成。

这里有一个使用示例。

cudaGraph_t graph;  // Preexisting graph

Object *object = new Object;  // C++ object with possibly nontrivial destructor
cudaUserObject_t cuObject;
cudaUserObjectCreate(
    &cuObject,
    object,  // Here we use a CUDA-provided template wrapper for this API,
             // which supplies a callback to delete the C++ object pointer
    1,  // Initial refcount
    cudaUserObjectNoDestructorSync  // Acknowledge that the callback cannot be
                                    // waited on via CUDA
);
cudaGraphRetainUserObject(
    graph,
    cuObject,
    1,  // Number of references
    cudaGraphUserObjectMove  // Transfer a reference owned by the caller (do
                             // not modify the total reference count)
);
// No more references owned by this thread; no need to call release API
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);  // Will retain a
                                                               // new reference
cudaGraphDestroy(graph);  // graphExec still owns a reference
cudaGraphLaunch(graphExec, 0);  // Async launch has access to the user objects
cudaGraphExecDestroy(graphExec);  // Launch is not synchronized; the release
                                  // will be deferred if needed
cudaStreamSynchronize(0);  // After the launch is synchronized, the remaining
                           // reference is released and the destructor will
                           // execute. Note this happens asynchronously.
// If the destructor callback had signaled a synchronization object, it would
// be safe to wait on it at this point.

子图节点中图所拥有的引用与子图相关联，而不是父图。如果更新或删除子图，引用也会相应更改。如果使用`cudaGraphExecUpdate`或`cudaGraphExecChildGraphNodeSetParams`更新可执行图或子图，则将克隆新源图中的引用并替换目标图中的引用。无论哪种情况，如果之前的启动没有同步，任何将要发布的引用都会被保留到启动完成执行。

目前没有通过CUDA API等待用户对象析构器的机制。用户可以从解构代码中手动发出同步对象的信号。此外，从析构函数调用CUDA API是不合法的，类似于对`cudaLaunchHostFunc`的限制。这是为了避免阻止CUDA内部共享线程并阻止前进。如果依赖性是单向的，并且进行调用的线程无法阻止CUDA工作的前进进度，则向另一个线程发出API调用信号是合法的。

用户对象是使用`cudaUserObjectCreate`创建的，这是浏览相关API的良好起点。

##### 6.2.8.7.5.更新实例化图形[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#updating-instantiated-graphs "这个标题的永久链接")

使用图表提交工作分为三个不同的阶段：定义、实例化和执行。在工作流程没有变化的情况下，定义和实例化的开销可以在许多执行中摊销，图形比流提供了明显的优势。

图形是工作流程的快照，包括内核、参数和依赖项，以便尽可能快速高效地重播它。在工作流程发生变化的情况下，图表会过时，必须修改。对图结构的重大更改，如拓扑结构或节点类型，将需要重新实例化源图，因为必须重新应用各种与拓扑相关的优化技术。

重复实例化的成本可能会降低图执行的整体性能收益，但只有节点参数（如内核参数和`cudaMemcpy`地址）发生变化，而图拓扑保持不变是很常见的。在这种情况下，CUDA提供了一个称为“图更新”的轻量级机制，该机制允许在原地修改某些节点参数，而无需重建整个图。这比恢复效率高得多。

更新将在下次启动图表时生效，因此它们不会影响之前的图表启动，即使它们在更新时正在运行。图表可以反复更新和重新启动，因此可以在流上排队多次更新/启动。

CUDA提供了两种更新实例化图形参数的机制，即整个图形更新和单个节点更新。整个图形更新允许用户提供拓扑相同的`cudaGraph_t`对象，其节点包含更新的参数。单个节点更新允许用户明确更新单个节点的参数。当大量节点正在更新时，或者当调用者不知道图形拓扑结构时（即图是库调用的流捕获结果）时，使用更新的`cudaGraph_t`更方便。当更改数量较少且用户拥有需要更新的节点的句柄时，首选使用单个节点更新。单个节点更新跳过了未更改节点的拓扑检查和比较，因此在许多情况下可以更高效。

CUDA还提供了一种机制，用于在不影响其当前参数的情况下启用和禁用单个节点。

以下部分将更详细地解释每种方法。

###### 6.2.8.7.5.1.图表更新限制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-update-limitations "这个标题的永久链接")

内核节点：

- 函数的拥有上下文不能改变。
    
- 功能最初不使用CUDA动态并行的节点不能更新为使用CUDA动态并行的函数。
    

`cudaMemset`和`cudaMemcpy`节点：

- 分配/映射操作数的CUDA设备无法更改。
    
- 源/目的地内存必须从与原始源/目的地内存相同的上下文中分配。
    
- 只能更改1D `cudaMemset`节点。
    

额外的memcpy节点限制：

- 不支持更改源或目标内存类型（即`cudaPitchedPtr`、`cudaArray_t`等）或传输类型（即`cudaMemcpyKind`）。
    

外部訊號燈等待节点和记录节点：

- 不支持更改信号灯的数量。
    

条件节点：

- 句柄创建和分配的顺序必须在图表之间匹配。
    
- 不支持更改节点参数（即条件、节点上下文等中的图形数量）。
    
- 更改条件体图中节点的参数受制于上述规则。
    

内存节点：

- 如果`cudaGraph_t`当前实例化为不同的`cudaGraphExec_t`，则无法使用`cudaGraph_t`更新`cudaGraphExec_t`。
    

对主机节点、事件记录节点或事件等待节点的更新没有限制。

###### 6.2.8.7.5.2.整体图表更新[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#whole-graph-update "这个标题的永久链接")

`cudaGraphExecUpdate()`允许使用拓扑相同图（“更新”图）的参数更新实例化图（“原始图”）。更新图的拓扑结构必须与用于实例化`cudaGraphExec_t`的原始图相同。此外，指定依赖项的顺序必须匹配。最后，CUDA需要始终如一地对汇节点（无依赖节点）进行排序。CUDA依赖于特定api调用的顺序来实现一致的汇节点排序。

更明确地说，遵循以下规则会导致`cudaGraphExecUpdate()`确定性地将原始图和更新图中的节点配对：

1. 对于任何捕获流，在该流上运行的API调用必须以相同的顺序进行，包括事件等待和其他与节点创建不直接对应的API调用。
    
2. 直接操作给定图节点的传入边缘（包括捕获的流API、节点添加API和边缘添加/删除API）的API调用必须按照相同的顺序进行。此外，当在数组中指定依赖关系到这些API时，这些数组中指定依赖关系的顺序必须匹配。
    
3. 汇节点必须按顺序排列。汇节点是调用`cudaGraphExecUpdate()`时最终图形中没有依赖节点/出站边缘的节点。以下操作会影响汇节点排序（如果存在），并且必须（作为组合集）以相同的顺序进行：
    
    - 节点添加API导致一个汇节点。
        
    - 边缘移除导致节点变成汇节点。
        
    - `cudaStreamUpdateCaptureDependencies()`，如果它从捕获流的依赖集中删除了一个汇节点。
        
    - `cudaStreamEndCapture()`.
        

以下示例展示了如何使用API来更新实例化图：

cudaGraphExec_t graphExec = NULL;

for (int i = 0; i < 10; i++) {
    cudaGraph_t graph;
    cudaGraphExecUpdateResult updateResult;
    cudaGraphNode_t errorNode;

    // In this example we use stream capture to create the graph.
    // You can also use the Graph API to produce a graph.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Call a user-defined, stream based workload, for example
    do_cuda_work(stream);

    cudaStreamEndCapture(stream, &graph);

    // If we've already instantiated the graph, try to update it directly
    // and avoid the instantiation overhead
    if (graphExec != NULL) {
        // If the graph fails to update, errorNode will be set to the
        // node causing the failure and updateResult will be set to a
        // reason code.
        cudaGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    }

    // Instantiate during the first iteration or whenever the update
    // fails for any reason
    if (graphExec == NULL || updateResult != cudaGraphExecUpdateSuccess) {

        // If a previous update failed, destroy the cudaGraphExec_t
        // before re-instantiating it
        if (graphExec != NULL) {
            cudaGraphExecDestroy(graphExec);
        }
        // Instantiate graphExec from graph. The error node and
        // error message parameters are unused here.
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    }

    cudaGraphDestroy(graph);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}

一个典型的工作流程是使用流捕获或图形API创建初始`cudaGraph_t`。然后，`cudaGraph_t`被实例化并正常启动。初始启动后，使用与初始图相同的方法创建新的`cudaGraph_t`，并调用`cudaGraphExecUpdate()`）。如果图形更新成功，由上述示例中的`updateResult`参数指示，则启动更新的`cudaGraphExec_t`。如果更新因任何原因失败，则调用`cudaGraphExecDestroy()`和`cudaGraphInstantiate()`来销毁原始`cudaGraphExec_t`并实例化一个新的。

也可以直接更新`cudaGraph_t`节点（即使用`cudaGraphKernelNodeSetParams()`，然后更新`cudaGraphExec_t`，但是使用下一节中涵盖的显式节点更新API更有效。

条件句柄标志和默认值作为图形更新的一部分进行更新。

有关使用情况和当前限制的更多信息，请参阅[Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)。

###### 6.2.8.7.5.3.单个节点更新[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#individual-node-update "这个标题的永久链接")

实例化图形节点参数可以直接更新。这消除了实例化的开销以及创建新`cudaGraph_t`的开销。如果需要更新的节点数量相对于图表中的节点总数来说很小，最好单独更新节点。以下方法可用于更新`cudaGraphExec_t`节点：

- `cudaGraphExecKernelNodeSetParams()`
    
- `cudaGraphExecMemcpyNodeSetParams()`
    
- `cudaGraphExecMemsetNodeSetParams()`
    
- `cudaGraphExecHostNodeSetParams()`
    
- `cudaGraphExecChildGraphNodeSetParams()`
    
- `cudaGraphExecEventRecordNodeSetEvent()`
    
- `cudaGraphExecEventWaitNodeSetEvent()`
    
- `cudaGraphExecExternalSemaphoresSignalNodeSetParams()`
    
- `cudaGraphExecExternalSemaphoresWaitNodeSetParams()`
    

有关使用情况和当前限制的更多信息，请参阅[Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)。

###### 6.2.8.7.5.4.单个节点启用[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#individual-node-enable "这个标题的永久链接")

实例化图中的内核、memset和memcpy节点可以使用`cudaGraphNodeSetEnabled()`API启用或禁用。这允许创建一个包含所需功能的超集的图表，该功能可以为每次启动进行定制。可以使用`cudaGraphNodeGetEnabled()`API查询节点的启用状态。

禁用的节点在功能上等同于空节点，直到重新启用。节点参数不受启用/禁用节点的影响。启用状态不受单个节点更新或使用`cudaGraphExecUpdate()`整个图形更新的影响。节点被禁用时的参数更新将在节点重新启用时生效。

以下方法可用于启用/禁用`cudaGraphExec_t`节点，以及查询其状态：

- `cudaGraphNodeSetEnabled()`
    
- `cudaGraphNodeGetEnabled()`
    

有关使用情况和当前限制的更多信息，请参阅[Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)。

##### 6.2.8.7.6.使用Graph API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-graph-apis "这个标题的永久链接")

`cudaGraph_t`对象不是线程安全的。用户有责任确保多个线程不会同时访问samecudaGraph_t。

`cudaGraphExec_t`不能与自身同时运行。`cudaGraphExec_t`的启动将在之前启动同一可执行图后订购。

图形执行在流中完成，用于与其他异步工作进行排序。然而，该流仅用于排序；它不限制图形的内部并行性，也不影响图形节点的执行位置。

请参阅[图形API。](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)

##### 6.2.8.7.7.设备图形启动[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-launch "这个标题的永久链接")

有许多工作流程需要在运行时做出依赖数据的决策，并根据这些决策执行不同的操作。用户可能更喜欢在设备上执行此决策过程，而不是将此决策过程卸载到主机，主机可能需要从设备往返。为此，CUDA提供了一个从设备发射图形的机制。

设备图形启动提供了一种从设备执行动态控制流的便捷方式，无论是像循环一样简单还是像设备端工作调度器一样复杂的东西。此功能仅在支持[统一寻址](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space)的系统上可用。

从现在开始，可以从设备启动的图形将被称为设备图形，而无法从设备启动的图形将被称为主机图。

设备图可以从主机和设备启动，而主机图只能从主机启动。与主机启动不同，在上次启动图形运行时从设备启动设备图形将导致错误，返回`cudaErrorInvalidValue`；因此，设备图形不能同时从设备启动两次。同时从主机和设备启动设备图将导致未定义的行为。

###### 6.2.8.7.7.1.设备图创建[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-creation "这个标题的永久链接")

为了从设备启动图形，它必须明确地实例化以供设备启动。这是通过将`cudaGraphInstantiateFlagDeviceLaunch`标志传递给`cudaGraphInstantiate()`调用来实现的。与主机图一样，设备图结构在实例化时是固定的，如果不重新实例化就无法更新，实例化只能在主机上执行。为了能够实例化设备启动的图形，它必须遵守各种要求。

6.2.8.7.7.1.1.设备图要求[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-requirements "这个标题的永久链接")

一般要求：

- 图形的节点必须全部位于单个设备上。
    
- 该图只能包含内核节点、memcpy节点、memset节点和子图节点。
    

内核节点：

- 不允许在图形中使用内核CUDA动态并行。
    
- 只要不使用MPS，就允许合作发射。
    

Memcpy节点：

- 仅允许涉及设备内存和/或固定设备映射的主机内存的副本。
    
- 不允许复制涉及CUDA阵列的副本。
    
- 两个操作数必须在实例化时从当前设备访问。请注意，复制操作将从图形所在的设备执行，即使它针对的是其他设备上的内存。
    

6.2.8.7.7.1.2.设备图形上传[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-upload "这个标题的永久链接")

为了在设备上启动图表，必须首先将其上传到设备，以填充必要的设备资源。这可以通过两种方式之一来实现。

首先，图形可以通过`cudaGraphUpload()`或通过`cudaGraphInstantiateWithParams()`作为实例化的一部分请求上传来显式上传。

或者，图形可以首先从主机启动，主机将隐式执行此上传步骤，作为启动的一部分。

所有三种方法的示例如下：

// Explicit upload after instantiation
cudaGraphInstantiate(&deviceGraphExec1, deviceGraph1, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphUpload(deviceGraphExec1, stream);

// Explicit upload as part of instantiation
cudaGraphInstantiateParams instantiateParams = {0};
instantiateParams.flags = cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUpload;
instantiateParams.uploadStream = stream;
cudaGraphInstantiateWithParams(&deviceGraphExec2, deviceGraph2, &instantiateParams);

// Implicit upload via host launch
cudaGraphInstantiate(&deviceGraphExec3, deviceGraph3, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphLaunch(deviceGraphExec3, stream);

6.2.8.7.7.1.3.设备图更新[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-update "这个标题的永久链接")

设备图形只能从主机更新，并且必须在可执行图形更新时重新上传到设备，才能使更改生效。这可以使用上一节中概述的相同方法来实现。与主机图不同，在应用更新时从设备启动设备图将导致未定义的行为。

###### 6.2.8.7.7.2.设备启动[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-launch "这个标题的永久链接")

设备图可以通过`cudaGraphLaunch()`从主机和设备启动，该设备在设备上的签名与在主机上的签名相同。设备图通过主机和设备上的相同句柄启动。从设备启动时，设备图必须从另一个图启动。

设备端图形启动是按线程进行的，多个启动可能同时从不同的线程发生，因此用户需要选择一个线程来启动给定的图形。

6.2.8.7.7.2.1.设备启动模式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-launch-modes "这个标题的永久链接")

与主机启动不同，设备图不能启动到常规CUDA流中，只能启动到不同的命名流中，每个流都表示特定的启动模式：

表5 仅设备图形启动流[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id456 "此表的永久链接")
|流动|启动模式|
|---|---|
|`cudaStreamGraphFireAndForget`|发射并忘记发射|
|`cudaStreamGraphTailLaunch`|尾部发射|
|`cudaStreamGraphFireAndForgetAsSibling`|兄弟姐妹启动|

6.2.8.7.7.2.1.1.开火和忘记发射[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fire-and-forget-launch "这个标题的永久链接")

顾名思义，启动和忘记启动会立即提交给GPU，并且它独立于启动图运行。在火和忘记的场景中，启动图是父图，启动图是子图。

[![_图像/fire-and-forget-simple.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-simple.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-simple.png)

图15 开火并忘记发射[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id457 "此图像的永久链接")

上面的图表可以通过下面的示例代码生成：

__global__ void launchFireAndForgetGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphFireAndForget);
}

void graphSetup() {
    cudaGraphExec_t gExec1, gExec2;
    cudaGraph_t g1, g2;

    // Create, instantiate, and upload the device graph.
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // Create and instantiate the launching graph.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchFireAndForgetGraph<<<1, 1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    // Launch the host graph, which will in turn launch the device graph.
    cudaGraphLaunch(gExec1, stream);
}

一个图形在执行过程中总共可以有120个触发和忘记的图形。此总数在同一父图的启动之间重置。

6.2.8.7.7.2.1.2.图形执行环境[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-execution-environments "这个标题的永久链接")

为了充分理解设备端同步模型，首先有必要了解执行环境的概念。

当图形从设备启动时，它会启动到自己的执行环境中。给定图的执行环境封装了图中的所有工作以及所有生成的火灾和遗忘工作。当图形完成执行且所有生成的子工作都完成时，该图可以被视为完成。

下图显示了上一节中触发和忘记示例代码生成的环境封装。

[![_图像/火灾和忘记环境.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-environments.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-environments.png)

图16 启动并忘记启动，带有执行环境[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id458 "此图像的永久链接")

这些环境也是分层的，因此图形环境可以包括多个级别的子环境，从火灾和忘记启动。

[![_图像/火和忘记-巢-环境.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-nested-environments.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-nested-environments.png)

图17 嵌套的火和遗忘环境[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id459 "此图像的永久链接")

当图形从主机启动时，存在一个流环境，该环境是启动图形的执行环境的父。流环境封装了作为整体启动的一部分生成的所有工作。当整体流环境被标记为完成时，流启动就完成了（即下游依赖工作现在可以运行）。

[![_图像/设备-图形-流-环境.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/device-graph-stream-environment.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/device-graph-stream-environment.png)

图18 溪流环境，可视化[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id460 "此图像的永久链接")

6.2.8.7.7.2.1.3.尾部发射[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tail-launch "这个标题的永久链接")

Unlike on the host, it is not possible to synchronize with device graphs from the GPU via traditional methods such as `cudaDeviceSynchronize()` or `cudaStreamSynchronize()`. Rather, in order to enable serial work dependencies, a different launch mode - tail launch - is offered, to provide similar functionality.

当图形的环境被认为是完整的时，尾部启动就会执行——即当图形及其所有子图都已完成时。当一个图形完成时，尾部启动列表中下一个图形的环境将替换完成的环境作为父环境的子环境。与火和忘记发射一样，一个图形可以有多个图形排成一排用于尾部发射。

[![_图像/tail-launch-simple.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-simple.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-simple.png)

图19 一个简单的尾部发射[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id461 "此图像的永久链接")

上述执行流程可以通过以下代码生成：

__global__ void launchTailGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphTailLaunch);
}

void graphSetup() {
    cudaGraphExec_t gExec1, gExec2;
    cudaGraph_t g1, g2;

    // Create, instantiate, and upload the device graph.
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // Create and instantiate the launching graph.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchTailGraph<<<1, 1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    // Launch the host graph, which will in turn launch the device graph.
    cudaGraphLaunch(gExec1, stream);
}

由给定图排成一排的尾部发射将一次执行一个，按照排成排的顺序。因此，第一个排成一排的图将首先运行，然后是第二个，以此以此为。

![_图像/尾部-启动-排序-简单.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-ordering-simple.png)

图20 尾部发射顺序[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id462 "此图像的永久链接")

由尾图排成一排的尾部发射将在尾部发射列表中由前一个图排成一排的尾部发射前执行。这些新的尾翼发射将按照排成一排的顺序执行。

![_图像/尾部-启动-订购-复杂.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-ordering-complex.png)

图21 从多个图形排队时的尾部启动顺序[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id463 "此图像的永久链接")

一个图表可以有多达255个待定的尾部发射。

6.2.8.7.7.2.1.3.1.尾部自发[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tail-self-launch "这个标题的永久链接")

设备图可以自行排成一排进行尾部发射，尽管给定的图一次只能排一个自启动。为了查询当前运行的设备图，以便重新启动，添加了一个新的设备端函数：

cudaGraphExec_t cudaGetCurrentGraphExec();

如果当前运行的图形是设备图，则该函数返回当前运行的图形的句柄。如果当前正在执行的内核不是设备图中的节点，则此函数将返回NULL。

以下是示例代码，显示了此函数在重新启动循环中的使用情况：

__device__ int relaunchCount = 0;

__global__ void relaunchSelf() {
    int relaunchMax = 100;

    if (threadIdx.x == 0) {
        if (relaunchCount < relaunchMax) {
            cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        }

        relaunchCount++;
    }
}

6.2.8.7.7.2.1.4.兄弟姐妹发射[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sibling-launch "这个标题的永久链接")

兄弟姐妹启动是触发和忘记启动的变体，其中图形不是作为启动图执行环境的子程序启动，而是作为启动图父环境的子程序启动。兄弟启动相当于从启动图的父环境启动并忘记启动。

![_图像/兄弟姐妹-启动-简单.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/sibling-launch-simple.png)

图22 一个简单的兄弟姐妹发射[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id464 "此图像的永久链接")

上面的图表可以通过下面的示例代码生成：

__global__ void launchSiblingGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphFireAndForgetAsSibling);
}

void graphSetup() {
    cudaGraphExec_t gExec1, gExec2;
    cudaGraph_t g1, g2;

    // Create, instantiate, and upload the device graph.
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream);

    // Create and instantiate the launching graph.
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launchSiblingGraph<<<1, 1, 0, stream>>>(gExec2);
    cudaStreamEndCapture(stream, &g1);
    cudaGraphInstantiate(&gExec1, g1);

    // Launch the host graph, which will in turn launch the device graph.
    cudaGraphLaunch(gExec1, stream);
}

由于兄弟启动不会启动到启动图的执行环境中，它们不会在启动图中排队的尾部发射。

##### 6.2.8.7.8.条件图节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-graph-nodes "这个标题的永久链接")

条件节点允许条件执行和循环条件节点中包含的图形。这允许动态和迭代工作流程在图形中完全表示，并释放主机CPU来并行执行其他工作。

当满足条件节点的依赖性时，在设备上进行条件值的评估。条件节点可以是以下类型之一：

- 如果节点执行时条件值非零，则条件[IF节点](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-if-nodes)执行一次其身体图。可以提供可选的第二个身体图，如果节点执行时条件值为零，则将执行一次。
    
- 如果节点执行时条件值为非零，则条件[WILE节点](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-while-nodes)执行其身体图，并将继续执行其身体图，直到条件值为零。
    
- 如果条件值等于n，条件[开关节点](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-switch-nodes)执行第n个身体图一次。如果条件值与身体图不对应，则不会启动身体图。
    

条件值由[条件句柄](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-handles)访问，[条件句柄](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-handles)必须在节点之前创建。条件值可以通过使用`cudaGraphSetConditional()`的设备代码进行设置。在创建句柄时，也可以指定应用于每个图形启动的默认值。

创建条件节点时，会创建一个空图形，并将句柄返回给用户，以便可以填充图形。可以使用[graph API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis)或[cudaStreamBeginCaptureToGraph()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture)填充此条件正文图。

条件节点可以嵌套。

###### 6.2.8.7.8.1.条件手柄[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-handles "这个标题的永久链接")

条件值由`cudaGraphConditionalHandle`表示，并由`cudaGraphConditionalHandleCreate()`创建。

句柄必须与单个条件节点相关联。手柄不能被破坏。

如果在创建句柄时指定了`cudaGraphCondAssignDefault`，则条件值将在每次图形执行开始时初始化为指定的默认值。如果没有提供此标志，条件值在每个图形执行开始时都是未定义的，代码不应假设条件值在执行中持续存在。

与句柄关联的默认值和标志将在[整个图形更新](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#whole-graph-update)期间更新。

###### 6.2.8.7.8.2.条件节点正体图要求[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-node-body-graph-requirements "这个标题的永久链接")

一般要求：

- 图形的节点必须全部位于单个设备上。
    
- 该图只能包含内核节点、空节点、memcpy节点、memset节点、子图节点和条件节点。
    

内核节点：

- 不允许在图表中使用CUDA动态并行或设备图形启动内核。
    
- 只要不使用MPS，就允许合作发射。
    

Memcpy/Memset节点：

- 仅允许涉及设备内存和/或固定设备映射主机内存的副本/内存集。
    
- 不允许使用涉及CUDA阵列的副本/memset。
    
- 两个操作数必须在实例化时从当前设备访问。请注意，复制操作将从图形所在的设备执行，即使它针对的是其他设备上的内存。
    

###### 6.2.8.7.8.3.条件IF节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-if-nodes "这个标题的永久链接")

如果执行节点时条件为非零，则IF节点的正体图将执行一次。下图描绘了一个3个节点图，其中中间节点B是一个条件节点：

![_图像/条件-如果-节点.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/conditional-if-node.png)

图23条件IF节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id465 "此图像的永久链接")

以下代码说明了包含IF条件节点的图形的创建。条件的默认值是使用上游内核设置的。条件的主体使用[图形API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis)填充。

__global__ void setHandle(cudaGraphConditionalHandle handle)
{
    ...
    cudaGraphSetConditional(handle, value);
    ...
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[1];
    int value = 1;

    cudaGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Use a kernel upstream of the conditional to set the handle value
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    cudaGraphAddNode(&node, graph, NULL, NULL, 0, &params);

    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeIf;
    cParams.conditional.size   = 1;
    cudaGraphAddNode(&node, graph, &node, NULL, 1, &cParams);

    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    // Populate the body of the conditional node
    ...
    cudaGraphAddNode(&node, bodyGraph, NULL, NULL, 0, &params);

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

从CUDA 12.8开始，IF节点还可以有一个可选的第二个身体图，如果条件值为零，则在节点执行时执行一次。

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[1];
    int value = 1;

    cudaGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Use a kernel upstream of the conditional to set the handle value
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    cudaGraphAddNode(&node, graph, NULL, NULL, 0, &params);

    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeIf;
    cParams.conditional.size   = 2; // Note that size is now set to '2'
    cudaGraphAddNode(&node, graph, &node, NULL, 1, &cParams);

    cudaGraph_t ifBodyGraph = cParams.conditional.phGraph_out[0];
    cudaGraph_t elseBodyGraph = cParams.conditional.phGraph_out[1];

    // Populate the body graphs of the conditional node
    ...
    cudaGraphAddNode(&node, ifBodyGraph, NULL, NULL, 0, &params);
    ...
    cudaGraphAddNode(&node, elseBodyGraph, NULL, NULL, 0, &params);

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

###### 6.2.8.7.8.4.条件的WILE节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-while-nodes "这个标题的永久链接")

只要条件为非零，就会执行 WHILE节点的正体图。当节点执行时和完成正文图后，将对条件进行评估。下图描绘了一个3个节点图，其中中间节点B是一个条件节点：

![_图像/条件-while-node.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/conditional-while-node.png)

图24条件的WILE节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id466 "此图像的永久链接")

以下代码说明了包含WHILE条件节点的图形的创建。手柄是使用_cudaGraphCondAssignDefault_创建的，以避免对上游内核的需求。条件的主体使用[图形API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis)填充。

__global__ void loopKernel(cudaGraphConditionalHandle handle)
{
    static int count = 10;
    cudaGraphSetConditional(handle, --count ? 1 : 0);
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[1];

    cuGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault);

    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeWhile;
    cParams.conditional.size   = 1;
    cudaGraphAddNode(&node, graph, NULL, NULL, 0, &cParams);

    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];

    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)loopKernel;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    cudaGraphAddNode(&node, bodyGraph, NULL, NULL, 0, &params);

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

###### 6.2.8.7.8.5.条件开关节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-switch-nodes "这个标题的永久链接")

CUDA 12.8中添加的SWITCH节点在条件节点内执行n个不同的图形中的1个。如果条件值为n，则当评估SWITCH节点时，将执行第n个图形。如果条件值大于或等于n，则不会执行任何图形。下图描绘了一个3个节点图，其中中间节点B是一个条件节点：

![_图像/条件开关-节点.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/conditional-switch-node.png)

图25条件开关节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id467 "此图像的永久链接")

以下代码说明了创建包含SWITCH条件节点的图形。条件的值是使用上游内核设置的。条件的主体使用[图形API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis)填充。

__global__ void setHandle(cudaGraphConditionalHandle handle)
{
    ...
    cudaGraphSetConditional(handle, value);
    ...
}

void graphSetup() {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[1];
    int value = 1;

    cudaGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph);

    // Use a kernel upstream of the conditional to set the handle value
    cudaGraphNodeParams params = { cudaGraphNodeTypeKernel };
    params.kernel.func = (void *)setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    cudaGraphAddNode(&node, graph, NULL, NULL, 0, &params);

    cudaGraphNodeParams cParams = { cudaGraphNodeTypeConditional };
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeSwitch;
    cParams.conditional.size   = 5;
    cudaGraphAddNode(&node, graph, &node, NULL, 1, &cParams);

    cudaGraph_t *bodyGraphs = cParams.conditional.phGraph_out;

    // Populate the first body of the conditional node
    ...
    cudaGraphAddNode(&node, bodyGraphs[0], NULL, NULL, 0, &params);
    ...
    // Populate the last body of the conditional node
    cudaGraphAddNode(&node, bodyGraphs[4], NULL, NULL, 0, &params);

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}

#### 6.2.8.8.事件[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events "这个标题的永久链接")

运行时还提供了一种密切监控设备进度的方法，以及通过让应用程序在程序中的任何点异步记录_事件_，并查询这些事件何时完成，从而执行准确的计时。当事件之前的所有任务——或可选的给定流中的所有命令——都已完成时，事件就完成了。流零中的事件在完成所有流中的所有前述任务和命令后完成。

##### 6.2.8.8.1.事件的创建和销毁[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-of-events "这个标题的永久链接")

以下代码示例创建了两个事件：

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

它们被这样摧毁：

cudaEventDestroy(start);
cudaEventDestroy(stop);

##### 6.2.8.8.2.流逝的时间[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#elapsed-time "这个标题的永久链接")

在[事件的创建和销毁](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-events)中创建的事件可用于以以下方式为[创建和销毁流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-streams)的代码样本计时：

cudaEventRecord(start, 0);
for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDev + i * size, inputHost + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>
               (outputDev + i * size, inputDev + i * size, size);
    cudaMemcpyAsync(outputHost + i * size, outputDev + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);

#### 6.2.8.9.同步呼叫[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronous-calls "这个标题的永久链接")

当调用同步函数时，在设备完成请求的任务之前，控制不会返回到主机线程。在主机线程执行任何其他CUDA调用之前，可以通过调用带有一些特定标志的`cudaSetDeviceFlags()`来指定主机线程是否会屈服、块或旋转。

### 6.2.9.多设备系统[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-device-system "这个标题的永久链接")

#### 6.2.9.1.设备枚举[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration "这个标题的永久链接")

主机系统可以有多个设备。以下代码示例展示了如何枚举这些设备、查询其属性以及确定启用CUDA的设备数量。

int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}

#### 6.2.9.2.设备选择[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-selection "这个标题的永久链接")

主机线程可以通过调用`cudaSetDevice()`随时设置其运行的设备。设备内存分配和内核启动在当前设置的设备上进行；流和事件与当前设置的设备相关联创建。如果没有调用`cudaSetDevice()`，当前设备为设备0。

以下代码示例说明了设置当前设备如何影响内存分配和内核执行。

size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // Set device 0 as current
float* p0;
cudaMalloc(&p0, size);       // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0
cudaSetDevice(1);            // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);       // Allocate memory on device 1
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1

#### 6.2.9.3.流和事件行为[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-and-event-behavior "这个标题的永久链接")

如果内核启动发布到未与当前设备关联的流，则内核启动将失败，如以下代码示例所示。

cudaSetDevice(0);               // Set device 0 as current
cudaStream_t s0;
cudaStreamCreate(&s0);          // Create stream s0 on device 0
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 0 in s0
cudaSetDevice(1);               // Set device 1 as current
cudaStream_t s1;
cudaStreamCreate(&s1);          // Create stream s1 on device 1
MyKernel<<<100, 64, 0, s1>>>(); // Launch kernel on device 1 in s1

// This kernel launch will fail:
MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 1 in s0

即使内存副本发布到与当前设备未关联的流，它也会成功。

`cudaEventRecord()`如果输入事件和输入流与不同的设备相关联，将失败。

`cudaEventElapsedTime()`如果两个输入事件与不同的设备相关联，将失败。

`cudaEventSynchronize()`和`cudaEventQuery()`即使输入事件与当前设备关联，也会成功。

`cudaStreamWaitEvent()`即使输入流和输入事件与不同的设备相关联，也会成功。因此，可以使用`cudaStreamWaitEvent()`将多个设备相互同步。

每个设备都有自己的默认流（请参阅[默认流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#default-stream)），因此发给设备默认流的命令可能会不按顺序或与发到任何其他设备的默认流的命令同时执行。

#### 6.2.9.4.点对点内存访问[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-to-peer-memory-access "这个标题的永久链接")

根据系统属性，特别是PCIe和/或NVLINK拓扑结构，设备能够处理彼此的内存（即，在一个设备上执行的内核可以将指针非引用到另一个设备内存）。如果`cudaDeviceCanAccessPeer()`为这两个设备返回true，则在两个设备之间支持这种点对点内存访问功能。

点对点内存访问仅支持64位应用程序，并且必须通过调用`cudaDeviceEnablePeerAccess()`两个设备之间启用，如以下代码示例所示。在非NVSwitch启用的系统上，每个设备最多可以支持8个全系统对等连接。

统一地址空间用于两个设备（请参阅[统一虚拟地址空间](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space)），因此可以使用相同的指针来定址来自两个设备的内存，如下文代码示例所示。

cudaSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0
cudaSetDevice(1);                   // Set device 1 as current
cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                    // with device 0

// Launch kernel on device 1
// This kernel launch can access memory on device 0 at address p0
MyKernel<<<1000, 128>>>(p0);

##### 6.2.9.4.1.Linux上的IOMMU[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#iommu-on-linux "这个标题的永久链接")

仅在Linux上，CUDA和显示驱动程序不支持支持IOMMU的裸机PCIe对等内存副本。然而，CUDA和显示驱动程序确实通过VM通过支持IOMMU。因此，Linux用户在原生裸机系统上运行时，应禁用IOMMU。应启用IOMMU，并将VFIO驱动程序用作虚拟机的PCIe通道。

在Windows上，上述限制不存在。

另请参阅[在64位平台上分配DMA缓冲器](https://download.nvidia.com/XFree86/Linux-x86_64/396.51/README/dma_issues.html)。

#### 6.2.9.5.点对点内存副本[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-to-peer-memory-copy "这个标题的永久链接")

内存复制可以在两个不同设备的内存之间执行。

当两个设备都使用统一地址空间（请参阅[统一虚拟地址空间](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space)）时，这是使用[设备内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory)中提到的常规内存复制函数完成的。

否则，这是使用`cudaMemcpyPeer()``cudaMemcpyPeerAsync()``cudaMemcpy3DPeer()`或`cudaMemcpy3DPeerAsync()`完成，如以下代码示例所示。

cudaSetDevice(0);                   // Set device 0 as current
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // Allocate memory on device 0
cudaSetDevice(1);                   // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);              // Allocate memory on device 1
cudaSetDevice(0);                   // Set device 0 as current
MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0
cudaSetDevice(1);                   // Set device 1 as current
cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
MyKernel<<<1000, 128>>>(p1);        // Launch kernel on device 1

两个不同设备内存之间的副本（在隐式_NULL_流中）：

- 直到之前向任一设备发出的所有命令都完成并且
    
- 在复制到任一设备后发出的任何命令（请参阅[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)）之前运行到完成。
    

与流的正常行为一致，两个设备内存之间的异步副本可能与另一个流中的副本或内核重叠。

请注意，如点[对点内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-to-peer-memory-access)中所述，如果通过`cudaDeviceEnablePeerAccess()`在两个设备之间启用点对点访问，则这两个设备之间的点对点内存副本不再需要通过主机进行分阶段，因此速度更快。

### 6.2.10.统一的虚拟地址空间[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space "这个标题的永久链接")

当应用程序作为64位进程运行时，主机和所有计算能力2.0及更高版本的设备将使用单个地址空间。通过CUDA API调用进行的所有主机内存分配以及受支持设备上的所有设备内存分配都在此虚拟地址范围内。因此：

- 通过CUDA分配的主机上或使用统一地址空间的任何设备上的任何内存的位置，都可以使用`cudaPointerGetAttributes()`从指针的值中确定。
    
- 当复制到或从任何使用统一地址空间的设备内存中复制时，`cudaMemcpy*()`的`cudaMemcpyKind`参数可以设置为`cudaMemcpyDefault`以确定指针的位置。这也适用于未通过CUDA分配的主机指针，只要当前设备使用统一寻址。
    
- 通过`cudaHostAlloc()`的分配在使用统一地址空间的所有设备上自动可移植（请参阅[便携式内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#portable-memory)），`cudaHostAlloc()`返回的指针可以直接从这些设备上运行的内核内使用（即，无需通过`cudaHostGetDevicePointer()`获取设备指针，如[映射内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory)中所述）。
    

应用程序可以通过检查`unifiedAddressing`设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)）是否等于1来查询统一地址空间是否用于特定设备。

### 6.2.11.进程间通信[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interprocess-communication "这个标题的永久链接")

主机线程创建的任何设备内存指针或事件句柄都可以由同一进程中的任何其他线程直接引用。然而，它在此过程之外是无效的，因此不能被属于不同过程的线程直接引用。

要跨进程共享设备内存指针和事件，应用程序必须使用进程间通信API，该API在参考手册中进行了详细描述。IPC API仅支持Linux上的64位进程和计算能力2.0及更高版本的设备。请注意，iPC API不支持`cudaMallocManaged`分配。

使用此API，应用程序可以使用`cudaIpcGetMemHandle()`获取给定设备内存指针的IPC句柄，使用标准IPC机制（例如，进程间共享内存或文件）将其传递给另一个进程，并使用`cudaIpcOpenMemHandle()`从IPC句柄中检索设备指针，该指针是此其他进程中的有效指针。事件句柄可以使用类似的入口点进行共享。

请注意，出于性能原因，`cudaMalloc()`进行的分配可能会从更大的内存块中进行子分配。在这种情况下，CUDA IPC API将共享整个底层内存块，这可能会导致其他子分配被共享，这可能会导致进程之间的信息泄露。为了防止这种行为，建议仅共享2MiB对齐大小的分配。

使用IPC API的一个例子是，单个主进程生成一批输入数据，使数据可供多个次级进程使用，而无需再生或复制。

使用CUDA IPC相互通信的应用程序应使用相同的CUDA驱动程序和运行时进行编译、链接和运行。

笔记

自CUDA 11.5以来，L4T和具有7.x及以上计算能力的嵌入式Linux Tegra设备仅支持事件共享IPC API。Tegra平台上仍然不支持内存共享IPC API。

### 6.2.12.错误检查[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#error-checking "这个标题的永久链接")

所有运行时函数都会返回一个错误代码，但对于异步函数（请参阅[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)），此错误代码不可能报告设备上可能发生的任何异步错误，因为该函数在设备完成任务之前返回；错误代码仅报告在执行任务之前在主机上发生的错误，通常与参数验证有关；如果发生异步错误，它将由后续一些不相关的运行时函数调用报告。

因此，在某些异步函数调用后检查异步错误的唯一方法是，在调用后通过调用`cudaDeviceSynchronize()`（或使用[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)中描述的任何其他同步机制）并检查`cudaDeviceSynchronize()`返回的错误代码进行同步。

运行时为每个主机线程维护一个错误变量，该变量初始化为`cudaSuccess`，每次出现错误（无论是参数验证错误还是异步错误）都会被错误代码覆盖。`cudaPeekAtLastError()`返回此变量。`cudaGetLastError()`返回此变量并将其重置为`cudaSuccess`。

内核启动不会返回任何错误代码，因此必须在内核启动后调用`cudaPeekAtLastError()`或`cudaGetLastError()`，以检索任何启动前错误。为了确保`cudaPeekAtLastError()`或`cudaGetLastError()`返回的任何错误不是来自内核启动前的调用，必须确保运行时错误变量在内核启动前设置为`cudaSuccess`，例如，在内核启动前调用`cudaGetLastError()`。内核启动是非同步的，因此要检查异步错误，应用程序必须在内核启动和调用`cudaPeekAtLastError()`或`cudaGetLastError()`之间同步。

请注意，`cudaStreamQuery()`和`cudaEventQuery()`可能返回的`cudaErrorNotReady`不被视为错误，因此`cudaPeekAtLastError()`或`cudaGetLastError()`不会报告。

### 6.2.13.呼叫堆栈[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#call-stack "这个标题的永久链接")

在具有计算能力2.x及更高版本的设备上，可以使用`cudaDeviceGetLimit()`查询调用堆栈的大小，并使用`cudaDeviceSetLimit()`进行设置。

当调用堆栈溢出时，如果应用程序是通过CUDA调试器（CUDA-GDB，Nsight）运行的，否则，内核调用将失败并出现堆栈溢出错误，否则。当编译器无法确定堆栈大小时，它会发出警告，表示堆栈大小无法静态确定。递归函数通常就是这种情况。一旦发出此警告，如果默认堆栈大小不够，用户将需要手动设置堆栈大小。

### 6.2.14.纹理和表面记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory "这个标题的永久链接")

CUDA支持GPU用于图形访问纹理和表面内存的纹理硬件的子集。如[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)中所述，从纹理或表面内存而不是全局内存读取数据可以有几个性能优势。

#### 6.2.14.1.纹理记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory "这个标题的永久链接")

使用[纹理函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-functions)中描述的设备函数从内核读取纹理内存。读取调用这些函数之一的纹理的过程称为_纹理获取_。每个纹理获取都为纹理对象API指定一个称为_纹理对象_的参数。

纹理对象指定：

- _纹理_，这是获取的纹理内存的一部分。纹理对象在运行时创建，并在创建纹理对象时指定纹理，如[纹理对象API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api)中所述。
    
- 它的_维度_指定了纹理是使用一个纹理坐标的一维数组、使用两个纹理坐标的二维数组，还是使用三个纹理坐标的三维数组。数组的元素称为_texels_，是_纹理元素_的缩写。_纹理宽度_、_高度_和_深度_是指每个维度中数组的大小。[表27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability)列出了根据设备的计算能力的最大纹理宽度、高度和深度。
    
- texel的类型，仅限于基本整数和单精度浮点类型，以及从基本整数和单精度浮点类型衍生的[内置向量类型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-vector-types)中定义的任何1、2和4分量向量类型。
    
- _读取模式_，等于`cudaReadModeNormalizedFloat`或`cudaReadModeElementType`。如果是`cudaReadModeNormalizedFloat`，并且texel的类型是16位或8位整数类型，则纹理获取返回的值实际上返回为浮点类型，整数类型的整个范围对映到[0.0, 1.0]对于无符号整数类型和[-1.0, 1.0]对于有符号整数类型；例如，值为0xff的无符号8位纹理元素读作1。如果是`cudaReadModeElementType`，则不会执行转换。
    
- 纹理坐标是否归一化。默认情况下，使用[0, N-1]范围内的浮点坐标引用纹理（由[纹理函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-functions)的函数），其中N是与坐标相对应的维度中纹理的大小。例如，对于x和y维度，尺寸为64x32的纹理将分别引用[0, 63]和[0, 31]范围内的坐标。规范化纹理坐标导致坐标在[0.0, 1.0-1/N]范围内指定，而不是[0, N-1]，因此相同的64x32纹理将在x和y维度中由[0, 1-1/N]范围内的规范化坐标处理。如果纹理坐标独立于纹理大小，则规范化纹理坐标自然适合某些应用程序的要求。
    
- _寻址模式_。调用B.8节的设备函数的坐标不在范围范围内是有效的。寻址模式定义了在这种情况下会发生什么。默认寻址模式是将坐标夹到有效范围：非规范化坐标为[0，N]，规范化坐标为[0.0，1.0）。如果指定了边框模式，则使用范围外纹理坐标的纹理获取将返回零。对于归一化坐标，还可以使用换行模式和镜像模式。使用包装模式时，每个坐标x都转换为_frac(x)=x - floor(x)_，其中_floor(x)_是不超过_x_的最大整数。使用镜像模式时，如果_floor(x)_为偶数，则每个坐标_x_转换为_frac(x)_，如果_floor(x)_为奇数，则转换为_1-frac(x)_。寻址模式被指定为大小为三的数组，其第一、第二和第三元素分别指定第一、第二和第三纹理坐标的寻址模式；寻址模式是`cudaAddressModeBorder`、`cudaAddressModeClamp`、`cudaAddressModeWrap`和`cudaAddressModeMirror`和`cudaAddressModeMirror`仅支持规范化纹理坐标
    
- 指定获取纹理时返回值的_过滤_模式是根据输入纹理坐标计算的。线性纹理过滤只能对配置为返回浮点数据的纹理进行。它在相邻的文本之间执行低精度插值。启用后，会读取围绕纹理获取位置的文本，并根据纹理坐标在文本之间的位置插值纹理的返回值。一维纹理执行简单线性插值，二维纹理执行双线性插值，三维纹理执行三线性插值。[纹理获取](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-fetching)提供了有关纹理获取的更多详细信息。过滤模式等于`cudaFilterModePoint`或`cudaFilterModeLinear`。如果是`cudaFilterModePoint`，则返回的值是纹理坐标最接近输入纹理坐标的文本。如果是`cudaFilterModeLinear`，返回的值是两个（对于一维纹理）、四个（对于二维纹理）或八个（对于三维纹理）文本的线性插值，其纹理坐标最接近输入纹理坐标。`cudaFilterModeLinear`仅适用于浮点类型的返回值。
    

[纹理对象API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api)引入了纹理对象API。

[16位浮点纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sixteen-bit-floating-point-textures)解释了如何处理16位浮点纹理。

纹理也可以分层，如[分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)中所述。

[立方体图纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures)和[立方体图分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures)描述了一种特殊的纹理类型，即立方体图纹理。

[Texture Gather](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-gather)描述了一种特殊的纹理获取，纹理收集。

##### 6.2.14.1.1.紋理物件API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api "这个标题的永久链接")

A texture object is created using `cudaCreateTextureObject()` from a resource description of type `struct cudaResourceDesc`, which specifies the texture, and from a texture description defined as such:

struct cudaTextureDesc
{
    enum cudaTextureAddressMode addressMode[3];
    enum cudaTextureFilterMode  filterMode;
    enum cudaTextureReadMode    readMode;
    int                         sRGB;
    int                         normalizedCoords;
    unsigned int                maxAnisotropy;
    enum cudaTextureFilterMode  mipmapFilterMode;
    float                       mipmapLevelBias;
    float                       minMipmapLevelClamp;
    float                       maxMipmapLevelClamp;
};

- `addressMode`指定寻址模式；
    
- `filterMode`指定过滤模式；
    
- `readMode`指定读取模式；
    
- `normalizedCoords`指定纹理坐标是否归一化；
    
- 请参阅`sRGB`、`maxAnisotropy`、`mipmapFilterMode`、`mipmapLevelBias`、`minMipmapLevelClamp`和`maxMipmapLevelClamp`的参考手册。
    

以下代码示例将一些简单的转换内核应用于纹理。

// Simple transformation kernel
__global__ void transformKernel(float* output,
                                cudaTextureObject_t texObj,
                                int width, int height,
                                float theta)
{
    // Calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)width;
    float v = y / (float)height;

    // Transform coordinates
    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // Read from texture and write to global memory
    output[y * width + x] = tex2D<float>(texObj, tu, tv);
}

// Host code
int main()
{
    const int height = 1024;
    const int width = 1024;
    float angle = 0.5;

    // Allocate and set some host data
    float *h_data = (float *)std::malloc(sizeof(float) * width * height);
    for (int i = 0; i < height * width; ++i)
        h_data[i] = i;

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof(float);
    // Copy data located at address h_data in host memory to device memory
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float),
                        height, cudaMemcpyHostToDevice);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // Allocate result of transformation in device memory
    float *output;
    cudaMalloc(&output, width * height * sizeof(float));

    // Invoke kernel
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                    (height + threadsperBlock.y - 1) / threadsperBlock.y);
    transformKernel<<<numBlocks, threadsperBlock>>>(output, texObj, width, height,
                                                    angle);
    // Copy data from device back to host
    cudaMemcpy(h_data, output, width * height * sizeof(float),
                cudaMemcpyDeviceToHost);

    // Destroy texture object
    cudaDestroyTextureObject(texObj);

    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);

    // Free host memory
    free(h_data);

    return 0;
}

##### 6.2.14.1.2.16位浮点纹理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#bit-floating-point-textures "这个标题的永久链接")

CUDA数组支持的16位浮点或_半格式_与IEEE 754-2008二进制2格式相同。

CUDA C++不支持匹配数据类型，但提供内在函数，通过`unsignedshort`转换为32位浮点格式：`__float2half_rn(float)`和`__half2float(unsignedshort)`这些功能仅在设备代码中支持。例如，可以在OpenEXR库中找到主机代码的等效函数。

在执行任何过滤之前，在纹理获取过程中，16位浮点组件被提升为32位浮点。

可以通过调用`cudaCreateChannelDescHalf*()`函数之一来创建16位浮点格式的通道描述。

##### 6.2.14.1.3.分层纹理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures "这个标题的永久链接")

一维或二维分层纹理（在Direct3D中也称为_纹理阵列_，在OpenGL中称为_阵列纹理_）是由一系列图层组成的纹理，所有这些图层都是相同维度、大小和数据类型的常规纹理。

一维分层纹理使用整数索引和浮点纹理坐标处理；索引表示序列中的一层，坐标处理该层中的文本。二维分层纹理使用整数索引和两个浮点纹理坐标处理；索引表示序列中的一层，坐标处理该层中的文本。

分层纹理只能通过使用`cudaArrayLayered`标志（以及一维分层纹理的高度为零）调用`cudaMalloc3DArray()`成为CUDA数组。

分层纹理是使用[tex1DLayered（）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlayered-object)和[tex2DLayered（）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayered-object)中描述的设备函数获取的。纹理过滤（请参阅[纹理获取](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-fetching)）仅在图层内完成，而不是跨图层。

分层纹理仅在具有计算能力2.0及更高版本的设备上支持。

##### 6.2.14.1.4.立方体图纹理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures "这个标题的永久链接")

_立方体图_纹理是一种特殊类型的二维分层纹理，有六层代表立方体的面：

- 一层的宽度等于其高度。
    
- 立方体映射使用三个纹理坐标_x、y_和_z来_处理，这些坐标被解释为从立方体中心发出的方向向量，指向立方体的一面和与该面相对应的层内的文本。更具体地说，面由最大幅度_m_的坐标选择，使用坐标_（s/m+1）/2_和_（t/m+1）/2_处理相应的层，其中_s_和_t_定义在[表6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures-cubemap-fetch)中。
    

表6 立方体地图獲取[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures-cubemap-fetch "此表的永久链接")
||   |面容|M|s|吨|
|---|---|---|---|---|---|
|`\|x\| > \|y\|`和`\|x\| > \|z\|`|X ≥ 0|0|x|-z|-y|
|X < 0|1|-x|Z|-y|
|`\|y\| > \|x\|`和`\|y\| > \|z\|`|Y≥0|2|y|x|Z|
|Y < 0|3|-y|x|-z|
|`\|z\| > \|x\|`和`\|z\| > \|y\|`|Z ≥ 0|4|Z|x|-y|
|Z < 0|5|-z|-x|-y|

立方体地图纹理只能通过用`cudaArrayCubemap`标志调用`cudaMalloc3DArray()`成为CUDA数组。

使用[texCubemap（）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemap-object)中描述的设备函数获取立方体图纹理。

立方体图纹理仅在计算能力2.0及更高版本的设备上受支持。

##### 6.2.14.1.5.立方体图分层纹理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures "这个标题的永久链接")

_立方体图分层_纹理是一种分层纹理，其图层是相同尺寸的立方体图。

立方体图分层纹理使用整数索引和三个浮点纹理坐标来处理；索引表示序列中的立方体图，坐标地址为该立方体图中的文本。

立方体地图分层纹理只能通过使用`cudaArrayLayered`和`cudaArrayCubemap`标志调用`cudaMalloc3DArray()`成为CUDA数组。

使用[texCubemapLayered（）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplayered-object)中描述的设备函数获取Cubemap分层纹理。纹理过滤（请参阅[纹理获取](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-fetching)）仅在图层内完成，而不是跨图层。

立方体图分层纹理仅支持计算能力2.0及更高版本的设备。

##### 6.2.14.1.6.纹理集合[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-gather "这个标题的永久链接")

纹理收集是一种特殊的纹理获取，仅适用于二维纹理。它由`tex2Dgather()`函数执行，该函数具有与`tex2D()`相同的参数，外加一个等于0、1、2或3的附加`comp`参数（见[tex2Dgather（））](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgather-object)。它返回四个32位数字，对应于在常规纹理获取期间用于双线性过滤的四个文本中每个文本的组件`comp`的值。例如，如果这些texels是值（253、20、31、255）、（250、25、29、254）、（249、16、37、253）、（251、22、30、250），并且`comp`是2，`tex2Dgather()`返回（31、29、37、30）。

请注意，纹理坐标仅以8位的分数精度计算。因此，`tex2Dgather()`可能会返回意外结果，如果`tex2D()`将1.0用于其权重之一（α或β，请参阅线性[过滤](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#linear-filtering)）。例如，_x_纹理坐标为2.49805：_xB=x-0.5=1.99805_，但是_xB_的分数部分以8位固定点格式存储。由于0.99805比255.f/256.f更接近256.f/256.f，_xB_的值为2。因此，在这种情况下，`tex2Dgather()`将返回_x_中的索引2和3，而不是索引1和2。

纹理收集仅支持使用`cudaArrayTextureGather`标志创建的CUDA数组，并且宽度和高度小于[表27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability)中为纹理收集指定的最大值，该数组小于常规纹理获取。

仅在计算能力2.0及更高版本的设备上支持纹理收集。

#### 6.2.14.2.表面存储器[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-memory "这个标题的永久链接")

对于具有计算能力2.0及更高版本的设备，使用`cudaArraySurfaceLoadStore`标志创建的CUDA数组（在[Cubemap Surfaces](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-surfaces)中描述）可以使用[Surface Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-functions)中描述的函数通过_表面对象_读取和写入。

[表27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability)列出了根据设备的计算能力的最大表面宽度、高度和深度。

##### 6.2.14.2.1.表面对象API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-object-api "这个标题的永久链接")

A surface object is created using `cudaCreateSurfaceObject()` from a resource description of type `struct cudaResourceDesc`. Unlike texture memory, surface memory uses byte addressing. This means that the x-coordinate used to access a texture element via texture functions needs to be multiplied by the byte size of the element to access the same element via a surface function. For example, the element at texture coordinate x of a one-dimensional floating-point CUDA array bound to a texture object `texObj` and a surface object `surfObj` is read using `tex1d(texObj, x)` via `texObj`, but `surf1Dread(surfObj, 4*x)` via `surfObj`. Similarly, the element at texture coordinate x and y of a two-dimensional floating-point CUDA array bound to a texture object `texObj` and a surface object `surfObj` is accessed using `tex2d(texObj, x, y)` via `texObj`, but `surf2Dread(surfObj,4*x, y)` via `surObj` (the byte offset of the y-coordinate is internally calculated from the underlying line pitch of the CUDA array).

以下代码示例将一些简单的变换内核应用于曲面。

// Simple copy kernel
__global__ void copyKernel(cudaSurfaceObject_t inputSurfObj,
                           cudaSurfaceObject_t outputSurfObj,
                           int width, int height)
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 data;
        // Read from input surface
        surf2Dread(&data,  inputSurfObj, x * 4, y);
        // Write to output surface
        surf2Dwrite(data, outputSurfObj, x * 4, y);
    }
}

// Host code
int main()
{
    const int height = 1024;
    const int width = 1024;

    // Allocate and set some host data
    unsigned char *h_data =
        (unsigned char *)std::malloc(sizeof(unsigned char) * width * height * 4);
    for (int i = 0; i < height * width * 4; ++i)
        h_data[i] = i;

    // Allocate CUDA arrays in device memory
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaArray_t cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);
    cudaArray_t cuOutputArray;
    cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);

    // Set pitch of the source (the width in memory in bytes of the 2D array
    // pointed to by src, including padding), we dont have any padding
    const size_t spitch = 4 * width * sizeof(unsigned char);
    // Copy data located at address h_data in host memory to device memory
    cudaMemcpy2DToArray(cuInputArray, 0, 0, h_data, spitch,
                        4 * width * sizeof(unsigned char), height,
                        cudaMemcpyHostToDevice);

    // Specify surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    // Create the surface objects
    resDesc.res.array.array = cuInputArray;
    cudaSurfaceObject_t inputSurfObj = 0;
    cudaCreateSurfaceObject(&inputSurfObj, &resDesc);
    resDesc.res.array.array = cuOutputArray;
    cudaSurfaceObject_t outputSurfObj = 0;
    cudaCreateSurfaceObject(&outputSurfObj, &resDesc);

    // Invoke kernel
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                    (height + threadsperBlock.y - 1) / threadsperBlock.y);
    copyKernel<<<numBlocks, threadsperBlock>>>(inputSurfObj, outputSurfObj, width,
                                                height);

    // Copy data from device back to host
    cudaMemcpy2DFromArray(h_data, spitch, cuOutputArray, 0, 0,
                            4 * width * sizeof(unsigned char), height,
                            cudaMemcpyDeviceToHost);

    // Destroy surface objects
    cudaDestroySurfaceObject(inputSurfObj);
    cudaDestroySurfaceObject(outputSurfObj);

    // Free device memory
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);

    // Free host memory
    free(h_data);

  return 0;
}

##### 6.2.14.2.2.立方体图表面[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-surfaces "这个标题的永久链接")

使用`surfCubemapread()`和`surfCubemapwrite()`（[surfCubemapread（）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemapread-object)和[surfCubemapwrite（））](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemapwrite-object)作为二维分层表面访问立方体地图曲面，即使用表示面的整数索引和两个浮点纹理坐标，地址对应于该面的图层内的texel。脸的顺序如[表6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures-cubemap-fetch)所示。

##### 6.2.14.2.3.立方体图分层曲面[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-surfaces "这个标题的永久链接")

使用`surfCubemapLayeredread()`和`surfCubemapLayeredwrite()`（[surfCubemapLayeredread（）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemaplayeredread-object)和[surfCubemapLayeredwrite（））](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemaplayeredwrite-object)作为二维分层表面访问Cubemap分层曲面，即使用表示其中一个立方体映射面的整数索引和两个浮点纹理坐标，地址与该面对应的图层内的texel。面的顺序如[表6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures-cubemap-fetch)所示，因此索引（（2 * 6）+ 3）例如访问第三个立方体图的第四个面。

#### 6.2.14.3.CUDA阵列[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arrays "这个标题的永久链接")

CUDA数组是为纹理获取而优化的不透明内存布局。它们是一维、二维或三维，由元素组成，每个元素都有1、2或4个成分，可以是有符号或无符号的8位、16位或32位整数、16位浮点或32位浮点。CUDA阵列只能通过[纹理内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory)中描述的纹理获取或[表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-memory)中描述的表面读取和写入来访问。

#### 6.2.14.4.读/写一致性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#read-write-coherency "这个标题的永久链接")

纹理和表面内存被缓存（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)），在同一内核调用中，缓存与全局内存写入和表面内存写入没有保持一致性，因此任何纹理获取或表面读取到同一内核调用中通过全局写入或表面写入的地址都会返回未定义的数据。换句话说，只有当该内存位置已由上一个内核调用或内存副本更新时，线程才能安全地读取某些纹理或表面内存位置，但当该内存位置已由同一线程或同一内核调用中的另一个线程更新时，则不能安全读取该内存位置。

### 6.2.15.图形互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graphics-interoperability "这个标题的永久链接")

来自OpenGL和Direct3D的一些资源可以映射到CUDA的地址空间，要么使CUDA能够读取OpenGL或Direct3D编写的数据，要么使CUDA能够写入OpenGL或Direct3D消耗的数据。

资源必须注册到CUDA，然后才能使用[OpenGL互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability)和[Direct3D互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-interoperability)中提到的功能进行映射。这些函数返回指向`struct`类型为`cudaGraphicsResource`的CUDA图形资源的指针。注册资源可能高昂，因此通常每个资源只能调用一次。使用`cudaGraphicsUnregisterResource()`未注册CUDA图形资源。每个打算使用该资源的CUDA上下文都需要单独注册。

一旦资源注册到CUDA，可以使用`cudaGraphicsMapResources()`和`cudaGraphicsUnmapResources()`根据需要多次映射和取消映射。可以调用`cudaGraphicsResourceSetMapFlags()`来指定使用提示（仅写，仅读），CUDA驱动程序可用于优化资源管理。

映射的资源可以由内核使用`cudaGraphicsResourceGetMappedPointer()`返回的缓冲区和`cudaGraphicsSubResourceGetMappedArray()`的CUDA数组返回的设备内存地址读取或写入。

在映射时通过OpenGL、Direct3D或其他CUDA上下文访问资源会产生未定义的结果。[OpenGL互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability)和[Direct3D互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-interoperability)为每个图形API和一些代码示例提供了具体说明。[SLI互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sli-interoperability)提供了系统何时处于SLI模式的具体说明。

#### 6.2.15.1.OpenGL互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability "这个标题的永久链接")

可以映射到CUDA地址空间的OpenGL资源是OpenGL缓冲区、纹理和渲染缓冲区对象。

使用`cudaGraphicsGLRegisterBuffer()`注册缓冲对象。在CUDA中，它显示为设备指针，因此可以通过内核或通过`cudaMemcpy()`调用读取和写入。

使用`cudaGraphicsGLRegisterImage()`注册纹理或渲染缓冲对象。在CUDA中，它显示为CUDA数组。内核可以通过将数组绑定到纹理或表面引用来读取数组。如果资源已注册为`cudaGraphicsRegisterFlagsSurfaceLoadStore`标志，他们也可以通过表面写入函数写入它。数组也可以通过`cudaMemcpy2D()`calls.cudaGraphicsGLRegisterImage`cudaGraphicsGLRegisterImage()`读取和写入，支持所有具有1、2、2或4个组件的纹理格式，以及内部类型的浮点（例如，`GL_RGBA_FLOAT32`）、规范化整数（例如，`GL_RGBA8,GL_INTENSITY16`）和非规范化整数（例如，`GL_RGBA8UI`）（请注意，由于未规范化整数格式需要OpenGL 3.0，它们只能由着色器编写，而不是固定函数管道）。

资源共享的OpenGL上下文必须是当前的主机线程，才能进行任何OpenGL互操作性API调用。

请注意：当OpenGL纹理无绑定时（例如，使用`glGetTextureHandle`*/`glGetImageHandle`* API请求图像或纹理句柄），它无法在CUDA注册。在请求图像或纹理句柄之前，应用程序需要注册互操作的纹理。

以下代码示例使用内核动态修改存储在顶点缓冲对象中的顶点的2D`width`x`height`网格：

GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;

int main()
{
    // Initialize OpenGL and GLUT for device 0
    // and make the OpenGL context current
    ...
    glutDisplayFunc(display);

    // Explicitly set device 0
    cudaSetDevice(0);

    // Create buffer object and register it with CUDA
    glGenBuffers(1, &positionsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    unsigned int size = width * height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA,
                                 positionsVBO,
                                 cudaGraphicsMapFlagsWriteDiscard);

    // Launch rendering loop
    glutMainLoop();

    ...
}

void display()
{
    // Map buffer object for writing from CUDA
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,
                                         positionsVBO_CUDA));

    // Execute kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);

    // Render from buffer object
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, width * height);
    glDisableClientState(GL_VERTEX_ARRAY);

    // Swap buffers
    glutSwapBuffers();
    glutPostRedisplay();
}

void deleteVBO()
{
    cudaGraphicsUnregisterResource(positionsVBO_CUDA);
    glDeleteBuffers(1, &positionsVBO);
}

__global__ void createVertices(float4* positions, float time,
                               unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // Write positions
    positions[y * width + x] = make_float4(u, w, v, 1.0f);
}

在Windows和Quadro GPU上，可以使用`cudaWGLGetDevice()`检索与`wglEnumGpusNV()`返回的句柄相关的CUDA设备。Quadro GPU在多GPU配置中比GeForce和Tesla GPU提供更高的OpenGL互操作性，OpenGL渲染在Quadro GPU上执行，CUDA计算在系统中的其他GPU上执行。

#### 6.2.15.2.Direct3D互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-interoperability "这个标题的永久链接")

Direct3D 9Ex、Direct3D 10和Direct3D 11支持Direct3D互操作性。

CUDA上下文只能与满足以下标准的Direct3D设备互操作：Direct3D 9Ex设备必须使用`DeviceType`设置为`D3DDEVTYPE_HAL`和带有`D3DCREATE_HARDWARE_VERTEXPROCESSING`标志的`BehaviorFlags`创建；Direct3D 10和Direct3D 11设备必须使用`DriverType`设置为`D3D_DRIVER_TYPE_HARDWARE`创建。

可能映射到CUDA地址空间的Direct3D资源是Direct3D缓冲区、纹理和曲面。这些资源使用`cudaGraphicsD3D9RegisterResource()``cudaGraphicsD3D10RegisterResource()`和`cudaGraphicsD3D11RegisterResource()`注册。

以下代码示例使用内核来动态修改存储在顶点缓冲对象中的顶点的2D`width`x`height`网格。

##### 6.2.15.2.1.Direct3D 9版本[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-9-version "这个标题的永久链接")

IDirect3D9* D3D;
IDirect3DDevice9* device;
struct CUSTOMVERTEX {
    FLOAT x, y, z;
    DWORD color;
};
IDirect3DVertexBuffer9* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;

int main()
{
    int dev;
    // Initialize Direct3D
    D3D = Direct3DCreate9Ex(D3D_SDK_VERSION);

    // Get a CUDA-enabled adapter
    unsigned int adapter = 0;
    for (; adapter < g_pD3D->GetAdapterCount(); adapter++) {
        D3DADAPTER_IDENTIFIER9 adapterId;
        g_pD3D->GetAdapterIdentifier(adapter, 0, &adapterId);
        if (cudaD3D9GetDevice(&dev, adapterId.DeviceName)
            == cudaSuccess)
            break;
    }

     // Create device
    ...
    D3D->CreateDeviceEx(adapter, D3DDEVTYPE_HAL, hWnd,
                        D3DCREATE_HARDWARE_VERTEXPROCESSING,
                        &params, NULL, &device);

    // Use the same device
    cudaSetDevice(dev);

    // Create vertex buffer and register it with CUDA
    unsigned int size = width * height * sizeof(CUSTOMVERTEX);
    device->CreateVertexBuffer(size, 0, D3DFVF_CUSTOMVERTEX,
                               D3DPOOL_DEFAULT, &positionsVB, 0);
    cudaGraphicsD3D9RegisterResource(&positionsVB_CUDA,
                                     positionsVB,
                                     cudaGraphicsRegisterFlagsNone);
    cudaGraphicsResourceSetMapFlags(positionsVB_CUDA,
                                    cudaGraphicsMapFlagsWriteDiscard);

    // Launch rendering loop
    while (...) {
        ...
        Render();
        ...
    }
    ...
}

void Render()
{
    // Map vertex buffer for writing from CUDA
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVB_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,
                                         positionsVB_CUDA));

    // Execute kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // Unmap vertex buffer
    cudaGraphicsUnmapResources(1, &positionsVB_CUDA, 0);

    // Draw and present
    ...
}

void releaseVB()
{
    cudaGraphicsUnregisterResource(positionsVB_CUDA);
    positionsVB->Release();
}

__global__ void createVertices(float4* positions, float time,
                               unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // Calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // Write positions
    positions[y * width + x] =
                make_float4(u, w, v, __int_as_float(0xff00ff00));
}

##### 6.2.15.2.2.Direct3D 10版本[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-10-version "这个标题的永久链接")

ID3D10Device* device;
struct CUSTOMVERTEX {
    FLOAT x, y, z;
    DWORD color;
};
ID3D10Buffer* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;

int main()
{
    int dev;
    // Get a CUDA-enabled adapter
    IDXGIFactory* factory;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    IDXGIAdapter* adapter = 0;
    for (unsigned int i = 0; !adapter; ++i) {
        if (FAILED(factory->EnumAdapters(i, &adapter))
            break;
        if (cudaD3D10GetDevice(&dev, adapter) == cudaSuccess)
            break;
        adapter->Release();
    }
    factory->Release();

    // Create swap chain and device
    ...
    D3D10CreateDeviceAndSwapChain(adapter,
                                  D3D10_DRIVER_TYPE_HARDWARE, 0,
                                  D3D10_CREATE_DEVICE_DEBUG,
                                  D3D10_SDK_VERSION,
                                  &swapChainDesc, &swapChain,
                                  &device);
    adapter->Release();

    // Use the same device
    cudaSetDevice(dev);

    // Create vertex buffer and register it with CUDA
    unsigned int size = width * height * sizeof(CUSTOMVERTEX);
    D3D10_BUFFER_DESC bufferDesc;
    bufferDesc.Usage          = D3D10_USAGE_DEFAULT;
    bufferDesc.ByteWidth      = size;
    bufferDesc.BindFlags      = D3D10_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags      = 0;
    device->CreateBuffer(&bufferDesc, 0, &positionsVB);
    cudaGraphicsD3D10RegisterResource(&positionsVB_CUDA,
                                      positionsVB,
                                      cudaGraphicsRegisterFlagsNone);
                                      cudaGraphicsResourceSetMapFlags(positionsVB_CUDA,
                                      cudaGraphicsMapFlagsWriteDiscard);

    // Launch rendering loop
    while (...) {
        ...
        Render();
        ...
    }
    ...
}

void Render()
{
    // Map vertex buffer for writing from CUDA
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVB_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,
                                         positionsVB_CUDA));

    // Execute kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // Unmap vertex buffer
    cudaGraphicsUnmapResources(1, &positionsVB_CUDA, 0);

    // Draw and present
    ...
}

void releaseVB()
{
    cudaGraphicsUnregisterResource(positionsVB_CUDA);
    positionsVB->Release();
}

__global__ void createVertices(float4* positions, float time,
                               unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // Calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // Write positions
    positions[y * width + x] =
                make_float4(u, w, v, __int_as_float(0xff00ff00));
}

##### 6.2.15.2.3.Direct3D 11版本[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-11-version "这个标题的永久链接")

ID3D11Device* device;
struct CUSTOMVERTEX {
    FLOAT x, y, z;
    DWORD color;
};
ID3D11Buffer* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;

int main()
{
    int dev;
    // Get a CUDA-enabled adapter
    IDXGIFactory* factory;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    IDXGIAdapter* adapter = 0;
    for (unsigned int i = 0; !adapter; ++i) {
        if (FAILED(factory->EnumAdapters(i, &adapter))
            break;
        if (cudaD3D11GetDevice(&dev, adapter) == cudaSuccess)
            break;
        adapter->Release();
    }
    factory->Release();

    // Create swap chain and device
    ...
    sFnPtr_D3D11CreateDeviceAndSwapChain(adapter,
                                         D3D11_DRIVER_TYPE_HARDWARE,
                                         0,
                                         D3D11_CREATE_DEVICE_DEBUG,
                                         featureLevels, 3,
                                         D3D11_SDK_VERSION,
                                         &swapChainDesc, &swapChain,
                                         &device,
                                         &featureLevel,
                                         &deviceContext);
    adapter->Release();

    // Use the same device
    cudaSetDevice(dev);

    // Create vertex buffer and register it with CUDA
    unsigned int size = width * height * sizeof(CUSTOMVERTEX);
    D3D11_BUFFER_DESC bufferDesc;
    bufferDesc.Usage          = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth      = size;
    bufferDesc.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags      = 0;
    device->CreateBuffer(&bufferDesc, 0, &positionsVB);
    cudaGraphicsD3D11RegisterResource(&positionsVB_CUDA,
                                      positionsVB,
                                      cudaGraphicsRegisterFlagsNone);
    cudaGraphicsResourceSetMapFlags(positionsVB_CUDA,
                                    cudaGraphicsMapFlagsWriteDiscard);

    // Launch rendering loop
    while (...) {
        ...
        Render();
        ...
    }
    ...
}

void Render()
{
    // Map vertex buffer for writing from CUDA
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVB_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,
                                         positionsVB_CUDA));

    // Execute kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // Unmap vertex buffer
    cudaGraphicsUnmapResources(1, &positionsVB_CUDA, 0);

    // Draw and present
    ...
}

void releaseVB()
{
    cudaGraphicsUnregisterResource(positionsVB_CUDA);
    positionsVB->Release();
}

    __global__ void createVertices(float4* positions, float time,
                          unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

// Calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // Calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // Write positions
    positions[y * width + x] =
                make_float4(u, w, v, __int_as_float(0xff00ff00));
}

#### 6.2.15.3.SLI互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sli-interoperability "这个标题的永久链接")

在具有多个GPU的系统中，所有支持CUDA的GPU都可以通过CUDA驱动程序和运行时作为单独的设备访问。然而，当系统处于SLI模式时，有如下所述的特殊注意事项。

首先，在一个GPU上的一个CUDA设备中的分配将消耗其他GPU的内存，这些GPU是Direct3D或OpenGL设备的SLI配置的一部分。因此，分配可能会比预期更早失败。

其次，应用程序应创建多个CUDA上下文，为SLI配置中的每个GPU创建一个。虽然这不是严格的要求，但它避免了设备之间不必要的数据传输。该应用程序可以使用`cudaD3D[9|10|11]GetDevices()`的Direct3D和`cudaGLGetDevices()`的OpenGL调用集，以识别当前和下一帧中执行渲染的设备的CUDA设备句柄。给定此信息，当`deviceList`参数设置为`cudaD3D[9|10|11]DeviceListCurrentFrame`或`cudaGLDeviceListCurrentFrame`时，应用程序通常会选择合适的设备，并将Direct3D或OpenGL资源映射到`cudaD3D[9|10|11]GetDevices()`或`cudaGLGetDevices()`返回的CUDA设备。

请注意，从`cudaGraphicsD9D[9|10|11]RegisterResource`和`cudaGraphicsGLRegister[Buffer|Image]`返回的资源只能在注册发生的设备上使用。因此，在SLI配置中，当不同帧的数据在不同的CUDA设备上计算时，有必要分别注册每个帧的资源。

有关CUDA运行时如何分别与Direct3D和OpenGL互操作的详细信息，请参阅[Direct3D互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-interoperability)和[OpenGL互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability)。

### 6.2.16.外部资源互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#external-resource-interoperability "这个标题的永久链接")

外部资源互操作性允许CUDA导入由其他API显式导出的某些资源。这些对象通常由其他API使用操作系统原生的句柄导出，例如Linux上的文件描述符或Windows上的NT句柄。它们也可以使用其他统一接口（如NVIDIA软件通信接口）进行导出。可以导入两种类型的资源：内存对象和同步对象。

内存对象可以使用`cudaImportExternalMemory()`导入CUDA。可以使用通过`cudaExternalMemoryGetMappedBuffer()`或CUDA映射的viacudaExternalMemoryGetMappedMipmappedArray`cudaExternalMemoryGetMappedMipmappedArray()`映射到内存对象上的设备指针，从内核中访问导入的内存对象。根据内存对象的类型，有可能在单个内存对象上设置多个映射。映射必须与导出API中的映射设置相匹配。任何不匹配的映射都会导致未定义的行为。必须使用`cudaDestroyExternalMemory()`释放导入的内存对象。释放内存对象不会释放到该对象的任何映射。因此，映射到该对象的任何设备指针都必须使用`cudaFree()`显式释放，映射到该对象的任何CUDA mipmapp的数组必须使用`cudaFreeMipmappedArray()`显式释放。在物体被摧毁后访问对映是违法的。

可以使用`cudaImportExternalSemaphore()`将同步对象导入CUDA。然后可以使用`cudaSignalExternalSemaphoresAsync()`发出信号，并使用`cudaWaitExternalSemaphoresAsync()`等待导入的同步对象。在发出相应信号之前发出等待是违法的。此外，根据导入的同步对象的类型，可能会对如何发出信号和等待它们施加额外的限制，如后续章节所述。必须使用`cudaDestroyExternalSemaphore()`释放导入的semaphore对象。在信号灯对象被摧毁之前，所有未完成的信号和等待必须完成。

#### 6.2.16.1.Vulkan互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vulkan-interoperability "这个标题的永久链接")

##### 6.2.16.1.1.匹配设备UUID[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#matching-device-uuids "这个标题的永久链接")

导入Vulkan导出的内存和同步对象时，它们必须导入并映射到创建的同一设备上。与创建对象的Vulkan物理设备相对应的CUDA设备可以通过将CUDA设备的UUID与Vulkan物理设备的UUID进行比较来确定，如以下代码示例所示。请注意，Vulkan物理设备不应成为包含多个Vulkan物理设备的设备组的一部分。包含给定Vulkan物理设备的`vkEnumeratePhysicalDeviceGroups`返回的设备组必须具有1的物理设备计数。

int getCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice) {
    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
    vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

    vkGetPhysicalDeviceProperties2(vkPhysicalDevice, &vkPhysicalDeviceProperties2);

    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);

    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        if (!memcmp(&deviceProp.uuid, vkPhysicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE)) {
            return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}

##### 6.2.16.1.2.导入内存对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-memory-objects "这个标题的永久链接")

在Linux和Windows 10上，Vulkan导出的专用和非专用内存对象都可以导入CUDA。在Windows 7上，只能导入专用内存对象。导入Vulkan专用内存对象时，必须设置标志`cudaExternalMemoryDedicated`。

使用`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`导出的Vulkan内存对象可以使用与该对象关联的文件描述符导入CUDA，如下所示。请注意，一旦导入文件描述符，CUDA就承担其所有权。导入成功后使用文件描述符会导致未定义的行为。

cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(int fd, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);

    // Input parameter 'fd' should not be used beyond this point as CUDA has assumed ownership of it

    return extMem;
}

使用`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`导出的Vulkan内存对象可以使用与该对象关联的NT句柄导入CUDA，如下所示。请注意，CUDA不承担NT句柄的所有权，当不再需要时，应用程序有责任关闭句柄。NT句柄包含对资源的引用，因此在释放底层内存之前，必须明确释放它。

cudaExternalMemory_t importVulkanMemoryObjectFromNTHandle(HANDLE handle, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extMem;
}

使用`VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT`导出的Vulkan内存对象也可以使用命名句柄导入，如果存在，如下所示。

cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(LPCWSTR name, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.name = (void *)name;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);

    return extMem;
}

使用VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT导出的Vulkan内存对象可以使用与该对象关联的全球共享D3DKMT句柄导入CUDA，如下所示。由于全局共享的D3DKMT句柄不包含对底层内存的引用，因此当对资源的所有其他引用被销毁时，它会自动销毁。

cudaExternalMemory_t importVulkanMemoryObjectFromKMTHandle(HANDLE handle, unsigned long long size, bool isDedicated) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    if (isDedicated) {
        desc.flags |= cudaExternalMemoryDedicated;
    }

    cudaImportExternalMemory(&extMem, &desc);

    return extMem;
}

##### 6.2.16.1.3.将缓冲區映射到导入的内存对象上[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-buffers-onto-imported-memory-objects "这个标题的永久链接")

设备指针可以映射到导入的内存对象上，如下所示。映射的偏移量和大小必须与使用相应的Vulkan API创建映射时指定的偏移量和大小相匹配。必须使用`cudaFree()`释放所有映射的设备指针。

void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {

    void *ptr = NULL;

    cudaExternalMemoryBufferDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;

    desc.size = size;

    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);

    // Note: ‘ptr’ must eventually be freed using cudaFree()

    return ptr;

}

##### 6.2.16.1.4.将Mipmapped数组映射到导入的内存对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-mipmapped-arrays-onto-imported-memory-objects "这个标题的永久链接")

CUDA mipmapped数组可以映射到导入的内存对象上，如下所示。偏移量、尺寸、格式和mip级别必须与使用相应的Vulkan API创建映射时指定的值相匹配。此外，如果mipmapped数组在Vulkan中绑定为颜色目标，则必须设置flagcudaArrayColorAttachment。必须使用`cudaFreeMipmappedArray()`释放所有映射的mipmapped数组。以下代码示例展示了在将mipmapp的数组映射到导入的内存对象时，如何将Vulkan参数转换为相应的CUDA参数。

cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;

    // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);

    return mipmap;
}

cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format)
{
    cudaChannelFormatDesc d;

    memset(&d, 0, sizeof(d));

    switch (format) {
    case VK_FORMAT_R8_UINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R8_SINT:             d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R8G8_UINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R8G8_SINT:           d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R8G8B8A8_UINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R8G8B8A8_SINT:       d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R16_UINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R16_SINT:            d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R16G16_UINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R16G16_SINT:         d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R16G16B16A16_UINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R16G16B16A16_SINT:   d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R32_UINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R32_SINT:            d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R32_SFLOAT:          d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
    case VK_FORMAT_R32G32_UINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R32G32_SINT:         d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R32G32_SFLOAT:       d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
    case VK_FORMAT_R32G32B32A32_UINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
    case VK_FORMAT_R32G32B32A32_SINT:   d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned;   break;
    case VK_FORMAT_R32G32B32A32_SFLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat;    break;
    default: assert(0);
    }

    return d;
}

cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers, VkImageViewType vkImageViewType) {
    cudaExtent e = { 0, 0, 0 };

    switch (vkImageViewType) {
    case VK_IMAGE_VIEW_TYPE_1D:         e.width = vkExt.width; e.height = 0;            e.depth = 0;           break;
    case VK_IMAGE_VIEW_TYPE_2D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = 0;           break;
    case VK_IMAGE_VIEW_TYPE_3D:         e.width = vkExt.width; e.height = vkExt.height; e.depth = vkExt.depth; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:       e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   e.width = vkExt.width; e.height = 0;            e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: e.width = vkExt.width; e.height = vkExt.height; e.depth = arrayLayers; break;
    default: assert(0);
    }

    return e;
}

unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType vkImageViewType, VkImageUsageFlags vkImageUsageFlags, bool allowSurfaceLoadStore) {
    unsigned int flags = 0;

    switch (vkImageViewType) {
    case VK_IMAGE_VIEW_TYPE_CUBE:       flags |= cudaArrayCubemap;                    break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:   flags |= cudaArrayLayered;                    break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   flags |= cudaArrayLayered;                    break;
    default: break;
    }

    if (vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
        flags |= cudaArrayColorAttachment;
    }

    if (allowSurfaceLoadStore) {
        flags |= cudaArraySurfaceLoadStore;
    }
    return flags;
}

##### 6.2.16.1.5.导入同步对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects "这个标题的永久链接")

使用`VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT`导出的Vulkan semaphore对象可以使用与该对象关联的文件描述符导入CUDA，如下所示。请注意，一旦导入文件描述符，CUDA就承担其所有权。导入成功后使用文件描述符会导致未定义的行为。

cudaExternalSemaphore_t importVulkanSemaphoreObjectFromFileDescriptor(int fd) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    desc.handle.fd = fd;

    cudaImportExternalSemaphore(&extSem, &desc);

    // Input parameter 'fd' should not be used beyond this point as CUDA has assumed ownership of it

    return extSem;
}

使用`VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT`导出的Vulkan訊號器对象可以使用与该对象关联的NT句柄导入CUDA，如下所示。请注意，CUDA不承担NT句柄的所有权，当不再需要时，应用程序有责任关闭句柄。NT句柄包含对资源的引用，因此在释放底层信号之前，必须明确释放它。

cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;

    cudaImportExternalSemaphore(&extSem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extSem;
}

使用`VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT`导出的Vulkan信号器对象也可以使用命名句柄导入，如果存在，如下所示。

cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.name = (void *)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

使用`VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT`导出的Vulkan semaphore对象可以使用与该对象关联的全局共享D3DKMT句柄导入CUDA，如下所示。由于全局共享的D3DKMT控制代碼不包含对底层訊號燈的引用，因此当对资源的所有其他引用被销毁时，它会自动被销毁。

cudaExternalSemaphore_t importVulkanSemaphoreObjectFromKMTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    desc.handle.win32.handle = (void *)handle;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

##### 6.2.16.1.6.在导入的同步对象上发出信号/等待[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#signaling-waiting-on-imported-synchronization-objects "这个标题的永久链接")

导入的Vulkan信号灯对象可以发出信号，如下所示。发出这样的信号灯对象将其设置为信号状态。等待此信号的相应等待必须在Vulkan中发出。此外，在此信号发出后，必须发出等待此信号。

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));

    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

导入的Vulkan semaphore对象可以等待，如下所示。在这样的信号灯对象上等待，直到它达到信号状态，然后将其重置为无信号状态。必须用Vulkan发出等待的相应信号。此外，在发出此等待之前，必须发出信号。

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));

    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

#### 6.2.16.2.OpenGL互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability-ext-res-int "这个标题的永久链接")

[OpenGL互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability)中概述的传统OpenGL-CUDA互操作由CUDA直接消耗OpenGL中创建的句柄。然而，由于OpenGL也可以消耗在Vulkan中创建的内存和同步对象，因此存在一种替代方法来进行OpenGL-CUDA互操作。从本质上讲，Vulkan导出的内存和同步对象可以导入到OpenGL和CUDA中，然后用于协调OpenGL和CUDA之间的内存访问。有关如何导入Vulkan导出的内存和同步对象的更多详细信息，请参阅以下OpenGL扩展：

- GL_EXT_内存_对象
    
- GL_EXT_内存_对象_fd
    
- GL_EXT_内存_对象_win32
    
- GL_EXT_中光
    
- GL_EXT_semaphore_fd
    
- GL_EXT_semaphore_win32
    

#### 6.2.16.3.Direct3D 12互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-12-interoperability "这个标题的永久链接")

##### 6.2.16.3.1.匹配设备LUID[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#matching-device-luids "这个标题的永久链接")

导入Direct3D 12导出的内存和同步对象时，它们必须导入并映射到创建的同一设备上。与创建对象的Direct3D 12设备对应的CUDA设备可以通过将CUDA设备的LUID与Direct3D 12设备的LUID进行比较来确定，如以下代码示例所示。请注意，Direct3D 12设备不得在链接节点适配器上创建。即`ID3D12Device::GetNodeCount`返回的节点计数必须为1。

int getCudaDeviceForD3D12Device(ID3D12Device *d3d12Device) {
    LUID d3d12Luid = d3d12Device->GetAdapterLuid();

    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);

    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        char *cudaLuid = deviceProp.luid;

        if (!memcmp(&d3d12Luid.LowPart, cudaLuid, sizeof(d3d12Luid.LowPart)) &&
            !memcmp(&d3d12Luid.HighPart, cudaLuid + sizeof(d3d12Luid.LowPart), sizeof(d3d12Luid.HighPart))) {
            return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}

##### 6.2.16.3.2.导入内存对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-memory-objects-dir3d-12-int "这个标题的永久链接")

通过在调用`ID3D12Device::CreateHeap`中设置标志`D3D12_HEAP_FLAG_SHARED`创建的可共享Direct3D 12堆内存对象，可以使用与该对象关联的NT句柄导入CUDA，如下所示。请注意，当不再需要NT句柄时，应用程序有责任关闭它。NT句柄包含对资源的引用，因此在释放底层内存之前，必须明确释放它。

cudaExternalMemory_t importD3D12HeapFromNTHandle(HANDLE handle, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;

    cudaImportExternalMemory(&extMem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extMem;
}

如果存在命名句柄，也可以使用可共享的Direct3D 12堆内存对象导入，如下所示。

cudaExternalMemory_t importD3D12HeapFromNamedNTHandle(LPCWSTR name, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeD3D12Heap;
    desc.handle.win32.name = (void *)name;
    desc.size = size;

    cudaImportExternalMemory(&extMem, &desc);

    return extMem;
}

通过在调用`D3D12Device::CreateCommittedResource`设置标志`D3D12_HEAP_FLAG_SHARED`创建的可共享的Direct3D 12提交资源，可以使用与该对象关联的NT句柄导入CUDA，如下所示。导入Direct3D 12提交资源时，必须设置标志`cudaExternalMemoryDedicated`。请注意，当不再需要NT句柄时，应用程序有责任关闭它。NT句柄包含对资源的引用，因此在释放底层内存之前，必须明确释放它。

cudaExternalMemory_t importD3D12CommittedResourceFromNTHandle(HANDLE handle, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;

    cudaImportExternalMemory(&extMem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extMem;
}

如果存在可共享的Direct3D 12提交资源，也可以使用命名手柄导入，如下所示。

cudaExternalMemory_t importD3D12CommittedResourceFromNamedNTHandle(LPCWSTR name, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    desc.handle.win32.name = (void *)name;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;

    cudaImportExternalMemory(&extMem, &desc);

    return extMem;
}

##### 6.2.16.3.3.将缓冲區映射到导入的内存对象上[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-buffers-onto-imported-memory-objects-dir3d-12-int "这个标题的永久链接")

设备指针可以映射到导入的内存对象上，如下所示。映射的偏移量和大小必须与使用相应的Direct3D 12 API创建映射时指定的偏移量和大小相匹配。必须使用`cudaFree()`释放所有映射的设备指针。

void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.size = size;

    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);

    // Note: 'ptr' must eventually be freed using cudaFree()
    return ptr;
}

##### 6.2.16.3.4.将Mipmapped数组映射到导入的内存对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-mipmapped-arrays-onto-imported-memory-objects-dir3d-12-int "这个标题的永久链接")

CUDA mipmapped数组可以映射到导入的内存对象上，如下所示。偏移量、尺寸、格式和mip级别数量必须与使用相应的Direct3D 12 API创建映射时指定的偏移量、尺寸、格式和数量相匹配。此外，如果mipmapped数组可以在Direct3D 12中绑定为渲染目标，则必须设置标志`cudaArrayColorAttachment`。必须使用`cudaFreeMipmappedArray()`释放所有映射的mipmapped数组。以下代码示例展示了在将mipmapp的数组映射到导入的内存对象时，如何将Vulkan参数转换为相应的CUDA参数。

cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;

    // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);

    return mipmap;
}

cudaChannelFormatDesc getCudaChannelFormatDescForDxgiFormat(DXGI_FORMAT dxgiFormat)
{
    cudaChannelFormatDesc d;

    memset(&d, 0, sizeof(d));

    switch (dxgiFormat) {
    case DXGI_FORMAT_R8_UINT:            d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R8_SINT:            d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R8G8_UINT:          d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R8G8_SINT:          d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R8G8B8A8_UINT:      d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R8G8B8A8_SINT:      d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R16_UINT:           d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R16_SINT:           d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R16G16_UINT:        d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R16G16_SINT:        d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R16G16B16A16_UINT:  d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R16G16B16A16_SINT:  d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R32_UINT:           d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R32_SINT:           d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R32_FLOAT:          d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
    case DXGI_FORMAT_R32G32_UINT:        d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R32G32_SINT:        d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R32G32_FLOAT:       d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
    case DXGI_FORMAT_R32G32B32A32_UINT:  d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R32G32B32A32_SINT:  d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat;    break;
    default: assert(0);

    }

    return d;
}

cudaExtent getCudaExtentForD3D12Extent(UINT64 width, UINT height, UINT16 depthOrArraySize, D3D12_SRV_DIMENSION d3d12SRVDimension) {
    cudaExtent e = { 0, 0, 0 };

    switch (d3d12SRVDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE1D:        e.width = width; e.height = 0;      e.depth = 0;                break;
    case D3D12_SRV_DIMENSION_TEXTURE2D:        e.width = width; e.height = height; e.depth = 0;                break;
    case D3D12_SRV_DIMENSION_TEXTURE3D:        e.width = width; e.height = height; e.depth = depthOrArraySize; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE:      e.width = width; e.height = height; e.depth = depthOrArraySize; break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:   e.width = width; e.height = 0;      e.depth = depthOrArraySize; break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:   e.width = width; e.height = height; e.depth = depthOrArraySize; break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: e.width = width; e.height = height; e.depth = depthOrArraySize; break;
    default: assert(0);
    }

    return e;
}

unsigned int getCudaMipmappedArrayFlagsForD3D12Resource(D3D12_SRV_DIMENSION d3d12SRVDimension, D3D12_RESOURCE_FLAGS d3d12ResourceFlags, bool allowSurfaceLoadStore) {
    unsigned int flags = 0;

    switch (d3d12SRVDimension) {
    case D3D12_SRV_DIMENSION_TEXTURECUBE:      flags |= cudaArrayCubemap;                    break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:   flags |= cudaArrayLayered;                    break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:   flags |= cudaArrayLayered;                    break;
    default: break;
    }

    if (d3d12ResourceFlags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) {
        flags |= cudaArrayColorAttachment;
    }
    if (allowSurfaceLoadStore) {
        flags |= cudaArraySurfaceLoadStore;
    }

    return flags;
}

##### 6.2.16.3.5.导入同步对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects-dir3d-12-int "这个标题的永久链接")

通过在调用`ID3D12Device::CreateFence`设置标志`D3D12_FENCE_FLAG_SHARED`创建的可共享的Direct3D 12围栏对象，可以使用与该对象关联的NT句柄导入CUDA，如下所示。请注意，当不再需要时，应用程序有责任关闭手柄。NT句柄包含对资源的引用，因此在释放底层信号之前，必须明确释放它。

cudaExternalSemaphore_t importD3D12FenceFromNTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.handle = handle;

    cudaImportExternalSemaphore(&extSem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extSem;
}

如果存在可共享的Direct3D 12围栏对象，也可以使用命名手柄导入，如下所示。

cudaExternalSemaphore_t importD3D12FenceFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.name = (void *)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

##### 6.2.16.3.6.在导入的同步对象上发出信号/等待[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#signaling-waiting-on-imported-synchronization-objects-dir3d-12-int "这个标题的永久链接")

导入的Direct3D 12围栏对象可以发出信号，如下所示。发出这样的信号，将栅栏对象将其值设置为指定的值。等待此信号的相应等待必须在Direct3D 12中发出。此外，在此信号发出后，必须发出等待此信号。

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

导入的Direct3D 12围栏对象可以等待，如下所示。等待这样的栅栏对象，直到其值大于或等于指定值。必须在Direct3D 12中发出此等待的相应信号。此外，在发出此等待之前，必须发出信号。

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

#### 6.2.16.4.Direct3D 11互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-11-interoperability "这个标题的永久链接")

##### 6.2.16.4.1.匹配设备LUID[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#matching-device-luids-dir3d-11-int "这个标题的永久链接")

导入Direct3D 11导出的内存和同步对象时，它们必须导入并映射到创建的同一设备上。与创建对象的Direct3D 11设备对应的CUDA设备可以通过将CUDA设备的LUID与Direct3D 11设备的LUID进行比较来确定，如以下代码示例所示。

int getCudaDeviceForD3D11Device(ID3D11Device *d3d11Device) {
    IDXGIDevice *dxgiDevice;
    d3d11Device->QueryInterface(__uuidof(IDXGIDevice), (void **)&dxgiDevice);

    IDXGIAdapter *dxgiAdapter;
    dxgiDevice->GetAdapter(&dxgiAdapter);

    DXGI_ADAPTER_DESC dxgiAdapterDesc;
    dxgiAdapter->GetDesc(&dxgiAdapterDesc);

    LUID d3d11Luid = dxgiAdapterDesc.AdapterLuid;

    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);

    for (int cudaDevice = 0; cudaDevice < cudaDeviceCount; cudaDevice++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, cudaDevice);
        char *cudaLuid = deviceProp.luid;

        if (!memcmp(&d3d11Luid.LowPart, cudaLuid, sizeof(d3d11Luid.LowPart)) &&
            !memcmp(&d3d11Luid.HighPart, cudaLuid + sizeof(d3d11Luid.LowPart), sizeof(d3d11Luid.HighPart))) {
            return cudaDevice;
        }
    }
    return cudaInvalidDeviceId;
}

##### 6.2.16.4.2.导入内存对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-memory-objects-dir3d-11-int "这个标题的永久链接")

A shareable Direct3D 11 texture resource, viz, `ID3D11Texture1D`, `ID3D11Texture2D` or `ID3D11Texture3D`, can be created by setting either the `D3D11_RESOURCE_MISC_SHARED` or `D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX` (on Windows 7) or `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` (on Windows 10) when calling `ID3D11Device:CreateTexture1D`, `ID3D11Device:CreateTexture2D` or `ID3D11Device:CreateTexture3D` respectively. A shareable Direct3D 11 buffer resource, `ID3D11Buffer`, can be created by specifying either of the above flags when calling `ID3D11Device::CreateBuffer`. A shareable resource created by specifying the `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` can be imported into CUDA using the NT handle associated with that object as shown below. Note that it is the application’s responsibility to close the NT handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed. When importing a Direct3D 11 resource, the flag `cudaExternalMemoryDedicated` must be set.

cudaExternalMemory_t importD3D11ResourceFromNTHandle(HANDLE handle, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;

    cudaImportExternalMemory(&extMem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extMem;
}

如果存在可共享的Direct3D 11资源，也可以使用命名手柄导入，如下所示。

cudaExternalMemory_t importD3D11ResourceFromNamedNTHandle(LPCWSTR name, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
    desc.handle.win32.name = (void *)name;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;

    cudaImportExternalMemory(&extMem, &desc);

    return extMem;
}

通过指定`D3D11_RESOURCE_MISC_SHARED`或`D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX`创建的可共享Direct3D 11资源，可以使用与该对象关联的全球共享`D3DKMT`句柄导入CUDA，如下所示。由于全局共享的`D3DKMT`句柄不包含对底层内存的引用，因此当对资源的所有其他引用被销毁时，它会自动销毁。

cudaExternalMemory_t importD3D11ResourceFromKMTHandle(HANDLE handle, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeD3D11ResourceKmt;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;

    cudaImportExternalMemory(&extMem, &desc);

    return extMem;
}

##### 6.2.16.4.3.将缓冲區映射到导入的内存对象上[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-buffers-onto-imported-memory-objects-dir3d-11-int "这个标题的永久链接")

设备指针可以映射到导入的内存对象上，如下所示。映射的偏移量和大小必须与使用相应的Direct3D 11 API创建映射时指定的偏移量和大小相匹配。必须使用`cudaFree()`释放所有映射的设备指针。

void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.size = size;

    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);

    // Note: ‘ptr’ must eventually be freed using cudaFree()
    return ptr;
}

##### 6.2.16.4.4.将Mipmapped数组映射到导入的内存对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-mipmapped-arrays-onto-imported-memory-objects-dir3d-11-int "这个标题的永久链接")

CUDA mipmapped数组可以映射到导入的内存对象上，如下所示。偏移量、尺寸、格式和mip级别必须与使用相应的Direct3D 11 API创建映射时指定的值相匹配。此外，如果mipmapped数组可以在Direct3D 12中绑定为渲染目标，则必须设置标志`cudaArrayColorAttachment`。必须使用`cudaFreeMipmappedArray()`释放所有映射的mipmapped数组。以下代码示例展示了如何在将mipmapped数组映射到导入的内存对象时将Direct3D 11参数转换为相应的CUDA参数。

cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;

    // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);

    return mipmap;
}

cudaChannelFormatDesc getCudaChannelFormatDescForDxgiFormat(DXGI_FORMAT dxgiFormat)
{
    cudaChannelFormatDesc d;
    memset(&d, 0, sizeof(d));
    switch (dxgiFormat) {
    case DXGI_FORMAT_R8_UINT:            d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R8_SINT:            d.x = 8;  d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R8G8_UINT:          d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R8G8_SINT:          d.x = 8;  d.y = 8;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R8G8B8A8_UINT:      d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R8G8B8A8_SINT:      d.x = 8;  d.y = 8;  d.z = 8;  d.w = 8;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R16_UINT:           d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R16_SINT:           d.x = 16; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R16G16_UINT:        d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R16G16_SINT:        d.x = 16; d.y = 16; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R16G16B16A16_UINT:  d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R16G16B16A16_SINT:  d.x = 16; d.y = 16; d.z = 16; d.w = 16; d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R32_UINT:           d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R32_SINT:           d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R32_FLOAT:          d.x = 32; d.y = 0;  d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
    case DXGI_FORMAT_R32G32_UINT:        d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R32G32_SINT:        d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R32G32_FLOAT:       d.x = 32; d.y = 32; d.z = 0;  d.w = 0;  d.f = cudaChannelFormatKindFloat;    break;
    case DXGI_FORMAT_R32G32B32A32_UINT:  d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindUnsigned; break;
    case DXGI_FORMAT_R32G32B32A32_SINT:  d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindSigned;   break;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: d.x = 32; d.y = 32; d.z = 32; d.w = 32; d.f = cudaChannelFormatKindFloat;    break;
    default: assert(0);
    }

    return d;
}

cudaExtent getCudaExtentForD3D11Extent(UINT64 width, UINT height, UINT16 depthOrArraySize, D3D12_SRV_DIMENSION d3d11SRVDimension) {
    cudaExtent e = { 0, 0, 0 };

    switch (d3d11SRVDimension) {
    case D3D11_SRV_DIMENSION_TEXTURE1D:        e.width = width; e.height = 0;      e.depth = 0;                break;
    case D3D11_SRV_DIMENSION_TEXTURE2D:        e.width = width; e.height = height; e.depth = 0;                break;
    case D3D11_SRV_DIMENSION_TEXTURE3D:        e.width = width; e.height = height; e.depth = depthOrArraySize; break;
    case D3D11_SRV_DIMENSION_TEXTURECUBE:      e.width = width; e.height = height; e.depth = depthOrArraySize; break;
    case D3D11_SRV_DIMENSION_TEXTURE1DARRAY:   e.width = width; e.height = 0;      e.depth = depthOrArraySize; break;
    case D3D11_SRV_DIMENSION_TEXTURE2DARRAY:   e.width = width; e.height = height; e.depth = depthOrArraySize; break;
    case D3D11_SRV_DIMENSION_TEXTURECUBEARRAY: e.width = width; e.height = height; e.depth = depthOrArraySize; break;
    default: assert(0);
    }
    return e;
}

unsigned int getCudaMipmappedArrayFlagsForD3D12Resource(D3D11_SRV_DIMENSION d3d11SRVDimension, D3D11_BIND_FLAG d3d11BindFlags, bool allowSurfaceLoadStore) {
    unsigned int flags = 0;

    switch (d3d11SRVDimension) {
    case D3D11_SRV_DIMENSION_TEXTURECUBE:      flags |= cudaArrayCubemap;                    break;
    case D3D11_SRV_DIMENSION_TEXTURECUBEARRAY: flags |= cudaArrayCubemap | cudaArrayLayered; break;
    case D3D11_SRV_DIMENSION_TEXTURE1DARRAY:   flags |= cudaArrayLayered;                    break;
    case D3D11_SRV_DIMENSION_TEXTURE2DARRAY:   flags |= cudaArrayLayered;                    break;
    default: break;
    }

    if (d3d11BindFlags & D3D11_BIND_RENDER_TARGET) {
        flags |= cudaArrayColorAttachment;
    }

    if (allowSurfaceLoadStore) {
        flags |= cudaArraySurfaceLoadStore;
    }

    return flags;
}

##### 6.2.16.4.5.导入同步对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects-dir3d-11-int "这个标题的永久链接")

通过在调用`ID3D11Device5::CreateFence`中设置标志`D3D11_FENCE_FLAG_SHARED`创建的可共享的Direct3D 11围栏对象，可以使用与该对象关联的NT句柄导入CUDA，如下所示。请注意，当不再需要时，应用程序有责任关闭手柄。NT句柄包含对资源的引用，因此在释放底层信号之前，必须明确释放它。

cudaExternalSemaphore_t importD3D11FenceFromNTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
    desc.handle.win32.handle = handle;

    cudaImportExternalSemaphore(&extSem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extSem;
}

可共享的Direct3D 11围栏对象也可以使用命名句柄导入，如果存在，如下所示。

cudaExternalSemaphore_t importD3D11FenceFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
    desc.handle.win32.name = (void *)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

通过设置flagD3D11`D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX`创建的可共享Direct3D 11资源，即`IDXGIKeyedMutex`，可共享的Direct3D 11密钥互斥对象，可以使用与该对象关联的NT句柄导入CUDA，如下所示。请注意，当不再需要时，应用程序有责任关闭手柄。NT句柄包含对资源的引用，因此在释放底层信号之前，必须明确释放它。

cudaExternalSemaphore_t importD3D11KeyedMutexFromNTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeKeyedMutex;
    desc.handle.win32.handle = handle;

    cudaImportExternalSemaphore(&extSem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extSem;
}

也可以使用命名句柄导入可共享的Direct3D 11键互斥对象，如果存在如下所示。

cudaExternalSemaphore_t importD3D11KeyedMutexFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeKeyedMutex;
    desc.handle.win32.name = (void *)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

可共享的Direct3D 11键互斥对象可以使用与该对象关联的全球共享D3DKMT句柄导入CUDA，如下所示。由于全局共享的D3DKMT句柄不包含对底层内存的引用，因此当对资源的所有其他引用被销毁时，它会自动销毁。

cudaExternalSemaphore_t importD3D11FenceFromKMTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeKeyedMutexKmt;
    desc.handle.win32.handle = handle;

    cudaImportExternalSemaphore(&extSem, &desc);

    // Input parameter 'handle' should be closed if it's not needed anymore
    CloseHandle(handle);

    return extSem;
}

##### 6.2.16.4.6.在导入的同步对象上发出信号/等待[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#signaling-waiting-on-imported-synchronization-objects-dir3d-11-int "这个标题的永久链接")

导入的Direct3D 11围栏对象可以发出信号，如下所示。发出这样的信号，将栅栏对象将其值设置为指定的值。等待此信号的相應等待必須在Direct3D 11中发出。此外，在此信号发出后，必须发出等待此信号。

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

导入的Direct3D 11围栏对象可以等待，如下所示。等待这样的栅栏对象，直到其值大于或等于指定值。必须在Direct3D 11中发出此等待的相应信号。此外，在发出此等待之前，必须发出信号。

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

导入的Direct3D 11键互斥对象可以发出信号，如下所示。通过指定密钥值来发出此类密钥互斥体对象的信号，释放该值的密钥互斥体。等待此信号的相应等待必须在Direct3D 11中以相同的密钥值发出。此外，在发出此信号后，必须发出Direct3D 11等待。

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long key, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.keyedmutex.key = key;

    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

导入的Direct3D 11键互斥对象可以等待，如下所示。在等待这种键控互斥时，需要以毫秒为单位的超时值。等待操作等待，直到密钥互斥值等于指定的密钥值或超时已过。超时间隔也可以是无限值。如果指定了无限值，超时永远不会过去。必须使用windows INFINITE宏来指定无限超时。必须在Direct3D 11中发出此等待的相应信号。此外，在发出此等待之前，必须发出Direct3D 11信号。

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long key, unsigned int timeoutMs, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.keyedmutex.key = key;
    params.params.keyedmutex.timeoutMs = timeoutMs;

    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

#### 6.2.16.5.英伟达软件通信接口互操作性（NVSCI）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nvidia-software-communication-interface-interoperability-nvsci "这个标题的永久链接")

NvSciBuf和NvSciSync是为以下目的而开发的接口：

- NvSciBuf：允许应用程序在内存中分配和交换缓冲区
    
- NvSciSync：允许应用程序在操作边界管理同步对象
    

有关这些接口的更多详细信息，请访问：[https://docs.nvidia.com/drive](https://docs.nvidia.com/drive)。

##### 6.2.16.5.1.导入内存对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-memory-objects-nvsci "这个标题的永久链接")

要分配与给定CUDA设备兼容的NvSciBuf对象，必须在NvSciBuf属性列表中使用`NvSciBufGeneralAttrKey_GpuId`设置相应的GPU ID，如下所示。可选的，应用程序可以指定以下属性-

- `NvSciBufGeneralAttrKey_NeedCpuAccess`：指定缓冲区是否需要CPU访问
    
- `NvSciBufRawBufferAttrKey_Align`：指定对齐要求`NvSciBufType_RawBuffer`
    
- `NvSciBufGeneralAttrKey_RequiredPerm`：每个NvSciBuf内存对象实例可以为不同的UMD配置不同的访问权限。例如，为了为GPU提供对缓冲区的只读访问权限，请使用`NvSciBufObjDupWithReducePerm()`创建一个重复的NvSciBuf对象，`NvSciBufAccessPerm_Readonly`作为输入参数。然后将这个新创建的重复对象导入到CUDA中，权限减少，如图所示
    
- `NvSciBufGeneralAttrKey_EnableGpuCache`：控制GPU L2缓存性
    
- `NvSciBufGeneralAttrKey_EnableGpuCompression`：指定GPU压缩
    

笔记

有关这些属性及其有效输入选项的更多详细信息，请参阅NvSciBuf文档。

以下代码片段说明了它们的示例用法。

NvSciBufObj createNvSciBufObject() {
   // Raw Buffer Attributes for CUDA
    NvSciBufType bufType = NvSciBufType_RawBuffer;
    uint64_t rawsize = SIZE;
    uint64_t align = 0;
    bool cpuaccess_flag = true;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;

    NvSciRmGpuId gpuid[] ={};
    CUuuid uuid;
    cuDeviceGetUuid(&uuid, dev));

    memcpy(&gpuid[0].bytes, &uuid.bytes, sizeof(uuid.bytes));
    // Disable cache on dev
    NvSciBufAttrValGpuCache gpuCache[] = {{gpuid[0], false}};
    NvSciBufAttrValGpuCompression gpuCompression[] = {{gpuid[0], NvSciBufCompressionType_GenericCompressible}};
    // Fill in values
    NvSciBufAttrKeyValuePair rawbuffattrs[] = {
         { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
         { NvSciBufRawBufferAttrKey_Size, &rawsize, sizeof(rawsize) },
         { NvSciBufRawBufferAttrKey_Align, &align, sizeof(align) },
         { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuaccess_flag, sizeof(cpuaccess_flag) },
         { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
         { NvSciBufGeneralAttrKey_GpuId, &gpuid, sizeof(gpuid) },
         { NvSciBufGeneralAttrKey_EnableGpuCache &gpuCache, sizeof(gpuCache) },
         { NvSciBufGeneralAttrKey_EnableGpuCompression &gpuCompression, sizeof(gpuCompression) }
    };

    // Create list by setting attributes
    err = NvSciBufAttrListSetAttrs(attrListBuffer, rawbuffattrs,
            sizeof(rawbuffattrs)/sizeof(NvSciBufAttrKeyValuePair));

    NvSciBufAttrListCreate(NvSciBufModule, &attrListBuffer);

    // Reconcile And Allocate
    NvSciBufAttrListReconcile(&attrListBuffer, 1, &attrListReconciledBuffer,
                       &attrListConflictBuffer)
    NvSciBufObjAlloc(attrListReconciledBuffer, &bufferObjRaw);
    return bufferObjRaw;
}

NvSciBufObj bufferObjRo; // Readonly NvSciBuf memory obj
// Create a duplicate handle to the same memory buffer with reduced permissions
NvSciBufObjDupWithReducePerm(bufferObjRaw, NvSciBufAccessPerm_Readonly, &bufferObjRo);
return bufferObjRo;

分配的NvSciBuf内存对象可以使用NvSciBufObj句柄在CUDA中导入，如下所示。应用程序应查询分配的NvSciBufObj，以获取填充CUDA外部内存描述符所需的属性。请注意，属性列表和NvSciBuf对象应由应用程序维护。如果导入到CUDA的NvSciBuf对象也被其他驱动程序映射，那么基于`NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency`输出属性值，应用程序必须使用NvSciSync对象（参考[导入同步对象](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects-nvsci)）作为保持CUDA和其他驱动程序之间的一致性的适当屏障。

笔记

有关如何分配和维护NvSciBuf对象的更多详细信息，请参阅NvSciBuf API文档。

cudaExternalMemory_t importNvSciBufObject (NvSciBufObj bufferObjRaw) {

    /*************** Query NvSciBuf Object **************/
    NvSciBufAttrKeyValuePair bufattrs[] = {
                { NvSciBufRawBufferAttrKey_Size, NULL, 0 },
                { NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency, NULL, 0 },
                { NvSciBufGeneralAttrKey_EnableGpuCompression, NULL, 0 }
    };
    NvSciBufAttrListGetAttrs(retList, bufattrs,
        sizeof(bufattrs)/sizeof(NvSciBufAttrKeyValuePair)));
                ret_size = *(static_cast<const uint64_t*>(bufattrs[0].value));

    // Note cache and compression are per GPU attributes, so read values for specific gpu by comparing UUID
    // Read cacheability granted by NvSciBuf
    int numGpus = bufattrs[1].len / sizeof(NvSciBufAttrValGpuCache);
    NvSciBufAttrValGpuCache[] cacheVal = (NvSciBufAttrValGpuCache *)bufattrs[1].value;
    bool ret_cacheVal;
    for (int i = 0; i < numGpus; i++) {
        if (memcmp(gpuid[0].bytes, cacheVal[i].gpuId.bytes, sizeof(CUuuid)) == 0) {
            ret_cacheVal = cacheVal[i].cacheability);
        }
    }

    // Read compression granted by NvSciBuf
    numGpus = bufattrs[2].len / sizeof(NvSciBufAttrValGpuCompression);
    NvSciBufAttrValGpuCompression[] compVal = (NvSciBufAttrValGpuCompression *)bufattrs[2].value;
    NvSciBufCompressionType ret_compVal;
    for (int i = 0; i < numGpus; i++) {
        if (memcmp(gpuid[0].bytes, compVal[i].gpuId.bytes, sizeof(CUuuid)) == 0) {
            ret_compVal = compVal[i].compressionType);
        }
    }

    /*************** NvSciBuf Registration With CUDA **************/

    // Fill up CUDA_EXTERNAL_MEMORY_HANDLE_DESC
    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = bufferObjRaw;
    // Set the NvSciBuf object with required access permissions in this step
    memHandleDesc.handle.nvSciBufObject = bufferObjRo;
    memHandleDesc.size = ret_size;
    cudaImportExternalMemory(&extMemBuffer, &memHandleDesc);
    return extMemBuffer;
 }

##### 6.2.16.5.2.将缓冲區映射到导入的内存对象上[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-buffers-onto-imported-memory-objects-nvsci "这个标题的永久链接")

设备指针可以映射到导入的内存对象上，如下所示。映射的偏移量和大小可以根据分配的`NvSciBufObj`的属性进行填充。必须使用`cudaFree()`释放所有映射的设备指针。

void * mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size) {
    void *ptr = NULL;
    cudaExternalMemoryBufferDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.size = size;

    cudaExternalMemoryGetMappedBuffer(&ptr, extMem, &desc);

    // Note: 'ptr' must eventually be freed using cudaFree()
    return ptr;
}

##### 6.2.16.5.3.将Mipmapped数组映射到导入的内存对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-mipmapped-arrays-onto-imported-memory-objects-nvsci "这个标题的永久链接")

CUDA mipmapped数组可以映射到导入的内存对象上，如下所示。偏移量、尺寸和格式可以根据分配的`NvSciBufObj`的属性进行填充。必须使用`cudaFreeMipmappedArray()`释放所有映射的mipmapped数组。以下代码示例展示了在将mipmap的数组映射到导入的内存对象上时，如何将NvSciBuf属性转换为相应的CUDA参数。

笔记

mip级别的数量必须为1。

cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc *formatDesc, cudaExtent *extent, unsigned int flags, unsigned int numLevels) {
    cudaMipmappedArray_t mipmap = NULL;
    cudaExternalMemoryMipmappedArrayDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.offset = offset;
    desc.formatDesc = *formatDesc;
    desc.extent = *extent;
    desc.flags = flags;
    desc.numLevels = numLevels;

    // Note: 'mipmap' must eventually be freed using cudaFreeMipmappedArray()
    cudaExternalMemoryGetMappedMipmappedArray(&mipmap, extMem, &desc);

    return mipmap;
}

##### 6.2.16.5.4.导入同步对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects-nvsci "这个标题的永久链接")

可以使用`cudaDeviceGetNvSciSyncAttributes()`生成与给定CUDA设备兼容的NvSciSync属性。返回的属性列表可用于创建`NvSciSyncObj`，该NvSciSyncObj保证与给定的CUDA设备兼容。

NvSciSyncObj createNvSciSyncObject() {
    NvSciSyncObj nvSciSyncObj
    int cudaDev0 = 0;
    int cudaDev1 = 1;
    NvSciSyncAttrList signalerAttrList = NULL;
    NvSciSyncAttrList waiterAttrList = NULL;
    NvSciSyncAttrList reconciledList = NULL;
    NvSciSyncAttrList newConflictList = NULL;

    NvSciSyncAttrListCreate(module, &signalerAttrList);
    NvSciSyncAttrListCreate(module, &waiterAttrList);
    NvSciSyncAttrList unreconciledList[2] = {NULL, NULL};
    unreconciledList[0] = signalerAttrList;
    unreconciledList[1] = waiterAttrList;

    cudaDeviceGetNvSciSyncAttributes(signalerAttrList, cudaDev0, CUDA_NVSCISYNC_ATTR_SIGNAL);
    cudaDeviceGetNvSciSyncAttributes(waiterAttrList, cudaDev1, CUDA_NVSCISYNC_ATTR_WAIT);

    NvSciSyncAttrListReconcile(unreconciledList, 2, &reconciledList, &newConflictList);

    NvSciSyncObjAlloc(reconciledList, &nvSciSyncObj);

    return nvSciSyncObj;
}

NvSciSync对象（如上创建）可以使用如下所示的NvSciSyncObj句柄导入CUDA。请注意，即使导入后，NvSciSyncObj句柄的所有权仍与应用程序有。

cudaExternalSemaphore_t importNvSciSyncObject(void* nvSciSyncObj) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    desc.handle.nvSciSyncObj = nvSciSyncObj;

    cudaImportExternalSemaphore(&extSem, &desc);

    // Deleting/Freeing the nvSciSyncObj beyond this point will lead to undefined behavior in CUDA

    return extSem;
}

##### 6.2.16.5.5.在导入的同步对象上发出信号/等待[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#signaling-waiting-on-imported-synchronization-objects-nvsci "这个标题的永久链接")

导入的`NvSciSyncObj`对象可以发出如下所述的信号。信号NvSciSync支持的信号灯对象初始化作为输入传递的_栅栏_参数。这个围栏参数由与上述信号对应的等待操作等待。此外，在此信号发出后，必须发出等待此信号。如果标志设置为`cudaExternalSemaphoreSignalSkipNvSciBufMemSync`，则默认作为信号操作的一部分执行的内存同步操作（在此过程中导入的所有NvSciBuf）将被跳过。当`NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency`为FALSE时，应设置此标志。

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, void *fence) {
    cudaExternalSemaphoreSignalParams signalParams = {};

    memset(&signalParams, 0, sizeof(signalParams));

    signalParams.params.nvSciSync.fence = (void*)fence;
    signalParams.flags = 0; //OR cudaExternalSemaphoreSignalSkipNvSciBufMemSync

    cudaSignalExternalSemaphoresAsync(&extSem, &signalParams, 1, stream);

}

导入的`NvSciSyncObj`对象可以等待，如下所述。等待NvSciSync支持的信号灯对象，直到输入_围栏_参数由相应的信号器发出信号。此外，在发出等待之前，必须发出信号。如果标志设置为`cudaExternalSemaphoreWaitSkipNvSciBufMemSync`，则默认作为信号操作的一部分执行的内存同步操作（在此过程中导入的所有NvSciBuf）将被跳过。当`NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency`为FALSE时，应设置此标志。

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, void *fence) {
     cudaExternalSemaphoreWaitParams waitParams = {};

    memset(&waitParams, 0, sizeof(waitParams));

    waitParams.params.nvSciSync.fence = (void*)fence;
    waitParams.flags = 0; //OR cudaExternalSemaphoreWaitSkipNvSciBufMemSync

    cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, stream);
}

## 6.3.版本和兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#versioning-and-compatibility "这个标题的永久链接")

开发人员在开发CUDA应用程序时应该注意两个版本号：描述计算设备的一般规格和功能的计算能力（请参阅[计算能力](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability)）和描述驱动程序API和运行时支持的功能的CUDA驱动程序API版本。

驱动程序API的版本在驱动程序头文件中定义为`CUDA_VERSION`。它允许开发人员检查他们的应用程序是否需要比当前安装的更新的设备驱动程序。这很重要，因为驱动程序API向_后兼容_，这意味着根据特定版本的驱动程序API编译的应用程序、插件和库（包括CUDA运行时）将继续在后续设备驱动程序版本中工作，如[图26](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#versioning-and-compatibility-driver-api-is-backward-but-not-forward-compatible)所示。驱动程序API不_向前兼容_，这意味着针对特定版本的驱动程序API编译的应用程序、插件和库（包括CUDA运行时）将无法在以前版本的设备驱动程序上工作。

需要注意的是，支持的版本的混合和匹配有限制：

- 由于CUDA驱动程序一次只能在系统上安装一个版本，因此已安装的驱动程序必须与构建任何必须在该系统上运行的应用程序、插件或库的最大驱动程序API版本相同或更高的版本。
    
- 应用程序使用的所有插件和库都必须使用相同版本的CUDA运行时，除非它们静态链接到运行时，在这种情况下，运行时的多个版本可以在同一进程空间中共存。请注意，如果使用`nvcc`链接应用程序，默认将使用CUDA运行时库的静态版本，所有CUDA工具包库都与CUDA运行时静态链接。
    
- 应用程序使用的所有插件和库必须使用使用运行时的任何库的相同版本（如cuFFT、cuBLAS......），除非静态链接到这些库。
    

![驱动程序API向后兼容，但不向前兼容](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/compatibility-of-cuda-versions.png)

图26驱动程序API向后兼容，但不向前兼容[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#versioning-and-compatibility-driver-api-is-backward-but-not-forward-compatible "此图像的永久链接")

对于特斯拉GPU产品，CUDA 10为CUDA驱动程序的用户模式组件引入了新的向前兼容升级路径。[CUDA兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)中描述了此功能。此处描述的CUDA驱动程序版本要求适用于用户模式组件的版本。

## 6.4.计算模式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-modes "这个标题的永久链接")

在运行Windows Server 2008及更高版本或Linux的特斯拉解决方案上，可以使用NVIDIA的系统管理接口（nvidia-smi）在以下三种模式之一中设置系统中的任何设备，nvidia-smi是作为驱动程序的一部分分发的工具：

- _默认_计算模式：多个主机线程可以同时使用该设备（使用运行时API时，通过在此设备上调用`cudaSetDevice()`），或在使用驱动程序API时将当前设置为与设备关联的上下文）。
    
- _专属进程_计算模式：在系统中的所有进程中，只能在设备上创建一个CUDA上下文。在创建该上下文的进程中，上下文可以是最新的，可以到所需的尽可能多的线程。
    
- _禁止的_计算模式：无法在设备上创建CUDA上下文。
    

这特别意味着，如果设备0处于禁止模式或专属进程模式，并由另一个进程使用，而不明确调用`cudaSetDevice()`的运行时API的主机线程可能会与设备0以外的设备相关联。`cudaSetValidDevices()`可用于从优先级设备列表中设置设备。

Note also that, for devices featuring the Pascal architecture onwards (compute capability with major revision number 6 and higher), there exists support for Compute Preemption. This allows compute tasks to be preempted at instruction-level granularity, rather than thread block granularity as in prior Maxwell and Kepler GPU architecture, with the benefit that applications with long-running kernels can be prevented from either monopolizing the system or timing out. However, there will be context switch overheads associated with Compute Preemption, which is automatically enabled on those devices for which support exists. The individual attribute query function `cudaDeviceGetAttribute()` with the attribute `cudaDevAttrComputePreemptionSupported` can be used to determine if the device in use supports Compute Preemption. Users wishing to avoid context switch overheads associated with different processes can ensure that only one process is active on the GPU by selecting exclusive-process mode.

应用程序可以通过检查属性`cudaDevAttrComputeMode`来查询设备的计算模式。

## 6.5.模式开关[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mode-switches "这个标题的永久链接")

具有显示输出的GPU将一些DRAM内存用于所谓的_主表面_，用于刷新用户查看输出的显示设备。当用户通过更改显示器的分辨率或比特深度（使用NVIDIA控制面板或Windows上的显示器控制面板）来启动显示器的_模式切换_时，主表面所需的内存量会发生变化。例如，如果用户将显示分辨率从1280x1024x32位更改为1600x1200x32位，系统必须将7.68 MB用于主表面，而不是5.24 MB。（在启用抗锯齿的情况下运行的全屏图形应用程序可能需要更多的主表面显示内存。）在Windows上，可能启动显示模式切换的其他事件包括启动全屏DirectX应用程序，按Alt+Tab以将任务从全屏DirectX应用程序切换，或按Ctrl+Alt+Del锁定计算机。

如果模式开关增加了主表面所需的内存量，系统可能不得不吞噬专用于CUDA应用程序的内存分配。因此，模式切换会导致对CUDA运行时的任何调用都失败并返回无效的上下文错误。

## 6.6.适用于Windows的特斯拉计算集群模式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tesla-compute-cluster-mode-for-windows "这个标题的永久链接")

使用NVIDIA的系统管理接口（_nvidia-smi_），Windows设备驱动程序可以为特斯拉和Quadro系列设备进入TCC（特斯拉计算集群）模式。

TCC模式取消了对任何图形功能的支持。

# 7.硬件实施[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-implementation "这个标题的永久链接")

NVIDIA GPU架构是围绕多线程流_多处理器_（_SM_）的可扩展阵列构建的。当主机CPU上的CUDA程序调用内核网格时，网格的块被枚举并分发到具有可用执行能力的多处理器。线程块的线程在一个多处理器上同时执行，多个线程块可以在一个多处理器上同时执行。随着线程块的终止，新的块会在空的多处理器上启动。

多处理器被设计为同时执行数百个线程。为了管理如此大量的线程，它采用了[SIMT架构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture)中描述的名为_SIMT_（_单指令、多线程_）的独特架构。指令是流水线的，利用单个线程中的指令级并行性，以及通过同时硬件多线程进行广泛的线程级并行性，如[硬件多线程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading)中所述。与CPU内核不同，它们是按顺序发布的，没有分支预测或投机执行。

[SIMT架构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture)和[硬件多线程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading)描述了所有设备通用的流媒体多处理器的架构功能。[计算能力5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x)、[计算能力6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)和[计算能力7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)分别为计算能力5.x、6.x和7.x的设备提供了具体说明。

NVIDIA GPU架构使用小端表示。

## 7.1.SIMT架构[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture "这个标题的永久链接")

多处理器在一组称为_warps_的32个并行线程中创建、管理、调度和执行线程。组成扭曲的单个线程在同一程序地址一起启动，但它们有自己的指令地址计数器和寄存器状态，因此可以自由地独立分支和执行。_经编_一词源于编织，这是第一个平行线技术。_半翘曲_是翘曲的前半部分或后半部分。_四分之一经编_是经编的第一、第二、第三或第四四分之一。

当多处理器被赋予一个或多个线程块执行时，它会将它们划分为经编，每个经编由_经编调度器_安排执行。块被划分为经编的方式总是相同的；每个经编包含连续的线程，增加线程ID，第一个经编包含线程0。[线程层次结构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)描述了线程ID与块中的线程索引之间的关系。

经编一次执行一个通用指令，因此当经编的所有32个线程同意其执行路径时，可以实现完全的效率。如果经编的线程通过数据依赖的条件分支发散，经编会执行每个分支路径，禁用不在该路径上的线程。分支发散只发生在一个经编内；不同的经编独立执行，无论它们是执行共同的还是不连续的代码路径。

SIMT架构类似于SIMD（单指令，多数据）向量组织，即单个指令控制多个处理元素。一个关键的区别是，SIMD向量组织向软件公开SIMD宽度，而SIMT指令指定了单个线程的执行和分支行为。与SIMD向量机相比，SIMT使程序员能够为独立的标量线程编写线程级并行代码，以及为协调线程编写数据并行代码。为了正确性，程序员基本上可以忽略SIMT行为；然而，通过注意代码很少需要经编中的线程来发散，可以实现实质性能的性能改进。在实践中，这与快取行在传统代码中的作用相似：在设计正确性时可以安全地忽略缓存行大小，但在设计峰值性能时，必须在代码结构中考虑。另一方面，矢量架构要求软件将负载合并为矢量并手动管理发散。

在NVIDIA Volta之前，warps使用warp中所有32个线程共享的单个程序计数器，以及指定warp的活动线程的主动掩码。因此，来自不同区域或不同执行状态的同一经编的线程无法相互发出信号或交换数据，需要由锁或互斥保护的数据精细共享的算法很容易导致僵局，这取决于竞争线程来自哪个经编。

从NVIDIA Volta架构开始，_独立线程调度_允许线程之间完全并发，无论是否翘曲。借助独立线程调度，GPU可以维护每个线程的执行状态，包括程序计数器和调用堆栈，并且可以以每个线程的粒度生成执行，要么更好地利用执行资源，要么允许一个线程等待另一个线程生成数据。时间表优化器决定了如何将来自同一经编的活动线程分组为SIMT单元。这保留了与以前的NVIDIA GPU一样SIMT执行的高吞吐量，但具有更大的灵活性：线程现在可以在子扭曲粒度下发散和重新收敛。

如果开发人员对以前硬件架构的warp-synchronicity2做出假设，独立线程调度可能会导致参与执行代码的线程与预期不同。特别是，任何经编同步代码（如无同步、经编内还原）都应重新访问，以确保与NVIDIA Volta及以后的兼容性。有关更多详细信息[，](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)请参阅[计算能力7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)。

笔记

参与当前指令的经编线程称为_活动_线程，而不在当前指令中的线程则处于_非活动状态_（禁用）。线程可能因各种原因而处于非活动状态，包括比其他线程更早退出，采取与当前由经编执行的分支路径不同的分支路径，或者是线程数不是经编大小的倍数的块的最后一个线程。

如果由经编执行的非原子指令为经编的多个线程写入全局或共享内存中的同一位置，则发生在该位置的序列化写入次数因设备的计算能力而异（请参阅[计算能力5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x)、[计算能力6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)和[计算能力7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)），并且执行最终写入的线程尚未定义。

如果由经编执行的[原子](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)指令读取、修改和写入到全局内存中的多个线程的同一位置，则每次读取/修改/写入该位置都会发生，它们都是序列化，但它们发生的顺序未定义。

## 7.2.硬件多线程[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading "这个标题的永久链接")

多处理器处理的每个经编的执行上下文（程序计数器、寄存器等）在整个经编的整个生命周期内都保持在芯片上。因此，从一个执行上下文切换到另一个执行上下文是免费的，在每次指令发布时间，经编调度器都会选择一个线程准备执行下一个指令的经编（经编的[活动线程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes)），并向这些线程发出指令。

特别是，每个多处理器都有一组32位寄存器，这些寄存器被划分在经编之间，以及一个_并行数据缓存_或_共享内存_，这些寄存器被划分在线程块中。

对于给定的内核，可以在多处理器上一起驻留和处理的块和扭曲的数量取决于内核使用的寄存器和共享内存的数量以及多处理器上可用的寄存器和共享内存的数量。每个多处理器还有一个最大驻留块数量和最大驻留扭曲数量。这些限制以及多处理器上可用的寄存器和共享内存数量是设备计算能力的函数，并在[计算能力](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities)中给出。如果每个多处理器没有足够的寄存器或共享内存来处理至少一个块，内核将无法启动。

块中翘曲总数如下：

ceil(TWsize,1)

- _T是_每个块的线程数，
    
- _Wsize_是经度大小，等于32，
    
- ceil(x, y)等于x四舍五入到y的最接近的倍数。
    

分配给一个区块的寄存器总数和共享内存总数记录在CUDA工具包中提供的CUDA占用率计算器中。

[2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id126)

术语_warp-synchronous_是指隐式假设同一warp中的线程在每个指令中同步的代码。

# 8.绩效准则[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines "这个标题的永久链接")

## 8.1.整体绩效优化策略[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overall-performance-optimization-strategies "这个标题的永久链接")

性能优化围绕四个基本策略展开：

- 最大化并行执行，以实现最大利用率；
    
- 优化内存使用，以实现最大的内存吞吐量；
    
- 优化指令使用，以实现最大的指令吞吐量；
    
- 尽量减少内存崩溃。
    

哪种策略将为应用程序的特定部分产生最佳性能增益取决于该部分的性能限制器；例如，优化主要由内存访问限制的内核的指令使用不会产生任何显著的性能增益。因此，应通过测量和监测性能限制器，例如使用CUDA分析器来不断指导优化工作。此外，将特定内核的浮点操作吞吐量或内存吞吐量（以哪个更有意义）与设备的相应峰值理论吞吐量进行比较，表明内核有多少改进空间。

## 8.2.最大化利用[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximize-utilization "这个标题的永久链接")

为了最大限度地提高利用率，应用程序的结构应该尽可能多地暴露并行性，并有效地将这种并行性映射到系统的各个组件，以使它们大部分时间都处于忙碌。

### 8.2.1.应用级别[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-level "这个标题的永久链接")

在高层次上，应用程序应使用非[同步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)中所述的非同步函数调用和流，最大限度地在主机、设备和将主机连接到设备的总线之间实现并行执行。它应该为每个处理器分配它最能完成的工作类型：主机的串行工作负载；设备的并行工作负载。

对于并行工作负载，在算法中由于一些线程需要同步才能相互共享数据而中断的点上，有两种情况：要么这些线程属于同一块，在这种情况下，它们应该使用`__syncthreads()`并通过同一内核调用中的共享内存共享数据，要么它们属于不同的块，在这种情况下，它们必须使用两个单独的内核调用通过全局内存共享数据，一个用于写入，一个用于从全局内存读取。第二种情况不太理想，因为它增加了额外的内核调用和全局内存流量的开销。因此，应通过将算法映射到CUDA编程模型来尽量减少其发生，以便需要线程间通信的计算尽可能在单个线程块内执行。

### 8.2.2.设备级别[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-level "这个标题的永久链接")

在较低级别，应用程序应最大限度地在设备的多处理器之间实现并行执行。

多个内核可以在一台设备上同时执行，因此也可以通过使用流实现足够的内核同时执行，如[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)中所述。

### 8.2.3.多处理器级别[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multiprocessor-level "这个标题的永久链接")

在更低的级别，应用程序应该最大化多处理器中各种功能单元之间的并行执行。

如[硬件多线程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading)中所述，GPU多处理器主要依赖于线程级并行性，以最大限度地利用其功能单元。因此，利用率与居民翘曲的数量直接相关。在每个指令问题时，经编调度器都会选择一个准备执行的指令。该指令可以是同一经编的另一个独立指令，利用指令级并行性，或者更常见的是另一个经编的指令，利用线程级并行性。如果选择了准备执行的指令，它将发给经编的[活动](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes)线程。扭曲准备执行其下一个指令所需的时钟周期数称为_延迟_，当所有扭曲调度器在该延迟期间的每个时钟周期总是为一些扭曲发出一些指令时，或者换句话说，当延迟完全“隐藏”时，就可以实现充分利用。隐藏L时钟周期延迟所需的指令数量取决于这些指令的相应吞吐量（有关各种算术指令的吞吐量，请参阅[CUDA C++最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions)）。如果我们假设指令具有最大吞吐量，它等于：

- _4L_适用于计算能力5.x、6.1、6.2、7.x和8.x的设备，因为对于这些设备，多处理器在一次四个经编的时钟周期内，每次发出一个指令，如[计算能力](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities)中提到的。
    
- _2L_对于计算能力6.0的设备，因为对于这些设备，每个周期发出的两个指令是两个不同翘曲的一个指令。
    

扭曲尚未准备好执行其下一个指令的最常见原因是指令的输入操作数尚未可用。

如果所有输入操作数都是寄存器，则延迟是由寄存器依赖性引起的，即一些输入操作数是由一些尚未完成执行的先前指令编写的。在这种情况下，延迟等于上一个指令的执行时间，经编调度器必须在这段时间内安排其他经编的指令。执行时间因指令而异。在具有7.x计算能力的设备上，对于大多数算术指令，它通常是4个时钟周期。这意味着每个多处理器需要16个主动经编（4个周期，4个经编调度器）来隐藏算术指令延迟（假设经编以最大吞吐量执行指令，否则需要更少的经编）。如果单个经编表现出指令级并行性，即其指令流中有多个独立指令，则需要更少的经编，因为单个经编可以背靠背发出多个独立指令。

如果一些输入操作数位于芯片外内存中，延迟会高得多：通常是数百个时钟周期。在如此高延迟的延迟期间，保持经编调度程序繁忙所需的经编次数取决于内核代码及其指令级并行程度。一般来说，如果没有芯片外内存操作数的指令数量（即大多数时候是算术指令）与具有芯片外内存操作数的指令数量之比较低（这个比率通常称为程序的算术强度），则需要更多的翘曲。

扭曲还没有准备好执行下一个指令的另一个原因是，它正在某个内存围栏（[内存围栏函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-functions)）或同步点（[同步函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-functions)）等待。同步点可以迫使多处理器闲置，因为越来越多的扭曲等待同一块中的其他扭曲在同步点之前完成指令的执行。在这种情况下，每个多处理器有多个驻留块有助于减少工时间，因为来自不同块的翘曲不需要在同步点相互等待。

The number of blocks and warps residing on each multiprocessor for a given kernel call depends on the execution configuration of the call ([Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)), the memory resources of the multiprocessor, and the resource requirements of the kernel as described in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading). Register and shared memory usage are reported by the compiler when compiling with the `--ptxas-options=-v` option.

一个块所需的共享内存总量等于静态分配的共享内存量和动态分配的共享内存量之和。

内核使用的寄存器数量会对驻留翘曲的数量产生重大影响。例如，对于计算能力为6.x的设备，如果内核使用64个寄存器，每个块有512个线程，并且需要很少的共享内存，那么两个块（即32个经编）可以驻留在多处理器上，因为它们需要2x512x64寄存器，这与多处理器上可用的寄存器数量完全匹配。但是，一旦内核再使用一个寄存器，就只有一个块（即16个经编）可以驻留，因为两个块需要2x512x65寄存器，这些寄存器比多处理器上可用的寄存器多。因此，编译器试图最大限度地减少寄存器的使用，同时将寄存器溢出（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)）和指令数量保持在最低限度。寄存器的使用可以使用`maxrregcount`编译器选项、[Launch Binunds](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-bounds)中描述的`__launch_bounds__()`限定符或[每个线程的最大寄存器数量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximum-number-of-registers-per-thread)中描述的`__maxnreg__()`限定符来控制。

寄存器文件被组织成32位寄存器。因此，存储在寄存器中的每个变量至少需要一个32位寄存器，例如，一个`double`使用两个32位寄存器。

执行配置对给定内核调用性能的影响通常取决于内核代码。因此，建议进行实验。应用程序还可以根据寄存器文件大小和共享内存大小对执行配置进行参数化，这取决于设备的计算能力，以及设备的多处理器数量和内存带宽，所有这些都可以使用运行时进行查询（请参阅参考手册）。

每个块的线程数应选择为经编大小的倍数，以避免尽可能浪费计算资源的人口不足的经编。

#### 8.2.3.1.占用计算器[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#occupancy-calculator "这个标题的永久链接")

存在几个API功能，以帮助程序员根据寄存器和共享内存要求选择线程块大小和集群大小。

- 占用计算器API，`cudaOccupancyMaxActiveBlocksPerMultiprocessor`，可以根据内核的块大小和共享内存使用情况提供占用预测。该函数以每个多处理器的并发线程块数量报告占用率。
    
    - 请注意，此值可以转换为其他指标。乘以每个块的经编数，得出每个多处理器的并发经编数；进一步将并发经编除以每个多处理器的最大经编，得出占用率为百分比。
        
- 基于占用的启动配置器API，`cudaOccupancyMaxPotentialBlockSize`和`cudaOccupancyMaxPotentialBlockSizeVariableSMem`，以启发式方式计算实现最大多处理器级占用的执行配置。
    
- 占用计算器API，`cudaOccupancyMaxActiveClusters`，可以根据内核的集群大小、块大小和共享内存使用情况提供占用预测。该功能以系统中GPU上给定大小的最大活动集群数量来报告占用率。
    

以下代码示例计算了MyKernel的占用率。然后，它报告占用水平，以及每个多处理器的并发翘曲率与最大翘曲率之间的比率。

// Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

// Host code
int main()
{
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;

    return 0;
}

以下代码示例根据用户输入配置了MyKernel的基于占用的内核启动。

// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// Host code
int launchMyKernel(int *array, int arrayCount)
{
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device
                        // launch
    int gridSize;       // The actual grid size needed, based on input
                        // size

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)MyKernel,
        0,
        arrayCount);

    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
    cudaDeviceSynchronize();

    // If interested, the occupancy can be calculated with
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor

    return 0;
}

以下代码示例展示了如何使用集群占用API来查找给定大小的活动集群的最大数量。下面的示例代码计算大小为2的集群占用率，每个区块128个线程。

Cluster size of 8 is forward compatible starting compute capability 9.0, except on GPU hardware or MIG configurations which are too small to support 8 multiprocessors in which case the maximum cluster size will be reduced. But it is recommended that the users query the maximum cluster size before launching a cluster kernel. Max cluster size can be queried using `cudaOccupancyMaxPotentialClusterSize` API.

{
  cudaLaunchConfig_t config = {0};
  config.gridDim = number_of_blocks;
  config.blockDim = 128; // threads_per_block = 128
  config.dynamicSmemBytes = dynamic_shared_memory_size;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = 2; // cluster_size = 2
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;

  int max_cluster_size = 0;
  cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, (void *)kernel, &config);

  int max_active_clusters = 0;
  cudaOccupancyMaxActiveClusters(&max_active_clusters, (void *)kernel, &config);

  std::cout << "Max Active Clusters of size 2: " << max_active_clusters << std::endl;
}

CUDA Nsight计算用户界面还为任何不能依赖CUDA软件堆栈的用例提供了独立的占用计算器和`<CUDA_Toolkit_Path>/include/cuda_occupancy.h`中的启动配置器实现。Nsight计算版本的占用率计算器作为学习工具特别有用，可以可视化影响占用率的参数变化的影响（块大小、每个线程的寄存器和每个线程的共享内存）。

## 8.3.最大化内存吞吐量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximize-memory-throughput "这个标题的永久链接")

最大化应用程序整体内存吞吐量的第一步是尽量减少低带宽的数据传输。

这意味着尽量减少主机和设备之间的数据传输，如[主机和设备之间的数据传输](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-transfer-between-host-and-device)中所述，因为这些数据的带宽比全局内存和设备之间的数据传输低得多。

这也意味着通过最大限度地使用片上内存来最大限度地减少全局内存和设备之间的数据传输：共享内存和缓存（即在计算能力2.x及以上的设备上可用的L1缓存和L2缓存，所有设备上可用的纹理缓存和常量缓存）。

共享内存等同于用户管理的缓存：应用程序明确分配和访问它。如[CUDA运行时](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-runtime)所示，一个典型的编程模式是将来自设备内存的数据分阶段到共享内存中；换句话说，有一个块的每个线程：

- 将数据从设备内存加载到共享内存，
    
- 与块的所有其他线程同步，以便每个线程都可以安全地读取由不同线程填充的共享内存位置，
    
- 在共享内存中处理数据，
    
- 如有必要，请再次同步，以确保共享内存已更新结果，
    
- 将结果写回设备内存中。
    

对于一些应用程序（例如，全局内存访问模式与数据相关），传统的硬件管理缓存更适合利用数据局部性。如[计算能力7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)、[计算能力8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x)和[计算能力9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0)中提到的，对于计算能力7.x、8.x和9.0的设备，L1和共享内存都使用相同的片上内存，每个内核调用都可以配置多少内存专用于L1与共享内存。

内核对内存访问的吞吐量可能因每种内存类型的访问模式而异。因此，最大化内存吞吐量的下一步是根据[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)中描述的最佳内存访问模式，尽可能优化地组织内存访问。这种优化对全局内存访问尤为重要，因为与可用的片上带宽和算术指令吞吐量相比，全局内存带宽较低，因此非最佳全局内存访问通常对性能有重大影响。

### 8.3.1.主机和设备之间的数据传输[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-transfer-between-host-and-device "这个标题的永久链接")

应用程序应努力尽量减少主机和设备之间的数据传输。实现这一目标的一种方法是将更多代码从主机移动到设备，即使这意味着运行没有暴露足够的并行性以完全高效地在设备上执行的内核。中间数据结构可以在设备内存中创建，由设备操作，并被主机映射或复制到主机内存的情况下被销毁。

此外，由于与每次转移相关的开销，将许多小转移分批为单个大型转移总是比单独进行每次转移的效果更好。

在具有前端总线的系统上，如[页面锁定主机内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#page-locked-host-memory)中所述，通过使用页面锁定主机内存，可以实现主机和设备之间数据传输的更高性能。

此外，当使用映射的页面锁定内存（[映射的内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory)）时，不需要分配任何设备内存，也不需要在设备和主机内存之间显式复制数据。每次内核访问映射内存时，都会隐式执行数据传输。为了获得最大的性能，这些内存访问必须像访问全局内存一样合并（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)）。假设它们是，并且映射内存只读取或写入一次，使用映射的页面锁定内存而不是设备和主机内存之间的显式副本，可以赢得性能。

在设备内存和主机内存在物理上相同的集成系统上，主机和设备内存之间的任何副本都是多余的，应改用映射的页面锁定内存。应用程序可以通过检查集成设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)）等于1来查询设备是否`integrated`。

### 8.3.2.设备内存访问[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses "这个标题的永久链接")

访问可定址内存（即全局、本地、共享、常量或纹理内存）的指令可能需要多次重新发放，具体取决于内存地址在经编内线程中的分布情况。分布如何以这种方式影响指令吞吐量是特定于每种内存类型的，并在以下章节中进行了描述。例如，对于全局内存，一般来说，地址越分散，吞吐量就越低。

**全球记忆**

全局内存位于设备内存中，设备内存通过32、64或128字节的内存事务访问。这些内存事务必须自然对齐：只有与其大小对齐的32、64或128字节的设备内存段（即其第一个地址是其大小的倍数）才能通过内存事务读取或写入。

当经编执行访问全局内存的指令时，它根据每个线程访问的字的大小和内存地址在线程中的分布情况，将经编内线程的内存访问合并为一个或多个这些内存事务。一般来说，需要的交易越多，除了线程访问的单词外，传输的未使用的单词就越多，相应地降低了指令吞吐量。例如，如果为每个线程的4字节访问生成32字节内存事务，则吞吐量除以8。

需要多少笔交易以及最终影响多少吞吐量因设备的计算能力而异。[计算能力5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x)、[计算能力6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)、[计算能力7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)、[计算能力8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x)、[计算能力9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0)、[计算能力10.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-10-0)和[计算能力12.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-12-0)提供了有关如何处理各种计算能力的全局内存访问的更多详细信息。

因此，为了最大化全局内存吞吐量，通过以下方式最大化融合很重要：

- 遵循基于[计算能力5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x)、[计算能力6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)、[计算能力7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)、[计算能力8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x)、[计算能力9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0)、[计算能力10.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-10-0)和[计算能力12.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-12-0)的最优化访问模式。
    
- 使用符合以下“尺寸和对齐要求”部分中详述的尺寸和对齐要求的数据类型，
    
- 在某些情况下，例如，在访问二维数组时，如下文“二维数组”部分所述，填充数据。
    

**尺寸和对齐要求**

全局内存指令支持读取或写入大小等于1、2、4、8或16字节的单词。当且仅当数据类型的大小为1、2、4、8或16字节且数据自然对齐（即其地址是该大小的倍数）时，对全局内存中数据的任何访问（通过变量或指针）都会编译为单个全局内存指令。

如果未满足此大小和对齐要求，访问将编译为具有交错访问模式的多个指令，防止这些指令完全融合。因此，对于位于全局内存中的数据，建议使用符合此要求的类型。

[内置矢量类型的](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-vector-types)对齐要求会自动满足。

对于结构，编译器可以使用对齐指定符`__align__(8)or__align__(16)`来强制执行大小和对齐要求，例如

struct __align__(8) {
    float x;
    float y;
};

或者

struct __align__(16) {
    float x;
    float y;
    float z;
};

位于全局内存中的变量或由驱动程序或运行时API的内存分配例程之一返回的任何地址始终与至少256字节对齐。

读取非自然对齐的8字节或16字节单词会产生不正确的结果（仅几个单词），因此必须特别注意保持这些类型的任何值或值数组的起始地址对齐。这可能容易被忽视的典型情况是使用一些自定义全局内存分配方案时，其中多个数组的分配（对`cudaMalloc()`或`cuMemAlloc()`多次调用）被分配分割成多个数组的单个内存块所取代，在这种情况下，每个数组的起始地址都与块的起始地址相偏移。

**二维阵列**

一个常见的全局内存访问模式是，索引的每个线程`(tx,ty)`使用以下地址访问宽度为2D数组的一个元素，该数组位于类型类型`type*`的地址`BaseAddress`（其中`type`符合[最大化利用率](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximize-utilization)中描述的要求）：

BaseAddress + width * ty + tx

为了使这些访问完全合并，线程块的宽度和数组的宽度必须是经编大小的倍数。

特别是，这意味着宽度不是此大小的倍数的数组，如果实际分配的宽度四舍五入到最接近此大小的倍数，并相应地填充其行，则可以更有效地访问。参考手册中描述的`cudaMallocPitch()`和`cuMemAllocPitch()`函数以及相关的内存复制函数使程序员能够编写非硬件依赖性代码来分配符合这些约束的数组。

**本地记忆**

本地内存访问仅适用于[可变内存空间指定符中](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#variable-memory-space-specifiers)提到的一些自动变量。编译器可能放置在本地内存中的自动变量是：

- 无法确定它们以恒定数量索引的数组，
    
- 会占用过多寄存器空间的大型结构或数组，
    
- 如果核心使用比可用更多的寄存器（這也稱為_寄存器溢位_），任何變數。
    

对_PTX_汇编代码的检查（通过使用`-ptx`或`-keep`选项编译获得）将告诉变量在第一个编译阶段是否已放置在本地内存中，因为它将使用`.local`符声明，并使用`ld.local`和`st.local`记符访问。即使没有，后续的编译阶段可能仍然会做出不同的决定，尽管如果他们发现它为目标架构占用了太多寄存器空间：使用`cuobjdump`检查_cubin_对象将判断是否是这种情况。此外，当使用`--ptxas-options=-v`选项编译时，编译器会报告每个内核（`lmem`）的总本地内存使用量。请注意，一些数学函数的实现路径可能会访问本地内存。

本地内存空间位于设备内存中，因此本地内存访问具有与全局内存访问相同的高延迟和低带宽，并且受制于[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)中所述的相同内存融合要求。然而，本地内存的组织方式是连续的32位单词由连续的线程ID访问。因此，只要经编中的所有线程访问相同的相对地址（例如，数组变量中的相同索引，结构变量中的相同成员），访问就完全合并。

在具有计算能力5.x以后的设备上，本地内存访问总是以与全局内存访问相同的方式在L2中缓存（请参阅[计算能力5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x)和[计算能力6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)）。

**共享内存**

由于它是片上存储器，共享内存比本地或全局内存具有更高的带宽和更低的延迟。

为了实现高带宽，共享内存被划分为大小相等的内存模块，称为库，可以同时访问。因此，任何由_n个_地址组成的内存读取或写入请求都可以同时服务，产生的总体带宽是单个模块带宽的_n倍_。

然而，如果内存请求的两个地址位于同一内存库中，则存在库冲突，并且必须对访问进行序列化。硬件将具有银行冲突的内存请求拆分为必要的多个单独的无冲突请求，将吞吐量降低等于单独内存请求数量的系数。如果单独的内存请求数量为_n_，则初始内存请求将导致_n-way_银行冲突。

因此，为了获得最大的性能，了解内存地址如何映射到内存库，以便安排内存请求，从而最大限度地减少内存库冲突，这一点很重要。[计算能力5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x)、[计算能力6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)、[计算能力7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)、[计算能力8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x)、[计算能力9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0)、[计算能力10.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-10-0)和[计算能力12.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-12-0)中分别描述了这些计算能力的设备。

**恒定记忆**

恒定内存空间位于设备内存中，并缓存在恒定缓存中。

然后，一个请求被拆分为与初始请求中不同的内存地址一样多的单独请求，将吞吐量降低到与单独请求数量相等的系数。

如果缓存被击中，则以恒定缓存的吞吐量或设备内存的吞吐量对生成的请求进行服务。

**纹理和表面记忆**

纹理和表面内存空间位于设备内存中，并缓存在纹理缓存中，因此纹理获取或表面读取仅在缓存错过时需要从设备内存中读取一个内存，否则只需从纹理缓存中读取一次。纹理缓存针对2D空间局部进行了优化，因此读取2D中紧密的纹理或表面地址的相同经编线程将达到最佳性能。此外，它专为具有恒定延迟的流式获取而设计；缓存命中可以减少DRAM带宽需求，但不会降低获取延迟。

通过纹理或表面获取读取设备内存具有一些好处，使其成为从全局或常量内存读取设备内存的有利替代方案：

- 如果内存读取不遵循全局或常量内存读取必须遵循的访问模式以获得良好的性能，那么只要在纹理获取或表面读取中存在局部性，就可以实现更高的带宽；
    
- 定址计算由专用单元在内核之外执行；
    
- 打包的数据可以在单个操作中广播到单独的变量；
    
- 8位和16位整数输入数据可以选择转换为[0.0, 1.0]或[-1.0, 1.0]范围内的32位浮点值（见[纹理内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory)）。
    

## 8.4.最大化指令吞吐量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximize-instruction-throughput "这个标题的永久链接")

有关优化指令吞吐量的更多详细信息，请参阅[CUDA C++最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput)。

## 8.5.最小化内存粉碎[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#minimize-memory-thrashing "这个标题的永久链接")

经常不断分配和释放内存的应用程序可能会发现，随着时间的推移，分配调用往往会变慢，直到一定限制。由于将内存释放回操作系统供其自己使用的性质，这通常是意料之中的。为了在这方面的最佳表现，我们建议以下内容：

- Try to size your allocation to the problem at hand. Don’t try to allocate all available memory with `cudaMalloc` / `cudaMallocHost` / `cuMemCreate`, as this forces memory to be resident immediately and prevents other applications from being able to use that memory. This can put more pressure on operating system schedulers, or just prevent other applications using the same GPU from running entirely.
    
- 尝试在应用程序早期以适当大小的分配分配内存，并且仅在应用程序没有任何用途时才进行分配。减少应用程序中的`cudaMalloc`调用数量，特别是在性能关键区域。
    
- 如果应用程序无法分配足够的设备内存，请考虑使用其他内存类型，如`cudaMallocHost`或`cudaMallocManaged`，这可能没有性能，但将使应用程序取得进展。
    
- 对于支持该功能的平台，`cudaMallocManaged`允许过度订阅，启用正确的`cudaMemAdvise`策略后，将允许应用程序保留大部分（如果不是全部）`cudaMalloc`的性能。`cudaMallocManaged`也不会强制分配为常驻，直到需要或预取，从而减轻了操作系统调度器的整体压力，并更好地启用多原则用例。
    

# 9.支持CUDA的GPU[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-enabled-gpus "这个标题的永久链接")

[https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)列出了所有具有计算能力的CUDA设备。

可以使用运行时查询计算能力、多处理器数量、时钟频率、设备内存总量和其他属性（请参阅参考手册）。

# 10。C++语言扩展[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions "这个标题的永久链接")

## 10.1.函数执行空间指定符[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-execution-space-specifiers "这个标题的永久链接")

函数执行空间指定符表示函数是在主机上执行还是在设备上执行，以及它是可以从主机还是从设备调用。

### 10.1.1. __全球__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global "这个标题的永久链接")

`__global__`执行空间指定符将函数声明为内核。这样的功能是：

- 在设备上执行，
    
- 可由主机呼叫，
    
- 可从计算能力5.0或更高版本的设备调用（有关更多详细信息，请参阅[CUDA动态并行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-dynamic-parallelism)）。
    

`__global__`函数必须具有无效返回类型，并且不能成为类的成员。

任何对`__global__`函数的调用都必须按照[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)中所述指定其执行配置。

对`__global__`函数的调用是非同步的，这意味着它在设备完成执行之前返回。

### 10.1.2. __设备__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device "这个标题的永久链接")

`__device__`执行空间指定符声明一个函数：

- 在设备上执行，
    
- 只能从设备呼叫。
    

`__global__`和`__device__`执行空间指定符不能一起使用。

### 10.1.3. __主机__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#host "这个标题的永久链接")

`__host__`执行空间指定符声明一个函数：

- 在主机上执行，
    
- 只能从主机呼叫。
    

等同于声明仅使用`__host__`执行空间指定符的函数，或者声明没有任何`__host__`、`__device__`或`__global__`执行空间指定符的函数；无论哪种情况，该函数都仅为主机编译。

`__global__`和`__host__`执行空间指定符不能一起使用。

然而，`__device__`和`__host__`执行空间指定符可以一起使用，在这种情况下，该函数是为主机和设备编译的。[应用程序兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility)中引入的`__CUDA_ARCH__`宏可用于区分主机和设备之间的代码路径：

__host__ __device__ func()
{
#if __CUDA_ARCH__ >= 800
   // Device code path for compute capability 8.x
#elif __CUDA_ARCH__ >= 700
   // Device code path for compute capability 7.x
#elif __CUDA_ARCH__ >= 600
   // Device code path for compute capability 6.x
#elif __CUDA_ARCH__ >= 500
   // Device code path for compute capability 5.x
#elif !defined(__CUDA_ARCH__)
   // Host code path
#endif
}

### 10.1.4.未定义的行为[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#undefined-behavior "这个标题的永久链接")

在以下时候，“跨执行空间”调用具有未定义的行为：

- `__CUDA_ARCH__` is defined, a call from within a `__global__`, `__device__` or `__host__ __device__` function to a `__host__` function.
    
- `__CUDA_ARCH__`是未定义的，从`__host__`函数内调用`__device__`函数。[4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn11)
    

### 10.1.5. __noinline__和__forceinline__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#noinline-and-forceinline "这个标题的永久链接")

编译器在认为合适时内行任何`__device__`函数。

如果可能的话，`__noinline__`函数限定符可以作为编译器不内联函数的提示。

`__forceinline__`函数限定符可用于强制编译器内联函数。

`__noinline__`和`__forceinline__`函数限定符不能一起使用，也不能将函数限定符应用于内联函数。

### 10.1.6. __内联_提示__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#inline-hint "这个标题的永久链接")

`__inline_hint__`限定符可以在编译器中进行更积极的内联。`__forceinline__`不同，它并不意味着该函数是内联的。在使用LTO时，它可用于改进跨模块的内联。

无论是`__noinline__`还是`__forceinline__`函数限定符都不能与`__inline_hint__`函数限定符一起使用。

## 10.2.可变内存空间指定符[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#variable-memory-space-specifiers "这个标题的永久链接")

变量内存空间指定符表示变量设备上的内存位置。

在设备代码中声明的自动变量，没有任何本节中描述的`__device__`、`__shared__`和`__constant__`内存空间指定符，通常位于寄存器中。然而，在某些情况下，编译器可能会选择将其放置在本地内存中，这可能会对性能产生不利影响，如[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)中所述。

### 10.2.1. __设备__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-variable-specifier "这个标题的永久链接")

`__device__`内存空间指定符声明位于设备上的变量。

最多，接下来三个部分中定义的其他内存空间指定符之一可以与`__device__`一起使用，以进一步表示变量属于哪个内存空间。如果它们都不存在，变量：

- 居住在全球内存空间，
    
- 拥有创建CUDA上下文的生命周期，
    
- 每个设备都有一个不同的对象，
    
- Is accessible from all the threads within the grid and from the host through the runtime library `(cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`).
    

### 10.2.2. __恒定__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant "这个标题的永久链接")

`__constant__`内存空间指定符，可选择与`__device__`一起使用，声明一个变量：

- 位于恒定的内存空间中，
    
- 拥有创建CUDA上下文的生命周期，
    
- 每个设备都有一个不同的对象，
    
- Is accessible from all the threads within the grid and from the host through the runtime library (`cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`).
    

当有一个并发网格在该网格生命周期的任何点访问该常量时，从主机修改常量的行为是未定义的。

### 10.2.3. __共享__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared "这个标题的永久链接")

`__shared__`内存空间指定符，可选择与`__device__`一起使用，声明一个变量：

- 位于线程块的共享内存空间中，
    
- 拥有块的寿命，
    
- 每个块都有一个不同的对象，
    
- 只能从块内的所有线程访问，
    
- 没有固定的地址。
    

当将共享内存中的变量声明为外部数组时，例如

extern __shared__ float shared[];

阵列的大小在启动时确定（请参阅[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)）。以这种方式声明的所有变量，从内存中的同一地址开始，因此数组中变量的布局必须通过偏移量进行显式管理。例如，如果一个人想要等同于

short array0[128];
float array1[64];
int   array2[256];

在动态分配的共享内存中，可以通过以下方式声明和初始化数组：

extern __shared__ float array[];
__device__ void func()      // __device__ or __global__ function
{
    short* array0 = (short*)array;
    float* array1 = (float*)&array0[128];
    int*   array2 =   (int*)&array1[64];
}

请注意，指针需要与它们所指向的类型对齐，因此，例如，以下代码不起作用，因为数组1没有对齐到4个字节。

extern __shared__ float array[];
__device__ void func()      // __device__ or __global__ function
{
    short* array0 = (short*)array;
    float* array1 = (float*)&array0[127];
}

[表7](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types-alignment-requirements-in-device-code)列出了内置矢量类型的对齐要求。

### 10.2.4. __网格_常数__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#grid-constant "这个标题的永久链接")

大于或等于7.0的计算架构的`__grid_constant__`注释注释了非引用类型的`const`限定`__global__`函数参数：

- 拥有网格的寿命，
    
- 对网格是私有的，即该对象无法访问主机线程和其他网格（包括子网格）的线程，
    
- 每个网格都有一个不同的对象，即网格中的所有线程都看到相同的地址，
    
- 是只读的，即修改`__grid_constant__`对象或其任何子对象都是_未定义的行为_，包括`mutable`成员。
    

要求：

- 用`__grid_constant__`注释的内核参数必须具有`const`限定的非引用类型。
    
- 所有函数声明必须与任何`__grid_constant_`参数相匹配。
    
- 函数模板专业化必须与任何`__grid_constant__`参数的主要模板声明相匹配。
    
- 函数模板实例化指令必须与任何`__grid_constant__`参数的主模板声明相匹配。
    

如果获取了`__global__`函数参数的地址，编译器通常会在线程本地内存中复制内核参数，并使用副本的地址来部分支持C++语义，这允许每个线程修改函数参数的本地副本。用`__grid_constant__`注释`__global__`函数参数，确保编译器不会在线程本地内存中创建内核参数的副本，而是使用参数本身的通用地址。避免本地复制可能会提高性能。

__device__ void unknown_function(S const&);
__global__ void kernel(const __grid_constant__ S s) {
   s.x += threadIdx.x;  // Undefined Behavior: tried to modify read-only memory

   // Compiler will _not_ create a per-thread thread local copy of "s":
   unknown_function(s);
}

### 10.2.5. __管理__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#managed "这个标题的永久链接")

`__managed__`内存空间指定符，可选择与`__device__`一起使用，声明一个变量：

- 可以从设备和主机代码中引用，例如，可以获取其地址，也可以直接从设备或主机函数读取或写入。
    
- 具有应用程序的生命周期。
    

有关更多详细信息，请参阅[__managed__内存空间指定符](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#managed-specifier)。

### 10.2.6. __限制__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#restrict "这个标题的永久链接")

`nvcc`通过`__restrict__`关键字支持受限的指针。

C99中引入了受限的指针，以缓解C型语言中存在的锯齿问题，并抑制了从代码重新排序到常见子表达式消除的所有优化。

这是一个受别名问题约束的示例，使用受限指针可以帮助编译器减少指令数量：

void foo(const float* a,
         const float* b,
         float* c)
{
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] * b[0] * a[1];
    c[3] = a[0] * a[1];
    c[4] = a[0] * b[0];
    c[5] = b[0];
    ...
}

在C型语言中，指针`a`和`c`可以是别名的，因此任何通过`c`的写入都可以修改`a`或`b`的元素。这意味着，为了保证功能正确性，编译器不能将`a[0]`和`b[0]`加载到寄存器中，将它们相乘，并将结果存储在`c[0]`和`c[1]`中，因为如果`a[0]`与`c[0]`的位置确实相同，结果将与抽象执行模型不同。因此，编译器无法利用常见的子表达式。同样，编译器不能只是将`c[4]`的计算重新排序为`c[0]`和`c[1]`计算的接近，因为之前写入`c[3]`可能会将输入更改为`c[4]`的计算。

通过使`a`和`c`限制指针，程序员向编译器断言，指针实际上不是别名的，在这种情况下，这意味着通过`c`写入永远不会覆盖a或`b`的元素。这改变了函数原型如下：

void foo(const float* __restrict__ a,
         const float* __restrict__ b,
         float* __restrict__ c);

请注意，所有指针参数都需要受到限制，编译器优化器才能获得任何好处。添加`__restrict__`关键字后，编译器现在可以随心地重新排序和执行常见的子表达式消除，同时保留与抽象执行模型相同的功能：

void foo(const float* __restrict__ a,
         const float* __restrict__ b,
         float* __restrict__ c)
{
    float t0 = a[0];
    float t1 = b[0];
    float t2 = t0 * t1;
    float t3 = a[1];
    c[0] = t2;
    c[1] = t2;
    c[4] = t2;
    c[2] = t2 * t3;
    c[3] = t0 * t3;
    c[5] = t1;
    ...
}

这里的影响是内存访问数量减少和计算数量减少。这被由于“缓存”负载和常见的子表达式导致的寄存器压力的增加所平衡。

由于寄存器压力是许多CUDA代码中的关键问题，由于占用率降低，使用受限指针可能会对CUDA代码的性能产生负面影响。

## 10.3.内置矢量类型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-vector-types "这个标题的永久链接")

### 10.3.1. char, short, int, long, long long, float, double[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#char-short-int-long-longlong-float-double "这个标题的永久链接")

这些是从基本整数和浮点类型派生的向量类型。它们是结构，第1、第2、第3和第4个成分分别可以通过字段`x`、`y`、`z`访问。它们都带有`make_<typename>`形式的构造函数；例如，

int2 make_int2(int x, int y);

它创建了一个值为`(x,y)`的`int2`类型的向量。

向量类型的对齐要求详见表[7。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types-alignment-requirements-in-device-code)

表7对齐要求[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types-alignment-requirements-in-device-code "此表的永久链接")
|类型|对齐|
|---|---|
|char1，uchar1|1|
|字符2，uchar2|2|
|字符3，uchar3|1|
|char4，uchar4|4|
|短1，ushort1|2|
|短2，ushort2|4|
|短3，ushort3|2|
|短4，ushort4|8|
|int1，uint1|4|
|int2，uint2|8|
|int3，uint3|4|
|int4，uint4|16|
|长1，长1|4如果sizeof（long）等于sizeof（int）8，否则|
|长2，长2|8如果sizeof（长）等于sizeof（int），否则16|
|长3，长3|4如果sizeof（long）等于sizeof（int），8，否则|
|长4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|长4_16a|
|长4_32a|32|
|乌隆4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|乌隆4_16a|
|乌隆4_32a|32|
|龙龙1，乌龙龙1|8|
|龙龙2，乌龙龙2|16|
|龙龙3，龙龙3|8|
|龙龙4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|长长4_16a|
|长长4_32a|32|
|乌龙龙4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|乌龙龙4_16a|
|乌隆隆4_32a|32|
|浮动1|4|
|浮动2|8|
|浮动3|4|
|浮动4|16|
|双1|8|
|双2|16|
|双倍3|8|
|双4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|双4_16a|
|双4_32a|32|

3（1，2，3，4，5）

在CUDA Toolkit 13.0中，该向量类型已弃用。请使用`_16a`或`_32a`变体，具体取决于您的对齐要求。

### 10.3.2. dim3[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3 "这个标题的永久链接")

此类型是基于`uint3`的整数向量类型，用于指定维度。在定义`dim3`类型的变量时，任何未指定的组件都会初始化为1。

## 10.4.内置变量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables "这个标题的永久链接")

内置变量指定网格和块尺寸以及块和线程索引。它们仅在设备上执行的函数内有效。

### 10.4.1.网格Dim[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#griddim "这个标题的永久链接")

此变量类型为`dim3`（见[dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3)），包含网格的维度。

### 10.4.2.块Idx[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#blockidx "这个标题的永久链接")

此变量属于`uint3`类型（见[char、short、int、long、longlong、float、double](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types)），包含网格中的块索引。

### 10.4.3.块Dim[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#blockdim "这个标题的永久链接")

此变量类型为`dim3`（见[dim3）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3)，包含块的维度。

### 10.4.4.线程Idx[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#threadidx "这个标题的永久链接")

此变量类型为`uint3`（见[char、short、int、long、longlong、float、double](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types)），包含块中的线程索引。

### 10.4.5. 扭曲尺寸[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warpsize "这个标题的永久链接")

此变量类型为`int`，包含线程中的经编大小（请参阅[SIMT架构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture)以定义经编）。

## 10.5.内存围栏功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-functions "这个标题的永久链接")

CUDA编程模型假设一个具有弱有序内存模型的设备，即CUDA线程将数据写入共享内存、全局内存、页面锁定主机内存或对等设备的内存的顺序不一定是另一个CUDA或主机线程记录数据的顺序。两个线程在不同步的情况下读取或写入同一内存位置是未定义的行为。

在以下示例中，线程1执行`writeXY()`而线程2执行`readXY()`

__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    int A = X;
}

两个线程同时从相同的内存位置X和`Y`读取和写入。任何数据竞赛都是未定义的行为，并且没有定义的语义。`A`的结果值可以是任何东西。

内存围栏函数可用于对内存访问强制执行[顺序一致的](https://en.cppreference.com/w/cpp/atomic/memory_order)排序。内存围栏函数在强制排序的[范围](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes)上有所不同，但它们独立于访问的内存空间（共享内存、全局内存、页面锁定主机内存和对等设备的内存）。

void __threadfence_block();

等同于[cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html)，并确保：

- 调用`__threadfence_block()`之前，调用线程对所有内存的所有写入都被调用线程块中的所有线程观察到，在调用`__threadfence_block()`后对调用线程对所有内存的所有写入之前发生；
    
- 在调用`__threadfence_block()`之前，调用线程从所有内存中读取的所有读取都排序，在调用`__threadfence_block()`后，调用线程从所有内存中读取之前排序。
    

void __threadfence();

等同于[cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html)，并确保在调用`__threadfence()`后调用线程对所有内存的写入，设备中的任何线程都观察到在调用`__threadfence()`之前对调用线程的所有内存进行写入之前发生。

void __threadfence_system();

等同于[cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system)](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html)，并确保在调用`__threadfence_system()`之前调用线程对所有内存的所有写入都被设备中的所有线程、主机线程和对等设备中的所有线程视为在调用`__threadfence_system()`后对调用线程对所有内存的所有写入之前发生。

`__threadfence_system()`仅支持具有2.x及以上计算能力的设备。

在之前的代码示例中，我们可以在代码中插入栅栏如下：
```c++
__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    __threadfence();
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    __threadfence();
    int A = X;
}
```
对于这个代码，可以观察到以下结果：

- `A`等于1，B等于2，
    
- `A`等于10，B等于2，
    
- `A`等于10，B等于20。
    

第四个结果是不可能的，因为第一次写入必须在第二次写入之前可见。如果线程1和2属于同一个块，使用`__threadfence_block()`就足够了。如果线程1和2不属于同一块，如果它们是来自同一设备的CUDA线程，则必须使用`__threadfence()`），如果它们是来自两个不同设备的CUDA线程，则必须使用`__threadfence_system()`）。

一个常见的用例是当线程消耗其他线程生成的一些数据时，如以下内核代码示例所示，该代码示例计算在一次调用中N个数字数组的总和。每个块首先将数组的子集相加，并将结果存储在全局内存中。当所有块完成时，最后一个块从全局内存中读取这些部分和，并将它们相加以获得最终结果。为了确定哪个块最后完成，每个块原子递增一个计数器，以表示它是通过计算和存储其部分和完成的（见关于原子函数的[原子函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)）。最后一个块是接收计数器值等于`gridDim.x-1`的块。如果在存储部分总和和增加计数器之间没有设置栅栏，计数器可能会在部分总和存储之前增加，因此，可能会到达`gridDim.x-1`，并让最后一个块在内存中实际更新之前开始读取部分总和。

内存围栏函数只影响线程对内存操作的排序；它们本身并不能确保这些内存操作对其他线程可见（如`__syncthreads()`对块内线程所做的那样；请参阅[同步函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-functions)）。在下面的代码示例中，通过将其声明为易失性来确保`result`变量上内存操作的可见性（请参阅[易失性限定符](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#volatile-qualifier)）。
```c++
__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;
__global__ void sum(const float* array, unsigned int N,
                    volatile float* result)
{
    // Each block sums a subset of the input array.
    float partialSum = calculatePartialSum(array, N);

    if (threadIdx.x == 0) {

        // Thread 0 of each block stores the partial sum
        // to global memory. The compiler will use
        // a store operation that bypasses the L1 cache
        // since the "result" variable is declared as
        // volatile. This ensures that the threads of
        // the last block will read the correct partial
        // sums computed by all other blocks.
        result[blockIdx.x] = partialSum;

        // Thread 0 makes sure that the incrementing
        // of the "count" variable is only performed after
        // the partial sum has been written to global memory.
        __threadfence();

        // Thread 0 signals that it is done.
        unsigned int value = atomicInc(&count, gridDim.x);

        // Thread 0 determines if its block is the last
        // block to be done.
        isLastBlockDone = (value == (gridDim.x - 1));
    }

    // Synchronize to make sure that each thread reads
    // the correct value of isLastBlockDone.
    __syncthreads();

    if (isLastBlockDone) {

        // The last block sums the partial sums
        // stored in result[0 .. gridDim.x-1]
        float totalSum = calculateTotalSum(result);

        if (threadIdx.x == 0) {

            // Thread 0 of last block stores the total sum
            // to global memory and resets the count
            // variable, so that the next kernel call
            // works properly.
            result[0] = totalSum;
            count = 0;
        }
    }
}
```
## 10.6.同步功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-functions "这个标题的永久链接")
```c++
void __syncthreads();
```

等到线程块中的所有线程都达到这个点，并且这些线程在`__syncthreads()`之前进行的所有全局和共享内存访问都对块中的所有线程可见。

- `__syncthreads()`用于协调同一块线程之间的通信。当块中的一些线程访问共享或全局内存中的相同地址时，其中一些内存访问存在潜在的写后读、读后写或写后写入危险。通过在这些访问之间同步线程，可以避免这些数据危害。

- `__syncthreads()`在条件代码中是允许的，但前提是条件在整个线程块中评估相同，否则代码执行可能会挂起或产生意外副作用。

计算能力2.x及以上的设备支持下面描述的`__syncthreads()`的三种变体。
```c++
int __syncthreads_count(int predicate);
```

与`__syncthreads()`相同，具有额外的功能，即它评估块的所有线程的谓词，并返回谓词评估为非零的线程数。

```
int __syncthreads_and(int predicate);
```
与`__syncthreads()`相同，具有额外的功能，即它评估块的所有线程的谓词，并且仅当谓词评估为非零时（仅当）将其返回非零。

```c++
int __syncthreads_or(int predicate);
```

与`__syncthreads()`相同，具有额外的功能，即它评估块的所有线程的谓词，并且仅当谓词评估为非零时返回非零。

```c++
void __syncwarp(unsigned mask=0xffffffff);
```
将导致执行线程等到掩码中命名的所有扭曲通道都执行了`__syncwarp()`（具有相同的掩码），然后再恢复执行。每个调用线程必须在掩码中设置自己的位，掩码中命名的所有未退出线程必须使用相同的掩码执行相应的`__syncwarp()`），否则结果是未定义的。

执行`__syncwarp()`保证了参与障碍的线程之间的内存排序。因此，希望通过内存通信的经编中的线程可以存储在内存中，执行`__syncwarp()`然后安全地读取经编中其他线程存储的值。

对于.target sm_6x或更低，掩码中的所有线程必须在收敛中执行相同的`__syncwarp()`），掩码中所有值的并联必须等于活动掩码。否则，行为是未定义的。

## 10.7.数学函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mathematical-functions "这个标题的永久链接")

参考手册列出了设备代码中支持的所有C/C++标准库数学函数以及仅在设备代码中支持的所有内在函数。

[数学函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mathematical-functions-appendix)在相关时为其中一些函数提供准确性信息。

## 10.8.纹理函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-functions "这个标题的永久链接")

纹理对象在[纹理对象API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api)中进行了描述。

纹理获取在[“纹理获取”](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-fetching)中进行了描述。

### 10.8.1.紋理物件API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api-appendix "这个标题的永久链接")

#### 10.8.1.1. tex1D获取（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dfetch "这个标题的永久链接")
```c++
template<class T>
T tex1Dfetch(cudaTextureObject_t texObj, int x);
```

使用整数纹理坐标`x``tex1Dfetch()`从一维纹理对象`texObj`指定的线性内存区域获取，仅适用于非规范化坐标，因此仅支持边框和夹紧寻址模式。它不执行任何纹理过滤。对于整数类型，它可以选择将整数提升为单精度浮点。

#### 10.8.1.2. tex1D（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1d "这个标题的永久链接")
```c++
template<class T>
T tex1D(cudaTextureObject_t texObj, float x);
```

使用纹理坐标x从一维纹理对象`texObj`指定的CUDA数组中获取。

#### 10.8.1.3. tex1DLod（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlod "这个标题的永久链接")
```c++
template<class T>
T tex1DLod(cudaTextureObject_t texObj, float x, float level);
```

从一维纹理对象`texObj`指定的CUDA数组中获取，使用细节`level`的纹理坐标x。

#### 10.8.1.4. tex1DGrad（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dgrad "这个标题的永久链接")
```c++
template<class T>
T tex1DGrad(cudaTextureObject_t texObj, float x, float dx, float dy);
```

使用纹理坐标x从一维纹理对象`texObj`指定的CUDA数组中获取。细节水平来自X-梯度`dx`和Y-梯度`dy`。

#### 10.8.1.5. tex2D（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2d "这个标题的永久链接")
```c++
template<class T>
T tex2D(cudaTextureObject_t texObj, float x, float y);
```
使用纹理坐标`(x,y)`从CUDA数组或二维纹理对象`texObj`指定的线性内存区域获取。

#### 10.8.1.6. tex2D（）用于稀疏CUDA数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2d-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
template<class T>
T tex2D(cudaTextureObject_t texObj, float x, float y, bool* isResident);
```
使用纹理坐标`(x,y)`从二维纹理对象`texObj`指定的CUDA数组中获取。还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.7. tex2Dgather（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgather "这个标题的永久链接")
```c++
template<class T>
T tex2Dgather(cudaTextureObject_t texObj,
              float x, float y, int comp = 0);
```
使用纹理坐标x和`y`以及[Texture Gather](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-gather)中描述的`comp`参数，从2D纹理对象`texObj`指定的CUDA数组中获取。

#### 10.8.1.8. tex2Dgather（）用于稀疏CUDA数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgather-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
template<class T>
T tex2Dgather(cudaTextureObject_t texObj,
            float x, float y, bool* isResident, int comp = 0);
```
使用纹理坐标x和`y`以及[Texture Gather](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-gather)中描述的`comp`参数，从2D纹理对象`texObj`指定的CUDA数组中获取。还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.9. tex2DGrad（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgrad "这个标题的永久链接")
```c++
template<class T>
T tex2DGrad(cudaTextureObject_t texObj, float x, float y,
            float2 dx, float2 dy);
```
使用纹理坐标`(x,y)`从二维纹理对象`texObj`指定的CUDA数组中获取。细节水平来自`dx`和`dy`梯度。

#### 10.8.1.10. tex2DGrad（）用于稀疏CUDA数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgrad-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
template<class T>
T tex2DGrad(cudaTextureObject_t texObj, float x, float y,
        float2 dx, float2 dy, bool* isResident);
```
使用纹理坐标`(x,y)`从二维纹理对象`texObj`指定的CUDA数组中获取。细节水平来自`dx`和`dy`梯度。还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.11. tex2DLod（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlod "这个标题的永久链接")
```c++
template<class T>
tex2DLod(cudaTextureObject_t texObj, float x, float y, float level);
```
从CUDA数组或二维纹理对象`texObj`指定的线性内存区域获取，使用细节`level`纹理坐标`(x,y)`

#### 10.8.1.12. tex2DLod（）用于稀疏CUDA数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlod-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
template<class T>
tex2DLod(cudaTextureObject_t texObj, float x, float y, float level, bool* isResident);
```
从二维纹理对象`texObj`指定的CUDA数组中获取，使用细节级别的纹理坐标`(x,y)`还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.13. tex3D（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3d "这个标题的永久链接")
```c++
template<class T>
T tex3D(cudaTextureObject_t texObj, float x, float y, float z);
```
使用纹理坐标`(x,y,z)`从三维纹理对象`texObj`指定的CUDA数组中获取。

#### 10.8.1.14. tex3D（）用于稀疏CUDA数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3d-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
template<class T>
T tex3D(cudaTextureObject_t texObj, float x, float y, float z, bool* isResident);
```
使用纹理坐标`(x,y,z)`从三维纹理对象`texObj`指定的CUDA数组中获取。还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.15. tex3DLod（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3dlod "这个标题的永久链接")
```c++
template<class T>
T tex3DLod(cudaTextureObject_t texObj, float x, float y, float z, float level);
```
从CUDA数组或三维纹理对象`texObj`指定的线性内存区域获取，使用细节`level`纹理坐标`(x,y,z)`

#### 10.8.1.16. tex3DLod（）用于稀疏CUDA数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3dlod-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
template<class T>
T tex3DLod(cudaTextureObject_t texObj, float x, float y, float z, float level, bool* isResident);
```
从CUDA数组或三维纹理对象`texObj`指定的线性内存区域获取，使用细节`level`纹理坐标`(x,y,z)`还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.17. tex3DGrad（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3dgrad "这个标题的永久链接")
```c++
template<class T>
T tex3DGrad(cudaTextureObject_t texObj, float x, float y, float z,
            float4 dx, float4 dy);
```
从三维纹理对象`texObj`指定的CUDA数组中获取，使用纹理坐标`(x,y,z)`在X和Y梯度`dx`和`dy`衍生的细节级别上。

#### 10.8.1.18. tex3DGrad（）用于稀疏CUDA数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3dgrad-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
template<class T>
T tex3DGrad(cudaTextureObject_t texObj, float x, float y, float z,
        float4 dx, float4 dy, bool* isResident);
```
从三维纹理对象`texObj`指定的CUDA数组中获取，使用纹理坐标`(x,y,z)`在X和Y梯度`dx`和`dy`衍生的细节级别上。还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.19. tex1DLyered（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlayered "这个标题的永久链接")
```c++
template<class T>
T tex1DLayered(cudaTextureObject_t texObj, float x, int layer);
```
如[分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)中所述，使用纹理坐标x和索引层从一维纹理对象`texObj`指定的CUDA数组中获取。

#### 10.8.1.20. tex1DRayeredLod（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlayeredlod "这个标题的永久链接")
```c++
template<class T>
T tex1DLayeredLod(cudaTextureObject_t texObj, float x, int layer, float level);
```
使用纹理坐标x和detaillevel从图层上一维分[层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)指定的CUDA数组获取。

#### 10.8.1.21. tex1DLayeredGrad（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlayeredgrad "这个标题的永久链接")
```c++
template<class T>
T tex1DLayeredGrad(cudaTextureObject_t texObj, float x, int layer,
                   float dx, float dy);
```
使用纹理坐标x以及从`dx`和`dy`梯度得出的细节水平，从图层层中一维分[层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)指定的CUDA数组中获取。

#### 10.8.1.22. tex2DLyered（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayered "这个标题的永久链接")
```c++
template<class T>
T tex2DLayered(cudaTextureObject_t texObj,
               float x, float y, int layer);
```
如[分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)中所述，使用纹理坐标`(x,y)`和索引层从二维纹理对象`texObj`指定的CUDA数组中获取。

#### 10.8.1.23. 稀疏CUDA数组的tex2DLayered()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayered-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
template<class T>
T tex2DLayered(cudaTextureObject_t texObj,
            float x, float y, int layer, bool* isResident);
```
如[分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)中所述，使用纹理坐标`(x,y)`和索引层从二维纹理对象`texObj`指定的CUDA数组中获取。还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.24. tex2DLayeredLod（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayeredlod "这个标题的永久链接")
```c++
template<class T>
T tex2DLayeredLod(cudaTextureObject_t texObj, float x, float y, int layer,
                  float level);
```
使用纹理坐标`(x,y)`从层层二维[分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)指定的CUDA数组中获取。

#### 10.8.1.25. tex2DLayeredLod（）用于稀疏CUDA数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayeredlod-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
template<class T>
T tex2DLayeredLod(cudaTextureObject_t texObj, float x, float y, int layer,
                float level, bool* isResident);
```
使用纹理坐标`(x,y)`从层层二维[分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)指定的CUDA数组中获取。还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.26. tex2DLayeredGrad（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayeredgrad "这个标题的永久链接")
```c++
template<class T>
T tex2DLayeredGrad(cudaTextureObject_t texObj, float x, float y, int layer,
                   float2 dx, float2 dy);
```
使用纹理坐标`(x,y)`和从`dx`和`dy`梯度衍生的细节级别，从层层中指定的二维[分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)指定的CUDA数组中获取。

#### 10.8.1.27. tex2DLayeredGrad（）用于稀疏CUDA数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayeredgrad-for-sparse-cuda-arrays "这个标题的永久链接")
```c++
                template<class T>
T tex2DLayeredGrad(cudaTextureObject_t texObj, float x, float y, int layer,
                float2 dx, float2 dy, bool* isResident);
```
使用纹理坐标`(x,y)`和从`dx`和`dy`梯度衍生的细节级别，从层层中指定的二维[分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures)指定的CUDA数组中获取。还通过`isResident`指针返回texel是否驻留在内存中。如果没有，获取的值将是零。

#### 10.8.1.28. texCubemap（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemap "这个标题的永久链接")
```c++
template<class T>
T texCubemap(cudaTextureObject_t texObj, float x, float y, float z);
```
如[立方体图纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures)中所述，使用纹理坐标`(x,y,z)`获取立方体图纹理对象`texObj`指定的CUDA数组。

#### 10.8.1.29. texCubemapGrad（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemapgrad "这个标题的永久链接")
```c++
template<class T>
T texCubemapGrad(cudaTextureObject_t texObj, float x, float, y, float z,
                float4 dx, float4 dy);
```
使用[立方体图纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures)中描述的纹理坐标`(x,y,z)`从立方体图纹理对象`texObj`指定的CUDA数组中获取。使用的细节级别来自`dx`和`dy`梯度。

#### 10.8.1.30. texCubemapLod（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplod "这个标题的永久链接")
```c++
template<class T>
T texCubemapLod(cudaTextureObject_t texObj, float x, float, y, float z,
                float level);
```
使用[立方体图纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures)中描述的纹理坐标`(x,y,z)`从立方体图纹理对象`texObj`指定的CUDA数组中获取。使用的细节级别按`level`给出。

#### 10.8.1.31. texCubemap分层（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplayered "这个标题的永久链接")
```c++
template<class T>
T texCubemapLayered(cudaTextureObject_t texObj,
                    float x, float y, float z, int layer);
```
如[Cubemap分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures)中所述，使用纹理坐标`(x,y,z)`和索引层从立方体图分层纹理对象`texObj`指定的CUDA数组中获取。

#### 10.8.1.32. texCubemap分层毕业（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplayeredgrad "这个标题的永久链接")
```c++
template<class T>
T texCubemapLayeredGrad(cudaTextureObject_t texObj, float x, float y, float z,
                       int layer, float4 dx, float4 dy);
```
使用纹理坐标`(x,y,z)`和索引层从立方体图分层纹理对象`texObj`指定的CUDA数组中获取，如[立方体图分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures)中所述，在从`dx`和`dy`梯度中提取的细节级别。

#### 10.8.1.33. texCubemapLayeredLod（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplayeredlod "这个标题的永久链接")
```c++
template<class T>
T texCubemapLayeredLod(cudaTextureObject_t texObj, float x, float y, float z,
                       int layer, float level);
```
使用纹理坐标`(x,y,z)`和索引层从立方体图分层纹理对象`texObj`指定的CUDA数组中获取，如[立方体图分层纹理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures)中所述，在细节级别级别。

## 10.9.表面功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-functions "这个标题的永久链接")

表面功能仅由具有2.0及更高计算能力的设备支持。

表面对象在[表面对象API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-object-api-appendix)中描述。

在下面的章节中，`boundaryMode`指定了边界模式，即如何处理范围外表面坐标；它等于`cudaBoundaryModeClamp`，在这种情况下，范围外坐标被夹在有效范围内，或`cudaBoundaryModeZero`，在这种情况下，范围外读取返回零，范围外写入被忽略，或`cudaBoundaryModeTrap`，在这种情况下，范围外访问导致内核执行失败。

### 10.9.1.表面对象API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-object-api-appendix "这个标题的永久链接")

#### 10.9.1.1. surf1Dread（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf1dread "这个标题的永久链接")
```c++
template<class T>
T surf1Dread(cudaSurfaceObject_t surfObj, int x,
               boundaryMode = cudaBoundaryModeTrap);
```
使用字节坐标x读取一维表面对象`surfObj`指定的CUDA数组。

#### 10.9.1.2. surf1Dwrite[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf1dwrite "这个标题的永久链接")
```c++
template<class T>
void surf1Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x,
                  boundaryMode = cudaBoundaryModeTrap);
```
在字节坐标x处将值数据写入一维表面对象`surfObj`指定的CUDA数组。

#### 10.9.1.3. surf2Dread（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf2dread "这个标题的永久链接")
```c++
template<class T>
T surf2Dread(cudaSurfaceObject_t surfObj,
              int x, int y,
              boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf2Dread(T* data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y,
                 boundaryMode = cudaBoundaryModeTrap);
```
使用字节坐标x和y读取二维表面对象`surfObj`指定的CUDA数组。

#### 10.9.1.4. surf2Dwrite（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf2dwrite "这个标题的永久链接")
```c++
template<class T>
void surf2Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y,
                  boundaryMode = cudaBoundaryModeTrap);
```
在字节坐标x和y处将值数据写入由二维表面对象`surfObj`指定的CUDA数组。

#### 10.9.1.5. surf3Dread（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf3dread "这个标题的永久链接")
```c++
template<class T>
T surf3Dread(cudaSurfaceObject_t surfObj,
              int x, int y, int z,
              boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf3Dread(T* data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int z,
                 boundaryMode = cudaBoundaryModeTrap);
```
使用字节坐标x、y和z读取三维表面对象`surfObj`指定的CUDA数组。

#### 10.9.1.6. surf3Dwrite（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf3dwrite "这个标题的永久链接")
```c++
template<class T>
void surf3Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y, int z,
                  boundaryMode = cudaBoundaryModeTrap);
```
在字节坐标x、y和z处将值数据写入三维对象`surfObj`指定的CUDA数组。

#### 10.9.1.7. surf1DLayeredread（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf1dlayeredread "这个标题的永久链接")
```c++
template<class T>
T surf1DLayeredread(
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf1DLayeredread(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
```
使用字节坐标x和索引层读取一维分层表面对象`surfObj`指定的CUDA数组。

#### 10.9.1.8. surf1DLayeredwrite（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf1dlayeredwrite "这个标题的永久链接")
```c++
template<class Type>
void surf1DLayeredwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
```
将值数据写入由二维分层表面对象`surfObj`在字节坐标x和索引层指定的CUDA数组。

#### 10.9.1.9. surf2DLayeredread（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf2dlayeredread "这个标题的永久链接")
```c++
template<class T>
T surf2DLayeredread(
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int layer,
                 boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf2DLayeredread(T data,
                         cudaSurfaceObject_t surfObj,
                         int x, int y, int layer,
                         boundaryMode = cudaBoundaryModeTrap);
```
使用字节坐标x和y以及索引层读取二维分层表面对象`surfObj`指定的CUDA数组。

#### 10.9.1.10. surf2DLayeredwrite（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf2dlayeredwrite "这个标题的永久链接")
```c++
template<class T>
void surf2DLayeredwrite(T data,
                          cudaSurfaceObject_t surfObj,
                          int x, int y, int layer,
                          boundaryMode = cudaBoundaryModeTrap);
```
将值数据写入由一维分层表面对象`surfObj`在字节坐标x和y以及索引层指定的CUDA数组。

#### 10.9.1.11. surfCubemapread（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemapread "这个标题的永久链接")
```c++
template<class T>
T surfCubemapread(
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surfCubemapread(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);
```
使用字节坐标x和y以及面索引面读取立方体映射表面对象`surfObj`指定的CUDA数组。

#### 10.9.1.12. surfCubemapwrite（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemapwrite "这个标题的永久链接")
```c++
template<class T>
void surfCubemapwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);
```
将值数据写入立方体对象`surfObj`在字节坐标x和y和面索引面指定的CUDA数组。

#### 10.9.1.13. surfCubemapLayeredread（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemaplayeredread "这个标题的永久链接")
```c++
template<class T>
T surfCubemapLayeredread(
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surfCubemapLayeredread(T data,
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);
```
使用字节坐标x和y以及索引读取立方体图分层表面对象`surfObj`指定的CUDA数组`layerFace.`

#### 10.9.1.14. surfCubemapLayeredwrite（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemaplayeredwrite "这个标题的永久链接")
```c++
template<class T>
void surfCubemapLayeredwrite(T data,
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);
```
将值数据写入由立方体地图分层对象`surfObj`在字节坐标x和`y`和索引`layerFace`指定的CUDA数组。

## 10.10.只读数据缓存加载功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#read-only-data-cache-load-function "这个标题的永久链接")

只读数据缓存加载功能仅由计算能力5.0及更高版本的设备支持。

T __ldg(const T* address);

returns the data of type `T` located at address `address`, where `T` is `char`, `signed char`, `short`, `int`, `long`, `long long``unsigned char`, `unsignedshort`, `unsigned int`, `unsigned long`, `unsigned long long`, `char2`, `char4`, `short2`, `short4`, `int2`, `int4`, `longlong2``uchar2`, `uchar4`, `ushort2`, `ushort4`, `uint2`, `uint4`, `ulonglong2``float`, `float2`, `float4`, `double`, or `double2`. With the `cuda_fp16.h` header included, `T` can be `__half` or `__half2`. Similarly, with the `cuda_bf16.h` header included, `T` can also be `__nv_bfloat16` or `__nv_bfloat162`. The operation is cached in the read-only data cache (see [Global Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x)).

## 10.11.使用缓存提示加载函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#load-functions-using-cache-hints "这个标题的永久链接")

这些负载功能仅由具有5.0及更高计算能力的设备支持。

T __ldcg(const T* address);
T __ldca(const T* address);
T __ldcs(const T* address);
T __ldlu(const T* address);
T __ldcv(const T* address);

returns the data of type `T` located at address `address`, where `T` is `char`, `signed char`, `short`, `int`, `long`, `long long``unsigned char`, `unsignedshort`, `unsigned int`, `unsigned long`, `unsigned long long`, `char2`, `char4`, `short2`, `short4`, `int2`, `int4`, `longlong2``uchar2`, `uchar4`, `ushort2`, `ushort4`, `uint2`, `uint4`, `ulonglong2``float`, `float2`, `float4`, `double`, or `double2`. With the `cuda_fp16.h` header included, `T` can be `__half` or `__half2`. Similarly, with the `cuda_bf16.h` header included, `T` can also be `__nv_bfloat16` or `__nv_bfloat162`. The operation is using the corresponding cache operator (see [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators))

## 10.12.使用缓存提示存储功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#store-functions-using-cache-hints "这个标题的永久链接")

只有具有5.0及更高计算能力的设备支持这些存储功能。
```c++
void __stwb(T* address, T value);
void __stcg(T* address, T value);
void __stcs(T* address, T value);
void __stwt(T* address, T value);
```
stores the `value` argument of type `T` to the location at address `address`, where `T` is `char`, `signed char`, `short`, `int`, `long`, `long long``unsignedchar`, `unsigned short`, `unsigned int`, `unsigned long`, `unsigned long long`, `char2`, `char4`, `short2`, `short4`, `int2`, `int4`, `longlong2``uchar2`, `uchar4`, `ushort2`, `ushort4`, `uint2`, `uint4`, `ulonglong2``float`, `float2`, `float4`, `double`, or `double2`. With the `cuda_fp16.h` header included, `T` can be `__half` or `__half2`. Similarly, with the `cuda_bf16.h` header included, `T` can also be `__nv_bfloat16` or `__nv_bfloat162`. The operation is using the corresponding cache operator (see [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) )

## 10.13.时间函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#time-function "这个标题的永久链接")
```c++
clock_t clock();
long long int clock64();
```
在设备代码中执行时，返回每个时钟周期递增的每多处理器计数器的值。在内核的开头和结尾对这个计数器进行采样，取两个样本的差值，并记录每个线程的结果，为每个线程提供了设备为完全执行线程所采取的时钟周期数的度量，但不能为设备实际执行线程指令所花费的时钟周期数提供了衡量标准。前者的数量大于后者，因为线程是时间切片的。

## 10.14.原子函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions "这个标题的永久链接")

原子函数对位于全局或共享内存中的一个32位、64位或128位字执行读-修改-写原子操作。在`float2`或`float4`的情况下，对位于全局内存中的向量的每个元素执行读-修改-写操作。例如，`atomicAdd()`在全局或共享内存中的某个地址读取一个单词，向其添加一个数字，并将结果写回同一地址。原子函数只能在设备函数中使用。

本节中描述的原子函数具有排序[cuda::memory_order_relaxed](https://en.cppreference.com/w/cpp/atomic/memory_order)，并且仅在特定[范围内](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes)是原子的：

- 带有`_system`后缀的原子API（例如：`atomicAdd_system`）如果满足特定[条件，则](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#atomicity)在范围`cuda::thread_scope_system`是原子的。
    
- 没有后缀的原子API（例如：`atomicAdd`）在范围`cuda::thread_scope_device`上是原子的。
    
- 带有`_block`后缀的原子API（例如：`atomicAdd_block`）在范围`cuda::thread_scope_block`是原子的。
    

在以下示例中，CPU和GPU都以原子方式更新地址地址的整数值：
```c++
__global__ void mykernel(int *addr) {
  atomicAdd_system(addr, 10);       // only available on devices with compute capability 6.x
}

void foo() {
  int *addr;
  cudaMallocManaged(&addr, 4);
  *addr = 0;

   mykernel<<<...>>>(addr);
   __sync_fetch_and_add(addr, 10);  // CPU atomic operation
}
```
请注意，任何原子操作都可以基于`atomicCAS()`比较和交换）实现。例如，用于双精度浮点数的`atomicAdd()`在计算能力低于6.0的设备上不可用，但可以实现如下：
```c++
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
```
以下设备范围的原子API有系统范围和块范围的变体，但有以下例外：
- 计算能力小于6.0的设备仅支持设备范围的原子操作，
- 计算能力小于7.2的Tegra设备不支持全系统原子操作。

CUDA 12.8及更高版本支持CUDA编译器内置函数，用于具有内存顺序和线程范围的原子操作。我们遵循[GNU的原子内置函数签名](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html)，并附加了线程范围的参数。我们使用以下原子操作内存顺序和线程范围：
```c++
enum {
   __NV_ATOMIC_RELAXED,
   __NV_ATOMIC_CONSUME,
   __NV_ATOMIC_ACQUIRE,
   __NV_ATOMIC_RELEASE,
   __NV_ATOMIC_ACQ_REL,
   __NV_ATOMIC_SEQ_CST
};

enum {
   __NV_THREAD_SCOPE_THREAD,
   __NV_THREAD_SCOPE_BLOCK,
   __NV_THREAD_SCOPE_CLUSTER,
   __NV_THREAD_SCOPE_DEVICE,
   __NV_THREAD_SCOPE_SYSTEM
};
```
示例：
```c++
__device__ T __nv_atomic_load_n(T* ptr, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```
T可以是大小为1、2、4、8和16字节的任何整数类型。

这些原子函数不能在本地存储器上运行。例如：
```c++
__device__ void foo() {
   int a = 1; // defined in local memory
   int b;
   __nv_atomic_load(&a, &b, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
}
```
这些函数只能在`__device__`函数的块范围内使用。例如：
```c++
__device__ void foo() {
   __shared__ unsigned int u1 = 1;
   __shared__ unsigned int u2 = 2;
   __nv_atomic_load(&u1, &u2, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
}
```
并且无法获得这些函数的地址。以下是三个不受支持的例子：
```c++
// Not permitted to be used in a host function
__host__ void bar() {
   __shared__ unsigned int u1 = 1;
   __shared__ unsigned int u2 = 2;
   __nv_atomic_load(&u1, &u2, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
}

// Not permitted to be used as a template default argument.
// The function address cannot be taken.
template<void *F = __nv_atomic_load_n>
class X {
   void *f = F;
};

// Not permitted to be called in a constructor initialization list.
class Y {
   int a;
public:
   __device__ Y(int *b): a(__nv_atomic_load_n(b, __NV_ATOMIC_RELAXED)) {}
};
```
内存顺序对应于[C++标准原子运算的内存顺序](https://en.cppreference.com/w/cpp/atomic/memory_order)。对于线程范围，我们遵循cuda::thread_scope的[定义](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes)。

`__NV_ATOMIC_CONSUME`内存顺序目前使用更强的`__NV_ATOMIC_ACQUIRE`内存顺序实现。

`__NV_THREAD_SCOPE_THREAD`线程范围目前使用更宽的`__NV_THREAD_SCOPE_BLOCK`线程范围实现。

有关支持的数据类型，请参阅不同原子操作的相应部分。

### 10.14.1.算术函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-functions "这个标题的永久链接")

#### 10.14.1.1.原子添加（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd "这个标题的永久链接")
```c++
int atomicAdd(int* address, int val);
unsigned int atomicAdd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAdd(unsigned long long int* address,
                                 unsigned long long int val);
float atomicAdd(float* address, float val);
double atomicAdd(double* address, double val);
__half2 atomicAdd(__half2 *address, __half2 val);
__half atomicAdd(__half *address, __half val);
__nv_bfloat162 atomicAdd(__nv_bfloat162 *address, __nv_bfloat162 val);
__nv_bfloat16 atomicAdd(__nv_bfloat16 *address, __nv_bfloat16 val);
float2 atomicAdd(float2* address, float2 val);
float4 atomicAdd(float4* address, float4 val);
```
读取位于全局或共享内存中地址的16位、32位或64位`old`，计算`(oldval)`并将结果存储回同一地址的内存中。这三个操作在一次原子交易中执行。该函数返回`old`。

- `atomicAdd()`的32位浮点版本仅由具有2.x及更高计算能力的设备支持。
- `atomicAdd()`的64位浮点版本仅由具有6.x及更高计算能力的设备支持。

`atomicAdd()`的32位`__half2`浮点版本仅由具有6.x及以上计算能力的设备支持。`__half2`或`__nv_bfloat162`添加操作的原子性对两个`__half`或`__nv_bfloat16`元素中的每一个分别保证；整个`__half2`或`__nv_bfloat162`不能保证作为单个32位访问是原子的。

`atomicAdd()`的`float2`和`float4`浮点向量版本仅由计算能力9.x及以上的设备支持。`float2`或`float4`添加操作的原子性对两个或四个`float`元素中的每一个都单独保证；整个`float2`或`float4`不能保证作为单个64位或128位访问是原子的。

- `atomicAdd()`的16位`__half`版本仅由具有7.x及以上计算能力的设备支持。
- `atomicAdd()`的16位`__nv_bfloat16`浮点版本仅由计算能力8.x及更高级别的设备支持。
- `atomicAdd()`的`float2`和`float4`浮点向量版本仅由计算能力9.x及以上的设备支持。
- `atomicAdd()`的`float2`和`float4`浮点向量版本仅支持全局内存地址。

#### 10.14.1.2.原子子（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicsub "这个标题的永久链接")
```c++
int atomicSub(int* address, int val);
unsigned int atomicSub(unsigned int* address,
                       unsigned int val);
```
reads the 32-bit word `old` located at the address `address` in global or shared memory, computes `(old - val)`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

#### 10.14.1.3.原子Exch（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicexch "这个标题的永久链接")
```c++
int atomicExch(int* address, int val);
unsigned int atomicExch(unsigned int* address,
                        unsigned int val);
unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val);
float atomicExch(float* address, float val);
```
reads the 32-bit or 64-bit word `old` located at the address `address` in global or shared memory and stores `val` back to memory at the same address. These two operations are performed in one atomic transaction. The function returns `old`.
```c++
template<typename T> T atomicExch(T* address, T val);
```
读取位于全局或共享内存中地址的128位`old`字，并将`val`存储回同一地址的内存中。这两个操作在一个原子交易中执行。该函数返回`old`。`T`必须满足以下要求：
```c++
sizeof(T) == 16
alignof(T) >= 16
std::is_trivially_copyable<T>::value == true
// for C++03 and older
std::is_default_constructible<T>::value == true
```
因此，`T`必须是128位并正确对齐，可以简单复制，在C++03或更早版本上，它也必须是默认可建的。

128位`atomicExch()`仅受计算能力9.x及更高的设备支持。

#### 10.14.1.4.原子最小（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicmin "这个标题的永久链接")
```c++
int atomicMin(int* address, int val);
unsigned int atomicMin(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMin(long long int* address,
                                long long int val);
```
读取位于全局或共享内存中地址`address`的32位或64位字，计算`old`和`val`的最小值，并将结果存储回同一地址的内存中。这三个操作在一次原子交易中执行。该函数返回`old`。

64位版本的`atomicMin()`仅支持具有5.0及更高计算能力的设备。

#### 10.14.1.5.原子最大（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicmax "这个标题的永久链接")
```c++
int atomicMax(int* address, int val);
unsigned int atomicMax(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMax(long long int* address,
                                 long long int val);
```
读取位于全局或共享内存中地址`address`的32位或64位单词，计算`old`和`val`的最大值，并将结果存储回同一地址的内存中。这三个操作在一次原子交易中执行。该函数返回`old`。

`atomicMax()`的64位版本仅受计算能力5.0及更高版本的设备支持。

#### 10.14.1.6.原子公司（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicinc "这个标题的永久链接")
```c++
unsigned int atomicInc(unsigned int* address,
                       unsigned int val);
```
reads the 32-bit word `old` located at the address `address` in global or shared memory, computes `((old >= val) ? 0 : (old+1))`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

#### 10.14.1.7.原子十二（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicdec "这个标题的永久链接")
```c++
unsigned int atomicDec(unsigned int* address,
                       unsigned int val);
```
reads the 32-bit word `old` located at the address `address` in global or shared memory, computes `(((old == 0) || (old > val)) ? val : (old-1)` ), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

#### 10.14.1.8.原子CAS（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomiccas "这个标题的永久链接")
```c++
int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val);
unsigned short int atomicCAS(unsigned short int *address,
                             unsigned short int compare,
                             unsigned short int val);
```
读取位于全局或共享内存中地址的16位、32位或64位`old`字，计算`(old==compare?val:old)`并将结果存储在同一地址的内存中。这三个操作在一次原子交易中执行。该函数返回（比较和交换）。
```c++
template<typename T> T atomicCAS(T* address, T compare, T val);
```
读取位于全局或共享内存中地址`address`的128位单词，计算`(old==compare?val:old)`并将结果存储在同一地址的内存中。这三个操作在一次原子交易中执行。该函数返回`old`（比较和交换）。`T`必须满足以下要求：
```c++
sizeof(T) == 16
alignof(T) >= 16
std::is_trivially_copyable<T>::value == true
// for C++03 and older
std::is_default_constructible<T>::value == true
```
因此，`T`必须是128位并正确对齐，可以简单复制，在C++03或更早版本上，它也必须是默认可建的。

128位`atomicCAS()`仅受计算能力为9.x及更高的设备支持。

#### 10.14.1.9. __nv_原子_交换（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-exchange "这个标题的永久链接")
```c++
__device__ void __nv_atomic_exchange(T* ptr, T* val, T *ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
```
CUDA 12.8中引入了这个原子函数。它读取`ptr`指向的值，并将`ret`指向的值存储到ret指向的值。它读取`val`指向的值，并存储`ptr`指向的值。

这是一种通用的原子交换，这意味着`T`可以是大小为4、8或16字节的任何数据类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

架构`sm_90`及更高版本支持16字节数据类型。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

#### 10.14.1.10. __nv_原子_交换_n()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-exchange-n "这个标题的永久链接")

__device__ T __nv_atomic_exchange_n(T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这个原子函数。它读取`ptr`指向的值，并将此值用作返回值。它将`val`存储到`ptr`指向的地方。

这是一个非通用原子交换，这意味着`T`只能是大小为4、8或16字节的整数类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

架构`sm_90`及更高版本支持16字节数据类型。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

#### 10.14.1.11. __nv_原子_比较_交换（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-compare-exchange "这个标题的永久链接")

__device__ bool __nv_atomic_compare_exchange (T* ptr, T* expected, T* desired, bool weak, int success_order, int failure_order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这个原子函数。它读取`ptr`指向的值，并将其与`expected`指向的值进行比较。如果它们相等，则返回值为`true`，`desired`指向的值存储在`ptr`指向的位置。否则，它返回`false`，`ptr`指向的值存储在`expected`指向的位置。忽略参数`weak`，并在`success_order`和`failure_order`之间选择更强的内存顺序来执行比较和交换操作。

这是一个通用的原子比较和交换，这意味着`T`可以是任何大小为2、4、8或16字节的数据类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

架构`sm_90`及更高版本支持16字节数据类型。

`sm_70`及更高架构支持2字节数据类型。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

#### 10.14.1.12. __nv_原子_比较_交换_n（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-compare-exchange-n "这个标题的永久链接")

__device__ bool __nv_atomic_compare_exchange_n (T* ptr, T* expected, T desired, bool weak, int success_order, int failure_order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这个原子函数。它读取`ptr`指向的值，并将其与`expected`指向的值进行比较。如果它们相等，则返回值为`true`，`desired`存储在`ptr`指向的地方。否则，它返回`false`，`ptr`指向的值存储在`expected`指向的位置。忽略参数`weak`，并在`success_order`和`failure_order`之间选择更强的内存顺序来执行比较和交换操作。

这是一个非通用原子比较和交换，这意味着`T`只能是大小为2、4、8或16字节的整数类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

架构`sm_90`及更高版本支持16字节数据类型。

`sm_70`及更高架构支持2字节数据类型。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

#### 10.14.1.13. __nv_atomic_fetch_add（）和__nv_atomic_add（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-add-and-nv-atomic-add "这个标题的永久链接")

__device__ T __nv_atomic_fetch_add (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_add (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这两个原子函数。它读取`ptr`指向的值，用`val`添加，并将结果存储回`ptr`指向的值。`__nv_atomic_fetch_add`返回`ptr`指向的旧值。`__nv_atomic_add`没有返回值。

`T` can only be `unsigned int`, `int`, `unsigned long long`, `float` or `double`.

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

#### 10.14.1.14. __nv_atomic_fetch_sub（）和__nv_atomic_sub（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-sub-and-nv-atomic-sub "这个标题的永久链接")

__device__ T __nv_atomic_fetch_sub (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_sub (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这两个原子函数。它读取`ptr`指向的值，用`val`进行减去，并将结果存储回`ptr`指向的值。`__nv_atomic_fetch_sub`返回`ptr`指向的旧值。`__nv_atomic_sub`没有返回值。

`T` can only be `unsigned int`, `int`, `unsigned long long`, `float` or `double`.

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

#### 10.14.1.15. __nv_atomic_fetch_min（）和__nv_atomic_min（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-min-and-nv-atomic-min "这个标题的永久链接")

__device__ T __nv_atomic_fetch_min (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_min (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这两个原子函数。它读取`ptr`指向的值，与`val`进行比较，并将较小的值存储回`ptr`指向的值。`__nv_atomic_fetch_min`返回`ptr`指向的旧值。`__nv_atomic_min`没有返回值。

`T` can only be `unsigned int`, `int`, `unsigned long long` or `long long`.

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

#### 10.14.1.16. __nv_atomic_fetch_max（）和__nv_atomic_max（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-max-and-nv-atomic-max "这个标题的永久链接")

__device__ T __nv_atomic_fetch_max (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_max (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这两个原子函数。它读取`ptr`指向的值，与`val`进行比较，并将更大的值存储回`ptr`指向的位置。`__nv_atomic_fetch_max`返回`ptr`指向的旧值。`__nv_atomic_max`没有返回值。

`T` can only be `unsigned int`, `int`, `unsigned long long` or `long long`.

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

### 10.14.2.位函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#bitwise-functions "这个标题的永久链接")

#### 10.14.2.1.原子和（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicand "这个标题的永久链接")

int atomicAnd(int* address, int val);
unsigned int atomicAnd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val);

读取位于全局或共享内存中地址的32位或64位`old`字，计算`(old`），并将结果存储回同一地址的内存中。这三个操作在一次原子交易中执行。该函数返回`old`。

`atomicAnd()`的64位版本仅由具有5.0及更高计算能力的设备支持。

#### 10.14.2.2. 原子或（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicor "这个标题的永久链接")

int atomicOr(int* address, int val);
unsigned int atomicOr(unsigned int* address,
                      unsigned int val);
unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val);

reads the 32-bit or 64-bit word `old` located at the address `address` in global or shared memory, computes `(old | val)`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

`atomicOr()`的64位版本仅由计算能力5.0及更高版本的设备支持。

#### 10.14.2.3.原子Xor（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicxor "这个标题的永久链接")

int atomicXor(int* address, int val);
unsigned int atomicXor(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicXor(unsigned long long int* address,
                                 unsigned long long int val);

读取位于全局或共享内存中地址`address`的32位或64位`old`字，计算`(oldval)`并将结果存储回同一地址的内存中。这三个操作在一次原子交易中执行。该函数返回`old`。

仅计算能力5.0及更高版本的设备支持`atomicXor()`的64位版本。

#### 10.14.2.4. __nv_atomic_fetch_or()和__nv_atomic_or()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-or-and-nv-atomic-or "这个标题的永久链接")

__device__ T __nv_atomic_fetch_or (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_or (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这两个原子函数。它读取`ptr`指向的值，`or`带有`val`的值，并将结果存储回`ptr`指向的值。`__nv_atomic_fetch_or`返回`ptr`指向的旧值。`__nv_atomic_or`没有返回值。

`T`只能是大小为4或8字节的整数类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

#### 10.14.2.5. __nv_atomic_fetch_xor（）和__nv_atomic_xor（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-xor-and-nv-atomic-xor "这个标题的永久链接")

__device__ T __nv_atomic_fetch_xor (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_xor (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这两个原子函数。它读取`ptr`指向的值，`xor`与`val`，并将结果存储回`ptr`指向的值。`__nv_atomic_fetch_xor`返回`ptr`指向的旧值。`__nv_atomic_xor`没有返回值。

`T`只能是大小为4或8字节的整数类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

#### 10.14.2.6. __nv_atomic_fetch_and（）和__nv_atomic_and（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-and-and-nv-atomic-and "这个标题的永久链接")

__device__ T __nv_atomic_fetch_and (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_and (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这两个原子函数。它读取`ptr`指向的值，`and`带有`val`，并将结果存储回`ptr`指向的位置。`__nv_atomic_fetch_and`返回`ptr`指向的旧值。`__nv_atomic_and`没有返回值。

`T`只能是大小为4或8字节的整数类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

### 10.14.3.其他原子函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#other-atomic-functions "这个标题的永久链接")

#### 10.14.3.1. __nv_原子_负载（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-load "这个标题的永久链接")

__device__ void __nv_atomic_load(T* ptr, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这个原子函数。它加载`ptr`指向的值，并将`ret`指向的值写入。

这是一个通用的原子负载，这意味着`T`可以是大小为1、2、4、8或16字节的任何数据类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`sm_70`及更高版本的架构支持16字节数据类型。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。`order`不能是`__NV_ATOMIC_RELEASE`或`__NV_ATOMIC_ACQ_REL`。

#### 10.14.3.2. __nv_原子_负载_n（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-load-n "这个标题的永久链接")

__device__ T __nv_atomic_load_n(T* ptr, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这个原子函数。它加载`ptr`指向的值并返回此值。

这是一个非通用的原子荷载，这意味着`T`只能是大小为1、2、4、8或16字节的整数类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`sm_70`及更高版本的架构支持16字节数据类型。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。`order`不能是`__NV_ATOMIC_RELEASE`或`__NV_ATOMIC_ACQ_REL`。

#### 10.14.3.3. __nv_原子_商店（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-store "这个标题的永久链接")

__device__ void __nv_atomic_store(T* ptr, T* val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这个原子函数。它读取`val`指向的值，并存储到`ptr`指向的值。

这是一个通用的原子负载，这意味着`T`可以是大小为1、2、4、8或16字节的任何数据类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`sm_70`及更高版本的架构支持16字节数据类型。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。`order`不能是`__NV_ATOMIC_CONSUME`,`__NV_ATOMIC_ACQUIRE`或`__NV_ATOMIC_ACQ_REL`。

#### 10.14.3.4. __nv_原子_商店_n（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-store-n "这个标题的永久链接")

__device__ void __nv_atomic_store_n(T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

CUDA 12.8中引入了这个原子函数。它存储了`ptr`指向的`val`。

这是一个非通用的原子荷载，这意味着`T`只能是大小为1、2、4、8或16字节的整数类型。

架构`sm_60`及更高版本支持具有内存顺序和线程范围的原子操作。

`sm_70`及更高版本的架构支持16字节数据类型。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。`order`不能是`__NV_ATOMIC_CONSUME`,`__NV_ATOMIC_ACQUIRE`或`__NV_ATOMIC_ACQ_REL`。

#### 10.14.3.5. __nv_原子_线程_围栏（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-thread-fence "这个标题的永久链接")

__device__ void __nv_atomic_thread_fence (int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

该原子函数基于指定的内存顺序，在此线程请求的内存访问之间建立排序。线程范围参数指定了可以观察此操作的排序效果的线程集。

`cluster`的线程范围支持架构`sm_90`及更高版本。

参数`order`和`scope`需要是整数字面值，即参数不能是变量。

## 10.15.地址空间谓词函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#address-space-predicate-functions "这个标题的永久链接")

如果参数是空指针，则本节中描述的函数具有未指定的行为。

### 10.15.1.__是全局的（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isglobal "这个标题的永久链接")

__device__ unsigned int __isGlobal(const void *ptr);

如果`ptr`包含全局内存空间中对象的通用地址，则返回1，否则返回0。

### 10.15.2. __是共享的（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isshared "这个标题的永久链接")

__device__ unsigned int __isShared(const void *ptr);

如果`ptr`包含共享内存空间中对象的通用地址，则返回1，否则返回0。

### 10.15.3. __是常数（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isconstant "这个标题的永久链接")

__device__ unsigned int __isConstant(const void *ptr);

如果`ptr`包含常量内存空间中对象的通用地址，则返回1，否则返回0。

### 10.15.4. __是网格常数（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isgridconstant "这个标题的永久链接")

__device__ unsigned int __isGridConstant(const void *ptr);

如果`ptr`包含用`__grid_constant__`注释的内核参数的通用地址，则返回1，否则返回0。仅支持大于或等于7x或更高版本的计算架构。

### 10.15.5. __是本地的（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#islocal "这个标题的永久链接")

__device__ unsigned int __isLocal(const void *ptr);

如果`ptr`包含本地内存空间中对象的通用地址，则返回1，否则返回0。

## 10.16.地址空间转换功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#address-space-conversion-functions "这个标题的永久链接")

### 10.16.1. __cvta_通用_到_全局（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-generic-to-global "这个标题的永久链接")

__device__ size_t __cvta_generic_to_global(const void *ptr);

返回在`ptr`表示的通用地址上执行_PTXcvta.to.global_指令的结果。

### 10.16.2. __cvta_通用_共享（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-generic-to-shared "这个标题的永久链接")

__device__ size_t __cvta_generic_to_shared(const void *ptr);

返回在`ptr`表示的通用地址上执行_PTXcvta.to.shared_指令的结果。

### 10.16.3. __cvta_通用_到_常数（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-generic-to-constant "这个标题的永久链接")

__device__ size_t __cvta_generic_to_constant(const void *ptr);

返回在`ptr`表示的通用地址上执行_PTXcvta.to.const_指令的结果。

### 10.16.4. __cvta_通用_到_本地（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-generic-to-local "这个标题的永久链接")

__device__ size_t __cvta_generic_to_local(const void *ptr);

返回在`ptr`表示的通用地址上执行_PTXcvta.to.local_指令的结果。

### 10.16.5. __cvta_global_to_generic（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-global-to-generic "这个标题的永久链接")

__device__ void * __cvta_global_to_generic(size_t rawbits);

返回通过在`rawbits`提供的值上执行_PTXcvta.global_指令获得的通用指针。

### 10.16.6. __cvta_共享_到_通用（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-shared-to-generic "这个标题的永久链接")

__device__ void * __cvta_shared_to_generic(size_t rawbits);

返回通过在`rawbits`提供的值上执行_PTXcvta.shared_指令获得的通用指针。

### 10.16.7. __cvta_constant_to_generic（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-constant-to-generic "这个标题的永久链接")

__device__ void * __cvta_constant_to_generic(size_t rawbits);

返回通过在`rawbits`提供的值上执行_PTXcvta.const_指令获得的通用指针。

### 10.16.8. __cvta_本地_到_通用（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-local-to-generic "这个标题的永久链接")

__device__ void * __cvta_local_to_generic(size_t rawbits);

返回通过在`rawbits`提供的值上执行_PTXcvta.local_指令获得的通用指针。

## 10.17.Alloca函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#alloca-function "这个标题的永久链接")

### 10.17.1.大纲[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synopsis "这个标题的永久链接")

__host__ __device__ void * alloca(size_t size);

### 10.17.2.描述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#description "这个标题的永久链接")

`alloca()`函数在调用者的堆栈帧中分配内存`size`字节。返回的值是指向分配内存的指针，当从设备代码中调用函数时，内存的开头对齐16字节。当返回`alloca()`的呼叫者时，分配的内存会自动释放。

笔记

在Windows平台上，在使用`alloca()`之前必须包含`<malloc.h>`使用`alloca()`可能会导致堆栈溢出，用户需要相应地调整堆栈大小。

它支持5.2或更高的计算能力。

### 10.17.3.示例：[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#example "这个标题的永久链接")
```c++
__device__ void foo(unsigned int num) {
    int4 *ptr = (int4 *)alloca(num * sizeof(int4));
    // use of ptr
    ...
}
```
## 10.18.编译器优化提示功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compiler-optimization-hint-functions "这个标题的永久链接")

本节中描述的功能可用于向编译器优化器提供其他信息。

### 10.18.1. __内置_假设_对齐（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#builtin-assume-aligned "这个标题的永久链接")

void * __builtin_assume_aligned (const void *exp, size_t align)

允许编译器假设参数指针至少对齐字节，并返回参数指针。

示例：

void *res = __builtin_assume_aligned(ptr, 32); // compiler can assume 'res' is
                                               // at least 32-byte aligned

三个参数版本：
```c++
void * __builtin_assume_aligned (const void *exp, size_t align,
                                 <integral type> offset)
```
Allows the compiler to assume that `(char *)exp - offset` is aligned to at least `align` bytes, and returns the argument pointer.

示例：
```c++
void *res = __builtin_assume_aligned(ptr, 32, 8); // compiler can assume
                                                  // '(char *)res - 8' is
                                                  // at least 32-byte aligned.
```
### 10.18.2. __内置_假设（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#builtin-assume "这个标题的永久链接")

void __builtin_assume(bool exp)

允许编译器假设布尔参数为真。如果参数在运行时为真，则行为未定义。请注意，如果参数有副作用，则行为未指定。

示例：

 __device__ int get(int *ptr, int idx) {
   __builtin_assume(idx <= 2);
   return ptr[idx];
}

### 10.18.3. __假设（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#assume "这个标题的永久链接")

void __assume(bool exp)

允许编译器假设布尔参数为真。如果参数在运行时为真，则行为未定义。请注意，如果参数有副作用，则行为未指定。

示例：

 __device__ int get(int *ptr, int idx) {
   __assume(idx <= 2);
   return ptr[idx];
}

### 10.18.4. __内置_预期（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#builtin-expect "这个标题的永久链接")

long __builtin_expect (long exp, long c)

Indicates to the compiler that it is expected that `exp == c`, and returns the value of `exp`. Typically used to indicate branch prediction information to the compiler.

示例：

// indicate to the compiler that likely "var == 0",
// so the body of the if-block is unlikely to be
// executed at run time.
if (__builtin_expect (var, 0))
  doit ();

### 10.18.5. __内置_无法到达（）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#builtin-unreachable "这个标题的永久链接")

void __builtin_unreachable(void)

向编译器表示控制流从未到达调用此函数的点。如果控制流在运行时确实达到这个点，程序有未定义的行为。

示例：

// indicates to the compiler that the default case label is never reached.
switch (in) {
case 1: return 4;
case 2: return 10;
default: __builtin_unreachable();
}

### 10.18.6.限制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#restrictions "这个标题的永久链接")

`__assume()`仅在使用`cl.exe`主机编译器时才支持。所有平台都支持其他功能，但受以下限制：

- 如果主机编译器支持该函数，则可以从翻译单元的任何地方调用该函数。
    
- 否则，该函数必须从`__device__`/ `__global__`function的主体内调用，或者仅在定义了`__CUDA_ARCH__`宏[5](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn12)时调用。
    

## 10.19.扭曲投票函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-vote-functions "这个标题的永久链接")

int __all_sync(unsigned mask, int predicate);
int __any_sync(unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);
unsigned __activemask();

弃用通知：`__any`、`__all`和`__ballot`已在所有CUDA 9.0中对所有设备弃用。

删除通知：当目标计算机能力为7.x或更高的设备时，`__any`、`__all`和`__ballot`不再可用，应使用其同步变体。

经编投票函数允许给定[经编](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture)的线程执行还原和广播操作。这些函数将经编中每个线程的整数`predicate`作为输入，并将这些值与零进行比较。比较结果以以下方式之一在经编的[活跃](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes)线程之间组合（减少），向每个参与线程广播单个返回值：

`__all_sync(unsigned mask, predicate)`：

为`mask`中所有未退出的线程评估`predicate`，并返回非零，当且仅当谓`predicate`评估为非零时。

`__any_sync(unsigned mask, predicate)`：

为`mask`中所有未退出的线程评估`predicate`，并返回非零，当且仅当谓`predicate`对其中任何线程评估为非零时。

`__ballot_sync(unsigned mask, predicate)`：

为`mask`中所有未退出的线程评估`predicate`并返回一个整数，其第N位被设置，当且仅当谓`predicate`对经编的第N个线程评估为非零且第N个线程处于活动状态时。

`__activemask()`：

返回调用经编中所有当前活动线程的32位整数掩码。当调用`__activemask()`时，如果warp中的Nth通道处于活动状态，则设置第N位。[不活跃](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes)的线程在返回的掩码中由0位表示。退出程序的线程总是被标记为不活跃。请注意，在`__activemask()`调用时收敛的线程不能保证在后续指令时收敛，除非这些指令同步warp-builtin函数。

对于`__all_sync`、`__any_sync`和`__ballot_sync`，必须传递一个掩码来指定参与调用的线程。必须为每个参与线程设置一个表示线程通道ID的位，以确保它们在硬件执行内在之前正确收敛。每个调用线程必须在掩码中设置自己的位，掩码中命名的所有未退出线程必须使用相同的掩码执行相同的内在线程，否则结果是未定义的。

这些内在并不意味着记忆障碍。他们不保证任何内存排序。

## 10.20.扭曲匹配函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-match-functions "这个标题的永久链接")

`__match_any_sync`和`__match_all_sync`对[扭曲](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture)中的线程之间的变量执行广播和比较操作。

由计算能力为7倍或更高的设备支持。

### 10.20.1.大纲[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synopsis-match "这个标题的永久链接")
```c++
unsigned int __match_any_sync(unsigned mask, T value);
unsigned int __match_all_sync(unsigned mask, T value, int *pred);
```
`T` can be `int`, `unsigned int`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float` or `double`.

### 10.20.2.描述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-description-match "这个标题的永久链接")

`__match_sync()`内在允许在同步名为inmask的线程后，在扭曲中跨线程的值进行广播和比较。

`__match_any_sync`

返回具有相同值的线程掩码`mask`

`__match_all_sync`

如果`mask`中的所有线程的值相同，则返回`mask`；否则返回0。如果`mask`中的所有线程具有相同的值，则谓词`pred`设置为true；否则谓词设置为false。

新的`*_sync`匹配内在在掩码中表示参与呼叫的线程。必须为每个参与线程设置一个表示线程的通道ID的位，以确保它们在硬件执行内在之前正确收敛。每个调用线程必须在掩码中设置自己的位，掩码中命名的所有未退出线程必须使用相同的掩码执行相同的内在线程，否则结果是未定义的。

这些内在并不意味着记忆障碍。他们不保证任何内存排序。

## 10.21.扭曲减少功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-functions "这个标题的永久链接")

`__reduce_sync(unsignedmask,Tvalue)`内在在同步`mask`中命名的线程后，对以值提供的数据执行还原操作。对于{add, min, max}，T可以是无符号的或有符号的，对于{and, or, xor}操作可以是无符号的。

由计算能力为8.x或更高的设备支持。

### 10.21.1.大纲[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-synopsis "这个标题的永久链接")

// add/min/max
unsigned __reduce_add_sync(unsigned mask, unsigned value);
unsigned __reduce_min_sync(unsigned mask, unsigned value);
unsigned __reduce_max_sync(unsigned mask, unsigned value);
int __reduce_add_sync(unsigned mask, int value);
int __reduce_min_sync(unsigned mask, int value);
int __reduce_max_sync(unsigned mask, int value);

// and/or/xor
unsigned __reduce_and_sync(unsigned mask, unsigned value);
unsigned __reduce_or_sync(unsigned mask, unsigned value);
unsigned __reduce_xor_sync(unsigned mask, unsigned value);

### 10.21.2.描述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-reduce-description "这个标题的永久链接")

`__reduce_add_sync`，`__reduce_min_sync`，`__reduce_max_sync`

返回对每个线程在名为inmask的`value`提供的值应用算术加、最小值或最大减小运算的结果。

`__reduce_and_sync`，`__reduce_or_sync`，`__reduce_xor_sync`

返回对`mask`中命名的每个线程在`value`提供的值上应用逻辑AND、OR或XOR还原操作的结果。

`mask`表示参与通话的线程。必须为每个参与线程设置一个表示线程的通道ID的位，以确保它们在硬件执行内在之前正确收敛。每个调用线程必须在掩码中设置自己的位，掩码中命名的所有未退出线程必须使用相同的掩码执行相同的内在线程，否则结果是未定义的。

这些内在并不意味着记忆障碍。他们不保证任何内存排序。

## 10.22.扭曲洗牌功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions "这个标题的永久链接")

`__shfl_sync`，`__shfl_up_sync`、`__shfl_down_sync`和`__shfl_xor_sync`在[经编](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture)中的线程之间交换变量。

由计算能力5.0或更高版本的设备支持。

弃用通知：`__shfl`、`__shfl_up`、`__shfl_down`和`__shfl_xor`已在所有设备CUDA 9.0中弃用。

删除通知：当目标计算机能力为7.x或更高的设备时，`__shfl`、`__shfl_up`、`__shfl_down`和`__shfl_xor`不再可用，应使用其同步变体。

### 10.22.1.大纲[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-synopsis "这个标题的永久链接")

T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);

`T` can be `int`, `unsigned int`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float` or `double`. With the `cuda_fp16.h` header included, `T`can also be `__half` or `__half2`. Similarly, with the `cuda_bf16.h` header included, `T` can also be `__nv_bfloat16` or `__nv_bfloat162`.

### 10.22.2.描述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-description "这个标题的永久链接")

`__shfl_sync()`内在允许在不使用共享内存的情况下在经编中的线程之间交换变量。交换同时发生在经编中的所有[活动](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes)线程（并在`mask`中命名），根据类型，每个线程移动4或8字节的数据。

经编中的线程被称为_车道_，索引可能在0和`warpSize-1`（含）之间。支持四种源通道寻址模式：

`__shfl_sync()`

来自索引通道的直接复制

`__shfl_up_sync()`

从相对于呼叫者具有较低ID的通道复制

`__shfl_down_sync()`

从相对于呼叫者具有较高ID的通道复制

`__shfl_xor_sync()`

根据自身车道ID的按位XOR从车道复制

线程只能从另一个积极参与`__shfl_sync()`命令的线程中读取数据。如果目标线程处于[非活动状态](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes)，则检索到的值是未定义的。

所有`__shfl_sync()`内在都采取一个可选的`width`参数，该参数会改变内在的行为。`width`必须在[1，warpSize]范围内（即1、2、4、8、16或32）中具有二的弥的�力。其他值的结果未定义。

`__shfl_sync()`返回由`srcLane`给出的ID线程持有的`var`。如果宽度小于`warpSize`那么warp的每个子部分都表现为一个单独的实体，起始逻辑车道ID为0。如果`srcLane`超出`[0:width-1]`范围，则返回的值对应于`srcLanemodulo`（即在同一小节内）持有的var值。

`__shfl_up_sync()`通过从呼叫者的车道ID中减去`delta`来计算源车道ID。返回由结果车道ID持有的`var`：实际上，`var`被`delta`车道向翘曲移动。如果宽度小于`warpSize`那么warp的每个子部分都表现为一个单独的实体，起始逻辑车道ID为0。源车道索引不会环绕`width`值，因此实际上较低的`delta`车道将保持不变。

`__shfl_down_sync()`通过向呼叫者的车道ID添加`delta`来计算源车道ID。返回由生成的车道ID持有的`var`：这具有通过`delta`车道将`var`向下移动的效果。如果宽度小于`warpSize`那么warp的每个子部分都表现为一个单独的实体，起始逻辑车道ID为0。至于`__shfl_up_sync()`源通道的ID号不会环绕宽度值，因此上`delta`通道将保持不变。

`__shfl_xor_sync()`通过使用`laneMask`对呼叫者的车道ID执行位XOR来计算源线ID：返回由结果车道ID持有的`var`。如果`width`小于`warpSize`，那么每组`width`连续线程都可以访问早期线程组的元素，但是，如果他们尝试访问后期线程组的元素，则将返回自己的`var`。此模式实现了蝴蝶寻址模式，例如用于树减少和广播。

The new `*_sync` shfl intrinsics take in a mask indicating the threads participating in the call. A bit, representing the thread’s lane id, must be set for each participating thread to ensure they are properly converged before the intrinsic is executed by the hardware. Each calling thread must have its own bit set in the mask and all non-exited threads named in mask must execute the same intrinsic with the same mask, or the result is undefined.

线程只能从另一个积极参与`__shfl_sync()`命令的线程中读取数据。如果目标线程处于非活动状态，则检索到的值是未定义的。

这些内在并不意味着记忆障碍。他们不保证任何内存排序。

### 10.22.3.实例[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#examples "这个标题的永久链接")
```c++
#### 10.22.3.1.跨经编广播单个值[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#broadcast-of-a-single-value-across-a-warp "这个标题的永久链接")

#include <stdio.h>

__global__ void bcast(int arg) {
    int laneId = threadIdx.x & 0x1f;
    int value;
    if (laneId == 0)        // Note unused variable for
        value = arg;        // all threads except lane 0
    value = __shfl_sync(0xffffffff, value, 0);   // Synchronize all threads in warp, and get "value" from lane 0
    if (value != arg)
        printf("Thread %d failed.\n", threadIdx.x);
}

int main() {
    bcast<<< 1, 32 >>>(1234);
    cudaDeviceSynchronize();

    return 0;
}
```
#### 10.22.3.2.跨8个线程子分区的包容性加扫描[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#inclusive-plus-scan-across-sub-partitions-of-8-threads "这个标题的永久链接")
```c++
#include <stdio.h>

__global__ void scan4() {
    int laneId = threadIdx.x & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    int value = 31 - laneId;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i=1; i<=4; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        int n = __shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }

    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    scan4<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}
```

#### 10.22.3.3.跨越经编的减少[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#reduction-across-a-warp "这个标题的永久链接")

#include <stdio.h>

__global__ void warpReduce() {
    int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    int value = 31 - laneId;

    // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);

    // "value" now contains the sum across all threads
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    warpReduce<<< 1, 32 >>>();
    cudaDeviceSynchronize();

    return 0;
}

## 10.23.纳米睡眠功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nanosleep-function "这个标题的永久链接")

### 10.23.1.大纲[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nanosleep-synopsis "这个标题的永久链接")

void __nanosleep(unsigned ns);

### 10.23.2.描述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nanosleep-description "这个标题的永久链接")

`__nanosleep(ns)`暂停线程的睡眠时间约为`ns`秒。最长睡眠时间约为1毫秒。

它支持7.0或更高的计算能力。

### 10.23.3.示例：[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nanosleep-example "这个标题的永久链接")

以下代码实现了具有指数回退的互斥。

__device__ void mutex_lock(unsigned int *mutex) {
    unsigned int ns = 8;
    while (atomicCAS(mutex, 0, 1) == 1) {
        __nanosleep(ns);
        if (ns < 256) {
            ns *= 2;
        }
    }
}

__device__ void mutex_unlock(unsigned int *mutex) {
    atomicExch(mutex, 0);
}

## 10.24.扭曲矩阵函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-matrix-functions "这个标题的永久链接")

C++扭曲矩阵操作利用张量核心来加速`D=A*B+C`形式的矩阵问题。这些操作支持计算能力7.0或更高的设备的混合精度浮点数据。这需要所有线程的合作。此外，只有在条件在整个[经编](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture)中评估相同时，条件代码中才允许这些操作，否则代码执行可能会挂起。

### 10.24.1.描述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-description "这个标题的永久链接")

以下所有函数和类型都在命名空间`nvcuda::wmma`中定义。子字节操作被视为预览，即它们的数据结构和API可能会发生变化，并且可能与未来版本不兼容。这个额外的功能在thenvcuda`nvcuda::wmma::experimental`命名空间中定义。

template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);

`fragment`

一个超载类，包含分布在经编中所有线程上的矩阵部分。将矩阵元素映射到`fragment`内部存储是未指定的，并且可能会在未来架构中发生变化。

只允许某些模板参数组合。第一个模板参数指定片段将如何参与矩阵操作。`Use`可接受值是：

- `matrix_a`当片段用作第一个乘数时，`A`，
    
- `matrix_b`当片段用作第二乘数时，`B`，或者
    
- `accumulator`当片段用作源或目标累加器（分别为`C`或`D`。
    
    The `m`, `n` and `k` sizes describe the shape of the warp-wide matrix tiles participating in the multiply-accumulate operation. The dimension of each tile depends on its role. For `matrix_a` the tile takes dimension `m x k`; for `matrix_b` the dimension is `k x n`, and `accumulator` tiles are `m xn`.
    
    数据类型`T`可以是`double`、`float`、`__half`、`__nv_bfloat16`、`char`或乘数的`unsignedchar`，累加数可以是`double`、`float`、`int`或`__half`。如[元素类型和矩阵大小中](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-type-sizes)所述，支持累加器和乘数类型的有限组合。必须为`matrix_a`和`matrix_b`片段指定布局参数。`row_major`或`col_major`表示矩阵行或列中的元素在内存中分别是连续的。`accumulator`矩阵的`Layout`参数应保留`void`的默认值。仅当累加器加载或存储时，才会指定行或列布局，如下所述。
    

`load_matrix_sync`

等到所有经线通道到达load_matrix_sync，然后从内存中加载矩阵片段a。mptr必须是256位对齐指针，指向内存中矩阵的第一个元素。`ldm`描述连续行（行主要布局）或列（列主要布局）之间元素的步长，`__half`类型必须是8的倍数或`float`元素类型的4的倍数。（即在这两种情况下都是16字节的倍数）。如果片段是`accumulator`，则`layout`参数必须指定为`mem_row_major`或`mem_col_major`。对于`matrix_a`和`matrix_b`片段，布局是从片段的`layout`参数中推断出来的。`mptr`、`ldm`、`layout`和所有模板参数的值对于经编中的所有线程都必须相同。该函数必须由经编中的所有线程调用，否则结果是未定义的。

`store_matrix_sync`

等到所有经编通道到达store_matrix_sync，然后将矩阵片段a存储到内存中。`mptr`必须是指向内存中矩阵的第一个元素的256位对齐指针。`ldm`描述了连续行（对于行主要布局）或列（对于列主要布局）之间的元素的步长，并且必须是`__half`类型的8的倍数或`float`元素类型的4的倍数。（即在这两种情况下都是16字节的倍数）。输出矩阵的布局必须指定为`mem_row_major`或`mem_col_major`。对于经编中的所有线程，a的ofmptr、`ldm`、`layout`和所有模板参数的值必须相同。

`fill_fragment`

用常数值`v`填充矩阵片段。由于矩阵元素到每个片段的映射是未指定的，因此该函数通常由经编中所有线程调用，具有`v`的通用值。

`mma_sync`

等到所有经编通道到达mma_sync，然后执行经编同步矩阵乘积运算`D=A*B+C`。也支持原位操作，`C=A*B+C`。对于经编中的所有线程，每个矩阵片段的`satf`和模板参数的值必须相同。此外，模板参数`m`和`k`必须在片段`A`、`C`之间匹配。该函数必须由经编中的所有线程调用，否则结果是未定义的。

如果`satf`（饱和到有限值）模式为`true`，则以下附加数值属性适用于目标累加器：

- 如果元素结果是+Infinity，相应的累加器将包含`+MAX_NORM`
    
- 如果元素结果是-Infinity，相应的累加器将包含`-MAX_NORM`
    
- 如果元素结果是NaN，则相应的累加器将包含`+0`
    

由于矩阵元素进入每个线程`fragment`的映射未指定，因此在调用`store_matrix_sync`后，必须从内存（共享或全局）访问单个矩阵元素。在扭曲中的所有线程将统一对所有片段元素应用元素操作的特殊情况下，可以使用以下`fragment`类成员实现直接元素访问。

enum fragment<Use, m, n, k, T, Layout>::num_elements;
T fragment<Use, m, n, k, T, Layout>::x[num_elements];

例如，以下代码将`accumulator`矩阵图块缩放一半。

wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
float alpha = 0.5f; // Same value for all threads in warp
/*...*/
for(int t=0; t<frag.num_elements; t++)
frag.x[t] *= alpha;

### 10.24.2.备用浮点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#alternate-floating-point "这个标题的永久链接")

Tensor Cores支持计算能力8.0及更高级别的设备上的替代类型的浮点运算。

`__nv_bfloat16`

这种数据格式是一种替代的fp16格式，其范围与f32相同，但精度降低（7位）。您可以将此数据格式直接与`cuda_bf16.h`中可用的`__nv_bfloat16`类型一起使用。具有`__nv_bfloat16`数据类型的矩阵片段需要用`float`类型的累加器组成。支持的形状和操作与`__half`相同。

`tf32`

此数据格式是张量核心支持的特殊浮点格式，其范围与f32相同，精度降低（>=10位）。这种格式的内部布局是实现定义的。为了将这种浮点格式与WMMA操作一起使用，输入矩阵必须手动转换为tf32精度。

为了方便转换，提供了一个新的内在`__float_to_tf32`。虽然内在的输入和输出参数是`float`类型，但输出在数值上将是`tf32`。这个新的精度仅用于张量核心，如果与其他`float`类型运算混合，结果的精度和范围将未定义。

一旦输入矩阵（`matrix_a`或`matrix_b`）转换为tf32精度，将`fragment`与`precision::tf32`精度和`float`到`load_matrix_sync`的数据类型的组合将利用这一新功能。两个累加器片段都必须有`float`数据类型。唯一支持的矩阵大小是16x16x8（m-n-k）。

片段的元素表示为`float`，因此从`element_type<T>`到`storage_element_type<T>`的映射是：

precision::tf32 -> float

### 10.24.3.双重精度[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#double-precision "这个标题的永久链接")

Tensor Cores支持计算能力8.0及更高的设备上的双精度浮点操作。要使用这个新功能，必须使用具有`double`的`fragment`。`mma_sync`操作将使用.rn（四舍五入至最接近的偶）四舍五入修饰符执行。

### 10.24.4.子字节操作[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sub-byte-operations "这个标题的永久链接")

子字节WMMA操作提供了一种访问张量核心的低精度能力的方法。它们被视为预览功能，即它们的数据结构和API可能会发生变化，并且可能与未来的版本不兼容。此功能可通过thenvcuda`nvcuda::wmma::experimental`命名空间获得：
```c++
namespace experimental {
    namespace precision {
        struct u4; // 4-bit unsigned
        struct s4; // 4-bit signed
        struct b1; // 1-bit
   }
    enum bmmaBitOp {
        bmmaBitOpXOR = 1, // compute_75 minimum
        bmmaBitOpAND = 2  // compute_80 minimum
    };
    enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 };
}
```
对于4位精度，可用的API保持不变，但您必须指定`experimental::precision::u4`或`experimental::precision::s4`作为片段数据类型。由于片段的元素是打包在一起的，`num_storage_elements`将小于该片段的`num_elements`。子字节片段的`num_elements`变量，因此返回子字节类型`element_type<T>`的元素数。单位精度也是如此，在这种情况下，从`element_type<T>`到`storage_element_type<T>`的映射如下：

experimental::precision::u4 -> unsigned (8 elements in 1 storage element)
experimental::precision::s4 -> int (8 elements in 1 storage element)
experimental::precision::b1 -> unsigned (32 elements in 1 storage element)
T -> T  //all other types

子字节片段允许的布局始终是`matrix_a`的`row_major`和`matrix_b`的`col_major`。

For sub-byte operations the value of `ldm` in `load_matrix_sync` should be a multiple of 32 for element type `experimental::precision::u4` and `experimental::precision::s4` or a multiple of 128 for element type `experimental::precision::b1` (i.e., multiple of 16 bytes in both cases).

笔记

对MMA指令的以下变体的支持已被弃用，并将在sm_90中删除：

> - `experimental::precision::u4`
>     
> - `experimental::precision::s4`
>     
> - `experimental::precision::b1``bmmaBitOp`设置为`bmmaBitOpXOR`
>     

`bmma_sync`

Waits until all warp lanes have executed bmma_sync, and then performs the warp-synchronous bit matrix multiply-accumulate operation `D =(A op B) + C`, where `op` consists of a logical operation `bmmaBitOp` followed by the accumulation defined by `bmmaAccumulateOp`. The available operations are:

`bmmaBitOpXOR`，`matrix_a`中行的128位XOR，128位列为`matrix_b`

`bmmaBitOpAND`，`matrix_a`中带有`matrix_b`的128位列的128位AND，可用于具有8.0及更高计算能力的设备。

累积操作始终是`bmmaAccumulateOpPOPC`，它计算设置的位数。

### 10.24.5.限制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-restrictions "这个标题的永久链接")

张量核心所需的特殊格式可能因每个主要和次要设备架构而异。由于线程只持有整个矩阵的片段（不透明架构特定ABI数据结构），这更加复杂，开发人员不允许假设单个参数如何映射到参与矩阵乘积累的寄存器。

由于片段是特定于架构的，如果函数已为不同的链接兼容架构编译并链接到同一设备可执行文件，则将它们从函数A传递到函数B是不安全的。在这种情况下，片段的大小和布局将特定于一个架构，在另一个架构中使用WMMA API将导致不正确的结果或潜在的损坏。

两个链接兼容架构的一个例子是sm_70和sm_75，其中片段的布局不同。
```c++
fragA.cu: void foo() { wmma::fragment<...> mat_a; bar(&mat_a); }
fragB.cu: void bar(wmma::fragment<...> *mat_a) { // operate on mat_a }

// sm_70 fragment layout
$> nvcc -dc -arch=compute_70 -code=sm_70 fragA.cu -o fragA.o
// sm_75 fragment layout
$> nvcc -dc -arch=compute_75 -code=sm_75 fragB.cu -o fragB.o
// Linking the two together
$> nvcc -dlink -arch=sm_75 fragA.o fragB.o -o frag.o
```
这种未定义的行为在编译时和运行时可能无法被工具检测到，因此需要格外小心，以确保片段的布局一致。当与传统库链接时，最有可能出现这种链接危险，该库既是为不同的链接兼容架构而构建的，又期望传递WMMA片段。

请注意，在弱链接的情况下（例如，CUDA C++内联函数），链接器可以选择任何可用的函数定义，这可能会导致编译单元之间的隐式传递。

To avoid these sorts of problems, the matrix should always be stored out to memory for transit through external interfaces (e.g. `wmma::store_matrix_sync(dst, …);`) and then it can be safely passed to `bar()` as a pointer type [e.g. `float *dst`].

请注意，由于sm_70可以在sm_75上运行，因此上述示例sm_75代码可以更改为sm_70，并在sm_75上正常工作。但是，与其他单独编译的sm_75二进制文件链接时，建议在应用程序中包含sm_75本机代码。

### 10.24.6.元素类型和矩阵大小[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#element-types-and-matrix-sizes "这个标题的永久链接")

张量核心支持各种元素类型和矩阵大小。下表显示了支持的`matrix_a`、`matrix_b`和`accumulator`矩阵的各种组合：

| 矩阵A   | 矩阵B   | 累加器  | 矩阵大小（m-n-k） |
| ----- | ----- | ---- | ----------- |
| __一半  | __一半  | 漂浮   | 16x16x16    |
| __一半  | __一半  | 漂浮   | 32x8x16     |
| __一半  | __一半  | 漂浮   | 8x32x16     |
| __一半  | __一半  | __一半 | 16x16x16    |
| __一半  | __一半  | __一半 | 32x8x16     |
| __一半  | __一半  | __一半 | 8x32x16     |
| 无符号字符 | 无符号字符 | int  | 16x16x16    |
| 无符号字符 | 无符号字符 | int  | 32x8x16     |
| 无符号字符 | 无符号字符 | int  | 8x32x16     |
| 签名字符  | 签名字符  | int  | 16x16x16    |
| 签名字符  | 签名字符  | int  | 32x8x16     |
| 签名字符  | 签名字符  | int  | 8x32x16     |

备用浮点支持：

|矩阵A|矩阵B|累加器|矩阵大小（m-n-k）|
|---|---|---|---|
|__nv_bfloat16|__nv_bfloat16|漂浮|16x16x16|
|__nv_bfloat16|__nv_bfloat16|漂浮|32x8x16|
|__nv_bfloat16|__nv_bfloat16|漂浮|8x32x16|
|精度::tf32|精度::tf32|漂浮|16x16x8|

双重精密支持：

|矩阵A|矩阵B|累加器|矩阵大小（m-n-k）|
|---|---|---|---|
|双倍|双倍|双倍|8x8x4|

对子字节操作的实验支持：

|矩阵A|矩阵B|累加器|矩阵大小（m-n-k）|
|---|---|---|---|
|精度::u4|精度::u4|int|8x8x32|
|精度::s4|精度::s4|int|8x8x32|
|精度::b1|精度::b1|int|8x8x128|

### 10.24.7.示例：[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-example "这个标题的永久链接")

以下代码在单个经编中实现了16x16x16矩阵乘法。
```c++
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
   wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}
```
## 10.25.DPX[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dpx "这个标题的永久链接")

DPX是一组函数，可以查找最小值和最大值，以及融合加法和最小值/最大值，最多三个16位和32位有符号或无符号整数参数，可选ReLU（夹到零）：

- 三个参数：`__vimax3_s32`，`__vimax3_s16x2`，`__vimax3_u32`，`__vimax3_u16x2`，`__vimin3_s32`，`__vimin3_s16x2`，`__vimin3_u32`，`__vimin3_u16x2`
    
- 两个参数，带有ReLU：`__vimax_s32_relu`，`__vimax_s16x2_relu`，`__vimin_s32_relu`，`__vimin_s16x2_relu`
    
- 三个参数，带有ReLU：`__vimax3_s32_relu`，`__vimax3_s16x2_relu`，`__vimin3_s32_relu`，`__vimin3_s16x2_relu`
    
- 两个参数，也返回哪个参数更小/更大：`__vibmax_s32`，`__vibmax_u32`，`__vibmin_s32`，`__vibmin_u32`，`__vibmax_s16x2`，`__vibmax_u16x2`，`__vibmin_s16x2`，`__vibmin_u16x2`
    
- 三个参数，与第三个参数进行比较（第一+第二）：`__viaddmax_s32`，`__viaddmax_s16x2`，`__viaddmax_u32`，`__viaddmax_u16x2`，`__viaddmin_s32`，`__viaddmin_s16x2`，`__viaddmin_u32`，`__viaddmin_u16x2`
    
- 三个参数，使用ReLU，与第三个和零进行比较（第一+第二）：`__viaddmax_s32_relu`，`__viaddmax_s16x2_relu`，`__viaddmin_s32_relu`，`__viaddmin_s16x2_relu`
    

这些指令在具有计算能力9及更高的设备上进行硬件加速，在旧设备上进行软件仿真。

完整的API可以在[CUDA数学API文档](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html)中找到。

DPX在实现动态编程算法时非常有用，例如基因组学中的Smith-Waterman或Neederman-Wunsch和路线优化中的Floyd-Warshall。

### 10.25.1.实例[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dpx-example "这个标题的永久链接")

三个有符号的32位整数的最大值，带有ReLU

const int a = -15;
const int b = 8;
const int c = 5;
int max_value_0 = __vimax3_s32_relu(a, b, c); // max(-15, 8, 5, 0) = 8
const int d = -2;
const int e = -4;
int max_value_1 = __vimax3_s32_relu(a, d, e); // max(-15, -2, -4, 0) = 0

两个32位有符号整数、另一个32位有符号整数和零（ReLU）之和的最小值

const int a = -5;
const int b = 6;
const int c = -2;
int max_value_0 = __viaddmax_s32_relu(a, b, c); // max(-5 + 6, -2, 0) = max(1, -2, 0) = 1
const int d = 4;
int max_value_1 = __viaddmax_s32_relu(a, d, c); // max(-5 + 4, -2, 0) = max(-1, -2, 0) = 0

两个无符号32位整数的最小值，并确定哪个值更小

const unsigned int a = 9;
const unsigned int b = 6;
bool smaller_value;
unsigned int min_value = __vibmin_u32(a, b, &smaller_value); // min_value is 6, smaller_value is true

三对无符号16位整数的最大值

const unsigned a = 0x00050002;
const unsigned b = 0x00070004;
const unsigned c = 0x00020006;
unsigned int max_value = __vimax3_u16x2(a, b, c); // max(5, 7, 2) and max(2, 4, 6), so max_value is 0x00070006

## 10.26.异步屏障[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-barrier "这个标题的永久链接")

NVIDIA C++标准库引入了[std::barrier](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/barrier.html)的GPU实现。随着`std::barrier`的实现，该库提供了扩展，允许用户指定障碍对象的范围。屏障API范围记录在[线程范围](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes)下。计算能力8.0或更高版本的设备为障碍操作提供硬件加速，并将这些障碍与thememcpy_async功能集成。在计算能力低于8.0但从7.0开始的设备上，这些障碍可以在没有硬件加速的情况下使用。

`nvcuda::experimental::awbarrier`被弃用，有利于`cuda::barrier`。

### 10.26.1.简单的同步模式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simple-synchronization-pattern "这个标题的永久链接")

如果没有到达/等待障碍，使用[合作组](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups)时可以使用`__syncthreads()`（同步块中的所有线程）或`group.sync()`实现同步。

#include <cooperative_groups.h>

__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();

    for (int i = 0; i < iteration_count; ++i) {
        /* code before arrive */
        block.sync(); /* wait for all threads to arrive here */
        /* code after wait */
    }
}

线程在同步点（`block.sync()`）被阻止，直到所有线程都达到同步点。此外，同步点之前发生的内存更新保证在同步点之后对块中的所有线程可见，即等同于`atomic_thread_fence(memory_order_seq_cst,thread_scope_block)`以及`sync`。

这个模式有三个阶段：

- 同步**前的**代码执行内存更新，该更新将在同步**后**读取。
    
- 同步点
    
- 同步点**后的**代码，可见同步点**之前**发生的内存更新。
    

### 10.26.2.时间分割和同步的五个阶段[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#temporal-splitting-and-five-stages-of-synchronization "这个标题的永久链接")

与`std::barrier`的时间分割同步模式如下。

#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ void compute(float* data, int curr_iteration);

__global__ void split_arrive_wait(int iteration_count, float *data) {
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__  barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // Initialize the barrier with expected arrival count
    }
    block.sync();

    for (int curr_iter = 0; curr_iter < iteration_count; ++curr_iter) {
        /* code before arrive */
       barrier::arrival_token token = bar.arrive(); /* this thread arrives. Arrival does not block a thread */
       compute(data, curr_iter);
       bar.wait(std::move(token)); /* wait for all threads participating in the barrier to complete bar.arrive()*/
        /* code after wait */
    }
}

在此模式中，同步点（`block.sync()`被分为到达点（`bar.arrive()`和等待点（`bar.wait(std::move(token))`）。线程在第一次调用`bar.arrive()`开始参与`cuda::barrier`当线程调用`bar.wait(std::move(token))`时，它将被阻止，直到参与线程完成`bar.arrive()`传递给`init()`的预期到达计数参数指定的预期次数。在参与线程调用`bar.arrive()`之前发生的内存更新保证在调用`bar.wait(std::move(token))`后对参与线程可见。请注意，对`bar.arrive()`的调用不会阻止线程，它可以继续进行其他工作，这些工作不依赖于在其他参与线程调用`bar.arrive()`之前发生的内存更新。

_到达然后等待_模式有五个阶段，可以迭代重复：

- 到达**前的**代码执行内存更新，这些更新将在等待**后**读取。
    
- 带有隐式内存围栏的到达点（即，相当于`atomic_thread_fence(memory_order_seq_cst,thread_scope_block)`）。
    
- 到达和等待**之间的**代码。
    
- 等待点。
    
- 等待**后**的代码，以及到达**前**执行的更新的可见性。
    

### 10.26.3.引导初始化、预计到达计数和参与[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#bootstrap-initialization-expected-arrival-count-and-participation "这个标题的永久链接")

初始化必须在任何线程开始参与`cuda::barrier`之前进行。

#include <cuda/barrier>
#include <cooperative_groups.h>

__global__ void init_barrier() {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // Single thread initializes the total expected arrival count.
    }
    block.sync();
}

在任何线程参与`cuda::barrier`之前，必须使用具有**预期到达计数**的`init()`初始化屏障，在本例中是`block.size()`。初始化必须在任何线程调用`bar.arrive()`之前进行。这带来了一个引导挑战，因为线程在参与`cuda::barrier`之前必须同步，但线程正在创建`cuda::barrier`以进行同步。在本例中，将参与的线程是合作组的一部分，并使用`block.sync()`进行引导初始化。在本例中，整个线程块正在参与初始化，因此也可以使用`__syncthreads()`）。

`init()`的第二个参数是**预期到达计数**，即在参与线程从调用`bar.wait(std::move(token))`中解锁之前，参与线程将调用`bar.arrive()`的次数。在之前的示例中，`cuda::barrier`使用线程块中的线程数初始化，即`cooperative_groups::this_thread_block().size()`，线程块内的所有线程都参与了障碍。

A `cuda::barrier` is flexible in specifying how threads participate (split arrive/wait) and which threads participate. In contrast `this_thread_block.sync()` from cooperative groups or `__syncthreads()` is applicable to whole-thread-block and `__syncwarp(mask)` is a specified subset of a warp. If the intention of the user is to synchronize a full thread block or a full warp we recommend using `__syncthreads()` and `__syncwarp(mask)` respectively for performance reasons.

### 10.26.4.障碍阶段：到达、倒计时、完成和重置[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#a-barrier-s-phase-arrival-countdown-completion-and-reset "这个标题的永久链接")

当参与线程调用`bar.arrive()`，`cuda::barrier`从预期到达计数倒计时到零。当倒计时达到零时，当前阶段的`cuda::barrier`就完成了。当最后一次调用`bar.arrive()`导致倒计时达到零时，倒计时会自动和原子重置。重置将倒计时分配给预期到达计数，并将`cuda::barrier`移动到下一阶段。

从`token=bar.arrive()`返回的类`cuda::barrier::arrival_token`的`token`对象与障碍的当前阶段相关联。当`cuda::barrier`处于当前阶段时，对`bar.wait(std::move(token))`的调用会阻止调用线程，即与令牌关联的阶段与`cuda::barrier`的阶段相匹配。如果在调用tobar.wait(std`bar.wait(std::move(token))`之前阶段是高级的（因为倒计时达到零），那么线程不会被阻止；如果线程在`bar.wait(std::move(token))`中被阻止时阶段是高级的，线程将被阻止。

**知道何时可以或不能发生重置至关重要，特别是在非平凡的到达/等待同步模式中。**

- 线程对`token=bar.arrive()`和`bar.wait(std::move(token))`的调用必须排序，以便`token=bar.arrive()`在thecuda`cuda::barrier`的当前阶段发生，`bar.wait(std::move(token))`在同一阶段或下一个阶段发生。
    
- 当障碍计数器非零时，必须发生线程对`bar.arrive()`的调用。在屏障初始化后，如果线程对`bar.arrive()`调用导致倒计时达到零，那么必须调用`bar.wait(std::move(token))`，然后才能重复使用该屏障进行对`bar.arrive()`的后续调用。
    
- `bar.wait()`只能使用当前阶段或紧接着阶段的`token`对象进行调用。对于`token`对象的任何其他值，行为是未定义的。
    

对于简单的到达/等待同步模式，遵守这些使用规则是直接的。

### 10.26.5.空间分区（也称为Warp专业化）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#spatial-partitioning-also-known-as-warp-specialization "这个标题的永久链接")

线程块可以进行空间分区，这样扭曲就专门用于执行独立计算。空间分区用于生产者或消费者模式，其中一个线程子集生成的数据由线程的另一个（不相干）子集同时消耗。

生产者/消费者空间分区模式需要两个单边同步来管理生产者和消费者之间的数据缓冲区。

|生产者|消费者|
|---|---|
|等待缓冲区准备好填充|信号缓冲区已准备好填充|
|生成数据并填充缓冲区||
|信号缓冲区已填满|等待缓冲区被填满|
||消耗填充缓冲区中的数据|

生产者线程等待消费者线程发出缓冲区已准备好填充的信号；但是，消费者线程不会等待此信号。消费者线程等待生产者线程发出缓冲区已满的信号；然而，生产者线程不会等待此信号。对于完全的生产者/消费者并发性，该模式具有（至少）双缓冲，每个缓冲区需要两个`cuda::barrier`。
```c++
#include <cuda/barrier>
#include <cooperative_groups.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ void producer(barrier ready[], barrier filled[], float* buffer, float* in, int N, int buffer_len)
{
    for (int i = 0; i < (N/buffer_len); ++i) {
        ready[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
        /* produce, i.e., fill in, buffer_(i%2)  */
        barrier::arrival_token token = filled[i%2].arrive(); /* buffer_(i%2) is filled */
    }
}

__device__ void consumer(barrier ready[], barrier filled[], float* buffer, float* out, int N, int buffer_len)
{
    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
    for (int i = 0; i < (N/buffer_len); ++i) {
        filled[i%2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
        /* consume buffer_(i%2) */
        barrier::arrival_token token = ready[i%2].arrive(); /* buffer_(i%2) is ready to be re-filled */
    }
}

//N is the total number of float elements in arrays in and out
__global__ void producer_consumer_pattern(int N, int buffer_len, float* in, float* out) {

    // Shared memory buffer declared below is of size 2 * buffer_len
    // so that we can alternatively work between two buffers.
    // buffer_0 = buffer and buffer_1 = buffer + buffer_len
    __shared__ extern float buffer[];

    // bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
    // while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively
    __shared__ barrier bar[4];

    auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() < 4)
        init(bar + block.thread_rank(), block.size());
    block.sync();

    if (block.thread_rank() < warpSize)
        producer(bar, bar+2, buffer, in, N, buffer_len);
    else
        consumer(bar, bar+2, buffer, out, N, buffer_len);
}
```
在本例中，第一个经编是专门作为生产者，其余经编是专门作为消费者。所有生产者和消费者线程都参与（调用`bar.arrive()`或`bar.arrive_and_wait()`四个`cuda::barrier`，因此预期到达计数等于`block.size()`

生产者线程等待消费者线程发出共享内存缓冲区可以填充的信号。为了等待`cuda::barrier`，生产者线程必须首先到达`ready[i%2].arrive()`以获取令牌，然后使用该令牌`ready[i%2].wait(token)`。对于简单性，`ready[i%2].arrive_and_wait()`结合了这些操作。

bar.arrive_and_wait();
/* is equivalent to */
bar.wait(bar.arrive());

生产者线程计算并填充准备缓冲区，然后它们通过到达填充屏障，`filled[i%2].arrive()`发出信号，缓冲区已填充。生产者线程不会在这一点上等待，而是等到下一个迭代的缓冲区（双缓冲）准备好填充。

消费者线程首先发出信号，表明两个缓冲区都已准备好填充。此时，消费者线程不会等待，而是等待此迭代的缓冲区被填充，`filled[i%2].arrive_and_wait()`在消费者线程消耗缓冲区后，它们发出缓冲区准备再次填充的信号，`ready[i%2].arrive()`，然后等待下一个迭代的缓冲区被填充。

### 10.26.6.提前退出（退出参与）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#early-exit-dropping-out-of-participation "这个标题的永久链接")

当参与同步序列的线程必须提前退出该序列时，该线程在退出之前必须明确退出参与。其余参与线程可以正常进行后续的`cuda::barrier`到达和等待操作。
```c++
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ bool condition_check();

__global__ void early_exit_kernel(int N) {
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__ barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0)
        init(&bar , block.size());
    block.sync();

    for (int i = 0; i < N; ++i) {
        if (condition_check()) {
          bar.arrive_and_drop();
          return;
        }
        /* other threads can proceed normally */
        barrier::arrival_token token = bar.arrive();
        /* code between arrive and wait */
        bar.wait(std::move(token)); /* wait for all threads to arrive */
        /* code after wait */
    }
}
```
此操作到达`cuda::barrier`，以履行参与线程在**当前**阶段到达的义务，然后减少**下一**阶段的预期到达计数，以便该线程不再预计到达障碍。

### 10.26.7.完成功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#completion-function "这个标题的永久链接")

The `CompletionFunction` of `cuda::barrier<Scope, CompletionFunction>` is executed once per phase, after the last thread _arrives_ and before any thread is unblocked from the `wait`. Memory operations performed by the threads that arrived at the `barrier` during the phase are visible to the thread executing the `CompletionFunction`, and all memory operations performed within the `CompletionFunction` are visible to all threads waiting at the `barrier` once they are unblocked from the `wait`.
```c++
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <functional>
namespace cg = cooperative_groups;

__device__ int divergent_compute(int*, int);
__device__ int independent_computation(int*, int);

__global__ void psum(int* data, int n, int* acc) {
  auto block = cg::this_thread_block();

  constexpr int BlockSize = 128;
  __shared__ int smem[BlockSize];
  assert(BlockSize == block.size());
  assert(n % 128 == 0);

  auto completion_fn = [&] {
    int sum = 0;
    for (int i = 0; i < 128; ++i) sum += smem[i];
    *acc += sum;
  };

  // Barrier storage
  // Note: the barrier is not default-constructible because
  //       completion_fn is not default-constructible due
  //       to the capture.
  using completion_fn_t = decltype(completion_fn);
  using barrier_t = cuda::barrier<cuda::thread_scope_block,
                                  completion_fn_t>;
  __shared__ std::aligned_storage<sizeof(barrier_t),
                                  alignof(barrier_t)> bar_storage;

  // Initialize barrier:
  barrier_t* bar = (barrier_t*)&bar_storage;
  if (block.thread_rank() == 0) {
    assert(*acc == 0);
    assert(blockDim.x == blockDim.y == blockDim.y == 1);
    new (bar) barrier_t{block.size(), completion_fn};
    // equivalent to: init(bar, block.size(), completion_fn);
  }
  block.sync();

  // Main loop
  for (int i = 0; i < n; i += block.size()) {
    smem[block.thread_rank()] = data[i] + *acc;
    auto t = bar->arrive();
    // We can do independent computation here
    bar->wait(std::move(t));
    // shared-memory is safe to re-use in the next iteration
    // since all threads are done with it, including the one
    // that did the reduction
  }
}
```
### 10.26.8.内存屏障原始接口[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-barrier-primitives-interface "这个标题的永久链接")

内存屏障原语是与`cuda::barrier`功能的类似C的接口。这些原语可以通过包含`<cuda_awbarrier_primitives.h>`标题获得。

#### 10.26.8.1.数据类型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-types "这个标题的永久链接")

typedef /* implementation defined */ __mbarrier_t;
typedef /* implementation defined */ __mbarrier_token_t;

#### 10.26.8.2.内存屏障原语API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-barrier-primitives-api "这个标题的永久链接")

uint32_t __mbarrier_maximum_count();
void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);

- `bar`必须是`__shared__`内存的指针。
    
- `expected_count <= __mbarrier_maximum_count()`
    
- 将当前和下一阶段的预期到达计数初始化`*bar`到`expected_count`。
    

void __mbarrier_inval(__mbarrier_t* bar);

- `bar`必须是指向共享内存中的屏障对象的指针。
    
- 在重新利用相应的共享内存之前，需要对`*bar`进行无效。
    

__mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);

- `*bar`的初始化必须在此呼叫之前完成。
    
- 待定计数不得为零。
    
- 原子地减少屏障当前阶段的待定计数。
    
- 在递减之前返回与屏障状态相关的到达令牌。
    

__mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);

- `*bar`的初始化必须在此呼叫之前完成。
    
- 待定计数不得为零。
    
- 原子地减少当前阶段的待定计数和屏障下一阶段的预期计数。
    
- 在递减之前返回与屏障状态相关的到达令牌。
    

bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);

- `token`必须与`*this`的紧随其后阶段或当前阶段相关联。
    
- 如果`token`与`*bar`的紧接着阶段相关联，则返回`true`，否则返回`false`。
    

//Note: This API has been deprecated in CUDA 11.1
uint32_t __mbarrier_pending_count(__mbarrier_token_t token);

## 10.27.异步数据副本[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies "这个标题的永久链接")

CUDA 11 introduces Asynchronous Data operations with `memcpy_async` API to allow device code to explicitly manage the asynchronous copying of data. The `memcpy_async` feature enables CUDA kernels to overlap computation with data movement.

### 10.27.1. `memcpy_async` API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memcpy-async-api "这个标题的永久链接")

The `memcpy_async` APIs are provided in the `cuda/barrier`, `cuda/pipeline`, and `cooperative_groups/memcpy_async.h` header files.

The `cuda::memcpy_async` APIs work with `cuda::barrier` and `cuda::pipeline` synchronization primitives, while the `cooperative_groups::memcpy_async`synchronizes using `cooperative_groups::wait`.

These APIs have very similar semantics: copy objects from `src` to `dst` as-if performed by another thread which, on completion of the copy, can be synchronized through `cuda::pipeline`, `cuda::barrier`, or `cooperative_groups::wait`.

[libcudacxx API](https://nvidia.github.io/libcudacxx)文档中提供了`cuda::barrier`和`cuda::pipeline`的`cuda::memcpy_async`过载的完整API文档以及一些示例。

[Cooperative_groups::memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-memcpy-async)的API文档在[合作小组](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups)部分提供。

The `memcpy_async` APIs that use [cuda::barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier) and `cuda::pipeline` require compute capability 7.0 or higher. On devices with compute capability 8.0 or higher, `memcpy_async` operations from global to shared memory can benefit from hardware acceleration.

### 10.27.2.复制和计算模式-通过共享内存暂存数据[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#copy-and-compute-pattern-staging-data-through-shared-memory "这个标题的永久链接")

CUDA应用程序通常采用_复制和计算_模式：

- 从全局内存中获取数据，
    
- 将数据存储到共享内存中，以及
    
- 对共享内存数据执行计算，并有可能将结果写回全局内存。
    

以下章节说明了如何在没有`memcpy_async`功能的情况下表达此模式：

- [没有memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#without-memcpy-async)引入了一个不与数据移动重叠计算并使用中间寄存器复制数据的示例。
    
- [With memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#with-memcpy-async) improves the previous example by introducing the [memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-memcpy-async) and the `cuda::memcpy_async` APIs to directly copy data from global to shared memory without using intermediate registers.
    
- [使用cuda::barrier的异步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memcpy-async-barrier)显示带有合作组和屏障的memcpy。
    
- [使用cuda::pipeline的单级异步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#with-memcpy-async-pipeline-pattern-single)显示带有单级管道的memcpy。
    
- [使用cuda::pipeline的多阶段异步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#with-memcpy-async-pipeline-pattern-multi)显示带有多阶段管道的memcpy。
    

### 10.27.3.没有`memcpy_async`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#without-memcpy-async "这个标题的永久链接")

Without `memcpy_async`, the _copy_ phase of the _copy and compute_ pattern is expressed as `shared[local_idx] = global[global_idx]`. This global to shared memory copy is expanded to a read from global memory into a register, followed by a write to shared memory from the register.

When this pattern occurs within an iterative algorithm, each thread block needs to synchronize after the `shared[local_idx] = global[global_idx]`assignment, to ensure all writes to shared memory have completed before the compute phase can begin. The thread block also needs to synchronize again after the compute phase, to prevent overwriting shared memory before all threads have completed their computations. This pattern is illustrated in the following code snippet.

#include <cooperative_groups.h>
__device__ void compute(int* global_out, int const* shared_in) {
    // Computes using all values of current batch from shared memory.
    // Stores this thread's result back to global memory.
}

__global__ void without_memcpy_async(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  size_t local_idx = block.thread_rank();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    // Compute the index of the current batch for this block in global memory:
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    size_t global_idx = block_batch_idx + threadIdx.x;
    shared[local_idx] = global_in[global_idx];

    block.sync(); // Wait for all copies to complete

    compute(global_out + block_batch_idx, shared); // Compute and write result to global memory

    block.sync(); // Wait for compute using shared memory to finish
  }
}

### 10.27.4.与`memcpy_async`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#with-memcpy-async "这个标题的永久链接")

使用`memcpy_async`，从全局内存中分配共享内存

shared[local_idx] = global_in[global_idx];

被来自[合作组](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups)的异步复制操作所取代

cooperative_groups::memcpy_async(group, shared, global_in + batch_idx, sizeof(int) * block.size());

The [cooperative_groups::memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-memcpy-async) API copies `sizeof(int) * block.size()` bytes from global memory starting at `global_in + batch_idx` to the `shared` data. This operation happens as-if performed by another thread, which synchronizes with the current thread’s call to [cooperative_groups::wait](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-wait) after the copy has completed. Until the copy operation completes, modifying the global data or reading or writing the shared data introduces a data race.

在计算能力为8.0或更高版本的设备上，`memcpy_async`从全局内存传输到共享内存可以从硬件加速中受益，这避免了通过中间寄存器传输数据。

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

__device__ void compute(int* global_out, int const* shared_in);

__global__ void with_memcpy_async(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    // Whole thread-group cooperatively copies whole batch to shared memory:
    cooperative_groups::memcpy_async(block, shared, global_in + block_batch_idx, sizeof(int) * block.size());

    cooperative_groups::wait(block); // Joins all threads, waits for all copies to complete

    compute(global_out + block_batch_idx, shared);

    block.sync();
  }
}}

### 10.27.5.异步数据副本使用`cuda::barrier`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-cuda-barrier "这个标题的永久链接")

[cuda::barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier)的`cuda::memcpy_async`过载可以使用`barrier`同步异步数据传输。这种超载执行复制操作，就像由绑定到屏障的另一个线程执行一样：在创建时增加当前阶段的预期计数，并在复制操作完成后递减它，这样`barrier`的阶段只有在所有参与屏障的线程到达，并且绑定到屏障当前阶段的所有`memcpy_async`都已完成时才会前进。以下示例使用块宽`barrier`，其中所有块线程都参与其中，并将等待操作与屏障`arrive_and_wait`交换，同时提供与前一个示例相同的功能：

#include <cooperative_groups.h>
#include <cuda/barrier>
__device__ void compute(int* global_out, int const* shared_in);

__global__ void with_barrier(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();
  assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

  extern __shared__ int shared[]; // block.size() * sizeof(int) bytes

  // Create a synchronization object (C++20 barrier)
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
  if (block.thread_rank() == 0) {
    init(&barrier, block.size()); // Friend function initializes barrier
  }
  block.sync();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size() + grid.size() * batch;
    cuda::memcpy_async(block, shared, global_in + block_batch_idx, sizeof(int) * block.size(), barrier);

    barrier.arrive_and_wait(); // Waits for all copies to complete

    compute(global_out + block_batch_idx, shared);

    block.sync();
  }
}

### 10.27.6.绩效指导`memcpy_async`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidance-for-memcpy-async "这个标题的永久链接")

对于计算能力8.x，管道机制在同一CUDA扭曲中的CUDA线程之间共享。这种共享导致`memcpy_async`的批次被卷入经编中，在某些情况下可能会影响性能。

本节重点介绍了_提交_、_等待_和_到达_操作的扭曲纠缠效应。有关单个操作的概述，请参阅[管道接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pipeline-interface)和[管道原始接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pipeline-primitives-interface)。

#### 10.27.6.1.对齐[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#alignment "这个标题的永久链接")

在具有计算能力8.0的设备上，[cp.async系列指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)允许非同步地将数据从全局内存复制到共享内存。这些指令支持一次复制4、8和16字节。如果提供给`memcpy_async`的大小是4、8或16的倍数，并且传递给`memcpy_async`两个指针都与4、8或16对齐边界对齐，那么`memcpy_async`可以使用完全异步内存操作来实现。

Additionally for achieving best performance when using `memcpy_async` API, an alignment of 128 Bytes for both shared memory and global memory is required.

对于对齐要求为1或2的类型值的指针，通常无法证明指针始终与更高的对齐边界对齐。确定`cp.async`指令是否可以使用，必须推迟到运行时。执行这样的运行时对齐检查会增加代码大小并增加运行时开销。

The [cuda::aligned_size_t<size_t Align>(size_t size)](https://nvidia.github.io/libcudacxx)[Shape](https://nvidia.github.io/libcudacxx) can be used to supply a proof that both pointers passed to `memcpy_async` are aligned to an `Align` alignment boundary and that `size` is a multiple of `Align`, by passing it as an argument where the `memcpy_async` APIs expect a `Shape`:

cuda::memcpy_async(group, dst, src, cuda::aligned_size_t<16>(N * block.size()), pipeline);

如果证明不正确，则行为未定义。

#### 10.27.6.2.微不足道的可复制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#trivially-copyable "这个标题的永久链接")

在具有计算能力8.0的设备上，[cp.async系列指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)允许非同步地将数据从全局内存复制到共享内存。如果传递给`memcpy_async`的指针类型不指向[TriviallyCopyable](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable)类型，则需要调用每个输出元素的复制构造函数，这些指令不能用于加速`memcpy_async`。

#### 10.27.6.3.扭曲纠缠-提交[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-entanglement-commit "这个标题的永久链接")

`memcpy_async`批次的序列在经编中共享。提交操作是合并的，因此对于调用提交操作的所有收敛线程，序列会增加一次。如果经编完全收敛，序列递增一；如果经编完全发散，序列递增32。

- 让_PB_是扭曲共享管道_的实际_批次序列。
    
    `PB = {BP0, BP1, BP2, …, BPL}`
    
- 让_TB_是线程_感知_的批次序列，就好像序列只是通过该线程调用提交操作而增加的一样。
    
    `TB = {BT0, BT1, BT2, …, BTL}`
    
    `pipeline::producer_commit()`返回值来自线程_感知_的批处理序列。
    
- 线程感知序列中的索引总是与实际经编共享序列中相同或更大的索引对齐。只有当所有提交操作都从收敛线程调用时，序列才相等。
    
    `BTn ≡ BPm`地点`n <= m`
    

例如，当经编完全发散时：

- The warp-shared pipeline’s actual sequence would be: `PB = {0, 1, 2, 3, ..., 31}` (`PL=31`).
    
- 这个经编的每个线程的感知序列将是：
    
    - Thread 0: `TB = {0}` (`TL=0`)
        
    - Thread 1: `TB = {0}` (`TL=0`)
        
    - `…`
        
    - Thread 31: `TB = {0}` (`TL=0`)
        

#### 10.27.6.4.扭曲纠缠 - 等待[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-entanglement-wait "这个标题的永久链接")

A CUDA thread invokes either `pipeline_consumer_wait_prior<N>()` or `pipeline::consumer_wait()` to wait for batches in the _perceived_ sequence `TB` to complete. Note that `pipeline::consumer_wait()` is equivalent to `pipeline_consumer_wait_prior<N>()`, where `N =                                        PL`.

The `pipeline_consumer_wait_prior<N>()` function waits for batches in the _actual_ sequence at least up to and including `PL-N`. Since `TL <= PL`, waiting for batch up to and including `PL-N` includes waiting for batch `TL-N`. Thus, when `TL < PL`, the thread will unintentionally wait for additional, more recent batches.

在上述极端完全发散的经编示例中，每个线程都可以等待所有32批次。

#### 10.27.6.5.扭曲纠缠-到达[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-entanglement-arrive-on "这个标题的永久链接")

扭曲发散会影响arrar_on`arrive_on(bar)`操作更新屏障的次数。如果调用的扭曲完全收敛，那么屏障就会更新一次。如果调用经编完全发散，那么32个单独的更新将应用于屏障。

#### 10.27.6.6.保持承诺和到达操作趋同[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#keep-commit-and-arrive-on-operations-converged "这个标题的永久链接")

建议通过收敛线程进行提交和到达调用：

- 通过保持线程感知的批次序列与实际序列保持一致，从而不超时等待，以及
    
- 尽量减少对屏障对象的更新。
    

当这些操作之前的代码发散线程时，在调用提交或到达操作之前，应通过`__syncwarp`重新收敛经编。

## 10.28.异步数据副本使用`cuda::pipeline`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-cuda-pipeline "这个标题的永久链接")

CUDA提供`cuda::pipeline`同步对象，用于管理异步数据移动与计算重叠。

`cuda::pipeline`的API文档在[libcudacxx API](https://nvidia.github.io/libcudacxx)中提供。管道对象是一个具有_头部_和_尾部_的双端N级队列，用于以先入先出（FIFO）顺序处理工作。管道对象具有以下成员函数来管理管道的阶段。

|管道类成员功能|描述|
|---|---|
|`producer_acquire`|获取管道内部队列中的可用阶段。|
|`producer_commit`|承诺在`producer_acquire`调用后对管道的当前收购阶段发出的异步操作。|
|`consumer_wait`|等待管道最古老的阶段的所有异步操作完成。|
|`consumer_release`|将管道的最早阶段释放到管道对象中以供重复使用。然后，制作人可以获取发布的阶段。|

### 10.28.1.单阶段异步数据副本使用`cuda::pipeline`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#single-stage-asynchronous-data-copies-using-cuda-pipeline "这个标题的永久链接")

In previous examples we showed how to use [cooperative_groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-wait) and [cuda::barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier) to do asynchronous data transfers. In this section, we will use the `cuda::pipeline` API with a single stage to schedule asynchronous copies. And later we will expand this example to show multi staged overlapped compute and copy.

#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

__device__ void compute(int* global_out, int const* shared_in);
__global__ void with_single_stage(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 1; // Pipeline with one stage
    // One batch must fit in shared memory:
    extern __shared__ int shared[];  // block.size() * sizeof(int) bytes

    // Allocate shared storage for a single stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Each thread processes `batch_sz` elements.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
      return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (size_t batch = 0; batch < batch_sz; ++batch) {
        size_t global_idx = block_batch(batch);

        // Collectively acquire the pipeline head stage from all producer threads:
        pipeline.producer_acquire();

        // Submit async copies to the pipeline's head stage to be
        // computed in the next loop iteration
        cuda::memcpy_async(block, shared, global_in + global_idx, sizeof(int) * block.size(), pipeline);
        // Collectively commit (advance) the pipeline's head stage
        pipeline.producer_commit();

        // Collectively wait for the operations committed to the
        // previous `compute` stage to complete:
        pipeline.consumer_wait();

        // Computation overlapped with the memcpy_async of the "copy" stage:
        compute(global_out + global_idx, shared);

        // Collectively release the stage resources
        pipeline.consumer_release();
    }
}

### 10.28.2.多阶段异步数据副本使用`cuda::pipeline`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-stage-asynchronous-data-copies-using-cuda-pipeline "这个标题的永久链接")

在前面的[cooperative_groups::wait](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-wait)和[cuda::barrier的](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier)示例中，内核线程立即等待数据传输到共享内存完成。这避免了数据从全局内存传输到寄存器，但不会通过重叠计算来_隐藏_`memcpy_async`操作的延迟。

为此，我们在以下示例中使用CUDA[管道](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pipeline-interface)功能。它提供了一种管理`memcpy_async`批次序列的机制，使CUDA内核能够将内存传输与计算重叠。以下示例实现了一个两阶段的管道，该管道将数据传输与计算重叠。它：

- 初始化管道共享状态（更多内容见下文）
    
- 通过为第一批安排`memcpy_async`来启动管道。
    
- 循环所有批次：它为下一个批次安排`memcpy_async`，阻止上一个批次的`memcpy_async`完成的所有线程，然后将上一个批次的计算与下一个批次的内存异步副本重叠。
    
- 最后，它通过对最后一批进行计算来耗尽管道。
    

请注意，为了与`cuda::pipeline`的互操作性，这里使用了`cuda/pipeline`的`cuda::memcpy_async`。

#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

__device__ void compute(int* global_out, int const* shared_in);
__global__ void with_staging(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 2; // Pipeline with two stages
    // Two batches must fit in shared memory:
    extern __shared__ int shared[];  // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count] = { 0, block.size() }; // Offsets to each batch

    // Allocate shared storage for a two-stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // Each thread processes `batch_sz` elements.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
      return block.group_index().x * block.size() + grid.size() * batch;
    };

    // Initialize first pipeline stage by submitting a `memcpy_async` to fetch a whole batch for the block:
    if (batch_sz == 0) return;
    pipeline.producer_acquire();
    cuda::memcpy_async(block, shared + shared_offset[0], global_in + block_batch(0), sizeof(int) * block.size(), pipeline);
    pipeline.producer_commit();

    // Pipelined copy/compute:
    for (size_t batch = 1; batch < batch_sz; ++batch) {
        // Stage indices for the compute and copy stages:
        size_t compute_stage_idx = (batch - 1) % 2;
        size_t copy_stage_idx = batch % 2;

        size_t global_idx = block_batch(batch);

        // Collectively acquire the pipeline head stage from all producer threads:
        pipeline.producer_acquire();

        // Submit async copies to the pipeline's head stage to be
        // computed in the next loop iteration
        cuda::memcpy_async(block, shared + shared_offset[copy_stage_idx], global_in + global_idx, sizeof(int) * block.size(), pipeline);
        // Collectively commit (advance) the pipeline's head stage
        pipeline.producer_commit();

        // Collectively wait for the operations committed to the
        // previous `compute` stage to complete:
        pipeline.consumer_wait();

        // Computation overlapped with the memcpy_async of the "copy" stage:
        compute(global_out + global_idx, shared + shared_offset[compute_stage_idx]);

        // Collectively release the stage resources
        pipeline.consumer_release();
    }

    // Compute the data fetch by the last iteration
    pipeline.consumer_wait();
    compute(global_out + block_batch(batch_sz-1), shared + shared_offset[(batch_sz - 1) % 2]);
    pipeline.consumer_release();
}

[管道对象](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pipeline-interface)是一个具有_头部_和_尾部_的双端队列，用于按先入先出（FIFO）顺序处理工作。生产者线程将工作提交到管道的头部，而消费者线程则从管道的尾部提取工作。在上述示例中，所有线程都是生产者和消费者线程。线程首先_提交_`memcpy_async`操作来获取_下一个_批次，同时_等待_上一批`memcpy_async`操作完成。

- 将工作提交到管道阶段涉及：
    
    - 使用`pipeline.producer_acquire()`从一组生产者线程中集体_获取_管道头_。_
        
    - 向管道头提交`memcpy_async`操作。
        
    - 使用`pipeline.producer_commit()`集体_提交_（推进）管道头。
        
- 使用之前承诺的阶段包括：
    
    - 集体等待阶段完成，例如，使用`pipeline.consumer_wait()`在尾部（最旧）阶段等待。
        
    - 使用`pipeline.consumer_release()`集体_发布_阶段。
        

`cuda::pipeline_shared_state<scope, count>`封装有限资源，允许管道处理到`count`并发阶段。如果所有资源都在使用中，`pipeline.producer_acquire()`阻止生产者线程，直到下一个管道阶段的资源被消费者线程释放。

通过将循环的前体和尾部与循环本身合并，可以以更简洁的方式编写此示例，具体如下所示：

template <size_t stages_count = 2 /* Pipeline with stages_count stages */>
__global__ void with_staging_unified(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * block.size();

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    // compute_batch: next batch to process
    // fetch_batch:  next batch to fetch from global memory
    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
        // The outer loop iterates over the computation of the batches
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
            // This inner loop iterates over the memory transfers, making sure that the pipeline is always full
            pipeline.producer_acquire();
            size_t shared_idx = fetch_batch % stages_count;
            size_t batch_idx = fetch_batch;
            size_t block_batch_idx = block_batch(batch_idx);
            cuda::memcpy_async(block, shared + shared_offset[shared_idx], global_in + block_batch_idx, sizeof(int) * block.size(), pipeline);
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        int shared_idx = compute_batch % stages_count;
        int batch_idx = compute_batch;
        compute(global_out + block_batch(batch_idx), shared + shared_offset[shared_idx]);
        pipeline.consumer_release();
    }
}

上面使用的`pipeline<thread_scope_block>`原语非常灵活，并支持我们上面的例子没有使用的两个功能：块中线程的任何任意子集都可以参与`pipeline`，从参与的线程中，任何子集都可以是生产者、消费者或两者兼而有之。在以下示例中，线程排名为“偶数”的线程是生产者，而其他线程是消费者：
```c++
__device__ void compute(int* global_out, int shared_in);

template <size_t stages_count = 2>
__global__ void with_specialized_staging_unified(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    // In this example, threads with "even" thread rank are producers, while threads with "odd" thread rank are consumers:
    const cuda::pipeline_role thread_role
      = block.thread_rank() % 2 == 0? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;

    // Each thread block only has half of its threads as producers:
    auto producer_threads = block.size() / 2;

    // Map adjacent even and odd threads to the same id:
    const int thread_idx = block.thread_rank() / 2;

    auto elements_per_batch = size / batch_sz;
    auto elements_per_batch_per_block = elements_per_batch / grid.group_dim().x;

    extern __shared__ int shared[]; // stages_count * elements_per_batch_per_block * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * elements_per_batch_per_block;

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    cuda::pipeline pipeline = cuda::make_pipeline(block, &shared_state, thread_role);

    // Each thread block processes `batch_sz` batches.
    // Compute offset of the batch `batch` of this thread block in global memory:
    auto block_batch = [&](size_t batch) -> int {
      return elements_per_batch * batch + elements_per_batch_per_block * blockIdx.x;
    };

    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
        // The outer loop iterates over the computation of the batches
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
            // This inner loop iterates over the memory transfers, making sure that the pipeline is always full
            if (thread_role == cuda::pipeline_role::producer) {
                // Only the producer threads schedule asynchronous memcpys:
                pipeline.producer_acquire();
                size_t shared_idx = fetch_batch % stages_count;
                size_t batch_idx = fetch_batch;
                size_t global_batch_idx = block_batch(batch_idx) + thread_idx;
                size_t shared_batch_idx = shared_offset[shared_idx] + thread_idx;
                cuda::memcpy_async(shared + shared_batch_idx, global_in + global_batch_idx, sizeof(int), pipeline);
                pipeline.producer_commit();
            }
        }
        if (thread_role == cuda::pipeline_role::consumer) {
            // Only the consumer threads compute:
            pipeline.consumer_wait();
            size_t shared_idx = compute_batch % stages_count;
            size_t global_batch_idx = block_batch(compute_batch) + thread_idx;
            size_t shared_batch_idx = shared_offset[shared_idx] + thread_idx;
            compute(global_out + global_batch_idx, *(shared + shared_batch_idx));
            pipeline.consumer_release();
        }
    }
}

`pipeline`执行了一些优化，例如，当所有线程既是生产者也是消费者时，但一般来说，支持所有这些功能的成本无法完全消除。例如，`pipeline`在共享内存中存储和使用一组障碍物进行同步，如果块中的所有线程都参与管道，这实际上没有必要。

对于块中所有线程都参与`pipeline`的特殊情况，我们可以使用apipeline<thread_scope_thread>与`__syncthreads()`相结合，比`pipeline<thread_scope_block>`做得更好：

template<size_t stages_count>
__global__ void with_staging_scope_thread(int* global_out, int const* global_in, size_t size, size_t batch_sz) {
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    auto thread = cooperative_groups::this_thread();
    assert(size == batch_sz * grid.size()); // Assume input size fits batch_sz * grid_size

    extern __shared__ int shared[]; // stages_count * block.size() * sizeof(int) bytes
    size_t shared_offset[stages_count];
    for (int s = 0; s < stages_count; ++s) shared_offset[s] = s * block.size();

    // No pipeline::shared_state needed
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    auto block_batch = [&](size_t batch) -> int {
        return block.group_index().x * block.size() + grid.size() * batch;
    };

    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < batch_sz; ++compute_batch) {
        for (; fetch_batch < batch_sz && fetch_batch < (compute_batch + stages_count); ++fetch_batch) {
            pipeline.producer_acquire();
            size_t shared_idx = fetch_batch % stages_count;
            size_t batch_idx = fetch_batch;
            // Each thread fetches its own data:
            size_t thread_batch_idx = block_batch(batch_idx) + threadIdx.x;
            // The copy is performed by a single `thread` and the size of the batch is now that of a single element:
            cuda::memcpy_async(thread, shared + shared_offset[shared_idx] + threadIdx.x, global_in + thread_batch_idx, sizeof(int), pipeline);
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        block.sync(); // __syncthreads: All memcpy_async of all threads in the block for this stage have completed here
        int shared_idx = compute_batch % stages_count;
        int batch_idx = compute_batch;
        compute(global_out + block_batch(batch_idx), shared + shared_offset[shared_idx]);
        pipeline.consumer_release();
    }
}
```
如果`compute`操作仅读取与当前线程相同的经编中其他线程写入的共享内存，则`__syncwarp()`就足够了。

### 10.28.3.管道接口[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pipeline-interface "这个标题的永久链接")

[libcudacxx API](https://nvidia.github.io/libcudacxx)文档中提供了`cuda::memcpy_async`的完整API文档以及一些示例。

`pipeline`接口需要

- 至少CUDA 11.0，
    
- 至少ISO C++ 2011兼容性，例如，使用`-std=c++11`编译，以及
    
- `#include <cuda/pipeline>`.
    

对于类似C的接口，在没有ISO C++ 2011兼容性的情况下编译时，请参阅[管道原始接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pipeline-primitives-interface)。

### 10.28.4.管道原始接口[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pipeline-primitives-interface "这个标题的永久链接")

管道原语是`memcpy_async`功能的类似C的接口。管道原语接口可以通过包含`<cuda_pipeline.h>`标头获得。在没有ISO C++ 2011兼容性的情况下编译时，包括`<cuda_pipeline_primitives.h>`标题。

#### 10.28.4.1. `memcpy_async` Primitive[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memcpy-async-primitive "这个标题的永久链接")
```c++
void __pipeline_memcpy_async(void* __restrict__ dst_shared,
                             const void* __restrict__ src_global,
                             size_t size_and_align,
                             size_t zfill=0);
```
- 请求提交以下操作以进行异步评估：
    ```c++
    size_t i = 0;
    for (; i < size_and_align - zfill; ++i) ((char*)dst_shared)[i] = ((char*)src_global)[i]; /* copy */
    for (; i < size_and_align; ++i) ((char*)dst_shared)[i] = 0; /* zero-fill */
    ```
- 要求：
    
    - `dst_shared`必须是指向`memcpy_async`共享内存目的地的指针。
        
    - `src_global`必须是指向`memcpy_async`的全局内存源的指针。
        
    - `size_and_align`必须是4、8或16。
        
    - `zfill <= size_and_align`.
        
    - `size_and_align`必须是`dst_shared`和`src_global`的对齐。
        
- 在等待`memcpy_async`操作完成之前，任何线程修改源内存或观察目标内存都是一个竞赛条件。在提交`memcpy_async`操作和等待其完成之间，以下任何操作都引入了竞赛条件：
    
    - 从`dst_shared`加载。
        
    - 存储到`dst_shared`或`src_global`。
        
    - 将原子更新应用于`dst_shared`或`src_global`。
        

#### 10.28.4.2.承诺原始[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#commit-primitive "这个标题的永久链接")

void __pipeline_commit();

- 提交将`memcpy_async`作为当前批处理提交到管道。
    

#### 10.28.4.3.等待原始的[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wait-primitive "这个标题的永久链接")

void __pipeline_wait_prior(size_t N);

- Let `{0, 1, 2, ..., L}` be the sequence of indices associated with invocations of `__pipeline_commit()` by a given thread.
    
- 等待批次完成，_至少_包括`L-N`。
    

#### 10.28.4.4.到达屏障原始[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arrive-on-barrier-primitive "这个标题的永久链接")

void __pipeline_arrive_on(__mbarrier_t* bar);

- `bar`指向共享内存中的障碍。
    
- 将障碍到达计数增加一个，当此调用之前排序的所有memcpy_async操作都完成时，到达计数将减小一个，因此对到达计数的净影响为零。用户有责任确保到达计数的增量不超过`__mbarrier_maximum_count()`
    

## 10.29.使用张量内存加速器（TMA）进行异步数据复制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies-using-the-tensor-memory-accelerator-tma "这个标题的永久链接")

许多应用程序需要将大量数据从全局内存移动到全局内存。通常，数据作为具有非顺序数据访问模式的多维数组排列在全局内存中。为了减少全局内存使用，在用于计算之前，将此类数组的子图将子图复制到共享内存中。加载和存储涉及地址计算，这些地址计算可能容易出错且重复。为了卸载这些计算，计算能力9.0引入了张量内存加速器（TMA）。TMA的主要目标是为多维阵列提供从全局内存到共享内存的高效数据传输机制。

**命名**。张量内存加速器（TMA）是一个广义的术语，用于指代本节中描述的功能。为了向前兼容并减少与PTX ISA的差异，本节中的文本将TMA操作称为批量异步副本或批量张量异步副本，具体取决于所使用的副本类型。“散块”一词用于将这些操作与前几节中描述的异步内存操作进行对比。

**尺寸**。TMA支持复制一维和多维数组（最多5维）。一维连续数组的**批量异步副本**的编程模型与多维数组的**批量张量异步副本**的编程模型不同。要执行多维数组的批量张量异步复制，硬件需要[张量图](https://docs.nvidia.com/cuda/cuda-driver-api/structCUtensorMap.html#structCUtensorMap)。此对象描述了全局和共享内存中多维数组的布局。张量映射通常使用thecuTensorMapEncode [API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY)在主机上创建，然后作为`const`核心参数从主机传输到设备，并用`__grid_constant__`注释。张量映射作为用`__grid_constant__`注释的`const`核心参数从主机传输到设备，可以在设备上用于在共享内存和全局内存之间复制数据图块。相比之下，执行连续一维数组的批量异步复制不需要张量映射：它可以在设备上使用指针和大小参数执行。

**来源和目的地**。批量异步复制操作的源地址和目标地址可以在共享或全局内存中。操作可以将数据从全局内存读取到共享内存，将数据从共享内存写入全局内存，还可以将共享内存复制到同一集群中另一个块的[分布式共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#distributed-shared-memory)。此外，在集群中，批量异步操作可以指定为组播。在这种情况下，数据可以从全局内存传输到集群内多个块的共享内存。组播功能针对目标架构`sm_90a`进行了优化，可能[显著降低了](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)其他目标的[性能](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)。因此，建议与[计算架构](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)`sm_90a`一起使用。

**非同步**。使用TMA的数据传输[是非同步](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-simt-programming-model)的。这允许启动线程继续计算，而硬件则异步复制数据。**数据传输在实践中是否异步发生取决于硬件实现，并且在未来可能会发生变化**。批量异步操作可以使用几种[完成机制](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms)来表示已完成。当操作从全局读取到共享内存时，块中的任何线程都可以通过等待[共享内存屏障](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier)来等待共享内存中的数据可读。当批量异步操作将数据从共享内存写入全局或分布式共享内存时，只有启动线程可以等待操作完成。这是使用基于_批量异步组_的完成机制完成的。描述完成机制的表格可以在下面和[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)中找到。

表8 具有可能的源和目标内存空间以及完成机制的异步副本。空单元格表示不支持源-目的地对。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-tma-source-dest-state-spaces "此表的永久链接")
|方向|   |完成机制|   |
|---|---|---|---|
|目的地|来源|异步复制|批量异步复制（TMA）|
|---|---|---|---|
|全球|全球|||
|全球|共享::cta||批量异步组|
|共享::cta|全球|异步组，屏障|屏障|
|共享：集群|全球||Mbarrier（多播）|
|共享::cta|共享：集群||屏障|
|共享::cta|共享::cta|||

### 10.29.1.使用TMA传输一维数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-one-dimensional-arrays "这个标题的永久链接")

本节演示如何编写一个简单的内核，该内核使用TMA读取-修改-写入一维数组。这展示了如何使用批量异步副本加载和存储数据，以及如何将执行线程与这些副本同步。

内核的代码包含在下面。某些功能需要内联PTX程序集，目前通过[libcu++](https://nvidia.github.io/cccl/libcudacxx/ptx.html)提供。可以通过以下代码检查这些包装纸的可用性：

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
static_assert(false, "Device code is being compiled with older architectures that are incompatible with TMA.");
#endif // __CUDA_MINIMUM_ARCH__

内核经历以下阶段：

1. 初始化共享内存屏障。
    
2. 启动从全局内存到共享内存块的批量异步复制。
    
3. 到达并等待共享记忆屏障。
    
4. 增加共享内存缓冲值。
    
5. 等待共享内存写入对后续的批量异步副本可见，即在下一步之前，在[异步代理中](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#async-proxy)对共享内存写入进行排序。
    
6. 将共享内存中的缓冲区批量异步复制到全局内存。
    
7. 等待内核末尾的批量异步副本完成读取共享内存。
    

#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                      // a)
    ptx::fence_proxy_async(ptx::space_shared);   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
    cuda::memcpy_async(
        smem_data, 
        data + offset, 
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar
    );
  }
  // 3b. All threads arrive on the barrier
  barrier::arrival_token token = bar.arrive();
  
  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);   // b)
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk(
        ptx::space_global,
        ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}

**屏障初始化**。屏障以参与区块的线程数量初始化。因此，只有当所有线程都到达该屏障时，屏障才会翻转。共享内存障碍在[使用cuda::barrier的异步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memcpy-async-barrier)中进行了更详细的描述。为了使初始化的屏障对后续的批量异步副本可见，使用了`fence.proxy.async.shared::cta`指令。此指令确保后续的批量异步复制操作在初始化的屏障上运行。

**TMA阅读**。批量异步复制指令指示硬件将大量数据复制到共享内存中，并在完成读取后更新共享内存屏障的[事务计数](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations)。一般来说，尽可能少地发布大尺寸的批量副本，效果最好。由于副本可以通过硬件异步执行，因此没有必要将副本分成更小的块。

启动批量异步复制操作的线程使用`mbarrier.expect_tx`到达屏障。这是由`cuda::memcpy_async`自动执行的。这告诉了线程已经到达的障碍，以及预计到达的字节（tx/事务）有多少。只有一个线程需要更新预期的交易数量。如果多个线程更新交易计数，预期交易将是更新的总和。只有在所有线程到达**和所有**字节到达后，屏障才会翻转。一旦障碍翻转，字节就可以安全地从共享内存中读取，无论是线程还是通过后续的批量异步副本。有关障碍交易会计的更多信息可以在[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations)中找到。

**屏障等待**。使用`mbarrier.try_wait`完成等待翻转的障碍。它可以返回true，表示等待已经结束，也可以返回false，这可能意味着等待超时。while循环等待完成，并在超时重新重做。

**SMEM写入和同步**。缓冲值的增量读取和写入共享内存。为了使写入对后续的批量异步副本可见，使用了`fence.proxy.async.shared::cta`指令。这在从批量异步复制操作中读取后续读取之前，将写入顺序分配到共享内存，该操作通过异步代理读取。因此，每个线程首先通过`fence.proxy.async.shared::cta`在异步代理中命令写入共享内存中的对象，所有线程的这些操作都在使用`__syncthreads()`在线程0中执行异步操作之前进行排序。

**TMA写入和同步**。从共享到全局内存的写入再次由单个线程启动。共享内存屏障不会跟踪写入的完成。相反，使用线程本地机制。多个写入可以批量合并到所谓的_批量异步组中_。之后，线程可以等待此组中的所有操作完成从共享内存中读取（如上面的代码）或完成写入全局内存，使写入对启动线程可见。有关更多信息，请参阅[cp.async.bulk.wait_group](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)的PTX ISA文档。请注意，批量异步和非批量异步复制指令有不同的异步组：`cp.async.wait_group`和`cp.async.bulk.wait_group`指令都存在。

批量异步指令对其源地址和目标地址有特定的对齐要求。更多信息可以在下表中找到。

表9计算能力9.0中一维批量异步操作的对齐要求。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-alignment-one-dim-tma "此表的永久链接")
|地址/尺寸|对齐|
|---|---|
|全局内存地址|必须对齐16字节。|
|共享内存地址|必须对齐16字节。|
|共享内存屏障地址|必须对齐8字节（这由`cuda::barrier`保证）。|
|转移规模|必须是16字节的倍数。|

### 10.29.2.使用TMA传输多维数组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-multi-dimensional-arrays "这个标题的永久链接")

一维和多维情况之间的主要区别在于，必须在主机上创建张量映射，并传递给CUDA内核。本节介绍如何使用CUDA驱动程序API创建张量图，如何将其传递给设备，以及如何在设备上使用它。

**Driver API**. A tensor map is created using the [cuTensorMapEncodeTiled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) driver API. This API can be accessed by linking to the driver directly (`-lcuda`) or by using the [cudaGetDriverEntryPointByVersion](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER__ENTRY__POINT.html) API. Below, we show how to get a pointer to the `cuTensorMapEncodeTiled` API. For more information, refer to [Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-entry-point-access).

#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

**Creation**. Creating a tensor map requires many parameters. Among them are the base pointer to an array in global memory, the size of the array (in number of elements), the stride from one row to the next (in bytes), the size of the shared memory buffer (in number of elements). The code below creates a tensor map to describe a two-dimensional row-major array of size `GMEM_HEIGHT x GMEM_WIDTH`. Note the order of the parameters: the fastest moving dimension comes first.

  CUtensorMap tensor_map{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    rank,                       // cuuint32_t tensorRank,
    tensor_ptr,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

**主机到设备传输**。有三种方法可以让设备代码访问张量图。推荐的方法是将张量映射作为常量`__grid_constant__`参数传递给内核。其他可能性是使用`cudaMemcpyToSymbol`将张量映射复制到设备`__constant__`内存中，或者通过全局内存访问它。当将张量映射作为参数传递时，一些版本的GCC C++编译器会发出警告“在GCC 4.6中传递64字节对齐参数的ABI已经改变”。这个警告可以忽略。

#include <cuda.h>

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map)
{
   // Use tensor_map here.
}
int main() {
  CUtensorMap map;
  // [ ..Initialize map.. ]
  kernel<<<1, 1>>>(map);
}

作为`__grid_constant__`内核参数的替代方案，可以使用全局[常量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant)变量。下面包括一个例子。

#include <cuda.h>

__constant__ CUtensorMap global_tensor_map;
__global__ void kernel()
{
  // Use global_tensor_map here.
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
  kernel<<<1, 1>>>();
}

最后，可以将张量图复制到全局内存。在全局设备内存中使用指向张量映射的指针，需要在块中的任何线程中使用更新的张量映射之前，在每个线程块中都有一个栅栏。除非再次修改张量图，否则不需要对该线程块对张量图的进一步使用进行围栏。请注意，这种机制可能比上面描述的两种机制慢。

#include <cuda.h>
#include <cuda/ptx>
namespace ptx = cuda::ptx;

__device__ CUtensorMap global_tensor_map;
__global__ void kernel(CUtensorMap *tensor_map)
{
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  // Since the tensor map was modified from the host using cudaMemcpy,
  // the scope should be .sys.
  ptx::fence_proxy_tensormap_generic(
     ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes
 );
 // Safe to use tensor_map after fence inside this thread..
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..Initialize map.. ]
  cudaMemcpy(&global_tensor_map, &local_tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  kernel<<<1, 1>>>(global_tensor_map);
}

**Use**. The kernel below loads a 2D tile of size `SMEM_HEIGHT x SMEM_WIDTH` from a larger 2D array. The top-left corner of the tile is indicated by the indices `x` and `y`. The tile is loaded into shared memory, modified, and written back to global memory.

#include <cuda.h>         // CUtensormap
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
    // Make initialized barrier visible in async proxy.
    cde::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // Initiate bulk tensor copy.
    cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, x, y, bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  // Symbolically modify a value in shared memory.
  smem_buffer[0][threadIdx.x] += threadIdx.x;

  // Wait for shared memory writes to be visible to TMA engine.
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);
    // Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    cde::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    cde::cp_async_bulk_wait_group_read<0>();
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}

**负指数和范围外**。当从全局_读取_到共享内存的部分图块超出界时，对应于超出界区域的共享内存为零填充。瓷砖的左上角索引也可能是负的。从共享到全局内存_写入_时，瓷砖的部分内容可能超出界，但左上角不能有任何负指数。

**大小和步长**。张量的大小是一个维度上的元素数。所有尺寸必須大於1。步长是同一维度元素之间的字节数。例如，一个4 x 4的整数矩阵的大小为4和4。由于每个元素有4个字节，所以步数为4个字节和16个字节。由于对齐要求，4 x 3行整数大矩阵也必须有4和16字节的步长。每行都加了4个额外的字节，以确保下一行的开头与16字节对齐。有关对齐的更多信息，请参阅表 [10](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-alignment-multi-dim-tma)。

表10计算能力9.0中多维批量张量异步复制操作的对齐要求。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-alignment-multi-dim-tma "此表的永久链接")
|地址/尺寸|对齐|
|---|---|
|全局内存地址|必须对齐16字节。|
|全局内存大小|必须大于或等于1。不一定是16字节的倍数。|
|全局内存步长|必须是16字节的倍数。|
|共享内存地址|必须对齐128字节。|
|共享内存屏障地址|必须对齐8字节（这由`cuda::barrier`保证）。|
|转移规模|必须是16字节的倍数。|

#### 10.29.2.1.多维TMA PTX包装机[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-dimensional-tma-ptx-wrappers "这个标题的永久链接")

下面，PTX指令按上述示例代码中的使用顺序排列。

[cp.async.bulk.tensor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)指令在全局和共享内存之间启动批量张量异步复制。下面的包装器从全局读取到共享内存，从共享到全局内存写入。

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_1d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map , int c0, cuda::barrier<cuda::thread_scope_block> &bar
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map , int c0, int c1, cuda::barrier<cuda::thread_scope_block> &bar
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_3d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map, int c0, int c1, int c2, cuda::barrier<cuda::thread_scope_block> &bar
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_4d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map , int c0, int c1, int c2, int c3, cuda::barrier<cuda::thread_scope_block> &bar
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_5d_global_to_shared(
    void *dest, const CUtensorMap *tensor_map , int c0, int c1, int c2, int c3, int c4, cuda::barrier<cuda::thread_scope_block> &bar
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_1d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, const void *src
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_2d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, const void *src
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_3d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, int c2, const void *src
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_4d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, int c2, int c3, const void *src
);

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
inline __device__
void cuda::device::experimental::cp_async_bulk_tensor_5d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, int c2, int c3, int c4, const void *src
);

### 10.29.3.TMA Swizzle[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tma-swizzle "这个标题的永久链接")

默认情况下，TMA引擎将数据加载到共享内存中，顺序与在全局内存中排列的顺序相同。然而，这种布局可能不是某些共享内存访问模式的最佳方式，因为它可能会导致共享内存库冲突。为了提高性能并减少银行冲突，我们可以通过应用“swizzle模式”来更改共享内存布局。

共享内存有32个库，这些库被组织成连续的32位单词映射到连续的库。每个银行的带宽为每个时钟周期32位。在加载和存储共享内存时，如果在同一库在事务中多次使用，则会发生库冲突，从而导致带宽降低。请参阅[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x)，银行冲突。

为了确保数据以用户代码可以避免共享内存库冲突的方式布局在共享内存中，可以指示TMA引擎在将数据存储在共享内存中之前“sizzizz”数据，并在将数据从共享内存复制回全局内存时“取消swizz”数据。张量图编码“swizzle模式”，表示使用哪种swizzle模式。

#### 10.29.3.1.示例“矩阵转置”[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#example-matrix-transpose "这个标题的永久链接")

一个例子是矩阵的转置，其中数据首先从行映射到列访问。数据存储在全局内存中，但我们也想在共享内存中以列方式访问它，这会导致银行冲突。然而，通过使用128字节的“swizzle”模式和新的共享内存索引，它们被淘汰了。

在示例中，我们将`int4`类型的8x8矩阵加载到共享内存中，该矩阵作为行主要存储在全局内存中。然后，每组八个线程从共享内存缓冲区加载一行，并将其存储到单独的转置共享内存缓冲区中的一列中。这导致存储时发生八向银行冲突。最后，转置缓冲区被写回全局内存。

为了避免银行冲突，可以使用`CU_TENSOR_MAP_SWIZZLE_128B`布局。此布局与128字节的行长度相匹配，并更改共享内存布局，以便列和行方式访问不需要每笔交易相同的银行。

下面的两个表格，[图27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#figure-swizzle-example1)和[图28，](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#figure-swizzle-example2)显示了`int4`类型8x8矩阵及其转置矩阵的法线和旋转共享内存布局。颜色表示矩阵元素映射到四个银行的八组中的哪一组，边距行和边距列列出了全局内存行和列索引。条目显示了16字节矩阵元素的共享内存索引。

[![没有swizzle的共享内存数据布局](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/example1.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/example1.png)

图27在没有swizzle的共享内存数据布局中，共享内存指数等同于全局内存指数。根据加载指令，一行被读取并存储在转置缓冲区的一列中。由于转置中列的所有矩阵元素都位于同一库中，因此存储必须序列化，从而产生八个存储交易，每个存储的列都会产生八路库冲突。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id468 "此图像的永久链接")

[![与CU_TENSOR_MAP_SWIZZLE_128B swizzle共享内存数据布局。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/example2.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/example2.png)

Figure 28 The shared memory data layout with `CU_TENSOR_MAP_SWIZZLE_128B` swizzle. One row is stored in a column, each matrix element is from a different bank for both the rows and columns, and so without any bank conflicts.[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id469 "此图像的永久链接")
```c++
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   // The destination shared memory buffer of a bulk tensor operation
   // with the 128-byte swizzle mode, it should be 1024 bytes aligned.
   __shared__ alignas(1024) int4 smem_buffer[8][8];
   __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

   // Initialize shared memory barrier
   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ barrier bar;

   if (threadIdx.x == 0) {
     init(&bar, blockDim.x);
     cde::fence_proxy_async_shared_cta();
   }

   __syncthreads();

   barrier::arrival_token token;
   if (threadIdx.x == 0) {
     // Initiate bulk tensor copy from global to shared memory,
     // in the same way as without swizzle.
     cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_buffer, &tensor_map, 0, 0, bar);
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else {
     token = bar.arrive();
   }

   bar.wait(std::move(token));

   /* Matrix transpose
    *  When using the normal shared memory layout, there are eight
    *  8-way shared memory bank conflict when storing to the transpose.
    *  When enabling the 128-byte swizzle pattern and using the according access pattern,
    *  they are eliminated both for load and store. */
   for(int sidx_j =threadIdx.x; sidx_j < 8; sidx_j+= blockDim.x){
      for(int sidx_i = 0; sidx_i < 8; ++sidx_i){
         const int swiz_j_idx = (sidx_i % 8) ^ sidx_j;
         const int swiz_i_idx_tr = (sidx_j % 8) ^ sidx_i;
         smem_buffer_tr[sidx_j][swiz_i_idx_tr] = smem_buffer[sidx_i][swiz_j_idx];
      }
   }

   // Wait for shared memory writes to be visible to TMA engine.
   cde::fence_proxy_async_shared_cta();
   __syncthreads();

   /* Initiate TMA transfer to copy the transposed shared memory buffer back to global memory,
    * it will 'unswizzle' the data. */
   if (threadIdx.x == 0) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, 0, 0, &smem_buffer_tr);
      cde::cp_async_bulk_commit_group();
      cde::cp_async_bulk_wait_group_read<0>();
   }

   // Destroy barrier
   if (threadIdx.x == 0) {
     (&bar)->~barrier();
   }
}

// --------------------------------- main ----------------------------------------

int main(){

...
   void* tensor_ptr = d_data;

   CUtensorMap tensor_map{};
   // rank is the number of dimensions of the array.
   constexpr uint32_t rank = 2;
   // global memory size
   uint64_t size[rank] = {4*8, 8};
   // global memory stride, must be a multiple of 16.
   uint64_t stride[rank - 1] = {8 * sizeof(int4)};
   // The inner shared memory box dimension in bytes, equal to the swizzle span.
   uint32_t box_size[rank] = {4*8, 8};

   uint32_t elem_stride[rank] = {1, 1};

   // Create the tensor descriptor.
   CUresult res = cuTensorMapEncodeTiled(
       &tensor_map,                // CUtensorMap *tensorMap,
       CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
       rank,                       // cuuint32_t tensorRank,
       tensor_ptr,                 // void *globalAddress,
       size,                       // const cuuint64_t *globalDim,
       stride,                     // const cuuint64_t *globalStrides,
       box_size,                   // const cuuint32_t *boxDim,
       elem_stride,                // const cuuint32_t *elementStrides,
       CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
       // Using a swizzle pattern of 128 bytes.
       CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
       CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
       CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
   );

   kernel_tma<<<1, 8>>>(tensor_map);
 ...
}
```
**备注。**这个例子应该展示swizzle的使用，“as-is”没有性能，也没有超出给定维度的扩展范围。

解释。在数据传输过程中，TMA引擎根据抖动模式对数据进行洗牌，如下表所述。这些swizzle模式定义了沿swizzle宽度的16字节块映射到四个银行的子组。它的类型为`CUtensorMapSwizzle`，有四个选项：无、32字节、64字节和128字节。请注意，共享内存盒的内部尺寸必须小于或等于swizzle模式的跨度。

#### 10.29.3.2.Swizzle模式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-swizzle-modes "这个标题的永久链接")

如前所述，有四种swizzle模式。下表显示了不同的swizzle模式，包括新共享内存指数的关系。这些表定义了沿128字节的16字节块映射到四个银行的八个子组。

[![TMA Swizzle模式概述](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/swizzle-pattern.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/swizzle-pattern.png)

图29 TMA Swizzle模式概述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id470 "此图像的永久链接")

**考虑因素。**在应用TMA swizzle模式时，遵守特定的内存要求至关重要：

- **全局内存对齐：**全局内存必须对齐到128字节。
    
- **共享内存对齐：**为了简单起见，共享内存应根据字节数对齐，之后旋转模式重复。当共享内存缓冲区与swizzle模式重复的字节数不一致时，swizzle模式和共享内存之间存在偏移。请参阅下面的[评论](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#swizzle-pattern-pointer-offset-computation)。
    
- **内部尺寸：**共享内存块的内部尺寸必须满足表[12](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-swizzle-pattern-properties-and-requirements)中规定的尺寸要求。如果不符合这些要求，则该指令将被视为无效。此外，如果swizzle宽度超过内部尺寸，请确保分配共享内存以适应完整的swizzle宽度。
    
- 粒度：swizzle映射的粒度固定在16字节。这意味着数据以16字节的块形式组织和访问，在规划内存布局和访问模式时必须考虑这一点。
    

**Swizzle模式指针偏移计算**。在这里，我们描述了如何确定swizzle模式和共享内存之间的偏移，当共享内存缓冲区没有按swizzle模式重复的字节数对齐时。使用TMA时，共享内存需要与128字节对齐。要找出共享内存缓冲区相对于swizzle模式的移动次数，请应用相应的偏移公式。

表11 Swizzle模式指针偏移公式和指数关系[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-swizzle-pattern-offset "此表的永久链接")
|旋转模式|偏移公式|指数关系|
|---|---|---|
|CU_TENSOR_地图_SWIZZLE_128B|`(reinterpret_cast <uintptr_t>(smem_ptr)/128)%8`|`smem[y][x] <-> smem[y][((y+offset)%8)^x]`|
|CU_TENSOR_MAP_SWIZZLE_64B|`(reinterpret_cast <uintptr_t>(smem_ptr)/128)%4`|`smem[y][x] <-> smem[y][((y+offset)%4)^x]`|
|CU_TENSOR_地图_SWIZZLE_32B|`(reinterpret_cast <uintptr_t>(smem_ptr)/128)%2`|`smem[y][x] <-> smem[y][((y+offset)%2)^x]`|

在[图29中，](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#figure-swizzle-overview)这个偏移表示初始行偏移，因此，在swizzle指数计算中，它被添加到行索引y中。以下片段展示了如何在`CU_TENSOR_MAP_SWIZZLE_128B`模式下访问swizzled共享内存。
```c++
data_t* smem_ptr = &smem[0][0];
int offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8;
smem[y][((y+offset)%8)^x] = ...
```
**总结。**下表[12](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-swizzle-pattern-properties-and-requirements)总结了计算能力9的不同swizzle模式的要求和属性。

表12计算能力9的不同swizzle模式的要求和属性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-swizzle-pattern-properties-and-requirements "此表的永久链接")
|模式|旋转宽度|共享盒子的内部尺寸|重复之后|共享内存对齐|全局内存对齐|
|---|---|---|---|---|---|
|CU_TENSOR_地图_SWIZZLE_128B|128字节|<=128字节|1024字节|128字节|128字节|
|CU_TENSOR_MAP_SWIZZLE_64B|64字节|<=64字节|512字节|128字节|128字节|
|CU_TENSOR_地图_SWIZZLE_32B|32字节|<=32字节|256字节|128字节|128字节|
|CU_TENSOR_MAP_SWIZZLE_NONE（默认）||||128字节|16字节|

## 10.30.在设备上编码张量图[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#encoding-a-tensor-map-on-device "这个标题的永久链接")

前几节介绍了如何使用CUDA驱动程序API在主机上创建张量图。

This section explains how to encode a tiled-type tensor map on device. This is useful in situations where the typical way of transferring the tensor map (using `const __grid_constant__` kernel parameters) is undesirable, for instance, when processing a batch of tensors of various sizes in a single kernel launch.

推荐的模式如下：

1. 使用主机上的驱动程序API创建一个张量图“模板”，`template_tensor_map`。
    
2. 在设备内核中，复制`template_tensor_map`，修改副本，存储在全局内存中，并适当地围栏。
    
3. 在带有适当围栏的内核中使用张量映射。
    

高级代码结构如下：

// Initialize device context:
CUDA_CHECK(cudaDeviceSynchronize());

// Create a tensor map template using the cuTensorMapEncodeTiled driver function
CUtensorMap template_tensor_map = make_tensormap_template();

// Allocate tensor map and tensor in global memory
CUtensorMap* global_tensor_map;
CUDA_CHECK(cudaMalloc(&global_tensor_map, sizeof(CUtensorMap)));
char* global_buf;
CUDA_CHECK(cudaMalloc(&global_buf, 8 * 256));

// Fill global buffer with data.
fill_global_buf<<<1, 1>>>(global_buf);

// Define the parameters of the tensor map that will be created on device.
tensormap_params p{};
p.global_address    = global_buf;
p.rank              = 2;
p.box_dim[0]        = 128; // The box in shared memory has half the width of the full buffer
p.box_dim[1]        = 4;   // The box in shared memory has half the height of the full buffer
p.global_dim[0]     = 256; //
p.global_dim[1]     = 8;   //
p.global_stride[0]  = 256; //
p.element_stride[0] = 1;   //
p.element_stride[1] = 1;   //

// Encode global_tensor_map on device:
encode_tensor_map<<<1, 32>>>(template_tensor_map, p, global_tensor_map);

// Use it from another kernel:
consume_tensor_map<<<1, 1>>>(global_tensor_map);

// Check for errors:
CUDA_CHECK(cudaDeviceSynchronize());

以下部分介绍高级步骤。在整个示例中，以下`tensormap_params`结构包含要更新的字段的新值。它包含在此处，以便在阅读示例时参考。

struct tensormap_params {
  void* global_address;
  int rank;
  uint32_t box_dim[5];
  uint64_t global_dim[5];
  size_t global_stride[4];
  uint32_t element_stride[5];
};

### 10.30.1.张量图的设备端编码和修改[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-side-encoding-and-modification-of-a-tensor-map "这个标题的永久链接")

建议在全局内存中编码张量图的过程如下。

1. 将现有的张量映射，即`template_tensor_map`传递给内核。与在`cp.async.bulk.tensor`指令中使用张量映射的内核相反，这可以通过任何方式完成：指向全局内存的指针、内核参数、`__const___`变量等。
    
2. 使用template_tensor_map值在共享内存中复制-初始化张量映射。
    
3. 使用[cuda::ptx::tensormap_replace](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap.replace.html)函数修改共享内存中的张量图。这些函数包裹[tensormap.replace](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace) PTX指令，该指令可用于修改平铺式张量映射的任何字段，包括基本地址、大小、步长等。
    
4. 使用[cuda::ptx::tensormap_copy_fenceproxy](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap.cp_fenceproxy.html#tensormap-cp-fenceproxy)函数，将修改后的张量映射从共享内存复制到全局内存，并执行任何必要的围栏。
    

以下代码包含遵循这些步骤的内核。为了完整，它修改了张量图的所有字段。通常，内核只会修改几个字段。

在这个内核中，`template_tensor_map`作为内核参数传递。这是将`template_tensor_map`从主机移动到设备的首选方法。如果内核打算更新设备内存中的现有张量图，它可以获取指向现有张量图的指针进行修改。

笔记

The format of the tensor map may change over time. Therefore, the [cuda::ptx::tensormap_replace](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap.replace.html) functions and corresponding [tensormap.replace.tile](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-tensormap-replace) PTX instructions are marked as specific to sm_90a. To use them, compile using `nvcc -arch sm_90a ....`.

提示

在sm_90a上，共享内存中的零初始化緩衝區也可以用作初始張量映射值。这允许纯粹在设备上编码张量映射，而无需使用驱动程序API对`template_tensor_map`进行编码。

笔记

仅支持平铺式张量映射的设备端修改；其他张量映射类型无法在设备上进行修改。有关张量映射类型的更多信息，请参阅[驱动程序API参考](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY)。

#include <cuda/ptx>

namespace ptx = cuda::ptx;

// launch with 1 warp.
__launch_bounds__(32)
__global__ void encode_tensor_map(const __grid_constant__ CUtensorMap template_tensor_map, tensormap_params p, CUtensorMap* out) {
   __shared__ alignas(128) CUtensorMap smem_tmap;
   if (threadIdx.x == 0) {
      // Copy template to shared memory:
      smem_tmap = template_tensor_map;

      const auto space_shared = ptx::space_shared;
      ptx::tensormap_replace_global_address(space_shared, &smem_tmap, p.global_address);
      // For field .rank, the operand new_val must be ones less than the desired
      // tensor rank as this field uses zero-based numbering.
      ptx::tensormap_replace_rank(space_shared, &smem_tmap, p.rank - 1);

      // Set box dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.box_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.box_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.box_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.box_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_box_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.box_dim[4]); }
      // Set global dimensions:
      if (0 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<0>{}, (uint32_t) p.global_dim[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<1>{}, (uint32_t) p.global_dim[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<2>{}, (uint32_t) p.global_dim[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<3>{}, (uint32_t) p.global_dim[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_dim(space_shared, &smem_tmap, ptx::n32_t<4>{}, (uint32_t) p.global_dim[4]); }
      // Set global stride:
      if (1 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.global_stride[0]); }
      if (2 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.global_stride[1]); }
      if (3 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.global_stride[2]); }
      if (4 < p.rank) { ptx::tensormap_replace_global_stride(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.global_stride[3]); }
      // Set element stride:
      if (0 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<0>{}, p.element_stride[0]); }
      if (1 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<1>{}, p.element_stride[1]); }
      if (2 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<2>{}, p.element_stride[2]); }
      if (3 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<3>{}, p.element_stride[3]); }
      if (4 < p.rank) { ptx::tensormap_replace_element_size(space_shared, &smem_tmap, ptx::n32_t<4>{}, p.element_stride[4]); }

      // These constants are documented in this table:
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensormap-new-val-validity
      auto u8_elem_type = ptx::n32_t<0>{};
      ptx::tensormap_replace_elemtype(space_shared, &smem_tmap, u8_elem_type);
      auto no_interleave = ptx::n32_t<0>{};
      ptx::tensormap_replace_interleave_layout(space_shared, &smem_tmap, no_interleave);
      auto no_swizzle = ptx::n32_t<0>{};
      ptx::tensormap_replace_swizzle_mode(space_shared, &smem_tmap, no_swizzle);
      auto zero_fill = ptx::n32_t<0>{};
      ptx::tensormap_replace_fill_mode(space_shared, &smem_tmap, zero_fill);
   }
   // Synchronize the modifications with other threads in warp
   __syncwarp();
   // Copy the tensor map to global memory collectively with threads in the warp.
   // In addition: make the updated tensor map visible to other threads on device that
   // for use with cp.async.bulk.
   ptx::n32_t<128> bytes_128;
   ptx::tensormap_cp_fenceproxy(ptx::sem_release, ptx::scope_gpu, out, &smem_tmap, bytes_128);
}

### 10.30.2.修改后张量图的使用[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#usage-of-a-modified-tensor-map "这个标题的永久链接")

In contrast to using a tensor map that is passed as a `const __grid_constant__` kernel parameter, using a tensor map in global memory requires explicitly establishing a release-acquire pattern in the tensor map proxy between the threads that modify the tensor map and the threads that use it.

模式的释放部分在上一节中显示。它是使用[cuda::ptx::tensormap.cp_fenceproxy](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/tensormap.cp_fenceproxy.html)函数完成的。

获取部分使用[cuda::ptx::fence_proxy_tensormap_generic](https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/fence.html)函数完成，该函数包裹了[fence.proxy.tensormap::generic.acquire](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence)指令。如果参与发布-获取模式的两个线程在同一设备上，则`.gpu`范围就足够了。如果线程位于不同的设备上，则必须使用`.sys`范围。一旦一个线程获取了张量映射，在充分同步后，块中的其他线程就可以使用它，例如，使用`__syncthreads()`使用张量图的线程和执行围栏的线程必须在同一个块中。也就是说，如果线程在，例如，同一集群、同一网格或不同内核的两个不同线程块中，则同步API，如`cooperative_groups::cluster`或`grid_group::sync()`或流序同步不足以建立张量映射更新的顺序，也就是说，这些其他线程块中的线程在使用更新的张量映射之前仍然需要在正确的范围内获取张量映射代理。如果没有中间修改，则不必在每个`cp.async.bulk.tensor`指令之前重复栅栏。

`fence`和随后的张量图的使用如以下示例所示。

// Consumer of tensor map in global memory:
__global__ void consume_tensor_map(CUtensorMap* tensor_map) {
  // Fence acquire tensor map:
  ptx::n32_t<128> size_bytes;
  ptx::fence_proxy_tensormap_generic(ptx::sem_acquire, ptx::scope_sys, tensor_map, size_bytes);
  // Safe to use tensor_map after fence..

  __shared__ uint64_t bar;
  __shared__ alignas(128) char smem_buf[4][128];

  if (threadIdx.x == 0) {
    // Initialize barrier
    ptx::mbarrier_init(&bar, 1);
    // Make barrier init visible in async proxy, i.e., to TMA engine
    ptx::fence_proxy_async(ptx::space_shared);
    // Issue TMA request
    ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, smem_buf, tensor_map, {0, 0}, &bar);

    // Arrive on barrier. Expect 4 * 128 bytes.
    ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar, sizeof(smem_buf));
  }
  const int parity = 0;
  // Wait for load to have completed
  while (!ptx::mbarrier_try_wait_parity(&bar, parity)) {}

  // print items:
  printf("Got:\n\n");
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 128; ++i) {
      printf("%3d ", smem_buf[j][i]);
      if (i % 32 == 31) { printf("\n"); };
    }
    printf("\n");
  }
}

### 10.30.3.使用驱动程序API创建模板张量映射值[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-template-tensor-map-value-using-the-driver-api "这个标题的永久链接")

以下代码创建了一个最小平铺型张量图，随后可以在设备上进行修改。

CUtensorMap make_tensormap_template() {
  CUtensorMap template_tensor_map{};
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  uint32_t dims_32         = 16;
  uint64_t dims_strides_64 = 16;
  uint32_t elem_strides    = 1;

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &template_tensor_map, // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
    1,                // cuuint32_t tensorRank,
    nullptr,          // void *globalAddress,
    &dims_strides_64, // const cuuint64_t *globalDim,
    &dims_strides_64, // const cuuint64_t *globalStrides,
    &dims_32,         // const cuuint32_t *boxDim,
    &elem_strides,    // const cuuint32_t *elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CU_CHECK(res);
  return template_tensor_map;
}

## 10.31.分析器计数器功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#profiler-counter-function "这个标题的永久链接")

每个多处理器都有一组十六个硬件计数器，应用程序可以通过调用`__prof_trigger()`函数来增加单个指令。

void __prof_trigger(int counter);

索引`counter`每多处理器硬件计数器每翘曲一次增量。柜台8到15是保留的，应用程序不应使用。

The value of counters 0, 1, …, 7 can be obtained via `nvprof` by `nvprof --events prof_trigger_0x` where `x` is 0, 1, …, 7. All counters are reset before each kernel launch (note that when collecting counters, kernel launches are synchronous as mentioned in [Concurrent Execution between Host and Device](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-execution-host-device)).

## 10.32.断言[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#assertion "这个标题的永久链接")

断言仅由具有2.x及更高计算能力的设备支持。
```c++
void assert(int expression);
```
如果`expression`等于零，则停止内核执行。如果程序在调试器内运行，这会触发一个断点，调试器可用于检查设备的当前状态。否则，`expression`等于零的每个线程在通过`cudaDeviceSynchronize()``cudaStreamSynchronize()`或`cudaEventSynchronize()`与主机同步后向_stderr_打印消息。此消息的格式如下：
```c++
<filename>:<line number>:<function>:
block: [blockId.x,blockId.x,blockIdx.z],
thread: [threadIdx.x,threadIdx.y,threadIdx.z]
Assertion `<expression>` failed.
```
任何后续针对同一设备的主机端同步调用都将返回`cudaErrorAssert`。在调用`cudaDeviceReset()`重新初始化设备之前，不能再向该设备发送命令。

如果`expression`与零不同，则内核执行不受影响。

例如，源文件_test.cu_中的以下程序
```c++
#include <assert.h>

__global__ void testAssert(void)
{
    int is_one = 1;
    int should_be_one = 0;

    // This will have no effect
    assert(is_one);

    // This will halt kernel execution
    assert(should_be_one);
}

int main(int argc, char* argv[])
{
    testAssert<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}
```
将输出：

test.cu:19: void testAssert(): block: [0,0,0], thread: [0,0,0] Assertion `should_be_one` failed.

Assertions are for debugging purposes. They can affect performance and it is therefore recommended to disable them in production code. They can be disabled at compile time by defining the `NDEBUG` preprocessor macro before including `assert.h`. Note that `expression` should not be an expression with side effects (something like`(++i > 0)`, for example), otherwise disabling the assertion will affect the functionality of the code.

## 10.33.陷阱功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#trap-function "这个标题的永久链接")

可以通过从任何设备线程调用`__trap()`函数来启动陷阱操作。

void __trap();

内核的执行被中止，并在主机程序中引发中断。

## 10.34.断点函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#breakpoint-function "这个标题的永久链接")

可以通过从任何设备线程调用`__brkpt()`函数来暂停内核函数的执行。

void __brkpt();

## 10.35.格式化输出[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#formatted-output "这个标题的永久链接")

格式化输出仅支持具有2.x及更高计算能力的设备。

int printf(const char *format[, arg, ...]);

将格式化的输出从内核打印到主机端输出流。

内核`printf()`函数的行为方式与标准C库`printf()`函数相似，用户可以参考主机系统的手册页面，以获取`printf()`行为的完整描述。从本质上讲，作为`format`传递的字符串被输出到主机上的流中，在遇到格式指定符的地方，从参数列表中进行替换。支持的格式指定符如下所示。

`printf()`命令作为任何其他设备端函数执行：每个线程，以及在调用线程的上下文中。从多线程内核来看，这意味着每个线程将使用指定的线程数据执行对`printf()`的直接调用。然后，输出字符串的多个版本将出现在主机流中，每个遇到`printf()`线程都会出现一次。

如果只需要单个输出字符串，则由程序员将输出限制为单个线程（请参阅[示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#examples-per-thread)说明性示例）。

与C标准`printf()`不同，C标准printf（）返回打印的字符数，CUDA的`printf()`返回解析的参数数。如果格式字符串后面没有参数，则返回0。如果格式字符串为NULL，则返回-1。如果发生内部错误，则返回-2。

### 10.35.1.格式指定符[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#format-specifiers "这个标题的永久链接")

对于标准`printf()`格式指定符的形式是：`%[flags][width][.precision][size]type`

支持以下字段（有关所有行为的完整描述，请参阅广泛可用的文档）：

- 旗帜：`'#' ' ' '0' '+' '-'`
- 宽度：`'*' '0-9'`
- 精确度：`'0-9'`
- 尺寸：`'h' 'l' 'll'`
- 类型：`"%cdiouxXpeEfgGaAs"`

请注意，CUDA的`printf()`将接受标志、宽度、精度、大小和类型的任何组合，无论它们是否整体形成有效的格式指定符。换句话说，“`%hd`”将被接受，printf将期望在参数列表中的相应位置有一个双精度变量。

### 10.35.2.限制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#limitations "这个标题的永久链接")

`printf()`输出的最终格式在主机系统上进行。这意味着格式字符串必须由主机系统的编译器和C库理解。已经尽一切努力确保CUDA的printf函数支持的格式指定符形成来自最常见的主机编译器的通用子集，但确切的行为将取决于主机操作系统。

如[格式指定符中](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#format-specifiers)所述，`printf()`将接受有效标志和类型的_所有_组合。这是因为它无法确定在最终输出格式化的主机系统上什么有效和无效。这样做的效果是，如果程序发出包含无效组合的格式字符串，输出可能未定义。

除了格式字符串外，`printf()`命令最多可以接受32个参数。超出此范围的其他参数将被忽略，格式指定符按原位输出。

由于64位Windows平台上的`long`大小不同（64位Windows平台上为4字节，其他64位平台上为8字节），在非Windows 64位计算机上编译但然后在win64计算机上运行的内核将看到包含“`%ld`”的所有格式字符串的损坏输出。建议编译平台与执行平台相匹配，以确保安全。

`printf()`的输出缓冲区在内核启动前被设置为固定大小（请参阅[关联主机端API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#associated-host-side-api)）。它是循环的，如果在内核执行期间产生的输出超过缓冲区，旧输出就会被覆盖。仅当执行以下操作之一时，它才会冲洗：

- 通过`<<<>>>`或`cuLaunchKernel()`启动内核（在启动开始时，如果CUDA_LAUNCH_BLOCKING环境变量设置为1，也在启动结束时），
    
- 通过`cudaDeviceSynchronize()``cuCtxSynchronize()``cudaStreamSynchronize()``cuStreamSynchronize()``cudaEventSynchronize()`orcuEventSynchronize`cuEventSynchronize()`进行同步，
    
- 通过任何阻止版本的`cudaMemcpy*()`或`cuMemcpy*()`的内存复制，
    
- 通过`cuModuleLoad()`或`cuModuleUnload()`加载/卸载模块，
    
- 通过`cudaDeviceReset()`或`cuCtxDestroy()`破坏上下文。
    
- 在执行`cudaLaunchHostFunc`或`cuLaunchHostFunc`添加的流回调之前。
    

請注意，當程式退出時，緩衝區不會自動重新整理。用户必须显式调用`cudaDeviceReset()`或`cuCtxDestroy()`，如下例所示。

在内部，`printf()`使用共享数据结构，因此调用`printf()`可能会改变线程的执行顺序。特别是，调用`printf()`的线程可能比不调用`printf()`线程使用更长的执行路径，并且该路径长度取决于`printf()`的参数。然而，请注意，除了在explicit`__syncthreads()`障碍外，CUDA不保证线程执行顺序，因此无法判断执行顺序是否已被`printf()`或硬件中的其他调度行为修改。

### 10.35.3.关联的主机端API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#associated-host-side-api "这个标题的永久链接")

以下API函数获取并设置用于将`printf()`参数和内部元数据传输到主机的缓冲区大小（默认为1兆字节）：

- `cudaDeviceGetLimit(size_t* size,cudaLimitPrintfFifoSize)`
    
- `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_t size)`
    

### 10.35.4.实例[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#format-specifier-examples "这个标题的永久链接")

以下代码示例：
```c++
#include <stdio.h>

__global__ void helloCUDA(float f)
{
    printf("Hello thread %d, f=%f\n", threadIdx.x, f);
}

int main()
{
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    return 0;
}
```
将输出：

Hello thread 2, f=1.2345
Hello thread 1, f=1.2345
Hello thread 4, f=1.2345
Hello thread 0, f=1.2345
Hello thread 3, f=1.2345

注意每个线程如何遇到`printf()`命令，因此输出行与网格中启动的线程一样多。不出所料，全局值（即`float`）在所有线程之间是共用的，局部值（即`threadIdx.x`）是每个线程不同的。

以下代码示例：
```c++
#include <stdio.h>

__global__ void helloCUDA(float f)
{
    if (threadIdx.x == 0)
        printf("Hello thread %d, f=%f\n", threadIdx.x, f) ;
}

int main()
{
    helloCUDA<<<1, 5>>>(1.2345f);
    cudaDeviceSynchronize();
    return 0;
}
```
将输出：

Hello thread 0, f=1.2345

显然，`if()`语句限制了哪些线程将调用`printf`，因此只看到一行输出。

## 10.36.动态全局内存分配和操作[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dynamic-global-memory-allocation-and-operations "这个标题的永久链接")

动态全局内存分配和操作仅由具有2.x及以上计算能力的设备支持。
```c++
__host__ __device__ void* malloc(size_t size);
__device__ void *__nv_aligned_device_malloc(size_t size, size_t align);
__host__ __device__  void free(void* ptr);
```
从全局内存中的固定大小堆中动态分配和释放内存。

__host__ __device__ void* memcpy(void* dest, const void* src, size_t size);

从`src`指向的内存位置到`dest`指向的内存位置的复制`size`字节。

__host__ __device__ void* memset(void* ptr, int value, size_t size);

设置由`ptr`指向值的内存块`size`字节（解释为无符号字符）。

CUDA内核内`malloc()`函数从设备堆中分配至少`size`字节，如果内存不足以满足请求，则返回指向分配内存或NULL的指针。返回的指针保证与16字节的边界对齐。

CUDA内核`__nv_aligned_device_malloc()`函数从设备堆中至少分配`size`字节，如果内存不足以满足要求的大小或对齐，则返回指向分配内存或NULL的指针。分配的内存的地址将是`align`的倍数。`align`必须是2的非零次数。

CUDA内核`free()`函数去分配`ptr`指向的内存，该内存必须由之前对`malloc()`或`__nv_aligned_device_malloc()`的调用返回。如果`ptr`为NULL，则忽略对`free()`的调用。使用相同的`ptr`重复调用`free()`具有未定义的行为。

给定的CUDA线程通过`malloc()`或`__nv_aligned_device_malloc()`分配的内存在CUDA上下文的生命周期内保持分配，或直到调用`free()`明确释放。它可以被任何其他CUDA线程使用，甚至从随后的内核启动开始。任何CUDA线程都可能释放由另一个线程分配的内存，但应注意确保同一指针不会被释放多次。

### 10.36.1.堆内存分配[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heap-memory-allocation "这个标题的永久链接")

设备内存堆具有固定大小，在将任何使用`malloc()``__nv_aligned_device_malloc()`或`free()`程序加载到上下文中之前，必须指定该大小。如果任何程序使用`malloc()`或`__nv_aligned_device_malloc()`而不明确指定堆大小，则会分配8兆字节的默认堆。

以下API函数获取并设置堆大小：

- `cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize)`
    
- `cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)`
    

授予的堆大小至少为字节。`cuCtxGetLimit()`和`cudaDeviceGetLimit()`返回当前请求的堆大小。

当模块加载到上下文中时，堆的实际内存分配会发生，无论是通过CUDA驱动程序API（见[模块](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#module)）显式加载，还是通过CUDA运行时API（见[CUDA运行时](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-runtime)）隐式加载。如果内存分配失败，模块加载将产生`CUDA_ERROR_SHARED_OBJECT_INIT_FAILED`错误。

一旦发生模块加载，堆大小就无法更改，它不会根据需要动态调整大小。

除了通过主机端CUDA API调用（如`cudaMalloc()`分配的内存外，还为设备堆保留的内存。

### 10.36.2.与主机内存API的互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interoperability-with-host-memory-api "这个标题的永久链接")

通过设备`malloc()`或`__nv_aligned_device_malloc()`分配的内存不能使用运行时（即呼叫[设备内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory)中的任何可用内存函数）释放。

同样，通过运行时分配的内存（即通过从[设备内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory)调用任何内存分配函数）不能通过`free()`释放。

此外，设备代码中调用`malloc()`或`__nv_aligned_device_malloc()`分配的内存不能用于任何运行时或驱动程序API调用（即cudaMemcpy、cudaMemset等）。

### 10.36.3.实例[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#examples-per-thread "这个标题的永久链接")

#### 10.36.3.1.每个线程分配[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#per-thread-allocation "这个标题的永久链接")

以下代码示例：

#include <stdlib.h>
#include <stdio.h>

__global__ void mallocTest()
{
    size_t size = 123;
    char* ptr = (char*)malloc(size);
    memset(ptr, 0, size);
    printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
    free(ptr);
}

int main()
{
    // Set a heap size of 128 megabytes. Note that this must
    // be done before any kernel is launched.
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    mallocTest<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}

将输出：

Thread 0 got pointer: 00057020
Thread 1 got pointer: 0005708c
Thread 2 got pointer: 000570f8
Thread 3 got pointer: 00057164
Thread 4 got pointer: 000571d0

注意每个线程如何遇到`malloc()`和`memset()`命令，因此接收并初始化自己的分配。（确切的指针值会有所不同：这些是说明性的。）

#### 10.36.3.2.每个线程块分配[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#per-thread-block-allocation "这个标题的永久链接")

#include <stdlib.h>

__global__ void mallocTest()
{
    __shared__ int* data;

    // The first thread in the block does the allocation and then
    // shares the pointer with all other threads through shared memory,
    // so that access can easily be coalesced.
    // 64 bytes per thread are allocated.
    if (threadIdx.x == 0) {
        size_t size = blockDim.x * 64;
        data = (int*)malloc(size);
    }
    __syncthreads();

    // Check for failure
    if (data == NULL)
        return;

    // Threads index into the memory, ensuring coalescence
    int* ptr = data;
    for (int i = 0; i < 64; ++i)
        ptr[i * blockDim.x + threadIdx.x] = threadIdx.x;

    // Ensure all threads complete before freeing
    __syncthreads();

    // Only one thread may free the memory!
    if (threadIdx.x == 0)
        free(data);
}

int main()
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    mallocTest<<<10, 128>>>();
    cudaDeviceSynchronize();
    return 0;
}

#### 10.36.3.3.在内核启动之间持续分配[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#allocation-persisting-between-kernel-launches "这个标题的永久链接")
```c++
#include <stdlib.h>
#include <stdio.h>

#define NUM_BLOCKS 20

__device__ int* dataptr[NUM_BLOCKS]; // Per-block pointer

__global__ void allocmem()
{
    // Only the first thread in the block does the allocation
    // since we want only one allocation per block.
    if (threadIdx.x == 0)
        dataptr[blockIdx.x] = (int*)malloc(blockDim.x * 4);
    __syncthreads();

    // Check for failure
    if (dataptr[blockIdx.x] == NULL)
        return;

    // Zero the data with all threads in parallel
    dataptr[blockIdx.x][threadIdx.x] = 0;
}

// Simple example: store thread ID into each element
__global__ void usemem()
{
    int* ptr = dataptr[blockIdx.x];
    if (ptr != NULL)
        ptr[threadIdx.x] += threadIdx.x;
}

// Print the content of the buffer before freeing it
__global__ void freemem()
{
    int* ptr = dataptr[blockIdx.x];
    if (ptr != NULL)
        printf("Block %d, Thread %d: final value = %d\n",
                      blockIdx.x, threadIdx.x, ptr[threadIdx.x]);

    // Only free from one thread!
    if (threadIdx.x == 0)
        free(ptr);
}

int main()
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);

    // Allocate memory
    allocmem<<< NUM_BLOCKS, 10 >>>();

    // Use memory
    usemem<<< NUM_BLOCKS, 10 >>>();
    usemem<<< NUM_BLOCKS, 10 >>>();
    usemem<<< NUM_BLOCKS, 10 >>>();

    // Free memory
    freemem<<< NUM_BLOCKS, 10 >>>();

    cudaDeviceSynchronize();

    return 0;
}
```
## 10.37.执行配置[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration "这个标题的永久链接")

对`__global__`函数的任何调用都必须指定该调用的_执行配置_。执行配置定义了用于在设备上执行功能的网格和块的维度，以及相关的流（有关流的描述，请参阅[CUDA运行时](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-runtime)）。

The execution configuration is specified by inserting an expression of the form `<<< Dg, Db, Ns, S >>>` between the function name and the parenthesized argument list, where:

- `Dg` is of type `dim3` (see [dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3)) and specifies the dimension and size of the grid, such that `Dg.x * Dg.y * Dg.z` equals the number of blocks being launched;
    
- `Db` is of type `dim3` (see [dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3)) and specifies the dimension and size of each block, such that `Db.x * Db.y * Db.z` equals the number of threads per block;
    
- `Ns`类型为`size_t`，并指定除了静态分配的内存外，共享内存中每个块动态分配的字节数；如[__shared__](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared)中提到的，此动态分配内存由声明为外部数组的任何变量使用；`Ns`是一个可选参数，默认为0；
    
- `S`是`cudaStream_t`类型，并指定关联的流；S是一个可选参数，默认为0。
    

例如，一个声明为的函数

__global__ void Func(float* parameter);

必须这样叫：

Func<<< Dg, Db, Ns >>>(parameter);

执行配置的参数在实际函数参数之前进行评估。

如果`Dg`或`Db`大于[计算能力](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities)中指定的设备允许的最大尺寸，或者`Ns`大于设备上可用的最大共享内存量，减去静态分配所需的共享内存量，函数调用将失败。

Compute capability 9.0 and above allows users to specify compile time thread block cluster dimensions, so that the kernel can use the cluster hierarchy in CUDA. Compile time cluster dimension can be specified using `__cluster_dims__([x, [y, [z]]])`. The example below shows compile time cluster size of 2 in X dimension and 1 in Y and Z dimension.

__global__ void __cluster_dims__(2, 1, 1) Func(float* parameter);

`__cluster_dims__()`的默认形式指定内核将作为集群网格启动。通过不指定集群维度，用户可以在启动时自由指定维度。在启动时不指定维度将导致启动时间错误。

Thread block cluster dimensions can also be specified at runtime and kernel with the cluster can be launched using `cudaLaunchKernelEx` API. The API takes a configuration argument of type `cudaLaunchConfig_t`, kernel function pointer and kernel arguments. Runtime kernel configuration is shown in the example below.
```c++
__global__ void Func(float* parameter);

// Kernel invocation with runtime cluster size
{
    cudaLaunchConfig_t config = {0};
    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension should be a multiple of cluster size.
    config.gridDim = Dg;
    config.blockDim = Db;
    config.dynamicSmemBytes = Ns;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    float* parameter;
    cudaLaunchKernelEx(&config, Func, parameter);
}
```
## 10.38.发射边界[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-bounds "这个标题的永久链接")

正如在[多处理器级别](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multiprocessor-level)中详细讨论的那样，内核使用的寄存器越少，多处理器上可能存在的线程和线程块就越多，这可以提高性能。

因此，编译器使用启发式方法来最大限度地减少寄存器的使用，同时将寄存器溢出（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)）和指令计数保持在最低限度。应用程序可以选择通过在`__global__`函数定义中使用`__launch_bounds__()`限定符指定的启动边界的形式向编译器提供附加信息来帮助这些启发式方法：

__global__ void
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
MyKernel(...)
{
    ...
}

- `maxThreadsPerBlock`指定应用程序将启动`MyKernel()`的每个块的最大线程数；它编译为`.maxntid`指令。
    
- `minBlocksPerMultiprocessor`是可选的，并指定每个多处理器所需的最小驻留块数；它编译为`.minnctapersm`指令。
    
- `maxBlocksPerCluster`是可选的，并指定每个集群所需的最大线程块数，应用程序将启动`MyKernel()`它编译为`.maxclusterrank`指令。
    

如果指定了启动边界，编译器首先从它们中推导出内核应该使用的寄存器数量的上限_L_，以确保`maxThreadsPerBlock`线程的`minBlocksPerMultiprocessor`块（如果没有指定`minBlocksPerMultiprocessor`，则为单个块）可以驻留在多处理器上（请参阅[硬件多线程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading)，以了理内核使用的寄存器数量与每个块分配的寄存器数量之间的关系）。然后，编译器以以下方式优化寄存器的使用：

- 如果初始寄存器使用率高于_L_，编译器将进一步减少它，直到它小于或等于_L_，通常以牺牲更多的本地内存使用量和/或更高的指令数量为代价；
    
- 如果初始寄存器使用率低于_L_
    
    - 如果指定了`maxThreadsPerBlock`，而未指定`minBlocksPerMultiprocessor`，编译器将使用`maxThreadsPerBlock`来确定`n`和`n+1`驻留块之间转换的寄存器使用阈值（即，当少使用一个寄存器为额外的驻块留块留出空间时，如[多处理器级别](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multiprocessor-level)示例），然后应用与未指定启动边界时类似的启发式方法；
        
    - 如果同时指定了`minBlocksPerMultiprocessor`和`maxThreadsPerBlock`，编译器可能会将寄存器使用量增加到_L_，以减少指令数量并更好地隐藏单线程指令延迟。
        

如果内核执行的每个块线程数多于其启动绑定的`maxThreadsPerBlock`，则内核将无法启动。

如果每个集群执行的线程块比其启动绑定的`maxBlocksPerCluster`多，则内核将无法启动。

CUDA内核所需的每线程资源可能会以不需要的方式限制最大块大小。为了保持与未来硬件和工具包的向向兼容性，并确保至少一个线程块可以在SM上运行，开发人员应包含单个参数`__launch_bounds__(maxThreadsPerBlock)`该参数指定了内核将要启动的最大块大小。如果不这样做，可能会导致“启动资源请求太多”错误。在某些情况下，提供`__launch_bounds__(maxThreadsPerBlock,minBlocksPerMultiprocessor)`的两个参数版本可以提高性能。正确的值forminBlocksPerMultiprocessor应该使用详细的每个内核分析来确定。

给定内核的最佳启动边界通常会因主要架构修订版而异。下面的示例代码显示了如何使用[应用程序兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility)中引入的`__CUDA_ARCH__`宏在设备代码中通常如何处理此操作。

#define THREADS_PER_BLOCK          256
#if __CUDA_ARCH__ >= 200
    #define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
    #define MY_KERNEL_MIN_BLOCKS   3
#else
    #define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
    #define MY_KERNEL_MIN_BLOCKS   2
#endif

// Device code
__global__ void
__launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
MyKernel(...)
{
    ...
}

在使用每个块的最大线程数（指定为`__launch_bounds__()`的第一个参数）调用`MyKernel`的常见情况下，使用`MY_KERNEL_MAX_THREADS`作为执行配置中每个块的线程数是很诱人的：

// Host code
MyKernel<<<blocksPerGrid, MY_KERNEL_MAX_THREADS>>>(...);

然而，这不会起作用，因为`__CUDA_ARCH__`在主机代码中未定义，如[应用程序兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility)中提到，因此即使`__CUDA_ARCH__`大于或等于200，`MyKernel`也会以每个块256个线程启动。相反，应该确定每个块的线程数：

- 例如，在编译时使用不依赖于`__CUDA_ARCH__`的宏
    
    // Host code
    MyKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(...);
    
- 或者在运行时基于计算能力
    
    // Host code
    cudaGetDeviceProperties(&deviceProp, device);
    int threadsPerBlock =
              (deviceProp.major >= 2 ?
                        2 * THREADS_PER_BLOCK : THREADS_PER_BLOCK);
    MyKernel<<<blocksPerGrid, threadsPerBlock>>>(...);
    

注册使用情况由`--ptxas-options=-v`编译器选项报告。常驻块的数量可以从CUDA分析器报告的占用率中得出（有用定义，请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)）。

## 10.39.每个线程的最大寄存器数量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximum-number-of-registers-per-thread "这个标题的永久链接")

为了提供低级性能调整的机制，CUDA C++提供了`__maxnreg__()`函数限定符，将性能调整信息传递给后端优化编译器。`__maxnreg__()`限定符指定了分配给线程块中单个线程的最大寄存器数。在`__global__`函数的定义中：

__global__ void
__maxnreg__(maxNumberRegistersPerThread)
MyKernel(...)
{
    ...
}

- `maxNumberRegistersPerThread`指定在kernelMyKernel`MyKernel()`的线程块中分配给单个线程的最大寄存器数；它编译为`.maxnreg`指令。
    

`__launch_bounds__()`和`__maxnreg__()`限定符不能应用于同一内核。

也可以使用`maxrregcount`编译器选项控制文件中所有`__global__`函数的寄存器使用情况。对于具有`__maxnreg__`限定符的函数，`maxrregcount`的值会被忽略。

## 10.40. #pragma展开[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pragma-unroll "这个标题的永久链接")

默认情况下，编译器以已知的行程计数展开小循环。然而，`#pragmaunroll`指令可用于控制任何给定循环的展开。它必须紧挨着循环，并且仅适用于该循环。可选的后面是一个整数常数表达式（ICE）[6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn13)。如果没有ICE，如果它的行程计数是恒定的，循环将被完全展开。如果ICE评估为1，编译器将不会展开循环。如果ICE评估为非正整数或大于整数数据类型可表示的最大值的整数，则将忽略实用法。

示例：
```c++
struct S1_t { static const int value = 4; };
template <int X, typename T2>
__device__ void foo(int *p1, int *p2) {

// no argument specified, loop will be completely unrolled
#pragma unroll
for (int i = 0; i < 12; ++i)
  p1[i] += p2[i]*2;

// unroll value = 8
#pragma unroll (X+1)
for (int i = 0; i < 12; ++i)
  p1[i] += p2[i]*4;

// unroll value = 1, loop unrolling disabled
#pragma unroll 1
for (int i = 0; i < 12; ++i)
  p1[i] += p2[i]*8;

// unroll value = 4
#pragma unroll (T2::value)
for (int i = 0; i < 12; ++i)
  p1[i] += p2[i]*16;
}

__global__ void bar(int *p1, int *p2) {
foo<7, S1_t>(p1, p2);
}
```
## 10.41.SIMD视频说明[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simd-video-instructions "这个标题的永久链接")

PTX ISA版本3.0包括SIMD（单指令，多数据）视频指令，这些指令在16位值对和8位值的四元上操作。这些在计算能力3.0的设备上可用。

SIMD视频说明是：

- 瓦德2，瓦德4
    
- vsub2，vsub4
    
- vavrg2，vavrg4
    
- vabsdiff2，vabsdiff4
    
- vmin2，vmin4
    
- vmax2，vmax4
    
- vset2，vset4
    

PTX指令，如SIMD视频指令，可以通过汇编程序、`asm()`语句包含在CUDA程序中。

`asm()`语句的基本语法是：

asm("template-string" : "constraint"(output) : "constraint"(input)"));

An example of using the `vabsdiff4` PTX instruction is:

asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;": "=r" (result):"r" (A), "r" (B), "r" (C));

这使用`vabsdiff4`指令来计算绝对差分的整数四字节SIMD和。以SIMD方式计算未符号整数A和B的每个字节的绝对差值。指定了可选的累积操作（`.add`）来求和这些差异。

有关在代码中使用汇编语句的详细信息，请参阅文档“在CUDA中使用内联PTX汇编”。有关您正在使用的PTX版本的PTX说明的详细信息，请参阅PTX ISA文档（例如“并行线程执行ISA版本3.0”）。

## 10.42.诊断Pragmas[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#diagnostic-pragmas "这个标题的永久链接")

以下实用程序可用于控制发出给定诊断消息时使用的错误严重程度。
```c++
#pragma nv_diag_suppress
#pragma nv_diag_warning
#pragma nv_diag_error
#pragma nv_diag_default
#pragma nv_diag_once
```
这些词的用法有以下形式：

#pragma nv_diag_xxx error_number, error_number ...

The diagnostic affected is specified using an error number showed in a warning message. Any diagnostic may be overridden to be an error, but only warnings may have their severity suppressed or be restored to a warning after being promoted to an error. The `nv_diag_default` pragma is used to return the severity of a diagnostic to the one that was in effect before any pragmas were issued (i.e., the normal severity of the message as modified by any command-line options). The following example suppresses the `"declared but never referenced"` warning on the declaration of `foo`:

#pragma nv_diag_suppress 177
void foo()
{
  int i=0;
}
#pragma nv_diag_default 177
void bar()
{
  int i=0;
}

以下实用程序可用于保存和恢复当前的诊断实用程序状态：

#pragma nv_diagnostic push
#pragma nv_diagnostic pop

示例：

#pragma nv_diagnostic push
#pragma nv_diag_suppress 177
void foo()
{
  int i=0;
}
#pragma nv_diagnostic pop
void bar()
{
  int i=0;
}

请注意，pragmas只影响nvcc CUDA前端编译器；它们对主机编译器没有影响。

删除通知：没有`nv_`前缀的诊断pragmas支持从CUDA 12.0中删除，如果pragmas在设备代码中，将发出`devicecodeinunrecognized#pragma`警告，否则它们将被传递给主机编译器。如果它们用于CUDA代码，请使用带有`nv_`前缀的pragmas。

[4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id144)

当包含的__host__函数是模板时，nvcc目前在某些情况下可能无法发出诊断消息；这种行为可能会在未来发生变化。

[5](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id170)

如果主机编译器不支持函数，其目的是防止主机编译器遇到对函数的调用。

6（1，2，3）

关于积分常数表达式的定义，请参阅C++标准。

## 10.43.定制ABI Pragmas[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#custom-abi-pragmas "这个标题的永久链接")

The `#pragma nv_abi` directive enables applications compiled in separate compilation mode to achieve performance similar to that of whole program compilation.

使用此语法的语法如下，其中ICE指的是任何整数常数表达式（ICE）：6.

#pragma nv_abi preserve_n_data(ICE) preserve_n_control(ICE)

Note, the arguments that follow `#pragma nv_abi` are optional and can be provided in any order; however, at least one argument is required.

`preserve_n`参数对函数调用期间保留的寄存器数量设置了限制：

- `preserve_n_data(ICE)`限制数据寄存器的数量，以及
    
- `preserve_n_control(ICE)`限制控制寄存器的数量。
    

`#pragma nv_abi`可以紧挨着设备功能声明或定义。或者，它可以直接放在设备函数内的C++表达式语句中的间接函数调用之前。请注意，支持对自由函数的间接函数调用，但不支持通过函数参数引用或类成员函数进行间接调用。

When the pragma is applied to a device function declaration or definition, it modifies the custom ABI properties for any calls to that function. When placed at an indirect function call site, the pragma affects the ABI properties for that indirect function call. The key point is that unlike direct function calls, where you can place the pragma before a function declaration or definition, `#pragma nv_abi` only affects indirect function calls when the pragma is placed before a call site.

如下例所示，我们有两个设备函数，`foo()`和`bar()`在本例中，pragma被放置在函数指针fptr的调用站点之前，以修改间接函数调用的ABI属性。请注意，将pragma放在直接呼叫之前不会影响呼叫的ABI属性。要更改直接函数调用的ABI属性，必须将pragma放在函数声明或定义之前。

__device__ int foo()
{
  int value{0};
  ...
  return value;
}

__device__ int bar()
{
  int value{0};
  ...
  return value;
}

__device__ void baz()
{
  int result{0};
  int (*fptr)() = foo;  // function pointer

  #pragma nv_abi preserve_n_data(16) preserve_n_control(8)
  result = fptr();      // The pragma affects the indirect call to foo() via fptr

  #pragma nv_abi preserve_n_data(16) preserve_n_control(8)
  result = (*fptr)();   // Alternate syntax for the indirect call to foo()

  #pragma nv_abi preserve_n_data(16) preserve_n_control(8)
  result += bar();      // The pragma does NOT affect the direct call to bar()
}

如以下示例所示，要修改直接函数调用，您必须将实用程序应用于函数声明或定义。

#pragma nv_abi preserve_n_data(16)
__device__ void foo();

请注意，如果函数声明的语用参数及其相应定义不匹配，则程序就形成不良。

## 10.44.CUDA C++内存模型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-memory-model "这个标题的永久链接")

CUDA C++内存模型扩展了[CUDA C++内存模型文档](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html)中记录的ISO C++内存模型。

## 10.45.CUDA C++执行模型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-execution-model "这个标题的永久链接")

CUDA C++执行模型扩展了[CUDA C++执行模型文档](https://nvidia.github.io/cccl/libcudacxx/extended_api/execution_model.html)中记录的ISO C++执行模型。

# 11.合作团体[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups "这个标题的永久链接")

## 11.1.介绍[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#introduction-cg "这个标题的永久链接")

合作组是CUDA编程模型的扩展，在CUDA 9中引入，用于组织通信线程组。合作组允许开发人员表达线程通信的粒度，帮助他们表达更丰富、更高效的并行分解。

历史上，CUDA编程模型为同步合作线程提供了单一的简单结构：线程块所有线程的屏障，如`__syncthreads()`内在函数实现的那样。然而，程序员希望以其他粒度定义和同步线程组，以实现更高的性能、设计灵活性和以“集体”全组功能接口的形式重复使用软件。为了表达更广泛的并行交互模式，许多以性能为导向的程序员诉诸于编写自己的临时和不安全的原语，以同步单个经编中或跨单个GPU上运行的线程块集的线程。虽然所取得的性能改进通常很有价值，但这导致了不断增长的易碎代码集合，随着时间的推移和跨GPU几代编写、调整和维护成本很高。合作小组通过提供一个安全和面向未来的机制来实现高性能代码来解决这个问题。

## 11.2.合作团体的新内容[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#what-s-new-in-cooperative-groups "这个标题的永久链接")

### 11.2.1.CUDA 13.0[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-13-0 "这个标题的永久链接")

- `multi_grid_group`被移除了。
    

### 11.2.2.CUDA 12.2[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-12-2 "这个标题的永久链接")

- `barrier_arrive`并为[grid_group](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#grid-group-cg)和[thread_block](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-group-cg)添加了`barrier_wait`成员函数。API的描述可以[在这里](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-sync)找到。
    

### 11.2.3.CUDA 12.1[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-12-1 "这个标题的永久链接")

- 添加了[invoke_one和invoke_one_broadcast](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#invoke-one-and-invoke-one-broadcast) API。
    

### 11.2.4.库达12.0[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-12-0 "这个标题的永久链接")

- 以下实验性API现已移至主命名空间：
    
    - 在CUDA 11.7中添加了异步减少和扫描更新
        
    - `thread_block_tile`大于在CUDA 11.1中添加的32
        
- 为了在计算能力8.0或更高版本上创建这些大型图块，不再需要使用`block_tile_memory`对象提供内存。
    

## 11.3.编程模型概念[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model-concept "这个标题的永久链接")

合作组编程模型描述了CUDA线程块内和跨线程的同步模式。它既为应用程序提供了定义自己线程组的手段，也为同步它们提供了接口。它还提供了新的启动API，这些API强制执行某些限制，因此可以保证同步工作。这些原语使CUDA内的合作并行性新模式成为可能，包括生产者-消费者并行性、机会性并行性和整个网格的全球同步性。

合作小组编程模型由以下要素组成：

- 表示合作线程组的数据类型；
    
- 获取CUDA启动API定义的隐式组的操作（例如，线程块）；
    
- 将现有组划分为新组的集体；
    
- 数据移动和操作的集体算法（例如memcpy_async、reduce、scan）；
    
- 同步组内所有线程的操作；
    
- 检查组属性的操作；
    
- 暴露低级、特定于群体和经常硬件加速操作的集体。
    

合作组的主要概念是对象命名作为其一部分的线程集。组作为一流程序对象的表达方式改善了软件的构成，因为集合函数可以接收一个表示参与线程组的显式对象。这个对象还使程序员的意图明确，这消除了不健全的架构假设，这些假设导致代码变得脆弱、编译器优化的不良限制，以及与新一代GPU的更好兼容性。

要编写高效的代码，最好使用专用组（转为通用会损失很多编译时间优化），并通过引用这些组对象传递给打算以某种合作方式使用这些线程的函数。

合作团体需要CUDA 9.0或更高版本。要使用合作组，请包括标题文件：

// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
// Optionally include for inclusive_scan() and exclusive_scan() collectives
#include <cooperative_groups/scan.h>

并使用合作组命名空间：

using namespace cooperative_groups;
// Alternatively use an alias to avoid polluting the namespace with collective algorithms
namespace cg = cooperative_groups;

代码可以使用nvcc以正常方式编译，但是，如果您希望使用memcpy_async、减少或扫描功能，并且您的主机编译器的默认方言不是C++11或更高，那么您必须将`--std=c++11`添加到命令行中。

### 11.3.1.构图示例[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#composition-example "这个标题的永久链接")

为了说明组的概念，这个例子试图执行整个块的总和减少。以前，在编写此代码时，实现上存在隐藏的约束：

__device__ int sum(int *x, int n) {
    // ...
    __syncthreads();
    return total;
}

__global__ void parallel_kernel(float *x) {
    // ...
    // Entire thread block must call sum
    sum(x, n);
}

线程块中的所有线程必须到达`__syncthreads()`屏障，但是，这个约束对可能想要使用`sum(…)`的开发人员是隐藏的。对于合作团体，更好的写作方式是：

__device__ int sum(const thread_block& g, int *x, int n) {
    // ...
    g.sync()
    return total;
}

__global__ void parallel_kernel(...) {
    // ...
    // Entire thread block must call sum
    thread_block tb = this_thread_block();
    sum(tb, x, n);
    // ...
}

## 11.4.组类型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#group-types "这个标题的永久链接")

### 11.4.1.隐式组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#implicit-groups "这个标题的永久链接")

隐式组表示内核的启动配置。无论您的内核是如何编写的，它总是有一组线程、块和块尺寸，单个网格和网格尺寸。此外，如果使用多设备合作启动API，它可以有多个网格（每个设备单个网格）。这些组为分解为更精细的颗粒组提供了一个起点，这些组通常是硬件加速的，并且对开发人员正在解决的问题更加专业化。

尽管您可以在代码的任何地方创建一个隐式组，但这样做很危险。为隐式组创建句柄是一种集体操作——组中的所有线程都必须参与。如果该组是在并非所有线程都能到达的条件分支中创建的，这可能会导致死锁或数据损坏。出于这个原因，建议您提前为隐式组创建一个句柄（尽早，在任何分支发生之前），并在整个内核中使用该句柄。出于同样的原因，必须在声明时初始化组句柄（没有默认构造函数），不建议复制它们。

#### 11.4.1.1.线程块组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-group "这个标题的永久链接")

任何CUDA程序员都已经熟悉某一组线程：线程块。合作组扩展引入了一个新的数据类型thread_block，以在内核中显式表示此概念。

`class thread_block;`

通过以下方式构建：

thread_block g = this_thread_block();

**公共成员职能：**

`static void sync()`：同步组中命名的线程，相当于`g.barrier_wait(g.barrier_arrive())`

`thread_block::arrival_token barrier_arrive()`：到达thread_block屏障，返回一个需要传递到`barrier_wait()`的令牌。更多详情[在这里](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-sync)

`void barrier_wait(thread_block::arrival_token&& t)`：在`thread_block`屏障上等待，将从`barrier_arrive()`返回的到达令牌作为rvalue引用。更多详情[在这里](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-sync)

`static unsigned int thread_rank()`：[0，num_threads）内的调用线程的排名

`static dim3 group_index()`：发射网格内块的三维指数

`static dim3 thread_index()`：启动块内线程的三维索引

`static dim3 dim_threads()`：以线程为单位的启动块的尺寸

`static unsigned int num_threads()`：组中的线程总数

传统成员功能（别名）：

`static unsigned int size()`：组中的线程总数（`num_threads()`的别名）

`static dim3 group_dim()`：启动块的尺寸（`dim_threads()`的别名）

**示例：**

/// Loading an integer from global into shared memory
__global__ void kernel(int *globalInput) {
    __shared__ int x;
    thread_block g = this_thread_block();
    // Choose a leader in the thread block
    if (g.thread_rank() == 0) {
        // load from global into shared for all threads to work with
        x = (*globalInput);
    }
    // After loading data into shared memory, you want to synchronize
    // if all threads in your thread block need to see it
    g.sync(); // equivalent to __syncthreads();
}

**注意：**组中的所有线程都必须参与集体操作，否则行为是未定义的。

**相关：**`thread_block`数据类型来自更通用的`thread_group`数据类型，可用于表示更广泛的组类。

#### 11.4.1.2.集群组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-group "这个标题的永久链接")

此组对象表示在单个集群中启动的所有线程。参考线[程块集群](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters)。API适用于所有具有计算能力9.0+的硬件。在这种情况下，当启动非集群网格时，API假设一个1x1x1集群。

`class cluster_group;`

通过以下方式构建：

cluster_group g = this_cluster();

**公共成员职能：**

`static void sync()`：同步组中命名的线程，相当于`g.barrier_wait(g.barrier_arrive())`

`static cluster_group::arrival_token barrier_arrive()`：到达集群屏障，返回一个需要传递到`barrier_wait()`的令牌。更多详情[在这里](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-sync)

`static void barrier_wait(cluster_group::arrival_token&& t)`：在集群屏障上等待，将从`barrier_arrive()`返回的到达令牌作为rvalue引用。更多详情[在这里](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-sync)

`static unsigned int thread_rank()`：[0，num_threads）内的调用线程的排名

`static unsigned int block_rank()`：调用块在[0，num_blocks）内的排名

`static unsigned int num_threads()`：组中的线程总数

`static unsigned int num_blocks()`：组中的区块总数

`static dim3 dim_threads()`：以线程为单位的启动集群的尺寸

`static dim3 dim_blocks()`：以块为单位的发射集群的尺寸

`static dim3 block_index()`：启动集群内调用块的三维索引

`static unsigned int query_shared_rank(const void *addr)`：获取共享内存地址所属的块秩

`static T* map_shared_rank(T *addr, int rank)`：获取集群中另一个块的共享内存变量的地址

传统成员功能（别名）：

`static unsigned int size()`：组中的线程总数（`num_threads()`的别名）

#### 11.4.1.3.网格组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#grid-group "这个标题的永久链接")

这个组对象表示在单个网格中启动的所有线程。`sync()`以外的API始终可用，但要能够在网格上同步，您需要使用合作启动API。

`class grid_group;`

通过以下方式构建：

grid_group g = this_grid();

**公共成员职能：**

`bool is_valid() const`：返回grid_group是否可以同步

`void sync() const`：同步组中命名的线程，相当于`g.barrier_wait(g.barrier_arrive())`

`grid_group::arrival_token barrier_arrive()`：到达网格屏障，返回一个需要传递到`barrier_wait()`的令牌。更多详情请见

`void barrier_wait(grid_group::arrival_token&& t)`：在网格屏障上等待，将从`barrier_arrive()`返回的到达令牌作为rvalue引用。更多详情[在这里](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-sync)

`static unsigned long long thread_rank()`：[0，num_threads）内的调用线程的排名

`static unsigned long long block_rank()`：调用块在[0，num_blocks）内的排名

`static unsigned long long cluster_rank()`：调用集群在[0，num_clusters）中的排名

`static unsigned long long num_threads()`：组中的线程总数

`static unsigned long long num_blocks()`：组中的区块总数

`static unsigned long long num_clusters()`：组中的集群总数

`static dim3 dim_blocks()`：以块为单位的发射网格尺寸

`static dim3 dim_clusters()`：以集群为单位的发射网格尺寸

`static dim3 block_index()`：发射网格内块的三维指数

`static dim3 cluster_index()`：启动网格内集群的三维指数

传统成员功能（别名）：

`static unsigned long long size()`：组中的线程总数（`num_threads()`的别名）

`static dim3 group_dim()`：启动网格的尺寸（`dim_blocks()`的别名）

### 11.4.2.显式组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#explicit-groups "这个标题的永久链接")

#### 11.4.2.1.螺纹块瓷砖[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-tile "这个标题的永久链接")

平铺组的模板版本，其中模板参数用于指定平铺的大小——在编译时已知这一点，有可能实现更优化的执行。

template <unsigned int Size, typename ParentT = void>
class thread_block_tile;

通过以下方式构建：

template <unsigned int Size, typename ParentT>
_CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)

`Size`必须是2的次比，小于或等于1024。备注部分介绍了在具有计算能力 7.5 或更低的硬件上创建大小大于 32 的瓷砖所需的额外步骤。

`ParentT`是分割此組的父类型。它是自动推断的，但无效值将将此信息存储在组句柄中，而不是类型中。

**公共成员职能：**

`void sync() const`：同步组中命名的线程

`unsigned long long num_threads() const`：组中的线程总数

`unsigned long long thread_rank() const`：[0，num_threads）内的调用线程的排名

`unsigned long long meta_group_size() const`：返回父组分区时创建的组数。

`unsigned long long meta_group_rank() const`：从父组中分区的瓷砖集中组的线性排名（由meta_group_size限制）

`T shfl(T var, unsigned int src_rank) const`：参考[Warp Shuffle函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)，**注意：对于大于32的大小，组中的所有线程必须指定相同的src_rank，否则行为是未定义的。**

`T shfl_up(T var, int delta) const`：参考[Warp Shuffle功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)，仅适用于小于或等于32的尺寸。

`T shfl_down(T var, int delta) const`：参考[Warp Shuffle功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)，仅适用于小于或等于32的尺寸。

`T shfl_xor(T var, int delta) const`：参考[Warp Shuffle功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)，仅适用于小于或等于32的尺寸。

`int any(int predicate) const`：参考[Warp Vote函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)

`int all(int predicate) const`：参考[Warp Vote函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)

`unsigned int ballot(int predicate) const`：参考[Warp Vote Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)，仅适用于小于或等于32的尺寸。

`unsigned int match_any(T val) const`：参考[Warp Match功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-match-functions)，仅适用于小于或等于32的尺寸。

`unsigned int match_all(T val, int &pred) const`：参考[Warp Match功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-match-functions)，仅适用于小于或等于32的尺寸。

传统成员功能（别名）：

`unsigned long long size() const`：组中的线程总数（`num_threads()`的别名）

**备注：**

- `thread_block_tile`这里使用模板数据结构，组的大小作为模板参数而不是参数传递到`tiled_partition`调用。
    
- `shfl, shfl_up, shfl_down, and shfl_xor`当使用C++11或更高版本编译时，函数接受任何类型的对象。这意味着，只要非整体类型满足以下约束，就可以洗牌：
    
    - 符合可复制条件，即，`is_trivially_copyable<T>::value == true`
        
    - `sizeof(T) <= 32` for tile sizes lower or equal 32, `sizeof(T) <= 8` for larger tiles
        
- 在具有计算能力7.5或更低的硬件上，尺寸大于32的瓷砖需要为它们保留少量内存。这可以使用`cooperative_groups::block_tile_memory`结构模板来完成，该模板必须位于共享或全局内存中。
    ```c++
    template <unsigned int MaxBlockSize = 1024>
    struct block_tile_memory;
    ```
    `MaxBlockSize`指定当前线程块中线程的最大数量。此参数可用于最大限度地减少仅以较小线程计数启动的内核中`block_tile_memory`的共享内存使用。
    
    然后，这个`block_tile_memory`需要传递到`cooperative_groups::this_thread_block`，允许将生成的`thread_block`划分为大于32的瓷砖。接受`block_tile_memory`参数的`this_thread_block`过载是一个集体操作，必须与`thread_block`中的所有线程一起调用。
    
    `block_tile_memory`可以在具有计算能力8.0或更高版本的硬件上使用，以便能够针对多个不同的计算能力编写一个源。在不需要的情况下，在共享内存中实例化时，不应消耗内存。
    

**示例：**
```c++
/// The following code will create two sets of tiled groups, of size 32 and 4 respectively:
/// The latter has the provenance encoded in the type, while the first stores it in the handle
thread_block block = this_thread_block();
thread_block_tile<32> tile32 = tiled_partition<32>(block);
thread_block_tile<4, thread_block> tile4 = tiled_partition<4>(block);

/// The following code will create tiles of size 128 on all Compute Capabilities.
/// block_tile_memory can be omitted on Compute Capability 8.0 or higher.
__global__ void kernel(...) {
    // reserve shared memory for thread_block_tile usage,
    //   specify that block size will be at most 256 threads.
    __shared__ block_tile_memory<256> shared;
    thread_block thb = this_thread_block(shared);

    // Create tiles with 128 threads.
    auto tile = tiled_partition<128>(thb);

    // ...
}
```
##### 11.4.2.1.1.扭曲同步代码模式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-synchronous-code-pattern "这个标题的永久链接")

开发人员可能有经编同步代码，他们之前对经编大小做出了隐性假设，并将围绕该数字进行编码。现在这需要明确指定。

__global__ void cooperative_kernel(...) {
    // obtain default "current thread block" group
    thread_block my_block = this_thread_block();

    // subdivide into 32-thread, tiled subgroups
    // Tiled subgroups evenly partition a parent group into
    // adjacent sets of threads - in this case each one warp in size
    auto my_tile = tiled_partition<32>(my_block);

    // This operation will be performed by only the
    // first 32-thread tile of each block
    if (my_tile.meta_group_rank() == 0) {
        // ...
        my_tile.sync();
    }
}

##### 11.4.2.1.2.单线程组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#single-thread-group "这个标题的永久链接")

代表当前线程的组可以从`this_thread`函数获得：

thread_block_tile<1> this_thread();

The following `memcpy_async` API uses a `thread_group`, to copy an int element from source to destination:

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

cooperative_groups::memcpy_async(cooperative_groups::this_thread(), dest, src, sizeof(int));

使用`this_thread`执行异步副本的更详细示例可以在[使用cuda::pipeline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#with-memcpy-async-pipeline-pattern-single)的[单阶段异步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#with-memcpy-async-pipeline-pattern-single)和[使用cuda::pipeline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#with-memcpy-async-pipeline-pattern-multi)的[多阶段异步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#with-memcpy-async-pipeline-pattern-multi)中找到。

#### 11.4.2.2.合并的群体[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#coalesced-groups "这个标题的永久链接")

在CUDA的SIMT架构中，在硬件级别，多处理器以32组称为warps的线程形式执行线程。如果应用程序代码中存在一个数据依赖的条件分支，使经编中的线程发散，那么经编会连续执行每个分支禁用不在该路径上的线程。路径上仍然活跃的线程被称为合并。合作组具有发现和创建包含所有合并线程的组的功能。

通过`coalesced_threads()`构建组句柄是机会主义的。它在该时间点返回一组活动线程，并且不保证哪些线程被返回（只要它们处于活动状态），也不保证它们在整个执行过程中会保持凝聚（它们将被重新组合在一起执行，但之后可能会再次发散）。

`class coalesced_group;`

通过以下方式构建：

coalesced_group active = coalesced_threads();

**公共成员职能：**

`void sync() const`：同步组中命名的线程

`unsigned long long num_threads() const`：组中的线程总数

`unsigned long long thread_rank() const`：[0，num_threads）内的调用线程的排名

`unsigned long long meta_group_size() const`：返回父组分区时创建的组数。如果此组是通过查询活动线程集创建的，例如`coalesced_threads()`则`meta_group_size()`的值将是1。

`unsigned long long meta_group_rank() const`：从父组分区的瓷砖集中组的组的线性排名（由meta_group_size限制）。如果此组是通过查询一组活动线程创建的，例如`coalesced_threads()`则`meta_group_rank()`的值将始终为0。

`T shfl(T var, unsigned int src_rank) const`：参考[Warp Shuffle功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)

`T shfl_up(T var, int delta) const`：参考[Warp Shuffle功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)

`T shfl_down(T var, int delta) const`：参考[Warp Shuffle功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)

`int any(int predicate) const`：参考[Warp Vote函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)

`int all(int predicate) const`：参考[Warp Vote函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)

`unsigned int ballot(int predicate) const`：参考[Warp Vote函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions)

`unsigned int match_any(T val) const`：参考[Warp Match功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-match-functions)

`unsigned int match_all(T val, int &pred) const`：参考[Warp Match功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-match-functions)

传统成员功能（别名）：

`unsigned long long size() const`：组中的线程总数（`num_threads()`的别名）

**备注：**

`shfl, shfl_up, and shfl_down`当使用C++11或更高版本编译时，函数接受任何类型的对象。这意味着，只要非整体类型满足以下约束，就可以洗牌：

- 符合可复制条件，即`is_trivially_copyable<T>::value == true`
    
- `sizeof(T) <= 32`
    

**示例：**

/// Consider a situation whereby there is a branch in the
/// code in which only the 2nd, 4th and 8th threads in each warp are
/// active. The coalesced_threads() call, placed in that branch, will create (for each
/// warp) a group, active, that has three threads (with
/// ranks 0-2 inclusive).
__global__ void kernel(int *globalInput) {
    // Lets say globalInput says that threads 2, 4, 8 should handle the data
    if (threadIdx.x == *globalInput) {
        coalesced_group active = coalesced_threads();
        // active contains 0-2 inclusive
        active.sync();
    }
}

##### 11.4.2.2.1.发现模式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#discovery-pattern "这个标题的永久链接")

通常，开发人员需要使用当前的活动线程集。对存在的线程没有做出任何假设，而是开发人员使用恰好存在的线程。这在以下“在经编中跨线程的聚合原子增量”示例中可见（使用正确的CUDA 9.0内在集编写）：

{
    unsigned int writemask = __activemask();
    unsigned int total = __popc(writemask);
    unsigned int prefix = __popc(writemask & __lanemask_lt());
    // Find the lowest-numbered active lane
    int elected_lane = __ffs(writemask) - 1;
    int base_offset = 0;
    if (prefix == 0) {
        base_offset = atomicAdd(p, total);
    }
    base_offset = __shfl_sync(writemask, base_offset, elected_lane);
    int thread_offset = prefix + base_offset;
    return thread_offset;
}

这可以用合作小组重写如下：

{
    cg::coalesced_group g = cg::coalesced_threads();
    int prev;
    if (g.thread_rank() == 0) {
        prev = atomicAdd(p, g.num_threads());
    }
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}

## 11.5.分组[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#group-partitioning "这个标题的永久链接")

### 11.5.1.`tiled_partition`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tiled-partition "这个标题的永久链接")

template <unsigned int Size, typename ParentT>
thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);

thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);

`tiled_partition`方法是一种集体操作，将父组划分为一维、行主要、子组的平铺。将创建总共（（size（parent）/tilesz）子组，因此父组大小必须被`Size`整除。允许的父组是`thread_block`或`thread_block_tile`。

实现可能会导致调用线程等待父组的所有成员调用该操作，然后再恢复执行。功能仅限于原生硬件尺寸，1/2/4/8/16/32，`cg::size(parent)`必须大于`Size`参数。`tiled_partition`的模板版本也支持64/128/256/512尺寸，但在计算能力7.5或更低版本上需要一些额外的步骤，有关详细信息，请参阅线[程块瓷砖](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-tile-group-cg)。

**Codegen要求：**计算能力5.0最低，大于32的C++11

**示例：**

/// The following code will create a 32-thread tile
thread_block block = this_thread_block();
thread_block_tile<32> tile32 = tiled_partition<32>(block);

我们可以将这些组分成更小的组，每个组大小为4个线程：

auto tile4 = tiled_partition<4>(tile32);
// or using a general group
// thread_group tile4 = tiled_partition(tile32, 4);

例如，如果我们要包含以下代码行：

if (tile4.thread_rank()==0) printf("Hello from tile4 rank 0\n");

然后，该语句将由块中的每四个线程打印：每个`tile4`组中排名为0的线程，对应于`block`中排名为0、4、8、12等的线程。

### 11.5.2.`labeled_partition`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#labeled-partition "这个标题的永久链接")
```c++
template <typename Label>
coalesced_group labeled_partition(const coalesced_group& g, Label label);

template <unsigned int Size, typename Label>
coalesced_group labeled_partition(const thread_block_tile<Size>& g, Label label);
```
`labeled_partition`方法是一种集体操作，将父组划分为线程合并的一维子组。实现将评估条件标签，并将具有相同标签值的线程分配到同一组中。

`Label`可以是任何整数类型。

实现可能会导致调用线程等待父组的所有成员调用该操作，然后再恢复执行。

**注意：**此功能仍在评估中，将来可能会略有变化。

**Codegen要求：**计算能力最低7.0，C++11

### 11.5.3.`binary_partition`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#binary-partition "这个标题的永久链接")

coalesced_group binary_partition(const coalesced_group& g, bool pred);
```c++
template <unsigned int Size>
coalesced_group binary_partition(const thread_block_tile<Size>& g, bool pred);
```
`binary_partition()`方法是一种集体操作，将父组划分为一维子组，其中线程合并。实现将评估一个谓词，并将具有相同值的线程分配到同一组中。这是`labeled_partition()`的一种特殊形式，其中标签只能是0或1。

实现可能会导致调用线程等待父组的所有成员调用该操作，然后再恢复执行。

**注意：**此功能仍在评估中，将来可能会略有变化。

**Codegen要求：**计算能力最低7.0，C++11

**示例：**
```c++
/// This example divides a 32-sized tile into a group with odd
/// numbers and a group with even numbers
_global__ void oddEven(int *inputArr) {
    auto block = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(block);

    // inputArr contains random integers
    int elem = inputArr[block.thread_rank()];
    // after this, tile32 is split into 2 groups,
    // a subtile where elem&1 is true and one where its false
    auto subtile = cg::binary_partition(tile32, (elem & 1));
}
```
## 11.6.团体集体[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#group-collectives "这个标题的永久链接")

合作组库提供了一组可以由一组线程执行的集体操作。这些操作需要指定组中的所有线程参与才能完成操作。除非参数描述中明确允许不同的值，否则组中的所有线程都需要为每个集体调用传递相应的参数相同的值。否则，呼叫的行为是未定义的。

### 11.6.1.同步[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization "这个标题的永久链接")

#### 11.6.1.1. `barrier_arrive`和`barrier_wait`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#barrier-arrive-and-barrier-wait "这个标题的永久链接")

T::arrival_token T::barrier_arrive();
void T::barrier_wait(T::arrival_token&&);

`barrier_arrive`和`barrier_wait`成员函数提供了一个类似于`cuda::barrier`的同步API[（阅读更多）。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier)合作组自动初始化组屏障，但由于这些操作的集体性质，到达和等待操作有额外的限制：组中的所有线程必须每个阶段到达并等待一次屏障。当与一个组一起调用`barrier_arrive`时，调用任何集体操作或该组的另一个障碍到达的结果是未定义的，直到`barrier_wait`调用观察到障碍阶段的完成。在其他线程调用`barrier_wait`之前，`barrier_wait`上被阻止的线程可能会从同步中释放，但只有在名为`barrier_arrive`的组中的所有线程之后。组类型T可以是任何[隐式组](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#group-types-implicit-cg)。这允许线程在到达后和等待同步解决之前进行独立工作，从而隐藏一些同步延迟。`barrier_arrive`返回一个`arrival_token`对象，该对象必须传递到相应的`barrier_wait`。令牌以这种方式消耗，不能用于另一个`barrier_wait`调用。

**用于同步跨集群共享内存初始化的barrier_arrive和barrier_wait示例：**
```c++
#include <cooperative_groups.h>

using namespace cooperative_groups;

void __device__ init_shared_data(const thread_block& block, int *data);
void __device__ local_processing(const thread_block& block);
void __device__ process_shared_data(const thread_block& block, int *data);

__global__ void cluster_kernel() {
    extern __shared__ int array[];
    auto cluster = this_cluster();
    auto block   = this_thread_block();

    // Use this thread block to initialize some shared state
    init_shared_data(block, &array[0]);

    auto token = cluster.barrier_arrive(); // Let other blocks know this block is running and data was initialized

    // Do some local processing to hide the synchronization latency
    local_processing(block);

    // Map data in shared memory from the next block in the cluster
    int *dsmem = cluster.map_shared_rank(&array[0], (cluster.block_rank() + 1) % cluster.num_blocks());

    // Make sure all other blocks in the cluster are running and initialized shared data before accessing dsmem
    cluster.barrier_wait(std::move(token));

    // Consume data in distributed shared memory
    process_shared_data(block, dsmem);
    cluster.sync();
}
```
#### 11.6.1.2.`sync`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sync "这个标题的永久链接")
```c++
static void T::sync();

template <typename T>
void sync(T& group);
```
`sync`同步组中命名的线程。组类型T可以是任何现有的组类型，因为它们都支持同步。它可作为每个组类型的成员函数，或作为将组作为参数的自由函数。如果该组是`grid_group`，则内核必须使用适当的合作启动API启动。相当于`T.barrier_wait(T.barrier_arrive())`

### 11.6.2.数据传输[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-transfer "这个标题的永久链接")

#### 11.6.2.1.`memcpy_async`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memcpy-async "这个标题的永久链接")

`memcpy_async` is a group-wide collective memcpy that utilizes hardware accelerated support for non-blocking memory transactions from global to shared memory. Given a set of threads named in the group, `memcpy_async` will move specified amount of bytes or elements of the input type through a single pipeline stage. Additionally for achieving best performance when using the `memcpy_async` API, an alignment of 16 bytes for both shared memory and global memory is required. It is important to note that while this is a memcpy in the general case, it is only asynchronous if the source is global memory and the destination is shared memory and both can be addressed with 16, 8, or 4 byte alignments. Asynchronously copied data should only be read following a call to wait or wait_prior which signals that the corresponding stage has completed moving data to shared memory.

Having to wait on all outstanding requests can lose some flexibility (but gain simplicity). In order to efficiently overlap data transfer and execution, its important to be able to kick off an **N+1**`memcpy_async` request while waiting on and operating on request **N**. To do so, use `memcpy_async` and wait on it using the collective stage-based `wait_prior` API. See [wait and wait_prior](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-wait) for more details.

用法1
```c++
template <typename TyGroup, typename TyElem, typename TyShape>
void memcpy_async(
  const TyGroup &group,
  TyElem *__restrict__ _dst,
  const TyElem *__restrict__ _src,
  const TyShape &shape
);
```
执行**“形状”字节**的副本。

用法2

template <typename TyGroup, typename TyElem, typename TyDstLayout, typename TySrcLayout>
void memcpy_async(
  const TyGroup &group,
  TyElem *__restrict__ dst,
  const TyDstLayout &dstLayout,
  const TyElem *__restrict__ src,
  const TySrcLayout &srcLayout
);

执行**``min(dstLayout, srcLayout)``元素**的副本。如果布局类型为`cuda::aligned_size_t<N>`，则两者必须指定相同的对齐方式。

**Errata** The `memcpy_async` API introduced in CUDA 11.1 with both src and dst input layouts, expects the layout to be provided in elements rather than bytes. The element type is inferred from `TyElem` and has the size `sizeof(TyElem)`. If `cuda::aligned_size_t<N>` type is used as the layout, the number of elements specified times `sizeof(TyElem)` must be a multiple of N and it is recommended to use `std::byte` or `char` as the element type.

If specified shape or layout of the copy is of type `cuda::aligned_size_t<N>`, alignment will be guaranteed to be at least `min(16, N)`. In that case both `dst` and `src` pointers need to be aligned to N bytes and the number of bytes copied needs to be a multiple of N.

**Codegen要求：**最低计算能力5.0，非同步计算能力8.0，C++11

`cooperative_groups/memcpy_async.h`标题需要包含。

**示例：**

/// This example streams elementsPerThreadBlock worth of data from global memory
/// into a limited sized shared memory (elementsInShared) block to operate on.
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel(int* global_data) {
    cg::thread_block tb = cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024;
    const size_t elementsInShared = 128;
    __shared__ int local_smem[elementsInShared];

    size_t copy_count;
    size_t index = 0;
    while (index < elementsPerThreadBlock) {
        cg::memcpy_async(tb, local_smem, elementsInShared, global_data + index, elementsPerThreadBlock - index);
        copy_count = min(elementsInShared, elementsPerThreadBlock - index);
        cg::wait(tb);
        // Work with local_smem
        index += copy_count;
    }
}

#### 11.6.2.2.`wait and wait_prior`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wait-and-wait-prior "这个标题的永久链接")
```c++
template <typename TyGroup>
void wait(TyGroup & group);

template <unsigned int NumStages, typename TyGroup>
void wait_prior(TyGroup & group);
```
`wait`和`wait_prior`集合允许等待memcpy_async副本完成。`wait`呼叫线程的块，直到所有之前的副本都完成。`wait_prior`允许最新的NumStages仍未完成，并等待之前的所有请求。因此，在请求的`N`副本总数中，它会等到第一个`N-NumStages`完成，最后一个`NumStages`可能仍在进行中。`wait`和`wait_prior`都会同步命名组。

**Codegen要求：**最低计算能力5.0，非同步计算能力8.0，C++11
cooperative_groups/memcpy_async.h 标题需要包含。

**示例：**
```c++
/// This example streams elementsPerThreadBlock worth of data from global memory
/// into a limited sized shared memory (elementsInShared) block to operate on in
/// multiple (two) stages. As stage N is kicked off, we can wait on and operate on stage N-1.
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

__global__ void kernel(int* global_data) {
    cg::thread_block tb = cg::this_thread_block();
    const size_t elementsPerThreadBlock = 16 * 1024 + 64;
    const size_t elementsInShared = 128;
    __align__(16) __shared__ int local_smem[2][elementsInShared];
    int stage = 0;
    // First kick off an extra request
    size_t copy_count = elementsInShared;
    size_t index = copy_count;
    cg::memcpy_async(tb, local_smem[stage], elementsInShared, global_data, elementsPerThreadBlock - index);
    while (index < elementsPerThreadBlock) {
        // Now we kick off the next request...
        cg::memcpy_async(tb, local_smem[stage ^ 1], elementsInShared, global_data + index, elementsPerThreadBlock - index);
        // ... but we wait on the one before it
        cg::wait_prior<1>(tb);

        // Its now available and we can work with local_smem[stage] here
        // (...)
        //

        // Calculate the amount fo data that was actually copied, for the next iteration.
        copy_count = min(elementsInShared, elementsPerThreadBlock - index);
        index += copy_count;

        // A cg::sync(tb) might be needed here depending on whether
        // the work done with local_smem[stage] can release threads to race ahead or not
        // Wrap to the next stage
        stage ^= 1;
    }
    cg::wait(tb);
    // The last local_smem[stage] can be handled here
}
```
### 11.6.3.数据操作[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-manipulation "这个标题的永久链接")

#### 11.6.3.1.`reduce`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#reduce "这个标题的永久链接")

template <typename TyGroup, typename TyArg, typename TyOp>
auto reduce(const TyGroup& group, TyArg&& val, TyOp&& op) -> decltype(op(val, val));

`reduce`对传递的组中命名的每个线程提供的数据执行还原操作。这利用了计算加、最小或最大运算和逻辑AND、OR或XOR的硬件加速（在计算80及以上设备上），以及在旧一代硬件上提供软件回退。只有4B类型被硬件加速。

`group`：有效的组类型是`coalesced_group`和`thread_block_tile`。

`val`：满足以下要求的任何类型：

- 符合可复制条件，即`is_trivially_copyable<TyArg>::value == true`
    
- `sizeof(T) <= 32` for `coalesced_group` and tiles of size lower or equal 32, `sizeof(T) <= 8` for larger tiles
    
- 为给定的函数对象拥有合适的算术或比较运算符。
    

**注意：**组中不同的线程可以为这个参数传递不同的值。

`op`：将提供硬件加速的积分类型的有效函数对象是`plus(),less(),greater(),bit_and(),bit_xor(),bit_or()`这些必须构建，因此需要TyVal模板参数，即`plus<int>()`Reduce还支持lambdas和其他可以使用调用的功能对象`operator()`

异步减少

template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
void reduce_update_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
void reduce_store_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);

template <typename TyGroup, typename TyArg, typename TyOp>
void reduce_store_async(const TyGroup& group, TyArg* ptr, TyArg&& val, TyOp&& op);

`*_async`API的变体异步计算结果，通过其中一个参与线程存储或更新指定目的地，而不是按每个线程返回。为了观察这些异步调用的效果，需要同步调用线程组或包含线程的更大组。

- 在原子存储或更新变体的情况下，`atomic`参数可以是[CUDA C++标准库](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives.html)中可用的`cuda::atomic`或`cuda::atomic_ref`。API的这种变体仅在CUDA C++标准库支持的平台和设备上可用，这些类型。还原结果用于根据指定的`op`对原子进行原子更新，例如，在`cg::plus()`的情况下，结果被原子添加到原子中。`atomic`持有的类型必须与`TyArg`的类型相匹配。原子的范围必须包括组中的所有线程，如果多个组同时使用同一原子，范围必须包括使用该组的所有线程。原子更新以轻松的内存排序进行。
    
- 在指针存储变体的情况下，还原的结果将微弱地存储在`dst`指针中。
    

**代码生成要求：**最低计算能力5.0，用于硬件加速的计算能力8.0，C++11。

`cooperative_groups/reduce.h`标题需要包含。

**整数向量的近似标准差示例：**

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

/// Calculate approximate standard deviation of integers in vec
__device__ int std_dev(const cg::thread_block_tile<32>& tile, int *vec, int length) {
    int thread_sum = 0;

    // calculate average first
    for (int i = tile.thread_rank(); i < length; i += tile.num_threads()) {
        thread_sum += vec[i];
    }
    // cg::plus<int> allows cg::reduce() to know it can use hardware acceleration for addition
    int avg = cg::reduce(tile, thread_sum, cg::plus<int>()) / length;

    int thread_diffs_sum = 0;
    for (int i = tile.thread_rank(); i < length; i += tile.num_threads()) {
        int diff = vec[i] - avg;
        thread_diffs_sum += diff * diff;
    }

    // temporarily use floats to calculate the square root
    float diff_sum = static_cast<float>(cg::reduce(tile, thread_diffs_sum, cg::plus<int>())) / length;

    return static_cast<int>(sqrtf(diff_sum));
}

**块宽缩的例子：**
```c++
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg=cooperative_groups;

/// The following example accepts input in *A and outputs a result into *sum
/// It spreads the data equally within the block
__device__ void block_reduce(const int* A, int count, cuda::atomic<int, cuda::thread_scope_block>& total_sum) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int thread_sum = 0;

    // Stride loop over all values, each thread accumulates its part of the array.
    for (int i = block.thread_rank(); i < count; i += block.size()) {
        thread_sum += A[i];
    }

    // reduce thread sums across the tile, add the result to the atomic
    // cg::plus<int> allows cg::reduce() to know it can use hardware acceleration for addition
 cg::reduce_update_async(tile, total_sum, thread_sum, cg::plus<int>());

 // synchronize the block, to ensure all async reductions are ready
    block.sync();
}
```
#### 11.6.3.2.`Reduce`操作员[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#reduce-operators "这个标题的永久链接")

以下是一些基本操作的函数对象原型，可以完成`reduce`
```c++
namespace cooperative_groups {
  template <typename Ty>
  struct cg::plus;

  template <typename Ty>
  struct cg::less;

  template <typename Ty>
  struct cg::greater;

  template <typename Ty>
  struct cg::bit_and;

  template <typename Ty>
  struct cg::bit_xor;

  template <typename Ty>
  struct cg::bit_or;
}
```
减少仅限于编译时实现可用的信息。因此，为了利用CC 8.0中引入的内在，`cg::`命名空间暴露了几个反映硬件的功能对象。除了`less/greater`，这些对象看起来与C++ STL中呈现的对象相似。与STL的任何差异的原因是，这些功能对象被设计为实际反映硬件内在的操作。

**功能描述：**
- `cg::plus:`接受两个值，并使用运算符+返回两者的总和。
    
- `cg::less:`接受两个值，并使用运算符<返回较小的值。这不同之处在于**返回的是较低**的**值，**而不是布尔值。
    
- `cg::greater:`接受两个值，并使用运算符<返回更大的值。这不同之处在于**返回**的**是更大的值**而不是布尔值。
    
- `cg::bit_and:`接受两个值并返回运算符&的结果。
    
- `cg::bit_xor:`接受两个值并返回运算符^的结果。
    
- `cg::bit_or:`接受两个值并返回运算符|的结果。
    

**示例：**
```c++
{
    // cg::plus<int> is specialized within cg::reduce and calls __reduce_add_sync(...) on CC 8.0+
    cg::reduce(tile, (int)val, cg::plus<int>());

    // cg::plus<float> fails to match with an accelerator and instead performs a standard shuffle based reduction
    cg::reduce(tile, (float)val, cg::plus<float>());

    // While individual components of a vector are supported, reduce will not use hardware intrinsics for the following
    // It will also be necessary to define a corresponding operator for vector and any custom types that may be used
    int4 vec = {...};
    cg::reduce(tile, vec, cg::plus<int4>())

    // Finally lambdas and other function objects cannot be inspected for dispatch
    // and will instead perform shuffle based reductions using the provided function object.
    cg::reduce(tile, (int)val, [](int l, int r) -> int {return l + r;});
}
```
#### 11.6.3.3. `inclusive_scan`和`exclusive_scan`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#inclusive-scan-and-exclusive-scan "这个标题的永久链接")

template <typename TyGroup, typename TyVal, typename TyFn>
auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyVal>
TyVal inclusive_scan(const TyGroup& group, TyVal&& val);

template <typename TyGroup, typename TyVal, typename TyFn>
auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyVal>
TyVal exclusive_scan(const TyGroup& group, TyVal&& val);

`inclusive_scan`并且`exclusive_scan`对传递的组中命名的每个线程提供的数据执行扫描操作。在`exclusive_scan`的情况下，每个线程的结果都是从`thread_rank`低于该线程的线程中减少数据。`inclusive_scan`结果还包括减少中的调用线程数据。

`group`：有效的组类型是`coalesced_group`和`thread_block_tile`。

`val`：满足以下要求的任何类型：

- 符合可复制条件，即`is_trivially_copyable<TyArg>::value == true`
    
- `sizeof(T) <= 32` for `coalesced_group` and tiles of size lower or equal 32, `sizeof(T) <= 8` for larger tiles
    
- 为给定的函数对象拥有合适的算术或比较运算符。
    

**注意：**组中不同的线程可以为这个参数传递不同的值。

`op`：为方便起见定义的函数对象是“[减少运算符”](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#collectives-cg-reduce-operators)中描述的`plus(),less(),greater(),bit_and(),bit_xor(),bit_or()`）。这些必须构建，因此需要TyVal模板参数，即`plus<int>()``inclusive_scan`和`exclusive_scan`还支持lambda和其他可以使用`operator()`调用的函数对象。没有此参数的过载使用`cg::plus<TyVal>()`

**扫描更新**

template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
auto inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyAtomic, typename TyVal>
TyVal inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);

template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
auto exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));

template <typename TyGroup, typename TyAtomic, typename TyVal>
TyVal exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);

`*_scan_update`集体使用额外的参数`atomic`，可以是[CUDA C++标准库](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives.html)中的`cuda::atomic`或`cuda::atomic_ref`。API的这些变体仅在CUDA C++标准库支持的平台和设备上可用。这些变体将根据`op`对`atomic`进行更新，该组中所有线程的输入值之和的值。`atomic`的先前值将与每个线程的扫描结果相结合并返回。`atomic`持有的类型必须与`TyVal`的类型相匹配。原子的范围必须包括组中的所有线程，如果多个组同时使用同一原子，范围必须包括使用该组的所有线程。原子更新以轻松的内存排序进行。

以下伪代码说明了扫描的更新变体是如何工作的：

/*
 inclusive_scan_update behaves as the following block,
 except both reduce and inclusive_scan is calculated simultaneously.
auto total = reduce(group, val, op);
TyVal old;
if (group.thread_rank() == selected_thread) {
    atomically {
        old = atomic.load();
        atomic.store(op(old, total));
    }
}
old = group.shfl(old, selected_thread);
return op(inclusive_scan(group, val, op), old);
*/

**编程要求：**最低计算能力5.0，C++11。

`cooperative_groups/scan.h`标题需要包含。

**示例：**
```c++
#include <stdio.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    auto thread_block = cg::this_thread_block();
    auto tile = cg::tiled_partition<8>(thread_block);
    unsigned int val = cg::inclusive_scan(tile, tile.thread_rank());
    printf("%u: %u\n", tile.thread_rank(), val);
}

/*  prints for each group:
    0: 0
    1: 1
    2: 3
    3: 6
    4: 10
    5: 15
    6: 21
    7: 28
*/
```
**使用exclusive_scan进行流压缩的示例：**
```c++
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

// put data from input into output only if it passes test_fn predicate
template<typename Group, typename Data, typename TyFn>
__device__ int stream_compaction(Group &g, Data *input, int count, TyFn&& test_fn, Data *output) {
    int per_thread = count / g.num_threads();
    int thread_start = min(g.thread_rank() * per_thread, count);
    int my_count = min(per_thread, count - thread_start);

    // get all passing items from my part of the input
    //  into a contagious part of the array and count them.
    int i = thread_start;
    while (i < my_count + thread_start) {
        if (test_fn(input[i])) {
            i++;
        }
        else {
            my_count--;
            input[i] = input[my_count + thread_start];
        }
    }

    // scan over counts from each thread to calculate my starting
    //  index in the output
    int my_idx = cg::exclusive_scan(g, my_count);

    for (i = 0; i < my_count; ++i) {
        output[my_idx + i] = input[thread_start + i];
    }
    // return the total number of items in the output
    return g.shfl(my_idx + my_count, g.num_threads() - 1);
}
```
**使用exclusive_scan_update的动态缓冲空间分配示例：**

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

// Buffer partitioning is static to make the example easier to follow,
// but any arbitrary dynamic allocation scheme can be implemented by replacing this function.
__device__ int calculate_buffer_space_needed(cg::thread_block_tile<32>& tile) {
    return tile.thread_rank() % 2 + 1;
}

__device__ int my_thread_data(int i) {
    return i;
}

__global__ void kernel() {
    __shared__ extern int buffer[];
    __shared__ cuda::atomic<int, cuda::thread_scope_block> buffer_used;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    buffer_used = 0;
    block.sync();

    // each thread calculates buffer size it needs
    int buf_needed = calculate_buffer_space_needed(tile);

    // scan over the needs of each thread, result for each thread is an offset
    // of that thread’s part of the buffer. buffer_used is atomically updated with
    // the sum of all thread's inputs, to correctly offset other tile’s allocations
    int buf_offset =
        cg::exclusive_scan_update(tile, buffer_used, buf_needed);

    // each thread fills its own part of the buffer with thread specific data
    for (int i = 0 ; i < buf_needed ; ++i) {
        buffer[buf_offset + i] = my_thread_data(i);
    }

    block.sync();
    // buffer_used now holds total amount of memory allocated
    // buffer is {0, 0, 1, 0, 0, 1 ...};
}

### 11.6.4.执行控制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-control "这个标题的永久链接")

#### 11.6.4.1.`invoke_one`和`invoke_one_broadcast`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#invoke-one-and-invoke-one-broadcast "这个标题的永久链接")

template<typename Group, typename Fn, typename... Args>
void invoke_one(const Group& group, Fn&& fn, Args&&... args);

template<typename Group, typename Fn, typename... Args>
auto invoke_one_broadcast(const Group& group, Fn&& fn, Args&&... args) -> decltype(fn(args...));

`invoke_one`从调用组中选择单个任意线程，并使用该线程使用提供的参数`args`调用提供的可调用`fn`。在`invoke_one_broadcast`的情况下，调用的结果也会分发到组中的所有线程，并从该集合返回。

调用组可以在调用提供的可调用项之前和/或之后与选定的线程同步。这意味着呼叫组内的通信不允许在提供的可调用主体内进行，否则无法保证前进。允许在提供的可调用的正文中与调用组以外的线程进行通信。线程选择机制**不能**保证是确定的。

在具有计算能力9.0或更高版本的设备上，当调用[显式组类型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#group-types-explicit-cg)时，可以使用硬件加速来选择线程。

`group`：所有组类型都适用于`invoke_one`，`coalesced_group`和`thread_block_tile`适用于`invoke_one_broadcast`。

`fn`：可以使用`operator()`调用的函数或对象。

`args`：与提供的可调用`fn`的参数类型相匹配的类型参数包。

在`invoke_one_broadcast`的情况下，提供的可调用`fn`的返回类型必须满足以下要求：

- 符合可复制条件，即`is_trivially_copyable<T>::value == true`
    
- `sizeof(T) <= 32` for `coalesced_group` and tiles of size lower or equal 32, `sizeof(T) <= 8` for larger tiles
    

**Codegen要求：**最低计算能力5.0，用于硬件加速的计算能力9.0，C++11。

**重写为使用invoke_one_broadcast的**[发现模式部分的](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#discovery-pattern-cg)**聚合原子示例****：**

#include <cooperative_groups.h>
#include <cuda/atomic>
namespace cg = cooperative_groups;

template<cuda::thread_scope Scope>
__device__ unsigned int atomicAddOneRelaxed(cuda::atomic<unsigned int, Scope>& atomic) {
    auto g = cg::coalesced_threads();
    auto prev = cg::invoke_one_broadcast(g, [&] () {
        return atomic.fetch_add(g.num_threads(), cuda::memory_order_relaxed);
    });
    return prev + g.thread_rank();
}

## 11.7.网格同步[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#grid-synchronization "这个标题的永久链接")

在引入合作组之前，CUDA编程模型只允许在内核完成边界的线程块之间同步。内核边界带有隐性的状态无效，以及潜在的性能影响。

例如，在某些用例中，应用程序有大量的小内核，每个内核代表处理管道中的一个阶段。当前的CUDA编程模型要求这些内核的存在，以确保在一个管道阶段运行的线程块在下一个管道阶段运行的线程块准备好消耗之前生成数据。在这种情况下，提供全局线程间块同步的能力将允许应用程序重组为具有持久线程块，这些线程块能够在给定阶段完成后在设备上同步。

要跨网格同步，从内核内，您只需使用`grid.sync()`函数：

grid_group grid = this_grid();
grid.sync();

And when launching the kernel it is necessary to use, instead of the `<<<...>>>` execution configuration syntax, the `cudaLaunchCooperativeKernel`CUDA runtime launch API or the `CUDA driver equivalent`.

**示例：**

为了保证GPU上线程块的共同居住，需要仔细考虑启动的块数量。例如，SM的块数多，可以启动如下：

int dev = 0;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
// initialize, then launch
cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);

或者，您可以使用占用计算器计算每个SM可以同时容纳多少块，从而最大化暴露的并行性，如下所示：

/// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
int numBlocksPerSm = 0;
 // Number of threads my_kernel will be launched with
int numThreads = 128;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0);
// launch
void *kernelArgs[] = { /* add kernel args */ };
dim3 dimBlock(numThreads, 1, 1);
dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
cudaLaunchCooperativeKernel((void*)my_kernel, dimGrid, dimBlock, kernelArgs);

最佳做法是首先通过查询设备属性`cudaDevAttrCooperativeLaunch`来确保设备支持合作发布：

int dev = 0;
int supportsCoopLaunch = 0;
cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);

如果设备0支持该属性，则将`supportsCoopLaunch`设置为1。仅支持具有6.0及更高计算能力的设备。此外，您需要在以下任一方面运行：

- 没有MPS的Linux平台
    
- 带有MPS的Linux平台，在具有7.0或更高计算能力的设备上
    
- 最新的Windows平台
    

# 12.集群启动控制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-launch-control "这个标题的永久链接")

## 12.1.介绍[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#introduction-clc "这个标题的永久链接")

计算能力10.0引入了集群启动控制，这是一个新功能，通过取消线程块或线程块集群，为开发人员提供了对线程块调度的更多控制。

在处理可变大小的问题时，有两种主要方法来确定内核线程块的数量。

**方法1：每个线程块的固定工作：**

在这种方法中，线程块的数量由问题大小决定，而每个线程块完成的工作量保持不变或有限。

这种方法的主要优势：

- SM之间的负载平衡。
    
    特别是，当线程块运行时间表现出可变性和/或线程块的数量比GPU可以同时执行的要大得多（導致低尾效应），这种方法允许GPU调度器在某些SM上运行比其他SM上更多的线程块。
    
- 抢占。
    
    GPU调度器可以开始执行[高优先级内核](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-priorities)，即使它是在低优先级内核的执行已经开始后启动的，通过将高优先级内核的线程块安排为当前运行的低优先级内核的线程块完成。一旦高优先级内核完成，它就可以返回低优先级内核。
    

**方法2：固定数量的线程块：**

在这种方法中，通常作为块步幅或网格步幅循环实现，线程块的数量并不直接取决于问题的大小。相反，每个线程块完成的工作量是问题大小的函数。通常，线程块的数量基于执行内核的GPU上的SM数量和所需的占用率。

这种方法的主要优势：

- 减少线程块开销。
    
    这种方法不仅减少了摊销线程块启动延迟，而且还最大限度地减少了与所有线程块共享操作相关的计算开销。这些开销可能明显高于启动延迟开销。
    
    例如，在卷积内核中，由于线程块数量固定，计算卷积系数的序言——独立于线程块索引——可以减少计算次数，从而减少冗余计算。
    

**集群发射控制方法：**

集群启动控制允许内核请求（**取消**）尚未开始执行的块的线程块索引。

该机制允许线程块之间的工作窃取：线程块试图取消另一个尚未开始运行的线程块的启动。如果取消成功，它通过使用取消块索引来执行任务来“偷”其他线程块的工作。

如果没有更多的线程块索引可用，取消将失败，并且可能因其他原因而失败，例如正在安排更高优先级的内核。在后一种情况下，如果线程块在取消失败后退出，调度器可以开始执行优先级更高的内核，之后它将继续调度当前内核的剩余线程块以供执行。

下表总结了三种方法的优缺点：

||**每个线程块的固定工作**|**固定线程块数量**|**集群启动控制**|
|---|---|---|---|
|减少开销|**X**|**V**|**V**|
|抢占|**V**|**X**|**V**|
|负载平衡|**V**|**X**|**V**|

## 12.2.集群启动控制API详细信息[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-launch-control-api-details "这个标题的永久链接")

通过集群启动控制API取消线程块是异步的，并使用内存屏障同步，遵循类似于[异步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memcpy-async-barrier)的编程模式。

目前通过[libcu++](https://nvidia.github.io/cccl/libcudacxx/ptx.html)提供的API提供了一个请求指令，将编码的取消结果写入`__shared__`变量，以及将结果解码为_成功_/_失败_标志的指令，以及在_成功_的情况下取消线程块的索引。

### 12.2.1.线程块取消步骤[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-cancellation-steps "这个标题的永久链接")

使用集群启动控制的首选方式是来自单个线程，即一次一个请求。

以下是线程块取消过程的五个步骤。前两个步骤是取消结果和同步变量的声明和初始化，这些在工作窃取之前完成。最后三个步骤通常在线程块索引上的工作窃窃循环中执行。

1. 声明线程块取消的变量：
    
    __shared__ uint4 result; // Request result.
    __shared__ uint64_t bar; // Synchronization barrier.
    int phase = 0;           // Synchronization barrier phase.
    
2. 用单个到达计数初始化共享内存屏障：
    
    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&bar, 1);
    __syncthreads();
    
3. 通过单个线程提交异步取消请求并设置交易计数：
    
    if (cg::thread_block::thread_rank() == 0) {
        cg::invoke_one(cg::coalesced_threads(), ptx::clusterlaunchcontrol_try_cancel, &result, &bar);
        ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
    }
    
    笔记
    
    由于线程块取消是一个统一的指令，建议在[invoke_one](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#invoke-one-and-invoke-one-broadcast)线程选择器中提交。这允许编译器优化剥离环。
    
4. 同步（完成）异步取消请求：
    
    while (!ptx::mbarrier_try_wait_parity(&bar, phase))
    {}
    phase ^= 1;
    
5. 检索取消状态和取消的线程块索引：
    
    bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
    if (success) {
        // Don't need all three for 1D/2D thread blocks:
        int bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x(result);
        int by = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_y(result);
        int bz = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_z(result);
    }
    
6. 确保异步[代理](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies)和通用[代理](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies)之间的共享内存操作的可见性，并防止工作窃取循环迭代之间的数据竞赛。
    

### 12.2.2.线程块取消约束[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-cancellation-constraints "这个标题的永久链接")

约束与失败的取消请求有关：

- 在**观察到**之前失败的请求后提交另一个取消请求是未定义的行为。
    
    在下面的两个代码示例中，假设第一个取消请求失败，只有第一个示例表现出未定义的行为。第二个示例是正确的，因为取消请求之间没有观察：
    
    **无效代码：**
    
    // First request:
    ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);
    
    // First request query:
    [Synchronize bar0 code here.]
    bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
    assert(!success0); // Observed failure; second cacellation will be invalid.
    
    // Second request - next line is Undefined Behavior:
    ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);
    
    **有效代码：**
    
    // First request:
    ptx::clusterlaunchcontrol_try_cancel(&result0, &bar0);
    
    // Second request:
    ptx::clusterlaunchcontrol_try_cancel(&result1, &bar1);
    
    // First request query:
    [Synchronize bar0 code here.]
    bool success0 = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result0);
    assert(!success0); // Observed failure; second cacellation was valid.
    
- 检索失败取消请求的线程块索引是未定义行为。
    
- 不建议从多个线程提交取消请求。它导致多个线程块被取消，需要小心处理，例如：
    
    - 每个提交线程必须提供一个唯一的`__shared__`结果指针，以避免数据竞赛。
        
    - 如果使用相同的障碍进行同步，必须相应地调整到达和交易计数。
        

### 12.2.3.内核示例：向量标量乘法[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernel-example-vector-scalar-multiplication "这个标题的永久链接")

以下三个内核展示了每个线程块的固定工作、线程块的固定数量和矢量标量乘法的聚类启动控制方法v―:=αv―.

- 每个线程块的固定工作：
    
    __global__
    void kernel_fixed_work (float* data, int n)
    {
        // Prologue:
        float alpha = compute_scalar();
    
        // Computation:
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;
    }
    
    // Launch: kernel_fixed_work<<<1024, (n + 1023) / 1024>>>(data, n);
    
- 固定线程块数量：
    
    __global__
    void kernel_fixed_blocks (float* data, int n)
    {
        // Prologue:
        float alpha = compute_scalar();
    
        // Computation:
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < n) {
            data[i] *= alpha;
            i += gridDim.x * blockDim.x;
        }
    }
    
    // Launch: kernel_fixed_blocks<<<1024, SM_COUNT>>>(data, n);
    
- 集群启动控制：
    
    #include <cooperative_groups.h>
    #include <cuda/ptx>
    
    namespace cg = cooperative_groups;
    namespace ptx = cuda::ptx;
    
    __global__
    void kernel_cluster_launch_control (float* data, int n)
    {
        // Cluster launch control initialization:
        __shared__ uint4 result;
        __shared__ uint64_t bar;
        int phase = 0;
    
        if (cg::thread_block::thread_rank() == 0)
            ptx::mbarrier_init(&bar, 1);
    
        // Prologue:
        float alpha = compute_scalar(); // Device function not shown in this code snippet.
    
        // Work-stealing loop:
        int bx = blockIdx.x; // Assuming 1D x-axis thread blocks.
    
        while (true) {
            // Protect result from overwrite in the next iteration,
            // (also ensure barrier initialization at 1st iteration):
            __syncthreads();
    
            // Cancellation request:
            if (cg::thread_block::thread_rank() == 0) {
                // Acquire write of result in the async proxy:
                ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
    
                cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel(&result, &bar);});
                ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared, &bar, sizeof(uint4));
            }
    
            // Computation:
            int i = bx * blockDim.x + threadIdx.x;
            if (i < n)
                data[i] *= alpha;
    
            // Cancellation request synchronization:
            while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar, phase))
            {}
            phase ^= 1;
    
            // Cancellation request decoding:
            bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
            if (!success)
                break;
    
            bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
    
            // Release read of result to the async proxy:
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
        }
    }
    
    // Launch: kernel_cluster_launch_control<<<1024, (n + 1023) / 1024>>>(data, n);
    

### 12.2.4.线程块集群的集群启动控制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-launch-control-for-thread-block-clusters "这个标题的永久链接")

在线[程块集群](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters)的情况下，线程块取消步骤与非集群设置相同，但有轻微的调整。与非集群情况一样，不建议从**集群中的**多个线程提交取消请求，因为这将尝试取消多个集群。

- 取消由单个集群线程提交。
    
- The shared memory result of each cluster’s thread block will receive the same (encoded) value of the cancelled thread block index (i.e., the result value is multicasted). The result received by all thread blocks corresponds to the local block index `{0, 0, 0}` within a cluster. Therefore, thread blocks within the cluster need to add the local block index.
    
- 同步由每个集群的线程块使用本地`__shared__`内存屏障执行。必须使用`ptx::scope_cluster`范围执行屏障操作。
    
- 在集群案例中取消需要所有线程块都存在。用户可以通过使用来自[集群组](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-group-cg)API的`cg::cluster_group::sync()`来保证所有线程块都在运行。
    

以下核心演示了线程块群集案例中的群集启动控制方法：

#include <cooperative_groups.h>
#include <cuda/ptx>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

__global__ __cluster_dims__(2, 1, 1)
void kernel_cluster_launch_control (float* data, int n)
{
    // Cluster launch control initialization:
    __shared__ uint4 result;
    __shared__ uint64_t bar;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0) {
        ptx::mbarrier_init(&bar, 1);
        ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster); // CGA-level fence.
    }

    // Prologue:
    float alpha = compute_scalar(); // Device function not shown in this code snippet.

    // Work-stealing loop:
    int bx = blockIdx.x; // Assuming 1D x-axis thread blocks.

    while (true) {
        // Protect result from overwrite in the next iteration,
        // (also ensure all thread blocks have started at 1st iteration):
        cg::cluster_group::sync();

        // Cancellation request by a single cluster thread:
        if (cg::cluster_group::thread_rank() == 0) {
            // Acquire write of result in the async proxy:
            ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);

            cg::invoke_one(cg::coalesced_threads(), [&](){ptx::clusterlaunchcontrol_try_cancel_multicast(&result, &bar);});
        }

        // Cancellation completion tracked by each thread block:
        if (cg::thread_block::thread_rank() == 0)
            ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cluster, ptx::space_shared, &bar, sizeof(uint4));

        // Computation:
        int i = bx * blockDim.x + threadIdx.x;
        if (i < n)
            data[i] *= alpha;

        // Cancellation request synchronization:
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster, &bar, phase))
        {}
        phase ^= 1;

        // Cancellation request decoding:
        bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
        if (!success)
            break;

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result);
        bx += cg::cluster_group::block_index().x; // Add local offset.

        // Release read of result to the async proxy:
        ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}

// Launch: kernel_cluster_launch_control<<<1024, (n + 1023) / 1024>>>(data, n);

# 13.CUDA动态并行[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-dynamic-parallelism "这个标题的永久链接")

## 13.1.介绍[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#introduction-cuda-dynamic-parallelism "这个标题的永久链接")

### 13.1.1.一览表[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id214 "这个标题的永久链接")

_动态并行_是CUDA编程模型的扩展，使CUDA内核能够直接在GPU上创建和同步新工作。在程序中需要的任何点动态创建并行性，提供了令人兴奋的能力。

直接从GPU创建工作的能力可以减少在主机和设备之间传输执行控制和数据的需要，因为现在可以通过在设备上执行的线程在运行时做出启动配置决策。此外，数据依赖的并行工作可以在运行时在内核中内联生成，动态利用GPU的硬件调度器和负载平衡器，并适应数据驱动的决策或工作负载。以前需要修改以消除递归、不规则循环結構或其他不符合平面、单級平行的結構的算法和寫程式模式可以更透明地表达。

本文件描述了CUDA的扩展功能，这些功能实现了动态并行性，包括对CUDA编程模型的修改和添加，以及利用这种额外能力的指南和最佳实践。

动态并行性仅由计算能力3.5及更高的设备支持。

### 13.1.2.词汇[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#glossary "这个标题的永久链接")

本指南中使用的术语的定义。

网格

网格是_线程的_集合。网格中的线程执行内核_函数_，并被划分为_线程块_。

线程块

线程块是一组在同一多处理器（_SM_）上执行的线程。线程块中的线程可以访问共享内存，并且可以显式同步。

内核功能

内核函数是一个隐式并行子例程，在CUDA执行和内存模型下为网格中的每个线程执行。

主持人

主机是指最初调用CUDA的执行环境。通常在系统的CPU处理器上运行的线程。

父母

_父线程_、线程块或网格是启动了新网格，即子网格。在其所有启动的子网格也完成之前，父格不被视为已完成。

儿童

子线程、块或网格是由父网格启动的。子网格必须完成，然后父线程、线程块或网格才算成完成。

螺纹块范围

具有线程块范围的对象具有单个线程块的寿命。只有在创建对象的线程块中线程操作时，它们才会有定义的行为，并在创建它们的线程块完成后被销毁。

设备运行时

设备运行时是指可用于使内核函数使用动态并行的运行时系统和API。

## 13.2.执行环境和内存模型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-environment-and-memory-model "这个标题的永久链接")

### 13.2.1.执行环境[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-environment "这个标题的永久链接")

CUDA执行模型基于线程、线程块和网格的原语，内核函数定义了线程块和网格中单个线程执行的程序。当调用内核函数时，网格的属性由执行配置描述，该配置在CUDA中具有特殊的语法。CUDA中对动态并行的支持扩展了配置、启动和隐式同步到设备上运行的线程的新网格的能力。

#### 13.2.1.1.父网格和子网格[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parent-and-child-grids "这个标题的永久链接")

配置和启动新网格的设备线程属于父网格，调用创建的网格是子网格。

子网格的调用和完成是正确嵌套的，这意味着在其线程创建的所有子网格完成之前，父网格不被视为完成，并且运行时保证了父网格和子网格之间的隐式同步。

![父子启动嵌套](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/parent-child-launch-nesting.png)

图30 父子启动嵌套[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parent-child-launch-nesting-figure "此图像的永久链接")

#### 13.2.1.2.CUDA原始的范围[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#scope-of-cuda-primitives "这个标题的永久链接")

在主机和设备上，CUDA运行时提供了一个API，用于启动内核，并通过流和事件跟踪启动之间的依赖关系。在主机系统上，启动状态和引用流和事件的CUDA原语由进程中的所有线程共享；但是进程独立执行，可能不共享CUDA对象。

在设备上，网格中的所有线程都可以看到启动的内核和CUDA对象。这意味着，例如，流可以由一个线程创建，并由网格中的任何其他线程使用。

#### 13.2.1.3.同步[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dynamic-parallelism-synchronization "这个标题的永久链接")

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中已弃用，并删除用于compute_90+编译。对于计算能力<9.0，需要通过指定`-DCUDA_FORCE_CDP1_IF_SUPPORTED`来编译时选择加入，才能继续在设备代码中使用`cudaDeviceSynchronize()`）。请注意，这计划在未来的CUDA版本中完全删除。

来自任何线程的CUDA运行时操作，包括内核启动，都可以在网格中的所有线程中可见。这意味着父网格中的调用线程可以执行同步，以控制网格中任何线程在网格中的任何线程创建的流上启动的网格的启动顺序。在网格中所有线程的所有启动完成之前，网格的执行不被视为完成。如果网格中的所有线程在所有子启动完成之前退出，则将自动触发隐式同步操作。

#### 13.2.1.4.流和事件[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams-and-events "这个标题的永久链接")

CUDA_流_和_事件_允许控制网格启动之间的依赖关系：启动到同一流中的网格按顺序执行，事件可用于在流之间创建依赖关系。在设备上创建的流和事件具有完全相同的目的。

在网格中创建的流和事件存在于网格范围内，但当在创建它们的网格之外使用时，具有未定义的行为。如上所述，网格启动的所有工作在网格退出时都隐式同步；启动到流中的工作都包含在其中，所有依赖项都得到了适当的解决。在网格范围之外修改的流上的操作行为是未定义的。

在任何内核中使用时，在主机上创建的流和事件具有未定义的行为，就像由父网格创建的流和事件在子网格中使用时具有未定义的行为一样。

#### 13.2.1.5.订购和并发[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ordering-and-concurrency "这个标题的永久链接")

从设备运行时开始的内核启动顺序遵循CUDA流排序语义。在网格中，所有内核启动到同一流中（后面讨论的触发和忘记流除外）都是按顺序执行的。当同一网格中的多个线程启动到同一流中时，流中的排序取决于网格中的线程调度，该调度可以通过`__syncthreads()`等同步原语进行控制。

请注意，虽然命名流由网格中的所有线程共享，但隐式_NULL_流仅由线程块中的所有线程共享。如果线程块中的多个线程启动到隐式流中，那么这些启动将按顺序执行。如果不同线程块中的多个线程启动到隐式流中，那么这些启动可能会同时执行。如果线程块内多个线程的启动需要并发性，则应使用显式命名流。

_动态并行性_使并发性更容易在程序中表达；然而，设备运行时在CUDA执行模型中没有引入新的并发性保证。不能保证设备上任何数量的不同线程块之间同时执行。

缺乏并发保证延伸到父网格及其子网格。当父网格启动子网格时，一旦满足流依赖性并且硬件资源可用于托管子网格，子网格可能会开始执行，但在父网格达到隐式同步点之前，不能保证开始执行。

虽然并发通常很容易实现，但它可能会因设备配置、应用程序工作负载和运行时调度而有所不同。因此，依赖不同线程块之间的任何并发是不安全的。

#### 13.2.1.6.设备管理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-management "这个标题的永久链接")

设备运行时不支持多GPU；设备运行时只能在它当前正在执行的设备上运行。然而，允许查询系统中任何支持CUDA的设备的属性。

### 13.2.2.记忆模型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-model "这个标题的永久链接")

父网格和子网格共享相同的全局和常量内存存储，但具有不同的本地和共享内存。

#### 13.2.2.1.连贯性和一致性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#coherence-and-consistency "这个标题的永久链接")

##### 13.2.2.1.1.全球记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory "这个标题的永久链接")

父网格和子网格对全局内存有一致的访问，子网格和父网格之间的一致性保证较弱。在执行子网格时，只有一个时间点，其内存视图与父线程完全一致：在父网格调用子网格的点。

在子网格调用之前，父线程中的所有全局内存操作对子网格可见。删除`cudaDeviceSynchronize()`，无法再从父网格访问子网格中线程所做的修改。在父网格退出之前访问子网格中线程所做的修改的唯一方法是通过启动到`cudaStreamTailLaunch`流的内核。

在以下示例中，执行`child_launch`的子网格只能保证看到在子网格启动之前对`data`所做的修改。由于父线程0正在执行启动，因此子线程将与父线程0看到的内存一致。由于第一次`__syncthreads()`调用，子程序将看到`data[0]=0`，`data[1]=1`，...，`data[255]=255`（如果没有`__syncthreads()`调用，只能保证子程序看到`data[0]=0`）。子网格只能保证在隐式同步时返回。这意味着子网格中线程所做的修改永远不会保证父网格可用。要访问`child_launch`所做的修改，`tail_launch`内核被启动到`cudaStreamTailLaunch`流中。

__global__ void tail_launch(int *data) {
   data[threadIdx.x] = data[threadIdx.x]+1;
}

__global__ void child_launch(int *data) {
   data[threadIdx.x] = data[threadIdx.x]+1;
}

__global__ void parent_launch(int *data) {
   data[threadIdx.x] = threadIdx.x;

   __syncthreads();

   if (threadIdx.x == 0) {
       child_launch<<< 1, 256 >>>(data);
       tail_launch<<< 1, 256, 0, cudaStreamTailLaunch >>>(data);
   }
}

void host_launch(int *data) {
    parent_launch<<< 1, 256 >>>(data);
}

##### 13.2.2.1.2.零复制内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#zero-copy-memory "这个标题的永久链接")

零复制系统内存与全局内存具有相同的一致性和一致性保证，并遵循上述详述语义。内核可能不会分配或释放零复制内存，但可以使用指针从主机程序传递的零复制。

##### 13.2.2.1.3.恒定记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant-memory "这个标题的永久链接")

常量不能从设备中修改。它们只能从主机修改，但当有併發网格在其生命周期中的任何点访问常量时，从主机修改常量的行为是未定义的。

##### 13.2.2.1.4.共享和本地内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-and-local-memory "这个标题的永久链接")

共享和本地内存分别是线程块或线程的私有内存，父和子内存不可见或一致。当这些位置之一的对象被引用到其所属范围之外时，行为是未定义的，并可能导致错误。

NVIDIA编译器将尝试警告是否可以检测到指向本地或共享内存的指针正在作为参数传递给内核启动。在运行时，程序员可以使用`__isGlobal()`内在来确定指针是否引用全局内存，因此可以安全地传递给子启动。

请注意，调用`cudaMemcpy*Async()`或`cudaMemset*Async()`可能会在设备上调用新的子内核，以保留流语义。因此，将共享或本地内存指针传递给这些API是非法的，并且会返回错误。

##### 13.2.2.1.5.本地记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#local-memory "这个标题的永久链接")

本地内存是执行线程的私有存储，在该线程之外不可见。在启动子内核时，将指针传递给本地内存作为启动参数是非法的。从子程序取消引用此类本地内存指针的结果将是未定义的。

例如，如果`child_launch`访问`x_array`则以下内容是非法的，行为未定义：

int x_array[10];       // Creates x_array in parent's local memory
child_launch<<< 1, 1 >>>(x_array);

程序员有时很难意识到编译器何时将变量放入本地内存中。作为一般规则，传递给子内核的所有存储都应从全局内存堆中显式分配，无论是使用`cudaMalloc()``new()`还是通过在全局范围内声明`__device__`存储。例如：

// Correct - "value" is global storage
__device__ int value;
__device__ void x() {
    value = 5;
    child<<< 1, 1 >>>(&value);
}

// Invalid - "value" is local storage
__device__ void y() {
    int value = 5;
    child<<< 1, 1 >>>(&value);
}

##### 13.2.2.1.6.纹理记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory-cdp "这个标题的永久链接")

写入纹理映射的全局内存区域与纹理访问不连贯。纹理内存的一致性在调用子网格时和子网格完成后强制执行。这意味着在子内核启动之前写入内存会反映在子内核的纹理内存访问中。与上述全局内存类似，子内存的写入永远不会保证反映在父内存访问的纹理内存中。在父网格退出之前，访问子网格中线程所做的修改的唯一方法是通过启动到`cudaStreamTailLaunch`流中的内核。父母和子女同时访问可能会导致数据不一致。

## 13.3.编程接口[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface-cdp "这个标题的永久链接")

### 13.3.1.CUDA C++参考[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-reference "这个标题的永久链接")

本节介绍CUDA C++语言扩展的更改和添加，以支持_动态并行_。

使用CUDA C++进行动态并行的CUDA内核可用的语言接口和API，称为_设备运行时_，与主机上可用的CUDA运行时API基本相似。在可能的情况下，保留了CUDA运行时API的语法和语义，以便于在主机或设备环境中运行的例程的代码重复使用。

与CUDA C++中的所有代码一样，这里概述的API和代码是每线程代码。这使每个线程能够就接下来要执行的内核或操作做出独特的动态决定。块内线程之间没有同步要求来执行任何提供的设备运行时API，这使得设备运行时API函数可以在任意发散的内核代码中调用，而不会死锁。

#### 13.3.1.1.设备侧内核启动[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-side-kernel-launch "这个标题的永久链接")

内核可以使用标准CUDA <<< >>>语法从设备启动：

kernel_name<<< Dg, Db, Ns, S >>>([kernel arguments]);

- `Dg`类型为`dim3`，并指定网格的尺寸和大小
    
- `Db`类型为`dim3`，并指定每个线程块的尺寸和大小
    
- `Ns`类型为`size_t`，除了静态分配的内存外，还指定了该调用每个线程块动态分配的共享内存的字节数。`Ns`是一个可选参数，默认为0。
    
- `S`是`cudaStream_t`类型，并指定与此调用关联的流。流必须分配到正在呼叫的同一网格中。`S`是一个可选参数，默认为NULL流。
    

##### 13.3.1.1.1.发射是非同步的[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launches-are-asynchronous "这个标题的永久链接")

与主机端启动相同，所有设备端内核启动与启动线程是异步的。也就是说，`<<<>>>`启动命令将立即返回，启动线程将继续执行，直到它到达隐式启动同步点（例如在启动到`cudaStreamTailLaunch`流的内核中）。

子网格启动发布到设备上，并将独立于父线程执行。子网格可以在启动后随时开始执行，但不能保证在启动线程到达隐式启动同步点之前开始执行。

##### 13.3.1.1.2.启动环境配置[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-environment-configuration "这个标题的永久链接")

所有全局设备配置设置（例如，从`cudaDeviceGetCacheConfig()`返回的共享内存和L1缓存大小，以及从`cudaDeviceGetLimit()`返回的设备限制）都将从父级继承。同样，堆栈大小等设备限制将保持配置不变。

对于主机启动的内核，从主机设置的每个内核配置将优先于全局设置。当内核从设备启动时，也会使用这些配置。无法从设备重新配置内核环境。

#### 13.3.1.2.溪流[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams-cdp "这个标题的永久链接")

命名和未命名（NULL）流都可以从设备运行时获得。网格内的任何线程都可以使用命名流，但流句柄不能传递给其他子/父内核。换句话说，流应该被视为创建它的网格的私有。

与主机端启动类似，启动到单独流中的工作可能会同时运行，但不能保证实际的并发性。CUDA编程模型不支持依赖子内核之间并发性的程序，并且将具有未定义的行为。

设备不支持主机端NULL流的跨流屏障语义（详情见下文）。为了保持与主机运行时的语义兼容性，必须使用`cudaStreamCreateWithFlags()`API创建所有设备流，并传递`cudaStreamNonBlocking`标志。`cudaStreamCreate()`调用是主机运行时仅API，将无法为设备编译。

由于设备运行时不支持`cudaStreamSynchronize()`和`cudaStreamQuery()`，当应用程序需要知道流启动的子内核已完成时，应使用启动到`cudaStreamTailLaunch`流中的内核。

##### 13.3.1.2.1.隐式（空）流[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-implicit-null-stream "这个标题的永久链接")

在主机程序中，未命名（NULL）流与其他流有额外的屏障同步语义（详情请参阅[默认流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#default-stream)）。设备运行时提供线程块中所有线程之间共享的单个隐式、未命名的流，但由于所有命名流都必须使用`cudaStreamNonBlocking`标志创建，因此启动到NULL流中的工作不会在任何其他流（包括其他线程块的NULL流）中插入对待定工作的隐式依赖。

##### 13.3.1.2.2.火与遗忘之流[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-fire-and-forget-stream "这个标题的永久链接")

名为fire-and-forget的流（cudaStreamFireAndForget）允许用户以更少的模板和没有流跟踪开销启动fire-and-forget工作。它在功能上与每次启动创建新流并启动该流相比更快。

火和忘记的发射计划立即发射，而不依赖于之前发射的网格的完成。除了通过父网格末尾的隐式同步，否则其他网格启动不能依赖于火和忘记发射的完成。因此，在父网格的发射和忘记工作完成之前，父网格流中的尾部启动或下一个网格不会启动。

// In this example, C2's launch will not wait for C1's completion
__global__ void P( ... ) {
   C1<<< ... , cudaStreamFireAndForget >>>( ... );
   C2<<< ... , cudaStreamFireAndForget >>>( ... );
}

火和忘记流不能用于记录或等待事件。尝试这样做会导致`cudaErrorInvalidValue`。当使用定义的`CUDA_FORCE_CDP1_IF_SUPPORTED`编译时，不支持fire-and-forget流。启动并忘记流的使用需要以64位模式进行编译。

##### 13.3.1.2.3.尾巴发射流[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-tail-launch-stream "这个标题的永久链接")

名为流（`cudaStreamTailLaunch`）的尾部启动允许网格在完成后安排一个新的网格进行启动。在大多数情况下，应该可以使用尾部启动来实现与`cudaDeviceSynchronize()`相同的功能。

每个网格都有自己的尾部发射流。在尾流启动之前，所有由网格启动的非尾部发射工作都隐式同步。即父网格的尾部启动不会启动，直到父网格和父网格向普通流或每线程或触发和忘记流发射的所有工作完成。如果两个网格被发射到同一网格的尾部发射流，则在早期网格及其所有后代工作完成之前，后期网格不会启动。

// In this example, C2 will only launch after C1 completes.
__global__ void P( ... ) {
   C1<<< ... , cudaStreamTailLaunch >>>( ... );
   C2<<< ... , cudaStreamTailLaunch >>>( ... );
}

启动到尾启动流的网格不会启动，直到父网格完成所有工作，包括父网格在所有非尾启动流中启动的所有其他网格（及其后代），包括尾启动后执行或启动的工作。

// In this example, C will only launch after all X, F and P complete.
__global__ void P( ... ) {
   C<<< ... , cudaStreamTailLaunch >>>( ... );
   X<<< ... , cudaStreamPerThread >>>( ... );
   F<<< ... , cudaStreamFireAndForget >>>( ... )
}

在父网格的尾部启动工作完成之前，父网格流中的下一个网格将不会启动。换句话说，尾部启动流的行为就像它被插入其父网格和父网格流中的下一个网格之间。

// In this example, P2 will only launch after C completes.
__global__ void P1( ... ) {
   C<<< ... , cudaStreamTailLaunch >>>( ... );
}

__global__ void P2( ... ) {
}

int main ( ... ) {
   ...
   P1<<< ... >>>( ... );
   P2<<< ... >>>( ... );
   ...
}

每个网格只得到一个尾部发射流。要尾部启动并发网格，可以像下面的示例一样完成。

// In this example,  C1 and C2 will launch concurrently after P's completion
__global__ void T( ... ) {
   C1<<< ... , cudaStreamFireAndForget >>>( ... );
   C2<<< ... , cudaStreamFireAndForget >>>( ... );
}

__global__ void P( ... ) {
   ...
   T<<< ... , cudaStreamTailLaunch >>>( ... );
}

尾部启动流不能用于记录或等待事件。尝试这样做会导致`cudaErrorInvalidValue`。当定义了`CUDA_FORCE_CDP1_IF_SUPPORTED`时，不支持尾部启动流。尾部启动流的使用需要以64位模式进行编译。

#### 13.3.1.3.事件[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events-cdp "这个标题的永久链接")

仅支持CUDA事件的流间同步功能。这意味着支持`cudaStreamWaitEvent()`），但不支持`cudaEventSynchronize()``cudaEventElapsedTime()`和`cudaEventQuery()`）。由于不支持`cudaEventElapsedTime()`），因此必须通过`cudaEventCreateWithFlags()`创建cudaEvents，传递`cudaEventDisableTiming`标志。

与命名流一样，事件对象可能在创建事件的网格中的所有线程之间共享，但该线程是本地的，可能不会传递给其他内核。事件句柄不能保证在网格之间是唯一的，因此在网格中使用未创建的事件句柄将导致未定义的行为。

#### 13.3.1.4.同步[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-programming-interface "这个标题的永久链接")

如果调用线程旨在与其他线程调用的子网格同步，则由程序执行足够的线程间同步，例如通过CUDA事件。

由于不可能从父线程明确同步子工作，因此没有办法保证子网格中发生的更改对父网格中的线程可见。

#### 13.3.1.5.设备管理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-management-programming "这个标题的永久链接")

Only the device on which a kernel is running will be controllable from that kernel. This means that device APIs such as `cudaSetDevice()` are not supported by the device runtime. The active device as seen from the GPU (returned from `cudaGetDevice()`) will have the same device number as seen from the host system. The `cudaDeviceGetAttribute()` call may request information about another device as this API allows specification of a device ID as a parameter of the call. Note that the catch-all `cudaGetDeviceProperties()` API is not offered by the device runtime - properties must be queried individually.

#### 13.3.1.6.记忆声明[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-declarations "这个标题的永久链接")

##### 13.3.1.6.1.设备和恒定内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-and-constant-memory "这个标题的永久链接")

使用设备运行时，在文件范围内使用`__device__`或`__constant__`内存空间指定符的内存行为相同。所有内核都可以读取或写入设备变量，无论内核最初是由主机还是设备运行时启动的。同样，所有内核都将具有与模块范围内声明的`__constant__`s相同的视图。

##### 13.3.1.6.2.纹理和表面[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#textures-and-surfaces "这个标题的永久链接")

CUDA支持动态创建的纹理和表面对象[7](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn14)，其中可以在主机上创建纹理对象，传递给内核，由该内核使用，然后从主机中销毁。设备运行时不允许在设备代码中创建或销毁纹理或表面对象，但可以从主机创建的纹理和表面对象在设备上自由使用和传递。无论它们在哪里创建，动态创建的纹理对象始终有效，并且可以从父内核传递给子内核。

笔记

设备运行时不支持从设备启动的内核内的传统模块范围（即费米风格）纹理和表面。模块范围（传统）纹理可以从主机创建，并在设备代码中使用，就像任何内核一样，但只能由顶级内核（即从主机启动的内核）使用。

##### 13.3.1.6.3.共享内存变量声明[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-variable-declarations "这个标题的永久链接")

在CUDA中，C++共享内存可以声明为静态大小的文件范围或函数范围变量，也可以声明为运行时由内核调用者通过启动配置参数确定大小的`extern`变量。两种类型的声明在设备运行时都有效。

__global__ void permute(int n, int *data) {
   extern __shared__ int smem[];
   if (n <= 1)
       return;

   smem[threadIdx.x] = data[threadIdx.x];
   __syncthreads();

   permute_data(smem, n);
   __syncthreads();

   // Write back to GMEM since we can't pass SMEM to children.
   data[threadIdx.x] = smem[threadIdx.x];
   __syncthreads();

   if (threadIdx.x == 0) {
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
   }
}

void host_launch(int *data) {
    permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
}

##### 13.3.1.6.4.符号地址[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#symbol-addresses "这个标题的永久链接")

设备端符号（即标记为`__device__`的符号）可以通过`&`运算符从内核内引用，因为所有全局范围设备变量都在内核的可见地址空间中。这也适用于`__constant__`符号，尽管在这种情况下，指针将引用只读数据。

鉴于设备端符号可以直接引用，引用符号（例如，`cudaMemcpyToSymbol()`或`cudaGetSymbolAddress()`的CUDA运行时API是多余的，因此设备运行时不支持。请注意，这意味着即使在子内核启动之前，也无法从运行内核内更改常量数据，因为对`__constant__`空间的引用是只读的。

#### 13.3.1.7.API错误和启动失败[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-errors-and-launch-failures "这个标题的永久链接")

和CUDA运行时一样，任何函数都可能返回错误代码。最后一个返回的错误代码被记录下来，可以通过thecudaGetLastError`cudaGetLastError()`调用检索。错误按线程记录，以便每个线程可以识别它生成的最新错误。错误代码类型为`cudaError_t`。

与主机端启动类似，设备端启动可能会因多种原因（参数无效等）而失败。用户必须调用`cudaGetLastError()`来确定启动是否产生了错误，但是启动后没有错误并不意味着子内核成功完成。

对于设备端的例外情况，例如，访问无效地址，子网格中的错误将返回给主机。

##### 13.3.1.7.1.启动设置API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-setup-apis "这个标题的永久链接")

内核启动是通过设备运行时库公开的系统级机制，因此可以通过底层`cudaGetParameterBuffer()`和`cudaLaunchDevice()`API直接从PTX获得。允许CUDA应用程序自行调用这些API，其要求与PTX相同。在这两种情况下，用户都有责任根据规范以正确的格式正确填充所有必要的数据结构。这些数据结构保证了向后兼容性。

与主机端启动一样，设备端运算符`<<<>>>`映射到底层内核启动API。这样，针对PTX的用户将能够执行启动，并且编译器前端可以将`<<<>>>`翻译成这些调用。

表13新的设备专用启动实现功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id471 "此表的永久链接")
|运行时API启动功能|与主机运行时行为的不同描述（如果没有描述，则行为是相同的）|
|---|---|
|`cudaGetParameterBuffer`|从`<<<>>>`自动生成。注意与主机等效的API不同。|
|`cudaLaunchDevice`|从`<<<>>>`自动生成。注意与主机等效的API不同。|

这些启动函数的API与CUDA运行时API的API不同，定义如下：

extern   device   cudaError_t cudaGetParameterBuffer(void **params);
extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                        void *params, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize = 0,
                                        cudaStream_t stream = 0);

#### 13.3.1.8.API参考[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-reference "这个标题的永久链接")

此处详细介绍了设备运行时中支持的CUDA运行时API的部分。主机和设备运行时API具有相同的语法；除非另有说明，否则语义是相同的。下表概述了与主机提供的版本相关的API。

表14支持的API功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id472 "此表的永久链接")
|运行时API函数|详情|
|---|---|
|`cudaDeviceGetCacheConfig`||
|`cudaDeviceGetLimit`||
|`cudaGetLastError`|最后一个错误是每个线程状态，而不是每个块状态|
|`cudaPeekAtLastError`||
|`cudaGetErrorString`||
|`cudaGetDeviceCount`||
|`cudaDeviceGetAttribute`|将返回任何设备的属性|
|`cudaGetDevice`|始终返回从主机看到的当前设备ID|
|`cudaStreamCreateWithFlags`|必须通过`cudaStreamNonBlocking`标志|
|`cudaStreamDestroy`||
|`cudaStreamWaitEvent`||
|`cudaEventCreateWithFlags`|必须通过`cudaEventDisableTiming`标志|
|`cudaEventRecord`||
|`cudaEventDestroy`||
|`cudaFuncGetAttributes`||
|`cudaMemcpyAsync`|关于所有`memcpy/memset`函数的说明：<br><br>- 仅支持异步`memcpy/set`函数<br>    <br>- 仅允许设备到设备的`memcpy`<br>    <br>- 可能无法传递本地或共享内存指针|
|`cudaMemcpy2DAsync`|
|`cudaMemcpy3DAsync`|
|`cudaMemsetAsync`|
|`cudaMemset2DAsync`||
|`cudaMemset3DAsync`||
|`cudaRuntimeGetVersion`||
|`cudaMalloc`|不得在设备上在主机上创建的指针上调用`cudaFree`，反之亦然|
|`cudaFree`|
|`cudaOccupancyMaxActiveBlocksPerMultiprocessor`||
|`cudaOccupancyMaxPotentialBlockSize`||
|`cudaOccupancyMaxPotentialBlockSizeVariableSMem`||

### 13.3.2.来自PTX的设备端启动[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-side-launch-from-ptx "这个标题的永久链接")

本节适用于针对并行_线程执行_（PTX）并计划在其语言中支持_动态并行_的编程语言和编译器实现者。它提供了与支持PTX级别的内核启动相关的低级细节。

#### 13.3.2.1.内核启动API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernel-launch-apis "这个标题的永久链接")

可以使用从PTX访问的以下两个API实现设备端内核启动：`cudaLaunchDevice()`和`cudaGetParameterBuffer()``cudaLaunchDevice()`使用参数缓冲区启动指定的内核，该参数是通过调用`cudaGetParameterBuffer()`获得的，并填充启动内核的参数。参数缓冲区可以是NULL，即如果启动的内核不接受任何参数，则无需调用`cudaGetParameterBuffer()`

##### 13.3.2.1.1. 库达朗克设备[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudalaunchdevice "这个标题的永久链接")

在PTX级别，`cudaLaunchDevice()`在使用之前需要以下所示的两种形式之一声明。

// PTX-level Declaration of cudaLaunchDevice() when .address_size is 64
.extern .func(.param .b32 func_retval0) cudaLaunchDevice
(
  .param .b64 func,
  .param .b64 parameterBuffer,
  .param .align 4 .b8 gridDimension[12],
  .param .align 4 .b8 blockDimension[12],
  .param .b32 sharedMemSize,
  .param .b64 stream
)
;

下面的CUDA级声明映射到上述PTX级声明之一，并位于系统标题文件`cuda_device_runtime_api.h`中。该函数在`cudadevrt`系统库中定义，它必须与程序链接才能使用设备端内核启动功能。

// CUDA-level declaration of cudaLaunchDevice()
extern "C" __device__
cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer,
                             dim3 gridDimension, dim3 blockDimension,
                             unsigned int sharedMemSize,
                             cudaStream_t stream);

第一个参数是指向要启动的内核的指针，第二个参数是将实际参数保存到启动的内核的参数缓冲区。参数缓冲区的布局在下面的[参数缓冲区布局](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parameter-buffer-layout)中进行了解释。其他参数指定启动配置，即网格维度、块维度、共享内存大小和与启动相关的流（有关启动配置的详细说明，请参阅[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)。

##### 13.3.2.1.2. cudaGet参数缓冲区[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudagetparameterbuffer "这个标题的永久链接")

`cudaGetParameterBuffer()`在使用之前，需要在PTX级别声明。根据地址大小，PTX级声明必须以以下两种形式之一：

// PTX-level Declaration of cudaGetParameterBuffer() when .address_size is 64
.extern .func(.param .b64 func_retval0) cudaGetParameterBuffer
(
  .param .b64 alignment,
  .param .b64 size
)
;

以下`cudaGetParameterBuffer()`的CUDA级声明映射到上述PTX级声明：

// CUDA-level Declaration of cudaGetParameterBuffer()
extern "C" __device__
void *cudaGetParameterBuffer(size_t alignment, size_t size);

第一个参数指定了参数缓冲区的对齐要求，第二个参数指定了以字节为单位的大小要求。在当前实现中，`cudaGetParameterBuffer()`返回的参数缓冲区始终保证为64字节对齐，对齐要求参数被忽略。然而，建议将正确的对齐要求值（这是放置在参数缓冲区中的任何参数的最大对齐）传递给`cudaGetParameterBuffer()`以确保未来的可移植性。

#### 13.3.2.2.参数缓冲区布局[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parameter-buffer-layout "这个标题的永久链接")

禁止在参数缓冲区中重新排序参数，参数缓冲区中的每个参数都需要对齐。也就是说，每个参数必须放在参数缓冲区的第n个字节上，其中_n_是大于前一个参数取的最后一个字节偏移量的参数大小的最小倍数。参数缓冲区的最大大小为4KB。

有关CUDA编译器生成的PTX代码的更详细描述，请参阅PTX-3.5规范。

### 13.3.3.动态并行的工具包支持[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#toolkit-support-for-dynamic-parallelism "这个标题的永久链接")

#### 13.3.3.1.在CUDA代码中包括设备运行时API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#including-device-runtime-api-in-cuda-code "这个标题的永久链接")

与主机端运行时API类似，CUDA设备运行时API的原型在程序编译期间自动包含。不需要明确包含`cuda_device_runtime_api.h`。

#### 13.3.3.2.编译和链接[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compiling-and-linking "这个标题的永久链接")

当使用`nvcc`的动态并行性编译和链接CUDA程序时，该程序将自动与静态设备运行时库`libcudadevrt`链接。

设备运行时作为静态库提供（Windows上的`cudadevrt.lib`，Linux下的`libcudadevrt.a`），必须链接使用设备运行时的GPU应用程序。设备库的链接可以通过`nvcc`和/或`nvlink`完成。下面显示了两个简单的例子。

如果可以从命令行指定所有必需的源文件，则可以一步编译和链接设备运行时程序：

$ nvcc -arch=sm_75 -rdc=true hello_world.cu -o hello -lcudadevrt

也可以先将CUDA .cu源文件编译为对象文件，然后在两个阶段的过程中将它们链接在一起：

$ nvcc -arch=sm_75 -dc hello_world.cu -o hello_world.o
$ nvcc -arch=sm_75 -rdc=true hello_world.o -o hello -lcudadevrt

有关更多详细信息，请参阅CUDA驱动程序编译器NVCC指南的“使用单独编译”部分。

## 13.4.编程指南[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-guidelines "这个标题的永久链接")

### 13.4.1.基础[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#basics "这个标题的永久链接")

设备运行时是主机运行时的功能子集。API級裝置管理、核心啟動、裝置memcpy、流管理和事件管理從裝置執行時暴露出來。

已经使用CUDA经验的人应该熟悉设备运行时的编程。设备运行时语法和语义与主机API基本相同，但本文档前面详细说明的任何例外情况除外。

以下示例展示了一个包含动态并行性的简单_Hello World_程序：

#include <stdio.h>

__global__ void childKernel()
{
    printf("Hello ");
}

__global__ void tailKernel()
{
    printf("World!\n");
}

__global__ void parentKernel()
{
    // launch child
    childKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return;
    }

    // launch tail into cudaStreamTailLaunch stream
    // implicitly synchronizes: waits for child to complete
    tailKernel<<<1,1,0,cudaStreamTailLaunch>>>();

}

int main(int argc, char *argv[])
{
    // launch parent
    parentKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return 1;
    }

    // wait for parent to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
        return 2;
    }

    return 0;
}

该程序可以从命令行一步构建，如下所示：

$ nvcc -arch=sm_75 -rdc=true hello_world.cu -o hello -lcudadevrt

### 13.4.2.表演[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance "这个标题的永久链接")

#### 13.4.2.1.动态并行支持内核头顶[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dynamic-parallelism-enabled-kernel-overhead "这个标题的永久链接")

控制动态启动时处于活动状态的系统软件可能会对当时正在运行的任何内核施加开销，无论它是否调用自己的内核启动。此开销源于设备运行时的执行跟踪和管理软件，并可能导致性能下降。一般来说，与设备运行时库链接的应用程序会产生这种开销。

### 13.4.3.实施限制和限制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#implementation-restrictions-and-limitations "这个标题的永久链接")

_动态并行性_保证了本文档中描述的所有语义，但是，某些硬件和软件资源依赖于实现，并限制了使用设备运行时的程序的规模、性能和其他属性。

#### 13.4.3.1.运行时间[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#runtime "这个标题的永久链接")

##### 13.4.3.1.1.记忆足迹[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-footprint "这个标题的永久链接")

设备运行时系统软件为各种管理目的保留内存，特别是用于跟踪待定网格启动的保留。配置控制可用于减少此保留的规模，以换取某些发射限制。有关详细信息，请参阅下面的[配置选项](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#configuration-options)。

##### 13.4.3.1.2.等待内核启动[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pending-kernel-launches "这个标题的永久链接")

当内核启动时，所有相关的配置和参数数据都会被跟踪，直到内核完成。这些数据存储在系统管理的发射池中。

可以通过从主机调用`cudaDeviceSetLimit()`并指定`cudaLimitDevRuntimePendingLaunchCount`来配置固定大小启动池的大小。

##### 13.4.3.1.3.配置选项[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#configuration-options "这个标题的永久链接")

设备运行时系统软件的资源分配是通过主机程序的`cudaDeviceSetLimit()`API控制的。必须在启动任何内核之前设置限制，并且在GPU正在积极运行程序时不得更改。

可以设置以下指定限制：

|限制|行为|
|---|---|
|`cudaLimitDevRuntimePendingLaunchCount`|控制为缓冲内核启动和因未解决的依赖关系或缺乏执行资源而尚未开始执行的事件预留的内存量。当缓冲区已满时，在设备端内核启动期间分配启动插槽的尝试将失败并返回`cudaErrorLaunchOutOfResources`，而分配事件插槽的尝试将失败并返回`cudaErrorMemoryAllocation`。启动插槽的默认数量是2048。应用程序可以通过设置`cudaLimitDevRuntimePendingLaunchCount`来增加启动和/或活动时段的数量。分配的事件时段数量是该限制值的两倍。|
|`cudaLimitStackSize`|控制每个GPU线程的字节堆栈大小。CUDA驱动程序会根据需要自动增加每个内核启动的每线程堆栈大小。每次启动后，此大小不会重置为原始值。要将每个线程堆栈大小设置为不同的值，可以调用`cudaDeviceSetLimit()`来设置此限制。堆栈将立即调整大小，如有必要，设备将阻止，直到所有先前请求的任务都完成。可以调用`cudaDeviceGetLimit()`来获取当前的每线程堆栈大小。|

##### 13.4.3.1.4.内存分配和寿命[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-allocation-and-lifetime "这个标题的永久链接")

`cudaMalloc()`和`cudaFree()`在主机和设备环境之间具有不同的语义。当从主机调用时，`cudaMalloc()`从未使用的设备内存中分配一个新区域。当从设备运行时调用时，这些函数映射到设备端的`malloc()`和`free()`这意味着在设备环境中，总可分配内存仅限于设备`malloc()`堆大小，该堆可能小于可用的未使用设备内存。此外，在设备上由`cudaMalloc()`分配的指针上从主机程序调用`cudaFree()`是一个错误，反之亦然。

||`cudaMalloc()`在主机上|`cudaMalloc()`在设备上|
|---|---|---|
|`cudaFree()`在主机上|支持的|不支持|
|`cudaFree()`在设备上|不支持|支持的|
|分配限制|可用设备内存|`cudaLimitMallocHeapSize`|

##### 13.4.3.1.5.SM ID和Warp Id[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sm-id-and-warp-id "这个标题的永久链接")

请注意，在PTX中，`%smid`和`%warpid`被定义为挥发性值。设备运行时可能会将线程块重新安排到不同的SM上，以便更有效地管理资源。因此，依赖`%smid`或`%warpid`在线程或线程块的生命周期内保持不变是不安全的。

##### 13.4.3.1.6.ECC错误[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ecc-errors "这个标题的永久链接")

在CUDA内核中没有ECC错误的通知。一旦整个启动树完成，主机端就会报告ECC错误。在执行嵌套程序期间出现的任何ECC错误将生成异常或继续执行（取决于错误和配置）。

## 13.5.CDP2与CDP1[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cdp2-vs-cdp1 "这个标题的永久链接")

本节总结了新（CDP2）和旧（CDP1）CUDA动态并行接口之间的差异、兼容性和互操作性。它还展示了如何在计算能力小于9.0的设备上选择退出CDP2接口。

### 13.5.1.CDP1和CDP2之间的区别[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#differences-between-cdp1-and-cdp2 "这个标题的永久链接")

在CDP2或计算能力9.0或更高版本的设备上，不再能够进行显式设备端同步。必须改用隐式同步（如尾部发射）。

尝试使用CDP2或在计算能力9.0或更高版本的设备上查询或设置`cudaLimitDevRuntimeSyncDepth`（或`CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH`）会导致`cudaErrorUnsupportedLimit`。

CDP2不再有虚拟化池，用于不适合固定大小池的待定启动。`cudaLimitDevRuntimePendingLaunchCount`必须设置为足够大，以避免启动插槽耗尽。

对于CDP2，一次存在的事件总数是有限制的（请注意，事件只有在启动完成后才会被销毁），相当于待定启动计数的两倍。`cudaLimitDevRuntimePendingLaunchCount`必须设置为足够大，以避免事件插槽耗尽。

流是使用CDP2或计算能力9.0或更高的设备上每个网格跟踪的，而不是每个线程块。这允许将工作启动到另一个线程块创建的流中。尝试使用CDP1这样做会导致`cudaErrorInvalidValue`。

CDP2引入了名为tail launch（cudaStreamTailLaunch）和fire-and-forget（`cudaStreamFireAndForget`）的流。

仅在64位编译模式下支持CDP2。

### 13.5.2.兼容性和互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compatibility-and-interoperability "这个标题的永久链接")

CDP2是默认值。可以使用`-DCUDA_FORCE_CDP1_IF_SUPPORTED`编译函数，以选择在计算能力小于9.0的设备上选择不使用CDP2。

||带有CUDA 12.0及更新版本的函数编译器（默认）|使用CUDA 12.0前或CUDA 12.0及更新版本编译的函数，指定了`-DCUDA_FORCE_CDP1_IF_SUPPORTED`|
|---|---|---|
|编辑|如果设备代码引用`cudaDeviceSynchronize`编译错误。|如果代码引用`cudaStreamTailLaunch`或`cudaStreamFireAndForget`，则编译错误。如果设备代码引用`cudaDeviceSynchronize`，并且代码为sm_90或更新程序编译，则编译错误。|
|计算能力<9.0|使用了新的界面。|使用了传统界面。|
|计算能力9.0及更高|使用了新的界面。|使用了新的界面。如果函数在设备代码中引用`cudaDeviceSynchronize`，函数加载将返回`cudaErrorSymbolNotFound`（如果代码是为计算能力小于9.0的设备编译的，但使用JIT在计算能力为9.0或更高的设备上运行，则可能会发生这种情况）。|

使用CDP1和CDP2的函数可以在同一上下文中同时加载和运行。CDP1函数可以使用CDP1特定功能（例如`cudaDeviceSynchronize`），CDP2函数可以使用CDP2特定功能（例如尾部启动和触发并忘记启动）。

使用CDP1的函数不能启动使用CDP2的函数，反之亦然。如果使用CDP1的函数在其调用图中包含一个使用CDP2的函数，反之亦然，`cudaErrorCdpVersionMismatch`将在函数加载期间产生。

## 13.6.传统CUDA动态并行性（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#legacy-cuda-dynamic-parallelism-cdp1 "这个标题的永久链接")

请参阅上面的[CUDA动态并行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-dynamic-parallelism)，了解CDP2版本的文档。

### 13.6.1.执行环境和内存模型（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-environment-and-memory-model-cdp1 "这个标题的永久链接")

请参阅上面的[执行环境和内存模型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-environment-and-memory-model-cdp2)，了解文档的CDP2版本。

#### 13.6.1.1.执行环境 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-environment-cdp1 "这个标题的永久链接")

请参阅上面的[执行环境](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-environment-cdp2)，了解文档的CDP2版本。

CUDA执行模型基于线程、线程块和网格的原语，内核函数定义了线程块和网格中单个线程执行的程序。当调用内核函数时，网格的属性由执行配置描述，该配置在CUDA中具有特殊的语法。CUDA对动态并行支持扩展了在设备上运行的线程上配置、启动和同步新网格的能力。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`块在CUDA 11.6中已弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

##### 13.6.1.1.1.父网格和子网格（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parent-and-child-grids-cdp1 "这个标题的永久链接")

请参阅上面的[父网格和子网格](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parent-and-child-grids-cdp2)，以获取文档的 CDP2 版本。

配置和启动新网格的设备线程属于父网格，调用创建的网格是子网格。

子网格的调用和完成是正确嵌套的，这意味着在其线程创建的所有子网格完成之前，父网格不被视为完成。即使调用线程在启动的子网格上没有显式同步，运行时也保证了父和子网格之间的隐式同步。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

[![GPU将更多的晶体管用于数据处理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/parent-child-launch-nesting.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/parent-child-launch-nesting.png)

图31 父子启动嵌套[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parent-child-launch-nesting "此图像的永久链接")

##### 13.6.1.1.2.CUDA原始人（CDP1）的范围[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#scope-of-cuda-primitives-cdp1 "这个标题的永久链接")

请参阅上面的[CUDA原始文件范围](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#scope-of-cuda-primitives-cdp2)，了解CDP2版本的文件。

在主机和设备上，CUDA运行时提供了一个API，用于启动内核，等待启动工作完成，以及通过流和事件跟踪启动之间的依赖关系。在主机系统上，启动状态和引用流和事件的CUDA原语由进程中的所有线程共享；但是进程独立执行，可能不共享CUDA对象。

设备上存在类似的层次结构：启动的内核和CUDA对象对线程块中的所有线程可见，但线程块之间是独立的。这意味着，例如，流可以由一个线程创建，并由同一线程块中的任何其他线程使用，但不得与任何其他线程块中的线程共享。

##### 13.6.1.1.3.同步（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-cdp1 "这个标题的永久链接")

请参阅上面的[同步](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dynamic-parallelism-synchronization)，以获取文档的CDP2版本。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

来自任何线程的CUDA运行时操作，包括内核启动，都可以跨线程块看到。这意味着父网格中的调用线程可能会在该线程启动的网格、线程块中的其他线程或在同一线程块中创建的流上执行同步。在线程块中所有线程的所有启动完成之前，线程块的执行不被视为完成。如果在所有子启动完成之前，块中的所有线程退出，同步操作将自动触发。

##### 13.6.1.1.4.流和事件 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams-and-events-cdp1 "这个标题的永久链接")

請參閱上面的[流和事件](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams-and-events-cdp2)，以瞭解CDP2版本的文件。

CUDA_流_和_事件_允许控制网格启动之间的依赖关系：启动到同一流中的网格按顺序执行，事件可用于在流之间创建依赖关系。在设备上创建的流和事件具有完全相同的目的。

在网格中创建的流和事件存在于线程块范围内，但当在创建它们的线程块之外使用时，具有未定义的行为。如上所述，线程块启动的所有工作在块退出时都隐式同步；启动到流中的工作都包含在内，所有依赖项都得到了适当的解决。在线程块范围之外修改的流上的操作行为是未定义的。

在任何内核中使用时，在主机上创建的流和事件具有未定义的行为，就像由父网格创建的流和事件在子网格中使用时具有未定义的行为一样。

##### 13.6.1.1.5.订购和并发（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ordering-and-concurrency-cdp1 "这个标题的永久链接")

请参阅上面的[“订购和并发](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ordering-and-concurrency-cdp2)”，了解文档的 CDP2 版本。

从设备运行时开始的内核启动顺序遵循CUDA流排序语义。在线程块中，所有内核启动到同一流中都是按顺序执行的。当同一线程块中的多个线程启动到同一流中时，流内的排序取决于块内的线程调度，该调度可以通过`__syncthreads()`等同步原语进行控制。

请注意，由于流由线程块内的所有线程共享，因此隐式_NULL_流也共享。如果线程块中的多个线程启动到隐式流中，那么这些启动将按顺序执行。如果需要并发，则应使用显式命名流。

_动态并行性_使并发性更容易在程序中表达；然而，设备运行时在CUDA执行模型中没有引入新的并发性保证。不能保证设备上任何数量的不同线程块之间同时执行。

缺乏并发保证延伸到父线程块及其子网格。当父线程块启动子网格时，在父线程块达到显式同步点（例如`cudaDeviceSynchronize()`之前，不能保证子线程开始执行。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

虽然并发通常很容易实现，但它可能会因设备配置、应用程序工作负载和运行时调度而有所不同。因此，依赖不同线程块之间的任何并发是不安全的。

##### 13.6.1.1.6.设备管理 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-management-cdp1 "这个标题的永久链接")

请参阅上面的[设备管理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-management-programming)，了解CDP2版本的文档。

设备运行时不支持多GPU；设备运行时只能在它当前正在执行的设备上运行。然而，允许查询系统中任何支持CUDA的设备的属性。

#### 13.6.1.2.内存型号（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-model-cdp1 "这个标题的永久链接")

请参阅上面的[内存模型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-model)，了解文档的CDP2版本。

父网格和子网格共享相同的全局和常量内存存储，但具有不同的本地和共享内存。

##### 13.6.1.2.1.连贯性和一致性（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#coherence-and-consistency-cdp1 "这个标题的永久链接")

请参阅上面的[一致性和一致性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#coherence-and-consistency-cdp2)，了解CDP2版本的文档。

###### 13.6.1.2.1.1.全局内存（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-cdp1 "这个标题的永久链接")

请参阅上面的“[全局内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-cdp2)”，了解文档的 CDP2 版本。

父网格和子网格对全局内存有一致的访问，子网格和父网格之间的一致性保证较弱。当子网格的内存视图与父线程完全一致时，子网格的执行有两个点：当子网格被父网格调用时，以及当子网格完成时，由父线程中的同步API调用发出信号。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

在子网格调用之前，父线程中的所有全局内存操作对子网格可见。在子网格完成时同步后，子网格的所有内存操作对父网格可见。

在以下示例中，执行`child_launch`的子网格只能保证看到在子网格启动之前对`data`所做的修改。由于父线程0正在执行启动，因此子线程将与父线程0看到的内存一致。由于第一次`__syncthreads()`呼叫，子程序将看到`data[0]=0`，`data[1]=1`，...，`data[255]=255`（如果没有`__syncthreads()`调用，只能保证子程序看到`data[0]`）。当子网格返回时，保证线程0会看到线程在其子网格中所做的修改。只有在第二个`__syncthreads()`调用后，父网格的其他线程才能使用这些修改：

__global__ void child_launch(int *data) {
   data[threadIdx.x] = data[threadIdx.x]+1;
}

__global__ void parent_launch(int *data) {
   data[threadIdx.x] = threadIdx.x;

   __syncthreads();

   if (threadIdx.x == 0) {
       child_launch<<< 1, 256 >>>(data);
       cudaDeviceSynchronize();
   }

   __syncthreads();
}

void host_launch(int *data) {
    parent_launch<<< 1, 256 >>>(data);
}

###### 13.6.1.2.1.2.零复制内存 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#zero-copy-memory-cdp1 "这个标题的永久链接")

请参阅上文的[零拷贝内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#zero-copy-memory)，以获取CDP2版本的文档。

零复制系统内存与全局内存具有相同的一致性和一致性保证，并遵循上述详述语义。内核可能不会分配或释放零复制内存，但可以使用指针从主机程序传递的零复制。

###### 13.6.1.2.1.3.恒定内存（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant-memory-cdp1 "这个标题的永久链接")

请参阅上面的“[恒定内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant-memory)”，以获取文档的CDP2版本。

常量是不可变的，即使在父和子启动之间，也不可能从设备中修改。也就是说，在启动之前，必须从主机上设置all`__constant__`变量的值。恒定内存由所有子内核从各自的父内核那里自动继承。

从内核线程中获取常量内存对象的地址与所有CUDA程序具有相同的语义，自然支持将指针从父程序传递到子程序或从子程序传递给父程序。

###### 13.6.1.2.1.4.共享和本地内存（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-and-local-memory-cdp1 "这个标题的永久链接")

请参阅上面的[共享和本地内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-and-local-memory-cdp2)，了解文档的CDP2版本。

共享和本地内存分别是线程块或线程的私有内存，父和子内存不可见或一致。当这些位置之一的对象被引用到其所属范围之外时，行为是未定义的，并可能导致错误。

NVIDIA编译器将尝试警告是否可以检测到指向本地或共享内存的指针正在作为参数传递给内核启动。在运行时，程序员可以使用`__isGlobal()`内在来确定指针是否引用全局内存，因此可以安全地传递给子启动。

请注意，调用`cudaMemcpy*Async()`或`cudaMemset*Async()`可能会在设备上调用新的子内核，以保留流语义。因此，将共享或本地内存指针传递给这些API是非法的，并且会返回错误。

###### 13.6.1.2.1.5.本地内存（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#local-memory-cdp1 "这个标题的永久链接")

请参阅上面的[本地内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#local-memory-cdp2)，以获取CDP2版本的文档。

本地内存是执行线程的私有存储，在该线程之外不可见。在启动子内核时，将指针传递给本地内存作为启动参数是非法的。从子程序取消引用此类本地内存指针的结果将是未定义的。

例如，如果`child_launch`访问`x_array`则以下内容是非法的，行为未定义：

int x_array[10];       // Creates x_array in parent's local memory
child_launch<<< 1, 1 >>>(x_array);

程序员有时很难意识到编译器何时将变量放入本地内存中。作为一般规则，传递给子内核的所有存储都应从全局内存堆中显式分配，无论是使用`cudaMalloc()``new()`还是通过在全局范围内声明`__device__`存储。例如：

// Correct - "value" is global storage
__device__ int value;
__device__ void x() {
    value = 5;
    child<<< 1, 1 >>>(&value);
}

// Invalid - "value" is local storage
__device__ void y() {
    int value = 5;
    child<<< 1, 1 >>>(&value);
}

###### 13.6.1.2.1.6.纹理记忆（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory-cdp1 "这个标题的永久链接")

请参阅上面的[纹理内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory-cdp)，以获取文档的CDP2版本。

写入纹理映射的全局内存区域与纹理访问不连贯。纹理内存的一致性在调用子网格时和子网格完成后强制执行。这意味着在子内核启动之前写入内存会反映在子内核的纹理内存访问中。同样，子程序写入内存将反映在父级访问的纹理内存中，但只有在父级在子级完成时同步后。父母和子女同时访问可能会导致数据不一致。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

### 13.6.2.编程接口（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface-cdp1 "这个标题的永久链接")

请参阅上面的[编程界面](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface-cdp)，了解CDP2版本的文档。

#### 13.6.2.1.CUDA C++参考（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-reference-cdp1 "这个标题的永久链接")

请参阅上面的[CUDA C++参考](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-reference)，了解CDP2版本的文档。

本节介绍CUDA C++语言扩展的更改和添加，以支持_动态并行_。

使用CUDA C++进行动态并行的CUDA内核可用的语言接口和API，称为_设备运行时_，与主机上可用的CUDA运行时API基本相似。在可能的情况下，保留了CUDA运行时API的语法和语义，以便于在主机或设备环境中运行的例程的代码重复使用。

与CUDA C++中的所有代码一样，这里概述的API和代码是每线程代码。这使每个线程能够就接下来要执行的内核或操作做出独特的动态决定。块内线程之间没有同步要求来执行任何提供的设备运行时API，这使得设备运行时API函数可以在任意发散的内核代码中调用，而不会死锁。

##### 13.6.2.1.1.设备侧内核启动（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-side-kernel-launch-cdp1 "这个标题的永久链接")

请参阅上面的[内核启动API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id237)，了解文档的CDP2版本。

内核可以使用标准CUDA <<< >>>语法从设备启动：

kernel_name<<< Dg, Db, Ns, S >>>([kernel arguments]);

- `Dg`类型为`dim3`，并指定网格的尺寸和大小
    
- `Db`类型为`dim3`，并指定每个线程块的尺寸和大小
    
- `Ns`类型为`size_t`，并指定为该调用为每个线程块动态分配的共享内存字节数，并添加到静态分配的内存中。`Ns`是一个可选参数，默认为0。
    
- `S`是`cudaStream_t`类型，并指定与此调用关联的流。流必须分配到正在调用的同一线程块中。`S`是一个可选参数，默认为0。
    

###### 13.6.2.1.1.1.发射是非同步的（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launches-are-asynchronous-cdp1 "这个标题的永久链接")

请参阅上面的[“启动是异步](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launches-are-asynchronous)的”，以获取CDP2版本的文档。

与主机端启动相同，所有设备端内核启动与启动线程是异步的。也就是说，`<<<>>>`启动命令将立即返回，启动线程将继续执行，直到它达到明确的启动同步点，如`cudaDeviceSynchronize()`

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

网格启动发布到设备上，并将独立于父线程执行。子网格可以在启动后随时开始执行，但在启动线程达到明确的启动同步点之前不能保证开始执行。

###### 13.6.2.1.1.2.启动环境配置 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-environment-configuration-cdp1 "这个标题的永久链接")

请参阅上面的[启动环境配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-environment-configuration)，以获取CDP2版本的文档。

所有全局设备配置设置（例如，从`cudaDeviceGetCacheConfig()`返回的共享内存和L1缓存大小，以及从`cudaDeviceGetLimit()`返回的设备限制）都将从父级继承。同样，堆栈大小等设备限制将保持配置不变。

对于主机启动的内核，从主机设置的每个内核配置将优先于全局设置。当内核从设备启动时，也会使用这些配置。无法从设备重新配置内核环境。

##### 13.6.2.1.2.流（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams-cdp1 "这个标题的永久链接")

请参阅上面的[流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams-cdp)，了解CDP2版本的文档。

命名和未命名（NULL）流都可以从设备运行时获得。线程块内的任何线程都可以使用命名流，但流句柄不得传递给其他块或子/父内核。换句话说，流应该被视为创建它的块的私有。流句柄不能保证在块之间是唯一的，因此在未分配的块中使用流句柄会导致未定义的行为。

与主机端启动类似，启动到单独流中的工作可能会同时运行，但不能保证实际的并发性。CUDA编程模型不支持依赖子内核之间并发性的程序，并且将具有未定义的行为。

设备不支持主机端NULL流的跨流屏障语义（详情见下文）。为了保持与主机运行时的语义兼容性，必须使用`cudaStreamCreateWithFlags()`API创建所有设备流，并传递`cudaStreamNonBlocking`标志。`cudaStreamCreate()`调用是主机运行时仅API，将无法为设备编译。

由于设备运行时不支持`cudaStreamSynchronize()`和`cudaStreamQuery()`当应用程序需要知道流启动的子内核已完成时，应改用`cudaDeviceSynchronize()`。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

###### 13.6.2.1.2.1.隐式（空）流（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-implicit-null-stream-cdp1 "这个标题的永久链接")

请参阅上面的[隐式（NULL）流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-implicit-null-stream)，了解文档的CDP2版本。

在主机程序中，未命名（NULL）流与其他流有额外的屏障同步语义（详情请参阅[默认流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#default-stream)）。设备运行时提供了一个在块中所有线程之间共享的单个隐式、未命名的流，但由于所有命名流都必须使用`cudaStreamNonBlocking`标志创建，因此启动到NULL流中的工作不会在任何其他流（包括其他线程块的NULL流）中插入对待定工作的隐式依赖。

##### 13.6.2.1.3.事件（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events-cdp1 "这个标题的永久链接")

请参阅上面的[事件](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events-cdp)，了解CDP2版本的文档。

仅支持CUDA事件的流间同步功能。这意味着支持`cudaStreamWaitEvent()`），但不支持`cudaEventSynchronize()``cudaEventElapsedTime()`和`cudaEventQuery()`）。由于不支持`cudaEventElapsedTime()`），因此必须通过`cudaEventCreateWithFlags()`创建cudaEvents，传递`cudaEventDisableTiming`标志。

对于所有设备运行时对象，事件对象可以在创建它们的线程块内的所有线程之间共享，但该线程是本地的，不得传递给其他内核，或在同一内核内的块之间。事件句柄不能保证在块之间是唯一的，因此在没有创建事件句柄的块中使用事件句柄会导致未定义的行为。

##### 13.6.2.1.4.同步（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-programming-interface-cdp1 "这个标题的永久链接")

请参阅上面的[同步](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-programming-interface)，以获取文档的CDP2版本。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

`cudaDeviceSynchronize()`函数将同步线程块中任何线程启动的所有工作，直到调用`cudaDeviceSynchronize()`的点。请注意，`cudaDeviceSynchronize()`可以从发散代码中调用（请参阅[区块范围同步（CDP1）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#block-wide-synchronization-cdp1)）。

如果调用线程旨在与其他线程调用的子网格同步，则由程序执行足够的额外线程间同步，例如通过调用`__syncthreads()`。

###### 13.6.2.1.4.1.区块宽同步（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#block-wide-synchronization-cdp1 "这个标题的永久链接")

请参阅上面的[CUDA动态并行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-dynamic-parallelism)，了解CDP2版本的文档。

`cudaDeviceSynchronize()`函数并不意味着块内同步。特别是，如果没有通过a__syncthreads`__syncthreads()`指令进行显式同步，调用线程无法假设除自身以外的任何线程启动了哪些工作。例如，如果一个块中的多个线程是每个启动工作，并且需要同时同步所有工作（也许是因为基于事件的依赖关系），那么在调用`cudaDeviceSynchronize()`之前，由程序来保证所有线程提交此工作。

由于该实现允许在从块中的任何线程启动时同步，因此很有可能由多个线程同时调用tocudaDeviceSynchronize`cudaDeviceSynchronize()`将耗尽第一个调用中的所有工作，然后对随后的调用没有影响。

##### 13.6.2.1.5.设备管理 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-management-programming-cdp1 "这个标题的永久链接")

请参阅上面的[设备管理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-management-programming)，了解CDP2版本的文档。

Only the device on which a kernel is running will be controllable from that kernel. This means that device APIs such as `cudaSetDevice()` are not supported by the device runtime. The active device as seen from the GPU (returned from `cudaGetDevice()`) will have the same device number as seen from the host system. The `cudaDeviceGetAttribute()` call may request information about another device as this API allows specification of a device ID as a parameter of the call. Note that the catch-all `cudaGetDeviceProperties()` API is not offered by the device runtime - properties must be queried individually.

##### 13.6.2.1.6.内存声明（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-declarations-cdp1 "这个标题的永久链接")

请参阅上面的[内存声明](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-declarations)，[以](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-declarations)获取文档的CDP2版本。

###### 13.6.2.1.6.1.设备和恒定内存（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-and-constant-memory-cdp1 "这个标题的永久链接")

请参阅上面的[设备和常量内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-and-constant-memory)，了解文档的CDP2版本。

使用设备运行时，在文件范围内使用`__device__`或`__constant__`内存空间指定符的内存行为相同。所有内核都可以读取或写入设备变量，无论内核最初是由主机还是设备运行时启动的。同样，所有内核都将具有与模块范围内声明的`__constant__`s相同的视图。

###### 13.6.2.1.6.2.纹理和表面（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#textures-and-surfaces-cdp1 "这个标题的永久链接")

请参阅上面的[纹理和表面](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#textures-and-surfaces)，以获取文档的CDP2版本。

CUDA支持动态创建的纹理和表面对象[7](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn14)，其中可以在主机上创建纹理对象，传递给内核，由该内核使用，然后从主机中销毁。设备运行时不允许在设备代码中创建或销毁纹理或表面对象，但可以从主机创建的纹理和表面对象在设备上自由使用和传递。无论它们在哪里创建，动态创建的纹理对象始终有效，并且可以从父内核传递给子内核。

笔记

设备运行时不支持从设备启动的内核内的传统模块范围（即费米风格）纹理和表面。模块范围（传统）纹理可以从主机创建，并在设备代码中使用，就像任何内核一样，但只能由顶级内核（即从主机启动的内核）使用。

###### 13.6.2.1.6.3.共享内存变量声明（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-variable-declarations-cdp1 "这个标题的永久链接")

请参阅上面的[共享内存变量声明](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-variable-declarations)，了解文档的CDP2版本。

在CUDA中，C++共享内存可以声明为静态大小的文件范围或函数范围变量，也可以声明为运行时由内核调用者通过启动配置参数确定大小的`extern`变量。两种类型的声明在设备运行时都有效。

__global__ void permute(int n, int *data) {
   extern __shared__ int smem[];
   if (n <= 1)
       return;

   smem[threadIdx.x] = data[threadIdx.x];
   __syncthreads();

   permute_data(smem, n);
   __syncthreads();

   // Write back to GMEM since we can't pass SMEM to children.
   data[threadIdx.x] = smem[threadIdx.x];
   __syncthreads();

   if (threadIdx.x == 0) {
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
       permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
   }
}

void host_launch(int *data) {
    permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
}

###### 13.6.2.1.6.4.符号地址（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#symbol-addresses-cdp1 "这个标题的永久链接")

请参阅上面的[符号地址](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#symbol-addresses)，了解CDP2版本的文档。

设备端符号（即标记为`__device__`的符号）可以通过`&`运算符从内核内引用，因为所有全局范围设备变量都在内核的可见地址空间中。这也适用于`__constant__`符号，尽管在这种情况下，指针将引用只读数据。

鉴于设备端符号可以直接引用，引用符号（例如，`cudaMemcpyToSymbol()`或`cudaGetSymbolAddress()`的CUDA运行时API是多余的，因此设备运行时不支持。请注意，这意味着即使在子内核启动之前，也无法从运行内核内更改常量数据，因为对`__constant__`空间的引用是只读的。

##### 13.6.2.1.7.API错误和启动失败（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-errors-and-launch-failures-cdp1 "这个标题的永久链接")

请参阅上面的[API错误和启动失败](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-errors-and-launch-failures)，[以](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-errors-and-launch-failures)获取CDP2版本的文档。

和CUDA运行时一样，任何函数都可能返回错误代码。最后一个返回的错误代码被记录下来，可以通过thecudaGetLastError`cudaGetLastError()`调用检索。错误按线程记录，以便每个线程可以识别它生成的最新错误。错误代码类型为`cudaError_t`。

与主机端启动类似，设备端启动可能会因多种原因（参数无效等）而失败。用户必须调用`cudaGetLastError()`来确定启动是否产生了错误，但是启动后没有错误并不意味着子内核成功完成。

对于设备端的例外情况，例如，访问无效地址，子网格中的错误将返回给主机，而不是由父网格调用`cudaDeviceSynchronize()`返回。

###### 13.6.2.1.7.1.启动设置 API (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-setup-apis-cdp1 "这个标题的永久链接")

有关 CDP2 版本的文档，请参阅上面的[“启动设置 API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-setup-apis)”。

内核启动是通过设备运行时库公开的系统级机制，因此可以通过底层`cudaGetParameterBuffer()`和`cudaLaunchDevice()`API直接从PTX获得。允许CUDA应用程序自行调用这些API，其要求与PTX相同。在这两种情况下，用户都有责任根据规范以正确的格式正确填充所有必要的数据结构。这些数据结构保证了向后兼容性。

与主机端启动一样，设备端运算符`<<<>>>`映射到底层内核启动API。这样，针对PTX的用户将能够执行启动，并且编译器前端可以将`<<<>>>`翻译成这些调用。

表15新的仅设备启动实现功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id473 "此表的永久链接")
|运行时API启动功能|与主机运行时行为的不同描述（如果没有描述，则行为是相同的）|
|---|---|
|`cudaGetParameterBuffer`|从`<<<>>>`自动生成。注意与主机等效的API不同。|
|`cudaLaunchDevice`|从`<<<>>>`自动生成。注意与主机等效的API不同。|

这些启动函数的API与CUDA运行时API的API不同，定义如下：

extern   device   cudaError_t cudaGetParameterBuffer(void **params);
extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                        void *params, dim3 gridDim,
                                        dim3 blockDim,
                                        unsigned int sharedMemSize = 0,
                                        cudaStream_t stream = 0);

##### 13.6.2.1.8.API 参考 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-reference-cdp1 "这个标题的永久链接")

请参阅上面的[API参考](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-reference-cdp2)，了解CDP2版本的文档。

此处详细介绍了设备运行时中支持的CUDA运行时API的部分。主机和设备运行时API具有相同的语法；除非另有说明，否则语义是相同的。下表概述了与主机提供的版本相关的API。

表16支持的API功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id474 "此表的永久链接")
|运行时API函数|详情|
|---|---|
|`cudaDeviceSynchronize`|仅从线程自己的块启动的工作同步。<br><br>警告：请注意，从设备代码调用此API在CUDA 11.6中已弃用，用于compute_90+编译，并计划在未来的CUDA版本中完全删除。|
|`cudaDeviceGetCacheConfig`||
|`cudaDeviceGetLimit`||
|`cudaGetLastError`|最后一个错误是每个线程状态，而不是每个块状态|
|`cudaPeekAtLastError`||
|`cudaGetErrorString`||
|`cudaGetDeviceCount`||
|`cudaDeviceGetAttribute`|将返回任何设备的属性|
|`cudaGetDevice`|始终返回从主机看到的当前设备ID|
|`cudaStreamCreateWithFlags`|必须通过`cudaStreamNonBlocking`标志|
|`cudaStreamDestroy`||
|`cudaStreamWaitEvent`||
|`cudaEventCreateWithFlags`|必须通过`cudaEventDisableTiming`标志|
|`cudaEventRecord`||
|`cudaEventDestroy`||
|`cudaFuncGetAttributes`||
|`cudaMemcpyAsync`|关于所有`memcpy/memset`函数的说明：<br><br>- 仅支持异步`memcpy/set`函数<br>    <br>- 仅允许设备到设备的`memcpy`<br>    <br>- 可能无法传递本地或共享内存指针|
|`cudaMemcpy2DAsync`|
|`cudaMemcpy3DAsync`|
|`cudaMemsetAsync`|
|`cudaMemset2DAsync`||
|`cudaMemset3DAsync`||
|`cudaRuntimeGetVersion`||
|`cudaMalloc`|不得在设备上在主机上创建的指针上调用`cudaFree`，反之亦然|
|`cudaFree`|
|`cudaOccupancyMaxActiveBlocksPerMultiprocessor`||
|`cudaOccupancyMaxPotentialBlockSize`||
|`cudaOccupancyMaxPotentialBlockSizeVariableSMem`||

#### 13.6.2.2.来自PTX（CDP1）的设备端启动[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-side-launch-from-ptx-cdp1 "这个标题的永久链接")

请参阅上面的[PTX设备端启动](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-side-launch-from-ptx-cdp2)，了解文档的CDP2版本。

本节适用于针对并行_线程执行_（PTX）并计划在其语言中支持_动态并行_的编程语言和编译器实现者。它提供了与支持PTX级别的内核启动相关的低级细节。

##### 13.6.2.2.1.内核启动API（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernel-launch-apis-cdp1 "这个标题的永久链接")

请参阅上面的[内核启动API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id237)，了解文档的CDP2版本。

可以使用从PTX访问的以下两个API实现设备端内核启动：`cudaLaunchDevice()`和`cudaGetParameterBuffer()``cudaLaunchDevice()`使用参数缓冲区启动指定的内核，该参数是通过调用`cudaGetParameterBuffer()`获得的，并填充启动内核的参数。参数缓冲区可以是NULL，即如果启动的内核不接受任何参数，则无需调用`cudaGetParameterBuffer()`

###### 13.6.2.2.1.1. cudaLaunch设备（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudalaunchdevice-cdp1 "这个标题的永久链接")

请参阅上面的[cudaLaunchDevice](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudalaunchdevice-cdp2)，了解CDP2版本的文档。

在PTX级别，`cudaLaunchDevice()`在使用之前需要以下所示的两种形式之一声明。

// PTX-level Declaration of cudaLaunchDevice() when .address_size is 64
.extern .func(.param .b32 func_retval0) cudaLaunchDevice
(
  .param .b64 func,
  .param .b64 parameterBuffer,
  .param .align 4 .b8 gridDimension[12],
  .param .align 4 .b8 blockDimension[12],
  .param .b32 sharedMemSize,
  .param .b64 stream
)
;

// PTX-level Declaration of cudaLaunchDevice() when .address_size is 32
.extern .func(.param .b32 func_retval0) cudaLaunchDevice
(
  .param .b32 func,
  .param .b32 parameterBuffer,
  .param .align 4 .b8 gridDimension[12],
  .param .align 4 .b8 blockDimension[12],
  .param .b32 sharedMemSize,
  .param .b32 stream
)
;

下面的CUDA级声明映射到上述PTX级声明之一，并位于系统标题文件`cuda_device_runtime_api.h`中。该函数在`cudadevrt`系统库中定义，它必须与程序链接才能使用设备端内核启动功能。

// CUDA-level declaration of cudaLaunchDevice()
extern "C" __device__
cudaError_t cudaLaunchDevice(void *func, void *parameterBuffer,
                             dim3 gridDimension, dim3 blockDimension,
                             unsigned int sharedMemSize,
                             cudaStream_t stream);

第一个参数是指向要启动的内核的指针，第二个参数是将实际参数保存到启动的内核的参数缓冲区。参数缓冲区布局[（CDP1）在](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parameter-buffer-layout-cdp1)下面的[参数缓冲区布局](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parameter-buffer-layout-cdp1)中进行了解释。其他参数指定启动配置，即网格维度、块维度、共享内存大小和与启动相关的流（有关启动配置的详细说明，请参阅[执行配置](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)。

###### 13.6.2.2.1.2. cudaGetParameterBuffer（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudagetparameterbuffer-cdp1 "这个标题的永久链接")

请参阅上面的[cudaGetParameterBuffer](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudagetparameterbuffer-cdp2)，了解CDP2版本的文档。

`cudaGetParameterBuffer()`在使用之前，需要在PTX级别声明。根据地址大小，PTX级声明必须以以下两种形式之一：
```c++
// PTX-level Declaration of cudaGetParameterBuffer() when .address_size is 64
// When .address_size is 64
.extern .func(.param .b64 func_retval0) cudaGetParameterBuffer
(
  .param .b64 alignment,
  .param .b64 size
)
;

// PTX-level Declaration of cudaGetParameterBuffer() when .address_size is 32
.extern .func(.param .b32 func_retval0) cudaGetParameterBuffer
(
  .param .b32 alignment,
  .param .b32 size
)
;
```
以下`cudaGetParameterBuffer()`的CUDA级声明映射到上述PTX级声明：

// CUDA-level Declaration of cudaGetParameterBuffer()
extern "C" __device__
void *cudaGetParameterBuffer(size_t alignment, size_t size);

第一个参数指定了参数缓冲区的对齐要求，第二个参数指定了以字节为单位的大小要求。在当前实现中，`cudaGetParameterBuffer()`返回的参数缓冲区始终保证为64字节对齐，对齐要求参数被忽略。然而，建议将正确的对齐要求值（这是放置在参数缓冲区中的任何参数的最大对齐）传递给`cudaGetParameterBuffer()`以确保未来的可移植性。

##### 13.6.2.2.2.参数缓冲区布局（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parameter-buffer-layout-cdp1 "这个标题的永久链接")

请参阅上面的[参数缓冲区布局](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#parameter-buffer-layout)，了解文档的CDP2版本。

禁止在参数缓冲区中重新排序参数，参数缓冲区中的每个参数都需要对齐。也就是说，每个参数必须放在参数缓冲区的第n个字节上，其中_n_是大于前一个参数取的最后一个字节偏移量的参数大小的最小倍数。参数缓冲区的最大大小为4KB。

有关CUDA编译器生成的PTX代码的更详细描述，请参阅PTX-3.5规范。

#### 13.6.2.3.动态并行工具包支持（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#toolkit-support-for-dynamic-parallelism-cdp1 "这个标题的永久链接")

请参阅上面的[动态并行性工具包支持](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#toolkit-support-for-dynamic-parallelism)，以获取文档的CDP2版本。

##### 13.6.2.3.1.在CUDA代码（CDP1）中包含设备运行时API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#including-device-runtime-api-in-cuda-code-cdp1 "这个标题的永久链接")

有关文档的CDP2版本，请参阅上面的[CUDA代码中包含设备运行时API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#including-device-runtime-api-in-cuda-code-cdp2)。

与主机端运行时API类似，CUDA设备运行时API的原型在程序编译期间自动包含。无需明确包含`cuda_device_runtime_api.h`。

##### 13.6.2.3.2.编译和链接（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compiling-and-linking-cdp1 "这个标题的永久链接")

有关CDP2版本的文档，请参阅上面的[编译和链接](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compiling-and-linking)。

当使用`nvcc`的动态并行性编译和链接CUDA程序时，该程序将自动与静态设备运行时库`libcudadevrt`链接。

设备运行时作为静态库提供（Windows上的`cudadevrt.lib`，Linux下的`libcudadevrt.a`），必须链接使用设备运行时的GPU应用程序。设备库的链接可以通过`nvcc`和/或`nvlink`完成。下面显示了两个简单的例子。

如果可以从命令行指定所有必需的源文件，则可以一步编译和链接设备运行时程序：

$ nvcc -arch=sm_75 -rdc=true hello_world.cu -o hello -lcudadevrt

也可以先将CUDA .cu源文件编译为对象文件，然后在两个阶段的过程中将它们链接在一起：

$ nvcc -arch=sm_75 -dc hello_world.cu -o hello_world.o
$ nvcc -arch=sm_75 -rdc=true hello_world.o -o hello -lcudadevrt

有关更多详细信息，请参阅CUDA驱动程序编译器NVCC指南的“使用单独编译”部分。

### 13.6.3.编程指南（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-guidelines-cdp1 "这个标题的永久链接")

请参阅上面的[编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-guidelines)，了解CDP2版本的文档。

#### 13.6.3.1.基础知识（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#basics-cdp1 "这个标题的永久链接")

请参阅上面的[基础知识](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#basics)，了解文档的CDP2版本。

设备运行时是主机运行时的功能子集。API級裝置管理、核心啟動、裝置memcpy、流管理和事件管理從裝置執行時暴露出來。

已经使用CUDA经验的人应该熟悉设备运行时的编程。设备运行时语法和语义与主机API基本相同，但本文档前面详细说明的任何例外情况除外。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

以下示例展示了一个包含动态并行性的简单_Hello World_程序：

#include <stdio.h>

__global__ void childKernel()
{
    printf("Hello ");
}

__global__ void parentKernel()
{
    // launch child
    childKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return;
    }

    // wait for child to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
        return;
    }

    printf("World!\n");
}

int main(int argc, char *argv[])
{
    // launch parent
    parentKernel<<<1,1>>>();
    if (cudaSuccess != cudaGetLastError()) {
        return 1;
    }

    // wait for parent to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
        return 2;
    }

    return 0;
}

该程序可以从命令行一步构建，如下所示：

$ nvcc -arch=sm_75 -rdc=true hello_world.cu -o hello -lcudadevrt

#### 13.6.3.2.绩效（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-cdp1 "这个标题的永久链接")

请参阅上面的[性能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance)，了解CDP2版本的文档。

##### 13.6.3.2.1.同步（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-performance-cdp1 "这个标题的永久链接")

请参阅上面的[CUDA动态并行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-dynamic-parallelism)，了解CDP2版本的文档。

警告

与父块的子内核的显式同步（例如在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中已弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

一个线程的同步可能会影响同一线_程块_中其他线程的性能，即使这些其他线程不调用`cudaDeviceSynchronize()`本身。这种影响将取决于基本实施。一般来说，与显式调用`cudaDeviceSynchronize()`相比，线程块结束时对子内核进行隐式同步更有效。因此，建议仅在线程块结束前需要与子内核同步时调用`cudaDeviceSynchronize()`）。

##### 13.6.3.2.2.动态并行支持内核架空（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dynamic-parallelism-enabled-kernel-overhead-cdp1 "这个标题的永久链接")

请参阅上文的CDP2版本文档，以[启用动态并行性的内核架空](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dynamic-parallelism-enabled-kernel-overhead)。

控制动态启动时处于活动状态的系统软件可能会对当时正在运行的任何内核施加开销，无论它是否调用自己的内核启动。这种开销源于设备运行时的执行跟踪和管理软件，并可能导致性能下降，例如，与从主机端相比，从设备进行库调用。一般来说，与设备运行时库链接的应用程序会产生这种开销。

#### 13.6.3.3.实施限制和限制（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#implementation-restrictions-and-limitations-cdp1 "这个标题的永久链接")

请参阅上面的[实施限制和限制](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#implementation-restrictions-and-limitations)，以了解CDP2版本的文档。

_动态并行性_保证了本文档中描述的所有语义，但是，某些硬件和软件资源依赖于实现，并限制了使用设备运行时的程序的规模、性能和其他属性。

##### 13.6.3.3.1.运行时（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#runtime-cdp1 "这个标题的永久链接")

请参阅上面的[运行时](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#runtime)，了解文档的CDP2版本。

###### 13.6.3.3.1.1.内存足迹（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-footprint-cdp1 "这个标题的永久链接")

请参阅上面的[内存足迹](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-footprint)，以获取CDP2版本的文档。

设备运行时系统软件为各种管理目的保留内存，特别是用于在同步期间保存父网格状态的保留，以及用于跟踪待定网格启动的第二个保留。配置控件可用于减少这些保留的大小，以换取某些发射限制。有关详细信息，请参阅下面的[配置选项 (CDP1)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#configuration-options-cdp1)。

大部分保留内存被分配为父内核状态的备份存储，用于在子启动时同步。保守地说，该内存必须支持存储设备上可能的最大实时线程数量的状态。这意味着，`cudaDeviceSynchronize()`可调用的每个父代可能需要高达860MB的设备内存，这取决于设备配置，即使没有全部消耗，也无法用于程序使用。

###### 13.6.3.3.1.2.嵌套和同步深度（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nesting-and-synchronization-depth-cdp1 "这个标题的永久链接")

请参阅上面的[CUDA动态并行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-dynamic-parallelism)，了解CDP2版本的文档。

使用设备运行时，一个内核可以启动另一个内核，该内核可以启动另一个内核，以此以此为。每个从属发射都被认为是一个新的_嵌套级别_，级别总数是程序的_嵌套深度_。_同步深度_被定义为程序在子启动时显式同步的最深级别。通常，这比程序的嵌套深度少一个，但如果程序不需要在所有级别调用`cudaDeviceSynchronize()`那么同步深度可能与嵌套深度有很大不同。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

总体最大嵌套深度限制为24，但实际上，实际限制将是系统在每个新级别所需的内存量（见上文内[存容量（CDP1）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-footprint-cdp1)）。任何会导致核心深度高于最大值的启动都会失败。请注意，这也可能适用于`cudaMemcpyAsync()`它本身可能会生成内核启动。有关详细信息，请参阅[配置选项 (CDP1)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#configuration-options-cdp1)。

默认情况下，为两个级别的同步保留了足够的存储空间。可以通过调用`cudaDeviceSetLimit()`并指定`cudaLimitDevRuntimeSyncDepth`来控制这个最大同步深度（以及保留的存储）。在从主机启动顶级内核之前，必须配置要支持的级别数量，以保证嵌套程序的成功执行。在大于指定最大同步深度的深度调用`cudaDeviceSynchronize()`将返回错误。

如果父内核从不调用`cudaDeviceSynchronize()`系统检测到它不需要为父状态保留空间，则允许优化。在这种情况下，由于从未发生过显式父/子同步，程序所需的内存占用空间将遠遠低於保守的最大值。这样的程序可以指定更浅的最大同步深度，以避免备份存储的过度分配。

###### 13.6.3.3.1.3.等待内核启动 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pending-kernel-launches-cdp1 "这个标题的永久链接")

请参阅上面的[等待核心启动](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pending-kernel-launches)，以瞭解CDP2版本的文件。

当内核启动时，所有相关的配置和参数数据都会被跟踪，直到内核完成。这些数据存储在系统管理的发射池中。

发射池分为固定大小的池和性能较低的虚拟化池。设备运行时系统软件将首先尝试在固定大小的池中跟踪发射数据。当固定大小的池已满时，虚拟化池将用于跟踪新发布。

可以通过从主机调用`cudaDeviceSetLimit()`并指定`cudaLimitDevRuntimePendingLaunchCount`来配置固定大小启动池的大小。

###### 13.6.3.3.1.4.配置选项 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#configuration-options-cdp1 "这个标题的永久链接")

请参阅上面的[配置选项](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#configuration-options)，了解文档的CDP2版本。

设备运行时系统软件的资源分配是通过主机程序的`cudaDeviceSetLimit()`API控制的。必须在启动任何内核之前设置限制，并且在GPU正在积极运行程序时不得更改。

警告

与父块的子内核的显式同步（即在设备代码中使用`cudaDeviceSynchronize()`在CUDA 11.6中被弃用，为compute_90+编译而删除，并计划在未来的CUDA版本中完全删除。

可以设置以下指定限制：

|限制|行为|
|---|---|
|`cudaLimitDevRuntimeSyncDepth`|设置可以调用`cudaDeviceSynchronize()`的最大深度。启动可能会比这更深，但比这限制更深的显式同步将返回`cudaErrorLaunchMaxDepthExceeded`。默认最大同步深度为2。|
|`cudaLimitDevRuntimePendingLaunchCount`|控制为缓冲内核启动预留的内存量，这些内核启动由于未解决的依赖关系或缺乏执行资源而尚未开始执行。当缓冲区已满时，设备运行时系统软件将尝试在性能较低的虚拟化缓冲区中跟踪新的待定启动。如果虚拟化缓冲区也已满，即当所有可用的堆空间都被消耗时，不会发生启动，线程的最后一个错误将设置为tocudaErrorLaunchPendingCountExceeded。默认的待定启动计数为2048次启动。|
|`cudaLimitStackSize`|控制每个GPU线程的字节堆栈大小。CUDA驱动程序会根据需要自动增加每个内核启动的每线程堆栈大小。每次启动后，此大小不会重置为原始值。要将每个线程堆栈大小设置为不同的值，可以调用`cudaDeviceSetLimit()`来设置此限制。堆栈将立即调整大小，如有必要，设备将阻止，直到之前请求的所有任务完成。可以调用`cudaDeviceGetLimit()`来获取当前的每线程堆栈大小。|

###### 13.6.3.3.1.5.内存分配和寿命（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-allocation-and-lifetime-cdp1 "这个标题的永久链接")

请参阅上面的[内存分配和寿命](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-allocation-and-lifetime)，以获取文档的CDP2版本。

`cudaMalloc()`和`cudaFree()`在主机和设备环境之间具有不同的语义。当从主机调用时，`cudaMalloc()`从未使用的设备内存中分配一个新区域。当从设备运行时调用时，这些函数映射到设备端的`malloc()`和`free()`这意味着在设备环境中，总可分配内存仅限于设备`malloc()`堆大小，该堆可能小于可用的未使用设备内存。此外，在设备上由`cudaMalloc()`分配的指针上从主机程序调用`cudaFree()`是一个错误，反之亦然。

||`cudaMalloc()`在主机上|`cudaMalloc()`在设备上|
|---|---|---|
|`cudaFree()`在主机上|支持的|不支持|
|`cudaFree()`在设备上|不支持|支持的|
|分配限制|可用设备内存|`cudaLimitMallocHeapSize`|

###### 13.6.3.3.1.6.SM Id和Warp Id（CDP1）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sm-id-and-warp-id-cdp1 "这个标题的永久链接")

请参阅上面的[SM ID和Warp Id](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sm-id-and-warp-id)，以获取文档的CDP2版本。

请注意，在PTX中，`%smid`和`%warpid`被定义为挥发性值。设备运行时可能会将线程块重新安排到不同的SM上，以便更有效地管理资源。因此，依赖`%smid`或`%warpid`在线程或线程块的生命周期内保持不变是不安全的。

###### 13.6.3.3.3.1.7.ECC 错误 (CDP1)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ecc-errors-cdp1 "这个标题的永久链接")

请参阅上面的[ECC错误](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ecc-errors)，了解CDP2版本的文档。

在CUDA内核中没有ECC错误的通知。一旦整个启动树完成，主机端就会报告ECC错误。在执行嵌套程序期间出现的任何ECC错误将生成异常或继续执行（取决于错误和配置）。

7（1，2，3）

动态创建的纹理和表面物件是CUDA 5.0引入的CUDA内存模型的補充。有关详细信息，请参阅_CUDA编程指南_。

# 14.虚拟内存管理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-memory-management "这个标题的永久链接")

## 14.1.介绍[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#introduction-virtual-memory-management "这个标题的永久链接")

[虚拟内存管理API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html)为应用程序提供了一种直接管理统一虚拟地址空间的方法，CUDA提供的统一虚拟地址空间用于将物理内存映射到GPU可访问的虚拟地址。在CUDA 10.2中引入的这些API还提供了一种与其他进程和图形API（如OpenGL和Vulkan）互操作的新方式，并提供了用户可以调整以适应其应用程序的更新内存属性。

历史上，CUDA编程模型中的内存分配调用（如`cudaMalloc()`会返回指向GPU内存的内存地址。由此获得的地址可以与任何CUDA API或设备内核内使用。然而，分配的内存无法根据用户的内存需求调整大小。为了增加分配的大小，用户必须明确分配一个更大的缓冲区，从初始分配中复制数据，释放它，然后继续跟踪较新分配的地址。这通常会导致应用程序的性能降低和更高的峰值内存利用率。从本质上讲，用户有一个类似malloc的接口来分配GPU内存，但没有相应的realloc来补充它。虚拟内存管理API将地址和内存的概念解耦，并允许应用程序单独处理它们。API允许应用程序在其认为合适的虚拟地址范围内映射和取消映射内存。

如果使用`cudaEnablePeerAccess`启用对等设备访问内存分配，所有过去和未来的用户分配都映射到目标对等设备。这导致用户无意中支付将所有cudaMalloc分配映射到对等设备的运行时成本。然而，在大多数情况下，应用程序仅通过与其他设备共享一些分配来进行通信，并且不需要将所有分配映射到所有设备。借助虚拟内存管理，应用程序可以专门选择某些分配，以便从目标设备访问。

CUDA虚拟内存管理API向用户公开精细控制，用于管理应用程序中的GPU内存。它提供API，让用户：

- 将分配给不同设备的内存放入连续的VA范围内。
    
- 使用特定于平台的机制执行内存共享的进程间通信。
    
- 在支持它们的设备上选择较新的内存类型。
    

为了分配内存，虚拟内存管理编程模型公开了以下功能：

- 分配物理内存。
    
- 保留VA范围。
    
- 将分配的内存映射到VA范围。
    
- 控制映射范围内的访问权限。
    

请注意，本节中描述的API套件需要一个支持UVA的系统。

## 14.2.支持查询[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#query-for-support "这个标题的永久链接")

在尝试使用虚拟内存管理API之前，应用程序必须确保他们想要使用的设备支持CUDA虚拟内存管理。以下代码示例显示了对虚拟内存管理支持的查询：

int deviceSupportsVmm;
CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
if (deviceSupportsVmm != 0) {
    // `device` supports Virtual Memory Management
}

## 14.3.分配物理内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#allocating-physical-memory "这个标题的永久链接")

The first step in memory allocation using Virtual Memory Management APIs is to create a physical memory chunk that will provide a backing for the allocation. In order to allocate physical memory, applications must use the `cuMemCreate` API. The allocation created by this function does not have any device or host mappings. The function argument `CUmemGenericAllocationHandle` describes the properties of the memory to allocate such as the location of the allocation, if the allocation is going to be shared to another process (or other Graphics APIs), or the physical attributes of the memory to be allocated. Users must ensure the requested allocation’s size must be aligned to appropriate granularity. Information regarding an allocation’s granularity requirements can be queried using `cuMemGetAllocationGranularity`. The following code snippet shows allocating physical memory with `cuMemCreate`:

CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;

    size_t granularity = 0;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    // Allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}

The memory allocated by `cuMemCreate` is referenced by the `CUmemGenericAllocationHandle` it returns. This is a departure from the cudaMalloc-style of allocation, which returns a pointer to the GPU memory, which was directly accessible by CUDA kernel executing on the device. The memory allocated cannot be used for any operations other than querying properties using `cuMemGetAllocationPropertiesFromHandle`. In order to make this memory accessible, applications must map this memory into a VA range reserved by `cuMemAddressReserve` and provide suitable access rights to it. Applications must free the allocated memory using the `cuMemRelease` API.

### 14.3.1.可共享内存分配[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shareable-memory-allocations "这个标题的永久链接")

有了`cuMemCreate`，用户现在可以在分配时向CUDA表明，他们已为进程间通信和图形互操作目的指定了特定分配。应用程序可以通过将`CUmemAllocationProp::requestedHandleTypes`设置为平台特定字段来完成此任务。在Windows上，当`CUmemAllocationProp::requestedHandleTypes`设置为`CU_MEM_HANDLE_TYPE_WIN32`应用程序还必须在`CUmemAllocationProp::win32HandleMetaData`中指定LPSECURITYATTRIBUTES属性。此安全属性定义了导出的分配可以转移到其他进程的范围。

CUDA虚拟内存管理API函数不支持其内存的传统进程间通信函数。相反，他们展示了一种使用OS特定句柄的进程间通信的新机制。应用程序可以使用`cuMemExportToShareableHandle`获得与分配相对应的这些OS特定句柄。由此获得的句柄可以通过使用通常的操作系统原生机制进行进程间通信来传输。收件人进程应该使用`cuMemImportFromShareableHandle`导入分配。

在尝试导出使用`cuMemCreate`分配的内存之前，用户必须确保他们查询对请求的句柄类型的支持。以下代码片段以特定于平台的方式说明了对手柄类型支持的查询。

int deviceSupportsIpcHandle;
#if defined(__linux__)
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));
#else
    cuDeviceGetAttribute(&deviceSupportsIpcHandle, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED, device));
#endif

用户应适当设置`CUmemAllocationProp::requestedHandleTypes`，如下所示：

#if defined(__linux__)
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_WIN32;
    prop.win32HandleMetaData = // Windows specific LPSECURITYATTRIBUTES attribute.
#endif

[memMapIpcDrv](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/memMapIPCDrv/)样本可以作为将IPC与虚拟内存管理分配一起使用的示例。

### 14.3.2.记忆类型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-type "这个标题的永久链接")

在CUDA 10.2之前，应用程序没有用户控制的方式来分配某些设备可能支持的任何特殊类型的内存。WithcuMemCreate，应用程序可以使用`CUmemAllocationProp::allocFlags`额外指定内存类型要求，以选择任何特定的内存功能。应用程序还必须确保分配设备支持请求的内存类型。

#### 14.3.2.1.可压缩内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compressible-memory "这个标题的永久链接")

可压缩内存可用于加速对非结构化稀疏和其他可压缩数据模式的数据的访问。压缩可以节省DRAM带宽、L2读取带宽和L2容量，具体取决于正在运行的数据。想要在支持计算数据压缩的设备上分配可压缩内存的应用程序可以通过将`CUmemAllocationProp::allocFlags::compressionType`设置为`CU_MEM_ALLOCATION_COMP_GENERIC`完成。用户必须使用`CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`查询设备是否支持计算数据压缩。以下代码片段说明了查询可压缩内存supportcuDeviceGetAttribute。

int compressionSupported = 0;
cuDeviceGetAttribute(&compressionSupported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, device);

在支持计算数据压缩的设备上，用户必须在分配时间选择加入，如下所示：

prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;

由于硬件资源有限等各种原因，分配可能没有压缩属性，用户需要使用`cuMemGetAllocationPropertiesFromHandle`查询分配内存的属性，并检查压缩属性。

CUmemAllocationProp allocationProp = {};
cuMemGetAllocationPropertiesFromHandle(&allocationProp, allocationHandle);

if (allocationProp.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC)
{
    // Obtained compressible memory allocation
}

## 14.4.保留虚拟地址范围[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#reserving-a-virtual-address-range "这个标题的永久链接")

由于虚拟内存管理使地址和内存的概念是不同的，因此应用程序必须雕刻出一个地址范围，以容纳`cuMemCreate`的内存分配。保留的地址范围必须至少与用户计划放置的所有物理内存分配的大小之和一样大。

应用程序可以通过将适当的参数传递给`cuMemAddressReserve`保留虚拟地址范围。获得的地址范围不会包含与之关联的任何设备或主机物理内存。保留的虚拟地址范围可以映射到属于系统中任何设备的内存块，从而为应用程序提供由属于不同设备的内存支持和映射的连续VA范围。应用程序需要使用`cuMemAddressFree`将虚拟地址范围返回CUDA。在调用`cuMemAddressFree`之前，用户必须确保整个VA范围已取消映射。这些函数在概念上与mmap/munmap（在Linux上）或VirtualAlloc/VirtualFree（在Windows上）函数相似。以下代码片段说明了该函数的用法：

CUdeviceptr ptr;
// `ptr` holds the returned start of virtual address range reserved.
CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0 for default alignment

## 14.5.虚拟别名支持[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-aliasing-support "这个标题的永久链接")

虚拟内存管理API提供了一种使用具有不同虚拟地址的`cuMemMap`的多次调用来创建多个虚拟内存映射或同一分配的“代理”的方法，即所谓的虚拟别名。除非PTX ISA中另有说明，否则在写入设备操作（网格启动、memcpy、memset等）完成之前，写入分配的一个代理被视为与同一内存的任何其他代理不一致且不连贯。在写入设备操作之前出现在GPU上的网格，但在写入设备操作完成后读取的网格也被认为具有不一致和不连贯的代理。

例如，假设设备指针A和B是相同内存分配的虚拟别名，以下片段被视为未定义：

__global__ void foo(char *A, char *B) {
  *A = 0x1;
  printf("%d\n", *B);    // Undefined behavior!  *B can take on either
// the previous value or some value in-between.
}

以下是定义的行为，假设这两个内核是单调的（按流或事件）排序的。

__global__ void foo1(char *A) {
  *A = 0x1;
}

__global__ void foo2(char *B) {
  printf("%d\n", *B);    // *B == *A == 0x1 assuming foo2 waits for foo1
// to complete before launching
}

cudaMemcpyAsync(B, input, size, stream1);    // Aliases are allowed at
// operation boundaries
foo1<<<1,1,0,stream1>>>(A);                  // allowing foo1 to access A.
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);
foo2<<<1,1,0,stream2>>>(B);
cudaStreamWaitEvent(stream3, event);
cudaMemcpyAsync(output, B, size, stream3);  // Both launches of foo2 and
                                            // cudaMemcpy (which both
                                            // read) wait for foo1 (which writes)
                                            // to complete before proceeding

如果在同一内核中需要通过不同的“代理”访问相同的分配，则可以在两个访问之间使用`fence.proxy.alias`。因此，上述示例可以通过内联PTX组装合法化：

__global__ void foo(char *A, char *B) {
  *A = 0x1;
  asm volatile ("fence.proxy.alias;" ::: "memory");
  printf("%d\n", *B);    // *B == *A == 0x1
}

## 14.6.映射内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-memory "这个标题的永久链接")

分配的物理内存和前两个部分中雕刻的虚拟地址空间代表了虚拟内存管理API引入的内存和地址的区别。要使分配的内存可用，用户必须首先将内存放在地址空间中。从`cuMemAddressReserve`获得的地址范围和从`cuMemCreate`或`cuMemImportFromShareableHandle`获得的物理分配必须使用`cuMemMap`相互关联。

用户可以将来自多个设备的分配关联到相邻的虚拟地址范围内，只要它们已经挖出了足够的地址空间。为了分离物理分配和地址范围，用户必须使用`cuMemUnmap`取消映射地址。用户可以将内存映射和取消映射到同一地址范围，只要他们确保不尝试在已映射的VA范围保留上创建映射。以下代码片段说明了该函数的用法：

CUdeviceptr ptr;
// `ptr`: address in the address range previously reserved by cuMemAddressReserve.
// `allocHandle`: CUmemGenericAllocationHandle obtained by a previous call to cuMemCreate.
CUresult result = cuMemMap(ptr, size, 0, allocHandle, 0);

## 14.7.控制访问权限[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#controlling-access-rights "这个标题的永久链接")

虚拟内存管理API使应用程序能够通过访问控制机制明确保护其VA范围。使用`cuMemMap`将分配映射到地址范围的区域不会使地址可访问，如果由CUDA内核访问，将导致程序崩溃。用户必须使用`cuMemSetAccess`功能专门选择访问控制，该功能允许或限制特定设备访问映射地址范围。以下代码片段说明了该函数的用法：

void setAccessOnDevice(int device, CUdeviceptr ptr, size_t size) {
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Make the address accessible
    cuMemSetAccess(ptr, size, &accessDesc, 1);
}

虚拟内存管理公开的访问控制机制允许用户明确他们希望与系统上的其他对等设备共享哪些分配。如前所述，`cudaEnablePeerAccess`强制将所有之前和未来的cudaMalloc分配映射到目标对等设备。在许多情况下，这很方便，因为用户不必担心跟踪系统中每个设备的每个分配的映射状态。但对于关注应用程序性能的用户来说，这种方法[对性能有影响](https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/)。通过分配粒度访问控制，虚拟内存管理公开了一种机制，以最小的开销进行对等映射。

`vectorAddMMAP`样本可以作为使用虚拟内存管理API的示例。

## 14.8.织物记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fabric-memory "这个标题的永久链接")

CUDA 12.4引入了新的VMM分配句柄类型`CU_MEM_HANDLE_TYPE_FABRIC`。在受支持的平台上，如果NVIDIA IMEX守护程序正在运行，则此分配处理类型不仅可以与任何通信机制共享节点内分配，例如MPI，也是节点间。这允许多节点NVLINK系统中的GPU映射同一NVLINK结构中所有其他GPU的内存，即使它们位于不同的节点中，也大大增加了使用NVLINK进行多GPU编程的规模。

### 14.8.1.支持查询[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#querying-fabric-mem-support "这个标题的永久链接")

在尝试使用Fabric Memory之前，应用程序必须确保他们想要使用的设备支持Fabric Memory。以下代码示例显示了对结构内存支持的查询：

int deviceSupportsFabricMem;
CUresult result = cuDeviceGetAttribute(&deviceSupportsFabricMem, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
if (deviceSupportsFabricMem != 0) {
    // `device` supports Fabric Memory
}

除了使用`CU_MEM_HANDLE_TYPE_FABRIC`作为句柄类型，并且不需要OS原生机制进行进程间通信来交换可共享句柄外，与其他分配句柄类型相比，使用Fabric Memory没有区别。

## 14.9.多播支持[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multicast-support "这个标题的永久链接")

[多播对象管理API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST/)提供了一種建立多播对象的方法，並與上述[虛擬記憶體管理API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html/)相結合[，](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html/)允許應用程式在受支援的NVLINK連線的GPU上利用NVLINK SHARP，如果它們與NVSWITCH連線。NVLINK SHARP允许CUDA应用程序利用结构计算来加速广播和与NVSWITCH连接的GPU之间的减少等操作。为此，多个NVLINK连接的GPU组成了一个多播团队，团队的每个GPU都备份了带有物理内存的多播对象。因此，一个由N个GPU组成的多播团队有N个物理副本，每个副本都位于一个参与的GPU上，一个多播对象。使用多播对象映射的[多模因PTX指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/)适用于多播对象的所有副本。

要使用多播对象，应用程序需要

- 查询组播支持
    
- 使用`cuMulticastCreate`创建多播手柄。
    
- 与控制应参与组播团队的GPU的所有进程共享组播手柄。如上所述，这适用于`cuMemExportToShareableHandle`。
    
- 使用`cuMulticastAddDevice`添加所有应该参与组播团队的GPU。
    
- 对于每个参与的GPU，如上所述，将分配给`cuMemCreate`物理内存绑定到多播手柄。在任何设备上绑定内存之前，所有设备都需要添加到组播团队中。
    
- 保留地址范围，映射多播句柄，并设置上述常规单播映射的访问权限。可以将单播和多播映射到同一物理内存。请参阅上面的“[虚拟别名支持](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-aliasing-support)”部分，如何确保多个映射到同一物理内存之间的一致性。
    
- 将[multimem PTX指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/)与组播映射一起使用。
    

[Multi GPU编程模型](https://github.com/NVIDIA/multi-gpu-programming-models/)GitHub存储库中的`multi_node_p2p`示例包含一个使用结构内存（包括多播对象）来利用NVLINK SHARP的完整示例。请注意，此示例适用于NCCL或NVSHMEM等库的开发人员。它展示了像NVSHMEM这样的更高级别的编程模型如何在（多节点）NVLINK域内内部工作。应用程序开发人员通常应该使用更高级别的MPI、NCCL或NVSHMEM接口，而不是此API。

### 14.9.1.支持查询[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#querying-multicast-obj-mem-support "这个标题的永久链接")

在尝试使用组播对象之前，应用程序必须确保它们想要使用的设备支持它们。以下代码示例显示了对结构内存支持的查询：

int deviceSupportsMultiCast;
CUresult result = cuDeviceGetAttribute(&deviceSupportsMultiCast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
if (deviceSupportsMultiCast != 0) {
    // `device` supports Multicast Objects
}

### 14.9.2.分配组播对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#allocating-multicast-objects "这个标题的永久链接")

可以使用`cuMulticastCreate`创建多播对象：

CUmemGenericAllocationHandle createMCHandle(int numDevices, size_t size) {
    CUmemAllocationProp mcProp = {};
    mcProp.numDevices = numDevices;
    mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC; // or on single node CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

    size_t granularity = 0;
    cuMulticastGetGranularity(&granularity, &mcProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    mcProp.size = padded_size;

    // Create Multicast Object this has no devices and no physical memory associated yet
    CUmemGenericAllocationHandle mcHandle;
    cuMulticastCreate(&mcHandle, &mcProp);

    return mcHandle;
}

### 14.9.3.将设备添加到多播对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#add-devices-to-multicast-objects "这个标题的永久链接")

设备可以通过`cuMulticastAddDevice`添加到组播团队：

cuMulticastAddDevice(&mcHandle, device);

在任何设备上的内存绑定到组播对象之前，需要在控制应参与组播团队的所有进程设备上完成此步骤。

### 14.9.4.将内存绑定到组播对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#bind-memory-to-multicast-objects "这个标题的永久链接")

创建多播对象并将所有参与设备添加到多播对象后，需要为每个设备分配`cuMemCreate`的物理内存：

cuMulticastBindMem(mcHandle, mcOffset, memHandle, memOffset, size, 0 /*flags*/);

### 14.9.5.使用组播映射[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#use-multicast-mappings "这个标题的永久链接")

要在CUDA C++中使用多播映射，需要将[multimem PTX指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red/)与Inline PTX Assembly一起使用：

__global__ void all_reduce_norm_barrier_kernel(float* l2_norm,
                                               float* partial_l2_norm_mc,
                                               unsigned int* arrival_counter_uc, unsigned int* arrival_counter_mc,
                                               const unsigned int expected_count) {
    assert( 1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z );
    float l2_norm_sum = 0.0;
#if __CUDA_ARCH__ >= 900

    // atomic reduction to all replicas
    // this can be conceptually thought of as __threadfence_system(); atomicAdd_system(arrival_counter_mc, 1);
    asm volatile ("multimem.red.release.sys.global.add.u32 [%0], %1;" :: "l"(arrival_counter_mc), "n"(1) : "memory");

    // Need a fence between Multicast (mc) and Unicast (uc) access to the same memory `arrival_counter_uc` and `arrival_counter_mc`:
    // - fence.proxy instructions establish an ordering between memory accesses that may happen through different proxies
    // - Value .alias of the .proxykind qualifier refers to memory accesses performed using virtually aliased addresses to the same memory location.
    // from https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar
    asm volatile ("fence.proxy.alias;" ::: "memory");

    // spin wait with acquire ordering on UC mapping till all peers have arrived in this iteration
    // Note: all ranks need to reach another barrier after this kernel, such that it is not possible for the barrier to be unblocked by an
    // arrival of a rank for the next iteration if some other rank is slow.
    cuda::atomic_ref<unsigned int,cuda::thread_scope_system> ac(arrival_counter_uc);
    while (expected_count > ac.load(cuda::memory_order_acquire));

    // Atomic load reduction from all replicas. It does not provide ordering so it can be relaxed.
    asm volatile ("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(l2_norm_sum) : "l"(partial_l2_norm_mc) : "memory");

#else
    #error "ERROR: multimem instructions require compute capability 9.0 or larger."
#endif

    *l2_norm = std::sqrt(l2_norm_sum);
}

# 15.流有序内存分配器[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-ordered-memory-allocator "这个标题的永久链接")

## 15.1.介绍[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-ordered-memory-allocator-intro "这个标题的永久链接")

使用`cudaMalloc`和`cudaFree`管理内存分配会导致GPU在所有执行的CUDA流中同步。流顺序内存分配器使应用程序能够对启动到CUDA流中的其他工作进行排序内存分配和去分配，例如内核启动和非同步副本。这通过利用流排序语义来重复使用内存分配来改善应用程序内存的使用。分配器还允许应用程序控制分配器的内存缓存行为。当设置了适当的发布阈值时，缓存行为允许分配器在应用程序表示愿意接受更大的内存占用时避免对操作系统的昂贵调用。分配器还支持在流程之间轻松安全地共享分配。

对于许多应用程序来说，流有序内存分配器减少了对自定义内存管理抽象的需求，并使其更容易为需要它的应用程序创建高性能自定义内存管理。对于已经拥有自定义内存分配器的应用程序和库，采用流有序内存分配器使多个库能够共享由驱动程序管理的通用内存池，从而减少多余的内存消耗。此外，驱动程序可以根据其对分配器和其他流管理API的认识进行优化。最后，Nsight Compute和下一代CUDA调试器将分配器作为其CUDA 11.3工具包支持的一部分。

## 15.2.支持查询[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-ordered-querying-memory-support "这个标题的永久链接")

用户可以通过调用带有设备属性`cudaDevAttrMemoryPoolsSupported``cudaDeviceGetAttribute()`来确定设备是否支持流有序内存分配器。

从CUDA 11.3开始，IPC内存池支持可以通过`cudaDevAttrMemoryPoolSupportedHandleTypes`设备属性进行查询。之前的驱动程序将返回`cudaErrorInvalidValue`，因为这些驱动程序不知道属性枚舉。
```c++
int driverVersion = 0;
int deviceSupportsMemoryPools = 0;
int poolSupportedHandleTypes = 0;
cudaDriverGetVersion(&driverVersion);
if (driverVersion >= 11020) {
    cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                           cudaDevAttrMemoryPoolsSupported, device);
}
if (deviceSupportsMemoryPools != 0) {
    // `device` supports the Stream Ordered Memory Allocator
}

if (driverVersion >= 11030) {
    cudaDeviceGetAttribute(&poolSupportedHandleTypes,
              cudaDevAttrMemoryPoolSupportedHandleTypes, device);
}
if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
   // Pools on the specified device can be created with posix file descriptor-based IPC
}
```
在查询之前执行驱动程序版本检查，避免在尚未定义属性的驱动程序上遇到`cudaErrorInvalidValue`错误。人们可以使用`cudaGetLastError`来清除错误，而不是避免它。

## 15.3.API基础知识（cudaMallocAsync和cudaFreeAsync）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-fundamentals-cudamallocasync-and-cudafreeasync "这个标题的永久链接")

API `cudaMallocAsync`和`cudaFreeAsync`构成了分配器的核心。`cudaMallocAsync`返回分配，`cudaFreeAsync`释放分配。两个API都接受流参数来定义分配何时开始和停止使用。`cudaMallocAsync`返回的指针值是同步确定的，可用于构建未来的工作。需要注意的是，在确定分配的位置时，`cudaMallocAsync`忽略当前设备/上下文。相反，`cudaMallocAsync`指定的内存池或提供的流来确定驻留设备。最简单的使用模式是将内存分配、使用并释放回同一流中。

void *ptr;
size_t size = 512;
cudaMallocAsync(&ptr, size, cudaStreamPerThread);
// do work using the allocation
kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
// An asynchronous free can be specified without synchronizing the cpu and GPU
cudaFreeAsync(ptr, cudaStreamPerThread);

在分配流以外的流中使用分配时，用户必须保证访问将在分配操作后发生，否则行为是未定义的。用户可以通过同步分配流或使用CUDA事件来同步生产和消费流来做出此保证。

`cudaFreeAsync()`将自由操作插入到流中。用户必须保证在分配操作和分配的任何使用后进行自由操作。此外，在自由操作开始后，任何使用分配都会导致未定义的行为。事件和/或流同步操作应用于保证在释放流开始自由操作之前完成对其他流分配的任何访问。

cudaMallocAsync(&ptr, size, stream1);
cudaEventRecord(event1, stream1);
//stream2 must wait for the allocation to be ready before accessing
cudaStreamWaitEvent(stream2, event1);
kernel<<<..., stream2>>>(ptr, ...);
cudaEventRecord(event2, stream2);
// stream3 must wait for stream2 to finish accessing the allocation before
// freeing the allocation
cudaStreamWaitEvent(stream3, event2);
cudaFreeAsync(ptr, stream3);

用户可以通过`cudaFreeAsync()`释放使用`cudaMalloc()`分配的分配。在免费操作开始之前，用户必须对访问完成做出同样的保证。
```c++
cudaMalloc(&ptr, size);
kernel<<<..., stream>>>(ptr, ...);
cudaFreeAsync(ptr, stream);

The user can free memory allocated with `cudaMallocAsync` with `cudaFree()`. When freeing such allocations through the `cudaFree()` API, the driver assumes that all accesses to the allocation are complete and performs no further synchronization. The user can use `cudaStreamQuery` / `cudaStreamSynchronize` / `cudaEventQuery` / `cudaEventSynchronize` / `cudaDeviceSynchronize` to guarantee that the appropriate asynchronous work is complete and that the GPU will not try to access the allocation.

cudaMallocAsync(&ptr, size,stream);
kernel<<<..., stream>>>(ptr, ...);
// synchronize is needed to avoid prematurely freeing the memory
cudaStreamSynchronize(stream);
cudaFree(ptr);
```
## 15.4.记忆池和cudaMemPool_t[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-pools-and-the-cudamempool-t "这个标题的永久链接")

内存池封装了根据池属性和属性分配和管理的虚拟地址和物理内存资源。内存池的主要方面是它管理的内存类型和位置。

所有对`cudaMallocAsync`的调用都使用内存池的资源。在没有指定内存池的情况下，`cudaMallocAsync`使用提供的流设备的当前内存池。设备的当前内存池可以使用`cudaDeviceSetMempool`设置，并使用`cudaDeviceGetMempool`进行查询。默认情况下（在没有`cudaDeviceSetMempool`调用的情况下），当前内存池是设备的默认内存池。API `cudaMallocFromPoolAsync`和[cudaMallocAsync的c++过载](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb)允许用户指定用于分配的池，而无需将其设置为当前池。API `cudaDeviceGetDefaultMempool`和`cudaMemPoolCreate`为用户提供内存池的句柄。

笔记: 设备的内存池电流将是该设备的本地电流。因此，在不指定内存池的情况下进行分配将始终产生流设备本地的分配。

笔记: `cudaMemPoolSetAttribute`和`cudaMemPoolGetAttribute`控制内存池的属性。

## 15.5.默认/隐含池[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#default-implicit-pools "这个标题的永久链接")

The default memory pool of a device may be retrieved with the `cudaDeviceGetDefaultMempool` API. Allocations from the default memory pool of a device are non-migratable device allocation located on that device. These allocations will always be accessible from that device. The accessibility of the default memory pool may be modified with `cudaMemPoolSetAccess` and queried by `cudaMemPoolGetAccess`. Since the default pools do not need to be explicitly created, they are sometimes referred to as implicit pools. The default memory pool of a device does not support IPC.

## 15.6.明确的池[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#explicit-pools "这个标题的永久链接")

API `cudaMemPoolCreate`创建了一个显式池。这允许应用程序请求超出默认/隐含池提供的分配属性。这些包括IPC功能、最大池大小、驻留在受支持平台上的特定CPU NUMA节点上的分配等属性。

// create a pool similar to the implicit pool on device 0
int device = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = device;
poolProps.location.type = cudaMemLocationTypeDevice;

cudaMemPoolCreate(&memPool, &poolProps));

以下代码片段说明了在有效的CPU NUMA节点上创建支持IPC的内存池的示例。

// create a pool resident on a CPU NUMA node that is capable of IPC sharing (via a file descriptor).
int cpu_numa_id = 0;
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = cpu_numa_id;
poolProps.location.type = cudaMemLocationTypeHostNuma;
poolProps.handleType = cudaMemHandleTypePosixFileDescriptor;

cudaMemPoolCreate(&ipcMemPool, &poolProps));

## 15.7.物理页面缓存行为[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#physical-page-caching-behavior "这个标题的永久链接")

默认情况下，分配器试图最小化池拥有的物理内存。为了尽量减少操作系统调用来分配和释放物理内存，应用程序必须为每个池配置内存占用空间。应用程序可以使用释放阈值属性（`cudaMemPoolAttrReleaseThreshold`）来完成此工作。

释放阈值是池在尝试将内存释放回操作系统之前应该保留的内存量（以字节为本为本）。当内存池保留的内存超过释放阈值字节时，分配器将尝试在下一次调用流、事件或设备同步时将内存释放回操作系统。将释放阈值设置为UINT64_MAX将防止驱动程序在每次同步后尝试缩小池。

Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

将`cudaMemPoolAttrReleaseThreshold`设置为足够高以有效禁用内存池缩减的应用程序可能希望显式缩小内存池的内存占用。`cudaMemPoolTrimTo`允许此类应用程序这样做。在修剪内存池的足迹时，theminBytesToKeep参数允许应用程序保留它期望在后续执行阶段所需的内存量。
```c++
Cuuint64_t setVal = UINT64_MAX;
cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);

// application phase needing a lot of memory from the stream ordered allocator
for (i=0; i<10; i++) {
    for (j=0; j<10; j++) {
        cudaMallocAsync(&ptrs[j],size[j], stream);
    }
    kernel<<<...,stream>>>(ptrs,...);
    for (j=0; j<10; j++) {
        cudaFreeAsync(ptrs[j], stream);
    }
}

// Process does not need as much memory for the next phase.
// Synchronize so that the trim operation will know that the allocations are no
// longer in use.
cudaStreamSynchronize(stream);
cudaMemPoolTrimTo(mempool, 0);

// Some other process/allocation mechanism can now use the physical memory
// released by the trimming operation.
```
## 15.8.资源使用统计[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#resource-usage-statistics "这个标题的永久链接")

在CUDA 11.3中，添加了池属性`cudaMemPoolAttrReservedMemCurrent`、`cudaMemPoolAttrReservedMemHigh`、`cudaMemPoolAttrUsedMemCurrent`和`cudaMemPoolAttrUsedMemHigh`来查询池的内存使用情况。

查询池的`cudaMemPoolAttrReservedMemCurrent`属性报告池当前消耗的物理GPU内存总量。查询池的`cudaMemPoolAttrUsedMemCurrent`将返回池中分配的所有内存的总大小，并且无法重复使用。

The`cudaMemPoolAttr*MemHigh` attributes are watermarks recording the max value achieved by the respective `cudaMemPoolAttr*MemCurrent` attribute since last reset. They can be reset to the current value by using the `cudaMemPoolSetAttribute` API.
```c++
// sample helper functions for getting the usage statistics in bulk
struct usageStatistics {
    cuuint64_t reserved;
    cuuint64_t reservedHigh;
    cuuint64_t used;
    cuuint64_t usedHigh;
};

void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics)
{
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, statistics->reserved);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, statistics->reservedHigh);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, statistics->used);
    cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, statistics->usedHigh);
}

// resetting the watermarks will make them take on the current value.
void resetStatistics(cudaMemoryPool_t memPool)
{
    cuuint64_t value = 0;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
}
```
## 15.9.内存重复使用政策[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-reuse-policies "这个标题的永久链接")

为了服务分配请求，驱动程序在尝试从操作系统中分配更多内存之前，尝试重复使用之前通过`cudaFreeAsync()`释放的内存。例如，流中释放的内存可以立即重复使用，用于同一流中的后续分配请求。同样，当一个流与CPU同步时，之前在该流中释放的内存可以用于任何流中的分配。

流有序分配器有一些可控制的分配策略。池属性`cudaMemPoolReuseFollowEventDependencies`、`cudaMemPoolReuseAllowOpportunistic`和`cudaMemPoolReuseAllowInternalDependencies`控制这些策略。升级到较新的CUDA驱动程序可能会更改、增强、增强和/或重新排序重复使用策略。

### 15.9.1. cudaMemPoolReuseFollowEvent依赖[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudamempoolreusefolloweventdependencies "这个标题的永久链接")

在分配更多物理GPU内存之前，分配器会检查CUDA事件建立的依赖性信息，并尝试从另一个流中释放的内存中分配。

cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);
cudaEventRecord(event,originalStream);

// waiting on the event that captures the free in another stream
// allows the allocator to reuse the memory to satisfy
// a new allocation request in the other stream when
// cudaMemPoolReuseFollowEventDependencies is enabled.
cudaStreamWaitEvent(otherStream, event);
cudaMallocAsync(&ptr2, size, otherStream);

### 15.9.2. cudaMemPoolReuseAllowOpportunistic[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudamempoolreuseallowopportunistic "这个标题的永久链接")

根据`cudaMemPoolReuseAllowOpportunistic`政策，分配器检查自由分配，看看是否满足了自由流的流顺序语义（例如流已经通过了自由指示的执行点）。当禁用此功能时，分配器仍将重复使用流与CPU同步时可用的内存。禁用此策略不会阻止`cudaMemPoolReuseFollowEventDependencies`的应用。

cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// after some time, the kernel finishes running
wait(10);

// When cudaMemPoolReuseAllowOpportunistic is enabled this allocation request
// can be fulfilled with the prior allocation based on the progress of originalStream.
cudaMallocAsync(&ptr2, size, otherStream);

### 15.9.3. cudaMemPoolReuseAllow内部依赖[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudamempoolreuseallowinternaldependencies "这个标题的永久链接")

如果未能从操作系统分配和映射更多物理内存，驱动程序将查找内存，其可用性取决于另一个流的待定进度。如果找到此类内存，驱动程序将把所需的依赖项插入到分配流中，并重复使用内存。

cudaMallocAsync(&ptr, size, originalStream);
kernel<<<..., originalStream>>>(ptr, ...);
cudaFreeAsync(ptr, originalStream);

// When cudaMemPoolReuseAllowInternalDependencies is enabled
// and the driver fails to allocate more physical memory, the driver may
// effectively perform a cudaStreamWaitEvent in the allocating stream
// to make sure that future work in ‘otherStream’ happens after the work
// in the original stream that would be allowed to access the original allocation.
cudaMallocAsync(&ptr2, size, otherStream);

### 15.9.4.禁用重复使用政策[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#disabling-reuse-policies "这个标题的永久链接")

虽然可控制的重复使用策略改善了内存重用，但用户可能想要禁用它们。允许机会主义重用（如`cudaMemPoolReuseAllowOpportunistic`）引入了基于CPU和GPU执行交织的分配模式的运行到运行方差。内部依赖插入（如`cudaMemPoolReuseAllowInternalDependencies`）可以以意外和潜在的非确定性方式序列化工作，当用户宁愿在分配失败时明确同步事件或流时。

## 15.10.多GPU支持的设备辅助功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-accessibility-for-multi-gpu-support "这个标题的永久链接")

就像通过虚拟内存管理API控制的分配可访问性一样，内存池分配可访问性不遵循`cudaDeviceEnablePeerAccess`或`cuCtxEnablePeerAccess`。相反，API `cudaMemPoolSetAccess`修改了哪些设备可以从池中访问分配。默认情况下，可以从分配所在的设备访问分配。此访问权限无法撤销。要启用从其他设备访问，访问设备必须能够与内存池的设备对等；请与`cudaDeviceCanAccessPeer`核对。如果没有检查对等功能，设置访问可能会失败，`cudaErrorInvalidDevice`。如果没有从池中进行分配，即使设备没有对等功能，`cudaMemPoolSetAccess`调用也可能成功；在这种情况下，池中的下一次分配将失败。

值得注意的是，`cudaMemPoolSetAccess`会影响内存池的所有分配，而不仅仅是未来的分配。此外，`cudaMemPoolGetAccess`报告的可访问性适用于池中的所有分配，而不仅仅是未来的分配。建议不要频繁更改给定GPU池的可访问性设置；一旦从给定GPU访问池，在池的生命周期内应保持从该GPU访问。

// snippet showing usage of cudaMemPoolSetAccess:
cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
              int accessingDevice) {
    cudaMemAccessDesc accessDesc = {};
    accessDesc.location.type = cudaMemLocationTypeDevice;
    accessDesc.location.id = accessingDevice;
    accessDesc.flags = cudaMemAccessFlagsProtReadWrite;

    int canAccess = 0;
    cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice,
              residentDevice);
    if (error != cudaSuccess) {
        return error;
    } else if (canAccess == 0) {
        return cudaErrorPeerAccessUnsupported;
    }

    // Make the address accessible
    return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
}

## 15.11.IPC内存池[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ipc-memory-pools "这个标题的永久链接")

支持IPC的内存池允许在进程之间轻松、高效、安全地共享GPU内存。CUDA的IPC内存池提供与CUDA的虚拟内存管理API相同的安全优势。

在具有内存池的进程之间共享内存有两个阶段。流程首先需要共享池的访问权限，然后共享该池中的特定分配。第一阶段建立并强制执行安全。第二阶段协调每个流程中使用哪些虚拟地址，以及导入过程中何时需要有效的映射。

### 15.11.1.创建和共享IPC内存池[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-and-sharing-ipc-memory-pools "这个标题的永久链接")

共享池的访问权限涉及将OS本机句柄检索到池（使用`cudaMemPoolExportToShareableHandle()`API），使用通常的OS本机IPC机制将句柄传输到导入过程，以及创建导入的内存池（使用`cudaMemPoolImportFromShareableHandle()`API）。为了使`cudaMemPoolExportToShareableHandle`成功，必须使用池属性结构中指定的请求句柄类型创建内存池。请参考适当的IPC机制的样本，以便在进程之间传输操作系统本机句柄。程序的其余部分可以在以下代码片段中找到。

// in exporting process
// create an exportable IPC capable pool on device 0
cudaMemPoolProps poolProps = { };
poolProps.allocType = cudaMemAllocationTypePinned;
poolProps.location.id = 0;
poolProps.location.type = cudaMemLocationTypeDevice;

// Setting handleTypes to a non zero value will make the pool exportable (IPC capable)
poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

cudaMemPoolCreate(&memPool, &poolProps));

// FD based handles are integer types
int fdHandle = 0;

// Retrieve an OS native handle to the pool.
// Note that a pointer to the handle memory is passed in here.
cudaMemPoolExportToShareableHandle(&fdHandle,
             memPool,
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
             0);

// The handle must be sent to the importing process with the appropriate
// OS specific APIs.

// in importing process
 int fdHandle;
// The handle needs to be retrieved from the exporting process with the
// appropriate OS specific APIs.
// Create an imported pool from the shareable handle.
// Note that the handle is passed by value here.
cudaMemPoolImportFromShareableHandle(&importedMemPool,
          (void*)fdHandle,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
          0);

### 15.11.2.在导入过程中设置访问权限[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#set-access-in-the-importing-process "这个标题的永久链接")

导入的内存池最初只能从其常驻设备访问。导入的内存池不会继承导出过程设置的任何可访问性。导入过程需要从它计划访问内存的任何GPU启用访问（使用`cudaMemPoolSetAccess`）。

If the imported memory pool belongs to a non-visible device in the importing process, the user must use the `cudaMemPoolSetAccess` API to enable access from the GPUs the allocations will be used on.

### 15.11.3.从导出池中创建和共享分配[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-and-sharing-allocations-from-an-exported-pool "这个标题的永久链接")

池共享后，在导出过程中使用库中的`cudaMallocAsync()`分配可以与导入池的其他进程共享。由于池的安全策略是在池级别建立和验证的，操作系统不需要额外的簿记来为特定的池分配提供安全性；换句话说，导入池分配所需的不透明的`cudaMemPoolPtrExportData`可以使用任何机制发送到导入过程。

虽然分配可以导出甚至导入而不以任何方式与分配流同步，但导入过程在访问分配时必须遵循与导出过程相同的规则。也就是说，在分配流中分配操作的流排序之后，必须访问分配。以下两个代码片段显示`cudaMemPoolExportPointer()`和`cudaMemPoolImportPointer()`与IPC事件共享分配，用于保证在分配准备就绪之前不会在导入过程中访问分配。

// preparing an allocation in the exporting process
cudaMemPoolPtrExportData exportData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t readyIpcEventHandle;

// ipc event for coordinating between processes
// cudaEventInterprocess flag makes the event an ipc event
// cudaEventDisableTiming  is set for performance reasons

cudaEventCreate(
        &readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess)

// allocate from the exporting mem pool
cudaMallocAsync(&ptr, size,exportMemPool, stream);

// event for sharing when the allocation is ready.
cudaEventRecord(readyIpcEvent, stream);
cudaMemPoolExportPointer(&exportData, ptr);
cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);

// Share IPC event and pointer export data with the importing process using
//  any mechanism. Here we copy the data into shared memory
shmem->ptrData = exportData;
shmem->readyIpcEventHandle = readyIpcEventHandle;
// signal consumers data is ready

// Importing an allocation
cudaMemPoolPtrExportData *importData = &shmem->prtData;
cudaEvent_t readyIpcEvent;
cudaIpcEventHandle_t *readyIpcEventHandle = &shmem->readyIpcEventHandle;

// Need to retrieve the ipc event handle and the export data from the
// exporting process using any mechanism.  Here we are using shmem and just
// need synchronization to make sure the shared memory is filled in.

cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);

// import the allocation. The operation does not block on the allocation being ready.
cudaMemPoolImportPointer(&ptr, importedMemPool, importData);

// Wait for the prior stream operations in the allocating stream to complete before
// using the allocation in the importing process.
cudaStreamWaitEvent(stream, readyIpcEvent);
kernel<<<..., stream>>>(ptr, ...);

释放分配时，需要在导入过程中释放分配，然后再在出口过程中释放分配。以下代码片段演示了使用CUDA IPC事件来提供两个进程中`cudaFreeAsync`操作之间所需的同步。导入过程中对分配的访问显然受到导入过程中自由操作的限制。值得注意的是，`cudaFree`可用於釋放兩個程序中的分配，並且可以使用其他流同步API代替CUDA IPC事件。

// The free must happen in importing process before the exporting process
kernel<<<..., stream>>>(ptr, ...);

// Last access in importing process
cudaFreeAsync(ptr, stream);

// Access not allowed in the importing process after the free
cudaIpcEventRecord(finishedIpcEvent, stream);

// Exporting process
// The exporting process needs to coordinate its free with the stream order
// of the importing process’s free.
cudaStreamWaitEvent(stream, finishedIpcEvent);
kernel<<<..., stream>>>(ptrInExportingProcess, ...);

// The free in the importing process doesn’t stop the exporting process
// from using the allocation.
cudFreeAsync(ptrInExportingProcess,stream);

### 15.11.4.IPC出口池限制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ipc-export-pool-limitations "这个标题的永久链接")

IPC pools currently do not support releasing physical blocks back to the OS. As a result the `cudaMemPoolTrimTo` API acts as a no-op and the `cudaMemPoolAttrReleaseThreshold` effectively gets ignored. This behavior is controlled by the driver, not the runtime and may change in a future driver update.

### 15.11.5.IPC导入池限制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ipc-import-pool-limitations "这个标题的永久链接")

Allocating from an import pool is not allowed; specifically, import pools cannot be set current and cannot be used in the `cudaMallocFromPoolAsync`API. As such, the allocation reuse policy attributes are meaningless for these pools.

IPC pools currently do not support releasing physical blocks back to the OS. As a result the `cudaMemPoolTrimTo` API acts as a no-op and the `cudaMemPoolAttrReleaseThreshold` effectively gets ignored.

资源使用统计属性查询仅反映导入到进程和相关物理内存中的分配。

## 15.12.同步API操作[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-api-actions "这个标题的永久链接")

分配器作为CUDA驱动程序的一部分的优化之一是与同步API的集成。当用户请求CUDA驱动程序同步时，驱动程序会等待异步工作完成。在返回之前，司机将确定哪些可以释放保证完成的同步。无论指定流还是禁用分配策略，这些分配都可以进行分配。驱动程序还在这里检查`cudaMemPoolAttrReleaseThreshold`，并释放任何多余的物理内存。

## 15.13.附录[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#addendums "这个标题的永久链接")

### 15.13.1. cudaMemcpyAsync当前上下文/设备灵敏度[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudamemcpyasync-current-context-device-sensitivity "这个标题的永久链接")

在当前CUDA驱动程序中，任何涉及`cudaMallocAsync`内存的异步`memcpy`都应使用指定流的上下文作为调用线程的当前上下文来完成。这对于`cudaMemcpyPeerAsync`来说没有必要，因为引用了API中指定的设备主上下文，而不是当前上下文。

### 15.13.2. cuPointerGet属性查询[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cupointergetattribute-query "这个标题的永久链接")

在调用`cudaFreeAsync`后，在分配上调用`cuPointerGetAttribute`导致未定义的行为。具体来说，分配是否仍然可以从给定的流访问并不重要：行为仍然未定义。

### 15.13.3. cuGraphAddMemsetNode[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cugraphaddmemsetnode "这个标题的永久链接")

`cuGraphAddMemsetNode`不适用于通过流有序分配器分配的内存。然而，分配的memset可以被流捕获。

### 15.13.4.指针属性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pointer-attributes "这个标题的永久链接")

`cuPointerGetAttributes`查询适用于流有序分配。由于流有序分配没有上下文关联，查询`CU_POINTER_ATTRIBUTE_CONTEXT`将成功，但在`*data`中返回NULL。属性`CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL`可用于确定分配的位置：在使用`cudaMemcpyPeerAsync`选择制作p2h2p副本的上下文时，这非常有用。属性`CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE`是在CUDA 11.3中添加的，在进行IPC之前，它可用于调试和确认分配来自哪个池。

### 15.13.5.中央处理器虚拟内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cpu-virtual-memory "这个标题的永久链接")

使用CUDA流有序内存分配器API时，避免使用“ulimit -v”设置VRAM限制，因为这不受支持。

# 16.图形内存节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-memory-nodes "这个标题的永久链接")

## 16.1.介绍[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-memory-nodes-intro "这个标题的永久链接")

图内存节点允许图创建和拥有内存分配。图形内存节点具有GPU有序的生命周期语义，它决定了何时允许在设备上访问内存。这些GPU有序的生命周期语义可以实现驱动程序管理的内存重用，并匹配流有序分配API `cudaMallocAsync`和`cudaFreeAsync`，这些语义可以在创建图形时捕获。

图形分配在图形生命周期内具有固定地址，包括重复实例化和启动。这允许内存被图形中的其他操作直接引用，而无需图形更新，即使CUDA更改了备份物理内存。在图中，图有序寿命不重叠的分配可能会使用相同的底层物理内存。

CUDA可能会重复使用相同的物理内存进行跨多个图形的分配，根据GPU有序的生命周期语义进行虚拟地址映射。例如，当不同的图被启动到同一流中时，CUDA可能会虚拟地使用相同的物理内存，以满足具有单图寿命的分配的需求。

## 16.2.支持和兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#support-and-compatibility "这个标题的永久链接")

图形内存节点需要一个具有11.4功能的CUDA驱动程序，并支持GPU上的流有序分配器。以下片段展示了如何检查给定设备上的支持。

int driverVersion = 0;
int deviceSupportsMemoryPools = 0;
int deviceSupportsMemoryNodes = 0;
cudaDriverGetVersion(&driverVersion);
if (driverVersion >= 11020) { // avoid invalid value error in cudaDeviceGetAttribute
    cudaDeviceGetAttribute(&deviceSupportsMemoryPools, cudaDevAttrMemoryPoolsSupported, device);
}
deviceSupportsMemoryNodes = (driverVersion >= 11040) && (deviceSupportsMemoryPools != 0);

在驱动程序版本检查中进行属性查询可以避免在11.0和11.1驱动程序上出现无效的值返回代码。请注意，当计算机消毒器检测到CUDA返回错误代码时，它会发出警告，在读取属性之前进行版本检查可以避免这种情况。图形内存节点仅支持驱动程序版本11.4及更高版本。

## 16.3.API基础知识[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-fundamentals "这个标题的永久链接")

图形内存节点是表示内存分配或自由动作的图形节点。作为简写，分配内存的节点被称为分配节点。同样，释放内存的节点被称为自由节点。由分配节点创建的分配称为图形分配。CUDA在节点建立時為圖形分配分配虛擬地址。虽然这些虚拟地址在分配节点的生命周期内是固定的，但分配内容在释放操作后不会持久，并且可能会被引用不同分配的访问覆盖。

每次运行图表时，都会认为图表分配是重新创建的。图形分配的寿命与节点的寿命不同，从GPU执行到达分配的图形节点时开始，并在发生以下情况之一时结束：

- GPU执行到达释放图节点
    
- GPU执行到达释放`cudaFreeAsync()`流调用
    
- 立即释放电话`cudaFree()`
    

笔记

图销毁不会自动释放任何实时图分配的内存，即使它结束了分配节点的寿命。分配随后必须在另一个图中释放，或使用`cudaFreeAsync()``/cudaFree()`

与其他[图结构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-structure)一样，图内存节点在图中按依赖边缘排序。程序必须保证访问图形内存的操作：

- 在分配节点之后排序
    
- 在释放内存的操作之前被订购
    

图形分配寿命根据GPU执行开始，通常根据GPU执行（而不是API调用）结束。GPU排序是工作在GPU上运行的顺序，而不是工作被列入或描述的顺序。因此，图形分配被认为是“GPU有序”。

### 16.3.1.图形节点API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-node-apis "这个标题的永久链接")

图形内存节点可以使用内存节点创建API、`cudaGraphAddMemAllocNode`和`cudaGraphAddMemFreeNode`显式创建。由`cudaGraphAddMemAllocNode`分配的地址在传递的`CUDA_MEM_ALLOC_NODE_PARAMS`结构的`dptr`字段中返回给用户。在分配图中使用图分配的所有操作必须在分配节点之后排序。同样，任何空闲节点必须在图形中分配的所有使用后进行排序。`cudaGraphAddMemFreeNode`创建空闲节点。

在下图中，有一个带有分配和自由节点的示例图。内核节点**a、b**和**c**在分配节点之后和自由节点之前排序，这样内核就可以访问分配。内核节点**e**在alloc节点之后没有排序，因此无法安全地访问内存。内核节点**d**不在自由节点之前排序，因此它无法安全地访问内存。

![内核节点](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/kernel-nodes.png)

图32内核节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id475 "此图像的永久链接")

以下代码片段建立了本图中的图形：

// Create the graph - it starts out empty
cudaGraphCreate(&graph, 0);

// parameters for a basic allocation
cudaMemAllocNodeParams params = {};
params.poolProps.allocType = cudaMemAllocationTypePinned;
params.poolProps.location.type = cudaMemLocationTypeDevice;
// specify device 0 as the resident device
params.poolProps.location.id = 0;
params.bytesize = size;

cudaGraphAddMemAllocNode(&allocNode, graph, NULL, 0, &params);
nodeParams->kernelParams[0] = params.dptr;
cudaGraphAddKernelNode(&a, graph, &allocNode, 1, &nodeParams);
cudaGraphAddKernelNode(&b, graph, &a, 1, &nodeParams);
cudaGraphAddKernelNode(&c, graph, &a, 1, &nodeParams);
cudaGraphNode_t dependencies[2];
// kernel nodes b and c are using the graph allocation, so the freeing node must depend on them.  Since the dependency of node b on node a establishes an indirect dependency, the free node does not need to explicitly depend on node a.
dependencies[0] = b;
dependencies[1] = c;
cudaGraphAddMemFreeNode(&freeNode, graph, dependencies, 2, params.dptr);
// free node does not depend on kernel node d, so it must not access the freed graph allocation.
cudaGraphAddKernelNode(&d, graph, &c, 1, &nodeParams);

// node e does not depend on the allocation node, so it must not access the allocation.  This would be true even if the freeNode depended on kernel node e.
cudaGraphAddKernelNode(&e, graph, NULL, 0, &nodeParams);

### 16.3.2.流捕获[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-capture "这个标题的永久链接")

可以通过捕获相应的流有序分配和自由调用`cudaMallocAsync`和`cudaFreeAsync`来创建图形内存节点。在这种情况下，捕获的分配API返回的虚拟地址可以用于图形中的其他操作。由于流有序依赖关系将被捕获到图中，因此流有序分配API的排序要求保证了图内存节点将与捕获的流操作（对于正确编写的流代码）进行正确排序。

忽略内核节点d和**e**，为了清楚起见，以下代码片段展示了如何使用流捕获来创建上图中的图形：

cudaMallocAsync(&dptr, size, stream1);
kernel_A<<< ..., stream1 >>>(dptr, ...);

// Fork into stream2
cudaEventRecord(event1, stream1);
cudaStreamWaitEvent(stream2, event1);

kernel_B<<< ..., stream1 >>>(dptr, ...);
// event dependencies translated into graph dependencies, so the kernel node created by the capture of kernel C will depend on the allocation node created by capturing the cudaMallocAsync call.
kernel_C<<< ..., stream2 >>>(dptr, ...);

// Join stream2 back to origin stream (stream1)
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

// Free depends on all work accessing the memory.
cudaFreeAsync(dptr, stream1);

// End capture in the origin stream
cudaStreamEndCapture(stream1, &graph);

### 16.3.3.在分配图之外访问和释放图内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#accessing-and-freeing-graph-memory-outside-of-the-allocating-graph "这个标题的永久链接")

图形分配不必通过分配图来释放。当图形没有释放分配时，该分配会持续到图形的执行之外，并且可以通过后续的CUDA操作访问。只要访问操作是在通过CUDA事件和其他流排序机制分配后排序的，就可以在另一个图表中访问，也可以直接使用流操作。随后，可以通过定期调用`cudaFree`、`cudaFreeAsync`或启动具有相应自由节点的另一个图，或随后启动分配图（如果使用[cudaGraphInstantiateFlagAutoFreeOnLaunch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-memory-nodes-cudagraphinstantiateflagautofreeonlaunch)标志实例化）来释放分配。释放内存后访问内存是非法的——在使用图形依赖性、CUDA事件和其他流排序机制访问内存的所有操作后，必须对自由操作进行排序。

笔记

由于图形分配可能相互共享底层物理内存，因此必须考虑与一致性和一致性相关的[虚拟别名支持](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-aliasing-support)规则。简单来说，自由操作必须在完整的设备操作（例如，计算内核/memcpy）完成后订购。具体来说，带外同步——例如，作为访问图分配内存的计算内核的一部分，通过内存进行握手——不足以在内存写入图内存和该图内存的自由操作之间提供排序保证。

以下代码片段演示了在分配图之外访问图分配，顺序正确建立：使用单个流，使用流之间的事件，以及使用嵌入分配和释放图的事件。

**使用单一流建立的排序：**

void *dptr;
cudaGraphAddMemAllocNode(&allocNode, allocGraph, NULL, 0, &params);
dptr = params.dptr;

cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, stream);
kernel<<< …, stream >>>(dptr, …);
cudaFreeAsync(dptr, stream);

**通过记录和等待CUDA活动建立的订单：**

void *dptr;

// Contents of allocating graph
cudaGraphAddMemAllocNode(&allocNode, allocGraph, NULL, 0, &params);
dptr = params.dptr;

// contents of consuming/freeing graph
nodeParams->kernelParams[0] = params.dptr;
cudaGraphAddKernelNode(&a, graph, NULL, 0, &nodeParams);
cudaGraphAddMemFreeNode(&freeNode, freeGraph, &a, 1, dptr);

cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);
cudaGraphInstantiate(&freeGraphExec, freeGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, allocStream);

// establish the dependency of stream2 on the allocation node
// note: the dependency could also have been established with a stream synchronize operation
cudaEventRecord(allocEvent, allocStream)
cudaStreamWaitEvent(stream2, allocEvent);

kernel<<< …, stream2 >>> (dptr, …);

// establish the dependency between the stream 3 and the allocation use
cudaStreamRecordEvent(streamUseDoneEvent, stream2);
cudaStreamWaitEvent(stream3, streamUseDoneEvent);

// it is now safe to launch the freeing graph, which may also access the memory
cudaGraphLaunch(freeGraphExec, stream3);

**通过使用图形外部事件节点建立排序：**

void *dptr;
cudaEvent_t allocEvent; // event indicating when the allocation will be ready for use.
cudaEvent_t streamUseDoneEvent; // event indicating when the stream operations are done with the allocation.

// Contents of allocating graph with event record node
cudaGraphAddMemAllocNode(&allocNode, allocGraph, NULL, 0, &params);
dptr = params.dptr;
// note: this event record node depends on the alloc node
cudaGraphAddEventRecordNode(&recordNode, allocGraph, &allocNode, 1, allocEvent);
cudaGraphInstantiate(&allocGraphExec, allocGraph, NULL, NULL, 0);

// contents of consuming/freeing graph with event wait nodes
cudaGraphAddEventWaitNode(&streamUseDoneEventNode, waitAndFreeGraph, NULL, 0, streamUseDoneEvent);
cudaGraphAddEventWaitNode(&allocReadyEventNode, waitAndFreeGraph, NULL, 0, allocEvent);
nodeParams->kernelParams[0] = params.dptr;

// The allocReadyEventNode provides ordering with the alloc node for use in a consuming graph.
cudaGraphAddKernelNode(&kernelNode, waitAndFreeGraph, &allocReadyEventNode, 1, &nodeParams);

// The free node has to be ordered after both external and internal users.
// Thus the node must depend on both the kernelNode and the
// streamUseDoneEventNode.
dependencies[0] = kernelNode;
dependencies[1] = streamUseDoneEventNode;
cudaGraphAddMemFreeNode(&freeNode, waitAndFreeGraph, &dependencies, 2, dptr);
cudaGraphInstantiate(&waitAndFreeGraphExec, waitAndFreeGraph, NULL, NULL, 0);

cudaGraphLaunch(allocGraphExec, allocStream);

// establish the dependency of stream2 on the event node satisfies the ordering requirement
cudaStreamWaitEvent(stream2, allocEvent);
kernel<<< …, stream2 >>> (dptr, …);
cudaStreamRecordEvent(streamUseDoneEvent, stream2);

// the event wait node in the waitAndFreeGraphExec establishes the dependency on the “readyForFreeEvent” that is needed to prevent the kernel running in stream two from accessing the allocation after the free node in execution order.
cudaGraphLaunch(waitAndFreeGraphExec, stream3);

### 16.3.4. cudaGraphInstantiateFlagAutoFreeOnLaunch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cudagraphinstantiateflagautofreeonlaunch "这个标题的永久链接")

在正常情况下，如果图形有未释放的内存分配，CUDA将阻止重新启动，因为同一地址的多个分配会泄漏内存。使用`cudaGraphInstantiateFlagAutoFreeOnLaunch`标志实例化图形，允许在图形仍然有未释放的分配时重新启动。在这种情况下，启动会自动插入非同步，没有未释放的分配。

启动时自动免费对单生产者多消费者算法很有用。在每次迭代时，生产者图都会创建几个分配，根据运行时条件，一组不同的消费者访问这些分配。这种类型的变量执行序列意味着消费者无法释放分配，因为后续消费者可能需要访问权限。启动时自动释放意味着启动循环不需要跟踪制作者的分配——相反，该信息仍然与制作者的创建和销毁逻辑隔离。一般来说，启动时自动免费简化了算法，否则需要在每次重新启动之前释放图形拥有的所有分配。

笔记

`cudaGraphInstantiateFlagAutoFreeOnLaunch`标志不会改变图破坏的行为。应用程序必须明确释放未释放的内存，以避免内存泄漏，即使是用标志实例化的图形。以下代码显示了使用`cudaGraphInstantiateFlagAutoFreeOnLaunch`来简化单生产者/多消费者算法：

// Create producer graph which allocates memory and populates it with data
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
cudaMallocAsync(&data1, blocks * threads, cudaStreamPerThread);
cudaMallocAsync(&data2, blocks * threads, cudaStreamPerThread);
produce<<<blocks, threads, 0, cudaStreamPerThread>>>(data1, data2);
...
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&producer,
                              graph,
                              cudaGraphInstantiateFlagAutoFreeOnLaunch);
cudaGraphDestroy(graph);

// Create first consumer graph by capturing an asynchronous library call
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
consumerFromLibrary(data1, cudaStreamPerThread);
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&consumer1, graph, 0); //regular instantiation
cudaGraphDestroy(graph);

// Create second consumer graph
cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeGlobal);
consume2<<<blocks, threads, 0, cudaStreamPerThread>>>(data2);
...
cudaStreamEndCapture(cudaStreamPerThread, &graph);
cudaGraphInstantiateWithFlags(&consumer2, graph, 0);
cudaGraphDestroy(graph);

// Launch in a loop
bool launchConsumer2 = false;
do {
    cudaGraphLaunch(producer, myStream);
    cudaGraphLaunch(consumer1, myStream);
    if (launchConsumer2) {
        cudaGraphLaunch(consumer2, myStream);
    }
} while (determineAction(&launchConsumer2));

cudaFreeAsync(data1, myStream);
cudaFreeAsync(data2, myStream);

cudaGraphExecDestroy(producer);
cudaGraphExecDestroy(consumer1);
cudaGraphExecDestroy(consumer2);

## 16.4.优化内存重复使用[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#optimized-memory-reuse "这个标题的永久链接")

CUDA通过两种方式重复使用内存：

- 图形中的虚拟和物理内存重用基于虚拟地址分配，如流有序分配器。
    
- 图形之间的物理内存重用通过虚拟锯齿完成：不同的图形可以将相同的物理内存映射到其唯一的虚拟地址。
    

### 16.4.1.图表中的地址重复使用[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#address-reuse-within-a-graph "这个标题的永久链接")

CUDA可以通过将相同的虚拟地址范围分配给寿命不重叠的不同分配来重复使用图形中的内存。由于虚拟地址可以重复使用，因此指向具有不连续寿命的不同分配的指针不能保证是唯一的。

下图显示了添加一个新的分配节点（2），该节点可以重复使用依赖节点（1）释放的地址。

![添加新的分配节点2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/new-alloc-node.png)

图33添加新的分配节点2[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id476 "此图像的永久链接")

下图显示了添加一个新的分配节点（4）。新的分配节点不依赖于自由节点（2），因此无法重复使用关联分配节点（2）的地址。如果分配节点（2）使用自由节点（1）释放的地址，则新的分配节点3将需要一个新的地址。

![添加新的Alloc节点3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/adding-new-alloc-nodes.png)

图34添加新的Alloc节点3[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id477 "此图像的永久链接")

### 16.4.2.物理记忆管理和共享[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#physical-memory-management-and-sharing "这个标题的永久链接")

CUDA负责在GPU顺序到达分配节点之前将物理内存映射到虚拟地址。作为内存占用和映射开销的优化，如果多个图形不会同时运行，它们可能会使用相同的物理内存进行不同的分配；但是，如果物理页面同时绑定到多个执行图，或绑定到未释放的图分配，则无法重复使用。

CUDA可以在图形实例化、启动或执行期间随时更新物理内存映射。CUDA还可能在未来的图形发射之间引入同步，以防止实时图形分配引用相同的物理内存。对于任何分配自由分配模式，如果程序在分配的生命周期之外访问指针，错误的访问可能会无声读取或写入另一个分配拥有的实时数据（即使分配的虚拟地址是唯一的）。使用计算机消毒工具可以捕获此错误。

下图显示了在同一流中依次启动的图形。在本例中，每个图形都释放了它分配的所有内存。由于同一流中的图形永远不会同时运行，CUDA可以而且应该使用相同的物理内存来满足所有分配。

![顺序启动的图形](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/sequentially-launched-graphs.png)

图35顺序启动的图形[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id478 "此图像的永久链接")

## 16.5.绩效考虑因素[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-considerations "这个标题的永久链接")

当多个图形被启动到同一流中时，CUDA会尝试为它们分配相同的物理内存，因为这些图形的执行不能重叠。图形的物理映射在发布之间保留，作为优化，以避免重新映射的成本。如果稍后启动其中一个图形，使其执行可能与其他图形重叠（例如，如果它启动到不同的流中），那么CUDA必须执行一些重新映射，因为并发图形需要不同的内存来避免数据损坏。

一般来说，CUDA中图形内存的重新映射可能是由以下操作引起的：

- 更改启动图形的流
    
- 图形内存池上的修剪操作，明确释放未使用的内存（在[物理内存占地面积](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-memory-nodes-physical-memory-footprint)中讨论）
    
- 重新启动一个图形，而另一个图形的未自由分配映射到同一内存，将导致重新启动前重新映射内存
    

重新映射必须按照执行顺序进行，但在该图形之前的任何执行完成后（否则仍在使用的内存可能会被取消映射）。由于这种排序依赖性，以及由于映射操作是操作系统调用，映射操作可能相对昂贵。应用程序可以通过在同一流中持续启动包含分配内存节点的图形来避免这种成本。

### 16.5.1.首次发布/cudaGraph上传[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#first-launch-cudagraphupload "这个标题的永久链接")

在图实例化期间无法分配或映射物理内存，因为图将执行的流是未知的。而是在图形启动期间进行映射。调用`cudaGraphUpload`可以通过立即执行该图的所有映射并将图与上传流相关联，从启动中分离分配成本。如果图形随后被启动到同一流中，它将在没有任何额外的重新映射的情况下启动。

使用不同的流进行图形上传和图形启动的行为与切换流相似，可能会导致重新映射操作。此外，允许不相关的内存池管理从空闲流中提取内存，这可能会抵消上传的影响。

## 16.6.物理内存足迹[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#physical-memory-footprint "这个标题的永久链接")

The pool-management behavior of asynchronous allocation means that destroying a graph which contains memory nodes (even if their allocations are free) will not immediately return physical memory to the OS for use by other processes. To explicitly release memory back to the OS, an application should use the `cudaDeviceGraphMemTrim` API.

`cudaDeviceGraphMemTrim` will unmap and release any physical memory reserved by graph memory nodes that is not actively in use. Allocations that have not been freed and graphs that are scheduled or running are considered to be actively using the physical memory and will not be impacted. Use of the trim API will make physical memory available to other allocation APIs and other applications or processes, but will cause CUDA to reallocate and remap memory when the trimmed graphs are next launched. Note that `cudaDeviceGraphMemTrim` operates on a different pool from `cudaMemPoolTrimTo()`. The graph memory pool is not exposed to the steam ordered memory allocator. CUDA allows applications to query their graph memory footprint through the `cudaDeviceGetGraphMemAttribute` API. Querying the attribute `cudaGraphMemAttrReservedMemCurrent` returns the amount of physical memory reserved by the driver for graph allocations in the current process. Querying `cudaGraphMemAttrUsedMemCurrent` returns the amount of physical memory currently mapped by at least one graph. Either of these attributes can be used to track when new physical memory is acquired by CUDA for the sake of an allocating graph. Both of these attributes are useful for examining how much memory is saved by the sharing mechanism.

## 16.7.同行访问[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-access "这个标题的永久链接")

图形分配可以配置为从多个GPU访问，在这种情况下，CUDA将根据需要将分配映射到对等GPU上。CUDA允许需要不同映射的图形分配来重复使用相同的虚拟地址。当这种情况发生时，地址范围会映射到不同分配所需的所有GPU上。这意味着分配有时可能允许比创建期间请求的更多的对等访问；然而，依赖这些额外的映射仍然是一个错误。

### 16.7.1.使用Graph Node API的对等访问[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-access-with-graph-node-apis "这个标题的永久链接")

The `cudaGraphAddMemAllocNode` API accepts mapping requests in the `accessDescs` array field of the node parameters structures. The `poolProps.location` embedded structure specifies the resident device for the allocation. Access from the allocating GPU is assumed to be needed, thus the application does not need to specify an entry for the resident device in the `accessDescs` array.

cudaMemAllocNodeParams params = {};
params.poolProps.allocType = cudaMemAllocationTypePinned;
params.poolProps.location.type = cudaMemLocationTypeDevice;
// specify device 1 as the resident device
params.poolProps.location.id = 1;
params.bytesize = size;

// allocate an allocation resident on device 1 accessible from device 1
cudaGraphAddMemAllocNode(&allocNode, graph, NULL, 0, &params);

accessDescs[2];
// boilerplate for the access descs (only ReadWrite and Device access supported by the add node api)
accessDescs[0].flags = cudaMemAccessFlagsProtReadWrite;
accessDescs[0].location.type = cudaMemLocationTypeDevice;
accessDescs[1].flags = cudaMemAccessFlagsProtReadWrite;
accessDescs[1].location.type = cudaMemLocationTypeDevice;

// access being requested for device 0 & 2.  Device 1 access requirement left implicit.
accessDescs[0].location.id = 0;
accessDescs[1].location.id = 2;

// access request array has 2 entries.
params.accessDescCount = 2;
params.accessDescs = accessDescs;

// allocate an allocation resident on device 1 accessible from devices 0, 1 and 2. (0 & 2 from the descriptors, 1 from it being the resident device).
cudaGraphAddMemAllocNode(&allocNode, graph, NULL, 0, &params);

### 16.7.2.带有流捕获的对等访问[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-access-with-stream-capture "这个标题的永久链接")

对于流捕获，分配节点在捕获时记录分配池的对等可访问性。在捕获`cudaMallocFromPoolAsync`调用后更改分配池的对等可访问性不会影响图形将为分配所做的映射。

// boilerplate for the access descs (only ReadWrite and Device access supported by the add node api)
accessDesc.flags = cudaMemAccessFlagsProtReadWrite;
accessDesc.location.type = cudaMemLocationTypeDevice;
accessDesc.location.id = 1;

// let memPool be resident and accessible on device 0

cudaStreamBeginCapture(stream);
cudaMallocAsync(&dptr1, size, memPool, stream);
cudaStreamEndCapture(stream, &graph1);

cudaMemPoolSetAccess(memPool, &accessDesc, 1);

cudaStreamBeginCapture(stream);
cudaMallocAsync(&dptr2, size, memPool, stream);
cudaStreamEndCapture(stream, &graph2);

//The graph node allocating dptr1 would only have the device 0 accessibility even though memPool now has device 1 accessibility.
//The graph node allocating dptr2 will have device 0 and device 1 accessibility, since that was the pool accessibility at the time of the cudaMallocAsync call.

## 16.8.子图中的内存节点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-nodes-in-child-graphs "这个标题的永久链接")

CUDA 12.9引入了将子图所有权转移到父图的能力。移至父图形的子图形允许包含内存分配和自由节点。这允许在添加到父图中之前独立构建包含分配或自由节点的子图。

移动后，以下限制适用于子图：

- 不能独立实例化或销毁。
    
- 不能添加为单独的父图的子图。
    
- 不能用作cuGraphExecUpdate的参数。
    
- 不能添加额外的内存分配或空闲节点。
    

// Create the child graph
cudaGraphCreate(&child, 0);

// parameters for a basic allocation
cudaMemAllocNodeParams params = {};
params.poolProps.allocType = cudaMemAllocationTypePinned;
params.poolProps.location.type = cudaMemLocationTypeDevice;
// specify device 0 as the resident device
params.poolProps.location.id = 0;
params.bytesize = size;

cudaGraphAddMemAllocNode(&allocNode, graph, NULL, 0, &params);
// Additional nodes using the allocation could be added here
cudaGraphAddMemFreeNode(&freeNode, graph, &allocNode, 1, params.dptr);

// Create the parent graph
cudaGraphCreate(&parent, 0);

// Move the child graph to the parent graph
cudaGraphNodeParams childNodeParams = { cudaGraphNodeTypeGraph };
childNodeParams.graph.graph = child;
childNodeParams.graph.ownership = cudaGraphChildGraphOwnershipMove;
cudaGraphAddNode(&parentNode, parent, NULL, NULL, 0, &childNodeParams);

# 17.数学函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mathematical-functions-appendix "这个标题的永久链接")

参考手册及其描述列出了设备代码中支持的C/C++标准库数学函数的所有功能，以及所有内在函数（仅在设备代码中支持）。

本节在适用时提供其中一些功能的准确性信息。它使用ULP进行量化。有关最后一名单位（ULP）定义的更多信息，请参阅Jean-Michel Muller_关于ulp（x）定义的_论文，RR-5504，LIP RR-2005-09，INRIA，LIP。2005，第16页，网址为[https://hal.inria.fr/inria-00070503/document](https://hal.inria.fr/inria-00070503/document)。

设备代码中支持的数学函数不设置全局`errno`变量，也不报告任何浮点异常来表示错误；因此，如果需要错误诊断机制，用户应对函数的输入和输出进行额外的筛选。用户对指针参数的有效性负责。用户不得将未初始化的参数传递给数学函数，因为这可能会导致未定义的行为：函数在用户程序中内联，因此受制于编译器优化。

## 17.1.标准函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#standard-functions "这个标题的永久链接")

本节的功能可以在主机和设备代码中使用。

本节指定了在设备上执行时以及在主机不提供函数的情况下在主机上执行时的每个函数的错误边界。

误差边界是由广泛但非详尽的测试产生的，因此它们不是保证的边界。

**单精密浮点函数**

加法和乘法符合IEEE标准，因此最大误差为0.5 ulp。

将单精度浮点操作数四舍五入为整数的推荐方法，结果为单精度浮点数为`rintf()`而不是`roundf()`原因是`roundf()`映射到设备上的4指令序列，而`rintf()`映射到单个指令。`truncf()``ceilf()`和`floorf()`也映射到单个指令。

表17具有最大ULP误差的单精密数学标准库函数。最大误差表示为CUDA库函数返回的结果与根据最接近的溣到偶四舍五入模式获得的正确四舍五入的单精度结果之间的ulps差的绝对值。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id479 "此表的永久链接")
|功能|最大ulp误差|
|---|---|
|`x+y`|0（IEEE-754四舍五入至最接近偶七）|
|`x*y`|0（IEEE-754四舍五入至最接近偶七）|
|`x/y`|0用于计算能力≥2当编译时`-prec-div=true`<br><br>2（全范围），否则|
|`1/x`|0用于计算能力≥2当编译时`-prec-div=true`<br><br>1（全范围），否则|
|`rsqrtf(x)`<br><br>`1/sqrtf(x)`|2（全范围）<br><br>Applies to `1/sqrtf(x)` only when it is converted to `rsqrtf(x)` by the compiler.|
|`sqrtf(x)`|0当编译时`-prec-sqrt=true`<br><br>否则1用于计算能力≥5.2<br><br>和3个用于旧架构|
|`cbrtf(x)`|1（全范围）|
|`rcbrtf(x)`|1（全范围）|
|`hypotf(x,y)`|3（全范围）|
|`rhypotf(x,y)`|2（全范围）|
|`norm3df(x,y,z)`|3（全范围）|
|`rnorm3df(x,y,z)`|2（全范围）|
|`norm4df(x,y,z,t)`|3（全范围）|
|`rnorm4df(x,y,z,t)`|2（全范围）|
|`normf(dim,arr)`|无法提供误差边界，因为使用快速算法时，由于四舍五入而导致精度损失。|
|`rnormf(dim,arr)`|无法提供误差边界，因为使用快速算法时，由于四舍五入而导致精度损失。|
|`expf(x)`|2（全范围）|
|`exp2f(x)`|2（全范围）|
|`exp10f(x)`|2（全范围）|
|`expm1f(x)`|1（全范围）|
|`logf(x)`|1（全范围）|
|`log2f(x)`|1（全范围）|
|`log10f(x)`|2（全范围）|
|`log1pf(x)`|1（全范围）|
|`sinf(x)`|2（全范围）|
|`cosf(x)`|2（全范围）|
|`tanf(x)`|4（全范围）|
|`sincosf(x,sptr,cptr)`|2（全范围）|
|`sinpif(x)`|1（全范围）|
|`cospif(x)`|1（全范围）|
|`sincospif(x,sptr,cptr)`|1（全范围）|
|`asinf(x)`|2（全范围）|
|`acosf(x)`|2（全范围）|
|`atanf(x)`|2（全范围）|
|`atan2f(y,x)`|3（全范围）|
|`sinhf(x)`|3（全范围）|
|`coshf(x)`|2（全范围）|
|`tanhf(x)`|2（全范围）|
|`asinhf(x)`|3（全范围）|
|`acoshf(x)`|4（全范围）|
|`atanhf(x)`|3（全范围）|
|`powf(x,y)`|4（全范围）|
|`erff(x)`|2（全范围）|
|`erfcf(x)`|4（全范围）|
|`erfinvf(x)`|2（全范围）|
|`erfcinvf(x)`|4（全范围）|
|`erfcxf(x)`|4（全范围）|
|`normcdff(x)`|5（全范围）|
|`normcdfinvf(x)`|5（全范围）|
|`lgammaf(x)`|6（外部间隔-10.001 ... -2.264；内部更大）|
|`tgammaf(x)`|5（全范围）|
|`fmaf(x,y,z)`|0（全范围）|
|`frexpf(x,exp)`|0（全范围）|
|`ldexpf(x,exp)`|0（全范围）|
|`scalbnf(x,n)`|0（全范围）|
|`scalblnf(x,l)`|0（全范围）|
|`logbf(x)`|0（全范围）|
|`ilogbf(x)`|0（全范围）|
|`j0f(x)`|9为\|x\| < 8<br><br>否则，最大绝对误差为2.2 x 10-6|
|`j1f(x)`|9为\|x\| < 8<br><br>否则，最大绝对误差为2.2 x 10-6|
|`jnf(n,x)`|对于n = 128，最大绝对误差为2.2 x 10-6|
|`y0f(x)`|9为\|x\| < 8<br><br>否则，最大绝对误差为2.2 x 10-6|
|`y1f(x)`|9为\|x\| < 8<br><br>否则，最大绝对误差为2.2 x 10-6|
|`ynf(n,x)`|ceil(2 + 2.5n) for \|x\| < n<br><br>否则，最大绝对误差为2.2 x 10-6|
|`cyl_bessel_i0f(x)`|6（全范围）|
|`cyl_bessel_i1f(x)`|6（全范围）|
|`fmodf(x,y)`|0（全范围）|
|`remainderf(x,y)`|0（全范围）|
|`remquof(x,y,iptr)`|0（全范围）|
|`modff(x,iptr)`|0（全范围）|
|`fdimf(x,y)`|0（全范围）|
|`truncf(x)`|0（全范围）|
|`roundf(x)`|0（全范围）|
|`rintf(x)`|0（全范围）|
|`nearbyintf(x)`|0（全范围）|
|`ceilf(x)`|0（全范围）|
|`floorf(x)`|0（全范围）|
|`lrintf(x)`|0（全范围）|
|`lroundf(x)`|0（全范围）|
|`llrintf(x)`|0（全范围）|
|`llroundf(x)`|0（全范围）|

**双精密浮点函数**

将双精度浮点操作数四舍五入到整数的推荐方法，结果为双精度浮点数是`rint()`而不是`round()`原因是`round()`映射到设备上的5个指令序列，而`rint()`映射到单个指令。`trunc()``ceil()`和`floor()`也映射到单个指令。

表18最大ULP误差的双精密数学标准库函数。最大误差表示为CUDA库函数返回的结果与根据四舍五入到最近的偶数四舍五入模式获得的正确四舍五入双精度结果之间的ulps差的绝对值。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id480 "此表的永久链接")
|功能|最大ulp误差|
|---|---|
|`x+y`|0（IEEE-754四舍五入至最接近偶七）|
|`x*y`|0（IEEE-754四舍五入至最接近偶七）|
|`x/y`|0（IEEE-754四舍五入至最接近偶七）|
|`1/x`|0（IEEE-754四舍五入至最接近偶七）|
|`sqrt(x)`|0（IEEE-754四舍五入至最接近偶七）|
|`rsqrt(x)`|1（全范围）|
|`cbrt(x)`|1（全范围）|
|`rcbrt(x)`|1（全范围）|
|`hypot(x,y)`|2（全范围）|
|`rhypot(x,y)`|1（全范围）|
|`norm3d(x,y,z)`|2（全范围）|
|`rnorm3d(x,y,z)`|1（全范围）|
|`norm4d(x,y,z,t)`|2（全范围）|
|`rnorm4d(x,y,z,t)`|1（全范围）|
|`norm(dim,arr)`|无法提供误差边界，因为使用快速算法时会因四舍五入而损失准确性。|
|`rnorm(dim,arr)`|无法提供误差边界，因为使用快速算法时会因四舍五入而损失准确性。|
|`exp(x)`|1（全范围）|
|`exp2(x)`|1（全范围）|
|`exp10(x)`|1（全范围）|
|`expm1(x)`|1（全范围）|
|`log(x)`|1（全范围）|
|`log2(x)`|1（全范围）|
|`log10(x)`|1（全范围）|
|`log1p(x)`|1（全范围）|
|`sin(x)`|2（全范围）|
|`cos(x)`|2（全范围）|
|`tan(x)`|2（全范围）|
|`sincos(x,sptr,cptr)`|2（全范围）|
|`sinpi(x)`|2（全范围）|
|`cospi(x)`|2（全范围）|
|`sincospi(x,sptr,cptr)`|2（全范围）|
|`asin(x)`|2（全范围）|
|`acos(x)`|2（全范围）|
|`atan(x)`|2（全范围）|
|`atan2(y,x)`|2（全范围）|
|`sinh(x)`|2（全范围）|
|`cosh(x)`|1（全范围）|
|`tanh(x)`|1（全范围）|
|`asinh(x)`|3（全范围）|
|`acosh(x)`|3（全范围）|
|`atanh(x)`|2（全范围）|
|`pow(x,y)`|2（全范围）|
|`erf(x)`|2（全范围）|
|`erfc(x)`|5（全范围）|
|`erfinv(x)`|5（全范围）|
|`erfcinv(x)`|6（全范围）|
|`erfcx(x)`|4（全范围）|
|`normcdf(x)`|5（全范围）|
|`normcdfinv(x)`|8（全范围）|
|`lgamma(x)`|4（外部间隔-23.0001 ... -2.2637；内部更大）|
|`tgamma(x)`|10（全范围）|
|`fma(x,y,z)`|0（IEEE-754四舍五入至最接近偶七）|
|`frexp(x,exp)`|0（全范围）|
|`ldexp(x,exp)`|0（全范围）|
|`scalbn(x,n)`|0（全范围）|
|`scalbln(x,l)`|0（全范围）|
|`logb(x)`|0（全范围）|
|`ilogb(x)`|0（全范围）|
|`j0(x)`|7为\|x\| < 8<br><br>否则，最大绝对误差为5 x 10-12|
|`j1(x)`|7为\|x\| < 8<br><br>否则，最大绝对误差为5 x 10-12|
|`jn(n,x)`|对于n = 128，最大绝对误差为5 x 10-12|
|`y0(x)`|7为\|x\| < 8<br><br>否则，最大绝对误差为5 x 10-12|
|`y1(x)`|7为\|x\| < 8<br><br>否则，最大绝对误差为5 x 10-12|
|`yn(n,x)`|对于\|x\|>1.5n，最大绝对误差为5 x 10-12|
|`cyl_bessel_i0(x)`|6（全范围）|
|`cyl_bessel_i1(x)`|6（全范围）|
|`fmod(x,y)`|0（全范围）|
|`remainder(x,y)`|0（全范围）|
|`remquo(x,y,iptr)`|0（全范围）|
|`modf(x,iptr)`|0（全范围）|
|`fdim(x,y)`|0（全范围）|
|`trunc(x)`|0（全范围）|
|`round(x)`|0（全范围）|
|`rint(x)`|0（全范围）|
|`nearbyint(x)`|0（全范围）|
|`ceil(x)`|0（全范围）|
|`floor(x)`|0（全范围）|
|`lrint(x)`|0（全范围）|
|`lround(x)`|0（全范围）|
|`llrint(x)`|0（全范围）|
|`llround(x)`|0（全范围）|

**四精密浮点函数**

请注意，四精度数学函数目前仅适用于具有计算能力10.0及更高版本的设备。由于实现的细节，设备代码中对`__float128`和`_Float128`类型的支持也仅限于选择主机平台的组合，另请参阅[主机编译器扩展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#host-compiler-extensions)。

表19最大ULP误差的四精密数学标准库函数。最大误差表示为CUDA库函数返回的结果与根据四舍五入到最近的平复四舍五入模式获得的正确四精度结果之间的ulps差异的绝对值。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id481 "此表的永久链接")
|功能|最大ulp误差|
|---|---|
|`x+y` `__nv_fp128_add(x, y)`|0（IEEE-754四舍五入至最接近偶七）|
|`x-y` `__nv_fp128_sub(x, y)`|0（IEEE-754四舍五入至最接近偶七）|
|`x*y` `__nv_fp128_mul(x, y)`|0（IEEE-754四舍五入至最接近偶七）|
|`x/y` `__nv_fp128_div(x, y)`|0（IEEE-754四舍五入至最接近偶七）|
|`__nv_fp128_sqrt(x)`|0（IEEE-754四舍五入至最接近偶七）|
|`__nv_fp128_fma(x, y, z)`|0（IEEE-754四舍五入至最接近偶七）|
|`__nv_fp128_sin(x)`|1（全范围）|
|`__nv_fp128_cos(x)`|1（全范围）|
|`__nv_fp128_tan(x)`|1（全范围）|
|`__nv_fp128_asin(x)`|1（全范围）|
|`__nv_fp128_acos(x)`|1（全范围）|
|`__nv_fp128_atan(x)`|1（全范围）|
|`__nv_fp128_exp(x)`|1（全范围）|
|`__nv_fp128_exp2(x)`|1（全范围）|
|`__nv_fp128_exp10(x)`|1（全范围）|
|`__nv_fp128_expm1(x)`|1（全范围）|
|`__nv_fp128_log(x)`|1（全范围）|
|`__nv_fp128_log2(x)`|1（全范围）|
|`__nv_fp128_log10(x)`|1（全范围）|
|`__nv_fp128_log1p(x)`|1（全范围）|
|`__nv_fp128_pow(x, y)`|1（全范围）|
|`__nv_fp128_sinh(x)`|1（全范围）|
|`__nv_fp128_cosh(x)`|1（全范围）|
|`__nv_fp128_tanh(x)`|1（全范围）|
|`__nv_fp128_asinh(x)`|1（全范围）|
|`__nv_fp128_acosh(x)`|1（全范围）|
|`__nv_fp128_atanh(x)`|1（全范围）|
|`__nv_fp128_hypot(x, y)`|1（全范围）|
|`__nv_fp128_ceil(x)`|0（全范围）|
|`__nv_fp128_trunc(x)`|0（全范围）|
|`__nv_fp128_floor(x)`|0（全范围）|
|`__nv_fp128_round(x)`|0（全范围）|
|`__nv_fp128_rint(x)`|0（全范围）|
|`__nv_fp128_fabs(x)`|0（全范围）|
|`__nv_fp128_copysign(x, y)`|0（全范围）|
|`__nv_fp128_fmax(x, y)`|0（全范围）|
|`__nv_fp128_fmin(x, y)`|0（全范围）|
|`__nv_fp128_fdim(x, y)`|0（全范围）|
|`__nv_fp128_fmod(x, y)`|0（全范围）|
|`__nv_fp128_remainder(x, y)`|0（全范围）|
|`__nv_fp128_frexp(x, nptr)`|0（全范围）|
|`__nv_fp128_modf(x, iptr)`|0（全范围）|
|`__nv_fp128_ldexp(x, exp)`|0（全范围）|
|`__nv_fp128_ilogb(x)`|0（全范围）|

## 17.2.内在功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#intrinsic-functions "这个标题的永久链接")

本节的功能只能在设备代码中使用。

在这些功能中，有一些[标准功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mathematical-functions-appendix-standard-functions)的功能不太准确，但速度更快。它们的名字相同，前缀为`__`（例如`__sinf(x)`）。它们的速度更快，因为它们映射到更少的本机指令。编译器有一个选项（`-use_fast_math`），该选项强制[表20](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#intrinsic-functions-functions-affected-use-fast-math)中的每个函数编译为其内在对应函数。除了降低受影响功能的准确性外，它还可能导致特殊情况处理方面的一些差异。一种更稳健的方法是选择性地用调用替换内在函数的数学函数调用，仅当性能增益值得并且可以容忍更改的属性，如精度降低和不同的特殊情况处理。

表20受-use_fast_math影响的函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#intrinsic-functions-functions-affected-use-fast-math "此表的永久链接")
|操作员/功能|设备功能|
|---|---|
|`x/y`|`__fdividef(x,y)`|
|`sinf(x)`|`__sinf(x)`|
|`cosf(x)`|`__cosf(x)`|
|`tanf(x)`|`__tanf(x)`|
|`sincosf(x,sptr,cptr)`|`__sincosf(x,sptr,cptr)`|
|`logf(x)`|`__logf(x)`|
|`log2f(x)`|`__log2f(x)`|
|`log10f(x)`|`__log10f(x)`|
|`expf(x)`|`__expf(x)`|
|`exp10f(x)`|`__exp10f(x)`|
|`powf(x,y)`|`__powf(x,y)`|
|`tanhf(x)`|`__tanhf(x)`|

**单精密浮点函数**

`__fadd_[rn,rz,ru,rd]()`和`__fmul_[rn,rz,ru,rd]()`映射到编译器永远不会合并到FMAD的加法和乘法运算。相比之下，由“*”和“+”运算符生成的加法和乘法将经常合并到FMAD中。

后缀为`_rn`的函数使用四舍五入到最接近的偶数四舍五入模式进行操作。

后缀为`_rz`的函数使用向零四舍五入模式进行操作。

后缀为`_ru`的函数使用四舍五入（到正无穷大）四舍五入模式进行操作。

后缀为`_rd`的函数使用四舍五入（到负无穷大）四舍五入模式进行操作。

The accuracy of floating-point division varies depending on whether the code is compiled with `-prec-div=false` or `-prec-div=true`. When the code is compiled with `-prec-div=false`, both the regular division `/` operator and `__fdividef(x,y)` have the same accuracy, but for 2126 < `|y|` < 2128,`__fdividef(x,y)` delivers a result of zero, whereas the `/` operator delivers the correct result to within the accuracy stated in [Table 21](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#intrinsic-functions-single-precision-floating-point-intrinsic-functions-supported-by-cuda-runtime-library). Also, for 2126 < `|y|` < 2128, if `x` is infinity, `__fdividef(x,y)` delivers a `NaN` (as a result of multiplying infinity by zero), while the `/` operator returns infinity. On the other hand, the `/` operator is IEEE-compliant when the code is compiled with `-prec-div=true` or without any `-prec-div` option at all since its default value is true.

表21单精密浮点内在函数。（由CUDA运行时库支持，具有各自的错误边界）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#intrinsic-functions-single-precision-floating-point-intrinsic-functions-supported-by-cuda-runtime-library "此表的永久链接")
|功能|误差范围|
|---|---|
|`__fadd_[rn,rz,ru,rd](x,y)`|符合IEEE。|
|`__fsub_[rn,rz,ru,rd](x,y)`|符合IEEE。|
|`__fmul_[rn,rz,ru,rd](x,y)`|符合IEEE。|
|`__fmaf_[rn,rz,ru,rd](x,y,z)`|符合IEEE。|
|`__frcp_[rn,rz,ru,rd](x)`|符合IEEE。|
|`__fsqrt_[rn,rz,ru,rd](x)`|符合IEEE。|
|`__frsqrt_rn(x)`|符合IEEE。|
|`__fdiv_[rn,rz,ru,rd](x,y)`|符合IEEE。|
|`__fdividef(x,y)`|对于`\|y\|`在[2−126,2126]，最大ulp误差为2。|
|`__expf(x)`|The maximum ulp error is `2 + floor(abs(1.173 * x))`.|
|`__exp10f(x)`|The maximum ulp error is `2 + floor(abs(2.97 * x))`.|
|`__logf(x)`|对于[0.5，2]中的`x`，最大绝对误差为2−21.41，否则，最大ulp误差为3。|
|`__log2f(x)`|对于[0.5，2]中的`x`，最大绝对误差为2−22，否则，最大ulp误差为2。|
|`__log10f(x)`|对于[0.5，2]中的`x`，最大绝对误差为2−24，否则，最大ulp误差为3。|
|`__sinf(x)`|对于`x`在[−π,π]，最大绝对误差是2−21.41，否则更大。|
|`__cosf(x)`|对于`x`在[−π,π]，最大绝对误差是2−21.19，否则更大。|
|`__sincosf(x,sptr,cptr)`|与`__sinf(x)`和`__cosf(x)`相同。|
|`__tanf(x)`|Derived from its implementation as `__sinf(x) * (1/__cosf(x))`.|
|`__powf(x, y)`|Derived from its implementation as `exp2f(y * __log2f(x))`.|
|`__tanhf(x)`|当前实现的最大相对误差是2−11.即使在`-ftz=true`编译器设置下，这种快速内在的亚常态结果也不会刷新到零。适用于计算能力至少为7.5的设备；默认为其他设备上的常规`tanhf()`函数行为。|

**双精密浮点函数**

`__dadd_rn()`和`__dmul_rn()`映射到编译器永远不会合并到FMAD的加法和乘法运算。相比之下，由“*”和“+”运算符生成的加法和乘法将经常合并到FMAD中。

表22双精制浮点内在函数。（由CUDA运行时库支持，具有各自的错误边界）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id482 "此表的永久链接")
|功能|误差范围|
|---|---|
|`__dadd_[rn,rz,ru,rd](x,y)`|符合IEEE。|
|`__dsub_[rn,rz,ru,rd](x,y)`|符合IEEE。|
|`__dmul_[rn,rz,ru,rd](x,y)`|符合IEEE。|
|`__fma_[rn,rz,ru,rd](x,y,z)`|符合IEEE。|
|`__ddiv_[rn,rz,ru,rd](x,y)(x,y)`|符合IEEE。<br><br>需要计算能力_>_ 2。|
|`__drcp_[rn,rz,ru,rd](x)`|符合IEEE。<br><br>需要计算能力_>_ 2。|
|`__dsqrt_[rn,rz,ru,rd](x)`|符合IEEE。<br><br>需要计算能力_>_ 2。|

# 18.C++语言支持[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-support "这个标题的永久链接")

如[使用NVCC编译](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-with-nvcc)中所述，使用`nvcc`编译的CUDA源文件可以包含主机代码和设备代码的组合。CUDA前端编译器旨在模拟与C++输入代码相关的主机编译器行为。输入源代码是根据C++ ISO/IEC 14882:2003、C++ ISO/IEC 14882:2011、C++ ISO/IEC 14882:2014或C++ ISO/IEC 14882:2017规范进行处理，CUDA前端编译器旨在模拟任何与ISO规范的主机编译器的差异。此外，支持的语言使用本文档6中描述的CUDA特定结构进行扩展，并受以下描述的限制。

[C++11语言功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cpp11-language-features)、[C++14语言](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cpp14-language-features)功能和[C++17语言功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cpp17-language-features)分别为C++11、C++14、C++17和C++20功能提供支持矩阵。[限制](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#language-restrictions)列出了语言限制。[多态函数包装器](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#polymorphic-function-wrappers)和[扩展Lambdas](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-lambda)描述了其他功能。[代码样本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#code-samples)提供代码样本。

## 18.1.C++11语言功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-11-language-features "这个标题的永久链接")

下表列出了已被C++11标准接受的新语言功能。“提案”列提供了描述该功能的ISO C++委员会提案的链接，而“在nvcc（设备代码）中可用”列表示nvcc的第一个版本，其中包含该功能（如果已实现）的设备代码实现。

表23 C++11语言功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id483 "此表的永久链接")
|语言特征|C++11提案|在nvcc（设备代码）中可用|
|---|---|---|
|R值引用|[N2118号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html)|7.0|
|Rvalue参考`*this`|[N2439号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm)|7.0|
|通过rvalues初始化类对象|[N1610号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html)|7.0|
|非静态数据成员初始化器|[N2756号](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm)|7.0|
|可变模板|[N2242号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf)|7.0|
|扩展可调模板模板参数|[N2555号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf)|7.0|
|初始化器列表|[N2672号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm)|7.0|
|静态断言|[N1720号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html)|7.0|
|`auto`-类型变量|[N1984](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf)|7.0|
|多重声明器`auto`|[N1737](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf)|7.0|
|删除汽车作为存储类指定符|[N2546号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm)|7.0|
|新函数声明器语法|[N2541号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm)|7.0|
|Lambda表达|[N2927](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf)|7.0|
|表达式的声明类型|[N2343号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf)|7.0|
|退货类型不完整|[N3276号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3276.pdf)|7.0|
|直角括号|[N1757号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html)|7.0|
|函数模板的默认模板参数|[博士226](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226)|7.0|
|解决表达式的SFINAE问题|[DR339](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html)|7.0|
|别名模板|[N2258号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf)|7.0|
|外部模板|[N1987](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm)|7.0|
|空指针常数|[N2431号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf)|7.0|
|强型枚列表|[N2347](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf)|7.0|
|枚舉的转发声明|[N2764](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf) [DR1206](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1206)|7.0|
|标准化属性语法|[N2761号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf)|7.0|
|广义常数表达式|[N2235号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf)|7.0|
|对齐支持|[N2341号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf)|7.0|
|有条件支持行为|[N1627号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf)|7.0|
|将未定义的行为更改为可诊断的错误|[N1727号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1727.pdf)|7.0|
|委托构造函数|[N1986](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf)|7.0|
|继承构造函数|[N2540](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm)|7.0|
|显式转换运算符|[N2437](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf)|7.0|
|新角色类型|[N2249号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html)|7.0|
|Unicode字符串字面值|[N2442号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm)|7.0|
|原始字符串字面值|[N2442号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm)|7.0|
|字面的通用字符名称|[N2170号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html)|7.0|
|用户定义的字面值|[N2765号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf)|7.0|
|标准布局类型|[N2342号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm)|7.0|
|默认函数|[N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm)|7.0|
|已删除的功能|[N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm)|7.0|
|延长的朋友声明|[N1791号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf)|7.0|
|延伸`sizeof`|[N2253](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html) [DR850](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#850)|7.0|
|内联命名空间|[N2535号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm)|7.0|
|不受限制的工会|[N2544号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf)|7.0|
|本地和未命名的类型作为模板参数|[N2657号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm)|7.0|
|基于范围的|[N2930号](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html)|7.0|
|显式虚拟覆盖|[N2928](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm) [N3206N3272](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm)|7.0|
|对垃圾收集和基于可访问性的泄漏检测的最低支持|[N2670号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm)|不适用（见[限制](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#language-restrictions)）|
|允许移动构造函数投掷[noexcept]|[N3050](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html)|7.0|
|定义移动特殊成员功能|[N3053](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3053.html)|7.0|
|**并发性**|   |   |
|序列点|[N2239号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html)||
|原子操作|[N2427号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html)||
|强有力的比较和交换|[N2748](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2748.html)||
|双向围栏|[N2752号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm)||
|记忆模型|[N2429号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm)||
|数据依赖排序：原子和内存模型|[N2664](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm)||
|传播例外情况|[N2179号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html)||
|允许在信号处理程序中使用原子|[N2547](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm)||
|线程本地存储|[N2659号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm)||
|并发的动态初始化和销毁|[N2660号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm)||
|**C++11中的C99功能**|   |   |
|`__func__`预定义标识符|[N2340](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm)|7.0|
|C99预处理器|[N1653号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm)|7.0|
|`long long`|[N1811](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf)|7.0|
|扩展积分类型|[N1988](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf)||

## 18.2.C++14语言功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-14-language-features "这个标题的永久链接")

下表列出了已被C++14标准接受的新语言功能。

表24 C++14语言功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id484 "此表的永久链接")
|语言特征|C++14提案|在nvcc（设备代码）中可用|
|---|---|---|
|调整某些C++上下文转换|[N3323号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3323.pdf)|9.0|
|二进制字面值|[N3472号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3472.pdf)|9.0|
|具有推断返回类型的函数|[N3638号](https://isocpp.org/files/papers/N3638.html)|9.0|
|广义lambda捕获（init-capture）|[N3648号](https://isocpp.org/files/papers/N3648.html)|9.0|
|通用（多态）lambda表达式|[N3649号](https://isocpp.org/files/papers/N3649.html)|9.0|
|可变模板|[N3651号](https://isocpp.org/files/papers/N3651.pdf)|9.0|
|放松对constexpr函数的要求|[N3652号](https://isocpp.org/files/papers/N3652.html)|9.0|
|成员初始化器和聚合|[N3653号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3653.html)|9.0|
|澄清内存分配|[N3664号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3664.html)||
|规模的去分配|[N3778号](https://isocpp.org/files/papers/n3778.html)||
|`[[deprecated]]`属性|[N3760号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html)|9.0|
|单引号作为数字分隔符|[N3781号](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3781.pdf)|9.0|

## 18.3.C++17语言功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-17-language-features "这个标题的永久链接")

nvcc版本11.0及更高版本支持所有C++17语言功能，但受[此处](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cpp17)描述的限制。

## 18.4.C++20语言功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-20-language-features "这个标题的永久链接")

nvcc版本12.0及更高版本支持所有C++20语言功能，但受[此处](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cpp20)描述的限制。

## 18.5.限制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#language-restrictions "这个标题的永久链接")

### 18.5.1.主机编译器扩展[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#host-compiler-extensions "这个标题的永久链接")

设备代码不支持主机编译器特定语言扩展。

`__Complex`类型仅在主机代码中支持。

`__int128`当与支持它的主机编译器一起编译时，设备代码中支持类型。

`__float128`当与支持该类型的主机编译器一起编译时，支持具有计算能力10.0及更高版本的设备。编译器可以在精度较低的浮点表示中处理`__float128`类型的常量表达式。

### 18.5.2.预处理器符号[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#preprocessor-symbols "这个标题的永久链接")

#### 18.5.2.1.__库达_拱门__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arch "这个标题的永久链接")

1. 以下实体的类型签名不应取决于是否定义了`__CUDA_ARCH__`，也不取决于`__CUDA_ARCH__`的特定值：
    
    - `__global__`函数和函数模板
        
    - `__device__`和`__constant__`变量
        
    - 纹理和表面
        
    
    示例：
    ```c++
    #if !defined(__CUDA_ARCH__)
    typedef int mytype;
    #else
    typedef double mytype;
    #endif
    
    __device__ mytype xxx;         // error: xxx's type depends on __CUDA_ARCH__
    __global__ void foo(mytype in, // error: foo's type depends on __CUDA_ARCH__
                        mytype *ptr)
    {
      *ptr = in;
    }
    ```
1. 如果`__global__`函数模板被实例化并从主机启动，那么无论是否定义了`__CUDA_ARCH__`，也无论`__CUDA_ARCH__`的值如何，函数模板都必须使用相同的模板参数实例化。
    
    示例：
    ```c++
    __device__ int result;
    template <typename T>
    __global__ void kern(T in)
    {
      result = in;
    }
    
    __host__ __device__ void foo(void)
    {
    #if !defined(__CUDA_ARCH__)
      kern<<<1,1>>>(1);      // error: "kern<int>" instantiation only
                             // when __CUDA_ARCH__ is undefined!
    #endif
    }
    
    int main(void)
    {
      foo();
      cudaDeviceSynchronize();
      return 0;
    }
    ```
1. 在单独的编译模式下，是否存在具有外部链接的函数或变量的定义不应取决于是否定义了`__CUDA_ARCH__`或`__CUDA_ARCH__`的特定值。
    
    示例：
    
    #if !defined(__CUDA_ARCH__)
    void foo(void) { }                  // error: The definition of foo()
                                        // is only present when __CUDA_ARCH__
                                        // is undefined
    #endif
    
2. 在单独的编译中，`__CUDA_ARCH__`不得用于标题，这样不同的对象可能包含不同的行为。或者，必须保证所有对象都将为相同的compute_arch编译。如果在标题中定义了弱函数或模板函数，并且其行为取决于`__CUDA_ARCH__`，那么如果对象为不同的计算arch编译，则对象中该函数的实例可能会发生冲突。
    
    例如，如果a.h包含：
    ```c++
    template<typename T>
    __device__ T* getptr(void)
    {
    #if __CUDA_ARCH__ == 700
      return NULL; /* no address */
    #else
      __shared__ T arr[256];
      return arr;
    #endif
    }
    ```
    然后，如果`a.cu`和`b.cu`都包含`a.h`并实例化相同类型的`getptr`，`b.cu`期望一个非空地址，并用以下方式编译：
    
    nvcc –arch=compute_70 –dc a.cu
    nvcc –arch=compute_80 –dc b.cu
    nvcc –arch=sm_80 a.o b.o
    
    在链接时，只使用一个版本的`getptr`，因此行为将取决于选择哪个版本。为了避免这种情况，`a.cu`和`b.cu`必须为相同的计算拱进行编译，或者`__CUDA_ARCH__`不应在共享标头函数中使用。
    

编译器不保证会为上述`__CUDA_ARCH__`的不受支持的用途生成诊断。

### 18.5.3.资格赛[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#qualifiers "这个标题的永久链接")

#### 18.5.3.1.设备内存空间指定符[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-space-specifiers "这个标题的永久链接")

`__device__`、`__shared__`、`__managed__`和`__constant__`内存空间指定符不允许：

- `class`，`struct`和`union`数据成员，
    
- 形式参数，
    
- 在主机上执行的函数中的非外部变量声明。
    

`__device__`、`__constant__`和`__managed__`内存空间指定符不允许在设备上执行的函数中既不是外部的也不是静态的变量声明。

`__device__`、`__constant__`、`__managed__`或`__shared__`变量定义不能具有非空构造函数或非空析构函数的类类型。类类型的构造函数在翻译单元的一点上被视为空，如果它要么是微不足道的构造函数，要么满足以下所有条件：

- 建構函式已經定義。
    
- 构造函数没有参数，初始化器列表是空的，函数主体是一个空的复合语句。
    
- 它的类没有虚拟函数，没有虚拟基类，也没有非静态数据成员初始化器。
    
- 其类所有基类的默认构造函数都可以视为空。
    
- 对于其类中所有类类型（或其数组）的非静态数据成员，默认构造函数可以视为空。
    

类的析构器在翻译单元的一点上被视为空，如果它要么是微不足道的析构器，要么满足以下所有条件：

- 析构函数已经定义。
    
- 析构函数体是一个空的复合语句。
    
- 它的类没有虚拟函数，也没有虚拟基类。
    
- 其类的所有基类的析构器都可以视为空。
    
- 对于其类中所有类类型（或其数组）的非静态数据成员，析构器可以视为空。
    

在整个程序编译模式下编译时（有关此模式的描述，请参阅nvcc用户手册），`__device__`、`__shared__`、`__managed__`和`__constant__`变量不能使用`extern`关键字定义为外部变量。唯一的例外是动态分配的`__shared__`变量，如[__shared__](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared)中所述。

在单独的编译模式下编译时（有关此模式的描述，请参阅nvcc用户手册），`__device__`、`__shared__`、`__managed__`和`__constant__`变量可以使用外部关键字定义为外部变量。当`nvlink`找不到外部变量的定义时，将生成错误（除非它是动态分配的`__shared__`变量）。

#### 18.5.3.2. __管理__内存空间指定符[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#managed-memory-space-specifier "这个标题的永久链接")

用`__managed__`内存空间指定符（“托管”变量）标记的变量有以下限制：

- 托管变量的地址不是常数表达式。
    
- 托管变量不应具有恒定限定类型。
    
- 托管变量不应具有参考类型。
    
- 当CUDA运行时可能不处于有效状态时，不得使用托管变量的地址或值，包括以下情况：
    
    - 在静态/动态初始化或销毁具有静态或线程本地存储持续时间的对象中。
        
    - 在调用exit（）后执行的代码中（例如，用gcc的“`__attribute__((destructor))`”标记的函数）。
        
    - 在CUDA运行时可能未初始化时执行的代码中（例如，用gcc的“`__attribute__((constructor))`”标记的函数）。
        
- 托管变量不能用作`decltype()`表达式的无括号id表达式参数。
    
- 托管变量具有与动态分配托管内存相同的一致性和一致性行为。
    
- 当包含托管变量的CUDA程序在具有多个GPU的执行平台上运行时，变量只分配一次，而不是每个GPU。
    
- 在主机上执行的函数中不允许没有外部链接的托管变量声明。
    
- 在设备上执行的函数中，不允许在没有外部或静态链接的情况下进行托管变量声明。
    

以下是合法和非法使用托管变量的示例：

__device__ __managed__ int xxx = 10;         // OK

int *ptr = &xxx;                             // error: use of managed variable
                                             // (xxx) in static initialization
struct S1_t {
  int field;
  S1_t(void) : field(xxx) { };
};
struct S2_t {
  ~S2_t(void) { xxx = 10; }
};

S1_t temp1;                                 // error: use of managed variable
                                            // (xxx) in dynamic initialization

S2_t temp2;                                 // error: use of managed variable
                                            // (xxx) in the destructor of
                                            // object with static storage
                                            // duration

__device__ __managed__ const int yyy = 10;  // error: const qualified type

__device__ __managed__ int &zzz = xxx;      // error: reference type

template <int *addr> struct S3_t { };
S3_t<&xxx> temp;                            // error: address of managed
                                            // variable(xxx) not a
                                            // constant expression

__global__ void kern(int *ptr)
{
  assert(ptr == &xxx);                      // OK
  xxx = 20;                                 // OK
}
int main(void)
{
  int *ptr = &xxx;                          // OK
  kern<<<1,1>>>(ptr);
  cudaDeviceSynchronize();
  xxx++;                                    // OK
  decltype(xxx) qqq;                        // error: managed variable(xxx) used
                                            // as unparenthized argument to
                                            // decltype

  decltype((xxx)) zzz = yyy;                // OK
}

#### 18.5.3.3.挥发性限定符[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#volatile-qualifier "这个标题的永久链接")

笔记

支持`volatile`关键字，以保持与ISO C++的兼容性；然而，其[剩余的非弃用用法](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1152r0.html#prop)很少适用于GPU。

读取和写入易失性限定对象不是原子的，并被编译成一个或多个[.易失性指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation)，不保证：

- 内存操作的排序，或者
    
- 硬件执行的内存操作数量与PTX指令的数量相匹配。
    

也就是说，CUDA C++不稳定不适合：

- **线程间同步**：通过[cuda::atomic_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html)、[cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html)或Atomic[函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)使用原子运算。原子内存操作提供了线程间同步保证，并提供比易失性操作更好的性能。CUDA C++易失性操作不提供任何线程间同步保证，因此线程间同步不正确。以下示例展示了如何使用原子运算在两个线程中传递消息。
    
    > cuda::原子_ref
    > 
    > |   |
    > |---|
    > |__global__ void kernel(int* flag, int* data) {<br>  cuda::atomic_ref<int, cuda::thread_scope_device> f{*flag};<br>  if (threadIdx.x == 0) {<br>    // Consumer: blocks until flag is set by producer, then reads data<br>    while(f.load(cuda::memory_order_acquire) == 0);<br>    if (*data != 42) __trap(); // Errors if wrong data read<br>  } else if (threadIdx.x == 1) {<br>    // Producer: writes data then sets flag<br>    *data = 42;<br>    f.store(1, cuda::memory_order_release);<br>  }<br>}|
    > 
    > 库达::原子原子函数（`atomicAdd`和`atomicExch`）
    
- **内存映射IO（MMIO**）：通过内联PTX使用[PTX MMIO操作](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mmio-operation)。PTX MMIO操作严格保留执行的内存访问次数。CUDA C++易失性操作不保留执行的内存访问数量，并且可能会以非确定性方式执行比请求的访问多或少，使它们对MMIO不正确。以下示例展示了如何使用PTX mmio操作从寄存器中读取和写入。
    
    > __global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
    >   // Write to MMIO register:
    >   int value = 13;
    >   asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;" :: "l"(mmio_reg0), "r"(value) : "memory");
    > 
    >   // Read MMIO register:
    >   asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];" : "=r"(value) : "l"(mmio_reg1) : "memory");
    >   
    >   if (value != 42) __trap(); // Errors if wrong data read
    > }
    

### 18.5.4.指针[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#pointers "这个标题的永久链接")

将指针非引用到在主机上执行的代码中的全局或共享内存，或在设备上执行的代码中的主机内存会导致未定义的行为，最常见的是分割故障和应用程序终止。

通过获取`__device__`、`__shared__`或`__constant__`变量的地址获得的地址只能在设备代码中使用。如[设备内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory)中所述，通过`cudaGetSymbolAddress()`获得的`__device__`或`__constant__`变量的地址只能在主机代码中使用。

### 18.5.5.操作员[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#operators "这个标题的永久链接")

#### 18.5.5.1.分配操作员[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#assignment-operator "这个标题的永久链接")

`__constant__`变量只能通过运行时函数（[设备内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory)）从主机代码中分配；它们不能从设备代码中分配。

`__shared__`变量不能有初始化作为其声明的一部分。

不允许为[内置变量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables)中定义的任何内置变量分配值。

#### 18.5.5.2.地址操作员[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#address-operator "这个标题的永久链接")

不允许取[内置变量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables)中定义的任何内置变量的地址。

### 18.5.6.运行时类型信息（RTTI）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#run-time-type-information-rtti "这个标题的永久链接")

主机代码支持以下与 RTTI 相关的功能，但设备代码不支持。

- `typeid`操作员
    
- `std::type_info`
    
- `dynamic_cast`操作员
    

### 18.5.7.例外处理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#exception-handling "这个标题的永久链接")

例外处理仅支持主机代码，不支持设备代码。

`__global__`函数不支持异常规范。

### 18.5.8.标准图书馆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#standard-library "这个标题的永久链接")

除非另有说明，否则标准库仅在主机代码中支持，但在设备代码中不受支持。

### 18.5.9.命名空间保留[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#namespace-reservations "这个标题的永久链接")

除非另有例外，否则将任何声明或定义添加到`cuda::`, `nv::`, `cooperative_groups::`或嵌套的任何命名空间都是未定义的行为。

示例：

namespace cuda{
   // Bad: class declaration added to namespace cuda
   struct foo{};

   // Bad: function definition added to namespace cuda
   cudaStream_t make_stream(){
      cudaStream_t s;
      cudaStreamCreate(&s);
      return s;
   }
} // namespace cuda

namespace cuda{
   namespace utils{
      // Bad: function definition added to namespace nested within cuda
      cudaStream_t make_stream(){
          cudaStream_t s;
          cudaStreamCreate(&s);
          return s;
      }
   } // namespace utils
} // namespace cuda

namespace utils{
   namespace cuda{
     // Okay: namespace cuda may be used nested within a non-reserved namespace
     cudaStream_t make_stream(){
          cudaStream_t s;
          cudaStreamCreate(&s);
          return s;
      }
   } // namespace cuda
} // namespace utils

// Bad: Equivalent to adding symbols to namespace cuda at global scope
using namespace utils;

### 18.5.10.功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#functions "这个标题的永久链接")

#### 18.5.10.1.外部链接[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#external-linkage "这个标题的永久链接")

仅当函数在与设备代码相同的编译单元中定义时，才允许在某些设备代码中调用使用外部限定符声明的函数，即单个文件或与可重新定位的设备代码和nvlink链接的多个文件。

#### 18.5.10.2.隐式声明和非虚拟显式默认函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#implicitly-declared-and-non-virtual-explicitly-defaulted-functions "这个标题的永久链接")

让F表示一个隐式声明或第一个声明时显式默认的非虚拟函数的函数。`F`的执行空间指定符（`__host__`，`__device__`）是调用它的所有函数的执行空间指定符的组合（请注意，`__global__`调用者将被视为此分析的`__device__`调用者）。例如：

class Base {
  int x;
public:
  __host__ __device__ Base(void) : x(10) {}
};

class Derived : public Base {
  int y;
};

class Other: public Base {
  int z;
};

__device__ void foo(void)
{
  Derived D1;
  Other D2;
}

__host__ void bar(void)
{
  Other D3;
}

Here, the implicitly-declared constructor function “Derived::Derived” will be treated as a `__device__` function, since it is invoked only from the `__device__` function “foo”. The implicitly-declared constructor function “Other::Other” will be treated as a `__host__ __device__` function, since it is invoked both from a `__device__` function “foo” and a `__host__` function “bar”.

此外，如果`F`是隐式声明的虚拟函数（例如，虚拟构造函数），那么如果`D`没有隐式声明，则被`F`覆盖的每个虚拟函数`D`的执行空间都会添加到`F`的执行空间集中。

例如：

struct Base1 { virtual __host__ __device__ ~Base1() { } };
struct Derived1 : Base1 { }; // implicitly-declared virtual destructor
                             // ~Derived1 has __host__ __device__
                             // execution space specifiers

struct Base2 { virtual __device__ ~Base2() = default; };
struct Derived2 : Base2 { }; // implicitly-declared virtual destructor
                             // ~Derived2 has __device__ execution
                             // space specifiers

#### 18.5.10.3.函数参数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-parameters "这个标题的永久链接")

`__global__`函数参数通过常量内存传递到设备，从Volta开始限制为32,764字节，在旧架构上限制为4 KB。

`__global__`函数不能有可变的参数。

`__global__`函数参数不能按引用传递。

在单独的编译模式下，如果`__device__`或`__global__`函数在特定翻译单元中使用ODR，那么该函数的参数和返回类型必须在该翻译单元中完成。

示例：

//first.cu:
struct S;
__device__ void foo(S); // error: type 'S' is incomplete
__device__ auto *ptr = foo;

int main() { }

//second.cu:
struct S { int x; };
__device__ void foo(S) { }

//compiler invocation
$nvcc -std=c++14 -rdc=true first.cu second.cu -o first
nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o', first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
nvlink fatal   : merge_elf failed

##### 18.5.10.3.1. `__global__`函数参数处理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-function-argument-processing "这个标题的永久链接")

当从设备代码中启动`__global__`函数时，每个参数必须是微不足道的可复制和微不足道的可销毁。

当从主机代码中启动`__global__`函数时，每个参数类型都允许非平凡地复制或非平凡地可销毁，但此类类型的处理不遵循标准C++模型，如下所述。用户代码必须确保此工作流程不会影响程序的正确性。工作流程在两个方面与标准C++不同：

1. **Memcpy而不是复制构造函数调用**
    
    当从主机代码中降低`__global__`函数启动时，编译器会生成存根函数，这些函数按值复制参数一次或多次，然后最终使用`memcpy`将参数复制到设备上`__global__`函数的参数内存。即使参数不能平凡地复制，也会发生这种情况，因此可能会破坏复制构造函数有副作用的程序。
    
    示例：
    ```c++
    #include <cassert>
    struct S {
     int x;
     int *ptr;
     __host__ __device__ S() { }
     __host__ __device__ S(const S &) { ptr = &x; }
    };
    
    __global__ void foo(S in) {
     // this assert may fail, because the compiler
     // generated code will memcpy the contents of "in"
     // from host to kernel parameter memory, so the
     // "in.ptr" is not initialized to "&in.x" because
     // the copy constructor is skipped.
     assert(in.ptr == &in.x);
    }
    
    int main() {
      S tmp;
      foo<<<1,1>>>(tmp);
      cudaDeviceSynchronize();
    }
    ```
    示例：
    ```c++
    #include <cassert>
    
    __managed__ int counter;
    struct S1 {
    S1() { }
    S1(const S1 &) { ++counter; }
    };
    
    __global__ void foo(S1) {
    
    /* this assertion may fail, because
       the compiler generates stub
       functions on the host for a kernel
       launch, and they may copy the
       argument by value more than once.
    */
    assert(counter == 1);
    }
    
    int main() {
    S1 V;
    foo<<<1,1>>>(V);
    cudaDeviceSynchronize();
    }
    ```
1. **在``__global__``函数完成之前可以调用析构函数**
    
    内核启动与主机执行异步。因此，如果`__global__`函数参数具有非平凡的析构函数，则在`__global__`函数完成执行之前，析构函数也可能在主机代码中执行。这可能会破坏破坏者有副作用的程序。
    
    示例：
    
    struct S {
     int *ptr;
     S() : ptr(nullptr) { }
     S(const S &) { cudaMallocManaged(&ptr, sizeof(int)); }
     ~S() { cudaFree(ptr); }
    };
    
    __global__ void foo(S in) {
    
      //error: This store may write to memory that has already been
      //       freed (see below).
      *(in.ptr) = 4;
    
    }
    
    int main() {
     S V;
    
     /* The object 'V' is first copied by value to a compiler-generated
      * stub function that does the kernel launch, and the stub function
      * bitwise copies the contents of the argument to kernel parameter
      * memory.
      * However, GPU kernel execution is asynchronous with host
      * execution.
      * As a result, S::~S() will execute when the stub function   returns, releasing allocated memory, even though the kernel may not have finished execution.
      */
     foo<<<1,1>>>(V);
     cudaDeviceSynchronize();
    }
    

##### 18.5.10.3.2.工具包和驱动程序兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#toolkit-and-driver-compatibility "这个标题的永久链接")

开发人员必须使用12.1工具包和r530驱动程序或更高版本来编译、启动和调试接受大于4KB的参数的内核。如果在旧驱动程序上启动此类内核，CUDA将发出错误`CUDA_ERROR_NOT_SUPPORTED`。

##### 18.5.10.3.3.跨工具包修订版的链接兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#link-compatibility-across-toolkit-revisions "这个标题的永久链接")

链接设备对象时，如果至少一个设备对象包含参数大于4KB的内核，开发人员必须在将它们链接在一起之前使用12.1工具包或更高版本重新编译各自设备源的所有对象。如果不这样做，将导致链接器错误。

#### 18.5.10.4.函数中的静态变量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#static-variables-within-function "这个标题的永久链接")

在函数`F`的直接或嵌套块范围内的静态变量`V`的声明中允许变量内存空间指定符，其中：

- `F`是一个`__global__`或`__device__`-only函数。
    
- `F` is a `__host__ __device__` function and `__CUDA_ARCH__` is defined [11](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn17).
    

如果`V`的声明中没有显式内存空间指定符，则在设备编译期间假定隐式`__device__`指定符。

`V`具有与在命名空间范围内声明的具有相同内存空间指定符的变量相同的初始化限制，例如`__device__`变量不能有“非空”构造函数（请参阅[设备内存空间指定符](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-specifiers)）。

函数范围静态变量的合法和非法使用示例如下所示。
```c++
struct S1_t {
  int x;
};

struct S2_t {
  int x;
  __device__ S2_t(void) { x = 10; }
};

struct S3_t {
  int x;
  __device__ S3_t(int p) : x(p) { }
};

__device__ void f1() {
  static int i1;              // OK, implicit __device__ memory space specifier
  static int i2 = 11;         // OK, implicit __device__ memory space specifier
  static __managed__ int m1;  // OK
  static __device__ int d1;   // OK
  static __constant__ int c1; // OK

  static S1_t i3;             // OK, implicit __device__ memory space specifier
  static S1_t i4 = {22};      // OK, implicit __device__ memory space specifier

  static __shared__ int i5;   // OK

  int x = 33;
  static int i6 = x;          // error: dynamic initialization is not allowed
  static S1_t i7 = {x};       // error: dynamic initialization is not allowed

  static S2_t i8;             // error: dynamic initialization is not allowed
  static S3_t i9(44);         // error: dynamic initialization is not allowed
}

__host__ __device__ void f2() {
  static int i1;              // OK, implicit __device__ memory space specifier
                              // during device compilation.
#ifdef __CUDA_ARCH__
  static __device__ int d1;   // OK, declaration is only visible during device
                              // compilation  (__CUDA_ARCH__ is defined)
#else
  static int d0;              // OK, declaration is only visible during host
                              // compilation (__CUDA_ARCH__ is not defined)
#endif

  static __device__ int d2;   // error: __device__ variable inside
                              // a host function during host compilation
                              // i.e. when __CUDA_ARCH__ is not defined

  static __shared__ int i2;  // error: __shared__ variable inside
                             // a host function during host compilation
                             // i.e. when __CUDA_ARCH__ is not defined
}
```
#### 18.5.10.5.函数指针[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-pointers "这个标题的永久链接")

主机代码中的`__global__`函数的地址不能用于设备代码（例如启动内核）。同样，设备代码中的a__global`__global__`函数的地址不能用于主机代码。

不允许在主机代码中取`__device__`函数的地址。

#### 18.5.10.6.函数递归[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-recursion "这个标题的永久链接")

`__global__`函数不支持递归。

#### 18.5.10.7.朋友功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#friend-functions "这个标题的永久链接")

`__global__`函数或函数模板不能在朋友声明中定义。

示例：
```c++
struct S1_t {
  friend __global__
  void foo1(void);  // OK: not a definition
  template<typename T>
  friend __global__
  void foo2(void); // OK: not a definition

  friend __global__
  void foo3(void) { } // error: definition in friend declaration

  template<typename T>
  friend __global__
  void foo4(void) { } // error: definition in friend declaration
};
```
#### 18.5.10.8.操作符功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#operator-function "这个标题的永久链接")

运算符函数不能是`__global__`函数。

#### 18.5.10.9.分配和取消分配功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#allocation-and-deallocation-functions "这个标题的永久链接")

用户定义的`operatornew`、`operatornew[]`、`operatordelete`或`operatordelete[]`不能用于替换编译器提供的相应`__host__`或`__device__`内置。

### 18.5.11.级别[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#classes "这个标题的永久链接")

#### 18.5.11.1.数据成员[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-members "这个标题的永久链接")

不支持静态数据成员，但那些也是const限定的变量除外（请参阅[Const限定变量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#const-variables)）。

#### 18.5.11.2.功能成员[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-members "这个标题的永久链接")

静态成员函数不能是`__global__`函数。

#### 18.5.11.3.虚拟函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-functions "这个标题的永久链接")

当派生类中的函数覆盖基类中的虚拟函数时，覆盖和覆盖函数上的执行空间指定符（即`__host__`、`__device__`）必须匹配。

不允许将具有虚拟函数的类的对象作为参数传递给`__global__`函数。

如果在主机代码中创建了对象，则在设备代码中为该对象调用虚拟函数具有未定义的行为。

如果在设备代码中创建了对象，则在主机代码中为该对象调用虚拟函数具有未定义的行为。

有关使用微软主机编译器时的其他限制，请参阅[Windows-Specific](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#windows-specific)。

示例：
```c++
struct S1 { virtual __host__ __device__ void foo() { } };

__managed__ S1 *ptr1, *ptr2;

__managed__ __align__(16) char buf1[128];
__global__ void kern() {
  ptr1->foo();     // error: virtual function call on a object
                   //        created in host code.
  ptr2 = new(buf1) S1();
}

int main(void) {
  void *buf;
  cudaMallocManaged(&buf, sizeof(S1), cudaMemAttachGlobal);
  ptr1 = new (buf) S1();
  kern<<<1,1>>>();
  cudaDeviceSynchronize();
  ptr2->foo();  // error: virtual function call on an object
                //        created in device code.
}```

#### 18.5.11.4.虚拟基类[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-base-classes "这个标题的永久链接")

不允许将从虚拟基类派生的类的对象作为参数传递给`__global__`函数。

有关使用微软主机编译器时的其他限制，请参阅[Windows-Specific](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#windows-specific)。

#### 18.5.11.5.匿名工会[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#anonymous-unions "这个标题的永久链接")

名称空间范围匿名联合的成员变量不能在`__global__`或`__device__`函数中引用。

#### 18.5.11.6.特定于Windows的[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#windows-specific "这个标题的永久链接")

CUDA编译器遵循IA64 ABI进行类布局，而微软主机编译器则不遵循。让T表示指向成员类型的指针，或满足以下任何条件的类类型：

- `T`具有虚拟功能。
    
- `T`有一个虚拟基类。
    
- `T`具有多个直接或间接空基类的多重继承。
    
- `T`的所有直接和间接基类`B`都是空的，`T`的第一个字段`F`的类型在其定义中使用`B`，因此`B`在`F`的定义中以偏移0排列。
    

让C表示`T`或以`T`为字段类型或基类类型的类类型。CUDA编译器可能计算类布局和大小可能与`C`类型的微软主机编译器不同。

只要`C`专门用于主机或设备代码，程序就应该可以正常工作。

在主机和设备代码之间传递`C`对象具有未定义的行为，例如，作为`__global__`函数的参数或通过`cudaMemcpy*()`调用。

如果对象是在主机代码中创建的，则访问`C`类型的对象或设备代码中的任何子对象，或调用设备代码中的成员函数，则具有未定义的行为。

如果对象是在设备代码[12](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn19)中创建的，则访问`C`类型的对象或主机代码中的任何子对象，或调用主机代码中的成员函数，则具有未定义的行为。

### 18.5.12.模板[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#templates "这个标题的永久链接")

类型或模板不能用于`__global__`函数模板实例化或`__device__/__constant__`变量实例化的类型、非类型或模板模板参数，如果：

- The type or template is defined within a `__host__` or `__host__ __device__`.
    
- 类型或模板是具有`private`或`protected`访问权限的类成员，其父类未在`__device__`或`__global__`函数中定义。
    
- 类型未命名。
    
- 该类型由上述任何类型复合。
    

示例：
```c++
template <typename T>
__global__ void myKernel(void) { }

class myClass {
private:
    struct inner_t { };
public:
    static void launch(void)
    {
       // error: inner_t is used in template argument
       // but it is private
       myKernel<inner_t><<<1,1>>>();
    }
};

// C++14 only
template <typename T> __device__ T d1;

template <typename T1, typename T2> __device__ T1 d2;

void fn() {
  struct S1_t { };
  // error (C++14 only): S1_t is local to the function fn
  d1<S1_t> = {};

  auto lam1 = [] { };
  // error (C++14 only): a closure type cannot be used for
  // instantiating a variable template
  d2<int, decltype(lam1)> = 10;
}
```
### 18.5.13.三叉戔和二叉戳[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#trigraphs-and-digraphs "这个标题的永久链接")

任何平台都不支持Trigraphs。Windows不支持双图。

### 18.5.14.限定变量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#const-qualified-variables "这个标题的永久链接")

让“V”表示具有const限定类型且没有执行空间注释的命名空间范围变量或类静态成员变量（例如，`__device__,__constant__,__shared__`）。V被认为是一个主机代码变量。

V的值可以直接用于设备代码，如果

- V在使用点之前已用常量表达式初始化，
    
- V型不是挥发性限定的，并且
    
- 它有以下类型之一：
    
    - 内置浮点类型，除非微软编译器用作主机编译器，
        
    - 内置积分类型。
        

设备源代码不能包含对V的引用或取V的地址。

示例：

const int xxx = 10;
struct S1_t {  static const int yyy = 20; };

extern const int zzz;
const float www = 5.0;
__device__ void foo(void) {
  int local1[xxx];          // OK
  int local2[S1_t::yyy];    // OK

  int val1 = xxx;           // OK

  int val2 = S1_t::yyy;     // OK

  int val3 = zzz;           // error: zzz not initialized with constant
                            // expression at the point of use.

  const int &val3 = xxx;    // error: reference to host variable
  const int *val4 = &xxx;   // error: address of host variable
  const float val5 = www;   // OK except when the Microsoft compiler is used as
                            // the host compiler.
}
const int zzz = 20;

### 18.5.15.长双[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#long-double "这个标题的永久链接")

设备代码不支持使用`longdouble`。

### 18.5.16.弃用注释[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#deprecation-annotation "这个标题的永久链接")

nvcc支持在使用`gcc`、`clang`、`xlC`、`icc`或`pgcc`主机编译器时使用`deprecated`属性，以及使用`cl.exe`主机编译器时使用`deprecated`declspec。当启用C++14方言时，它还支持`[[deprecated]]`标准属性。当定义`__CUDA_ARCH__`（即在设备编译阶段），CUDA前端编译器将从`__device__`、`__global__`或`__host____device__`函数的主体中生成弃用诊断，用于引用弃用实体。对弃用实体的其他引用将由主机编译器处理，例如，来自`__host__`函数中的引用。

The CUDA frontend compiler does not support the `#pragma gcc diagnostic` or `#pragma warning` mechanisms supported by various host compilers. Therefore, deprecation diagnostics generated by the CUDA frontend compiler are not affected by these pragmas, but diagnostics generated by the host compiler will be affected. To suppress the warning for device-code, user can use NVIDIA specific pragma [#pragma nv_diag_suppress](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-diagnostic-pragmas). The `nvcc` flag `-Wno-deprecated-declarations` can be used to suppress all deprecation warnings, and the flag `-Werror=deprecated-declarations` can be used to turn deprecation warnings into errors.

### 18.5.17.无返回注释[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#noreturn-annotation "这个标题的永久链接")

nvcc supports the use of `noreturn` attribute when using `gcc`, `clang`, `xlC`, `icc` or `pgcc` host compilers, and the use of `noreturn` declspec when using the `cl.exe` host compiler. It also supports the `[[noreturn]]` standard attribute when the C++11 dialect has been enabled.

属性/declspec可用于主机和设备代码。

### 18.5.18. [[可能]] / [[不太可能]] 标准属性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#likely-unlikely-standard-attributes "这个标题的永久链接")

所有支持C++標準属性语法的配置都接受这些属性。属性可用于向设备编译器优化器提示，与任何不包含该语句的替代路径相比，语句被执行的可能性是否更大或更小。

示例：

__device__ int foo(int x) {

 if (i < 10) [[likely]] { // the 'if' block will likely be entered
  return 4;
 }
 if (i < 20) [[unlikely]] { // the 'if' block will not likely be entered
  return 1;
 }
 return 0;
}

如果这些属性在`__CUDA_ARCH__`未定义时在主机代码中使用，那么它们将出现在主机编译器解析的代码中，如果属性不受支持，可能会生成警告。例如，`clang`主机编译器将生成一个“未知属性”警告。

### 18.5.19. const和纯GNU属性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#const-and-pure-gnu-attributes "这个标题的永久链接")

当使用语言方言和主机编译器时，这些属性都支持主机和设备功能，例如g++主机编译器。

对于带有`pure`注释的设备函数，设备代码优化器假设该函数不会改变调用函数可见的任何可变状态（例如内存）。

对于用`const`属性注释的设备函数，设备代码优化器假定该函数不会访问或更改调用函数可见的任何可变状态（例如内存）。

示例：

__attribute__((const)) __device__ int get(int in);

__device__ int doit(int in) {
int sum = 0;

//because 'get' is marked with 'const' attribute
//device code optimizer can recognize that the
//second call to get() can be commoned out.
sum = get(in);
sum += get(in);

return sum;
}

### 18.5.20. __nv_pure__属性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-pure-attribute "这个标题的永久链接")

`__nv_pure__`属性支持主机和设备功能。对于主机函数，当使用支持`pure`属性的语言方言时，`__nv_pure__`属性被翻译成`pure`GNU属性。同样，当使用MSVC作为主机编译器时，该属性被转换为MSVC `noalias`属性。

当设备函数用`__nv_pure__`属性进行注释时，设备代码优化器假定该函数不会改变调用函数可见的任何可变状态（例如内存）。

### 18.5.21.英特尔主机编译器特定[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#intel-host-compiler-specific "这个标题的永久链接")

CUDA前端编译器解析器无法识别英特尔编译器支持的一些内在功能（例如`icc`）。因此，当使用英特尔编译器作为主机编译器时，`nvcc`将在预处理期间启用宏`__INTEL_COMPILER_USE_INTRINSIC_PROTOTYPES`。此宏可以在关联的标头文件中明确声明英特尔编译器的内在函数，允许`nvcc`支持在主机代码[13](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn20)中使用此类函数。

### 18.5.22.C++11功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-11-features "这个标题的永久链接")

nvcc也支持主机编译器默认启用的C++11功能，但受本文档中描述的限制。此外，使用`-std=c++11`标志调用nvcc可以打开所有C++11功能，还可以使用相应的C++11方言选项14调用主机预处理器、编译器和链接器。

#### 18.5.22.1.Lambda表示式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#lambda-expressions "这个标题的永久链接")

编译器导出了与lambda表达式关联的闭包类的所有成员函数[15](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn22)的执行空间指定符如下。如C++11标准所述，编译器在包含lambda表达式的最小块范围、类范围或命名空间范围内创建闭包类型。计算包含闭包类型的最内层函数范围，并将相应函数的执行空间指定符分配给闭包类成员函数。如果没有包含函数范围，执行空间指定符为`__host__`。

lambda表达式和计算执行空间指定符的示例如下所示（在注释中）。

auto globalVar = [] { return 0; }; // __host__

void f1(void) {
  auto l1 = [] { return 1; };      // __host__
}

__device__ void f2(void) {
  auto l2 = [] { return 2; };      // __device__
}

__host__ __device__ void f3(void) {
  auto l3 = [] { return 3; };      // __host__ __device__
}

__device__ void f4(int (*fp)() = [] { return 4; } /* __host__ */) {
}

__global__ void f5(void) {
  auto l5 = [] { return 5; };      // __device__
}

__device__ void f6(void) {
  struct S1_t {
    static void helper(int (*fp)() = [] {return 6; } /* __device__ */) {
    }
  };
}

lambda表达式的闭包类型不能用于`__global__`函数模板实例化的类型或非类型参数，除非lambda是在`__device__`或`__global__`函数中定义的。

示例：
```c++
template <typename T>
__global__ void foo(T in) { };

template <typename T>
struct S1_t { };

void bar(void) {
  auto temp1 = [] { };

  foo<<<1,1>>>(temp1);                    // error: lambda closure type used in
                                          // template type argument
  foo<<<1,1>>>( S1_t<decltype(temp1)>()); // error: lambda closure type used in
                                          // template type argument
}
```
#### 18.5.22.2. std::初始化器_列表[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#std-initializer-list "这个标题的永久链接")

By default, the CUDA compiler will implicitly consider the member functions of `std::initializer_list` to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code. The nvcc flag `--no-host-device-initializer-list` will disable this behavior; member functions of `std::initializer_list` will then be considered as `__host__` functions and will not be directly invokable from device code.

示例：
```c++
#include <initializer_list>

__device__ int foo(std::initializer_list<int> in);

__device__ void bar(void)
  {
    foo({4,5,6});   // (a) initializer list containing only
                    // constant expressions.

    int i = 4;
    foo({i,5,6});   // (b) initializer list with at least one
                    // non-constant element.
                    // This form may have better performance than (a).
  }
```
#### 18.5.22.3.R值引用[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#rvalue-references "这个标题的永久链接")

By default, the CUDA compiler will implicitly consider `std::move` and `std::forward` function templates to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code. The nvcc flag `--no-host-device-move-forward` will disable this behavior; `std::move` and `std::forward` will then be considered as `__host__` functions and will not be directly invokable from device code.

#### 18.5.22.4.Constexpr函数和函数模板[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constexpr-functions-and-function-templates "这个标题的永久链接")

By default, a constexpr function cannot be called from a function with incompatible execution space [16](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn23). The experimental nvcc flag `--expt-relaxed-constexpr` removes this restriction [17](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn24). When this flag is specified, host code can invoke a `__device__` constexpr function and device code can invoke a `__host__` constexpr function. nvcc will define the macro `__CUDACC_RELAXED_CONSTEXPR__` when `--expt-relaxed-constexpr` has been specified. Note that a function template instantiation may not be a constexpr function even if the corresponding template is marked with the keyword `constexpr` (C++11 Standard Section `[dcl.constexpr.p6]`).

#### 18.5.22.5.Constexpr变量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constexpr-variables "这个标题的永久链接")

让“V”表示已标记为constexpr且没有执行空间注释的命名空间范围变量或类静态成员变量（例如，`__device__,__constant__,__shared__`）。V被认为是一个主机代码变量。

如果V是`long`以外的標量类型[18，](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn25)并且该类型不是挥发性限定的，则V的值可以直接在设备代码中使用。此外，如果V是非scalar类型，那么V的scal元素可以在constexpr `__device__`或`__host____device__`函数中使用，如果对函数的调用是常数表达式[19](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn26)。设备源代码不能包含对V的引用或取V的地址。

示例：

constexpr int xxx = 10;
constexpr int yyy = xxx + 4;
struct S1_t { static constexpr int qqq = 100; };

constexpr int host_arr[] = { 1, 2, 3};
constexpr __device__ int get(int idx) { return host_arr[idx]; }

__device__ int foo(int idx) {
  int v1 = xxx + yyy + S1_t::qqq;  // OK
  const int &v2 = xxx;             // error: reference to host constexpr
                                   // variable
  const int *v3 = &xxx;            // error: address of host constexpr
                                   // variable
  const int &v4 = S1_t::qqq;       // error: reference to host constexpr
                                   // variable
  const int *v5 = &S1_t::qqq;      // error: address of host constexpr
                                   // variable

  v1 += get(2);                    // OK: 'get(2)' is a constant
                                   // expression.
  v1 += get(idx);                  // error: 'get(idx)' is not a constant
                                   // expression
  v1 += host_arr[2];               // error: 'host_arr' does not have
                                   // scalar type.
  return v1;
}

#### 18.5.22.6.内联命名空间[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#inline-namespaces "这个标题的永久链接")

对于输入CUDA翻译单元，CUDA编译器可以调用主机编译器来编译翻译单元中的主机代码。在传递给主机编译器的代码中，如果输入CUDA翻译单元包含以下任何实体的定义，CUDA编译器将注入额外的编译器生成代码：

- `__global__`函数或函数模板实例化
    
- `__device__`，`__constant__`
    
- 具有表面或纹理类型的变量
    

编译器生成的代码包含对定义实体的引用。如果实体在内联命名空间中定义，而同一名称和类型签名的另一个实体在封闭命名空间中定义，则此引用可能被主机编译器视为模棱两可，主机编译将失败。

通过为内联命名空间中定义的此类实体使用唯一名称，可以避免这种限制。

示例：

__device__ int Gvar;
inline namespace N1 {
  __device__ int Gvar;
}

// <-- CUDA compiler inserts a reference to "Gvar" at this point in the
// translation unit. This reference will be considered ambiguous by the
// host compiler and compilation will fail.

示例：

inline namespace N1 {
  namespace N2 {
    __device__ int Gvar;
  }
}

namespace N2 {
  __device__ int Gvar;
}

// <-- CUDA compiler inserts reference to "::N2::Gvar" at this point in
// the translation unit. This reference will be considered ambiguous by
// the host compiler and compilation will fail.

##### 18.5.22.6.1.内联未命名的命名空间[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#inline-unnamed-namespaces "这个标题的永久链接")

无法在内联未命名命名空间的命名空间范围内声明以下实体：

- `__managed__`，`__device__`，`__shared__`和`__constant__`变量
    
- `__global__`函数和函数模板
    
- 具有表面或纹理类型的变量
    

示例：
```c++
inline namespace {
  namespace N2 {
    template <typename T>
    __global__ void foo(void);            // error

    __global__ void bar(void) { }         // error

    template <>
    __global__ void foo<int>(void) { }    // error

    __device__ int x1b;                   // error
    __constant__ int x2b;                 // error
    __shared__ int x3b;                   // error

    texture<int> q2;                      // error
    surface<int> s2;                      // error
  }
};```

#### 18.5.22.7.线程_本地[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-local "这个标题的永久链接")

设备代码中不允许使用`thread_local`存储指定符。

#### 18.5.22.8. __global__函数和函数模板[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-functions-and-function-templates "这个标题的永久链接")

如果与lambda表达式关联的闭包类型用于`__global__`函数模板实例化的模板参数，则lambda表达式必须在`__device__`或`__global__`函数的直接或嵌套块范围内定义，或者必须是anextended [lambda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-lambda)。

示例：
```c++
template <typename T>
__global__ void kernel(T in) { }

__device__ void foo_device(void)
{
  // All kernel instantiations in this function
  // are valid, since the lambdas are defined inside
  // a __device__ function.

  kernel<<<1,1>>>( [] __device__ { } );
  kernel<<<1,1>>>( [] __host__ __device__ { } );
  kernel<<<1,1>>>( []  { } );
}
auto lam1 = [] { };

auto lam2 = [] __host__ __device__ { };

void foo_host(void)
{
   // OK: instantiated with closure type of an extended __device__ lambda
   kernel<<<1,1>>>( [] __device__ { } );

   // OK: instantiated with closure type of an extended __host__ __device__
   // lambda
   kernel<<<1,1>>>( [] __host__ __device__ { } );

   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( []  { } );

   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( lam1);

   // error: unsupported: instantiated with closure type of a lambda
   // that is not an extended lambda
   kernel<<<1,1>>>( lam2);
}
```
`__global__`函数或函数模板不能声明为`constexpr`。

`__global__`函数或函数模板不能具有`std::initializer_list`或`va_list`类型的参数。

`__global__`函数不能有rvalue引用类型的参数。

可变量`__global__`函数模板有以下限制：

- 只允许单个包参数。
    
- 包参数必须列在模板参数列表中的最后。
    

示例：
```c++
// ok
template <template <typename...> class Wrapper, typename... Pack>
__global__ void foo1(Wrapper<Pack...>);

// error: pack parameter is not last in parameter list
template <typename... Pack, template <typename...> class Wrapper>
__global__ void foo2(Wrapper<Pack...>);

// error: multiple parameter packs
template <typename... Pack1, int...Pack2, template<typename...> class Wrapper1,
          template<int...> class Wrapper2>
__global__ void foo3(Wrapper1<Pack1...>, Wrapper2<Pack2...>);
```
#### 18.5.22.9. __管理__和__共享__变量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#managed-and-shared-variables "这个标题的永久链接")

`` `__managed__ ``和`__shared__`变量不能用关键字`constexpr`标记。

#### 18.5.22.10.默认函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#defaulted-functions "这个标题的永久链接")

CUDA编译器会忽略在第一个声明中明确默认的非虚拟函数上的执行空间指定符。相反，CUDA编译器将推断[隐式声明和非虚拟显式默认函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compiler-generated-functions)中描述的执行空间指定符。

如果函数是：

- 明确默认，但不是在其第一次声明中。
    
- 明确默认和虚拟。
    

示例：

 struct S1 {
   // warning: __host__ annotation is ignored on a non-virtual function that
   //          is explicitly-defaulted on its first declaration
   __host__ S1() = default;
 };

 __device__ void foo1() {
   //note: __device__ execution space is derived for S1::S1
   //       based on implicit call from within __device__ function
   //       foo1
   S1 s1;
 }

 struct S2 {
   __host__ S2();
 };

 //note: S2::S2 is not defaulted on its first declaration, and
 //      its execution space is fixed to __host__  based on its
 //      first declaration.
 S2::S2() = default;

 __device__ void foo2() {
    // error: call from __device__ function 'foo2' to
    //        __host__ function 'S2::S2'
    S2 s2;
 }

struct S3 {
  //note: S3::~S3 has __host__ execution space
  virtual __host__ ~S3() = default;
};

__device__ void foo3() {
  S3 qqq;
}  /*(implicit destructor call for 'qqq'):
      error: call from a __device__ fuction 'foo3' to a
     __host__ function 'S3::~S3' */

### 18.5.23.C++14功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-14-features "这个标题的永久链接")

nvcc也支持主机编译器默认启用的C++14功能。传递nvcc `-std=c++14`标志会打开所有C++14功能，还可以调用主机预处理器、编译器和链接器以及相应的C++14方言选项[20](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn27)。本节介绍对受支持的C++14功能的限制。

#### 18.5.23.1.具有推断返回类型的函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#functions-with-deduced-return-type "这个标题的永久链接")

`__global__`函数不能有推断的返回类型。

如果`__device__`函数推导出了返回类型，CUDA前端编译器将在调用主机编译器之前将函数声明更改为`void`返回类型。这可能会导致在主机代码中内省`__device__`函数的推断返回类型时出现问题。因此，CUDA编译器将发出编译时错误，用于在设备函数体之外引用此类推断的返回类型，除非`__CUDA_ARCH__`未定义时没有引用。

示例：
```c++
__device__ auto fn1(int x) {
  return x;
}

__device__ decltype(auto) fn2(int x) {
  return x;
}

__device__ void device_fn1() {
  // OK
  int (*p1)(int) = fn1;
}

// error: referenced outside device function bodies
decltype(fn1(10)) g1;

void host_fn1() {
  // error: referenced outside device function bodies
  int (*p1)(int) = fn1;

  struct S_local_t {
    // error: referenced outside device function bodies
    decltype(fn2(10)) m1;

    S_local_t() : m1(10) { }
  };
}

// error: referenced outside device function bodies
template <typename T = decltype(fn2)>
void host_fn2() { }

template<typename T> struct S1_t { };

// error: referenced outside device function bodies
struct S1_derived_t : S1_t<decltype(fn1)> { };
```
#### 18.5.23.2.可变模板[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#variable-templates "这个标题的永久链接")

使用微软主机编译器时，`__device__/__constant__`变量模板不能具有常量限定类型。

示例：
```c++
// error: a __device__ variable template cannot
// have a const qualified type on Windows
template <typename T>
__device__ const T d1(2);

int *const x = nullptr;
// error: a __device__ variable template cannot
// have a const qualified type on Windows
template <typename T>
__device__ T *const d2(x);

// OK
template <typename T>
__device__ const T *d3;

__device__ void fn() {
  int t1 = d1<int>;

  int *const t2 = d2<int>;

  const int *t3 = d3<int>;
}
```
### 18.5.24.C++17功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-17-features "这个标题的永久链接")

nvcc也支持主机编译器默认启用的C++17功能。传递nvcc `-std=c++17`标志可以打开所有C++17功能，并使用相应的C++17方言选项[21](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn28)调用主机预处理器、编译器和链接器。本节介绍受支持的C++17功能的限制。

#### 18.5.24.1.内联变量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#inline-variable "这个标题的永久链接")

- 如果代码在整个程序编译模式下使用nvcc编译，则使用`__device__`或`__constant__`或`__managed__`内存空间指定符声明的命名空间范围内联变量必须具有内部链接。
    
    示例：
    
    inline __device__ int xxx; //error when compiled with nvcc in
                               //whole program compilation mode.
                               //ok when compiled with nvcc in
                               //separate compilation mode.
    
    inline __shared__ int yyy0; // ok.
    
    static inline __device__ int yyy; // ok: internal linkage
    namespace {
    inline __device__ int zzz; // ok: internal linkage
    }
    
- 使用g++主机编译器时，调试器可能看不到使用`__managed__`内存空间指定符声明的内联变量。
    

#### 18.5.24.2.结构化绑定[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#structured-binding "这个标题的永久链接")

结构化绑定不能用可变内存空间指定符声明。

示例：

struct S { int x; int y; };
__device__ auto [a1, b1] = S{4,5}; // error

### 18.5.25.C++20功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-20-features "这个标题的永久链接")

nvcc也支持主机编译器默认启用的C++20功能。传递nvcc `-std=c++20`标志可以打开所有C++20功能，并使用相应的C++20方言选项[22](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn29)调用主机预处理器、编译器和链接器。本节介绍受支持的C++20功能的限制。

#### 18.5.25.1.模块支持[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#module-support "这个标题的永久链接")

CUDA C++、主机或设备代码都不支持模块。`module`、`export`和`import`关键字的使用被诊断为错误。

#### 18.5.25.2.辅助支持[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#coroutine-support "这个标题的永久链接")

设备代码不支持Coroutine。在设备编译期间，在设备函数范围内使用`co_await`、`co_yield`和`co_return`关键字被诊断为错误。

#### 18.5.25.3.三方比较运算符[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#three-way-comparison-operator "这个标题的永久链接")

主机和设备代码都支持三方比较运算符，但一些用途隐含地依赖于主机实现提供的标准模板库的功能。使用这些运算符可能需要指定标志`--expt-relaxed-constexpr`来静音警告，该功能要求主机实现满足设备代码的要求。

示例：
```c++
#include<compare>
struct S {
  int x, y, z;
  auto operator<=>(const S& rhs) const = default;
  __host__ __device__ bool operator<=>(int rhs) const { return false; }
};
__host__ __device__ bool f(S a, S b) {
  if (a <=> 1) // ok, calls a user-defined host-device overload
    return true;
  return a < b; // call to an implicitly-declared function and requires
                // a device-compatible std::strong_ordering implementation
}
```
#### 18.5.25.4.节数函数[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#consteval-functions "这个标题的永久链接")

通常，不允许交叉执行空间调用，并导致编译器诊断（警告或错误）。当调用的函数使用`consteval`指定符声明时，此限制不适用。因此，`__device__`或`__global__`函数可以调用`__host__``consteval`函数，`__host__`函数可以调用`__device__consteval`函数。

示例：
```c++
namespace N1 {
//consteval host function
consteval int hcallee() { return 10; }

__device__ int dfunc() { return hcallee(); /* OK */ }
__global__ void gfunc() { (void)hcallee(); /* OK */ }
__host__ __device__ int hdfunc() { return hcallee();  /* OK */ }
int hfunc() { return hcallee(); /* OK */ }
} // namespace N1

namespace N2 {
//consteval device function
consteval __device__ int dcallee() { return 10; }

__device__ int dfunc() { return dcallee(); /* OK */ }
__global__ void gfunc() { (void)dcallee(); /* OK */ }
__host__ __device__ int hdfunc() { return dcallee();  /* OK */ }
int hfunc() { return dcallee(); /* OK */ }
}
```
## 18.6.多态函数包装器[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#polymorphic-function-wrappers "这个标题的永久链接")

`nvfunctional`标题中提供了多态函数包装器类模板`nvstd::function`。此类模板的实例可用于存储、复制和调用任何可调用的目标，例如lambda表达式。`nvstd::function`可用于主机和设备代码。

示例：
```c++
#include <nvfunctional>

__device__ int foo_d() { return 1; }
__host__ __device__ int foo_hd () { return 2; }
__host__ int foo_h() { return 3; }

__global__ void kernel(int *result) {
  nvstd::function<int()> fn1 = foo_d;
  nvstd::function<int()> fn2 = foo_hd;
  nvstd::function<int()> fn3 =  []() { return 10; };

  *result = fn1() + fn2() + fn3();
}

__host__ __device__ void hostdevice_func(int *result) {
  nvstd::function<int()> fn1 = foo_hd;
  nvstd::function<int()> fn2 =  []() { return 10; };

  *result = fn1() + fn2();
}

__host__ void host_func(int *result) {
  nvstd::function<int()> fn1 = foo_h;
  nvstd::function<int()> fn2 = foo_hd;
  nvstd::function<int()> fn3 =  []() { return 10; };

  *result = fn1() + fn2() + fn3();
}
```
主机代码中`nvstd::function`的实例不能用`__device__`函数的地址或`operator()`是`__device__`函数的函子初始化。设备代码中的`nvstd::function`实例不能用`__host__`函数的地址或`operator()`是`__host__`函数的函子初始化。

`nvstd::function`实例不能在运行时从主机代码传递到设备代码（反之亦然）。如果`__global__`函数是从主机代码启动的，则不能在`__global__`函数的参数类型中使用`nvstd::function`。

示例：
```c++
#include <nvfunctional>

__device__ int foo_d() { return 1; }
__host__ int foo_h() { return 3; }
auto lam_h = [] { return 0; };

__global__ void k(void) {
  // error: initialized with address of __host__ function
  nvstd::function<int()> fn1 = foo_h;

  // error: initialized with address of functor with
  // __host__ operator() function
  nvstd::function<int()> fn2 = lam_h;
}

__global__ void kern(nvstd::function<int()> f1) { }

void foo(void) {
  // error: initialized with address of __device__ function
  nvstd::function<int()> fn1 = foo_d;

  auto lam_d = [=] __device__ { return 1; };

  // error: initialized with address of functor with
  // __device__ operator() function
  nvstd::function<int()> fn2 = lam_d;

  // error: passing nvstd::function from host to device
  kern<<<1,1>>>(fn2);
}

`nvstd::function`在`nvfunctional`标题中定义如下：

namespace nvstd {
  template <class _RetType, class ..._ArgTypes>
  class function<_RetType(_ArgTypes...)>
  {
    public:
      // constructors
      __device__ __host__  function() noexcept;
      __device__ __host__  function(nullptr_t) noexcept;
      __device__ __host__  function(const function &);
      __device__ __host__  function(function &&);

      template<class _F>
      __device__ __host__  function(_F);

      // destructor
      __device__ __host__  ~function();

      // assignment operators
      __device__ __host__  function& operator=(const function&);
      __device__ __host__  function& operator=(function&&);
      __device__ __host__  function& operator=(nullptr_t);
      __device__ __host__  function& operator=(_F&&);

      // swap
      __device__ __host__  void swap(function&) noexcept;

      // function capacity
      __device__ __host__  explicit operator bool() const noexcept;

      // function invocation
      __device__ _RetType operator()(_ArgTypes...) const;
  };

  // null pointer comparisons
  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator==(const function<_R(_ArgTypes...)>&, nullptr_t) noexcept;

  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator==(nullptr_t, const function<_R(_ArgTypes...)>&) noexcept;

  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator!=(const function<_R(_ArgTypes...)>&, nullptr_t) noexcept;

  template <class _R, class... _ArgTypes>
  __device__ __host__
  bool operator!=(nullptr_t, const function<_R(_ArgTypes...)>&) noexcept;

  // specialized algorithms
  template <class _R, class... _ArgTypes>
  __device__ __host__
  void swap(function<_R(_ArgTypes...)>&, function<_R(_ArgTypes...)>&);
}
```
## 18.7.扩展的Lambdas[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-lambdas "这个标题的永久链接")

nvcc标志`'--extended-lambda'`允许在lambda表达式[23](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn30)中显式执行空间注释。执行空间注释应出现在`'--extended-lambda'`之后和可选的“lambda-declarator”之前。当指定了“--extended-lambda”标志时，nvcc将定义宏__CUDACC_EXTENDED_LAMBDA__。

An ‘extended `__device__` lambda’ is a lambda expression that is annotated explicitly with ‘`__device__`’, and is defined within the immediate or nested block scope of a `__host__` or `__host__ __device__` function.

An ‘extended `__host__ __device__` lambda’ is a lambda expression that is annotated explicitly with both ‘`__host__`’ and ‘`__device__`’, and is defined within the immediate or nested block scope of a `__host__` or `__host__ __device__` function.

An ‘extended lambda’ denotes either an extended `__device__` lambda or an extended `__host__ __device__` lambda. Extended lambdas can be used in the type arguments of [__global__ function template instantiation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cpp11-global).

如C++11支持部分所述，如果执行空间注释未明确指定，则根据包含与lambda关联的闭包类的范围进行计算。执行空间注释应用于与lambda关联的闭包类的所有方法。

示例：

void foo_host(void) {
  // not an extended lambda: no explicit execution space annotations
  auto lam1 = [] { };

  // extended __device__ lambda
  auto lam2 = [] __device__ { };

  // extended __host__ __device__ lambda
  auto lam3 = [] __host__ __device__ { };

  // not an extended lambda: explicitly annotated with only '__host__'
  auto lam4 = [] __host__ { };
}

__host__ __device__ void foo_host_device(void) {
  // not an extended lambda: no explicit execution space annotations
  auto lam1 = [] { };

  // extended __device__ lambda
  auto lam2 = [] __device__ { };

  // extended __host__ __device__ lambda
  auto lam3 = [] __host__ __device__ { };

  // not an extended lambda: explicitly annotated with only '__host__'
  auto lam4 = [] __host__ { };
}

__device__ void foo_device(void) {
  // none of the lambdas within this function are extended lambdas,
  // because the enclosing function is not a __host__ or __host__ __device__
  // function.
  auto lam1 = [] { };
  auto lam2 = [] __device__ { };
  auto lam3 = [] __host__ __device__ { };
  auto lam4 = [] __host__ { };
}

// lam1 and lam2 are not extended lambdas because they are not defined
// within a __host__ or __host__ __device__ function.
auto lam1 = [] { };
auto lam2 = [] __host__ __device__ { };

### 18.7.1.扩展的Lambda类型特征[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-lambda-type-traits "这个标题的永久链接")

编译器提供类型特征，以在编译时检测扩展lambda的闭包类型：

`__nv_is_extended_device_lambda_closure_type(type)`：如果“类型”是为扩展的`__device__`lambda创建的闭包类，则该特征为真，否则为假。

`__nv_is_extended_device_lambda_with_preserved_return_type(type)`: If ‘type’ is the closure class created for an extended `__device__` lambda and the lambda is defined with trailing return type (with restriction), then the trait is true, otherwise it is false. If the trailing return type definition refers to any lambda parameter name, the return type is not preserved.

`__nv_is_extended_host_device_lambda_closure_type(type)`: If ‘type’ is the closure class created for an extended `__host__ __device__` lambda, then the trait is true, otherwise it is false.

这些特征可以在所有编译模式下使用，无论是否启用了lambda或扩展lambda24。

示例：

#define IS_D_LAMBDA(X) __nv_is_extended_device_lambda_closure_type(X)
#define IS_DPRT_LAMBDA(X) __nv_is_extended_device_lambda_with_preserved_return_type(X)
#define IS_HD_LAMBDA(X) __nv_is_extended_host_device_lambda_closure_type(X)

auto lam0 = [] __host__ __device__ { };

void foo(void) {
  auto lam1 = [] { };
  auto lam2 = [] __device__ { };
  auto lam3 = [] __host__ __device__ { };
  auto lam4 = [] __device__ () --> double { return 3.14; }
  auto lam5 = [] __device__ (int x) --> decltype(&x) { return 0; }

  // lam0 is not an extended lambda (since defined outside function scope)
  static_assert(!IS_D_LAMBDA(decltype(lam0)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam0)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam0)), "");

  // lam1 is not an extended lambda (since no execution space annotations)
  static_assert(!IS_D_LAMBDA(decltype(lam1)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam1)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam1)), "");

  // lam2 is an extended __device__ lambda
  static_assert(IS_D_LAMBDA(decltype(lam2)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam2)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam2)), "");

  // lam3 is an extended __host__ __device__ lambda
  static_assert(!IS_D_LAMBDA(decltype(lam3)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam3)), "");
  static_assert(IS_HD_LAMBDA(decltype(lam3)), "");

  // lam4 is an extended __device__ lambda with preserved return type
  static_assert(IS_D_LAMBDA(decltype(lam4)), "");
  static_assert(IS_DPRT_LAMBDA(decltype(lam4)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam4)), "");

  // lam5 is not an extended __device__ lambda with preserved return type
  // because it references the operator()'s parameter types in the trailing return type.
  static_assert(IS_D_LAMBDA(decltype(lam5)), "");
  static_assert(!IS_DPRT_LAMBDA(decltype(lam5)), "");
  static_assert(!IS_HD_LAMBDA(decltype(lam5)), "");
}

### 18.7.2.扩展的Lambda限制[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-lambda-restrictions "这个标题的永久链接")

在调用主机编译器之前，CUDA编译器将用命名空间范围内定义的占位符类型的实例替换扩展的lambda表达式。占位符类型的模板参数需要取包含原始扩展lambda表达式的函数的地址。正确执行任何`__global__`函数模板都需要这个，其模板参数涉及扩展lambda的闭包类型。_封闭函数_的计算方式如下。

根据定义，扩展的lambda存在于`__host__`或`__host____device__`函数的直接或嵌套块范围内。如果这个函数不是lambda表达式的`operator()`），那么它被认为是扩展lambda的包围函数。否则，扩展的lambda是在一个或多个包含lambda表达式的`operator()`的直接或嵌套块范围内定义的。如果最外层的这种lambda表达式是在函数`F`的直接或嵌套块范围内定义的，那么`F`是计算的包围函数，否则包围函数不存在。

示例：

void foo(void) {
  // enclosing function for lam1 is "foo"
  auto lam1 = [] __device__ { };

  auto lam2 = [] {
     auto lam3 = [] {
        // enclosing function for lam4 is "foo"
        auto lam4 = [] __host__ __device__ { };
     };
  };
}

auto lam6 = [] {
  // enclosing function for lam7 does not exist
  auto lam7 = [] __host__ __device__ { };
};

以下是对扩展lambda的限制：

1. 扩展的lambda不能在另一个扩展的lambda表达式中定义。
    
    示例：
    
    void foo(void) {
      auto lam1 = [] __host__ __device__  {
        // error: extended lambda defined within another extended lambda
        auto lam2 = [] __host__ __device__ { };
      };
    }
    
2. 扩展的lambda不能在通用lambda表示式中定义。
    
    示例：
    
    void foo(void) {
      auto lam1 = [] (auto) {
        // error: extended lambda defined within a generic lambda
        auto lam2 = [] __host__ __device__ { };
      };
    }
    
3. 如果扩展的lambda是在一个或多个嵌套lambda表达式的直接或嵌套块范围内定义的，则最外层的lambda表达式必须在函数的直接或嵌套块范围内定义。
    
    示例：
    
    auto lam1 = []  {
      // error: outer enclosing lambda is not defined within a
      // non-lambda-operator() function.
      auto lam2 = [] __host__ __device__ { };
    };
    
4. 必须命名扩展lambda的包围函数，并且可以获取其地址。如果包含函数是类成员，则必须满足以下条件：
    
    - 包含成员函数的所有类都必须有一个名称。
        
    - 成员函数不得在其父类内拥有私有或受保护的访问权限。
        
    - 所有封闭类不得在其各自的父类中拥有私有或受保护的访问权限。
        
    
    示例：
    
    void foo(void) {
      // OK
      auto lam1 = [] __device__ { return 0; };
      {
        // OK
        auto lam2 = [] __device__ { return 0; };
        // OK
        auto lam3 = [] __device__ __host__ { return 0; };
      }
    }
    
    struct S1_t {
      S1_t(void) {
        // Error: cannot take address of enclosing function
        auto lam4 = [] __device__ { return 0; };
      }
    };
    
    class C0_t {
      void foo(void) {
        // Error: enclosing function has private access in parent class
        auto temp1 = [] __device__ { return 10; };
      }
      struct S2_t {
        void foo(void) {
          // Error: enclosing class S2_t has private access in its
          // parent class
          auto temp1 = [] __device__ { return 10; };
        }
      };
    };
    
5. 在定义了扩展的lambda时，必须能够明确地取包围例程的地址。在某些情况下，这可能不可行，例如，当类typedef阴影相同名称的模板类型参数时。
    
    示例：
    ```c++
    template <typename> struct A {
      typedef void Bar;
      void test();
    };
    
    template<> struct A<void> { };
    
    template <typename Bar>
    void A<Bar>::test() {
      /* In code sent to host compiler, nvcc will inject an
         address expression here, of the form:
         (void (A< Bar> ::*)(void))(&A::test))
    
         However, the class typedef 'Bar' (to void) shadows the
         template argument 'Bar', causing the address
         expression in A<int>::test to actually refer to:
         (void (A< void> ::*)(void))(&A::test))
    
         ..which doesn't take the address of the enclosing
         routine 'A<int>::test' correctly.
      */
      auto lam1 = [] __host__ __device__ { return 4; };
    }
    
    int main() {
      A<int> xxx;
      xxx.test();
    }
    ```
1. 扩展的lambda无法在函数本地的类中定义。
    
    示例：
    
    void foo(void) {
      struct S1_t {
        void bar(void) {
          // Error: bar is member of a class that is local to a function.
          auto lam4 = [] __host__ __device__ { return 0; };
        }
      };
    }
    
2. 扩展lambda的包围函数不能有推断的返回类型。
    
    示例：
    
    auto foo(void) {
      // Error: the return type of foo is deduced.
      auto lam1 = [] __host__ __device__ { return 0; };
    }
    
3. __host__ __device__ extended lambda不能是通用的lambda。
    
    示例：
    
    void foo(void) {
      // Error: __host__ __device__ extended lambdas cannot be
      // generic lambdas.
      auto lam1 = [] __host__ __device__ (auto i) { return i; };
    
      // Error: __host__ __device__ extended lambdas cannot be
      // generic lambdas.
      auto lam2 = [] __host__ __device__ (auto ...i) {
                   return sizeof...(i);
                  };
    }
    
4. 如果包含函数是函数模板或成员函数模板的实例化，并且/或者该函数是类模板的成员，则该模板必须满足以下约束：
    
    - 模板最多必须有一个变量参数，并且它必须列在模板参数列表中的最后。
        
    - 模板参数必须命名。
        
    - 模板实例化参数类型不能涉及函数的本地类型（扩展lambda的闭包类型除外）或私有或受保护的类成员。
        
    
    示例：
    ```c++
    template <typename T>
    __global__ void kern(T in) { in(); }
    
    template <typename... T>
    struct foo {};
    
    template < template <typename...> class T, typename... P1,
              typename... P2>
    void bar1(const T<P1...>, const T<P2...>) {
      // Error: enclosing function has multiple parameter packs
      auto lam1 =  [] __device__ { return 10; };
    }
    
    template < template <typename...> class T, typename... P1,
              typename T2>
    void bar2(const T<P1...>, T2) {
      // Error: for enclosing function, the
      // parameter pack is not last in the template parameter list.
      auto lam1 =  [] __device__ { return 10; };
    }
    
    template <typename T, T>
    void bar3(void) {
      // Error: for enclosing function, the second template
      // parameter is not named.
      auto lam1 =  [] __device__ { return 10; };
    }
    
    int main() {
      foo<char, int, float> f1;
      foo<char, int> f2;
      bar1(f1, f2);
      bar2(f1, 10);
      bar3<int, 10>();
    }
    ```
    示例：
```c++
    template <typename T>
    __global__ void kern(T in) { in(); }
    
    template <typename T>
    void bar4(void) {
      auto lam1 =  [] __device__ { return 10; };
      kern<<<1,1>>>(lam1);
    }
    
    struct C1_t { struct S1_t { }; friend int main(void); };
    int main() {
      struct S1_t { };
      // Error: enclosing function for device lambda in bar4
      // is instantiated with a type local to main.
      bar4<S1_t>();
    
      // Error: enclosing function for device lambda in bar4
      // is instantiated with a type that is a private member
      // of a class.
      bar4<C1_t::S1_t>();
    }
    ```
5. 使用Visual Studio主机编译器，包含函数必须有外部链接。存在限制是因为这个主机编译器不支持使用非外部链接函数的地址作为模板参数，CUDA编译器转换需要这个地址来支持扩展的lambda。
    
6. 使用Visual Studio主机编译器，不应在“if-constexpr”块的主体中定义扩展的lambda。
    
7. 扩展lambda对捕获的变量有以下限制：
    
    - 在发送到主机编译器的代码中，变量可以通过值传递给帮助函数序列，然后用于直接初始化用于表示扩展lambda25的闭包类型的类类型的字段。
        
    - 变量只能通过值捕获。
        
    - 如果陣列尺寸的數量大於7，則無法捕獲陣列型別的變數。
        
    - 对于数组类型的变量，在发送到主机编译器的代码中，闭包类型的数组字段首先默认初始化，然后从捕获的数组变量的相应元素中复制分配数组字段的每个元素。因此，阵列元素类型必须在主机代码中默认可构造和可复制分配。
        
    - 不能捕获作为可变量参数包元素的函数参数。
        
    - 捕获变量的类型不能涉及函数的本地类型（扩展lambda的闭包类型除外）或私有或受保护的类成员。
        
    - 对于__host__ __device__扩展lambda，lambda表达式`operator()`的返回或参数类型中使用的类型不能涉及函数的本地类型（扩展lambda的闭包类型除外）或私有或受保护的类成员。
        
    - __host__ __device__ extended lambdas不支持Init-capture。init-capture支持__device__扩展lambdas，除非init-capture为数组类型或`std::initializer_list`类型。
        
    - 扩展lambda的函数调用运算符不是constexpr。扩展lambda的闭包类型不是字面类型。constexpr和consteval指定符不能用于扩展lambda的声明。
        
    - 变量不能隐式捕获在词汇嵌套在扩展lambda中的if-constexpr块中，除非它早些时候在if-constexpr块外隐式捕获或出现在扩展lambda的显式捕获列表中（见下面的示例）。
        
    
    示例：
    ```c++
    void foo(void) {
      // OK: an init-capture is allowed for an
      // extended __device__ lambda.
      auto lam1 = [x = 1] __device__ () { return x; };
    
      // Error: an init-capture is not allowed for
      // an extended __host__ __device__ lambda.
      auto lam2 = [x = 1] __host__ __device__ () { return x; };
    
      int a = 1;
      // Error: an extended __device__ lambda cannot capture
      // variables by reference.
      auto lam3 = [&a] __device__ () { return a; };
    
      // Error: by-reference capture is not allowed
      // for an extended __device__ lambda.
      auto lam4 = [&x = a] __device__ () { return x; };
    
      struct S1_t { };
      S1_t s1;
      // Error: a type local to a function cannot be used in the type
      // of a captured variable.
      auto lam6 = [s1] __device__ () { };
    
      // Error: an init-capture cannot be of type std::initializer_list.
      auto lam7 = [x = {11}] __device__ () { };
    
      std::initializer_list<int> b = {11,22,33};
      // Error: an init-capture cannot be of type std::initializer_list.
      auto lam8 = [x = b] __device__ () { };
    
      // Error scenario (lam9) and supported scenarios (lam10, lam11)
      // for capture within 'if-constexpr' block
      int yyy = 4;
      auto lam9 = [=] __device__ {
        int result = 0;
        if constexpr(false) {
          //Error: An extended __device__ lambda cannot first-capture
          //      'yyy' in constexpr-if context
          result += yyy;
        }
        return result;
      };
    
      auto lam10 = [yyy] __device__ {
        int result = 0;
        if constexpr(false) {
          //OK: 'yyy' already listed in explicit capture list for the extended lambda
          result += yyy;
        }
        return result;
      };
    
      auto lam11 = [=] __device__ {
        int result = yyy;
        if constexpr(false) {
          //OK: 'yyy' already implicit captured outside the 'if-constexpr' block
          result += yyy;
        }
        return result;
      };
    }
    
8. 解析函数时，CUDA编译器会为该函数中的每个扩展lambda分配一个计数器值。此计数器值用于传递给主机编译器的替换命名类型。因此，是否在函数中定义扩展lambda不应取决于`__CUDA_ARCH__`的特定值，或`__CUDA_ARCH__`未定义。
    
    示例：
    ```c++
    template <typename T>
    __global__ void kernel(T in) { in(); }
    
    __host__ __device__ void foo(void) {
      // Error: the number and relative declaration
      // order of extended lambdas depends on
      // __CUDA_ARCH__
    #if defined(__CUDA_ARCH__)
      auto lam1 = [] __device__ { return 0; };
      auto lam1b = [] __host___ __device__ { return 10; };
    #endif
      auto lam2 = [] __device__ { return 4; };
      kernel<<<1,1>>>(lam2);
    }
    ```
5. As described above, the CUDA compiler replaces a `__device__` extended lambda defined in a host function with a placeholder type defined in namespace scope. Unless the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns true for the closure type of the extended lambda, the placeholder type does not define a `operator()` function equivalent to the original lambda declaration. An attempt to determine the return type or parameter types of the `operator()` function of such a lambda may therefore work incorrectly in host code, as the code processed by the host compiler will be semantically different than the input code processed by the CUDA compiler. However, it is OK to introspect the return type or parameter types of the `operator()` function within device code. Note that this restriction does not apply to `__host__ __device__` extended lambdas, or to `__device__` extended lambdas for which the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns true.
    
    示例：
    
    #include <type_traits>
    const char& getRef(const char* p) { return *p; }
    
    void foo(void) {
      auto lam1 = [] __device__ { return "10"; };
    
      // Error: attempt to extract the return type
      // of a __device__ lambda in host code
      std::result_of<decltype(lam1)()>::type xx1 = "abc";
    
      auto lam2 = [] __host__ __device__  { return "10"; };
    
      // OK : lam2 represents a __host__ __device__ extended lambda
      std::result_of<decltype(lam2)()>::type xx2 = "abc";
    
      auto lam3 = []  __device__ () -> const char * { return "10"; };
    
      // OK : lam3 represents a __device__ extended lambda with preserved return type
      std::result_of<decltype(lam3)()>::type xx2 = "abc";
      static_assert( std::is_same_v< std::result_of<decltype(lam3)()>::type, const char *>);
    
      auto lam4 = [] __device__ (char x) -> decltype(getRef(&x)) { return 0; };
      // lam4's return type is not preserved because it references the operator()'s
      // parameter types in the trailing return type.
      static_assert( ! __nv_is_extended_device_lambda_with_preserved_return_type(decltype(lam4)), "" );
    }
    
6. 对于扩展设备lambda：-内省运算符（）的参数类型仅在设备代码中支持。-内省运算符（）的返回类型仅在设备代码中支持，除非特征函数__nv_is_extended_device_lambda_with_preserved_return_type（）返回true。
    
7. 如果由扩展lambda表示的函子对象从主机传递到设备代码（例如，作为`__global__`函数的参数），那么lambda表达式正文中捕获变量的任何表达式都必须保持不变，无论`__CUDA_ARCH__`宏是否定义，以及宏是否具有特定值。出现此限制是因为lambda的闭包类布局取决于编译器处理lambda表达式时遇到捕获变量的顺序；如果设备和主机编译中的闭包类布局不同，程序可能会错误地执行。
    
    示例：
    ```c++
    __device__ int result;
    
    template <typename T>
    __global__ void kernel(T in) { result = in(); }
    
    void foo(void) {
      int x1 = 1;
      auto lam1 = [=] __host__ __device__ {
        // Error: "x1" is only captured when __CUDA_ARCH__ is defined.
    #ifdef __CUDA_ARCH__
        return x1 + 1;
    #else
        return 10;
    #endif
      };
      kernel<<<1,1>>>(lam1);
    }
    ```
5. As described previously, the CUDA compiler replaces an extended `__device__` lambda expression with an instance of a placeholder type in the code sent to the host compiler. This placeholder type does not define a pointer-to-function conversion operator in host code, however the conversion operator is provided in device code. Note that this restriction does not apply to `__host__ __device__` extended lambdas.
    
    示例：
    ```c++
    template <typename T>
    __global__ void kern(T in) {
      int (*fp)(double) = in;
    
      // OK: conversion in device code is supported
      fp(0);
      auto lam1 = [](double) { return 1; };
    
      // OK: conversion in device code is supported
      fp = lam1;
      fp(0);
    }
    
    void foo(void) {
      auto lam_d = [] __device__ (double) { return 1; };
      auto lam_hd = [] __host__ __device__ (double) { return 1; };
      kern<<<1,1>>>(lam_d);
      kern<<<1,1>>>(lam_hd);
    
      // OK : conversion for __host__ __device__ lambda is supported
      // in host code
      int (*fp)(double) = lam_hd;
    
      // Error: conversion for __device__ lambda is not supported in
      // host code.
      int (*fp2)(double) = lam_d;
    }
    ```
5. As described previously, the CUDA compiler replaces an extended `__device__` or `__host__ __device__` lambda expression with an instance of a placeholder type in the code sent to the host compiler. This placeholder type may define C++ special member functions (e.g. constructor, destructor). As a result, some standard C++ type traits may return different results for the closure type of the extended lambda, in the CUDA frontend compiler versus the host compiler. The following type traits are affected: `std::is_trivially_copyable`, `std::is_trivially_constructible`, `std::is_trivially_copy_constructible`, `std::is_trivially_move_constructible`, `std::is_trivially_destructible`.
    
    Care must be taken that the results of these type traits are not used in `__global__` function template instantiation or in `__device__ /__constant__ / __managed__` variable template instantiation.
    
    示例：
    ```c++
    template <bool b>
    void __global__ foo() { printf("hi"); }
    
    template <typename T>
    void dolaunch() {
    
    // ERROR: this kernel launch may fail, because CUDA frontend compiler
    // and host compiler may disagree on the result of
    // std::is_trivially_copyable() trait on the closure type of the
    // extended lambda
    foo<std::is_trivially_copyable<T>::value><<<1,1>>>();
    cudaDeviceSynchronize();
    }
    
    int main() {
    int x = 0;
    auto lam1 = [=] __host__ __device__ () { return x; };
    dolaunch<decltype(lam1)>();
    }
    
```
CUDA编译器将为1-12中描述的案例子集生成编译器诊断；不会为案例13-17生成诊断，但主机编译器可能无法编译生成的代码。

### 18.7.3.关于__host__ __device__ lambdas的注释[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#notes-on-host-device-lambdas "这个标题的永久链接")

Unlike `__device__` lambdas, `__host__ __device__` lambdas can be called from host code. As described earlier, the CUDA compiler replaces an extended lambda expression defined in host code with an instance of a named placeholder type. The placeholder type for an extended `__host____device__` lambda invokes the original lambda’s `operator()` with an indirect function call [24](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn31).

The presence of the indirect function call may cause an extended `__host__ __device__` lambda to be less optimized by the host compiler than lambdas that are implicitly or explicitly `__host__` only. In the latter case, the host compiler can easily inline the body of the lambda into the calling context. But in case of an extended `__host__                                  __device__` lambda, the host compiler encounters the indirect function call and may not be able to easily inline the original `__host__ __device__` lambda body.

### 18.7.4. *按值捕获此[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#this-capture-by-value "这个标题的永久链接")

When a lambda is defined within a non-static class member function, and the body of the lambda refers to a class member variable, C++11/C++14 rules require that the `this` pointer of the class is captured by value, instead of the referenced member variable. If the lambda is an extended `__device__` or `__host__``__device__` lambda defined in a host function, and the lambda is executed on the GPU, accessing the referenced member variable on the GPU will cause a run time error if the `this` pointer points to host memory.

示例：
```c++
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("\n value = %d", in()); }

struct S1_t {
  int xxx;
  __host__ __device__ S1_t(void) : xxx(10) { };

  void doit(void) {

    auto lam1 = [=] __device__ {
       // reference to "xxx" causes
       // the 'this' pointer (S1_t*) to be captured by value
       return xxx + 1;

    };

    // Kernel launch fails at run time because 'this->xxx'
    // is not accessible from the GPU
    foo<<<1,1>>>(lam1);
    cudaDeviceSynchronize();
  }
};

int main(void) {
  S1_t s1;
  s1.doit();
}
```
C++17通过添加一个新的“*this”捕获模式来解决这个问题。在此模式下，编译器复制了用“*this”表示的对象，而不是通过值捕获指针`this`。“*this”捕获模式在此处进行了更详细的描述：`http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html`。

The CUDA compiler supports the “*this” capture mode for lambdas defined within `__device__` and `__global__` functions and for extended `__device__` lambdas defined in host code, when the `--extended-lambda` nvcc flag is used.

以下是修改为使用“*this”捕获模式的上述示例：
```c++
#include <cstdio>

template <typename T>
__global__ void foo(T in) { printf("\n value = %d", in()); }

struct S1_t {
  int xxx;
  __host__ __device__ S1_t(void) : xxx(10) { };

  void doit(void) {

    // note the "*this" capture specification
    auto lam1 = [=, *this] __device__ {

       // reference to "xxx" causes
       // the object denoted by '*this' to be captured by
       // value, and the GPU code will access copy_of_star_this->xxx
       return xxx + 1;

    };

    // Kernel launch succeeds
    foo<<<1,1>>>(lam1);
    cudaDeviceSynchronize();
  }
};

int main(void) {
  S1_t s1;
  s1.doit();
}
```
除非所选语言方言启用了“*this”捕获，否则不允许用于主机代码中定义的未注释的lambda或扩展的`__host__``__device__`lambdas。支持和不支持的使用示例：

struct S1_t {
  int xxx;
  __host__ __device__ S1_t(void) : xxx(10) { };

  void host_func(void) {

    // OK: use in an extended __device__ lambda
    auto lam1 = [=, *this] __device__ { return xxx; };

    // Use in an extended __host__ __device__ lambda
    // Error if *this capture not enabled by language dialect
    auto lam2 = [=, *this] __host__ __device__ { return xxx; };

    // Use in an unannotated lambda in host function
    // Error if *this capture not enabled by language dialect
    auto lam3 = [=, *this]  { return xxx; };
  }

  __device__ void device_func(void) {

    // OK: use in a lambda defined in a __device__ function
    auto lam1 = [=, *this] __device__ { return xxx; };

    // OK: use in a lambda defined in a __device__ function
    auto lam2 = [=, *this] __host__ __device__ { return xxx; };

    // OK: use in a lambda defined in a __device__ function
    auto lam3 = [=, *this]  { return xxx; };
  }

   __host__ __device__ void host_device_func(void) {

    // OK: use in an extended __device__ lambda
    auto lam1 = [=, *this] __device__ { return xxx; };

    // Use in an extended __host__ __device__ lambda
    // Error if *this capture not enabled by language dialect
    auto lam2 = [=, *this] __host__ __device__ { return xxx; };

    // Use in an unannotated lambda in a __host__ __device__ function
    // Error if *this capture not enabled by language dialect
    auto lam3 = [=, *this]  { return xxx; };
  }
};

### 18.7.5.附加备注[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#additional-notes "这个标题的永久链接")

1. `ADL Lookup`：如前所述，在调用主机编译器之前，CUDA编译器将用占位符类型的实例替换扩展的lambda表达式。占位符类型的一个模板参数使用包含原始lambda表达式的函数地址。这可能会导致其他命名空间参与参数依赖查找（ADL），对于任何参数类型涉及扩展lambda表达式的闭包类型的主机函数调用。这可能会导致主机编译器选择不正确的函数。
    
    示例：
    ```c++
    namespace N1 {
      struct S1_t { };
      template <typename T>  void foo(T);
    };
    
    namespace N2 {
      template <typename T> int foo(T);
    
      template <typename T>  void doit(T in) {     foo(in);  }
    }
    
    void bar(N1::S1_t in) {
      /* extended __device__ lambda. In the code sent to the host compiler, this
         is replaced with the placeholder type instantiation expression
         ' __nv_dl_wrapper_t< __nv_dl_tag<void (*)(N1::S1_t in),(&bar),1> > { }'
    
         As a result, the namespace 'N1' participates in ADL lookup of the
         call to "foo" in the body of N2::doit, causing ambiguity.
      */
      auto lam1 = [=] __device__ { };
      N2::doit(lam1);
    }```
    
    在上述示例中，CUDA编译器用涉及`N1`命名空间的占位符类型取代了扩展的lambda。因此，命名空间`N1`参与了`N2::doit`正文中`foo(in)`的ADL查找，主机编译失败，因为发现了多个过载候选者`N1::foo`和`N2::foo`。
    

## 18.8.放松的Constexpr（-expt-relaxed-constexpr）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#relaxed-constexpr-expt-relaxed-constexpr "这个标题的永久链接")

默认情况下，不支持以下交叉执行空间调用：

1. 在主机代码生成阶段（即`__CUDA_ARCH__`宏未定义时）从`__host__`函数调用`__device__`only `constexpr`函数。示例：
    
    > constexpr __device__ int D() { return 0; }
    > int main() {
    >     int x = D();  //ERROR: calling a __device__-only constexpr function from host code
    > }
    
2. 在设备代码生成阶段（即定义`__CUDA_ARCH__`宏时），从`__device__`或`__global__`函数调用`__host__`-only `constexpr`函数。示例：
    
    > constexpr  int H() { return 0; }
    > __device__ void dmain()
    > {
    >     int x = H();  //ERROR: calling a __host__-only constexpr function from device code
    > }
    

实验标志`-expt-relaxed-constexpr`可用于放松这个约束。当指定此标志时，编译器将支持上述交叉执行空间调用，如下所示：

1. 如果constexpr函数的跨执行空间调用发生在需要持续评估的上下文中，例如在constexpr变量的初始化器中，则支持对constexpr函数的交叉执行空间调用。示例：
    
    > constexpr __host__ int H(int x) { return x+1; };
    > __global__ void doit() {
    > constexpr int val = H(1); // OK: call is in a context that
    >                           // requires constant evaluation.
    > }
    > 
    > constexpr __device__ int D(int x) { return x+1; }
    > int main() {
    > constexpr int val = D(1); // OK: call is in a context that
    >                           // requires constant evaluation.
    > }
    
2. 否则：
    
    > 1. 在设备代码生成过程中，为`__host__`-only constexpr函数`H`的主体生成设备代码，除非`H`不使用或仅在constexpr上下文中调用。示例：
    >     
    >     > // NOTE: "H" is emitted in generated device code because it is
    >     > // called from device code in a non-constexpr context
    >     > constexpr __host__ int H(int x) { return x+1; }
    >     > 
    >     > __device__ int doit(int in) {
    >     >   in = H(in);  // OK, even though argument is not a constant expression
    >     >   return in;
    >     > }
    >     
    > 2. **适用于“__device__”函数的所有代码限制也适用于从设备代码调用的“constexpr host”-only函数“H”。然而，对于这些限制**[8，](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#frelaxedconstexpr1)**编译器可能不会为“H”发出任何构建时间诊断**。
    >     
    >     > 例如，以下代码模式在`H`的主体中不受支持（与任何`__device__`函数一样），但不会生成编译器诊断：
    >     > 
    >     > > - ODR-使用主机变量或`__host__`-only非constexpr函数。示例：
    >     > >     
    >     > >     > int qqq, www;
    >     > >     > 
    >     > >     > constexpr __host__ int* H(bool b) { return b ? &qqq : &www; };
    >     > >     > 
    >     > >     > __device__ int doit(bool flag) {
    >     > >     >   int *ptr;
    >     > >     >   ptr = H(flag); // ERROR: H() attempts to refer to host variables 'qqq' and 'www'.
    >     > >     >                  // code will compile, but will NOT execute correctly.
    >     > >     >   return *ptr;
    >     > >     > }
    >     > >     
    >     > > - 使用异常（`throw/catch`）和RTTI（typeid`typeid,dynamic_cast`）。示例：
    >     > >     
    >     > >     > struct Base { };
    >     > >     > struct Derived : public Base { };
    >     > >     > 
    >     > >     > // NOTE: "H" is emitted in generated device code
    >     > >     > constexpr int H(bool b, Base *ptr) {
    >     > >     >   if (b) {
    >     > >     >     return 1;
    >     > >     >   } else if (typeid(ptr) == typeid(Derived)) { // ERROR: use of typeid in code executing on the GPU
    >     > >     >     return 2;
    >     > >     >   } else {
    >     > >     >     throw int{4}; // ERROR: use of throw in code executing on the GPU
    >     > >     >   }
    >     > >     > }
    >     > >     > __device__ void doit(bool flag) {
    >     > >     >   int val;
    >     > >     >   Derived d;
    >     > >     >   val = H(flag, &d); //ERROR: H() attempts use typeid and throw(), which are not allowed in code that executes on the GPU
    >     > >     > }
    >     > >     
    >     
    > 3. 在主机代码生成过程中，`__device__`-only constexpr函数`D`的主体保留在发送到主机编译器的代码中。如果`D`的主体尝试使用命名空间范围设备变量或`__device__`-only非constexpr函数，则不支持从主机代码调用toD（代码可能在没有编译器诊断的情况下构建，但在运行时可能表现不正确）。示例：
    >     
    >     > __device__ int qqq, www;
    >     > constexpr __device__ int* D(bool b) { return b ? &qqq : &www; };
    >     > 
    >     > int doit(bool flag) {
    >     >   int *ptr;
    >     >   ptr = D(flag); // ERROR: D() attempts to refer to device variables 'qqq' and 'www'
    >     >                  // code will compile, but will NOT execute correctly.
    >     >   return *ptr;
    >     > }
    >     
    > 4. **注意：鉴于上述限制和缺乏使用不当的编译器诊断，从设备代码中在标准C++标头中调用constexpr __host__函数时要小心**，因为该函数的实现将因主机平台而异，例如，基于gcc主机编译器的`libstdc++`版本。此类代码在移植到其他平台或主机编译器版本时可能会无声中断（如前所述，如果目标C++库实现odr-使用主机代码变量或函数）。
    >     
    >     > 示例：
    >     > 
    >     > __device__ int get(int in) {
    >     >  int val = std::foo(in); // "std::foo" is constexpr function defined in the host compiler's standard library header
    >     >                          // WARNING: if std::foo implementation ODR-uses host variables or functions,
    >     >                          // code will not work correctly
    >     > }
    >     
    

[8](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id380)

诊断通常在解析期间生成，但在稍后在翻译单元中遇到从设备代码调用`H`之前，可能已经解析了仅主机函数`H`。

## 18.9.代码样本[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#code-samples "这个标题的永久链接")

### 18.9.1.数据聚合类[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-aggregation-class "这个标题的永久链接")

class PixelRGBA {
public:
    __device__ PixelRGBA(): r_(0), g_(0), b_(0), a_(0) { }

    __device__ PixelRGBA(unsigned char r, unsigned char g,
                         unsigned char b, unsigned char a = 255):
                         r_(r), g_(g), b_(b), a_(a) { }

private:
    unsigned char r_, g_, b_, a_;

    friend PixelRGBA operator+(const PixelRGBA&, const PixelRGBA&);
};

__device__
PixelRGBA operator+(const PixelRGBA& p1, const PixelRGBA& p2)
{
    return PixelRGBA(p1.r_ + p2.r_, p1.g_ + p2.g_,
                     p1.b_ + p2.b_, p1.a_ + p2.a_);
}

__device__ void func(void)
{
    PixelRGBA p1, p2;
    // ...      // Initialization of p1 and p2 here
    PixelRGBA p3 = p1 + p2;
}

### 18.9.2.衍生类[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#derived-class "这个标题的永久链接")

__device__ void* operator new(size_t bytes, MemoryPool& p);
__device__ void operator delete(void*, MemoryPool& p);
class Shape {
public:
    __device__ Shape(void) { }
    __device__ void putThis(PrintBuffer *p) const;
    __device__ virtual void Draw(PrintBuffer *p) const {
         p->put("Shapeless");
    }
    __device__ virtual ~Shape() {}
};
class Point : public Shape {
public:
    __device__ Point() : x(0), y(0) {}
    __device__ Point(int ix, int iy) : x(ix), y(iy) { }
    __device__ void PutCoord(PrintBuffer *p) const;
    __device__ void Draw(PrintBuffer *p) const;
    __device__ ~Point() {}
private:
    int x, y;
};
__device__ Shape* GetPointObj(MemoryPool& pool)
{
    Shape* shape = new(pool) Point(rand(-20,10), rand(-100,-20));
    return shape;
}

### 18.9.3.班级模板[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#class-template "这个标题的永久链接")
```c++
template <class T>
class myValues {
    T values[MAX_VALUES];
public:
    __device__ myValues(T clear) { ... }
    __device__ void setValue(int Idx, T value) { ... }
    __device__ void putToMemory(T* valueLocation) { ... }
};

template <class T>
void __global__ useValues(T* memoryBuffer) {
    myValues<T> myLocation(0);
    ...
}

__device__ void* buffer;

int main()
{
    ...
    useValues<int><<<blocks, threads>>>(buffer);
    ...
}
```
### 18.9.4.功能模板[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-template "这个标题的永久链接")
```c++
template <typename T>
__device__ bool func(T x)
{
   ...
   return (...);
}

template <>
__device__ bool func<int>(T x) // Specialization
{
   return true;
}

// Explicit argument specification
bool result = func<double>(0.5);

// Implicit argument deduction
int x = 1;
bool result = func(x);
```
### 18.9.5.函子类[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#functor-class "这个标题的永久链接")
```c++
class Add {
public:
    __device__  float operator() (float a, float b) const
    {
        return a + b;
    }
};

class Sub {
public:
    __device__  float operator() (float a, float b) const
    {
        return a - b;
    }
};

// Device code
template<class O> __global__
void VectorOperation(const float * A, const float * B, float * C,
                     unsigned int N, O op)
{
    unsigned int iElement = blockDim.x * blockIdx.x + threadIdx.x;
    if (iElement < N)
        C[iElement] = op(A[iElement], B[iElement]);
}

// Host code
int main()
{
    ...
    VectorOperation<<<blocks, threads>>>(v1, v2, v3, N, Add());
    ...
}```

9例如，`<<<...>>>`启动内核的语法。

10 这不适用于可能在多个翻译单元中定义的实体，例如编译器生成的模板实例化。

[11](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id339) The intent is to allow variable memory space specifiers for static variables in a `__host__ __device__` function during device compilation, but disallow it during host compilation

[12](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id349) One way to debug suspected layout mismatch of a type `C` is to use `printf` to output the values of `sizeof(C)` and `offsetof(C, field)` in host and device code.

[13](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id355) 请注意，由于存在额外的声明，这可能会对编译时间产生负面影响。

[14](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id356) 目前，`-std=c++11`标志仅支持以下主机编译器：gcc版本>= 4.7、clang、icc>= 15和xlc >= 13.1

[15](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id358) 包括`operator()`

[16](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id360) 限制与非constexpr被调用函数相同。

[17](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id361)

请注意，实验标志的行为可能会在未来的编译器版本中发生变化。

[18](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id363)

C++标准部分`[basic.types]`

[19](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id364)

C++标准部分`[expr.const]`

[20](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id368)

目前，`-std=c++14`标志仅支持以下主机编译器：gcc版本>= 5.1，clang版本>= 3.7和icc版本>= 17

[21](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id370)

目前，`-std=c++17`标志仅支持以下主机编译器：gcc版本>= 7.0，clang版本>= 8.0，Visual Studio版本>= 2017，pgi编译器版本>= 19.0，icc编译器版本>= 19.0

[22](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id373)

目前，`-std=c++20`标志仅支持以下主机编译器：gcc版本>= 10.0，clang版本>= 10.0，Visual Studio版本>= 2022和nvc++版本>= 20.7。

[23](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id375)

使用icc主机编译器时，此标志仅支持icc >= 1800。

24（1,2）

如果扩展的lambda模式未激活，特征将始终返回false。

[25](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id378)

相比之下，C++标准指定捕获的变量用于直接初始化闭包类型的字段。

# 19.纹理获取[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-fetching "这个标题的永久链接")

本节给出了用于计算[纹理函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-functions)的纹理函数根据纹理对象的各种属性返回的值的公式（请参阅[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)）。

绑定到纹理对象的纹理表示为数组T

- _N_ texels用于一维纹理，
    
- _N x M_ texels用于二维纹理，
    
- _N x M x L_ texels用于三维纹理。
    

它使用非规范化纹理坐标_x、y_和_z_，或[纹理内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory)中描述的规范化纹理坐标_x/N_、_y/M_和_z/L_获取。在本节中，假定坐标在有效范围内。[纹理记忆](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory)解释了如何根据寻址模式将范围外的坐标重新映射到有效范围。

## 19.1.最近点采样[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nearest-point-sampling "这个标题的永久链接")

在此过滤模式下，纹理获取返回的值是

- _tex(x)=T[i]_用于一维纹理，
    
- _tex(x,y)=T[i,j]_用于二维纹理，
    
- 三维纹理的_tex(x,y,z)=T[i,j,k]_，
    

其中_i=floor(x)，__j=floor(y)_，_k=floor(z)。_

[图36](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nearest-point-sampling-nearest-point-sampling-fig)说明了_N=4_的一维纹理的最近点采样。

![_图像/4-texels的1-d-纹理的最近点采样.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/nearest-point-sampling-of-1-d-texture-of-4-texels.png)

图36最近点采样过滤模式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nearest-point-sampling-nearest-point-sampling-fig "此图像的永久链接")

对于整数纹理，纹理获取返回的值可以选择重新映射为[0.0，1.0]（见[纹理内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory)）。

## 19.2.线性过滤[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#linear-filtering "这个标题的永久链接")

在此过滤模式下，仅适用于浮点纹理，纹理获取返回的值是

- tex(x)=(1−α)T[i]+αT[i+1]对于一维纹理，
    
- tex(x)=(1−α)T[i]+αT[i+1]对于一维纹理，
    
- tex(x,y)=(1−α)(1−β)T[i,j]+α(1−β)T[i+1,j]+(1−α)βT[i,j+1]+αβT[i+1,j+1]对于二维纹理，
    
- tex(x,y,z)=
    
    (1−α)(1−β)(1−γ)T[i,j,k]+α(1−β)(1−γ)T[i+1,j,k]+
    
    (1−α)β(1−γ)T[i,j+1,k]+αβ(1−γ)T[i+1,j+1,k]+
    
    (1−α)(1−β)γT[i,j,k+1]+α(1−β)γT[i+1,j,k+1]+
    
    (1−α)βγT[i,j+1,k+1]+αβγT[i+1,j+1,k+1]
    
    对于三维纹理，
    

地点：

- i=floor(x B)∗,α=frac(x B)∗,∗x B =x−0.5,
    
- j=floor(y B)∗,β=frac(y B)∗,∗y B =y−0.5,
    
- k=floor(z B)∗,γ=frac(z B)∗,∗z B =z−0.5,
    

α，β，以及γ以9位固定点格式存储，具有8位分数值（因此1.0被精确表示）。

[图37](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#linear-filtering-of-1-d-texture-of-4-texels)说明了_N=4_的一维纹理的线性过滤。

![_图像/线性过滤-1-d-纹理-of-4-texels.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/linear-filtering-of-1-d-texture-of-4-texels.png)

图37线性过滤模式[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#linear-filtering-of-1-d-texture-of-4-texels "此图像的永久链接")

## 19.3.表格查找[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-lookup "这个标题的永久链接")

表查找_TL(x)_，其中_x_跨越区间_[0,R]_可以实现为_TL(x)=tex((N-1)/R)x+0.5)_，以确保_TL(0)=T[0]_和_TL(R)=T[N-1]_。

[图38](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-lookup-1-d-table-lookup-using-linear-filtering)说明了使用纹理过滤从_N=4_的一维纹理中实现_R=4_或_R=1_的表格查找。

![_图像/1-d-表-查找-使用-线性-过滤.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/1-d-table-lookup-using-linear-filtering.png)

图38使用线性过滤的一维表格查找[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-lookup-1-d-table-lookup-using-linear-filtering "此图像的永久链接")

# 20.计算能力[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities "这个标题的永久链接")

计算设备的一般规格和功能取决于其计算能力（请参阅[计算能力](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability)）。

[表26](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-feature-support-per-compute-capability)和[表27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability)显示了与目前支持的每个计算能力相关的功能和技术规格。

[浮点标准](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#floating-point-standard)部分审查了IEEE浮点标准的合规性。

[计算能力5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x)、[计算能力6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)、[计算能力7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)、[计算能力8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x)、[计算能力9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0)、[计算能力10.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-10-0)和[计算能力12.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-12-0)部分提供了有关具有这些各自计算能力的设备架构的更多详细信息。

## 20.1.功能可用性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#feature-availability "这个标题的永久链接")

计算架构中引入的大多数计算功能都打算在所有后续架构上提供。[表26](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-feature-support-per-compute-capability)中显示了计算功能引入后功能可用性的“是”。

### 20.1.1.特定于架构的功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-specific-features "这个标题的永久链接")

从计算能力9.0的设备开始，在架构中引入的专业计算功能可能无法保证在所有后续计算功能上都可用。这些功能被称为_特定于架构_的功能，并针对专业操作的加速，例如Tensor Core操作，这些操作并不适用于所有类别的计算能力，或者可能会在下一代中发生重大变化。必须使用特定于架构的编译器目标（请参阅[功能集编译器目标](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#feature-set-compiler-targets)）编译代码，才能启用特定于架构的功能。使用特定于架构的编译器目标编译的代码只能在编译的确切计算能力上运行。

### 20.1.2.特定于家族的特征[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-features "这个标题的永久链接")

从计算能力10.0的设备开始，一些特定于架构的功能与具有多个计算能力的设备是通用的。包含这些功能的设备是同一家族的一部分，这些功能也可以称为_家族特定_功能。保证同一家庭中的所有设备上都提供特定于家庭的功能。需要特定于家族的编译器目标才能启用特定于家族的功能。参见[第20.1.3节](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#feature-set-compiler-targets)。为特定于家族的目标编译的代码只能在该家族成员的GPU上运行。

### 20.1.3.功能集编译器目标[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#feature-set-compiler-targets "这个标题的永久链接")

编译器可以针对三组计算功能：

**基线功能集**：引入的主要计算功能集，旨在用于后续计算架构。[表26](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-feature-support-per-compute-capability)总结了这些功能及其可用性。

**特定于架构的功能集**：一组小型且高度专业化的功能，称为特定于架构，引入这些功能是为了加速专业操作，但不能保证在后续计算架构上可用或可能发生重大变化。这些功能在各自的“计算能力#.”子节中进行了总结。特定于架构的功能集是特定于家族的功能集的超集。特定于架构的编译器目标与计算能力9.0设备一起引入，并通过在编译目标中使用**a**后缀来选择，例如指定`compute_100a`或`compute_120a`作为计算目标。

**特定于家族的功能集**：一些特定于架构的功能是具有多个计算能力的GPU的通用。这些功能在各自的“计算能力#.”子节中进行了总结。除了少数例外，具有相同主要计算能力的后代设备都在同一家族中。[表25](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-compatibility)显示了特定于家族的目标与设备计算能力的兼容性，包括例外情况。特定于家族的特征集是基线特征集的超集。特定于家族的编译器目标是与计算能力10.0设备一起引入的，并通过在编译目标中使用**f**后缀来选择，例如指定`compute_100f`或`compute_120f`作为计算目标。

从计算能力9.0开始的所有设备都具有一组特定于架构的功能。要在特定GPU上使用这些功能的完整集，必须使用带有后缀a的架构特定编译器目标。此外，从计算能力10.0开始，有一组功能出现在具有不同次要计算能力的多个设备上。这些指令集称为家族特定功能，共享这些功能的设备被称为同一家族的一部分。特定于家族的功能是该GPU家族的所有成员共享的特定于架构的特征的子集。带有后缀**f**的家族特定编译器目标允许编译器生成代码，该代码使用该架构特定功能的通用子集。

例如：

- `compute_100`编译目标不允许使用特定于架构的功能。此目标将与所有具有计算能力10.0及更高版本的设备兼容。
    
- `compute_100f`_家族特定的_编译目标允许使用整个GPU家族通用的架构特定特征的子集。此目标仅与属于GPU系列的设备兼容。在本例中，它与计算能力10.0和计算能力10.3的设备兼容。特定于家族的`compute_100f`目标中可用的功能是基线`compute_100`目标中可用的功能的超集。
    
- `compute_100a`_特定于架构的_编译目标允许在Compute Capability 10.0设备中使用一整套特定于架构的功能。此目标仅与计算能力10.0的设备兼容，不兼容其他设备。`compute_100a`目标中可用的功能构成了`compute_100f`目标中可用的功能的超集。
    

表25家族特定兼容性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-compatibility "此表的永久链接")
|汇编目标|与计算功能兼容|   |
|---|---|---|
|`compute_100f`|10.0|10.3|
|`compute_103f`|10.3 [26](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family2)|   |
|`compute_110f`|11.0 [26](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family2)|   |
|`compute_120f`|12.0|12.1|
|`compute_121f`|12.1 [26](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family2)|   |

26（1,2,3）

一些家族在创建时只包含一个成员。它们将来可能会扩展，以包括更多的设备。

## 20.2.特点和技术规格[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications "这个标题的永久链接")

表26每个计算功能的功能支持[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-feature-support-per-compute-capability "此表的永久链接")
|**功能支持**|**计算能力**|   |   |   |   |   |
|---|---|---|---|---|---|---|
|（所有计算功能都支持未列出的功能）|7.x|8.x|9.0|10.0|11.0|12.0|
|在全局内存中128位整数值上运行的原子函数（[原子函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)）|不|   |是|   |   |   |
|在共享内存中128位整数值上操作的原子函数（[原子函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)）|不|   |是|   |   |   |
|在全局内存中对float2和float4浮点向量操作的原子加法（[atomicAdd（））](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd)|不|   |是|   |   |   |
|Bfloat16精度浮点运算：加法、减法、乘法、比较、扭曲洗牌函数、转换|不|   |是|   |   |   |
|硬件加速的`memcpy_async`（[使用cuda::pipeline的非同步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memcpy-async-pipeline)）|不|   |是|   |   |   |
|硬件加速的拆分到达/等待屏障（[异步屏障](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier)）|不|是|   |   |   |   |
|L2缓存驻留管理（[设备内存L2访问管理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-access-intro)）|不|是|   |   |   |   |
|加速动态编程的DPX说明|不|   |是|   |   |   |
|分布式共享内存|不|   |是|   |   |   |
|线程块集群|不|   |是|   |   |   |
|张量内存加速器（TMA）单元|不|   |是|   |   |   |

请注意，下表中使用的KB和K单位分别对应于1024字节（即KiB）和1024字节。

表27每个计算能力的技术规格[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability "此表的永久链接")
||**计算能力**|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|---|---|
|技术规格|7.5|8.0|8.6|8.7|8.9|9.0|10.0|11.0|12.0|
|每个设备的最大驻留网格数（并发内核执行）|128|   |   |   |   |   |   |   |   |
|螺纹块网格的最大维度|3|   |   |   |   |   |   |   |   |
|螺纹块网格的最大x-维|231-1|   |   |   |   |   |   |   |   |
|线程块网格的最大y或z维|65535|   |   |   |   |   |   |   |   |
|螺纹块的最大尺寸|3|   |   |   |   |   |   |   |   |
|块的最大x或y维|1024|   |   |   |   |   |   |   |   |
|块的最大z维|64|   |   |   |   |   |   |   |   |
|每个块的最大线程数|1024|   |   |   |   |   |   |   |   |
|扭曲尺寸|32|   |   |   |   |   |   |   |   |
|每个SM的最大居民区块数|16|32|16|   |24|32|   |24|   |
|每个SM的最大常驻翘曲数|32|64|48|   |   |64|   |48|   |
|每个SM的最大常驻线程数|1024|2048|1536|   |   |2048|   |1536|   |
|每个SM的32位寄存器数量|64K|   |   |   |   |   |   |   |   |
|每个线程块的最大32位寄存器数量|64K|   |   |   |   |   |   |   |   |
|每个线程的32位寄存器的最大数量|255|   |   |   |   |   |   |   |   |
|每个SM的最大共享内存量|64KB|164 KB|100KB|164 KB|100KB|228千字节|   |   |100KB|
|每个线程块的最大共享内存量[27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn33)|64KB|163KB|99千字节|163KB|99千字节|227 千字节|   |   |99千字节|
|共享内存库的数量|32|   |   |   |   |   |   |   |   |
|每个线程的最大本地内存量|512千分|   |   |   |   |   |   |   |   |
|恒定内存大小|64KB|   |   |   |   |   |   |   |   |
|缓存工作设置每个SM的恒定内存|8千分|   |   |   |   |   |   |   |   |
|纹理内存的每个SM的缓存工作集|32或64 KB|28 KB ~ 192 KB|28 KB ~ 128 KB|28 KB ~ 192 KB|28 KB ~ 128 KB|28 KB ~ 256 KB|   |   |28 KB ~ 128 KB|
|使用CUDA数组的1D纹理对象的最大宽度|131072|   |   |   |   |   |   |   |   |
|使用线性内存的一维纹理对象的最大宽度|228|   |   |   |   |   |   |   |   |
|1D分层纹理对象的最大宽度和层数|32768 x 2048|   |   |   |   |   |   |   |   |
|使用CUDA数组的2D纹理对象的最大宽度和高度|131072 x 65536|   |   |   |   |   |   |   |   |
|使用线性内存的二维纹理对象的最大宽度和高度|131072 x 65000|   |   |   |   |   |   |   |   |
|使用支持纹理收集的CUDA阵列的2D纹理对象的最大宽度和高度|32768 x 32768|   |   |   |   |   |   |   |   |
|2D分层纹理对象的最大宽度、高度和层数|32768 x 32768 x 2048|   |   |   |   |   |   |   |   |
|使用CUDA阵列的3D纹理对象的最大宽度、高度和深度|16384 x 16384 x 16384|   |   |   |   |   |   |   |   |
|立方体地图纹理对象的最大宽度（和高度）|32768|   |   |   |   |   |   |   |   |
|立方体图分层纹理对象的最大宽度（和高度）和层数|32768 x 2046|   |   |   |   |   |   |   |   |
|可以绑定到内核的最大纹理数|256|   |   |   |   |   |   |   |   |
|使用CUDA数组的1D表面对象的最大宽度|32768|   |   |   |   |   |   |   |   |
|一维分层表面对象的最大宽度和层数|32768 x 2048|   |   |   |   |   |   |   |   |
|使用CUDA阵列的二维表面对象的最大宽度和高度|131072 x 65536|   |   |   |   |   |   |   |   |
|2D分层表面对象的最大宽度、高度和层数|32768 x 32768 x 1048|   |   |   |   |   |   |   |   |
|使用CUDA阵列的3D表面对象的最大宽度、高度和深度|16384 x 16384 x 16384|   |   |   |   |   |   |   |   |
|使用CUDA数组的立方体地图表面对象的最大宽度（和高度）|32768|   |   |   |   |   |   |   |   |
|立方体图分层表面对象的最大宽度（和高度）和层数|32768 x 2046|   |   |   |   |   |   |   |   |
|可以使用内核的最大表面数量|32|   |   |   |   |   |   |   |   |

## 20.3.浮点标准[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#floating-point-standard "这个标题的永久链接")

所有计算设备都遵循IEEE 754-2008二进制浮点算术标准，偏差如下：

- 没有可动态配置的四舍五入模式；但是，大多数操作都支持多种IEEE四舍五入模式，通过设备内在暴露。
    
- 没有机制来检测是否发生了浮点异常，所有操作的行为都像IEEE-754异常总是被屏蔽一样，如果存在异常事件，则按照IEEE-754定义的屏蔽响应。出于同样的原因，虽然支持SNaN编码，但它们不是信号，而是作为静音处理。
    
- 涉及一个或多个输入NaN的单精度浮点操作的结果是位模式0x7ffffffff的安静NaN。
    
- 关于NaNs的双精度浮点绝对值和否定与IEEE-754不符合；这些传递不变。
    

代码必须使用`-ftz=false`、`-prec-div=true`和`-prec-sqrt=true`编译，以确保IEEE合规性（这是默认设置；请参阅thenvcc用户手册以了解这些编译标志的描述）。

无论编译器标志`-ftz`的设置如何，

- 全局内存上的原子单精度浮点加法总是在齐平到零模式下运行，即行为等同于`FADD.F32.FTZ.RN`，
    
- 共享内存上的原子单精度浮点加法总是在非正态支持下运行，即行为等同于`FADD.F32.RN`。
    

根据IEEE-754R标准，如果`fminf()``fmin()``fmaxf()`或`fmax()`的输入参数之一是NaN，而不是另一个，则结果是非NaN参数。

IEEE-754未定义浮点值在浮点数值超出整数格式范围的情况下，将浮点数值转换为整数值。对于计算设备，行为是将夹到支持范围的末端。这与x86架构行为不同。

IEEE-754未定义整数除以零和整数溢出的行为。对于计算设备，没有检测此类整数运算异常的机制。整数除以零会产生一个未指定的、特定于机器的值。

[https://developer.nvidia.com/content/precision-performance-floating-point-and-ieee-754-compliance-nvidia-gpus](https://developer.nvidia.com/content/precision-performance-floating-point-and-ieee-754-compliance-nvidia-gpus)包含有关NVIDIA GPU浮点精度和合规性的更多信息。

## 20.4.计算能力5.x[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x "这个标题的永久链接")

### 20.4.1.建筑[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture "这个标题的永久链接")

SM由以下部分组成：

- 用于算术运算的128个CUDA核心（有关算术运算的吞吐量，请参阅[CUDA C++最佳实践指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions)），
    
- 32个用于单精度浮点超越函数的特殊函数单元，
    
- 4个扭曲调度器。
    

当SM被赋予要执行的经编时，它首先将它们分配到四个调度器之间。然后，在每次指令发布时间，每个调度器都会为其分配的其中一个准备执行的扭曲发布一个指令（如果有的话）。

SM有：

- 一个只读的常量缓存，由所有功能单元共享，并加快来自常量内存空间的读取速度，该空间位于设备内存中，
    
- 24 KB的统一L1/纹理缓存，用于缓存全局内存中的读取，
    
- 计算能力5.0设备的64 KB共享内存或计算能力5.2的设备96KB的共享内存。
    

统一的L1/纹理缓存也被纹理单元使用，该单元实现了[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)中提到的各种寻址模式和数据过滤。

还有一个由所有SM共享的L2缓存，用于缓存对本地或全局内存的访问，包括临时寄存器泄漏。应用程序可以通过检查`l2CacheSize`设备属性来查询L2缓存大小（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)）。

缓存行为（例如，读取是否同时缓存在统一的L1/纹理缓存和L2中，还是仅在L2中缓存）可以使用加载指令的修饰符在每次访问的基础上进行部分配置。

### 20.4.2.全球记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x "这个标题的永久链接")

全局内存访问总是缓存在L2中。

内核整个生命周期内都是只读的数据也可以通过使用`__ldg()`函数读取（请参阅[只读数据缓存加载函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ldg-function)），在上一节中描述的统一L1/texture缓存中缓存。当编译器检测到某些数据满足只读条件时，它将使用`__ldg()`来读取它。编译器可能并不总是能够检测到某些数据的只读条件是否满足。用`const`和`__restrict__`限定符标记用于加载此类数据的指针，增加了编译器检测只读条件的可能性。

内核整个生命周期内非只读的数据不能缓存在计算能力5.0设备的统一L1/texture缓存中。对于具有计算能力5.2的设备，默认情况下，它不会缓存在统一的L1/纹理缓存中，但可以使用以下机制启用缓存：

- 如PTX参考手册中所述，使用适当的修饰符使用内联装配进行读取；
    
- Compile with the `-Xptxas -dlcm=ca` compilation flag, in which case all reads are cached, except reads that are performed using inline assembly with a modifier that disables caching;
    
- Compile with the `-Xptxas -fscm=ca` compilation flag, in which case all reads are cached, including reads that are performed using inline assembly regardless of the modifier used.
    

当使用上述三种机制之一启用缓存时，具有计算能力5.2的设备将在统一的L1/texture缓存中缓存全局内存读取，用于所有内核启动，但线程块消耗太多SM寄存器文件的内核启动除外。这些例外情况由分析器报告。

### 20.4.3.共享内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x "这个标题的永久链接")

共享内存有32个库，这些库被组织成连续的32位单词映射到连续的库。每个银行的带宽为每个时钟周期32位。

对扭曲的共享内存请求不会在访问同一32位字内任何地址的两个线程之间产生银行冲突（即使两个地址属于同一银行）。在这种情况下，对于读取访问，该单词被广播到请求线程，对于写入访问，每个地址仅由其中一个线程写入（执行写入的线程未定义）。

[图39](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x-examples-of-strided-shared-memory-accesses)显示了一些步步访问的例子。

[图40](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x-examples-of-irregular-shared-memory-accesses)显示了一些涉及广播机制的内存读取访问示例。

![在32位银行大小模式下的共享内存访问。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/examples-of-strided-shared-memory-accesses.png)

图39在32位银行大小模式下的Strided共享内存访问。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x-examples-of-strided-shared-memory-accesses "此图像的永久链接")

左边

具有一步跨32位字的線性定址（無銀行衝突）。

中间的

线性寻址，带两个32位字的步长（双向银行冲突）。

右

线性寻址，三个32位字的步数（无银行冲突）。

![不规则的共享内存访问。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/examples-of-irregular-shared-memory-accesses.png)

图40 不规则的共享内存访问。[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x-examples-of-irregular-shared-memory-accesses "此图像的永久链接")

左边

通过随机排列进行无冲突访问。

中间的

无冲突访问，因为线程3、4、6、7和9在银行5中访问了相同的单词。

右

无冲突广播访问（线程访问银行内的相同单词）。

## 20.5.计算能力6.x[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x "这个标题的永久链接")

### 20.5.1.建筑[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-6-x "这个标题的永久链接")

SM由以下部分组成：

- 64（计算能力6.0）或128（6.1和6.2）CUDA核心用于算术运算，
    
- 用于单精度浮点超验函数的16（6.0）或32（6.1和6.2）特殊函数单元，
    
- 2（6.0）或4（6.1和6.2）翘曲调度器。
    

当SM被赋予要执行的扭曲时，它首先在调度器之间分发它们。然后，在每次指令发布时间，每个调度器都会为其分配的其中一个准备执行的扭曲发布一个指令（如果有的话）。

SM有：

- 一个只读的常量缓存，由所有功能单元共享，并加快来自常量内存空间的读取速度，该空间位于设备内存中，
    
- 统一的L1/纹理缓存，用于从大小为24 KB（6.0和6.2）或48 KB（6.1）的全局内存中读取，
    
- 大小为64 KB（6.0和6.2）或96 KB（6.1）的共享内存。
    

统一的L1/纹理缓存也被纹理单元使用，该单元实现了[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)中提到的各种寻址模式和数据过滤。

还有一个由所有SM共享的L2缓存，用于缓存对本地或全局内存的访问，包括临时寄存器泄漏。应用程序可以通过检查`l2CacheSize`设备属性来查询L2缓存大小（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)）。

缓存行为（例如，读取是否同时缓存在统一的L1/纹理缓存和L2中，还是仅在L2中）可以使用加载指令的修饰符在每次访问的基础上进行部分配置。

### 20.5.2.全球记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-6-x "这个标题的永久链接")

全局内存的行为方式与计算能力5.x的设备相同（见[全局内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x)）。

### 20.5.3.共享内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-6-x "这个标题的永久链接")

共享内存的行为方式与计算能力5.x的设备相同（请参阅[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x)）。

## 20.6.计算能力7.x[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x "这个标题的永久链接")

### 20.6.1.建筑[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-7-x "这个标题的永久链接")

SM由以下部分组成：

- 用于单精度算术运算的64个FP32内核，
    
- 32个FP64核心，用于双精度算术运算，[28](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn35)
    
- 用于整数数学的64个INT32核心，
    
- 用于深度学习矩阵算术的8个混合精度张量核心
    
- 用于单精度浮点超越函数的16个特殊函数单元，
    
- 4个扭曲调度器。
    

SM在其调度器之间静态地分配其翘曲。然后，在每次指令发布时间，每个调度器都会为其分配的其中一个准备执行的扭曲发布一个指令（如果有的话）。

SM有：

- 一个只读的常量缓存，由所有功能单元共享，并加快来自常量内存空间的读取速度，该空间位于设备内存中，
    
- 统一的数据缓存和共享内存，总大小为128 KB（Volta）或96 KB（_Turing_）。
    

共享内存由统一数据缓存分区，可以配置为各种尺寸（请参阅[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x)。）剩余的数据缓存用作L1缓存，也用于纹理单元，该单元实现了[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)中提到的各种寻址和数据过滤模式。

### 20.6.2.独立线程调度[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#independent-thread-scheduling "这个标题的永久链接")

**NVIDIA Volta GPU架构**在扭曲的线程之间引入了_独立线程调度_，使以前不可用的扭曲内同步模式成为可能，并简化了移植CPU代码时的代码更改。然而，如果开发人员对以前的硬件架构的经编同步性做出假设，这可能会导致一组参与执行代码的线程与预期的相当不同。

以下是Volta-safe代码的担忧代码模式和建议的纠正措施。

1. 对于使用warp intrinsics（`__shfl*`、`__any`、`__all`、`__ballot`）的应用程序，开发人员有必要将代码移植到带有`*_sync`后缀的新、安全的同步对应物。新的扭曲内在特征采取了线程掩码，明确定义了哪些车道（扭曲的线程）必须参与扭曲内在性。有关详细信息，请参阅[Warp Vote函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-vote-functions)和[Warp Shuffle函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)。
    

由于CUDA 9.0+提供内在，（如有必要）代码可以使用以下预处理器宏有条件地执行：

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
// *_sync intrinsic
#endif

这些内在功能适用于所有架构，而不仅仅是**NVIDIA Volta GPU架构**或**NVIDIA Turing GPU架构**，在大多数情况下，单个代码库就足够了。然而，请注意，对于_Pascal_和早期架构，掩码中的所有线程必须在收敛中执行相同的经编内在指令，并且掩码中所有值的组合必须等于经编的主动掩码。以下代码模式适用于**NVIDIA Volta GPU架构**，但不适用于_Pascal_或更早的架构。

> if (tid % warpSize < 16) {
>     ...
>     float swapped = __shfl_xor_sync(0xffffffff, val, 16);
>     ...
> } else {
>     ...
>     float swapped = __shfl_xor_sync(0xffffffff, val, 16);
>     ...
> }

`__ballot(1)`的替代品是`__activemask()`请注意，即使在单个代码路径内，经编中的线程也可以发散。因此，`__activemask()`和`__ballot(1)`可能只返回当前代码路径上线程的子集。当`data[i]`大于阈值时，以下无效代码示例将`output`的位i设置为1。`__activemask()`用于尝试启用`dataLen`不是32的倍数的情况。

> // Sets bit in output[] to 1 if the correspond element in data[i]
> // is greater than 'threshold', using 32 threads in a warp.
> 
> for (int i = warpLane; i < dataLen; i += warpSize) {
>     unsigned active = __activemask();
>     unsigned bitPack = __ballot_sync(active, data[i] > threshold);
>     if (warpLane == 0) {
>         output[i / 32] = bitPack;
>     }
> }

此代码无效，因为CUDA不保证经编仅在循环条件下发散。当因其他原因发生发散时，扭曲中线程的不同子集将为同一32位输出元素计算冲突结果。正确的代码可能会使用非发散循环条件与`__ballot_sync()`一起安全地枚举参与阈值计算的经编中的线程集，如下所示。

> for (int i = warpLane; i - warpLane < dataLen; i += warpSize) {
>     unsigned active = __ballot_sync(0xFFFFFFFF, i < dataLen);
>     if (i < dataLen) {
>         unsigned bitPack = __ballot_sync(active, data[i] > threshold);
>         if (warpLane == 0) {
>             output[i / 32] = bitPack;
>         }
>     }
> }

[发现模式](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#discovery-pattern-cg)展示了`__activemask()`的有效用例。

1. 如果应用程序有扭曲同步代码，它们需要在通过全局或共享内存在线程之间交换数据的任何步骤之间插入新的`__syncwarp()`扭曲范围屏障同步指令。假设代码是按锁步骤执行的，或者从单独的线程读取/写入在没有同步的情况下在经编中可见的假设是无效的。
    
    __shared__ float s_buff[BLOCK_SIZE];
    s_buff[tid] = val;
    __syncthreads();
    
    // Inter-warp reduction
    for (int i = BLOCK_SIZE / 2; i >= 32; i /= 2) {
        if (tid < i) {
            s_buff[tid] += s_buff[tid+i];
        }
        __syncthreads();
    }
    
    // Intra-warp reduction
    // Butterfly reduction simplifies syncwarp mask
    if (tid < 32) {
        float temp;
        temp = s_buff[tid ^ 16]; __syncwarp();
        s_buff[tid] += temp;     __syncwarp();
        temp = s_buff[tid ^ 8];  __syncwarp();
        s_buff[tid] += temp;     __syncwarp();
        temp = s_buff[tid ^ 4];  __syncwarp();
        s_buff[tid] += temp;     __syncwarp();
        temp = s_buff[tid ^ 2];  __syncwarp();
        s_buff[tid] += temp;     __syncwarp();
    }
    
    if (tid == 0) {
        *output = s_buff[0] + s_buff[1];
    }
    __syncthreads();
    
2. 尽管`__syncthreads()`一直被记录为同步线程块中的所有线程，但_Pascal_和以前的架构只能在经编级别强制同步。在某些情况下，只要每个经编中至少有一些线程到达障碍，这允许障碍物成功而不被每条线执行。从**NVIDIA Volta GPU架构**开始，CUDA内置`__syncthreads()`和PTX指令`bar.sync`（及其衍生物）是按线程强制执行的，因此在块中所有未退出的线程到达之前不会成功。利用先前行为的代码可能会陷入僵局，必须进行修改，以确保所有未退出的线程都达到障碍。
    

`compute-saniter`提供的`racecheck`和`synccheck`工具可以帮助定位违规。

为了在实施上述纠正措施的同时帮助迁移，开发人员可以选择不支持独立线程调度的Pascal调度模型。有关详细信息，请参阅[应用程序兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility)。

### 20.6.3.全球记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-7-x "这个标题的永久链接")

全局内存的行为方式与计算能力5.x的设备相同（见[全局内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x)）。

### 20.6.4.共享内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x "这个标题的永久链接")

为共享内存保留的统一数据缓存的数量可以按内核进行配置。对于_Volta_架构（计算功能7.0），统一数据缓存的大小为128 KB，共享内存容量可以设置为0、8、16、32、64或96 KB。对于_图灵_架构（计算能力7.5），统一数据缓存的大小为96 KB，共享内存容量可以设置为32 KB或64 KB。与Kepler不同，驱动程序会自动配置每个内核的共享内存容量，以避免共享内存占用瓶颈，同时允许在可能的情况下与已启动的内核同时执行。在大多数情况下，驱动程序的默认行为应该提供最佳性能。

由于驱动程序并不总是意识到全部工作负载，因此有时对应用程序来说，提供有关所需共享内存配置的额外提示是有用的。例如，一个共享内存使用很少或没有共享内存的内核可能会要求更大的分叉，以鼓励与需要更多共享内存的后来内核同时执行。新的`cudaFuncSetAttribute()`API允许应用程序设置首选共享内存容量或`carveout`，作为最大支持共享内存容量（_Volta_为96 KB，_Turing_为64 KB）的百分比。

`cudaFuncSetAttribute()`与Kepler引入的传统`cudaFuncSetCacheConfig()`API相比，放宽了对首选共享容量的强制执行。传统API将共享内存容量视为内核启动的硬性要求。因此，将具有不同共享内存配置的内核交织在一起，将不必要地将共享内存重新配置背后的启动序列化。使用新的API，分出被视为提示。如果需要执行功能或避免崩溃，驱动程序可以选择不同的配置。

// Device code
__global__ void MyKernel(...)
{
    __shared__ float buffer[BLOCK_DIM];
    ...
}

// Host code
int carveout = 50; // prefer shared memory capacity 50% of maximum
// Named Carveout Values:
// carveout = cudaSharedmemCarveoutDefault;   //  (-1)
// carveout = cudaSharedmemCarveoutMaxL1;     //   (0)
// carveout = cudaSharedmemCarveoutMaxShared; // (100)
cudaFuncSetAttribute(MyKernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
MyKernel <<<gridDim, BLOCK_DIM>>>(...);

除了整数百分比外，还提供了上面代码注释中列出的几个方便枚举。如果所选的整数百分比没有完全映射到支持的容量（SM 7.0设备支持0、8、16、32、64或96 KB的共享容量），则使用下一个更大的容量。例如，在上述示例中，96 KB的最大值的50%是48 KB，这不是支持的共享内存容量。因此，偏好四舍五入到64 KB。

计算能力7.x设备允许单个线程块处理共享内存的全部容量：_Volta_上96 KB，_Turing_上64 KB。依赖每个块超过48 KB的共享内存分配的内核是特定于架构的，因此它们必须使用动态共享内存（而不是静态大小的数组），并且需要使用`cudaFuncSetAttribute()`进行显式选择，如下所示。

// Device code
__global__ void MyKernel(...)
{
    extern __shared__ float buffer[];
    ...
}

// Host code
int maxbytes = 98304; // 96 KB
cudaFuncSetAttribute(MyKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
MyKernel <<<gridDim, blockDim, maxbytes>>>(...);

否则，共享内存的行为方式与计算能力5.x的设备相同（请参阅[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-5-x)）。

## 20.7.计算能力8.x[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x "这个标题的永久链接")

### 20.7.1.建筑[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-8-x "这个标题的永久链接")

流媒体多处理器（SM）由以下部分组成：

- 计算能力8.0设备中的64个FP32内核用于单精度算术运算，计算能力8.6、8.7和8.9设备中的128个FP32内核，
    
- 32个FP64内核，用于计算能力8.0的设备中的双精度算术运算，以及计算能力8.6、8.7和8.9的设备中的2个FP64内核
    
- 用于整数数学的64个INT32核心，
    
- 4个混合精度第三代张量核心，支持半精度（fp16）、`__nv_bfloat16`、`tf32`、子字节和双精度（fp64）矩阵算术，用于计算能力8.0、8.6和8.7（详情见[Warp矩阵函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma)），
    
- 4个混合精度第四代张量核心支持`fp8`、`fp16`、`__nv_bfloat16`、`tf32`、sub-byte和`fp64`，用于计算能力8.9（详情请参阅[Warp矩阵函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma)），
    
- 用于单精度浮点超越函数的16个特殊函数单元，
    
- 4个扭曲调度器。
    

SM在其调度器之间静态地分配其翘曲。然后，在每次指令发布时间，每个调度器都会为其分配的其中一个准备执行的扭曲发布一个指令（如果有的话）。

SM有：

- 一个只读的常量缓存，由所有功能单元共享，并加快来自常量内存空间的读取速度，该空间位于设备内存中，
    
- 统一的数据缓存和共享内存，计算能力为8.0和8.7的设备（1.5倍_Volta的_128 KB容量）的总大小为192 KB，计算能力8.6和8.9的设备为128 KB。
    

共享内存被分区到统一数据缓存中，可以配置为各种大小（请参阅[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x)）。剩余的数据缓存作为L1缓存，也用于实现[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)中提到的各种寻址和数据过滤模式的纹理单元。

### 20.7.2.全球记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-8-x "这个标题的永久链接")

全局内存的行为方式与计算能力5.x的设备相同（请参阅[全局内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x)）。

### 20.7.3.共享内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x "这个标题的永久链接")

与[Volta架构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-7-x)类似，为共享内存保留的统一数据缓存量可以按每个内核配置。对于**NVIDIA Ampere GPU架构**，对于计算能力8.0和8.7的设备，统一数据缓存的大小为192 KB，对于计算能力8.6和8.9的设备，统一数据缓存大小为128 KB。对于具有计算能力8.0和8.7的设备，共享内存容量可以设置为0、8、16、32、64、100、132或164 KB，对于具有计算能力8.6和8.9的设备，可以设置为0、8、16、32、64或100 KB。

应用程序可以使用`cudaFuncSetAttribute()`设置`carveout`，即首选的共享内存容量。

cudaFuncSetAttribute(kernel_name, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);

API可以指定计算能力8.0和8.7设备最大支持共享内存容量164 KB的整数百分比，计算能力8.6和8.9设备100KB的整数百分比，或指定为以下值之一：`{cudaSharedmemCarveoutDefault`，`cudaSharedmemCarveoutMaxL1`或`cudaSharedmemCarveoutMaxShared`。使用百分比时，分段四舍五入到最接近的支持共享内存容量。例如，对于具有计算能力8.0的设备，50%将映射到100 KB的雕刻，而不是82 KB的雕刻。设置`cudaFuncAttributePreferredSharedMemoryCarveout`被驱动程序视为提示；如果需要，驱动程序可以选择其他配置。

计算能力8.0和8.7的设备允许单个线程块处理高达163 KB的共享内存，而计算能力8.6和8.9的设备允许高达99 KB的共享内存。依赖每个块超过48 KB的共享内存分配的内核是特定于架构的，必须使用动态共享内存，而不是静态大小的共享内存阵列。这些内核需要使用`cudaFuncSetAttribute()`来设置`cudaFuncAttributeMaxDynamicSharedMemorySize`明确选择加入；请参阅**NVIDIA Volta GPU架构**的[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x)。

请注意，每个线程块的最大共享内存量小于每个SM可用的最大共享内存分区。线程块未可用的1 KB共享内存保留给系统使用。

## 20.8.计算能力9.0[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0 "这个标题的永久链接")

### 20.8.1.建筑[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-9-0 "这个标题的永久链接")

流媒体多处理器（SM）由以下部分组成：

- 128个FP32核心，用于单精度算术运算，
    
- 用于双精度算术运算的64个FP64内核，
    
- 用于整数数学的64个INT32核心，
    
- 4个混合精度第四代张量核心支持`E4M3`或`E5M2`中的新`FP8`输入类型，用于指数（E）和曼蒂萨（M）、半精度（fp16）、`__nv_bfloat16`、`tf32`、INT8和双精度（fp64）矩阵算术（详情见[扭曲矩阵函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma)），并支持稀疏性，
    
- 用于单精度浮点超越函数的16个特殊函数单元，
    
- 4个扭曲调度器。
    

SM在其调度器之间静态地分配其翘曲。然后，在每次指令发布时间，每个调度器都会为其分配的其中一个准备执行的扭曲发布一个指令（如果有的话）。

SM有：

- 一个只读的常量缓存，由所有功能单元共享，并加快来自常量内存空间的读取速度，该空间位于设备内存中，
    
- 统一的数据缓存和共享内存，总大小为256 KB，适用于计算能力9.0的设备（1.33x **NVIDIA Ampere GPU架构的**192 KB容量）。
    

共享内存被分区到统一数据缓存中，可以配置为各种大小（请参阅[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-9-0)）。剩余的数据缓存作为L1缓存，也用于实现[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)中提到的各种寻址和数据过滤模式的纹理单元。

### 20.8.2.全球记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-9-0 "这个标题的永久链接")

全局内存的行为方式与计算能力5.x的设备相同（请参阅[全局内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x)）。

### 20.8.3.共享内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-9-0 "这个标题的永久链接")

与[NVIDIA Ampere GPU架构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-8-x)类似，为共享内存保留的统一数据缓存量可以按内核配置。对于_NVIDIA H100 Tensor Core GPU架构_，对于具有9.0計算能力的设备，统一数据缓存的大小为256 KB。共享内存容量可以设置为0、8、16、32、64、100、132、164、196或228 KB。

与[NVIDIA Ampere GPU架构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x)一样，应用程序可以配置其首选的共享内存容量，即`carveout`。计算能力9.0的设备允许单个线程块处理高达227 KB的共享内存。依赖每个块超过48 KB的共享内存分配的内核是特定于架构的，必须使用动态共享内存，而不是静态大小的共享内存阵列。这些内核需要使用`cudaFuncSetAttribute()`来设置`cudaFuncAttributeMaxDynamicSharedMemorySize`明确选择加入；请参阅**NVIDIA Volta GPU架构**的[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x)。

请注意，每个线程块的最大共享内存量小于每个SM可用的最大共享内存分区。线程块未可用的1 KB共享内存保留给系统使用。

### 20.8.4.加速专业计算的特点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-accelerating-specialized-computations "这个标题的永久链接")

NVIDIA Hopper GPU架构包括加速矩阵乘积（MMA）计算的功能：

- MMA指令的非同步执行
    
- MMA指令作用于跨越经编组的大矩阵
    
- 在扭曲群之间动态重新分配寄存器容量，以支持更大的矩阵，以及
    
- 直接从共享内存访问的操作数矩阵
    

有关更多详细信息，请参阅[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-set)。

此功能集仅通过内联PTX在CUDA编译工具链中可用。

强烈建议应用程序通过cuBLAS、cuDNN或cuFFT等CUDA-X库来利用这个复杂的功能集。

强烈建议设备内核通过[CUTLASS](https://github.com/NVIDIA/cutlass)利用这个复杂的功能集，CUTLASS是一个CUDA C++模板抽象的集合，用于在CUDA内实现高性能矩阵乘法（GEMM）和相关计算的所有级别和规模。

## 20.9.计算能力10.0[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-10-0 "这个标题的永久链接")

### 20.9.1.建筑[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-10-0 "这个标题的永久链接")

流媒体多处理器（SM）由以下部分组成：

- 128个FP32核心，用于单精度算术运算，
    
- 用于双精度算术运算的64个FP64内核，
    
- 用于整数数学的64个INT32核心，
    
- 4个混合精度第五代张量核心支持`E4M3`或`E5M2``FP8`输入类型，用于指数（E）和曼蒂萨（M）、半精度（fp16）、`__nv_bfloat16`、`tf32`、INT8和双精度（fp64）矩阵算术（详情见[Warp矩阵函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma)），支持稀疏性，
    
- 用于单精度浮点超越函数的16个特殊函数单元，
    
- 4个扭曲调度器。
    

SM在其调度器之间静态地分配其翘曲。然后，在每次指令发布时间，每个调度器都会为其分配的其中一个准备执行的扭曲发布一个指令（如果有的话）。

SM有：

- 一个只读的常量缓存，由所有功能单元共享，并加快来自常量内存空间的读取速度，该空间位于设备内存中，
    
- 统一的数据缓存和共享内存，总大小为256 KB，适用于计算能力10.0的设备
    

共享内存被分区到统一数据缓存中，可以配置为各种大小（请参阅[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-10-0)）。剩余的数据缓存作为L1缓存，也用于实现[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)中提到的各种寻址和数据过滤模式的纹理单元。

### 20.9.2.全球记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-10-0 "这个标题的永久链接")

全局内存的行为方式与计算能力5.x的设备相同（请参阅[全局内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x)）。

### 20.9.3.共享内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-10-0 "这个标题的永久链接")

为共享内存保留的统一数据缓存量可按内核配置，与[计算能力9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-9-0)相同。对于计算能力为10.0的设备，统一数据缓存的大小为256 KB。共享内存容量可以设置为0、8、16、32、64、100、132、164、196或228 KB。

与[NVIDIA Ampere GPU架构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x)一样，应用程序可以配置其首选的共享内存容量，即`carveout`。计算能力10.0的设备允许单个线程块处理高达227 KB的共享内存。依赖每个块超过48 KB的共享内存分配的内核是特定于架构的，必须使用动态共享内存，而不是静态大小的共享内存阵列。这些内核需要使用`cudaFuncSetAttribute()`来设置`cudaFuncAttributeMaxDynamicSharedMemorySize`明确选择加入；请参阅Volta架构的[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x)。

请注意，每个线程块的最大共享内存量小于每个SM可用的最大共享内存分区。线程块未可用的1 KB共享内存保留给系统使用。

### 20.9.4.加速专业计算的特点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-accelerating-specialized-computations-10-0 "这个标题的永久链接")

NVIDIA Blackwell GPU架构扩展了功能，从NVIDIA Hopper GPU架构加速矩阵乘积（MMA）。

有关更多详细信息，请参阅[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-set)。

此功能集仅通过内联PTX在CUDA编译工具链中可用。

强烈建议应用程序通过cuBLAS、cuDNN或cuFFT等CUDA-X库来利用这个复杂的功能集。

强烈建议设备内核通过[CUTLASS](https://github.com/NVIDIA/cutlass)利用这个复杂的功能集，CUTLASS是一个CUDA C++模板抽象的集合，用于在CUDA内实现高性能矩阵乘法（GEMM）和相关计算的所有级别和规模。

## 20.10.计算能力12.0[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-12-0 "这个标题的永久链接")

### 20.10.1.建筑[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-12-0 "这个标题的永久链接")

流媒体多处理器（SM）由以下部分组成：

- 128个FP32核心，用于单精度算术运算，
    
- 2个FP64内核，用于双精度算术运算，
    
- 用于整数数学的64个INT32核心，
    
- 混合精度第五代张量核心支持`E4M3`或`E5M2``FP8`输入类型，用于指数（E）和曼蒂萨（M）、半精度（fp16）、`__nv_bfloat16`、`tf32`、INT8和双精度（fp64）矩阵算术（详情请参阅[Warp矩阵函数](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma)），支持稀疏性，
    
- 用于单精度浮点超越函数的16个特殊函数单元，
    
- 4个扭曲调度器。
    

SM在其调度器之间静态地分配其翘曲。然后，在每次指令发布时间，每个调度器都会为其分配的其中一个准备执行的扭曲发布一个指令（如果有的话）。

SM有：

- 一个只读的常量缓存，由所有功能单元共享，并加快来自常量内存空间的读取速度，该空间位于设备内存中，
    
- 统一的数据缓存和共享内存，总大小为100 KB，适用于具有计算能力的设备12.0
    

共享内存被分区到统一数据缓存中，可以配置为各种大小（请参阅[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-12-0)）。剩余的数据缓存作为L1缓存，也用于实现[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)中提到的各种寻址和数据过滤模式的纹理单元。

### 20.10.2.全球记忆[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-12-0 "这个标题的永久链接")

全局内存的行为方式与计算能力5.x的设备相同（请参阅[全局内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x)）。

### 20.10.3.共享内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-12-0 "这个标题的永久链接")

为共享内存保留的统一数据缓存量可按内核配置，与[计算能力9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-9-0)相同。对于具有12.0计算能力的设备，统一数据缓存的大小为100 KB。共享内存容量可以设置为0、8、16、32、64或100 KB。

与[NVIDIA Ampere GPU架构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x)一样，应用程序可以配置其首选的共享内存容量，即`carveout`。计算能力12.0的设备允许单个线程块处理高达99 KB的共享内存。依赖每个块超过48 KB的共享内存分配的内核是特定于架构的，必须使用动态共享内存，而不是静态大小的共享内存阵列。这些内核需要使用`cudaFuncSetAttribute()`来设置`cudaFuncAttributeMaxDynamicSharedMemorySize`明确选择加入；请参阅Volta架构的[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-7-x)。

请注意，每个线程块的最大共享内存量小于每个SM可用的最大共享内存分区。线程块未可用的1 KB共享内存保留给系统使用。

### 20.10.4.加速专业计算的特点[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-accelerating-specialized-computations-12-0 "这个标题的永久链接")

NVIDIA Blackwell GPU架构扩展了功能，从NVIDIA Hopper GPU架构加速矩阵乘积（MMA）。

有关更多详细信息，请参阅[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-set)。

此功能集仅通过内联PTX在CUDA编译工具链中可用。

强烈建议应用程序通过cuBLAS、cuDNN或cuFFT等CUDA-X库来利用这个复杂的功能集。

强烈建议设备内核通过[CUTLASS](https://github.com/NVIDIA/cutlass)利用这个复杂的功能集，CUTLASS是一个CUDA C++模板抽象的集合，用于在CUDA内实现高性能矩阵乘法（GEMM）和相关计算的所有级别和规模。

[27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id399)

超过48 KB需要动态共享内存

[28](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id410)

2个FP64内核，用于计算能力7.5的设备的双精度算术操作

# 21.驱动程序API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-api "这个标题的永久链接")

本节假设了解[CUDA运行时](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-runtime)中描述的概念。

驱动程序API在`cuda`动态库（`cuda.dll`或`cuda.so`）中实现，该库在安装设备驱动程序期间复制到系统上。它的所有入口点都以cu为前缀。

这是一个基于句柄的命令式API：大多数对象由不透明句柄引用，这些句柄可以指定为操作对象的函数。

[表28](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-api-objects-available-in-cuda-driver-api)总结了驱动程序API中可用的对象。

表28 CUDA驱动程序API中可用的对象[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-api-objects-available-in-cuda-driver-api "此表的永久链接")
|对象|处理|描述|
|---|---|---|
|设备|CU设备|支持CUDA的设备|
|上下文|CU上下文|大致相当于CPU进程|
|模块|CU模块|大致等同于动态库|
|功能|功能|内核|
|堆内存|CUdeviceptr|指向设备内存的指针|
|CUDA阵列|CU阵列|设备上一维或二维数据的不透明容器，可通过纹理或表面引用读取|
|纹理对象|切割参考|描述如何解释纹理内存数据的对象|
|表面参考|CUsurfref|描述如何读取或写入CUDA数组的对象|
|流动|CU流|描述CUDA流的对象|
|事件|CU事件|描述CUDA事件的对象|

在调用驱动程序API的任何函数之前，必须使用`cuInit()`初始化驱动程序API。然后，必须创建一个CUDA上下文，该上下文附加到特定设备，并使调用主机线程处于当前，如[上下文](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#context)中所述。

在CUDA上下文中，如[模块](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#module)中所述，内核被主机代码显式加载为PTX或二进制对象。因此，用C++编写的内核必须单独编译成_PTX_或二进制对象。如[内核执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernel-execution)中所述，内核使用API入口点启动。

任何想要在未来设备架构上运行的应用程序都必须加载_PTX_，而不是二进制代码。这是因为二进制代码是特定于架构的，因此与未来的架构不兼容，而_PTX_代码在加载时由设备驱动程序编译为二进制代码。

这是使用驱动程序API编写的[Kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernels)样本的主机代码：

int main()
{
    int N = ...;
    size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Initialize input vectors
    ...

    // Initialize
    cuInit(0);

    // Get number of devices supporting CUDA
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA.\n");
        exit (0);
    }

    // Get handle for device 0
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, 0);

    // Create context
    CUcontext cuContext;
    cuCtxCreate(&cuContext, NULL, 0, cuDevice);

    // Create module from binary file
    CUmodule cuModule;
    cuModuleLoad(&cuModule, "VecAdd.ptx");

    // Allocate vectors in device memory
    CUdeviceptr d_A;
    cuMemAlloc(&d_A, size);
    CUdeviceptr d_B;
    cuMemAlloc(&d_B, size);
    CUdeviceptr d_C;
    cuMemAlloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cuMemcpyHtoD(d_A, h_A, size);
    cuMemcpyHtoD(d_B, h_B, size);

    // Get function handle from module
    CUfunction vecAdd;
    cuModuleGetFunction(&vecAdd, cuModule, "VecAdd");

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
            (N + threadsPerBlock - 1) / threadsPerBlock;
    void* args[] = { &d_A, &d_B, &d_C, &N };
    cuLaunchKernel(vecAdd,
                   blocksPerGrid, 1, 1, threadsPerBlock, 1, 1,
                   0, 0, args, 0);

    ...
}

Full code can be found in the `vectorAddDrv` CUDA sample.

## 21.1.上下文[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#context "这个标题的永久链接")

CUDA上下文类似于CPU进程。在驱动程序API中执行的所有资源和操作都封装在CUDA上下文中，当上下文被销毁时，系统会自动清理这些资源。除了模块和纹理或表面引用等对象外，每个上下文都有自己独特的地址空间。因此，来自不同上下文的`CUdeviceptr`引用不同的内存位置。

主机线程一次可能只有一个设备上下文当前。当使用`cuCtxCreate()`创建上下文时，它将成为调用主机线程的当前上下文。如果在线程中没有有效的上下文，则在上下文中运行的CUDA函数（大多数不涉及设备枚举或上下文管理的函数）将返回`CUDA_ERROR_INVALID_CONTEXT`。

每个主机线程都有一个当前上下文的堆栈。`cuCtxCreate()`将新上下文推送到堆栈的顶部。可以调用`cuCtxPopCurrent()`将上下文与主机线程分离。然后，上下文是“浮动的”，并且可以推送，因为任何主机的当前上下文thread.cuCtxPopCurrent`cuCtxPopCurrent()`也会恢复之前的当前上下文（如果有的话）。

每个上下文还维护使用计数。`cuCtxCreate()`创建了使用计数为1的上下文。`cuCtxAttach()`增加使用计数，`cuCtxDetach()`将其递减。调用`cuCtxDetach()`或`cuCtxDestroy()`时，当使用计数变为0时，上下文就会被破坏。

驱动程序API与运行时可互操作，并且可以通过`cuDevicePrimaryCtxRetain()`访问运行时从驱动程序API管理_的主要上下文_（请参阅[初始化](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#initialization)）。

使用计数促进了在同一上下文中运行的第三方编写代码之间的互操作性。例如，如果加载三个库以使用相同的上下文，则当库使用上下文完成时，每个库将调用`cuCtxAttach()`来增加使用计数，并调用`cuCtxDetach()`来使用计数。对于大多数库来说，预计应用程序在加载或初始化库之前会创建一个上下文；这样，应用程序可以使用自己的启发式方法创建上下文，并且库只需在交给它的上下文上操作。希望创建自己的上下文的库——他们的API客户端可能创建了也可能没有创建了自己的上下文——将使用`cuCtxPushCurrent()`和`cuCtxPopCurrent()`如下图所示。

![图书馆上下文管理](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/library-context-management.png)

图41 图书馆上下文管理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#library-context-management "此图像的永久链接")

## 21.2.模块[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#module "这个标题的永久链接")

模块是设备代码和数据的动态可加载包，类似于Windows中的DLL，由nvcc输出（请参阅[使用NVCC编译](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-with-nvcc)）。所有符号的名称，包括函数、全局变量和纹理或表面引用，都保留在模块范围内，以便由独立第三方编写的模块可以在同一CUDA上下文中互操作。

此代码示例加载一个模块，并将一个句柄检索到一些内核：

CUmodule cuModule;
cuModuleLoad(&cuModule, "myModule.ptx");
CUfunction myKernel;
cuModuleGetFunction(&myKernel, cuModule, "MyKernel");

此代码示例从PTX代码中编译并加载一个新模块，并解析编译错误：

#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[3];
void* values[3];
char* PTXCode = "some PTX code";
char error_log[BUFFER_SIZE];
int err;
options[0] = CU_JIT_ERROR_LOG_BUFFER;
values[0]  = (void*)error_log;
options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[1]  = (void*)BUFFER_SIZE;
options[2] = CU_JIT_TARGET_FROM_CUCONTEXT;
values[2]  = 0;
err = cuModuleLoadDataEx(&cuModule, PTXCode, 3, options, values);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);

此代码示例从多个PTX代码编译、链接和加载新模块，并解析链接和编译错误：

#define BUFFER_SIZE 8192
CUmodule cuModule;
CUjit_option options[6];
void* values[6];
float walltime;
char error_log[BUFFER_SIZE], info_log[BUFFER_SIZE];
char* PTXCode0 = "some PTX code";
char* PTXCode1 = "some other PTX code";
CUlinkState linkState;
int err;
void* cubin;
size_t cubinSize;
options[0] = CU_JIT_WALL_TIME;
values[0] = (void*)&walltime;
options[1] = CU_JIT_INFO_LOG_BUFFER;
values[1] = (void*)info_log;
options[2] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
values[2] = (void*)BUFFER_SIZE;
options[3] = CU_JIT_ERROR_LOG_BUFFER;
values[3] = (void*)error_log;
options[4] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
values[4] = (void*)BUFFER_SIZE;
options[5] = CU_JIT_LOG_VERBOSE;
values[5] = (void*)1;
cuLinkCreate(6, options, values, &linkState);
err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                    (void*)PTXCode0, strlen(PTXCode0) + 1, 0, 0, 0, 0);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
err = cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
                    (void*)PTXCode1, strlen(PTXCode1) + 1, 0, 0, 0, 0);
if (err != CUDA_SUCCESS)
    printf("Link error:\n%s\n", error_log);
cuLinkComplete(linkState, &cubin, &cubinSize);
printf("Link completed in %fms. Linker Output:\n%s\n", walltime, info_log);
cuModuleLoadData(cuModule, cubin);
cuLinkDestroy(linkState);

Full code can be found in the `ptxjit` CUDA sample.

## 21.3.内核执行[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernel-execution "这个标题的永久链接")

`cuLaunchKernel()`启动具有给定执行配置的内核。

参数要么作为指针数组（在`cuLaunchKernel()`的最后一个参数旁边）传递，其中第n个指针对应于第n个参数，并指向从中复制参数的内存区域，要么作为额外选项之一（`cuLaunchKernel()`的最后一个参数）。

当参数作为额外选项传递时（CU_LAUNCH`CU_LAUNCH_PARAM_BUFFER_POINTER`选项），它们作为指向单个缓冲区的指针传递，通过匹配设备代码中每个参数类型的对齐要求，假设参数相互正确偏移。

[表7](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types-alignment-requirements-in-device-code)列出了内置矢量类型的设备代码中的对齐要求。对于所有其他基本类型，设备代码中的对齐要求与主机代码中的对齐要求相匹配，因此可以使用`__alignof()`获得。唯一的例外是，当主机编译器在单字边界上对齐双`double``longlong`（在64位系统上对齐）而不是在两个字边界上（例如，使用`gcc`编译标志`-mno-align-double`），因为在设备代码中，这些类型总是在两个字边界上对齐。

`CUdeviceptr`是一个整数，但表示一个指针，因此它的对齐要求是`__alignof(void*)`

以下代码示例使用宏（`ALIGN_UP()`调整每个参数的偏移量，以满足其对齐要求，另一个宏（`ADD_TO_PARAM_BUFFER()`将每个参数添加到传递给`CU_LAUNCH_PARAM_BUFFER_POINTER`选项的参数缓冲区。

#define ALIGN_UP(offset, alignment) \
      (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

char paramBuffer[1024];
size_t paramBufferSize = 0;

#define ADD_TO_PARAM_BUFFER(value, alignment)                   \
    do {                                                        \
        paramBufferSize = ALIGN_UP(paramBufferSize, alignment); \
        memcpy(paramBuffer + paramBufferSize,                   \
               &(value), sizeof(value));                        \
        paramBufferSize += sizeof(value);                       \
    } while (0)

int i;
ADD_TO_PARAM_BUFFER(i, __alignof(i));
float4 f4;
ADD_TO_PARAM_BUFFER(f4, 16); // float4's alignment is 16
char c;
ADD_TO_PARAM_BUFFER(c, __alignof(c));
float f;
ADD_TO_PARAM_BUFFER(f, __alignof(f));
CUdeviceptr devPtr;
ADD_TO_PARAM_BUFFER(devPtr, __alignof(devPtr));
float2 f2;
ADD_TO_PARAM_BUFFER(f2, 8); // float2's alignment is 8

void* extra[] = {
    CU_LAUNCH_PARAM_BUFFER_POINTER, paramBuffer,
    CU_LAUNCH_PARAM_BUFFER_SIZE,    &paramBufferSize,
    CU_LAUNCH_PARAM_END
};
cuLaunchKernel(cuFunction,
               blockWidth, blockHeight, blockDepth,
               gridWidth, gridHeight, gridDepth,
               0, 0, 0, extra);

结构的对齐要求等于其字段的最大对齐要求。因此，包含内置向量类型、`CUdeviceptr`或非对齐双和`long`结构的对齐要求在设备代码和主机代码之间可能有所不同。这样的结构也可能有不同的填充。例如，以下结构在主机代码中根本没有填充，但在设备代码中填充了字段`f`12个字节，因为字段`f4`的对齐要求为16。

typedef struct {
    float  f;
    float4 f4;
} myStruct;

## 21.4.运行时和驱动程序API之间的互操作性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interoperability-between-runtime-and-driver-apis "这个标题的永久链接")

应用程序可以将运行时API代码与驱动程序API代码混合。

如果通过驱动程序API创建并使上下文成为当前，则后续的运行时调用将拾取此上下文，而不是创建新的上下文。

如果运行时被初始化（如[CUDA运行时](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-runtime)中提到的隐式），可以使用`cuCtxGetCurrent()`来检索初始化期间创建的上下文。此上下文可用于后续的驱动程序API调用。

运行时隐式创建的上下文称为主_上下文_（请参阅[初始化](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#initialization)）。它可以通过主上[下文管理](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html)功能从驱动程序API进行管理。

设备内存可以使用任一API进行分配和释放。`CUdeviceptr`可以投向常规指针，反之亦然：

CUdeviceptr devPtr;
float* d_data;

// Allocation using driver API
cuMemAlloc(&devPtr, size);
d_data = (float*)devPtr;

// Allocation using runtime API
cudaMalloc(&d_data, size);
devPtr = (CUdeviceptr)d_data;

特别是，这意味着使用驱动程序API编写的应用程序可以调用使用运行时API编写的库（如cuFFT、cuBLAS......）。

参考手册的设备和版本管理部分的所有功能都可以互换使用。

## 21.5.司机入口点访问[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-entry-point-access "这个标题的永久链接")

### 21.5.1.介绍[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#introduction-driver-entry-point-access "这个标题的永久链接")

`DriverEntryAccessAPIs`提供了一种检索CUDA驱动程序函数地址的方法。从CUDA 11.3开始，用户可以使用从这些API获得的函数指针调用可用的CUDA驱动程序API。

这些API提供的功能与POSIX平台上的dlsym和Windows上的GetProcAddress相似。提供的API将让用户：

- 使用检索驱动程序函数的地址`CUDA Driver API.`
    
- 使用检索驱动程序函数的地址`CUDA Runtime API.`
    
- 请求CUDA驱动程序函数_的每线程默认流_版本。有关更多详细信息，请参阅[检索每线程默认流版本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#retrieve-per-thread-default-stream-versions)。
    
- 在旧工具包上访问新的CUDA功能，但使用较新的驱动程序。
    

### 21.5.2.驱动程序功能类型defs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-function-typedefs "这个标题的永久链接")

为了帮助检索CUDA驱动程序API入口点，CUDA工具包提供了对包含所有CUDA驱动程序API的函数指针定义的标题的访问。这些标题与CUDA工具包一起安装，并在工具包的`include/`目录中可用。下表总结了包含每个CUDA API标头文件的`typedefs`的标头文件。

表29 CUDA驱动程序API的Typedefs标头文件[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id485 "此表的永久链接")
|API标头文件|API Typedef标头文件|
|---|---|
|`cuda.h`|`cudaTypedefs.h`|
|`cudaGL.h`|`cudaGLTypedefs.h`|
|`cudaProfiler.h`|`cudaProfilerTypedefs.h`|
|`cudaVDPAU.h`|`cudaVDPAUTypedefs.h`|
|`cudaEGL.h`|`cudaEGLTypedefs.h`|
|`cudaD3D9.h`|`cudaD3D9Typedefs.h`|
|`cudaD3D10.h`|`cudaD3D10Typedefs.h`|
|`cudaD3D11.h`|`cudaD3D11Typedefs.h`|

上述标题本身没有定义实际的函数指针；它们定义了函数指针的typedefs。例如，`cudaTypedefs.h`具有以下驱动程序API `cuMemAlloc`的typedefs：

typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v3020)(CUdeviceptr_v2 *dptr, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2000)(CUdeviceptr_v1 *dptr, unsigned int bytesize);

除了第一个版本外，CUDA驱动程序符号有一个基于版本的命名方案，其名称中带有`_v*`扩展名。当特定CUDA驱动程序API的签名或语义发生变化时，我们会增加相应驱动程序符号的版本号。在`cuMemAlloc`驱动程序API的情况下，第一个驱动程序符号名称是`cuMemAlloc`，下一个符号名称是`cuMemAlloc_v2`。CUDA 2.0（2000）中引入的第一个版本的typedef是`PFN_cuMemAlloc_v2000`。CUDA 3.2（3020）中引入的下一个版本的类型定义是`PFN_cuMemAlloc_v3020`。

`typedefs`可用于更轻松地定义代码中适当类型的函数指针：

PFN_cuMemAlloc_v3020 pfn_cuMemAlloc_v2;
PFN_cuMemAlloc_v2000 pfn_cuMemAlloc_v1;

### 21.5.3.驱动程序功能检索[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-function-retrieval "这个标题的永久链接")

使用驱动程序入口点访问API和适当的typedef，我们可以获取指向任何CUDA驱动程序API的函数指针。

#### 21.5.3.1.使用驱动程序API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-the-driver-api "这个标题的永久链接")

驱动程序API需要CUDA版本作为参数来获取请求的驱动程序符号的ABI兼容版本。CUDA驱动程序API有一个用`_v*`扩展表示的per-function ABI。例如，考虑`cuStreamBeginCapture`的版本及其来自`cudaTypedefs.h`相应`typedefs`：

// cuda.h
CUresult CUDAAPI cuStreamBeginCapture(CUstream hStream);
CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode);

// cudaTypedefs.h
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10000)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamBeginCapture_v10010)(CUstream hStream, CUstreamCaptureMode mode);

从代码片段中的上述`typedefs`，版本后缀`_v10000`和`_v10010`表明上述API分别在CUDA 10.0和CUDA 10.1中引入。

#include <cudaTypedefs.h>

// Declare the entry points for cuStreamBeginCapture
PFN_cuStreamBeginCapture_v10000 pfn_cuStreamBeginCapture_v1;
PFN_cuStreamBeginCapture_v10010 pfn_cuStreamBeginCapture_v2;

// Get the function pointer to the cuStreamBeginCapture driver symbol
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v1, 10000, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
// Get the function pointer to the cuStreamBeginCapture_v2 driver symbol
cuGetProcAddress("cuStreamBeginCapture", &pfn_cuStreamBeginCapture_v2, 10010, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

Referring to the code snippet above, to retrieve the address to the `_v1` version of the driver API `cuStreamBeginCapture`, the CUDA version argument should be exactly 10.0 (10000). Similarly, the CUDA version for retrieving the address to the `_v2` version of the API should be 10.1 (10010). Specifying a higher CUDA version for retrieving a specific version of a driver API might not always be portable. For example, using 11030 here would still return the `_v2` symbol, but if a hypothetical `_v3` version is released in CUDA 11.3, the `cuGetProcAddress` API would start returning the newer `_v3` symbol instead when paired with a CUDA 11.3 driver. Since the ABI and function signatures of the `_v2` and `_v3` symbols might differ, calling the `_v3` function using the `_v10010` typedef intended for the `_v2` symbol would exhibit undefined behavior.

请注意，请求具有无效CUDA版本的驱动程序API将返回错误`CUDA_ERROR_NOT_FOUND`。在上述代码示例中，传递低于10000（CUDA 10.0）的版本将无效。

#### 21.5.3.2.使用运行时API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-the-runtime-api "这个标题的永久链接")

运行时API `cudaGetDriverEntryPointByVersion`使用提供的CUDA版本，以与`cuGetProcAddress`相同的方式获取请求的驱动程序符号的ABI兼容版本。在下面的代码片段中，最低CUDA版本要求是CUDA 11.2 ascuMemAllocAsync。

#include <cudaTypedefs.h>

int cudaVersion;
// Ensure a CUDA driver >= 11.2 is installed or we will get an error from cuGetProcAddress
status = cuDriverGetVersion(&cudaVersion);
if (cudaVersion >= 11020) {

   // Declare the entry point
   PFN_cuMemAllocAsync_v11020 pfn_cuMemAllocAsync;

   // Intialize the entry point
   cudaGetDriverEntryPointByVersion("cuMemAllocAsync", &pfn_cuMemAllocAsync, 11020, cudaEnableDefault, &driverStatus);

   // Call the entry point
   if(driverStatus == cudaDriverEntryPointSuccess && pfn_cuMemAllocAsync) {
       pfn_cuMemAllocAsync(...);
   }
}

#### 21.5.3.3.检索每个线程的默认流版本[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#retrieve-per-thread-default-stream-versions "这个标题的永久链接")

一些CUDA驱动程序API可以配置为具有_默认流_或_每线程默认流_语义。具有_每线程默认流_语义的驱动程序API在其名称中带有__ptsz_或__ptds_的后缀。例如，`cuLaunchKernel`有一个名为`cuLaunchKernel_ptsz`的_每线程默认流_变体。使用驱动程序入口点访问API，用户可以请求驱动程序APIcuLaunchKernel的_每线程默认流_版本，而不是_默认流_版本。为默_认流_或_每线程默认流_语义配置CUDA驱动程序API会影响同步行为。更多详情可以[在这里](https://docs.nvidia.com/cuda/cuda-driver-api/stream-sync-behavior.html#stream-sync-behavior__default-stream)找到。

驱动程序API的_默认流_或_每线程默认流_版本可以通过以下方式之一获得：

- Use the compilation flag `--default-stream per-thread` or define the macro `CUDA_API_PER_THREAD_DEFAULT_STREAM` to get _per-thread default stream_behavior.
    
- 分别使用标志`CU_GET_PROC_ADDRESS_LEGACY_STREAM/cudaEnableLegacyStream`或`CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM/cudaEnablePerThreadDefaultStream`强制执行_默认流_或_每线程默认流_行为。
    

#### 21.5.3.4.访问新的CUDA功能[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#access-new-cuda-features "这个标题的永久链接")

始终建议安装最新的CUDA工具包来访问新的CUDA驱动程序功能，但如果由于某种原因，用户不想更新或无法访问最新的工具包，则可以使用API仅使用更新的CUDA驱动程序访问新的CUDA功能。为了讨论，让我们假设用户使用CUDA 12.3，并希望使用CUDA 12.5驱动程序中可用的新驱动程序API `cuFoo`。下面的代码片段说明了这个用例：

int main()
{
    // Manually define the prototype as cudaTypedefs.h in CUDA 12.3 does not have the cuFoo typedef
    typedef CUresult (CUDAAPI *PFN_cuFoo_v12050)(...);
    PFN_cuFoo_v12050 pfn_cuFoo = NULL;
    CUdriverProcAddressQueryResult driverStatus;
    int cudaVersion;

    // Ensure a CUDA driver >= 12.5 is installed or we will get an error from cuGetProcAddress
    status = cuDriverGetVersion(&cudaVersion);
    if (cudaVersion >= 12050) {
        // Get the address for cuFoo API using cuGetProcAddress. Specify CUDA version as
        // 12050 since cuFoo was introduced then
        CUresult status = cuGetProcAddress("cuFoo", &pfn_cuFoo, 12050, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);

        if (status == CUDA_SUCCESS && pfn_cuFoo) {
            pfn_cuFoo(...);
        }
        else {
            printf("Cannot retrieve the address to cuFoo - driverStatus = %d\n", driverStatus);
            assert(0);
        }
    }

    // rest of code here
}

在下一个示例中，我们讨论了如何在CUDA工具包的次要版本中获取API的新版本。请注意，在cuda.h标题中，将`cuDeviceGetUuid`提升到_v2的版本宏直到主要边界才完成。因此，在11.4+版本中，以下示例说明了如何获取_v2版本。

请注意，在这种情况下，原始（不是_v2版本）typedef看起来像：

typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v9020)(CUuuid *uuid, CUdevice_v1 dev);

但_v2版本typedef看起来像：

typedef CUresult (CUDAAPI *PFN_cuDeviceGetUuid_v11040)(CUuuid *uuid, CUdevice_v1 dev);

#include <cudaTypedefs.h>

CUuuid uuid;
CUdevice dev;
CUresult status;
int cudaVersion;
CUdriverProcAddressQueryResult driverStatus;

status = cuDeviceGet(&dev, 0); // Get device 0
// handle status

status = cuDriverGetVersion(&cudaVersion);
// handle status

// Ensure a CUDA driver >= 11.4 is installed or we will get an error from cuGetProcAddress
status = cuDriverGetVersion(&cudaVersion);
if (cudaVersion >= 11040) {
   PFN_cuDeviceGetUuid_v11040 pfn_cuDeviceGetUuid;
   status = cuGetProcAddress("cuDeviceGetUuid", &pfn_cuDeviceGetUuid, 11040, CU_GET_PROC_ADDRESS_DEFAULT, &driverStatus);
   if(CUDA_SUCCESS == status && pfn_cuDeviceGetUuid) {
      pfn_cuDeviceGetUuid(&uuid, dev);
   }
}

### 21.5.4.cuGetProcAddress指南[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#guidelines-for-cugetprocaddress "这个标题的永久链接")

以下是使用`cuGetProcAddress`需要记住的指南。

- 编码传递给`cuGetProcAddress`的CUDA版本，以匹配typedef版本（不要使用`CUDA_VERSION`等编译时间常量或从`cuDriverGetVersion`返回的动态版本）
    
- 在调用`cuGetProcAddress`之前，请检查当前驱动程序版本（例如来自`cuDriverGetVersion`）是否足够，否则可能会出现错误或返回意外符号
    

#### 21.5.4.1.运行时API使用指南[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#guidelines-for-runtime-api-usage "这个标题的永久链接")

除非另有说明，否则CUDA运行时API `cudaGetDriverEntryPointByVersion`将具有与驱动程序条目pointcuGet`cuGetProcAddress`类似的指南，因为它允许用户请求特定的CUDA驱动程序版本。

### 21.5.5.确定cuGetProcAddress失败的原因[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#determining-cugetprocaddress-failure-reasons "这个标题的永久链接")

cuGetProcAddress有两种错误。这些是（1）API/使用错误和（2）无法找到请求的驱动程序API。第一个错误类型将通过CUresult返回值从API返回错误代码。例如将NULL传递为`pfn`变量或传递无效`flags`。

The second error type encodes in the `CUdriverProcAddressQueryResult *symbolStatus` and can be used to help distinguish potential issues with the driver not being able to find the symbol requested. Take the following example:

// cuDeviceGetExecAffinitySupport was introduced in release CUDA 11.4
#include <cuda.h>
CUdriverProcAddressQueryResult driverStatus;
cudaVersion = ...;
status = cuGetProcAddress("cuDeviceGetExecAffinitySupport", &pfn, cudaVersion, 0, &driverStatus);
if (CUDA_SUCCESS == status) {
    if (CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT == driverStatus) {
        printf("We can use the new feature when you upgrade cudaVersion to 11.4, but CUDA driver is good to go!\n");
        // Indicating cudaVersion was < 11.4 but run against a CUDA driver >= 11.4
    }
    else if (CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND == driverStatus) {
        printf("Please update both CUDA driver and cudaVersion to at least 11.4 to use the new feature!\n");
        // Indicating driver is < 11.4 since string not found, doesn't matter what cudaVersion was
    }
    else if (CU_GET_PROC_ADDRESS_SUCCESS == driverStatus && pfn) {
        printf("You're using cudaVersion and CUDA driver >= 11.4, using new feature!\n");
        pfn();
    }
}

The first case with the return code `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT` indicates that the `symbol` was found when searching in the CUDA driver but it was added later than the `cudaVersion` supplied. In the example, specifying `cudaVersion` as anything 11030 or less and when running against a CUDA driver >= CUDA 11.4 would give this result of `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`. This is because `cuDeviceGetExecAffinitySupport` was added in CUDA 11.4 (11040).

返回代码为`CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`的第二个情况表明在CUDA驱动程序中搜索时没有找到该`symbol`。这可能是由于一些原因，例如由于驱动程序较旧，无法支持CUDA功能，以及只是有错别字。在后者中，与上一个示例类似，如果用户将`symbol`作为CUDeviceGetExecAffinitySupport-注意大写字母CU以开始字符串-`cuGetProcAddress`将无法找到API，因为字符串不匹配。在前一种情况下，一个例子可能是用户针对支持新API的CUDA驱动程序开发应用程序，以及针对旧的CUDA驱动程序部署应用程序。使用最后一个例子，如果开发人员针对CUDA 11.4或更高版本进行开发，但针对CUDA 11.3驱动程序进行了部署，在开发过程中，他们可能有一个成功的`cuGetProcAddress`，但当部署针对CUDA 11.3驱动程序运行的应用程序时，该调用将不再与`driverStatus`中返回的`CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND`一起使用。

# 22.CUDA环境变量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-environment-variables "这个标题的永久链接")

下表列出了CUDA环境变量。与多过程服务相关的环境变量记录在GPU部署和管理指南的多过程服务部分。

表30 CUDA环境变量[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id486 "此表的永久链接")
|变量|价值取向|描述|
|---|---|---|
|**设备枚举和属性**|||
|CUDA_可见_设备|逗号分隔的GPU标识符MIG支持的序列：`MIG-<GPU-UUID>/<GPU instanceID>/<compute instanceID>`|GPU标识符以整数索引或UUID字符串的形式给出。GPU UUID字符串应遵循与_nvidia-smi_给出的相同格式，例如GPU-8932f937-d72c-4106-c12f-20bd9faed9f6。然而，为了方便起见，允许使用缩写形式；只需从GPU UUID开始指定足够的数字，即可唯一识别目标系统中的GPU。例如，CUDA_VISIBLE_DEVICES=GPU-8932f937可能是引用上述GPU UUID的有效方法，假设系统中没有其他GPU共享此前缀。只有序列中存在索引的设备对CUDA应用程序可见，并且它们按序列顺序枚举。如果其中一個索引無效，只有索引在無效索引之前的裝置才能在CUDA應用程式中看到。例如，将CUDA_VISIBLE_DEVICES设置为2,1会导致设备0不可见，设备2在设备1之前被枚举。将CUDA_VISIBLE_DEVICES设置为0,2,-1,1会导致设备0和2可见，设备1不可见。MIG格式以MIG关键字开头，GPU UUID应遵循与_nvidia-smi_给出的相同格式。例如，MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2。仅支持单个MIG实例枚举。|
|CUDA_MANAGED_FORCE_DEVICE_ALLOC|0或1（默认为0）|强制驱动程序将所有托管分配放在设备内存中。|
|CUDA_DEVICE_订单|FASTEST_FIRST，PCI_BUS_ID，（默认为FASTEST_FIRST）|FASTEST_FIRST使CUDA使用简单的启发式方法按最快到最慢的顺序列举可用的设备。PCI_BUS_ID按PCI总线ID按升序排序设备。|
|**编辑**|||
|CUDA_缓存_禁用|0或1（默认为0）|禁用缓存（设置为1时）或启用缓存（设置为0时）以及时编译。禁用时，不会将二进制代码添加到缓存中或从缓存中检索到。|
|CUDA_缓存_路径|文件路径|指定及时编译器缓存二进制代码的文件夹；默认值为：<br><br>- 在Windows上，`%APPDATA%\NVIDIA\ComputeCache`<br>    <br>- 在Linux上，`~/.nv/ComputeCache`|
|CUDA_缓存_最大尺寸|整数（桌面/服务器平台的默认值为1073741824（1 GiB），嵌入式平台的默认值为268435456（256 MiB），最大值为4294967296（4 GiB））|指定及时编译器使用的缓存的大小（以字节为单位）。大小超过缓存大小的二进制代码不缓存。旧的二进制代码会从缓存中驱逐，以便在需要时为较新的二进制代码腾出空间。|
|CUDA_FORCE_PTX_JIT|0或1（默认为0）|当设置为1时，强制设备驱动程序忽略应用程序中嵌入的任何二进制代码（请参阅[应用程序兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility)），而是及时编译嵌入式PTX代码。如果内核没有嵌入式PTX代码，它将无法加载。此环境变量可用于验证PTX代码是否嵌入到应用程序中，以及其即时编译是否按预期工作，以保证应用程序与未来架构的正向兼容性（请参阅[及时编译](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#just-in-time-compilation)）。|
|CUDA_禁用_PTX_JIT|0或1（默认为0）|当设置为1时，禁用嵌入式PTX代码的及时编译，并使用嵌入应用程序中的兼容二进制代码（请参阅[应用程序兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility)）。如果内核没有嵌入二进制代码，或者嵌入式二进制文件是为不兼容的架构编译的，那么它将无法加载。此环境变量可用于验证应用程序是否为每个内核生成的兼容_SASS_代码。（请参阅[二进制兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#binary-compatibility)）。|
|CUDA_FORCE_JIT|0或1（默认为0）|当设置为1时，强制设备驱动程序忽略应用程序中嵌入的任何二进制代码（请参阅[应用程序兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility)），而是及时编译嵌入式PTX代码。如果内核没有嵌入式PTX代码，它将无法加载。此环境变量可用于验证PTX代码是否嵌入到应用程序中，以及其即时编译是否按预期工作，以保证应用程序与未来架构的正向兼容性（请参阅[及时编译](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#just-in-time-compilation)）。可以通过设置`CUDA_FORCE_PTX_JIT=0`来覆盖嵌入式PTX的行为。|
|CUDA_禁用_JIT|0或1（默认为0）|当设置为1时，禁用嵌入式PTX代码的及时编译，并使用嵌入应用程序中的兼容二进制代码（请参阅[应用程序兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility)）。如果内核没有嵌入二进制代码，或者嵌入式二进制文件是为不兼容的架构编译的，那么它将无法加载。此环境变量可用于验证应用程序是否为每个内核生成的兼容SASS代码。（请参阅[二进制兼容性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#binary-compatibility)）。可以通过设置`CUDA_DISABLE_PTX_JIT=0`来覆盖嵌入式PTX的行为。|
|**执行**|||
|CUDA_LANUNCH_BLOCKING|0或1（默认为0）|禁用（设置为1）或启用（设置为0时）异步内核启动。|
|CUDA_DEVICE_最大_连接|1到32（默认为8）|设置从主机到计算能力3.5及以上的每个设备的计算和复制引擎并发连接（工作队列）的数量。|
|CUDA_DEVICE_MAX_COPY_连接|1到32（默认为8）|设置从主机到计算能力8.0及以上的每个设备的异步复制引擎的复制引擎并发连接（工作队列）的数量。当CUDA_DEVICE_MAX_CONNECTIONS和CUDA_DEVICE_MAX_COPY_CONNECTIONS同时设置时，只有CUDA_DEVICE_MAX_CONNECTIONS设置的复制连接数量将被覆盖。|
|CUDA_自动_提升|0或1|覆盖nvidia-smi的-auto-boost-default选项设置的自动提升行为。如果应用程序通过此环境变量请求与nvidia-smi不同的行为，如果目前没有在同一GPU上运行的其他应用程序成功请求不同行为，则该请求将受到尊重，否则将忽略。|
|CUDA_规模_LAUNCH_QUEUES|“0.25x”、“0.5x”、“2x”或“4x”|按固定乘数缩放可用于启动工作的队列大小。|
|**cuda-gdb（在Linux平台上）**|||
|CUDA_DEVICE_等待_例外|0或1（默认为0）|当设置为1时，CUDA应用程序将在发生设备异常时停止，允许附加调试器以进行进一步调试。|
|**MPS服务（在Linux平台上）**|||
|CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT|百分比值（在0-100之间，默认值为0）|Devices of compute capability 8.x allow, a portion of L2 cache to be set-aside for persisting data accesses to global memory. When using CUDA MPS service, the set-aside size can only be controlled using this environment variable, before starting CUDA MPS control daemon. I.e., the environment variable should be set before running the command `nvidia-cuda-mps-control -d`.|
|**模块加载**|||
|CUDA_模块_加载|默认、懒惰、渴望（默认为懒惰）|Specifies the module loading mode for the application. When set to EAGER, all kernels and data from a cubin, fatbin or a PTX file are fully loaded upon corresponding `cuModuleLoad*` and `cuLibraryLoad*` API call. When set to LAZY, loading of specific kernels is delayed to the point a CUfunc handle is extracted with `cuModuleGetFunction`or `cuKernelGetFunction` API calls and data from the cubin is loaded at load of first kernel in the cubin or at first access of variables in the cubin. Default behavior may change in future CUDA releases.|
|CUDA_模块_数据_加载|默认、懒惰、渴望（默认为懒惰）|指定应用程序的数据加载模式。当设置为EAGER时，来自cubin、fatbin或PTX文件的所有数据都会在相应的`cuLibraryLoad*`上完全加载到内存中。这不会影响内核的LAZY或EAGER加载。设置为LAZY时，数据加载会延迟到需要句柄的点。默认行为可能会在未来的CUDA版本中发生变化。如果未设置此环境变量，数据加载行为将从`CUDA_MODULE_LOADING`继承。|
|**预加载依赖库**|||
|CUDA_FORCE_PRELOAD_LIBRARIES|0或1（默认为0）|当设置为1时，强制驱动程序在驱动程序初始化期间预加载NVVM和PTX及时编译所需的库。这将增加内存占用和CUDA驱动程序初始化所花时间。需要设置此环境变量，以避免某些涉及多个CUDA线程的死锁情况。|
|**CUDA图表**|||
|CUDA_GRAPHS_USE_NODE_优先权|0或1|覆盖图形实例化上的cudaGraphInstantiateFlagUseNodePriority标志。当设置为1时，将为所有图形设置标志，当设置为0时，所有图形的标志将被清除。|
|**CUDA错误日志管理**|||
|CUDA_LOG_文件|stdout、stderr或有效文件路径|提供打印错误日志的位置。有关更多详细信息，请参阅“`ErrorLogManagement`”部分。|

# 23.错误日志管理[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#error-log-management "这个标题的永久链接")

_错误日志管理_机制允许以描述问题原因的纯英语格式向开发人员报告CUDA API错误。

## 23.1.背景[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id446 "这个标题的永久链接")

传统上，CUDA API调用失败的唯一迹象是返回非零代码。截至CUDA工具包12.9，CUDA运行时为错误条件定义了100多个不同的返回代码，但其中许多是通用的，在调试原因时没有为开发人员提供帮助。

## 23.2.激活[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#activation "这个标题的永久链接")

设置_CUDA_LOG_FILE_环境变量。可接受的值是_stdout_、_stderr_或系统上写入文件的有效路径。即使在程序执行前没有设置_CUDA_LOG_FILE_，也可以通过API转储日志缓冲区。注意：无差错的执行可能不会打印任何日志。

## 23.3.输出[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#output "这个标题的永久链接")

日志以以下格式输出：

[Time][TID][Source][Severity][API Entry Point] Message

以下一行是当开发人员尝试将错误日志管理日志转储到未分配的缓冲区时生成的实际错误消息：

[22:21:32.099][25642][CUDA][E][cuLogsDumpToMemory] buffer cannot be NULL

之前，开发人员在返回代码中得到的只是_CUDA_ERROR_INVALID_VALUE_，如果调用_cuGetErrorString_，可能是“无效参数”。

## 23.4.API描述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id447 "这个标题的永久链接")

CUDA驱动程序提供两类API，用于与错误日志管理功能进行交互。

此功能允许开发人员注册回调函数，以便在生成错误日志时使用，其中回调签名为：

void callbackFunc(void *data, CUlogLevel logLevel, char *message, size_t length)

回调在此API中注册：

CUresult cuLogsRegisterCallback(CUlogsCallback callbackFunc, void *userData, CUlogsCallbackHandle *callback_out)

_用户数据_在不修改的情况下传递到回调函数。_调用者_应存储_callback_out_，以便使用incuLogsUnregisterCallback。

CUresult cuLogsUnregisterCallback(CUlogsCallbackHandle callback)

另一组API函数用于管理日志的输出。一个重要的概念是日志迭代器，它指向缓冲区的当前端：

CUresult cuLogsCurrent(CUlogIterator *iterator_out, unsigned int flags)

在不需要转储整个日志缓冲区的情况下，调用软件可以保留迭代器位置。目前，标志参数必须为0，为未来的CUDA版本保留了额外的选项。

随时都可以使用以下功能将错误日志缓冲区转储到文件或内存中：

CUresult cuLogsDumpToFile(CUlogIterator *iterator, const char *pathToFile, unsigned int flags)
CUresult cuLogsDumpToMemory(CUlogIterator *iterator, char *buffer, size_t *size, unsigned int flags)

如果_迭代器_为空，整个缓冲区将被转储，最多100个条目。如果_迭代器_不是空值，日志将从该条目开始转储，_迭代器_的值将更新到日志的当前末尾，就像_cuLogsCurrent_被调用一样。如果缓冲区中有超过100个日志条目，将在转储开始时添加注释，说明这一点。

标志参数必须为0，为未来的CUDA版本保留了其他选项。

_cuLogsDumpToMemory_函数还有其他注意事项：

1. 缓冲区本身将以空终止，但每个单独的日志条目将仅用一个換行符（n）字符分隔。
    
2. 缓冲区的最大大小为25600字节。
    
3. 如果_大小_中提供的值不足以存储所有所需的日志，则将添加一个注释作为第一个条目，并且不合适的最古老的条目将不会被转储。
    
4. 返回后，_大小_将包含写入提供缓冲区的实际字节数。
    

## 23.5.限制和已知问题[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#limitations-and-known-issues "这个标题的永久链接")

1. 日志缓冲区限制为100个条目。达到此限制后，最古老的条目将被替换，日志转储将包含一行，其中记录了滚动。
    
2. 尚未涵盖所有CUDA API。这是一个正在进行的项目，为所有API提供更好的使用错误报告。
    
3. 错误日志管理日志位置（如果给定）在/除非生成日志之前不会进行有效性测试。
    
4. 错误日志管理API目前只能通过CUDA驱动程序获得。等效的API将在未来的版本中添加到CUDA运行时。
    
5. 日志消息未本地化为任何语言，所有提供的日志均为美国英语。
    

# 24.统一内存编程[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-programming "这个标题的永久链接")

笔记

除非另有说明，否则本章适用于具有5.0或更高计算能力的设备。对于计算能力低于5.0的设备，请参考CUDA 11.8的CUDA工具包文档。

这份关于统一内存的文档分为3个部分：

- [统一内存的一般描述](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-introduction)
    
- [完全支持CUDA统一内存的设备上的统一内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-pageable-systems)
    
- [没有完全CUDA统一内存支持的设备上的统一内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-no-pageable-systems)
    

## 24.1.统一内存介绍[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-introduction "这个标题的永久链接")

CUDA统一内存为所有处理器提供：

- 单个_统一_内存池，即单个指针值使系统中的所有处理器（所有CPU、所有GPU等）能够通过其所有原生内存操作（指针去引用、原子等）访问此内存。
    
- 从系统中的所有处理器同时访问统一内存池。
    

统一内存通过多种方式改进了GPU编程：

- **生产力**：GPU程序可以同时从GPU和CPU线程访问统一内存，而无需创建单独的分配（`cudaMalloc()`和手动来回复制内存（`cudaMemcpy*()`）。
    
- **表现**：
    
    - 通过将数据迁移到最频繁访问数据的处理器，可以最大限度地提高数据访问速度。应用程序可以触发数据的手动迁移，并可能使用提示来控制迁移启发式方法。
        
    - 通过避免在CPU和GPU上重复内存，可以减少系统内存的总使用。
        
- **功能**：它使GPU程序能够处理超出GPU内存容量的数据。
    

使用CUDA统一内存，数据移动仍然会发生，提示可能会提高性能。正确性或功能不需要这些提示，即程序员可能会首先专注于跨GPU和CPU并行应用程序，并在开发周期的后期担心数据移动作为性能优化。请注意，数据的物理位置对程序是看不见的，可以随时更改，但无论位置如何，对数据虚拟地址的访问都将保持有效和一致。

获取CUDA统一内存主要有两种方法：

- [系统分配内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-implicit-allocation)：使用系统API分配到主机上的内存：堆栈变量、全局/文件范围变量、`malloc()`/`mmap()`（请参阅[系统分配内存：深入示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-system-allocator)的深入示例）、线程本地等。
    
- [明确分配统一内存的CUDA API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-explicit-allocation)：使用`cudaMallocManaged()`分配的内存，例如，可以在更多系统上使用，并且可能比系统分配的内存表现更好。
    

### 24.1.1.统一内存的系统要求[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#system-requirements-for-unified-memory "这个标题的永久链接")

下表显示了对CUDA统一内存的不同支持级别、检测这些支持级别所需的设备属性以及特定于每个支持级别的文档链接：

表31统一内存支持级别概述[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-unified-memory-levels "此表的永久链接")
|统一内存支持级别|系统设备属性|进一步的文件|
|---|---|---|
|完整的CUDA统一内存：所有内存都有完整的支持。这包括系统分配和CUDA管理内存。|设置为1：`pageableMemoryAccess`<br><br>[具有硬件加速的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-system-allocator)还具有以下属性设置为1：<br><br>`hostNativeAtomicSupported`，`pageableMemoryAccessUsesHostPageTables`，`directManagedMemAccessFromHost`|[完全支持CUDA统一内存的设备上的统一内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-pageable-systems)|
|只有CUDA托管内存有全面支持。|设置为1：`concurrentManagedAccess`<br><br>设置为0：`pageableMemoryAccess`|[仅支持CUDA托管内存的设备上的统一内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-cc60)|
|没有完全支持的CUDA托管内存：统一寻址，但没有并发访问。|设置为1：`managedMemory`<br><br>设置为0：`concurrentManagedAccess`|[Windows或具有计算能力的设备上的统一内存5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-legacy-devices)<br><br>[用于Tegra内存管理的CUDA](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management)<br><br>[Tegra上的统一内存](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#effective-usage-of-unified-memory-on-tegra)|
|不支持统一内存。|设置为0：`managedMemory`|[用于Tegra内存管理的CUDA](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management)|

尝试在不支持统一内存的系统上使用统一内存的应用程序的行为是未定义的。以下属性使CUDA应用程序能够检查统一内存的系统支持级别，并在具有不同支持级别的系统之间进行移植：

- `pageableMemoryAccess`：在支持CUDA统一内存的系统上，此属性设置为1，所有线程都可以访问系统分配内存和CUDA托管内存。这些系统包括NVIDIA Grace Hopper、IBM Power9 + Volta和启用HMM的现代Linux系统（见下一个项目符号）等。
    
    - Linux HMM需要Linux内核版本6.1.24+、6.2.11+或6.3+，具有7.5或更高版本的计算能力的设备，以及安装了[开放内核模块](http://download.nvidia.com/XFree86/Linux-x86_64/515.43.04/README/kernel_open.html)的CUDA驱动程序版本535+。
        
- `concurrentManagedAccess`：在完全支持CUDA管理内存的系统上，此属性设置为1。当此属性设置为0时，CUDA管理内存中仅部分支持统一内存。有关统一内存的Tegra支持，请参阅[Tegra内存管理的CUDA](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management)。
    

程序可以使用`cudaGetDeviceProperties()`查询表[31](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#table-unified-memory-levels)中的属性来查询CUDA统一内存的GPU支持级别。

### 24.1.2.编程模型[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-opt-in "这个标题的永久链接")

有了CUDA统一内存，不再需要在主机和设备之间进行单独的分配，以及它们之间的显式内存传输。程序可以通过以下方式分配统一内存：

- [系统分配API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-implicit-allocation)：通过主机进程的任何系统分配（C的`malloc()`C++`new`算符、POSIX的`mmap`等）在[具有完全CUDA统一内存支持的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)上。
    
- [CUDA托管内存分配API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-explicit-allocation)：通过`cudaMallocManaged()`API，该API在语法上与`cudaMalloc()`相似。
    
- [CUDA管理变量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-language-integration)：用`__managed__`声明的变量，在语义上类似于`__device__`变量。
    

本章中的大多数示例至少提供了两个版本，一个使用CUDA托管内存，一个使用系统分配内存。选项卡允许您在它们之间进行选择。以下示例说明了统一内存如何简化CUDA程序：

系统（`malloc()`

|   |   |
|---|---|
|__global__ void write_value(int* ptr, int v) {<br>  *ptr = v;<br>}<br><br>int main() {<br>  int* d_ptr = nullptr;<br>  // Does not require any unified memory support<br>  cudaMalloc(&d_ptr, sizeof(int));<br>  write_value<<<1, 1>>>(d_ptr, 1);<br>  int h_value;<br>  // Copy memory back to the host and synchronize<br>  cudaMemcpy(&h_value, d_ptr, sizeof(int),<br>             cudaMemcpyDefault);<br>  printf("value = %d\n", h_value); <br>  cudaFree(d_ptr); <br>  return 0;<br>}|__global__ void write_value(int* ptr, int v) {<br>  *ptr = v;<br>}<br><br>int main() {<br>  // Requires System-Allocated Memory support<br>  int* ptr = (int*)malloc(sizeof(int));<br>  write_value<<<1, 1>>>(ptr, 1);<br>  // Synchronize required<br>  // (before, cudaMemcpy was synchronizing)<br>  cudaDeviceSynchronize();<br>  printf("value = %d\n", *ptr); <br>  free(ptr); <br>  return 0;<br>}|

系统（堆栈）管理（`cudaMallocManaged()`管理（`__managed__`）

在上述示例中，设备写入一个值，然后由主机读取：

- **如果没有统一内存**：需要编写值的主机端和设备端存储（示例中的`h_value`和`d_ptr`），以及使用`cudaMemcpy()`在两者之间提供显式副本。
    
- **使用统一内存**：设备直接从主机访问数据。`ptr`/`value`可以在没有单独的`h_value`分配的情况下使用，并且不需要复制例程，大大简化和缩小了程序的大小。和：
    
    - **系统分配**：无需其他更改。
        
    - **托管内存**：数据分配更改为使用`cudaMallocManaged()`该代码从主机和设备代码中返回一个有效的指针。
        

#### 24.1.2.1.系统分配内存的分配API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#allocation-apis-for-system-allocated-memory "这个标题的永久链接")

在[完全支持CUDA统一内存的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)上，所有内存都是统一内存。这包括分配给系统API分配的内存，如`malloc()``mmap()`C++ `new()`运算符，以及CPU线程堆栈、线程局部、全局变量等上的自动变量。

系统分配的内存可能会在第一次触摸时填充，具体取决于使用的API和系统设置。第一次触摸意味着：

- 分配API分配虚拟内存并立即返回，并且
    
- 当线程首次访问内存时，物理内存被填充。
    

通常，物理内存将被选择“靠近”线程正在运行的处理器。例如，

- GPU线程首先访问它：选择线程运行的GPU的物理GPU内存。
    
- CPU线程首先访问它：选择线程运行的CPU核心的内存NUMA节点中的物理CPU内存。
    

CUDA统一内存提示和预取API、`cudaMemAdvise`和`cudaMemPreftchAsync`可用于系统分配的内存。下面的“[数据使用提示”](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-tuning-usage)部分涵盖了这些API。

__global__ void printme(char *str) {
  printf(str);
}

int main() {
  // Allocate 100 bytes of memory, accessible to both Host and Device code
  char *s = (char*)malloc(100);
  // Physical allocation placed in CPU memory because host accesses "s" first
  strncpy(s, "Hello Unified Memory\n", 99);
  // Here we pass "s" to a kernel without explicitly copying
  printme<<< 1, 1 >>>(s);
  cudaDeviceSynchronize();
  // Free as for normal CUDA allocations
  cudaFree(s); 
  return  0;
}

#### 24.1.2.2.CUDA托管内存的分配API：`cudaMallocManaged()`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#allocation-api-for-cuda-managed-memory-cudamallocmanaged "这个标题的永久链接")

在支持CUDA托管内存的系统上，可以使用以下来分配统一内存：

__host__ cudaError_t cudaMallocManaged(void **devPtr, size_t size);

此API在语法上与`cudaMalloc()`相同：它分配托管内存`size`字节，并设置`devPtr`来引用分配。CUDA托管内存也与`cudaFree()`一起进行了分配。

在[完全支持CUDA托管内存的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)上，系统中的所有CPU和GPU都可以同时访问托管内存分配。用`cudaMallocManaged()`替换对`cudaMalloc()`的主机调用不会影响这些系统上的程序语义；设备代码无法调用`cudaMallocManaged()`

以下示例显示了`cudaMallocManaged()`的使用：

__global__ void printme(char *str) {
  printf(str);
}

int main() {
  // Allocate 100 bytes of memory, accessible to both Host and Device code
  char *s;
  cudaMallocManaged(&s, 100);
  // Note direct Host-code use of "s"
  strncpy(s, "Hello Unified Memory\n", 99);
  // Here we pass "s" to a kernel without explicitly copying
  printme<<< 1, 1 >>>(s);
  cudaDeviceSynchronize();
  // Free as for normal CUDA allocations
  cudaFree(s); 
  return  0;
}

笔记

对于支持CUDA托管内存分配但不提供全面支持的系统，请参阅[一致性和并发性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-coherency-hd)。实施细节（可能随时更改）：

- 计算能力5.x的设备在GPU上分配CUDA托管内存。
    
- 计算能力6.x及更高的设备在第一次触摸时填充内存，就像系统分配的内存API一样。
    

#### 24.1.2.3.全球范围管理变量使用`__managed__`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-scope-managed-variables-using-managed "这个标题的永久链接")

CUDA `__managed__`变量的行为就像它们通过`cudaMallocManaged()`分配一样（请参阅[CUDA托管内存的分配API：cudaMallocManaged（））](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-explicit-allocation)。它们简化了带有全局变量的程序，使主机和设备之间交换数据特别容易，而无需手动分配或复制。

在[完全支持CUDA统一内存的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)上，文件范围或全局范围变量无法通过设备代码直接访问。但指向这些变量的指针可以作为参数传递给内核，请参阅[系统分配内存：深度示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-system-allocator)。

系统分配器

__global__ void write_value(int* ptr, int v) {
  *ptr = v;
}

int main() {
  // Requires System-Allocated Memory support
  int value;
  write_value<<<1, 1>>>(&value, 1);
  // Synchronize required
  // (before, cudaMemcpy was synchronizing)
  cudaDeviceSynchronize();
  printf("value = %d\n", value);
  return 0;
}

管理的

请注意，没有明确的`cudaMemcpy()`命令，并且书面值在CPU和GPU上都可见。

CUDA `__managed__` variable implies `__device__` and is equivalent to `__managed__ __device__`, which is also allowed. Variables marked `__constant__`may not be marked as `__managed__`.

有效的CUDA上下文是`__managed__`变量的正确操作所必需的。如果当前设备的上下文尚未创建，访问`__managed__`变量可以触发CUDA上下文创建。在上述示例中，在内核启动前访问`value`触发默认设备上的上下文创建。在没有该访问的情况下，内核启动会触发上下文创建。

声明为`__managed__`C++对象受制于某些特定的约束，特别是在涉及静态初始化器时。有关这些约束的列表，请参阅[C++语言支持](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-cplusplus-language-support)。

笔记

对于[没有完全支持的CUDA托管内存的设备](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)，在“[管理数据可见性和并发CPU + GPU访问与流的](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-managing-data)部分中讨论了在CUDA流中执行异步操作的`__managed__`变量的可见性。

#### 24.1.2.4.统一内存和映射内存的区别[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#difference-between-unified-memory-and-mapped-memory "这个标题的永久链接")

统一内存和[映射内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory)的主要区别在于，CUDA映射内存不能保证所有系统都支持各种内存访问（例如原子），而统一内存可以。CUDA映射内存保证可移植支持的有限内存操作集可用于比统一内存更多系统。

#### 24.1.2.5.指针属性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-pointer-attributes "这个标题的永久链接")

CUDA程序可以通过调用`cudaPointerGetAttributes()`和测试指针属性值是否为`cudaMemoryTypeManaged`来检查指针是否处理CUDA托管内存分配。

此API返回`cudaMemoryTypeHost`，用于在`cudaHostRegister()`注册的系统分配内存，并返回`cudaMemoryTypeUnregistered`用于CUDA不知道的系统分配内存。

指针属性不说明内存的位置，它们说明内存是如何分配或注册的。

以下示例展示了如何在运行时检测指针的类型：

char const* kind(cudaPointerAttributes a, bool pma, bool cma) {
    switch(a.type) {
    case cudaMemoryTypeHost: return pma?
      "Unified: CUDA Host or Registered Memory" :
      "Not Unified: CUDA Host or Registered Memory";
    case cudaMemoryTypeDevice: return "Not Unified: CUDA Device Memory";
    case cudaMemoryTypeManaged: return cma?
      "Unified: CUDA Managed Memory" : "Not Unified: CUDA Managed Memory";
    case cudaMemoryTypeUnregistered: return pma?
      "Unified: System-Allocated Memory" :
      "Not Unified: System-Allocated Memory";
    default: return "unknown";
    }
}

void check_pointer(int i, void* ptr) {
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, ptr);
  int pma = 0, cma = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, device);
  cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, device);
  printf("Pointer %d: memory is %s\n", i, kind(attr, pma, cma));
}

__managed__ int managed_var = 5;

int main() {
  int* ptr[5];
  ptr[0] = (int*)malloc(sizeof(int));
  cudaMallocManaged(&ptr[1], sizeof(int));
  cudaMallocHost(&ptr[2], sizeof(int));
  cudaMalloc(&ptr[3], sizeof(int));
  ptr[4] = &managed_var;

  for (int i = 0; i < 5; ++i) check_pointer(i, ptr[i]);
  
  cudaFree(ptr[3]);
  cudaFreeHost(ptr[2]);
  cudaFree(ptr[1]);
  free(ptr[0]);
  return 0;
}

#### 24.1.2.6.统一内存支持级别的运行时检测[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#runtime-detection-of-unified-memory-support-level "这个标题的永久链接")

以下示例展示了如何在运行时检测统一内存支持级别：

int main() {
  int d;
  cudaGetDevice(&d);

  int pma = 0;
  cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, d);
  printf("Full Unified Memory Support: %s\n", pma == 1? "YES" : "NO");
  
  int cma = 0;
  cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, d);
  printf("CUDA Managed Memory with full support: %s\n", cma == 1? "YES" : "NO");

  return 0;
}

#### 24.1.2.7.GPU内存超额订阅[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#gpu-memory-oversubscription "这个标题的永久链接")

统一内存使应用程序能够_超订阅_任何单个处理器的内存：换句话说，它们可以分配和共享比系统中任何单个处理器的内存容量更大的数组，除其他外，可以实现不适合单个GPU的数据集的核心外处理，而不会给编程模型增加重大复杂性。

#### 24.1.2.8.性能提示[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-hints "这个标题的永久链接")

以下章节描述了可用的统一内存性能提示，这些提示可用于所有统一内存，例如CUDA管理内存，或者在[完全支持CUDA统一内存的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)上，也可以用于所有系统分配的内存。这些API是提示，即它们不影响应用程序的语义，只影响其性能。也就是说，它们可以在任何应用程序的任何地方添加或删除，而不会影响其结果。

CUDA统一内存可能并不总是拥有做出与统一内存相关的最佳性能决策所需的所有信息。这些性能提示使应用程序能够为CUDA提供更多信息。

请注意，应用程序只有在性能提高时才应使用这些提示。

##### 24.1.2.8.1.数据预取[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-prefetching "这个标题的永久链接")

The `cudaMemPrefetchAsync` API is an asynchronous stream-ordered API that may migrate data to reside closer to the specified processor. The data may be accessed while it is being prefetched. The migration does not begin until all prior operations in the stream have completed, and completes before any subsequent operation in the stream.

cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                 size_t count,
                                 struct cudaMemLocation location,
                                 unsigned int flags,
                                 cudaStream_t stream);

A memory region containing `[devPtr, devPtr + count)` may be migrated to the destination device `location.id` if `location.type` is `cudaMemLocationTypeDevice` - or CPU if `location.type` is `cudaMemLocationTypeHost` - when the prefetch task is executed in the given `stream`. For details on `flags`, see the current [CUDA Runtime API documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).

考虑下面的简单代码示例：

系统分配器

void test_prefetch_sam(cudaStream_t s) {
  char *data = (char*)malloc(N);
  init_data(data, N);                                         // execute on CPU
  cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemPrefetchAsync(data, N, location, s, 0 /* flags */);  // prefetch to GPU
  mykernel<<<(N + TPB - 1) / TPB, TPB, 0, s>>>(data, N);      // execute on GPU
  location = {.type = cudaMemLocationTypeHost};
  cudaMemPrefetchAsync(data, N, location, s, 0 /* flags */);  // prefetch to CPU
  cudaStreamSynchronize(s);
  use_data(data, N);
  free(data);
}

管理的

##### 24.1.2.8.2.数据使用提示[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-usage-hints "这个标题的永久链接")

When multiple processors simultaneously access the same data, `cudaMemAdvise` may be used to hint how the data at `[devPtr, devPtr + count)` will be accessed:

cudaError_t cudaMemAdvise(const void *devPtr,
                          size_t count,
                          enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);

`advice`可能采取以下价值观：

- `cudaMemAdviseSetReadMostly`：这意味着数据大多会被读取，只是偶尔写入。一般来说，它允许将读取带宽换成该区域的写入带宽。示例：
    

void test_advise_managed(cudaStream_t stream) {
  char *dataPtr;
  size_t dataSize = 64 * TPB;  // 16 KiB
  // Allocate memory using cudaMallocManaged
  // (malloc may be used on systems with full CUDA Unified memory support)
  cudaMallocManaged(&dataPtr, dataSize);
  // Set the advice on the memory region
  cudaMemLocation loc = {.type = cudaMemLocationTypeDevice, .id = myGpuId};
  cudaMemAdvise(dataPtr, dataSize, cudaMemAdviseSetReadMostly, loc);
  int outerLoopIter = 0;
  while (outerLoopIter < maxOuterLoopIter) {
    // The data is written to in the outer loop on the CPU
    init_data(dataPtr, dataSize);
    // The data is made available to all GPUs by prefetching.
    // Prefetching here causes read duplication of data instead
    // of data migration
    cudaMemLocation location;
    location.type = cudaMemLocationTypeDevice;
    for (int device = 0; device < maxDevices; device++) {
      location.id = device;
      cudaMemPrefetchAsync(dataPtr, dataSize, location, 0 /* flags */, stream);
    }
    // The kernel only reads this data in the inner loop
    int innerLoopIter = 0;
    while (innerLoopIter < maxInnerLoopIter) {
      mykernel<<<32, TPB, 0, stream>>>((const char *)dataPtr, dataSize);
      innerLoopIter++;
    }
    outerLoopIter++;
  }
  cudaFree(dataPtr);
}

- `cudaMemAdviseSetPreferredLocation`：一般来说，任何内存都可以随时迁移到任何位置，例如，当给定的处理器物理内存耗尽时。此提示告诉系统，通过将数据的首选位置设置为属于设备的物理内存，将此内存区域从其首选位置迁移是不受欢迎的。传递`cudaMemLocationTypeHost`的值，用于location.type将首选位置设置为CPU内存。其他提示，如`cudaMemPrefetchAsync`，可能会覆盖此提示，导致内存从其首选位置迁移。
    

- `cudaMemAdviseSetAccessedBy`: In some systems, it may be beneficial for performance to establish a mapping into memory before accessing the data from a given processor. This hint tells the system that the data will be frequently accessed by `location.id` when `location.type` is `cudaMemLocationTypeDevice`, enabling the system to assume that creating these mappings pays off. This hint does not imply where the data should reside, but it can be combined with `cudaMemAdviseSetPreferredLocation` to specify that.
    

每个建议也可以使用以下值之一取消设置：`cudaMemAdviseUnsetReadMostly`、`cudaMemAdviseUnsetPreferredLocation`和`cudaMemAdviseUnsetAccessedBy`。

##### 24.1.2.8.3.查询托管内存上的数据使用属性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#querying-data-usage-attributes-on-managed-memory "这个标题的永久链接")

程序可以使用以下API在CUDA托管内存上查询通过`cudaMemAdvise`或`cudaMemPrefetchAsync`分配的内存范围属性：

cudaMemRangeGetAttribute(void *data,
                         size_t dataSize,
                         enum cudaMemRangeAttribute attribute,
                         const void *devPtr,
                         size_t count);

该函数以`count`字节大小查询从`devPtr`开始的内存范围的属性。内存范围必须指通过`cudaMallocManaged`分配的托管内存或通过`__managed__`变量声明的存储器。可以查询以下属性：

- `cudaMemRangeAttributeReadMostly`：如果整个内存范围都设置了`cudaMemAdviseSetReadMostly`属性，则返回的结果将为1，否则返回的结果将为0。
    
- `cudaMemRangeAttributePreferredLocation`：如果整个内存范围有相应的处理器作为首选位置，则返回的结果将是GPU设备ID或`cudaCpuDeviceId`，否则将返回`cudaInvalidDeviceId`。应用程序可以使用此查询API来决定通过CPU或GPU暂存数据，具体取决于托管指针的首选位置属性。请注意，查询时内存范围的实际位置可能与首选位置不同。
    
- `cudaMemRangeAttributeAccessedBy`：将返回为该内存范围设置建议的设备列表。
    
- `cudaMemRangeAttributeLastPrefetchLocation`：将返回使用`cudaMemPrefetchAsync`明确预取内存范围的最后一个位置。请注意，这只是返回应用程序请求预取内存范围的最后一个位置。它没有指示到该位置的预取操作是否已完成甚至已经开始。
    
- `cudaMemRangeAttributePreferredLocationType`：
    
    将返回首选位置的位置类型，如果内存范围内的所有页面都具有与首选位置相同的GPU，则为`cudaMemLocationTypeDevice`，或者如果内存范围内的所有页面都有CPU作为首选位置，则为`cudaMemLocationTypeHost`，或者如果内存范围内的所有页面都具有与其首选位置相同的主机NUMA节点ID，则为`cudaMemLocationTypeHostNuma`，或者如果所有页面没有相同的首选位置或某些页面根本没有首选位置，则为`cudaMemLocationTypeInvalid`。
    
- `cudaMemRangeAttributePreferredLocationId`：
    
    如果同一地址范围的`cudaMemRangeAttributePreferredLocationType`查询返回`cudaMemLocationTypeDevice`，它将是一个有效的设备序数，或者如果它返回`cudaMemLocationTypeHostNuma`，它将是一个有效的主机NUMA节点ID，或者如果它返回任何其他位置类型，则应忽略ID。
    
- `cudaMemRangeAttributeLastPrefetchLocationType`：
    
    将是内存范围内所有页面通过`cudaMemPrefetchAsync`明确预取的最后一个位置类型，如果内存范围内的所有页面都预取到同一GPU，则该位置类型将是`cudaMemLocationTypeDevice`，或者如果内存范围内的所有页面都预取到CPU，则将是`cudaMemLocationTypeHost`，如果内存范围内的所有页面都预取到同一主机NUMA节点ID，则将是`cudaMemLocationTypeHostNuma`，或者如果所有页面没有预取到同一位置或某些页面从未预取过，它将是`cudaMemLocationTypeInvalid`。
    
- `cudaMemRangeAttributeLastPrefetchLocationId`：
    
    如果同一地址范围的`cudaMemRangeAttributeLastPrefetchLocationType`查询返回`cudaMemLocationTypeDevice`，它将是一个有效的设备序数，或者如果它返回`cudaMemLocationTypeHostNuma`，它将是一个有效的主机NUMA节点ID，或者如果它返回任何其他位置类型，则应忽略该ID。
    

此外，可以使用相应的`cudaMemRangeGetAttributes`函数来查询多个属性。

## 24.2.完全支持CUDA统一内存的设备上的统一内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-on-devices-with-full-cuda-unified-memory-support "这个标题的永久链接")

### 24.2.1.系统分配的内存：深入的例子[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#system-allocated-memory-in-depth-examples "这个标题的永久链接")

[具有完全CUDA统一内存支持](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)的[系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)允许设备访问与设备交互的主机进程拥有的任何内存。本节展示了一些高级用例，使用一个内核，只需将输入字符数组的前8个字符打印到标准输出流：

__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}

以下选项卡显示了如何调用此内核的各种方式：

马洛克

void test_malloc() {
  const char test_string[] = "Hello World";
  char* heap_data = (char*)malloc(sizeof(test_string));
  strncpy(heap_data, test_string, sizeof(test_string));
  kernel<<<1, 1>>>("malloc", heap_data);
  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  free(heap_data);
}

管理的堆栈变量文件范围静态变量全局范围变量全局范围外部变量

上面的前三个选项卡显示了“[编程模型”部分](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-programming-model)中已经详述的示例。接下来的三个选项卡显示了从设备访问文件范围或全局范围变量的各种方式。

请注意，对于外部变量，它可以由第三方库声明，其内存由第三方库拥有和管理，该库根本不与CUDA交互。

Also note that stack variables as well as file-scope and global-scope variables can only be accessed through a pointer by the GPU. In this specific example, this is convenient because the character array is already declared as a pointer: `const char*`. However, consider the following example with a global-scope integer:

// this variable is declared at global scope
int global_variable;

__global__ void kernel_uncompilable() {
  // this causes a compilation error: global (__host__) variables must not
  // be accessed from __device__ / __global__ code
  printf("%d\n", global_variable);
}

// On systems with pageableMemoryAccess set to 1, we can access the address
// of a global variable. The below kernel takes that address as an argument
__global__ void kernel(int* global_variable_addr) {
  printf("%d\n", *global_variable_addr);
}
int main() {
  kernel<<<1, 1>>>(&global_variable);
  ...
  return 0;
}

在上述示例中，我们需要确保将指向全局变量的_指针_传递给内核，而不是直接访问内核中的全局变量。这是因为默认情况下，没有`__managed__`指定符的全局变量被声明为`__host__`only，因此目前大多数编译器不允许直接在设备代码中使用这些变量。

#### 24.2.1.1.文件备份的统一内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#file-backed-unified-memory "这个标题的永久链接")

由于[具有完全CUDA统一内存支持](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)的[系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)允许设备访问主机进程拥有的任何内存，因此它们可以直接访问文件备份内存。

在這裡，我們展示了上一節中顯示的初始示例的修改版本，以使用文件背後的記憶體，以便直接從輸入檔案中讀取GPU列印字串。在以下示例中，内存由物理文件支持，但该示例也适用于内存支持的文件，如“[使用统一内存进行过程间通信（IPC）”](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-sam-ipc)部分所述。

__global__ void kernel(const char* type, const char* data) {
  static const int n_char = 8;
  printf("%s - first %d characters: '", type, n_char);
  for (int i = 0; i < n_char; ++i) printf("%c", data[i]);
  printf("'\n");
}

void test_file_backed() {
  int fd = open(INPUT_FILE_NAME, O_RDONLY);
  ASSERT(fd >= 0, "Invalid file handle");
  struct stat file_stat;
  int status = fstat(fd, &file_stat);
  ASSERT(status >= 0, "Invalid file stats");
  char* mapped = (char*)mmap(0, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);  ASSERT(mapped != MAP_FAILED, "Cannot map file into memory");
  kernel<<<1, 1>>>("file-backed", mapped);  ASSERT(cudaDeviceSynchronize() == cudaSuccess,
    "CUDA failed with '%s'", cudaGetErrorString(cudaGetLastError()));
  ASSERT(munmap(mapped, file_stat.st_size) == 0, "Cannot unmap file");
  ASSERT(close(fd) == 0, "Cannot close file");
}

请注意，在没有`hostNativeAtomicSupported`属性的系统上，包括[启用Linux HMM的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)，不支持对文件支持内存的原子访问。

#### 24.2.1.2.具有统一内存的过程间通信（IPC）[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#inter-process-communication-ipc-with-unified-memory "这个标题的永久链接")

笔记

截至目前，将IPC与统一内存一起使用可能会对性能产生重大影响。

许多应用程序更喜欢为每个进程管理一个GPU，但仍然需要使用统一内存，例如用于超额订阅，并从多个GPU访问它。

CUDA IPC（见[进程间通信](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interprocess-communication)）不支持托管内存：此类内存的处理程序可能无法通过本节中讨论的任何机制共享。在[完全支持CUDA统一内存的系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)上，系统分配的内存具有过程间通信（IPC）功能。一旦对系统分配内存的访问与其他进程共享，相同的[编程模型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-programming-model)就会适用，类似于[文件支持的统一内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-sam-file-backed)。

有关在Linux下创建IPC系统分配内存的各种方法的更多信息，请参阅以下参考资料：

- [带有MAP_SHARED的mmap](https://man7.org/linux/man-pages/man2/mmap.2.html)
    
- [POSIX IPC API](https://pubs.opengroup.org/onlinepubs/007904875/functions/shm_open.html)
    
- [Linux memfd_创建](https://man7.org/linux/man-pages/man2/memfd_create.2.html)
    

请注意，使用此技术无法在不同主机及其设备之间共享内存。

### 24.2.2.性能调整[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-tuning "这个标题的永久链接")

为了在统一内存中实现良好的性能，重要的是：

- 了解分页如何在您的系统上工作，以及如何避免不必要的页面故障。
    
- 了解各种机制，允许您将数据保持在访问处理器的本地。
    
- 考虑调整您的应用程序，以适应系统内存传输的粒度。
    

作为一般建议，[性能提示](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-perf-hints)可能会提供更好的性能，但与默认行为相比，错误使用它们可能会降低性能。另请注意，任何提示在主机上都有与之相关的性能成本，因此有用的提示必须至少提高性能，以克服这一成本。

#### 24.2.2.1.内存分页和页面大小[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-paging-and-page-sizes "这个标题的永久链接")

许多统一内存性能调整的部分假设了关于虚拟寻址、内存页面和页面大小的先验知识。本节试图定义所有必要的术语，并解释为什么分页对性能很重要。

目前支持的统一内存的所有系统都使用虚拟地址空间：这意味着应用程序使用的内存地址表示_虚拟_位置，该位置可能_映射_到内存实际所在的物理位置。

目前支持的所有处理器，包括CPU和GPU，还使用内存_分页_。由于所有系统都使用虚拟地址空间，因此有两种类型的内存页：

- 虚拟页面：这代表了操作系统跟踪的每个进程的固定大小连续虚拟内存块，可以_映射_到物理内存中。请注意，虚拟页面与_映射_相关联：例如，单个虚拟地址可能会使用不同的页面大小映射到物理内存中。
    
- 物理頁面：這代表了處理器的主記憶體管理單元（MMU）支援的固定大小的連續記憶體塊，並且可以對映虛擬頁面。
    

目前，所有x86_64 CPU都使用4KiB物理页面。Arm CPU支持多种物理页面大小——4KiB、16KiB、32KiB和64KiB——取决于确切的CPU。最后，NVIDIA GPU支持多种物理页面大小，但更喜欢2MiB或更大的物理页面。请注意，这些尺寸可能会在未来的硬件中发生变化。

虚拟页面的默认页面大小通常对应于物理页面大小，但应用程序可以使用不同的页面大小，只要操作系统和硬件支持它们。通常，支持的虚拟页面大小必须是物理页面大小的2的幂和倍数。

跟踪虚拟页面映射到物理页面的逻辑实体将被称为_页面表_，具有给定虚拟大小的给定虚拟页面到物理页面的每个映射都称为_页面表条目（PTE）。_所有支持的处理器都为页面表提供特定的缓存，以加快虚拟地址到物理地址的转换速度。这些缓存被称为_翻译旁看缓冲区（TLB）。_

应用程序的性能调整有两个重要方面：

- 虚拟页面大小的选择，
    
- 系统是提供CPU和GPU使用的组合页面表，还是为每个CPU和GPU单独提供单独的页面表。
    

##### 24.2.2.1.1.选择正确的页面大小[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#choosing-the-right-page-size "这个标题的永久链接")

一般来说，小页面大小会导致更少的（虚拟）内存碎片，但更多的TLB错过，而更大的页面大小会导致更多的内存碎片，但更少的TLB错过。此外，与较小的页面大小相比，较大的页面迁移通常更昂贵，因为我们通常迁移整个内存页面。这可能会导致使用大页面大小的应用程序中更大的延迟峰值。有关页面故障的更多详细信息，另请参阅下一节。

性能调整的一个重要方面是，与CPU相比，TLB错过在GPU上通常要贵得多。这意味着，如果GPU线程经常访问使用足够小的页面大小映射的统一内存的随机位置，与使用足够大的页面大小映射的相同访问统一内存相比，它可能会明显变慢。虽然对使用小页面大小随机访问大面积内存的CPU线程可能会出现类似的效果，但减速不那么明显，这意味着应用程序可能希望将这种减速与更少的内存碎片进行权衡。

请注意，一般来说，应用程序不应将其性能调整为给定处理器的物理页面大小，因为物理页面大小可能会因硬件而异。上述建议仅适用于虚拟页面大小。

##### 24.2.2.1.2.CPU和GPU页面表：硬件一致性与软件一致性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cpu-and-gpu-page-tables-hardware-coherency-vs-software-coherency "这个标题的永久链接")

笔记

在性能调整文档的其余部分，我们将把CPU和GPU的组合页面表的系统称为_硬件连贯_系统。具有CPU和GPU单独页面表的系统被称为_软件一致_。

硬件连贯系统，如NVIDIA Grace Hopper，为CPU和GPU提供了一个逻辑组合的页面表。这很重要，因为为了[从GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-system-allocator)访问[系统分配的内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-system-allocator)，GPU使用CPU为请求的内存创建的任何页面表条目。如果该页面表条目使用4KiB或64KiB的默认CPU页面大小，则访问大型虚拟内存区域将导致重大的TLB错过，从而导致显著减速。

有关如何确保系统分配内存使用足够大的页面大小以避免此类问题的示例，请参阅有关配置大页面的部分。

另一方面，在CPU和GPU都有自己的逻辑页面表的系统上，应考虑不同的性能调整方面：为了[保证一致性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-introduction)，这些系统通常使用_页面故障，_以防处理器访问映射到不同处理器的物理内存中的内存地址。这样的页面故障意味着：

- 需要确保当前拥有的处理器（物理页面当前所在的处理器）无法再访问此页面，无论是删除页面表条目还是更新它。
    
- 需要确保请求访问的处理器可以通过创建新的页面表条目或更新现有条目来访问此页面，使其变得有效/活跃。
    
- 支持此虚拟页面的物理页面必须移动/迁移到请求访问的处理器：这可能是一个昂贵的操作，工作量与页面大小成正比。
    

总体而言，在CPU和GPU线程频繁并发访问同一内存页面的情况下，与软件连贯系统相比，硬件连贯系统提供了显著的性能优势：

- 更少的页面故障：这些系统不需要使用页面故障来模拟一致性或迁移内存，
    
- 更少的争议：这些系统在缓存行粒度上是一致的，而不是页面大小的粒度，也就是说，当一个缓存行内有多个处理器的争议时，只有比最小页面大小小得多的缓存行被交换，当不同的处理器访问一个页面中的不同缓存行时，就没有争执。
    

这会影响以下情景的性能：

- 从CPU和GPU同时向同一地址进行原子更新。
    
- 从CPU线程发出GPU线程的信号，反之亦然。
    

#### 24.2.2.2.来自主机的直接统一内存访问[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct-unified-memory-access-from-host "这个标题的永久链接")

一些裝置在GPU駐留統一記憶體上具有一致讀取、儲存和原子訪問的硬體支援。这些设备将属性`cudaDevAttrDirectManagedMemAccessFromHost`设置为1。请注意，所有[硬件连贯系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency)都为NVLink连接的设备设置了此属性。在这些系统上，主机可以直接访问GPU驻留内存，而无需页面故障和数据迁移（有关内存使用提示的更多详细信息[，](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-tuning-usage)请参阅[数据使用提示](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-tuning-usage)）。请注意，使用CUDA托管内存，需要带有位置类型`cudaMemLocationTypeHost`的`cudaMemAdviseSetAccessedBy`提示，才能在没有页面故障的情况下实现此直接访问。

考虑以下示例代码：

系统分配器

__global__ void write(int *ret, int a, int b) {
  ret[threadIdx.x] = a + b + threadIdx.x;
}

__global__ void append(int *ret, int a, int b) {
  ret[threadIdx.x] += a + b + threadIdx.x;
}
void test_malloc() {
  int *ret = (int*)malloc(1000 * sizeof(int));
  // for shared page table systems, the following hint is not necesary
  cudaMemLocation location = {.type = cudaMemLocationTypeHost};
  cudaMemAdvise(ret, 1000 * sizeof(int), cudaMemAdviseSetAccessedBy, location);

  write<<< 1, 1000 >>>(ret, 10, 100);            // pages populated in GPU memory
  cudaDeviceSynchronize();
  for(int i = 0; i < 1000; i++)
      printf("%d: A+B = %d\n", i, ret[i]);        // directManagedMemAccessFromHost=1: CPU accesses GPU memory directly without migrations
                                                  // directManagedMemAccessFromHost=0: CPU faults and triggers device-to-host migrations
  append<<< 1, 1000 >>>(ret, 10, 100);            // directManagedMemAccessFromHost=1: GPU accesses GPU memory without migrations
  cudaDeviceSynchronize();                        // directManagedMemAccessFromHost=0: GPU faults and triggers host-to-device migrations
  free(ret);
}

管理的

`write`内核完成后，`ret`将在GPU内存中创建并初始化。接下来，CPU将访问`ret`，然后再次使用相同的`ret`内存`append`内核。此代码将根据系统架构和硬件一致性支持显示不同的行为：

- 在具有`directManagedMemAccessFromHost=1`的系统上：CPU对托管缓冲区的访问不会触发任何迁移；数据将保留在GPU内存中，任何后续的GPU内核都可以继续直接访问它，而不会造成故障或迁移。
    
- 在具有`directManagedMemAccessFromHost=0`的系统上：CPU访问托管缓冲区将出现页面故障并启动数据迁移；任何首次尝试访问相同数据的GPU内核都会出现页面故障并将页面迁移回GPU内存。
    

#### 24.2.2.3.主机原生原子[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#host-native-atomics "这个标题的永久链接")

一些设备，包括[硬件相干系统中](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency)的NVLink连接设备，支持对CPU驻留内存的硬件加速原子访问。这意味着对主机内存的原子访问不必用页面故障进行模拟。对于这些设备，属性`cudaDevAttrHostNativeAtomicSupported`被设置为1。

#### 24.2.2.4.原子访问和同步原语[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-accesses-synchronization-primitives "这个标题的永久链接")

CUDA统一内存支持主机和设备线程可用的所有原子操作，使所有线程能够通过同时访问相同的共享内存位置进行合作。[CUDA C++标准库](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html)提供了许多异构同步原语，这些原语为主机和设备线程之间并发使用而调整，包括`cuda::atomic`、`cuda::atomic_ref`、`cuda::barrier`、`cuda::semaphore`等。

在没有[CPU和GPU页面表](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency)的系统上[：硬件一致性与软件一致性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency)，不支持从设备到文件支持的主机内存的原子访问。以下示例代码在具有[CPU和GPU页面表](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency)的系统上有效[：硬件一致性与软件一致性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency)，但在其他系统上表现出未定义的行为：

#include <cuda/atomic>

#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>

#define ERR(msg, ...) { fprintf(stderr, msg, ##__VA_ARGS__); return EXIT_FAILURE; }

__global__ void kernel(int* ptr) {
  cuda::atomic_ref{*ptr}.store(2);
}

int main() {
  // this will be closed/deleted by default on exit
  FILE* tmp_file = tmpfile64();
  // need to allcate space in the file, we do this with posix_fallocate here
  int status = posix_fallocate(fileno(tmp_file), 0, 4096);
  if (status != 0) ERR("Failed to allocate space in temp file\n");
  int* ptr = (int*)mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fileno(tmp_file), 0);
  if (ptr == MAP_FAILED) ERR("Failed to map temp file\n");

  // initialize the value in our file-backed memory
  *ptr = 1;
  printf("Atom value: %d\n", *ptr);

  // device and host thread access ptr concurrently, using cuda::atomic_ref
  kernel<<<1, 1>>>(ptr);
  while (cuda::atomic_ref{*ptr}.load() != 2);
  // this will always be 2
  printf("Atom value: %d\n", *ptr);

  return EXIT_SUCCESS;
}

On systems without [CPU and GPU page tables: hardware coherency vs. software coherency](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency), atomic accesses to unified memory may incur page faults which can lead to significant latencies. Note that this is not the case for all GPU atomics to CPU memory on these systems: operations listed by `nvidia-smi -q | grep "Atomic Caps Outbound"` may avoid page faults.

在具有[CPU和GPU页面表](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency)的系统上[：硬件一致性与软件一致性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency)，主机和设备之间的原子不需要页面故障，但仍可能因任何内存访问可能故障的其他原因而出现故障。

#### 24.2.2.5.具有统一内存的Memcpy（）/Memset（）行为[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memcpy-memset-behavior-with-unified-memory "这个标题的永久链接")

`cudaMemcpy*()`和`cudaMemset*()`接受任何统一的内存指针作为参数。

对于`cudaMemcpy*()`指定为`cudaMemcpyKind`的方向是性能提示，如果任何参数是统一的内存指针，则可能会对性能产生更高的影响。

因此，建议遵循以下绩效建议：

- 当已知统一内存的物理位置时，请使用准确的`cudaMemcpyKind`提示。
    
- 比起不准确的`cudaMemcpyKind`提示，更喜欢`cudaMemcpyDefault`。
    
- 始终使用填充（初始化）缓冲区：避免使用这些API来初始化内存。
    
- 如果两个指针都指向系统分配的内存，请避免使用`cudaMemcpy*()`）：启动内核或使用CPU内存复制算法，如asstd`std::memcpy`。
    

## 24.3.没有完全CUDA统一内存支持的设备上的统一内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-on-devices-without-full-cuda-unified-memory-support "这个标题的永久链接")

### 24.3.1.仅支持CUDA托管内存的设备上的统一内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-on-devices-with-only-cuda-managed-memory-support "这个标题的永久链接")

对于具有6.x或更高计算能力但没有[可分页内存访问的](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-requirements)设备，CUDA托管内存完全支持且一致。统一内存的编程模型和性能调整与[完全支持CUDA统一内存的设备上的统一内存中](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-pageable-systems)描述的模型基本相似，但值得注意的例外是，系统分配器不能用于分配内存。因此，以下子部分列表不适用：

- [系统分配的内存：深入的例子](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-system-allocator)
    
- [硬件/软件一致性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-hw-coherency)
    

### 24.3.2.Windows或具有计算能力的设备上的统一内存5.x[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-on-windows-or-devices-with-compute-capability-5-x "这个标题的永久链接")

计算能力低于6.0或Windows平台的设备支持CUDA托管内存v1.0，对数据迁移和一致性以及内存超额订阅的支持有限。以下小节更详细地描述了如何在这些平台上使用和优化托管内存。

#### 24.3.2.1.数据迁移和一致性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-migration-and-coherency "这个标题的永久链接")

计算能力低于6.0的GPU架构不支持按需将托管数据精细地移动到GPU。每当启动GPU内核时，通常必须将所有托管内存传输到GPU内存，以避免内存访问时出现故障。借助计算功能6.x，引入了新的GPU页面故障机制，提供更无缝的统一内存功能。结合全系统虚拟地址空间，页面故障提供了几个好处。首先，页面故障意味着CUDA系统软件不需要在每次内核启动之前将所有托管内存分配同步到GPU。如果在GPU上运行的内核访问了不在其内存中的页面，它就会出现故障，允许该页面按需自动迁移到GPU内存。或者，该页面可以映射到GPU地址空间，以便通过PCIe或NVLink互连进行访问（访问映射有时可能比迁移更快）。请注意，统一内存是全系统范围的：GPU（和CPU）可以从CPU内存或系统中其他GPU的内存中故障和迁移内存页面。

#### 24.3.2.2.GPU内存超额订阅[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-legacy-oversubscription "这个标题的永久链接")

计算能力低于6.0的设备不能分配比GPU内存的物理大小更多的托管内存。

#### 24.3.2.3.多图形处理器[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-gpu "这个标题的永久链接")

在具有低于6.0的计算能力的设备的系统上，通过GPU的点对点功能，系统中的所有GPU都会自动看到托管分配。托管内存分配的行为类似于使用`cudaMalloc()`分配的非托管内存：当前活动设备是物理分配的家，但系统中的其他GPU将通过PCIe总线以减少的带宽访问内存。

在Linux上，只要程序积极使用的所有GPU都有点对点支持，托管内存就会分配到GPU内存中。如果应用程序在任何时候开始使用没有点对点支持的GPU，并且具有管理分配的任何其他GPU，那么驱动程序将将所有托管分配迁移到系统内存。在这种情况下，所有GPU都受到PCIe带宽限制。

在Windows上，如果对等映射不可用（例如，在不同架构的GPU之间），那么系统将自动恢复使用零复制内存，无论两个GPU是否实际被一个程序使用。如果实际上只使用一个GPU，则有必要在启动程序之前设置`CUDA_VISIBLE_DEVICES`环境变量。这限制了哪些GPU是可见的，并允许在GPU内存中分配托管内存。

或者，在Windows上，用户还可以将`CUDA_MANAGED_FORCE_DEVICE_ALLOC`设置为非零值，以强制驱动程序始终使用设备内存进行物理存储。当此环境变量设置为非零值时，该过程中使用的所有支持托管内存的设备必须彼此对点兼容。如果使用支持托管内存的设备，并且该设备与之前在该过程中使用的任何其他托管内存支持设备不兼容，则将返回错误`::cudaErrorInvalidDevice`即使已在这些设备上调用`::cudaDeviceReset`。[CUDA環境變數](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#env-vars)中描述了這些環境變數。请注意，从CUDA 8.0开始，`CUDA_MANAGED_FORCE_DEVICE_ALLOC`对Linux操作系统没有影响。

#### 24.3.2.4.一致性和并发性[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#coherency-and-concurrency "这个标题的永久链接")

不可能在计算能力低于6.0的设备上同时访问托管内存，因为如果CPU在GPU内核处于活动状态时访问统一内存分配，则无法保证一致性。

##### 24.3.2.4.1.GPU独家访问托管内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#gpu-exclusive-access-to-managed-memory "这个标题的永久链接")

为了确保pre-6.x GPU架构的一致性，统一内存编程模型对数据访问施加了限制，而CPU和GPU同时执行。实际上，无论特定内核是否在积极使用数据，GPU在执行任何内核操作时都可以独家访问所有托管数据。当与`cudaMemcpy*()`或`cudaMemset*()`一起使用托管数据时，系统可以选择从主机或设备访问源或目标，这将在`cudaMemcpy*()`或`cudaMemset*()`执行时对该数据的并发CPU访问施加限制。有关更多详细信息，请参阅[Memcpy（）/Memset（）使用统一内存的行为](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-memcpy-memset)。

对于`concurrentManagedAccess`属性设置为0的设备，当GPU处于活动状态时，CPU不允许访问任何托管分配或变量。在這些系統上，併發的CPU/GPU訪問，即使是不同的託管記憶體分配，也會導致分割故障，因為CPU被認為無法訪問該頁面。

__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    y = 20;            // Error on GPUs not supporting concurrent access

    cudaDeviceSynchronize();
    return  0;
}

在上述示例中，当CPU接触`y`，GPU程序`kernel`仍然处于活动状态。（注意它在`cudaDeviceSynchronize()`之前是如何发生的。）由于GPU页面故障功能，该代码在具有计算能力6.x的设备上成功运行，该功能解除了对同时访问的所有限制。然而，即使CPU访问的数据与GPU不同，这种内存访问在pre-6.x架构上也是无效的。在访问`y`之前，程序必须明确地与GPU同步：

__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    kernel<<< 1, 1 >>>();
    cudaDeviceSynchronize();
    y = 20;            //  Success on GPUs not supporing concurrent access
    return  0;
}

如本例所示，在具有pre-6.x GPU架构的系统上，无论GPU内核是否实际接触过相同的数据（或任何托管数据），CPU线程在执行内核启动和随后的同步调用之间可能无法访问任何托管数据。仅仅是CPU和GPU并发访问的潜力就足以引发进程级异常。

请注意，如果在GPU处于活动状态时使用`cudaMallocManaged()`或`cuMemAllocManaged()`动态分配内存，则在启动其他工作或GPU同步之前，内存的行为是未指定的。在此期间尝试访问CPU上的内存可能会导致也可能不会导致分割故障。这不适用于使用标志`cudaMemAttachHost`或`CU_MEM_ATTACH_HOST`分配的内存。

##### 24.3.2.4.2.显式同步和逻辑GPU活动[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#explicit-synchronization-and-logical-gpu-activity "这个标题的永久链接")

请注意，在上述示例中，即使`kernel`快速运行并在CPU接触y之前完成，也需要显式同步。统一内存使用逻辑活动来确定GPU是否处于空闲状态。这与CUDA编程模型一致，该模型指定内核可以在启动后随时运行，在主机发出同步调用之前不能保证完成。

任何在逻辑上保证GPU完成工作的函数调用都是有效的。这包括`cudaDeviceSynchronize()``cudaStreamSynchronize()`和`cudaStreamQuery()`（前提是它返回`cudaSuccess`而不是`cudaErrorNotReady`），其中指定的流是唯一仍在GPU上执行的流；在指定事件没有遵循任何设备工作的情况下，`cudaEventSynchronize()`和`cudaEventQuery()`）；以及`cudaMemcpy()`和`cudaMemset()`的使用，这些文件被记录为相对于主机完全同步。

将遵循流之间创建的依赖关系，通过同步流或事件来推断其他流的完成。依赖项可以通过`cudaStreamWaitEvent()`创建，也可以在使用默认（NULL）流时隐式创建。

CPU从流回调内访问托管数据是合法的，前提是GPU上没有其他可能访问托管数据的流处于活动状态。此外，不跟随任何设备工作的回调可用于同步：例如，通过从回调内部发出条件变量的信号；否则，CPU访问仅在回调期间有效。

有几个重要的注意事项：

- 當GPU處於活動狀態時，始終允許CPU訪問非託管的零複製資料。
    
- 当GPU运行任何内核时，即使该内核不使用托管数据，它也被视为处于活动状态。如果内核可能使用数据，则禁止访问，除非设备属性`concurrentManagedAccess`为1。
    
- 除了适用于非托管内存的多GPU访问之外，对托管内存的并发GPU间访问没有限制。
    
- 访问托管数据的并发GPU内核没有限制。
    

请注意最后一点如何允许GPU内核之间的竞争，就像目前非托管GPU内存的情况一样。如前所述，从GPU的角度来看，托管内存的功能与非托管内存相同。以下代码示例说明了这些要点：

int main() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    int *non_managed, *managed, *also_managed;
    cudaMallocHost(&non_managed, 4);    // Non-managed, CPU-accessible memory
    cudaMallocManaged(&managed, 4);
    cudaMallocManaged(&also_managed, 4);
    // Point 1: CPU can access non-managed data.
    kernel<<< 1, 1, 0, stream1 >>>(managed);
    *non_managed = 1;
    // Point 2: CPU cannot access any managed data while GPU is busy,
    //          unless concurrentManagedAccess = 1
    // Note we have not yet synchronized, so "kernel" is still active.
    *also_managed = 2;      // Will issue segmentation fault
    // Point 3: Concurrent GPU kernels can access the same data.
    kernel<<< 1, 1, 0, stream2 >>>(managed);
    // Point 4: Multi-GPU concurrent access is also permitted.
    cudaSetDevice(1);
    kernel<<< 1, 1 >>>(managed);
    return  0;
}

##### 24.3.2.4.3.使用流管理数据可见性和并发CPU + GPU访问[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#managing-data-visibility-and-concurrent-cpu-gpu-access-with-streams "这个标题的永久链接")

到目前为止，人们认为，对于6.x之前的SM架构：1）任何活动内核都可以使用任何托管内存，2）在内核处于活动状态时使用CPU的托管内存是无效的。在这里，我们提出了一个更精细的托管内存控制系统，该系统设计用于支持托管内存的所有设备，包括具有`concurrentManagedAccess`等于0的旧架构。

CUDA编程模型提供流作为程序指示内核启动之间的依赖性和独立性的机制。启动到同一流中的内核保证连续执行，而启动到不同流中的内核允许同时执行。流描述了工作项目之间的独立性，因此通过并发性可以提高潜在的效率。

统一内存基于流独立模型，允许CUDA程序显式将托管分配与CUDA流相关联。通过这种方式，程序员根据内核是否被启动到指定流中来指示数据的使用。这为基于程序特定数据访问模式的并发机会提供了机会。控制这种行为的功能是：

cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream,
                                     void *ptr,
                                     size_t length=0,
                                     unsigned int flags=0);

`cudaStreamAttachMemAsync()`函数将从`ptr`开始的内存`length`字节与指定的`stream`关联。（目前，`length`必须始终为0，以表明整个区域应该被连接。）由于这种关联，只要`stream`的所有操作都已完成，统一内存系统就允许CPU访问此内存区域，无论其他流是否处于活动状态。实际上，这限制了活动GPU对托管内存区域的独家所有权，以限制每个流活动，而不是整个GPU活动。

最重要的是，如果分配与特定流无关，则所有运行的内核都可见，无论其流如何。这是`cudaMallocManaged()`分配或`__managed__`变量的默认可见性；因此，简单案例规则是，在任何内核运行时，CPU不得接触数据。

通过将分配与特定流相关联，该程序保证只有启动到该流的内核才会接触该数据。统一内存系统不进行错误检查：程序员有责任确保保证得到尊重。

除了允许更大的并发性外，使用`cudaStreamAttachMemAsync()`可以（通常确实）在统一内存系统中实现数据传输优化，这可能会影响延迟和其他开销。

##### 24.3.2.4.4.流关联示例[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-association-examples "这个标题的永久链接")

将数据与流关联可以对CPU + GPU并发进行精细控制，但在使用计算能力低于6.0的设备时，必须记住哪些数据对哪些数据可见。查看之前的同步示例：

__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &y, 0, cudaMemAttachHost);
    cudaDeviceSynchronize();          // Wait for Host attachment to occur.
    kernel<<< 1, 1, 0, stream1 >>>(); // Note: Launches into stream1.
    y = 20;                           // Success – a kernel is running but “y”
                                      // has been associated with no stream.
    return  0;
}

在这里，我们明确地将`y`与主机可访问性相关联，从而始终能够从CPU进行访问。（和以前一样，请注意在访问之前没有`cudaDeviceSynchronize()`）。）运行的GPU`kernel`访问`y`现在将产生未定义的结果。

请注意，将变量与流关联不会改变任何其他变量的关联。例如，将`x`与`stream1`关联并不能确保在`stream1`中启动的内核只访问`x`，因此此代码会导致错误：

__device__ __managed__ int x, y=2;
__global__  void  kernel() {
    x = 10;
}
int main() {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    cudaStreamAttachMemAsync(stream1, &x);// Associate “x” with stream1.
    cudaDeviceSynchronize();              // Wait for “x” attachment to occur.
    kernel<<< 1, 1, 0, stream1 >>>();     // Note: Launches into stream1.
    y = 20;                               // ERROR: “y” is still associated globally
                                          // with all streams by default
    return  0;
}

请注意，访问`y`如何导致错误，因为即使`x`已与流相关联，我们也没有告诉系统谁可以看到`y`。因此，系统保守地假设`kernel`可能会访问它，并阻止CPU这样做。

##### 24.3.2.4.5.使用多執行緒主機程式進行流附加[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-attach-with-multithreaded-host-programs "这个标题的永久链接")

`cudaStreamAttachMemAsync()`的主要用途是使用CPU线程启用独立任务并行。通常在这样的程序中，CPU线程为其生成的所有工作创建自己的流，因为使用CUDA的NULL流会导致线程之间存在依赖关系。

托管数据对任何GPU流的默认全局可见性可能会使多线程程序中难以避免CPU线程之间的交互。因此，函数`cudaStreamAttachMemAsync()`用于将线程的托管分配与该线程自己的流相关联，并且该关联通常不会在线程的生命周期内更改。

这样的程序只需向`cudaStreamAttachMemAsync()`添加一个调用，以使用统一内存进行数据访问：

// This function performs some task, in its own private stream.
void run_task(int *in, int *out, int length) {
    // Create a stream for us to use.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Allocate some managed data and associate with our stream.
    // Note the use of the host-attach flag to cudaMallocManaged();
    // we then associate the allocation with our stream so that
    // our GPU kernel launches can access it.
    int *data;
    cudaMallocManaged((void **)&data, length, cudaMemAttachHost);
    cudaStreamAttachMemAsync(stream, data);
    cudaStreamSynchronize(stream);
    // Iterate on the data in some way, using both Host & Device.
    for(int i=0; i<N; i++) {
        transform<<< 100, 256, 0, stream >>>(in, data, length);
        cudaStreamSynchronize(stream);
        host_process(data, length);    // CPU uses managed data.
        convert<<< 100, 256, 0, stream >>>(out, data, length);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(data);
}

在本例中，分配-流关联只建立一次，然后`data`被主机和设备反复使用。结果是比在主机和设备之间显式复制数据要简单得多的代码，尽管结果是一样的。

##### 24.3.2.4.6.高级主题：模块化程序和数据访问约束[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#advanced-topic-modular-programs-and-data-access-constraints "这个标题的永久链接")

在前面的示例中，`cudaMallocManaged()`指定了`cudaMemAttachHost`标志，该标志创建了一个最初对设备端执行不可见的分配。（默认分配将对所有流上的所有GPU内核可见。）这确保了在数据分配和为特定流获取数据之间的间隔内，不会与另一个线程的执行意外互动。

如果没有此标志，如果另一个线程启动的内核恰好正在运行，新的分配将被视为在GPU上使用中。这可能会影响线程在能够明确地将其附加到私有流之前，从CPU访问新分配的数据（例如，在基类构造函数中）的能力。因此，为了实现线程之间的安全独立性，应指定此标志进行分配。

笔记

另一种选择是在分配附加到流后，在所有线程中设置一个全进程的障碍。这将确保所有线程在启动任何内核之前完成其数据/流关联，从而避免危险。在流被摧毁之前，需要第二个障碍，因为流被摧毁会导致分配恢复到默认可见性。ThecudaMemAttachHost标志的存在既是为了简化此过程，也因为并不总是可以在需要时插入全局屏障。

##### 24.3.2.4.7.带有流关联统一内存的Memcpy（）/Memset（）行为[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memcpy-memset-behavior-with-stream-associated-unified-memory "这个标题的永久链接")

See [Memcpy()/Memset() Behavior With Unified Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-memcpy-memset) for a general overview of `cudaMemcpy*` / `cudaMemset*` behavior on devices with `concurrentManagedAccess` set. On devices where `concurrentManagedAccess` is not set, the following rules apply:

如果指定了`cudaMemcpyHostTo*`，并且源数据是统一的内存，那么如果它可以从复制流[（1）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-legacy-memcpy-cit1)中的主机一致地访问，它将从主机访问它；否则它将从设备访问。当指定了`cudaMemcpy*ToHost`并且目的地是统一内存时，类似的规则也适用于目的地。

如果指定了`cudaMemcpyDeviceTo*`，并且源数据是统一的内存，那么它将从设备访问。源必须能够从复制流[（2）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-legacy-memcpy-cit2)中的设备连贯地访问；否则，将返回错误。当指定了`cudaMemcpy*ToDevice`并且目的地是统一内存时，类似的规则也适用于目的地。

如果指定了`cudaMemcpyDefault`，那么如果无法从复制流[（2）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-legacy-memcpy-cit2)中的设备连贯地访问统一内存，或者如果数据的首选位置是`cudaCpuDeviceId`，并且可以从复制流[（1）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-legacy-memcpy-cit1)中的主机连贯地访问统一内存；否则，它将从设备访问。

当使用带有统一内存的`cudaMemset*()`，数据必须能够从用于thecudaMemset`cudaMemset()`操作[（2）](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-legacy-memcpy-cit2)的流中的设备一致地访问；否则，将返回错误。

当`cudaMemcpy*`或`cudaMemset*`从设备访问数据时，该操作流在GPU上被视为处于活动状态。在此期间，如果GPU的设备属性`concurrentManagedAccess`为零，则与该流或具有全局可见性的数据关联的任何CPU访问都会导致分割故障。程序必须适当同步，以确保在从CPU访问任何相关数据之前完成操作。

> 1. 在给定流中从主机可连贯访问意味着内存既没有全局可见性，也不与给定流相关联。
>     

> 2. 从给定流中的设备可连贯访问意味着内存要么具有全局可见性，要么与给定流相关联。
>     

# 25.懒惰加载[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#lazy-loading "这个标题的永久链接")

## 25.1.什么是懒惰加载？[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#what-is-lazy-loading "这个标题的永久链接")

延迟加载CUDA模块和内核的加载，从程序初始化到核心执行。如果一个程序不使用它包含的每个内核，那么一些内核将被不必要地加载。这很常见，特别是如果你包括任何库。大多数时候，程序只使用它们包含的库中的少量内核。

多亏了Lazy Loading，程序只能加载它们实际要使用的内核，从而节省了初始化时间。这减少了GPU内存和主机内存的内存开销。

通过将`CUDA_MODULE_LOADING`环境变量设置为`LAZY`来启用Lazy Loading。

首先，CUDA运行时在程序初始化期间将不再加载所有模块，包含托管变量的模块除外。每个模块将在首次使用该模块的变量或内核时加载。此优化仅与CUDA运行时用户相关，使用`cuModuleLoad`的CUDA驱动程序用户不受影响。此优化在CUDA 11.8中发布。使用`cuLibraryLoad`将模块数据加载到内存中的CUDA驱动程序用户的行为可以通过设置`CUDA_MODULE_DATA_LOADING`环境变量来更改。

其次，加载模块（`cuModuleLoad*()`函数家族）不会立即加载内核，而是会延迟内核的加载，直到调用`cuModuleGetFunction()`）。这里存在某些例外情况，一些内核必须在`cuModuleLoad*()`期间加载，例如指针存储在全局变量中的内核。这种优化与CUDA运行时和CUDA驱动程序用户都相关。CUDA运行时仅在首次使用/引用内核时才会调用`cuModuleGetFunction()`）。此优化在CUDA 11.7中发布。

假设遵循CUDA编程模型，这两种优化都被设计成对用户隐形。

## 25.2.懒惰加载版本支持[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#lazy-loading-version-support "这个标题的永久链接")

惰性加载是CUDA运行时和CUDA驱动程序的功能。可能需要升级到两者才能利用该功能。

### 25.2.1.司机[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver "这个标题的永久链接")

懒惰加载需要R515+用户模式库，但它支持正向兼容性，这意味着它可以在旧的内核模式驱动程序上运行。

如果没有R515+用户模式库，即使工具包版本为11.7+，Lazy Loading也不会以任何形式或形式可用。

### 25.2.2.工具包[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#toolkit "这个标题的永久链接")

CUDA 11.7中引入了Lazy Loading，并在CUDA 11.8中进行了重大升级。

如果您的应用程序使用CUDA运行时，那么为了看到懒惰加载的好处，您的应用程序必须使用11.7+ CUDA运行时。

由于CUDA运行时通常静态链接到程序和库中，这意味着您必须使用CUDA 11.7+工具包重新编译程序并使用CUDA 11.7+库。

否则，即使您的驱动程序版本支持它，您也不会看到Lazy Loading的好处。

如果您的一些库是11.7+，您只会在这些库中看到Lazy Loading的好处。其他图书馆仍然会急切地加载所有内容。

### 25.2.3.编译器[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compiler "这个标题的永久链接")

懒惰加载不需要任何编译器支持。使用pre-11.7编译器编译的SASS和PTX都可以在启用Lazy Loading的情况下加载，并将看到该功能的全部好处。然而，如上所述，仍然需要11.7+ CUDA运行时。

## 25.3.在惰性模式下触发内核的加载[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#triggering-loading-of-kernels-in-lazy-mode "这个标题的永久链接")

加载内核和变量是自动发生的，不需要任何显式加载。只需启动内核或引用变量或内核即可自动加载相关模块和内核。

但是，如果出于任何原因，您希望在不执行或以任何方式修改内核的情况下加载内核，我们建议以下内容。

### 25.3.1.CUDA驱动程序API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-driver-api "这个标题的永久链接")

内核的加载发生在`cuModuleGetFunction()`调用期间。即使没有Lazy Loading，此调用也是必要的，因为这是获得内核句柄的唯一方法。

但是，当内核加载时，您还可以使用此API以更精细的粒度进行控制。

### 25.3.2.CUDA运行时API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-runtime-api "这个标题的永久链接")

CUDA Runtime API自动管理模块管理，因此我们建议只需使用`cudaFuncGetAttributes()`来引用内核。

这将确保在不改变状态的情况下加载内核。

## 25.4.查询是否打开了懒惰加载[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#querying-whether-lazy-loading-is-turned-on "这个标题的永久链接")

In order to check whether user enabled Lazy Loading, `CUresult cuModuleGetLoadingMode ( CUmoduleLoadingMode* mode )` can be used.

需要注意的是，在运行此函数之前，必须初始化CUDA。示例用法可以在下面的片段中看到。

#include "cuda.h"
#include "assert.h"
#include "iostream"

int main() {
        CUmoduleLoadingMode mode;

        assert(CUDA_SUCCESS == cuInit(0));
        assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));

        std::cout << "CUDA Module Loading Mode is " << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager") << std::endl;

        return 0;
}

## 25.5.采用惰性加载时可能存在的问题[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#possible-issues-when-adopting-lazy-loading "这个标题的永久链接")

懒惰加载的设计不需要对应用程序进行任何修改即可使用。也就是说，有一些注意事项，特别是当应用程序不完全符合CUDA编程模型时。

### 25.5.1.并发执行[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-execution "这个标题的永久链接")

加载内核可能需要上下文同步。一些程序错误地将同时执行内核的可能性视为保证。在这种情况下，如果程序假设两个内核能够同时执行，而另一个内核在没有执行的情况下，其中一个内核不会返回，则可能存在死锁。

如果内核A将在无限循环中旋转，直到内核B执行。在这种情况下，启动内核B将触发内核B的延迟加载。如果此加载需要上下文同步，那么我们有一个死锁：内核A正在等待内核B，但加载内核B卡住，等待内核A完成上下文同步。

此类程序是反模式的，但如果出于任何原因您想保留它，您可以进行以下操作：

- 在启动之前预加载您希望同时执行的所有内核
    
- 使用`CUDA_MODULE_DATA_LOADING=EAGER`运行应用程序，以强制急切地加载数据，而不强制每个函数急切加载
    

### 25.5.2.分配器[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#allocators "这个标题的永久链接")

延迟加载将加载代码从程序的初始化阶段延迟到执行阶段。将代码加载到GPU上需要内存分配。

如果您的应用程序在启动时尝试分配整个VRAM，例如，将其用于自己的分配器，那么可能会证明没有剩余的内存来加载内核。尽管总体上，Lazy Loading为用户释放了更多内存。CUDA需要分配一些内存来加载每个内核，这通常发生在每个内核的首次启动时。如果您的应用程序分配器贪婪地分配了所有内容，CUDA将无法分配内存。

可能的解决方案：

- 使用`cudaMallocAsync()`，而不是在启动时分配整个VRAM的分配器
    
- 添加一些缓冲区来补偿内核加载的延迟
    
- 在尝试初始化分配器之前，预加载程序中将要使用的所有内核
    

### 25.5.3.自动调谐[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#autotuning "这个标题的永久链接")

一些应用程序会启动几个实现相同功能的内核，以确定哪个是最快的。虽然总体上建议至少运行一次预热迭代，但对于懒惰加载来说，这变得尤为重要。毕竟，包括加载内核所花时间，会扭曲您的结果。

可能的解决方案：

- 在测量之前至少进行一次热身互动
    
- 在启动之前预加载基准内核
    

# 26.扩展的GPU内存[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#extended-gpu-memory "这个标题的永久链接")

扩展GPU内存（EGM）功能利用高带宽NVLink-C2C，促进GPU在单节点系统中高效访问所有系统内存。EGM适用于集成的CPU-GPU NVIDIA系统，允许物理内存分配可以从设置中的任何GPU线程访问。EGM确保所有GPU都能以GPU-GPU NVLink或NVLink-C2C的速度访问其资源。

[![EGM](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/egm-c2c-intro.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/egm-c2c-intro.png)

在此设置中，内存访问通过本地高带宽NVLink-C2C进行。对于远程内存访问，使用GPU NVLink，在某些情况下，使用NVLink-C2C。借助EGM，GPU线程可以在NVSwitch结构上访问所有可用的内存资源，包括CPU附加内存和HBM3。

## 26.1.初步[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#preliminaries "这个标题的永久链接")

在深入研究EGM功能的API更改之前，我们将介绍目前支持的拓扑结构、标识符分配、虚拟内存管理的先决条件以及EGM的CUDA类型。

### 26.1.1.EGM平台：系统拓扑[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#egm-platforms-system-topology "这个标题的永久链接")

目前，EGM可以在三个平台上启用：**（1）单节点、单GPU**：由基于Arm的CPU、CPU连接内存和GPU组成。在CPU和GPU之间有一个高带宽的C2C（芯片对芯片）互连。**（2）单节点、多GPU**：由四个完全连接的单节点、单GPU平台组成。**（3）多节点、单GPU**：两个或多个单节点多插槽系统。

笔记

使用`cgroups`来限制可用设备将阻止通过EGM的路由，并导致性能问题。改为使用`CUDA_VISIBLE_DEVICES`。

### 26.1.2.套接字标识符：它们是什么？如何访问它们？[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#socket-identifiers-what-are-they-how-to-access-them "这个标题的永久链接")

NUMA（非均匀内存访问）是一种用于多处理器计算机系统的内存架构，将内存分为多个节点。每个节点都有自己的处理器和内存。在这样的系统中，NUMA将系统划分为节点，并为每个节点分配一个唯一的标识符（numaID）。

EGM使用操作系统分配的NUMA节点标识符。请注意，此标识符与设备的序数不同，它与最近的主机节点相关联。除了现有方法外，用户可以通过调用[cuDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266)和`CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`属性类型来获取主机节点的标识符（numaID），如下所示：

int numaId;
cuDeviceGetAttribute(&numaId, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, deviceOrdinal);

### 26.1.3.分配器和EGM支持[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#allocators-and-egm-support "这个标题的永久链接")

将系统内存映射为 EGM 不会造成任何性能问题。事实上，访问映射为EGM的远程套接字系统内存会更快。因为，EGM流量保证通过NVLinks路由。目前，`cuMemCreate`和`cudaMemPoolCreate`配器支持适当的位置类型和NUMA标识符。

### 26.1.4.当前API的内存管理扩展[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-management-extensions-to-current-apis "这个标题的永久链接")

目前，EGM内存可以用虚拟内存（`cuMemCreate`）或流有序内存（`cudaMemPoolCreate`）分配器进行映射。用户负责分配物理内存并将其映射到所有套接字上的虚拟内存地址空间。

笔记

多节点、单GPU平台需要进程间通信。因此，我们鼓励读者查看[第3章](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interprocess-communication)

笔记

我们鼓励读者阅读CUDA编程指南的第10[章](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-memory-management)和第11[章](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-ordered-memory-allocator)，以便更好地理解。

新的CUDA属性类型已添加到API中，允许这些方法使用类似NUMA的节点标识符来理解分配位置：

|   |   |
|---|---|
|**CUDA类型**|**与...一起使用**|
|`CU_MEM_LOCATION_TYPE_HOST_NUMA`|`CUmemAllocationProp`为了`cuMemCreate`|
|`cudaMemLocationTypeHostNuma`|`cudaMemPoolProps`为了`cudaMemPoolCreate`|

笔记

请参阅[CUDA驱动程序API](https://www.google.com/url?q=https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html&sa=D&source=editors&ust=1696873412599124&usg=AOvVaw0Ru93Acs_FpJG0gl02BLMX)和[CUDA运行时数据类型](https://www.google.com/url?q=https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html%23group__CUDART__TYPES_1gg2279aa08666f329f3ba4afe397fa60f024dc63fb938dee27b41e3842da35d2d0&sa=D&source=editors&ust=1696873412599344&usg=AOvVaw2O-SyvDt1G37IjcpFzc-4C)，以了解有关NUMA特定CUDA类型的更多信息。

## 26.2.使用EGM接口[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-the-egm-interface "这个标题的永久链接")

### 26.2.1.单节点，单GPU[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#single-node-single-gpu "这个标题的永久链接")

任何现有的CUDA主机分配器以及系统分配的内存都可以用于从高带宽C2C中受益。对用户来说，本地访问是当今的主机分配。

笔记

有关内存分配器和页面大小的更多信息，请参阅调整指南。

### 26.2.2.单节点，多GPU[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#single-node-multi-gpu "这个标题的永久链接")

在多GPU系统中，用户必须为放置提供主机信息。正如我们提到的，表达该信息的自然方式是使用NUMA节点ID，EGM遵循这种方法。因此，使用`cuDeviceGetAttribute`函数，用户应该能够学习最近的NUMA节点ID。（请参阅[套接字标识符：它们是什么？如何访问它们？](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#socket-identifiers-what-are-they-how-to-access-them)）。然后，用户可以使用VMM（虚拟内存管理）API或CUDA内存池分配和管理EGM内存。

#### 26.2.2.1.使用VMM API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-vmm-apis "这个标题的永久链接")

使用虚拟内存管理API进行内存分配的第一步是创建一个物理内存块，为分配提供支持。有关更多详细信息，请参阅CUDA编程指南的[虚拟内存管理部分](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-memory-management)。在EGM分配中，用户必须明确提供`CU_MEM_LOCATION_TYPE_HOST_NUMA`作为位置类型和numaID作为位置标识符。同样在EGM中，分配必须与平台的适当粒度保持一致。以下代码片段显示了使用`cuMemCreate`分配物理内存：

CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop, MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);

物理内存分配后，我们必须保留一个地址空间并将其映射到指针。这些规程没有特定于 EGM 的更改：

CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);

最后，用户必须明确保护映射的虚拟地址范围。否则，访问映射空间将导致崩溃。与内存分配类似，用户必须提供`CU_MEM_LOCATION_TYPE_HOST_NUMA`作为位置类型，提供numaId作为位置标识符。以下代码片段为主机节点和GPU创建访问描述符，为映射内存提供读取和写入访问权限：

CUmemAccessDesc accessDesc[2]{{}};
accessDesc[0].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
accessDesc[0].location.id = numaId;
accessDesc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
accessDesc[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
accessDesc[1].location.id = currentDev;
accessDesc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
cuMemSetAccess(dptr, size, accessDesc, 2);

#### 26.2.2.2.使用CUDA内存池[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-cuda-memory-pool "这个标题的永久链接")

要定义EGM，用户可以在节点上创建一个内存池，并授予对等的访问权限。在这种情况下，用户必须明确定义`cudaMemLocationTypeHostNuma`作为位置类型，numaId作为位置标识符。以下代码片段显示了创建内存池`cudaMemPoolCreate`：

cudaSetDevice(homeDevice);
cudaMemPoolProps props{};
props.allocType = cudaMemAllocationTypePinned;
props.location.type = cudaMemLocationTypeHostNuma;
props.location.id = numaId;
cudaMemPoolCreate(&memPool, &props);

此外，对于直接连接对等访问，也可以使用现有的对等访问API，`cudaMemPoolSetAccess`。访问设备的示例如下代码片段所示：

cudaMemAccessDesc desc{};
desc.flags = cudaMemAccessFlagsProtReadWrite;
desc.location.type = cudaMemLocationTypeDevice;
desc.location.id = accessingDevice;
cudaMemPoolSetAccess(memPool, &desc, 1);

创建内存池并授予访问权限后，用户可以将创建的内存池设置为residentDevice，并开始使用`cudaMallocAsync`分配内存：

cudaDeviceSetMemPool(residentDevice, memPool);
cudaMallocAsync(&ptr, size, memPool, stream);

笔记

EGM映射为2MB页面。因此，用户在访问非常大的分配时可能会遇到更多的TLB错过。

### 26.2.3.多节点，单GPU[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-node-single-gpu "这个标题的永久链接")

除了内存分配外，远程对等访问没有特定于EGM的修改，它遵循CUDA进程间（IPC）协议。有关IPC的更多详细信息，请参阅[CUDA编程指南](https://www.google.com/url?q=https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23allocating-physical-memory&sa=D&source=editors&ust=1696873412606850&usg=AOvVaw0IF8bdtDWgRlAiW3tIoyXg)。

用户应使用`cuMemCreate`分配内存，用户必须再次明确提供`CU_MEM_LOCATION_TYPE_HOST_NUMA`作为位置类型和numaID作为位置标识符。此外，`CU_MEM_HANDLE_TYPE_FABRIC`应定义为请求的手柄类型。以下代码片段显示了在节点A上分配物理内存：

CUmemAllocationProp prop{};
prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
prop.location.id = numaId;
size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop,
                              MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUmemGenericAllocationHandle allocHandle;
cuMemCreate(&allocHandle, padded_size, &prop, 0);

使用`cuMemCreate`创建分配句柄后，用户可以将该句柄导出到另一个节点B节点，调用`cuMemExportToShareableHandle`：

cuMemExportToShareableHandle(&fabricHandle, allocHandle,
                             CU_MEM_HANDLE_TYPE_FABRIC, 0);
// At this point, fabricHandle should be sent to Node B via TCP/IP.

在节点B上，手柄可以使用`cuMemImportFromShareableHandle`导入，并像任何其他织物手柄一样处理

// At this point, fabricHandle should be received from Node A via TCP/IP.
CUmemGenericAllocationHandle allocHandle;
cuMemImportFromShareableHandle(&allocHandle, &fabricHandle,
                               CU_MEM_HANDLE_TYPE_FABRIC);

当在节点B导入手柄时，用户可以保留一个地址空间，并以常规方式将其映射到本地：

size_t granularity = 0;
cuMemGetAllocationGranularity(&granularity, &prop,
                              MEM_ALLOC_GRANULARITY_MINIMUM);
size_t padded_size = ROUND_UP(size, granularity);
size_t page_size = ...;
assert(padded_size % page_size == 0);
CUdeviceptr dptr;
cuMemAddressReserve(&dptr, padded_size, 0, 0, 0);
cuMemMap(dptr, padded_size, 0, allocHandle, 0);

作为最后一步，用户应该对节点B的每个本地GPU提供适当的访问权限。一个代码片段示例，提供对八个本地GPU的读写访问权限：

// Give all 8 local  GPUS access to exported EGM memory located on Node A.                                                               |
CUmemAccessDesc accessDesc[8];
for (int i = 0; i < 8; i++) {
   accessDesc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
   accessDesc[i].location.id = i;
   accessDesc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}
cuMemSetAccess(dptr, size, accessDesc, 8);

# 27.通知[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#notices "这个标题的永久链接")

## 27.1.通知[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#notice "这个标题的永久链接")

本文件仅供参考，不应被视为对产品某些功能、状况或质量的保证。NVIDIA公司（“NVIDIA”）对本文档中包含信息的准确性或完整性不作任何明示或暗示的陈述或保证，并且对此处包含的任何错误不承担任何责任。NVIDIA对此类信息的后果或使用，或因使用此类信息而可能对第三方的专利或其他权利的侵犯不承担任何责任。本文档不是开发、发布或交付任何材料（定义如下）、代码或功能的承诺。

NVIDIA保留随时对本文档进行更正、修改、增强、改进和任何其他更改的权利，恕不另行通知。

客户应在下订单前获取最新的相关信息，并应验证此类信息是否是最新的和完整的。

NVIDIA产品受订单确认时提供的NVIDIA标准销售条款和条件的约束，除非NVIDIA授权代表和客户签署的个人销售协议（“销售条款”）另有约定。NVIDIA特此明确反对在购买本文档中提到的NVIDIA产品时适用任何客户一般条款和条件。本文件没有直接或间接地形成任何合同义务。

NVIDIA产品的设计、授权或保证不适合用于医疗、军事、飞机、太空或生命支持设备，也不适用于NVIDIA产品故障或故障可能导致人身伤害、死亡或财产或环境损害的应用。NVIDIA对将NVIDIA产品包含在此类设备或应用程序中和/或使用不承担任何责任，因此此类包含和/或使用由客户自行承担风险。

NVIDIA不声明或保证基于本文档的产品适用于任何特定用途。NVIDIA不一定对每个产品的所有参数进行测试。客户全权负责评估和确定本文档中包含的任何信息的适用性，确保产品适合客户计划的应用，并对应用进行必要的测试，以避免应用程序或产品的默认。客户产品设计中的弱点可能会影响NVIDIA产品的质量和可靠性，并可能导致超出本文档中包含的附加或不同的条件和/或要求。NVIDIA对可能基于或归因于以下原因的任何违约、损害、成本或问题不承担任何责任：（i）以任何违反本文档的方式使用NVIDIA产品或（ii）客户产品设计。

本文档下的任何NVIDIA专利权、版权或其他NVIDIA知识产权均不授予任何明示或暗示的许可。NVIDIA发布的有关第三方产品或服务的信息并不构成NVIDIA使用此类产品或服务的许可，也不构成对此类产品或服务的保证或认可。使用此类信息可能需要第三方根据第三方的专利或其他知识产权获得许可，或根据英伟达的专利或其他知识产权获得英伟达的许可。

只有在事先获得英伟达的书面批准、未经更改、完全遵守所有适用的出口法律和法规，并附有所有相关条件、限制和通知的情况下，才允许复制本文档中的信息。

本文档和所有NVIDIA设计规范、参考板、文件、图纸、诊断、列表和其他文档（一起单独称为“材料”）均按“原样”提供。NVIDIA对材料不作任何明示、暗示、法定或其他保证，并明确否认对非侵权、适销性和特定用途的适用性的所有暗示保证。在法律不禁止的范围内，NVIDIA在任何情况下均不对任何损害负责，包括但不限于任何直接、间接、特殊、附带、懲罰性或后果性损害，无论其原因如何，无论责任理论如何，均因使用本文档而产生，即使NVIDIA已被告知此类损害的可能性。尽管客户可能因任何原因遭受任何损害，但NVIDIA对此处所述产品对客户的总和累积责任应根据产品销售条款进行限制。

## 27.2.开放CL[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opencl "这个标题的永久链接")

OpenCL是苹果公司的商标，在Khronos Group Inc.的许可下使用。

## 27.3.商标[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#trademarks "这个标题的永久链接")

NVIDIA和NVIDIA徽标是NVIDIA公司在美国和其他国家的商标或注册商标。其他公司和产品名称可能是与其关联的各自公司的商标。