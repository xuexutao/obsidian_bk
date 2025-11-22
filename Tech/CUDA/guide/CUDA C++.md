CUDA C++ Programming Guide

# 1. Overview[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overview "Permalink to this headline")

CUDA is a parallel computing platform and programming model developed by NVIDIA that enables dramatic increases in computing performance by harnessing the power of the GPU. It allows developers to accelerate compute-intensive applications using C, C++, and Fortran, and is widely adopted in fields such as deep learning, scientific computing, and high-performance computing (HPC).

# 2. What Is the CUDA C Programming Guide?[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#what-is-the-cuda-c-programming-guide "Permalink to this headline")

The CUDA C Programming Guide is the official, comprehensive resource that explains how to write programs using the CUDA platform. It provides detailed documentation of the CUDA architecture, programming model, language extensions, and performance guidelines. Whether you’re just getting started or optimizing complex GPU kernels, this guide is an essential reference for effectively leveraging CUDA’s full capabilities.

# 3. Introduction[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#introduction "Permalink to this headline")

## 3.1. The Benefits of Using GPUs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-benefits-of-using-gpus "Permalink to this headline")

The Graphics Processing Unit (GPU)[1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn1) provides much higher instruction throughput and memory bandwidth than the CPU within a similar price and power envelope. Many applications leverage these higher capabilities to run faster on the GPU than on the CPU (see [GPU Applications](https://www.nvidia.com/object/gpu-applications.html)). Other computing devices, like FPGAs, are also very energy efficient, but offer much less programming flexibility than GPUs.

This difference in capabilities between the GPU and the CPU exists because they are designed with different goals in mind. While the CPU is designed to excel at executing a sequence of operations, called a _thread_, as fast as possible and can execute a few tens of these threads in parallel, the GPU is designed to excel at executing thousands of them in parallel (amortizing the slower single-thread performance to achieve greater throughput).

The GPU is specialized for highly parallel computations and therefore designed such that more transistors are devoted to data processing rather than data caching and flow control. The schematic [Figure 1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing) shows an example distribution of chip resources for a CPU versus a GPU.

[![The GPU Devotes More Transistors to Data Processing](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-devotes-more-transistors-to-data-processing.png)

Figure 1 The GPU Devotes More Transistors to Data Processing[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#from-graphics-processing-to-general-purpose-parallel-computing-gpu-devotes-more-transistors-to-data-processing "Permalink to this image")

Devoting more transistors to data processing, for example, floating-point computations, is beneficial for highly parallel computations; the GPU can hide memory access latencies with computation, instead of relying on large data caches and complex flow control to avoid long memory access latencies, both of which are expensive in terms of transistors.

In general, an application has a mix of parallel parts and sequential parts, so systems are designed with a mix of GPUs and CPUs in order to maximize overall performance. Applications with a high degree of parallelism can exploit this massively parallel nature of the GPU to achieve higher performance than on the CPU.

[1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id2)

The _graphics_ qualifier comes from the fact that when the GPU was originally created, two decades ago, it was designed as a specialized processor to accelerate graphics rendering. Driven by the insatiable market demand for real-time, high-definition, 3D graphics, it has evolved into a general processor used for many more workloads than just graphics rendering.

## 3.2. CUDA®: A General-Purpose Parallel Computing Platform and Programming Model[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-a-general-purpose-parallel-computing-platform-and-programming-model "Permalink to this headline")

In November 2006, NVIDIA® introduced CUDA®, a general purpose parallel computing platform and programming model that leverages the parallel compute engine in NVIDIA GPUs to solve many complex computational problems in a more efficient way than on a CPU.

CUDA comes with a software environment that allows developers to use C++ as a high-level programming language. As illustrated by [Figure 2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-general-purpose-parallel-computing-architecture-cuda-is-designed-to-support-various-languages-and-application-programming-interfaces), other languages, application programming interfaces, or directives-based approaches are supported, such as FORTRAN, DirectCompute, OpenACC.

[![GPU Computing Applications. CUDA is designed to support various languages and application programming interfaces.](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-computing-applications.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-computing-applications.png)

Figure 2 GPU Computing Applications. CUDA is designed to support various languages and application programming interfaces.[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-general-purpose-parallel-computing-architecture-cuda-is-designed-to-support-various-languages-and-application-programming-interfaces "Permalink to this image")

## 3.3. A Scalable Programming Model[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#a-scalable-programming-model "Permalink to this headline")

The advent of multicore CPUs and manycore GPUs means that mainstream processor chips are now parallel systems. The challenge is to develop application software that transparently scales its parallelism to leverage the increasing number of processor cores, much as 3D graphics applications transparently scale their parallelism to manycore GPUs with widely varying numbers of cores.

The CUDA parallel programming model is designed to overcome this challenge while maintaining a low learning curve for programmers familiar with standard programming languages such as C.

At its core are three key abstractions — a hierarchy of thread groups, shared memories, and barrier synchronization — that are simply exposed to the programmer as a minimal set of language extensions.

These abstractions provide fine-grained data parallelism and thread parallelism, nested within coarse-grained data parallelism and task parallelism. They guide the programmer to partition the problem into coarse sub-problems that can be solved independently in parallel by blocks of threads, and each sub-problem into finer pieces that can be solved cooperatively in parallel by all threads within the block.

This decomposition preserves language expressivity by allowing threads to cooperate when solving each sub-problem, and at the same time enables automatic scalability. Indeed, each block of threads can be scheduled on any of the available multiprocessors within a GPU, in any order, concurrently or sequentially, so that a compiled CUDA program can execute on any number of multiprocessors as illustrated by [Figure 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#scalable-programming-model-automatic-scalability), and only the runtime system needs to know the physical multiprocessor count.

This scalable programming model allows the GPU architecture to span a wide market range by simply scaling the number of multiprocessors and memory partitions: from the high-performance enthusiast GeForce GPUs and professional Quadro and Tesla computing products to a variety of inexpensive, mainstream GeForce GPUs (see [CUDA-Enabled GPUs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-enabled-gpus) for a list of all CUDA-enabled GPUs).

![Automatic Scalability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/automatic-scalability.png)

Figure 3 Automatic Scalability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#scalable-programming-model-automatic-scalability "Permalink to this image")

Note

A GPU is built around an array of Streaming Multiprocessors (SMs) (see [Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-implementation) for more details). A multithreaded program is partitioned into blocks of threads that execute independently from each other, so that a GPU with more multiprocessors will automatically execute the program in less time than a GPU with fewer multiprocessors.

# 4. Changelog[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#changelog "Permalink to this headline")

Table 1 Change Log[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id452 "Permalink to this table")
|Version|Changes|
|---|---|
|13.0|Moved the instruction throughput table from the _Performance Guidelines_ section of the CUDA C++ Programming Guide to the [Instruction-optimization](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#instruction-optimization) section of the CUDA C++ Best Practices Guide. Removed unsupported architectures and corrected entries for integer arithmetic and type conversion.|
|12.9|Added section [Error Log Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#error-log-management) and CUDA_LOG_FILE to [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#env-vars)|
|12.8|Added section [TMA Swizzle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tma-swizzle)|

# 5. Programming Model[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model "Permalink to this headline")

This chapter introduces the main concepts behind the CUDA programming model by outlining how they are exposed in C++.

An extensive description of CUDA C++ is given in [Programming Interface](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface).

Full code for the vector addition example used in this chapter and the next can be found in the [vectorAdd CUDA sample](https://docs.nvidia.com/cuda/cuda-samples/index.html#vector-addition).

## 5.1. Kernels[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernels "Permalink to this headline")

CUDA C++ extends C++ by allowing the programmer to define C++ functions, called _kernels_, that, when called, are executed N times in parallel by N different _CUDA threads_, as opposed to only once like regular C++ functions.

A kernel is defined using the `__global__` declaration specifier and the number of CUDA threads that execute that kernel for a given kernel call is specified using a new `<<<...>>>`_execution configuration_ syntax (see [Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)). Each thread that executes the kernel is given a unique _thread ID_ that is accessible within the kernel through built-in variables.

As an illustration, the following sample code, using the built-in variable `threadIdx`, adds two vectors _A_ and _B_ of size _N_ and stores the result into vector _C_.

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

Here, each of the _N_ threads that execute `VecAdd()` performs one pair-wise addition.

## 5.2. Thread Hierarchy[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy "Permalink to this headline")

For convenience, `threadIdx` is a 3-component vector, so that threads can be identified using a one-dimensional, two-dimensional, or three-dimensional _thread index_, forming a one-dimensional, two-dimensional, or three-dimensional block of threads, called a _thread block_. This provides a natural way to invoke computation across the elements in a domain such as a vector, matrix, or volume.

The index of a thread and its thread ID relate to each other in a straightforward way: For a one-dimensional block, they are the same; for a two-dimensional block of size _(Dx, Dy)_, the thread ID of a thread of index _(x, y)_ is _(x + y Dx)_; for a three-dimensional block of size _(Dx, Dy, Dz)_, the thread ID of a thread of index _(x, y, z)_ is _(x + y Dx + z Dx Dy)_.

As an example, the following code adds two matrices _A_ and _B_ of size _NxN_ and stores the result into matrix _C_.

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

There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same streaming multiprocessor core and must share the limited memory resources of that core. On current GPUs, a thread block may contain up to 1024 threads.

However, a kernel can be executed by multiple equally-shaped thread blocks, so that the total number of threads is equal to the number of threads per block times the number of blocks.

Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional _grid_ of thread blocks as illustrated by [Figure 4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy-grid-of-thread-blocks). The number of thread blocks in a grid is usually dictated by the size of the data being processed, which typically exceeds the number of processors in the system.

[![Grid of Thread Blocks](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png)

Figure 4 Grid of Thread Blocks[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy-grid-of-thread-blocks "Permalink to this image")

The number of threads per block and the number of blocks per grid specified in the `<<<...>>>` syntax can be of type `int` or `dim3`. Two-dimensional blocks or grids can be specified as in the example above.

Each block within the grid can be identified by a one-dimensional, two-dimensional, or three-dimensional unique index accessible within the kernel through the built-in `blockIdx` variable. The dimension of the thread block is accessible within the kernel through the built-in `blockDim` variable.

Extending the previous `MatAdd()` example to handle multiple blocks, the code becomes as follows.

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

A thread block size of 16x16 (256 threads), although arbitrary in this case, is a common choice. The grid is created with enough blocks to have one thread per matrix element as before. For simplicity, this example assumes that the number of threads per grid in each dimension is evenly divisible by the number of threads per block in that dimension, although that need not be the case.

Thread blocks are required to execute independently. It must be possible to execute blocks in any order, in parallel or in series. This independence requirement allows thread blocks to be scheduled in any order and across any number of cores as illustrated by [Figure 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#scalable-programming-model-automatic-scalability), enabling programmers to write code that scales with the number of cores.

Threads within a block can cooperate by sharing data through some _shared memory_ and by synchronizing their execution to coordinate memory accesses. More precisely, one can specify synchronization points in the kernel by calling the `__syncthreads()` intrinsic function; `__syncthreads()` acts as a barrier at which all threads in the block must wait before any is allowed to proceed. [Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory) gives an example of using shared memory. In addition to `__syncthreads()`, the [Cooperative Groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups) provides a rich set of thread-synchronization primitives.

For efficient cooperation, shared memory is expected to be a low-latency memory near each processor core (much like an L1 cache) and `__syncthreads()` is expected to be lightweight.

### 5.2.1. Thread Block Clusters[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters "Permalink to this headline")

With the introduction of NVIDIA [Compute Capability 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0), the CUDA programming model introduces an optional level of hierarchy called Thread Block Clusters that are made up of thread blocks. Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU.

Similar to thread blocks, clusters are also organized into a one-dimension, two-dimension, or three-dimension grid of thread block clusters as illustrated by [Figure 5](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters-grid-of-clusters). The number of thread blocks in a cluster can be user-defined, and a maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA. Note that on GPU hardware or MIG configurations which are too small to support 8 multiprocessors the maximum cluster size will be reduced accordingly. Identification of these smaller configurations, as well as of larger configurations supporting a thread block cluster size beyond 8, is architecture-specific and can be queried using the `cudaOccupancyMaxPotentialClusterSize` API.

[![Grid of Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png)

Figure 5 Grid of Thread Block Clusters[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters-grid-of-clusters "Permalink to this image")

Note

In a kernel launched using cluster support, the gridDim variable still denotes the size in terms of number of thread blocks, for compatibility purposes. The rank of a block in a cluster can be found using the [Cluster Group](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-group-cg) API.

A thread block cluster can be enabled in a kernel either using a compile-time kernel attribute using `__cluster_dims__(X,Y,Z)` or using the CUDA kernel launch API `cudaLaunchKernelEx`. The example below shows how to launch a cluster using a compile-time kernel attribute. The cluster size using kernel attribute is fixed at compile time and then the kernel can be launched using the classical `<<< , >>>`. If a kernel uses compile-time cluster size, the cluster size cannot be modified when launching the kernel.

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

A thread block cluster size can also be set at runtime and the kernel can be launched using the CUDA kernel launch API `cudaLaunchKernelEx`. The code example below shows how to launch a cluster kernel using the extensible API.

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

Thread blocks that belong to a cluster have access to the Distributed Shared Memory. Thread blocks in a cluster have the ability to read, write, and perform atomics to any address in the distributed shared memory. [Distributed Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#distributed-shared-memory) gives an example of performing histograms in distributed shared memory.

### 5.2.2. Blocks as Clusters[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#blocks-as-clusters "Permalink to this headline")

With `__cluster_dims__`, the number of launched clusters is kept implicit and can only be calculated manually.

__cluster_dims__((2, 2, 2)) __global__ void foo();

// 8x8x8 clusters each with 2x2x2 thread blocks.
foo<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();

In the above example, the kernel is launched as a grid of 16x16x16 thread blocks, or in fact a grid of 8x8x8 clusters. Alternatively, with another compile-time kernel attribute `__block_size__`, one is allowed to launch a grid explicitly configured with the number of thread block clusters.

// Implementation detail of how many threads per block and blocks per cluster
// is handled as an attribute of the kernel.
__block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();

// 8x8x8 clusters.
foo<<<dim3(8, 8, 8)>>>();

`__block_size__` requires two fields each being a tuple of 3 elements. The first tuple denotes block dimension and second cluster size. The second tuple is assumed to be `(1,1,1)` if it’s not passed. To specify the stream, one must pass `1` and `0` as the second and third arguments within `<<<>>>` and lastly the stream. Passing other values would lead to undefined behavior.

Note that it is illegal for the second tuple of `__block_size__` and `__cluster_dims__` to be specified at the same time. When the second tuple of `__block_size__` is specified, it implies the “Blocks as Clusters” being enabled and the compiler would recognize the first argument inside `<<<>>>` as the number of clusters instead of thread blocks.

## 5.3. Memory Hierarchy[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-hierarchy "Permalink to this headline")

CUDA threads may access data from multiple memory spaces during their execution as illustrated by [Figure 6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-hierarchy-memory-hierarchy-figure). Each thread has private local memory. Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block. Thread blocks in a thread block cluster can perform read, write, and atomics operations on each other’s shared memory. All threads have access to the same global memory.

There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces. The global, constant, and texture memory spaces are optimized for different memory usages (see [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)). Texture memory also offers different addressing modes, as well as data filtering, for some specific data formats (see [Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory)).

The global, constant, and texture memory spaces are persistent across kernel launches by the same application.

![Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png)

Figure 6 Memory Hierarchy[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-hierarchy-memory-hierarchy-figure "Permalink to this image")

## 5.4. Heterogeneous Programming[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming "Permalink to this headline")

As illustrated by [Figure 7](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming-heterogeneous-programming), the CUDA programming model assumes that the CUDA threads execute on a physically separate _device_ that operates as a coprocessor to the _host_ running the C++ program. This is the case, for example, when the kernels execute on a GPU and the rest of the C++ program executes on a CPU.

The CUDA programming model also assumes that both the host and the device maintain their own separate memory spaces in DRAM, referred to as _host memory_ and _device memory_, respectively. Therefore, a program manages the global, constant, and texture memory spaces visible to kernels through calls to the CUDA runtime (described in [Programming Interface](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface)). This includes device memory allocation and deallocation as well as data transfer between host and device memory.

Unified Memory provides _managed memory_ to bridge the host and device memory spaces. Managed memory is accessible from all CPUs and GPUs in the system as a single, coherent memory image with a common address space. This capability enables oversubscription of device memory and can greatly simplify the task of porting applications by eliminating the need to explicitly mirror data on host and device. See [Unified Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#um-unified-memory-programming-hd) for an introduction to Unified Memory.

![Heterogeneous Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/heterogeneous-programming.png)

Figure 7 Heterogeneous Programming[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming-heterogeneous-programming "Permalink to this image")

Note

Serial code executes on the host while parallel code executes on the device.

## 5.5. Asynchronous SIMT Programming Model[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-simt-programming-model "Permalink to this headline")

In the CUDA programming model a thread is the lowest level of abstraction for doing a computation or a memory operation. Starting with devices based on the **NVIDIA Ampere GPU Architecture**, the CUDA programming model provides acceleration to memory operations via the asynchronous programming model. The asynchronous programming model defines the behavior of asynchronous operations with respect to CUDA threads.

The asynchronous programming model defines the behavior of [Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier) for synchronization between CUDA threads. The model also explains and defines how [cuda::memcpy_async](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies) can be used to move data asynchronously from global memory while computing in the GPU.

### 5.5.1. Asynchronous Operations[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-operations "Permalink to this headline")

An asynchronous operation is defined as an operation that is initiated by a CUDA thread and is executed asynchronously as-if by another thread. In a well formed program one or more CUDA threads synchronize with the asynchronous operation. The CUDA thread that initiated the asynchronous operation is not required to be among the synchronizing threads.

Such an asynchronous thread (an as-if thread) is always associated with the CUDA thread that initiated the asynchronous operation. An asynchronous operation uses a synchronization object to synchronize the completion of the operation. Such a synchronization object can be explicitly managed by a user (e.g., `cuda::memcpy_async`) or implicitly managed within a library (e.g., `cooperative_groups::memcpy_async`).

A synchronization object could be a `cuda::barrier` or a `cuda::pipeline`. These objects are explained in detail in [Asynchronous Barrier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#aw-barrier) and [Asynchronous Data Copies using cuda::pipeline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies). These synchronization objects can be used at different thread scopes. A scope defines the set of threads that may use the synchronization object to synchronize with the asynchronous operation. The following table defines the thread scopes available in CUDA C++ and the threads that can be synchronized with each.

|Thread Scope|Description|
|---|---|
|`cuda::thread_scope::thread_scope_thread`|Only the CUDA thread which initiated asynchronous operations synchronizes.|
|`cuda::thread_scope::thread_scope_block`|All or any CUDA threads within the same thread block as the initiating thread synchronizes.|
|`cuda::thread_scope::thread_scope_device`|All or any CUDA threads in the same GPU device as the initiating thread synchronizes.|
|`cuda::thread_scope::thread_scope_system`|All or any CUDA or CPU threads in the same system as the initiating thread synchronizes.|

These thread scopes are implemented as extensions to standard C++ in the [CUDA Standard C++](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes) library.

## 5.6. Compute Capability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability "Permalink to this headline")

The _compute capability_ of a device is represented by a version number, also sometimes called its “SM version”. This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU.

The compute capability comprises a major revision number _X_ and a minor revision number _Y_ and is denoted by _X.Y_.

The major revision number indicates the core GPU architecture of a device. Devices with the same major revision number share the same fundamental architecture. The table below lists the major revision numbers corresponding to each NVIDIA GPU architecture.

Table 2 GPU Architecture and Major Revision Numbers[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id453 "Permalink to this table")
|Major Revision Number|NVIDIA GPU Architecture|
|---|---|
|9|NVIDIA Hopper GPU Architecture|
|8|NVIDIA Ampere GPU Architecture|
|7|NVIDIA Volta GPU Architecture|
|6|NVIDIA Pascal GPU Architecture|
|5|NVIDIA Maxwell GPU Architecture|
|3|NVIDIA Kepler GPU Architecture|

The minor revision number corresponds to an incremental improvement to the core architecture, possibly including new features.

Table 3 Incremental Updates in GPU Architectures[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id454 "Permalink to this table")
|Compute Capability|NVIDIA GPU Architecture|Based On|
|---|---|---|
|7.5|NVIDIA Turing GPU Architecture|NVIDIA Volta GPU Architecture|

[CUDA-Enabled GPUs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-enabled-gpus) lists of all CUDA-enabled devices along with their compute capability. [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities) gives the technical specifications of each compute capability.

Note

The compute capability version of a particular GPU should not be confused with the CUDA version (for example, CUDA 7.5, CUDA 8, CUDA 9), which is the version of the CUDA _software platform_. The CUDA platform is used by application developers to create applications that run on many generations of GPU architectures, including future GPU architectures yet to be invented. While new versions of the CUDA platform often add native support for a new GPU architecture by supporting the compute capability version of that architecture, new versions of the CUDA platform typically also include software features that are independent of hardware generation.

The _Tesla_ and _Fermi_ architectures are no longer supported starting with CUDA 7.0 and CUDA 9.0, respectively.

# 6. Programming Interface[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-interface "Permalink to this headline")

CUDA C++ provides a simple path for users familiar with the C++ programming language to easily write programs for execution by the device.

It consists of a minimal set of extensions to the C++ language and a runtime library.

The core language extensions have been introduced in [Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model). They allow programmers to define a kernel as a C++ function and use some new syntax to specify the grid and block dimension each time the function is called. A complete description of all extensions can be found in [C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions). Any source file that contains some of these extensions must be compiled with `nvcc` as outlined in [Compilation with NVCC](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-with-nvcc).

The runtime is introduced in [CUDA Runtime](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-runtime). It provides C and C++ functions that execute on the host to allocate and deallocate device memory, transfer data between host memory and device memory, manage systems with multiple devices, etc. A complete description of the runtime can be found in the CUDA reference manual.

The runtime is built on top of a lower-level C API, the CUDA driver API, which is also accessible by the application. The driver API provides an additional level of control by exposing lower-level concepts such as CUDA contexts - the analogue of host processes for the device - and CUDA modules - the analogue of dynamically loaded libraries for the device. Most applications do not use the driver API as they do not need this additional level of control and when using the runtime, context and module management are implicit, resulting in more concise code. As the runtime is interoperable with the driver API, most applications that need some driver API features can default to use the runtime API and only use the driver API where needed. The driver API is introduced in [Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-api) and fully described in the reference manual.

## 6.1. Compilation with NVCC[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-with-nvcc "Permalink to this headline")

Kernels can be written using the CUDA instruction set architecture, called _PTX_, which is described in the PTX reference manual. It is however usually more effective to use a high-level programming language such as C++. In both cases, kernels must be compiled into binary code by `nvcc` to execute on the device.

`nvcc` is a compiler driver that simplifies the process of compiling _C++_ or _PTX_ code: It provides simple and familiar command line options and executes them by invoking the collection of tools that implement the different compilation stages. This section gives an overview of `nvcc` workflow and command options. A complete description can be found in the `nvcc` user manual.

### 6.1.1. Compilation Workflow[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-workflow "Permalink to this headline")

#### 6.1.1.1. Offline Compilation[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#offline-compilation "Permalink to this headline")

Source files compiled with `nvcc` can include a mix of host code (i.e., code that executes on the host) and device code (i.e., code that executes on the device). `nvcc`’s basic workflow consists in separating device code from host code and then:

- compiling the device code into an assembly form (_PTX_ code) and/or binary form (_cubin_ object),
    
- and modifying the host code by replacing the `<<<...>>>` syntax introduced in [Kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernels) (and described in more details in [Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)) by the necessary CUDA runtime function calls to load and launch each compiled kernel from the _PTX_ code and/or _cubin_ object.
    

The modified host code is output either as C++ code that is left to be compiled using another tool or as object code directly by letting `nvcc` invoke the host compiler during the last compilation stage.

Applications can then:

- Either link to the compiled host code (this is the most common case),
    
- Or ignore the modified host code (if any) and use the CUDA driver API (see [Driver API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#driver-api)) to load and execute the _PTX_ code or _cubin_ object.
    

#### 6.1.1.2. Just-in-Time Compilation[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#just-in-time-compilation "Permalink to this headline")

Any _PTX_ code loaded by an application at runtime is compiled further to binary code by the device driver. This is called _just-in-time compilation_. Just-in-time compilation increases application load time, but allows the application to benefit from any new compiler improvements coming with each new device driver. It is also the only way for applications to run on devices that did not exist at the time the application was compiled, as detailed in [Application Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility).

When the device driver just-in-time compiles some _PTX_ code for some application, it automatically caches a copy of the generated binary code in order to avoid repeating the compilation in subsequent invocations of the application. The cache - referred to as _compute cache_ - is automatically invalidated when the device driver is upgraded, so that applications can benefit from the improvements in the new just-in-time compiler built into the device driver.

Environment variables are available to control just-in-time compilation as described in [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#env-vars)

As an alternative to using `nvcc` to compile CUDA C++ device code, NVRTC can be used to compile CUDA C++ device code to PTX at runtime. NVRTC is a runtime compilation library for CUDA C++; more information can be found in the NVRTC User guide.

### 6.1.2. Binary Compatibility[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#binary-compatibility "Permalink to this headline")

Binary code is architecture-specific. A _cubin_ object is generated using the compiler option `-code` that specifies the targeted architecture: For example, compiling with `-code=sm_80` produces binary code for devices of [compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability) 8.0. Binary compatibility is guaranteed from one minor revision to the next one, but not from one minor revision to the previous one or across major revisions. In other words, a _cubin_ object generated for compute capability _X.y_ will only execute on devices of compute capability _X.z_ where _z≥y_.

Note

Binary compatibility is supported only for the desktop. It is not supported for Tegra. Also, the binary compatibility between desktop and Tegra is not supported.

### 6.1.3. PTX Compatibility[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ptx-compatibility "Permalink to this headline")

Some _PTX_ instructions are only supported on devices of higher compute capabilities. For example, [Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions) are only supported on devices of compute capability 5.0 and above. The `-arch` compiler option specifies the compute capability that is assumed when compiling C++ to _PTX_ code. So, code that contains warp shuffle, for example, must be compiled with `-arch=compute_50` (or higher).

_PTX_ code produced for some specific compute capability can always be compiled to binary code of greater or equal compute capability. Note that a binary compiled from an earlier PTX version may not make use of some hardware features. For example, a binary targeting devices of compute capability 7.0 (Volta) compiled from PTX generated for compute capability 6.0 (Pascal) will not make use of Tensor Core instructions, since these were not available on Pascal. As a result, the final binary may perform worse than would be possible if the binary were generated using the latest version of PTX.

_PTX_ code compiled to target [Architecture-Specific Features](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-specific-features) only runs on the exact same physical architecture and nowhere else. Architecture-specific _PTX_ code is not forward and backward compatible. Example code compiled with `sm_90a` or `compute_90a` only runs on devices with compute capability 9.0 and is not backward or forward compatible.

_PTX_ code compiled to target [Family-Specific Features](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-features) only runs on the exact same physical architecture and other architectures in the same family. Family-specific _PTX_ code is forward compatible with other devices in the same family, and is not backward compatible. Example code compiled with `sm_100f` or `compute_100f` only runs on devices with compute capability 10.0 and 10.3. [Table 25](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-compatibility) shows the compatibility of family-specific targets with compute capability.

### 6.1.4. Application Compatibility[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility "Permalink to this headline")

To execute code on devices of specific compute capability, an application must load binary or _PTX_ code that is compatible with this compute capability as described in [Binary Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#binary-compatibility) and [PTX Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#ptx-compatibility). In particular, to be able to execute code on future architectures with higher compute capability (for which no binary code can be generated yet), an application must load _PTX_ code that will be just-in-time compiled for these devices (see [Just-in-Time Compilation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#just-in-time-compilation)).

Which _PTX_ and binary code gets embedded in a CUDA C++ application is controlled by the `-arch` and `-code` compiler options or the `-gencode` compiler option as detailed in the `nvcc` user manual. For example,

nvcc x.cu
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_70,code=\"compute_70,sm_70\"

embeds binary code compatible with compute capability 5.0 and 6.0 (first and second `-gencode` options) and _PTX_ and binary code compatible with compute capability 7.0 (third `-gencode` option).

Host code is generated to automatically select at runtime the most appropriate code to load and execute, which, in the above example, will be:

- 5.0 binary code for devices with compute capability 5.0 and 5.2,
    
- 6.0 binary code for devices with compute capability 6.0 and 6.1,
    
- 7.0 binary code for devices with compute capability 7.0 and 7.5,
    
- _PTX_ code which is compiled to binary code at runtime for devices with compute capability later than 7.5
    

`x.cu` can have an optimized code path that uses warp reduction operations, for example, which are only supported in devices of compute capability 8.0 and higher. The `__CUDA_ARCH__` macro can be used to differentiate various code paths based on compute capability. It is only defined for device code. When compiling with `-arch=compute_80` for example, `__CUDA_ARCH__` is equal to `800`.

If `x.cu` is compiled for [Family-Specific Features](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#family-specific-features) with `sm_100f` or `compute_100f`, the code can only run on devices in that specific family, which are devices with compute capability 10.0 and 10.3. For family-specific code targets an additional macro `__CUDA_ARCH_FAMILY_SPECIFIC__` is defined. In this example, `__CUDA_ARCH_FAMILY_SPECIFIC__` is equal to `1000`.

If `x.cu` is compiled for [Architecture-Specific Features](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#architecture-specific-features) with `sm_100a` or `compute_100a`, the code can only run on devices with compute capability 10.0. For architecture-specific code targets an additional macro `__CUDA_ARCH_SPECIFIC__` is defined. In this example, `__CUDA_ARCH_SPECIFIC__` is equal to `1000`. Because architecture-specific features are a superset of family-specific features, the family-specific macro `__CUDA_ARCH_FAMILY_SPECIFIC__` is also defined and is equal to `1000`.

Applications using the driver API must compile code to separate files and explicitly load and execute the most appropriate file at runtime.

The Volta architecture introduces _Independent Thread Scheduling_ which changes the way threads are scheduled on the GPU. For code relying on specific behavior of [SIMT scheduling](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture) in previous architectures, Independent Thread Scheduling may alter the set of participating threads, leading to incorrect results. To aid migration while implementing the corrective actions detailed in [Independent Thread Scheduling](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#independent-thread-scheduling-7-x), Volta developers can opt-in to Pascal’s thread scheduling with the compiler option combination `-arch=compute_60 -code=sm_70`.

The `nvcc` user manual lists various shorthands for the `-arch`, `-code`, and `-gencode` compiler options. For example, `-arch=sm_70` is a shorthand for `-arch=compute_70 -code=compute_70,sm_70` (which is the same as `-gencode arch=compute_70,code=\"compute_70,sm_70\"`).

### 6.1.5. C++ Compatibility[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-compatibility "Permalink to this headline")

The front end of the compiler processes CUDA source files according to C++ syntax rules. Full C++ is supported for the host code. However, only a subset of C++ is fully supported for the device code as described in [C++ Language Support](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-cplusplus-language-support).

### 6.1.6. 64-Bit Compatibility[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#bit-compatibility "Permalink to this headline")

The 64-bit version of `nvcc` compiles device code in 64-bit mode (i.e., pointers are 64-bit). Device code compiled in 64-bit mode is only supported with host code compiled in 64-bit mode.

## 6.2. CUDA Runtime[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-runtime "Permalink to this headline")

The runtime is implemented in the `cudart` library, which is linked to the application, either statically via `cudart.lib` or `libcudart.a`, or dynamically via `cudart.dll` or `libcudart.so`. Applications that require `cudart.dll` and/or `cudart.so` for dynamic linking typically include them as part of the application installation package. It is only safe to pass the address of CUDA runtime symbols between components that link to the same instance of the CUDA runtime.

All its entry points are prefixed with `cuda`.

As mentioned in [Heterogeneous Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming), the CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. [Device Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory) gives an overview of the runtime functions used to manage device memory.

[Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory) illustrates the use of shared memory, introduced in [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy), to maximize performance.

[Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#page-locked-host-memory) introduces page-locked host memory that is required to overlap kernel execution with data transfers between host and device memory.

[Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution) describes the concepts and API used to enable asynchronous concurrent execution at various levels in the system.

[Multi-Device System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-device-system) shows how the programming model extends to a system with multiple devices attached to the same host.

[Error Checking](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#error-checking) describes how to properly check the errors generated by the runtime.

[Call Stack](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#call-stack) mentions the runtime functions used to manage the CUDA C++ call stack.

[Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory) presents the texture and surface memory spaces that provide another way to access device memory; they also expose a subset of the GPU texturing hardware.

[Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graphics-interoperability) introduces the various functions the runtime provides to interoperate with the two main graphics APIs, OpenGL and Direct3D.

### 6.2.1. Initialization[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#initialization "Permalink to this headline")

As of CUDA 12.0, the `cudaInitDevice()` and `cudaSetDevice()` calls initialize the runtime and the primary context associated with the specified device. Absent these calls, the runtime will implicitly use device 0 and self-initialize as needed to process other runtime API requests. One needs to keep this in mind when timing runtime function calls and when interpreting the error code from the first call into the runtime. Before 12.0, `cudaSetDevice()` would not initialize the runtime and applications would often use the no-op runtime call `cudaFree(0)` to isolate the runtime initialization from other api activity (both for the sake of timing and error handling).

The runtime creates a CUDA context for each device in the system (see [Context](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#context) for more details on CUDA contexts). This context is the _primary context_ for this device and is initialized at the first runtime function which requires an active context on this device. It is shared among all the host threads of the application. As part of this context creation, the device code is just-in-time compiled if necessary (see [Just-in-Time Compilation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#just-in-time-compilation)) and loaded into device memory. This all happens transparently. If needed, for example, for driver API interoperability, the primary context of a device can be accessed from the driver API as described in [Interoperability between Runtime and Driver APIs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interoperability-between-runtime-and-driver-apis).

When a host thread calls `cudaDeviceReset()`, this destroys the primary context of the device the host thread currently operates on (that is, the current device as defined in [Device Selection](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-selection)). The next runtime function call made by any host thread that has this device as current will create a new primary context for this device.

Note

The CUDA interfaces use global state that is initialized during host program initiation and destroyed during host program termination. The CUDA runtime and driver cannot detect if this state is invalid, so using any of these interfaces (implicitly or explicitly) during program initiation or termination after main) will result in undefined behavior.

As of CUDA 12.0, `cudaSetDevice()` will now explicitly initialize the runtime after changing the current device for the host thread. Previous versions of CUDA delayed runtime initialization on the new device until the first runtime call was made after `cudaSetDevice()`. This change means that it is now very important to check the return value of `cudaSetDevice()` for initialization errors.

The runtime functions from the error handling and version management sections of the reference manual do not initialize the runtime.

### 6.2.2. Device Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory "Permalink to this headline")

As mentioned in [Heterogeneous Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#heterogeneous-programming), the CUDA programming model assumes a system composed of a host and a device, each with their own separate memory. Kernels operate out of device memory, so the runtime provides functions to allocate, deallocate, and copy device memory, as well as transfer data between host memory and device memory.

Device memory can be allocated either as _linear memory_ or as _CUDA arrays_.

CUDA arrays are opaque memory layouts optimized for texture fetching. They are described in [Texture and Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory).

Linear memory is allocated in a single unified address space, which means that separately allocated entities can reference one another via pointers, for example, in a binary tree or linked list. The size of the address space depends on the host system (CPU) and the compute capability of the used GPU:

Table 4 Linear Memory Address Space[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id455 "Permalink to this table")
||x86_64 (AMD64)|POWER (ppc64le)|ARM64|
|---|---|---|---|
|up to compute capability 5.3 (Maxwell)|40bit|40bit|40bit|
|compute capability 6.0 (Pascal) or newer|up to 47bit|up to 49bit|up to 48bit|

Note

On devices of compute capability 5.3 (Maxwell) and earlier, the CUDA driver creates an uncommitted 40bit virtual address reservation to ensure that memory allocations (pointers) fall into the supported range. This reservation appears as reserved virtual memory, but does not occupy any physical memory until the program actually allocates memory.

Linear memory is typically allocated using `cudaMalloc()` and freed using `cudaFree()` and data transfer between host memory and device memory are typically done using `cudaMemcpy()`. In the vector addition code sample of [Kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernels), the vectors need to be copied from host memory to device memory:

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

Linear memory can also be allocated through `cudaMallocPitch()` and `cudaMalloc3D()`. These functions are recommended for allocations of 2D or 3D arrays as it makes sure that the allocation is appropriately padded to meet the alignment requirements described in [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses), therefore ensuring best performance when accessing the row addresses or performing copies between 2D arrays and other regions of device memory (using the `cudaMemcpy2D()` and `cudaMemcpy3D()` functions). The returned pitch (or stride) must be used to access array elements. The following code sample allocates a `width` x `height` 2D array of floating-point values and shows how to loop over the array elements in device code:

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

The following code sample allocates a `width` x `height` x `depth` 3D array of floating-point values and shows how to loop over the array elements in device code:

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

Note

To avoid allocating too much memory and thus impacting system-wide performance, request the allocation parameters from the user based on the problem size. If the allocation fails, you can fallback to other slower memory types (`cudaMallocHost()`, `cudaHostRegister()`, etc.), or return an error telling the user how much memory was needed that was denied. If your application cannot request the allocation parameters for some reason, we recommend using `cudaMallocManaged()` for platforms that support it.

The reference manual lists all the various functions used to copy memory between linear memory allocated with `cudaMalloc()`, linear memory allocated with `cudaMallocPitch()` or `cudaMalloc3D()`, CUDA arrays, and memory allocated for variables declared in global or constant memory space.

The following code sample illustrates various ways of accessing global variables via the runtime API:

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

`cudaGetSymbolAddress()` is used to retrieve the address pointing to the memory allocated for a variable declared in global memory space. The size of the allocated memory is obtained through `cudaGetSymbolSize()`.

### 6.2.3. Device Memory L2 Access Management[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-l2-access-management "Permalink to this headline")

When a CUDA kernel accesses a data region in the global memory repeatedly, such data accesses can be considered to be _persisting_. On the other hand, if the data is only accessed once, such data accesses can be considered to be _streaming_.

Starting with CUDA 11.0, devices of compute capability 8.0 and above have the capability to influence persistence of data in the L2 cache, potentially providing higher bandwidth and lower latency accesses to global memory.

#### 6.2.3.1. L2 Cache Set-Aside for Persisting Accesses[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-cache-set-aside-for-persisting-accesses "Permalink to this headline")

A portion of the L2 cache can be set aside to be used for persisting data accesses to global memory. Persisting accesses have prioritized use of this set-aside portion of L2 cache, whereas normal or streaming, accesses to global memory can only utilize this portion of L2 when it is unused by persisting accesses.

The L2 cache set-aside size for persisting accesses may be adjusted, within limits:

cudaGetDeviceProperties(&prop, device_id);
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); /* set-aside 3/4 of L2 cache for persisting accesses or the max allowed*/

When the GPU is configured in Multi-Instance GPU (MIG) mode, the L2 cache set-aside functionality is disabled.

When using the Multi-Process Service (MPS), the L2 cache set-aside size cannot be changed by `cudaDeviceSetLimit`. Instead, the set-aside size can only be specified at start up of MPS server through the environment variable `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`.

#### 6.2.3.2. L2 Policy for Persisting Accesses[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-policy-for-persisting-accesses "Permalink to this headline")

An access policy window specifies a contiguous region of global memory and a persistence property in the L2 cache for accesses within that region.

The code example below shows how to set an L2 persisting access window using a CUDA Stream.

**CUDA Stream Example**

cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                              // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA stream of type cudaStream_t
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

When a kernel subsequently executes in CUDA `stream`, memory accesses within the global memory extent `[ptr..ptr+num_bytes)` are more likely to persist in the L2 cache than accesses to other global memory locations.

L2 persistence can also be set for a CUDA Graph Kernel Node as shown in the example below:

**CUDA GraphKernelNode Example**

cudaKernelNodeAttrValue node_attribute;                                     // Kernel level attributes data structure
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
node_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persistence access.
                                                                            // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
node_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // Hint for cache hit ratio
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.

//Set the attributes to a CUDA Graph Kernel node of type cudaGraphNode_t
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);

The `hitRatio` parameter can be used to specify the fraction of accesses that receive the `hitProp` property. In both of the examples above, 60% of the memory accesses in the global memory region `[ptr..ptr+num_bytes)` have the persisting property and 40% of the memory accesses have the streaming property. Which specific memory accesses are classified as persisting (the `hitProp`) is random with a probability of approximately `hitRatio`; the probability distribution depends upon the hardware architecture and the memory extent.

For example, if the L2 set-aside cache size is 16KB and the `num_bytes` in the `accessPolicyWindow` is 32KB:

- With a `hitRatio` of 0.5, the hardware will select, at random, 16KB of the 32KB window to be designated as persisting and cached in the set-aside L2 cache area.
    
- With a `hitRatio` of 1.0, the hardware will attempt to cache the whole 32KB window in the set-aside L2 cache area. Since the set-aside area is smaller than the window, cache lines will be evicted to keep the most recently used 16KB of the 32KB data in the set-aside portion of the L2 cache.
    

The `hitRatio` can therefore be used to avoid thrashing of cache lines and overall reduce the amount of data moved into and out of the L2 cache.

A `hitRatio` value below 1.0 can be used to manually control the amount of data different `accessPolicyWindow`s from concurrent CUDA streams can cache in L2. For example, let the L2 set-aside cache size be 16KB; two concurrent kernels in two different CUDA streams, each with a 16KB `accessPolicyWindow`, and both with `hitRatio` value 1.0, might evict each others’ cache lines when competing for the shared L2 resource. However, if both `accessPolicyWindows` have a hitRatio value of 0.5, they will be less likely to evict their own or each others’ persisting cache lines.

#### 6.2.3.3. L2 Access Properties[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-access-properties "Permalink to this headline")

Three types of access properties are defined for different global memory data accesses:

1. `cudaAccessPropertyStreaming`: Memory accesses that occur with the streaming property are less likely to persist in the L2 cache because these accesses are preferentially evicted.
    
2. `cudaAccessPropertyPersisting`: Memory accesses that occur with the persisting property are more likely to persist in the L2 cache because these accesses are preferentially retained in the set-aside portion of L2 cache.
    
3. `cudaAccessPropertyNormal`: This access property forcibly resets previously applied persisting access property to a normal status. Memory accesses with the persisting property from previous CUDA kernels may be retained in L2 cache long after their intended use. This persistence-after-use reduces the amount of L2 cache available to subsequent kernels that do not use the persisting property. Resetting an access property window with the `cudaAccessPropertyNormal` property removes the persisting (preferential retention) status of the prior access, as if the prior access had been without an access property.
    

#### 6.2.3.4. L2 Persistence Example[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#l2-persistence-example "Permalink to this headline")

The following example shows how to set-aside L2 cache for persistent accesses, use the set-aside L2 cache in CUDA kernels via CUDA Stream and then reset the L2 cache.

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

#### 6.2.3.5. Reset L2 Access to Normal[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#reset-l2-access-to-normal "Permalink to this headline")

A persisting L2 cache line from a previous CUDA kernel may persist in L2 long after it has been used. Hence, a reset to normal for L2 cache is important for streaming or normal memory accesses to utilize the L2 cache with normal priority. There are three ways a persisting access can be reset to normal status.

1. Reset a previous persisting memory region with the access property, `cudaAccessPropertyNormal`.
    
2. Reset all persisting L2 cache lines to normal by calling `cudaCtxResetPersistingL2Cache()`.
    
3. **Eventually** untouched lines are automatically reset to normal. Reliance on automatic reset is strongly discouraged because of the undetermined length of time required for automatic reset to occur.
    

#### 6.2.3.6. Manage Utilization of L2 set-aside cache[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#manage-utilization-of-l2-set-aside-cache "Permalink to this headline")

Multiple CUDA kernels executing concurrently in different CUDA streams may have a different access policy window assigned to their streams. However, the L2 set-aside cache portion is shared among all these concurrent CUDA kernels. As a result, the net utilization of this set-aside cache portion is the sum of all the concurrent kernels’ individual use. The benefits of designating memory accesses as persisting diminish as the volume of persisting accesses exceeds the set-aside L2 cache capacity.

To manage utilization of the set-aside L2 cache portion, an application must consider the following:

- Size of L2 set-aside cache.
    
- CUDA kernels that may concurrently execute.
    
- The access policy window for all the CUDA kernels that may concurrently execute.
    
- When and how L2 reset is required to allow normal or streaming accesses to utilize the previously set-aside L2 cache with equal priority.
    

#### 6.2.3.7. Query L2 cache Properties[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#query-l2-cache-properties "Permalink to this headline")

Properties related to L2 cache are a part of `cudaDeviceProp` struct and can be queried using CUDA runtime API `cudaGetDeviceProperties`

CUDA Device Properties include:

- `l2CacheSize`: The amount of available L2 cache on the GPU.
    
- `persistingL2CacheMaxSize`: The maximum amount of L2 cache that can be set-aside for persisting memory accesses.
    
- `accessPolicyMaxWindowSize`: The maximum size of the access policy window.
    

#### 6.2.3.8. Control L2 Cache Set-Aside Size for Persisting Memory Access[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#control-l2-cache-set-aside-size-for-persisting-memory-access "Permalink to this headline")

The L2 set-aside cache size for persisting memory accesses is queried using CUDA runtime API `cudaDeviceGetLimit` and set using CUDA runtime API `cudaDeviceSetLimit` as a `cudaLimit`. The maximum value for setting this limit is `cudaDeviceProp::persistingL2CacheMaxSize`.

enum cudaLimit {
    /* other fields not shown */
    cudaLimitPersistingL2CacheSize
};

### 6.2.4. Shared Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory "Permalink to this headline")

As detailed in [Variable Memory Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#variable-memory-space-specifiers) shared memory is allocated using the `__shared__` memory space specifier.

Shared memory is expected to be much faster than global memory as mentioned in [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy) and detailed in [Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory). It can be used as scratchpad memory (or software managed cache) to minimize global memory accesses from a CUDA block as illustrated by the following matrix multiplication example.

The following code sample is a straightforward implementation of matrix multiplication that does not take advantage of shared memory. Each thread reads one row of _A_ and one column of _B_ and computes the corresponding element of _C_ as illustrated in [Figure 8](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-no-shared-memory). _A_ is therefore read _B.width_ times from global memory and _B_ is read _A.height_ times.

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

![_images/matrix-multiplication-without-shared-memory.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-without-shared-memory.png)

Figure 8 Matrix Multiplication without Shared Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-no-shared-memory "Permalink to this image")

The following code sample is an implementation of matrix multiplication that does take advantage of shared memory. In this implementation, each thread block is responsible for computing one square sub-matrix _Csub_ of _C_ and each thread within the block is responsible for computing one element of _Csub_. As illustrated in [Figure 9](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-shared-memory), _Csub_ is equal to the product of two rectangular matrices: the sub-matrix of _A_ of dimension (_A.width, block_size_) that has the same row indices as _Csub_, and the sub-matrix of _B_ of dimension (_block_size, A.width_ )that has the same column indices as _Csub_. In order to fit into the device’s resources, these two rectangular matrices are divided into as many square matrices of dimension _block_size_ as necessary and _Csub_ is computed as the sum of the products of these square matrices. Each of these products is performed by first loading the two corresponding square matrices from global memory to shared memory with one thread loading one element of each matrix, and then by having each thread compute one element of the product. Each thread accumulates the result of each of these products into a register and once done writes the result to global memory.

By blocking the computation this way, we take advantage of fast shared memory and save a lot of global memory bandwidth since _A_ is only read (_B.width / block_size_) times from global memory and _B_ is read (_A.height / block_size_) times.

The _Matrix_ type from the previous code sample is augmented with a _stride_ field, so that sub-matrices can be efficiently represented with the same type. [__device__](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-function-specifier) functions are used to get and set elements and build any sub-matrix from a matrix.

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

![_images/matrix-multiplication-with-shared-memory.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-with-shared-memory.png)

Figure 9 Matrix Multiplication with Shared Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-shared-memory "Permalink to this image")

### 6.2.5. Distributed Shared Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#distributed-shared-memory "Permalink to this headline")

Thread block clusters introduced in compute capability 9.0 provide the ability for threads in a thread block cluster to access shared memory of all the participating thread blocks in a cluster. This partitioned shared memory is called _Distributed Shared Memory_, and the corresponding address space is called Distributed shared memory address space. Threads that belong to a thread block cluster, can read, write or perform atomics in the distributed address space, regardless whether the address belongs to the local thread block or a remote thread block. Whether a kernel uses distributed shared memory or not, the shared memory size specifications, static or dynamic is still per thread block. The size of distributed shared memory is just the number of thread blocks per cluster multiplied by the size of shared memory per thread block.

Accessing data in distributed shared memory requires all the thread blocks to exist. A user can guarantee that all thread blocks have started executing using `cluster.sync()` from [Cluster Group](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cluster-group-cg) API. The user also needs to ensure that all distributed shared memory operations happen before the exit of a thread block, e.g., if a remote thread block is trying to read a given thread block’s shared memory, user needs to ensure that the shared memory read by remote thread block is completed before it can exit.

CUDA provides a mechanism to access to distributed shared memory, and applications can benefit from leveraging its capabilities. Lets look at a simple histogram computation and how to optimize it on the GPU using thread block cluster. A standard way of computing histograms is do the computation in the shared memory of each thread block and then perform global memory atomics. A limitation of this approach is the shared memory capacity. Once the histogram bins no longer fit in the shared memory, a user needs to directly compute histograms and hence the atomics in the global memory. With distributed shared memory, CUDA provides an intermediate step, where a depending on the histogram bins size, histogram can be computed in shared memory, distributed shared memory or global memory directly.

The CUDA kernel example below shows how to compute histograms in shared memory or distributed shared memory, depending on the number of histogram bins.

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

The above kernel can be launched at runtime with a cluster size depending on the amount of distributed shared memory required. If histogram is small enough to fit in shared memory of just one block, user can launch kernel with cluster size 1. The code snippet below shows how to launch a cluster kernel dynamically based depending on shared memory requirements.

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

### 6.2.6. Page-Locked Host Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#page-locked-host-memory "Permalink to this headline")

The runtime provides functions to allow the use of _page-locked_ (also known as _pinned_) host memory (as opposed to regular pageable host memory allocated by `malloc()`):

- `cudaHostAlloc()` and `cudaFreeHost()` allocate and free page-locked host memory;
    
- `cudaHostRegister()` page-locks a range of memory allocated by `malloc()` (see reference manual for limitations).
    

Using page-locked host memory has several benefits:

- Copies between page-locked host memory and device memory can be performed concurrently with kernel execution for some devices as mentioned in [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution).
    
- On some devices, page-locked host memory can be mapped into the address space of the device, eliminating the need to copy it to or from device memory as detailed in [Mapped Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory).
    
- On systems with a front-side bus, bandwidth between host memory and device memory is higher if host memory is allocated as page-locked and even higher if in addition it is allocated as write-combining as described in [Write-Combining Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#write-combining-memory).
    

Note

Page-locked host memory is not cached on non I/O coherent Tegra devices. Also, `cudaHostRegister()` is not supported on non I/O coherent Tegra devices.

The simple zero-copy CUDA sample comes with a detailed document on the page-locked memory APIs.

#### 6.2.6.1. Portable Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#portable-memory "Permalink to this headline")

A block of page-locked memory can be used in conjunction with any device in the system (see [Multi-Device System](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-device-system) for more details on multi-device systems), but by default, the benefits of using page-locked memory described above are only available in conjunction with the device that was current when the block was allocated (and with all devices sharing the same unified address space, if any, as described in [Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space)). To make these advantages available to all devices, the block needs to be allocated by passing the flag `cudaHostAllocPortable` to `cudaHostAlloc()` or page-locked by passing the flag `cudaHostRegisterPortable` to `cudaHostRegister()`.

#### 6.2.6.2. Write-Combining Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#write-combining-memory "Permalink to this headline")

By default page-locked host memory is allocated as cacheable. It can optionally be allocated as _write-combining_ instead by passing flag `cudaHostAllocWriteCombined` to `cudaHostAlloc()`. Write-combining memory frees up the host’s L1 and L2 cache resources, making more cache available to the rest of the application. In addition, write-combining memory is not snooped during transfers across the PCI Express bus, which can improve transfer performance by up to 40%.

Reading from write-combining memory from the host is prohibitively slow, so write-combining memory should in general be used for memory that the host only writes to.

Using CPU atomic instructions on WC memory should be avoided because not all CPU implementations guarantee that functionality.

#### 6.2.6.3. Mapped Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory "Permalink to this headline")

A block of page-locked host memory can also be mapped into the address space of the device by passing flag `cudaHostAllocMapped` to `cudaHostAlloc()` or by passing flag `cudaHostRegisterMapped` to `cudaHostRegister()`. Such a block has therefore in general two addresses: one in host memory that is returned by `cudaHostAlloc()` or `malloc()`, and one in device memory that can be retrieved using `cudaHostGetDevicePointer()` and then used to access the block from within a kernel. The only exception is for pointers allocated with `cudaHostAlloc()` and when a unified address space is used for the host and the device as mentioned in [Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space).

Accessing host memory directly from within a kernel does not provide the same bandwidth as device memory, but does have some advantages:

- There is no need to allocate a block in device memory and copy data between this block and the block in host memory; data transfers are implicitly performed as needed by the kernel;
    
- There is no need to use streams (see [Concurrent Data Transfers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-data-transfers)) to overlap data transfers with kernel execution; the kernel-originated data transfers automatically overlap with kernel execution.
    

Since mapped page-locked memory is shared between host and device however, the application must synchronize memory accesses using streams or events (see [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)) to avoid any potential read-after-write, write-after-read, or write-after-write hazards.

To be able to retrieve the device pointer to any mapped page-locked memory, page-locked memory mapping must be enabled by calling `cudaSetDeviceFlags()` with the `cudaDeviceMapHost` flag before any other CUDA call is performed. Otherwise, `cudaHostGetDevicePointer()` will return an error.

`cudaHostGetDevicePointer()` also returns an error if the device does not support mapped page-locked host memory. Applications may query this capability by checking the `canMapHostMemory` device property (see [Device Enumeration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)), which is equal to 1 for devices that support mapped page-locked host memory.

Note that atomic functions (see [Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions)) operating on mapped page-locked memory are not atomic from the point of view of the host or other devices.

Also note that CUDA runtime requires that 1-byte, 2-byte, 4-byte, 8-byte, and 16-byte naturally aligned loads and stores to host memory initiated from the device are preserved as single accesses from the point of view of the host and other devices. On some platforms, atomics to memory may be broken by the hardware into separate load and store operations. These component load and store operations have the same requirements on preservation of naturally aligned accesses. The CUDA runtime does not support a PCI Express bus topology where a PCI Express bridge splits 8-byte naturally aligned operations and NVIDIA is not aware of any topology that splits 16-byte naturally aligned operations.

### 6.2.7. Memory Synchronization Domains[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-synchronization-domains "Permalink to this headline")

#### 6.2.7.1. Memory Fence Interference[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-interference "Permalink to this headline")

Some CUDA applications may see degraded performance due to memory fence/flush operations waiting on more transactions than those necessitated by the CUDA memory consistency model.

|   |   |   |
|---|---|---|
|__managed__ int x = 0;<br>__device__  cuda::atomic<int, cuda::thread_scope_device> a(0);<br>__managed__ cuda::atomic<int, cuda::thread_scope_system> b(0);|||
|Thread 1 (SM)<br><br>x = 1;<br>a = 1;|Thread 2 (SM)<br><br>while (a != 1) ;<br>assert(x == 1);<br>b = 1;|Thread 3 (CPU)<br><br>while (b != 1) ;<br>assert(x == 1);|

Consider the example above. The CUDA memory consistency model guarantees that the asserted condition will be true, so the write to `x` from thread 1 must be visible to thread 3, before the write to `b` from thread 2.

The memory ordering provided by the release and acquire of `a` is only sufficient to make `x` visible to thread 2, not thread 3, as it is a device-scope operation. The system-scope ordering provided by release and acquire of `b`, therefore, needs to ensure not only writes issued from thread 2 itself are visible to thread 3, but also writes from other threads that are visible to thread 2. This is known as cumulativity. As the GPU cannot know at the time of execution which writes have been guaranteed at the source level to be visible and which are visible only by chance timing, it must cast a conservatively wide net for in-flight memory operations.

This sometimes leads to interference: because the GPU is waiting on memory operations it is not required to at the source level, the fence/flush may take longer than necessary.

Note that fences may occur explicitly as intrinsics or atomics in code, like in the example, or implicitly to implement _synchronizes-with_ relationships at task boundaries.

A common example is when a kernel is performing computation in local GPU memory, and a parallel kernel (e.g. from NCCL) is performing communications with a peer. Upon completion, the local kernel will implicitly flush its writes to satisfy any _synchronizes-with_ relationships to downstream work. This may unnecessarily wait, fully or partially, on slower nvlink or PCIe writes from the communication kernel.

#### 6.2.7.2. Isolating Traffic with Domains[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isolating-traffic-with-domains "Permalink to this headline")

Beginning with Hopper architecture GPUs and CUDA 12.0, the memory synchronization domains feature provides a way to alleviate such interference. In exchange for explicit assistance from code, the GPU can reduce the net cast by a fence operation. Each kernel launch is given a domain ID. Writes and fences are tagged with the ID, and a fence will only order writes matching the fence’s domain. In the concurrent compute vs communication example, the communication kernels can be placed in a different domain.

When using domains, code must abide by the rule that **ordering or synchronization between distinct domains on the same GPU requires system-scope fencing**. Within a domain, device-scope fencing remains sufficient. This is necessary for cumulativity as one kernel’s writes will not be encompassed by a fence issued from a kernel in another domain. In essence, cumulativity is satisfied by ensuring that cross-domain traffic is flushed to the system scope ahead of time.

Note that this modifies the definition of `thread_scope_device`. However, because kernels will default to domain 0 as described below, backward compatibility is maintained.

#### 6.2.7.3. Using Domains in CUDA[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-domains-in-cuda "Permalink to this headline")

Domains are accessible via the new launch attributes `cudaLaunchAttributeMemSyncDomain` and `cudaLaunchAttributeMemSyncDomainMap`. The former selects between logical domains `cudaLaunchMemSyncDomainDefault` and `cudaLaunchMemSyncDomainRemote`, and the latter provides a mapping from logical to physical domains. The remote domain is intended for kernels performing remote memory access in order to isolate their memory traffic from local kernels. Note, however, the selection of a particular domain does not affect what memory access a kernel may legally perform.

The domain count can be queried via device attribute `cudaDevAttrMemSyncDomainCount`. Hopper has 4 domains. To facilitate portable code, domains functionality can be used on all devices and CUDA will report a count of 1 prior to Hopper.

Having logical domains eases application composition. An individual kernel launch at a low level in the stack, such as from NCCL, can select a semantic logical domain without concern for the surrounding application architecture. Higher levels can steer logical domains using the mapping. The default value for the logical domain if it is not set is the default domain, and the default mapping is to map the default domain to 0 and the remote domain to 1 (on GPUs with more than 1 domain). Specific libraries may tag launches with the remote domain in CUDA 12.0 and later; for example, NCCL 2.16 will do so. Together, this provides a beneficial use pattern for common applications out of the box, with no code changes needed in other components, frameworks, or at application level. An alternative use pattern, for example in an application using nvshmem or with no clear separation of kernel types, could be to partition parallel streams. Stream A may map both logical domains to physical domain 0, stream B to 1, and so on.

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

As with other launch attributes, these are exposed uniformly on CUDA streams, individual launches using `cudaLaunchKernelEx`, and kernel nodes in CUDA graphs. A typical use would set the mapping at stream level and the logical domain at launch level (or bracketing a section of stream use) as described above.

Both attributes are copied to graph nodes during stream capture. Graphs take both attributes from the node itself, essentially an indirect way of specifying a physical domain. Domain-related attributes set on the stream a graph is launched into are not used in execution of the graph.

### 6.2.8. Asynchronous Concurrent Execution[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution "Permalink to this headline")

CUDA exposes the following operations as independent tasks that can operate concurrently with one another:

- Computation on the host;
    
- Computation on the device;
    
- Memory transfers from the host to the device;
    
- Memory transfers from the device to the host;
    
- Memory transfers within the memory of a given device;
    
- Memory transfers among devices.
    

The level of concurrency achieved between these operations will depend on the feature set and compute capability of the device as described below.

#### 6.2.8.1. Concurrent Execution between Host and Device[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-execution-between-host-and-device "Permalink to this headline")

Concurrent host execution is facilitated through asynchronous library functions that return control to the host thread before the device completes the requested task. Using asynchronous calls, many device operations can be queued up together to be executed by the CUDA driver when appropriate device resources are available. This relieves the host thread of much of the responsibility to manage the device, leaving it free for other tasks. The following device operations are asynchronous with respect to the host:

- Kernel launches;
    
- Memory copies within a single device’s memory;
    
- Memory copies from host to device of a memory block of 64 KB or less;
    
- Memory copies performed by functions that are suffixed with `Async`;
    
- Memory set function calls.
    

Programmers can globally disable asynchronicity of kernel launches for all CUDA applications running on a system by setting the `CUDA_LAUNCH_BLOCKING` environment variable to 1. This feature is provided for debugging purposes only and should not be used as a way to make production software run reliably.

Kernel launches are synchronous if hardware counters are collected via a profiler (Nsight Compute) unless concurrent kernel profiling is enabled. `Async` memory copies might also be synchronous if they involve host memory that is not page-locked.

#### 6.2.8.2. Concurrent Kernel Execution[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-kernel-execution "Permalink to this headline")

Some devices of compute capability 2.x and higher can execute multiple kernels concurrently. Applications may query this capability by checking the `concurrentKernels` device property (see [Device Enumeration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)), which is equal to 1 for devices that support it.

The maximum number of kernel launches that a device can execute concurrently depends on its compute capability and is listed in [Table 27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability).

A kernel from one CUDA context cannot execute concurrently with a kernel from another CUDA context. The GPU may time slice to provide forward progress to each context. If a user wants to run kernels from multiple process simultaneously on the SM, one must enable MPS.

Kernels that use many textures or a large amount of local memory are less likely to execute concurrently with other kernels.

#### 6.2.8.3. Overlap of Data Transfer and Kernel Execution[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlap-of-data-transfer-and-kernel-execution "Permalink to this headline")

Some devices can perform an asynchronous memory copy to or from the GPU concurrently with kernel execution. Applications may query this capability by checking the `asyncEngineCount` device property (see [Device Enumeration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)), which is greater than zero for devices that support it. If host memory is involved in the copy, it must be page-locked.

It is also possible to perform an intra-device copy simultaneously with kernel execution (on devices that support the `concurrentKernels` device property) and/or with copies to or from the device (for devices that support the `asyncEngineCount` property). Intra-device copies are initiated using the standard memory copy functions with destination and source addresses residing on the same device.

#### 6.2.8.4. Concurrent Data Transfers[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-data-transfers "Permalink to this headline")

Some devices of compute capability 2.x and higher can overlap copies to and from the device. Applications may query this capability by checking the `asyncEngineCount` device property (see [Device Enumeration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)), which is equal to 2 for devices that support it. In order to be overlapped, any host memory involved in the transfers must be page-locked.

#### 6.2.8.5. Streams[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams "Permalink to this headline")

Applications manage the concurrent operations described above through _streams_. A stream is a sequence of commands (possibly issued by different host threads) that execute in order. Different streams, on the other hand, may execute their commands out of order with respect to one another or concurrently; this behavior is not guaranteed and should therefore not be relied upon for correctness (for example, inter-kernel communication is undefined). The commands issued on a stream may execute when all the dependencies of the command are met. The dependencies could be previously launched commands on same stream or dependencies from other streams. The successful completion of synchronize call guarantees that all the commands launched are completed.

##### 6.2.8.5.1. Creation and Destruction of Streams[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-of-streams "Permalink to this headline")

A stream is defined by creating a stream object and specifying it as the stream parameter to a sequence of kernel launches and host `<->` device memory copies. The following code sample creates two streams and allocates an array `hostPtr` of `float` in page-locked memory.

cudaStream_t stream[2];
for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);
float* hostPtr;
cudaMallocHost(&hostPtr, 2 * size);

Each of these streams is defined by the following code sample as a sequence of one memory copy from host to device, one kernel launch, and one memory copy from device to host:

for (int i = 0; i < 2; ++i) {
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
    MyKernel <<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}

Each stream copies its portion of input array `hostPtr` to array `inputDevPtr` in device memory, processes `inputDevPtr` on the device by calling `MyKernel()`, and copies the result `outputDevPtr` back to the same portion of `hostPtr`. [Overlapping Behavior](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlapping-behavior) describes how the streams overlap in this example depending on the capability of the device. Note that `hostPtr` must point to page-locked host memory for any overlap to occur.

Streams are released by calling `cudaStreamDestroy()`.

for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);

In case the device is still doing work in the stream when `cudaStreamDestroy()` is called, the function will return immediately and the resources associated with the stream will be released automatically once the device has completed all work in the stream.

##### 6.2.8.5.2. Default Stream[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#default-stream "Permalink to this headline")

Kernel launches and host `<->` device memory copies that do not specify any stream parameter, or equivalently that set the stream parameter to zero, are issued to the default stream. They are therefore executed in order.

For code that is compiled using the `--default-stream per-thread` compilation flag (or that defines the `CUDA_API_PER_THREAD_DEFAULT_STREAM` macro before including CUDA headers (`cuda.h` and `cuda_runtime.h`)), the default stream is a regular stream and each host thread has its own default stream.

Note

`#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1` cannot be used to enable this behavior when the code is compiled by `nvcc` as `nvcc` implicitly includes `cuda_runtime.h` at the top of the translation unit. In this case the `--default-stream per-thread` compilation flag needs to be used or the `CUDA_API_PER_THREAD_DEFAULT_STREAM` macro needs to be defined with the `-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1` compiler flag.

For code that is compiled using the `--default-stream legacy` compilation flag, the default stream is a special stream called the _NULL stream_ and each device has a single NULL stream used for all host threads. The NULL stream is special as it causes implicit synchronization as described in [Implicit Synchronization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#implicit-synchronization).

For code that is compiled without specifying a `--default-stream` compilation flag, `--default-stream legacy` is assumed as the default.

##### 6.2.8.5.3. Explicit Synchronization[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#explicit-synchronization "Permalink to this headline")

There are various ways to explicitly synchronize streams with each other.

`cudaDeviceSynchronize()` waits until all preceding commands in all streams of all host threads have completed.

`cudaStreamSynchronize()`takes a stream as a parameter and waits until all preceding commands in the given stream have completed. It can be used to synchronize the host with a specific stream, allowing other streams to continue executing on the device.

`cudaStreamWaitEvent()`takes a stream and an event as parameters (see [Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events) for a description of events)and makes all the commands added to the given stream after the call to `cudaStreamWaitEvent()`delay their execution until the given event has completed.

`cudaStreamQuery()`provides applications with a way to know if all preceding commands in a stream have completed.

##### 6.2.8.5.4. Implicit Synchronization[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#implicit-synchronization "Permalink to this headline")

Two operations from different streams cannot run concurrently if any CUDA operation on the NULL stream is submitted in-between them, unless the streams are non-blocking streams (created with the `cudaStreamNonBlocking` flag).

Applications should follow these guidelines to improve their potential for concurrent kernel execution:

- All independent operations should be issued before dependent operations,
    
- Synchronization of any kind should be delayed as long as possible.
    

##### 6.2.8.5.5. Overlapping Behavior[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlapping-behavior "Permalink to this headline")

The amount of execution overlap between two streams depends on the order in which the commands are issued to each stream and whether or not the device supports overlap of data transfer and kernel execution (see [Overlap of Data Transfer and Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlap-of-data-transfer-and-kernel-execution)), concurrent kernel execution (see [Concurrent Kernel Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-kernel-execution)), and/or concurrent data transfers (see [Concurrent Data Transfers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#concurrent-data-transfers)).

For example, on devices that do not support concurrent data transfers, the two streams of the code sample of [Creation and Destruction of Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-streams) do not overlap at all because the memory copy from host to device is issued to stream[1] after the memory copy from device to host is issued to stream[0], so it can only start once the memory copy from device to host issued to stream[0] has completed. If the code is rewritten the following way (and assuming the device supports overlap of data transfer and kernel execution)

for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size,
                    size, cudaMemcpyHostToDevice, stream[i]);
for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>
          (outputDevPtr + i * size, inputDevPtr + i * size, size);
for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);

then the memory copy from host to device issued to stream[1] overlaps with the kernel launch issued to stream[0].

On devices that do support concurrent data transfers, the two streams of the code sample of [Creation and Destruction of Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-streams) do overlap: The memory copy from host to device issued to stream[1] overlaps with the memory copy from device to host issued to stream[0] and even with the kernel launch issued to stream[0] (assuming the device supports overlap of data transfer and kernel execution).

##### 6.2.8.5.6. Host Functions (Callbacks)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#host-functions-callbacks "Permalink to this headline")

The runtime provides a way to insert a CPU function call at any point into a stream via `cudaLaunchHostFunc()`. The provided function is executed on the host once all commands issued to the stream before the callback have completed.

The following code sample adds the host function `MyCallback` to each of two streams after issuing a host-to-device memory copy, a kernel launch and a device-to-host memory copy into each stream. The function will begin execution on the host after each of the device-to-host memory copies completes.

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

The commands that are issued in a stream after a host function do not start executing before the function has completed.

A host function enqueued into a stream must not make CUDA API calls (directly or indirectly), as it might end up waiting on itself if it makes such a call leading to a deadlock.

##### 6.2.8.5.7. Stream Priorities[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-priorities "Permalink to this headline")

The relative priorities of streams can be specified at creation using `cudaStreamCreateWithPriority()`. The range of allowable priorities, ordered as [ greatest priority, least priority ] can be obtained using the `cudaDeviceGetStreamPriorityRange()` function. At runtime, the GPU scheduler utilizes stream priorities to determine task execution order, but these priorities serve as hints rather than guarantees. When selecting work to launch, pending tasks in higher-priority streams take precedence over those in lower-priority streams. Higher-priority tasks do not preempt already running lower-priority tasks. The GPU does not reassess work queues during task execution, and increasing a stream’s priority will not interrupt ongoing work. Stream priorities influence task execution without enforcing strict ordering, so users can leverage stream priorities to influence task execution without relying on strict ordering guarantees.

The following code sample obtains the allowable range of priorities for the current device, and creates streams with the highest and lowest available priorities.

// get the range of stream priorities for this device
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
// create streams with highest and lowest available priorities
cudaStream_t st_high, st_low;
cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, greatestPriority));
cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, leastPriority);

#### 6.2.8.6. Programmatic Dependent Launch and Synchronization[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization "Permalink to this headline")

The _Programmatic Dependent Launch_ mechanism allows for a dependent _secondary_ kernel to launch before the _primary_ kernel it depends on in the same CUDA stream has finished executing. Available starting with devices of compute capability 9.0, this technique can provide performance benefits when the _secondary_ kernel can complete significant work that does not depend on the results of the _primary_ kernel.

##### 6.2.8.6.1. Background[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#background "Permalink to this headline")

A CUDA application utilizes the GPU by launching and executing multiple kernels on it. A typical GPU activity timeline is shown in [Figure 10](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#gpu-activity).

[![GPU activity timeline](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-activity.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-activity.png)

Figure 10 GPU activity timeline[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#gpu-activity "Permalink to this image")

Here, `secondary_kernel` is launched after `primary_kernel` finishes its execution. Serialized execution is usually necessary because `secondary_kernel` depends on result data produced by `primary_kernel`. If `secondary_kernel` has no dependency on `primary_kernel`, both of them can be launched concurrently by using [Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams). Even if `secondary_kernel` is dependent on `primary_kernel`, there is some potential for concurrent execution. For example, almost all the kernels have some sort of _preamble_ section during which tasks such as zeroing buffers or loading constant values are performed.

[![Preamble section of ``secondary_kernel``](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/secondary-kernel-preamble.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/secondary-kernel-preamble.png)

Figure 11 Preamble section of `secondary_kernel`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#secondary-kernel-preamble "Permalink to this image")

[Figure 11](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#secondary-kernel-preamble) demonstrates the portion of `secondary_kernel` that could be executed concurrently without impacting the application. Note that concurrent launch also allows us to hide the launch latency of `secondary_kernel` behind the execution of `primary_kernel`.

[![Concurrent execution of ``primary_kernel`` and ``secondary_kernel``](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/preamble-overlap.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/preamble-overlap.png)

Figure 12 Concurrent execution of `primary_kernel` and `secondary_kernel`[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#preamble-overlap "Permalink to this image")

The concurrent launch and execution of `secondary_kernel` shown in [Figure 12](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#preamble-overlap) is achievable using _Programmatic Dependent Launch_.

_Programmatic Dependent Launch_ introduces changes to the CUDA kernel launch APIs as explained in following section. These APIs require at least compute capability 9.0 to provide overlapping execution.

##### 6.2.8.6.2. API Description[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#api-description "Permalink to this headline")

In Programmatic Dependent Launch, a primary and a secondary kernel are launched in the same CUDA stream. The primary kernel should execute `cudaTriggerProgrammaticLaunchCompletion` with all thread blocks when it’s ready for the secondary kernel to launch. The secondary kernel must be launched using the extensible launch API as shown.

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

When the secondary kernel is launched using the `cudaLaunchAttributeProgrammaticStreamSerialization` attribute, the CUDA driver is safe to launch the secondary kernel early and not wait on the completion and memory flush of the primary before launching the secondary.

The CUDA driver can launch the secondary kernel when all primary thread blocks have launched and executed `cudaTriggerProgrammaticLaunchCompletion`. If the primary kernel doesn’t execute the trigger, it implicitly occurs after all thread blocks in the primary kernel exit.

In either case, the secondary thread blocks might launch before data written by the primary kernel is visible. As such, when the secondary kernel is configured with _Programmatic Dependent Launch_, it must always use `cudaGridDependencySynchronize` or other means to verify that the result data from the primary is available.

Please note that these methods provide the opportunity for the primary and secondary kernels to execute concurrently, however this behavior is opportunistic and not guaranteed to lead to concurrent kernel execution. Reliance on concurrent execution in this manner is unsafe and can lead to deadlock.

##### 6.2.8.6.3. Use in CUDA Graphs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#use-in-cuda-graphs "Permalink to this headline")

Programmatic Dependent Launch can be used in [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs) via [stream capture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture) or directly via [edge data](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#edge-data). To program this feature in a CUDA Graph with edge data, use a `cudaGraphDependencyType` value of `cudaGraphDependencyTypeProgrammatic` on an edge connecting two kernel nodes. This edge type makes the upstream kernel visible to a `cudaGridDependencySynchronize()` in the downstream kernel. This type must be used with an outgoing port of either `cudaGraphKernelNodePortLaunchCompletion` or `cudaGraphKernelNodePortProgrammatic`.

The resulting graph equivalents for stream capture are as follows:

|Stream code (abbreviated)|Resulting graph edge|
|---|---|
|cudaLaunchAttribute attribute;<br>attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;<br>attribute.val.programmaticStreamSerializationAllowed = 1;|cudaGraphEdgeData edgeData;<br>edgeData.type = cudaGraphDependencyTypeProgrammatic;<br>edgeData.from_port = cudaGraphKernelNodePortProgrammatic;|
|cudaLaunchAttribute attribute;<br>attribute.id = cudaLaunchAttributeProgrammaticEvent;<br>attribute.val.programmaticEvent.triggerAtBlockStart = 0;|cudaGraphEdgeData edgeData;<br>edgeData.type = cudaGraphDependencyTypeProgrammatic;<br>edgeData.from_port = cudaGraphKernelNodePortProgrammatic;|
|cudaLaunchAttribute attribute;<br>attribute.id = cudaLaunchAttributeProgrammaticEvent;<br>attribute.val.programmaticEvent.triggerAtBlockStart = 1;|cudaGraphEdgeData edgeData;<br>edgeData.type = cudaGraphDependencyTypeProgrammatic;<br>edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion;|

#### 6.2.8.7. CUDA Graphs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs "Permalink to this headline")

CUDA Graphs present a new model for work submission in CUDA. A graph is a series of operations, such as kernel launches, connected by dependencies, which is defined separately from its execution. This allows a graph to be defined once and then launched repeatedly. Separating out the definition of a graph from its execution enables a number of optimizations: first, CPU launch costs are reduced compared to streams, because much of the setup is done in advance; second, presenting the whole workflow to CUDA enables optimizations which might not be possible with the piecewise work submission mechanism of streams.

To see the optimizations possible with graphs, consider what happens in a stream: when you place a kernel into a stream, the host driver performs a sequence of operations in preparation for the execution of the kernel on the GPU. These operations, necessary for setting up and launching the kernel, are an overhead cost which must be paid for each kernel that is issued. For a GPU kernel with a short execution time, this overhead cost can be a significant fraction of the overall end-to-end execution time.

Work submission using graphs is separated into three distinct stages: definition, instantiation, and execution.

- During the definition phase, a program creates a description of the operations in the graph along with the dependencies between them.
    
- Instantiation takes a snapshot of the graph template, validates it, and performs much of the setup and initialization of work with the aim of minimizing what needs to be done at launch. The resulting instance is known as an _executable graph._
    
- An executable graph may be launched into a stream, similar to any other CUDA work. It may be launched any number of times without repeating the instantiation.
    

##### 6.2.8.7.1. Graph Structure[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-structure "Permalink to this headline")

An operation forms a node in a graph. The dependencies between the operations are the edges. These dependencies constrain the execution sequence of the operations.

An operation may be scheduled at any time once the nodes on which it depends are complete. Scheduling is left up to the CUDA system.

###### 6.2.8.7.1.1. Node Types[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#node-types "Permalink to this headline")

A graph node can be one of:

- kernel
    
- CPU function call
    
- memory copy
    
- memset
    
- empty node
    
- waiting on an [event](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events)
    
- recording an [event](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events)
    
- signalling an [external semaphore](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#external-resource-interoperability)
    
- waiting on an [external semaphore](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#external-resource-interoperability)
    
- [conditional node](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-graph-nodes)
    
- [Graph Memory Nodes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-memory-nodes)
    
- child graph: To execute a separate nested graph, as shown in the following figure.
    

[![Child Graph Example](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/child-graph.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/child-graph.png)

Figure 13 Child Graph Example[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#node-types-fig-child-graph "Permalink to this image")

###### 6.2.8.7.1.2. Edge Data[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#edge-data "Permalink to this headline")

CUDA 12.3 introduced edge data on CUDA Graphs. Edge data modifies a dependency specified by an edge and consists of three parts: an outgoing port, an incoming port, and a type. An outgoing port specifies when an associated edge is triggered. An incoming port specifies what portion of a node is dependent on an associated edge. A type modifies the relation between the endpoints.

Port values are specific to node type and direction, and edge types may be restricted to specific node types. In all cases, zero-initialized edge data represents default behavior. Outgoing port 0 waits on an entire task, incoming port 0 blocks an entire task, and edge type 0 is associated with a full dependency with memory synchronizing behavior.

Edge data is optionally specified in various graph APIs via a parallel array to the associated nodes. If it is omitted as an input parameter, zero-initialized data is used. If it is omitted as an output (query) parameter, the API accepts this if the edge data being ignored is all zero-initialized, and returns `cudaErrorLossyQuery` if the call would discard information.

Edge data is also available in some stream capture APIs: `cudaStreamBeginCaptureToGraph()`, `cudaStreamGetCaptureInfo()`, and `cudaStreamUpdateCaptureDependencies()`. In these cases, there is not yet a downstream node. The data is associated with a dangling edge (half edge) which will either be connected to a future captured node or discarded at termination of stream capture. Note that some edge types do not wait on full completion of the upstream node. These edges are ignored when considering if a stream capture has been fully rejoined to the origin stream, and cannot be discarded at the end of capture. See [Creating a Graph Using Stream Capture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture).

Currently, no node types define additional incoming ports, and only kernel nodes define additional outgoing ports. There is one non-default dependency type, `cudaGraphDependencyTypeProgrammatic`, which enables [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization) between two kernel nodes.

##### 6.2.8.7.2. Creating a Graph Using Graph APIs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis "Permalink to this headline")

Graphs can be created via two mechanisms: explicit API and stream capture. The following is an example of creating and executing the below graph.

[![Creating a Graph Using Graph APIs Example](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/create-a-graph.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/create-a-graph.png)

Figure 14 Creating a Graph Using Graph APIs Example[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-api-fig-creating-using-graph-apis "Permalink to this image")

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

##### 6.2.8.7.3. Creating a Graph Using Stream Capture[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture "Permalink to this headline")

Stream capture provides a mechanism to create a graph from existing stream-based APIs. A section of code which launches work into streams, including existing code, can be bracketed with calls to `cudaStreamBeginCapture()` and `cudaStreamEndCapture()`. See below.

cudaGraph_t graph;

cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);

A call to `cudaStreamBeginCapture()` places a stream in capture mode. When a stream is being captured, work launched into the stream is not enqueued for execution. It is instead appended to an internal graph that is progressively being built up. This graph is then returned by calling `cudaStreamEndCapture()`, which also ends capture mode for the stream. A graph which is actively being constructed by stream capture is referred to as a _capture graph._

Stream capture can be used on any CUDA stream except `cudaStreamLegacy` (the “NULL stream”). Note that it _can_ be used on `cudaStreamPerThread`. If a program is using the legacy stream, it may be possible to redefine stream 0 to be the per-thread stream with no functional change. See [Default Stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#default-stream).

Whether a stream is being captured can be queried with `cudaStreamIsCapturing()`.

Work can be captured to an existing graph using `cudaStreamBeginCaptureToGraph()`. Instead of capturing to an internal graph, work is captured to a graph provided by the user.

###### 6.2.8.7.3.1. Cross-stream Dependencies and Events[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cross-stream-dependencies-and-events "Permalink to this headline")

Stream capture can handle cross-stream dependencies expressed with `cudaEventRecord()` and `cudaStreamWaitEvent()`, provided the event being waited upon was recorded into the same capture graph.

When an event is recorded in a stream that is in capture mode, it results in a _captured event._ A captured event represents a set of nodes in a capture graph.

When a captured event is waited on by a stream, it places the stream in capture mode if it is not already, and the next item in the stream will have additional dependencies on the nodes in the captured event. The two streams are then being captured to the same capture graph.

When cross-stream dependencies are present in stream capture, `cudaStreamEndCapture()` must still be called in the same stream where `cudaStreamBeginCapture()` was called; this is the _origin stream_. Any other streams which are being captured to the same capture graph, due to event-based dependencies, must also be joined back to the origin stream. This is illustrated below. All streams being captured to the same capture graph are taken out of capture mode upon `cudaStreamEndCapture()`. Failure to rejoin to the origin stream will result in failure of the overall capture operation.

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

Graph returned by the above code is shown in [Figure 14](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-api-fig-creating-using-graph-apis).

Note

When a stream is taken out of capture mode, the next non-captured item in the stream (if any) will still have a dependency on the most recent prior non-captured item, despite intermediate items having been removed.

###### 6.2.8.7.3.2. Prohibited and Unhandled Operations[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#prohibited-and-unhandled-operations "Permalink to this headline")

It is invalid to synchronize or query the execution status of a stream which is being captured or a captured event, because they do not represent items scheduled for execution. It is also invalid to query the execution status of or synchronize a broader handle which encompasses an active stream capture, such as a device or context handle when any associated stream is in capture mode.

When any stream in the same context is being captured, and it was not created with `cudaStreamNonBlocking`, any attempted use of the legacy stream is invalid. This is because the legacy stream handle at all times encompasses these other streams; enqueueing to the legacy stream would create a dependency on the streams being captured, and querying it or synchronizing it would query or synchronize the streams being captured.

It is therefore also invalid to call synchronous APIs in this case. Synchronous APIs, such as `cudaMemcpy()`, enqueue work to the legacy stream and synchronize it before returning.

Note

As a general rule, when a dependency relation would connect something that is captured with something that was not captured and instead enqueued for execution, CUDA prefers to return an error rather than ignore the dependency. An exception is made for placing a stream into or out of capture mode; this severs a dependency relation between items added to the stream immediately before and after the mode transition.

It is invalid to merge two separate capture graphs by waiting on a captured event from a stream which is being captured and is associated with a different capture graph than the event. It is invalid to wait on a non-captured event from a stream which is being captured without specifying the cudaEventWaitExternal flag.

A small number of APIs that enqueue asynchronous operations into streams are not currently supported in graphs and will return an error if called with a stream which is being captured, such as `cudaStreamAttachMemAsync()`.

###### 6.2.8.7.3.3. Invalidation[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#invalidation "Permalink to this headline")

When an invalid operation is attempted during stream capture, any associated capture graphs are _invalidated_. When a capture graph is invalidated, further use of any streams which are being captured or captured events associated with the graph is invalid and will return an error, until stream capture is ended with `cudaStreamEndCapture()`. This call will take the associated streams out of capture mode, but will also return an error value and a NULL graph.

##### 6.2.8.7.4. CUDA User Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-user-objects "Permalink to this headline")

CUDA User Objects can be used to help manage the lifetime of resources used by asynchronous work in CUDA. In particular, this feature is useful for [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs) and [stream capture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture).

Various resource management schemes are not compatible with CUDA graphs. Consider for example an event-based pool or a synchronous-create, asynchronous-destroy scheme.

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

These schemes are difficult with CUDA graphs because of the non-fixed pointer or handle for the resource which requires indirection or graph update, and the synchronous CPU code needed each time the work is submitted. They also do not work with stream capture if these considerations are hidden from the caller of the library, and because of use of disallowed APIs during capture. Various solutions exist such as exposing the resource to the caller. CUDA user objects present another approach.

A CUDA user object associates a user-specified destructor callback with an internal refcount, similar to C++ `shared_ptr`. References may be owned by user code on the CPU and by CUDA graphs. Note that for user-owned references, unlike C++ smart pointers, there is no object representing the reference; users must track user-owned references manually. A typical use case would be to immediately move the sole user-owned reference to a CUDA graph after the user object is created.

When a reference is associated to a CUDA graph, CUDA will manage the graph operations automatically. A cloned `cudaGraph_t` retains a copy of every reference owned by the source `cudaGraph_t`, with the same multiplicity. An instantiated `cudaGraphExec_t` retains a copy of every reference in the source `cudaGraph_t`. When a `cudaGraphExec_t` is destroyed without being synchronized, the references are retained until the execution is completed.

Here is an example use.

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

References owned by graphs in child graph nodes are associated to the child graphs, not the parents. If a child graph is updated or deleted, the references change accordingly. If an executable graph or child graph is updated with `cudaGraphExecUpdate` or `cudaGraphExecChildGraphNodeSetParams`, the references in the new source graph are cloned and replace the references in the target graph. In either case, if previous launches are not synchronized, any references which would be released are held until the launches have finished executing.

There is not currently a mechanism to wait on user object destructors via a CUDA API. Users may signal a synchronization object manually from the destructor code. In addition, it is not legal to call CUDA APIs from the destructor, similar to the restriction on `cudaLaunchHostFunc`. This is to avoid blocking a CUDA internal shared thread and preventing forward progress. It is legal to signal another thread to perform an API call, if the dependency is one way and the thread doing the call cannot block forward progress of CUDA work.

User objects are created with `cudaUserObjectCreate`, which is a good starting point to browse related APIs.

##### 6.2.8.7.5. Updating Instantiated Graphs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#updating-instantiated-graphs "Permalink to this headline")

Work submission using graphs is separated into three distinct stages: definition, instantiation, and execution. In situations where the workflow is not changing, the overhead of definition and instantiation can be amortized over many executions, and graphs provide a clear advantage over streams.

A graph is a snapshot of a workflow, including kernels, parameters, and dependencies, in order to replay it as rapidly and efficiently as possible. In situations where the workflow changes the graph becomes out of date and must be modified. Major changes to graph structure such as topology or types of nodes will require re-instantiation of the source graph because various topology-related optimization techniques must be re-applied.

The cost of repeated instantiation can reduce the overall performance benefit from graph execution, but it is common for only node parameters, such as kernel parameters and `cudaMemcpy` addresses, to change while graph topology remains the same. For this case, CUDA provides a lightweight mechanism known as “Graph Update,” which allows certain node parameters to be modified in-place without having to rebuild the entire graph. This is much more efficient than re-instantiation.

Updates will take effect the next time the graph is launched, so they will not impact previous graph launches, even if they are running at the time of the update. A graph may be updated and relaunched repeatedly, so multiple updates/launches can be queued on a stream.

CUDA provides two mechanisms for updating instantiated graph parameters, whole graph update and individual node update. Whole graph update allows the user to supply a topologically identical `cudaGraph_t` object whose nodes contain updated parameters. Individual node update allows the user to explicitly update the parameters of individual nodes. Using an updated `cudaGraph_t` is more convenient when a large number of nodes are being updated, or when the graph topology is unknown to the caller (i.e., The graph resulted from stream capture of a library call). Using individual node update is preferred when the number of changes is small and the user has the handles to the nodes requiring updates. Individual node update skips the topology checks and comparisons for unchanged nodes, so it can be more efficient in many cases.

CUDA also provides a mechanism for enabling and disabling individual nodes without affecting their current parameters.

The following sections explain each approach in more detail.

###### 6.2.8.7.5.1. Graph Update Limitations[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-update-limitations "Permalink to this headline")

Kernel nodes:

- The owning context of the function cannot change.
    
- A node whose function originally did not use CUDA dynamic parallelism cannot be updated to a function which uses CUDA dynamic parallelism.
    

`cudaMemset` and `cudaMemcpy` nodes:

- The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.
    
- The source/destination memory must be allocated from the same context as the original source/destination memory.
    
- Only 1D `cudaMemset`/`cudaMemcpy` nodes can be changed.
    

Additional memcpy node restrictions:

- Changing either the source or destination memory type (i.e., `cudaPitchedPtr`, `cudaArray_t`, etc.), or the type of transfer (i.e., `cudaMemcpyKind`) is not supported.
    

External semaphore wait nodes and record nodes:

- Changing the number of semaphores is not supported.
    

Conditional nodes:

- The order of handle creation and assignment must match between the graphs.
    
- Changing node parameters is not supported (i.e. number of graphs in the conditional, node context, etc).
    
- Changing parameters of nodes within the conditional body graph is subject to the rules above.
    

Memory nodes:

- It is not possible to update a `cudaGraphExec_t` with a `cudaGraph_t` if the `cudaGraph_t` is currently instantiated as a different `cudaGraphExec_t`.
    

There are no restrictions on updates to host nodes, event record nodes, or event wait nodes.

###### 6.2.8.7.5.2. Whole Graph Update[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#whole-graph-update "Permalink to this headline")

`cudaGraphExecUpdate()` allows an instantiated graph (the “original graph”) to be updated with the parameters from a topologically identical graph (the “updating” graph). The topology of the updating graph must be identical to the original graph used to instantiate the `cudaGraphExec_t`. In addition, the order in which the dependencies are specified must match. Finally, CUDA needs to consistently order the sink nodes (nodes with no dependencies). CUDA relies on the order of specific api calls to achieve consistent sink node ordering.

More explicitly, following the following rules will cause `cudaGraphExecUpdate()` to pair the nodes in the original graph and the updating graph deterministically:

1. For any capturing stream, the API calls operating on that stream must be made in the same order, including event wait and other api calls not directly corresponding to node creation.
    
2. The API calls which directly manipulate a given graph node’s incoming edges (including captured stream APIs, node add APIs, and edge addition / removal APIs) must be made in the same order. Moreover, when dependencies are specified in arrays to these APIs, the order in which the dependencies are specified inside those arrays must match.
    
3. Sink nodes must be consistently ordered. Sink nodes are nodes without dependent nodes / outgoing edges in the final graph at the time of the `cudaGraphExecUpdate()` invocation. The following operations affect sink node ordering (if present) and must (as a combined set) be made in the same order:
    
    - Node add APIs resulting in a sink node.
        
    - Edge removal resulting in a node becoming a sink node.
        
    - `cudaStreamUpdateCaptureDependencies()`, if it removes a sink node from a capturing stream’s dependency set.
        
    - `cudaStreamEndCapture()`.
        

The following example shows how the API could be used to update an instantiated graph:

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

A typical workflow is to create the initial `cudaGraph_t` using either the stream capture or graph API. The `cudaGraph_t` is then instantiated and launched as normal. After the initial launch, a new `cudaGraph_t` is created using the same method as the initial graph and `cudaGraphExecUpdate()` is called. If the graph update is successful, indicated by the `updateResult` parameter in the above example, the updated `cudaGraphExec_t` is launched. If the update fails for any reason, the `cudaGraphExecDestroy()` and `cudaGraphInstantiate()` are called to destroy the original `cudaGraphExec_t` and instantiate a new one.

It is also possible to update the `cudaGraph_t` nodes directly (i.e., Using `cudaGraphKernelNodeSetParams()`) and subsequently update the `cudaGraphExec_t`, however it is more efficient to use the explicit node update APIs covered in the next section.

Conditional handle flags and default values are updated as part of the graph update.

Please see the [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH) for more information on usage and current limitations.

###### 6.2.8.7.5.3. Individual Node Update[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#individual-node-update "Permalink to this headline")

Instantiated graph node parameters can be updated directly. This eliminates the overhead of instantiation as well as the overhead of creating a new `cudaGraph_t`. If the number of nodes requiring update is small relative to the total number of nodes in the graph, it is better to update the nodes individually. The following methods are available for updating `cudaGraphExec_t` nodes:

- `cudaGraphExecKernelNodeSetParams()`
    
- `cudaGraphExecMemcpyNodeSetParams()`
    
- `cudaGraphExecMemsetNodeSetParams()`
    
- `cudaGraphExecHostNodeSetParams()`
    
- `cudaGraphExecChildGraphNodeSetParams()`
    
- `cudaGraphExecEventRecordNodeSetEvent()`
    
- `cudaGraphExecEventWaitNodeSetEvent()`
    
- `cudaGraphExecExternalSemaphoresSignalNodeSetParams()`
    
- `cudaGraphExecExternalSemaphoresWaitNodeSetParams()`
    

Please see the [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH) for more information on usage and current limitations.

###### 6.2.8.7.5.4. Individual Node Enable[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#individual-node-enable "Permalink to this headline")

Kernel, memset and memcpy nodes in an instantiated graph can be enabled or disabled using the `cudaGraphNodeSetEnabled()` API. This allows the creation of a graph which contains a superset of the desired functionality which can be customized for each launch. The enable state of a node can be queried using the `cudaGraphNodeGetEnabled()` API.

A disabled node is functionally equivalent to empty node until it is reenabled. Node parameters are not affected by enabling/disabling a node. Enable state is unaffected by individual node update or whole graph update with `cudaGraphExecUpdate()`. Parameter updates while the node is disabled will take effect when the node is reenabled.

The following methods are available for enabling/disabling `cudaGraphExec_t` nodes, as well as querying their status:

- `cudaGraphNodeSetEnabled()`
    
- `cudaGraphNodeGetEnabled()`
    

Refer to the [Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH) for more information on usage and current limitations.

##### 6.2.8.7.6. Using Graph APIs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-graph-apis "Permalink to this headline")

`cudaGraph_t` objects are not thread-safe. It is the responsibility of the user to ensure that multiple threads do not concurrently access the same `cudaGraph_t`.

A `cudaGraphExec_t` cannot run concurrently with itself. A launch of a `cudaGraphExec_t` will be ordered after previous launches of the same executable graph.

Graph execution is done in streams for ordering with other asynchronous work. However, the stream is for ordering only; it does not constrain the internal parallelism of the graph, nor does it affect where graph nodes execute.

See [Graph API.](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)

##### 6.2.8.7.7. Device Graph Launch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-launch "Permalink to this headline")

There are many workflows which need to make data-dependent decisions during runtime and execute different operations depending on those decisions. Rather than offloading this decision-making process to the host, which may require a round-trip from the device, users may prefer to perform it on the device. To that end, CUDA provides a mechanism to launch graphs from the device.

Device graph launch provides a convenient way to perform dynamic control flow from the device, be it something as simple as a loop or as complex as a device-side work scheduler. This functionality is only available on systems which support [unified addressing](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space).

Graphs which can be launched from the device will henceforth be referred to as device graphs, and graphs which cannot be launched from the device will be referred to as host graphs.

Device graphs can be launched from both the host and device, whereas host graphs can only be launched from the host. Unlike host launches, launching a device graph from the device while a previous launch of the graph is running will result in an error, returning `cudaErrorInvalidValue`; therefore, a device graph cannot be launched twice from the device at the same time. Launching a device graph from the host and device simultaneously will result in undefined behavior.

###### 6.2.8.7.7.1. Device Graph Creation[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-creation "Permalink to this headline")

In order for a graph to be launched from the device, it must be instantiated explicitly for device launch. This is achieved by passing the `cudaGraphInstantiateFlagDeviceLaunch` flag to the `cudaGraphInstantiate()` call. As is the case for host graphs, device graph structure is fixed at time of instantiation and cannot be updated without re-instantiation, and instantiation can only be performed on the host. In order for a graph to be able to be instantiated for device launch, it must adhere to various requirements.

6.2.8.7.7.1.1. Device Graph Requirements[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-requirements "Permalink to this headline")

General requirements:

- The graph’s nodes must all reside on a single device.
    
- The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.
    

Kernel nodes:

- Use of CUDA Dynamic Parallelism by kernels in the graph is not permitted.
    
- Cooperative launches are permitted so long as MPS is not in use.
    

Memcpy nodes:

- Only copies involving device memory and/or pinned device-mapped host memory are permitted.
    
- Copies involving CUDA arrays are not permitted.
    
- Both operands must be accessible from the current device at time of instantiation. Note that the copy operation will be performed from the device on which the graph resides, even if it is targeting memory on another device.
    

6.2.8.7.7.1.2. Device Graph Upload[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-upload "Permalink to this headline")

In order to launch a graph on the device, it must first be uploaded to the device to populate the necessary device resources. This can be achieved in one of two ways.

Firstly, the graph can be uploaded explicitly, either via `cudaGraphUpload()` or by requesting an upload as part of instantiation via `cudaGraphInstantiateWithParams()`.

Alternatively, the graph can first be launched from the host, which will perform this upload step implicitly as part of the launch.

Examples of all three methods can be seen below:

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

6.2.8.7.7.1.3. Device Graph Update[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-graph-update "Permalink to this headline")

Device graphs can only be updated from the host, and must be re-uploaded to the device upon executable graph update in order for the changes to take effect. This can be achieved using the same methods outlined in the previous section. Unlike host graphs, launching a device graph from the device while an update is being applied will result in undefined behavior.

###### 6.2.8.7.7.2. Device Launch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-launch "Permalink to this headline")

Device graphs can be launched from both the host and the device via `cudaGraphLaunch()`, which has the same signature on the device as on the host. Device graphs are launched via the same handle on the host and the device. Device graphs must be launched from another graph when launched from the device.

Device-side graph launch is per-thread and multiple launches may occur from different threads at the same time, so the user will need to select a single thread from which to launch a given graph.

6.2.8.7.7.2.1. Device Launch Modes[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-launch-modes "Permalink to this headline")

Unlike host launch, device graphs cannot be launched into regular CUDA streams, and can only be launched into distinct named streams, which each denote a specific launch mode:

Table 5 Device-only Graph Launch Streams[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id456 "Permalink to this table")
|Stream|Launch Mode|
|---|---|
|`cudaStreamGraphFireAndForget`|Fire and forget launch|
|`cudaStreamGraphTailLaunch`|Tail launch|
|`cudaStreamGraphFireAndForgetAsSibling`|Sibling launch|

6.2.8.7.7.2.1.1. Fire and Forget Launch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fire-and-forget-launch "Permalink to this headline")

As the name suggests, a fire and forget launch is submitted to the GPU immediately, and it runs independently of the launching graph. In a fire-and-forget scenario, the launching graph is the parent, and the launched graph is the child.

[![_images/fire-and-forget-simple.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-simple.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-simple.png)

Figure 15 Fire and forget launch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id457 "Permalink to this image")

The above diagram can be generated by the sample code below:

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

A graph can have up to 120 total fire-and-forget graphs during the course of its execution. This total resets between launches of the same parent graph.

6.2.8.7.7.2.1.2. Graph Execution Environments[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graph-execution-environments "Permalink to this headline")

In order to fully understand the device-side synchronization model, it is first necessary to understand the concept of an execution environment.

When a graph is launched from the device, it is launched into its own execution environment. The execution environment of a given graph encapsulates all work in the graph as well as all generated fire and forget work. The graph can be considered complete when it has completed execution and when all generated child work is complete.

The below diagram shows the environment encapsulation that would be generated by the fire-and-forget sample code in the previous section.

[![_images/fire-and-forget-environments.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-environments.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-environments.png)

Figure 16 Fire and forget launch, with execution environments[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id458 "Permalink to this image")

These environments are also hierarchical, so a graph environment can include multiple levels of child-environments from fire and forget launches.

[![_images/fire-and-forget-nested-environments.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-nested-environments.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-nested-environments.png)

Figure 17 Nested fire and forget environments[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id459 "Permalink to this image")

When a graph is launched from the host, there exists a stream environment that parents the execution environment of the launched graph. The stream environment encapsulates all work generated as part of the overall launch. The stream launch is complete (i.e. downstream dependent work may now run) when the overall stream environment is marked as complete.

[![_images/device-graph-stream-environment.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/device-graph-stream-environment.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/device-graph-stream-environment.png)

Figure 18 The stream environment, visualized[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id460 "Permalink to this image")

6.2.8.7.7.2.1.3. Tail Launch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tail-launch "Permalink to this headline")

Unlike on the host, it is not possible to synchronize with device graphs from the GPU via traditional methods such as `cudaDeviceSynchronize()` or `cudaStreamSynchronize()`. Rather, in order to enable serial work dependencies, a different launch mode - tail launch - is offered, to provide similar functionality.

A tail launch executes when a graph’s environment is considered complete - ie, when the graph and all its children are complete. When a graph completes, the environment of the next graph in the tail launch list will replace the completed environment as a child of the parent environment. Like fire-and-forget launches, a graph can have multiple graphs enqueued for tail launch.

[![_images/tail-launch-simple.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-simple.png)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-simple.png)

Figure 19 A simple tail launch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id461 "Permalink to this image")

The above execution flow can be generated by the code below:

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

Tail launches enqueued by a given graph will execute one at a time, in order of when they were enqueued. So the first enqueued graph will run first, and then the second, and so on.

![_images/tail-launch-ordering-simple.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-ordering-simple.png)

Figure 20 Tail launch ordering[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id462 "Permalink to this image")

Tail launches enqueued by a tail graph will execute before tail launches enqueued by previous graphs in the tail launch list. These new tail launches will execute in the order they are enqueued.

![_images/tail-launch-ordering-complex.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-ordering-complex.png)

Figure 21 Tail launch ordering when enqueued from multiple graphs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id463 "Permalink to this image")

A graph can have up to 255 pending tail launches.

6.2.8.7.7.2.1.3.1. Tail Self-launch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tail-self-launch "Permalink to this headline")

It is possible for a device graph to enqueue itself for a tail launch, although a given graph can only have one self-launch enqueued at a time. In order to query the currently running device graph so that it can be relaunched, a new device-side function is added:

cudaGraphExec_t cudaGetCurrentGraphExec();

This function returns the handle of the currently running graph if it is a device graph. If the currently executing kernel is not a node within a device graph, this function will return NULL.

Below is sample code showing usage of this function for a relaunch loop:

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

6.2.8.7.7.2.1.4. Sibling Launch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sibling-launch "Permalink to this headline")

Sibling launch is a variation of fire-and-forget launch in which the graph is launched not as a child of the launching graph’s execution environment, but rather as a child of the launching graph’s parent environment. Sibling launch is equivalent to a fire-and-forget launch from the launching graph’s parent environment.

![_images/sibling-launch-simple.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/sibling-launch-simple.png)

Figure 22 A simple sibling launch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id464 "Permalink to this image")

The above diagram can be generated by the sample code below:

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

Since sibling launches are not launched into the launching graph’s execution environment, they will not gate tail launches enqueued by the launching graph.

##### 6.2.8.7.8. Conditional Graph Nodes[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-graph-nodes "Permalink to this headline")

Conditional nodes allow conditional execution and looping of a graph contained within the conditional node. This allows dynamic and iterative workflows to be represented completely within a graph and frees up the host CPU to perform other work in parallel.

Evaluation of the condition value is performed on the device when the dependencies of the conditional node have been met. Conditional nodes can be one of the following types:

- Conditional [IF nodes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-if-nodes) execute their body graph once if the condition value is non-zero when the node is executed. An optional second body graph can be provided and this will be executed once if the condition value is zero when the node is executed.
    
- Conditional [WHILE nodes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-while-nodes) execute their body graph if the condition value is non-zero when the node is executed and will continue to execute their body graph until the condition value is zero.
    
- Conditional [SWITCH nodes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-switch-nodes) execute the nth body graph once if the condition value is equal to n. If the condition value does not correspond to a body graph, no body graph is launched.
    

A condition value is accessed by a [conditional handle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-handles) , which must be created before the node. The condition value can be set by device code using `cudaGraphSetConditional()`. A default value, applied on each graph launch, can also be specified when the handle is created.

When the conditional node is created, an empty graph is created and the handle is returned to the user so that the graph can be populated. This conditional body graph can be populated using either the [graph APIs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis) or [cudaStreamBeginCaptureToGraph()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-stream-capture).

Conditional nodes can be nested.

###### 6.2.8.7.8.1. Conditional Handles[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-handles "Permalink to this headline")

A condition value is represented by `cudaGraphConditionalHandle` and is created by `cudaGraphConditionalHandleCreate()`.

The handle must be associated with a single conditional node. Handles cannot be destroyed.

If `cudaGraphCondAssignDefault` is specified when the handle is created, the condition value will be initialized to the specified default at the beginning of each graph execution. If this flag is not provided, the condition value is undefined at the start of each graph execution and code should not assume that the condition value persists across executions.

The default value and flags associated with a handle will be updated during [whole graph update](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#whole-graph-update).

###### 6.2.8.7.8.2. Conditional Node Body Graph Requirements[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-node-body-graph-requirements "Permalink to this headline")

General requirements:

- The graph’s nodes must all reside on a single device.
    
- The graph can only contain kernel nodes, empty nodes, memcpy nodes, memset nodes, child graph nodes, and conditional nodes.
    

Kernel nodes:

- Use of CUDA Dynamic Parallelism or Device Graph Launch by kernels in the graph is not permitted.
    
- Cooperative launches are permitted so long as MPS is not in use.
    

Memcpy/Memset nodes:

- Only copies/memsets involving device memory and/or pinned device-mapped host memory are permitted.
    
- Copies/memsets involving CUDA arrays are not permitted.
    
- Both operands must be accessible from the current device at time of instantiation. Note that the copy operation will be performed from the device on which the graph resides, even if it is targeting memory on another device.
    

###### 6.2.8.7.8.3. Conditional IF Nodes[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-if-nodes "Permalink to this headline")

The body graph of an IF node will be executed once if the condition is non-zero when the node is executed. The following diagram depicts a 3 node graph where the middle node, B, is a conditional node:

![_images/conditional-if-node.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/conditional-if-node.png)

Figure 23 Conditional IF Node[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id465 "Permalink to this image")

The following code illustrates the creation of a graph containing an IF conditional node. The default value of the condition is set using an upstream kernel. The body of the conditional is populated using the [graph API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis).

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

Starting in CUDA 12.8, IF nodes can also have an optional second body graph which is executed once when the node is executed if the condition value is zero.

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

###### 6.2.8.7.8.4. Conditional WHILE Nodes[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-while-nodes "Permalink to this headline")

The body graph of a WHILE node will be executed as long as the condition is non-zero. The condition will be evaluated when the node is executed and after completion of the body graph. The following diagram depicts a 3 node graph where the middle node, B, is a conditional node:

![_images/conditional-while-node.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/conditional-while-node.png)

Figure 24 Conditional WHILE Node[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id466 "Permalink to this image")

The following code illustrates the creation of a graph containing a WHILE conditional node. The handle is created using _cudaGraphCondAssignDefault_ to avoid the need for an upstream kernel. The body of the conditional is populated using the [graph API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis).

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

###### 6.2.8.7.8.5. Conditional SWITCH Nodes[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#conditional-switch-nodes "Permalink to this headline")

SWITCH nodes, added in CUDA 12.8, execute 1 of n different graphs within the conditional node. The nth graph will be executed when the SWITCH node is evaluated if the condition value is n. If the condition value is greater than or equal to n, no graph will be executed. The following diagram depicts a 3 node graph where the middle node, B, is a conditional node:

![_images/conditional-switch-node.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/conditional-switch-node.png)

Figure 25 Conditional SWITCH Node[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id467 "Permalink to this image")

The following code illustrates the creation of a graph containing a SWITCH conditional node. The value of the condition is set using an upstream kernel. The bodies of the conditional are populated using the [graph API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creating-a-graph-using-graph-apis).

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

#### 6.2.8.8. Events[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#events "Permalink to this headline")

The runtime also provides a way to closely monitor the device’s progress, as well as perform accurate timing, by letting the application asynchronously record _events_ at any point in the program, and query when these events are completed. An event has completed when all tasks - or optionally, all commands in a given stream - preceding the event have completed. Events in stream zero are completed after all preceding tasks and commands in all streams are completed.

##### 6.2.8.8.1. Creation and Destruction of Events[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-of-events "Permalink to this headline")

The following code sample creates two events:

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

They are destroyed this way:

cudaEventDestroy(start);
cudaEventDestroy(stop);

##### 6.2.8.8.2. Elapsed Time[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#elapsed-time "Permalink to this headline")

The events created in [Creation and Destruction of Events](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-events) can be used to time the code sample of [Creation and Destruction of Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-streams) the following way:

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

#### 6.2.8.9. Synchronous Calls[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronous-calls "Permalink to this headline")

When a synchronous function is called, control is not returned to the host thread before the device has completed the requested task. Whether the host thread will then yield, block, or spin can be specified by calling `cudaSetDeviceFlags()`with some specific flags (see reference manual for details) before any other CUDA call is performed by the host thread.

### 6.2.9. Multi-Device System[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multi-device-system "Permalink to this headline")

#### 6.2.9.1. Device Enumeration[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration "Permalink to this headline")

A host system can have multiple devices. The following code sample shows how to enumerate these devices, query their properties, and determine the number of CUDA-enabled devices.

int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}

#### 6.2.9.2. Device Selection[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-selection "Permalink to this headline")

A host thread can set the device it operates on at any time by calling `cudaSetDevice()`. Device memory allocations and kernel launches are made on the currently set device; streams and events are created in association with the currently set device. If no call to `cudaSetDevice()` is made, the current device is device 0.

The following code sample illustrates how setting the current device affects memory allocation and kernel execution.

size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // Set device 0 as current
float* p0;
cudaMalloc(&p0, size);       // Allocate memory on device 0
MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0
cudaSetDevice(1);            // Set device 1 as current
float* p1;
cudaMalloc(&p1, size);       // Allocate memory on device 1
MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1

#### 6.2.9.3. Stream and Event Behavior[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#stream-and-event-behavior "Permalink to this headline")

A kernel launch will fail if it is issued to a stream that is not associated to the current device as illustrated in the following code sample.

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

A memory copy will succeed even if it is issued to a stream that is not associated to the current device.

`cudaEventRecord()` will fail if the input event and input stream are associated to different devices.

`cudaEventElapsedTime()` will fail if the two input events are associated to different devices.

`cudaEventSynchronize()` and `cudaEventQuery()` will succeed even if the input event is associated to a device that is different from the current device.

`cudaStreamWaitEvent()` will succeed even if the input stream and input event are associated to different devices. `cudaStreamWaitEvent()` can therefore be used to synchronize multiple devices with each other.

Each device has its own default stream (see [Default Stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#default-stream)), so commands issued to the default stream of a device may execute out of order or concurrently with respect to commands issued to the default stream of any other device.

#### 6.2.9.4. Peer-to-Peer Memory Access[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-to-peer-memory-access "Permalink to this headline")

Depending on the system properties, specifically the PCIe and/or NVLINK topology, devices are able to address each other’s memory (i.e., a kernel executing on one device can dereference a pointer to the memory of the other device). This peer-to-peer memory access feature is supported between two devices if `cudaDeviceCanAccessPeer()` returns true for these two devices.

Peer-to-peer memory access is only supported in 64-bit applications and must be enabled between two devices by calling `cudaDeviceEnablePeerAccess()` as illustrated in the following code sample. On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections.

A unified address space is used for both devices (see [Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space)), so the same pointer can be used to address memory from both devices as shown in the code sample below.

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

##### 6.2.9.4.1. IOMMU on Linux[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#iommu-on-linux "Permalink to this headline")

On Linux only, CUDA and the display driver does not support IOMMU-enabled bare-metal PCIe peer to peer memory copy. However, CUDA and the display driver does support IOMMU via VM pass through. As a consequence, users on Linux, when running on a native bare metal system, should disable the IOMMU. The IOMMU should be enabled and the VFIO driver be used as a PCIe pass through for virtual machines.

On Windows the above limitation does not exist.

See also [Allocating DMA Buffers on 64-bit Platforms](https://download.nvidia.com/XFree86/Linux-x86_64/396.51/README/dma_issues.html).

#### 6.2.9.5. Peer-to-Peer Memory Copy[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-to-peer-memory-copy "Permalink to this headline")

Memory copies can be performed between the memories of two different devices.

When a unified address space is used for both devices (see [Unified Virtual Address Space](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space)), this is done using the regular memory copy functions mentioned in [Device Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory).

Otherwise, this is done using `cudaMemcpyPeer()`, `cudaMemcpyPeerAsync()`, `cudaMemcpy3DPeer()`, or `cudaMemcpy3DPeerAsync()` as illustrated in the following code sample.

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

A copy (in the implicit _NULL_ stream) between the memories of two different devices:

- does not start until all commands previously issued to either device have completed and
    
- runs to completion before any commands (see [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)) issued after the copy to either device can start.
    

Consistent with the normal behavior of streams, an asynchronous copy between the memories of two devices may overlap with copies or kernels in another stream.

Note that if peer-to-peer access is enabled between two devices via `cudaDeviceEnablePeerAccess()` as described in [Peer-to-Peer Memory Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#peer-to-peer-memory-access), peer-to-peer memory copy between these two devices no longer needs to be staged through the host and is therefore faster.

### 6.2.10. Unified Virtual Address Space[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-virtual-address-space "Permalink to this headline")

When the application is run as a 64-bit process, a single address space is used for the host and all the devices of compute capability 2.0 and higher. All host memory allocations made via CUDA API calls and all device memory allocations on supported devices are within this virtual address range. As a consequence:

- The location of any memory on the host allocated through CUDA, or on any of the devices which use the unified address space, can be determined from the value of the pointer using `cudaPointerGetAttributes()`.
    
- When copying to or from the memory of any device which uses the unified address space, the `cudaMemcpyKind` parameter of `cudaMemcpy*()` can be set to `cudaMemcpyDefault` to determine locations from the pointers. This also works for host pointers not allocated through CUDA, as long as the current device uses unified addressing.
    
- Allocations via `cudaHostAlloc()` are automatically portable (see [Portable Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#portable-memory)) across all the devices for which the unified address space is used, and pointers returned by `cudaHostAlloc()` can be used directly from within kernels running on these devices (i.e., there is no need to obtain a device pointer via `cudaHostGetDevicePointer()` as described in [Mapped Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory).
    

Applications may query if the unified address space is used for a particular device by checking that the `unifiedAddressing` device property (see [Device Enumeration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)) is equal to 1.

### 6.2.11. Interprocess Communication[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interprocess-communication "Permalink to this headline")

Any device memory pointer or event handle created by a host thread can be directly referenced by any other thread within the same process. It is not valid outside this process however, and therefore cannot be directly referenced by threads belonging to a different process.

To share device memory pointers and events across processes, an application must use the Inter Process Communication API, which is described in detail in the reference manual. The IPC API is only supported for 64-bit processes on Linux and for devices of compute capability 2.0 and higher. Note that the IPC API is not supported for `cudaMallocManaged` allocations.

Using this API, an application can get the IPC handle for a given device memory pointer using `cudaIpcGetMemHandle()`, pass it to another process using standard IPC mechanisms (for example, interprocess shared memory or files), and use `cudaIpcOpenMemHandle()` to retrieve a device pointer from the IPC handle that is a valid pointer within this other process. Event handles can be shared using similar entry points.

Note that allocations made by `cudaMalloc()` may be sub-allocated from a larger block of memory for performance reasons. In such case, CUDA IPC APIs will share the entire underlying memory block which may cause other sub-allocations to be shared, which can potentially lead to information disclosure between processes. To prevent this behavior, it is recommended to only share allocations with a 2MiB aligned size.

An example of using the IPC API is where a single primary process generates a batch of input data, making the data available to multiple secondary processes without requiring regeneration or copying.

Applications using CUDA IPC to communicate with each other should be compiled, linked, and run with the same CUDA driver and runtime.

Note

Since CUDA 11.5, only events-sharing IPC APIs are supported on L4T and embedded Linux Tegra devices with compute capability 7.x and higher. The memory-sharing IPC APIs are still not supported on Tegra platforms.

### 6.2.12. Error Checking[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#error-checking "Permalink to this headline")

All runtime functions return an error code, but for an asynchronous function (see [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)), this error code cannot possibly report any of the asynchronous errors that could occur on the device since the function returns before the device has completed the task; the error code only reports errors that occur on the host prior to executing the task, typically related to parameter validation; if an asynchronous error occurs, it will be reported by some subsequent unrelated runtime function call.

The only way to check for asynchronous errors just after some asynchronous function call is therefore to synchronize just after the call by calling `cudaDeviceSynchronize()` (or by using any other synchronization mechanisms described in [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)) and checking the error code returned by `cudaDeviceSynchronize()`.

The runtime maintains an error variable for each host thread that is initialized to `cudaSuccess` and is overwritten by the error code every time an error occurs (be it a parameter validation error or an asynchronous error). `cudaPeekAtLastError()` returns this variable. `cudaGetLastError()` returns this variable and resets it to `cudaSuccess`.

Kernel launches do not return any error code, so `cudaPeekAtLastError()` or `cudaGetLastError()` must be called just after the kernel launch to retrieve any pre-launch errors. To ensure that any error returned by `cudaPeekAtLastError()` or `cudaGetLastError()` does not originate from calls prior to the kernel launch, one has to make sure that the runtime error variable is set to `cudaSuccess` just before the kernel launch, for example, by calling `cudaGetLastError()` just before the kernel launch. Kernel launches are asynchronous, so to check for asynchronous errors, the application must synchronize in-between the kernel launch and the call to `cudaPeekAtLastError()` or `cudaGetLastError()`.

Note that `cudaErrorNotReady` that may be returned by `cudaStreamQuery()` and `cudaEventQuery()` is not considered an error and is therefore not reported by `cudaPeekAtLastError()` or `cudaGetLastError()`.

### 6.2.13. Call Stack[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#call-stack "Permalink to this headline")

On devices of compute capability 2.x and higher, the size of the call stack can be queried using`cudaDeviceGetLimit()` and set using `cudaDeviceSetLimit()`.

When the call stack overflows, the kernel call fails with a stack overflow error if the application is run via a CUDA debugger (CUDA-GDB, Nsight) or an unspecified launch error, otherwise. When the compiler cannot determine the stack size, it issues a warning saying Stack size cannot be statically determined. This is usually the case with recursive functions. Once this warning is issued, user will need to set stack size manually if default stack size is not sufficient.

### 6.2.14. Texture and Surface Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-and-surface-memory "Permalink to this headline")

CUDA supports a subset of the texturing hardware that the GPU uses for graphics to access texture and surface memory. Reading data from texture or surface memory instead of global memory can have several performance benefits as described in [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses).

#### 6.2.14.1. Texture Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory "Permalink to this headline")

Texture memory is read from kernels using the device functions described in [Texture Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-functions). The process of reading a texture calling one of these functions is called a _texture fetch_. Each texture fetch specifies a parameter called a _texture object_ for the texture object API.

The texture object specifies:

- The _texture_, which is the piece of texture memory that is fetched. Texture objects are created at runtime and the texture is specified when creating the texture object as described in [Texture Object API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api).
    
- Its _dimensionality_ that specifies whether the texture is addressed as a one dimensional array using one texture coordinate, a two-dimensional array using two texture coordinates, or a three-dimensional array using three texture coordinates. Elements of the array are called _texels_, short for _texture elements_. The _texture width_, _height_, and _depth_ refer to the size of the array in each dimension. [Table 27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability) lists the maximum texture width, height, and depth depending on the compute capability of the device.
    
- The type of a texel, which is restricted to the basic integer and single-precision floating-point types and any of the 1-, 2-, and 4-component vector types defined in [Built-in Vector Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-vector-types) that are derived from the basic integer and single-precision floating-point types.
    
- The _read mode_, which is equal to `cudaReadModeNormalizedFloat` or `cudaReadModeElementType`. If it is `cudaReadModeNormalizedFloat` and the type of the texel is a 16-bit or 8-bit integer type, the value returned by the texture fetch is actually returned as floating-point type and the full range of the integer type is mapped to [0.0, 1.0] for unsigned integer type and [-1.0, 1.0] for signed integer type; for example, an unsigned 8-bit texture element with the value 0xff reads as 1. If it is `cudaReadModeElementType`, no conversion is performed.
    
- Whether texture coordinates are normalized or not. By default, textures are referenced (by the functions of [Texture Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-functions)) using floating-point coordinates in the range [0, N-1] where N is the size of the texture in the dimension corresponding to the coordinate. For example, a texture that is 64x32 in size will be referenced with coordinates in the range [0, 63] and [0, 31] for the x and y dimensions, respectively. Normalized texture coordinates cause the coordinates to be specified in the range [0.0, 1.0-1/N] instead of [0, N-1], so the same 64x32 texture would be addressed by normalized coordinates in the range [0, 1-1/N] in both the x and y dimensions. Normalized texture coordinates are a natural fit to some applications’ requirements, if it is preferable for the texture coordinates to be independent of the texture size.
    
- The _addressing mode_. It is valid to call the device functions of Section B.8 with coordinates that are out of range. The addressing mode defines what happens in that case. The default addressing mode is to clamp the coordinates to the valid range: [0, N) for non-normalized coordinates and [0.0, 1.0) for normalized coordinates. If the border mode is specified instead, texture fetches with out-of-range texture coordinates return zero. For normalized coordinates, the wrap mode and the mirror mode are also available. When using the wrap mode, each coordinate x is converted to _frac(x)=x - floor(x)_ where _floor(x)_ is the largest integer not greater than _x_. When using the mirror mode, each coordinate _x_ is converted to _frac(x)_ if _floor(x)_ is even and _1-frac(x)_ if _floor(x)_ is odd. The addressing mode is specified as an array of size three whose first, second, and third elements specify the addressing mode for the first, second, and third texture coordinates, respectively; the addressing mode are `cudaAddressModeBorder`, `cudaAddressModeClamp`, `cudaAddressModeWrap`, and `cudaAddressModeMirror`; `cudaAddressModeWrap` and `cudaAddressModeMirror` are only supported for normalized texture coordinates
    
- The _filtering_ mode which specifies how the value returned when fetching the texture is computed based on the input texture coordinates. Linear texture filtering may be done only for textures that are configured to return floating-point data. It performs low-precision interpolation between neighboring texels. When enabled, the texels surrounding a texture fetch location are read and the return value of the texture fetch is interpolated based on where the texture coordinates fell between the texels. Simple linear interpolation is performed for one-dimensional textures, bilinear interpolation for two-dimensional textures, and trilinear interpolation for three-dimensional textures. [Texture Fetching](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-fetching) gives more details on texture fetching. The filtering mode is equal to `cudaFilterModePoint` or `cudaFilterModeLinear`. If it is `cudaFilterModePoint`, the returned value is the texel whose texture coordinates are the closest to the input texture coordinates. If it is `cudaFilterModeLinear`, the returned value is the linear interpolation of the two (for a one-dimensional texture), four (for a two dimensional texture), or eight (for a three dimensional texture) texels whose texture coordinates are the closest to the input texture coordinates. `cudaFilterModeLinear` is only valid for returned values of floating-point type.
    

[Texture Object API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api) introduces the texture object API.

[16-Bit Floating-Point Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sixteen-bit-floating-point-textures) explains how to deal with 16-bit floating-point textures.

Textures can also be layered as described in [Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures).

[Cubemap Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures) and [Cubemap Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures) describe a special type of texture, the cubemap texture.

[Texture Gather](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-gather) describes a special texture fetch, texture gather.

##### 6.2.14.1.1. Texture Object API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api "Permalink to this headline")

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

- `addressMode` specifies the addressing mode;
    
- `filterMode` specifies the filter mode;
    
- `readMode` specifies the read mode;
    
- `normalizedCoords` specifies whether texture coordinates are normalized or not;
    
- See reference manual for `sRGB`, `maxAnisotropy`, `mipmapFilterMode`, `mipmapLevelBias`, `minMipmapLevelClamp`, and `maxMipmapLevelClamp`.
    

The following code sample applies some simple transformation kernel to a texture.

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

##### 6.2.14.1.2. 16-Bit Floating-Point Textures[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#bit-floating-point-textures "Permalink to this headline")

The 16-bit floating-point or _half_ format supported by CUDA arrays is the same as the IEEE 754-2008 binary2 format.

CUDA C++ does not support a matching data type, but provides intrinsic functions to convert to and from the 32-bit floating-point format via the `unsigned short` type: `__float2half_rn(float)` and `__half2float(unsigned short)`. These functions are only supported in device code. Equivalent functions for the host code can be found in the OpenEXR library, for example.

16-bit floating-point components are promoted to 32 bit float during texture fetching before any filtering is performed.

A channel description for the 16-bit floating-point format can be created by calling one of the `cudaCreateChannelDescHalf*()` functions.

##### 6.2.14.1.3. Layered Textures[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures "Permalink to this headline")

A one-dimensional or two-dimensional layered texture (also known as _texture array_ in Direct3D and _array texture_ in OpenGL) is a texture made up of a sequence of layers, all of which are regular textures of same dimensionality, size, and data type.

A one-dimensional layered texture is addressed using an integer index and a floating-point texture coordinate; the index denotes a layer within the sequence and the coordinate addresses a texel within that layer. A two-dimensional layered texture is addressed using an integer index and two floating-point texture coordinates; the index denotes a layer within the sequence and the coordinates address a texel within that layer.

A layered texture can only be a CUDA array by calling `cudaMalloc3DArray()` with the `cudaArrayLayered` flag (and a height of zero for one-dimensional layered texture).

Layered textures are fetched using the device functions described in [tex1DLayered()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlayered-object) and [tex2DLayered()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayered-object). Texture filtering (see [Texture Fetching](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-fetching)) is done only within a layer, not across layers.

Layered textures are only supported on devices of compute capability 2.0 and higher.

##### 6.2.14.1.4. Cubemap Textures[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures "Permalink to this headline")

A _cubemap_ texture is a special type of two-dimensional layered texture that has six layers representing the faces of a cube:

- The width of a layer is equal to its height.
    
- The cubemap is addressed using three texture coordinates _x_, _y_, and _z_ that are interpreted as a direction vector emanating from the center of the cube and pointing to one face of the cube and a texel within the layer corresponding to that face. More specifically, the face is selected by the coordinate with largest magnitude _m_ and the corresponding layer is addressed using coordinates _(s/m+1)/2_ and _(t/m+1)/2_ where _s_ and _t_ are defined in [Table 6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures-cubemap-fetch).
    

Table 6 Cubemap Fetch[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures-cubemap-fetch "Permalink to this table")
||   |face|m|s|t|
|---|---|---|---|---|---|
|`\|x\| > \|y\|` and `\|x\| > \|z\|`|x ≥ 0|0|x|-z|-y|
|x < 0|1|-x|z|-y|
|`\|y\| > \|x\|` and `\|y\| > \|z\|`|y ≥ 0|2|y|x|z|
|y < 0|3|-y|x|-z|
|`\|z\| > \|x\|` and `\|z\| > \|y\|`|z ≥ 0|4|z|x|-y|
|z < 0|5|-z|-x|-y|

A cubemap texture can only be a CUDA array by calling `cudaMalloc3DArray()` with the `cudaArrayCubemap` flag.

Cubemap textures are fetched using the device function described in [texCubemap()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemap-object).

Cubemap textures are only supported on devices of compute capability 2.0 and higher.

##### 6.2.14.1.5. Cubemap Layered Textures[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures "Permalink to this headline")

A _cubemap layered_ texture is a layered texture whose layers are cubemaps of same dimension.

A cubemap layered texture is addressed using an integer index and three floating-point texture coordinates; the index denotes a cubemap within the sequence and the coordinates address a texel within that cubemap.

A cubemap layered texture can only be a CUDA array by calling `cudaMalloc3DArray()` with the `cudaArrayLayered` and `cudaArrayCubemap` flags.

Cubemap layered textures are fetched using the device function described in [texCubemapLayered()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplayered-object). Texture filtering (see [Texture Fetching](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-fetching)) is done only within a layer, not across layers.

Cubemap layered textures are only supported on devices of compute capability 2.0 and higher.

##### 6.2.14.1.6. Texture Gather[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-gather "Permalink to this headline")

Texture gather is a special texture fetch that is available for two-dimensional textures only. It is performed by the `tex2Dgather()` function, which has the same parameters as `tex2D()`, plus an additional `comp` parameter equal to 0, 1, 2, or 3 (see [tex2Dgather()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgather-object)). It returns four 32-bit numbers that correspond to the value of the component `comp` of each of the four texels that would have been used for bilinear filtering during a regular texture fetch. For example, if these texels are of values (253, 20, 31, 255), (250, 25, 29, 254), (249, 16, 37, 253), (251, 22, 30, 250), and `comp` is 2, `tex2Dgather()` returns (31, 29, 37, 30).

Note that texture coordinates are computed with only 8 bits of fractional precision. `tex2Dgather()` may therefore return unexpected results for cases where `tex2D()` would use 1.0 for one of its weights (α or β, see [Linear Filtering](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#linear-filtering)). For example, with an _x_ texture coordinate of 2.49805: _xB=x-0.5=1.99805_, however the fractional part of _xB_ is stored in an 8-bit fixed-point format. Since 0.99805 is closer to 256.f/256.f than it is to 255.f/256.f, _xB_ has the value 2. A `tex2Dgather()` in this case would therefore return indices 2 and 3 in _x_, instead of indices 1 and 2.

Texture gather is only supported for CUDA arrays created with the `cudaArrayTextureGather` flag and of width and height less than the maximum specified in [Table 27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability) for texture gather, which is smaller than for regular texture fetch.

Texture gather is only supported on devices of compute capability 2.0 and higher.

#### 6.2.14.2. Surface Memory[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-memory "Permalink to this headline")

For devices of compute capability 2.0 and higher, a CUDA array (described in [Cubemap Surfaces](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-surfaces)), created with the `cudaArraySurfaceLoadStore` flag, can be read and written via a _surface object_ using the functions described in [Surface Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-functions).

[Table 27](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability) lists the maximum surface width, height, and depth depending on the compute capability of the device.

##### 6.2.14.2.1. Surface Object API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-object-api "Permalink to this headline")

A surface object is created using `cudaCreateSurfaceObject()` from a resource description of type `struct cudaResourceDesc`. Unlike texture memory, surface memory uses byte addressing. This means that the x-coordinate used to access a texture element via texture functions needs to be multiplied by the byte size of the element to access the same element via a surface function. For example, the element at texture coordinate x of a one-dimensional floating-point CUDA array bound to a texture object `texObj` and a surface object `surfObj` is read using `tex1d(texObj, x)` via `texObj`, but `surf1Dread(surfObj, 4*x)` via `surfObj`. Similarly, the element at texture coordinate x and y of a two-dimensional floating-point CUDA array bound to a texture object `texObj` and a surface object `surfObj` is accessed using `tex2d(texObj, x, y)` via `texObj`, but `surf2Dread(surfObj, 4*x, y)` via `surObj` (the byte offset of the y-coordinate is internally calculated from the underlying line pitch of the CUDA array).

The following code sample applies some simple transformation kernel to a surface.

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

##### 6.2.14.2.2. Cubemap Surfaces[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-surfaces "Permalink to this headline")

Cubemap surfaces are accessed using`surfCubemapread()` and `surfCubemapwrite()` ([surfCubemapread()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemapread-object) and [surfCubemapwrite()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemapwrite-object)) as a two-dimensional layered surface, i.e., using an integer index denoting a face and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. Faces are ordered as indicated in [Table 6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures-cubemap-fetch).

##### 6.2.14.2.3. Cubemap Layered Surfaces[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-surfaces "Permalink to this headline")

Cubemap layered surfaces are accessed using `surfCubemapLayeredread()` and `surfCubemapLayeredwrite()` ([surfCubemapLayeredread()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemaplayeredread-object) and [surfCubemapLayeredwrite()](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemaplayeredwrite-object)) as a two-dimensional layered surface, i.e., using an integer index denoting a face of one of the cubemaps and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. Faces are ordered as indicated in [Table 6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures-cubemap-fetch), so index ((2 * 6) + 3), for example, accesses the fourth face of the third cubemap.

#### 6.2.14.3. CUDA Arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arrays "Permalink to this headline")

CUDA arrays are opaque memory layouts optimized for texture fetching. They are one dimensional, two dimensional, or three-dimensional and composed of elements, each of which has 1, 2 or 4 components that may be signed or unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats. CUDA arrays are only accessible by kernels through texture fetching as described in [Texture Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory) or surface reading and writing as described in [Surface Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-memory).

#### 6.2.14.4. Read/Write Coherency[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#read-write-coherency "Permalink to this headline")

The texture and surface memory is cached (see [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)) and within the same kernel call, the cache is not kept coherent with respect to global memory writes and surface memory writes, so any texture fetch or surface read to an address that has been written to via a global write or a surface write in the same kernel call returns undefined data. In other words, a thread can safely read some texture or surface memory location only if this memory location has been updated by a previous kernel call or memory copy, but not if it has been previously updated by the same thread or another thread from the same kernel call.

### 6.2.15. Graphics Interoperability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#graphics-interoperability "Permalink to this headline")

Some resources from OpenGL and Direct3D may be mapped into the address space of CUDA, either to enable CUDA to read data written by OpenGL or Direct3D, or to enable CUDA to write data for consumption by OpenGL or Direct3D.

A resource must be registered to CUDA before it can be mapped using the functions mentioned in [OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability) and [Direct3D Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-interoperability). These functions return a pointer to a CUDA graphics resource of type `struct cudaGraphicsResource`. Registering a resource is potentially high-overhead and therefore typically called only once per resource. A CUDA graphics resource is unregistered using `cudaGraphicsUnregisterResource()`. Each CUDA context which intends to use the resource is required to register it separately.

Once a resource is registered to CUDA, it can be mapped and unmapped as many times as necessary using `cudaGraphicsMapResources()` and `cudaGraphicsUnmapResources()`. `cudaGraphicsResourceSetMapFlags()` can be called to specify usage hints (write-only, read-only) that the CUDA driver can use to optimize resource management.

A mapped resource can be read from or written to by kernels using the device memory address returned by `cudaGraphicsResourceGetMappedPointer()` for buffers and`cudaGraphicsSubResourceGetMappedArray()` for CUDA arrays.

Accessing a resource through OpenGL, Direct3D, or another CUDA context while it is mapped produces undefined results. [OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability) and [Direct3D Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-interoperability) give specifics for each graphics API and some code samples. [SLI Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sli-interoperability) gives specifics for when the system is in SLI mode.

#### 6.2.15.1. OpenGL Interoperability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability "Permalink to this headline")

The OpenGL resources that may be mapped into the address space of CUDA are OpenGL buffer, texture, and renderbuffer objects.

A buffer object is registered using `cudaGraphicsGLRegisterBuffer()`. In CUDA, it appears as a device pointer and can therefore be read and written by kernels or via `cudaMemcpy()` calls.

A texture or renderbuffer object is registered using `cudaGraphicsGLRegisterImage()`. In CUDA, it appears as a CUDA array. Kernels can read from the array by binding it to a texture or surface reference. They can also write to it via the surface write functions if the resource has been registered with the `cudaGraphicsRegisterFlagsSurfaceLoadStore` flag. The array can also be read and written via `cudaMemcpy2D()` calls. `cudaGraphicsGLRegisterImage()` supports all texture formats with 1, 2, or 4 components and an internal type of float (for example, `GL_RGBA_FLOAT32`), normalized integer (for example, `GL_RGBA8, GL_INTENSITY16`), and unnormalized integer (for example, `GL_RGBA8UI`) (please note that since unnormalized integer formats require OpenGL 3.0, they can only be written by shaders, not the fixed function pipeline).

The OpenGL context whose resources are being shared has to be current to the host thread making any OpenGL interoperability API calls.

Please note: When an OpenGL texture is made bindless (say for example by requesting an image or texture handle using the `glGetTextureHandle`*/`glGetImageHandle`* APIs) it cannot be registered with CUDA. The application needs to register the texture for interop before requesting an image or texture handle.

The following code sample uses a kernel to dynamically modify a 2D `width` x `height` grid of vertices stored in a vertex buffer object:

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

On Windows and for Quadro GPUs, `cudaWGLGetDevice()` can be used to retrieve the CUDA device associated to the handle returned by `wglEnumGpusNV()`. Quadro GPUs offer higher performance OpenGL interoperability than GeForce and Tesla GPUs in a multi-GPU configuration where OpenGL rendering is performed on the Quadro GPU and CUDA computations are performed on other GPUs in the system.

#### 6.2.15.2. Direct3D Interoperability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-interoperability "Permalink to this headline")

Direct3D interoperability is supported for Direct3D 9Ex, Direct3D 10, and Direct3D 11.

A CUDA context may interoperate only with Direct3D devices that fulfill the following criteria: Direct3D 9Ex devices must be created with `DeviceType` set to `D3DDEVTYPE_HAL` and `BehaviorFlags` with the `D3DCREATE_HARDWARE_VERTEXPROCESSING` flag; Direct3D 10 and Direct3D 11 devices must be created with `DriverType` set to `D3D_DRIVER_TYPE_HARDWARE`.

The Direct3D resources that may be mapped into the address space of CUDA are Direct3D buffers, textures, and surfaces. These resources are registered using `cudaGraphicsD3D9RegisterResource()`, `cudaGraphicsD3D10RegisterResource()`, and `cudaGraphicsD3D11RegisterResource()`.

The following code sample uses a kernel to dynamically modify a 2D `width` x `height` grid of vertices stored in a vertex buffer object.

##### 6.2.15.2.1. Direct3D 9 Version[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-9-version "Permalink to this headline")

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

##### 6.2.15.2.2. Direct3D 10 Version[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-10-version "Permalink to this headline")

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

##### 6.2.15.2.3. Direct3D 11 Version[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-11-version "Permalink to this headline")

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

#### 6.2.15.3. SLI Interoperability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#sli-interoperability "Permalink to this headline")

In a system with multiple GPUs, all CUDA-enabled GPUs are accessible via the CUDA driver and runtime as separate devices. There are however special considerations as described below when the system is in SLI mode.

First, an allocation in one CUDA device on one GPU will consume memory on other GPUs that are part of the SLI configuration of the Direct3D or OpenGL device. Because of this, allocations may fail earlier than otherwise expected.

Second, applications should create multiple CUDA contexts, one for each GPU in the SLI configuration. While this is not a strict requirement, it avoids unnecessary data transfers between devices. The application can use the `cudaD3D[9|10|11]GetDevices()` for Direct3D and `cudaGLGetDevices()` for OpenGL set of calls to identify the CUDA device handle(s) for the device(s) that are performing the rendering in the current and next frame. Given this information the application will typically choose the appropriate device and map Direct3D or OpenGL resources to the CUDA device returned by `cudaD3D[9|10|11]GetDevices()` or `cudaGLGetDevices()` when the `deviceList` parameter is set to `cudaD3D[9|10|11]DeviceListCurrentFrame` or `cudaGLDeviceListCurrentFrame`.

Please note that resource returned from `cudaGraphicsD9D[9|10|11]RegisterResource` and `cudaGraphicsGLRegister[Buffer|Image]` must be only used on device the registration happened. Therefore on SLI configurations when data for different frames is computed on different CUDA devices it is necessary to register the resources for each separately.

See [Direct3D Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-interoperability) and [OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability) for details on how the CUDA runtime interoperate with Direct3D and OpenGL, respectively.

### 6.2.16. External Resource Interoperability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#external-resource-interoperability "Permalink to this headline")

External resource interoperability allows CUDA to import certain resources that are explicitly exported by other APIs. These objects are typically exported by other APIs using handles native to the Operating System, like file descriptors on Linux or NT handles on Windows. They could also be exported using other unified interfaces such as the NVIDIA Software Communication Interface. There are two types of resources that can be imported: memory objects and synchronization objects.

Memory objects can be imported into CUDA using `cudaImportExternalMemory()`. An imported memory object can be accessed from within kernels using device pointers mapped onto the memory object via `cudaExternalMemoryGetMappedBuffer()`or CUDA mipmapped arrays mapped via `cudaExternalMemoryGetMappedMipmappedArray()`. Depending on the type of memory object, it may be possible for more than one mapping to be setup on a single memory object. The mappings must match the mappings setup in the exporting API. Any mismatched mappings result in undefined behavior. Imported memory objects must be freed using `cudaDestroyExternalMemory()`. Freeing a memory object does not free any mappings to that object. Therefore, any device pointers mapped onto that object must be explicitly freed using `cudaFree()` and any CUDA mipmapped arrays mapped onto that object must be explicitly freed using `cudaFreeMipmappedArray()`. It is illegal to access mappings to an object after it has been destroyed.

Synchronization objects can be imported into CUDA using `cudaImportExternalSemaphore()`. An imported synchronization object can then be signaled using `cudaSignalExternalSemaphoresAsync()` and waited on using `cudaWaitExternalSemaphoresAsync()`. It is illegal to issue a wait before the corresponding signal has been issued. Also, depending on the type of the imported synchronization object, there may be additional constraints imposed on how they can be signaled and waited on, as described in subsequent sections. Imported semaphore objects must be freed using `cudaDestroyExternalSemaphore()`. All outstanding signals and waits must have completed before the semaphore object is destroyed.

#### 6.2.16.1. Vulkan Interoperability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vulkan-interoperability "Permalink to this headline")

##### 6.2.16.1.1. Matching device UUIDs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#matching-device-uuids "Permalink to this headline")

When importing memory and synchronization objects exported by Vulkan, they must be imported and mapped on the same device as they were created on. The CUDA device that corresponds to the Vulkan physical device on which the objects were created can be determined by comparing the UUID of a CUDA device with that of the Vulkan physical device, as shown in the following code sample. Note that the Vulkan physical device should not be part of a device group that contains more than one Vulkan physical device. The device group as returned by `vkEnumeratePhysicalDeviceGroups` that contains the given Vulkan physical device must have a physical device count of 1.

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

##### 6.2.16.1.2. Importing Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-memory-objects "Permalink to this headline")

On Linux and Windows 10, both dedicated and non-dedicated memory objects exported by Vulkan can be imported into CUDA. On Windows 7, only dedicated memory objects can be imported. When importing a Vulkan dedicated memory object, the flag `cudaExternalMemoryDedicated` must be set.

A Vulkan memory object exported using `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` can be imported into CUDA using the file descriptor associated with that object as shown below. Note that CUDA assumes ownership of the file descriptor once it is imported. Using the file descriptor after a successful import results in undefined behavior.

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

A Vulkan memory object exported using `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` can be imported into CUDA using the NT handle associated with that object as shown below. Note that CUDA does not assume ownership of the NT handle and it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed.

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

A Vulkan memory object exported using `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` can also be imported using a named handle if one exists as shown below.

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

A Vulkan memory object exported using VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT can be imported into CUDA using the globally shared D3DKMT handle associated with that object as shown below. Since a globally shared D3DKMT handle does not hold a reference to the underlying memory it is automatically destroyed when all other references to the resource are destroyed.

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

##### 6.2.16.1.3. Mapping Buffers onto Imported Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-buffers-onto-imported-memory-objects "Permalink to this headline")

A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping must match that specified when creating the mapping using the corresponding Vulkan API. All mapped device pointers must be freed using `cudaFree()`.

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

##### 6.2.16.1.4. Mapping Mipmapped Arrays onto Imported Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-mipmapped-arrays-onto-imported-memory-objects "Permalink to this headline")

A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions, format and number of mip levels must match that specified when creating the mapping using the corresponding Vulkan API. Additionally, if the mipmapped array is bound as a color target in Vulkan, the flag`cudaArrayColorAttachment` must be set. All mapped mipmapped arrays must be freed using `cudaFreeMipmappedArray()`. The following code sample shows how to convert Vulkan parameters into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects.

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

##### 6.2.16.1.5. Importing Synchronization Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects "Permalink to this headline")

A Vulkan semaphore object exported using `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT`can be imported into CUDA using the file descriptor associated with that object as shown below. Note that CUDA assumes ownership of the file descriptor once it is imported. Using the file descriptor after a successful import results in undefined behavior.

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

A Vulkan semaphore object exported using `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT` can be imported into CUDA using the NT handle associated with that object as shown below. Note that CUDA does not assume ownership of the NT handle and it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying semaphore can be freed.

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

A Vulkan semaphore object exported using `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT` can also be imported using a named handle if one exists as shown below.

cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.name = (void *)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

A Vulkan semaphore object exported using `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT` can be imported into CUDA using the globally shared D3DKMT handle associated with that object as shown below. Since a globally shared D3DKMT handle does not hold a reference to the underlying semaphore it is automatically destroyed when all other references to the resource are destroyed.

cudaExternalSemaphore_t importVulkanSemaphoreObjectFromKMTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    desc.handle.win32.handle = (void *)handle;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

##### 6.2.16.1.6. Signaling/Waiting on Imported Synchronization Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#signaling-waiting-on-imported-synchronization-objects "Permalink to this headline")

An imported Vulkan semaphore object can be signaled as shown below. Signaling such a semaphore object sets it to the signaled state. The corresponding wait that waits on this signal must be issued in Vulkan. Additionally, the wait that waits on this signal must be issued after this signal has been issued.

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));

    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

An imported Vulkan semaphore object can be waited on as shown below. Waiting on such a semaphore object waits until it reaches the signaled state and then resets it back to the unsignaled state. The corresponding signal that this wait is waiting on must be issued in Vulkan. Additionally, the signal must be issued before this wait can be issued.

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));

    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

#### 6.2.16.2. OpenGL Interoperability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability-ext-res-int "Permalink to this headline")

Traditional OpenGL-CUDA interop as outlined in [OpenGL Interoperability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#opengl-interoperability) works by CUDA directly consuming handles created in OpenGL. However, since OpenGL can also consume memory and synchronization objects created in Vulkan, there exists an alternative approach to doing OpenGL-CUDA interop. Essentially, memory and synchronization objects exported by Vulkan could be imported into both, OpenGL and CUDA, and then used to coordinate memory accesses between OpenGL and CUDA. Please refer to the following OpenGL extensions for further details on how to import memory and synchronization objects exported by Vulkan:

- GL_EXT_memory_object
    
- GL_EXT_memory_object_fd
    
- GL_EXT_memory_object_win32
    
- GL_EXT_semaphore
    
- GL_EXT_semaphore_fd
    
- GL_EXT_semaphore_win32
    

#### 6.2.16.3. Direct3D 12 Interoperability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-12-interoperability "Permalink to this headline")

##### 6.2.16.3.1. Matching Device LUIDs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#matching-device-luids "Permalink to this headline")

When importing memory and synchronization objects exported by Direct3D 12, they must be imported and mapped on the same device as they were created on. The CUDA device that corresponds to the Direct3D 12 device on which the objects were created can be determined by comparing the LUID of a CUDA device with that of the Direct3D 12 device, as shown in the following code sample. Note that the Direct3D 12 device must not be created on a linked node adapter. I.e. the node count as returned by `ID3D12Device::GetNodeCount` must be 1.

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

##### 6.2.16.3.2. Importing Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-memory-objects-dir3d-12-int "Permalink to this headline")

A shareable Direct3D 12 heap memory object, created by setting the flag `D3D12_HEAP_FLAG_SHARED` in the call to `ID3D12Device::CreateHeap`, can be imported into CUDA using the NT handle associated with that object as shown below. Note that it is the application’s responsibility to close the NT handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed.

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

A shareable Direct3D 12 heap memory object can also be imported using a named handle if one exists as shown below.

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

A shareable Direct3D 12 committed resource, created by setting the flag `D3D12_HEAP_FLAG_SHARED` in the call to `D3D12Device::CreateCommittedResource`, can be imported into CUDA using the NT handle associated with that object as shown below. When importing a Direct3D 12 committed resource, the flag `cudaExternalMemoryDedicated` must be set. Note that it is the application’s responsibility to close the NT handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying memory can be freed.

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

A shareable Direct3D 12 committed resource can also be imported using a named handle if one exists as shown below.

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

##### 6.2.16.3.3. Mapping Buffers onto Imported Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-buffers-onto-imported-memory-objects-dir3d-12-int "Permalink to this headline")

A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping must match that specified when creating the mapping using the corresponding Direct3D 12 API. All mapped device pointers must be freed using `cudaFree()`.

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

##### 6.2.16.3.4. Mapping Mipmapped Arrays onto Imported Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-mipmapped-arrays-onto-imported-memory-objects-dir3d-12-int "Permalink to this headline")

A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions, format and number of mip levels must match that specified when creating the mapping using the corresponding Direct3D 12 API. Additionally, if the mipmapped array can be bound as a render target in Direct3D 12, the flag `cudaArrayColorAttachment` must be set. All mapped mipmapped arrays must be freed using `cudaFreeMipmappedArray()`. The following code sample shows how to convert Vulkan parameters into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects.

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

##### 6.2.16.3.5. Importing Synchronization Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects-dir3d-12-int "Permalink to this headline")

A shareable Direct3D 12 fence object, created by setting the flag `D3D12_FENCE_FLAG_SHARED` in the call to `ID3D12Device::CreateFence`, can be imported into CUDA using the NT handle associated with that object as shown below. Note that it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying semaphore can be freed.

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

A shareable Direct3D 12 fence object can also be imported using a named handle if one exists as shown below.

cudaExternalSemaphore_t importD3D12FenceFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    desc.handle.win32.name = (void *)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

##### 6.2.16.3.6. Signaling/Waiting on Imported Synchronization Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#signaling-waiting-on-imported-synchronization-objects-dir3d-12-int "Permalink to this headline")

An imported Direct3D 12 fence object can be signaled as shown below. Signaling such a fence object sets its value to the one specified. The corresponding wait that waits on this signal must be issued in Direct3D 12. Additionally, the wait that waits on this signal must be issued after this signal has been issued.

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

An imported Direct3D 12 fence object can be waited on as shown below. Waiting on such a fence object waits until its value becomes greater than or equal to the specified value. The corresponding signal that this wait is waiting on must be issued in Direct3D 12. Additionally, the signal must be issued before this wait can be issued.

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

#### 6.2.16.4. Direct3D 11 Interoperability[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#direct3d-11-interoperability "Permalink to this headline")

##### 6.2.16.4.1. Matching Device LUIDs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#matching-device-luids-dir3d-11-int "Permalink to this headline")

When importing memory and synchronization objects exported by Direct3D 11, they must be imported and mapped on the same device as they were created on. The CUDA device that corresponds to the Direct3D 11 device on which the objects were created can be determined by comparing the LUID of a CUDA device with that of the Direct3D 11 device, as shown in the following code sample.

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

##### 6.2.16.4.2. Importing Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-memory-objects-dir3d-11-int "Permalink to this headline")

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

A shareable Direct3D 11 resource can also be imported using a named handle if one exists as shown below.

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

A shareable Direct3D 11 resource, created by specifying the `D3D11_RESOURCE_MISC_SHARED` or `D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX`, can be imported into CUDA using the globally shared `D3DKMT` handle associated with that object as shown below. Since a globally shared `D3DKMT` handle does not hold a reference to the underlying memory it is automatically destroyed when all other references to the resource are destroyed.

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

##### 6.2.16.4.3. Mapping Buffers onto Imported Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-buffers-onto-imported-memory-objects-dir3d-11-int "Permalink to this headline")

A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping must match that specified when creating the mapping using the corresponding Direct3D 11 API. All mapped device pointers must be freed using `cudaFree()`.

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

##### 6.2.16.4.4. Mapping Mipmapped Arrays onto Imported Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-mipmapped-arrays-onto-imported-memory-objects-dir3d-11-int "Permalink to this headline")

A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions, format and number of mip levels must match that specified when creating the mapping using the corresponding Direct3D 11 API. Additionally, if the mipmapped array can be bound as a render target in Direct3D 12, the flag `cudaArrayColorAttachment` must be set. All mapped mipmapped arrays must be freed using `cudaFreeMipmappedArray()`. The following code sample shows how to convert Direct3D 11 parameters into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects.

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

##### 6.2.16.4.5. Importing Synchronization Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects-dir3d-11-int "Permalink to this headline")

A shareable Direct3D 11 fence object, created by setting the flag `D3D11_FENCE_FLAG_SHARED` in the call to `ID3D11Device5::CreateFence`, can be imported into CUDA using the NT handle associated with that object as shown below. Note that it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying semaphore can be freed.

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

A shareable Direct3D 11 fence object can also be imported using a named handle if one exists as shown below.

cudaExternalSemaphore_t importD3D11FenceFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
    desc.handle.win32.name = (void *)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

A shareable Direct3D 11 keyed mutex object associated with a shareable Direct3D 11 resource, viz, `IDXGIKeyedMutex`, created by setting the flag `D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX`, can be imported into CUDA using the NT handle associated with that object as shown below. Note that it is the application’s responsibility to close the handle when it is not required anymore. The NT handle holds a reference to the resource, so it must be explicitly freed before the underlying semaphore can be freed.

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

A shareable Direct3D 11 keyed mutex object can also be imported using a named handle if one exists as shown below.

cudaExternalSemaphore_t importD3D11KeyedMutexFromNamedNTHandle(LPCWSTR name) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeKeyedMutex;
    desc.handle.win32.name = (void *)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}

A shareable Direct3D 11 keyed mutex object can be imported into CUDA using the globally shared D3DKMT handle associated with that object as shown below. Since a globally shared D3DKMT handle does not hold a reference to the underlying memory it is automatically destroyed when all other references to the resource are destroyed.

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

##### 6.2.16.4.6. Signaling/Waiting on Imported Synchronization Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#signaling-waiting-on-imported-synchronization-objects-dir3d-11-int "Permalink to this headline")

An imported Direct3D 11 fence object can be signaled as shown below. Signaling such a fence object sets its value to the one specified. The corresponding wait that waits on this signal must be issued in Direct3D 11. Additionally, the wait that waits on this signal must be issued after this signal has been issued.

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

An imported Direct3D 11 fence object can be waited on as shown below. Waiting on such a fence object waits until its value becomes greater than or equal to the specified value. The corresponding signal that this wait is waiting on must be issued in Direct3D 11. Additionally, the signal must be issued before this wait can be issued.

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.fence.value = value;

    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

An imported Direct3D 11 keyed mutex object can be signaled as shown below. Signaling such a keyed mutex object by specifying a key value releases the keyed mutex for that value. The corresponding wait that waits on this signal must be issued in Direct3D 11 with the same key value. Additionally, the Direct3D 11 wait must be issued after this signal has been issued.

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long key, cudaStream_t stream) {
    cudaExternalSemaphoreSignalParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.keyedmutex.key = key;

    cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

An imported Direct3D 11 keyed mutex object can be waited on as shown below. A timeout value in milliseconds is needed when waiting on such a keyed mutex. The wait operation waits until the keyed mutex value is equal to the specified key value or until the timeout has elapsed. The timeout interval can also be an infinite value. In case an infinite value is specified the timeout never elapses. The windows INFINITE macro must be used to specify an infinite timeout. The corresponding signal that this wait is waiting on must be issued in Direct3D 11. Additionally, the Direct3D 11 signal must be issued before this wait can be issued.

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, unsigned long long key, unsigned int timeoutMs, cudaStream_t stream) {
    cudaExternalSemaphoreWaitParams params = {};

    memset(&params, 0, sizeof(params));

    params.params.keyedmutex.key = key;
    params.params.keyedmutex.timeoutMs = timeoutMs;

    cudaWaitExternalSemaphoresAsync(&extSem, &params, 1, stream);
}

#### 6.2.16.5. NVIDIA Software Communication Interface Interoperability (NVSCI)[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nvidia-software-communication-interface-interoperability-nvsci "Permalink to this headline")

NvSciBuf and NvSciSync are interfaces developed for serving the following purposes:

- NvSciBuf: Allows applications to allocate and exchange buffers in memory
    
- NvSciSync: Allows applications to manage synchronization objects at operation boundaries
    

More details on these interfaces are available at: [https://docs.nvidia.com/drive](https://docs.nvidia.com/drive).

##### 6.2.16.5.1. Importing Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-memory-objects-nvsci "Permalink to this headline")

For allocating an NvSciBuf object compatible with a given CUDA device, the corresponding GPU id must be set with `NvSciBufGeneralAttrKey_GpuId` in the NvSciBuf attribute list as shown below. Optionally, applications can specify the following attributes -

- `NvSciBufGeneralAttrKey_NeedCpuAccess`: Specifies if CPU access is required for the buffer
    
- `NvSciBufRawBufferAttrKey_Align`: Specifies the alignment requirement of `NvSciBufType_RawBuffer`
    
- `NvSciBufGeneralAttrKey_RequiredPerm`: Different access permissions can be configured for different UMDs per NvSciBuf memory object instance. For example, to provide the GPU with read-only access permissions to the buffer, create a duplicate NvSciBuf object using `NvSciBufObjDupWithReducePerm()` with `NvSciBufAccessPerm_Readonly` as the input parameter. Then import this newly created duplicate object with reduced permission into CUDA as shown
    
- `NvSciBufGeneralAttrKey_EnableGpuCache`: To control GPU L2 cacheability
    
- `NvSciBufGeneralAttrKey_EnableGpuCompression`: To specify GPU compression
    

Note

For more details on these attributes and their valid input options, refer to NvSciBuf Documentation.

The following code snippet illustrates their sample usage.

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

The allocated NvSciBuf memory object can be imported in CUDA using the NvSciBufObj handle as shown below. Application should query the allocated NvSciBufObj for attributes required for filling CUDA External Memory Descriptor. Note that the attribute list and NvSciBuf objects should be maintained by the application. If the NvSciBuf object imported into CUDA is also mapped by other drivers, then based on `NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency` output attribute value the application must use NvSciSync objects (refer to [Importing Synchronization Objects](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects-nvsci)) as appropriate barriers to maintain coherence between CUDA and the other drivers.

Note

For more details on how to allocate and maintain NvSciBuf objects refer to NvSciBuf API Documentation.

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

##### 6.2.16.5.2. Mapping Buffers onto Imported Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-buffers-onto-imported-memory-objects-nvsci "Permalink to this headline")

A device pointer can be mapped onto an imported memory object as shown below. The offset and size of the mapping can be filled as per the attributes of the allocated `NvSciBufObj`. All mapped device pointers must be freed using `cudaFree()`.

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

##### 6.2.16.5.3. Mapping Mipmapped Arrays onto Imported Memory Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapping-mipmapped-arrays-onto-imported-memory-objects-nvsci "Permalink to this headline")

A CUDA mipmapped array can be mapped onto an imported memory object as shown below. The offset, dimensions and format can be filled as per the attributes of the allocated `NvSciBufObj`. All mapped mipmapped arrays must be freed using `cudaFreeMipmappedArray()`. The following code sample shows how to convert NvSciBuf attributes into the corresponding CUDA parameters when mapping mipmapped arrays onto imported memory objects.

Note

The number of mip levels must be 1.

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

##### 6.2.16.5.4. Importing Synchronization Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#importing-synchronization-objects-nvsci "Permalink to this headline")

NvSciSync attributes that are compatible with a given CUDA device can be generated using `cudaDeviceGetNvSciSyncAttributes()`. The returned attribute list can be used to create a `NvSciSyncObj` that is guaranteed compatibility with a given CUDA device.

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

An NvSciSync object (created as above) can be imported into CUDA using the NvSciSyncObj handle as shown below. Note that ownership of the NvSciSyncObj handle continues to lie with the application even after it is imported.

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

##### 6.2.16.5.5. Signaling/Waiting on Imported Synchronization Objects[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#signaling-waiting-on-imported-synchronization-objects-nvsci "Permalink to this headline")

An imported `NvSciSyncObj` object can be signaled as outlined below. Signaling NvSciSync backed semaphore object initializes the _fence_ parameter passed as input. This fence parameter is waited upon by a wait operation that corresponds to the aforementioned signal. Additionally, the wait that waits on this signal must be issued after this signal has been issued. If the flags are set to `cudaExternalSemaphoreSignalSkipNvSciBufMemSync` then memory synchronization operations (over all the imported NvSciBuf in this process) that are executed as a part of the signal operation by default are skipped. When `NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency` is FALSE, this flag should be set.

void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, void *fence) {
    cudaExternalSemaphoreSignalParams signalParams = {};

    memset(&signalParams, 0, sizeof(signalParams));

    signalParams.params.nvSciSync.fence = (void*)fence;
    signalParams.flags = 0; //OR cudaExternalSemaphoreSignalSkipNvSciBufMemSync

    cudaSignalExternalSemaphoresAsync(&extSem, &signalParams, 1, stream);

}

An imported `NvSciSyncObj` object can be waited upon as outlined below. Waiting on NvSciSync backed semaphore object waits until the input _fence_ parameter is signaled by the corresponding signaler. Additionally, the signal must be issued before the wait can be issued. If the flags are set to `cudaExternalSemaphoreWaitSkipNvSciBufMemSync` then memory synchronization operations (over all the imported NvSciBuf in this process) that are executed as a part of the signal operation by default are skipped. When `NvsciBufGeneralAttrKey_GpuSwNeedCacheCoherency` is FALSE, this flag should be set.

void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream, void *fence) {
     cudaExternalSemaphoreWaitParams waitParams = {};

    memset(&waitParams, 0, sizeof(waitParams));

    waitParams.params.nvSciSync.fence = (void*)fence;
    waitParams.flags = 0; //OR cudaExternalSemaphoreWaitSkipNvSciBufMemSync

    cudaWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, stream);
}

## 6.3. Versioning and Compatibility[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#versioning-and-compatibility "Permalink to this headline")

There are two version numbers that developers should care about when developing a CUDA application: The compute capability that describes the general specifications and features of the compute device (see [Compute Capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability)) and the version of the CUDA driver API that describes the features supported by the driver API and runtime.

The version of the driver API is defined in the driver header file as `CUDA_VERSION`. It allows developers to check whether their application requires a newer device driver than the one currently installed. This is important, because the driver API is _backward compatible_, meaning that applications, plug-ins, and libraries (including the CUDA runtime) compiled against a particular version of the driver API will continue to work on subsequent device driver releases as illustrated in [Figure 26](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#versioning-and-compatibility-driver-api-is-backward-but-not-forward-compatible). The driver API is not _forward compatible_, which means that applications, plug-ins, and libraries (including the CUDA runtime) compiled against a particular version of the driver API will not work on previous versions of the device driver.

It is important to note that there are limitations on the mixing and matching of versions that is supported:

- Since only one version of the CUDA Driver can be installed at a time on a system, the installed driver must be of the same or higher version than the maximum Driver API version against which any application, plug-ins, or libraries that must run on that system were built.
    
- All plug-ins and libraries used by an application must use the same version of the CUDA Runtime unless they statically link to the Runtime, in which case multiple versions of the runtime can coexist in the same process space. Note that if `nvcc` is used to link the application, the static version of the CUDA Runtime library will be used by default, and all CUDA Toolkit libraries are statically linked against the CUDA Runtime.
    
- All plug-ins and libraries used by an application must use the same version of any libraries that use the runtime (such as cuFFT, cuBLAS, …) unless statically linking to those libraries.
    

![The Driver API Is Backward but Not Forward Compatible](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/compatibility-of-cuda-versions.png)

Figure 26 The Driver API Is Backward but Not Forward Compatible[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#versioning-and-compatibility-driver-api-is-backward-but-not-forward-compatible "Permalink to this image")

For Tesla GPU products, CUDA 10 introduced a new forward-compatible upgrade path for the user-mode components of the CUDA Driver. This feature is described in [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html). The requirements on the CUDA Driver version described here apply to the version of the user-mode components.

## 6.4. Compute Modes[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-modes "Permalink to this headline")

On Tesla solutions running Windows Server 2008 and later or Linux, one can set any device in a system in one of the three following modes using NVIDIA’s System Management Interface (nvidia-smi), which is a tool distributed as part of the driver:

- _Default_ compute mode: Multiple host threads can use the device (by calling `cudaSetDevice()` on this device, when using the runtime API, or by making current a context associated to the device, when using the driver API) at the same time.
    
- _Exclusive-process_ compute mode: Only one CUDA context may be created on the device across all processes in the system. The context may be current to as many threads as desired within the process that created that context.
    
- _Prohibited_ compute mode: No CUDA context can be created on the device.
    

This means, in particular, that a host thread using the runtime API without explicitly calling `cudaSetDevice()` might be associated with a device other than device 0 if device 0 turns out to be in prohibited mode or in exclusive-process mode and used by another process. `cudaSetValidDevices()` can be used to set a device from a prioritized list of devices.

Note also that, for devices featuring the Pascal architecture onwards (compute capability with major revision number 6 and higher), there exists support for Compute Preemption. This allows compute tasks to be preempted at instruction-level granularity, rather than thread block granularity as in prior Maxwell and Kepler GPU architecture, with the benefit that applications with long-running kernels can be prevented from either monopolizing the system or timing out. However, there will be context switch overheads associated with Compute Preemption, which is automatically enabled on those devices for which support exists. The individual attribute query function `cudaDeviceGetAttribute()` with the attribute `cudaDevAttrComputePreemptionSupported` can be used to determine if the device in use supports Compute Preemption. Users wishing to avoid context switch overheads associated with different processes can ensure that only one process is active on the GPU by selecting exclusive-process mode.

Applications may query the compute mode of a device by checking the attribute `cudaDevAttrComputeMode`.

## 6.5. Mode Switches[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mode-switches "Permalink to this headline")

GPUs that have a display output dedicate some DRAM memory to the so-called _primary surface_, which is used to refresh the display device whose output is viewed by the user. When users initiate a _mode switch_ of the display by changing the resolution or bit depth of the display (using NVIDIA control panel or the Display control panel on Windows), the amount of memory needed for the primary surface changes. For example, if the user changes the display resolution from 1280x1024x32-bit to 1600x1200x32-bit, the system must dedicate 7.68 MB to the primary surface rather than 5.24 MB. (Full-screen graphics applications running with anti-aliasing enabled may require much more display memory for the primary surface.) On Windows, other events that may initiate display mode switches include launching a full-screen DirectX application, hitting Alt+Tab to task switch away from a full-screen DirectX application, or hitting Ctrl+Alt+Del to lock the computer.

If a mode switch increases the amount of memory needed for the primary surface, the system may have to cannibalize memory allocations dedicated to CUDA applications. Therefore, a mode switch results in any call to the CUDA runtime to fail and return an invalid context error.

## 6.6. Tesla Compute Cluster Mode for Windows[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tesla-compute-cluster-mode-for-windows "Permalink to this headline")

Using NVIDIA’s System Management Interface (_nvidia-smi_), the Windows device driver can be put in TCC (Tesla Compute Cluster) mode for devices of the Tesla and Quadro Series.

TCC mode removes support for any graphics functionality.

# 7. Hardware Implementation[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-implementation "Permalink to this headline")

The NVIDIA GPU architecture is built around a scalable array of multithreaded _Streaming Multiprocessors_ (_SMs_). When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity. The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. As thread blocks terminate, new blocks are launched on the vacated multiprocessors.

A multiprocessor is designed to execute hundreds of threads concurrently. To manage such a large number of threads, it employs a unique architecture called _SIMT_ (_Single-Instruction, Multiple-Thread_) that is described in [SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture). The instructions are pipelined, leveraging instruction-level parallelism within a single thread, as well as extensive thread-level parallelism through simultaneous hardware multithreading as detailed in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading). Unlike CPU cores, they are issued in order and there is no branch prediction or speculative execution.

[SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture) and [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading) describe the architecture features of the streaming multiprocessor that are common to all devices. [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x), [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x), and [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x) provide the specifics for devices of compute capabilities 5.x, 6.x, and 7.x respectively.

The NVIDIA GPU architecture uses a little-endian representation.

## 7.1. SIMT Architecture[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture "Permalink to this headline")

The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called _warps_. Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. The term _warp_ originates from weaving, the first parallel thread technology. A _half-warp_ is either the first or second half of a warp. A _quarter-warp_ is either the first, second, third, or fourth quarter of a warp.

When a multiprocessor is given one or more thread blocks to execute, it partitions them into warps and each warp gets scheduled by a _warp scheduler_ for execution. The way a block is partitioned into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0. [Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy) describes how thread IDs relate to thread indices in the block.

A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, disabling threads that are not on that path. Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjoint code paths.

The SIMT architecture is akin to SIMD (Single Instruction, Multiple Data) vector organizations in that a single instruction controls multiple processing elements. A key difference is that SIMD vector organizations expose the SIMD width to the software, whereas SIMT instructions specify the execution and branching behavior of a single thread. In contrast with SIMD vector machines, SIMT enables programmers to write thread-level parallel code for independent, scalar threads, as well as data-parallel code for coordinated threads. For the purposes of correctness, the programmer can essentially ignore the SIMT behavior; however, substantial performance improvements can be realized by taking care that the code seldom requires threads in a warp to diverge. In practice, this is analogous to the role of cache lines in traditional code: Cache line size can be safely ignored when designing for correctness but must be considered in the code structure when designing for peak performance. Vector architectures, on the other hand, require the software to coalesce loads into vectors and manage divergence manually.

Prior to NVIDIA Volta, warps used a single program counter shared amongst all 32 threads in the warp together with an active mask specifying the active threads of the warp. As a result, threads from the same warp in divergent regions or different states of execution cannot signal each other or exchange data, and algorithms requiring fine-grained sharing of data guarded by locks or mutexes can easily lead to deadlock, depending on which warp the contending threads come from.

Starting with the NVIDIA Volta architecture, _Independent Thread Scheduling_ allows full concurrency between threads, regardless of warp. With Independent Thread Scheduling, the GPU maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity, either to make better use of execution resources or to allow one thread to wait for data to be produced by another. A schedule optimizer determines how to group active threads from the same warp together into SIMT units. This retains the high throughput of SIMT execution as in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at sub-warp granularity.

Independent Thread Scheduling can lead to a rather different set of threads participating in the executed code than intended if the developer made assumptions about warp-synchronicity[2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn2) of previous hardware architectures. In particular, any warp-synchronous code (such as synchronization-free, intra-warp reductions) should be revisited to ensure compatibility with NVIDIA Volta and beyond. See [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x) for further details.

Note

The threads of a warp that are participating in the current instruction are called the _active_ threads, whereas threads not on the current instruction are _inactive_ (disabled). Threads can be inactive for a variety of reasons including having exited earlier than other threads of their warp, having taken a different branch path than the branch path currently executed by the warp, or being the last threads of a block whose number of threads is not a multiple of the warp size.

If a non-atomic instruction executed by a warp writes to the same location in global or shared memory for more than one of the threads of the warp, the number of serialized writes that occur to that location varies depending on the compute capability of the device (see [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x), [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x), and [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x)), and which thread performs the final write is undefined.

If an [atomic](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions) instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined.

## 7.2. Hardware Multithreading[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading "Permalink to this headline")

The execution context (program counters, registers, and so on) for each warp processed by a multiprocessor is maintained on-chip during the entire lifetime of the warp. Therefore, switching from one execution context to another has no cost, and at every instruction issue time, a warp scheduler selects a warp that has threads ready to execute its next instruction (the [active threads](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes) of the warp) and issues the instruction to those threads.

In particular, each multiprocessor has a set of 32-bit registers that are partitioned among the warps, and a _parallel data cache_ or _shared memory_ that is partitioned among the thread blocks.

The number of blocks and warps that can reside and be processed together on the multiprocessor for a given kernel depends on the amount of registers and shared memory used by the kernel and the amount of registers and shared memory available on the multiprocessor. There are also a maximum number of resident blocks and a maximum number of resident warps per multiprocessor. These limits as well the amount of registers and shared memory available on the multiprocessor are a function of the compute capability of the device and are given in [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities). If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch.

The total number of warps in a block is as follows:

- _T_ is the number of threads per block,
    
- _Wsize_ is the warp size, which is equal to 32,
    
- ceil(x, y) is equal to x rounded up to the nearest multiple of y.
    

The total number of registers and total amount of shared memory allocated for a block are documented in the CUDA Occupancy Calculator provided in the CUDA Toolkit.

[2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id126)

The term _warp-synchronous_ refers to code that implicitly assumes threads in the same warp are synchronized at every instruction.

# 8. Performance Guidelines[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#performance-guidelines "Permalink to this headline")

## 8.1. Overall Performance Optimization Strategies[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overall-performance-optimization-strategies "Permalink to this headline")

Performance optimization revolves around four basic strategies:

- Maximize parallel execution to achieve maximum utilization;
    
- Optimize memory usage to achieve maximum memory throughput;
    
- Optimize instruction usage to achieve maximum instruction throughput;
    
- Minimize memory thrashing.
    

Which strategies will yield the best performance gain for a particular portion of an application depends on the performance limiters for that portion; optimizing instruction usage of a kernel that is mostly limited by memory accesses will not yield any significant performance gain, for example. Optimization efforts should therefore be constantly directed by measuring and monitoring the performance limiters, for example using the CUDA profiler. Also, comparing the floating-point operation throughput or memory throughput—whichever makes more sense—of a particular kernel to the corresponding peak theoretical throughput of the device indicates how much room for improvement there is for the kernel.

## 8.2. Maximize Utilization[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximize-utilization "Permalink to this headline")

To maximize utilization the application should be structured in a way that it exposes as much parallelism as possible and efficiently maps this parallelism to the various components of the system to keep them busy most of the time.

### 8.2.1. Application Level[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-level "Permalink to this headline")

At a high level, the application should maximize parallel execution between the host, the devices, and the bus connecting the host to the devices, by using asynchronous functions calls and streams as described in [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution). It should assign to each processor the type of work it does best: serial workloads to the host; parallel workloads to the devices.

For the parallel workloads, at points in the algorithm where parallelism is broken because some threads need to synchronize in order to share data with each other, there are two cases: Either these threads belong to the same block, in which case they should use `__syncthreads()` and share data through shared memory within the same kernel invocation, or they belong to different blocks, in which case they must share data through global memory using two separate kernel invocations, one for writing to and one for reading from global memory. The second case is much less optimal since it adds the overhead of extra kernel invocations and global memory traffic. Its occurrence should therefore be minimized by mapping the algorithm to the CUDA programming model in such a way that the computations that require inter-thread communication are performed within a single thread block as much as possible.

### 8.2.2. Device Level[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-level "Permalink to this headline")

At a lower level, the application should maximize parallel execution between the multiprocessors of a device.

Multiple kernels can execute concurrently on a device, so maximum utilization can also be achieved by using streams to enable enough kernels to execute concurrently as described in [Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution).

### 8.2.3. Multiprocessor Level[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#multiprocessor-level "Permalink to this headline")

At an even lower level, the application should maximize parallel execution between the various functional units within a multiprocessor.

As described in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading), a GPU multiprocessor primarily relies on thread-level parallelism to maximize utilization of its functional units. Utilization is therefore directly linked to the number of resident warps. At every instruction issue time, a warp scheduler selects an instruction that is ready to execute. This instruction can be another independent instruction of the same warp, exploiting instruction-level parallelism, or more commonly an instruction of another warp, exploiting thread-level parallelism. If a ready to execute instruction is selected it is issued to the [active](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes) threads of the warp. The number of clock cycles it takes for a warp to be ready to execute its next instruction is called the _latency_, and full utilization is achieved when all warp schedulers always have some instruction to issue for some warp at every clock cycle during that latency period, or in other words, when latency is completely “hidden”. The number of instructions required to hide a latency of L clock cycles depends on the respective throughputs of these instructions (see the [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions) for the throughputs of various arithmetic instructions). If we assume instructions with maximum throughput, it is equal to:

- _4L_ for devices of compute capability 5.x, 6.1, 6.2, 7.x and 8.x since for these devices, a multiprocessor issues one instruction per warp over one clock cycle for four warps at a time, as mentioned in [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capabilities).
    
- _2L_ for devices of compute capability 6.0 since for these devices, the two instructions issued every cycle are one instruction for two different warps.
    

The most common reason a warp is not ready to execute its next instruction is that the instruction’s input operands are not available yet.

If all input operands are registers, latency is caused by register dependencies, i.e., some of the input operands are written by some previous instruction(s) whose execution has not completed yet. In this case, the latency is equal to the execution time of the previous instruction and the warp schedulers must schedule instructions of other warps during that time. Execution time varies depending on the instruction. On devices of compute capability 7.x, for most arithmetic instructions, it is typically 4 clock cycles. This means that 16 active warps per multiprocessor (4 cycles, 4 warp schedulers) are required to hide arithmetic instruction latencies (assuming that warps execute instructions with maximum throughput, otherwise fewer warps are needed). If the individual warps exhibit instruction-level parallelism, i.e. have multiple independent instructions in their instruction stream, fewer warps are needed because multiple independent instructions from a single warp can be issued back to back.

If some input operand resides in off-chip memory, the latency is much higher: typically hundreds of clock cycles. The number of warps required to keep the warp schedulers busy during such high latency periods depends on the kernel code and its degree of instruction-level parallelism. In general, more warps are required if the ratio of the number of instructions with no off-chip memory operands (i.e., arithmetic instructions most of the time) to the number of instructions with off-chip memory operands is low (this ratio is commonly called the arithmetic intensity of the program).

Another reason a warp is not ready to execute its next instruction is that it is waiting at some memory fence ([Memory Fence Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-functions)) or synchronization point ([Synchronization Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-functions)). A synchronization point can force the multiprocessor to idle as more and more warps wait for other warps in the same block to complete execution of instructions prior to the synchronization point. Having multiple resident blocks per multiprocessor can help reduce idling in this case, as warps from different blocks do not need to wait for each other at synchronization points.

The number of blocks and warps residing on each multiprocessor for a given kernel call depends on the execution configuration of the call ([Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)), the memory resources of the multiprocessor, and the resource requirements of the kernel as described in [Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading). Register and shared memory usage are reported by the compiler when compiling with the `--ptxas-options=-v` option.

The total amount of shared memory required for a block is equal to the sum of the amount of statically allocated shared memory and the amount of dynamically allocated shared memory.

The number of registers used by a kernel can have a significant impact on the number of resident warps. For example, for devices of compute capability 6.x, if a kernel uses 64 registers and each block has 512 threads and requires very little shared memory, then two blocks (i.e., 32 warps) can reside on the multiprocessor since they require 2x512x64 registers, which exactly matches the number of registers available on the multiprocessor. But as soon as the kernel uses one more register, only one block (i.e., 16 warps) can be resident since two blocks would require 2x512x65 registers, which are more registers than are available on the multiprocessor. Therefore, the compiler attempts to minimize register usage while keeping register spilling (see [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)) and the number of instructions to a minimum. Register usage can be controlled using the `maxrregcount` compiler option, the `__launch_bounds__()` qualifier as described in [Launch Bounds](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#launch-bounds), or the `__maxnreg__()` qualifier as described in [Maximum Number of Registers per Thread](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximum-number-of-registers-per-thread).

The register file is organized as 32-bit registers. So, each variable stored in a register needs at least one 32-bit register, for example, a `double` variable uses two 32-bit registers.

The effect of execution configuration on performance for a given kernel call generally depends on the kernel code. Experimentation is therefore recommended. Applications can also parametrize execution configurations based on register file size and shared memory size, which depends on the compute capability of the device, as well as on the number of multiprocessors and memory bandwidth of the device, all of which can be queried using the runtime (see reference manual).

The number of threads per block should be chosen as a multiple of the warp size to avoid wasting computing resources with under-populated warps as much as possible.

#### 8.2.3.1. Occupancy Calculator[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#occupancy-calculator "Permalink to this headline")

Several API functions exist to assist programmers in choosing thread block size and cluster size based on register and shared memory requirements.

- The occupancy calculator API, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, can provide an occupancy prediction based on the block size and shared memory usage of a kernel. This function reports occupancy in terms of the number of concurrent thread blocks per multiprocessor.
    
    - Note that this value can be converted to other metrics. Multiplying by the number of warps per block yields the number of concurrent warps per multiprocessor; further dividing concurrent warps by max warps per multiprocessor gives the occupancy as a percentage.
        
- The occupancy-based launch configurator APIs, `cudaOccupancyMaxPotentialBlockSize` and `cudaOccupancyMaxPotentialBlockSizeVariableSMem`, heuristically calculate an execution configuration that achieves the maximum multiprocessor-level occupancy.
    
- The occupancy calculator API, `cudaOccupancyMaxActiveClusters`, can provided occupancy prediction based on the cluster size, block size and shared memory usage of a kernel. This function reports occupancy in terms of number of max active clusters of a given size on the GPU present in the system.
    

The following code sample calculates the occupancy of MyKernel. It then reports the occupancy level with the ratio between concurrent warps versus maximum warps per multiprocessor.

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

The following code sample configures an occupancy-based kernel launch of MyKernel according to the user input.

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

The following code sample shows how to use the cluster occupancy API to find the max number of active clusters of a given size. Example code below calcualtes occupancy for cluster of size 2 and 128 threads per block.

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

The CUDA Nsight Compute User Interface also provides a standalone occupancy calculator and launch configurator implementation in `<CUDA_Toolkit_Path>/include/cuda_occupancy.h` for any use cases that cannot depend on the CUDA software stack. The Nsight Compute version of the occupancy calculator is particularly useful as a learning tool that visualizes the impact of changes to the parameters that affect occupancy (block size, registers per thread, and shared memory per thread).

## 8.3. Maximize Memory Throughput[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximize-memory-throughput "Permalink to this headline")

The first step in maximizing overall memory throughput for the application is to minimize data transfers with low bandwidth.

That means minimizing data transfers between the host and the device, as detailed in [Data Transfer between Host and Device](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-transfer-between-host-and-device), since these have much lower bandwidth than data transfers between global memory and the device.

That also means minimizing data transfers between global memory and the device by maximizing use of on-chip memory: shared memory and caches (i.e., L1 cache and L2 cache available on devices of compute capability 2.x and higher, texture cache and constant cache available on all devices).

Shared memory is equivalent to a user-managed cache: The application explicitly allocates and accesses it. As illustrated in [CUDA Runtime](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-c-runtime), a typical programming pattern is to stage data coming from device memory into shared memory; in other words, to have each thread of a block:

- Load data from device memory to shared memory,
    
- Synchronize with all the other threads of the block so that each thread can safely read shared memory locations that were populated by different threads,
    
- Process the data in shared memory,
    
- Synchronize again if necessary to make sure that shared memory has been updated with the results,
    
- Write the results back to device memory.
    

For some applications (for example, for which global memory access patterns are data-dependent), a traditional hardware-managed cache is more appropriate to exploit data locality. As mentioned in [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x), [Compute Capability 8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x) and [Compute Capability 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0), for devices of compute capability 7.x, 8.x and 9.0, the same on-chip memory is used for both L1 and shared memory, and how much of it is dedicated to L1 versus shared memory is configurable for each kernel call.

The throughput of memory accesses by a kernel can vary by an order of magnitude depending on access pattern for each type of memory. The next step in maximizing memory throughput is therefore to organize memory accesses as optimally as possible based on the optimal memory access patterns described in [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses). This optimization is especially important for global memory accesses as global memory bandwidth is low compared to available on-chip bandwidths and arithmetic instruction throughput, so non-optimal global memory accesses generally have a high impact on performance.

### 8.3.1. Data Transfer between Host and Device[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#data-transfer-between-host-and-device "Permalink to this headline")

Applications should strive to minimize data transfer between the host and the device. One way to accomplish this is to move more code from the host to the device, even if that means running kernels that do not expose enough parallelism to execute on the device with full efficiency. Intermediate data structures may be created in device memory, operated on by the device, and destroyed without ever being mapped by the host or copied to host memory.

Also, because of the overhead associated with each transfer, batching many small transfers into a single large transfer always performs better than making each transfer separately.

On systems with a front-side bus, higher performance for data transfers between host and device is achieved by using page-locked host memory as described in [Page-Locked Host Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#page-locked-host-memory).

In addition, when using mapped page-locked memory ([Mapped Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mapped-memory)), there is no need to allocate any device memory and explicitly copy data between device and host memory. Data transfers are implicitly performed each time the kernel accesses the mapped memory. For maximum performance, these memory accesses must be coalesced as with accesses to global memory (see [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)). Assuming that they are and that the mapped memory is read or written only once, using mapped page-locked memory instead of explicit copies between device and host memory can be a win for performance.

On integrated systems where device memory and host memory are physically the same, any copy between host and device memory is superfluous and mapped page-locked memory should be used instead. Applications may query a device is `integrated` by checking that the integrated device property (see [Device Enumeration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-enumeration)) is equal to 1.

### 8.3.2. Device Memory Accesses[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses "Permalink to this headline")

An instruction that accesses addressable memory (i.e., global, local, shared, constant, or texture memory) might need to be re-issued multiple times depending on the distribution of the memory addresses across the threads within the warp. How the distribution affects the instruction throughput this way is specific to each type of memory and described in the following sections. For example, for global memory, as a general rule, the more scattered the addresses are, the more reduced the throughput is.

**Global Memory**

Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned: Only the 32-, 64-, or 128-byte segments of device memory that are aligned to their size (i.e., whose first address is a multiple of their size) can be read or written by memory transactions.

When a warp executes an instruction that accesses global memory, it coalesces the memory accesses of the threads within the warp into one or more of these memory transactions depending on the size of the word accessed by each thread and the distribution of the memory addresses across the threads. In general, the more transactions are necessary, the more unused words are transferred in addition to the words accessed by the threads, reducing the instruction throughput accordingly. For example, if a 32-byte memory transaction is generated for each thread’s 4-byte access, throughput is divided by 8.

How many transactions are necessary and how much throughput is ultimately affected varies with the compute capability of the device. [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x), [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x), [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x), [Compute Capability 8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x), [Compute Capability 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0), [Compute Capability 10.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-10-0), and [Compute Capability 12.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-12-0) give more details on how global memory accesses are handled for various compute capabilities.

To maximize global memory throughput, it is therefore important to maximize coalescing by:

- Following the most optimal access patterns based on [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x), [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x), [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x), [Compute Capability 8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x), [Compute Capability 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0), [Compute Capability 10.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-10-0), and [Compute Capability 12.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-12-0).
    
- Using data types that meet the size and alignment requirement detailed in the section Size and Alignment Requirement below,
    
- Padding data in some cases, for example, when accessing a two-dimensional array as described in the section Two-Dimensional Arrays below.
    

**Size and Alignment Requirement**

Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to data residing in global memory compiles to a single global memory instruction if and only if the size of the data type is 1, 2, 4, 8, or 16 bytes and the data is naturally aligned (i.e., its address is a multiple of that size).

If this size and alignment requirement is not fulfilled, the access compiles to multiple instructions with interleaved access patterns that prevent these instructions from fully coalescing. It is therefore recommended to use types that meet this requirement for data that resides in global memory.

The alignment requirement is automatically fulfilled for the [Built-in Vector Types](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-vector-types).

For structures, the size and alignment requirements can be enforced by the compiler using the alignment specifiers`__align__(8) or __align__(16)`, such as

struct __align__(8) {
    float x;
    float y;
};

or

struct __align__(16) {
    float x;
    float y;
    float z;
};

Any address of a variable residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is always aligned to at least 256 bytes.

Reading non-naturally aligned 8-byte or 16-byte words produces incorrect results (off by a few words), so special care must be taken to maintain alignment of the starting address of any value or array of values of these types. A typical case where this might be easily overlooked is when using some custom global memory allocation scheme, whereby the allocations of multiple arrays (with multiple calls to `cudaMalloc()` or `cuMemAlloc()`) is replaced by the allocation of a single large block of memory partitioned into multiple arrays, in which case the starting address of each array is offset from the block’s starting address.

**Two-Dimensional Arrays**

A common global memory access pattern is when each thread of index `(tx,ty)` uses the following address to access one element of a 2D array of width `width`, located at address `BaseAddress` of type `type*` (where `type` meets the requirement described in [Maximize Utilization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximize-utilization)):

BaseAddress + width * ty + tx

For these accesses to be fully coalesced, both the width of the thread block and the width of the array must be a multiple of the warp size.

In particular, this means that an array whose width is not a multiple of this size will be accessed much more efficiently if it is actually allocated with a width rounded up to the closest multiple of this size and its rows padded accordingly. The `cudaMallocPitch()` and `cuMemAllocPitch()` functions and associated memory copy functions described in the reference manual enable programmers to write non-hardware-dependent code to allocate arrays that conform to these constraints.

**Local Memory**

Local memory accesses only occur for some automatic variables as mentioned in [Variable Memory Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#variable-memory-space-specifiers). Automatic variables that the compiler is likely to place in local memory are:

- Arrays for which it cannot determine that they are indexed with constant quantities,
    
- Large structures or arrays that would consume too much register space,
    
- Any variable if the kernel uses more registers than available (this is also known as _register spilling_).
    

Inspection of the _PTX_ assembly code (obtained by compiling with the `-ptx` or`-keep` option) will tell if a variable has been placed in local memory during the first compilation phases as it will be declared using the `.local` mnemonic and accessed using the `ld.local` and `st.local` mnemonics. Even if it has not, subsequent compilation phases might still decide otherwise though if they find it consumes too much register space for the targeted architecture: Inspection of the _cubin_ object using `cuobjdump` will tell if this is the case. Also, the compiler reports total local memory usage per kernel (`lmem`) when compiling with the `--ptxas-options=-v` option. Note that some mathematical functions have implementation paths that might access local memory.

The local memory space resides in device memory, so local memory accesses have the same high latency and low bandwidth as global memory accesses and are subject to the same requirements for memory coalescing as described in [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses). Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address (for example, same index in an array variable, same member in a structure variable).

On devices of compute capability 5.x onwards, local memory accesses are always cached in L2 in the same way as global memory accesses (see [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x) and [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x)).

**Shared Memory**

Because it is on-chip, shared memory has much higher bandwidth and much lower latency than local or global memory.

To achieve high bandwidth, shared memory is divided into equally-sized memory modules, called banks, which can be accessed simultaneously. Any memory read or write request made of _n_ addresses that fall in _n_ distinct memory banks can therefore be serviced simultaneously, yielding an overall bandwidth that is _n_ times as high as the bandwidth of a single module.

However, if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized. The hardware splits a memory request with bank conflicts into as many separate conflict-free requests as necessary, decreasing throughput by a factor equal to the number of separate memory requests. If the number of separate memory requests is _n_, the initial memory request is said to cause _n_-way bank conflicts.

To get maximum performance, it is therefore important to understand how memory addresses map to memory banks in order to schedule the memory requests so as to minimize bank conflicts. This is described in [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-5-x), [Compute Capability 6.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-6-x), [Compute Capability 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-7-x), [Compute Capability 8.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-8-x), [Compute Capability 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-9-0), [Compute Capability 10.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-10-0), and [Compute Capability 12.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compute-capability-12-0) for devices of these compute capabilities respectively.

**Constant Memory**

The constant memory space resides in device memory and is cached in the constant cache.

A request is then split into as many separate requests as there are different memory addresses in the initial request, decreasing throughput by a factor equal to the number of separate requests.

The resulting requests are then serviced at the throughput of the constant cache in case of a cache hit, or at the throughput of device memory otherwise.

**Texture and Surface Memory**

The texture and surface memory spaces reside in device memory and are cached in texture cache, so a texture fetch or surface read costs one memory read from device memory only on a cache miss, otherwise it just costs one read from texture cache. The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture or surface addresses that are close together in 2D will achieve best performance. Also, it is designed for streaming fetches with a constant latency; a cache hit reduces DRAM bandwidth demand but not fetch latency.

Reading device memory through texture or surface fetching present some benefits that can make it an advantageous alternative to reading device memory from global or constant memory:

- If the memory reads do not follow the access patterns that global or constant memory reads must follow to get good performance, higher bandwidth can be achieved providing that there is locality in the texture fetches or surface reads;
    
- Addressing calculations are performed outside the kernel by dedicated units;
    
- Packed data may be broadcast to separate variables in a single operation;
    
- 8-bit and 16-bit integer input data may be optionally converted to 32 bit floating-point values in the range [0.0, 1.0] or [-1.0, 1.0] (see [Texture Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory)).
    

## 8.4. Maximize Instruction Throughput[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximize-instruction-throughput "Permalink to this headline")

See the [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput) for more details on optimizing instruction throughput.

## 8.5. Minimize Memory Thrashing[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#minimize-memory-thrashing "Permalink to this headline")

Applications that constantly allocate and free memory too often may find that the allocation calls tend to get slower over time up to a limit. This is typically expected due to the nature of releasing memory back to the operating system for its own use. For best performance in this regard, we recommend the following:

- Try to size your allocation to the problem at hand. Don’t try to allocate all available memory with `cudaMalloc` / `cudaMallocHost` / `cuMemCreate`, as this forces memory to be resident immediately and prevents other applications from being able to use that memory. This can put more pressure on operating system schedulers, or just prevent other applications using the same GPU from running entirely.
    
- Try to allocate memory in appropriately sized allocations early in the application and allocations only when the application does not have any use for it. Reduce the number of `cudaMalloc`+`cudaFree` calls in the application, especially in performance-critical regions.
    
- If an application cannot allocate enough device memory, consider falling back on other memory types such as `cudaMallocHost` or `cudaMallocManaged`, which may not be as performant, but will enable the application to make progress.
    
- For platforms that support the feature, `cudaMallocManaged` allows for oversubscription, and with the correct `cudaMemAdvise` policies enabled, will allow the application to retain most if not all the performance of `cudaMalloc`. `cudaMallocManaged` also won’t force an allocation to be resident until it is needed or prefetched, reducing the overall pressure on the operating system schedulers and better enabling multi-tenet use cases.
    

# 9. CUDA-Enabled GPUs[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-enabled-gpus "Permalink to this headline")

[https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) lists all CUDA-enabled devices with their compute capability.

The compute capability, number of multiprocessors, clock frequency, total amount of device memory, and other properties can be queried using the runtime (see reference manual).

# 10. C++ Language Extensions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions "Permalink to this headline")

## 10.1. Function Execution Space Specifiers[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#function-execution-space-specifiers "Permalink to this headline")

Function execution space specifiers denote whether a function executes on the host or on the device and whether it is callable from the host or from the device.

### 10.1.1. __global__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global "Permalink to this headline")

The `__global__` execution space specifier declares a function as being a kernel. Such a function is:

- Executed on the device,
    
- Callable from the host,
    
- Callable from the device for devices of compute capability 5.0 or higher (see [CUDA Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-dynamic-parallelism) for more details).
    

A `__global__` function must have void return type, and cannot be a member of a class.

Any call to a `__global__` function must specify its execution configuration as described in [Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration).

A call to a `__global__` function is asynchronous, meaning it returns before the device has completed its execution.

### 10.1.2. __device__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device "Permalink to this headline")

The `__device__` execution space specifier declares a function that is:

- Executed on the device,
    
- Callable from the device only.
    

The `__global__` and `__device__` execution space specifiers cannot be used together.

### 10.1.3. __host__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#host "Permalink to this headline")

The `__host__` execution space specifier declares a function that is:

- Executed on the host,
    
- Callable from the host only.
    

It is equivalent to declare a function with only the `__host__` execution space specifier or to declare it without any of the `__host__`, `__device__`, or `__global__` execution space specifier; in either case the function is compiled for the host only.

The `__global__` and `__host__` execution space specifiers cannot be used together.

The `__device__` and `__host__` execution space specifiers can be used together however, in which case the function is compiled for both the host and the device. The `__CUDA_ARCH__` macro introduced in [Application Compatibility](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#application-compatibility) can be used to differentiate code paths between host and device:

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

### 10.1.4. Undefined behavior[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#undefined-behavior "Permalink to this headline")

A ‘cross-execution space’ call has undefined behavior when:

- `__CUDA_ARCH__` is defined, a call from within a `__global__`, `__device__` or `__host__ __device__` function to a `__host__` function.
    
- `__CUDA_ARCH__` is undefined, a call from within a `__host__` function to a `__device__` function. [4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn11)
    

### 10.1.5. __noinline__ and __forceinline__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#noinline-and-forceinline "Permalink to this headline")

The compiler inlines any `__device__` function when deemed appropriate.

The `__noinline__` function qualifier can be used as a hint for the compiler not to inline the function if possible.

The `__forceinline__` function qualifier can be used to force the compiler to inline the function.

The `__noinline__` and `__forceinline__` function qualifiers cannot be used together, and neither function qualifier can be applied to an inline function.

### 10.1.6. __inline_hint__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#inline-hint "Permalink to this headline")

The `__inline_hint__` qualifier enables more aggressive inlining in the compiler. Unlike `__forceinline__`, it does not imply that the function is inline. It can be used to improve inlining across modules when using LTO.

Neither the `__noinline__` nor the `__forceinline__` function qualifier can be used with the `__inline_hint__` function qualifier.

## 10.2. Variable Memory Space Specifiers[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#variable-memory-space-specifiers "Permalink to this headline")

Variable memory space specifiers denote the memory location on the device of a variable.

An automatic variable declared in device code without any of the `__device__`, `__shared__` and `__constant__` memory space specifiers described in this section generally resides in a register. However in some cases the compiler might choose to place it in local memory, which can have adverse performance consequences as detailed in [Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses).

### 10.2.1. __device__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-variable-specifier "Permalink to this headline")

The `__device__` memory space specifier declares a variable that resides on the device.

At most one of the other memory space specifiers defined in the next three sections may be used together with `__device__` to further denote which memory space the variable belongs to. If none of them is present, the variable:

- Resides in global memory space,
    
- Has the lifetime of the CUDA context in which it is created,
    
- Has a distinct object per device,
    
- Is accessible from all the threads within the grid and from the host through the runtime library `(cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`).
    

### 10.2.2. __constant__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant "Permalink to this headline")

The `__constant__` memory space specifier, optionally used together with `__device__`, declares a variable that:

- Resides in constant memory space,
    
- Has the lifetime of the CUDA context in which it is created,
    
- Has a distinct object per device,
    
- Is accessible from all the threads within the grid and from the host through the runtime library (`cudaGetSymbolAddress()` / `cudaGetSymbolSize()` / `cudaMemcpyToSymbol()` / `cudaMemcpyFromSymbol()`).
    

The behavior of modifying a constant from the host while there is a concurrent grid that access that constant at any point of this grid’s lifetime is undefined.

### 10.2.3. __shared__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared "Permalink to this headline")

The `__shared__` memory space specifier, optionally used together with `__device__`, declares a variable that:

- Resides in the shared memory space of a thread block,
    
- Has the lifetime of the block,
    
- Has a distinct object per block,
    
- Is only accessible from all the threads within the block,
    
- Does not have a constant address.
    

When declaring a variable in shared memory as an external array such as

extern __shared__ float shared[];

the size of the array is determined at launch time (see [Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#execution-configuration)). All variables declared in this fashion, start at the same address in memory, so that the layout of the variables in the array must be explicitly managed through offsets. For example, if one wants the equivalent of

short array0[128];
float array1[64];
int   array2[256];

in dynamically allocated shared memory, one could declare and initialize the arrays the following way:

extern __shared__ float array[];
__device__ void func()      // __device__ or __global__ function
{
    short* array0 = (short*)array;
    float* array1 = (float*)&array0[128];
    int*   array2 =   (int*)&array1[64];
}

Note that pointers need to be aligned to the type they point to, so the following code, for example, does not work since array1 is not aligned to 4 bytes.

extern __shared__ float array[];
__device__ void func()      // __device__ or __global__ function
{
    short* array0 = (short*)array;
    float* array1 = (float*)&array0[127];
}

Alignment requirements for the built-in vector types are listed in [Table 7](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types-alignment-requirements-in-device-code).

### 10.2.4. __grid_constant__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#grid-constant "Permalink to this headline")

The `__grid_constant__` annotation for compute architectures greater or equal to 7.0 annotates a `const`-qualified `__global__` function parameter of non-reference type that:

- Has the lifetime of the grid,
    
- Is private to the grid, i.e., the object is not accessible to host threads and threads from other grids, including sub-grids,
    
- Has a distinct object per grid, i.e., all threads in the grid see the same address,
    
- Is read-only, i.e., modifying a `__grid_constant__` object or any of its sub-objects is _undefined behavior_, including `mutable` members.
    

Requirements:

- Kernel parameters annotated with `__grid_constant__` must have `const`-qualified non-reference types.
    
- All function declarations must match with respect to any `__grid_constant_` parameters.
    
- A function template specialization must match the primary template declaration with respect to any `__grid_constant__` parameters.
    
- A function template instantiation directive must match the primary template declaration with respect to any `__grid_constant__` parameters.
    

If the address of a `__global__` function parameter is taken, the compiler will ordinarily make a copy of the kernel parameter in thread local memory and use the address of the copy, to partially support C++ semantics, which allow each thread to modify its own local copy of function parameters. Annotating a `__global__` function parameter with `__grid_constant__` ensures that the compiler will not create a copy of the kernel parameter in thread local memory, but will instead use the generic address of the parameter itself. Avoiding the local copy may result in improved performance.

__device__ void unknown_function(S const&);
__global__ void kernel(const __grid_constant__ S s) {
   s.x += threadIdx.x;  // Undefined Behavior: tried to modify read-only memory

   // Compiler will _not_ create a per-thread thread local copy of "s":
   unknown_function(s);
}

### 10.2.5. __managed__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#managed "Permalink to this headline")

The `__managed__` memory space specifier, optionally used together with `__device__`, declares a variable that:

- Can be referenced from both device and host code, for example, its address can be taken or it can be read or written directly from a device or host function.
    
- Has the lifetime of an application.
    

See [__managed__ Memory Space Specifier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#managed-specifier) for more details.

### 10.2.6. __restrict__[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#restrict "Permalink to this headline")

`nvcc` supports restricted pointers via the `__restrict__` keyword.

Restricted pointers were introduced in C99 to alleviate the aliasing problem that exists in C-type languages, and which inhibits all kind of optimization from code re-ordering to common sub-expression elimination.

Here is an example subject to the aliasing issue, where use of restricted pointer can help the compiler to reduce the number of instructions:

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

In C-type languages, the pointers `a`, `b`, and `c` may be aliased, so any write through `c` could modify elements of `a` or `b`. This means that to guarantee functional correctness, the compiler cannot load `a[0]` and `b[0]` into registers, multiply them, and store the result to both `c[0]` and `c[1]`, because the results would differ from the abstract execution model if, say, `a[0]` is really the same location as `c[0]`. So the compiler cannot take advantage of the common sub-expression. Likewise, the compiler cannot just reorder the computation of `c[4]` into the proximity of the computation of `c[0]` and `c[1]` because the preceding write to `c[3]` could change the inputs to the computation of `c[4]`.

By making `a`, `b`, and `c` restricted pointers, the programmer asserts to the compiler that the pointers are in fact not aliased, which in this case means writes through `c` would never overwrite elements of `a` or `b`. This changes the function prototype as follows:

void foo(const float* __restrict__ a,
         const float* __restrict__ b,
         float* __restrict__ c);

Note that all pointer arguments need to be made restricted for the compiler optimizer to derive any benefit. With the `__restrict__` keywords added, the compiler can now reorder and do common sub-expression elimination at will, while retaining functionality identical with the abstract execution model:

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

The effects here are a reduced number of memory accesses and reduced number of computations. This is balanced by an increase in register pressure due to “cached” loads and common sub-expressions.

Since register pressure is a critical issue in many CUDA codes, use of restricted pointers can have negative performance impact on CUDA code, due to reduced occupancy.

## 10.3. Built-in Vector Types[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-vector-types "Permalink to this headline")

### 10.3.1. char, short, int, long, longlong, float, double[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#char-short-int-long-longlong-float-double "Permalink to this headline")

These are vector types derived from the basic integer and floating-point types. They are structures and the 1st, 2nd, 3rd, and 4th components are accessible through the fields `x`, `y`, `z`, and `w`, respectively. They all come with a constructor function of the form `make_<type name>`; for example,

int2 make_int2(int x, int y);

which creates a vector of type `int2` with value`(x, y)`.

The alignment requirements of the vector types are detailed in [Table 7](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types-alignment-requirements-in-device-code).

Table 7 Alignment Requirements[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types-alignment-requirements-in-device-code "Permalink to this table")
|Type|Alignment|
|---|---|
|char1, uchar1|1|
|char2, uchar2|2|
|char3, uchar3|1|
|char4, uchar4|4|
|short1, ushort1|2|
|short2, ushort2|4|
|short3, ushort3|2|
|short4, ushort4|8|
|int1, uint1|4|
|int2, uint2|8|
|int3, uint3|4|
|int4, uint4|16|
|long1, ulong1|4 if sizeof(long) is equal to sizeof(int) 8, otherwise|
|long2, ulong2|8 if sizeof(long) is equal to sizeof(int), 16, otherwise|
|long3, ulong3|4 if sizeof(long) is equal to sizeof(int), 8, otherwise|
|long4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|long4_16a|
|long4_32a|32|
|ulong4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|ulong4_16a|
|ulong4_32a|32|
|longlong1, ulonglong1|8|
|longlong2, ulonglong2|16|
|longlong3, ulonglong3|8|
|longlong4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|longlong4_16a|
|longlong4_32a|32|
|ulonglong4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|ulonglong4_16a|
|ulonglong4_32a|32|
|float1|4|
|float2|8|
|float3|4|
|float4|16|
|double1|8|
|double2|16|
|double3|8|
|double4 [3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn32a)|16|
|double4_16a|
|double4_32a|32|

3([1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id155),[2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id156),[3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id157),[4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id158),[5](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#id159))

This vector type was deprecated in CUDA Toolkit 13.0. Please use the `_16a` or `_32a` variant instead, depending on your alignment requirements.

### 10.3.2. dim3[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3 "Permalink to this headline")

This type is an integer vector type based on `uint3` that is used to specify dimensions. When defining a variable of type `dim3`, any component left unspecified is initialized to 1.

## 10.4. Built-in Variables[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables "Permalink to this headline")

Built-in variables specify the grid and block dimensions and the block and thread indices. They are only valid within functions that are executed on the device.

### 10.4.1. gridDim[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#griddim "Permalink to this headline")

This variable is of type `dim3` (see [dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3)) and contains the dimensions of the grid.

### 10.4.2. blockIdx[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#blockidx "Permalink to this headline")

This variable is of type `uint3` (see [char, short, int, long, longlong, float, double](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types)) and contains the block index within the grid.

### 10.4.3. blockDim[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#blockdim "Permalink to this headline")

This variable is of type `dim3` (see [dim3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3)) and contains the dimensions of the block.

### 10.4.4. threadIdx[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#threadidx "Permalink to this headline")

This variable is of type `uint3` (see [char, short, int, long, longlong, float, double](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#vector-types)) and contains the thread index within the block.

### 10.4.5. warpSize[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warpsize "Permalink to this headline")

This variable is of type `int` and contains the warp size in threads (see [SIMT Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture) for the definition of a warp).

## 10.5. Memory Fence Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-fence-functions "Permalink to this headline")

The CUDA programming model assumes a device with a weakly-ordered memory model, that is the order in which a CUDA thread writes data to shared memory, global memory, page-locked host memory, or the memory of a peer device is not necessarily the order in which the data is observed being written by another CUDA or host thread. It is undefined behavior for two threads to read from or write to the same memory location without synchronization.

In the following example, thread 1 executes `writeXY()`, while thread 2 executes `readXY()`.

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

The two threads read and write from the same memory locations `X` and `Y` simultaneously. Any data-race is undefined behavior, and has no defined semantics. The resulting values for `A` and `B` can be anything.

Memory fence functions can be used to enforce a [sequentially-consistent](https://en.cppreference.com/w/cpp/atomic/memory_order) ordering on memory accesses. The memory fence functions differ in the [scope](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes) in which the orderings are enforced but they are independent of the accessed memory space (shared memory, global memory, page-locked host memory, and the memory of a peer device).

void __threadfence_block();

is equivalent to [cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block)](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html) and ensures that:

- All writes to all memory made by the calling thread before the call to `__threadfence_block()` are observed by all threads in the block of the calling thread as occurring before all writes to all memory made by the calling thread after the call to `__threadfence_block()`;
    
- All reads from all memory made by the calling thread before the call to `__threadfence_block()` are ordered before all reads from all memory made by the calling thread after the call to `__threadfence_block()`.
    

void __threadfence();

is equivalent to [cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device)](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html) and ensures that no writes to all memory made by the calling thread after the call to `__threadfence()` are observed by any thread in the device as occurring before any write to all memory made by the calling thread before the call to `__threadfence()`.

void __threadfence_system();

is equivalent to [cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system)](https://nvidia.github.io/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html) and ensures that all writes to all memory made by the calling thread before the call to `__threadfence_system()` are observed by all threads in the device, host threads, and all threads in peer devices as occurring before all writes to all memory made by the calling thread after the call to `__threadfence_system()`.

`__threadfence_system()` is only supported by devices of compute capability 2.x and higher.

In the previous code sample, we can insert fences in the codes as follows:

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

For this code, the following outcomes can be observed:

- `A` equal to 1 and `B` equal to 2,
    
- `A` equal to 10 and `B` equal to 2,
    
- `A` equal to 10 and `B` equal to 20.
    

The fourth outcome is not possible, because the first write must be visible before the second write. If thread 1 and 2 belong to the same block, it is enough to use `__threadfence_block()`. If thread 1 and 2 do not belong to the same block, `__threadfence()` must be used if they are CUDA threads from the same device and `__threadfence_system()` must be used if they are CUDA threads from two different devices.

A common use case is when threads consume some data produced by other threads as illustrated by the following code sample of a kernel that computes the sum of an array of N numbers in one call. Each block first sums a subset of the array and stores the result in global memory. When all blocks are done, the last block done reads each of these partial sums from global memory and sums them to obtain the final result. In order to determine which block is finished last, each block atomically increments a counter to signal that it is done with computing and storing its partial sum (see [Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions) about atomic functions). The last block is the one that receives the counter value equal to `gridDim.x-1`. If no fence is placed between storing the partial sum and incrementing the counter, the counter might increment before the partial sum is stored and therefore, might reach `gridDim.x-1` and let the last block start reading partial sums before they have been actually updated in memory.

Memory fence functions only affect the ordering of memory operations by a thread; they do not, by themselves, ensure that these memory operations are visible to other threads (like `__syncthreads()` does for threads within a block; see [Synchronization Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-functions)). In the code sample below, the visibility of memory operations on the `result` variable is ensured by declaring it as volatile (see [Volatile Qualifier](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#volatile-qualifier)).

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

## 10.6. Synchronization Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-functions "Permalink to this headline")

void __syncthreads();

waits until all threads in the thread block have reached this point and all global and shared memory accesses made by these threads prior to `__syncthreads()` are visible to all threads in the block.

`__syncthreads()` is used to coordinate communication between the threads of the same block. When some threads within a block access the same addresses in shared or global memory, there are potential read-after-write, write-after-read, or write-after-write hazards for some of these memory accesses. These data hazards can be avoided by synchronizing threads in-between these accesses.

`__syncthreads()` is allowed in conditional code but only if the conditional evaluates identically across the entire thread block, otherwise the code execution is likely to hang or produce unintended side effects.

Devices of compute capability 2.x and higher support three variations of `__syncthreads()` described below.

int __syncthreads_count(int predicate);

is identical to `__syncthreads()` with the additional feature that it evaluates predicate for all threads of the block and returns the number of threads for which predicate evaluates to non-zero.

int __syncthreads_and(int predicate);

is identical to `__syncthreads()` with the additional feature that it evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non-zero for all of them.

int __syncthreads_or(int predicate);

is identical to `__syncthreads()` with the additional feature that it evaluates predicate for all threads of the block and returns non-zero if and only if predicate evaluates to non-zero for any of them.

void __syncwarp(unsigned mask=0xffffffff);

will cause the executing thread to wait until all warp lanes named in mask have executed a `__syncwarp()` (with the same mask) before resuming execution. Each calling thread must have its own bit set in the mask and all non-exited threads named in mask must execute a corresponding `__syncwarp()` with the same mask, or the result is undefined.

Executing `__syncwarp()` guarantees memory ordering among threads participating in the barrier. Thus, threads within a warp that wish to communicate via memory can store to memory, execute `__syncwarp()`, and then safely read values stored by other threads in the warp.

Note

For .target sm_6x or below, all threads in mask must execute the same `__syncwarp()` in convergence, and the union of all values in mask must be equal to the active mask. Otherwise, the behavior is undefined.

## 10.7. Mathematical Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mathematical-functions "Permalink to this headline")

The reference manual lists all C/C++ standard library mathematical functions that are supported in device code and all intrinsic functions that are only supported in device code.

[Mathematical Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#mathematical-functions-appendix) provides accuracy information for some of these functions when relevant.

## 10.8. Texture Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-functions "Permalink to this headline")

Texture objects are described in [Texture Object API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api).

Texture fetching is described in [Texture Fetching](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-fetching).

### 10.8.1. Texture Object API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-object-api-appendix "Permalink to this headline")

#### 10.8.1.1. tex1Dfetch()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dfetch "Permalink to this headline")

template<class T>
T tex1Dfetch(cudaTextureObject_t texObj, int x);

fetches from the region of linear memory specified by the one-dimensional texture object `texObj` using integer texture coordinate `x`. `tex1Dfetch()` only works with non-normalized coordinates, so only the border and clamp addressing modes are supported. It does not perform any texture filtering. For integer types, it may optionally promote the integer to single-precision floating point.

#### 10.8.1.2. tex1D()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1d "Permalink to this headline")

template<class T>
T tex1D(cudaTextureObject_t texObj, float x);

fetches from the CUDA array specified by the one-dimensional texture object `texObj` using texture coordinate `x`.

#### 10.8.1.3. tex1DLod()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlod "Permalink to this headline")

template<class T>
T tex1DLod(cudaTextureObject_t texObj, float x, float level);

fetches from the CUDA array specified by the one-dimensional texture object `texObj` using texture coordinate `x` at the level-of-detail `level`.

#### 10.8.1.4. tex1DGrad()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dgrad "Permalink to this headline")

template<class T>
T tex1DGrad(cudaTextureObject_t texObj, float x, float dx, float dy);

fetches from the CUDA array specified by the one-dimensional texture object `texObj` using texture coordinate `x`. The level-of-detail is derived from the X-gradient `dx` and Y-gradient `dy`.

#### 10.8.1.5. tex2D()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2d "Permalink to this headline")

template<class T>
T tex2D(cudaTextureObject_t texObj, float x, float y);

fetches from the CUDA array or the region of linear memory specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)`.

#### 10.8.1.6. tex2D() for sparse CUDA arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2d-for-sparse-cuda-arrays "Permalink to this headline")

                template<class T>
T tex2D(cudaTextureObject_t texObj, float x, float y, bool* isResident);

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.7. tex2Dgather()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgather "Permalink to this headline")

template<class T>
T tex2Dgather(cudaTextureObject_t texObj,
              float x, float y, int comp = 0);

fetches from the CUDA array specified by the 2D texture object `texObj` using texture coordinates `x` and `y` and the `comp` parameter as described in [Texture Gather](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-gather).

#### 10.8.1.8. tex2Dgather() for sparse CUDA arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgather-for-sparse-cuda-arrays "Permalink to this headline")

                template<class T>
T tex2Dgather(cudaTextureObject_t texObj,
            float x, float y, bool* isResident, int comp = 0);

fetches from the CUDA array specified by the 2D texture object `texObj` using texture coordinates `x` and `y` and the `comp` parameter as described in [Texture Gather](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-gather). Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.9. tex2DGrad()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgrad "Permalink to this headline")

template<class T>
T tex2DGrad(cudaTextureObject_t texObj, float x, float y,
            float2 dx, float2 dy);

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)`. The level-of-detail is derived from the `dx` and `dy` gradients.

#### 10.8.1.10. tex2DGrad() for sparse CUDA arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dgrad-for-sparse-cuda-arrays "Permalink to this headline")

                template<class T>
T tex2DGrad(cudaTextureObject_t texObj, float x, float y,
        float2 dx, float2 dy, bool* isResident);

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)`. The level-of-detail is derived from the `dx` and `dy` gradients. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.11. tex2DLod()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlod "Permalink to this headline")

template<class T>
tex2DLod(cudaTextureObject_t texObj, float x, float y, float level);

fetches from the CUDA array or the region of linear memory specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)` at level-of-detail `level`.

#### 10.8.1.12. tex2DLod() for sparse CUDA arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlod-for-sparse-cuda-arrays "Permalink to this headline")

        template<class T>
tex2DLod(cudaTextureObject_t texObj, float x, float y, float level, bool* isResident);

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)` at level-of-detail `level`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.13. tex3D()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3d "Permalink to this headline")

template<class T>
T tex3D(cudaTextureObject_t texObj, float x, float y, float z);

fetches from the CUDA array specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)`.

#### 10.8.1.14. tex3D() for sparse CUDA arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3d-for-sparse-cuda-arrays "Permalink to this headline")

                template<class T>
T tex3D(cudaTextureObject_t texObj, float x, float y, float z, bool* isResident);

fetches from the CUDA array specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.15. tex3DLod()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3dlod "Permalink to this headline")

template<class T>
T tex3DLod(cudaTextureObject_t texObj, float x, float y, float z, float level);

fetches from the CUDA array or the region of linear memory specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)` at level-of-detail `level`.

#### 10.8.1.16. tex3DLod() for sparse CUDA arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3dlod-for-sparse-cuda-arrays "Permalink to this headline")

                template<class T>
T tex3DLod(cudaTextureObject_t texObj, float x, float y, float z, float level, bool* isResident);

fetches from the CUDA array or the region of linear memory specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)` at level-of-detail `level`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.17. tex3DGrad()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3dgrad "Permalink to this headline")

template<class T>
T tex3DGrad(cudaTextureObject_t texObj, float x, float y, float z,
            float4 dx, float4 dy);

fetches from the CUDA array specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)` at a level-of-detail derived from the X and Y gradients `dx` and `dy`.

#### 10.8.1.18. tex3DGrad() for sparse CUDA arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex3dgrad-for-sparse-cuda-arrays "Permalink to this headline")

                template<class T>
T tex3DGrad(cudaTextureObject_t texObj, float x, float y, float z,
        float4 dx, float4 dy, bool* isResident);

fetches from the CUDA array specified by the three-dimensional texture object `texObj` using texture coordinate `(x,y,z)` at a level-of-detail derived from the X and Y gradients `dx` and `dy`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.19. tex1DLayered()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlayered "Permalink to this headline")

template<class T>
T tex1DLayered(cudaTextureObject_t texObj, float x, int layer);

fetches from the CUDA array specified by the one-dimensional texture object `texObj` using texture coordinate `x` and index `layer`, as described in [Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures).

#### 10.8.1.20. tex1DLayeredLod()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlayeredlod "Permalink to this headline")

template<class T>
T tex1DLayeredLod(cudaTextureObject_t texObj, float x, int layer, float level);

fetches from the CUDA array specified by the one-dimensional [Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures) at layer `layer` using texture coordinate `x` and level-of-detail `level`.

#### 10.8.1.21. tex1DLayeredGrad()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex1dlayeredgrad "Permalink to this headline")

template<class T>
T tex1DLayeredGrad(cudaTextureObject_t texObj, float x, int layer,
                   float dx, float dy);

fetches from the CUDA array specified by the one-dimensional [layered texture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures) at layer `layer` using texture coordinate `x` and a level-of-detail derived from the `dx` and `dy` gradients.

#### 10.8.1.22. tex2DLayered()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayered "Permalink to this headline")

template<class T>
T tex2DLayered(cudaTextureObject_t texObj,
               float x, float y, int layer);

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)` and index `layer`, as described in [Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures).

#### 10.8.1.23. tex2DLayered() for Sparse CUDA Arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayered-for-sparse-cuda-arrays "Permalink to this headline")

                template<class T>
T tex2DLayered(cudaTextureObject_t texObj,
            float x, float y, int layer, bool* isResident);

fetches from the CUDA array specified by the two-dimensional texture object `texObj` using texture coordinate `(x,y)` and index `layer`, as described in [Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures). Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.24. tex2DLayeredLod()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayeredlod "Permalink to this headline")

template<class T>
T tex2DLayeredLod(cudaTextureObject_t texObj, float x, float y, int layer,
                  float level);

fetches from the CUDA array specified by the two-dimensional [layered texture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures) at layer `layer` using texture coordinate `(x,y)`.

#### 10.8.1.25. tex2DLayeredLod() for sparse CUDA arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayeredlod-for-sparse-cuda-arrays "Permalink to this headline")

                template<class T>
T tex2DLayeredLod(cudaTextureObject_t texObj, float x, float y, int layer,
                float level, bool* isResident);

fetches from the CUDA array specified by the two-dimensional [layered texture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures) at layer `layer` using texture coordinate `(x,y)`. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.26. tex2DLayeredGrad()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayeredgrad "Permalink to this headline")

template<class T>
T tex2DLayeredGrad(cudaTextureObject_t texObj, float x, float y, int layer,
                   float2 dx, float2 dy);

fetches from the CUDA array specified by the two-dimensional [layered texture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures) at layer `layer` using texture coordinate `(x,y)` and a level-of-detail derived from the `dx` and `dy` gradients.

#### 10.8.1.27. tex2DLayeredGrad() for sparse CUDA arrays[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tex2dlayeredgrad-for-sparse-cuda-arrays "Permalink to this headline")

                template<class T>
T tex2DLayeredGrad(cudaTextureObject_t texObj, float x, float y, int layer,
                float2 dx, float2 dy, bool* isResident);

fetches from the CUDA array specified by the two-dimensional [layered texture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#layered-textures) at layer `layer` using texture coordinate `(x,y)` and a level-of-detail derived from the `dx` and `dy` gradients. Also returns whether the texel is resident in memory via `isResident` pointer. If not, the values fetched will be zeros.

#### 10.8.1.28. texCubemap()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemap "Permalink to this headline")

template<class T>
T texCubemap(cudaTextureObject_t texObj, float x, float y, float z);

fetches the CUDA array specified by the cubemap texture object `texObj` using texture coordinate `(x,y,z)`, as described in [Cubemap Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures).

#### 10.8.1.29. texCubemapGrad()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemapgrad "Permalink to this headline")

template<class T>
T texCubemapGrad(cudaTextureObject_t texObj, float x, float, y, float z,
                float4 dx, float4 dy);

fetches from the CUDA array specified by the cubemap texture object `texObj` using texture coordinate `(x,y,z)` as described in [Cubemap Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures). The level-of-detail used is derived from the `dx` and `dy` gradients.

#### 10.8.1.30. texCubemapLod()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplod "Permalink to this headline")

template<class T>
T texCubemapLod(cudaTextureObject_t texObj, float x, float, y, float z,
                float level);

fetches from the CUDA array specified by the cubemap texture object `texObj` using texture coordinate `(x,y,z)` as described in [Cubemap Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-textures). The level-of-detail used is given by `level`.

#### 10.8.1.31. texCubemapLayered()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplayered "Permalink to this headline")

template<class T>
T texCubemapLayered(cudaTextureObject_t texObj,
                    float x, float y, float z, int layer);

fetches from the CUDA array specified by the cubemap layered texture object `texObj` using texture coordinates `(x,y,z)`, and index `layer`, as described in [Cubemap Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures).

#### 10.8.1.32. texCubemapLayeredGrad()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplayeredgrad "Permalink to this headline")

template<class T>
T texCubemapLayeredGrad(cudaTextureObject_t texObj, float x, float y, float z,
                       int layer, float4 dx, float4 dy);

fetches from the CUDA array specified by the cubemap layered texture object `texObj` using texture coordinate `(x,y,z)` and index `layer`, as described in [Cubemap Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures), at level-of-detail derived from the `dx` and `dy` gradients.

#### 10.8.1.33. texCubemapLayeredLod()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texcubemaplayeredlod "Permalink to this headline")

template<class T>
T texCubemapLayeredLod(cudaTextureObject_t texObj, float x, float y, float z,
                       int layer, float level);

fetches from the CUDA array specified by the cubemap layered texture object `texObj` using texture coordinate `(x,y,z)` and index `layer`, as described in [Cubemap Layered Textures](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cubemap-layered-textures), at level-of-detail level `level`.

## 10.9. Surface Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-functions "Permalink to this headline")

Surface functions are only supported by devices of compute capability 2.0 and higher.

Surface objects are described in described in [Surface Object API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-object-api-appendix).

In the sections below, `boundaryMode` specifies the boundary mode, that is how out-of-range surface coordinates are handled; it is equal to either `cudaBoundaryModeClamp`, in which case out-of-range coordinates are clamped to the valid range, or `cudaBoundaryModeZero`, in which case out-of-range reads return zero and out-of-range writes are ignored, or `cudaBoundaryModeTrap`, in which case out-of-range accesses cause the kernel execution to fail.

### 10.9.1. Surface Object API[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surface-object-api-appendix "Permalink to this headline")

#### 10.9.1.1. surf1Dread()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf1dread "Permalink to this headline")

template<class T>
T surf1Dread(cudaSurfaceObject_t surfObj, int x,
               boundaryMode = cudaBoundaryModeTrap);

reads the CUDA array specified by the one-dimensional surface object `surfObj` using byte coordinate x.

#### 10.9.1.2. surf1Dwrite[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf1dwrite "Permalink to this headline")

template<class T>
void surf1Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x,
                  boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the one-dimensional surface object `surfObj` at byte coordinate x.

#### 10.9.1.3. surf2Dread()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf2dread "Permalink to this headline")

template<class T>
T surf2Dread(cudaSurfaceObject_t surfObj,
              int x, int y,
              boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf2Dread(T* data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y,
                 boundaryMode = cudaBoundaryModeTrap);

reads the CUDA array specified by the two-dimensional surface object `surfObj` using byte coordinates x and y.

#### 10.9.1.4. surf2Dwrite()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf2dwrite "Permalink to this headline")

template<class T>
void surf2Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y,
                  boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the two-dimensional surface object `surfObj` at byte coordinate x and y.

#### 10.9.1.5. surf3Dread()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf3dread "Permalink to this headline")

template<class T>
T surf3Dread(cudaSurfaceObject_t surfObj,
              int x, int y, int z,
              boundaryMode = cudaBoundaryModeTrap);
template<class T>
void surf3Dread(T* data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int z,
                 boundaryMode = cudaBoundaryModeTrap);

reads the CUDA array specified by the three-dimensional surface object `surfObj` using byte coordinates x, y, and z.

#### 10.9.1.6. surf3Dwrite()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf3dwrite "Permalink to this headline")

template<class T>
void surf3Dwrite(T data,
                  cudaSurfaceObject_t surfObj,
                  int x, int y, int z,
                  boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the three-dimensional object `surfObj` at byte coordinate x, y, and z.

#### 10.9.1.7. surf1DLayeredread()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf1dlayeredread "Permalink to this headline")

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

reads the CUDA array specified by the one-dimensional layered surface object `surfObj` using byte coordinate x and index `layer`.

#### 10.9.1.8. surf1DLayeredwrite()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf1dlayeredwrite "Permalink to this headline")

template<class Type>
void surf1DLayeredwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int layer,
                 boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the two-dimensional layered surface object `surfObj` at byte coordinate x and index `layer`.

#### 10.9.1.9. surf2DLayeredread()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf2dlayeredread "Permalink to this headline")

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

reads the CUDA array specified by the two-dimensional layered surface object `surfObj` using byte coordinate x and y, and index `layer`.

#### 10.9.1.10. surf2DLayeredwrite()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surf2dlayeredwrite "Permalink to this headline")

template<class T>
void surf2DLayeredwrite(T data,
                          cudaSurfaceObject_t surfObj,
                          int x, int y, int layer,
                          boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the one-dimensional layered surface object `surfObj` at byte coordinate x and y, and index `layer`.

#### 10.9.1.11. surfCubemapread()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemapread "Permalink to this headline")

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

reads the CUDA array specified by the cubemap surface object `surfObj` using byte coordinate x and y, and face index face.

#### 10.9.1.12. surfCubemapwrite()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemapwrite "Permalink to this headline")

template<class T>
void surfCubemapwrite(T data,
                 cudaSurfaceObject_t surfObj,
                 int x, int y, int face,
                 boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the cubemap object `surfObj` at byte coordinate x and y, and face index face.

#### 10.9.1.13. surfCubemapLayeredread()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemaplayeredread "Permalink to this headline")

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

reads the CUDA array specified by the cubemap layered surface object `surfObj` using byte coordinate x and y, and index `layerFace.`

#### 10.9.1.14. surfCubemapLayeredwrite()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#surfcubemaplayeredwrite "Permalink to this headline")

template<class T>
void surfCubemapLayeredwrite(T data,
             cudaSurfaceObject_t surfObj,
             int x, int y, int layerFace,
             boundaryMode = cudaBoundaryModeTrap);

writes value data to the CUDA array specified by the cubemap layered object `surfObj` at byte coordinate `x` and `y`, and index `layerFace`.

## 10.10. Read-Only Data Cache Load Function[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#read-only-data-cache-load-function "Permalink to this headline")

The read-only data cache load function is only supported by devices of compute capability 5.0 and higher.

T __ldg(const T* address);

returns the data of type `T` located at address `address`, where `T` is `char`, `signed char`, `short`, `int`, `long`, `long long``unsigned char`, `unsigned short`, `unsigned int`, `unsigned long`, `unsigned long long`, `char2`, `char4`, `short2`, `short4`, `int2`, `int4`, `longlong2``uchar2`, `uchar4`, `ushort2`, `ushort4`, `uint2`, `uint4`, `ulonglong2``float`, `float2`, `float4`, `double`, or `double2`. With the `cuda_fp16.h` header included, `T` can be `__half` or `__half2`. Similarly, with the `cuda_bf16.h` header included, `T` can also be `__nv_bfloat16` or `__nv_bfloat162`. The operation is cached in the read-only data cache (see [Global Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-memory-5-x)).

## 10.11. Load Functions Using Cache Hints[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#load-functions-using-cache-hints "Permalink to this headline")

These load functions are only supported by devices of compute capability 5.0 and higher.

T __ldcg(const T* address);
T __ldca(const T* address);
T __ldcs(const T* address);
T __ldlu(const T* address);
T __ldcv(const T* address);

returns the data of type `T` located at address `address`, where `T` is `char`, `signed char`, `short`, `int`, `long`, `long long``unsigned char`, `unsigned short`, `unsigned int`, `unsigned long`, `unsigned long long`, `char2`, `char4`, `short2`, `short4`, `int2`, `int4`, `longlong2``uchar2`, `uchar4`, `ushort2`, `ushort4`, `uint2`, `uint4`, `ulonglong2``float`, `float2`, `float4`, `double`, or `double2`. With the `cuda_fp16.h` header included, `T` can be `__half` or `__half2`. Similarly, with the `cuda_bf16.h` header included, `T` can also be `__nv_bfloat16` or `__nv_bfloat162`. The operation is using the corresponding cache operator (see [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators))

## 10.12. Store Functions Using Cache Hints[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#store-functions-using-cache-hints "Permalink to this headline")

These store functions are only supported by devices of compute capability 5.0 and higher.

void __stwb(T* address, T value);
void __stcg(T* address, T value);
void __stcs(T* address, T value);
void __stwt(T* address, T value);

stores the `value` argument of type `T` to the location at address `address`, where `T` is `char`, `signed char`, `short`, `int`, `long`, `long long``unsigned char`, `unsigned short`, `unsigned int`, `unsigned long`, `unsigned long long`, `char2`, `char4`, `short2`, `short4`, `int2`, `int4`, `longlong2``uchar2`, `uchar4`, `ushort2`, `ushort4`, `uint2`, `uint4`, `ulonglong2``float`, `float2`, `float4`, `double`, or `double2`. With the `cuda_fp16.h` header included, `T` can be `__half` or `__half2`. Similarly, with the `cuda_bf16.h` header included, `T` can also be `__nv_bfloat16` or `__nv_bfloat162`. The operation is using the corresponding cache operator (see [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) )

## 10.13. Time Function[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#time-function "Permalink to this headline")

clock_t clock();
long long int clock64();

when executed in device code, returns the value of a per-multiprocessor counter that is incremented every clock cycle. Sampling this counter at the beginning and at the end of a kernel, taking the difference of the two samples, and recording the result per thread provides a measure for each thread of the number of clock cycles taken by the device to completely execute the thread, but not of the number of clock cycles the device actually spent executing thread instructions. The former number is greater than the latter since threads are time sliced.

## 10.14. Atomic Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions "Permalink to this headline")

An atomic function performs a read-modify-write atomic operation on one 32-bit, 64-bit, or 128-bit word residing in global or shared memory. In the case of `float2` or `float4`, the read-modify-write operation is performed on each element of the vector residing in global memory. For example, `atomicAdd()` reads a word at some address in global or shared memory, adds a number to it, and writes the result back to the same address. Atomic functions can only be used in device functions.

The atomic functions described in this section have ordering [cuda::memory_order_relaxed](https://en.cppreference.com/w/cpp/atomic/memory_order) and are only atomic at a particular [scope](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#thread-scopes):

- Atomic APIs with `_system` suffix (example: `atomicAdd_system`) are atomic at scope `cuda::thread_scope_system` if they meet particular [conditions](https://nvidia.github.io/libcudacxx/extended_api/memory_model.html#atomicity).
    
- Atomic APIs without a suffix (example: `atomicAdd`) are atomic at scope `cuda::thread_scope_device`.
    
- Atomic APIs with `_block` suffix (example: `atomicAdd_block`) are atomic at scope `cuda::thread_scope_block`.
    

In the following example both the CPU and the GPU atomically update an integer value at address `addr`:

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

Note that any atomic operation can be implemented based on `atomicCAS()` (Compare And Swap). For example, `atomicAdd()` for double-precision floating-point numbers is not available on devices with compute capability lower than 6.0 but it can be implemented as follows:

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

There are system-wide and block-wide variants of the following device-wide atomic APIs, with the following exceptions:

- Devices with compute capability less than 6.0 only support device-wide atomic operations,
    
- Tegra devices with compute capability less than 7.2 do not support system-wide atomic operations.
    

CUDA 12.8 and later support CUDA compiler builtin functions for atomic operations with memory order and thread scope. We follows the [GNU’s atomic built-in function signature](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) with an extra argument of thread scope. We use the following atomic operation memory orders and thread scopes:

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

Example:

__device__ T __nv_atomic_load_n(T* ptr, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

T can be any integral type that is size of 1, 2, 4, 8 and 16 bytes.

These atomic functions cannot operate on local memory. For example:

__device__ void foo() {
   int a = 1; // defined in local memory
   int b;
   __nv_atomic_load(&a, &b, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
}

These functions must only be used within the block scope of a `__device__` function. For example:

__device__ void foo() {
   __shared__ unsigned int u1 = 1;
   __shared__ unsigned int u2 = 2;
   __nv_atomic_load(&u1, &u2, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
}

And these functions’ address cannot be taken. Here are three unsupported examples:

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

The memory order corresponds to [C++ standard atomic operation’s memory order](https://en.cppreference.com/w/cpp/atomic/memory_order). And for thread scope, we follows cuda::thread_scope’s [definition](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes).

`__NV_ATOMIC_CONSUME` memory order is currently implemented using stronger `__NV_ATOMIC_ACQUIRE` memory order.

`__NV_THREAD_SCOPE_THREAD` thread scope is currently implemented using wider `__NV_THREAD_SCOPE_BLOCK` thread scope.

For the supported data types, please refer to the corresponding section of different atomic operations.

### 10.14.1. Arithmetic Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-functions "Permalink to this headline")

#### 10.14.1.1. atomicAdd()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicadd "Permalink to this headline")

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

reads the 16-bit, 32-bit or 64-bit `old` located at the address `address` in global or shared memory, computes `(old + val)`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

The 32-bit floating-point version of `atomicAdd()` is only supported by devices of compute capability 2.x and higher.

The 64-bit floating-point version of `atomicAdd()` is only supported by devices of compute capability 6.x and higher.

The 32-bit `__half2` floating-point version of `atomicAdd()` is only supported by devices of compute capability 6.x and higher. The atomicity of the `__half2` or `__nv_bfloat162` add operation is guaranteed separately for each of the two `__half` or `__nv_bfloat16` elements; the entire `__half2` or `__nv_bfloat162` is not guaranteed to be atomic as a single 32-bit access.

The `float2` and `float4` floating-point vector versions of `atomicAdd()` are only supported by devices of compute capability 9.x and higher. The atomicity of the `float2` or `float4` add operation is guaranteed separately for each of the two or four `float` elements; the entire `float2` or `float4` is not guaranteed to be atomic as a single 64-bit or 128-bit access.

The 16-bit `__half` floating-point version of `atomicAdd()` is only supported by devices of compute capability 7.x and higher.

The 16-bit `__nv_bfloat16` floating-point version of `atomicAdd()` is only supported by devices of compute capability 8.x and higher.

The `float2` and `float4` floating-point vector versions of `atomicAdd()` are only supported by devices of compute capability 9.x and higher.

The `float2` and `float4` floating-point vector versions of `atomicAdd()` are only supported for global memory addresses.

#### 10.14.1.2. atomicSub()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicsub "Permalink to this headline")

int atomicSub(int* address, int val);
unsigned int atomicSub(unsigned int* address,
                       unsigned int val);

reads the 32-bit word `old` located at the address `address` in global or shared memory, computes `(old - val)`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

#### 10.14.1.3. atomicExch()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicexch "Permalink to this headline")

int atomicExch(int* address, int val);
unsigned int atomicExch(unsigned int* address,
                        unsigned int val);
unsigned long long int atomicExch(unsigned long long int* address,
                                  unsigned long long int val);
float atomicExch(float* address, float val);

reads the 32-bit or 64-bit word `old` located at the address `address` in global or shared memory and stores `val` back to memory at the same address. These two operations are performed in one atomic transaction. The function returns `old`.

template<typename T> T atomicExch(T* address, T val);

reads the 128-bit word `old` located at the address `address` in global or shared memory and stores `val` back to memory at the same address. These two operations are performed in one atomic transaction. The function returns `old`. The type `T` must meet the following requirements:

sizeof(T) == 16
alignof(T) >= 16
std::is_trivially_copyable<T>::value == true
// for C++03 and older
std::is_default_constructible<T>::value == true

So, `T` must be 128-bit and properly aligned, be trivially copyable, and on C++03 or older, it must also be default constructible.

The 128-bit `atomicExch()` is only supported by devices of compute capability 9.x and higher.

#### 10.14.1.4. atomicMin()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicmin "Permalink to this headline")

int atomicMin(int* address, int val);
unsigned int atomicMin(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMin(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMin(long long int* address,
                                long long int val);

reads the 32-bit or 64-bit word `old` located at the address `address` in global or shared memory, computes the minimum of `old` and `val`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

The 64-bit version of `atomicMin()` is only supported by devices of compute capability 5.0 and higher.

#### 10.14.1.5. atomicMax()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicmax "Permalink to this headline")

int atomicMax(int* address, int val);
unsigned int atomicMax(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicMax(unsigned long long int* address,
                                 unsigned long long int val);
long long int atomicMax(long long int* address,
                                 long long int val);

reads the 32-bit or 64-bit word `old` located at the address `address` in global or shared memory, computes the maximum of `old` and `val`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

The 64-bit version of `atomicMax()` is only supported by devices of compute capability 5.0 and higher.

#### 10.14.1.6. atomicInc()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicinc "Permalink to this headline")

unsigned int atomicInc(unsigned int* address,
                       unsigned int val);

reads the 32-bit word `old` located at the address `address` in global or shared memory, computes `((old >= val) ? 0 : (old+1))`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

#### 10.14.1.7. atomicDec()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicdec "Permalink to this headline")

unsigned int atomicDec(unsigned int* address,
                       unsigned int val);

reads the 32-bit word `old` located at the address `address` in global or shared memory, computes `(((old == 0) || (old > val)) ? val : (old-1)` ), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

#### 10.14.1.8. atomicCAS()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomiccas "Permalink to this headline")

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

reads the 16-bit, 32-bit or 64-bit word `old` located at the address `address` in global or shared memory, computes `(old == compare ? val : old)`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old` (Compare And Swap).

template<typename T> T atomicCAS(T* address, T compare, T val);

reads the 128-bit word `old` located at the address `address` in global or shared memory, computes `(old == compare ? val : old)`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old` (Compare And Swap). The type `T` must meet the following requirements:

sizeof(T) == 16
alignof(T) >= 16
std::is_trivially_copyable<T>::value == true
// for C++03 and older
std::is_default_constructible<T>::value == true

So, `T` must be 128-bit and properly aligned, be trivially copyable, and on C++03 or older, it must also be default constructible.

The 128-bit `atomicCAS()` is only supported by devices of compute capability 9.x and higher.

#### 10.14.1.9. __nv_atomic_exchange()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-exchange "Permalink to this headline")

__device__ void __nv_atomic_exchange(T* ptr, T* val, T *ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

This atomic function is introduced in CUDA 12.8. It reads the value where `ptr` points to and stores the value to where `ret` points to. And it reads the value where `val` points to and stores the value to where `ptr` points to.

This is a generic atomic exchange, which means that `T` can be any data type that is size of 4, 8 or 16 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

16-byte data type is supported on the architecture `sm_90` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

#### 10.14.1.10. __nv_atomic_exchange_n()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-exchange-n "Permalink to this headline")

__device__ T __nv_atomic_exchange_n(T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

This atomic function is introduced in CUDA 12.8. It reads the value where `ptr` points to and use this value as the return value. And it stores `val` to where `ptr` points to.

This is a non-generic atomic exchange, which means that `T` can only be an integral type that is size of 4, 8 or 16 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

16-byte data type is supported on the architecture `sm_90` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

#### 10.14.1.11. __nv_atomic_compare_exchange()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-compare-exchange "Permalink to this headline")

__device__ bool __nv_atomic_compare_exchange (T* ptr, T* expected, T* desired, bool weak, int success_order, int failure_order, int scope = __NV_THREAD_SCOPE_SYSTEM);

This atomic function is introduced in CUDA 12.8. It reads the value where `ptr` points to and compare it with the value where `expected` points to. If they are equal, the return value is `true` and the value where `desired` points to is stored to where `ptr` points to. Otherwise, it returns `false` and the value where `ptr` points to is stored to where `expected` points to. The parameter `weak` is ignored and it picks the stronger memory order between `success_order` and `failure_order` to execute the compare-and-exchange operation.

This is a generic atomic compare-and-exchange, which means that `T` can be any data type that is size of 2, 4, 8 or 16 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

16-byte data type is supported on the architecture `sm_90` and higher.

2-byte data type is supported on the architecture `sm_70` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

#### 10.14.1.12. __nv_atomic_compare_exchange_n()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-compare-exchange-n "Permalink to this headline")

__device__ bool __nv_atomic_compare_exchange_n (T* ptr, T* expected, T desired, bool weak, int success_order, int failure_order, int scope = __NV_THREAD_SCOPE_SYSTEM);

This atomic function is introduced in CUDA 12.8. It reads the value where `ptr` points to and compare it with the value where `expected` points to. If they are equal, the return value is `true` and `desired` is stored to where `ptr` points to. Otherwise, it returns `false` and the value where `ptr` points to is stored to where `expected` points to. The parameter `weak` is ignored and it picks the stronger memory order between `success_order` and `failure_order` to execute the compare-and-exchange operation.

This is a non-generic atomic compare-and-exchange, which means that `T` can only be an integral type that is size of 2, 4, 8 or 16 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

16-byte data type is supported on the architecture `sm_90` and higher.

2-byte data type is supported on the architecture `sm_70` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

#### 10.14.1.13. __nv_atomic_fetch_add() and __nv_atomic_add()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-add-and-nv-atomic-add "Permalink to this headline")

__device__ T __nv_atomic_fetch_add (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_add (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

These two atomic functions are introduced in CUDA 12.8. It reads the value where `ptr` points to, adds with `val`, and stores the result back to where `ptr` points to. `__nv_atomic_fetch_add` returns the old value where `ptr` points to. `__nv_atomic_add` does not have return value.

`T` can only be `unsigned int`, `int`, `unsigned long long`, `float` or `double`.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

#### 10.14.1.14. __nv_atomic_fetch_sub() and __nv_atomic_sub()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-sub-and-nv-atomic-sub "Permalink to this headline")

__device__ T __nv_atomic_fetch_sub (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_sub (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

These two atomic functions are introduced in CUDA 12.8. It reads the value where `ptr` points to, subtracts with `val`, and stores the result back to where `ptr` points to. `__nv_atomic_fetch_sub` returns the old value where `ptr` points to. `__nv_atomic_sub` does not have return value.

`T` can only be `unsigned int`, `int`, `unsigned long long`, `float` or `double`.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

#### 10.14.1.15. __nv_atomic_fetch_min() and __nv_atomic_min()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-min-and-nv-atomic-min "Permalink to this headline")

__device__ T __nv_atomic_fetch_min (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_min (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

These two atomic functions are introduced in CUDA 12.8. It reads the value where `ptr` points to, compares with `val`, and stores the smaller value back to where `ptr` points to. `__nv_atomic_fetch_min` returns the old value where `ptr` points to. `__nv_atomic_min` does not have return value.

`T` can only be `unsigned int`, `int`, `unsigned long long` or `long long`.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

#### 10.14.1.16. __nv_atomic_fetch_max() and __nv_atomic_max()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-max-and-nv-atomic-max "Permalink to this headline")

__device__ T __nv_atomic_fetch_max (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_max (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

These two atomic functions are introduced in CUDA 12.8. It reads the value where `ptr` points to, compares with `val`, and stores the bigger value back to where `ptr` points to. `__nv_atomic_fetch_max` returns the old value where `ptr` points to. `__nv_atomic_max` does not have return value.

`T` can only be `unsigned int`, `int`, `unsigned long long` or `long long`.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

### 10.14.2. Bitwise Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#bitwise-functions "Permalink to this headline")

#### 10.14.2.1. atomicAnd()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicand "Permalink to this headline")

int atomicAnd(int* address, int val);
unsigned int atomicAnd(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicAnd(unsigned long long int* address,
                                 unsigned long long int val);

reads the 32-bit or 64-bit word `old` located at the address `address` in global or shared memory, computes `(old & val`), and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

The 64-bit version of `atomicAnd()` is only supported by devices of compute capability 5.0 and higher.

#### 10.14.2.2. atomicOr()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicor "Permalink to this headline")

int atomicOr(int* address, int val);
unsigned int atomicOr(unsigned int* address,
                      unsigned int val);
unsigned long long int atomicOr(unsigned long long int* address,
                                unsigned long long int val);

reads the 32-bit or 64-bit word `old` located at the address `address` in global or shared memory, computes `(old | val)`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

The 64-bit version of `atomicOr()` is only supported by devices of compute capability 5.0 and higher.

#### 10.14.2.3. atomicXor()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomicxor "Permalink to this headline")

int atomicXor(int* address, int val);
unsigned int atomicXor(unsigned int* address,
                       unsigned int val);
unsigned long long int atomicXor(unsigned long long int* address,
                                 unsigned long long int val);

reads the 32-bit or 64-bit word `old` located at the address `address` in global or shared memory, computes `(old ^ val)`, and stores the result back to memory at the same address. These three operations are performed in one atomic transaction. The function returns `old`.

The 64-bit version of `atomicXor()` is only supported by devices of compute capability 5.0 and higher.

#### 10.14.2.4. __nv_atomic_fetch_or() and __nv_atomic_or()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-or-and-nv-atomic-or "Permalink to this headline")

__device__ T __nv_atomic_fetch_or (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_or (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

These two atomic functions are introduced in CUDA 12.8. It reads the value where `ptr` points to, `or` with `val`, and stores the result back to where `ptr` points to. `__nv_atomic_fetch_or` returns the old value where `ptr` points to. `__nv_atomic_or` does not have return value.

`T` can only be an integral type that is size of 4 or 8 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

#### 10.14.2.5. __nv_atomic_fetch_xor() and __nv_atomic_xor()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-xor-and-nv-atomic-xor "Permalink to this headline")

__device__ T __nv_atomic_fetch_xor (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_xor (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

These two atomic functions are introduced in CUDA 12.8. It reads the value where `ptr` points to, `xor` with `val`, and stores the result back to where `ptr` points to. `__nv_atomic_fetch_xor` returns the old value where `ptr` points to. `__nv_atomic_xor` does not have return value.

`T` can only be an integral type that is size of 4 or 8 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

#### 10.14.2.6. __nv_atomic_fetch_and() and __nv_atomic_and()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-fetch-and-and-nv-atomic-and "Permalink to this headline")

__device__ T __nv_atomic_fetch_and (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
__device__ void __nv_atomic_and (T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

These two atomic functions are introduced in CUDA 12.8. It reads the value where `ptr` points to, `and` with `val`, and stores the result back to where `ptr` points to. `__nv_atomic_fetch_and` returns the old value where `ptr` points to. `__nv_atomic_and` does not have return value.

`T` can only be an integral type that is size of 4 or 8 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

### 10.14.3. Other atomic functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#other-atomic-functions "Permalink to this headline")

#### 10.14.3.1. __nv_atomic_load()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-load "Permalink to this headline")

__device__ void __nv_atomic_load(T* ptr, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

This atomic function is introduced in CUDA 12.8. It loads the value where `ptr` points to and writes the value to where `ret` points to.

This is a generic atomic load, which means that `T` can be any data type that is size of 1, 2, 4, 8 or 16 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

16-byte data type is supported on the architecture `sm_70` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables. `order` cannot be `__NV_ATOMIC_RELEASE` or `__NV_ATOMIC_ACQ_REL`.

#### 10.14.3.2. __nv_atomic_load_n()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-load-n "Permalink to this headline")

__device__ T __nv_atomic_load_n(T* ptr, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

This atomic function is introduced in CUDA 12.8. It loads the value where `ptr` points to and returns this value.

This is a non-generic atomic load, which means that `T` can only be an integral type that is size of 1, 2, 4, 8 or 16 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

16-byte data type is supported on the architecture `sm_70` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables. `order` cannot be `__NV_ATOMIC_RELEASE` or `__NV_ATOMIC_ACQ_REL`.

#### 10.14.3.3. __nv_atomic_store()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-store "Permalink to this headline")

__device__ void __nv_atomic_store(T* ptr, T* val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

This atomic function is introduced in CUDA 12.8. It reads the value where `val` points to and stores to where `ptr` points to.

This is a generic atomic load, which means that `T` can be any data type that is size of 1, 2, 4, 8 or 16 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

16-byte data type is supported on the architecture `sm_70` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables. `order` cannot be `__NV_ATOMIC_CONSUME`, `__NV_ATOMIC_ACQUIRE` or `__NV_ATOMIC_ACQ_REL`.

#### 10.14.3.4. __nv_atomic_store_n()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-store-n "Permalink to this headline")

__device__ void __nv_atomic_store_n(T* ptr, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

This atomic function is introduced in CUDA 12.8. It stores `val` to where `ptr` points to.

This is a non-generic atomic load, which means that `T` can only be an integral type that is size of 1, 2, 4, 8 or 16 bytes.

The atomic operation with memory order and thread scope is supported on the architecture `sm_60` and higher.

16-byte data type is supported on the architecture `sm_70` and higher.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables. `order` cannot be `__NV_ATOMIC_CONSUME`, `__NV_ATOMIC_ACQUIRE` or `__NV_ATOMIC_ACQ_REL`.

#### 10.14.3.5. __nv_atomic_thread_fence()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#nv-atomic-thread-fence "Permalink to this headline")

__device__ void __nv_atomic_thread_fence (int order, int scope = __NV_THREAD_SCOPE_SYSTEM);

This atomic function establishes an ordering between memory accesses requested by this thread based on the specified memory order. And the thread scope parameter specifies the set of threads that may observe the ordering effect of this operation.

The thread scope of `cluster` is supported on the architecture `sm_90` and higher.

The arguments `order` and `scope` need to be integer literals, i.e., the arguments cannot be variables.

## 10.15. Address Space Predicate Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#address-space-predicate-functions "Permalink to this headline")

The functions described in this section have unspecified behavior if the argument is a null pointer.

### 10.15.1. __isGlobal()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isglobal "Permalink to this headline")

__device__ unsigned int __isGlobal(const void *ptr);

Returns 1 if `ptr` contains the generic address of an object in global memory space, otherwise returns 0.

### 10.15.2. __isShared()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isshared "Permalink to this headline")

__device__ unsigned int __isShared(const void *ptr);

Returns 1 if `ptr` contains the generic address of an object in shared memory space, otherwise returns 0.

### 10.15.3. __isConstant()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isconstant "Permalink to this headline")

__device__ unsigned int __isConstant(const void *ptr);

Returns 1 if `ptr` contains the generic address of an object in constant memory space, otherwise returns 0.

### 10.15.4. __isGridConstant()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#isgridconstant "Permalink to this headline")

__device__ unsigned int __isGridConstant(const void *ptr);

Returns 1 if `ptr` contains the generic address of a kernel parameter annotated with `__grid_constant__`, otherwise returns 0. Only supported for compute architectures greater than or equal to 7.x or later.

### 10.15.5. __isLocal()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#islocal "Permalink to this headline")

__device__ unsigned int __isLocal(const void *ptr);

Returns 1 if `ptr` contains the generic address of an object in local memory space, otherwise returns 0.

## 10.16. Address Space Conversion Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#address-space-conversion-functions "Permalink to this headline")

### 10.16.1. __cvta_generic_to_global()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-generic-to-global "Permalink to this headline")

__device__ size_t __cvta_generic_to_global(const void *ptr);

Returns the result of executing the _PTX_`cvta.to.global` instruction on the generic address denoted by `ptr`.

### 10.16.2. __cvta_generic_to_shared()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-generic-to-shared "Permalink to this headline")

__device__ size_t __cvta_generic_to_shared(const void *ptr);

Returns the result of executing the _PTX_`cvta.to.shared` instruction on the generic address denoted by `ptr`.

### 10.16.3. __cvta_generic_to_constant()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-generic-to-constant "Permalink to this headline")

__device__ size_t __cvta_generic_to_constant(const void *ptr);

Returns the result of executing the _PTX_`cvta.to.const` instruction on the generic address denoted by `ptr`.

### 10.16.4. __cvta_generic_to_local()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-generic-to-local "Permalink to this headline")

__device__ size_t __cvta_generic_to_local(const void *ptr);

Returns the result of executing the _PTX_`cvta.to.local` instruction on the generic address denoted by `ptr`.

### 10.16.5. __cvta_global_to_generic()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-global-to-generic "Permalink to this headline")

__device__ void * __cvta_global_to_generic(size_t rawbits);

Returns the generic pointer obtained by executing the _PTX_`cvta.global` instruction on the value provided by `rawbits`.

### 10.16.6. __cvta_shared_to_generic()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-shared-to-generic "Permalink to this headline")

__device__ void * __cvta_shared_to_generic(size_t rawbits);

Returns the generic pointer obtained by executing the _PTX_`cvta.shared` instruction on the value provided by `rawbits`.

### 10.16.7. __cvta_constant_to_generic()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-constant-to-generic "Permalink to this headline")

__device__ void * __cvta_constant_to_generic(size_t rawbits);

Returns the generic pointer obtained by executing the _PTX_`cvta.const` instruction on the value provided by `rawbits`.

### 10.16.8. __cvta_local_to_generic()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cvta-local-to-generic "Permalink to this headline")

__device__ void * __cvta_local_to_generic(size_t rawbits);

Returns the generic pointer obtained by executing the _PTX_`cvta.local` instruction on the value provided by `rawbits`.

## 10.17. Alloca Function[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#alloca-function "Permalink to this headline")

### 10.17.1. Synopsis[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synopsis "Permalink to this headline")

__host__ __device__ void * alloca(size_t size);

### 10.17.2. Description[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#description "Permalink to this headline")

The `alloca()` function allocates `size` bytes of memory in the stack frame of the caller. The returned value is a pointer to allocated memory, the beginning of the memory is 16 bytes aligned when the function is invoked from device code. The allocated memory is automatically freed when the caller to `alloca()` is returned.

Note

On Windows platform, `<malloc.h>` must be included before using `alloca()`. Using `alloca()` may cause the stack to overflow, user needs to adjust stack size accordingly.

It is supported with compute capability 5.2 or higher.

### 10.17.3. Example[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#example "Permalink to this headline")

__device__ void foo(unsigned int num) {
    int4 *ptr = (int4 *)alloca(num * sizeof(int4));
    // use of ptr
    ...
}

## 10.18. Compiler Optimization Hint Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compiler-optimization-hint-functions "Permalink to this headline")

The functions described in this section can be used to provide additional information to the compiler optimizer.

### 10.18.1. __builtin_assume_aligned()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#builtin-assume-aligned "Permalink to this headline")

void * __builtin_assume_aligned (const void *exp, size_t align)

Allows the compiler to assume that the argument pointer is aligned to at least `align` bytes, and returns the argument pointer.

Example:

void *res = __builtin_assume_aligned(ptr, 32); // compiler can assume 'res' is
                                               // at least 32-byte aligned

Three parameter version:

void * __builtin_assume_aligned (const void *exp, size_t align,
                                 <integral type> offset)

Allows the compiler to assume that `(char *)exp - offset` is aligned to at least `align` bytes, and returns the argument pointer.

Example:

void *res = __builtin_assume_aligned(ptr, 32, 8); // compiler can assume
                                                  // '(char *)res - 8' is
                                                  // at least 32-byte aligned.

### 10.18.2. __builtin_assume()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#builtin-assume "Permalink to this headline")

void __builtin_assume(bool exp)

Allows the compiler to assume that the Boolean argument is true. If the argument is not true at run time, then the behavior is undefined. Note that if the argument has side effects, the behavior is unspecified.

Example:

 __device__ int get(int *ptr, int idx) {
   __builtin_assume(idx <= 2);
   return ptr[idx];
}

### 10.18.3. __assume()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#assume "Permalink to this headline")

void __assume(bool exp)

Allows the compiler to assume that the Boolean argument is true. If the argument is not true at run time, then the behavior is undefined. Note that if the argument has side effects, the behavior is unspecified.

Example:

 __device__ int get(int *ptr, int idx) {
   __assume(idx <= 2);
   return ptr[idx];
}

### 10.18.4. __builtin_expect()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#builtin-expect "Permalink to this headline")

long __builtin_expect (long exp, long c)

Indicates to the compiler that it is expected that `exp == c`, and returns the value of `exp`. Typically used to indicate branch prediction information to the compiler.

Example:

// indicate to the compiler that likely "var == 0",
// so the body of the if-block is unlikely to be
// executed at run time.
if (__builtin_expect (var, 0))
  doit ();

### 10.18.5. __builtin_unreachable()[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#builtin-unreachable "Permalink to this headline")

void __builtin_unreachable(void)

Indicates to the compiler that control flow never reaches the point where this function is being called from. The program has undefined behavior if the control flow does actually reach this point at run time.

Example:

// indicates to the compiler that the default case label is never reached.
switch (in) {
case 1: return 4;
case 2: return 10;
default: __builtin_unreachable();
}

### 10.18.6. Restrictions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#restrictions "Permalink to this headline")

`__assume()` is only supported when using `cl.exe` host compiler. The other functions are supported on all platforms, subject to the following restrictions:

- If the host compiler supports the function, the function can be invoked from anywhere in translation unit.
    
- Otherwise, the function must be invoked from within the body of a `__device__`/ `__global__`function, or only when the `__CUDA_ARCH__` macro is defined[5](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#fn12).
    

## 10.19. Warp Vote Functions[](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-vote-functions "Permalink to this headline")

int __all_sync(unsigned mask, int predicate);
int __any_sync(unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);
unsigned __activemask();

Deprecation notice: `__any`, `__all`, and `__ballot` have been deprecated in CUDA 9.0 for all devices.

Removal notice: When targeting devices with compute capability 7.x or higher, `__any`, `__all`, and `__ballot` are no longer available and their sync variants should be used instead.

The warp vote functions allow the threads of a given [warp](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture) to perform a reduction-and-broadcast operation. These functions take as input an integer `predicate` from each thread in the warp and compare those values with zero. The results of the comparisons are combined (reduced) across the [active](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes) threads of the warp in one of the following ways, broadcasting a single return value to each participating thread:

`__all_sync(unsigned mask, predicate)`:

Evaluate `predicate` for all non-exited threads in `mask` and return non-zero if and only if `predicate` evaluates to non-zero for all of them.

`__any_sync(unsigned mask, predicate)`:

Evaluate `predicate` for all non-exited threads in `mask` and return non-zero if and only if `predicate` evaluates to non-zero for any of them.

`__ballot_sync(unsigned mask, predicate)`:

Evaluate `predicate` for all non-exited threads in `mask` and return an integer whose Nth bit is set if and only if `predicate` evaluates to non-zero for the Nth thread of the warp and the Nth thread is active.

`__activemask()`:

Returns a 32-bit integer mask of all currently active threads in the calling warp. The Nth bit is set if the Nth lane in the warp is active when `__activemask()` is called. [Inactive](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture-notes) threads are represented by 0 bits in the returned mask. Threads which have exited the program are always marked as inactive. Note that threads that are convergent at an `__activemask()` call are not guaranteed to be convergent at subsequent instructions unless those instructions are synchronizing warp-builtin functions.

For `__all_sync`, `__any_sync`, and `__ballot_sync`, a mask must be passed that specifies the threads participating in the call. A bit, representing the thread’s lane ID, must be set for each participating thread to ensure they are properly converged before the intrinsic is executed by the hardware. Each calling thread must have its own bit set in the mask and all non-exited threads named in mask must execute the same intrinsic with the same mask, or the result is undefined.

These intrinsics do not imply a memory barrier. They do not guarantee any memory ordering.