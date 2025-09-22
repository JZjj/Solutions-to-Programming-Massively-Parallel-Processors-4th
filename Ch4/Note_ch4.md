# Ch4
CPUs are designed to minimize the latency of instruction and GPUs are designed to maximize the throughput of executing instructions.

When a kernel is called, the CUDA runtime system launches a grid of threads that execute the kernel code. These threads are assigned to SMs on a block-by-block basis. That is, all threads in a block are simultaneously assigned to the same SM.

Multiple blocks are likely to be simultaneously assigned to the same SM. However, blocks need to reserve hardware resources to execute, so only a limited number of blocks can be simultaneously assigned to a given SM.

Limited number of SMs, limited number of blocks->assigned to each SM. To ensure that all blocks in a grid get executed, the runtime system maintains a list of blocks that need to execute and assigns new blocks to SMs when previously assigned blocks complete execution. Thread in the same block can communicate through shared memory.

Thread in the same block can coordinate their activities using barrier synchronization
```
__syncthreads()
```

Not all threads in a block are guaranteed to execute either of the barriers->undefined execution behavior. The ability to execute the same application code on different hardware with different amounts of execution resources is referred to as transparent scalability, which reduces the burden on application developers and improves the usability of applications.

Thread scheduling is a hardware implementation concept.

If each block has 256 threads, we can determine that each block has 256/32 or 8 wraps. For a block whose size is not a multiple of 32, the last warp will be padded with inactive threads to fill up the 32 thread positions. For example, if a block has 45 threads, it will be partitioned into two warps, and the second warp will be padded with 16 inactive threads.

For blocks that consist of multiple dimensions of threads, the dimensions will be projected into a linearized row-major layour before partitioning into warps. The linear layout is determined by placing the rows with larger y and z coordinates.

Each thread is shown as T_yx. T_zyx.

The advantage of SIMD is that the cost of the control hardware, such as the instruction fetch/dispatch unit, is shared across many execution units. The design choice allows for a smaller percentage of the hardware to be dedicated to control and a larger percentage to be dedicated to increasing arithmetic throughput

**warps and SIMD Hardware**
To execute a program, the computer first inputs the program and its data into the memory. The program consist of a collection of instructions. The Control Unit maintains a Program Counter, which contains the memory address of the next instruction to be executed. In each "instruction cycle", the Control Unit uses the PC to fetch an instruction into the Instruction Register. The instruction bits are then examined to determine the action to be taken by all components of the computer. This is the reason why the model is also called the "stored program" model, which means that a user can change the behavior of a computer by storing a different program into its memory.

For GPU, the processor, which correspond to a processing block has only one control unit that fetches and dispatches instructions. The same control signals go to multiple processing units that each correspond to a core in the SM, each of which executes one of the threads in a warps. Since all processing units are controlled by the same instruction in the Instruction Register of the Control Unit, execution differences are due to the different data operand values in the register files.

Control units in modern processors are quite complex, including sophisticated logic for fetching instructions and access ports to the instruction cache. Having multiple processing units to share a control unit can result in significant reduction in hardware manufacturing cost and power consumption.

**control divergence**
when threads within a warp take different control flow paths, the SIMD hardware will take multiple passes through these paths, one pass for each path. However, for an if-else construct, if some threads in a warp follow the if-path while others follow the else path, the hardware will take two passes. One pass executes the threads that follow the if-path, and the other executes the threads that follow the else-path. During each pass, the threads that follow the other path are not allowed to take effect.

When thread in the same warp follow different execution paths, we say that these threads exhibit control divergence, that is, they diverge in their execution. The multi-pass approach to divergent warp execution extends the SIMD hardware's ability to implement full CUDA threads. The cost of divergence is the extra passes the hardware needs to take to allow different threads in a wrap to make their own decisions as well as the execution resources that are consumed by the inactive threads in each pass.

Performance impact of control divergence decreases as the size of the vector being processed increases. For a vector length of 100, one of the four warps will have control divergence. For a size of 1000, only one of 32 warps will have c.d. (handle boundary conditions)

Long operation latencies is the main reason why GPUs do not dedicate nearly as much chip area to cache memories and branch prediction mechanisms as CPUs do. As a result, GPU can dedicate more chip area to floating-point execution and memory access channel resources.
In a computer based von Neumann model, the code of the program is stored in the memory. The PC keeps track of the address of the instruction of the program that is being executed. The IR holds the instruction that is being executed. The register and memory hold the values of the variables and data structures.

Modern processors are designed to allow context switching, where multiple threads can time-share a processor by taking turns to make progress. By carefully saving and restoring the PC value and the contents of registers and memory, we can suspend the execution of a thread and correctly resume the execution of the tread later. However, saving and restoring register contents during context-switching in these processors can incur significant overhead in terms of added execution time.

Zero-overhead scheduling refers to the GPU's ability to put a warp that needs to wait for a long-latency instruction result to sleep and activate a warp that is ready to go without introducing any extra idle cycles because switching the execution from one thread to another requires saving the execution state (such as register contents of the out-going thread) to memory and loading the execution state of the incoming thread from memory. GPU SMs achieves zero-overhead scheduling by holding all the execution states for the assigned warps in the hardware register so there is no need to save and restore states when switching from one warp to another.

good -> assign many warps to an SM in order to tolerate long-latency operations. However, it may not always be possible to assign to the SM the maximum number of warps that the SM supports. The ratio of the number of warps assigned to an SM to the maximum number it supports is referred to as **occupancy**.

How SM resources are partitioned?
**The execution resources in an SM include registers, shared memory, thread block slots, and thread slots. These resources are dynamically partitioned across threads to support their execution. **

For example, an Ampere A100 GPU can support a maximum of 32 blocks per SM, 64 warps (2048 threads) per SM, and 1024 threads per block. If a grid is launched with a block size of 1024 threads, the 2048 thread slots each SM are partitioned and assigned to 2 blocks. In this case, each SM can accommodate up to 2 blocks. Similarly, if a grid is launched with a block size of 512, 256, 128, or 64 threads, the 2048 thread slots are partitioned and assigned to 4, 8, 16, or 32 blocks.

The ability to dynamically partition thread slots among blocks makes SM versatile -> thread slots and achieve maximum occupancy

Another negatively affect occupancy occurs when the maximum number of threads per block is not divisible by the block size. In the example of the Ampere A100. If a block size 768 is selected, the SM will be able to accomodate only 2 thread blocks, leaving 512 thread slots untuilized. In this case, neither the maximum threads per SM nor the maximum blocks per SM are reached.

One does, however, need to be aware of potential impact of register resource limitation on occupancy. For example, the Ampere A100 GPU allows a maximum of registers -> resource occupancy. CUDA running system.

It should be clear to the reader that the constraints of all the dynamically partitioned resourcs interact with each other in a complex manner. Accurate determination of number of threads running in each SM can be difficult.

```
cudaDeviceProp devProp;
for (unsigned int i = 0; i < devCount; i++){
	cudaGetDeviceProperties(&devProp, i);6
}
```
