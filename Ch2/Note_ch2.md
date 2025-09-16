# Note Ch2 
__Data parallelism__

__Task parallelism__

Structure of a CUDA C program reflects the coexistence of a ***host(CPU)*** and one or more ***devices (GPUs)*** in the computer. One can add device code into any source file. The device code includes functions, kernels, whose code is executed in a data-parallel manner.

__Grid__: all the threads that are launched by a kernel call.

On GPU, programmer can assume threads take very few clock cycles to generate and schedule, CPU threads typically take thousands of clock cycles to generate.

Vector addition is arguably the "Hello World" of parallel computation.
Input vectors A and B, output vector C are allocated in the main program.

```
void vecadd(float* A, float* B, float* C, int n){
	int size = n * sizeof(float);
	float *d_A, *d_B, *d_C;
	
	//P1: allocate device memory
	//P2: kernel function
	//P3: copy result from device to host, free device memory
}
```

In current CUDA, devices are often hardware cards with own DRAM - global memory. Following are two API functions for allocating and freeing device global memory from the host
```
cudaMalloc(address of a pointer to the allocated object, size in terms of bytes)
cudaFree()

#example
void vecAdd(float* A_h, float* B_h, float* C_h, int n){
	int size = n * sizeof(float);
	float *A_d, *B_d, *C_d;
	
	cudaMalloc((void **)&A_d, size);
	cudaMalloc((void **)&B_d, size);
	cudaMalloc((void **)&C_d, size);
	
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
	
	//kernel invocation code - to be shown later
	
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
	
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}
```

A kernel can launch a grid of threads into two hierarchy. Each grid is organized as an array of thread blocks, which are referred to as blocks for brevity. All block are of the same size, each block can contain up to 1024 threads.

For a given grid of threads, the number of threads in a block is available in ***blockDim***. Dimensionality of data. ***blockIdx*** gives all threads in a block a common block coordinate.
A kernel function is executed on a device and can be called from the host. 