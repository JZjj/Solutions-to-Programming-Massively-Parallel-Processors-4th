# Note Ch3

A grid consists of one or more blocks, and each block consists of one or more threads. A grid is a 3-D array of blocks, and each block is a 3D array of threads. When calling a kernel, the program needs to specify the size of the grid and the blocks in each dimension.

```
<<< x, y >>>
x is gridDim
y is blockDim
```

The first execution specify the dimensions of the grid and the dimensions of each block. These dimensions are available via the gridDim and blockDim variables. The type of parameter is dim3. For example, the following host code generate 1D grid that consists of 32 blocks, each of which consists of 128 threads. The total number of threads is 128 * 32 = 4096

```
dim3 dimGrid(32, 1, 1);
dim3 dimBlock(128, 1, 1);
vecAddKernel<<<dimGrid, dimBlock>>>(...);
```

CUDA provides a special shortcut for calling a kernel with one-dimensional (1D) grids and blocks.

```
vecAddKernel<<<ceil(n/256), 256>>>
```

In CUDA C, the allowed values of gridDim.x range from 1 to 2^31 - 1, and gridDim.y and gridDim.z range from 1 to 2^16 - 1.

The total size of a block is limited to 1024 threads, meaning dimBlock <= 1024.

These threads can be distributed across the three dimensions in any way as long as the total number does not exceed 1024. For example, blockDim(512, 1, 1,), (8, 16, 4) and (32, 16, 2) are all allowed, but (32, 32, 2) is not allowed because the total number of threads would exceed 1024.
32 * 32 * 2 = 2048.

A grid and its block do not need to have the same dimensionality. Also, the ordering of the block and thread labels is highest dimension comes first.

The ANSI C standard on the basis of which CUDA C was developed requires the number of columns in Pin to be known at compile time for Pin to be accessed as a 2D array. But this info is not known at compile time for dynamically allocated arrays. In fact, dynamically array allows the sizes and dimensions of these arrays to vary according to the data size at runtime. Programmers need to explicitly linearize a dynamically allocated 2D array into an equivalent 1D array in the current CUDA C.

In reality, all multidimensional arrays in C are linearized. This is due to the use of flat memory space in modern computers.

CUDA C uses row major rather than column major.
```
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < Width) && (col < Width)){
		float Pvalue = 0;
		for (int k = 0; k < Width; k++){
			Pvalue += M[row * Width + k] * N[k * Width + col];
			}
		P[row * Width + col] = Pvalue;
			}	
	}


__global__ void MatrixRow(float* M, float* N, float* P, int Width){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < Width){
		float Pvalue = 0;
		for (int k = 0; k < Width; k++){
			Pvalue += M[]
			}	
	}
}
```