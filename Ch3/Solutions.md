# Exercises



### 1.
In this chapter we implemented a matrix multiplication kernel that has
each thread produce one output matrix element. In this question, you
will implement different matrix-matrix multiplication kernels and
compare them.



a.  Write a kernel that has each thread produce one output matrix row.
Fill in the execution configuration parameters for the design.
```agsl
__global__ void matMulRowKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}
```


b.  Write a kernel that has each thread produce one output matrix
column. Fill in the execution configuration parameters for the design.
```agsl
__global__ void matMulColKernel(float* A, float* B, float* C, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        for (int row = 0; row < N; row++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}
```

c.  Analyze the pros and cons of each of the two kernel designs.

For thread per row, since CUDA underlying compiler is using row-major layout, it has better memory 
coalescing, while thread per col could has bad performance since the memory access
is non-coalesced.


------------------------------------------------------------------------



### 2.
A matrix-vector multiplication takes an input matrix ****B**** and a vector
****C**** and produces one output vector ****A****. Each element of the output
vector ****A**** is the dot product of one row of the input matrix ****B**** and
****C****, that is,\\[ A\[i\] = `\sum`{=tex}\_j B\[i\]\[j\] `\times `{=tex}C\[j\] \]

For simplicity we will handle only square matrices whose elements are
single-precision floating-point numbers. Write a matrix-vector
multiplication kernel and the host stub function that can be called with
four parameters:
- pointer to the output matrix,
- pointer to the input matrix,
- pointer to the input vector, and
- the number of elements in each dimension.

Use one thread to calculate an output vector element.
```agsl
__global__ void matMulKernel(float* A, float* B, float* C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        float sum = 0.0f;
        for (int j = 0; j < N; j++){
            sum += B[i * N + j] * C[j];
        }
        A[row] = sum;
    }
}
```





------------------------------------------------------------------------



### 3.

Consider the following CUDA kernel and the corresponding host function
that calls it:

``` cpp

__global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N) {

        b[row * N + col] = a[row * N + col] / 2.1f + 4.8f;

    }

}

  

void foo(float* a_d, float* b_d) {

    unsigned int M = 150;

    unsigned int N = 300;

    dim3 bd(16, 32);

    dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);

    foo_kernel<<<gd, bd>>>(a_d, b_d, M, N);

}

```



a.  What is the number of threads per block?

16 x 32 = 512

b.  What is the number of threads in the grid?

Grid dimensions:

(300 - 1)/16 + 1 x (150 - 1)/32 + 1 = 19 x 5 = 95

95 x 512 = 48, 640

c.  What is the number of blocks in the grid?

19 x 5 = 95

d.  What is the number of threads that execute the code on line 05?

Because runs only is row < M && col < N, so valid threads
are 150 x 300 = 45,000 threads.


------------------------------------------------------------------------



### 4.
Consider a 2D matrix with a width of 400 and a height of 500. The matrix
is stored as a one-dimensional array. Specify the array index of the
matrix element at row 20 and column 10:

a.  If the matrix is stored in ****row-major order****.

index = row x width + col = 20 x 400 + 10 = 8010

b.  If the matrix is stored in ****column-major order****.

index = col x height + row = 10 x 500 + 20 = 5020


------------------------------------------------------------------------


### 5.

Consider a 3D tensor with a width of 400, a height of 500, and a depth
of 300. The tensor is stored as a one-dimensional array in ****row-major**
**order****. Specify the array index of the tensor element at ****x = 10, y =**
**20, z = 5****.

index = z x (height x width) + y x width + x 
      = 5 x (500 x 400) + 20 x 400 + 10
      = 5 x 200,000 + 8,000 + 10
      = 1,008,010