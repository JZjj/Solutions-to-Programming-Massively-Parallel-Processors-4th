# Chapter 2: Heterogeneous Data Parallel Computing
## Exercises with Answers & Explanations

---

### 1. Thread/Block Index for Vector Addition
If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?

- (A) `i = threadIdx.x + threadIdx.y;`
- (B) `i = blockIdx.x + threadIdx.x;`
- **(C) `i = blockIdx.x * blockDim.x + threadIdx.x;`**
- (D) `i = blockIdx.x * threadIdx.x;`

**Explanation:**  
Each thread index within a block is given by `threadIdx.x`, and each block contributes `blockDim.x` threads. So the global thread index is `(blockIdx.x * blockDim.x) + threadIdx.x`.

---

### 2. Two Adjacent Elements per Thread
Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

- (A) `i = blockIdx.x * blockDim.x + threadIdx.x + 2;`
- (B) `i = blockIdx.x * threadIdx.x * 2;`
- **(C) `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`**
- (D) `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

**Explanation:**  
If each thread handles **2 elements**, the base index for a thread is doubled. For example, for thread handling (0, 0), (0, 1),
i = 0 and for threadIdx.x = 1, which handling (0, 2), (0, 3), i = 2.

---

### 3. Two Elements per Thread (Section Processing)
We want to use each thread to calculate two elements of a vector addition. Each thread block processes `2 * blockDim.x` consecutive elements that form sections. All threads in each block will process a section first, each processing one element. Assume that variable `i` should be the index for the first element to be processed by a thread. What would be the expression?

- (A) `i = blockIdx.x * blockDim.x + threadIdx.x + 2;`
- (B) `i = blockIdx.x * threadIdx.x * 2;`
- **(C) `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`**
- (D) `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

**Explanation:**  
Each thread block processes 2 * blockDim.x consecutive elements that form two sections.

---

### 4. Threads Needed in Grid
For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

- (A) 8000
- (B) 8196
- **(C) 8192**
- (D) 8200

**Explanation:**  
We need at least `ceil(8000 / 1024) = 8` blocks.  
`8 * 1024 = 8192` threads in total. The last 192 threads will be unused, but CUDA launches whole blocks.

---

### 5. `cudaMalloc` for Integer Array
If we want to allocate an array of `v` integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the `cudaMalloc` call?

- (A) `n`
- (B) `v`
- (C) `n * sizeof(int)`
- **(D) `v * sizeof(int)`**

**Explanation:**  
`cudaMalloc(&ptr, size_in_bytes);`  
We want `v` integers, each of size `sizeof(int)`. Hence `v * sizeof(int)`.

---

### 6. `cudaMalloc` for Floating-Point Array
If we want to allocate an array of `n` floating-point elements and have a floating-point pointer variable `A_d` to point to the allocated memory, what would be an appropriate expression for the first argument of the `cudaMalloc` call?

- (A) `n`
- (B) `(void*) A_d`
- (C) `*A_d`
- **(D) `(void**) &A_d`**

**Explanation:**  
`cudaMalloc` needs the **address of the pointer** (so it can modify it to point to GPU memory). That’s why we pass `(void**)&A_d`.

---

### 7. Copy Data with `cudaMemcpy`
If we want to copy 3000 bytes of data from host array `A_h` (pointer to element 0 of the source array) to device array `A_d` (pointer to element 0 of the destination array), what would be an appropriate API call?

- (A) `cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);`
- (B) `cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);`
- **(C) `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`**
- (D) `cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);`

**Explanation:**  
The correct prototype is `cudaMemcpy(dst, src, size, direction)`.  
Here: destination = `A_d` (device), source = `A_h` (host), size = 3000, direction = `cudaMemcpyHostToDevice`.

---

### 8. Declaring CUDA Error Variable
How would one declare a variable `err` that can appropriately receive the returned value of a CUDA API call?

- (A) `int err;`
- (B) `cudaError err;`
- **(C) `cudaError_t err;`**
- (D) `cudaSuccess_t err;`

**Explanation:**  
All CUDA runtime API calls return an error code of type `cudaError_t`.

---

### 9. CUDA Kernel Analysis

#### Kernel Code
```cpp
__global__ void foo_kernel(float* a, float* b, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        b[i] = 2.7f * a[i] - 4.3f;
    }
}

void foo(float* a_d, float* b_d) {
    unsigned int N = 200000;
    foo_kernel <<< (N + 128 - 1) / 128, 128 >>> (a_d, b_d, N);
}
```
a. What is the number of threads per block?

__128__

b. What is the number of threads in the grid?

__floor( ( 200000 + 128 - 1 ) / 128 ) * 128 = 200,192__

c. What is the number of blocks in the grid?
__1564__

d. What is the number of threads that execute the code on line 02?

__200,192__

e. What is the number of threads that execute the code on line 04?

__200,000__ ( because i < N )

### 10.

The intern can avoid duplication by declaring with
```
__host__ __device__ 
```

