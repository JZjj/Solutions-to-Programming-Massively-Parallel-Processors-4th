# Exercise

## 1) Kernel analysis

Consider the following CUDA kernel and the corresponding host function that calls it:

```cpp
__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < 40 || threadIdx.x >= 104) {   // line 04
        b[i] = a[i] + 1;
    }

    if (i % 2 == 0) {                               // line 07
        a[i] = b[i] * 2;
    }

    for (unsigned int j = 0; j < 5 - (i % 3); ++j) { // line 09
        b[i] += j;
    }
}

void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<< (N + 128 - 1) / 128, 128 >>> (a_d, b_d);
}
```

**Given/Assumptions:**
- `N = 1024`, `blockDim.x = 128`, `gridDim.x = (N + 127) / 128 = 8` blocks.
- A warp has 32 threads.
- Global index: `i = blockIdx.x * 128 + threadIdx.x`.

---

### Q1(a) What is the number of warps per block?
**Answer:** `128 / 32 = 4` warps per block.  
**Explanation:** Warp = 32 lanes. A 128-thread block splits into four 32-lane warps.

---

### Q1(b) What is the number of warps in the grid?
**Answer:** `8 blocks × 4 warps/block = 32` warps.  
**Explanation:** `N` is exactly divisible by the block size (128), so no partial warps; just `blocks × warps_per_block`.

---

### Q1(c) For the statement on line 04: `if (threadIdx.x < 40 || threadIdx.x >= 104)`

Per block, warp lane ranges are:
- **Warp 0:** lanes 0–31
- **Warp 1:** lanes 32–63
- **Warp 2:** lanes 64–95
- **Warp 3:** lanes 96–127

Condition is **true** for lanes `0–39` and `104–127`.

| Warp | Lanes affected | True lanes | Divergent? | SIMD efficiency |
|---|---|---|---|---|
| 0 | 0–31 | 32 | No | 32/32 = **100%** |
| 1 | 32–63 | 32–39 → 8 true | Yes | 8/32 = **25%** |
| 2 | 64–95 | none | (inactive) | 0/32 (does not execute) |
| 3 | 96–127 | 104–127 → 24 true | Yes | 24/32 = **75%** |

**Answers:**  
i) **Active warps in the grid:** `3 per block × 8 blocks = 24`.  
ii) **Divergent warps in the grid:** `2 per block × 8 blocks = 16`.  
iii) **Warp 0 (block 0) SIMD eff:** **100%**.  
iv) **Warp 1 (block 0) SIMD eff:** **25%**.  
v) **Warp 3 (block 0) SIMD eff:** **75%**.

**Explanation:** A warp is *active* for this statement if at least one lane evaluates the predicate `true`. Warp 2 has zero true lanes, so it doesn’t execute this instruction; it’s neither active nor divergent for this statement.

---

### Q1(d) For the statement on line 07: `if (i % 2 == 0)`
**Answers:**  
i) **Active warps:** all 32 warps.  
ii) **Divergent warps:** all 32 warps.  
iii) **Warp 0 SIMD eff:** 16/32 = **50%**.

**Explanation:** Within any 32-consecutive global indices inside a warp, exactly half are even and half are odd. Thus every warp has lanes both taking and skipping the instruction → divergence with 50% efficiency.

---

### Q1(e) For the loop on line 09: `for (j = 0; j < 5 - (i % 3); ++j)`
`i % 3` determines the trip count:
- remainder 0 → 5 iterations (j = 0..4)
- remainder 1 → 4 iterations (j = 0..3)
- remainder 2 → 3 iterations (j = 0..2)

**Answers:**  
i) **No-divergence iterations:** **3** (j = 0,1,2) — all lanes still active.  
ii) **With divergence:** **2** (j = 3,4) — lanes with smaller trip counts have already exited.

**Explanation:** Up to `j = 2`, every lane still satisfies the loop condition. At `j = 3`, lanes with `i % 3 == 2` have finished; at `j = 4`, lanes with `i % 3 == 1` have also finished, leaving only lanes with `i % 3 == 0`.

---

## 2) Vector addition sizing

**Question:** For `N = 2000`, block size `= 512`, each thread computes one output. How many threads are launched?  
**Answer:** Blocks = `ceil(2000 / 512) = 4`, so threads launched = `4 × 512 =` **2048**.

**Explanation:** CUDA rounds up the number of blocks to cover all elements. The extra threads are typically guarded by `if (i < N)`.

---

## 3) Divergence due to boundary check

**Question:** With the setup from Q2, how many warps diverge due to the `if (i < N)` guard?  
**Answer:** **1** warp (the partial warp in the last block).

**Explanation:** The last block covers indices `1536..2047`. Valid work is `2000 - 1536 = 464` threads → `14` full warps (448 lanes) + `1` half warp (16 lanes). Only that half warp simultaneously has true/false lanes for the boundary check. The final warp with zero valid lanes is inactive for the guarded instruction and doesn’t count as divergent.

---

## 4) Barrier waiting percentage

**Question:** A block with 8 threads runs a section taking (µs): `2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9`, then synchronizes at a barrier. What percentage of total thread-time is waiting?  
**Answer:** ≈ **17.1%**.

**Explanation:** Let `T_barrier = max = 3.0 µs`. Each thread waits `T_barrier − t_i`.
- Total waiting = `8×3.0 − (2.0+2.3+3.0+2.8+2.4+1.9+2.6+2.9) = 24 − 19.9 = 4.1`.
- Total thread-time (busy + wait) = `8×3.0 = 24`.
- Waiting percentage = `4.1 / 24 ≈ 17.1%`.

---

## 5) Omitting `__syncthreads()` with 32-thread blocks

**Question:** Is it safe to skip `__syncthreads()` if each block has only 32 threads?  
**Answer:** **No.** Use `__syncthreads()` for block-wide sync (or `__syncwarp()` for warp-local sync).

**Explanation:** Modern GPUs support **independent thread scheduling**.

---

## 6) Best block size for given SM limits

**Question:** An SM can hold up to **1536 threads** and **4 blocks**. Which block size yields the most resident threads?  
**Answer:** **512 threads per block** (3 blocks × 512 = **1536** threads).

**Explanation:** Concurrency is limited by both threads and blocks per SM. The number of resident blocks is  
`min( floor(1536 / tpb), 4 )`.
- 128 t/b → min(12,4)=4 blocks → 4×128=512 threads
- 256 t/b → min(6,4)=4 blocks → 4×256=1024 threads
- **512 t/b → min(3,4)=3 blocks → 3×512=1536 threads (best)**
- 1024 t/b → min(1,4)=1 block → 1024 threads

---

## 7) Possibility & occupancy (limits: 64 blocks/SM, 2048 threads/SM)

**Answer:**
- (a) 8×128 = **1024** threads → **possible**, **50%** occupancy.
- (b) 16×64 = **1024** → **possible**, **50%**.
- (c) 32×32 = **1024** → **possible**, **50%**.
- (d) 64×32 = **2048** → **possible**, **100%**.
- (e) 32×64 = **2048** → **possible**, **100%**.

**Explanation:** Occupancy (by threads) = `resident_threads / 2048`. All cases respect the 64-block/SM cap.

---

## 8) Full-occupancy check under register and block limits

**Limits:** 2048 threads/SM, 32 blocks/SM, 65,536 registers/SM.  
Let `R_b = (threads per block) × (registers per thread)`. Blocks limited by registers = `floor(65536 / R_b)`.

- **(a)** 128 tpb, 30 regs → `R_b = 3840`. `floor(65536/3840) = 17` blocks (by regs).  
  Need 16 blocks to reach 2048 threads (since 16×128=2048). **Achievable → Full occupancy**.

- **(b)** 32 tpb, 29 regs → `R_b = 928`. Registers would allow 70 blocks, but **block limit** is 32.  
  Max resident threads = `32 × 32 = 1024` → **50% occupancy**. **Limiting factor:** blocks/SM.

- **(c)** 256 tpb, 34 regs → `R_b = 8704`. Registers allow `floor(65536/8704) = 7` blocks.  
  Threads would need 8 blocks to hit 2048, but regs cap at 7 → `7×256=1792` threads = **87.5%**. **Limiting factor:** registers.

---

## 9) Matrix-multiply claim sanity-check

**Question:** Student claims using **32×32 = 1024 threads per block** on a device whose max is **512**.  
**Answer:** **Not possible** on that device.

**Explanation:** The per-block thread limit is hardware-enforced. Use smaller blocks (e.g., 16×16 = 256) or have each thread compute multiple output elements via tiling to respect the limit and still cover the workload.

---

## Bonus: Handy formulas

- `gridDim = ceil(N / blockDim)`; `i = blockIdx*blockDim + threadIdx`.
- Warps per block = `blockDim / 32`; warps in grid = `gridDim × warps_per_block`.
- SIMD (warp) efficiency for an instruction = `active_lanes / 32`.
- Resident blocks per SM ≈ `min( floor(threads_limit/tpb), blocks_limit, floor(regs_SM / (tpb × regs_per_thread)) , sharedMem constraints )`.
- Occupancy (by threads) = `resident_threads / max_threads_per_SM`.
