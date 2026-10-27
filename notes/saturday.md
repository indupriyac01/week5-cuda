# Week 5 Saturday - cuBLAS SGEMM

## Concepts
- cuBLAS
- cublasSgemm
- Row-major and column-major memory
- Leading dimensions
- CUDA events
- GFLOP/s and GPU speedup

### Notes on the concepts

**Row-major vs column-major.** cuBLAS inherits the Fortran BLAS convention and reads
every matrix as column-major. C and C++ arrays are row-major. Rather than transposing
the data, the wrapper exploits the identity `(A·B)ᵀ = Bᵀ·Aᵀ`: passing B first, then A,
with `CUBLAS_OP_N` on both, makes cuBLAS compute the column-major Cᵀ, which occupies
exactly the same bytes as the row-major C we want. No copies, no transposes.

**Leading dimensions.** `lda`/`ldb`/`ldc` are the stride in elements between consecutive
columns in cuBLAS's column-major view. Because of the swap above, the row-major row
lengths get passed: B gets N, A gets K, C gets N.

**CUDA events vs wall clock.** Kernel launches are asynchronous, so `std::chrono` around
a launch measures the time to queue the work, not to do it. `cudaEventRecord` inserts
markers into the stream, so the elapsed time is GPU-side. The host-side timer is still
the right tool for `cudaMemcpy`, which is synchronous here.

## Environment
GPU: Tesla T4 (Turing, TU104)
Compute capability: 7.5
SM count: 40
Driver: 580.178.04
CUDA Toolkit version: <`nvcc --version`>
cuBLAS version: <`cublasGetVersion(handle, &v)`, or check the libcublas package>
Math mode: `CUBLAS_DEFAULT_MATH` (FP32 path, no tensor cores)

Build flags: `-std=c++17 -O3 -lineinfo -arch=sm_75 --ptxas-options=-v`
ptxas on `naive_matmul`: 40 registers, 0 bytes spill, 0 bytes stack frame.

Relevant T4 hardware limits:
- FP32 peak: ~8.1 TFLOP/s
- Memory bandwidth: ~320 GB/s
- Power cap: 70 W

## Results

| Size | H2D copies | CPU baseline | Naive CUDA | cuBLAS | Naive GFLOP/s | cuBLAS GFLOP/s | Speedup | Naive % of cuBLAS |
|------|-----------|--------------|------------|--------|---------------|----------------|---------|-------------------|
| 256  | 0.1666 ms | 20.66 ms     | 0.0929 ms  | 0.0299 ms | 361.20 | 1120.51 | 3.10x  | 32.24% |
| 512  | 0.6913 ms | 257.50 ms    | 0.6742 ms  | 0.0782 ms | 398.16 | 3430.64 | 8.62x  | 11.61% |
| 1024 | 1.9690 ms | not run      | 5.2577 ms  | 0.4791 ms | 408.45 | 4482.53 | 10.97x | 9.11%  |

Timings are the mean of 20 iterations after 5 warmup launches, measured with CUDA events.

Correctness: all comparisons PASS. Max absolute error was 7.15e-07 (256), 2.98e-07 (512),
and 2.15e-06 (1024, naive vs cuBLAS only). CPU reference was skipped at 1024 by design.

## Analysis

### 1. Why was cuBLAS faster?

The naive kernel is memory-bound and the gap widens with size because of it.

Each thread computes one output element by streaming a full row of A and a full column
of B from global memory: 2K loads for 2K flops, an arithmetic intensity of 0.25 flop/byte.
The T4 needs roughly 25 flop/byte to saturate its FP32 units, so the kernel is off by two
orders of magnitude and the 400 GFLOP/s ceiling is a bandwidth ceiling, not a compute one.
That ceiling sits above the 80 GFLOP/s that raw DRAM bandwidth alone would allow, which
means L1 and L2 are already absorbing a large share of the redundant traffic — the warp
reads B coalesced, and threads in a block reuse the same rows of A. But the reuse is
accidental and cache-sized, not planned.

cuBLAS plans it. The library tiles the output into blocks, stages tiles of A and B in
shared memory, and has each thread compute an 8×8 or similar patch of C held entirely in
registers. Arithmetic intensity rises with the tile dimension, so the same bytes feed far
more flops. On top of that it uses vectorized 128-bit loads, double-buffers the shared-memory
stages so loads overlap with math, and picks a different tile shape per problem size.

The flat profile of the naive kernel is the clearest evidence: 361 → 398 → 408 GFLOP/s
across a 64x increase in work. It found its bandwidth limit at 256 and stayed there.
cuBLAS went 1120 → 3431 → 4483, still climbing.

### 2. Which matrix size gave the highest GFLOP/s?

1024 for both, but for different reasons.

For cuBLAS, 4482 GFLOP/s is about 55% of the T4's FP32 peak — a respectable number for a
non-tensor-core SGEMM. The smaller sizes underperform because there isn't enough work to
fill 40 SMs and amortize launch overhead: at 256 the whole GEMM is 33 MFLOP and finishes
in 30 µs, which is only a few times the kernel launch cost itself.

For the naive kernel the improvement from 361 to 408 is marginal and is mostly occupancy
and launch-overhead amortization, not better memory behavior.

Extrapolating: cuBLAS would likely keep improving to 2048 and 4096 before flattening.
Worth testing — `./saturday_cublas 2048` already works.

### 3. Were the numerical results correct?

Yes. Every comparison passed, with maximum absolute errors at the level expected from
FP32 rounding (~1e-7 to ~2e-6, growing with K as more terms accumulate).

Two caveats about how much this proves:

The test inputs are small, deterministic values in roughly [-0.5, 0.5] built from
`i % 13` and `i % 17`. They are well-conditioned and never produce large intermediate
magnitudes, so this exercises correctness of indexing far more than numerical robustness.
A matrix with wide dynamic range, or one engineered for cancellation, would separate the
two accumulation orders much more visibly.

The `verify` tolerance is `1e-3 + 1e-3 * |ref|`, which is generous for FP32 at these
magnitudes. It will catch a transposed index or an off-by-one in the leading dimension —
the bugs this test is really guarding against — but it would not catch a subtle precision
regression.

The exact-zero error at 512 between cuBLAS and naive is a coincidence of these particular
inputs, not a guarantee. Different summation orders generally give different last bits.

### 4. What are the limitations of this benchmark?

**Transfer time is excluded from the comparison.** The H2D copies cost 1.97 ms at 1024,
four times the 0.479 ms cuBLAS GEMM. For any workload where the data isn't already resident
on the device, the PCIe transfer dominates and the GEMM choice barely matters. The benchmark
measures kernel time, which is the right thing for comparing kernels and the wrong thing for
predicting end-to-end speedup.

**The CPU baseline is a strawman.** `cpu_matmul` is a single-threaded triple loop with the
`k` loop innermost over B's column — the worst possible access pattern for cache. A blocked,
vectorized, multi-threaded CPU GEMM (what OpenBLAS or MKL would do) is perhaps 50-100x faster
than this. The "222x / 382x faster than CPU" figures should be read as "faster than naive
scalar CPU code", not "faster than a CPU".

**No tensor cores.** `CUBLAS_DEFAULT_MATH` keeps SGEMM on the FP32 pipeline. The T4's tensor
cores are FP16/INT8 and would roughly 8x the matmul throughput at reduced precision. Both
numbers here are FP32-vs-FP32, which is the fair comparison, but it understates what the
hardware can do.

**Clock behavior isn't controlled.** The T4 is a 70 W passively cooled card. Sustained runs
can downclock, and `nvidia-smi` showed 8% utilization before the run started, so the box
isn't necessarily idle. 20 iterations is enough to average out scheduling noise but not a
thermal ramp. Locking clocks with `nvidia-smi -lgc` would make runs comparable across days.

**Square matrices only.** Real workloads are frequently skewed — a transformer's QKV
projection is tall and thin. Tile selection and performance both change with shape.

**cuBLAS first-call cost is hidden.** `cublasCreate` and the first `cublasSgemm` of each
shape pay a setup cost that the warmup loop absorbs. That's correct for steady-state
benchmarking, but a program doing one GEMM would feel it.

### 5. Which optimizations should I implement next?

In the order the performance gains come:

1. **Shared-memory tiling.** Load a 32×32 tile of A and of B into shared memory per block,
   have the block compute a 32×32 tile of C. Cuts global loads by ~32x. Expect to reach
   roughly 1000-1500 GFLOP/s — around where cuBLAS was at size 256.

2. **Register tiling / thread coarsening.** One thread computes a 4×4 or 8×8 patch of C
   rather than a single element, keeping the accumulators in registers. This is the single
   biggest step: it raises arithmetic intensity past the point where the kernel stops being
   bandwidth-bound. Watch the ptxas register count — the baseline is 40, and spilling here
   would undo the gain.

3. **Vectorized loads.** `float4` loads for the shared-memory staging, so each instruction
   moves 128 bits. Requires the leading dimensions to be multiples of 4.

4. **Shared-memory bank conflict avoidance.** Pad the tile stride by one element, or store
   the A tile transposed, so the column reads don't serialize across banks.

5. **Double buffering.** Prefetch tile *k+1* into a second shared-memory buffer while
   computing on tile *k*, hiding the load latency behind the math.

6. **Tensor cores (FP16 WMMA).** A different exercise — mixed precision changes the error
   analysis, and the `verify` tolerance would need revisiting.

Profiling to guide it: `ncu --set full ./saturday_cublas 1024` on the naive kernel should
show DRAM throughput near peak with FP32 pipe utilization in the single digits. After
tiling, that balance should invert. The roofline chart in Nsight Compute is the fastest way
to see which side of the limit each version sits on.

## Open questions
- Where does cuBLAS flatten out? Test 2048, 4096.
- How close can a hand-written kernel get to 4482 GFLOP/s on this card?
- How much of the cuBLAS advantage at 256 is tile selection vs. just launch overhead?
