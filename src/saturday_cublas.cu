
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------
// Error checking helpers
// ---------------------------------------

#define CUDA_CHECK(call) do {                            \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",         \
                __FILE__, __LINE__,                      \
                cudaGetErrorString(err));                \
        exit(EXIT_FAILURE);                              \
    }                                                    \
} while (0)

#define CUBLAS_CHECK(call) do {                          \
    cublasStatus_t status = (call);                      \
    if (status != CUBLAS_STATUS_SUCCESS) {              \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",      \
                __FILE__, __LINE__, (int)status);       \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while (0)

constexpr int REPEATS = 20;
constexpr int WARMUP = 5;

// ---------------------------------------
// 1. CPU matrix multiplication
// ---------------------------------------

// A is M x K
// B is K x N
// C is M x N
void cpu_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k]
                     * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}

// ---------------------------------------
// 2. Naive CUDA matrix multiplication
// ---------------------------------------

// Every CUDA thread produces one output value.
__global__ void naive_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k]
             * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// ---------------------------------------
// 3. cuBLAS wrapper for ROW-MAJOR input
// ---------------------------------------

void cublas_matmul(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS interprets matrices as column-major.
    // Reverse A and B to calculate the row-major C.
    //
    // The column-major operation is:
    // C_transpose = B_transpose * A_transpose

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,         // Rows of column-major C^T
        M,         // Columns of column-major C^T
        K,         // Shared dimension
        &alpha,
        B, N,      // B is passed first
        A, K,      // A is passed second
        &beta,
        C, N
    ));
}

// ---------------------------------------
// 4. CUDA event benchmark helper
// ---------------------------------------

// mode 0 = naive CUDA
// mode 1 = cuBLAS
float benchmark(
    int mode,
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
) {
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    auto launch = [&]() {
        if (mode == 0) {
            naive_matmul<<<grid, block>>>(
                A, B, C, M, N, K);

            CUDA_CHECK(cudaGetLastError());
        } else {
            cublas_matmul(
                handle, A, B, C, M, N, K);
        }
    };

    // Warmup before measurement.
    for (int i = 0; i < WARMUP; ++i) {
        launch();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i) {
        launch();
    }

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Average GPU-stream elapsed time per call.
    return elapsed_ms / REPEATS;
}

// ---------------------------------------
// 5. Compare numerical results
// ---------------------------------------

bool verify(
    const std::vector<float>& reference,
    const std::vector<float>& result,
    const char* name
) {
    double max_error = 0.0;

    for (size_t i = 0; i < reference.size(); ++i) {

        double ref = reference[i];
        double actual = result[i];

        if (!std::isfinite(actual)) {
            printf("%s contains non-finite output\n", name);
            return false;
        }

        double difference = std::fabs(ref - actual);

        max_error = std::max(max_error, difference);

        // Mixed absolute and relative tolerance.
        double tolerance =
            1e-3 + 1e-3 * std::fabs(ref);

        if (difference > tolerance) {
            printf(
                "%s FAILED at %zu: ref=%f actual=%f\n",
                name, i, ref, actual);

            return false;
        }
    }

    printf("%s PASS (max abs error: %.6g)\n",
           name, max_error);

    return true;
}

// ---------------------------------------
// 6. Run a matrix-size experiment
// ---------------------------------------

bool run_test(int size, cublasHandle_t handle) {

    int M = size;
    int N = size;
    int K = size;

    size_t count = size_t(size) * size;
    size_t bytes = count * sizeof(float);

    printf("\n================================\n");
    printf("Matrix size: %d x %d\n", size, size);
    printf("================================\n");

    std::vector<float> A(count);
    std::vector<float> B(count);

    std::vector<float> C_naive(count);
    std::vector<float> C_cublas(count);

    // Small deterministic values.
    for (size_t i = 0; i < count; ++i) {
        A[i] = float((i % 13) - 6.0f) / 13.0f;
        B[i] = float((i % 17) - 8.0f) / 17.0f;
    }

    // Allocate memory on the GPU.
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to GPU.
    auto transfer_start = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaMemcpy(
        d_A, A.data(), bytes,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B, B.data(), bytes,
        cudaMemcpyHostToDevice));

    auto transfer_stop = std::chrono::steady_clock::now();

    double h2d_ms =
        std::chrono::duration<double, std::milli>(
            transfer_stop - transfer_start).count();

    // Time the naive CUDA kernel.
    float naive_ms = benchmark(
        0, handle, d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaMemcpy(
        C_naive.data(), d_C, bytes,
        cudaMemcpyDeviceToHost));

    // Time cuBLAS.
    float cublas_ms = benchmark(
        1, handle, d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaMemcpy(
        C_cublas.data(), d_C, bytes,
        cudaMemcpyDeviceToHost));

    // Compare the two GPU results.
    bool gpu_ok = verify(
        C_naive, C_cublas, "cuBLAS vs naive");

    // For smaller cases, also check a CPU reference.
    bool cpu_ok = true;
    double cpu_ms = 0.0;

    if (size <= 512) {
        std::vector<float> C_cpu(count);

        auto start = std::chrono::steady_clock::now();

        cpu_matmul(
            A.data(), B.data(), C_cpu.data(),
            M, N, K);

        auto stop = std::chrono::steady_clock::now();

        cpu_ms =
            std::chrono::duration<double, std::milli>(
                stop - start).count();

        cpu_ok = verify(
            C_cpu, C_naive, "CPU vs naive");

        cpu_ok = verify(
            C_cpu, C_cublas, "CPU vs cuBLAS")
            && cpu_ok;
    }

    // GEMM floating-point operation estimate.
    double flops = 2.0 * M * N * K;

    double naive_gflops =
        flops / (naive_ms * 1e6);

    double cublas_gflops =
        flops / (cublas_ms * 1e6);

    double library_speedup = naive_ms / cublas_ms;

    double percent_of_cublas =
        (naive_gflops / cublas_gflops) * 100.0;

    printf("\nPERFORMANCE RESULTS\n");

    printf("H2D input copies: %.4f ms\n", h2d_ms);

    if (size <= 512) {
        printf("CPU baseline: %.4f ms\n", cpu_ms);
    }

    printf("Naive CUDA: %.4f ms\n", naive_ms);
    printf("cuBLAS:     %.4f ms\n", cublas_ms);

    printf("Naive CUDA: %.2f GFLOP/s\n",
           naive_gflops);

    printf("cuBLAS:     %.2f GFLOP/s\n",
           cublas_gflops);

    printf("cuBLAS speedup: %.2fx\n",
           library_speedup);

    printf("Naive CUDA as %% of cuBLAS: %.2f%%\n",
           percent_of_cublas);

    if (size <= 512) {
        printf("CPU / naive kernel time ratio: %.2fx\n",
               cpu_ms / naive_ms);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return gpu_ok && cpu_ok;
}

// ---------------------------------------
// 7. Program entry point
// ---------------------------------------

int main(int argc, char** argv) {

    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n",
           prop.major, prop.minor);
    printf("SM count: %d\n", prop.multiProcessorCount);

    // Create a cuBLAS library context.
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Use cuBLAS default math mode.
    CUBLAS_CHECK(cublasSetMathMode(
        handle, CUBLAS_DEFAULT_MATH));

    std::vector<int> sizes = {256, 512, 1024};

    // Allow a custom matrix size from the command line.
    if (argc == 2) {
        int size = std::atoi(argv[1]);

        if (size < 1 || size > 8192) {
            fprintf(stderr,
                    "Matrix size must be 1..8192\n");

            CUBLAS_CHECK(cublasDestroy(handle));
            return EXIT_FAILURE;
        }

        sizes = {size};
    }

    bool all_ok = true;

    for (int size : sizes) {
        if (!run_test(size, handle)) {
            all_ok = false;
        }
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    printf("\nFinal result: %s\n",
           all_ok ? "PASS" : "FAIL");

    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

