// ============================================================================
// Preshuffle Weight GEMM - 完整优化版本
// ============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>

using fp16_t = _Float16;
using fp16x4_t = fp16_t __attribute__((ext_vector_type(4)));
using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp32x16_t = float __attribute__((ext_vector_type(16)));
using fp16x16_t = fp16_t __attribute__((ext_vector_type(16)));

#define HIP_CALL(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        printf("HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Baseline GEMM（已验证正确）
// ============================================================================

__global__ void gemm_mfma_baseline(
    const fp16_t* __restrict__ A,
    const fp16_t* __restrict__ B,
    fp16_t* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_b, int stride_c)
{
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 8;
    
    int block_m = blockIdx.x * MFMA_M;
    int block_n = blockIdx.y * MFMA_N;
    
    if (threadIdx.x >= 64) return;
    int tid = threadIdx.x;
    
    fp32x16_t v_c = {0.0f};
    
    for (int k = 0; k < K; k += MFMA_K) {
        int a_row = block_m + (tid % 32);
        int a_col = k + (tid / 32) * 4;
        
        fp16x4_t v_a = {0, 0, 0, 0};
        if (a_row < M && a_col + 3 < K) {
            v_a[0] = A[a_row * stride_a + a_col + 0];
            v_a[1] = A[a_row * stride_a + a_col + 1];
            v_a[2] = A[a_row * stride_a + a_col + 2];
            v_a[3] = A[a_row * stride_a + a_col + 3];
        }
        
        int row_group_id_b = (tid % 32) / 4;
        int row_id_b = (tid % 4) + 
                       (row_group_id_b % 2) * 8 + 
                       ((row_group_id_b % 4) / 2) * 4 + 
                       (row_group_id_b / 4) * 16;
        
        int b_col = block_n + row_id_b;
        int b_row = k + (tid / 32) * 4;
        
        fp16x4_t v_b = {0, 0, 0, 0};
        if (b_col < N && b_row + 3 < K) {
            v_b[0] = B[b_col * stride_b + b_row + 0];
            v_b[1] = B[b_col * stride_b + b_row + 1];
            v_b[2] = B[b_col * stride_b + b_row + 2];
            v_b[3] = B[b_col * stride_b + b_row + 3];
        }
        
        v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_b, v_a, v_c, 0, 0, 0);
    }
    
    fp16x16_t v_c_f16;
    for(int i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }
    
    int col_id_c = (tid / 32) * 8;
    int row_id_c = tid % 32;
    int global_row = block_m + row_id_c;
    int global_col = block_n + col_id_c;
    
    if (global_row < M && global_col + 15 < N) {
        for(int i = 0; i < 2; i++) {
            int col_offset = i * 16;
            fp16x8_t tmp;
            for(int j = 0; j < 8; j++) {
                tmp[j] = v_c_f16[i * 8 + j];
            }
            *reinterpret_cast<fp16x8_t*>(&C[global_row * stride_c + global_col + col_offset]) = tmp;
        }
    } else if (global_row < M && global_col < N) {
        for(int i = 0; i < 16; i++) {
            int col = global_col + (i % 8) + (i / 8) * 16;
            if (col < N) {
                C[global_row * stride_c + col] = v_c_f16[i];
            }
        }
    }
}

// ============================================================================
// 高效 Preshuffle Kernel: 完全并行化
// ============================================================================

__global__ void preshuffle_weight_b_fast(
    const fp16_t* __restrict__ B_orig,  // [N×K] ColumnMajor
    fp16_t* __restrict__ B_shuffled,
    int N, int K)
{
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_K = 8;
    constexpr int THREADS = 64;
    
    int n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    int k_blocks = (K + BLOCK_K - 1) / BLOCK_K;
    
    // 每个线程直接处理它对应的元素
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads_needed = n_blocks * k_blocks * THREADS;
    
    // 网格跨步循环
    for (int idx = global_tid; idx < total_threads_needed; idx += gridDim.x * blockDim.x) {
        // 解码：这是哪个 block 的哪个线程
        int block_id = idx / THREADS;
        int tid = idx % THREADS;
        
        int n_block = block_id / k_blocks;
        int k_block = block_id % k_blocks;
        
        int n_base = n_block * BLOCK_N;
        int k_base = k_block * BLOCK_K;
        
        // 计算这个线程要访问的原始位置
        int row_group_id_b = (tid % 32) / 4;
        int row_id_b = (tid % 4) + 
                      (row_group_id_b % 2) * 8 + 
                      ((row_group_id_b % 4) / 2) * 4 + 
                      (row_group_id_b / 4) * 16;
        
        int k_group = tid / 32;
        
        int n_orig = n_base + row_id_b;
        int k_orig = k_base + k_group * 4;
        
        // 计算新位置
        int new_offset = n_block * (k_blocks * BLOCK_N * BLOCK_K) +
                        k_block * (BLOCK_N * BLOCK_K) +
                        tid * 4;
        
        // 向量化复制 4 个元素
        if (n_orig < N && k_orig + 3 < K) {
            fp16x4_t data;
            data[0] = B_orig[n_orig * K + k_orig + 0];
            data[1] = B_orig[n_orig * K + k_orig + 1];
            data[2] = B_orig[n_orig * K + k_orig + 2];
            data[3] = B_orig[n_orig * K + k_orig + 3];
            *reinterpret_cast<fp16x4_t*>(&B_shuffled[new_offset]) = data;
        } else {
            // 边界情况
            for (int i = 0; i < 4; i++) {
                if (n_orig < N && k_orig + i < K) {
                    B_shuffled[new_offset + i] = B_orig[n_orig * K + k_orig + i];
                } else {
                    B_shuffled[new_offset + i] = 0;
                }
            }
        }
    }
}

// ============================================================================
// Preshuffle GEMM: 连续加载
// ============================================================================

__global__ void gemm_mfma_preshuffle(
    const fp16_t* __restrict__ A,
    const fp16_t* __restrict__ B_shuffled,
    fp16_t* __restrict__ C,
    int M, int N, int K,
    int stride_a, int stride_c)
{
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 8;
    
    int block_m = blockIdx.x * MFMA_M;
    int block_n = blockIdx.y * MFMA_N;
    
    if (threadIdx.x >= 64) return;
    int tid = threadIdx.x;
    
    fp32x16_t v_c = {0.0f};
    
    int n_block = block_n / MFMA_N;
    int k_blocks = K / MFMA_K;
    
    for (int k = 0; k < K; k += MFMA_K) {
        // Load A (same as baseline)
        int a_row = block_m + (tid % 32);
        int a_col = k + (tid / 32) * 4;
        
        fp16x4_t v_a = {0, 0, 0, 0};
        if (a_row < M && a_col + 3 < K) {
            v_a[0] = A[a_row * stride_a + a_col + 0];
            v_a[1] = A[a_row * stride_a + a_col + 1];
            v_a[2] = A[a_row * stride_a + a_col + 2];
            v_a[3] = A[a_row * stride_a + a_col + 3];
        }
        
        // Load B: 每个线程的数据是连续存储的
        int k_block = k / MFMA_K;
        int b_offset = n_block * (k_blocks * MFMA_N * MFMA_K) +
                      k_block * (MFMA_N * MFMA_K) +
                      tid * 4;
        
        fp16x4_t v_b;
        if (b_offset + 3 < N * K) {
            v_b = *reinterpret_cast<const fp16x4_t*>(&B_shuffled[b_offset]);
        } else {
            v_b = {0, 0, 0, 0};
        }
        
        v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_b, v_a, v_c, 0, 0, 0);
    }
    
    // Write back (same as baseline)
    fp16x16_t v_c_f16;
    for(int i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }
    
    int col_id_c = (tid / 32) * 8;
    int row_id_c = tid % 32;
    int global_row = block_m + row_id_c;
    int global_col = block_n + col_id_c;
    
    if (global_row < M && global_col + 15 < N) {
        for(int i = 0; i < 2; i++) {
            int col_offset = i * 16;
            fp16x8_t tmp;
            for(int j = 0; j < 8; j++) {
                tmp[j] = v_c_f16[i * 8 + j];
            }
            *reinterpret_cast<fp16x8_t*>(&C[global_row * stride_c + global_col + col_offset]) = tmp;
        }
    } else if (global_row < M && global_col < N) {
        for(int i = 0; i < 16; i++) {
            int col = global_col + (i % 8) + (i / 8) * 16;
            if (col < N) {
                C[global_row * stride_c + col] = v_c_f16[i];
            }
        }
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

void rand_vector_2d(float* v, int row, int col, int ld, 
                    float min_v = 0, float max_v = 1) {
    static int flag = 0;
    if(!flag) { srand(time(NULL)); flag = 1; }
    for(int r = 0; r < row; r++) {
        for(int c = 0; c < col; c++) {
            float tmp = float(std::rand()) / float(RAND_MAX);
            v[r*ld+c] = min_v + tmp * (max_v - min_v);
        }
    }
}

void gemm_rcr_cpu(const float* a, const float* b, float* c,
                  int m, int n, int k, int lda, int ldb, int ldc) {
    for(int im = 0; im < m; im++) {
        for(int in = 0; in < n; in++) {
            float acc = 0.0f;
            for(int ik = 0; ik < k; ik++) {
                acc += a[im * lda + ik] * b[in * ldb + ik];
            }
            c[im * ldc + in] = acc;
        }
    }
}

bool validate(const float* ref, const fp16_t* target, 
              int m, int n, int ldc, float threshold = 5e-3) {
    int errors = 0;
    for(int i = 0; i < m * n; i++) {
        float ref_val = ref[i];
        float target_val = static_cast<float>(target[i]);
        float abs_diff = std::abs(ref_val - target_val);
        float rel_diff = (std::abs(ref_val) > 1e-5) ? abs_diff / std::abs(ref_val) : abs_diff;
        
        if(rel_diff > threshold && abs_diff > 0.01) {
            errors++;
        }
    }
    
    if(errors > 0) {
        printf("  Errors: %d / %d (%.2f%%)\n", errors, m * n, 100.0f * errors / (m * n));
        return false;
    }
    return true;
}

// ============================================================================
// 性能对比测试
// ============================================================================

void benchmark_gemm(int m, int n, int k) {
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Preshuffle Weight GEMM Benchmark                             ║\n");
    printf("║  Problem: M=%d, N=%d, K=%d                               %s║\n", 
           m, n, k, (m < 1000) ? " " : "");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    int lda = k, ldb = k, ldc = n;
    
    // 准备数据
    float* host_a = (float*)malloc(m * lda * sizeof(float));
    float* host_b = (float*)malloc(n * ldb * sizeof(float));
    float* host_c = (float*)malloc(m * ldc * sizeof(float));
    
    rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
    rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
    gemm_rcr_cpu(host_a, host_b, host_c, m, n, k, lda, ldb, ldc);
    
    fp16_t* fp16_a = (fp16_t*)malloc(m * lda * sizeof(fp16_t));
    fp16_t* fp16_b = (fp16_t*)malloc(n * ldb * sizeof(fp16_t));
    fp16_t* fp16_c = (fp16_t*)malloc(m * ldc * sizeof(fp16_t));
    
    for(int i = 0; i < m * lda; i++) fp16_a[i] = __float2half_rn(host_a[i]);
    for(int i = 0; i < n * ldb; i++) fp16_b[i] = __float2half_rn(host_b[i]);
    
    // 设备内存
    fp16_t *dev_a, *dev_b, *dev_b_shuffled, *dev_c;
    HIP_CALL(hipMalloc(&dev_a, m * lda * sizeof(fp16_t)));
    HIP_CALL(hipMalloc(&dev_b, n * ldb * sizeof(fp16_t)));
    HIP_CALL(hipMalloc(&dev_b_shuffled, n * ldb * sizeof(fp16_t)));
    HIP_CALL(hipMalloc(&dev_c, m * ldc * sizeof(fp16_t)));
    
    HIP_CALL(hipMemcpy(dev_a, fp16_a, m * lda * sizeof(fp16_t), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, n * ldb * sizeof(fp16_t), hipMemcpyHostToDevice));
    
    dim3 grid((m + 31) / 32, (n + 31) / 32);
    dim3 block(64);
    
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));
    
    // ===== Baseline =====
    printf("Testing Baseline GEMM...\n");
    
    for(int i = 0; i < 10; i++) {
        gemm_mfma_baseline<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipDeviceSynchronize());
    
    int repeat = 100;
    HIP_CALL(hipEventRecord(start));
    for(int i = 0; i < repeat; i++) {
        gemm_mfma_baseline<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipEventRecord(stop));
    HIP_CALL(hipEventSynchronize(stop));
    
    float baseline_time;
    HIP_CALL(hipEventElapsedTime(&baseline_time, start, stop));
    baseline_time /= repeat;
    
    HIP_CALL(hipMemcpy(fp16_c, dev_c, m * ldc * sizeof(fp16_t), hipMemcpyDeviceToHost));
    bool baseline_correct = validate(host_c, fp16_c, m, n, ldc);
    
    printf("  Time: %.3f us, Correct: %s\n", baseline_time * 1000, 
           baseline_correct ? "✅" : "❌");
    
    // ===== Preshuffle =====
    printf("\nTesting Preshuffle GEMM...\n");
    
    int n_blocks = (n + 31) / 32;
    int k_blocks = (k + 7) / 8;
    int total_threads_needed = n_blocks * k_blocks * 64;
    
    // 使用更多的线程和块来提高并行度
    dim3 pre_block(256);
    dim3 pre_grid((total_threads_needed + pre_block.x - 1) / pre_block.x);
    // 限制最大 grid 大小
    if (pre_grid.x > 65535) pre_grid.x = 65535;
    
    printf("  Preshuffling... (grid: %d, block: %d, total_threads: %d)\n", 
           pre_grid.x, pre_block.x, total_threads_needed);
    
    // 计时 preshuffle
    hipEvent_t pre_start, pre_stop;
    HIP_CALL(hipEventCreate(&pre_start));
    HIP_CALL(hipEventCreate(&pre_stop));
    
    HIP_CALL(hipEventRecord(pre_start));
    preshuffle_weight_b_fast<<<pre_grid, pre_block>>>(dev_b, dev_b_shuffled, n, k);
    HIP_CALL(hipEventRecord(pre_stop));
    HIP_CALL(hipEventSynchronize(pre_stop));
    
    float preshuffle_time_ms;
    HIP_CALL(hipEventElapsedTime(&preshuffle_time_ms, pre_start, pre_stop));
    printf("  Preshuffle time: %.3f us\n", preshuffle_time_ms * 1000);
    
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("  ❌ Preshuffle failed: %s\n", hipGetErrorString(err));
        HIP_CALL(hipEventDestroy(pre_start));
        HIP_CALL(hipEventDestroy(pre_stop));
        free(host_a); free(host_b); free(host_c);
        free(fp16_a); free(fp16_b); free(fp16_c);
        HIP_CALL(hipFree(dev_a));
        HIP_CALL(hipFree(dev_b));
        HIP_CALL(hipFree(dev_b_shuffled));
        HIP_CALL(hipFree(dev_c));
        HIP_CALL(hipEventDestroy(start));
        HIP_CALL(hipEventDestroy(stop));
        return;
    }
    
    HIP_CALL(hipEventDestroy(pre_start));
    HIP_CALL(hipEventDestroy(pre_stop));
    
    // Warmup GEMM
    for(int i = 0; i < 10; i++) {
        gemm_mfma_preshuffle<<<grid, block>>>(dev_a, dev_b_shuffled, dev_c, m, n, k, lda, ldc);
    }
    HIP_CALL(hipDeviceSynchronize());
    
    err = hipGetLastError();
    if (err != hipSuccess) {
        printf("  ❌ GEMM failed: %s\n", hipGetErrorString(err));
        free(host_a); free(host_b); free(host_c);
        free(fp16_a); free(fp16_b); free(fp16_c);
        HIP_CALL(hipFree(dev_a));
        HIP_CALL(hipFree(dev_b));
        HIP_CALL(hipFree(dev_b_shuffled));
        HIP_CALL(hipFree(dev_c));
        HIP_CALL(hipEventDestroy(start));
        HIP_CALL(hipEventDestroy(stop));
        return;
    }
    
    // Benchmark GEMM
    HIP_CALL(hipEventRecord(start));
    for(int i = 0; i < repeat; i++) {
        gemm_mfma_preshuffle<<<grid, block>>>(dev_a, dev_b_shuffled, dev_c, m, n, k, lda, ldc);
    }
    HIP_CALL(hipEventRecord(stop));
    HIP_CALL(hipEventSynchronize(stop));
    
    float preshuffle_gemm_time;
    HIP_CALL(hipEventElapsedTime(&preshuffle_gemm_time, start, stop));
    preshuffle_gemm_time /= repeat;
    
    HIP_CALL(hipMemcpy(fp16_c, dev_c, m * ldc * sizeof(fp16_t), hipMemcpyDeviceToHost));
    bool preshuffle_correct = validate(host_c, fp16_c, m, n, ldc);
    
    printf("  GEMM time: %.3f us, Correct: %s\n", preshuffle_gemm_time * 1000, 
           preshuffle_correct ? "✅" : "❌");
    
    // ===== 结果 =====
    double gflops = 2.0 * m * n * k / 1e6;
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Performance Summary                                          ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Baseline GEMM:     %8.3f us  (%7.2f GFLOPS)          ║\n", 
           baseline_time * 1000, gflops / baseline_time);
    printf("║  Preshuffle (once): %8.3f us  (one-time cost)          ║\n", 
           preshuffle_time_ms * 1000);
    printf("║  Preshuffle GEMM:   %8.3f us  (%7.2f GFLOPS)          ║\n", 
           preshuffle_gemm_time * 1000, gflops / preshuffle_gemm_time);
    printf("║  Speedup:           %8.2fx                                ║\n", 
           baseline_time / preshuffle_gemm_time);
    printf("║  Break-even after:  %8.1f calls                        ║\n", 
           preshuffle_time_ms / (baseline_time - preshuffle_gemm_time));
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    // 清理
    free(host_a); free(host_b); free(host_c);
    free(fp16_a); free(fp16_b); free(fp16_c);
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_b_shuffled));
    HIP_CALL(hipFree(dev_c));
    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Preshuffle Weight GEMM - Optimized Implementation           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    benchmark_gemm(8192, 64, 256);
    // benchmark_gemm(4096, 64, 256);
    // benchmark_gemm(2048, 32, 128);
    // benchmark_gemm(1024, 32, 64);
    
    return 0;
}