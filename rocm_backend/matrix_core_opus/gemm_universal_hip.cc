// ============================================================================
// 文件: gemm_universal_hip.cc
// 功能: CK-style GEMM Universal 的简化 HIP 实现
// ============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// ----------------------------------------------------------------------------
// 1. 类型定义
// ----------------------------------------------------------------------------
using fp16_t = _Float16;
using fp32_t = float;

// MFMA 结果类型 (32x32x8 产生 16 个 FP32)
typedef fp32_t fp32x16_t __attribute__((ext_vector_type(16)));
typedef fp16_t fp16x4_t __attribute__((ext_vector_type(4)));

// ----------------------------------------------------------------------------
// 2. GEMM 配置 (类似 CK 的 GemmConfigComputeV3)
// ----------------------------------------------------------------------------
template<typename PrecType>
struct GemmConfig {
    // Tile 尺寸
    static constexpr int M_TILE = 128;  // Block 在 M 维度处理的行数
    static constexpr int N_TILE = 64;   // Block 在 N 维度处理的列数
    static constexpr int K_TILE = 128;  // 每次迭代处理的 K
    
    // Wave 配置
    static constexpr int M_WARP = 4;    // M 方向的 wave 数量
    static constexpr int N_WARP = 1;    // N 方向的 wave 数量
    static constexpr int WAVES_PER_BLOCK = M_WARP * N_WARP;
    static constexpr int THREADS_PER_BLOCK = WAVES_PER_BLOCK * 64;
    
    // MFMA 配置 (32x32x8)
    static constexpr int M_MFMA = 32;
    static constexpr int N_MFMA = 32;
    static constexpr int K_MFMA = 8;
    
    // 每个 wave 的 MFMA 数量
    static constexpr int M_MFMA_PER_WAVE = M_TILE / M_WARP / M_MFMA;  // 1
    static constexpr int N_MFMA_PER_WAVE = N_TILE / N_WARP / N_MFMA;  // 2
    
    // LDS 配置
    static constexpr bool USE_LDS = true;
    static constexpr bool DOUBLE_BUFFER = false;
};

// ----------------------------------------------------------------------------
// 3. Pipeline - AgBgCrCompV3 (A/B Global, C Register, Compute V3)
// ----------------------------------------------------------------------------

// LDS 布局: [M_TILE][K_TILE] 和 [K_TILE][N_TILE]
// Padding 以避免 bank conflict
template<typename Config>
struct SharedMemory {
    static constexpr int LDS_A_STRIDE = Config::K_TILE + 8;  // Padding
    static constexpr int LDS_B_STRIDE = Config::N_TILE + 8;
    
    __device__ static int get_lds_a_offset(int m, int k) {
        return m * LDS_A_STRIDE + k;
    }
    
    __device__ static int get_lds_b_offset(int k, int n) {
        return k * LDS_B_STRIDE + n;
    }
};

// ----------------------------------------------------------------------------
// 4. 核心 GEMM Kernel (简化版 CK Universal Pipeline)
// ----------------------------------------------------------------------------

template<typename Config>
__global__ void gemm_universal_kernel(
    const fp16_t* __restrict__ A,  // [M, K] RowMajor
    const fp16_t* __restrict__ B,  // [K, N] ColumnMajor
    fp32_t* __restrict__ C,        // [M, N] RowMajor
    int M, int N, int K)
{
    // Block/Wave ID
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int wave_id = threadIdx.x / 64;
    const int lane_id = threadIdx.x % 64;
    
    // Wave 在 block 内的位置
    const int wave_m = wave_id / Config::N_WARP;
    const int wave_n = wave_id % Config::N_WARP;
    
    // 全局坐标
    const int global_m = block_m * Config::M_TILE + wave_m * Config::M_MFMA;
    const int global_n = block_n * Config::N_TILE + wave_n * Config::N_MFMA * Config::N_MFMA_PER_WAVE;
    
    // 边界检查
    if (global_m >= M) return;
    
    // LDS 分配
    __shared__ fp16_t smem_a[Config::M_TILE * (Config::K_TILE + 8)];
    __shared__ fp16_t smem_b[Config::K_TILE * (Config::N_TILE + 8)];
    
    // 累加器 (每个 wave 处理 1x2 个 MFMA)
    fp32x16_t acc[Config::M_MFMA_PER_WAVE][Config::N_MFMA_PER_WAVE];
    for (int i = 0; i < Config::M_MFMA_PER_WAVE; i++) {
        for (int j = 0; j < Config::N_MFMA_PER_WAVE; j++) {
            acc[i][j] = {0};
        }
    }
    
    // --------------------------------------------------------------------
    // K-Loop: 迭代 K 维度
    // --------------------------------------------------------------------
    const int num_k_tiles = (K + Config::K_TILE - 1) / Config::K_TILE;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_start = k_tile * Config::K_TILE;
        const int k_end = min(k_start + Config::K_TILE, K);
        const int k_size = k_end - k_start;
        
        // ================================================================
        // Stage 1: Cooperative Load A/B into LDS
        // ================================================================
        
        // 每个线程负责加载多个元素 (向量化)
        const int total_threads = Config::THREADS_PER_BLOCK;
        const int elements_per_thread = 4;  // 使用 float4 向量化
        
        // Load A: [M_TILE, K_TILE]
        {
            const int total_elements = Config::M_TILE * k_size;
            const int loads_per_thread = (total_elements + total_threads * elements_per_thread - 1) 
                                       / (total_threads * elements_per_thread);
            
            for (int i = 0; i < loads_per_thread; i++) {
                const int idx = (i * total_threads + threadIdx.x) * elements_per_thread;
                if (idx + elements_per_thread <= total_elements) {
                    const int m = idx / k_size;
                    const int k = idx % k_size;
                    const int global_m_offset = block_m * Config::M_TILE + m;
                    
                    if (global_m_offset < M) {
                        // 向量化加载 (4 个 FP16)
                        const fp16_t* src = &A[global_m_offset * K + k_start + k];
                        fp16_t* dst = &smem_a[m * (Config::K_TILE + 8) + k];
                        
                        if (k + 4 <= k_size) {
                            *reinterpret_cast<fp16x4_t*>(dst) = 
                                *reinterpret_cast<const fp16x4_t*>(src);
                        } else {
                            // 处理边界
                            for (int e = 0; e < 4 && k + e < k_size; e++) {
                                dst[e] = src[e];
                            }
                        }
                    }
                }
            }
        }
        
        // Load B: [K_TILE, N_TILE] (B 是 ColumnMajor)
        {
            const int total_elements = k_size * Config::N_TILE;
            const int loads_per_thread = (total_elements + total_threads * elements_per_thread - 1) 
                                       / (total_threads * elements_per_thread);
            
            for (int i = 0; i < loads_per_thread; i++) {
                const int idx = (i * total_threads + threadIdx.x) * elements_per_thread;
                if (idx + elements_per_thread <= total_elements) {
                    const int k = idx / Config::N_TILE;
                    const int n = idx % Config::N_TILE;
                    const int global_n_offset = block_n * Config::N_TILE + n;
                    
                    if (global_n_offset < N) {
                        // B 是 ColumnMajor: B[k][n] = B[global_n_offset * K + (k_start + k)]
                        const fp16_t* src = &B[global_n_offset * K + k_start + k];
                        fp16_t* dst = &smem_b[k * (Config::N_TILE + 8) + n];
                        
                        // 非连续访问，逐个加载
                        for (int e = 0; e < 4 && k + e < k_size; e++) {
                            dst[e * (Config::N_TILE + 8)] = src[e];
                        }
                    }
                }
            }
        }
        
        __syncthreads();
        
        // ================================================================
        // Stage 2: MFMA Computation
        // ================================================================
        
        // 对于 K_TILE 内的每个 K_MFMA 块
        const int num_k_mfma = k_size / Config::K_MFMA;
        
        for (int k_mfma = 0; k_mfma < num_k_mfma; k_mfma++) {
            const int k_offset = k_mfma * Config::K_MFMA;
            
            // 对于每个 MFMA tile (1x2)
            for (int i = 0; i < Config::M_MFMA_PER_WAVE; i++) {
                for (int j = 0; j < Config::N_MFMA_PER_WAVE; j++) {
                    // 从 LDS 加载 A 和 B 片段
                    const int m_base = wave_m * Config::M_MFMA + i * Config::M_MFMA;
                    const int n_base = wave_n * Config::N_MFMA * Config::N_MFMA_PER_WAVE + j * Config::N_MFMA;
                    
                    // A fragment: [32, 8] 从 LDS 加载
                    fp16x4_t a_frag;
                    {
                        const int m_local = m_base + (lane_id / 2) % 32;
                        const int k_local = k_offset + (lane_id % 2) * 4;
                        a_frag = *reinterpret_cast<fp16x4_t*>(
                            &smem_a[m_local * (Config::K_TILE + 8) + k_local]);
                    }
                    
                    // B fragment: [8, 32] 从 LDS 加载
                    fp16x4_t b_frag;
                    {
                        const int k_local = k_offset + (lane_id / 16);
                        const int n_local = n_base + (lane_id % 16) * 2;
                        const int k_stride = Config::N_TILE + 8;
                        
                        // 手动收集 (非连续)
                        b_frag[0] = smem_b[k_local * k_stride + n_local];
                        b_frag[1] = smem_b[k_local * k_stride + n_local + 1];
                        b_frag[2] = smem_b[(k_local + 1) * k_stride + n_local];
                        b_frag[3] = smem_b[(k_local + 1) * k_stride + n_local + 1];
                    }
                    
                    // MFMA 指令
                    acc[i][j] = __builtin_amdgcn_mfma_f32_32x32x8f16(
                        a_frag, b_frag, acc[i][j], 0, 0, 0);
                }
            }
        }
        
        __syncthreads();
    }
    
    // ====================================================================
    // Stage 3: Epilogue - Write C
    // ====================================================================
    
    for (int i = 0; i < Config::M_MFMA_PER_WAVE; i++) {
        for (int j = 0; j < Config::N_MFMA_PER_WAVE; j++) {
            const int m_base = global_m + i * Config::M_MFMA;
            const int n_base = global_n + j * Config::N_MFMA;
            
            // MFMA 32x32x8 输出布局: 每个 lane 产生 16 个结果
            // 分布在 32x32 的 tile 中
            for (int e = 0; e < 16; e++) {
                const int m_offset = (e / 4) * 8 + (lane_id / 8) % 4;
                const int n_offset = (e % 4) * 8 + (lane_id % 8);
                
                const int m_global = m_base + m_offset;
                const int n_global = n_base + n_offset;
                
                if (m_global < M && n_global < N) {
                    C[m_global * N + n_global] = acc[i][j][e];
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 5. Host 函数和工具
// ----------------------------------------------------------------------------

// 随机初始化
void rand_vector_2d(std::vector<fp16_t>& vec, int rows, int cols) {
    for (auto& v : vec) {
        v = static_cast<fp16_t>((rand() % 2000 - 1000) / 1000.0f);
    }
}

// CPU 参考实现
void gemm_cpu(const std::vector<fp16_t>& A, const std::vector<fp16_t>& B,
              std::vector<fp32_t>& C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            fp32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += static_cast<fp32_t>(A[m * K + k]) * 
                       static_cast<fp32_t>(B[n * K + k]);  // B is ColumnMajor
            }
            C[m * N + n] = sum;
        }
    }
}

// 验证
bool validate(const std::vector<fp32_t>& C_gpu, const std::vector<fp32_t>& C_cpu, 
              int M, int N, float threshold = 1e-2) {
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(C_gpu[i] - C_cpu[i]);
        float rel_err = diff / (std::abs(C_cpu[i]) + 1e-6);
        if (rel_err > threshold) {
            if (errors < 10) {
                printf("  Mismatch at %d: GPU=%.6f, CPU=%.6f, RelErr=%.6f\n",
                       i, C_gpu[i], C_cpu[i], rel_err);
            }
            errors++;
        }
    }
    return errors == 0;
}

// ----------------------------------------------------------------------------
// 6. Main 测试函数
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    using Config = GemmConfig<fp16_t>;
    
    // 问题规模
    const int M = (argc > 1) ? atoi(argv[1]) : 8192;
    const int N = (argc > 2) ? atoi(argv[2]) : 64;
    const int K = (argc > 3) ? atoi(argv[3]) : 256;
    
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  GEMM Universal HIP Implementation                   ║\n");
    printf("║  Problem: M=%d, N=%d, K=%d                      ║\n", M, N, K);
    printf("║  Config: Tile=%dx%d, Waves=%dx%d                    ║\n",
           Config::M_TILE, Config::N_TILE, Config::M_WARP, Config::N_WARP);
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    // 分配和初始化
    std::vector<fp16_t> h_A(M * K), h_B(N * K);
    std::vector<fp32_t> h_C_gpu(M * N), h_C_cpu(M * N);
    
    rand_vector_2d(h_A, M, K);
    rand_vector_2d(h_B, N, K);
    
    // 设备内存
    fp16_t *d_A, *d_B;
    fp32_t *d_C;
    hipMalloc(&d_A, M * K * sizeof(fp16_t));
    hipMalloc(&d_B, N * K * sizeof(fp16_t));
    hipMalloc(&d_C, M * N * sizeof(fp32_t));
    
    hipMemcpy(d_A, h_A.data(), M * K * sizeof(fp16_t), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), N * K * sizeof(fp16_t), hipMemcpyHostToDevice);
    
    // Grid/Block 配置
    dim3 block(Config::THREADS_PER_BLOCK);
    dim3 grid((M + Config::M_TILE - 1) / Config::M_TILE,
              (N + Config::N_TILE - 1) / Config::N_TILE);
    
    printf("Launch: grid=(%d,%d), block=%d\n\n", grid.x, grid.y, block.x);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        gemm_universal_kernel<Config><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    hipDeviceSynchronize();
    
    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    for (int i = 0; i < 100; i++) {
        gemm_universal_kernel<Config><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    ms /= 100;
    
    // 计算性能
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    
    printf("Performance:\n");
    printf("  Time: %.3f us\n", ms * 1000);
    printf("  GFLOPS: %.2f\n\n", gflops);
    
    // 验证
    hipMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(fp32_t), hipMemcpyDeviceToHost);
    
    printf("Running CPU reference...\n");
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K);
    
    bool correct = validate(h_C_gpu, h_C_cpu, M, N);
    printf("Validation: %s\n", correct ? "✅ PASS" : "❌ FAIL");
    
    // 清理
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipEventDestroy(start);
    hipEventDestroy(stop);
    
    return correct ? 0 : 1;
}