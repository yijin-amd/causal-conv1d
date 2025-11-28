/*
 * HIP Kernel性能对比测试
 * 对比两个kernel的性能:
 * 1. causal_conv1d_fwd_kernel (channel-first布局)
 * 2. causal_conv1d_channellast_fwd_kernel (channel-last布局)
 * 
 * 测试不同的配置参数: batch, dim, seqlen, width, silu
 * 
 * 注意事项:
 * - 当前kernel实现仅支持width=4
 * - width=2和width=3的测试用例已被注释
 * - 共27个活跃测试配置 (原始35个配置，8个被注释)
 */

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <fstream>

// 包含两个kernel文件 (相对路径会在编译时通过-I指定)
#include "causal_conv1d_kernel.hip"
#include "causal_conv1d_kernel_channellast.hip"

// ==================== Helper Macros ====================

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP错误 " << __FILE__ << ":" << __LINE__ << " - " \
                      << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==================== Constexpr Helpers ====================

template<typename T>
constexpr T const_max(T a, T b) {
    return (a > b) ? a : b;
}

// ==================== Channel-First Kernel Traits ====================

template<int kNThreads_, int kWidth_, int kNElts_>
struct CausalConv1dKernelTraits {
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNElts = kNElts_;
    static constexpr int kChunkSize = kNThreads * kNElts;
    
    using vec_t = float4;  // 4 floats
    
    using BlockLoadT = hipcub::BlockLoad<float, kNThreads, kNElts, hipcub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = hipcub::BlockStore<float, kNThreads, kNElts, hipcub::BLOCK_STORE_WARP_TRANSPOSE>;
    
    static constexpr int kSmemIOSize = const_max(sizeof(typename BlockLoadT::TempStorage), 
                                                  sizeof(typename BlockStoreT::TempStorage));
    static constexpr int kSmemExchangeSize = kNThreads * sizeof(vec_t);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

// ==================== 测试配置结构 ====================

struct TestConfig {
    const char* name;
    int batch;
    int dim;
    int seqlen;
    int width;
    bool use_silu;
};

struct BenchmarkResult {
    std::string kernel_name;
    double time_ms;
    double bandwidth_gb_s;
    double gflops;
    
    BenchmarkResult(const std::string& name, double t, double bw, double gf)
        : kernel_name(name), time_ms(t), bandwidth_gb_s(bw), gflops(gf) {}
};

// ==================== 辅助函数 ====================

double calculate_bandwidth(size_t bytes, double time_ms) {
    return (bytes / 1e9) / (time_ms / 1000.0);
}

double calculate_gflops(size_t num_ops, double time_ms) {
    return (num_ops / 1e9) / (time_ms / 1000.0);
}

// ==================== Channel-First性能测试 ====================

template<int kNThreads, int kWidth, int kNElts>
BenchmarkResult benchmark_channel_first(
    const float* d_x, const float* d_weight, const float* d_bias, float* d_out,
    int batch, int dim, int seqlen, int width, bool use_silu,
    int warmup_iters, int bench_iters
) {
    using Ktraits = CausalConv1dKernelTraits<kNThreads, kWidth, kNElts>;
    
    // 计算grid和block配置
    dim3 grid(batch, dim);
    dim3 block(Ktraits::kNThreads);
    size_t smem_size = Ktraits::kSmemSize;
    
    // 参数设置 (channel-first: [batch, dim, seqlen])
    int x_batch_stride = dim * seqlen;
    int x_c_stride = seqlen;
    int weight_c_stride = width;
    int weight_width_stride = 1;
    int out_batch_stride = dim * seqlen;
    int out_c_stride = seqlen;
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        hipLaunchKernelGGL(
            (causal_conv1d_fwd_kernel<Ktraits>),
            grid, block, smem_size, 0,
            d_x, d_weight, d_bias, d_out,
            batch, dim, seqlen, width,
            x_batch_stride, x_c_stride,
            weight_c_stride, weight_width_stride,
            out_batch_stride, out_c_stride,
            use_silu
        );
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < bench_iters; i++) {
        hipLaunchKernelGGL(
            (causal_conv1d_fwd_kernel<Ktraits>),
            grid, block, smem_size, 0,
            d_x, d_weight, d_bias, d_out,
            batch, dim, seqlen, width,
            x_batch_stride, x_c_stride,
            weight_c_stride, weight_width_stride,
            out_batch_stride, out_c_stride,
            use_silu
        );
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_time_ms = elapsed_ms / bench_iters;
    
    // 计算性能指标
    size_t x_bytes = batch * dim * seqlen * sizeof(float);
    size_t weight_bytes = dim * width * sizeof(float);
    size_t bias_bytes = d_bias ? dim * sizeof(float) : 0;
    size_t out_bytes = batch * dim * seqlen * sizeof(float);
    size_t total_bytes = x_bytes + weight_bytes + bias_bytes + out_bytes;
    
    size_t num_ops = (size_t)batch * dim * seqlen * width * 2; // multiply-add
    
    double bandwidth = calculate_bandwidth(total_bytes, avg_time_ms);
    double gflops = calculate_gflops(num_ops, avg_time_ms);
    
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    return BenchmarkResult("Channel-First", avg_time_ms, bandwidth, gflops);
}

// ==================== Channel-Last性能测试 ====================

template<int kNThreads, int kWidth, int kChunkSizeL>
BenchmarkResult benchmark_channel_last(
    const float* d_x, const float* d_weight, const float* d_bias, float* d_out,
    int batch, int dim, int seqlen, int width, bool use_silu,
    int warmup_iters, int bench_iters
) {
    using Ktraits = Causal_conv1d_channellast_fwd_kernel_traits<kNThreads, kWidth, kChunkSizeL>;
    
    // 设置参数结构
    ConvParamsChannelLast params;
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = use_silu;
    
    // Channel-last strides: [batch, seqlen, dim]
    params.x_batch_stride = seqlen * dim;
    params.x_l_stride = dim;
    params.x_c_stride = 1;
    params.weight_c_stride = width;
    params.weight_width_stride = 1;
    params.out_batch_stride = seqlen * dim;
    params.out_l_stride = dim;
    params.out_c_stride = 1;
    
    params.x_ptr = const_cast<float*>(d_x);
    params.weight_ptr = const_cast<float*>(d_weight);
    params.bias_ptr = const_cast<float*>(d_bias);
    params.out_ptr = d_out;
    
    params.seq_idx_ptr = nullptr;
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = nullptr;
    
    // 计算grid配置
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    const int n_chunks_L = (seqlen + kChunkSizeL - 1) / kChunkSizeL;
    const int n_chunks_C = (dim + kChunkSizeC - 1) / kChunkSizeC;
    
    dim3 grid(batch, n_chunks_L, n_chunks_C);
    dim3 block(kNThreads);
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_fwd_kernel<Ktraits, false>),
            grid, block, 0, 0,
            params
        );
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < bench_iters; i++) {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_fwd_kernel<Ktraits, false>),
            grid, block, 0, 0,
            params
        );
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_time_ms = elapsed_ms / bench_iters;
    
    // 计算性能指标 (与channel-first相同的总字节数)
    size_t x_bytes = batch * dim * seqlen * sizeof(float);
    size_t weight_bytes = dim * width * sizeof(float);
    size_t bias_bytes = d_bias ? dim * sizeof(float) : 0;
    size_t out_bytes = batch * dim * seqlen * sizeof(float);
    size_t total_bytes = x_bytes + weight_bytes + bias_bytes + out_bytes;
    
    size_t num_ops = (size_t)batch * dim * seqlen * width * 2;
    
    double bandwidth = calculate_bandwidth(total_bytes, avg_time_ms);
    double gflops = calculate_gflops(num_ops, avg_time_ms);
    
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    return BenchmarkResult("Channel-Last", avg_time_ms, bandwidth, gflops);
}

// ==================== 数据转换函数 ====================

void convert_channel_first_to_last(
    const float* src, float* dst,
    int batch, int dim, int seqlen
) {
    // src: [batch, dim, seqlen] -> dst: [batch, seqlen, dim]
    for (int b = 0; b < batch; b++) {
        for (int d = 0; d < dim; d++) {
            for (int t = 0; t < seqlen; t++) {
                int src_idx = b * (dim * seqlen) + d * seqlen + t;
                int dst_idx = b * (seqlen * dim) + t * dim + d;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

// ==================== 主测试函数 ====================

void run_comparison_test(const TestConfig& config, int warmup, int iters) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "【测试配置】 " << config.name << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "  参数: batch=" << config.batch << ", dim=" << config.dim 
              << ", seqlen=" << config.seqlen << ", width=" << config.width << std::endl;
    std::cout << "  SiLU激活: " << (config.use_silu ? "是" : "否") << std::endl;
    std::cout << "  预热次数: " << warmup << ", 测试次数: " << iters << std::endl;
    
    int batch = config.batch;
    int dim = config.dim;
    int seqlen = config.seqlen;
    int width = config.width;
    bool use_silu = config.use_silu;
    
    // 分配host内存
    size_t x_size_cf = batch * dim * seqlen;
    size_t x_size_cl = batch * seqlen * dim;
    size_t weight_size = dim * width;
    size_t bias_size = dim;
    
    std::vector<float> h_x_cf(x_size_cf);
    std::vector<float> h_x_cl(x_size_cl);
    std::vector<float> h_weight(weight_size);
    std::vector<float> h_bias(bias_size);
    
    // 初始化数据
    std::random_device rd;
    std::mt19937 gen(42);  // 固定seed保证可重复
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < x_size_cf; i++) {
        h_x_cf[i] = dist(gen);
    }
    for (size_t i = 0; i < weight_size; i++) {
        h_weight[i] = dist(gen);
    }
    for (size_t i = 0; i < bias_size; i++) {
        h_bias[i] = dist(gen);
    }
    
    // 转换为channel-last布局
    convert_channel_first_to_last(h_x_cf.data(), h_x_cl.data(), batch, dim, seqlen);
    
    // 分配device内存
    float *d_x_cf, *d_x_cl, *d_weight, *d_bias, *d_out_cf, *d_out_cl;
    HIP_CHECK(hipMalloc(&d_x_cf, x_size_cf * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x_cl, x_size_cl * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_cf, x_size_cf * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_cl, x_size_cl * sizeof(float)));
    
    // 拷贝数据到device
    HIP_CHECK(hipMemcpy(d_x_cf, h_x_cf.data(), x_size_cf * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x_cl, h_x_cl.data(), x_size_cl * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    
    // ==================== Benchmark Channel-First ====================
    BenchmarkResult result_cf = benchmark_channel_first<128, 4, 4>(
        d_x_cf, d_weight, d_bias, d_out_cf,
        batch, dim, seqlen, width, use_silu,
        warmup, iters
    );
    
    // ==================== Benchmark Channel-Last ====================
    BenchmarkResult result_cl = benchmark_channel_last<256, 4, 64>(
        d_x_cl, d_weight, d_bias, d_out_cl,
        batch, dim, seqlen, width, use_silu,
        warmup, iters
    );
    
    // ==================== 输出结果 ====================
    std::cout << "\n【性能结果】" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(20) << "Kernel" 
              << std::setw(15) << "时间 (ms)" 
              << std::setw(20) << "带宽 (GB/s)" 
              << std::setw(20) << "算力 (GFLOPS)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(20) << result_cf.kernel_name
              << std::setw(15) << result_cf.time_ms
              << std::setw(20) << result_cf.bandwidth_gb_s
              << std::setw(20) << result_cf.gflops << std::endl;
    
    std::cout << std::setw(20) << result_cl.kernel_name
              << std::setw(15) << result_cl.time_ms
              << std::setw(20) << result_cl.bandwidth_gb_s
              << std::setw(20) << result_cl.gflops << std::endl;
    
    std::cout << std::string(80, '-') << std::endl;
    
    // 对比分析
    double speedup = result_cf.time_ms / result_cl.time_ms;
    std::cout << "\n【对比分析】" << std::endl;
    if (speedup > 1.0) {
        std::cout << "  ✓ Channel-Last 更快: " << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "  时间节省: " << std::setprecision(4) << (result_cf.time_ms - result_cl.time_ms) 
                  << " ms (" << std::setprecision(1) << ((result_cf.time_ms - result_cl.time_ms) / result_cf.time_ms * 100) 
                  << "%)" << std::endl;
    } else {
        std::cout << "  ✓ Channel-First 更快: " << std::setprecision(2) << (1.0 / speedup) << "x" << std::endl;
        std::cout << "  时间节省: " << std::setprecision(4) << (result_cl.time_ms - result_cf.time_ms) 
                  << " ms (" << std::setprecision(1) << ((result_cl.time_ms - result_cf.time_ms) / result_cl.time_ms * 100) 
                  << "%)" << std::endl;
    }
    
    // 清理
    HIP_CHECK(hipFree(d_x_cf));
    HIP_CHECK(hipFree(d_x_cl));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_out_cf));
    HIP_CHECK(hipFree(d_out_cl));
}

// ==================== Main函数 ====================

int main(int argc, char** argv) {
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        HIP Kernel 性能对比测试 - AMD ROCm                                  ║" << std::endl;
    std::cout << "║  对比 causal_conv1d_fwd_kernel vs causal_conv1d_channellast_fwd_kernel   ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    // 获取GPU信息
    int device = 0;
    HIP_CHECK(hipSetDevice(device));
    
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    
    std::cout << "\n【GPU信息】" << std::endl;
    std::cout << "  设备名称: " << prop.name << std::endl;
    std::cout << "  计算能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  全局内存: " << std::fixed << std::setprecision(2) << (prop.totalGlobalMem / 1e9) << " GB" << std::endl;
    std::cout << "  最大线程数/Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Wavefront大小: " << prop.warpSize << std::endl;
    
    // 测试参数
    const int warmup = 10;
    const int iters = 100;
    
    // 测试配置列表 - 与test_cuda_fwd_kernels.cu完全一致
    // 注意: width=2和width=3的用例暂时注释，因为当前kernel实现仅支持width=4
    std::vector<TestConfig> configs = {
        // ========== dim=64 测试 (小规模, 覆盖所有width和seqlen范围) ==========
        {"dim64-seq128-w4", 2, 64, 128, 4, false},   // 极小
        {"dim64-seq512-w4", 2, 64, 512, 4, false},   // 小
        {"dim64-seq512-w4-silu", 2, 64, 512, 4, true},    // 小+SiLU
        {"dim64-seq1024-w4", 2, 64, 1024, 4, false},  // 中
        {"dim64-seq1024-w4-silu", 2, 64, 1024, 4, true},   // 中+SiLU
        // {"dim64-seq2048-w2", 2, 64, 2048, 2, false},  // 中大+width=2 [暂不支持width=2]
        // {"dim64-seq2048-w3", 2, 64, 2048, 3, false},  // 中大+width=3 [暂不支持width=3]
        {"dim64-seq2048-w4", 2, 64, 2048, 4, false},  // 中大+width=4
        
        // ========== dim=256 测试 (中等规模, 不同seqlen+width) ==========
        {"dim256-seq256-w4", 4, 256, 256, 4, false},  // 短seq
        {"dim256-seq512-w4", 4, 256, 512, 4, false},  // 中短seq
        {"dim256-seq512-w4-silu", 4, 256, 512, 4, true},   // 中短seq+SiLU
        {"dim256-seq1024-w4", 4, 256, 1024, 4, false}, // 中seq
        {"dim256-seq1024-w4-silu", 4, 256, 1024, 4, true},  // 中seq+SiLU
        // {"dim256-seq2048-w2", 4, 256, 2048, 2, false}, // 长seq+width=2 [暂不支持width=2]
        // {"dim256-seq2048-w3", 4, 256, 2048, 3, false}, // 长seq+width=3 [暂不支持width=3]
        {"dim256-seq2048-w4", 4, 256, 2048, 4, false}, // 长seq+width=4
        
        // ========== dim=512 测试 (中大规模, 关键性能区) ==========
        {"dim512-seq512-w4", 4, 512, 512, 4, false},  // 短seq
        {"dim512-seq1024-w4", 4, 512, 1024, 4, false}, // 中seq
        {"dim512-seq1024-w4-silu", 4, 512, 1024, 4, true},  // 中seq+SiLU
        {"dim512-seq2048-w4", 4, 512, 2048, 4, false}, // 长seq (峰值性能点)
        {"dim512-seq2048-w4-silu", 4, 512, 2048, 4, true},  // 长seq+SiLU
        // {"dim512-seq2048-w2", 4, 512, 2048, 2, false}, // 长seq+width=2 [暂不支持width=2]
        // {"dim512-seq2048-w3-silu", 4, 512, 2048, 3, true},  // 长seq+width=3+SiLU [暂不支持width=3]
        
        // ========== dim=1024 测试 (大规模) ==========
        {"dim1024-seq512-w4", 8, 1024, 512, 4, false},  // 短seq
        {"dim1024-seq1024-w4", 8, 1024, 1024, 4, false}, // 中seq
        {"dim1024-seq1024-w4-silu", 8, 1024, 1024, 4, true},  // 中seq+SiLU
        {"dim1024-seq2048-w4", 8, 1024, 2048, 4, false}, // 长seq
        {"dim1024-seq2048-w4-silu", 8, 1024, 2048, 4, true},  // 长seq+SiLU
        {"dim1024-seq4096-w4", 8, 1024, 4096, 4, false}, // 超长seq
        
        // ========== dim=2048 测试 (超大规模) ==========
        {"dim2048-seq1024-w4", 8, 2048, 1024, 4, false}, // 中seq
        {"dim2048-seq2048-w4", 8, 2048, 2048, 4, false}, // 长seq
        {"dim2048-seq2048-w4-silu", 8, 2048, 2048, 4, true},  // 长seq+SiLU
        {"dim2048-seq4096-w4", 8, 2048, 4096, 4, false}, // 超长seq
        // {"dim2048-seq2048-w2", 8, 2048, 2048, 2, false}, // 长seq+width=2 [暂不支持width=2]
        // {"dim2048-seq2048-w3-silu", 8, 2048, 2048, 3, true},  // 长seq+width=3+SiLU [暂不支持width=3]
    };
    
    // 运行所有测试
    for (const auto& config : configs) {
        try {
            run_comparison_test(config, warmup, iters);
        } catch (const std::exception& e) {
            std::cerr << "测试 " << config.name << " 失败: " << e.what() << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "✓ 所有性能对比测试完成!" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    return 0;
}

