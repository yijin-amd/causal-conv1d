/*
 * 直接调用 causal_conv1d_fwd_cuda 和 causal_conv1d_channellast_fwd_cuda 测试性能
 * 这两个函数来自 causal_conv1d.cpp
 * 不修改原有kernel，直接测试包装函数
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// 复制 ConvParamsBase 结构体定义
// ============================================================================

struct ConvParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, width;
    bool silu_activation;

    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    index_t weight_c_stride;
    index_t weight_width_stride;
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;

    int conv_state_len;
    index_t conv_state_batch_stride;
    index_t conv_state_c_stride;
    index_t conv_state_l_stride;

    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;

    void *__restrict__ conv_state_ptr;
    int32_t *__restrict__ cache_seqlens;
    int32_t *__restrict__ conv_state_indices_ptr;
    void *__restrict__ seq_idx_ptr;

    void * initial_states_ptr;
    index_t initial_states_batch_stride;
    index_t initial_states_l_stride;
    index_t initial_states_c_stride;

    void * final_states_ptr;
    index_t final_states_batch_stride;
    index_t final_states_l_stride;
    index_t final_states_c_stride;
};

// ============================================================================
// 声明外部函数 (来自 causal_conv1d_fwd.cu)
// 这些函数在链接时会从 causal_conv1d_fwd.cu 中找到
// ============================================================================

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

// ============================================================================
// 性能测试函数
// ============================================================================

template<typename input_t, typename weight_t>
void benchmark_both_wrapper_functions(int batch, int dim, int seqlen, int width, bool silu, 
                                      int warmup, int iters, const char* dtype_name) {
    printf("\n========================================\n");
    printf("配置: batch=%d, dim=%d, seqlen=%d, width=%d\n", batch, dim, seqlen, width);
    printf("数据类型: %s, SiLU: %s\n", dtype_name, silu ? "是" : "否");
    printf("========================================\n");
    
    // 分配内存
    size_t x_size_cf = batch * dim * seqlen * sizeof(input_t);  // channel-first
    size_t x_size_cl = batch * seqlen * dim * sizeof(input_t);  // channel-last
    size_t weight_size = dim * width * sizeof(weight_t);
    size_t bias_size = dim * sizeof(weight_t);
    size_t out_size_cf = batch * dim * seqlen * sizeof(input_t);
    size_t out_size_cl = batch * seqlen * dim * sizeof(input_t);
    
    input_t *d_x_cf, *d_x_cl, *d_out_cf, *d_out_cl;
    weight_t *d_weight, *d_bias;
    
    CUDA_CHECK(cudaMalloc(&d_x_cf, x_size_cf));
    CUDA_CHECK(cudaMalloc(&d_x_cl, x_size_cl));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size));
    CUDA_CHECK(cudaMalloc(&d_out_cf, out_size_cf));
    CUDA_CHECK(cudaMalloc(&d_out_cl, out_size_cl));
    
    // 初始化随机数据
    CUDA_CHECK(cudaMemset(d_x_cf, 0, x_size_cf));
    CUDA_CHECK(cudaMemset(d_x_cl, 0, x_size_cl));
    CUDA_CHECK(cudaMemset(d_weight, 0, weight_size));
    CUDA_CHECK(cudaMemset(d_bias, 0, bias_size));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // ========== 测试 Channel-First (causal_conv1d_fwd_cuda) ==========
    ConvParamsBase params_cf;
    params_cf.batch = batch;
    params_cf.dim = dim;
    params_cf.seqlen = seqlen;
    params_cf.width = width;
    params_cf.silu_activation = silu;
    params_cf.x_batch_stride = dim * seqlen;
    params_cf.x_c_stride = seqlen;
    params_cf.x_l_stride = 1;
    params_cf.weight_c_stride = width;
    params_cf.weight_width_stride = 1;
    params_cf.out_batch_stride = dim * seqlen;
    params_cf.out_c_stride = seqlen;
    params_cf.out_l_stride = 1;
    params_cf.x_ptr = d_x_cf;
    params_cf.weight_ptr = d_weight;
    params_cf.bias_ptr = d_bias;
    params_cf.out_ptr = d_out_cf;
    params_cf.conv_state_ptr = nullptr;
    params_cf.cache_seqlens = nullptr;
    params_cf.conv_state_indices_ptr = nullptr;
    params_cf.seq_idx_ptr = nullptr;
    params_cf.initial_states_ptr = nullptr;
    params_cf.final_states_ptr = nullptr;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        causal_conv1d_fwd_cuda<input_t, weight_t>(params_cf, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        causal_conv1d_fwd_cuda<input_t, weight_t>(params_cf, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_cf = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_cf, start, stop));
    time_cf /= iters;
    
    // ========== 测试 Channel-Last (causal_conv1d_channellast_fwd_cuda) ==========
    ConvParamsBase params_cl;
    params_cl.batch = batch;
    params_cl.dim = dim;
    params_cl.seqlen = seqlen;
    params_cl.width = width;
    params_cl.silu_activation = silu;
    params_cl.x_batch_stride = seqlen * dim;
    params_cl.x_c_stride = 1;
    params_cl.x_l_stride = dim;
    params_cl.weight_c_stride = width;
    params_cl.weight_width_stride = 1;
    params_cl.out_batch_stride = seqlen * dim;
    params_cl.out_c_stride = 1;
    params_cl.out_l_stride = dim;
    params_cl.x_ptr = d_x_cl;
    params_cl.weight_ptr = d_weight;
    params_cl.bias_ptr = d_bias;
    params_cl.out_ptr = d_out_cl;
    params_cl.conv_state_ptr = nullptr;
    params_cl.cache_seqlens = nullptr;
    params_cl.conv_state_indices_ptr = nullptr;
    params_cl.seq_idx_ptr = nullptr;
    params_cl.initial_states_ptr = nullptr;
    params_cl.final_states_ptr = nullptr;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        causal_conv1d_channellast_fwd_cuda<input_t, weight_t>(params_cl, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; i++) {
        causal_conv1d_channellast_fwd_cuda<input_t, weight_t>(params_cl, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float time_cl = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_cl, start, stop));
    time_cl /= iters;
    
    // 计算性能指标
    size_t total_elements = (size_t)batch * dim * seqlen;
    size_t bytes_read = x_size_cf + weight_size + bias_size;
    size_t bytes_write = out_size_cf;
    size_t total_bytes = bytes_read + bytes_write;
    
    float bandwidth_cf = (total_bytes / 1e9) / (time_cf / 1000.0f);
    float bandwidth_cl = (total_bytes / 1e9) / (time_cl / 1000.0f);
    float throughput_cf = (total_elements * width * 2) / 1e9 / (time_cf / 1000.0f);
    float throughput_cl = (total_elements * width * 2) / 1e9 / (time_cl / 1000.0f);
    
    printf("\n【causal_conv1d_fwd_cuda (Channel-First)】\n");
    printf("  执行时间: %.4f ms\n", time_cf);
    printf("  带宽: %.2f GB/s\n", bandwidth_cf);
    printf("  吞吐量: %.2f GFLOPS\n", throughput_cf);
    
    printf("\n【causal_conv1d_channellast_fwd_cuda (Channel-Last)】\n");
    printf("  执行时间: %.4f ms\n", time_cl);
    printf("  带宽: %.2f GB/s\n", bandwidth_cl);
    printf("  吞吐量: %.2f GFLOPS\n", throughput_cl);
    
    printf("\n【对比】\n");
    if (time_cf < time_cl) {
        printf("  ✓ Channel-First 更快: %.2fx\n", time_cl / time_cf);
        printf("  时间节省: %.4f ms (%.1f%%)\n", time_cl - time_cf, (time_cl - time_cf) / time_cl * 100);
    } else {
        printf("  ✓ Channel-Last 更快: %.2fx\n", time_cf / time_cl);
        printf("  时间节省: %.4f ms (%.1f%%)\n", time_cf - time_cl, (time_cf - time_cl) / time_cf * 100);
    }
    
    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x_cf));
    CUDA_CHECK(cudaFree(d_x_cl));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_out_cf));
    CUDA_CHECK(cudaFree(d_out_cl));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    printf("============================================================\n");
    printf("直接调用 causal_conv1d_fwd_cuda 性能对比测试\n");
    printf("测试函数: causal_conv1d_fwd_cuda vs causal_conv1d_channellast_fwd_cuda\n");
    printf("============================================================\n");
    
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("\nGPU信息:\n");
    printf("  设备名称: %s\n", prop.name);
    printf("  计算能力: %d.%d\n", prop.major, prop.minor);
    printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / 1e9);
    printf("  内存带宽: %.2f GB/s\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    
    const int warmup = 10;
    const int iters = 100;
    
    printf("\n测试参数: warmup=%d, iterations=%d\n", warmup, iters);
    
    struct TestConfig {
        int batch, dim, seqlen, width;
        bool silu;
    };
    
    TestConfig configs[] = {
        // ========== dim=64 测试 (小规模, 覆盖所有width和seqlen范围) ==========
        {2, 64, 128, 4, false},   // 极小
        {2, 64, 512, 4, false},   // 小
        {2, 64, 512, 4, true},    // 小+SiLU
        {2, 64, 1024, 4, false},  // 中
        {2, 64, 1024, 4, true},   // 中+SiLU
        {2, 64, 2048, 2, false},  // 中大+width=2
        {2, 64, 2048, 3, false},  // 中大+width=3
        {2, 64, 2048, 4, false},  // 中大+width=4
        
        // ========== dim=256 测试 (中等规模, 不同seqlen+width) ==========
        {4, 256, 256, 4, false},  // 短seq
        {4, 256, 512, 4, false},  // 中短seq
        {4, 256, 512, 4, true},   // 中短seq+SiLU
        {4, 256, 1024, 4, false}, // 中seq
        {4, 256, 1024, 4, true},  // 中seq+SiLU
        {4, 256, 2048, 2, false}, // 长seq+width=2
        {4, 256, 2048, 3, false}, // 长seq+width=3
        {4, 256, 2048, 4, false}, // 长seq+width=4
        
        // ========== dim=512 测试 (中大规模, 关键性能区) ==========
        {4, 512, 512, 4, false},  // 短seq
        {4, 512, 1024, 4, false}, // 中seq
        {4, 512, 1024, 4, true},  // 中seq+SiLU
        {4, 512, 2048, 4, false}, // 长seq (峰值性能点)
        {4, 512, 2048, 4, true},  // 长seq+SiLU
        {4, 512, 2048, 2, false}, // 长seq+width=2
        {4, 512, 2048, 3, true},  // 长seq+width=3+SiLU
        
        // ========== dim=1024 测试 (大规模) ==========
        {8, 1024, 512, 4, false},  // 短seq
        {8, 1024, 1024, 4, false}, // 中seq
        {8, 1024, 1024, 4, true},  // 中seq+SiLU
        {8, 1024, 2048, 4, false}, // 长seq
        {8, 1024, 2048, 4, true},  // 长seq+SiLU
        {8, 1024, 4096, 4, false}, // 超长seq
        
        // ========== dim=2048 测试 (超大规模) ==========
        {8, 2048, 1024, 4, false}, // 中seq
        {8, 2048, 2048, 4, false}, // 长seq
        {8, 2048, 2048, 4, true},  // 长seq+SiLU
        {8, 2048, 4096, 4, false}, // 超长seq
        {8, 2048, 2048, 2, false}, // 长seq+width=2
        {8, 2048, 2048, 3, true},  // 长seq+width=3+SiLU
    };
    
    printf("\n开始性能对比测试...\n");
    printf("============================================================\n");
    
    // FP32测试
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("【FP32 测试】\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    for (const auto& cfg : configs) {
        benchmark_both_wrapper_functions<float, float>(
            cfg.batch, cfg.dim, cfg.seqlen, cfg.width, cfg.silu,
            warmup, iters, "float32"
        );
    }
    
    // 注意: FP16和BF16需要PyTorch类型支持，这里只测试FP32
    printf("\n注意: 由于需要PyTorch类型支持，此版本只测试FP32\n");
    printf("如需测试FP16/BF16，请使用完整的PyTorch编译流程\n");
    
    printf("\n============================================================\n");
    printf("✓ 所有性能对比测试完成!\n");
    printf("============================================================\n");
    
    return 0;
}

