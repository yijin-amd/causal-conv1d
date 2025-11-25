/*
 * Causal Conv1D Backward - Layout Comparison Test
 * 
 * 对比两种内存布局的性能和精度：
 * 1. Channel-First: [Batch, Channel, Length]
 * 2. Channel-Last:  [Batch, Length, Channel]
 * 
 * 测试内容：
 * - 精度验证（确保两种实现结果一致）
 * - 性能对比（吞吐量、带宽）
 * - 不同配置下的性能表现
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>
#include <chrono>

// 包含内核头文件
#include "causal_conv1d_bwd_kernel.h"
#include "causal_conv1d_channellast_bwd_kernel.h"

// ============================================================================
// 错误检查宏
// ============================================================================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// 性能计时器
// ============================================================================

class Timer {
private:
    hipEvent_t start_event, stop_event;
    
public:
    Timer() {
        HIP_CHECK(hipEventCreate(&start_event));
        HIP_CHECK(hipEventCreate(&stop_event));
    }
    
    ~Timer() {
        HIP_CHECK(hipEventDestroy(start_event));
        HIP_CHECK(hipEventDestroy(stop_event));
    }
    
    void start() {
        HIP_CHECK(hipEventRecord(start_event));
    }
    
    float stop() {
        HIP_CHECK(hipEventRecord(stop_event));
        HIP_CHECK(hipEventSynchronize(stop_event));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

// ============================================================================
// CPU参考实现
// ============================================================================

void causal_conv1d_bwd_cpu(
    const std::vector<float>& x,
    const std::vector<float>& weight,
    const std::vector<float>& dout,
    std::vector<float>& dx,
    std::vector<float>& dweight,
    std::vector<float>& dbias,
    int batch, int dim, int seqlen, int width,
    bool silu_activation,
    bool channel_last
) {
    // 初始化
    std::fill(dx.begin(), dx.end(), 0.0f);
    std::fill(dweight.begin(), dweight.end(), 0.0f);
    std::fill(dbias.begin(), dbias.end(), 0.0f);
    
    // Forward pass to get intermediate values for SiLU gradient
    std::vector<float> out(batch * dim * seqlen);
    
    auto get_x = [&](int b, int c, int l, bool is_channel_last) -> float {
        if (is_channel_last) {
            int idx = b * seqlen * dim + l * dim + c;
            return (idx >= 0 && idx < x.size()) ? x[idx] : 0.0f;
        } else {
            int idx = b * dim * seqlen + c * seqlen + l;
            return (idx >= 0 && idx < x.size()) ? x[idx] : 0.0f;
        }
    };
    
    auto get_dout = [&](int b, int c, int l, bool is_channel_last) -> float {
        if (is_channel_last) {
            return dout[b * seqlen * dim + l * dim + c];
        } else {
            return dout[b * dim * seqlen + c * seqlen + l];
        }
    };
    
    // Forward pass
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < dim; ++c) {
            for (int l = 0; l < seqlen; ++l) {
                float sum = 0.0f;
                for (int w = 0; w < width; ++w) {
                    int l_src = l - (width - 1) + w;
                    float x_val = get_x(b, c, l_src, channel_last);
                    sum += weight[c * width + w] * x_val;
                }
                int out_idx = channel_last ? 
                    (b * seqlen * dim + l * dim + c) : 
                    (b * dim * seqlen + c * seqlen + l);
                out[out_idx] = sum;
            }
        }
    }
    
    // Backward pass
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < dim; ++c) {
            for (int l = 0; l < seqlen; ++l) {
                float dout_val = get_dout(b, c, l, channel_last);
                
                // Apply SiLU gradient if needed
                if (silu_activation) {
                    int out_idx = channel_last ? 
                        (b * seqlen * dim + l * dim + c) : 
                        (b * dim * seqlen + c * seqlen + l);
                    float out_val = out[out_idx];
                    float sigmoid = 1.0f / (1.0f + expf(-out_val));
                    float silu_grad = sigmoid * (1.0f + out_val * (1.0f - sigmoid));
                    dout_val *= silu_grad;
                }
                
                // Compute dbias
                dbias[c] += dout_val;
                
                // Compute dweight and dx
                for (int w = 0; w < width; ++w) {
                    int l_src = l - (width - 1) + w;
                    float x_val = get_x(b, c, l_src, channel_last);
                    
                    // dweight
                    dweight[c * width + w] += dout_val * x_val;
                    
                    // dx
                    if (l_src >= 0 && l_src < seqlen) {
                        int dx_idx = channel_last ?
                            (b * seqlen * dim + l_src * dim + c) :
                            (b * dim * seqlen + c * seqlen + l_src);
                        dx[dx_idx] += dout_val * weight[c * width + w];
                    }
                }
            }
        }
    }
}

// ============================================================================
// 测试配置
// ============================================================================

struct TestConfig {
    std::string name;
    int batch;
    int dim;
    int seqlen;
    int width;
    bool silu;
    
    TestConfig(const std::string& n, int b, int d, int l, int w, bool s = false)
        : name(n), batch(b), dim(d), seqlen(l), width(w), silu(s) {}
};

// ============================================================================
// 误差计算
// ============================================================================

struct ErrorStats {
    float max_err;
    float avg_err;
    float rel_err;
};

ErrorStats compute_error(const std::vector<float>& a, const std::vector<float>& b) {
    ErrorStats stats = {0.0f, 0.0f, 0.0f};
    
    float sum_err = 0.0f;
    float sum_abs_b = 0.0f;
    
    for (size_t i = 0; i < a.size(); ++i) {
        float err = std::abs(a[i] - b[i]);
        stats.max_err = std::max(stats.max_err, err);
        sum_err += err;
        sum_abs_b += std::abs(b[i]);
    }
    
    stats.avg_err = sum_err / a.size();
    stats.rel_err = (sum_abs_b > 1e-6f) ? (sum_err / sum_abs_b) : 0.0f;
    
    return stats;
}

// ============================================================================
// 布局转换函数
// ============================================================================

void convert_channel_first_to_last(
    const std::vector<float>& src,
    std::vector<float>& dst,
    int batch, int dim, int seqlen
) {
    dst.resize(batch * dim * seqlen);
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < dim; ++c) {
            for (int l = 0; l < seqlen; ++l) {
                int src_idx = b * dim * seqlen + c * seqlen + l;
                int dst_idx = b * seqlen * dim + l * dim + c;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

void convert_channel_last_to_first(
    const std::vector<float>& src,
    std::vector<float>& dst,
    int batch, int dim, int seqlen
) {
    dst.resize(batch * dim * seqlen);
    for (int b = 0; b < batch; ++b) {
        for (int l = 0; l < seqlen; ++l) {
            for (int c = 0; c < dim; ++c) {
                int src_idx = b * seqlen * dim + l * dim + c;
                int dst_idx = b * dim * seqlen + c * seqlen + l;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

// ============================================================================
// 主测试函数
// ============================================================================

void run_comparison_test(const TestConfig& config, int warmup, int iterations) {
    const int batch = config.batch;
    const int dim = config.dim;
    const int seqlen = config.seqlen;
    const int width = config.width;
    const bool silu = config.silu;
    
    const size_t x_size = batch * dim * seqlen;
    const size_t weight_size = dim * width;
    const size_t bias_size = dim;
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Test: " << config.name << "\n";
    std::cout << "  Config: [B=" << batch << ", D=" << dim << ", L=" << seqlen 
              << ", W=" << width << "]" << (silu ? " + SiLU" : "") << "\n";
    std::cout << std::string(80, '=') << "\n";
    
    // 初始化随机数生成器
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // 生成输入数据 (Channel-First format)
    std::vector<float> x_cf(x_size);
    std::vector<float> weight(weight_size);
    std::vector<float> bias(bias_size);
    std::vector<float> dout_cf(x_size);
    
    for (auto& v : x_cf) v = dis(gen);
    for (auto& v : weight) v = dis(gen);
    for (auto& v : bias) v = dis(gen);
    for (auto& v : dout_cf) v = dis(gen);
    
    // 转换为 Channel-Last format
    std::vector<float> x_cl, dout_cl;
    convert_channel_first_to_last(x_cf, x_cl, batch, dim, seqlen);
    convert_channel_first_to_last(dout_cf, dout_cl, batch, dim, seqlen);
    
    // CPU参考结果 (使用Channel-First布局计算)
    std::vector<float> dx_cpu(x_size, 0.0f);
    std::vector<float> dweight_cpu(weight_size, 0.0f);
    std::vector<float> dbias_cpu(bias_size, 0.0f);
    
    causal_conv1d_bwd_cpu(x_cf, weight, dout_cf, dx_cpu, dweight_cpu, dbias_cpu,
                          batch, dim, seqlen, width, silu, false);
    
    // ========================================================================
    // Test Channel-First Kernel
    // ========================================================================
    
    std::cout << "\n--- Channel-First Layout ---\n";
    
    // 分配GPU内存 (Channel-First)
    float *d_x_cf, *d_weight, *d_bias, *d_dout_cf, *d_dx_cf, *d_dweight_cf, *d_dbias_cf;
    HIP_CHECK(hipMalloc(&d_x_cf, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dout_cf, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dx_cf, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dweight_cf, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dbias_cf, bias_size * sizeof(float)));
    
    // 拷贝数据到GPU
    HIP_CHECK(hipMemcpy(d_x_cf, x_cf.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias, bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dout_cf, dout_cf.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    
    // 设置参数
    ConvParamsBwd params_cf;
    params_cf.x_ptr = d_x_cf;
    params_cf.weight_ptr = d_weight;
    params_cf.bias_ptr = d_bias;
    params_cf.dout_ptr = d_dout_cf;
    params_cf.dx_ptr = d_dx_cf;
    params_cf.dweight_ptr = d_dweight_cf;
    params_cf.dbias_ptr = d_dbias_cf;
    params_cf.batch = batch;
    params_cf.dim = dim;
    params_cf.seqlen = seqlen;
    params_cf.width = width;
    params_cf.silu_activation = silu;
    params_cf.x_batch_stride = dim * seqlen;
    params_cf.x_c_stride = seqlen;
    params_cf.weight_c_stride = width;
    params_cf.weight_width_stride = 1;
    params_cf.dout_batch_stride = dim * seqlen;
    params_cf.dout_c_stride = seqlen;
    params_cf.dx_batch_stride = dim * seqlen;
    params_cf.dx_c_stride = seqlen;
    params_cf.dweight_c_stride = width;
    params_cf.dweight_width_stride = 1;
    
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        HIP_CHECK(hipMemset(d_dx_cf, 0, x_size * sizeof(float)));
        HIP_CHECK(hipMemset(d_dweight_cf, 0, weight_size * sizeof(float)));
        HIP_CHECK(hipMemset(d_dbias_cf, 0, bias_size * sizeof(float)));
        causal_conv1d_bwd_launch<256, 4, float, float>(params_cf, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Performance test
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        HIP_CHECK(hipMemset(d_dx_cf, 0, x_size * sizeof(float)));
        HIP_CHECK(hipMemset(d_dweight_cf, 0, weight_size * sizeof(float)));
        HIP_CHECK(hipMemset(d_dbias_cf, 0, bias_size * sizeof(float)));
        causal_conv1d_bwd_launch<256, 4, float, float>(params_cf, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    float time_cf = timer.stop() / iterations;
    
    // 单次运行用于验证
    HIP_CHECK(hipMemset(d_dx_cf, 0, x_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dweight_cf, 0, weight_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dbias_cf, 0, bias_size * sizeof(float)));
    causal_conv1d_bwd_launch<256, 4, float, float>(params_cf, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // 拷贝结果回CPU
    std::vector<float> dx_cf(x_size);
    std::vector<float> dweight_cf(weight_size);
    std::vector<float> dbias_cf(bias_size);
    HIP_CHECK(hipMemcpy(dx_cf.data(), d_dx_cf, x_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dweight_cf.data(), d_dweight_cf, weight_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dbias_cf.data(), d_dbias_cf, bias_size * sizeof(float), hipMemcpyDeviceToHost));
    
    // 计算带宽
    size_t bytes_cf = (2 * x_size + weight_size + bias_size) * sizeof(float); // 输入
    bytes_cf += (x_size + weight_size + bias_size) * sizeof(float); // 输出
    float bw_cf = (bytes_cf / 1e9) / (time_cf / 1000.0f);
    
    std::cout << "  Performance:\n";
    std::cout << "    Time: " << time_cf << " ms\n";
    std::cout << "    Bandwidth: " << bw_cf << " GB/s\n";
    
    // 精度验证
    auto err_dx_cf = compute_error(dx_cf, dx_cpu);
    auto err_dw_cf = compute_error(dweight_cf, dweight_cpu);
    auto err_db_cf = compute_error(dbias_cf, dbias_cpu);
    
    std::cout << "  Accuracy:\n";
    std::cout << "    dx  - Max: " << err_dx_cf.max_err << ", Avg: " << err_dx_cf.avg_err 
              << ", Rel: " << err_dx_cf.rel_err << "\n";
    std::cout << "    dW  - Max: " << err_dw_cf.max_err << ", Avg: " << err_dw_cf.avg_err 
              << ", Rel: " << err_dw_cf.rel_err << "\n";
    std::cout << "    db  - Max: " << err_db_cf.max_err << ", Avg: " << err_db_cf.avg_err 
              << ", Rel: " << err_db_cf.rel_err << "\n";
    
    // ========================================================================
    // Test Channel-Last Kernel
    // ========================================================================
    
    std::cout << "\n--- Channel-Last Layout ---\n";
    
    // 分配GPU内存 (Channel-Last)
    float *d_x_cl, *d_dout_cl, *d_dx_cl, *d_dweight_cl, *d_dbias_cl;
    HIP_CHECK(hipMalloc(&d_x_cl, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dout_cl, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dx_cl, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dweight_cl, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dbias_cl, bias_size * sizeof(float)));
    
    // 拷贝数据到GPU
    HIP_CHECK(hipMemcpy(d_x_cl, x_cl.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dout_cl, dout_cl.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    
    // 设置参数
    ConvParamsChannellastBwd params_cl;
    params_cl.batch = batch;
    params_cl.dim = dim;
    params_cl.seqlen = seqlen;
    params_cl.width = width;
    params_cl.silu_activation = silu;
    params_cl.x_batch_stride = seqlen * dim;
    params_cl.x_l_stride = dim;
    params_cl.x_c_stride = 1;
    params_cl.weight_c_stride = width;
    params_cl.weight_width_stride = 1;
    params_cl.dout_batch_stride = seqlen * dim;
    params_cl.dout_l_stride = dim;
    params_cl.dout_c_stride = 1;
    params_cl.dx_batch_stride = seqlen * dim;
    params_cl.dx_l_stride = dim;
    params_cl.dx_c_stride = 1;
    params_cl.dweight_c_stride = width;
    params_cl.dweight_width_stride = 1;
    params_cl.x_ptr = d_x_cl;
    params_cl.weight_ptr = d_weight;
    params_cl.bias_ptr = d_bias;
    params_cl.dout_ptr = d_dout_cl;
    params_cl.dx_ptr = d_dx_cl;
    params_cl.dweight_ptr = d_dweight_cl;
    params_cl.dbias_ptr = d_dbias_cl;
    params_cl.initial_states_ptr = nullptr;
    params_cl.dinitial_states_ptr = nullptr;
    params_cl.dfinal_states_ptr = nullptr;
    params_cl.seq_idx_ptr = nullptr;
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        HIP_CHECK(hipMemset(d_dx_cl, 0, x_size * sizeof(float)));
        HIP_CHECK(hipMemset(d_dweight_cl, 0, weight_size * sizeof(float)));
        HIP_CHECK(hipMemset(d_dbias_cl, 0, bias_size * sizeof(float)));
        causal_conv1d_channellast_bwd_launch_full<128>(params_cl, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Performance test
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        HIP_CHECK(hipMemset(d_dx_cl, 0, x_size * sizeof(float)));
        HIP_CHECK(hipMemset(d_dweight_cl, 0, weight_size * sizeof(float)));
        HIP_CHECK(hipMemset(d_dbias_cl, 0, bias_size * sizeof(float)));
        causal_conv1d_channellast_bwd_launch_full<128>(params_cl, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    float time_cl = timer.stop() / iterations;
    
    // 单次运行用于验证
    HIP_CHECK(hipMemset(d_dx_cl, 0, x_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dweight_cl, 0, weight_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dbias_cl, 0, bias_size * sizeof(float)));
    causal_conv1d_channellast_bwd_launch_full<128>(params_cl, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // 拷贝结果回CPU (Channel-Last)
    std::vector<float> dx_cl_raw(x_size);
    std::vector<float> dweight_cl(weight_size);
    std::vector<float> dbias_cl(bias_size);
    HIP_CHECK(hipMemcpy(dx_cl_raw.data(), d_dx_cl, x_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dweight_cl.data(), d_dweight_cl, weight_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(dbias_cl.data(), d_dbias_cl, bias_size * sizeof(float), hipMemcpyDeviceToHost));
    
    // 转换回Channel-First用于比较
    std::vector<float> dx_cl;
    convert_channel_last_to_first(dx_cl_raw, dx_cl, batch, dim, seqlen);
    
    // 计算带宽
    size_t bytes_cl = (2 * x_size + weight_size + bias_size) * sizeof(float);
    bytes_cl += (x_size + weight_size + bias_size) * sizeof(float);
    float bw_cl = (bytes_cl / 1e9) / (time_cl / 1000.0f);
    
    std::cout << "  Performance:\n";
    std::cout << "    Time: " << time_cl << " ms\n";
    std::cout << "    Bandwidth: " << bw_cl << " GB/s\n";
    
    // 精度验证
    auto err_dx_cl = compute_error(dx_cl, dx_cpu);
    auto err_dw_cl = compute_error(dweight_cl, dweight_cpu);
    auto err_db_cl = compute_error(dbias_cl, dbias_cpu);
    
    std::cout << "  Accuracy:\n";
    std::cout << "    dx  - Max: " << err_dx_cl.max_err << ", Avg: " << err_dx_cl.avg_err 
              << ", Rel: " << err_dx_cl.rel_err << "\n";
    std::cout << "    dW  - Max: " << err_dw_cl.max_err << ", Avg: " << err_dw_cl.avg_err 
              << ", Rel: " << err_dw_cl.rel_err << "\n";
    std::cout << "    db  - Max: " << err_db_cl.max_err << ", Avg: " << err_db_cl.avg_err 
              << ", Rel: " << err_db_cl.rel_err << "\n";
    
    // ========================================================================
    // 性能对比总结
    // ========================================================================
    
    std::cout << "\n--- Performance Comparison ---\n";
    std::cout << "  Channel-First: " << time_cf << " ms (" << bw_cf << " GB/s)\n";
    std::cout << "  Channel-Last:  " << time_cl << " ms (" << bw_cl << " GB/s)\n";
    
    if (time_cf < time_cl) {
        float speedup = time_cl / time_cf;
        std::cout << "  Winner: Channel-First (" << std::fixed << std::setprecision(2) 
                  << speedup << "x faster)\n";
    } else {
        float speedup = time_cf / time_cl;
        std::cout << "  Winner: Channel-Last (" << std::fixed << std::setprecision(2) 
                  << speedup << "x faster)\n";
    }
    
    // 清理
    HIP_CHECK(hipFree(d_x_cf));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_dout_cf));
    HIP_CHECK(hipFree(d_dx_cf));
    HIP_CHECK(hipFree(d_dweight_cf));
    HIP_CHECK(hipFree(d_dbias_cf));
    HIP_CHECK(hipFree(d_x_cl));
    HIP_CHECK(hipFree(d_dout_cl));
    HIP_CHECK(hipFree(d_dx_cl));
    HIP_CHECK(hipFree(d_dweight_cl));
    HIP_CHECK(hipFree(d_dbias_cl));
    HIP_CHECK(hipStreamDestroy(stream));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Causal Conv1D Backward - Layout Comparison Test              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    // GPU信息
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "\nGPU: " << prop.name << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / 1e9) << " GB\n";
    
    const int warmup = 10;
    const int iterations = 100;
    
    // 测试配置
    std::vector<TestConfig> configs = {
        TestConfig("Small", 2, 64, 128, 4, false),
        TestConfig("Small + SiLU", 2, 64, 128, 4, true),
        TestConfig("Medium", 4, 256, 512, 4, false),
        TestConfig("Medium + SiLU", 4, 256, 512, 4, true),
        TestConfig("Large", 8, 512, 1024, 4, false),
        TestConfig("Large + SiLU", 8, 512, 1024, 4, true),
        TestConfig("XLarge", 16, 1024, 2048, 4, false),
        TestConfig("XLarge + SiLU", 16, 1024, 2048, 4, true),
    };
    
    for (const auto& config : configs) {
        run_comparison_test(config, warmup, iterations);
    }
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "All comparison tests completed!\n";
    std::cout << std::string(80, '=') << "\n";
    
    return 0;
}

