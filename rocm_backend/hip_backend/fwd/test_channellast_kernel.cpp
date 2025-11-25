/*
 * Standalone Test for causal_conv1d_channellast_fwd_kernel
 * Tests both accuracy (vs CPU reference) and performance (timing + bandwidth)
 * 
 * Compile:
 *   hipcc -O2 -std=c++17 --offload-arch=gfx942 test_channellast_kernel.cpp -o test_channellast_kernel
 * 
 * Run:
 *   ./test_channellast_kernel [test_id]
 *   test_id: optional, 0=all, 1=accuracy, 2=performance (default: 0)
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <chrono>

// ==================== Error Checking ====================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// ==================== Kernel Parameters ====================

struct ConvParamsChannelLast {
    int batch;
    int dim;
    int seqlen;
    int width;
    bool silu_activation;
    
    // Strides
    int x_batch_stride;
    int x_l_stride;
    int x_c_stride;
    
    int weight_c_stride;
    int weight_width_stride;
    
    int out_batch_stride;
    int out_l_stride;
    int out_c_stride;
    
    // Pointers
    float* x_ptr;
    float* weight_ptr;
    float* bias_ptr;
    float* out_ptr;
    int* seq_idx_ptr;
    float* initial_states_ptr;
    float* final_states_ptr;
};

// ==================== Kernel Declaration ====================

template<int kNThreads, int kWidth>
__global__ void causal_conv1d_channellast_fwd_kernel(ConvParamsChannelLast params);

// ==================== Kernel Launcher ====================

template <int kNThreads, int kWidth>
void causal_conv1d_channellast_fwd_launch(ConvParamsChannelLast &params, hipStream_t stream) {
    dim3 grid((params.batch * params.seqlen + kNThreads - 1) / kNThreads);
    dim3 block(kNThreads);
    
    causal_conv1d_channellast_fwd_kernel<kNThreads, kWidth><<<grid, block, 0, stream>>>(params);
}

// ==================== Kernel Implementation ====================

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

template<int kNThreads, int kWidth>
__global__ void causal_conv1d_channellast_fwd_kernel(ConvParamsChannelLast params) {
    constexpr int K_UNROLL = 4;
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.batch * params.seqlen) return;
    
    const int batch_id = tid / params.seqlen;
    const int t = tid % params.seqlen;
    
    // Shared memory for weights
    __shared__ float smem_weight[K_UNROLL][kWidth];
    
    // Load weights to shared memory
    for (int k_iter = 0; k_iter < (params.dim + K_UNROLL - 1) / K_UNROLL; ++k_iter) {
        for (int w = 0; w < kWidth; ++w) {
            int k = k_iter * K_UNROLL + threadIdx.x % K_UNROLL;
            if (k < params.dim && threadIdx.x < K_UNROLL) {
                smem_weight[threadIdx.x][w] = params.weight_ptr[k * params.weight_c_stride + w * params.weight_width_stride];
            }
        }
        __syncthreads();
        
        // Compute convolution for K_UNROLL channels
        for (int k_offset = 0; k_offset < K_UNROLL; ++k_offset) {
            int k = k_iter * K_UNROLL + k_offset;
            if (k >= params.dim) break;
            
            float sum = 0.0f;
            
            // Convolution loop
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                int t_src = t - (kWidth - 1) + w;
                
                float x_val = 0.0f;
                if (t_src >= 0) {
                    x_val = params.x_ptr[batch_id * params.x_batch_stride + 
                                         t_src * params.x_l_stride + 
                                         k * params.x_c_stride];
                }
                
                sum += x_val * smem_weight[k_offset][w];
            }
            
            // Add bias
            if (params.bias_ptr != nullptr) {
                sum += params.bias_ptr[k];
            }
            
            // Apply SiLU activation
            if (params.silu_activation) {
                sum = silu(sum);
            }
            
            // Write output
            params.out_ptr[batch_id * params.out_batch_stride + 
                          t * params.out_l_stride + 
                          k * params.out_c_stride] = sum;
        }
        __syncthreads();
    }
}

// ==================== CPU Reference Implementation ====================

void causal_conv1d_channellast_cpu(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int batch, int dim, int seqlen, int width,
    bool use_silu
) {
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seqlen; ++t) {
            for (int d = 0; d < dim; ++d) {
                float sum = 0.0f;
                
                for (int w = 0; w < width; ++w) {
                    int t_src = t - (width - 1) + w;
                    if (t_src >= 0) {
                        sum += x[b * seqlen * dim + t_src * dim + d] * 
                               weight[d * width + w];
                    }
                }
                
                if (bias != nullptr) {
                    sum += bias[d];
                }
                
                if (use_silu) {
                    sum = sum / (1.0f + std::exp(-sum));
                }
                
                out[b * seqlen * dim + t * dim + d] = sum;
            }
        }
    }
}

// ==================== Test Configuration ====================

struct TestConfig {
    const char* name;
    int batch;
    int dim;
    int seqlen;
    int width;
    bool use_bias;
    bool use_silu;
};

// ==================== Accuracy Test ====================

bool test_accuracy(const TestConfig& cfg) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Accuracy Test: " << cfg.name << std::endl;
    std::cout << "  Config: batch=" << cfg.batch << ", dim=" << cfg.dim 
              << ", seqlen=" << cfg.seqlen << ", width=" << cfg.width << std::endl;
    std::cout << "  Bias=" << (cfg.use_bias ? "Yes" : "No") 
              << ", SiLU=" << (cfg.use_silu ? "Yes" : "No") << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int x_size = cfg.batch * cfg.seqlen * cfg.dim;
    const int weight_size = cfg.dim * cfg.width;
    const int bias_size = cfg.dim;
    
    // Allocate host memory
    std::vector<float> h_x(x_size);
    std::vector<float> h_weight(weight_size);
    std::vector<float> h_bias(cfg.use_bias ? bias_size : 0);
    std::vector<float> h_out_gpu(x_size, 0.0f);
    std::vector<float> h_out_cpu(x_size, 0.0f);
    
    // Initialize with random data
    srand(42);
    for (int i = 0; i < x_size; ++i) {
        h_x[i] = (rand() % 200 - 100) / 100.0f;  // [-1, 1]
    }
    for (int i = 0; i < weight_size; ++i) {
        h_weight[i] = (rand() % 200 - 100) / 100.0f;
    }
    if (cfg.use_bias) {
        for (int i = 0; i < bias_size; ++i) {
            h_bias[i] = (rand() % 100 - 50) / 100.0f;
        }
    }
    
    // Allocate device memory
    float *d_x, *d_weight, *d_bias = nullptr, *d_out;
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    if (cfg.use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(&d_out, x_size * sizeof(float)));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    if (cfg.use_bias) {
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    }
    
    // Setup parameters
    ConvParamsChannelLast params;
    params.batch = cfg.batch;
    params.dim = cfg.dim;
    params.seqlen = cfg.seqlen;
    params.width = cfg.width;
    params.silu_activation = cfg.use_silu;
    
    params.x_batch_stride = cfg.seqlen * cfg.dim;
    params.x_l_stride = cfg.dim;
    params.x_c_stride = 1;
    params.weight_c_stride = cfg.width;
    params.weight_width_stride = 1;
    params.out_batch_stride = cfg.seqlen * cfg.dim;
    params.out_l_stride = cfg.dim;
    params.out_c_stride = 1;
    
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = d_bias;
    params.out_ptr = d_out;
    params.seq_idx_ptr = nullptr;
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = nullptr;
    
    // Run GPU kernel
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    causal_conv1d_channellast_fwd_launch<128, 4>(params, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Copy result back
    HIP_CHECK(hipMemcpy(h_out_gpu.data(), d_out, x_size * sizeof(float), hipMemcpyDeviceToHost));
    
    // Run CPU reference
    causal_conv1d_channellast_cpu(
        h_x.data(), h_weight.data(),
        cfg.use_bias ? h_bias.data() : nullptr,
        h_out_cpu.data(),
        cfg.batch, cfg.dim, cfg.seqlen, cfg.width,
        cfg.use_silu
    );
    
    // Compare results
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int num_errors = 0;
    const float tolerance = cfg.use_silu ? 1e-3f : 1e-4f;
    
    for (int i = 0; i < x_size; ++i) {
        float diff = std::abs(h_out_gpu[i] - h_out_cpu[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        
        if (diff > tolerance) {
            if (num_errors < 3) {
                int b = i / (cfg.seqlen * cfg.dim);
                int remaining = i % (cfg.seqlen * cfg.dim);
                int t = remaining / cfg.dim;
                int d = remaining % cfg.dim;
                std::cout << "  Mismatch [b=" << b << ",t=" << t << ",d=" << d << "]: "
                          << "GPU=" << h_out_gpu[i] << " CPU=" << h_out_cpu[i]
                          << " diff=" << diff << std::endl;
            }
            num_errors++;
        }
    }
    
    float avg_diff = sum_diff / x_size;
    
    std::cout << "\n[Results]" << std::endl;
    std::cout << "  Max difference:  " << std::scientific << max_diff << std::endl;
    std::cout << "  Avg difference:  " << avg_diff << std::endl;
    std::cout << "  Errors (>" << tolerance << "): " << num_errors << " / " << x_size << std::endl;
    
    bool passed = (max_diff < tolerance);
    std::cout << "  Status: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    if (d_bias) HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipStreamDestroy(stream));
    
    return passed;
}

// ==================== Performance Test ====================

struct PerfResult {
    float mean_ms;
    float min_ms;
    float max_ms;
    float std_dev_ms;
    float bandwidth_gb_s;
};

PerfResult test_performance(const TestConfig& cfg, int warmup = 10, int iters = 100) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Performance Test: " << cfg.name << std::endl;
    std::cout << "  Config: batch=" << cfg.batch << ", dim=" << cfg.dim 
              << ", seqlen=" << cfg.seqlen << ", width=" << cfg.width << std::endl;
    std::cout << "  Warmup: " << warmup << ", Iterations: " << iters << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int x_size = cfg.batch * cfg.seqlen * cfg.dim;
    const int weight_size = cfg.dim * cfg.width;
    const int bias_size = cfg.dim;
    
    // Allocate device memory
    float *d_x, *d_weight, *d_bias = nullptr, *d_out;
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    if (cfg.use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(&d_out, x_size * sizeof(float)));
    
    // Setup parameters
    ConvParamsChannelLast params;
    params.batch = cfg.batch;
    params.dim = cfg.dim;
    params.seqlen = cfg.seqlen;
    params.width = cfg.width;
    params.silu_activation = cfg.use_silu;
    
    params.x_batch_stride = cfg.seqlen * cfg.dim;
    params.x_l_stride = cfg.dim;
    params.x_c_stride = 1;
    params.weight_c_stride = cfg.width;
    params.weight_width_stride = 1;
    params.out_batch_stride = cfg.seqlen * cfg.dim;
    params.out_l_stride = cfg.dim;
    params.out_c_stride = 1;
    
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = d_bias;
    params.out_ptr = d_out;
    params.seq_idx_ptr = nullptr;
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = nullptr;
    
    // Create stream and events
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < warmup; ++i) {
        causal_conv1d_channellast_fwd_launch<128, 4>(params, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    std::vector<float> times_ms(iters);
    
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(start, stream));
        causal_conv1d_channellast_fwd_launch<128, 4>(params, stream);
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times_ms[i] = ms;
    }
    
    // Calculate statistics
    float sum = 0, sum_sq = 0;
    float min_time = times_ms[0], max_time = times_ms[0];
    
    for (float t : times_ms) {
        sum += t;
        sum_sq += t * t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    
    float mean = sum / iters;
    float variance = (sum_sq / iters) - (mean * mean);
    float std_dev = std::sqrt(std::max(0.0f, variance));
    
    // Calculate bandwidth
    size_t bytes_read = x_size * sizeof(float) + weight_size * sizeof(float);
    if (cfg.use_bias) bytes_read += bias_size * sizeof(float);
    size_t bytes_written = x_size * sizeof(float);
    size_t total_bytes = bytes_read + bytes_written;
    
    float bandwidth_gb_s = (total_bytes / (mean * 1e-3)) / 1e9;
    
    // Print results
    std::cout << "\n[Performance Results]" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Mean time:     " << mean << " ms" << std::endl;
    std::cout << "  Min time:      " << min_time << " ms" << std::endl;
    std::cout << "  Max time:      " << max_time << " ms" << std::endl;
    std::cout << "  Std dev:       " << std_dev << " ms" << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "  Bandwidth:     " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    if (d_bias) HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipStreamDestroy(stream));
    
    return {mean, min_time, max_time, std_dev, bandwidth_gb_s};
}

// ==================== Main ====================

int main(int argc, char* argv[]) {
    int test_mode = 0;  // 0=all, 1=accuracy, 2=performance
    
    if (argc > 1) {
        test_mode = std::atoi(argv[1]);
    }
    
    std::cout << "╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Channel-Last Causal Conv1D Kernel Test                       ║" << std::endl;
    std::cout << "║  Test Mode: ";
    if (test_mode == 0) std::cout << "ALL (Accuracy + Performance)";
    else if (test_mode == 1) std::cout << "Accuracy Only";
    else if (test_mode == 2) std::cout << "Performance Only";
    std::cout << std::string(70 - 14 - (test_mode == 0 ? 29 : test_mode == 1 ? 14 : 18), ' ') << "║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    
    // Test configurations
    std::vector<TestConfig> configs = {
        {"Tiny", 1, 32, 64, 4, true, true},
        {"Small", 2, 64, 256, 4, true, true},
        {"Medium", 4, 128, 512, 4, true, true},
        {"Large", 4, 64, 2048, 4, true, true},
        {"Width=2", 2, 128, 512, 2, true, true},
        {"Width=8", 2, 128, 512, 8, true, true},
        {"No Bias", 2, 64, 256, 4, false, true},
        {"No SiLU", 2, 64, 256, 4, true, false},
    };
    
    // Run accuracy tests
    if (test_mode == 0 || test_mode == 1) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  ACCURACY TESTS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        int passed = 0;
        for (const auto& cfg : configs) {
            if (test_accuracy(cfg)) {
                passed++;
            }
        }
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Accuracy Summary: " << passed << " / " << configs.size() << " PASSED" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }
    
    // Run performance tests
    if (test_mode == 0 || test_mode == 2) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  PERFORMANCE TESTS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        std::vector<PerfResult> results;
        for (const auto& cfg : configs) {
            results.push_back(test_performance(cfg));
        }
        
        // Print summary table
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  PERFORMANCE SUMMARY" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << std::endl;
        
        std::cout << std::left << std::setw(20) << "Configuration"
                  << std::right << std::setw(12) << "Mean(ms)"
                  << std::setw(12) << "Min(ms)"
                  << std::setw(12) << "BW(GB/s)" << std::endl;
        std::cout << std::string(56, '-') << std::endl;
        
        for (size_t i = 0; i < configs.size(); ++i) {
            std::cout << std::left << std::setw(20) << configs[i].name
                      << std::right << std::fixed << std::setprecision(4)
                      << std::setw(12) << results[i].mean_ms
                      << std::setw(12) << results[i].min_ms
                      << std::setprecision(2)
                      << std::setw(12) << results[i].bandwidth_gb_s << std::endl;
        }
        std::cout << std::string(70, '=') << std::endl;
    }
    
    std::cout << "\n✓ Test completed!" << std::endl;
    
    return 0;
}

