/*
 * Standalone Test for causal_conv1d_channellast_bwd_kernel (HIP Implementation)
 * Tests both accuracy (vs CPU reference) and performance (timing + bandwidth)
 * 
 * Compile:
 *   hipcc -O2 -std=c++17 --offload-arch=gfx942 test_channellast_bwd_kernel.cpp -o test_channellast_bwd_kernel
 * 
 * Run:
 *   ./test_channellast_bwd_kernel [test_id]
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

// ==================== Helper Structures ====================

struct ConvParamsBwd {
    int batch;
    int dim;
    int seqlen;
    int width;
    bool silu_activation;
    
    // Input strides
    int x_batch_stride;
    int x_l_stride;
    int x_c_stride;
    
    // Weight strides
    int weight_c_stride;
    int weight_width_stride;
    
    // dout strides
    int dout_batch_stride;
    int dout_l_stride;
    int dout_c_stride;
    
    // dx strides
    int dx_batch_stride;
    int dx_l_stride;
    int dx_c_stride;
    
    // dweight strides
    int dweight_c_stride;
    int dweight_width_stride;
    
    // Pointers
    float* x_ptr;
    float* weight_ptr;
    float* bias_ptr;
    float* dout_ptr;
    float* dx_ptr;
    float* dweight_ptr;
    float* dbias_ptr;
};

// ==================== Device Helper Functions ====================

__device__ __forceinline__ float silu_grad(float x, float out) {
    // out = x / (1 + exp(-x)) = x * sigmoid(x)
    // d(out)/d(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    float sigmoid = out / (x + 1e-10f);  // out / x = sigmoid
    return sigmoid * (1.0f + x * (1.0f - sigmoid));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ==================== Simplified Channel-Last Backward Kernel ====================

template<int kNThreads, int kWidth, int kChunkSizeL, int kChunkSizeC>
__global__ void causal_conv1d_channellast_bwd_kernel(ConvParamsBwd params) {
    constexpr int kNElts = 4;  // Process 4 elements at a time
    constexpr int kNThreadsPerC = kChunkSizeC / kNElts;
    constexpr int kLPerLoad = kChunkSizeL / (kNThreads / kNThreadsPerC);
    
    // Shared memory
    __shared__ float dout_smem[kChunkSizeL + kWidth - 1][kChunkSizeC];
    __shared__ float x_smem[kWidth - 1 + kChunkSizeL + kWidth - 1][kChunkSizeC];
    
    const int batch_id = blockIdx.x;
    const int chunk_l_id = blockIdx.y;
    const int chunk_c_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    
    const int base_l = chunk_l_id * kChunkSizeL;
    const int base_c = chunk_c_id * kChunkSizeC;
    
    // Load dout and x to shared memory
    for (int l = l_idx; l < kChunkSizeL; l += kLPerLoad) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_l = base_l + l;
            const int global_c = base_c + c_idx * kNElts + c;
            
            if (global_l < params.seqlen && global_c < params.dim) {
                const int idx = batch_id * params.dout_batch_stride + 
                               global_l * params.dout_l_stride + 
                               global_c * params.dout_c_stride;
                dout_smem[l][c_idx * kNElts + c] = params.dout_ptr[idx];
                
                const int x_idx = batch_id * params.x_batch_stride + 
                                 global_l * params.x_l_stride + 
                                 global_c * params.x_c_stride;
                x_smem[kWidth - 1 + l][c_idx * kNElts + c] = params.x_ptr[x_idx];
            } else {
                dout_smem[l][c_idx * kNElts + c] = 0.0f;
                x_smem[kWidth - 1 + l][c_idx * kNElts + c] = 0.0f;
            }
        }
    }
    
    // Load boundary elements for x (previous kWidth-1 elements)
    if (l_idx < kWidth - 1) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_l = base_l + l_idx - (kWidth - 1);
            const int global_c = base_c + c_idx * kNElts + c;
            
            if (global_l >= 0 && global_l < params.seqlen && global_c < params.dim) {
                const int x_idx = batch_id * params.x_batch_stride + 
                                 global_l * params.x_l_stride + 
                                 global_c * params.x_c_stride;
                x_smem[l_idx][c_idx * kNElts + c] = params.x_ptr[x_idx];
            } else {
                x_smem[l_idx][c_idx * kNElts + c] = 0.0f;
            }
        }
    }
    
    // Load boundary elements for dout (next kWidth-1 elements)
    if (l_idx < kWidth - 1) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_l = base_l + kChunkSizeL + l_idx;
            const int global_c = base_c + c_idx * kNElts + c;
            
            if (global_l < params.seqlen && global_c < params.dim) {
                const int idx = batch_id * params.dout_batch_stride + 
                               global_l * params.dout_l_stride + 
                               global_c * params.dout_c_stride;
                dout_smem[kChunkSizeL + l_idx][c_idx * kNElts + c] = params.dout_ptr[idx];
            } else {
                dout_smem[kChunkSizeL + l_idx][c_idx * kNElts + c] = 0.0f;
            }
        }
    }
    
    // Load extra x for SiLU recomputation
    if (params.silu_activation && l_idx < kWidth - 1) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_l = base_l + kChunkSizeL + l_idx;
            const int global_c = base_c + c_idx * kNElts + c;
            
            if (global_l < params.seqlen && global_c < params.dim) {
                const int x_idx = batch_id * params.x_batch_stride + 
                                 global_l * params.x_l_stride + 
                                 global_c * params.x_c_stride;
                x_smem[kWidth - 1 + kChunkSizeL + l_idx][c_idx * kNElts + c] = params.x_ptr[x_idx];
            } else {
                x_smem[kWidth - 1 + kChunkSizeL + l_idx][c_idx * kNElts + c] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // Reorganize threads for computation
    constexpr int kLPerThread = kChunkSizeL * kChunkSizeC / kNThreads;
    constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
    
    const int row_idx = tid / kNThreadsPerRow;  // which channel
    const int col_idx = tid % kNThreadsPerRow;  // which time group
    
    if (base_c + row_idx >= params.dim) return;
    
    // Load weight and bias for this channel
    float weight_vals[kWidth];
    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        weight_vals[w] = params.weight_ptr[(base_c + row_idx) * params.weight_c_stride + 
                                           w * params.weight_width_stride];
    }
    
    float bias_val = (params.bias_ptr != nullptr) ? params.bias_ptr[base_c + row_idx] : 0.0f;
    
    // Load dout and x for this thread
    float dout_vals[kLPerThread + kWidth - 1];
    float x_vals[kWidth - 1 + kLPerThread + kWidth - 1];
    
    #pragma unroll
    for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
        dout_vals[i] = dout_smem[col_idx * kLPerThread + i][row_idx];
        x_vals[i] = x_smem[col_idx * kLPerThread + i][row_idx];
    }
    
    // Recompute output for SiLU gradient
    if (params.silu_activation) {
        #pragma unroll
        for (int i = kWidth - 1 + kLPerThread; i < kWidth - 1 + kLPerThread + kWidth - 1; ++i) {
            x_vals[i] = x_smem[col_idx * kLPerThread + i][row_idx];
        }
        
        #pragma unroll
        for (int i = 0; i < kLPerThread + kWidth - 1; ++i) {
            float out_val = bias_val;
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                out_val += weight_vals[w] * x_vals[i + w];
            }
            // Apply SiLU gradient
            float sigmoid = 1.0f / (1.0f + expf(-out_val));
            dout_vals[i] *= sigmoid * (1.0f + out_val * (1.0f - sigmoid));
        }
    }
    
    // Compute dweight
    float dweight_vals[kWidth] = {0};
    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        #pragma unroll
        for (int i = 0; i < kLPerThread; ++i) {
            dweight_vals[w] += x_vals[i + w] * dout_vals[i];
        }
    }
    
    // Reduce dweight across threads in the same row
    __shared__ float dweight_reduce[kNThreads][kWidth];
    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        dweight_reduce[tid][w] = dweight_vals[w];
    }
    __syncthreads();
    
    // Simple reduction
    for (int stride = kNThreadsPerRow / 2; stride > 0; stride /= 2) {
        if (col_idx < stride) {
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                dweight_reduce[tid][w] += dweight_reduce[tid + stride][w];
            }
        }
        __syncthreads();
    }
    
    // Atomically add to global dweight
    if (col_idx == 0) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            atomicAdd(&params.dweight_ptr[(base_c + row_idx) * params.dweight_c_stride + 
                                          w * params.dweight_width_stride], 
                      dweight_reduce[tid][w]);
        }
    }
    
    // Compute dbias
    if (params.dbias_ptr != nullptr) {
        float dbias_val = 0.0f;
        #pragma unroll
        for (int i = 0; i < kLPerThread; ++i) {
            dbias_val += dout_vals[i];
        }
        
        __shared__ float dbias_reduce[kNThreads];
        dbias_reduce[tid] = dbias_val;
        __syncthreads();
        
        for (int stride = kNThreadsPerRow / 2; stride > 0; stride /= 2) {
            if (col_idx < stride) {
                dbias_reduce[tid] += dbias_reduce[tid + stride];
            }
            __syncthreads();
        }
        
        if (col_idx == 0) {
            atomicAdd(&params.dbias_ptr[base_c + row_idx], dbias_reduce[tid]);
        }
    }
    
    // Compute dx
    float dx_vals[kLPerThread] = {0};
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            dx_vals[i] += weight_vals[kWidth - 1 - w] * dout_vals[i + w];
        }
    }
    
    // Store dx back to shared memory
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        x_smem[kWidth - 1 + col_idx * kLPerThread + i][row_idx] = dx_vals[i];
    }
    __syncthreads();
    
    // Write dx to global memory
    for (int l = l_idx; l < kChunkSizeL; l += kLPerLoad) {
        const int global_l = base_l + l;
        if (global_l < params.seqlen) {
            for (int c = 0; c < kNElts; ++c) {
                const int global_c = base_c + c_idx * kNElts + c;
                if (global_c < params.dim) {
                    const int idx = batch_id * params.dx_batch_stride + 
                                   global_l * params.dx_l_stride + 
                                   global_c * params.dx_c_stride;
                    params.dx_ptr[idx] = x_smem[kWidth - 1 + l][c_idx * kNElts + c];
                }
            }
        }
    }
}

// ==================== Kernel Launcher ====================

template<int kNThreads, int kWidth>
void causal_conv1d_channellast_bwd_launch(ConvParamsBwd& params, hipStream_t stream) {
    constexpr int kChunkSizeL = 64;
    constexpr int kChunkSizeC = 64;
    
    const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
    const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
    
    dim3 grid(params.batch, n_chunks_L, n_chunks_C);
    dim3 block(kNThreads);
    
    hipLaunchKernelGGL(
        (causal_conv1d_channellast_bwd_kernel<kNThreads, kWidth, kChunkSizeL, kChunkSizeC>),
        grid, block, 0, stream, params);
}

// ==================== CPU Reference Implementation ====================

void causal_conv1d_channellast_bwd_cpu(
    const float* x,           // [batch, seqlen, dim]
    const float* weight,      // [dim, width]
    const float* bias,        // [dim] or nullptr
    const float* dout,        // [batch, seqlen, dim]
    float* dx,                // [batch, seqlen, dim]
    float* dweight,           // [dim, width]
    float* dbias,             // [dim] or nullptr
    int batch, int dim, int seqlen, int width,
    bool use_silu
) {
    // Initialize gradients
    for (int d = 0; d < dim * width; ++d) {
        dweight[d] = 0.0f;
    }
    if (dbias != nullptr) {
        for (int d = 0; d < dim; ++d) {
            dbias[d] = 0.0f;
        }
    }
    
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seqlen; ++t) {
            for (int d = 0; d < dim; ++d) {
                const int idx = b * seqlen * dim + t * dim + d;
                
                // Recompute output if SiLU
                float out_val = 0.0f;
                float dout_val = dout[idx];
                
                if (use_silu) {
                    out_val = (bias != nullptr) ? bias[d] : 0.0f;
                    for (int w = 0; w < width; ++w) {
                        int t_src = t - (width - 1) + w;
                        if (t_src >= 0) {
                            out_val += x[b * seqlen * dim + t_src * dim + d] * weight[d * width + w];
                        }
                    }
                    // Apply SiLU gradient
                    float sigmoid = 1.0f / (1.0f + std::exp(-out_val));
                    dout_val *= sigmoid * (1.0f + out_val * (1.0f - sigmoid));
                }
                
                // Compute dx
                float dx_val = 0.0f;
                for (int w = 0; w < width; ++w) {
                    int t_dst = t + w;
                    if (t_dst < seqlen) {
                        dx_val += weight[d * width + (width - 1 - w)] * 
                                 dout[b * seqlen * dim + t_dst * dim + d];
                    }
                }
                
                if (use_silu) {
                    // For SiLU, need to use the modified dout_val
                    dx_val = 0.0f;
                    for (int w = 0; w < width; ++w) {
                        int t_dst = t + w;
                        if (t_dst < seqlen) {
                            // Recompute dout for t_dst
                            float out_dst = (bias != nullptr) ? bias[d] : 0.0f;
                            for (int w2 = 0; w2 < width; ++w2) {
                                int t_src2 = t_dst - (width - 1) + w2;
                                if (t_src2 >= 0) {
                                    out_dst += x[b * seqlen * dim + t_src2 * dim + d] * weight[d * width + w2];
                                }
                            }
                            float sigmoid_dst = 1.0f / (1.0f + std::exp(-out_dst));
                            float dout_dst = dout[b * seqlen * dim + t_dst * dim + d];
                            dout_dst *= sigmoid_dst * (1.0f + out_dst * (1.0f - sigmoid_dst));
                            
                            dx_val += weight[d * width + (width - 1 - w)] * dout_dst;
                        }
                    }
                }
                
                dx[idx] = dx_val;
                
                // Compute dweight and dbias
                for (int w = 0; w < width; ++w) {
                    int t_dst = t + w;
                    if (t_dst < seqlen) {
                        float dout_use = dout[b * seqlen * dim + t_dst * dim + d];
                        
                        if (use_silu) {
                            // Recompute with SiLU gradient
                            float out_dst = (bias != nullptr) ? bias[d] : 0.0f;
                            for (int w2 = 0; w2 < width; ++w2) {
                                int t_src2 = t_dst - (width - 1) + w2;
                                if (t_src2 >= 0) {
                                    out_dst += x[b * seqlen * dim + t_src2 * dim + d] * weight[d * width + w2];
                                }
                            }
                            float sigmoid_dst = 1.0f / (1.0f + std::exp(-out_dst));
                            dout_use *= sigmoid_dst * (1.0f + out_dst * (1.0f - sigmoid_dst));
                        }
                        
                        dweight[d * width + (width - 1 - w)] += x[idx] * dout_use;
                    }
                }
                
                if (dbias != nullptr) {
                    dbias[d] += dout_val;
                }
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
    std::vector<float> h_dout(x_size);
    
    std::vector<float> h_dx_gpu(x_size, 0.0f);
    std::vector<float> h_dweight_gpu(weight_size, 0.0f);
    std::vector<float> h_dbias_gpu(cfg.use_bias ? bias_size : 0, 0.0f);
    
    std::vector<float> h_dx_cpu(x_size, 0.0f);
    std::vector<float> h_dweight_cpu(weight_size, 0.0f);
    std::vector<float> h_dbias_cpu(cfg.use_bias ? bias_size : 0, 0.0f);
    
    // Initialize with random data
    srand(42);
    for (int i = 0; i < x_size; ++i) {
        h_x[i] = (rand() % 200 - 100) / 100.0f;
        h_dout[i] = (rand() % 200 - 100) / 100.0f;
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
    float *d_x, *d_weight, *d_bias = nullptr, *d_dout;
    float *d_dx, *d_dweight, *d_dbias = nullptr;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    if (cfg.use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(&d_dout, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dx, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dweight, weight_size * sizeof(float)));
    if (cfg.use_bias) {
        HIP_CHECK(hipMalloc(&d_dbias, bias_size * sizeof(float)));
    }
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    if (cfg.use_bias) {
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemcpy(d_dout, h_dout.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    
    // Zero out gradients
    HIP_CHECK(hipMemset(d_dx, 0, x_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dweight, 0, weight_size * sizeof(float)));
    if (cfg.use_bias) {
        HIP_CHECK(hipMemset(d_dbias, 0, bias_size * sizeof(float)));
    }
    
    // Setup parameters
    ConvParamsBwd params;
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
    
    params.dout_batch_stride = cfg.seqlen * cfg.dim;
    params.dout_l_stride = cfg.dim;
    params.dout_c_stride = 1;
    
    params.dx_batch_stride = cfg.seqlen * cfg.dim;
    params.dx_l_stride = cfg.dim;
    params.dx_c_stride = 1;
    
    params.dweight_c_stride = cfg.width;
    params.dweight_width_stride = 1;
    
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = d_bias;
    params.dout_ptr = d_dout;
    params.dx_ptr = d_dx;
    params.dweight_ptr = d_dweight;
    params.dbias_ptr = d_dbias;
    
    // Run GPU kernel
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    causal_conv1d_channellast_bwd_launch<128, 4>(params, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Copy results back
    HIP_CHECK(hipMemcpy(h_dx_gpu.data(), d_dx, x_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_dweight_gpu.data(), d_dweight, weight_size * sizeof(float), hipMemcpyDeviceToHost));
    if (cfg.use_bias) {
        HIP_CHECK(hipMemcpy(h_dbias_gpu.data(), d_dbias, bias_size * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    // Run CPU reference
    causal_conv1d_channellast_bwd_cpu(
        h_x.data(), h_weight.data(),
        cfg.use_bias ? h_bias.data() : nullptr,
        h_dout.data(),
        h_dx_cpu.data(), h_dweight_cpu.data(),
        cfg.use_bias ? h_dbias_cpu.data() : nullptr,
        cfg.batch, cfg.dim, cfg.seqlen, cfg.width,
        cfg.use_silu
    );
    
    // Compare results
    const float tolerance = cfg.use_silu ? 5e-2f : 1e-3f;
    
    // Check dx
    float max_diff_dx = 0.0f;
    int num_errors_dx = 0;
    for (int i = 0; i < x_size; ++i) {
        float diff = std::abs(h_dx_gpu[i] - h_dx_cpu[i]);
        max_diff_dx = std::max(max_diff_dx, diff);
        if (diff > tolerance) {
            num_errors_dx++;
            if (num_errors_dx <= 3) {
                int b = i / (cfg.seqlen * cfg.dim);
                int remaining = i % (cfg.seqlen * cfg.dim);
                int t = remaining / cfg.dim;
                int d = remaining % cfg.dim;
                std::cout << "  dx mismatch [b=" << b << ",t=" << t << ",d=" << d << "]: "
                          << "GPU=" << h_dx_gpu[i] << " CPU=" << h_dx_cpu[i]
                          << " diff=" << diff << std::endl;
            }
        }
    }
    
    // Check dweight
    float max_diff_dweight = 0.0f;
    int num_errors_dweight = 0;
    for (int i = 0; i < weight_size; ++i) {
        float diff = std::abs(h_dweight_gpu[i] - h_dweight_cpu[i]);
        max_diff_dweight = std::max(max_diff_dweight, diff);
        if (diff > tolerance * 10) {  // dweight accumulates over batch
            num_errors_dweight++;
            if (num_errors_dweight <= 3) {
                int d = i / cfg.width;
                int w = i % cfg.width;
                std::cout << "  dweight mismatch [d=" << d << ",w=" << w << "]: "
                          << "GPU=" << h_dweight_gpu[i] << " CPU=" << h_dweight_cpu[i]
                          << " diff=" << diff << std::endl;
            }
        }
    }
    
    // Check dbias
    float max_diff_dbias = 0.0f;
    int num_errors_dbias = 0;
    if (cfg.use_bias) {
        for (int i = 0; i < bias_size; ++i) {
            float diff = std::abs(h_dbias_gpu[i] - h_dbias_cpu[i]);
            max_diff_dbias = std::max(max_diff_dbias, diff);
            if (diff > tolerance * 10) {
                num_errors_dbias++;
                if (num_errors_dbias <= 3) {
                    std::cout << "  dbias mismatch [d=" << i << "]: "
                              << "GPU=" << h_dbias_gpu[i] << " CPU=" << h_dbias_cpu[i]
                              << " diff=" << diff << std::endl;
                }
            }
        }
    }
    
    std::cout << "\n[Results]" << std::endl;
    std::cout << "  dx:      max_diff=" << std::scientific << max_diff_dx 
              << ", errors=" << num_errors_dx << "/" << x_size << std::endl;
    std::cout << "  dweight: max_diff=" << max_diff_dweight 
              << ", errors=" << num_errors_dweight << "/" << weight_size << std::endl;
    if (cfg.use_bias) {
        std::cout << "  dbias:   max_diff=" << max_diff_dbias 
                  << ", errors=" << num_errors_dbias << "/" << bias_size << std::endl;
    }
    
    bool passed = (max_diff_dx < tolerance) && 
                  (max_diff_dweight < tolerance * 10) && 
                  (!cfg.use_bias || max_diff_dbias < tolerance * 10);
    
    std::cout << "  Status: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    if (d_bias) HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_dout));
    HIP_CHECK(hipFree(d_dx));
    HIP_CHECK(hipFree(d_dweight));
    if (d_dbias) HIP_CHECK(hipFree(d_dbias));
    HIP_CHECK(hipStreamDestroy(stream));
    
    return passed;
}

// ==================== Performance Test ====================

struct PerfResult {
    float mean_ms;
    float min_ms;
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
    float *d_x, *d_weight, *d_bias = nullptr, *d_dout;
    float *d_dx, *d_dweight, *d_dbias = nullptr;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    if (cfg.use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(&d_dout, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dx, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dweight, weight_size * sizeof(float)));
    if (cfg.use_bias) {
        HIP_CHECK(hipMalloc(&d_dbias, bias_size * sizeof(float)));
    }
    
    // Setup parameters
    ConvParamsBwd params;
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
    params.dout_batch_stride = cfg.seqlen * cfg.dim;
    params.dout_l_stride = cfg.dim;
    params.dout_c_stride = 1;
    params.dx_batch_stride = cfg.seqlen * cfg.dim;
    params.dx_l_stride = cfg.dim;
    params.dx_c_stride = 1;
    params.dweight_c_stride = cfg.width;
    params.dweight_width_stride = 1;
    
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = d_bias;
    params.dout_ptr = d_dout;
    params.dx_ptr = d_dx;
    params.dweight_ptr = d_dweight;
    params.dbias_ptr = d_dbias;
    
    // Create stream and events
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < warmup; ++i) {
        causal_conv1d_channellast_bwd_launch<128, 4>(params, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    std::vector<float> times_ms(iters);
    
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(start, stream));
        causal_conv1d_channellast_bwd_launch<128, 4>(params, stream);
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        times_ms[i] = ms;
    }
    
    // Calculate statistics
    float sum = 0, min_time = times_ms[0];
    for (float t : times_ms) {
        sum += t;
        min_time = std::min(min_time, t);
    }
    float mean = sum / iters;
    
    // Calculate bandwidth
    size_t bytes_read = x_size * sizeof(float) + weight_size * sizeof(float) + x_size * sizeof(float);
    if (cfg.use_bias) bytes_read += bias_size * sizeof(float);
    size_t bytes_written = x_size * sizeof(float) + weight_size * sizeof(float);
    if (cfg.use_bias) bytes_written += bias_size * sizeof(float);
    size_t total_bytes = bytes_read + bytes_written;
    
    float bandwidth_gb_s = (total_bytes / (mean * 1e-3)) / 1e9;
    
    std::cout << "\n[Performance Results]" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Mean time:     " << mean << " ms" << std::endl;
    std::cout << "  Min time:      " << min_time << " ms" << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "  Bandwidth:     " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    if (d_bias) HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_dout));
    HIP_CHECK(hipFree(d_dx));
    HIP_CHECK(hipFree(d_dweight));
    if (d_dbias) HIP_CHECK(hipFree(d_dbias));
    HIP_CHECK(hipStreamDestroy(stream));
    
    return {mean, min_time, bandwidth_gb_s};
}

// ==================== Main ====================

int main(int argc, char* argv[]) {
    int test_mode = 0;
    if (argc > 1) {
        test_mode = std::atoi(argv[1]);
    }
    
    std::cout << "╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Channel-Last Causal Conv1D Backward Kernel Test (HIP)        ║" << std::endl;
    std::cout << "║  Test Mode: ";
    if (test_mode == 0) std::cout << "ALL";
    else if (test_mode == 1) std::cout << "Accuracy Only";
    else if (test_mode == 2) std::cout << "Performance Only";
    std::cout << std::string(70 - 14 - (test_mode == 0 ? 3 : test_mode == 1 ? 14 : 18), ' ') << "║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    
    std::vector<TestConfig> configs = {
        {"Tiny", 1, 32, 64, 4, true, false},
        {"Small", 2, 64, 256, 4, true, false},
        {"Medium", 4, 64, 512, 4, true, false},
        {"Large", 4, 64, 2048, 4, true, false},
        {"No Bias", 2, 64, 256, 4, false, false},
        {"With SiLU", 2, 64, 256, 4, true, true},
    };
    
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
    
    if (test_mode == 0 || test_mode == 2) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  PERFORMANCE TESTS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        std::vector<PerfResult> results;
        for (const auto& cfg : configs) {
            results.push_back(test_performance(cfg));
        }
        
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

