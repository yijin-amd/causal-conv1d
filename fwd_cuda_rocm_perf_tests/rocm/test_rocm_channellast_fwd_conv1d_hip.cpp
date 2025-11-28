#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

#include "causal_conv1d_kernel_channellast.hip"

// ==================== Helper Macros ====================

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "  " << hipGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==================== CPU Reference Implementation ====================

void causal_conv1d_channellast_cpu_ref(
    const float* x,      // [batch, seqlen, dim] - channel-last layout
    const float* weight, // [dim, width]
    const float* bias,   // [dim] or nullptr
    float* out,          // [batch, seqlen, dim]
    int batch,
    int dim,
    int seqlen,
    int width,
    bool use_silu
) {
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seqlen; ++t) {
            for (int d = 0; d < dim; ++d) {
                float acc = 0.0f;
                
                // Causal convolution
                for (int w = 0; w < width; ++w) {
                    int input_t = t - (width - 1) + w;
                    
                    if (input_t >= 0) {
                        // Channel-last layout: x[batch, seqlen, dim]
                        int input_idx = b * (seqlen * dim) + input_t * dim + d;
                        acc += x[input_idx] * weight[d * width + w];
                    }
                }
                
                // Add bias
                if (bias != nullptr) {
                    acc += bias[d];
                }
                
                // Apply SiLU: x / (1 + exp(-x))
                if (use_silu) {
                    acc = acc / (1.0f + expf(-acc));
                }
                
                int out_idx = b * (seqlen * dim) + t * dim + d;
                out[out_idx] = acc;
            }
        }
    }
}

// ==================== Test Function ====================

bool run_test(const char* name, int batch, int dim, int seqlen, int width, 
              bool use_bias, bool use_silu) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test: " << name << std::endl;
    std::cout << "  Config: batch=" << batch << ", dim=" << dim 
              << ", seqlen=" << seqlen << ", width=" << width << std::endl;
    std::cout << "  Bias=" << (use_bias ? "Yes" : "No") 
              << ", SiLU=" << (use_silu ? "Yes" : "No") << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Sizes
    const int x_size = batch * seqlen * dim;
    const int weight_size = dim * width;
    const int bias_size = dim;
    
    // Allocate host memory
    std::vector<float> h_x(x_size);
    std::vector<float> h_weight(weight_size);
    std::vector<float> h_bias(use_bias ? bias_size : 0);
    std::vector<float> h_out_gpu(x_size, 0.0f);
    std::vector<float> h_out_cpu(x_size, 0.0f);
    
    // Initialize with simple pattern
    std::cout << "Initializing data..." << std::endl;
    for (int i = 0; i < x_size; ++i) {
        h_x[i] = 0.01f * (i % 100 - 50);  // Range: -0.5 to 0.49
    }
    for (int i = 0; i < weight_size; ++i) {
        h_weight[i] = 0.1f * (i % 10);  // Range: 0.0 to 0.9
    }
    if (use_bias) {
        for (int i = 0; i < bias_size; ++i) {
            h_bias[i] = 0.01f * (i % 20);  // Small bias
        }
    }
    
    // Allocate device memory
    std::cout << "Allocating GPU memory..." << std::endl;
    float *d_x, *d_weight, *d_bias = nullptr, *d_out;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    if (use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(&d_out, x_size * sizeof(float)));
    
    // Copy to device
    std::cout << "Copying data to GPU..." << std::endl;
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    if (use_bias) {
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    }
    
    // Setup parameters
    ConvParamsChannelLast params;
    
    // Basic parameters
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = use_silu;
    
    // Channel-last layout: [batch, seqlen, dim]
    params.x_batch_stride = seqlen * dim;
    params.x_l_stride = dim;
    params.x_c_stride = 1;
    
    params.weight_c_stride = width;
    params.weight_width_stride = 1;
    
    params.out_batch_stride = seqlen * dim;
    params.out_l_stride = dim;
    params.out_c_stride = 1;
    
    // Data pointers
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = d_bias;
    params.out_ptr = d_out;
    
    // Not using seq_idx or states in basic tests
    params.seq_idx_ptr = nullptr;
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = nullptr;
    
    // Create stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    
    // Launch kernel
    std::cout << "Launching GPU kernel..." << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    causal_conv1d_channellast_fwd_launch<128, 4>(params, stream);
    HIP_CHECK(hipGetLastError());  // Check for kernel launch errors
    
    HIP_CHECK(hipStreamSynchronize(stream));
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    
    std::cout << "  GPU time: " << std::fixed << std::setprecision(3) 
              << gpu_time.count() / 1000.0 << " ms" << std::endl;
    
    // Copy results back
    std::cout << "Copying results from GPU..." << std::endl;
    HIP_CHECK(hipMemcpy(h_out_gpu.data(), d_out, x_size * sizeof(float), hipMemcpyDeviceToHost));
    
    // Compute CPU reference
    std::cout << "Computing CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    causal_conv1d_channellast_cpu_ref(
        h_x.data(), h_weight.data(), 
        use_bias ? h_bias.data() : nullptr,
        h_out_cpu.data(),
        batch, dim, seqlen, width, use_silu
    );
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    std::cout << "  CPU time: " << std::fixed << std::setprecision(3) 
              << cpu_time.count() / 1000.0 << " ms" << std::endl;
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int num_errors = 0;
    const float tolerance = use_silu ? 1e-3f : 1e-4f;
    
    for (int i = 0; i < x_size; ++i) {
        float diff = std::abs(h_out_gpu[i] - h_out_cpu[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        
        if (diff > tolerance) {
            if (num_errors < 5) {
                int b = i / (seqlen * dim);
                int remaining = i % (seqlen * dim);
                int t = remaining / dim;
                int d = remaining % dim;
                std::cout << "  Mismatch [b=" << b << ",t=" << t << ",d=" << d << "]: "
                          << "GPU=" << std::scientific << h_out_gpu[i] 
                          << " CPU=" << h_out_cpu[i]
                          << " diff=" << diff << std::endl;
            }
            num_errors++;
        }
    }
    
    float avg_diff = sum_diff / x_size;
    float speedup = (float)cpu_time.count() / gpu_time.count();
    
    // Print results
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Max difference:  " << std::scientific << max_diff << std::endl;
    std::cout << "  Avg difference:  " << avg_diff << std::endl;
    std::cout << "  Speedup:         " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
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

// ==================== Test with seq_idx ====================

// CPU reference with seq_idx support
void causal_conv1d_channellast_cpu_ref_with_seq_idx(
    const float* x,      // [batch, seqlen, dim]
    const float* weight, // [dim, width]
    const float* bias,   // [dim] or nullptr
    const int32_t* seq_idx, // [batch, seqlen]
    float* out,          // [batch, seqlen, dim]
    int batch,
    int dim,
    int seqlen,
    int width,
    bool use_silu
) {
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seqlen; ++t) {
            const int seq_idx_cur = seq_idx[b * seqlen + t];
            
            for (int d = 0; d < dim; ++d) {
                float acc = 0.0f;
                
                // Skip computation for padding positions (seq_idx < 0)
                if (seq_idx_cur < 0) {
                    int out_idx = b * (seqlen * dim) + t * dim + d;
                    out[out_idx] = 0.0f;
                    continue;
                }
                
                // Causal convolution with sequence boundary checking
                for (int w = 0; w < width; ++w) {
                    int input_t = t - (width - 1) + w;
                    
                    if (input_t >= 0) {
                        int input_seq_idx = seq_idx[b * seqlen + input_t];
                        // Only accumulate if within same sub-sequence
                        if (input_seq_idx == seq_idx_cur) {
                            int input_idx = b * (seqlen * dim) + input_t * dim + d;
                            acc += x[input_idx] * weight[d * width + w];
                        }
                    }
                }
                
                // Add bias
                if (bias != nullptr) {
                    acc += bias[d];
                }
                
                // Apply SiLU
                if (use_silu) {
                    acc = acc / (1.0f + expf(-acc));
                }
                
                int out_idx = b * (seqlen * dim) + t * dim + d;
                out[out_idx] = acc;
            }
        }
    }
}

bool run_test_seq_idx(const char* name, int batch, int dim, int seqlen, int width,
                      bool use_bias, bool use_silu) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test (seq_idx): " << name << std::endl;
    std::cout << "  Config: batch=" << batch << ", dim=" << dim 
              << ", seqlen=" << seqlen << ", width=" << width << std::endl;
    std::cout << "  Bias=" << (use_bias ? "Yes" : "No") 
              << ", SiLU=" << (use_silu ? "Yes" : "No") << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int x_size = batch * seqlen * dim;
    const int weight_size = dim * width;
    const int bias_size = dim;
    const int seq_idx_size = batch * seqlen;
    
    // Allocate host memory
    std::vector<float> h_x(x_size);
    std::vector<float> h_weight(weight_size);
    std::vector<float> h_bias(use_bias ? bias_size : 0);
    std::vector<int32_t> h_seq_idx(seq_idx_size);
    std::vector<float> h_out_gpu(x_size, 0.0f);
    std::vector<float> h_out_cpu(x_size, 0.0f);
    
    std::cout << "Initializing data with seq_idx..." << std::endl;
    
    // Initialize input
    for (int i = 0; i < x_size; ++i) {
        h_x[i] = 0.01f * (i % 100 - 50);
    }
    for (int i = 0; i < weight_size; ++i) {
        h_weight[i] = 0.1f * (i % 10);
    }
    if (use_bias) {
        for (int i = 0; i < bias_size; ++i) {
            h_bias[i] = 0.01f * (i % 20);
        }
    }
    
    // Initialize seq_idx with multiple sub-sequences per batch
    // Example: [0, 0, 0, 1, 1, 1, 2, 2, ...] or with -1 for padding
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seqlen; ++t) {
            int idx = b * seqlen + t;
            // Create sub-sequences of length ~seqlen/4, with some padding
            if (t < seqlen / 8) {
                h_seq_idx[idx] = -1;  // Padding at start
            } else {
                h_seq_idx[idx] = (t - seqlen / 8) / (seqlen / 4);
            }
        }
    }
    
    // Allocate device memory
    std::cout << "Allocating GPU memory..." << std::endl;
    float *d_x, *d_weight, *d_bias = nullptr, *d_out;
    int32_t *d_seq_idx;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_seq_idx, seq_idx_size * sizeof(int32_t)));
    if (use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(&d_out, x_size * sizeof(float)));
    
    // Copy to device
    std::cout << "Copying data to GPU..." << std::endl;
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_seq_idx, h_seq_idx.data(), seq_idx_size * sizeof(int32_t), hipMemcpyHostToDevice));
    if (use_bias) {
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    }
    
    // Setup parameters
    ConvParamsChannelLast params;
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = use_silu;
    
    params.x_batch_stride = seqlen * dim;
    params.x_l_stride = dim;
    params.x_c_stride = 1;
    params.weight_c_stride = width;
    params.weight_width_stride = 1;
    params.out_batch_stride = seqlen * dim;
    params.out_l_stride = dim;
    params.out_c_stride = 1;
    
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = d_bias;
    params.out_ptr = d_out;
    params.seq_idx_ptr = d_seq_idx;  // Enable seq_idx
    params.initial_states_ptr = nullptr;
    params.final_states_ptr = nullptr;
    
    // Create stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    
    // Launch kernel
    std::cout << "Launching GPU kernel with seq_idx..." << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    causal_conv1d_channellast_fwd_launch<128, 4>(params, stream);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
    
    std::cout << "  GPU time: " << std::fixed << std::setprecision(3) 
              << gpu_time.count() / 1000.0 << " ms" << std::endl;
    
    // Copy results back
    std::cout << "Copying results from GPU..." << std::endl;
    HIP_CHECK(hipMemcpy(h_out_gpu.data(), d_out, x_size * sizeof(float), hipMemcpyDeviceToHost));
    
    // Compute CPU reference
    std::cout << "Computing CPU reference with seq_idx..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    causal_conv1d_channellast_cpu_ref_with_seq_idx(
        h_x.data(), h_weight.data(), 
        use_bias ? h_bias.data() : nullptr,
        h_seq_idx.data(),
        h_out_cpu.data(),
        batch, dim, seqlen, width, use_silu
    );
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    std::cout << "  CPU time: " << std::fixed << std::setprecision(3) 
              << cpu_time.count() / 1000.0 << " ms" << std::endl;
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int num_errors = 0;
    const float tolerance = use_silu ? 1e-3f : 1e-4f;
    
    for (int i = 0; i < x_size; ++i) {
        float diff = std::abs(h_out_gpu[i] - h_out_cpu[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        
        if (diff > tolerance) {
            if (num_errors < 5) {
                int b = i / (seqlen * dim);
                int remaining = i % (seqlen * dim);
                int t = remaining / dim;
                int d = remaining % dim;
                std::cout << "  Mismatch [b=" << b << ",t=" << t << ",d=" << d 
                          << ",seq_idx=" << h_seq_idx[b * seqlen + t] << "]: "
                          << "GPU=" << std::scientific << h_out_gpu[i] 
                          << " CPU=" << h_out_cpu[i]
                          << " diff=" << diff << std::endl;
            }
            num_errors++;
        }
    }
    
    float avg_diff = sum_diff / x_size;
    float speedup = (float)cpu_time.count() / gpu_time.count();
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Max difference:  " << std::scientific << max_diff << std::endl;
    std::cout << "  Avg difference:  " << avg_diff << std::endl;
    std::cout << "  Speedup:         " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "  Errors (>" << tolerance << "): " << num_errors << " / " << x_size << std::endl;
    
    bool passed = (max_diff < tolerance);
    std::cout << "  Status: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_seq_idx));
    if (d_bias) HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipStreamDestroy(stream));
    
    return passed;
}

// ==================== Test with final_states ====================

bool run_test_final_states(const char* name, int batch, int dim, int seqlen, int width,
                           bool use_bias, bool use_silu) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test (final_states): " << name << std::endl;
    std::cout << "  Config: batch=" << batch << ", dim=" << dim 
              << ", seqlen=" << seqlen << ", width=" << width << std::endl;
    std::cout << "  Bias=" << (use_bias ? "Yes" : "No") 
              << ", SiLU=" << (use_silu ? "Yes" : "No") << std::endl;
    std::cout << "  Testing chunked processing with states" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int x_size = batch * seqlen * dim;
    const int weight_size = dim * width;
    const int bias_size = dim;
    const int states_size = batch * (width - 1) * dim;
    
    // Allocate host memory
    std::vector<float> h_x(x_size);
    std::vector<float> h_weight(weight_size);
    std::vector<float> h_bias(use_bias ? bias_size : 0);
    std::vector<float> h_out_full(x_size, 0.0f);      // Full sequence output
    std::vector<float> h_out_chunked(x_size, 0.0f);   // Chunked output
    std::vector<float> h_final_states(states_size, 0.0f);
    
    std::cout << "Initializing data..." << std::endl;
    
    for (int i = 0; i < x_size; ++i) {
        h_x[i] = 0.01f * (i % 100 - 50);
    }
    for (int i = 0; i < weight_size; ++i) {
        h_weight[i] = 0.1f * (i % 10);
    }
    if (use_bias) {
        for (int i = 0; i < bias_size; ++i) {
            h_bias[i] = 0.01f * (i % 20);
        }
    }
    
    // Allocate device memory
    std::cout << "Allocating GPU memory..." << std::endl;
    float *d_x, *d_weight, *d_bias = nullptr, *d_out, *d_states;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_states, states_size * sizeof(float)));
    if (use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
    }
    HIP_CHECK(hipMalloc(&d_out, x_size * sizeof(float)));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    if (use_bias) {
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    }
    
    // Create stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    
    // ========== Step 1: Process full sequence as baseline ==========
    std::cout << "\n[Step 1] Processing full sequence..." << std::endl;
    
    ConvParamsChannelLast params_full;
    params_full.batch = batch;
    params_full.dim = dim;
    params_full.seqlen = seqlen;
    params_full.width = width;
    params_full.silu_activation = use_silu;
    
    params_full.x_batch_stride = seqlen * dim;
    params_full.x_l_stride = dim;
    params_full.x_c_stride = 1;
    params_full.weight_c_stride = width;
    params_full.weight_width_stride = 1;
    params_full.out_batch_stride = seqlen * dim;
    params_full.out_l_stride = dim;
    params_full.out_c_stride = 1;
    
    params_full.x_ptr = d_x;
    params_full.weight_ptr = d_weight;
    params_full.bias_ptr = d_bias;
    params_full.out_ptr = d_out;
    params_full.seq_idx_ptr = nullptr;
    params_full.initial_states_ptr = nullptr;
    params_full.final_states_ptr = nullptr;
    
    causal_conv1d_channellast_fwd_launch<128, 4>(params_full, stream);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    
    HIP_CHECK(hipMemcpy(h_out_full.data(), d_out, x_size * sizeof(float), hipMemcpyDeviceToHost));
    std::cout << "  Full sequence processing complete" << std::endl;
    
    // ========== Step 2: Process in two chunks with states ==========
    std::cout << "\n[Step 2] Processing in chunks with states..." << std::endl;
    
    int chunk1_len = seqlen / 2;
    int chunk2_len = seqlen - chunk1_len;
    
    std::cout << "  Chunk 1 length: " << chunk1_len << std::endl;
    std::cout << "  Chunk 2 length: " << chunk2_len << std::endl;
    
    // Zero the states
    HIP_CHECK(hipMemset(d_states, 0, states_size * sizeof(float)));
    
    // Process Chunk 1
    std::cout << "  Processing chunk 1..." << std::endl;
    ConvParamsChannelLast params_chunk1;
    params_chunk1.batch = batch;
    params_chunk1.dim = dim;
    params_chunk1.seqlen = chunk1_len;
    params_chunk1.width = width;
    params_chunk1.silu_activation = use_silu;
    
    params_chunk1.x_batch_stride = seqlen * dim;
    params_chunk1.x_l_stride = dim;
    params_chunk1.x_c_stride = 1;
    params_chunk1.weight_c_stride = width;
    params_chunk1.weight_width_stride = 1;
    params_chunk1.out_batch_stride = seqlen * dim;
    params_chunk1.out_l_stride = dim;
    params_chunk1.out_c_stride = 1;
    
    params_chunk1.x_ptr = d_x;
    params_chunk1.weight_ptr = d_weight;
    params_chunk1.bias_ptr = d_bias;
    params_chunk1.out_ptr = d_out;
    params_chunk1.seq_idx_ptr = nullptr;
    params_chunk1.initial_states_ptr = nullptr;
    params_chunk1.final_states_ptr = d_states;  // Save final states
    params_chunk1.final_states_batch_stride = (width - 1) * dim;
    params_chunk1.final_states_l_stride = dim;
    params_chunk1.final_states_c_stride = 1;
    
    causal_conv1d_channellast_fwd_launch<128, 4>(params_chunk1, stream);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Process Chunk 2 using states from Chunk 1
    std::cout << "  Processing chunk 2 with initial states..." << std::endl;
    ConvParamsChannelLast params_chunk2;
    params_chunk2.batch = batch;
    params_chunk2.dim = dim;
    params_chunk2.seqlen = chunk2_len;
    params_chunk2.width = width;
    params_chunk2.silu_activation = use_silu;
    
    params_chunk2.x_batch_stride = seqlen * dim;
    params_chunk2.x_l_stride = dim;
    params_chunk2.x_c_stride = 1;
    params_chunk2.weight_c_stride = width;
    params_chunk2.weight_width_stride = 1;
    params_chunk2.out_batch_stride = seqlen * dim;
    params_chunk2.out_l_stride = dim;
    params_chunk2.out_c_stride = 1;
    
    // Point to second chunk of input/output
    params_chunk2.x_ptr = reinterpret_cast<float*>(d_x) + chunk1_len * dim;
    params_chunk2.weight_ptr = d_weight;
    params_chunk2.bias_ptr = d_bias;
    params_chunk2.out_ptr = reinterpret_cast<float*>(d_out) + chunk1_len * dim;
    params_chunk2.seq_idx_ptr = nullptr;
    params_chunk2.initial_states_ptr = d_states;  // Use states from chunk 1
    params_chunk2.initial_states_batch_stride = (width - 1) * dim;
    params_chunk2.initial_states_l_stride = dim;
    params_chunk2.initial_states_c_stride = 1;
    params_chunk2.final_states_ptr = nullptr;
    
    causal_conv1d_channellast_fwd_launch<128, 4>(params_chunk2, stream);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    
    HIP_CHECK(hipMemcpy(h_out_chunked.data(), d_out, x_size * sizeof(float), hipMemcpyDeviceToHost));
    std::cout << "  Chunked processing complete" << std::endl;
    
    // Copy final states back for inspection
    HIP_CHECK(hipMemcpy(h_final_states.data(), d_states, states_size * sizeof(float), hipMemcpyDeviceToHost));
    
    // ========== Verify that chunked output matches full output ==========
    std::cout << "\n[Verification] Comparing full vs chunked outputs..." << std::endl;
    
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int num_errors = 0;
    const float tolerance = use_silu ? 1e-3f : 1e-4f;
    
    for (int i = 0; i < x_size; ++i) {
        float diff = std::abs(h_out_full[i] - h_out_chunked[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        
        if (diff > tolerance) {
            if (num_errors < 5) {
                int b = i / (seqlen * dim);
                int remaining = i % (seqlen * dim);
                int t = remaining / dim;
                int d = remaining % dim;
                std::cout << "  Mismatch [b=" << b << ",t=" << t << ",d=" << d << "]: "
                          << "Full=" << std::scientific << h_out_full[i] 
                          << " Chunked=" << h_out_chunked[i]
                          << " diff=" << diff << std::endl;
            }
            num_errors++;
        }
    }
    
    float avg_diff = sum_diff / x_size;
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "  Max difference:  " << std::scientific << max_diff << std::endl;
    std::cout << "  Avg difference:  " << avg_diff << std::endl;
    std::cout << "  Errors (>" << tolerance << "): " << num_errors << " / " << x_size << std::endl;
    
    bool passed = (max_diff < tolerance);
    std::cout << "  Status: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    if (passed) {
        std::cout << "\n  ✓ Chunked processing with states produces identical results!" << std::endl;
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_states));
    if (d_bias) HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipStreamDestroy(stream));
    
    return passed;
}

// ==================== Main ====================

int main(int argc, char* argv[]) {
    // Parse command line arguments
    // 0 or no arg: run all tests
    // 1: run only basic functionality tests
    // 2: run only seq_idx tests
    // 3: run only final_states tests
    int test_mode = 0;  // default: all
    if (argc > 1) {
        test_mode = std::atoi(argv[1]);
        if (test_mode < 0 || test_mode > 3) {
            std::cerr << "Invalid test mode: " << test_mode << std::endl;
            std::cerr << "Usage: " << argv[0] << " [mode]" << std::endl;
            std::cerr << "  mode 0 or omit: Run all tests (default)" << std::endl;
            std::cerr << "  mode 1: Run basic functionality tests only" << std::endl;
            std::cerr << "  mode 2: Run seq_idx tests only" << std::endl;
            std::cerr << "  mode 3: Run final_states tests only" << std::endl;
            return 1;
        }
    }
    
    std::cout << "╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Channel-Last Causal Conv1D for AMD MI308                     ║" << std::endl;
    if (test_mode == 0) {
        std::cout << "║  Comprehensive Test Suite (Basic + seq_idx + states)          ║" << std::endl;
    } else if (test_mode == 1) {
        std::cout << "║  Test Suite: Basic Functionality Only                         ║" << std::endl;
    } else if (test_mode == 2) {
        std::cout << "║  Test Suite: seq_idx Tests Only                               ║" << std::endl;
    } else if (test_mode == 3) {
        std::cout << "║  Test Suite: States Tests Only                                ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    // Check GPU
    std::cout << "Checking GPU..." << std::endl;
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    
    if (err != hipSuccess || device_count == 0) {
        std::cerr << "✗ Error: No GPU found!" << std::endl;
        std::cerr << "  " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "✓ Found " << device_count << " GPU(s)" << std::endl;
    
    // Get device properties
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    
    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "  Name:              " << prop.name << std::endl;
    std::cout << "  Compute Units:     " << prop.multiProcessorCount << std::endl;
    std::cout << "  Wavefront Size:    " << prop.warpSize << std::endl;
    std::cout << "  Global Memory:     " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "  Shared Mem/Block:  " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << std::endl;
    
    // Run tests based on mode
    int total = 0;
    int passed = 0;
    
    // PART 1: Basic Functionality Tests
    if (test_mode == 0 || test_mode == 1) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  PART 1: Basic Functionality Tests" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        // Test 1: Tiny (sanity check)
        total++;
        if (run_test("Tiny Test", 1, 32, 64, 4, true, false)) passed++;
        
        // Test 2: Small without bias
        total++;
        if (run_test("Small (No Bias)", 2, 64, 128, 4, false, false)) passed++;
        
        // Test 3: Small with bias
        total++;
        if (run_test("Small (With Bias)", 2, 64, 128, 4, true, false)) passed++;
        
        // Test 4: With SiLU
        total++;
        if (run_test("With SiLU", 2, 64, 128, 4, true, true)) passed++;
        
        // Test 5: Medium size
        total++;
        if (run_test("Medium Size", 2, 128, 256, 4, true, false)) passed++;
        
        // Test 6: Larger dimensions
        total++;
        if (run_test("Large Dimensions", 2, 256, 512, 4, true, true)) passed++;
    }
    
    // PART 2: seq_idx Tests
    if (test_mode == 0 || test_mode == 2) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  PART 2: seq_idx Tests (Sub-sequence Handling)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        // Test 7: seq_idx with small size
        total++;
        if (run_test_seq_idx("seq_idx Small", 2, 64, 128, 4, true, false)) passed++;
        
        // Test 8: seq_idx with SiLU
        total++;
        if (run_test_seq_idx("seq_idx with SiLU", 2, 64, 128, 4, true, true)) passed++;
        
        // Test 9: seq_idx with larger size
        total++;
        if (run_test_seq_idx("seq_idx Medium", 2, 128, 256, 4, true, false)) passed++;
    }
    
    // PART 3: States Tests
    if (test_mode == 0 || test_mode == 3) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  PART 3: States Tests (Streaming/Chunked Processing)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        // Test 10: final_states with small size
        total++;
        if (run_test_final_states("States Small", 2, 64, 128, 4, true, false)) passed++;
        
        // Test 11: final_states with SiLU
        total++;
        if (run_test_final_states("States with SiLU", 2, 64, 128, 4, true, true)) passed++;
        
        // Test 12: final_states with larger size
        total++;
        if (run_test_final_states("States Medium", 2, 128, 256, 4, true, false)) passed++;
    }
    
    // Summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Total:  " << total << std::endl;
    std::cout << "  Passed: " << passed << " ✓" << std::endl;
    std::cout << "  Failed: " << (total - passed) << " ✗" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return (passed == total) ? 0 : 1;
}

