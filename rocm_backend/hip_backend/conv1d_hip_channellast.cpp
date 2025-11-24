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

// ==================== Main ====================

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Channel-Last Causal Conv1D for AMD MI308                     ║" << std::endl;
    std::cout << "║  Basic Functionality Test (No seq_idx, No states)             ║" << std::endl;
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
    
    // Run tests
    int total = 0;
    int passed = 0;
    
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
    
    // Summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Total:  " << total << std::endl;
    std::cout << "  Passed: " << passed << " ✓" << std::endl;
    std::cout << "  Failed: " << (total - passed) << " ✗" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return (passed == total) ? 0 : 1;
}

