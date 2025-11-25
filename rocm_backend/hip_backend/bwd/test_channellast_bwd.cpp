/*
 * Channel-Last Backward Kernel - Test Suite
 * 
 * 测试功能：
 * - Basic: 基础功能测试（无 states, 无 seq_idx）
 * - seq_idx: 变长序列测试
 * - States: 流式处理测试（initial_states, dinitial_states, dfinal_states）
 * 
 * Compile:
 *   hipcc -O2 -std=c++17 --offload-arch=gfx942 \
 *         causal_conv1d_channellast_bwd_kernel.hip \
 *         test_channellast_bwd.cpp \
 *         -o test_channellast_bwd
 * 
 * Run:
 *   ./test_channellast_bwd [mode]
 *   mode: 0=all (default), 1=basic_only, 2=seq_idx_only, 3=states_only
 */

#include "causal_conv1d_channellast_bwd_kernel.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <string>

// ==================== Error Checking ====================

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << hipGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// ==================== CPU Reference ====================

void causal_conv1d_channellast_bwd_cpu(
    const float* x, const float* weight, const float* bias,
    const float* dout,
    const float* initial_states,
    const float* dfinal_states,
    const int32_t* seq_idx,
    float* dx, float* dweight, float* dbias,
    float* dinitial_states,
    int batch, int dim, int seqlen, int width, bool silu_activation
) {
    for (int i = 0; i < batch * seqlen * dim; ++i) dx[i] = 0.0f;
    for (int i = 0; i < dim * width; ++i) dweight[i] = 0.0f;
    if (dbias) for (int i = 0; i < dim; ++i) dbias[i] = 0.0f;
    if (dinitial_states) for (int i = 0; i < batch * (width - 1) * dim; ++i) dinitial_states[i] = 0.0f;
    
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < seqlen; ++t) {
            int32_t seq_idx_cur = 0;
            if (seq_idx) {
                seq_idx_cur = seq_idx[b * seqlen + t];
                if (seq_idx_cur < 0) continue;
            }
            
            for (int d = 0; d < dim; ++d) {
                float out_val = 0.0f;
                if (silu_activation) {
                    out_val = bias ? bias[d] : 0.0f;
                    for (int w = 0; w < width; ++w) {
                        int input_t = t - (width - 1) + w;
                        float x_val = 0.0f;
                        bool valid = false;
                        
                        if (input_t >= 0 && input_t < seqlen) {
                            if (seq_idx) {
                                int32_t seq_idx_input = seq_idx[b * seqlen + input_t];
                                valid = (seq_idx_input == seq_idx_cur);
                            } else {
                                valid = true;
                            }
                            if (valid) {
                                x_val = x[b * seqlen * dim + input_t * dim + d];
                            }
                        } else if (input_t < 0 && initial_states && !seq_idx) {
                            x_val = initial_states[b * (width - 1) * dim + (input_t + width - 1) * dim + d];
                            valid = true;
                        }
                        
                        if (valid) {
                            out_val += x_val * weight[d * width + w];
                        }
                    }
                }
                
                float dy = dout[b * seqlen * dim + t * dim + d];
                if (silu_activation) {
                    float sigmoid_val = 1.0f / (1.0f + expf(-out_val));
                    dy *= sigmoid_val * (1.0f + out_val * (1.0f - sigmoid_val));
                }
                
                for (int w = 0; w < width; ++w) {
                    int input_t = t - (width - 1) + w;
                    
                    if (input_t >= 0 && input_t < seqlen) {
                        bool valid = true;
                        if (seq_idx) {
                            int32_t seq_idx_input = seq_idx[b * seqlen + input_t];
                            valid = (seq_idx_input == seq_idx_cur);
                        }
                        if (valid) {
                            dx[b * seqlen * dim + input_t * dim + d] += weight[d * width + w] * dy;
                        }
                    } else if (input_t < 0 && dinitial_states && !seq_idx) {
                        dinitial_states[b * (width - 1) * dim + (input_t + width - 1) * dim + d] += weight[d * width + w] * dy;
                    }
                }
                
                if (dfinal_states && t >= seqlen - width + 1) {
                    int state_idx = t - (seqlen - width + 1);
                    dx[b * seqlen * dim + t * dim + d] += dfinal_states[b * (width - 1) * dim + state_idx * dim + d];
                }
                
                for (int w = 0; w < width; ++w) {
                    int input_t = t - (width - 1) + w;
                    float x_val = 0.0f;
                    bool valid = false;
                    
                    if (input_t >= 0 && input_t < seqlen) {
                        if (seq_idx) {
                            int32_t seq_idx_input = seq_idx[b * seqlen + input_t];
                            valid = (seq_idx_input == seq_idx_cur);
                        } else {
                            valid = true;
                        }
                        if (valid) {
                            x_val = x[b * seqlen * dim + input_t * dim + d];
                        }
                    } else if (input_t < 0 && initial_states && !seq_idx) {
                        x_val = initial_states[b * (width - 1) * dim + (input_t + width - 1) * dim + d];
                        valid = true;
                    }
                    
                    if (valid) {
                        dweight[d * width + w] += x_val * dy;
                    }
                }
                
                if (dbias) {
                    dbias[d] += dy;
                }
            }
        }
    }
}

// ==================== Test 1: Basic (No States, No seq_idx) ====================

bool run_test(const std::string& name, int batch, int dim, int seqlen, int width,
              bool use_bias, bool use_silu) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test: " << name << std::endl;
    std::cout << "  batch=" << batch << ", dim=" << dim << ", seqlen=" << seqlen 
              << ", width=" << width << std::endl;
    std::cout << "  bias=" << (use_bias ? "Yes" : "No") 
              << ", silu=" << (use_silu ? "Yes" : "No") << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int x_size = batch * seqlen * dim;
    const int weight_size = dim * width;
    const int bias_size = dim;
    
    std::vector<float> h_x(x_size), h_weight(weight_size), h_bias(use_bias ? bias_size : 0);
    std::vector<float> h_dout(x_size);
    std::vector<float> h_dx_cpu(x_size), h_dx_gpu(x_size);
    std::vector<float> h_dweight_cpu(weight_size), h_dweight_gpu(weight_size);
    std::vector<float> h_dbias_cpu(use_bias ? bias_size : 0), h_dbias_gpu(use_bias ? bias_size : 0);
    
    srand(42);
    for (auto& v : h_x) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_weight) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_bias) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_dout) v = (rand() % 200 - 100) / 100.0f;
    
    float *d_x, *d_weight, *d_bias, *d_dout, *d_dx, *d_dweight, *d_dbias;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dout, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dx, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dweight, weight_size * sizeof(float)));
    if (use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_dbias, bias_size * sizeof(float)));
    }
    
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dout, h_dout.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    if (use_bias) {
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    }
    
    HIP_CHECK(hipMemset(d_dx, 0, x_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dweight, 0, weight_size * sizeof(float)));
    if (use_bias) HIP_CHECK(hipMemset(d_dbias, 0, bias_size * sizeof(float)));
    
    ConvParamsChannellastBwd params;
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
    params.dout_batch_stride = seqlen * dim;
    params.dout_l_stride = dim;
    params.dout_c_stride = 1;
    params.dx_batch_stride = seqlen * dim;
    params.dx_l_stride = dim;
    params.dx_c_stride = 1;
    params.dweight_c_stride = width;
    params.dweight_width_stride = 1;
    
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = use_bias ? d_bias : nullptr;
    params.dout_ptr = d_dout;
    params.dx_ptr = d_dx;
    params.dweight_ptr = d_dweight;
    params.dbias_ptr = use_bias ? d_dbias : nullptr;
    params.initial_states_ptr = nullptr;
    params.dinitial_states_ptr = nullptr;
    params.dfinal_states_ptr = nullptr;
    params.seq_idx_ptr = nullptr;
    
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    causal_conv1d_channellast_bwd_launch_full<128>(params, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    
    HIP_CHECK(hipMemcpy(h_dx_gpu.data(), d_dx, x_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_dweight_gpu.data(), d_dweight, weight_size * sizeof(float), hipMemcpyDeviceToHost));
    if (use_bias) {
        HIP_CHECK(hipMemcpy(h_dbias_gpu.data(), d_dbias, bias_size * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    causal_conv1d_channellast_bwd_cpu(
        h_x.data(), h_weight.data(), use_bias ? h_bias.data() : nullptr,
        h_dout.data(), nullptr, nullptr, nullptr,
        h_dx_cpu.data(), h_dweight_cpu.data(), use_bias ? h_dbias_cpu.data() : nullptr,
        nullptr, batch, dim, seqlen, width, use_silu
    );
    
    auto check_error = [](const std::vector<float>& cpu, const std::vector<float>& gpu, 
                         float tol, const std::string& name) {
        float max_diff = 0.0f;
        int errors = 0;
        for (size_t i = 0; i < cpu.size(); ++i) {
            float diff = std::abs(cpu[i] - gpu[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > tol) errors++;
        }
        bool passed = errors == 0;
        std::cout << "  " << std::setw(10) << std::left << name 
                  << " max_diff=" << std::scientific << std::setprecision(2) << max_diff
                  << " errors=" << errors << "/" << cpu.size()
                  << (passed ? " ✓" : " ✗") << std::endl;
        return passed;
    };
    
    std::cout << "\nResults:" << std::endl;
    bool passed = true;
    passed &= check_error(h_dx_cpu, h_dx_gpu, 1e-5f, "dx");
    passed &= check_error(h_dweight_cpu, h_dweight_gpu, batch * seqlen * 1e-4f, "dweight");
    if (use_bias) {
        passed &= check_error(h_dbias_cpu, h_dbias_gpu, batch * seqlen * 1e-4f, "dbias");
    }
    
    std::cout << "  Status: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_dout));
    HIP_CHECK(hipFree(d_dx));
    HIP_CHECK(hipFree(d_dweight));
    if (use_bias) {
        HIP_CHECK(hipFree(d_bias));
        HIP_CHECK(hipFree(d_dbias));
    }
    HIP_CHECK(hipStreamDestroy(stream));
    
    return passed;
}

// ==================== Test 2: seq_idx ====================

bool run_test_seq_idx(const std::string& name, int batch, int dim, int seqlen, int width,
                      bool use_bias, bool use_silu) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test (seq_idx): " << name << std::endl;
    std::cout << "  batch=" << batch << ", dim=" << dim << ", seqlen=" << seqlen 
              << ", width=" << width << std::endl;
    std::cout << "  bias=" << (use_bias ? "Yes" : "No") 
              << ", silu=" << (use_silu ? "Yes" : "No") << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int x_size = batch * seqlen * dim;
    const int weight_size = dim * width;
    const int bias_size = dim;
    const int seq_idx_size = batch * seqlen;
    
    std::vector<float> h_x(x_size), h_weight(weight_size), h_bias(use_bias ? bias_size : 0);
    std::vector<float> h_dout(x_size);
    std::vector<int32_t> h_seq_idx(seq_idx_size);
    std::vector<float> h_dx_cpu(x_size), h_dx_gpu(x_size);
    std::vector<float> h_dweight_cpu(weight_size), h_dweight_gpu(weight_size);
    std::vector<float> h_dbias_cpu(use_bias ? bias_size : 0), h_dbias_gpu(use_bias ? bias_size : 0);
    
    srand(42);
    for (auto& v : h_x) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_weight) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_bias) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_dout) v = (rand() % 200 - 100) / 100.0f;
    
    // Generate seq_idx
    for (int b = 0; b < batch; ++b) {
        int pos = 0;
        int seq_id = 0;
        while (pos < seqlen) {
            int len = 10 + rand() % 20;
            len = std::min(len, seqlen - pos);
            for (int i = 0; i < len; ++i) {
                h_seq_idx[b * seqlen + pos + i] = seq_id;
            }
            pos += len;
            seq_id++;
            
            if (pos < seqlen && rand() % 5 == 0) {
                int pad_len = 1 + rand() % 5;
                pad_len = std::min(pad_len, seqlen - pos);
                for (int i = 0; i < pad_len; ++i) {
                    h_seq_idx[b * seqlen + pos + i] = -1;
                }
                pos += pad_len;
            }
        }
    }
    
    float *d_x, *d_weight, *d_bias, *d_dout, *d_dx, *d_dweight, *d_dbias;
    int32_t *d_seq_idx;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dout, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dx, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dweight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_seq_idx, seq_idx_size * sizeof(int32_t)));
    if (use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_dbias, bias_size * sizeof(float)));
    }
    
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dout, h_dout.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_seq_idx, h_seq_idx.data(), seq_idx_size * sizeof(int32_t), hipMemcpyHostToDevice));
    if (use_bias) {
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    }
    
    HIP_CHECK(hipMemset(d_dx, 0, x_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dweight, 0, weight_size * sizeof(float)));
    if (use_bias) HIP_CHECK(hipMemset(d_dbias, 0, bias_size * sizeof(float)));
    
    ConvParamsChannellastBwd params;
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
    params.dout_batch_stride = seqlen * dim;
    params.dout_l_stride = dim;
    params.dout_c_stride = 1;
    params.dx_batch_stride = seqlen * dim;
    params.dx_l_stride = dim;
    params.dx_c_stride = 1;
    params.dweight_c_stride = width;
    params.dweight_width_stride = 1;
    
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = use_bias ? d_bias : nullptr;
    params.dout_ptr = d_dout;
    params.dx_ptr = d_dx;
    params.dweight_ptr = d_dweight;
    params.dbias_ptr = use_bias ? d_dbias : nullptr;
    params.initial_states_ptr = nullptr;
    params.dinitial_states_ptr = nullptr;
    params.dfinal_states_ptr = nullptr;
    params.seq_idx_ptr = d_seq_idx;
    
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    causal_conv1d_channellast_bwd_launch_full<128>(params, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    
    HIP_CHECK(hipMemcpy(h_dx_gpu.data(), d_dx, x_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_dweight_gpu.data(), d_dweight, weight_size * sizeof(float), hipMemcpyDeviceToHost));
    if (use_bias) {
        HIP_CHECK(hipMemcpy(h_dbias_gpu.data(), d_dbias, bias_size * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    causal_conv1d_channellast_bwd_cpu(
        h_x.data(), h_weight.data(), use_bias ? h_bias.data() : nullptr,
        h_dout.data(), nullptr, nullptr, h_seq_idx.data(),
        h_dx_cpu.data(), h_dweight_cpu.data(), use_bias ? h_dbias_cpu.data() : nullptr,
        nullptr, batch, dim, seqlen, width, use_silu
    );
    
    auto check_error = [](const std::vector<float>& cpu, const std::vector<float>& gpu, 
                         float tol, const std::string& name) {
        float max_diff = 0.0f;
        int errors = 0;
        for (size_t i = 0; i < cpu.size(); ++i) {
            float diff = std::abs(cpu[i] - gpu[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > tol) errors++;
        }
        bool passed = errors == 0;
        std::cout << "  " << std::setw(10) << std::left << name 
                  << " max_diff=" << std::scientific << std::setprecision(2) << max_diff
                  << " errors=" << errors << "/" << cpu.size()
                  << (passed ? " ✓" : " ✗") << std::endl;
        return passed;
    };
    
    std::cout << "\nResults:" << std::endl;
    bool passed = true;
    passed &= check_error(h_dx_cpu, h_dx_gpu, 1e-5f, "dx");
    passed &= check_error(h_dweight_cpu, h_dweight_gpu, batch * seqlen * 1e-4f, "dweight");
    if (use_bias) {
        passed &= check_error(h_dbias_cpu, h_dbias_gpu, batch * seqlen * 1e-4f, "dbias");
    }
    
    std::cout << "  Status: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_dout));
    HIP_CHECK(hipFree(d_dx));
    HIP_CHECK(hipFree(d_dweight));
    HIP_CHECK(hipFree(d_seq_idx));
    if (use_bias) {
        HIP_CHECK(hipFree(d_bias));
        HIP_CHECK(hipFree(d_dbias));
    }
    HIP_CHECK(hipStreamDestroy(stream));
    
    return passed;
}

// ==================== Test 3: final_states ====================

bool run_test_final_states(const std::string& name, int batch, int dim, int seqlen, int width,
                           bool use_bias, bool use_silu) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test (states): " << name << std::endl;
    std::cout << "  batch=" << batch << ", dim=" << dim << ", seqlen=" << seqlen 
              << ", width=" << width << std::endl;
    std::cout << "  bias=" << (use_bias ? "Yes" : "No") 
              << ", silu=" << (use_silu ? "Yes" : "No") << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    const int x_size = batch * seqlen * dim;
    const int weight_size = dim * width;
    const int bias_size = dim;
    const int states_size = batch * (width - 1) * dim;
    
    std::vector<float> h_x(x_size), h_weight(weight_size), h_bias(use_bias ? bias_size : 0);
    std::vector<float> h_dout(x_size);
    std::vector<float> h_initial_states(states_size), h_dfinal_states(states_size);
    std::vector<float> h_dx_cpu(x_size), h_dx_gpu(x_size);
    std::vector<float> h_dweight_cpu(weight_size), h_dweight_gpu(weight_size);
    std::vector<float> h_dbias_cpu(use_bias ? bias_size : 0), h_dbias_gpu(use_bias ? bias_size : 0);
    std::vector<float> h_dinitial_states_cpu(states_size), h_dinitial_states_gpu(states_size);
    
    srand(42);
    for (auto& v : h_x) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_weight) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_bias) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_dout) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_initial_states) v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_dfinal_states) v = (rand() % 200 - 100) / 100.0f;
    
    float *d_x, *d_weight, *d_bias, *d_dout, *d_dx, *d_dweight, *d_dbias;
    float *d_initial_states, *d_dinitial_states, *d_dfinal_states;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dout, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dx, x_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dweight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_initial_states, states_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dinitial_states, states_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dfinal_states, states_size * sizeof(float)));
    if (use_bias) {
        HIP_CHECK(hipMalloc(&d_bias, bias_size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_dbias, bias_size * sizeof(float)));
    }
    
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dout, h_dout.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_initial_states, h_initial_states.data(), states_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dfinal_states, h_dfinal_states.data(), states_size * sizeof(float), hipMemcpyHostToDevice));
    if (use_bias) {
        HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), hipMemcpyHostToDevice));
    }
    
    HIP_CHECK(hipMemset(d_dx, 0, x_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dweight, 0, weight_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dinitial_states, 0, states_size * sizeof(float)));
    if (use_bias) HIP_CHECK(hipMemset(d_dbias, 0, bias_size * sizeof(float)));
    
    ConvParamsChannellastBwd params;
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
    params.dout_batch_stride = seqlen * dim;
    params.dout_l_stride = dim;
    params.dout_c_stride = 1;
    params.dx_batch_stride = seqlen * dim;
    params.dx_l_stride = dim;
    params.dx_c_stride = 1;
    params.dweight_c_stride = width;
    params.dweight_width_stride = 1;
    
    params.initial_states_batch_stride = (width - 1) * dim;
    params.initial_states_l_stride = dim;
    params.initial_states_c_stride = 1;
    params.dinitial_states_batch_stride = (width - 1) * dim;
    params.dinitial_states_l_stride = dim;
    params.dinitial_states_c_stride = 1;
    params.dfinal_states_batch_stride = (width - 1) * dim;
    params.dfinal_states_l_stride = dim;
    params.dfinal_states_c_stride = 1;
    
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = use_bias ? d_bias : nullptr;
    params.dout_ptr = d_dout;
    params.dx_ptr = d_dx;
    params.dweight_ptr = d_dweight;
    params.dbias_ptr = use_bias ? d_dbias : nullptr;
    params.initial_states_ptr = d_initial_states;
    params.dinitial_states_ptr = d_dinitial_states;
    params.dfinal_states_ptr = d_dfinal_states;
    params.seq_idx_ptr = nullptr;
    
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    causal_conv1d_channellast_bwd_launch_full<128>(params, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    
    HIP_CHECK(hipMemcpy(h_dx_gpu.data(), d_dx, x_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_dweight_gpu.data(), d_dweight, weight_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_dinitial_states_gpu.data(), d_dinitial_states, states_size * sizeof(float), hipMemcpyDeviceToHost));
    if (use_bias) {
        HIP_CHECK(hipMemcpy(h_dbias_gpu.data(), d_dbias, bias_size * sizeof(float), hipMemcpyDeviceToHost));
    }
    
    causal_conv1d_channellast_bwd_cpu(
        h_x.data(), h_weight.data(), use_bias ? h_bias.data() : nullptr,
        h_dout.data(), h_initial_states.data(), h_dfinal_states.data(), nullptr,
        h_dx_cpu.data(), h_dweight_cpu.data(), use_bias ? h_dbias_cpu.data() : nullptr,
        h_dinitial_states_cpu.data(), batch, dim, seqlen, width, use_silu
    );
    
    auto check_error = [](const std::vector<float>& cpu, const std::vector<float>& gpu, 
                         float tol, const std::string& name) {
        float max_diff = 0.0f;
        int errors = 0;
        for (size_t i = 0; i < cpu.size(); ++i) {
            float diff = std::abs(cpu[i] - gpu[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > tol) errors++;
        }
        bool passed = errors == 0;
        std::cout << "  " << std::setw(10) << std::left << name 
                  << " max_diff=" << std::scientific << std::setprecision(2) << max_diff
                  << " errors=" << errors << "/" << cpu.size()
                  << (passed ? " ✓" : " ✗") << std::endl;
        return passed;
    };
    
    std::cout << "\nResults:" << std::endl;
    bool passed = true;
    passed &= check_error(h_dx_cpu, h_dx_gpu, 1e-5f, "dx");
    passed &= check_error(h_dweight_cpu, h_dweight_gpu, batch * seqlen * 1e-4f, "dweight");
    if (use_bias) {
        passed &= check_error(h_dbias_cpu, h_dbias_gpu, batch * seqlen * 1e-4f, "dbias");
    }
    passed &= check_error(h_dinitial_states_cpu, h_dinitial_states_gpu, 1e-5f, "dinitial");
    
    std::cout << "  Status: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_dout));
    HIP_CHECK(hipFree(d_dx));
    HIP_CHECK(hipFree(d_dweight));
    HIP_CHECK(hipFree(d_initial_states));
    HIP_CHECK(hipFree(d_dinitial_states));
    HIP_CHECK(hipFree(d_dfinal_states));
    if (use_bias) {
        HIP_CHECK(hipFree(d_bias));
        HIP_CHECK(hipFree(d_dbias));
    }
    HIP_CHECK(hipStreamDestroy(stream));
    
    return passed;
}

// ==================== Main ====================

int main(int argc, char* argv[]) {
    int test_mode = 0;  // 0=all, 1=basic_only, 2=seq_idx_only, 3=states_only
    if (argc > 1) {
        test_mode = std::atoi(argv[1]);
        if (test_mode < 0 || test_mode > 3) {
            std::cerr << "Invalid test mode: " << test_mode << std::endl;
            std::cerr << "Usage: " << argv[0] << " [mode]" << std::endl;
            std::cerr << "  mode 0 or omit: Run all tests (default)" << std::endl;
            std::cerr << "  mode 1: Run basic tests only" << std::endl;
            std::cerr << "  mode 2: Run seq_idx tests only" << std::endl;
            std::cerr << "  mode 3: Run states tests only" << std::endl;
            return 1;
        }
    }
    
    std::cout << "╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Channel-Last Backward - Full Feature Test Suite              ║" << std::endl;
    if (test_mode == 0) {
        std::cout << "║  Mode: ALL (Basic + seq_idx + States)                         ║" << std::endl;
    } else if (test_mode == 1) {
        std::cout << "║  Mode: BASIC ONLY                                             ║" << std::endl;
    } else if (test_mode == 2) {
        std::cout << "║  Mode: SEQ_IDX ONLY                                           ║" << std::endl;
    } else if (test_mode == 3) {
        std::cout << "║  Mode: STATES ONLY                                            ║" << std::endl;
    }
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    int total = 0;
    int passed = 0;
    
    // PART 1: Basic Tests
    if (test_mode == 0 || test_mode == 1) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  PART 1: Basic Functionality Tests" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        total++;
        if (run_test("Tiny", 1, 32, 64, 4, true, false)) passed++;
        
        total++;
        if (run_test("Small (No Bias)", 2, 64, 128, 4, false, false)) passed++;
        
        total++;
        if (run_test("Small (With Bias)", 2, 64, 128, 4, true, false)) passed++;
        
        total++;
        if (run_test("With SiLU", 2, 64, 128, 4, true, true)) passed++;
        
        total++;
        if (run_test("Medium", 2, 128, 256, 4, true, false)) passed++;
        
        total++;
        if (run_test("Large", 2, 256, 512, 4, true, true)) passed++;
        
        total++;
        if (run_test("XLarge", 4, 512, 1024, 4, true, false)) passed++;
        
        total++;
        if (run_test("XLarge with SiLU", 4, 512, 1024, 4, true, true)) passed++;
        
        total++;
        if (run_test("XXLarge", 8, 1024, 2048, 4, true, false)) passed++;
        
        total++;
        if (run_test("XXLarge Wide Dim", 4, 2048, 1024, 4, true, true)) passed++;
    }
    
    // PART 2: seq_idx Tests
    if (test_mode == 0 || test_mode == 2) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  PART 2: seq_idx Tests (Variable-Length Sequences)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        total++;
        if (run_test_seq_idx("seq_idx Small", 2, 64, 128, 4, true, false)) passed++;
        
        total++;
        if (run_test_seq_idx("seq_idx with SiLU", 2, 64, 128, 4, true, true)) passed++;
        
        total++;
        if (run_test_seq_idx("seq_idx Medium", 2, 128, 256, 4, true, false)) passed++;
        
        total++;
        if (run_test_seq_idx("seq_idx Large", 2, 256, 512, 4, true, false)) passed++;
        
        total++;
        if (run_test_seq_idx("seq_idx XLarge", 4, 512, 1024, 4, true, true)) passed++;
    }
    
    // PART 3: States Tests
    if (test_mode == 0 || test_mode == 3) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  PART 3: States Tests (Streaming/Chunked Processing)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        total++;
        if (run_test_final_states("States Small", 2, 64, 128, 4, true, false)) passed++;
        
        total++;
        if (run_test_final_states("States with SiLU", 2, 64, 128, 4, true, true)) passed++;
        
        total++;
        if (run_test_final_states("States Medium", 2, 128, 256, 4, true, false)) passed++;
        
        total++;
        if (run_test_final_states("States Large", 2, 256, 512, 4, true, true)) passed++;
        
        total++;
        if (run_test_final_states("States XLarge", 4, 512, 1024, 4, true, false)) passed++;
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

