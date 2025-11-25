/*
 * Channel-Last Backward Kernel - Full Feature Support (HIP Implementation)
 * 
 * 支持的功能：
 * - States: initial_states, dinitial_states, dfinal_states
 * - seq_idx: 变长序列和padding支持
 * - Multi-width: width=2,3,4 (模板特化)
 * - SiLU activation
 * - Bias
 * 
 * 注意：seq_idx 和 initial_states 不能同时使用（遵循原始CUDA实现）
 * 
 * Compile:
 *   hipcc -O2 -std=c++17 --offload-arch=gfx942 test_channellast_bwd_full.cpp -o test_channellast_bwd_full
 * 
 * Run:
 *   ./test_channellast_bwd_full [mode]
 *   mode: 0=all (default), 1=basic_only, 2=seq_idx_only, 3=states_only
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
    
    // Strides
    int x_batch_stride;
    int x_l_stride;
    int x_c_stride;
    
    int weight_c_stride;
    int weight_width_stride;
    
    int dout_batch_stride;
    int dout_l_stride;
    int dout_c_stride;
    
    int dx_batch_stride;
    int dx_l_stride;
    int dx_c_stride;
    
    int dweight_c_stride;
    int dweight_width_stride;
    
    // States strides
    int initial_states_batch_stride;
    int initial_states_l_stride;
    int initial_states_c_stride;
    
    int dinitial_states_batch_stride;
    int dinitial_states_l_stride;
    int dinitial_states_c_stride;
    
    int dfinal_states_batch_stride;
    int dfinal_states_l_stride;
    int dfinal_states_c_stride;
    
    // Pointers
    float* x_ptr;
    float* weight_ptr;
    float* bias_ptr;
    float* dout_ptr;
    float* dx_ptr;
    float* dweight_ptr;
    float* dbias_ptr;
    
    // States pointers
    float* initial_states_ptr;
    float* dinitial_states_ptr;
    float* dfinal_states_ptr;
    
    // seq_idx pointer
    int32_t* seq_idx_ptr;
};

// ==================== Device Helper Functions ====================

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ==================== Channel-Last Backward Kernel (Full Features) ====================

template<int kNThreads, int kWidth, int kChunkSizeL, int kChunkSizeC, bool kHasDfinalStates, bool kHasSeqIdx>
__global__ void causal_conv1d_channellast_bwd_kernel_full(ConvParamsBwd params) {
    constexpr int kNElts = 4;
    constexpr int kNThreadsPerC = kChunkSizeC / kNElts;
    constexpr int kLPerLoad = kChunkSizeL / (kNThreads / kNThreadsPerC);
    
    __shared__ float dout_smem[kChunkSizeL + kWidth - 1][kChunkSizeC];
    __shared__ float x_smem[kWidth - 1 + kChunkSizeL + kWidth - 1][kChunkSizeC];
    __shared__ int32_t seq_idx_smem[kWidth - 1 + kChunkSizeL + kWidth - 1];
    
    const int batch_id = blockIdx.x;
    const int chunk_l_id = blockIdx.y;
    const int chunk_c_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    
    const int base_l = chunk_l_id * kChunkSizeL;
    const int base_c = chunk_c_id * kChunkSizeC;
    
    const bool use_initial_states = (params.initial_states_ptr != nullptr) && (chunk_l_id == 0);
    const bool use_dinitial_states = (params.dinitial_states_ptr != nullptr) && (chunk_l_id == 0);
    
    // Load seq_idx - need previous, current, and next kWidth-1 elements
    if constexpr (kHasSeqIdx) {
        if (c_idx == 0) {
            // Load previous kWidth-1 elements
            if (l_idx < kWidth - 1) {
                const int global_l = base_l + l_idx - (kWidth - 1);
                if (global_l >= 0 && global_l < params.seqlen) {
                    seq_idx_smem[l_idx] = params.seq_idx_ptr[batch_id * params.seqlen + global_l];
                } else {
                    seq_idx_smem[l_idx] = -1;
                }
            }
            
            // Load current chunk
            for (int l = l_idx; l < kChunkSizeL; l += kLPerLoad) {
                const int global_l = base_l + l;
                seq_idx_smem[kWidth - 1 + l] = (global_l < params.seqlen) ? 
                    params.seq_idx_ptr[batch_id * params.seqlen + global_l] : -1;
            }
            
            // Load next kWidth-1 elements
            if (l_idx < kWidth - 1) {
                const int global_l = base_l + kChunkSizeL + l_idx;
                seq_idx_smem[kWidth - 1 + kChunkSizeL + l_idx] = (global_l < params.seqlen) ?
                    params.seq_idx_ptr[batch_id * params.seqlen + global_l] : -1;
            }
        }
    }
    
    // Load dout and x to shared memory
    for (int l = l_idx; l < kChunkSizeL; l += kLPerLoad) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_l = base_l + l;
            const int global_c = base_c + c_idx * kNElts + c;
            
            if (global_l < params.seqlen && global_c < params.dim) {
                const int dout_idx = batch_id * params.dout_batch_stride + 
                                    global_l * params.dout_l_stride + 
                                    global_c * params.dout_c_stride;
                dout_smem[l][c_idx * kNElts + c] = params.dout_ptr[dout_idx];
                
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
    
    // Load boundary elements
    if (l_idx < kWidth - 1) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_l = base_l + l_idx - (kWidth - 1);
            const int global_c = base_c + c_idx * kNElts + c;
            
            float x_val = 0.0f;
            if (global_l >= 0 && global_l < params.seqlen && global_c < params.dim) {
                const int x_idx = batch_id * params.x_batch_stride + 
                                 global_l * params.x_l_stride + 
                                 global_c * params.x_c_stride;
                x_val = params.x_ptr[x_idx];
            } else if (use_initial_states && global_l < 0 && global_c < params.dim) {
                const int state_idx = batch_id * params.initial_states_batch_stride +
                                     (global_l + (kWidth - 1)) * params.initial_states_l_stride +
                                     global_c * params.initial_states_c_stride;
                x_val = params.initial_states_ptr[state_idx];
            }
            x_smem[l_idx][c_idx * kNElts + c] = x_val;
        }
    }
    
    if (l_idx < kWidth - 1) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_l = base_l + kChunkSizeL + l_idx;
            const int global_c = base_c + c_idx * kNElts + c;
            
            float dout_val = 0.0f;
            if (global_l < params.seqlen && global_c < params.dim) {
                const int dout_idx = batch_id * params.dout_batch_stride + 
                                    global_l * params.dout_l_stride + 
                                    global_c * params.dout_c_stride;
                dout_val = params.dout_ptr[dout_idx];
            }
            dout_smem[kChunkSizeL + l_idx][c_idx * kNElts + c] = dout_val;
        }
    }
    
    if (params.silu_activation && l_idx < kWidth - 1) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_l = base_l + kChunkSizeL + l_idx;
            const int global_c = base_c + c_idx * kNElts + c;
            
            float x_val = 0.0f;
            if (global_l < params.seqlen && global_c < params.dim) {
                const int x_idx = batch_id * params.x_batch_stride + 
                                 global_l * params.x_l_stride + 
                                 global_c * params.x_c_stride;
                x_val = params.x_ptr[x_idx];
            }
            x_smem[kWidth - 1 + kChunkSizeL + l_idx][c_idx * kNElts + c] = x_val;
        }
    }
    
    __syncthreads();
    
    // Compute phase
    constexpr int kLPerThread = std::min(kChunkSizeL * kChunkSizeC / kNThreads, kChunkSizeL);
    constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
    
    const int row_idx = tid / kNThreadsPerRow;
    const int col_idx = tid % kNThreadsPerRow;
    
    bool thread_valid = (base_c + row_idx < params.dim);
    
    float weight_vals[kWidth];
    float bias_val = 0.0f;
    
    if (thread_valid) {
        const int global_c = base_c + row_idx;
        if (params.bias_ptr != nullptr) {
            bias_val = params.bias_ptr[global_c];
        }
        for (int w = 0; w < kWidth; ++w) {
            const int weight_idx = global_c * params.weight_c_stride + w * params.weight_width_stride;
            weight_vals[w] = params.weight_ptr[weight_idx];
        }
    }
    
    float dout_vals[kLPerThread + kWidth - 1];
    float x_vals[kLPerThread + 2 * kWidth - 2];
    
    #pragma unroll
    for (int i = 0; i < kLPerThread + kWidth - 1; ++i) {
        dout_vals[i] = dout_smem[col_idx * kLPerThread + i][row_idx];
    }
    
    #pragma unroll
    for (int i = 0; i < kLPerThread + 2 * kWidth - 2; ++i) {
        x_vals[i] = x_smem[col_idx * kLPerThread + i][row_idx];
    }
    
    // Load seq_idx values - following CUDA implementation pattern
    // Need seq_idx for range [col_idx*kLPerThread - (kWidth-1), col_idx*kLPerThread + kLPerThread + kWidth - 1)
    int32_t seq_idx_vals[kWidth - 1 + kLPerThread + kWidth - 1];
    if constexpr (kHasSeqIdx) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread + kWidth - 1; ++i) {
            // Global position for this seq_idx element
            const int global_l = base_l + col_idx * kLPerThread + i - (kWidth - 1);
            // Index in shared memory
            // seq_idx_smem[kWidth-1] corresponds to base_l
            // So global_l corresponds to seq_idx_smem[kWidth - 1 + (global_l - base_l)]
            //                        = seq_idx_smem[kWidth - 1 + col_idx * kLPerThread + i - (kWidth - 1) - base_l]
            //                        = seq_idx_smem[col_idx * kLPerThread + i]
            const int smem_idx = col_idx * kLPerThread + i;
            
            if (global_l >= 0 && global_l < params.seqlen && 
                smem_idx >= 0 && smem_idx < kWidth - 1 + kChunkSizeL + kWidth - 1) {
                seq_idx_vals[i] = seq_idx_smem[smem_idx];
            } else {
                seq_idx_vals[i] = -1;  // Out of range or padding
            }
        }
    }
    
    // SiLU gradient - need to recompute output
    // Note: We need to apply SiLU gradient to kLPerThread + kWidth - 1 positions
    // because dx computation uses dout_vals[i + w] where w can be up to kWidth - 1
    if (params.silu_activation) {
        #pragma unroll
        for (int i = 0; i < kLPerThread + kWidth - 1; ++i) {
            const int global_l = base_l + col_idx * kLPerThread + i;
            if (global_l >= params.seqlen) continue;
            
            bool is_padding = false;
            int32_t seq_idx_cur = 0;
            if constexpr (kHasSeqIdx) {
                // seq_idx_vals is indexed starting from position -(kWidth-1) relative to col_idx*kLPerThread
                // So current output position i corresponds to seq_idx_vals[kWidth - 1 + i]
                seq_idx_cur = seq_idx_vals[i + kWidth - 1];
                is_padding = (seq_idx_cur < 0);
            }
            
            if (!is_padding && thread_valid) {
                float out_val = bias_val;
                
                // Recompute forward pass output for SiLU gradient
                #pragma unroll
                for (int w = 0; w < kWidth; ++w) {
                    if constexpr (kHasSeqIdx) {
                        // Input position w for output i: seq_idx_vals[i + w]
                        if (seq_idx_vals[i + w] == seq_idx_cur) {
                            out_val += weight_vals[w] * x_vals[i + w];
                        }
                    } else {
                        out_val += weight_vals[w] * x_vals[i + w];
                    }
                }
                
                // Apply SiLU gradient: d(silu(x))/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                float sigmoid_val = 1.0f / (1.0f + expf(-out_val));
                float silu_grad = sigmoid_val * (1.0f + out_val * (1.0f - sigmoid_val));
                dout_vals[i] *= silu_grad;
            }
        }
    }
    
    // Compute dx
    float dx_vals[kLPerThread] = {0};
    
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        const int global_l = base_l + col_idx * kLPerThread + i;
        if (global_l >= params.seqlen) continue;
        
        bool is_padding = false;
        int32_t seq_idx_cur = 0;
        if constexpr (kHasSeqIdx) {
            // For dx, current input position i has seq_idx at seq_idx_vals[i + kWidth - 1]
            seq_idx_cur = seq_idx_vals[i + kWidth - 1];
            is_padding = (seq_idx_cur < 0);
        }
        
        if (!is_padding && thread_valid) {
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                if constexpr (kHasSeqIdx) {
                    // Output position for this weight: seq_idx_vals[kWidth - 1 + i + w]
                    if (seq_idx_vals[kWidth - 1 + i + w] == seq_idx_cur) {
                        dx_vals[i] += weight_vals[kWidth - 1 - w] * dout_vals[i + w];
                    }
                } else {
                    dx_vals[i] += weight_vals[kWidth - 1 - w] * dout_vals[i + w];
                }
            }
            
            if constexpr (kHasDfinalStates) {
                if (global_l >= params.seqlen - kWidth + 1 && thread_valid) {
                    const int state_idx = batch_id * params.dfinal_states_batch_stride +
                                         (global_l - (params.seqlen - kWidth + 1)) * params.dfinal_states_l_stride +
                                         (base_c + row_idx) * params.dfinal_states_c_stride;
                    dx_vals[i] += params.dfinal_states_ptr[state_idx];
                }
            }
        }
    }
    
    // Compute dweight
    float dweight_vals[kWidth] = {0};
    
    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        #pragma unroll
        for (int i = 0; i < kLPerThread; ++i) {
            const int global_l = base_l + col_idx * kLPerThread + i;
            if (global_l >= params.seqlen) continue;
            
            bool is_padding = false;
            if constexpr (kHasSeqIdx) {
                // Output position i has seq_idx at seq_idx_vals[kWidth - 1 + i]
                is_padding = (seq_idx_vals[kWidth - 1 + i] < 0);
            }
            
            if (!is_padding && thread_valid) {
                if constexpr (kHasSeqIdx) {
                    // Input position for weight w: seq_idx_vals[i + w]
                    // Output position: seq_idx_vals[kWidth - 1 + i]
                    if (seq_idx_vals[i + w] == seq_idx_vals[kWidth - 1 + i]) {
                        dweight_vals[w] += x_vals[i + w] * dout_vals[i];
                    }
                } else {
                    dweight_vals[w] += x_vals[i + w] * dout_vals[i];
                }
            }
        }
    }
    
    // Compute dbias
    float dbias_val = 0.0f;
    if (params.bias_ptr != nullptr && thread_valid) {
        #pragma unroll
        for (int i = 0; i < kLPerThread; ++i) {
            const int global_l = base_l + col_idx * kLPerThread + i;
            if (global_l >= params.seqlen) continue;
            
            bool is_padding = false;
            if constexpr (kHasSeqIdx) {
                // Output position i has seq_idx at seq_idx_vals[kWidth - 1 + i]
                is_padding = (seq_idx_vals[kWidth - 1 + i] < 0);
            }
            
            if (!is_padding) {
                dbias_val += dout_vals[i];
            }
        }
    }
    
    // Compute dinitial_states
    float dxinit_vals[kWidth - 1] = {0};
    if (use_dinitial_states && col_idx == 0 && thread_valid) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                if (i + w >= kWidth - 1) {
                    int out_offset = i + w - (kWidth - 1);
                    if (out_offset < kLPerThread && base_l + out_offset < params.seqlen) {
                        dxinit_vals[i] += weight_vals[kWidth - 1 - w] * dout_vals[out_offset];
                    }
                }
            }
            
            if constexpr (kHasDfinalStates) {
                if (i >= params.seqlen) {
                    const int state_idx = batch_id * params.dfinal_states_batch_stride +
                                         (i - params.seqlen) * params.dfinal_states_l_stride +
                                         (base_c + row_idx) * params.dfinal_states_c_stride;
                    dxinit_vals[i] += params.dfinal_states_ptr[state_idx];
                }
            }
        }
    }
    
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        x_smem[col_idx * kLPerThread + i][row_idx] = dx_vals[i];
    }
    
    if (use_dinitial_states && col_idx == 0) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            x_smem[kChunkSizeL + i][row_idx] = dxinit_vals[i];
        }
    }
    
    __syncthreads();
    
    // Write dx
    for (int l = l_idx; l < kChunkSizeL; l += kLPerLoad) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_l = base_l + l;
            const int global_c = base_c + c_idx * kNElts + c;
            
            if (global_l < params.seqlen && global_c < params.dim) {
                const int dx_idx = batch_id * params.dx_batch_stride + 
                                  global_l * params.dx_l_stride + 
                                  global_c * params.dx_c_stride;
                params.dx_ptr[dx_idx] = x_smem[l][c_idx * kNElts + c];
            }
        }
    }
    
    // Write dweight
    if (thread_valid) {
        const int global_c = base_c + row_idx;
        for (int w = 0; w < kWidth; ++w) {
            const int dweight_idx = global_c * params.dweight_c_stride + w * params.dweight_width_stride;
            atomicAdd(&params.dweight_ptr[dweight_idx], dweight_vals[w]);
        }
    }
    
    // Write dbias
    if (params.bias_ptr != nullptr && thread_valid) {
        const int global_c = base_c + row_idx;
        atomicAdd(&params.dbias_ptr[global_c], dbias_val);
    }
    
    // Write dinitial_states
    if (use_dinitial_states && l_idx < kWidth - 1) {
        for (int c = 0; c < kNElts; ++c) {
            const int global_c = base_c + c_idx * kNElts + c;
            if (global_c < params.dim) {
                const int idx = batch_id * params.dinitial_states_batch_stride +
                               l_idx * params.dinitial_states_l_stride +
                               global_c * params.dinitial_states_c_stride;
                params.dinitial_states_ptr[idx] = x_smem[kChunkSizeL + l_idx][c_idx * kNElts + c];
            }
        }
    }
}

// ==================== Kernel Launcher ====================

template<int kNThreads, int kWidth>
void causal_conv1d_channellast_bwd_launch_width_full(ConvParamsBwd& params, hipStream_t stream) {
    constexpr int kChunkSizeL = 64;
    constexpr int kChunkSizeC = 64;
    
    const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
    const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
    
    dim3 grid(params.batch, n_chunks_L, n_chunks_C);
    dim3 block(kNThreads);
    
    if (params.seq_idx_ptr != nullptr && params.initial_states_ptr != nullptr) {
        std::cerr << "Error: seq_idx and initial_states cannot be used together!" << std::endl;
        exit(1);
    }
    
    bool has_dfinal = (params.dfinal_states_ptr != nullptr);
    bool has_seq_idx = (params.seq_idx_ptr != nullptr);
    
    if (has_dfinal && has_seq_idx) {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_bwd_kernel_full<kNThreads, kWidth, kChunkSizeL, kChunkSizeC, true, true>),
            grid, block, 0, stream, params);
    } else if (has_dfinal && !has_seq_idx) {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_bwd_kernel_full<kNThreads, kWidth, kChunkSizeL, kChunkSizeC, true, false>),
            grid, block, 0, stream, params);
    } else if (!has_dfinal && has_seq_idx) {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_bwd_kernel_full<kNThreads, kWidth, kChunkSizeL, kChunkSizeC, false, true>),
            grid, block, 0, stream, params);
    } else {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_bwd_kernel_full<kNThreads, kWidth, kChunkSizeL, kChunkSizeC, false, false>),
            grid, block, 0, stream, params);
    }
}

template<int kNThreads>
void causal_conv1d_channellast_bwd_launch_full(ConvParamsBwd& params, hipStream_t stream) {
    switch (params.width) {
        case 2:
            causal_conv1d_channellast_bwd_launch_width_full<kNThreads, 2>(params, stream);
            break;
        case 3:
            causal_conv1d_channellast_bwd_launch_width_full<kNThreads, 3>(params, stream);
            break;
        case 4:
            causal_conv1d_channellast_bwd_launch_width_full<kNThreads, 4>(params, stream);
            break;
        default:
            std::cerr << "Unsupported width: " << params.width << std::endl;
            exit(1);
    }
}

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
    
    ConvParamsBwd params;
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
    
    ConvParamsBwd params;
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
    
    ConvParamsBwd params;
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
