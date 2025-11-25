/*
 * Channel-Last Backward Kernel - Implementation File (HIP)
 * 
 * 支持的功能：
 * - States: initial_states, dinitial_states, dfinal_states
 * - seq_idx: 变长序列和padding支持
 * - Multi-width: width=2,3,4 (模板特化)
 * - SiLU activation
 * - Bias
 * 
 * 注意：seq_idx 和 initial_states 不能同时使用（遵循原始CUDA实现）
 */

#include "causal_conv1d_channellast_bwd_kernel.h"
#include <iostream>

// ==================== Device Helper Functions ====================

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ==================== Channel-Last Backward Kernel (Full Features) ====================

template<int kNThreads, int kWidth, int kChunkSizeL, int kChunkSizeC, bool kHasDfinalStates, bool kHasSeqIdx>
__global__ void causal_conv1d_channellast_bwd_kernel_full(ConvParamsChannellastBwd params) {
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
void causal_conv1d_channellast_bwd_launch_width_full(ConvParamsChannellastBwd& params, hipStream_t stream) {
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
void causal_conv1d_channellast_bwd_launch_full(ConvParamsChannellastBwd& params, hipStream_t stream) {
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

// Explicit instantiation
template void causal_conv1d_channellast_bwd_launch_full<128>(ConvParamsChannellastBwd& params, hipStream_t stream);

