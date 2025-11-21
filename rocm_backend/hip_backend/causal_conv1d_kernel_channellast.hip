// Channel-Last Causal Conv1D Kernel for AMD MI308
// 
// Features:
// - Optimized for 64-wide wavefronts (CDNA3 architecture)
// - Support for seq_idx (sub-sequence handling)
// - Support for initial_states/final_states (streaming/chunked processing)
// - Currently supports float32 only

#include <hip/hip_runtime.h>

// Helper for constexpr min
template<int kChunkSizeL>
constexpr int constexpr_min(int a, int b) {
    return (a < b) ? a : b;
}

// Byte-to-type mapping for vectorized loads
template<int N> struct BytesToType {};
template<> struct BytesToType<16> { using Type = float4; };

// ==================== Parameter Structure ====================

struct ConvParamsChannelLast {
    using index_t = uint32_t;
    
    // Basic parameters
    int batch, dim, seqlen, width;
    bool silu_activation;
    
    // Strides
    index_t x_batch_stride, x_l_stride, x_c_stride;
    index_t weight_c_stride, weight_width_stride;
    index_t out_batch_stride, out_l_stride, out_c_stride;
    
    // Data pointers
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;
    
    // Sequence index for handling variable-length sequences or sub-sequences
    int32_t *__restrict__ seq_idx_ptr;
    
    // Initial and final states for streaming/chunked processing
    // No __restrict__ since initial_states could be the same as final_states
    void * initial_states_ptr;
    index_t initial_states_batch_stride;
    index_t initial_states_l_stride;
    index_t initial_states_c_stride;
    
    void * final_states_ptr;
    index_t final_states_batch_stride;
    index_t final_states_l_stride;
    index_t final_states_c_stride;
};

// ==================== Kernel Traits ====================

template<int kNThreads_, int kWidth_, int kChunkSizeL_>
struct Causal_conv1d_channellast_fwd_kernel_traits {
    using input_t = float;
    using weight_t = float;
    
    static constexpr int kNThreads = kNThreads_;
    static_assert(kNThreads % 64 == 0, "MI308 uses 64-wide wavefronts");
    static constexpr int kNWarps = kNThreads / 64;  // AMD MI308 wavefront size
    static constexpr int kWidth = kWidth_;
    static constexpr int kChunkSizeL = kChunkSizeL_;
    static constexpr int kNBytes = sizeof(input_t);  // 4 for float
    
    // Vectorization: 128-byte cache line, 16 bytes per load (float4)
    static constexpr int kNElts = 4;  // float4
    static constexpr int kNEltsPerRow = 128 / kNBytes;  // 32 elements per cache line
    static constexpr int kNThreadsPerRow = kNEltsPerRow / kNElts;  // 8 threads per row
    static_assert(kNThreadsPerRow * kNBytes * kNElts == 128, "Cache line alignment");
    
    static constexpr int kNColsPerWarp = 64 / kNThreadsPerRow;  // MI308: 64-wide wavefront, so 8 cols per warp
    static_assert(kNColsPerWarp * kNThreadsPerRow == 64, "Wavefront coverage");
    static constexpr int kNColsPerLoad = kNColsPerWarp * kNWarps;
    static constexpr int kNLoads = kChunkSizeL / kNColsPerLoad;
    static_assert(kNLoads * kNColsPerLoad == kChunkSizeL, "Chunk coverage");
    
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;  // float4
};

// ==================== Channel-Last Forward Kernel ====================

template<typename Ktraits, bool kHasSeqIdx>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_channellast_fwd_kernel(ConvParamsChannelLast params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNThreadsPerC = Ktraits::kNThreadsPerRow;
    constexpr int kLPerLoad = Ktraits::kNColsPerLoad;
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory with padding to avoid LDS bank conflicts
    __shared__ input_t x_smem[kWidth - 1 + kChunkSizeL][kChunkSizeC + kNElts];

    // Block and thread indices
    const int batch_id = blockIdx.x;
    const int chunk_l_id = blockIdx.y;
    const int chunk_c_id = blockIdx.z;
    const int tid = threadIdx.x;
    
    // Thread mapping for loading (channel-wise)
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    
    // Global memory pointers
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.x_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr)
        + chunk_c_id * kChunkSizeC * params.weight_c_stride;
    
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.out_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    
    // Sequence index pointer (for handling sub-sequences)
    int *seq_idx = !kHasSeqIdx ? nullptr : reinterpret_cast<int *>(params.seq_idx_ptr)
        + batch_id * params.seqlen + chunk_l_id * kChunkSizeL;
    
    // Initial states pointer (for first L-chunk only)
    input_t *initial_states = params.initial_states_ptr == nullptr || chunk_l_id > 0 ? nullptr
        : reinterpret_cast<input_t *>(params.initial_states_ptr) + batch_id * params.initial_states_batch_stride 
        + l_idx * params.initial_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    
    // Final states pointer (for last L-chunk only)
    // The last L-chunk will have enough info to write to final states, since it also contains
    // a few x values from the previous L-chunk.
    input_t *final_states = params.final_states_ptr == nullptr || chunk_l_id < gridDim.y - 1 ? nullptr
        : reinterpret_cast<input_t *>(params.final_states_ptr) + batch_id * params.final_states_batch_stride 
        + l_idx * params.final_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;

    // ==================== Load input data into shared memory ====================
    
    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t x_vals_load[kNElts] = {0};
        
        // Vectorized load (128-byte aligned)
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + l * kLPerLoad * params.x_l_stride);
        }
        
        reinterpret_cast<vec_t *>(x_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx] = 
            reinterpret_cast<vec_t *>(x_vals_load)[0];
    }
    
    // Load elements from previous chunk (for causal dependency)
    if (l_idx < kWidth - 1) {
        input_t x_vals_load[kNElts] = {0};
        
        // Try to load from previous positions in the sequence
        if (chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) >= 0
            && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            // Load from global memory (previous part of sequence)
            reinterpret_cast<vec_t *>(x_vals_load)[0] = 
                *reinterpret_cast<vec_t *>(x - (kWidth - 1) * params.x_l_stride);
        } else if (initial_states != nullptr
                   && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < 0
                   && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            // Load from initial_states (for first chunk)
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(initial_states);
        }
        
        reinterpret_cast<vec_t *>(x_smem[l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }

    __syncthreads();
    
    // Write final states (for last L-chunk only)
    if (final_states != nullptr
        && l_idx < kWidth - 1
        && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
        // x_smem[0] contains element at index chunk_l_id * kChunkSizeL - (kWidth - 1)
        // So last few elements (index params.seqlen - kWidth + 1 + l_idx) are stored in
        // x_smem[params.seqlen + l_idx - chunk_l_id * kChunkSizeL]
        *reinterpret_cast<vec_t *>(final_states) = 
            reinterpret_cast<vec_t *>(x_smem[params.seqlen + l_idx - chunk_l_id * kChunkSizeL])[c_idx];
    }

    // ==================== Compute convolution ====================
    
    constexpr int kLPerThread = constexpr_min<kChunkSizeL>(kChunkSizeL * kChunkSizeC / kNThreads, kChunkSizeL);
    static_assert(kLPerThread * kNThreads == kChunkSizeL * kChunkSizeC);
    constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
    static_assert(kNThreadsPerRow * kLPerThread == kChunkSizeL);
    
    // Power of 2 checks for efficiency
    static_assert((kChunkSizeL & (kChunkSizeL - 1)) == 0);
    static_assert((kLPerThread & (kLPerThread - 1)) == 0);
    static_assert((kNThreadsPerRow & (kNThreadsPerRow - 1)) == 0);
    static_assert(kNThreadsPerRow <= 64);

    // Thread remapping for computation (time-wise)
    const int row_idx = tid / kNThreadsPerRow;  // Channel index
    const int col_idx = tid % kNThreadsPerRow;  // Time index

    // Load bias
    float bias_val = params.bias_ptr == nullptr || chunk_c_id * kChunkSizeC + row_idx >= params.dim 
        ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[chunk_c_id * kChunkSizeC + row_idx]);
    
    // Load weights into registers
    float weight_vals[kWidth] = {0};
    if (chunk_c_id * kChunkSizeC + row_idx < params.dim) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            weight_vals[w] = float(weight[row_idx * params.weight_c_stride + w * params.weight_width_stride]);
        }
    }
    
    // Load input values from shared memory
    float x_vals[kWidth - 1 + kLPerThread];
    #pragma unroll
    for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
        x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
    }
    
    // Load sequence indices (for handling sub-sequences)
    int seq_idx_thread[kWidth - 1 + kLPerThread];
    if constexpr (kHasSeqIdx) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
            // seq_idx is -1 for positions before the start of the sequence
            seq_idx_thread[i] = chunk_l_id * kChunkSizeL + col_idx * kLPerThread + i - (kWidth - 1) >= 0 
                ? seq_idx[col_idx * kLPerThread + i - (kWidth - 1)] : -1;
        }
    }

    // Perform causal convolution
    float out_vals[kLPerThread];
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        out_vals[i] = bias_val;
        
        const int seq_idx_cur = !kHasSeqIdx ? 0 : seq_idx_thread[i + kWidth - 1];
        
        // For padding tokens (seq_idx < 0), skip computation and set output to 0
        if (seq_idx_cur < 0) {
            out_vals[i] = 0.f;
            continue;
        }
        
        // Convolve
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            if constexpr (!kHasSeqIdx) {
                // Normal case: no sequence boundaries
                out_vals[i] += weight_vals[w] * x_vals[i + w];
            } else {
                // With seq_idx: only accumulate if within same sub-sequence
                out_vals[i] += seq_idx_thread[i + w] == seq_idx_cur ? weight_vals[w] * x_vals[i + w] : 0.f;
            }
        }
        
        // Apply SiLU activation: x / (1 + exp(-x))
        if (params.silu_activation) {
            out_vals[i] = out_vals[i] / (1.f + expf(-out_vals[i]));
        }
    }

    // ==================== Write output ====================
    
    __syncthreads();
    
    // Store to shared memory (transposed for coalesced writes)
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        x_smem[col_idx * kLPerThread + i][row_idx] = static_cast<input_t>(out_vals[i]);
    }
    
    __syncthreads();
    
    // Vectorized write to global memory
    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t out_vals_store[kNElts];
        reinterpret_cast<vec_t *>(out_vals_store)[0] = reinterpret_cast<vec_t *>(x_smem[l * kLPerLoad + l_idx])[c_idx];
        
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            *reinterpret_cast<vec_t *>(out + l * kLPerLoad * params.out_l_stride) = 
                reinterpret_cast<vec_t *>(out_vals_store)[0];
        }
    }
}

// ==================== Launch Helper ====================

template<int kNThreads, int kWidth>
void causal_conv1d_channellast_fwd_launch(ConvParamsChannelLast &params, hipStream_t stream) {
    using Ktraits = Causal_conv1d_channellast_fwd_kernel_traits<kNThreads, kWidth, 64>;
    
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    
    const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
    const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
    
    dim3 grid(params.batch, n_chunks_L, n_chunks_C);
    dim3 block(Ktraits::kNThreads);
    
    // Dispatch based on whether seq_idx is present
    if (params.seq_idx_ptr != nullptr) {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_fwd_kernel<Ktraits, true>),
            grid, block, 0, stream,
            params
        );
    } else {
        hipLaunchKernelGGL(
            (causal_conv1d_channellast_fwd_kernel<Ktraits, false>),
            grid, block, 0, stream,
            params
        );
    }
}

