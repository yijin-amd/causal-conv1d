/******************************************************************************
 * Optimized Causal Conv1D Kernel for ROCm/HIP
 * Based on CUDA implementation with vectorized load/store support
 ******************************************************************************/

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

// Helper for type conversion based on byte size
template<int N> struct BytesToType {};
template<> struct BytesToType<16> { using Type = float4; };
template<> struct BytesToType<8> { using Type = float2; };
template<> struct BytesToType<4> { using Type = float; };

// Helper for constexpr max
constexpr int custom_max(std::initializer_list<int> vals) {
    int max_val = 0;
    for (int v : vals) {
        if (v > max_val) max_val = v;
    }
    return max_val;
}

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Parameter structure for unified interface
struct ConvParamsBase {
    void *x_ptr;
    void *weight_ptr;
    void *bias_ptr;
    void *out_ptr;
    
    int batch;
    int dim;
    int seqlen;
    int width;
    
    int x_batch_stride;
    int x_c_stride;
    int weight_c_stride;
    int weight_width_stride;
    int out_batch_stride;
    int out_c_stride;
    
    bool silu_activation;
};

// Kernel traits with vectorized load support
template<int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct causal_conv1d_fwd_opt_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    
    // BlockLoad/Store types
    using BlockLoadT = hipcub::BlockLoad<input_t, kNThreads, kNElts, hipcub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = hipcub::BlockLoad<vec_t, kNThreads, 1, hipcub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = hipcub::BlockStore<input_t, kNThreads, kNElts, hipcub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = hipcub::BlockStore<vec_t, kNThreads, 1, hipcub::BLOCK_STORE_DIRECT>;
    
    // Shared memory size calculation
    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : custom_max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

// Optimized Causal Conv1D kernel with vectorized load support
template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_fwd_opt_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    
    // Setup pointers
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    // Thread 0 initializes the boundary
    if (tidx == 0) {
        input_t zeros[kNElts] = {0};
        smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t *>(zeros)[0];
    }

    // Load weights into registers
    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { 
        weight_vals[i] = float(weight[i * params.weight_width_stride]); 
    }

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        input_t x_vals_load[2 * kNElts] = {0};
        
        // Vectorized or standard load based on alignment
        if constexpr(kIsVecLoad) {
            // Vectorized load - faster for aligned sequences
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(
                reinterpret_cast<vec_t*>(x), 
                *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), 
                (params.seqlen - chunk * kChunkSize) / kNElts
            );
        } else {
            // Standard load - works for any sequence length
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(
                x, 
                *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), 
                params.seqlen - chunk * kChunkSize
            );
        }
        x += kChunkSize;
        __syncthreads();
        
        // Share data with neighboring threads for causal convolution
        if (tidx < kNThreads - 1) { 
            smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; 
        }
        __syncthreads();
        reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
        __syncthreads();
        if (tidx == kNThreads - 1) { 
            smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; 
        }

        // Convert to float for computation
        float x_vals[2 * kNElts];
        #pragma unroll
        for (int i = 0; i < 2 * kNElts; ++i) { 
            x_vals[i] = float(x_vals_load[i]); 
        }

        // Perform causal convolution
        float out_vals[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            out_vals[i] = bias_val;
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                out_vals[i] += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
            }
        }

        // Apply SiLU activation if requested
        if (params.silu_activation) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
            }
        }

        // Convert back to input_t for storage
        input_t out_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { 
            out_vals_store[i] = out_vals[i]; 
        }
        
        // Vectorized or standard store
        if constexpr(kIsVecLoad) {
            // Vectorized store
            typename Ktraits::BlockStoreVecT(smem_store_vec).Store(
                reinterpret_cast<vec_t*>(out), 
                reinterpret_cast<vec_t (&)[1]>(out_vals_store), 
                (params.seqlen - chunk * kChunkSize) / kNElts
            );
        } else {
            // Standard store
            typename Ktraits::BlockStoreT(smem_store).Store(
                out, 
                out_vals_store, 
                params.seqlen - chunk * kChunkSize
            );
        }
        out += kChunkSize;
    }
}

// Launch function with runtime optimization selection
template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_fwd_opt_launch(ConvParamsBase &params, hipStream_t stream) {
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    
    // Runtime check: use vectorized load if sequence length is aligned
    bool is_vec_load = (params.seqlen % kNElts == 0);
    
    if (is_vec_load) {
        using Ktraits = causal_conv1d_fwd_opt_kernel_traits<kNThreads, kWidth, true, input_t, weight_t>;
        constexpr int kSmemSize = Ktraits::kSmemSize;
        dim3 grid(params.batch, params.dim);
        
        auto kernel = &causal_conv1d_fwd_opt_kernel<Ktraits>;
        
        // Configure dynamic shared memory if needed
        if (kSmemSize >= 48 * 1024) {
            hipError_t err = hipFuncSetAttribute((void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
            (void)err; // Suppress warning - this is a non-op on ROCm <= 6.1
        }
        
        hipLaunchKernelGGL(kernel, grid, Ktraits::kNThreads, kSmemSize, stream, params);
    } else {
        using Ktraits = causal_conv1d_fwd_opt_kernel_traits<kNThreads, kWidth, false, input_t, weight_t>;
        constexpr int kSmemSize = Ktraits::kSmemSize;
        dim3 grid(params.batch, params.dim);
        
        auto kernel = &causal_conv1d_fwd_opt_kernel<Ktraits>;
        
        if (kSmemSize >= 48 * 1024) {
            hipError_t err = hipFuncSetAttribute((void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, kSmemSize);
            (void)err; // Suppress warning - this is a non-op on ROCm <= 6.1
        }
        
        hipLaunchKernelGGL(kernel, grid, Ktraits::kNThreads, kSmemSize, stream, params);
    }
}

// Top-level dispatch function
template<typename input_t, typename weight_t>
void causal_conv1d_fwd_hip(ConvParamsBase &params, hipStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_fwd_opt_launch<128, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_fwd_opt_launch<128, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_fwd_opt_launch<128, 4, input_t, weight_t>(params, stream);
    }
}

// Explicit template instantiations for float
template void causal_conv1d_fwd_hip<float, float>(ConvParamsBase &params, hipStream_t stream);

// Simple naive kernel for comparison
__global__ void causal_conv1d_kernel_naive(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int batch,
    int dim,
    int seqlen,
    int width,
    bool use_silu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * dim * seqlen;
    
    if (idx >= total_elements) return;
    
    int t = idx % seqlen;
    int d = (idx / seqlen) % dim;
    int b = idx / (dim * seqlen);
    
    float acc = 0.0f;
    
    for (int i = 0; i < width; ++i) {
        int input_t = t - width + 1 + i;
        if (input_t >= 0) {
            int input_idx = b * (dim * seqlen) + d * seqlen + input_t;
            acc += x[input_idx] * weight[d * width + i];
        }
    }
    
    if (bias != nullptr) {
        acc += bias[d];
    }
    
    if (use_silu) {
        acc = silu(acc);
    }
    
    out[idx] = acc;
}

