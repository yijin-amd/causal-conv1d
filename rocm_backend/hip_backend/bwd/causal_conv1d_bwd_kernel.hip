/*
 * Causal Conv1D Backward Kernel - Implementation (Channel-First Layout)
 * 
 * 特性：
 * - Channel-First 内存布局: [Batch, Channel, Length]
 * - 支持 FP32 和 FP16
 * - 支持 SiLU activation
 * - 向量化加载优化
 */

#include "causal_conv1d_bwd_kernel.h"
#include <hipcub/hipcub.hpp>
#include <type_traits>

// ============================================================================
// Kernel Traits (Channel-First)
// ============================================================================

template<int kNThreads_, int kWidth_, bool kSiluAct_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_bwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr bool kSiluAct = kSiluAct_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts);
    
    static constexpr int kNExchangeRounds = sizeof(float) / sizeof(input_t);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    
    // 修复：使用条件类型来处理 FP16 和 FP32
    using vec_t = typename std::conditional<
        std::is_same<input_t, __half>::value,
        half8_t,
        typename BytesToType<kNBytes * kNElts>::Type
    >::type;
    
    using BlockLoadT = hipcub::BlockLoad<input_t, kNThreads, kNElts, hipcub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = hipcub::BlockLoad<vec_t, kNThreads, 1, hipcub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = hipcub::BlockStore<input_t, kNThreads, kNElts, hipcub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = hipcub::BlockStore<vec_t, kNThreads, 1, hipcub::BLOCK_STORE_DIRECT>;
    using BlockReduceFloatT = hipcub::BlockReduce<float, kNThreads>;
    
    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : custom_max<sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)>();
    
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts * (!kSiluAct ? 1 : kNExchangeRounds + 1);
    
    static constexpr int kSmemSize = custom_max<kSmemExchangeSize,
            int(sizeof(typename BlockReduceFloatT::TempStorage))>() + (kIsVecLoad ? 0 : kSmemIOSize);
};

// ============================================================================
// Backward Kernel (Channel-First)
// ============================================================================

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_bwd_kernel(ConvParamsBwd params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr bool kSiluAct = Ktraits::kSiluAct;
    static constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
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
    vec_t *smem_exchange_x = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize) + kNThreads * kNExchangeRounds;
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + dim_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + dim_id * params.weight_c_stride;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + dim_id * params.dout_c_stride;
    input_t *dx = reinterpret_cast<input_t *>(params.dx_ptr) + batch_id * params.dx_batch_stride
        + dim_id * params.dx_c_stride;
    float *dweight = reinterpret_cast<float *>(params.dweight_ptr) + dim_id * params.dweight_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[dim_id]);

    // Initialize shared memory
    if (tidx == 0) {
        if constexpr (!kSiluAct) {
            input_t zeros[kNElts] = {0};
            smem_exchange[0] = reinterpret_cast<vec_t *>(zeros)[0];
        } else {
            float zeros[kNElts] = {0};
            #pragma unroll
            for (int r = 0; r < kNExchangeRounds; ++r) {
                smem_exchange[r * kNThreads] = reinterpret_cast<vec_t *>(zeros)[r];
            }
        }
    }

    // Load weights
    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { 
        weight_vals[i] = float(weight[i * params.weight_width_stride]); 
    }

    float dweight_vals[kWidth] = {0};
    float dbias_val = 0;

    // Process in reverse order (backward pass)
    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    x += (n_chunks - 1) * kChunkSize;
    dout += (n_chunks - 1) * kChunkSize;
    dx += (n_chunks - 1) * kChunkSize;
    
    for (int chunk = n_chunks - 1; chunk >= 0; --chunk) {
        input_t x_vals_load[2 * kNElts] = {0};
        input_t dout_vals_load[2 * kNElts] = {0};
        
        // Load x and dout
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(
                reinterpret_cast<vec_t*>(x), 
                *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), 
                (params.seqlen - chunk * kChunkSize) / kNElts);
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(
                reinterpret_cast<vec_t*>(dout), 
                *reinterpret_cast<vec_t (*)[1]>(&dout_vals_load[0]), 
                (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(
                x, *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), 
                params.seqlen - chunk * kChunkSize);
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(
                dout, *reinterpret_cast<input_t (*)[kNElts]>(&dout_vals_load[0]), 
                params.seqlen - chunk * kChunkSize);
        }
        
        float dout_vals[2 * kNElts], x_vals[2 * kNElts];
        
        if constexpr (!kSiluAct) {
            // Exchange dout values between threads
            __syncthreads();
            if (tidx > 0) { 
                smem_exchange[tidx] = reinterpret_cast<vec_t *>(dout_vals_load)[0]; 
            }
            __syncthreads();
            reinterpret_cast<vec_t *>(dout_vals_load)[1] = smem_exchange[tidx < kNThreads - 1 ? tidx + 1 : 0];
            __syncthreads();
            if (tidx == 0) { 
                smem_exchange[tidx] = reinterpret_cast<vec_t *>(dout_vals_load)[0]; 
            }
            
            #pragma unroll
            for (int i = 0; i < 2 * kNElts; ++i) {
                dout_vals[i] = float(dout_vals_load[i]);
                x_vals[i] = float(x_vals_load[i]);
            }
        } else {
            // SiLU: need to load extra x values and recompute gradients
            if (tidx == 0 && chunk > 0) {
                if constexpr(kIsVecLoad) {
                    reinterpret_cast<vec_t *>(x_vals_load)[0] = reinterpret_cast<vec_t *>(x)[-1];
                } else {
                    #pragma unroll
                    for (int i = 0; i < kNElts; ++i) {
                        if (chunk * kChunkSize + i < params.seqlen) { 
                            x_vals_load[i] = x[-kNElts + i]; 
                        }
                    }
                }
            }
            
            __syncthreads();
            smem_exchange_x[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1];
            __syncthreads();
            if (tidx > 0) { 
                reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange_x[tidx - 1]; 
            }
            
            #pragma unroll
            for (int i = 0; i < 2 * kNElts; ++i) { x_vals[i] = float(x_vals_load[i]); }
            
            // Recompute output and apply SiLU gradient
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                float out_val = bias_val;
                #pragma unroll
                for (int w = 0; w < kWidth; ++w) {
                    out_val += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
                }
                float out_sigmoid_val = 1.0f / (1.0f + expf(-out_val));
                dout_vals[i] = float(dout_vals_load[i]) * out_sigmoid_val
                               * (1.0f + out_val * (1.0f - out_sigmoid_val));
            }
            
            // Exchange dout_vals
            __syncthreads();
            if (tidx > 0) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    smem_exchange[r * kNThreads + tidx] = reinterpret_cast<vec_t *>(dout_vals)[r];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int r = 0; r < kNExchangeRounds; ++r) {
                reinterpret_cast<vec_t *>(dout_vals)[kNExchangeRounds + r]
                    = smem_exchange[r * kNThreads + (tidx < kNThreads - 1 ? tidx + 1 : 0)];
            }
            __syncthreads();
            if (tidx == 0) {
                #pragma unroll
                for (int r = 0; r < kNExchangeRounds; ++r) {
                    smem_exchange[r * kNThreads + tidx] = reinterpret_cast<vec_t *>(dout_vals)[r];
                }
            }
        }
        
        dout -= kChunkSize;
        x -= kChunkSize;

        // Accumulate bias gradient
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { dbias_val += dout_vals[i]; }

        // Compute dx (input gradient)
        float dx_vals[kNElts] = {0};
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                dx_vals[i] += weight_vals[w] * dout_vals[i + kWidth - w - 1];
            }
        }

        // Store dx
        input_t dx_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { dx_vals_store[i] = input_t(dx_vals[i]); }
        
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockStoreVecT(smem_store_vec).Store(
                reinterpret_cast<vec_t*>(dx), 
                reinterpret_cast<vec_t (&)[1]>(dx_vals_store), 
                (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            typename Ktraits::BlockStoreT(smem_store).Store(
                dx, dx_vals_store, params.seqlen - chunk * kChunkSize);
        }
        dx -= kChunkSize;

        // Accumulate weight gradients
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                dweight_vals[w] += x_vals[kNElts + i] * dout_vals[i + kWidth - w - 1];
            }
        }
    }

    // Reduce and store weight gradients
    #pragma unroll
    for (int w = 0; w < kWidth; ++w) {
        __syncthreads();
        dweight_vals[w] = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dweight_vals[w]);
        if (tidx == 0) {
            atomicAdd(&dweight[w * params.dweight_width_stride], dweight_vals[w]);
        }
    }
    
    // Reduce and store bias gradient
    if (params.dbias_ptr != nullptr) {
        __syncthreads();
        dbias_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dbias_val);
        if (tidx == 0) {
            atomicAdd(&reinterpret_cast<float *>(params.dbias_ptr)[dim_id], dbias_val);
        }
    }
}

// ============================================================================
// Launch wrapper
// ============================================================================

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_bwd_launch(ConvParamsBwd &params, hipStream_t stream) {
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    
    const bool kIsVecLoad = (params.seqlen % kNElts == 0);
    const bool kSiluAct = params.silu_activation;
    
    // Macro to instantiate kernel
    #define LAUNCH_KERNEL(SiluAct, VecLoad) \
        using Ktraits = Causal_conv1d_bwd_kernel_traits<kNThreads, kWidth, SiluAct, VecLoad, input_t, weight_t>; \
        constexpr int kSmemSize = Ktraits::kSmemSize; \
        dim3 grid(params.batch, params.dim); \
        auto kernel = &causal_conv1d_bwd_kernel<Ktraits>; \
        hipLaunchKernelGGL(kernel, grid, kNThreads, kSmemSize, stream, params);
    
    if (kSiluAct) {
        if (kIsVecLoad) {
            LAUNCH_KERNEL(true, true);
        } else {
            LAUNCH_KERNEL(true, false);
        }
    } else {
        if (kIsVecLoad) {
            LAUNCH_KERNEL(false, true);
        } else {
            LAUNCH_KERNEL(false, false);
        }
    }
    
    #undef LAUNCH_KERNEL
}

// 显式实例化
// Width 2
template void causal_conv1d_bwd_launch<128, 2, float, float>(ConvParamsBwd &params, hipStream_t stream);
template void causal_conv1d_bwd_launch<128, 2, __half, float>(ConvParamsBwd &params, hipStream_t stream);
template void causal_conv1d_bwd_launch<256, 2, float, float>(ConvParamsBwd &params, hipStream_t stream);
template void causal_conv1d_bwd_launch<256, 2, __half, float>(ConvParamsBwd &params, hipStream_t stream);

// Width 3
template void causal_conv1d_bwd_launch<128, 3, float, float>(ConvParamsBwd &params, hipStream_t stream);
template void causal_conv1d_bwd_launch<128, 3, __half, float>(ConvParamsBwd &params, hipStream_t stream);
template void causal_conv1d_bwd_launch<256, 3, float, float>(ConvParamsBwd &params, hipStream_t stream);
template void causal_conv1d_bwd_launch<256, 3, __half, float>(ConvParamsBwd &params, hipStream_t stream);

// Width 4
template void causal_conv1d_bwd_launch<128, 4, float, float>(ConvParamsBwd &params, hipStream_t stream);
template void causal_conv1d_bwd_launch<128, 4, __half, float>(ConvParamsBwd &params, hipStream_t stream);
template void causal_conv1d_bwd_launch<256, 4, float, float>(ConvParamsBwd &params, hipStream_t stream);
template void causal_conv1d_bwd_launch<256, 4, __half, float>(ConvParamsBwd &params, hipStream_t stream);

