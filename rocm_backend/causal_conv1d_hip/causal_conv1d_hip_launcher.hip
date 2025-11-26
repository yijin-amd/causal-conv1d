/******************************************************************************
 * HIP Kernel Launcher for Causal Conv1D Forward
 ******************************************************************************/

#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

// Helper function
template<typename T>
constexpr T const_max(T a, T b) {
    return (a > b) ? a : b;
}

// Kernel traits - must match the definition in causal_conv1d_kernel.hip
template<int kNThreads_, int kWidth_, int kNElts_>
struct CausalConv1dKernelTraits {
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNElts = kNElts_;
    static constexpr int kChunkSize = kNThreads * kNElts;
    
    using vec_t = float4;  // 4 floats for vectorization
    
    // BlockLoad and BlockStore for coalesced memory access
    using BlockLoadT = hipcub::BlockLoad<float, kNThreads, kNElts, hipcub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = hipcub::BlockStore<float, kNThreads, kNElts, hipcub::BLOCK_STORE_WARP_TRANSPOSE>;
    
    static constexpr int kSmemIOSize = const_max(sizeof(typename BlockLoadT::TempStorage), 
                                                  sizeof(typename BlockStoreT::TempStorage));
    static constexpr int kSmemExchangeSize = kNThreads * sizeof(vec_t);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

// Include the kernel definition
#include "causal_conv1d_kernel.hip"

// Template launcher function - Width 2
extern "C" void causal_conv1d_fwd_hip_launch_w2(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int batch,
    int dim,
    int seqlen,
    int width,
    int x_batch_stride,
    int x_c_stride,
    int weight_c_stride,
    int weight_width_stride,
    int out_batch_stride,
    int out_c_stride,
    bool use_silu,
    hipStream_t stream) {
    
    constexpr int kNThreads = 128;
    constexpr int kWidth = 2;
    constexpr int kNElts = 4;
    using Ktraits = CausalConv1dKernelTraits<kNThreads, kWidth, kNElts>;
    
    dim3 grid(batch, dim);
    dim3 block(kNThreads);
    size_t smem_size = Ktraits::kSmemSize;
    
    hipLaunchKernelGGL(
        causal_conv1d_fwd_kernel<Ktraits>,
        grid, block, smem_size, stream,
        x, weight, bias, out,
        batch, dim, seqlen, width,
        x_batch_stride, x_c_stride,
        weight_c_stride, weight_width_stride,
        out_batch_stride, out_c_stride,
        use_silu
    );
}

// Template launcher function - Width 3
extern "C" void causal_conv1d_fwd_hip_launch_w3(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int batch,
    int dim,
    int seqlen,
    int width,
    int x_batch_stride,
    int x_c_stride,
    int weight_c_stride,
    int weight_width_stride,
    int out_batch_stride,
    int out_c_stride,
    bool use_silu,
    hipStream_t stream) {
    
    constexpr int kNThreads = 128;
    constexpr int kWidth = 3;
    constexpr int kNElts = 4;
    using Ktraits = CausalConv1dKernelTraits<kNThreads, kWidth, kNElts>;
    
    dim3 grid(batch, dim);
    dim3 block(kNThreads);
    size_t smem_size = Ktraits::kSmemSize;
    
    hipLaunchKernelGGL(
        causal_conv1d_fwd_kernel<Ktraits>,
        grid, block, smem_size, stream,
        x, weight, bias, out,
        batch, dim, seqlen, width,
        x_batch_stride, x_c_stride,
        weight_c_stride, weight_width_stride,
        out_batch_stride, out_c_stride,
        use_silu
    );
}

// Template launcher function - Width 4
extern "C" void causal_conv1d_fwd_hip_launch_w4(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int batch,
    int dim,
    int seqlen,
    int width,
    int x_batch_stride,
    int x_c_stride,
    int weight_c_stride,
    int weight_width_stride,
    int out_batch_stride,
    int out_c_stride,
    bool use_silu,
    hipStream_t stream) {
    
    constexpr int kNThreads = 128;
    constexpr int kWidth = 4;
    constexpr int kNElts = 4;
    using Ktraits = CausalConv1dKernelTraits<kNThreads, kWidth, kNElts>;
    
    dim3 grid(batch, dim);
    dim3 block(kNThreads);
    size_t smem_size = Ktraits::kSmemSize;
    
    hipLaunchKernelGGL(
        causal_conv1d_fwd_kernel<Ktraits>,
        grid, block, smem_size, stream,
        x, weight, bias, out,
        batch, dim, seqlen, width,
        x_batch_stride, x_c_stride,
        weight_c_stride, weight_width_stride,
        out_batch_stride, out_c_stride,
        use_silu
    );
}

