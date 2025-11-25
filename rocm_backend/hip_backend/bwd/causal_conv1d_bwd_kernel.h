/*
 * Causal Conv1D Backward Kernel - Header File (Channel-First Layout)
 * 
 * 特性：
 * - Channel-First 内存布局: [Batch, Channel, Length]
 * - 支持 FP32 和 FP16
 * - 支持 SiLU activation
 * - 向量化加载优化
 */

#ifndef CAUSAL_CONV1D_BWD_KERNEL_H
#define CAUSAL_CONV1D_BWD_KERNEL_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ============================================================================
// 辅助结构和宏
// ============================================================================

// 编译时 max
template<int N>
constexpr int constexpr_max_impl() { return N; }

template<int N1, int N2, int... Ns>
constexpr int constexpr_max_impl() {
    return constexpr_max_impl<(N1 > N2 ? N1 : N2), Ns...>();
}

template<int... Ns>
constexpr int custom_max() {
    return constexpr_max_impl<Ns...>();
}

// 编译时 min
template<int N>
constexpr int constexpr_min_impl() { return N; }

template<int N1, int N2, int... Ns>
constexpr int constexpr_min_impl() {
    return constexpr_min_impl<(N1 < N2 ? N1 : N2), Ns...>();
}

template<int... Ns>
constexpr int constexpr_min() {
    return constexpr_min_impl<Ns...>();
}

// 字节到类型的转换
template<int N> struct BytesToType {};
template<> struct BytesToType<16> { using Type = float4; };
template<> struct BytesToType<8> { using Type = float2; };
template<> struct BytesToType<4> { using Type = float; };

// half8_t 定义 (16 bytes = 8 fp16)
struct half8_t {
    __half data[8];
    
    __host__ __device__ half8_t& operator=(const half8_t& other) {
        for (int i = 0; i < 8; ++i) {
            data[i] = other.data[i];
        }
        return *this;
    }
};

// ============================================================================
// 参数结构
// ============================================================================

struct ConvParamsBwd {
    void *x_ptr;
    void *weight_ptr;
    void *bias_ptr;
    void *dout_ptr;
    void *dx_ptr;
    void *dweight_ptr;
    void *dbias_ptr;
    
    int batch, dim, seqlen, width;
    bool silu_activation;
    
    int x_batch_stride;
    int x_c_stride;
    int weight_c_stride;
    int weight_width_stride;
    int dout_batch_stride;
    int dout_c_stride;
    int dx_batch_stride;
    int dx_c_stride;
    int dweight_c_stride;
    int dweight_width_stride;
};

// ============================================================================
// Kernel Launcher 声明
// ============================================================================

/**
 * Launch causal conv1d backward kernel (Channel-First layout)
 * 
 * @param params: Backward convolution parameters
 * @param stream: HIP stream for kernel execution
 */
template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_bwd_launch(ConvParamsBwd &params, hipStream_t stream);

// 显式实例化声明
extern template void causal_conv1d_bwd_launch<256, 4, float, float>(ConvParamsBwd &params, hipStream_t stream);
extern template void causal_conv1d_bwd_launch<256, 4, __half, float>(ConvParamsBwd &params, hipStream_t stream);

#endif // CAUSAL_CONV1D_BWD_KERNEL_H

