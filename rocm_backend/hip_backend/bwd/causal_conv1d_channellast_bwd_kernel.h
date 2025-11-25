/*
 * Channel-Last Backward Kernel - Header File
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

#ifndef CAUSAL_CONV1D_CHANNELLAST_BWD_KERNEL_H
#define CAUSAL_CONV1D_CHANNELLAST_BWD_KERNEL_H

#include <hip/hip_runtime.h>

// ==================== Helper Structures ====================

struct ConvParamsChannellastBwd {
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

// ==================== Kernel Launcher ====================

/**
 * Launch the channel-last backward kernel with full feature support
 * 
 * @param params: Backward convolution parameters
 * @param stream: HIP stream for kernel execution
 */
template<int kNThreads>
void causal_conv1d_channellast_bwd_launch_full(ConvParamsChannellastBwd& params, hipStream_t stream);

// Explicit instantiation declaration
extern template void causal_conv1d_channellast_bwd_launch_full<128>(ConvParamsChannellastBwd& params, hipStream_t stream);

#endif // CAUSAL_CONV1D_CHANNELLAST_BWD_KERNEL_H

