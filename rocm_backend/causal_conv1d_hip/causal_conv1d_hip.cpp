/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted for ROCm/HIP
 ******************************************************************************/

#include <Python.h>
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <torch/python.h>
#include <vector>

// HIP kernel launcher declarations (extern "C" from causal_conv1d_hip_launcher.hip)
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
    hipStream_t stream);

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
    hipStream_t stream);

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
    hipStream_t stream);

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_WIDTH(WIDTH, NAME, ...)                                      \
    if (WIDTH == 2) {                                                         \
        constexpr int kWidth = 2;                                             \
        __VA_ARGS__();                                                        \
    } else if (WIDTH == 3) {                                                  \
        constexpr int kWidth = 3;                                             \
        __VA_ARGS__();                                                        \
    } else if (WIDTH == 4) {                                                  \
        constexpr int kWidth = 4;                                             \
        __VA_ARGS__();                                                        \
    } else {                                                                  \
        AT_ERROR(#NAME, " not implemented for width '", WIDTH, "'");          \
    }

// Wrapper to call HIP kernel launcher
void causal_conv1d_fwd_hip_internal(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &bias,
    at::Tensor &out,
    bool silu_activation) {
    
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "Only float32 is supported for now");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float, "Only float32 is supported for now");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA/HIP device");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA/HIP device");
    TORCH_CHECK(out.is_cuda(), "Output must be on CUDA/HIP device");

    const auto sizes = x.sizes();
    const int batch = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(-1);

    CHECK_SHAPE(x, batch, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);
    CHECK_SHAPE(out, batch, dim, seqlen);

    // Only support channel-first layout for now
    TORCH_CHECK(x.stride(2) == 1, "x must be contiguous in last dimension");
    TORCH_CHECK(out.stride(2) == 1, "out must be contiguous in last dimension");
    TORCH_CHECK(width >= 2 && width <= 4, "Only support width between 2 and 4");

    if (bias.defined()) {
        TORCH_CHECK(bias.scalar_type() == at::ScalarType::Float, "Bias must be float32");
        TORCH_CHECK(bias.is_cuda(), "Bias must be on CUDA/HIP device");
        TORCH_CHECK(bias.stride(-1) == 1, "Bias must be contiguous");
        CHECK_SHAPE(bias, dim);
    }

    // Calculate strides
    const int x_batch_stride = x.stride(0);
    const int x_c_stride = x.stride(1);
    const int weight_c_stride = weight.stride(0);
    const int weight_width_stride = weight.stride(1);
    const int out_batch_stride = out.stride(0);
    const int out_c_stride = out.stride(1);

    // Get HIP stream  
    c10::DeviceGuard device_guard(x.device());
    hipStream_t stream = 0;  // Use default stream for now

    // Get raw pointers
    const float* x_ptr = x.data_ptr<float>();
    const float* weight_ptr = weight.data_ptr<float>();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    float* out_ptr = out.data_ptr<float>();

    // Dispatch based on width
    DISPATCH_WIDTH(width, "causal_conv1d_fwd_hip", [&] {
        constexpr int kNThreads = 128;
        constexpr int kNElts = 4;
        
        // Note: We can't use the full traits struct here because it's in the .hip file
        // Instead, we call a templated launcher that will be compiled with the kernel
        if constexpr (kWidth == 2) {
            causal_conv1d_fwd_hip_launch_w2(
                x_ptr, weight_ptr, bias_ptr, out_ptr,
                batch, dim, seqlen, width,
                x_batch_stride, x_c_stride,
                weight_c_stride, weight_width_stride,
                out_batch_stride, out_c_stride,
                silu_activation, stream);
        } else if constexpr (kWidth == 3) {
            causal_conv1d_fwd_hip_launch_w3(
                x_ptr, weight_ptr, bias_ptr, out_ptr,
                batch, dim, seqlen, width,
                x_batch_stride, x_c_stride,
                weight_c_stride, weight_width_stride,
                out_batch_stride, out_c_stride,
                silu_activation, stream);
        } else if constexpr (kWidth == 4) {
            causal_conv1d_fwd_hip_launch_w4(
                x_ptr, weight_ptr, bias_ptr, out_ptr,
                batch, dim, seqlen, width,
                x_batch_stride, x_c_stride,
                weight_c_stride, weight_width_stride,
                out_batch_stride, out_c_stride,
                silu_activation, stream);
        }
    });
}

// Python-facing function
at::Tensor causal_conv1d_fwd_hip(
    const at::Tensor &x,
    const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias_,
    bool silu_activation) {
    
    // Create output tensor with same shape as input
    auto out = torch::empty_like(x);
    
    // Handle bias
    at::Tensor bias = bias_.has_value() ? bias_.value() : at::Tensor();
    
    // Call internal implementation
    causal_conv1d_fwd_hip_internal(x, weight, bias, out, silu_activation);
    
    return out;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_conv1d_fwd_hip", &causal_conv1d_fwd_hip, "Causal conv1d forward (HIP)");
}

