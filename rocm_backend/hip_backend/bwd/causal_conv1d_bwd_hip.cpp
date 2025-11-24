// causal_conv1d_bwd_hip.cpp
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <iostream>
#include <cmath>
#include <cstring>
#include <type_traits>

// ============================================================================
// 辅助结构和宏
// ============================================================================

#define HIP_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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
};

// 删除了 59-61 行的错误特化

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

// ============================================================================
// Test driver
// ============================================================================

template<typename T>
void fill_random(T* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    for (int i = 0; i < size; ++i) {
        data[i] = T(min_val + (max_val - min_val) * (rand() / float(RAND_MAX)));
    }
}

void test_conv1d_backward() {
    const int batch = 4;
    const int dim = 64;
    const int seqlen = 2048;
    const int width = 4;
    const bool silu_activation = true;
    
    using input_t = float;
    using weight_t = float;
    
    // Host memory
    const int x_size = batch * dim * seqlen;
    const int weight_size = dim * width;
    const int grad_size = x_size;
    
    input_t *h_x = new input_t[x_size];
    weight_t *h_weight = new weight_t[weight_size];
    weight_t *h_bias = new weight_t[dim];
    input_t *h_dout = new input_t[grad_size];
    input_t *h_dx = new input_t[grad_size];
    float *h_dweight = new float[weight_size];
    float *h_dbias = new float[dim];
    
    // Fill with random data
    srand(42);
    fill_random(h_x, x_size);
    fill_random(h_weight, weight_size);
    fill_random(h_bias, dim);
    fill_random(h_dout, grad_size);
    memset(h_dx, 0, grad_size * sizeof(input_t));
    memset(h_dweight, 0, weight_size * sizeof(float));
    memset(h_dbias, 0, dim * sizeof(float));
    
    // Device memory
    input_t *d_x, *d_dout, *d_dx;
    weight_t *d_weight, *d_bias;
    float *d_dweight, *d_dbias;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(input_t)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(weight_t)));
    HIP_CHECK(hipMalloc(&d_bias, dim * sizeof(weight_t)));
    HIP_CHECK(hipMalloc(&d_dout, grad_size * sizeof(input_t)));
    HIP_CHECK(hipMalloc(&d_dx, grad_size * sizeof(input_t)));
    HIP_CHECK(hipMalloc(&d_dweight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dbias, dim * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_x, h_x, x_size * sizeof(input_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight, weight_size * sizeof(weight_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias, h_bias, dim * sizeof(weight_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dout, h_dout, grad_size * sizeof(input_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dx, h_dx, grad_size * sizeof(input_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dweight, h_dweight, weight_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dbias, h_dbias, dim * sizeof(float), hipMemcpyHostToDevice));
    
    // Setup params
    ConvParamsBwd params;
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = d_bias;
    params.dout_ptr = d_dout;
    params.dx_ptr = d_dx;
    params.dweight_ptr = d_dweight;
    params.dbias_ptr = d_dbias;
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.silu_activation = silu_activation;
    params.x_batch_stride = dim * seqlen;
    params.x_c_stride = seqlen;
    params.weight_c_stride = width;
    params.weight_width_stride = 1;
    params.dout_batch_stride = dim * seqlen;
    params.dout_c_stride = seqlen;
    params.dx_batch_stride = dim * seqlen;
    params.dx_c_stride = seqlen;
    params.dweight_c_stride = width;
    params.dweight_width_stride = 1;
    
    // Launch kernel
    std::cout << "Launching backward kernel..." << std::endl;
    std::cout << "  Batch: " << batch << ", Dim: " << dim << ", SeqLen: " << seqlen << ", Width: " << width << std::endl;
    std::cout << "  SiLU: " << (silu_activation ? "Yes" : "No") << std::endl;
    
    causal_conv1d_bwd_launch<128, 4, input_t, weight_t>(params, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy results back
    HIP_CHECK(hipMemcpy(h_dx, d_dx, grad_size * sizeof(input_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_dweight, d_dweight, weight_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_dbias, d_dbias, dim * sizeof(float), hipMemcpyDeviceToHost));
    
    // Print sample results
    std::cout << "\n✓ Backward pass completed!" << std::endl;
    std::cout << "\nSample dx (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "  dx[" << i << "] = " << h_dx[i] << std::endl;
    }
    
    std::cout << "\nSample dweight (first channel, all widths):" << std::endl;
    for (int w = 0; w < width; ++w) {
        std::cout << "  dweight[0][" << w << "] = " << h_dweight[w] << std::endl;
    }
    
    std::cout << "\nSample dbias (first 5 channels):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  dbias[" << i << "] = " << h_dbias[i] << std::endl;
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_dout));
    HIP_CHECK(hipFree(d_dx));
    HIP_CHECK(hipFree(d_dweight));
    HIP_CHECK(hipFree(d_dbias));
    
    delete[] h_x;
    delete[] h_weight;
    delete[] h_bias;
    delete[] h_dout;
    delete[] h_dx;
    delete[] h_dweight;
    delete[] h_dbias;
}

// ============================================================================
// 性能测试工具 (Timer 类的析构函数修复)
// ============================================================================

class Timer {
private:
    hipEvent_t start_, stop_;
    
public:
    Timer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }
    
    ~Timer() {
        HIP_CHECK(hipEventDestroy(start_));  // 修复：添加错误检查
        HIP_CHECK(hipEventDestroy(stop_));   // 修复：添加错误检查
    }
    
    void start() {
        HIP_CHECK(hipEventRecord(start_));
    }
    
    float stop() {
        HIP_CHECK(hipEventRecord(stop_));
        HIP_CHECK(hipEventSynchronize(stop_));
        float milliseconds = 0;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }
};

// ============================================================================
// CPU 参考实现（用于正确性验证）
// ============================================================================

void cpu_causal_conv1d_backward(
    const float* x,
    const float* weight,
    const float* bias,
    const float* dout,
    float* dx,
    float* dweight,
    float* dbias,
    int batch, int dim, int seqlen, int width,
    bool silu_activation
) {
    // Initialize gradients to zero
    const int x_size = batch * dim * seqlen;
    const int weight_size = dim * width;
    memset(dx, 0, x_size * sizeof(float));
    memset(dweight, 0, weight_size * sizeof(float));
    memset(dbias, 0, dim * sizeof(float));
    
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < dim; ++c) {
            for (int t = 0; t < seqlen; ++t) {
                int idx = b * dim * seqlen + c * seqlen + t;
                
                // Forward pass to compute output (needed for SiLU)
                float out = bias ? bias[c] : 0.0f;
                for (int w = 0; w < width; ++w) {
                    int t_in = t - (width - 1 - w);
                    if (t_in >= 0) {
                        int x_idx = b * dim * seqlen + c * seqlen + t_in;
                        out += weight[c * width + w] * x[x_idx];
                    }
                }
                
                // Apply SiLU gradient if needed
                float grad_out = dout[idx];
                if (silu_activation) {
                    float sigmoid = 1.0f / (1.0f + expf(-out));
                    grad_out *= sigmoid * (1.0f + out * (1.0f - sigmoid));
                }
                
                // Bias gradient
                if (dbias) {
                    dbias[c] += grad_out;
                }
                
                // Weight and input gradients
                for (int w = 0; w < width; ++w) {
                    int t_in = t - (width - 1 - w);
                    if (t_in >= 0) {
                        int x_idx = b * dim * seqlen + c * seqlen + t_in;
                        // Weight gradient
                        dweight[c * width + w] += x[x_idx] * grad_out;
                        // Input gradient
                        dx[x_idx] += weight[c * width + w] * grad_out;
                    }
                }
            }
        }
    }
}

// ============================================================================
// 测试用例配置
// ============================================================================

struct TestConfig {
    std::string name;
    int batch;
    int dim;
    int seqlen;
    int width;
    bool silu_activation;
    
    TestConfig(std::string n, int b, int d, int s, int w, bool silu = false)
        : name(n), batch(b), dim(d), seqlen(s), width(w), silu_activation(silu) {}
};

// ============================================================================
// 单个测试用例（修复版本）
// ============================================================================

template<typename input_t, typename weight_t>
bool run_test_case(const TestConfig& config, bool verify_correctness = true, int num_iters = 100) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Test: " << config.name << std::endl;
    std::cout << "  Config: [B=" << config.batch << ", D=" << config.dim 
              << ", L=" << config.seqlen << ", W=" << config.width << "]" << std::endl;
    std::cout << "  SiLU: " << (config.silu_activation ? "Yes" : "No") << std::endl;
    std::cout << "  Data Type: " << (sizeof(input_t) == 4 ? "FP32" : "FP16") << std::endl;
    
    // Sizes
    const int x_size = config.batch * config.dim * config.seqlen;
    const int weight_size = config.dim * config.width;
    
    // Host memory
    float *h_x = new float[x_size];
    float *h_weight = new float[weight_size];
    float *h_bias = new float[config.dim];
    float *h_dout = new float[x_size];
    float *h_dx_gpu = new float[x_size];
    float *h_dweight_gpu = new float[weight_size];
    float *h_dbias_gpu = new float[config.dim];
    
    // Fill with random data
    srand(42);
    for (int i = 0; i < x_size; ++i) h_x[i] = -1.0f + 2.0f * (rand() / float(RAND_MAX));
    for (int i = 0; i < weight_size; ++i) h_weight[i] = -1.0f + 2.0f * (rand() / float(RAND_MAX));
    for (int i = 0; i < config.dim; ++i) h_bias[i] = -0.5f + 1.0f * (rand() / float(RAND_MAX));
    for (int i = 0; i < x_size; ++i) h_dout[i] = -1.0f + 2.0f * (rand() / float(RAND_MAX));
    
    // Device memory
    input_t *d_x, *d_dout, *d_dx;
    weight_t *d_weight, *d_bias;
    float *d_dweight, *d_dbias;
    
    HIP_CHECK(hipMalloc(&d_x, x_size * sizeof(input_t)));
    HIP_CHECK(hipMalloc(&d_weight, weight_size * sizeof(weight_t)));
    HIP_CHECK(hipMalloc(&d_bias, config.dim * sizeof(weight_t)));
    HIP_CHECK(hipMalloc(&d_dout, x_size * sizeof(input_t)));
    HIP_CHECK(hipMalloc(&d_dx, x_size * sizeof(input_t)));
    HIP_CHECK(hipMalloc(&d_dweight, weight_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dbias, config.dim * sizeof(float)));
    
    // Convert and copy to device
    if (sizeof(input_t) == 4) {
        HIP_CHECK(hipMemcpy(d_x, h_x, x_size * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_dout, h_dout, x_size * sizeof(float), hipMemcpyHostToDevice));
    } else {
        // Convert to FP16
        __half *h_x_fp16 = new __half[x_size];
        __half *h_dout_fp16 = new __half[x_size];
        for (int i = 0; i < x_size; ++i) h_x_fp16[i] = __float2half(h_x[i]);
        for (int i = 0; i < x_size; ++i) h_dout_fp16[i] = __float2half(h_dout[i]);
        HIP_CHECK(hipMemcpy(d_x, h_x_fp16, x_size * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_dout, h_dout_fp16, x_size * sizeof(__half), hipMemcpyHostToDevice));
        delete[] h_x_fp16;
        delete[] h_dout_fp16;
    }
    
    if (sizeof(weight_t) == 4) {
        HIP_CHECK(hipMemcpy(d_weight, h_weight, weight_size * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_bias, h_bias, config.dim * sizeof(float), hipMemcpyHostToDevice));
    } else {
        __half *h_weight_fp16 = new __half[weight_size];
        __half *h_bias_fp16 = new __half[config.dim];
        for (int i = 0; i < weight_size; ++i) h_weight_fp16[i] = __float2half(h_weight[i]);
        for (int i = 0; i < config.dim; ++i) h_bias_fp16[i] = __float2half(h_bias[i]);
        HIP_CHECK(hipMemcpy(d_weight, h_weight_fp16, weight_size * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_bias, h_bias_fp16, config.dim * sizeof(__half), hipMemcpyHostToDevice));
        delete[] h_weight_fp16;
        delete[] h_bias_fp16;
    }
    
    // Setup params
    ConvParamsBwd params;
    params.x_ptr = d_x;
    params.weight_ptr = d_weight;
    params.bias_ptr = d_bias;
    params.dout_ptr = d_dout;
    params.dx_ptr = d_dx;
    params.dweight_ptr = d_dweight;
    params.dbias_ptr = d_dbias;
    params.batch = config.batch;
    params.dim = config.dim;
    params.seqlen = config.seqlen;
    params.width = config.width;
    params.silu_activation = config.silu_activation;
    params.x_batch_stride = config.dim * config.seqlen;
    params.x_c_stride = config.seqlen;
    params.weight_c_stride = config.width;
    params.weight_width_stride = 1;
    params.dout_batch_stride = config.dim * config.seqlen;
    params.dout_c_stride = config.seqlen;
    params.dx_batch_stride = config.dim * config.seqlen;
    params.dx_c_stride = config.seqlen;
    params.dweight_c_stride = config.width;
    params.dweight_width_stride = 1;
    
    // Helper lambda to launch kernel
    auto launch_kernel = [&]() {
        if (config.width == 2) {
            causal_conv1d_bwd_launch<128, 2, input_t, weight_t>(params, nullptr);
        } else if (config.width == 3) {
            causal_conv1d_bwd_launch<128, 3, input_t, weight_t>(params, nullptr);
        } else if (config.width == 4) {
            causal_conv1d_bwd_launch<128, 4, input_t, weight_t>(params, nullptr);
        }
    };
    
    // Warmup: zero gradients and run once
    HIP_CHECK(hipMemset(d_dx, 0, x_size * sizeof(input_t)));
    HIP_CHECK(hipMemset(d_dweight, 0, weight_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dbias, 0, config.dim * sizeof(float)));
    launch_kernel();
    HIP_CHECK(hipDeviceSynchronize());
    
    // Performance test: zero gradients and run multiple times
    HIP_CHECK(hipMemset(d_dx, 0, x_size * sizeof(input_t)));
    HIP_CHECK(hipMemset(d_dweight, 0, weight_size * sizeof(float)));
    HIP_CHECK(hipMemset(d_dbias, 0, config.dim * sizeof(float)));
    
    Timer timer;
    timer.start();
    for (int i = 0; i < num_iters; ++i) {
        launch_kernel();
    }
    float elapsed_ms = timer.stop();
    float avg_time_us = (elapsed_ms * 1000.0f) / num_iters;
    
    // Calculate bandwidth and throughput
    size_t bytes_read = x_size * sizeof(input_t) * 2 + weight_size * sizeof(weight_t) + 
                        config.dim * sizeof(weight_t);  // x, dout, weight, bias
    size_t bytes_write = x_size * sizeof(input_t) + weight_size * sizeof(float) + 
                         config.dim * sizeof(float);  // dx, dweight, dbias
    size_t total_bytes = bytes_read + bytes_write;
    float bandwidth_gb_s = (total_bytes * num_iters) / (elapsed_ms / 1000.0f) / 1e9;
    
    std::cout << "\n  Performance:" << std::endl;
    std::cout << "    Avg Time: " << avg_time_us << " μs" << std::endl;
    std::cout << "    Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // ========================================================================
    // CRITICAL FIX: Re-run once for correctness verification
    // The performance test accumulated gradients over multiple iterations
    // ========================================================================
    if (verify_correctness) {
        // Clear all gradients before single verification run
        HIP_CHECK(hipMemset(d_dx, 0, x_size * sizeof(input_t)));
        HIP_CHECK(hipMemset(d_dweight, 0, weight_size * sizeof(float)));
        HIP_CHECK(hipMemset(d_dbias, 0, config.dim * sizeof(float)));
        
        // Run once for verification
        launch_kernel();
        HIP_CHECK(hipDeviceSynchronize());
    }
    // ========================================================================
    
    // Copy results back
    if (sizeof(input_t) == 4) {
        HIP_CHECK(hipMemcpy(h_dx_gpu, d_dx, x_size * sizeof(float), hipMemcpyDeviceToHost));
    } else {
        __half *h_dx_fp16 = new __half[x_size];
        HIP_CHECK(hipMemcpy(h_dx_fp16, d_dx, x_size * sizeof(__half), hipMemcpyDeviceToHost));
        for (int i = 0; i < x_size; ++i) h_dx_gpu[i] = __half2float(h_dx_fp16[i]);
        delete[] h_dx_fp16;
    }
    HIP_CHECK(hipMemcpy(h_dweight_gpu, d_dweight, weight_size * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_dbias_gpu, d_dbias, config.dim * sizeof(float), hipMemcpyDeviceToHost));
    
    // Track test result
    bool passed = true;
    
    // Correctness verification
    if (verify_correctness) {
        std::cout << "\n  Correctness Check:" << std::endl;
        
        float *h_dx_cpu = new float[x_size];
        float *h_dweight_cpu = new float[weight_size];
        float *h_dbias_cpu = new float[config.dim];
        
        cpu_causal_conv1d_backward(
            h_x, h_weight, h_bias, h_dout,
            h_dx_cpu, h_dweight_cpu, h_dbias_cpu,
            config.batch, config.dim, config.seqlen, config.width,
            config.silu_activation
        );
        
        // Calculate errors
        float max_err_dx = 0.0f, max_err_dw = 0.0f, max_err_db = 0.0f;
        float avg_err_dx = 0.0f, avg_err_dw = 0.0f, avg_err_db = 0.0f;
        
        for (int i = 0; i < x_size; ++i) {
            float err = fabs(h_dx_gpu[i] - h_dx_cpu[i]);
            max_err_dx = std::max(max_err_dx, err);
            avg_err_dx += err;
        }
        avg_err_dx /= x_size;
        
        for (int i = 0; i < weight_size; ++i) {
            float err = fabs(h_dweight_gpu[i] - h_dweight_cpu[i]);
            max_err_dw = std::max(max_err_dw, err);
            avg_err_dw += err;
        }
        avg_err_dw /= weight_size;
        
        for (int i = 0; i < config.dim; ++i) {
            float err = fabs(h_dbias_gpu[i] - h_dbias_cpu[i]);
            max_err_db = std::max(max_err_db, err);
            avg_err_db += err;
        }
        avg_err_db /= config.dim;

        std::cout << "    dx  - Max Err: " << max_err_dx << ", Avg Err: " << avg_err_dx << std::endl;
        std::cout << "    dW  - Max Err: " << max_err_dw << ", Avg Err: " << avg_err_dw << std::endl;
        std::cout << "    db  - Max Err: " << max_err_db << ", Avg Err: " << avg_err_db << std::endl;
        
        // Adaptive tolerance based on problem size
        float base_tolerance = sizeof(input_t) == 4 ? 1e-4f : 1e-2f;
        
        // For weight/bias gradients, use relaxed tolerance for large problems
        // due to accumulation order differences between CPU and GPU
        int total_accumulations = config.batch * config.seqlen;
        float weight_tolerance = base_tolerance;
        if (total_accumulations > 2000) {
            // Scale tolerance logarithmically with problem size
            weight_tolerance *= std::max(1.0f, std::log10f(total_accumulations / 1000.0f));
        }
        
        passed = (max_err_dx < base_tolerance) && 
                 (max_err_dw < weight_tolerance * 10.0f) &&
                 (max_err_db < weight_tolerance * 10.0f);
        
        if (passed) {
            std::cout << "    ✓ PASSED (dx tol: " << base_tolerance 
                      << ", dW/db tol: " << (weight_tolerance * 10.0f) << ")" << std::endl;
        } else {
            std::cout << "    ✗ FAILED (dx tol: " << base_tolerance 
                      << ", dW/db tol: " << (weight_tolerance * 10.0f) << ")" << std::endl;
            std::cout << "    Note: Small numerical differences are expected for large accumulations" << std::endl;
        }
        
        delete[] h_dx_cpu;
        delete[] h_dweight_cpu;
        delete[] h_dbias_cpu;
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_dout));
    HIP_CHECK(hipFree(d_dx));
    HIP_CHECK(hipFree(d_dweight));
    HIP_CHECK(hipFree(d_dbias));
    
    delete[] h_x;
    delete[] h_weight;
    delete[] h_bias;
    delete[] h_dout;
    delete[] h_dx_gpu;
    delete[] h_dweight_gpu;
    delete[] h_dbias_gpu;
    
    return passed;
}

// ============================================================================
// Main test suite with summary report
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Causal Conv1D Backward - HIP Implementation Test Suite       ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    
    // Get device info
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "\nGPU: " << prop.name << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;
    
    // Test configurations
    std::vector<TestConfig> test_configs = {
        // Small tests
        TestConfig("Small - Width 4", 2, 64, 128, 4, false),
        TestConfig("Small - Width 4 with SiLU", 2, 64, 128, 4, true),
        // Medium tests
        TestConfig("Medium - Standard", 4, 256, 512, 4, false),
        TestConfig("Medium - With SiLU", 4, 256, 512, 4, true),
        
        // Large tests (typical transformer sizes)
        TestConfig("Large - Mamba-like", 8, 2048, 1024, 4, false),
        TestConfig("Large - Mamba-like + SiLU", 8, 2048, 1024, 4, true),
        
        // Very long sequence
        TestConfig("Long Sequence", 4, 256, 4096, 4, false),
        TestConfig("Long Sequence + SiLU", 4, 256, 4096, 4, true),
        
        // Wide channel
        TestConfig("Wide Channel", 4, 4096, 512, 4, false),
        TestConfig("Wide Channel + SiLU", 4, 4096, 512, 4, true),
    };
    
    bool verify_correctness = (argc > 1 && std::string(argv[1]) == "--verify");
    int num_iters = 100;
    
    std::cout << "\nTest Mode: " << (verify_correctness ? "Verification + Performance" : "Performance Only") << std::endl;
    std::cout << "Iterations: " << num_iters << std::endl;
    
    // Track results
    int fp32_passed = 0, fp32_failed = 0;
    int fp16_passed = 0, fp16_failed = 0;
    
    // Run FP32 tests
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  FP32 Tests" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    for (const auto& config : test_configs) {
        bool passed = run_test_case<float, float>(config, verify_correctness, num_iters);
        if (verify_correctness) {
            if (passed) fp32_passed++;
            else fp32_failed++;
        }
    }
    
    // Run FP16 tests
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "  FP16 Tests" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    for (const auto& config : test_configs) {
        bool passed = run_test_case<__half, float>(config, verify_correctness, num_iters);
        if (verify_correctness) {
            if (passed) fp16_passed++;
            else fp16_failed++;
        }
    }
    
    // ========================================================================
    // Print Summary Report
    // ========================================================================
    std::cout << "\n\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                 Causal Conv1D Backward Tests                 ║" << std::endl;
    std::cout << "║                   HIP Implementation Summary                 ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    int total_tests = (fp32_passed + fp32_failed + fp16_passed + fp16_failed);
    int total_passed = fp32_passed + fp16_passed;
    int total_failed = fp32_failed + fp16_failed;
    
    if (verify_correctness) {
        std::cout << "🎯 Total Tests Run:    " << total_tests << std::endl;
        std::cout << "✅ PASSED:            " << total_passed 
                  << " (" << (total_tests > 0 ? (total_passed * 100 / total_tests) : 0) << "%)" << std::endl;
        std::cout << "❌ FAILED:            " << total_failed 
                  << " (" << (total_tests > 0 ? (total_failed * 100 / total_tests) : 0) << "%)" << std::endl;
        std::cout << std::endl;
        
        std::cout << "📊 Breakdown by Data Type:" << std::endl;
        std::cout << "   FP32:  " << fp32_passed << " passed, " << fp32_failed << " failed" << std::endl;
        std::cout << "   FP16:  " << fp16_passed << " passed, " << fp16_failed << " failed" << std::endl;
        std::cout << std::endl;
    } else {
        std::cout << "⚡ Performance Mode" << std::endl;
        std::cout << "   Total Benchmarks Run: " << (test_configs.size() * 2) << std::endl;
        std::cout << "   (FP32: " << test_configs.size() << ", FP16: " << test_configs.size() << ")" << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "🖥️  Platform: " << prop.name << std::endl;
    std::cout << "💾 Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;
    std::cout << "🔢 Test Configurations: " << test_configs.size() << std::endl;
    std::cout << "🔄 Iterations per test: " << num_iters << std::endl;
    std::cout << std::endl;
    
    if (verify_correctness) {
        if (total_failed == 0) {
            std::cout << "🎉 All tests PASSED! Implementation is correct! 🎉" << std::endl;
        } else {
            std::cout << "⚠️  Some tests failed. Please review the results above." << std::endl;
        }
    } else {
        std::cout << "ℹ️  Run with '--verify' flag to enable correctness checking." << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << std::string(66, '=') << std::endl;
    
    return (total_failed == 0) ? 0 : 1;
}