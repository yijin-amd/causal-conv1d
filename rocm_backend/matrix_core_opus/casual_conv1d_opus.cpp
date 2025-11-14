#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#include <torch/torch.h>
#include <cstring>
#define HALF
#ifdef HALF
#include "half.hpp"
#endif

#include <opus/opus.hpp>

#define LOCAL_SCRATCH 0
#define RAND_INT 0
#define CUSTOMIZED_INT 1
#define ENABLE_HOST_VERIFICATION 1  // 设置为 0 则完全跳过 host 端预处理验证

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

#define ABS(x) ((x) > 0 ? (x) : -(x))

using fp32_t = float;
using fp16_t = _Float16;
using float16 = half_float::half; // cpu type

using fp16x2_t = fp16_t __attribute__((ext_vector_type(2)));
using fp16x4_t = fp16_t __attribute__((ext_vector_type(4)));
using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp16x16_t = fp16_t __attribute__((ext_vector_type(16)));
using fp32x4_t = fp32_t __attribute__((ext_vector_type(4)));
using fp32x16_t = fp32_t __attribute__((ext_vector_type(16)));

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
#define BUFFER_LOAD_DWORD3 0x00020000   // This is valid for 
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};

__device__ int32x4_t make_buffer_resource(const void * ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res {ptr, size, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(int32x4_t, res);
}

#ifdef RAND_INT
#define PER_PIXEL_CHECK
#endif

static inline bool valid_vector( const float* ref, const float16* pred, int n, double nrms = 1e-3 )
{    
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end=n;
    
    for( int i=i_start; i<i_end; ++i ){
        double ri=(double)ref[i];
        double pi=(double)pred[i];
        double d=ri-pi;
        double dd=d*d;
        double rr=2.0*ri*ri;
        s0+=dd;
        s1+=rr;
        
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri-pi)/ri;
        if(delta>nrms){
            if(pp_err<100)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n",i,ri,pi,((uint16_t*)pred)[i],delta);
            pp_err++;
        }
#endif
    }
    // int i_num = i_end - i_start;
    // printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

// ---------------- GEMM ----------------
void gemm(const float* A, const float* B, const float* bias,
          float* C, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = bias ? bias[m] : 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}
 
// ---------------- causal img2col ----------------
void img2col_causal_conv1d(const float* input, float* output,
                           int N, int C, int H,
                           int kernel_size, int stride) {
    int padding = kernel_size - 1;
    int out_len = (H + padding - kernel_size) / stride + 1;
 
    for (int n = 0; n < N; ++n) {
        for (int out_idx = 0; out_idx < out_len; ++out_idx) {
            for (int c = 0; c < C; ++c) {
                for (int k = 0; k < kernel_size; ++k) {
                    int in_pos = out_idx * stride + k - padding;
                    float val = 0.0f;
                    if (in_pos >= 0 && in_pos <= out_idx * stride && in_pos < H) {
                        val = input[n * C * H + c * H + in_pos];
                    }
                    output[n * (C * kernel_size * out_len) +
                           (c * kernel_size + k) * out_len +
                           out_idx] = val;
                }
            }
        }
    }
}
 
// ---------------- causal conv1d (img2col+gemm) ----------------
void causal_conv1d_img2col(const float* input, const float* weight, const float* bias,
                           float* output,
                           int N, int C_in, int H,
                           int C_out, int kernel_size, int stride) {
    int padding = kernel_size - 1;
    int out_len = (H + padding - kernel_size) / stride + 1;
 
    std::vector<float> col(N * C_in * kernel_size * out_len);
    img2col_causal_conv1d(input, col.data(), N, C_in, H, kernel_size, stride);
 
    // 权重矩阵 (C_out x C_in*kernel_size)
    std::vector<float> W(C_out * C_in * kernel_size);
    for (int co = 0; co < C_out; ++co) {
        for (int ci = 0; ci < C_in; ++ci) {
            for (int k = 0; k < kernel_size; ++k) {
                W[co * (C_in * kernel_size) + ci * kernel_size + k] =
                    weight[co * (C_in * kernel_size) + ci * kernel_size + k];
            }
        }
    }
 
    for (int n = 0; n < N; ++n) {
        gemm(W.data(),
             col.data() + n * (C_in * kernel_size * out_len),
             bias,
             output + n * (C_out * out_len),
             C_out, C_in * kernel_size, out_len);
    }
}

template<int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TILE_M, int TILE_N, int TILE_K, int WAVE_M, int WAVE_N, int WAVE_K>
__global__ void matrix_core_kernel_block_v2(const void* __restrict__ ptr_a,
                                         const void* __restrict__ ptr_b,
                                         void* __restrict__ ptr_c,
                                         int k,
                                         int stride_a, // stride in unit of pixel
                                         int stride_b,
                                         int stride_c)
{
    using opus::operator""_I;
    constexpr int W_M = WAVE_M;
    constexpr int W_N = WAVE_N;
    constexpr int W_K = WAVE_K;

    constexpr int T_M = TILE_M;
    constexpr int T_N = TILE_N;
    constexpr int T_K = TILE_K;

    constexpr int E_M = BLOCK_M / (W_M * T_M);
    constexpr int E_N = BLOCK_N / (W_N * T_N);
    constexpr int E_K = BLOCK_K / (W_K * T_K);
    static_assert(E_K == 1);

    using d_a = opus::fp16_t;
    using d_b = opus::fp16_t;
    using d_c = opus::fp32_t;

    int lane_id = threadIdx.x % opus::get_warp_size();
    int wave_id = threadIdx.x / opus::get_warp_size();
    int g_im = blockIdx.x * BLOCK_M;
    int g_in = blockIdx.y * BLOCK_N;

    // NOTE: the shape merge is per-dim
    //
    // A:[(expd_m<y>, tile_m<p>), (expd_k<y>, tile_k<p>)] * [(grpm_a<p>), (rept_a<y>, grpk_a<p>, pack_a<y>)]
    // B:[(expd_n<y>, tile_n<p>), (expd_k<y>, tile_k<p>)] * [(grpn_b<p>), (rept_b<y>, grpk_b<p>, pack_b<y>)]
    // C:[(expd_m<y>, tile_m<p>), (expd_n<y>, tile_n<p>)] * [(grpn_c<p>), (rept_c<y>, grpm_c<p>, pack_c<y>)]
    //
    // A:[(expd_m<y>, tile_m<p>, grpm_a<p>), (expd_k<y>, tile_k<p>, rept_a<y>, grpk_a<p>, pack_a<y>)]
    // B:[(expd_n<y>, tile_n<p>, grpn_b<p>), (expd_k<y>, tile_k<p>, rept_b<y>, grpk_b<p>, pack_b<y>)]
    // C:[(expd_m<y>, tile_m<p>, grpn_c<p>), (expd_n<y>, tile_n<p>, rept_c<y>, grpm_c<p>, pack_c<y>)]
    //
    auto mma  = opus::make_tiled_mma<d_a, d_b, d_c>(opus::seq<E_M, E_N, E_K>{}, opus::seq<T_M, T_N, T_K>{}, opus::seq<W_M, W_N, W_K>{}, opus::mfma_adaptor_swap_ab{});

    auto u_a = opus::partition_layout_a(mma, opus::make_tuple(stride_a, 1_I), opus::make_tuple(wave_id / 2, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a) /*tile_m<p>, grpm_a<p>, tile_k<p>, grpk_a<p>*/);
    auto u_b = opus::partition_layout_b(mma, opus::make_tuple(stride_b, 1_I), opus::make_tuple(wave_id % 2, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b) /*tile_n<p>, grpn_b<p>, tile_k<p>, grpk_b<p>*/);
    auto u_c = opus::partition_layout_c(mma, opus::make_tuple(stride_c, 1_I), opus::make_tuple(wave_id / 2, lane_id % mma.grpn_c, wave_id % 2, lane_id / mma.grpn_c) /*tile_m<p>, grpn_c<p> tile_n<p>, grpm_c<p>*/);
    auto g_a = opus::make_gmem(reinterpret_cast<const d_a*>(ptr_a) + g_im * stride_a);
    auto g_b = opus::make_gmem(reinterpret_cast<const d_b*>(ptr_b) + g_in * stride_b);
    auto g_c = opus::make_gmem(reinterpret_cast<opus::fp16_t*>(ptr_c) + g_im * stride_c + g_in);

    // start of kernel
    int loops = (k + BLOCK_K - 1) / BLOCK_K;
#if 1
    typename decltype(mma)::vtype_c v_c;
    opus::clear(v_c);

    for(auto i = 0; i < loops; i++ ) {
        auto v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
        auto v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    auto v_c_f16 = opus::cast<fp16_t>(v_c);
    g_c.store<4>(v_c_f16, u_c);
#else
    auto v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
    auto v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
    auto v_c = mma(v_a, v_b);   // first time, C is always zero

    for(auto i = 0; i < loops - 1; i++ ) {
        v_a = g_a.load<4>(u_a);  u_a += BLOCK_K;
        v_b = g_b.load<4>(u_b);  u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    auto v_c_f16 = opus::cast<fp16_t>(v_c);
    g_c.store<4>(v_c_f16, u_c);
#endif
}

// Depthwise Causal Conv1D with explicit left padding
// input:  N x C x H
// weight: C x kernel_size   (flattened as c*kernel_size)
// bias:   C
// output: N x C x H
void causal_conv1d_depthwise(
    const float* input,    // 输入
    const float* weight,   // 权重 (C * kernel_size)
    const float* bias,     // 偏置 (C)
    float* output,         // 输出
    int N, int C, int H,
    int kernel_size
) {
    int pad = kernel_size - 1;
    int H_pad = H + pad; // padded length
 
    // 构造 padded 输入
    std::vector<float> padded(N * C * H_pad, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            // 拷贝原始数据到 padded 的右侧
            for (int h = 0; h < H; ++h) {
                padded[n * C * H_pad + c * H_pad + pad + h] =
                    input[n * C * H + c * H + h];
            }
        }
    }
 
    // 卷积计算
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < H; ++t) {
                float sum = bias ? bias[c] : 0.0f;
                for (int k = 0; k < kernel_size; ++k) {
                    float val = padded[n * C * H_pad + c * H_pad + t + k];
                    float w = weight[c * kernel_size + k];
                    sum += val * w;
                }
                output[n * C * H + c * H + t] = sum;
            }
        }
    }
}

// 通用 Conv1d 函数
void casual_conv1d_rcr(float* input, float* output,
                float* weight, float* bias,
                int N, int C_in, int C_out, int L,
                int kernel_size, int pad, int stride = 1, int groups = 1) {
    // 1. 定义 Conv1d
    torch::nn::Conv1d conv(
        torch::nn::Conv1dOptions(C_in, C_out, kernel_size)
            .stride(stride)
            .padding(pad)
            .groups(groups)
    );
 
    // 2. 设置权重和偏置
    torch::Tensor w_tensor = torch::from_blob(weight, {C_out, C_in / groups, kernel_size}, torch::kFloat).clone();
    torch::Tensor b_tensor = torch::from_blob(bias, {C_out}, torch::kFloat).clone();
    conv->weight = w_tensor;
    conv->bias   = b_tensor;
 
    // 3. 输入张量
    torch::Tensor input_tensor = torch::from_blob(input, {N, C_in, L}, torch::kFloat).clone();
 
    // 4. 前向计算
    torch::Tensor output_tensor = conv->forward(input_tensor);
 
    // 5. 拷贝结果到一维指针
    // int out_size = 1;
    // for (auto s : output_tensor.sizes()) out_size *= s;
    int out_size = N * C_in * L;
    std::memcpy(output, output_tensor.data_ptr<float>(), out_size * sizeof(float));
 
    // 打印输出形状
    std::cout << "Output shape: " << output_tensor.sizes() << " out_size: " << out_size << std::endl;
}

void transpose(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // input(i, j) -> output(j, i)
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void transpose_fp16(const float16* input, float16* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // input(i, j) -> output(j, i)
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

void customized_vector_2d_in(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(r*ld+c)/100);
        }
    }
}

void customized_vector_2d_weight(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(r*ld+c)/100);
        }
    }
}

void rand_vector_2d(float* v, int batch, int row, int col, int ld, float min_v = 0, float max_v = 1){
    int b,r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(b=0;b<batch;b++){
        for(r=0;r<row;r++){
            for(c=0;c<col;c++){
                float tmp = float(std::rand()) / float(RAND_MAX);
                // v[r*ld+c] = static_cast<float>(min_v + tmp * (max_v - min_v));
                v[b*row*col+r*ld+c] = static_cast<float>(min_v + tmp * (max_v - min_v));
                // v[r*ld+c] =   ((float)(r*ld+c)) / (row/2 * col/2) - 5;
            }
        }
    }
}

void gemm_rcr(
    const float*  __restrict__ ptr_a,
    const float*  __restrict__ ptr_b,
    float*  ptr_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc)
{
    for(auto i_m = 0 ; i_m < m; i_m++) {
        for(auto i_n = 0; i_n < n; i_n++) {
            float acc = 0;
            for(auto i_k = 0; i_k < k; i_k++) {
                acc += ptr_a[i_m * lda + i_k] * ptr_b[i_n * ldb + i_k];
            }
            ptr_c[i_m * ldc + i_n] = acc;
        }
    }
}

// GPU Kernel: 输入预处理 - padding + img2col 操作
// 输入: (hi, ci) -> 输出: (ho, hk*ci)
// 
// 注意：此 kernel 接收已经在 host 端转置好的数据 [hi, ci]
// 只需要处理 padding 和 img2col 操作
__global__ void preprocess_input_kernel(
    const fp16_t* __restrict__ input,  // 输入 [hi, ci] - 已在 host 端转置
    fp16_t* __restrict__ output,       // 输出 [ho, hk*ci] - GEMM A 矩阵
    int ci, int hi, int ho, int hk, int pad)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 输出位置索引 [0, ho)
    int k_c_idx = blockIdx.y * blockDim.y + threadIdx.y;  // 输出特征索引 [0, hk*ci)
    
    if (out_idx >= ho || k_c_idx >= hk * ci) return;
    
    int k = k_c_idx / ci;  // kernel 位置
    int c = k_c_idx % ci;  // channel 位置
    
    // 计算输入位置（考虑 padding）
    int in_pos = out_idx + k;  // 在 padded 输入中的位置
    
    fp16_t val = 0.0f;
    if (in_pos >= pad && in_pos < hi + pad) {
        // 从已转置的 [hi, ci] 格式读取
        // 数据已经是时间主序，直接访问即可
        int h_idx = in_pos - pad;
        val = input[h_idx * ci + c];
    }
    
    // 输出是 [ho, hk*ci]
    output[out_idx * (hk * ci) + k_c_idx] = val;
}

// GPU Kernel: 权重转换
// depthwise 权重 [ci, hk] -> 普通卷积权重 [ci, hk*ci]
__global__ void preprocess_weight_kernel(
    const fp16_t* __restrict__ weight_dw,  // depthwise 权重 [ci, hk]
    fp16_t* __restrict__ weight_conv,      // 转换后权重 [ci, hk*ci]
    int ci, int hk)
{
    int out_c = blockIdx.x * blockDim.x + threadIdx.x;  // 输出 channel [0, ci)
    int k_in_c = blockIdx.y * blockDim.y + threadIdx.y; // 输入特征索引 [0, hk*ci)
    
    if (out_c >= ci || k_in_c >= hk * ci) return;
    
    int k = k_in_c / ci;      // kernel 位置
    int in_c = k_in_c % ci;   // 输入 channel
    
    fp16_t val = 0.0f;
    if (in_c == out_c) {
        // depthwise: 只有当输入输出 channel 相同时才有权重
        val = weight_dw[out_c * hk + k];
    }
    
    // 输出是 [ci, hk*ci]
    weight_conv[out_c * (hk * ci) + k_in_c] = val;
}

void casual_conv1d_block_run()
{
    int batch = 1;
    int hi = 2048;
    int ci = 64;
    int hk = 4;
    int pad = hk - 1;

    // define gemm inputs a,b,c
    int ho = hi;
    int m = ho;
    int n = ci;
    int k = hk * ci;

    int lda = k;
    int ldb = k;
    int ldc = n;

    // init fp32 input and weight
    float *host_in, *host_w, *host_c, *host_bias;

    //fp32 input[batch, ci, hi] and weight[ci, hk] on host
    host_in = (float*)malloc(batch*ci*hi*sizeof(float));
    host_w = (float*)malloc(ci*hk*sizeof(float));
    host_bias = (float*)malloc(ci*sizeof(float));
    host_c = (float*)malloc(batch*ldc*m*sizeof(float));
    int ld_in = hi;
    int ld_w = hk;

// #ifdef RAND_INT
//     rand_vector_2d_int(host_in, hi, ci, ld_in);
//     rand_vector_2d_int(host_w, ci, hk, ld_w);
// #ifdef CUSTOMIZED_INT
    // customized_vector_2d_in(host_in, ci, hi, ld_in);
    // customized_vector_2d_weight(host_w, ci, hk, ld_w);
    for (int i = 0; i < ci; i++) host_bias[i] = 0.0f;
// #else
    rand_vector_2d(host_in, batch, ci, hi, ld_in, 1.0, 2.0);
    rand_vector_2d(host_w, batch, ci, hk, ld_w, 1.0, 2.0);
// #endif
    // for(int i=0; i<hi*ci; i++) {if (i<100) {printf("in[%d], %f \n", i, host_in[i]);}}
    // for(int i=0; i<ci*hk; i++) {printf("w[%d], %f \n", i, host_w[i]);}
    float16 *fp16_in, *fp16_w;
    //convert fp32 input into fp16 on host
    fp16_in = (float16*)malloc((batch*hi*ci)*sizeof(float16));
    fp16_w = (float16*)malloc((batch*ci*hk)*sizeof(float16));
    for(int i=0; i<batch*hi*ci; i++)fp16_in[i]=__float2half_rn(host_in[i]);
    for(int i=0; i<batch*ci*hk; i++)fp16_w[i]=__float2half_rn(host_w[i]);
    // for(int i=0; i<ci*hk; i++) {printf("fp16_w[%d], %f \n", i, (float)(fp16_w[i]));}

    // ========== Host 端预处理：转置 ==========
    printf("在 Host 上执行转置操作...\n");
    
    float16 *fp16_in_transposed;  // 转置后的输入 [hi, ci]
    fp16_in_transposed = (float16*)malloc((batch*hi * ci)*sizeof(float16));
    
    // input [ci,hi] 转置为 [hi,ci] - 在 host 上显式执行
    transpose_fp16(fp16_in, fp16_in_transposed, ci, hi);
    printf("✓ 转置完成：[%d, %d] -> [%d, %d]\n", ci, hi, hi, ci);

#if ENABLE_HOST_VERIFICATION
    // ========== Host 端完整预处理（用于验证 GPU kernel 正确性）==========
    printf("执行完整 Host 端预处理用于验证...\n");
    
    float16 *fp16_in_pad, *fp16_a_host, *fp16_b_host;
    fp16_in_pad = (float16*)malloc((batch*(hi+pad) * ci)*sizeof(float16));
    fp16_a_host = (float16*)malloc(lda*m*sizeof(float16));
    fp16_b_host = (float16*)malloc(ldb*n*sizeof(float16));
    
    // add pad for input
    for(int i = 0; i < pad * ci; i++) {
        fp16_in_pad[i] = 0;
    }
    for(int i = 0; i < hi * ci; i++) {
        fp16_in_pad[i + pad * ci] = fp16_in_transposed[i];
    }
    
    // input with pad, img2col
    for(int i=0; i < ho; i++) {
        for (int j = 0; j < hk * ci; j++) {
            fp16_a_host[i * hk *ci + j] = fp16_in_pad[i * ci + j];
        }
    }
    
    //convert dw weight to common weight
    for(int i=0; i < ci * hk * ci; i++) {
        fp16_b_host[i] = 0;
    }
    for(int i=0; i < ci; i++) {
        for (int j = 0; j < hk*ci; j++) {
            if ((j % ci) == i) {
                fp16_b_host[i * hk *ci + j] = fp16_w[i * hk + int(j / ci)];
            }
        }
    }

    float *host_a, *host_b;
    host_a = (float*)malloc(lda*m*sizeof(float));
    host_b = (float*)malloc(ldb*n*sizeof(float));
    for(int i=0; i<lda*m; i++)host_a[i]=(float)(fp16_a_host[i]);
    for(int i=0; i<ldb*n; i++)host_b[i]=(float)(fp16_b_host[i]);
#endif

    // ========== GPU 预处理 ==========
    float16 *dev_in_transposed, *dev_w, *dev_a, *dev_b, *dev_c;
    float16 *fp16_c, *fp16_c_nch;
    fp16_c = (float16*)malloc(ldc*m*sizeof(float16));
    fp16_c_nch = (float16*)malloc(ldc*m*sizeof(float16));
    
    // 分配 GPU 内存
    HIP_CALL(hipMalloc(&dev_in_transposed, hi*ci*sizeof(float16)));  // 转置后的输入 [hi, ci]
    HIP_CALL(hipMalloc(&dev_w, ci*hk*sizeof(float16)));              // 权重 [ci, hk]
    HIP_CALL(hipMalloc(&dev_a, lda*m*sizeof(float16)));              // GEMM A [ho, hk*ci]
    HIP_CALL(hipMalloc(&dev_b, ldb*n*sizeof(float16)));              // GEMM B [ci, hk*ci]
    HIP_CALL(hipMalloc(&dev_c, ldc*m*sizeof(float16)));              // GEMM C [ho, ci]
    
    // 将转置后的数据拷贝到 GPU
    printf("传输转置后的数据到 GPU...\n");
    HIP_CALL(hipMemcpy(dev_in_transposed, fp16_in_transposed, hi*ci*sizeof(float16), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_w, fp16_w, ci*hk*sizeof(float16), hipMemcpyHostToDevice));
    
    // 调用预处理 kernel
    // 输入预处理：[hi, ci] -> [ho, hk*ci] (已转置)
    printf("在 GPU 上执行 padding + img2col...\n");
    {
        dim3 block_dim(16, 16);  // 线程块大小
        dim3 grid_dim((ho + block_dim.x - 1) / block_dim.x, 
                      (hk*ci + block_dim.y - 1) / block_dim.y);
        preprocess_input_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<fp16_t*>(dev_in_transposed), 
            reinterpret_cast<fp16_t*>(dev_a), 
            ci, hi, ho, hk, pad);
        HIP_CALL(hipGetLastError());
        HIP_CALL(hipDeviceSynchronize());
    }
    
    // 权重预处理：[ci, hk] -> [ci, hk*ci]
    printf("在 GPU 上执行权重转换...\n");
    {
        dim3 block_dim(16, 16);  // 线程块大小
        dim3 grid_dim((ci + block_dim.x - 1) / block_dim.x, 
                      (hk*ci + block_dim.y - 1) / block_dim.y);
        preprocess_weight_kernel<<<grid_dim, block_dim>>>(
            reinterpret_cast<fp16_t*>(dev_w), 
            reinterpret_cast<fp16_t*>(dev_b), 
            ci, hk);
        HIP_CALL(hipGetLastError());
        HIP_CALL(hipDeviceSynchronize());
    }
    
    printf("✓ GPU 预处理完成\n");
    
#if ENABLE_HOST_VERIFICATION
    // ========== 验证 GPU 预处理结果与 Host 端结果对比 ==========
    {
        float16 *fp16_a_gpu = (float16*)malloc(lda*m*sizeof(float16));
        float16 *fp16_b_gpu = (float16*)malloc(ldb*n*sizeof(float16));
        
        HIP_CALL(hipMemcpy(fp16_a_gpu, dev_a, lda*m*sizeof(float16), hipMemcpyDeviceToHost));
        HIP_CALL(hipMemcpy(fp16_b_gpu, dev_b, ldb*n*sizeof(float16), hipMemcpyDeviceToHost));
        
        // 验证输入预处理结果
        bool input_match = true;
        int input_err_count = 0;
        for(int i=0; i<lda*m && input_err_count < 10; i++) {
            float diff = fabs((float)fp16_a_gpu[i] - (float)fp16_a_host[i]);
            if(diff > 1e-5) {
                if(input_err_count < 5) {
                    printf("输入预处理差异 [%d]: GPU=%f, Host=%f, diff=%f\n", 
                           i, (float)fp16_a_gpu[i], (float)fp16_a_host[i], diff);
                }
                input_match = false;
                input_err_count++;
            }
        }
        printf("✓ 输入预处理验证（GPU vs Host）: %s (错误数=%d/%d)\n", 
               input_match?"通过":"失败", input_err_count, lda*m);
        
        // 验证权重预处理结果
        bool weight_match = true;
        int weight_err_count = 0;
        for(int i=0; i<ldb*n && weight_err_count < 10; i++) {
            float diff = fabs((float)fp16_b_gpu[i] - (float)fp16_b_host[i]);
            if(diff > 1e-5) {
                if(weight_err_count < 5) {
                    printf("权重预处理差异 [%d]: GPU=%f, Host=%f, diff=%f\n", 
                           i, (float)fp16_b_gpu[i], (float)fp16_b_host[i], diff);
                }
                weight_match = false;
                weight_err_count++;
            }
        }
        printf("✓ 权重预处理验证（GPU vs Host）: %s (错误数=%d/%d)\n", 
               weight_match?"通过":"失败", weight_err_count, ldb*n);
        
        free(fp16_a_gpu);
        free(fp16_b_gpu);
    }
#endif

    printf("m:%d,n:%d,k:%d,lda:%d,ldb:%d,ldc:%d\n",  m, n, k, lda, ldb, ldc); fflush(stdout);
    // gemm_rcr(host_a, host_b, host_c, m,n,k,lda,ldb,ldc);
    // casual_conv1d_rcr(host_in, host_c, host_w, host_bias, batch, ci, ci, hi, hk, pad, 1, ci);
     causal_conv1d_depthwise(host_in, host_w, host_bias, host_c, batch, ci, hi, hk);
    // causal_conv1d_img2col(host_in, host_w, host_bias, host_c, batch, ci, hi, ci, hk, 1);
    {
        constexpr int BLOCK_M = 32;
        constexpr int BLOCK_N = 32;
        constexpr int BLOCK_K = 16;
        constexpr int TILE_M = 2;
        constexpr int TILE_N = 2;
        constexpr int TILE_K = 1;
        constexpr int WAVE_M = 16;
        constexpr int WAVE_N = 16;
        constexpr int WAVE_K = 16;

        auto gdim = dim3(m / BLOCK_M, n / BLOCK_N);
        auto kernel = matrix_core_kernel_block_v2<256, BLOCK_M, BLOCK_N, BLOCK_K, TILE_M, TILE_N, TILE_K, WAVE_M, WAVE_N, WAVE_K>;
        kernel<<<gdim, 256>>>(dev_a, dev_b, dev_c, k, lda, ldb, ldc);

        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
        transpose_fp16(fp16_c, fp16_c_nch, m, n);
#if 1
        // bool res = valid_vector(host_c, fp16_c, m*n, 1e-3);
        bool res = valid_vector(host_c, fp16_c_nch, m*n, 1e-2);
        printf("[%dx%dx%d, block_gemm_%dx%dx%d_%dx%dx%d_%dx%dx%d], %s", m, n, k,
            BLOCK_M, BLOCK_N, BLOCK_K, TILE_M, TILE_N, TILE_K, WAVE_M, WAVE_N, WAVE_K,
            res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
#endif
    }


    // 释放 host 内存
    free(host_in);
    free(host_w);
    free(host_c);
    free(host_bias);
    free(fp16_in);
    free(fp16_w);
    free(fp16_in_transposed);
    free(fp16_c);
    free(fp16_c_nch);
    
#if ENABLE_HOST_VERIFICATION
    free(host_a);
    free(host_b);
    free(fp16_a_host);
    free(fp16_b_host);
    free(fp16_in_pad);
#endif
    
    // 释放 GPU 内存
    HIP_CALL(hipFree(dev_in_transposed));
    HIP_CALL(hipFree(dev_w));
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));
}
 
int main() {
    casual_conv1d_block_run();
}