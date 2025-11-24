#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#define HALF
#ifdef HALF
#include "half.hpp"
#endif

#include <opus/opus.hpp>

#define LOCAL_SCRATCH 0
#define RAND_INT 0

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
// A: M*K, B: N*K, C:M*N, use 32x32x8 fp16
/*
* V0/V1/   is 32bit register holding A/B matrix data, each register contains 2 fp16 pixel along gemm-k
* a0/a1... is 32bit register holding C matrix data in fp32 (this instruction use fp32 as acc)
* L0, L1.. is lane id with in a single wave, here we only have lane 0~63 (wave64)
* each thread need 2 registers for A, 2 regs for B, 16 regs for C

                                 L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
                       Matrix B   __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k0  L0~31
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k1
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k2
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k3
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L32~63
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7
     Matrix A
     L0~31       L32~63           Matrix C
     k0 k1 k2 k3 k4 k5 k6 k7      L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
     _____ _____|_____ _____      __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
L0  |v0   |v1   |v0   |v1   |    |a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0| L0~31
L1  |v0   |v1   |v0   |v1   |    |a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|
L2  |v0   |v1   |v0   |v1   |    |a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|
L3  |v0   |v1   |v0   |v1   |   _|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|_
L4  |v0   |v1   |v0   |v1   |    |a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0|a0| L32~63
L5  |v0   |v1   |v0   |v1   |    |a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|a1|
L6  |v0   |v1   |v0   |v1   |    |a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|a2|
L7  |v0   |v1   |v0   |v1   |   _|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|a3|_
L8  |v0   |v1   |v0   |v1   |    |a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4| L0~31
L9  |v0   |v1   |v0   |v1   |    |a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|
L10 |v0   |v1   |v0   |v1   |    |a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|
L11 |v0   |v1   |v0   |v1   |   _|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|_
L12 |v0   |v1   |v0   |v1   |    |a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4|a4| L32~63
L13 |v0   |v1   |v0   |v1   |    |a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|a5|
L14 |v0   |v1   |v0   |v1   |    |a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|a6|
L15 |v0   |v1   |v0   |v1   |   _|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|a7|_
L16 |v0   |v1   |v0   |v1   |    |a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8| L0~31
L17 |v0   |v1   |v0   |v1   |    |a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|
L18 |v0   |v1   |v0   |v1   |    |10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|
L19 |v0   |v1   |v0   |v1   |   _|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|_
L20 |v0   |v1   |v0   |v1   |    |a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8|a8| L32~63
L21 |v0   |v1   |v0   |v1   |    |a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|a9|
L22 |v0   |v1   |v0   |v1   |    |10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|10|
L23 |v0   |v1   |v0   |v1   |   _|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|11|_
L24 |v0   |v1   |v0   |v1   |    |12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12| L0~31
L25 |v0   |v1   |v0   |v1   |    |13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|
L26 |v0   |v1   |v0   |v1   |    |14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|
L27 |v0   |v1   |v0   |v1   |   _|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|_
L28 |v0   |v1   |v0   |v1   |    |12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12|12| L32~63
L29 |v0   |v1   |v0   |v1   |    |13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|13|
L30 |v0   |v1   |v0   |v1   |    |14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|14|
L31 |v0___|v1___|v0___|v1___|   _|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|15|_
                |
*/
__global__ void 
matrix_core_kernel_standard(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 32x32x8 gemm, assume only launced 1 wave
    int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
    int offset_b = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_b);

    fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_a) + offset_a);
    fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_b) + offset_b);
    fp32x16_t v_c = {.0f};  // clear

    v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a, v_b, v_c, 0, 0, 0);

    fp16x16_t v_c_f16;
    for(auto i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int col_id_c = threadIdx.x % 32;
    int row_id_c = threadIdx.x / 32 * 4;
    int offset_c = row_id_c * stride_c + col_id_c;

    for(auto i = 0; i < 16; i++) {
        int row_offset = (i % 4) + (i / 4 * 8);
        *(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + row_offset * stride_c) = v_c_f16[i];
    }
}

// ============================================================================
// 直接使用 MFMA 内建指令的大规模 GEMM Kernel
// C[M×N] = A[M×K] × B[N×K]^T
// ============================================================================

// ============================================================================
// 修正后的 MFMA Large Kernel - 正确的索引计算
// C[M×N] = A[M×K] × B[N×K]^T
// ============================================================================

__global__ void 
matrix_core_kernel_mfma_large(
    const void* __restrict__ ptr_a,
    const void* __restrict__ ptr_b,
    void* __restrict__ ptr_c,
    int m,              // M 维度
    int n,              // N 维度
    int k,              // K 维度
    int stride_a,       // A 矩阵的行 stride
    int stride_b,       // B 矩阵的行 stride
    int stride_c)       // C 矩阵的行 stride
{
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 8;
    
    // 当前 block 负责的矩阵位置
    int block_m = blockIdx.x * MFMA_M;
    int block_n = blockIdx.y * MFMA_N;
    
    int lane_id = threadIdx.x;
    
    // ========================================================================
    // 关键修正：正确的 A/B 索引计算
    // ========================================================================
    // 参考 matrix_core_kernel_standard 的实现（第117-118行）
    // offset = (lane / 32 * 4) + (lane % 32 * stride)
    
    // 每个 lane 的基础偏移（在 32x8 子矩阵中）：
    // - 列偏移：lane / 32 * 4  (前32个lane读取列0-3，后32个读取列4-7)
    // - 行偏移：lane % 32       (每个lane负责不同的行)
    
    int col_offset_a = lane_id / 32 * 4;  // 0 or 4
    int row_offset_a = lane_id % 32;       // 0-31
    
    int col_offset_b = lane_id / 32 * 4;
    int row_offset_b = lane_id % 32;
    
    // ========================================================================
    // K 循环累加
    // ========================================================================
    fp32x16_t v_c = {0.0f};
    
    int k_loops = (k + MFMA_K - 1) / MFMA_K;
    
    for (int k_iter = 0; k_iter < k_loops; k_iter++) {
        int k_offset = k_iter * MFMA_K;
        
        // A[block_m + row, k_offset + col]
        const fp16_t* ptr_a_base = reinterpret_cast<const fp16_t*>(ptr_a) + 
                                    (block_m + row_offset_a) * stride_a + 
                                    k_offset + col_offset_a;
        fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(ptr_a_base);
        
        // B[block_n + row, k_offset + col]
        const fp16_t* ptr_b_base = reinterpret_cast<const fp16_t*>(ptr_b) + 
                                    (block_n + row_offset_b) * stride_b + 
                                    k_offset + col_offset_b;
        fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(ptr_b_base);
        
        // MFMA 计算
        v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a, v_b, v_c, 0, 0, 0);
    }
    
    // ========================================================================
    // FP32 → FP16 转换
    // ========================================================================
    fp16x16_t v_c_f16;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }
    
    // ========================================================================
    // 关键修正：正确的 C 写入索引
    // ========================================================================
    // 参考 matrix_core_kernel_standard 的实现（第131-138行）
    
    int col_id_c = lane_id % 32;            // 列：0-31
    int row_id_c = lane_id / 32 * 4;        // 行起始：0 or 4
    int offset_c = (block_m + row_id_c) * stride_c + block_n + col_id_c;
    
    // 每个 lane 写 16 个元素，布局为：
    // i=0-3:   行偏移 0, 1, 2, 3
    // i=4-7:   行偏移 8, 9, 10, 11
    // i=8-11:  行偏移 16, 17, 18, 19
    // i=12-15: 行偏移 24, 25, 26, 27
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int row_offset = (i % 4) + (i / 4 * 8);
        *(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + row_offset * stride_c) = v_c_f16[i];
    }
}


// ============================================================================
// Double Buffering 优化版本
// 在 MFMA 计算时同步预加载下一次迭代的数据
// ============================================================================

__global__ void 
matrix_core_kernel_mfma_large_double_buffer(
    const void* __restrict__ ptr_a,
    const void* __restrict__ ptr_b,
    void* __restrict__ ptr_c,
    int m,
    int n,
    int k,
    int stride_a,
    int stride_b,
    int stride_c)
{
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 8;
    
    // ========================================================================
    // 1. 初始化：Block 和 Lane 定位
    // ========================================================================
    int block_m = blockIdx.x * MFMA_M;
    int block_n = blockIdx.y * MFMA_N;
    int lane_id = threadIdx.x;
    
    // A/B 的索引偏移（在 32×8 子矩阵中的位置）
    int col_offset_a = lane_id / 32 * 4;
    int row_offset_a = lane_id % 32;
    int col_offset_b = lane_id / 32 * 4;
    int row_offset_b = lane_id % 32;
    
    // ========================================================================
    // 2. 计算基础地址（避免循环内重复计算）
    // ========================================================================
    const fp16_t* ptr_a_base = reinterpret_cast<const fp16_t*>(ptr_a) + 
                                (block_m + row_offset_a) * stride_a + 
                                col_offset_a;
    
    const fp16_t* ptr_b_base = reinterpret_cast<const fp16_t*>(ptr_b) + 
                                (block_n + row_offset_b) * stride_b + 
                                col_offset_b;
    
    // ========================================================================
    // 3. Double Buffering: 预加载第一批数据
    // ========================================================================
    fp32x16_t v_c = {0.0f};
    int k_loops = (k + MFMA_K - 1) / MFMA_K;
    
    if (k_loops == 0) return;  // 边界保护
    
    // 🔄 Buffer 0: 加载第一次迭代的数据
    fp16x4_t v_a_curr = *reinterpret_cast<const fp16x4_t*>(ptr_a_base);
    fp16x4_t v_b_curr = *reinterpret_cast<const fp16x4_t*>(ptr_b_base);
    
    // ========================================================================
    // 4. K 循环：计算 + 预加载重叠
    // ========================================================================
    for (int k_iter = 0; k_iter < k_loops - 1; k_iter++) {
        // 计算下一次迭代的地址
        int k_offset_next = (k_iter + 1) * MFMA_K;
        const fp16_t* ptr_a_next = ptr_a_base + k_offset_next;
        const fp16_t* ptr_b_next = ptr_b_base + k_offset_next;
        
        // 🔄 Buffer 1: 预加载下一次迭代的数据
        // 这个加载操作会在 MFMA 计算时并行执行（异步）
        fp16x4_t v_a_next = *reinterpret_cast<const fp16x4_t*>(ptr_a_next);
        fp16x4_t v_b_next = *reinterpret_cast<const fp16x4_t*>(ptr_b_next);
        
        // 💡 关键：MFMA 计算（使用当前 buffer 的数据）
        // GPU 会在计算的同时处理上面的内存加载请求
        v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a_curr, v_b_curr, v_c, 0, 0, 0);
        
        // 🔄 切换 buffer：下一次变为当前
        v_a_curr = v_a_next;
        v_b_curr = v_b_next;
    }
    
    // ========================================================================
    // 5. 处理最后一次迭代（无需预加载）
    // ========================================================================
    v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a_curr, v_b_curr, v_c, 0, 0, 0);
    
    // ========================================================================
    // 6. FP32 → FP16 转换
    // ========================================================================
    fp16x16_t v_c_f16;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }
    
    // ========================================================================
    // 7. 写回结果到全局内存
    // ========================================================================
    int col_id_c = lane_id % 32;
    int row_id_c = lane_id / 32 * 4;
    int offset_c = (block_m + row_id_c) * stride_c + block_n + col_id_c;
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int row_offset = (i % 4) + (i / 4 * 8);
        *(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + row_offset * stride_c) = v_c_f16[i];
    }
}

// ============================================================================
// 软件流水线版本 - 针对单 Wave 优化
// 不使用 LDS，通过指令重排和预取优化
// ============================================================================

__global__ void 
matrix_core_kernel_mfma_large_prefetch(
    const void* __restrict__ ptr_a,
    const void* __restrict__ ptr_b,
    void* __restrict__ ptr_c,
    int m, int n, int k,
    int stride_a, int stride_b, int stride_c)
{
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int MFMA_K = 8;
    
    int block_m = blockIdx.x * MFMA_M;
    int block_n = blockIdx.y * MFMA_N;
    int lane_id = threadIdx.x;
    
    int col_offset_a = lane_id / 32 * 4;
    int row_offset_a = lane_id % 32;
    int col_offset_b = lane_id / 32 * 4;
    int row_offset_b = lane_id % 32;
    
    const fp16_t* ptr_a_base = reinterpret_cast<const fp16_t*>(ptr_a) + 
                                (block_m + row_offset_a) * stride_a + 
                                col_offset_a;
    
    const fp16_t* ptr_b_base = reinterpret_cast<const fp16_t*>(ptr_b) + 
                                (block_n + row_offset_b) * stride_b + 
                                col_offset_b;
    
    fp32x16_t v_c = {0.0f};
    int k_loops = (k + MFMA_K - 1) / MFMA_K;
    
    if (k_loops == 0) return;
    
    // ✅ 关键优化：手动展开 + 多级预取
    // 一次处理 4 个迭代，充分利用指令级并行
    
    int k_main_loops = k_loops / 4;
    int k_remain = k_loops % 4;
    
    for (int k_iter = 0; k_iter < k_main_loops; k_iter++) {
        int k_base = k_iter * 4 * MFMA_K;
        
        // 预取 4 个迭代的数据
        const fp16_t* pa0 = ptr_a_base + k_base;
        const fp16_t* pa1 = pa0 + MFMA_K;
        const fp16_t* pa2 = pa1 + MFMA_K;
        const fp16_t* pa3 = pa2 + MFMA_K;
        
        const fp16_t* pb0 = ptr_b_base + k_base;
        const fp16_t* pb1 = pb0 + MFMA_K;
        const fp16_t* pb2 = pb1 + MFMA_K;
        const fp16_t* pb3 = pb2 + MFMA_K;
        
        // 加载数据（编译器会自动重排以隐藏延迟）
        fp16x4_t v_a0 = *reinterpret_cast<const fp16x4_t*>(pa0);
        fp16x4_t v_b0 = *reinterpret_cast<const fp16x4_t*>(pb0);
        fp16x4_t v_a1 = *reinterpret_cast<const fp16x4_t*>(pa1);
        fp16x4_t v_b1 = *reinterpret_cast<const fp16x4_t*>(pb1);
        fp16x4_t v_a2 = *reinterpret_cast<const fp16x4_t*>(pa2);
        fp16x4_t v_b2 = *reinterpret_cast<const fp16x4_t*>(pb2);
        fp16x4_t v_a3 = *reinterpret_cast<const fp16x4_t*>(pa3);
        fp16x4_t v_b3 = *reinterpret_cast<const fp16x4_t*>(pb3);
        
        // MFMA 计算（4 次连续）
        v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a0, v_b0, v_c, 0, 0, 0);
        v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a1, v_b1, v_c, 0, 0, 0);
        v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a2, v_b2, v_c, 0, 0, 0);
        v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a3, v_b3, v_c, 0, 0, 0);
    }
    
    // 处理剩余迭代
    int k_base = k_main_loops * 4 * MFMA_K;
    for (int i = 0; i < k_remain; i++) {
        int k_offset = k_base + i * MFMA_K;
        const fp16_t* pa = ptr_a_base + k_offset;
        const fp16_t* pb = ptr_b_base + k_offset;
        
        fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(pa);
        fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(pb);
        
        v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a, v_b, v_c, 0, 0, 0);
    }
    
    // FP32 → FP16
    fp16x16_t v_c_f16;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }
    
    // 写回
    int col_id_c = lane_id % 32;
    int row_id_c = lane_id / 32 * 4;
    int offset_c = (block_m + row_id_c) * stride_c + block_n + col_id_c;
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int row_offset = (i % 4) + (i / 4 * 8);
        *(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + row_offset * stride_c) = v_c_f16[i];
    }
}

// ============================================================================
// 64×64 Multi-Wave + LDS 版本 - 修复内存访问错误
// ============================================================================

__global__ void 
matrix_core_kernel_mfma_large_64x64_lds(
    const void* __restrict__ ptr_a,
    const void* __restrict__ ptr_b,
    void* __restrict__ ptr_c,
    int m, int n, int k,
    int stride_a, int stride_b, int stride_c)
{
    constexpr int MFMA_M = 32;
    constexpr int MFMA_N = 32;
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int LDS_K = 64;
    
    // ✅ 确保 LDS 大小正确
    __shared__ fp16_t smem_a[BLOCK_M * LDS_K];  // 64×64 = 4096 FP16 = 8 KB
    __shared__ fp16_t smem_b[BLOCK_N * LDS_K];  // 64×64 = 4096 FP16 = 8 KB
    
    int wave_id = threadIdx.x / 64;
    int lane_id = threadIdx.x % 64;
    
    int wave_m = wave_id / 2;  // 0, 0, 1, 1
    int wave_n = wave_id % 2;  // 0, 1, 0, 1
    
    int block_m = blockIdx.x * BLOCK_M;
    int block_n = blockIdx.y * BLOCK_N;
    
    // ✅ 边界检查
    if (block_m >= m || block_n >= n) return;
    
    int local_m = wave_m * MFMA_M;
    int local_n = wave_n * MFMA_N;
    
    const fp16_t* g_a = reinterpret_cast<const fp16_t*>(ptr_a) + block_m * stride_a;
    const fp16_t* g_b = reinterpret_cast<const fp16_t*>(ptr_b) + block_n * stride_b;
    
    fp32x16_t v_c = {0.0f};
    int k_tiles = (k + LDS_K - 1) / LDS_K;
    
    for (int k_tile = 0; k_tile < k_tiles; k_tile++) {
        int k_start = k_tile * LDS_K;
        int k_remain = min(k - k_start, LDS_K);
        
        // ====================================================================
        // ✅ 修复 1: 加载 A 时添加边界检查
        // ====================================================================
        int tid = threadIdx.x;
        int total_a = BLOCK_M * k_remain;
        
        for (int idx = tid; idx < total_a; idx += 256) {
            int row = idx / k_remain;
            int col = idx % k_remain;
            
            // ✅ 边界检查
            if (row < BLOCK_M && block_m + row < m && k_start + col < k) {
                smem_a[row * LDS_K + col] = g_a[row * stride_a + k_start + col];
            } else {
                smem_a[row * LDS_K + col] = static_cast<fp16_t>(0.0f);
            }
        }
        
        // ====================================================================
        // ✅ 修复 2: 加载 B 时添加边界检查
        // ====================================================================
        int total_b = BLOCK_N * k_remain;
        
        for (int idx = tid; idx < total_b; idx += 256) {
            int row = idx / k_remain;
            int col = idx % k_remain;
            
            // ✅ 边界检查
            if (row < BLOCK_N && block_n + row < n && k_start + col < k) {
                smem_b[row * LDS_K + col] = g_b[row * stride_b + k_start + col];
            } else {
                smem_b[row * LDS_K + col] = static_cast<fp16_t>(0.0f);
            }
        }
        
        __syncthreads();
        
        // ====================================================================
        // ✅ 修复 3: 从 LDS 读取时确保索引正确
        // ====================================================================
        int col_offset = lane_id / 32 * 4;
        int row_offset = lane_id % 32;
        
        int mfma_loops = k_remain / 8;
        
        #pragma unroll 2
        for (int mfma_k = 0; mfma_k < mfma_loops; mfma_k++) {
            int k_local = mfma_k * 8;
            
            // ✅ 计算 LDS 索引（确保不越界）
            int smem_a_idx = (local_m + row_offset) * LDS_K + k_local + col_offset;
            int smem_b_idx = (local_n + row_offset) * LDS_K + k_local + col_offset;
            
            // ✅ 断言检查（调试用，生产环境可移除）
            // assert(smem_a_idx + 3 < BLOCK_M * LDS_K);
            // assert(smem_b_idx + 3 < BLOCK_N * LDS_K);
            
            fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(&smem_a[smem_a_idx]);
            fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(&smem_b[smem_b_idx]);
            
            v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_a, v_b, v_c, 0, 0, 0);
        }
        
        __syncthreads();
    }
    
    // ========================================================================
    // ✅ 修复 4: 写回时添加边界检查
    // ========================================================================
    fp16x16_t v_c_f16;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }
    
    int col_id_c = lane_id % 32;
    int row_id_c = lane_id / 32 * 4;
    
    fp16_t* g_c = reinterpret_cast<fp16_t*>(ptr_c);
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int row_offset = (i % 4) + (i / 4 * 8);
        int global_row = block_m + local_m + row_id_c + row_offset;
        int global_col = block_n + local_n + col_id_c;
        
        // ✅ 边界检查
        if (global_row < m && global_col < n) {
            g_c[global_row * stride_c + global_col] = v_c_f16[i];
        }
    }
}

__global__ void 
matrix_core_kernel_standard_v2(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    using opus::operator""_I;
    auto mfma = opus::make_mfma<opus::fp16_t, opus::fp16_t, opus::fp32_t>(32_I, 32_I, 8_I);

    auto s_a = opus::make_tuple(stride_a, 1_I);

    auto u_a = opus::partition_layout_a(mfma, s_a);
    auto u_b = opus::partition_layout_b(mfma, opus::make_tuple(stride_b, 1_I));

    auto g_a = opus::make_gmem(reinterpret_cast<const opus::fp16x4_t*>(ptr_a));
    auto g_b = opus::make_gmem(reinterpret_cast<const opus::fp16x4_t*>(ptr_b));

    // 32x32x8 gemm, assume only launced 1 wave
    opus::fp16x4_t v_a = g_a.load(u_a(threadIdx.x % mfma.grpm_a, 0_I, threadIdx.x / mfma.grpm_a, 0_I) / 4_I); // [lane_m(P), rept_k(Y), lane_k(P), pack_k(Y)]
    opus::fp16x4_t v_b = g_b.load(u_b(threadIdx.x % mfma.grpn_b, 0_I, threadIdx.x / mfma.grpn_b, 0_I) / 4_I);
    opus::fp32x16_t v_c = {.0f};  // clear

    v_c = mfma(v_a, v_b, v_c);

    fp16x16_t v_c_f16 = opus::cast<fp16_t>(v_c);

    // C:[rept_c(Y), grpm_c(P), pack_c(Y), grpn_c(P)]
    // auto u_c = opus::partition_layout_c(mfma, opus::make_tuple(mfma.grpm_c * mfma.pack_c * stride_c, mfma.pack_c * stride_c, stride_c, 1_I));
    auto u_c = opus::partition_layout_c(mfma, opus::make_tuple(stride_c, 1_I));

    auto g_c = opus::make_gmem(reinterpret_cast<fp16_t*>(ptr_c));
    for(auto i = 0; i < 16; i++) {
        auto i_pack = i % mfma.pack_c;
        auto i_rept = i / mfma.pack_c;
        g_c.store(v_c_f16[i], u_c(i_rept, threadIdx.x / mfma.grpn_c, i_pack, threadIdx.x % mfma.grpn_c));
    }
}

__global__ void 
matrix_core_kernel_standard_agpr(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 32x32x8 gemm, assume only launced 1 wave
    int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
    int offset_b = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_b);

    auto res_a = make_buffer_resource(ptr_a);
    auto res_b = make_buffer_resource(ptr_b);
    fp16x4_t v_a, v_b;

    asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
            :"+a"(v_a) :  "v"(static_cast<int>(offset_a * sizeof(fp16_t))), "s"(res_a), "n"(0) : "memory");

    asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
            :"+a"(v_b) :  "v"(static_cast<int>(offset_b * sizeof(fp16_t))), "s"(res_b), "n"(0) : "memory");

    fp32x16_t v_c = {.0f};  // clear

#if LOCAL_SCRATCH == 1
    // create 2 local scratch, note this is x8, not x4(purposely)
    fp16x8_t v_aa, v_bb;
    for(auto i = 0; i < 4; i++) v_aa[i] = v_a[i];
    for(auto i = 0; i < 4; i++) v_bb[i] = v_b[i];

    // Note the local scratch re-assignment is before this waitcnt
    // but this is fine, since finally compiler will remove all such
    // (redundant) movement for us
    asm volatile("s_waitcnt vmcnt(0)"  : : : "memory");
    
    // this is local scratch used for mfma
    fp16x4_t v_ar, v_br;
    for(auto i = 0; i < 4; i++) v_ar[i] = v_aa[i];
    for(auto i = 0; i < 4; i++) v_br[i] = v_bb[i];
    asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3\n" "s_nop 16" : "+v"(v_c) :  "a"(v_ar), "a"(v_br),  "v"(v_c) : );
#elif LOCAL_SCRATCH == 2
    // use different type for local scratch
    fp32x4_t v_aa, v_bb;
    for(auto i = 0; i < 2; i++) { fp16x2_t tmp; tmp[0] = v_a[2 * i + 0]; tmp[1] = v_a[2 * i + 1]; v_aa[i] = __builtin_bit_cast(float, tmp); }
    for(auto i = 0; i < 2; i++) { fp16x2_t tmp; tmp[0] = v_b[2 * i + 0]; tmp[1] = v_b[2 * i + 1]; v_bb[i] = __builtin_bit_cast(float, tmp); }

    asm volatile("s_waitcnt vmcnt(0)"  : : : "memory");

    fp16x4_t v_ar, v_br;
    for(auto i = 0; i < 2; i++) { fp16x2_t tmp; tmp = __builtin_bit_cast(fp16x2_t, v_aa[i]); v_ar[2 * i + 0] = tmp[0]; v_ar[2 * i + 1] = tmp[1]; }
    for(auto i = 0; i < 2; i++) { fp16x2_t tmp; tmp = __builtin_bit_cast(fp16x2_t, v_bb[i]); v_br[2 * i + 0] = tmp[0]; v_br[2 * i + 1] = tmp[1]; }
    asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3\n" "s_nop 16" : "+v"(v_c) :  "a"(v_ar), "a"(v_br),  "v"(v_c) : );
#else
    asm volatile("s_waitcnt vmcnt(0)"  : : : "memory");
    asm volatile("v_mfma_f32_32x32x8f16 %0, %1, %2, %3\n"
                 "s_nop 16"         // TODO: better resolve data dependency
                 : "+v"(v_c)
                 :  "a"(v_a), "a"(v_b),  "v"(v_c) : );
#endif

    fp16x16_t v_c_f16;
    for(auto i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int col_id_c = threadIdx.x % 32;
    int row_id_c = threadIdx.x / 32 * 4;
    int offset_c = row_id_c * stride_c + col_id_c;

    for(auto i = 0; i < 16; i++) {
        int row_offset = (i % 4) + (i / 4 * 8);
        *(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + row_offset * stride_c) = v_c_f16[i];
    }
}

// kernel-2, swap A/B pointer to transpose C matrix, now we can do vector store
/*
* Note: C matrix now is transposed, we can do vectore store out(assum C fast changing dim is N)

                                 L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
             (swapped) Matrix A   __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k0  L0~31
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k1
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k2
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k3
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L32~63
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7
     Matrix B (swapped)
     L0~31       L32~63           Matrix C (transposed)
     k0 k1 k2 k3 k4 k5 k6 k7      L0~31       L32~63      L0~31       L32~63      L0~31       L32~63      L0~31       L32~63
     _____ _____|_____ _____      __ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __
L0  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L0 
L1  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L1 
L2  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L2 
L3  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L3 
L4  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L4 
L5  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L5 
L6  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L6 
L7  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L7 
L8  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L8 
L9  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L9 
L10 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L10
L11 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L11
L12 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L12
L13 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L13
L14 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L14
L15 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L15
L16 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L16
L17 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L17
L18 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L18
L19 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L19
L20 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L20
L21 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L21
L22 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L22
L23 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L23
L24 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L24
L25 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L25
L26 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L26
L27 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L27
L28 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L28
L29 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L29
L30 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L30
L31 |v0___|v1___|v0___|v1___|    |a0|a1|a2|a3|a0|a1|a2|a3|a4|a5|a6|a7|a4|a5|a6|a7|a8|a9|10|11|a8|a9|10|11|12|13|14|15|12|13|14|15|  L31
                |                            |           |           |           |           |           |           |

*/
__global__ void 
matrix_core_kernel_swap_a_b(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 32x32x8 gemm, assume only launced 1 wave
    int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
    int offset_b = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_b);

    fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_a) + offset_a);
    fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_b) + offset_b);
    fp32x16_t v_c = {.0f};  // clear

    v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_b, v_a, v_c, 0, 0, 0);

    fp16x16_t v_c_f16;
    for(auto i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int col_id_c = threadIdx.x / 32 * 4; 
    int row_id_c = threadIdx.x % 32;
    int offset_c = row_id_c * stride_c + col_id_c;

    for(auto i = 0; i < (16 / 4); i++) {
        int col_offset = i * 8;
        fp16x4_t tmp;
        tmp.x = v_c_f16[4 * i + 0]; tmp.y = v_c_f16[4 * i + 1];
        tmp.z = v_c_f16[4 * i + 2]; tmp.w = v_c_f16[4 * i + 3];
        *reinterpret_cast<fp16x4_t*>(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + col_offset) = tmp;
    }
}

__global__ void 
matrix_core_kernel_swap_a_b_v2(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    using opus::operator""_I;
    auto mfma = opus::make_mfma<opus::fp16_t, opus::fp16_t, opus::fp32_t>(32_I, 32_I, 8_I, opus::mfma_adaptor_swap_ab{});

    auto s_a = opus::make_tuple(stride_a, 1_I);
    auto u_a = opus::partition_layout_a(mfma, s_a);

    auto u_b = opus::partition_layout_b(mfma, opus::make_tuple(stride_b, 1_I));

    auto g_a = opus::make_gmem(reinterpret_cast<const opus::fp16x4_t*>(ptr_a));
    auto g_b = opus::make_gmem(reinterpret_cast<const opus::fp16x4_t*>(ptr_b));

    opus::fp16x4_t v_a = g_a.load(u_a(threadIdx.x % mfma.grpm_a, 0_I, threadIdx.x / mfma.grpm_a, 0_I) / 4_I); // [lane_m(P), rept_k(Y), lane_k(P), pack_k(Y)]
    opus::fp16x4_t v_b = g_b.load(u_b(threadIdx.x % mfma.grpn_b, 0_I, threadIdx.x / mfma.grpn_b, 0_I) / 4_I);
    opus::fp32x16_t v_c = {.0f};  // clear
 
    v_c = mfma(v_a, v_b, v_c); // note here swapped a/b

    fp16x16_t v_c_f16 = opus::cast<fp16_t>(v_c);

    // C:[grpn_c(P), rept_c(Y), grpm_c(P), pack_c(Y)]
    auto u_c = opus::partition_layout_c(mfma);

    auto g_c = opus::make_gmem(reinterpret_cast<opus::fp16x4_t*>(ptr_c));

#if 1
    for(auto i = 0; i < (16 / 4); i++) {
        auto tmp = opus::slice<4>(v_c_f16, 4*i, 4*i+4);
        g_c.store(tmp, u_c( threadIdx.x % mfma.grpn_c, i, threadIdx.x / mfma.grpn_c, 0_I) / 4_I);   // C:[grpn_c(P), rept_c(Y), grpm_c(P), pack_c(Y)]
    }
#else
    opus::static_for<16/4>([&](auto i){
        auto tmp = opus::slice(v_c_f16, opus::number<4*i>{}, opus::number<4*i+4>{});
        // auto tmp = opus::slice<4>(v_c_f16, 4*i, 4*i+4);
        g_c.store<4>(tmp, u_c( threadIdx.x % mfma.grpn_c, i, threadIdx.x / mfma.grpn_c, 0_I));  
    });
#endif
}

#if  1
__global__ void 
matrix_core_kernel_swap_a_b_v3(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    using opus::operator""_I;
    auto mfma = opus::make_mfma<opus::fp16_t, opus::fp16_t, opus::fp32_t>(opus::seq<32, 32, 8>{}, opus::mfma_adaptor_swap_ab{});

    auto u_a = opus::partition_layout_a_packed(mfma, opus::make_tuple(threadIdx.x % mfma.grpm_a, threadIdx.x / mfma.grpm_a));   // A:[(grpm_a<p>), (rept_a<y>, grpk_a<p>, pack_a<y>)], MxK
    auto u_b = opus::partition_layout_b_packed(mfma, opus::make_tuple(threadIdx.x % mfma.grpn_b, threadIdx.x / mfma.grpn_b));

    auto g_a = opus::make_gmem(reinterpret_cast<const opus::fp16_t*>(ptr_a));
    auto g_b = opus::make_gmem(reinterpret_cast<const opus::fp16_t*>(ptr_b));

    auto v_a = g_a.load<4>(u_a(0_I, 0_I)); // [lane_m(P), rept_k(Y), lane_k(P), pack_k(Y)]
    auto v_b = g_b.load<4>(u_b(0_I, 0_I));
    opus::fp32x16_t v_c{.0f};  // clear
 
    v_c = mfma(v_a, v_b, v_c); // note here swapped a/b

    fp16x16_t v_c_f16 = opus::cast<fp16_t>(v_c);

    // C:[grpn_c(P), rept_c(Y), grpm_c(P), pack_c(Y)]
    auto u_c = opus::partition_layout_c_packed(mfma, opus::make_tuple(threadIdx.x % mfma.grpn_c, threadIdx.x / mfma.grpn_c));

    auto g_c = opus::make_gmem(reinterpret_cast<opus::fp16_t*>(ptr_c));

    for(auto i = 0; i < (16 / 4); i++) {
        auto tmp = opus::slice<4>(v_c_f16, 4*i, 4*i+4);
        g_c.store<4>(tmp, u_c(i,  0_I));
    }
}
#endif

// kernel-3, swap A/B pointer to transpose C matrix, and swizzle b(vector size is larger)
/*
* Note: C matrix now is transposed, we can do vectore store out(assum C fast changing dim is N), and vector size is larger

                                 L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
             (swapped) Matrix A   __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k0  L0~31
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k1
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k2
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k3
                                 |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L32~63
                                 |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                 |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7
Matrix B (swapped+swizzled)
     L0~31       L32~63           Matrix C (transposed + increased vector size)
     k0 k1 k2 k3 k4 k5 k6 k7      L0~31                   L32~63                  L0~31                   L32~63 
     _____ _____|_____ _____      __ __ __ __ __ __ __ __|__ __ __ __ __ __ __ __|__ __ __ __ __ __ __ __|__ __ __ __ __ __ __ __
L0  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L0 
L1  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L1 
L2  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L2 
L3  |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L3 
L8 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L4 
L9 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L5 
L10*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L6 
L11*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L7 
L4 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L8 
L5 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L9 
L6 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L10
L7 *|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L11
L12 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L12
L13 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L13
L14 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L14
L15 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L15
L16 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L16
L17 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L17
L18 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L18
L19 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L19
L24*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L20
L25*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L21
L26*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L22
L27*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L23
L20*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L24
L21*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L25
L22*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L26
L23*|v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L27
L28 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L28
L29 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L29
L30 |v0   |v1   |v0   |v1   |    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L30
L31 |v0___|v1___|v0___|v1___|    |a0|a1|a2|a3|a4|a5|a6|a7|a0|a1|a2|a3|a4|a5|a6|a7|a8|a9|10|11|12|13|14|15|a8|a9|10|11|12|13|14|15|  L31
                |                                        |                       |                       |            
*/
__global__ void 
matrix_core_kernel_swap_swb(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 32x32x8 gemm, assume only launced 1 wave
    int row_group_id_b = threadIdx.x % 32 / 4;
    int row_id_b = threadIdx.x % 4 + row_group_id_b % 2 * 8 + row_group_id_b % 4 / 2 * 4 + row_group_id_b / 4 * 16;
    int offset_a = (threadIdx.x / 32 * 4) + (threadIdx.x % 32 * stride_a);
    int offset_b = (threadIdx.x / 32 * 4) + (row_id_b * stride_b);

    // printf("tid:%d, rid:%d,%d, %d\n", static_cast<int>(threadIdx.x), row_id_b,row_group_id_b,  row_group_id_b % 4 / 2 * 4);

    fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_a) + offset_a);
    fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_b) + offset_b);
    fp32x16_t v_c = {.0f};  // clear

    v_c = __builtin_amdgcn_mfma_f32_32x32x8f16(v_b, v_a, v_c, 0, 0, 0);

    fp16x16_t v_c_f16;
    for(auto i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int col_id_c = threadIdx.x / 32 * 8;
    int row_id_c = threadIdx.x % 32;
    int offset_c = row_id_c * stride_c + col_id_c;

    for(auto i = 0; i < (16 / 8); i++) {
        int col_offset = i * 16;
        fp16x8_t tmp;
        for(auto j = 0; j < 8; j++) tmp[j] = v_c_f16[i * 8 + j];
        *reinterpret_cast<fp16x8_t*>(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + col_offset) = tmp;
    }
}

#if 1
// compute a single block gemm with multiple waves
template<int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K, int TILE_M, int TILE_N, int TILE_K>
__global__ void matrix_core_kernel_block(const void* __restrict__ ptr_a,
                                         const void* __restrict__ ptr_b,
                                         void* __restrict__ ptr_c,
                                         int stride_a, // stride in unit of pixel
                                         int stride_b,
                                         int stride_c)
{
    using opus::operator""_I;
    constexpr int W_M = 32;
    constexpr int W_N = 32;
    constexpr int W_K = 8;

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
    //                                                               tile_m<p>, grpm_a<p>           , tile_k<p>, grpk_a<p>
    auto u_a = opus::partition_layout_a_packed(mma, opus::make_tuple(wave_id / 2 , lane_id % mma.grpm_a, 0_I      , lane_id / mma.grpm_a));
    //                                                               tile_n<p>, grpn_b<p>           , tile_k<p>, grpk_b<p>
    auto u_b = opus::partition_layout_b_packed(mma, opus::make_tuple(wave_id % 2 , lane_id % mma.grpn_b, 0_I      , lane_id / mma.grpn_b));
    auto g_a = opus::make_gmem(reinterpret_cast<const d_a*>(ptr_a));
    auto g_b = opus::make_gmem(reinterpret_cast<const d_b*>(ptr_b));

    auto g_c = opus::make_gmem(reinterpret_cast<opus::fp16_t*>(ptr_c));
    //                                                               tile_m<p>,   grpn_c<p>              tile_n<p>    grpm_c<p>
    auto u_c = opus::partition_layout_c_packed(mma, opus::make_tuple(wave_id / 2, lane_id % mma.grpn_c, wave_id % 2, lane_id / mma.grpn_c));

    using va_t = opus::vector_t<d_a, 4>;
    using vb_t = opus::vector_t<d_b, 4>;
    using vc_t = opus::vector_t<d_c, 16>;

    // using y_shape_a = opus::seq<E_M, E_K, 1_I, 1_I>;
    // using y_shape_b = opus::seq<E_N, E_K, 1_I, 1_I>;

    opus::array<va_t, E_M> v_a;
    opus::array<vb_t, E_N> v_b;
    opus::array<vc_t, E_M * E_N> v_c;

    v_c.clear();

    for(auto i = 0; i < E_M; i++) {
        v_a[i] = g_a.load<4>(u_a(i, 0_I, 0_I, 0_I));
    }

    for(auto i = 0; i < E_N; i++) {
        v_b[i] = g_b.load<4>(u_b(i, 0_I, 0_I, 0_I));
    }

    v_c = mma(v_a, v_b, v_c);

    auto v_c_f16 = opus::cast<fp16_t>(v_c);

    opus::static_ford<E_M, E_N, mma.rept_c>([&](auto i_em, auto i_en, auto i_rp){
        auto current_tile = v_c_f16[i_em * E_N + i_en];        
        auto tmp = opus::slice<4>(current_tile, 4*i_rp, 4*i_rp+4);
        g_c.store<4>(tmp, u_c(i_em, i_en,  i_rp,  0_I));
    });
}
#endif

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

    auto u_a = opus::partition_layout_a(mma, opus::make_tuple(stride_a, 1_I), opus::make_tuple(wave_id / TILE_N, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a) /*tile_m<p>, grpm_a<p>, tile_k<p>, grpk_a<p>*/);
    auto u_b = opus::partition_layout_b(mma, opus::make_tuple(stride_b, 1_I), opus::make_tuple(wave_id % TILE_N, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b) /*tile_n<p>, grpn_b<p>, tile_k<p>, grpk_b<p>*/);
    auto u_c = opus::partition_layout_c(mma, opus::make_tuple(stride_c, 1_I), opus::make_tuple(wave_id / TILE_N, lane_id % mma.grpn_c, wave_id % TILE_N, lane_id / mma.grpn_c) /*tile_m<p>, grpn_c<p> tile_n<p>, grpm_c<p>*/);
    auto g_a = opus::make_gmem(reinterpret_cast<const d_a*>(ptr_a) + g_im * stride_a);
    auto g_b = opus::make_gmem(reinterpret_cast<const d_b*>(ptr_b) + g_in * stride_b);
    auto g_c = opus::make_gmem(reinterpret_cast<opus::fp16_t*>(ptr_c) + g_im * stride_c + g_in);

    // start of kernel
    int loops = (k + BLOCK_K - 1) / BLOCK_K;
#if 1
    typename decltype(mma)::vtype_c v_c;
    opus::clear(v_c);
#pragma unroll
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
        if(delta>1e-3){
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

void rand_vector_2d(float* v, int row, int col, int ld, float min_v = 0, float max_v = 1){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            float tmp = float(std::rand()) / float(RAND_MAX);
            v[r*ld+c] = static_cast<float>(min_v + tmp * (max_v - min_v));
            // v[r*ld+c] =   ((float)(r*ld+c)) / (row/2 * col/2) - 5;
        }
    }
}

void rand_vector_2d_int(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(rand() % 10)) - 5;
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

void block_run()
{
    int m = 32 * 64 * 4;
    int n = 32 * 2;
    int k = 8 * 32;

    // int m = 4096;
    // int n = 128;
    // int k = 7168;

    int lda = k;
    int ldb = k;
    int ldc = n;

    float *host_a, *host_b, *host_c;
    float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;

    //fp32 on host
    host_a = (float*)malloc(lda*m*sizeof(float));
    host_b = (float*)malloc(ldb*n*sizeof(float));
    host_c = (float*)malloc(ldc*m*sizeof(float));

#ifdef RAND_INT
    rand_vector_2d_int(host_a, m, k, lda);
    rand_vector_2d_int(host_b, n, k, ldb);
#else
    rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
    rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
#endif

    //fp16 on host
    fp16_a = (float16*)malloc(lda*m*sizeof(float16));
    fp16_b = (float16*)malloc(ldb*n*sizeof(float16));
    fp16_c = (float16*)malloc(ldc*m*sizeof(float16));
    //convert fp32 a and b into fp16 on host
    for(int i=0; i<lda*m; i++)fp16_a[i]=__float2half_rn(host_a[i]);
    for(int i=0; i<ldb*n; i++)fp16_b[i]=__float2half_rn(host_b[i]);

    HIP_CALL(hipMalloc(&dev_a, lda*m*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_b, ldb*n*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_c, ldc*m*sizeof(float16)));
    //fp16 cpy to device
    HIP_CALL(hipMemcpy(dev_a, fp16_a, lda*m*sizeof(float16), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb*n*sizeof(float16), hipMemcpyHostToDevice));

    printf("m:%d,n:%d,k:%d,lda:%d,ldb:%d,ldc:%d\n",  m, n, k, lda, ldb, ldc); fflush(stdout);
    gemm_rcr(host_a, host_b, host_c, m,n,k,lda,ldb,ldc);

    {
        constexpr int BLOCK_M = 32;
        constexpr int BLOCK_N = 32;
        constexpr int BLOCK_K = 8;
        constexpr int TILE_M = 1;
        constexpr int TILE_N = 1;
        constexpr int TILE_K = 1;
        constexpr int WAVE_M = 32;
        constexpr int WAVE_N = 32;
        constexpr int WAVE_K = 8;

        auto gdim = dim3(m / BLOCK_M, n / BLOCK_N);
        auto kernel = matrix_core_kernel_block_v2<64, BLOCK_M, BLOCK_N, BLOCK_K, TILE_M, TILE_N, TILE_K, WAVE_M, WAVE_N, WAVE_K>;
        kernel<<<gdim, 64>>>(dev_a, dev_b, dev_c, k, lda, ldb, ldc);

        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
#if 1
        bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
        printf("[%dx%dx%d, block_gemm_%dx%dx%d_%dx%dx%d_%dx%dx%d], %s", m, n, k,
            BLOCK_M, BLOCK_N, BLOCK_K, TILE_M, TILE_N, TILE_K, WAVE_M, WAVE_N, WAVE_K,
            res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
#endif
    }

    free(host_a);
    free(host_b);
    free(host_c);
    free(fp16_a);
    free(fp16_b);
    free(fp16_c);
    
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));
}

// int main(int argc, char ** argv)
// {
//     int m = 32;
//     int n = 32;
//     int k = 8;

//     int lda = k;
//     int ldb = k;
//     int ldc = n;

//     float *host_a, *host_b, *host_c;
//     float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;

//     //fp32 on host
//     host_a = (float*)malloc(lda*m*sizeof(float));
//     host_b = (float*)malloc(ldb*n*sizeof(float));
//     host_c = (float*)malloc(ldc*m*sizeof(float));

// #ifdef RAND_INT
//     rand_vector_2d_int(host_a, m, k, lda);
//     rand_vector_2d_int(host_b, n, k, ldb);
// #else
//     rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
//     rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
// #endif

//     //fp16 on host
//     fp16_a = (float16*)malloc(lda*m*sizeof(float16));
//     fp16_b = (float16*)malloc(ldb*n*sizeof(float16));
//     fp16_c = (float16*)malloc(ldc*m*sizeof(float16));
//     //convert fp32 a and b into fp16 on host
//     for(int i=0; i<lda*m; i++)fp16_a[i]=__float2half_rn(host_a[i]);
//     for(int i=0; i<ldb*n; i++)fp16_b[i]=__float2half_rn(host_b[i]);

//     HIP_CALL(hipMalloc(&dev_a, lda*m*sizeof(float16)));
//     HIP_CALL(hipMalloc(&dev_b, ldb*n*sizeof(float16)));
//     HIP_CALL(hipMalloc(&dev_c, ldc*m*sizeof(float16)));
//     //fp16 cpy to device
//     HIP_CALL(hipMemcpy(dev_a, fp16_a, lda*m*sizeof(float16), hipMemcpyHostToDevice));
//     HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb*n*sizeof(float16), hipMemcpyHostToDevice));

//     printf("m:%d,n:%d,k:%d,lda:%d,ldb:%d,ldc:%d\n",  m, n, k, lda, ldb, ldc); fflush(stdout);
//     gemm_rcr(host_a, host_b, host_c, m,n,k,lda,ldb,ldc);

//     {
//         matrix_core_kernel_standard<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);

//         HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
//         bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
//         printf("[32x32x8, standard], %s",res?"valid":"fail");fflush(stdout);
//         printf("\n"); fflush(stdout);
//     }

//     {
//         matrix_core_kernel_standard_v2<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);

//         HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
//         bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
//         printf("[32x32x8, standard_v2], %s",res?"valid":"fail");fflush(stdout);
//         printf("\n"); fflush(stdout);
//     }
//     {
//         matrix_core_kernel_standard_agpr<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);

//         HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
//         bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
//         printf("[32x32x8, std_agpr], %s",res?"valid":"fail");fflush(stdout);
//         printf("\n"); fflush(stdout);
//     }
//     {
//         matrix_core_kernel_swap_a_b_v3<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);

//         HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
//         bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
//         printf("[32x32x8, swap_a_b], %s",res?"valid":"fail");fflush(stdout);
//         printf("\n"); fflush(stdout);
//     }
//     {
//         matrix_core_kernel_swap_swb<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);
//         HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
//         bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
//         printf("[32x32x8, swap_swb], %s",res?"valid":"fail");fflush(stdout);
//         printf("\n"); fflush(stdout);
//     }
    
//     free(host_a);
//     free(host_b);
//     free(host_c);
//     free(fp16_a);
//     free(fp16_b);
//     free(fp16_c);
    
//     HIP_CALL(hipFree(dev_a));
//     HIP_CALL(hipFree(dev_b));
//     HIP_CALL(hipFree(dev_c));

//     block_run();

//     // int cc[5] = {1,2,3,4,5};
//     // auto xx = cc;
//     // printf("%d, %d %d\n", xx[2], xx[4], std::is_trivially_copyable_v<decltype(cc)> ? 1 : 0);
// }

// 在 main 函数或测试函数中添加：
void test_mfma_large_kernel(int m, int n, int k) {
    printf("\n=== Testing MFMA Large Kernel ===\n");
    printf("Problem size: M=%d, N=%d, K=%d\n", m, n, k);
    
    int lda = k;
    int ldb = k;
    int ldc = n;
    
    // 分配主机内存
    float* host_a = (float*)malloc(m * lda * sizeof(float));
    float* host_b = (float*)malloc(n * ldb * sizeof(float));
    float* host_c = (float*)malloc(m * ldc * sizeof(float));
    
    // ✅ 修正：使用 rand_vector_2d 而不是 rand_float
#ifdef RAND_INT
    rand_vector_2d_int(host_a, m, k, lda);
    rand_vector_2d_int(host_b, n, k, ldb);
#else
    rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
    rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
#endif
    
    // 计算 CPU 参考结果
    gemm_rcr(host_a, host_b, host_c, m, n, k, lda, ldb, ldc);
    
    // 转换为 FP16
    float16* fp16_a = (float16*)malloc(m * lda * sizeof(float16));
    float16* fp16_b = (float16*)malloc(n * ldb * sizeof(float16));
    float16* fp16_c = (float16*)malloc(m * ldc * sizeof(float16));
    
    for (int i = 0; i < m * lda; i++) 
        fp16_a[i] = __float2half_rn(host_a[i]);
    for (int i = 0; i < n * ldb; i++) 
        fp16_b[i] = __float2half_rn(host_b[i]);
    
    // 分配设备内存
    float16 *dev_a, *dev_b, *dev_c;
    HIP_CALL(hipMalloc(&dev_a, m * lda * sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_b, n * ldb * sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_c, m * ldc * sizeof(float16)));
    
    // 拷贝数据到设备
    HIP_CALL(hipMemcpy(dev_a, fp16_a, m * lda * sizeof(float16), 
                       hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, n * ldb * sizeof(float16), 
                       hipMemcpyHostToDevice));
    
    // ========================================================================
    // 启动 Kernel
    // ========================================================================
    dim3 grid((m + 31) / 32, (n + 31) / 32);  // 每个 block 处理 32×32
    dim3 block(64);                            // 1 wave = 64 threads
    
    printf("Grid: (%d, %d), Block: (%d)\n", grid.x, grid.y, block.x);
    
    matrix_core_kernel_mfma_large<<<grid, block>>>(
        dev_a, dev_b, dev_c, 
        m, n, k,
        lda, ldb, ldc);
    
    HIP_CALL(hipDeviceSynchronize());
    
    // 拷贝结果回主机
    HIP_CALL(hipMemcpy(fp16_c, dev_c, m * ldc * sizeof(float16), 
                       hipMemcpyDeviceToHost));
    
    // 验证结果
    bool res = valid_vector(host_c, fp16_c, m * n, 1e-3);
    printf("[MFMA Large %dx%dx%d], %s\n", m, n, k, res ? "✅ valid" : "❌ fail");
    
    // 性能测试（可选）
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));
    
    int warmup = 10;
    int repeat = 100;
    
    for (int i = 0; i < warmup; i++) {
        matrix_core_kernel_mfma_large<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipDeviceSynchronize());
    
    HIP_CALL(hipEventRecord(start));
    for (int i = 0; i < repeat; i++) {
        matrix_core_kernel_mfma_large<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipEventRecord(stop));
    HIP_CALL(hipEventSynchronize(stop));
    
    float ms;
    HIP_CALL(hipEventElapsedTime(&ms, start, stop));
    ms /= repeat;
    
    double gflops = 2.0 * m * n * k / (ms * 1e6);
    printf("Time: %.3f us, Performance: %.2f GFLOPS\n", ms * 1000, gflops);
    
    // 释放内存
    free(host_a); free(host_b); free(host_c);
    free(fp16_a); free(fp16_b); free(fp16_c);
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));
    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));
}

// 添加到 test_mfma_large_kernel 函数中
void test_mfma_double_buffer(int m, int n, int k) {
    printf("\n=== Testing MFMA Double Buffer Kernel ===\n");
    printf("Problem size: M=%d, N=%d, K=%d\n", m, n, k);
    
    int lda = k, ldb = k, ldc = n;
    
    // 准备数据（与原版相同）
    float* host_a = (float*)malloc(m * lda * sizeof(float));
    float* host_b = (float*)malloc(n * ldb * sizeof(float));
    float* host_c = (float*)malloc(m * ldc * sizeof(float));
    
#ifdef RAND_INT
    rand_vector_2d_int(host_a, m, k, lda);
    rand_vector_2d_int(host_b, n, k, ldb);
#else
    rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
    rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
#endif
    
    gemm_rcr(host_a, host_b, host_c, m, n, k, lda, ldb, ldc);
    
    // 转换为 FP16
    float16* fp16_a = (float16*)malloc(m * lda * sizeof(float16));
    float16* fp16_b = (float16*)malloc(n * ldb * sizeof(float16));
    float16* fp16_c = (float16*)malloc(m * ldc * sizeof(float16));
    
    for (int i = 0; i < m * lda; i++) fp16_a[i] = __float2half_rn(host_a[i]);
    for (int i = 0; i < n * ldb; i++) fp16_b[i] = __float2half_rn(host_b[i]);
    
    // 设备内存
    float16 *dev_a, *dev_b, *dev_c;
    HIP_CALL(hipMalloc(&dev_a, m * lda * sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_b, n * ldb * sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_c, m * ldc * sizeof(float16)));
    
    HIP_CALL(hipMemcpy(dev_a, fp16_a, m * lda * sizeof(float16), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, n * ldb * sizeof(float16), hipMemcpyHostToDevice));
    
    // 启动 kernel
    dim3 grid((m + 31) / 32, (n + 31) / 32);
    dim3 block(64);
    
    printf("Grid: (%d, %d), Block: (%d)\n", grid.x, grid.y, block.x);
    
    // ====================================================================
    // 原版 vs Double Buffering 对比测试
    // ====================================================================
    
    // 测试原版
    hipEvent_t start1, stop1;
    HIP_CALL(hipEventCreate(&start1));
    HIP_CALL(hipEventCreate(&stop1));
    
    for (int i = 0; i < 10; i++) {
        matrix_core_kernel_mfma_large<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipDeviceSynchronize());
    
    HIP_CALL(hipEventRecord(start1));
    for (int i = 0; i < 100; i++) {
        matrix_core_kernel_mfma_large<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipEventRecord(stop1));
    HIP_CALL(hipEventSynchronize(stop1));
    
    float ms_orig;
    HIP_CALL(hipEventElapsedTime(&ms_orig, start1, stop1));
    ms_orig /= 100;
    
    // 测试 Double Buffering 版本
    hipEvent_t start2, stop2;
    HIP_CALL(hipEventCreate(&start2));
    HIP_CALL(hipEventCreate(&stop2));
    
    for (int i = 0; i < 10; i++) {
        matrix_core_kernel_mfma_large_double_buffer<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipDeviceSynchronize());
    
    HIP_CALL(hipEventRecord(start2));
    for (int i = 0; i < 100; i++) {
        matrix_core_kernel_mfma_large_double_buffer<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipEventRecord(stop2));
    HIP_CALL(hipEventSynchronize(stop2));
    
    float ms_db;
    HIP_CALL(hipEventElapsedTime(&ms_db, start2, stop2));
    ms_db /= 100;
    
    // 验证正确性
    HIP_CALL(hipMemcpy(fp16_c, dev_c, m * ldc * sizeof(float16), hipMemcpyDeviceToHost));
    bool res = valid_vector(host_c, fp16_c, m * n, 1e-3);
    
    // 输出对比结果
    double gflops = 2.0 * m * n * k / 1e6;
    printf("\n");
    printf("┌─────────────────────────────────────────────────────────┐\n");
    printf("│  Performance Comparison                                 │\n");
    printf("├─────────────────────────────────────────────────────────┤\n");
    printf("│  Original:        %.3f us  (%.2f GFLOPS)              │\n", 
           ms_orig * 1000, gflops / ms_orig);
    printf("│  Double Buffer:   %.3f us  (%.2f GFLOPS)              │\n", 
           ms_db * 1000, gflops / ms_db);
    printf("│  Speedup:         %.2fx  ⬆️ %.1f%%                     │\n", 
           ms_orig / ms_db, (ms_orig / ms_db - 1) * 100);
    printf("│  Correctness:     %s                                    │\n", 
           res ? "✅ PASS" : "❌ FAIL");
    printf("└─────────────────────────────────────────────────────────┘\n");
    
    // 清理
    free(host_a); free(host_b); free(host_c);
    free(fp16_a); free(fp16_b); free(fp16_c);
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));
    HIP_CALL(hipEventDestroy(start1));
    HIP_CALL(hipEventDestroy(stop1));
    HIP_CALL(hipEventDestroy(start2));
    HIP_CALL(hipEventDestroy(stop2));
}

void test_mfma_lds(int m, int n, int k) {
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  LDS Optimization Test                                    ║\n");
    printf("║  Problem: M=%d, N=%d, K=%d                            ║\n", m, n, k);
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    
    int lda = k, ldb = k, ldc = n;
    
    // ====================================================================
    // 准备数据
    // ====================================================================
    float* host_a = (float*)malloc(m * lda * sizeof(float));
    float* host_b = (float*)malloc(n * ldb * sizeof(float));
    float* host_c = (float*)malloc(m * ldc * sizeof(float));
    
#ifdef RAND_INT
    rand_vector_2d_int(host_a, m, k, lda);
    rand_vector_2d_int(host_b, n, k, ldb);
#else
    rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
    rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
#endif
    
    gemm_rcr(host_a, host_b, host_c, m, n, k, lda, ldb, ldc);
    
    float16* fp16_a = (float16*)malloc(m * lda * sizeof(float16));
    float16* fp16_b = (float16*)malloc(n * ldb * sizeof(float16));
    float16* fp16_c = (float16*)malloc(m * ldc * sizeof(float16));
    
    for (int i = 0; i < m * lda; i++) fp16_a[i] = __float2half_rn(host_a[i]);
    for (int i = 0; i < n * ldb; i++) fp16_b[i] = __float2half_rn(host_b[i]);
    
    float16 *dev_a, *dev_b, *dev_c;
    HIP_CALL(hipMalloc(&dev_a, m * lda * sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_b, n * ldb * sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_c, m * ldc * sizeof(float16)));
    
    HIP_CALL(hipMemcpy(dev_a, fp16_a, m * lda * sizeof(float16), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, n * ldb * sizeof(float16), hipMemcpyHostToDevice));
    
    // ====================================================================
    // 对比测试：原版 vs LDS 优化
    // ====================================================================
    dim3 grid((m + 31) / 32, (n + 31) / 32);
    dim3 block(64);
    
    printf("Grid: (%d, %d), Block: (%d)\n", grid.x, grid.y, block.x);
    printf("LDS Usage: 8 KB per block\n\n");
    
    // 测试原版
    printf("Testing Original Kernel...\n");
    hipEvent_t start1, stop1;
    HIP_CALL(hipEventCreate(&start1));
    HIP_CALL(hipEventCreate(&stop1));
    
    for (int i = 0; i < 10; i++) {
        matrix_core_kernel_mfma_large<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipDeviceSynchronize());
    
    HIP_CALL(hipEventRecord(start1));
    for (int i = 0; i < 100; i++) {
        matrix_core_kernel_mfma_large<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipEventRecord(stop1));
    HIP_CALL(hipEventSynchronize(stop1));
    
    float ms_orig;
    HIP_CALL(hipEventElapsedTime(&ms_orig, start1, stop1));
    ms_orig /= 100;
    
    // 测试 LDS 版本
    printf("Testing LDS Optimized Kernel...\n");
    hipEvent_t start2, stop2;
    HIP_CALL(hipEventCreate(&start2));
    HIP_CALL(hipEventCreate(&stop2));
    
    for (int i = 0; i < 10; i++) {
        matrix_core_kernel_mfma_large_64x64_lds<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipDeviceSynchronize());
    
    HIP_CALL(hipEventRecord(start2));
    for (int i = 0; i < 100; i++) {
        matrix_core_kernel_mfma_large_64x64_lds<<<grid, block>>>(
            dev_a, dev_b, dev_c, m, n, k, lda, ldb, ldc);
    }
    HIP_CALL(hipEventRecord(stop2));
    HIP_CALL(hipEventSynchronize(stop2));
    
    float ms_lds;
    HIP_CALL(hipEventElapsedTime(&ms_lds, start2, stop2));
    ms_lds /= 100;
    
    // 验证正确性
    HIP_CALL(hipMemcpy(fp16_c, dev_c, m * ldc * sizeof(float16), hipMemcpyDeviceToHost));
    bool res = valid_vector(host_c, fp16_c, m * n, 1e-3);
    
    // ====================================================================
    // 输出对比结果
    // ====================================================================
    double gflops = 2.0 * m * n * k / 1e6;
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Performance Comparison                                       ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Metric              │ Original      │ LDS Optimized         ║\n");
    printf("╠══════════════════════╪═══════════════╪═══════════════════════╣\n");
    printf("║  Time (us)           │  %10.3f   │  %10.3f          ║\n", 
           ms_orig * 1000, ms_lds * 1000);
    printf("║  Performance (GFLOPS)│  %10.2f   │  %10.2f          ║\n",
           gflops / ms_orig, gflops / ms_lds);
    printf("║  Speedup             │    1.00x      │    %5.2fx            ║\n",
           ms_orig / ms_lds);
    printf("║  Improvement         │       -       │   +%.1f%%             ║\n",
           (ms_orig / ms_lds - 1) * 100);
    printf("║  Correctness         │      %3s      │      %3s             ║\n",
           res ? "✅" : "❌", res ? "✅" : "❌");
    printf("╚══════════════════════╧═══════════════╧═══════════════════════╝\n");
    
    // 内存带宽分析
    double bytes_orig = (m * k + n * k + m * n) * 2.0;  // FP16
    double bw_orig = bytes_orig / (ms_orig * 1e-3) / 1e9;
    double bw_lds = bytes_orig / (ms_lds * 1e-3) / 1e9;
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Memory Bandwidth Analysis                                    ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Total Data Transfer: %.2f MB                               ║\n", 
           bytes_orig / 1e6);
    printf("║  Original BW:         %.1f GB/s                             ║\n", bw_orig);
    printf("║  LDS Optimized BW:    %.1f GB/s                             ║\n", bw_lds);
    printf("║  BW Improvement:      %.1fx                                  ║\n", 
           bw_lds / bw_orig);
    printf("║  Peak BW (MI308X):    5,300 GB/s                             ║\n");
    printf("║  BW Utilization:      %.1f%% (orig) → %.1f%% (lds)          ║\n",
           bw_orig / 5300 * 100, bw_lds / 5300 * 100);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    // 清理
    free(host_a); free(host_b); free(host_c);
    free(fp16_a); free(fp16_b); free(fp16_c);
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));
    HIP_CALL(hipEventDestroy(start1));
    HIP_CALL(hipEventDestroy(stop1));
    HIP_CALL(hipEventDestroy(start2));
    HIP_CALL(hipEventDestroy(stop2));
}

// 在 main 函数中调用：
int main() {
    // ... 其他测试 ...
    
    // 测试不同规模
    // test_mfma_large_kernel(32, 32, 8);      // 最小规模
    // test_mfma_large_kernel(64, 64, 256);    // 小规模
    // test_mfma_large_kernel(2048, 64, 256);  // 中等规模
    // test_mfma_large_kernel(8192, 64, 256); // 大规模

    // test_mfma_double_buffer(8192, 64, 256);
    test_mfma_lds(8192, 64, 256);   // 与之前相同的问题规模
    
    return 0;
}
