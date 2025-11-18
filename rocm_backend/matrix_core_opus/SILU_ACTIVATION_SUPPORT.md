# SiLU Activation 支持

## 📋 概述

本文档详细说明了在 Causal Conv1D 实现中添加 SiLU (Sigmoid Linear Unit) activation 函数的实现方案。

---

## 什么是 SiLU？

### 数学定义

**SiLU** (Sigmoid Linear Unit)，也被称为 **Swish**，是一种平滑的非线性激活函数：

```
SiLU(x) = x · sigmoid(x) = x · (1 / (1 + e^(-x)))
```

### 函数特点

1. **平滑可微**：在整个定义域上连续且可微
2. **非单调**：与 ReLU 不同，SiLU 在负值区域不是恒为 0
3. **自门控**：通过 sigmoid 函数实现自适应的门控机制
4. **性能优异**：在许多深度学习任务中表现优于 ReLU

### 函数图像特性

- 当 `x → +∞` 时，`SiLU(x) → x` (线性增长)
- 当 `x → -∞` 时，`SiLU(x) → 0` (趋近于 0)
- 在 `x = 0` 附近平滑过渡

---

## 实现方案

### 架构选择：融合 Kernel

为了最大化性能，我们将 **bias 加法** 和 **SiLU activation** 融合到单个 GPU kernel 中：

```
传统方案（两次内存访问）:
  1. add_bias_kernel: C[h,c] += bias[c]
  2. silu_kernel: C[h,c] = SiLU(C[h,c])

融合方案（一次内存访问）:
  add_bias_silu_fused_kernel: 
    x = C[h,c] + bias[c]
    C[h,c] = x / (1 + exp(-x))
```

**性能优势：**
- ✅ 减少内存读写次数：从 4 次（读-写-读-写）降低到 2 次（读-写）
- ✅ 提高缓存命中率：中间结果无需写回显存
- ✅ 减少 kernel 启动开销：从 2 次 kernel 调用降低到 1 次

---

## GPU Kernel 实现

### 1. 融合 Kernel（推荐）

```cpp
__global__ void add_bias_silu_fused_kernel(
    fp16_t* __restrict__ output,       // 输出 [ho, ci] (in-place)
    const fp16_t* __restrict__ bias,   // bias [ci]
    int ho, int ci)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;  // 输出位置 [0, ho)
    int c = blockIdx.y * blockDim.y + threadIdx.y;  // channel [0, ci)
    
    if (h >= ho || c >= ci) return;
    
    int idx = h * ci + c;
    
    // 步骤1: 添加 bias
    float x = (float)output[idx] + (float)bias[c];
    
    // 步骤2: 应用 SiLU activation
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    float silu_x = x * sigmoid_x;
    
    output[idx] = (fp16_t)silu_x;
}
```

**实现细节：**

1. **FP16 → FP32 转换**：
   - 输入和 bias 从 FP16 转换为 FP32 进行计算
   - 避免 FP16 精度损失（特别是在 `exp` 计算中）

2. **数学稳定性**：
   - 使用 `expf(-x)` 而非 `exp(-x)`（FP32 指数函数）
   - 对于大的正值 `x`，`exp(-x)` 接近 0，`sigmoid(x)` 接近 1
   - 对于大的负值 `x`，`exp(-x)` 很大，但不会溢出（分母起保护作用）

3. **内存访问模式**：
   - 线程 `(h, c)` 访问 `output[h * ci + c]`：连续访问同一行的元素
   - 线程块内合并内存访问（coalesced access）
   - Bias 广播：所有处理相同 channel 的线程共享 `bias[c]`

### 2. 独立 SiLU Kernel（备选）

如果需要单独的 SiLU kernel（例如，在没有 bias 的情况下）：

```cpp
__global__ void silu_activation_kernel(
    fp16_t* __restrict__ output,       // 输出 [ho, ci] (in-place)
    int ho, int ci)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (h >= ho || c >= ci) return;
    
    int idx = h * ci + c;
    float x = (float)output[idx];
    
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    float silu_x = x * sigmoid_x;
    
    output[idx] = (fp16_t)silu_x;
}
```

---

## Host 端参考实现

### Depthwise Conv1D + SiLU

在 `causal_conv1d_depthwise` 函数中添加 SiLU：

```cpp
void causal_conv1d_depthwise(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C, int H, int kernel_size)
{
    int pad = kernel_size - 1;
    int H_pad = H + pad;
    
    // 1. Padding
    std::vector<float> padded(N * C * H_pad, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                padded[n * C * H_pad + c * H_pad + pad + h] =
                    input[n * C * H + c * H + h];
            }
        }
    }
    
    // 2. Convolution + Bias + SiLU
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < H; ++t) {
                // 卷积
                float sum = bias ? bias[c] : 0.0f;
                for (int k = 0; k < kernel_size; ++k) {
                    float val = padded[n * C * H_pad + c * H_pad + t + k];
                    float w = weight[c * kernel_size + k];
                    sum += val * w;
                }
                
                // SiLU activation
                float sigmoid_x = 1.0f / (1.0f + expf(-sum));
                float silu_x = sum * sigmoid_x;
                
                output[n * C * H + c * H + t] = silu_x;
            }
        }
    }
}
```

---

## Kernel 调用

### GPU 执行流程

```cpp
// 在 casual_conv1d_block_run 函数中

// ========== 执行 GEMM ==========
for (int b = 0; b < batch; b++) {
    matrix_core_kernel_block_v2<<<gdim, 256>>>(
        dev_a + b*lda*m,    // A [ho, hk*ci]
        dev_b,              // B [hk*ci, ci]
        dev_c + b*ldc*m,    // C [ho, ci]
        k, lda, ldb, ldc);
}

// ========== 添加 Bias + SiLU (融合) ==========
dim3 bias_block_dim(16, 16);
dim3 bias_grid_dim((ho + 15) / 16, (ci + 15) / 16);

for (int b = 0; b < batch; b++) {
    add_bias_silu_fused_kernel<<<bias_grid_dim, bias_block_dim>>>(
        reinterpret_cast<fp16_t*>(dev_c + b*ldc*m),  // [ho, ci]
        reinterpret_cast<fp16_t*>(dev_bias),          // [ci]
        ho, ci);
}
```

### 线程配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `block_dim` | `(16, 16)` | 每个线程块 256 个线程 |
| `grid_dim` | `((ho+15)/16, (ci+15)/16)` | 覆盖整个输出矩阵 |
| 每个线程 | 1 个输出元素 | `(h, c)` 处理 `output[h, c]` |

---

## 测试验证

### 验证流程

1. **Host 端参考计算**：
   - 执行完整的 Conv1D + Bias + SiLU（CPU）

2. **GPU 计算**：
   - GEMM → Add Bias + SiLU (fused kernel)

3. **结果对比**：
   - 使用 `valid_vector` 函数比较 Host 和 GPU 结果
   - 容差：`nrms < 1e-3`

### 测试结果

#### Batch = 1
```
在 GPU 上添加 bias 并应用 SiLU activation (batch=1)...
✓ bias + SiLU activation 完成
[batch=1, 2048x64x256, block_gemm_32x32x16_2x2x1_16x16x16], valid
✓ GPU GEMM 输出验证通过! (batch 0)
✓ GPU 输出验证通过! (batch 0)
```

#### Batch = 2
```
在 GPU 上添加 bias 并应用 SiLU activation (batch=2)...
✓ bias + SiLU activation 完成
[batch=2, 2048x64x256, block_gemm_32x32x16_2x2x1_16x16x16], valid
✓ GPU GEMM 输出验证通过! (batch 0)
✓ GPU 输出验证通过! (batch 0)
✓ GPU GEMM 输出验证通过! (batch 1)
✓ GPU 输出验证通过! (batch 1)
```

**结论：** ✅ SiLU activation 实现正确，数值验证通过。

---

## 性能考量

### 计算复杂度

对于输出 `[batch, ho, ci]`：

| 操作 | 计算量 | 内存访问 |
|------|--------|---------|
| GEMM | `O(batch · ho · ci · k)` | 读 A, B，写 C |
| Bias + SiLU | `O(batch · ho · ci)` | 读 C, bias，写 C |

**SiLU 额外开销：**
- 1 次 `expf` 调用（约 10-20 FLOPs）
- 1 次除法
- 1 次乘法
- 总计：每元素约 25 FLOPs

### 性能影响

假设 `ho=2048, ci=64, k=256, batch=1`：

- **GEMM 计算量**：`2048 × 64 × 256 × 2 ≈ 67M FLOPs`
- **SiLU 计算量**：`2048 × 64 × 25 ≈ 3.3M FLOPs`
- **SiLU 占比**：`3.3M / 67M ≈ 5%`

**结论：** SiLU 的计算开销相对于 GEMM 很小（< 5%），且融合 kernel 进一步减少了内存访问开销。

### 优化建议

1. **融合到 GEMM**：
   - 将 SiLU 直接融合到 GEMM kernel 的写回阶段
   - 进一步减少内存访问

2. **向量化**：
   - 使用 `fp16x2` 或 `fp16x4` 向量类型
   - 一次处理多个元素

3. **快速近似**：
   - 对于要求不高的场景，可以使用 `expf` 的快速近似版本
   - AMD GPU 提供 `__expf`（低精度快速版本）

---

## 其他 Activation 支持

基于当前实现，可以轻松添加其他 activation 函数：

### ReLU
```cpp
float relu_x = fmaxf(0.0f, x);
output[idx] = (fp16_t)relu_x;
```

### GELU
```cpp
// GELU(x) = x · Φ(x), Φ(x) 是标准正态分布的 CDF
// 近似：GELU(x) ≈ 0.5 · x · (1 + tanh(sqrt(2/π) · (x + 0.044715·x³)))
float x3 = x * x * x;
float inner = 0.7978845608f * (x + 0.044715f * x3);  // sqrt(2/π) ≈ 0.7978845608
float gelu_x = 0.5f * x * (1.0f + tanhf(inner));
output[idx] = (fp16_t)gelu_x;
```

### Mish
```cpp
// Mish(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + e^x))
float softplus_x = logf(1.0f + expf(x));
float mish_x = x * tanhf(softplus_x);
output[idx] = (fp16_t)mish_x;
```

---

## 使用指南

### 编译

```bash
cd /workspace/causal-conv1d/rocm_backend/matrix_core_opus

/opt/rocm/bin/hipcc -x hip -std=c++17 \
    casual_conv1d_opus.cpp \
    -o casual_conv1d_opus.exe \
    --offload-arch=gfx942 \
    -I/workspace/aiter/csrc/include \
    -I/root/libtorch/include \
    -I/root/libtorch/include/torch/csrc/api/include \
    -L/root/libtorch/lib \
    -Wl,-rpath=/root/libtorch/lib \
    -ltorch -lc10 -ltorch_cpu
```

### 运行

```bash
./casual_conv1d_opus.exe
```

### 配置

在 `casual_conv1d_opus.cpp` 中可以配置以下参数：

#### 1. 启用/禁用 SiLU Activation

```cpp
#define ENABLE_SILU_ACTIVATION 1  // 1=启用 SiLU, 0=只添加 bias
```

| 值 | 行为 | 用途 |
|----|------|------|
| `1` | 启用 SiLU | Conv1D + Bias + SiLU（默认） |
| `0` | 禁用 SiLU | Conv1D + Bias（无 activation） |

**测试输出对比：**

```bash
# ENABLE_SILU_ACTIVATION = 1
在 GPU 上添加 bias 并应用 SiLU activation (batch=1)...
✓ bias + SiLU activation 完成

# ENABLE_SILU_ACTIVATION = 0
在 GPU 上添加 bias (batch=1)...
✓ bias 添加完成
```

**应用场景：**
- 性能对比测试：比较有无 activation 的性能差异
- 功能验证：验证 activation 对精度的影响
- 调试：简化计算流程，便于问题定位

#### 2. 批处理大小

```cpp
int batch = 1;  // 支持 batch >= 1
```

#### 3. Host 端验证

```cpp
#define ENABLE_HOST_VERIFICATION 1  // 1=启用验证, 0=纯GPU模式
```

---

## 总结

### 实现要点

✅ **融合 Kernel**：Bias + SiLU 融合，减少内存访问  
✅ **数值稳定**：使用 FP32 进行中间计算  
✅ **批处理支持**：支持 `batch > 1`  
✅ **验证通过**：Host 和 GPU 结果一致  

### 性能特点

- **计算开销**：相对于 GEMM < 5%
- **内存访问**：融合后仅增加 1 次读（bias）
- **可扩展性**：易于添加其他 activation 函数

### 未来优化

1. 融合到 GEMM kernel
2. 向量化处理
3. 使用快速数学库函数

---

**文档版本：** 1.0  
**最后更新：** 2025-11-15  
**作者：** AI Assistant  
**相关文件：** `casual_conv1d_opus.cpp`



