# 快速参考：Causal Conv1D 预处理

## 当前实现：Host 端转置方式 ✅

### 数据流程图

```
输入 [ci=64, hi=2048]
    ↓
┌─────────────────────────────────────┐
│ Host 端处理                          │
├─────────────────────────────────────┤
│ 1. FP32 → FP16 转换                  │
│    fp16_in [64, 2048]                │
│                                     │
│ 2. 转置 (transpose_fp16)            │
│    ✓ 显式转置操作                    │
│    fp16_in_transposed [2048, 64]    │
└─────────────────────────────────────┘
    ↓ [H2D 传输 256KB]
┌─────────────────────────────────────┐
│ GPU 端处理                           │
├─────────────────────────────────────┤
│ 3. Padding + img2col Kernel         │
│    输入: [2048, 64]                  │
│    输出: [2048, 256]                 │
│                                     │
│ 4. 权重转换 Kernel                   │
│    输入: [64, 4]                     │
│    输出: [64, 256]                   │
│                                     │
│ 5. GEMM                             │
│    [2048, 256] × [64, 256]^T        │
└─────────────────────────────────────┘
    ↓
输出 [2048, 64]
```

## 关键代码位置

### 文件结构

```
/workspace/causal-conv1d/
├── rocm_backend/matrix_core_opus/
│   └── casual_conv1d_opus.cpp         ← 主要实现
├── HOST_TRANSPOSE_IMPLEMENTATION.md   ← 详细文档
├── IMPLEMENTATION_COMPARISON.md       ← 对比分析
├── QUICK_REFERENCE.md                ← 本文件
└── test_gpu_preprocess.sh            ← 测试脚本
```

### 关键函数

#### 1. Host 端转置（Line 522-523）

```cpp
// 在 casual_conv1d_block_run() 中
transpose_fp16(fp16_in, fp16_in_transposed, ci, hi);
// [64, 2048] → [2048, 64]
```

#### 2. GPU Kernel（Line 413-439）

```cpp
__global__ void preprocess_input_kernel(
    const fp16_t* __restrict__ input,  // [hi, ci] - 已转置
    fp16_t* __restrict__ output,       // [ho, hk*ci]
    int ci, int hi, int ho, int hk, int pad)
{
    // ...
    val = input[h_idx * ci + c];  // ← 合并访问
    // ...
}
```

## 运行步骤

### 1. 编译

```bash
cd /workspace/causal-conv1d/rocm_backend/matrix_core_opus

# 验证模式（默认，会对比 Host 和 GPU 结果）
hipcc casual_conv1d_opus.cpp -o casual_conv1d_opus -std=c++17 -O3

# 纯 GPU 模式（跳过验证）
hipcc -DENABLE_HOST_VERIFICATION=0 casual_conv1d_opus.cpp -o casual_conv1d_opus -std=c++17 -O3
```

### 2. 运行

```bash
./casual_conv1d_opus
```

### 3. 预期输出

```
在 Host 上执行转置操作...
✓ 转置完成：[64, 2048] -> [2048, 64]
执行完整 Host 端预处理用于验证...
传输转置后的数据到 GPU...
在 GPU 上执行 padding + img2col...
在 GPU 上执行权重转换...
✓ GPU 预处理完成
✓ 输入预处理验证（GPU vs Host）: 通过 (错误数=0/524288)
✓ 权重预处理验证（GPU vs Host）: 通过 (错误数=0/16384)
m:2048,n:64,k:256,lda:256,ldb:256,ldc:64
[2048x64x256, block_gemm_32x32x16_2x2x1_16x16x16], valid
```

## 配置选项

### 编译时宏定义

```cpp
// 在 casual_conv1d_opus.cpp 第 21 行
#define ENABLE_HOST_VERIFICATION 1  // 1=验证模式, 0=纯GPU模式
```

| 模式 | ENABLE_HOST_VERIFICATION | 说明 |
|------|-------------------------|------|
| 验证模式 | 1 | 保留 Host 端完整预处理用于验证 |
| 纯 GPU 模式 | 0 | 跳过 Host 端验证，最佳性能 |

## 性能特点

### 优点 ✅

1. **GPU 合并访问**
   - 线程束内相邻线程访问连续内存
   - 高效利用内存带宽

2. **清晰的数据流**
   - Host 转置 → GPU 处理
   - 职责分明，易于维护

3. **灵活优化**
   - Host 端可并行化转置
   - GPU 端可专注于核心计算

### 权衡 ⚠️

1. **Host 端开销**
   - 需要 0.1-1ms 转置时间
   - 可通过 OpenMP 并行化缓解

2. **额外内存**
   - 需要 `hi * ci * 2 bytes` Host 缓冲区
   - 对于大规模应用需要考虑

## 关键参数

### 默认配置

```cpp
int batch = 1;      // 批次大小
int hi = 2048;      // 输入序列长度
int ci = 64;        // 输入通道数
int hk = 4;         // 卷积核大小
int pad = 3;        // Padding 大小 (hk - 1)
```

### 计算参数

```cpp
int ho = hi;                // 输出长度 = 输入长度
int m = ho;                 // GEMM M 维度
int n = ci;                 // GEMM N 维度
int k = hk * ci;            // GEMM K 维度
```

### GPU 配置

```cpp
// Kernel launch 配置
dim3 block_dim(16, 16);                    // 线程块：16×16
dim3 grid_dim((ho+15)/16, (hk*ci+15)/16);  // 网格大小
```

## 内存占用

### Host 端

| Buffer | 大小 | 用途 |
|--------|------|------|
| `fp16_in` | 64×2048×2B = 256KB | 原始输入 |
| `fp16_in_transposed` | 64×2048×2B = 256KB | 转置后输入 |
| `fp16_w` | 64×4×2B = 512B | 权重 |

**总计**：约 512KB + 验证缓冲区

### GPU 端

| Buffer | 大小 | 用途 |
|--------|------|------|
| `dev_in_transposed` | 256KB | 转置后输入 |
| `dev_a` | 2048×256×2B = 1MB | GEMM A 矩阵 |
| `dev_b` | 64×256×2B = 32KB | GEMM B 矩阵 |
| `dev_c` | 2048×64×2B = 256KB | GEMM C 矩阵 |

**总计**：约 1.5MB

## 常见问题

### Q1: 为什么转置在 Host 上？

**A**: 为了让 GPU kernel 实现**合并内存访问**（coalesced access），提高 GPU 性能。转置后的数据布局 `[hi, ci]` 使得相邻线程可以访问连续内存地址。

### Q2: Host 端转置会成为瓶颈吗？

**A**: 对于小批量可能会，但可以通过以下方式优化：
- 使用 OpenMP 并行化
- 使用 SIMD 指令
- 对于大批量，转置开销被摊销

### Q3: 能否在 GPU 上做转置？

**A**: 可以，有两种方式：
1. **显式 GPU 转置 kernel**：先转置，再 padding/img2col
2. **隐式转置**（之前的实现）：在 padding/img2col kernel 中通过索引计算完成

当前选择 Host 端转置是为了最大化 GPU kernel 的内存访问效率。

### Q4: 如何切换到 GPU 隐式转置？

**A**: 需要修改：
1. GPU Kernel 读取方式：`input[c * hi + h_idx]` 而非 `input[h_idx * ci + c]`
2. 移除 Host 端 `transpose_fp16()` 调用
3. 直接传输原始数据到 GPU

详见 `IMPLEMENTATION_COMPARISON.md`

## 调试技巧

### 1. 验证转置正确性

```cpp
// 在转置后添加检查
for(int i = 0; i < 10; i++) {
    int c = i % ci;
    int h = i / ci;
    printf("原始[%d,%d]=%.4f, 转置[%d,%d]=%.4f\n",
           c, h, (float)fp16_in[c*hi + h],
           h, c, (float)fp16_in_transposed[h*ci + c]);
}
```

### 2. 检查 GPU kernel 输出

```cpp
// 从 GPU 拷贝回 host 检查
float16 *debug_a = (float16*)malloc(lda*m*sizeof(float16));
HIP_CALL(hipMemcpy(debug_a, dev_a, lda*m*sizeof(float16), hipMemcpyDeviceToHost));
for(int i = 0; i < 10; i++) {
    printf("dev_a[%d] = %.4f\n", i, (float)debug_a[i]);
}
free(debug_a);
```

### 3. 使用 HIP Profiler

```bash
rocprof --stats ./casual_conv1d_opus
```

## 相关文档

1. **HOST_TRANSPOSE_IMPLEMENTATION.md** - 详细实现说明
2. **IMPLEMENTATION_COMPARISON.md** - 与隐式转置方案对比
3. **GPU_PREPROCESS_README.md** - 原始 GPU 预处理文档

## 更新日志

- **2025-11-14**: 初始版本，实现 Host 端转置方式
  - 转置在 Host 上显式完成
  - GPU kernel 处理 padding + img2col
  - 支持验证模式和纯 GPU 模式

