# Causal Conv1D GPU 预处理改造说明

## 概述
本文档说明了如何将 `casual_conv1d_block_run` 函数中的 host 端预处理改写为 GPU kernel，从而提高性能。

## 主要改动

### 1. 新增 GPU Kernel

#### 1.1 输入预处理 Kernel
```cpp
__global__ void preprocess_input_kernel(
    const fp16_t* __restrict__ input,  // 输入 [ci, hi]
    fp16_t* __restrict__ output,       // 输出 [ho, hk*ci]
    int ci, int hi, int ho, int hk, int pad)
```

**功能**：将输入从 `[ci, hi]` 转换为 GEMM 的 A 矩阵 `[ho, hk*ci]`

**操作步骤**：
1. **转置**：从 `[ci, hi]` 隐式转置到 `[hi, ci]`
2. **Padding**：在输入序列前面添加 `pad` 个零（causal padding）
3. **img2col**：将 padded 输入展开成卷积窗口

**实现细节**：
- 使用 2D 线程网格，`(ho, hk*ci)` 维度
- 每个线程负责输出矩阵的一个元素
- 自动处理 padding 边界条件

#### 1.2 权重预处理 Kernel
```cpp
__global__ void preprocess_weight_kernel(
    const fp16_t* __restrict__ weight_dw,  // depthwise 权重 [ci, hk]
    fp16_t* __restrict__ weight_conv,      // 转换后权重 [ci, hk*ci]
    int ci, int hk)
```

**功能**：将 depthwise 卷积权重 `[ci, hk]` 转换为普通卷积权重 `[ci, hk*ci]`

**实现细节**：
- 使用 2D 线程网格，`(ci, hk*ci)` 维度
- Depthwise 卷积特性：只有当输入输出 channel 相同时才有非零权重
- 输出矩阵是稀疏的，每行只有 `hk` 个非零元素

### 2. 函数改造

#### 原始流程（Host 端预处理）
```
1. 转换输入到 fp16
2. 在 CPU 上进行转置：[ci, hi] -> [hi, ci]  ❌ 串行，慢
3. 在 CPU 上添加 padding                     ❌ 串行，慢
4. 在 CPU 上执行 img2col                     ❌ 串行，慢
5. 在 CPU 上转换权重格式                     ❌ 串行，慢
6. 将预处理结果拷贝到 GPU                    ❌ 大量数据传输
7. 执行 GEMM
```

#### 改造后流程（GPU 端预处理）
```
1. 转换输入到 fp16
2. 将原始输入和权重拷贝到 GPU               ✅ 最小数据传输
3. 在 GPU 上调用预处理 kernel                ✅ 并行，快
   - 转置（隐式，无需额外操作）              ✅ 通过索引计算完成
   - Padding（动态判断边界）                  ✅ 并行处理
   - img2col                                  ✅ 并行展开
   - 权重转换                                 ✅ 并行转换
4. 执行 GEMM
```

**重要说明**：
- ✅ **转置在 GPU 上隐式完成**：`preprocess_input_kernel` 直接从 `[ci, hi]` 格式读取输入（通过 `input[c * hi + h_idx]`），无需在 host 上预先转置
- ✅ **Host 端转置已移除**（在纯 GPU 模式下）：设置 `ENABLE_HOST_VERIFICATION=0` 即可完全跳过 host 端的转置、padding、img2col 操作

### 3. 性能优势

1. **减少 Host-Device 传输**：
   - 原始：需要传输预处理后的大矩阵（`lda*m` 和 `ldb*n`）
   - 改造后：只需传输原始输入和权重（`hi*ci` 和 `ci*hk`）
   - 数据量减少约 `hk` 倍

2. **并行化预处理**：
   - 原始：CPU 串行处理
   - 改造后：GPU 并行处理，利用数千个线程

3. **减少 Host 端开销**：
   - 原始：需要分配多个临时 host 缓冲区
   - 改造后：所有预处理在 GPU 上完成

### 4. 验证机制

代码中包含了验证逻辑，对比 GPU kernel 和原始 host 代码的结果：

```cpp
// 验证输入预处理结果
bool input_match = true;
int input_err_count = 0;
for(int i=0; i<lda*m && input_err_count < 10; i++) {
    float diff = fabs((float)fp16_a_gpu[i] - (float)fp16_a_host[i]);
    if(diff > 1e-5) {
        input_match = false;
        input_err_count++;
    }
}
printf("输入预处理验证: %s (错误数=%d/%d)\n", 
       input_match?"通过":"失败", input_err_count, lda*m);
```

### 5. 使用说明

#### 编译选项

代码提供了两种模式，通过宏 `ENABLE_HOST_VERIFICATION` 控制：

**模式 1：验证模式（默认）**
```cpp
#define ENABLE_HOST_VERIFICATION 1  // 在文件顶部
```
- ✅ 保留 host 端预处理，用于对比验证 GPU kernel 正确性
- ✅ 会打印 GPU 和 Host 结果的对比
- ❌ 会执行 host 端的转置、padding、img2col（仅用于验证）
- **适用场景**：开发、调试、验证阶段

**模式 2：纯 GPU 模式（生产环境）**
```cpp
#define ENABLE_HOST_VERIFICATION 0  // 在文件顶部
```
- ✅ 完全跳过 host 端预处理
- ✅ 所有预处理操作仅在 GPU 上执行
- ✅ 最小化 host-device 数据传输
- ✅ 最佳性能
- **适用场景**：生产环境、性能测试

#### 编译命令

```bash
cd /workspace/causal-conv1d/rocm_backend/matrix_core_opus

# 验证模式（默认）
hipcc -I/path/to/opus/include casual_conv1d_opus.cpp -o casual_conv1d_opus -std=c++17 -O3

# 或者通过命令行指定纯 GPU 模式
hipcc -I/path/to/opus/include -DENABLE_HOST_VERIFICATION=0 casual_conv1d_opus.cpp -o casual_conv1d_opus -std=c++17 -O3
```

#### 运行
```bash
./casual_conv1d_opus
```

#### 预期输出

**验证模式（ENABLE_HOST_VERIFICATION=1）**
```
执行 Host 端预处理用于验证...
GPU 预处理完成
✓ 输入预处理验证（GPU vs Host）: 通过 (错误数=0/524288)
✓ 权重预处理验证（GPU vs Host）: 通过 (错误数=0/16384)
m:2048,n:64,k:256,lda:256,ldb:256,ldc:64
[2048x64x256, block_gemm_32x32x16_2x2x1_16x16x16], valid
```

**纯 GPU 模式（ENABLE_HOST_VERIFICATION=0）**
```
跳过 Host 端预处理，使用纯 GPU 路径...
GPU 预处理完成
m:2048,n:64,k:256,lda:256,ldb:256,ldc:64
[2048x64x256, block_gemm_32x32x16_2x2x1_16x16x16], valid
```

### 6. 参数说明

测试配置：
- `batch = 1`：批次大小
- `hi = 2048`：输入序列长度
- `ci = 64`：输入通道数
- `hk = 4`：卷积核大小
- `pad = 3`：padding 大小（hk - 1）

Kernel 配置：
- **输入预处理 kernel**：`block_dim(16, 16)`，`grid_dim((ho+15)/16, (hk*ci+15)/16)`
- **权重预处理 kernel**：`block_dim(16, 16)`，`grid_dim((ci+15)/16, (hk*ci+15)/16)`

### 7. 代码结构

```
casual_conv1d_opus.cpp
├── preprocess_input_kernel()      # GPU 输入预处理 kernel
├── preprocess_weight_kernel()     # GPU 权重预处理 kernel
└── casual_conv1d_block_run()      # 主测试函数
    ├── 1. 数据初始化
    ├── 2. Host 端预处理（用于验证）
    ├── 3. GPU 端预处理（新增）
    ├── 4. 验证 GPU 预处理结果（新增）
    ├── 5. 执行 GEMM
    └── 6. 验证最终结果
```

### 8. 转置实现细节（重要）

#### GPU Kernel 中的隐式转置

在 `preprocess_input_kernel` 中，转置操作通过**索引计算隐式完成**，无需额外的转置 kernel：

```cpp
// 输入数据布局：[ci, hi] - channel 主序
// 对于位置 (c, h)，内存地址 = c * hi + h

// 在 kernel 中：
int c = k_c_idx % ci;  // 目标 channel
int h_idx = in_pos - pad;  // 目标时间位置

// 直接从 [ci, hi] 读取，效果等同于从转置后的 [hi, ci] 读取
val = input[c * hi + h_idx];
```

**关键点**：
1. ❌ **不需要**在 host 上预先执行 `transpose_fp16(fp16_in, fp16_in_nhc, ci, hi)`
2. ✅ GPU kernel 直接从原始 `[ci, hi]` 格式读取
3. ✅ 通过调整索引顺序，自然完成转置
4. ✅ 节省了一次完整的转置操作和内存拷贝

#### Host 端转置的作用

在 `ENABLE_HOST_VERIFICATION=1` 模式下：
- Host 端的 `transpose_fp16` 仅用于**验证 GPU kernel 的正确性**
- 通过对比 GPU 和 Host 的预处理结果，确保 kernel 实现无误
- 在生产环境中（`ENABLE_HOST_VERIFICATION=0`），这部分代码完全不会执行

### 9. 注意事项

1. **内存布局**：
   - 输入：`[ci, hi]` - 按 channel 主序
   - 输出 A：`[ho, hk*ci]` - 按输出位置主序
   - 权重：`[ci, hk]` - 按 channel 主序
   - 输出 B：`[ci, hk*ci]` - 按 channel 主序

2. **Causal 特性**：
   - Padding 在序列前面，确保每个位置只能看到之前的信息
   - img2col 时需要检查 `in_pos >= pad && in_pos < hi + pad`

3. **FP16 精度**：
   - 所有 GPU 计算使用 FP16
   - 验证阈值设置为 `1e-5`

4. **性能考虑**：
   - 转置操作的内存访问模式可能导致非合并访问
   - 但由于与 padding、img2col 融合在一起，总体性能仍优于分离的操作
   - 可以考虑使用 shared memory 优化内存访问模式（未来改进）

## 总结

### 核心改进

通过将预处理操作从 CPU 移到 GPU，实现了：
- ✅ **消除 Host 端转置**：转置通过 GPU kernel 中的索引计算隐式完成
- ✅ **减少数据传输**：只传输原始数据，不传输预处理后的大矩阵
- ✅ **并行化所有预处理**：转置、padding、img2col、权重转换全部并行
- ✅ **简化内存管理**：减少临时 buffer 分配
- ✅ **提高整体性能**：CPU 时间减少，GPU 利用率提高

### 两种模式

1. **验证模式**（`ENABLE_HOST_VERIFICATION=1`）
   - 保留 host 端预处理用于对比验证
   - 适合开发和调试阶段

2. **纯 GPU 模式**（`ENABLE_HOST_VERIFICATION=0`）
   - 完全跳过 host 端预处理
   - 最佳性能，适合生产环境

### 关键技术点

- **隐式转置**：通过 `input[c * hi + h_idx]` 直接从 `[ci, hi]` 读取，无需预先转置
- **融合操作**：转置、padding、img2col 在一个 kernel 中完成
- **最小数据传输**：只传输 `hi*ci + ci*hk` 而非 `lda*m + ldb*n`

