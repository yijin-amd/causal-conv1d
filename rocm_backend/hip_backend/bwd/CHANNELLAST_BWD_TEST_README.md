# Channel-Last Causal Conv1D Backward Kernel 测试

## 🎯 简介

这是一个独立的测试文件，用于测试 `causal_conv1d_channellast_bwd_kernel` 的 **HIP 实现**，包含完整的精度和性能测试。

## ✨ 特点

### 1. **HIP 实现**
- ✅ 从 CUDA 移植到 HIP
- ✅ 支持 AMD GPU (MI300 系列)
- ✅ 简化版本（不包含 seq_idx 和 states 功能）
- ✅ 使用 2D 共享内存优化

### 2. **精度测试**
- ✅ 与 CPU 参考实现对比
- ✅ 测试 dx, dweight, dbias 的正确性
- ✅ 支持 SiLU 激活函数
- ✅ 自适应容差

### 3. **性能测试**
- ✅ 10 次预热 + 100 次迭代
- ✅ hipEvent 精确计时
- ✅ 带宽计算

## 🚀 快速开始

### 方式 1: 使用脚本（推荐）

```bash
cd /workspace/causal-conv1d/rocm_backend/hip_backend/bwd

# 给脚本执行权限
chmod +x run_channellast_test.sh

# 运行所有测试
./run_channellast_test.sh

# 只测精度
./run_channellast_test.sh --accuracy

# 只测性能
./run_channellast_test.sh --performance
```

### 方式 2: 手动编译运行

```bash
# 编译
hipcc -O2 -std=c++17 --offload-arch=gfx942 \
    test_channellast_bwd_kernel.cpp -o test_channellast_bwd_kernel

# 运行所有测试
./test_channellast_bwd_kernel

# 只运行精度测试
./test_channellast_bwd_kernel 1

# 只运行性能测试
./test_channellast_bwd_kernel 2
```

## 📊 测试配置

包含 6 种预设配置：

| 配置 | batch | dim | seqlen | width | bias | silu |
|------|-------|-----|--------|-------|------|------|
| Tiny | 1 | 32 | 64 | 4 | ✓ | ✗ |
| Small | 2 | 64 | 256 | 4 | ✓ | ✗ |
| Medium | 4 | 64 | 512 | 4 | ✓ | ✗ |
| Large | 4 | 64 | 1024 | 4 | ✓ | ✗ |
| No Bias | 2 | 64 | 256 | 4 | ✗ | ✗ |
| With SiLU | 2 | 64 | 256 | 4 | ✓ | ✓ |

## 📈 输出示例

### 精度测试

```
======================================================================
Accuracy Test: Small
  Config: batch=2, dim=64, seqlen=256, width=4
  Bias=Yes, SiLU=No
======================================================================

[Results]
  dx:      max_diff=1.234e-05, errors=0/32768
  dweight: max_diff=5.678e-04, errors=0/256
  dbias:   max_diff=2.345e-04, errors=0/64
  Status: ✓ PASSED
```

### 性能测试

```
======================================================================
  PERFORMANCE SUMMARY
======================================================================

Configuration        Mean(ms)    Min(ms)    BW(GB/s)
--------------------------------------------------------
Tiny                   0.0234     0.0228       12.45
Small                  0.1234     0.1201       45.23
Medium                 0.4567     0.4512       67.89
Large                  0.9234     0.9123       89.12
No Bias                0.1198     0.1167       46.78
With SiLU              0.1345     0.1312       44.56
======================================================================
```

## 🔧 Kernel 实现细节

### 内存布局
- **输入格式**: `[Batch, Length, Channel]` (Channel-Last)
- **权重格式**: `[Channel, Width]`

### 分块策略
```cpp
kChunkSizeL = 64   // sequence length 块大小
kChunkSizeC = 64   // channel 块大小
kNThreads = 128    // 每个 block 的线程数
```

### Grid/Block 组织
```cpp
grid(batch, n_chunks_L, n_chunks_C)  // 3D grid
block(128)                            // 1D block
```

### 共享内存使用
```cpp
__shared__ float dout_smem[kChunkSizeL + kWidth - 1][kChunkSizeC];
__shared__ float x_smem[kWidth - 1 + kChunkSizeL + kWidth - 1][kChunkSizeC];
```

## 🆚 与 CUDA 版本的区别

| 特性 | CUDA 版本 | 这个 HIP 版本 |
|------|-----------|--------------|
| **seq_idx 支持** | ✅ | ✗ (简化) |
| **states 支持** | ✅ | ✗ (简化) |
| **核心算法** | ✅ | ✅ 相同 |
| **共享内存** | ✅ | ✅ 相同策略 |
| **优化** | 高级 | 基础版 |

## ⚠️ 注意事项

### 1. **简化版本**
这个实现是**简化版**，不包含：
- ❌ `seq_idx` 支持（变长序列）
- ❌ `initial_states` / `dinitial_states`
- ❌ `dfinal_states`

如果需要完整功能，请参考原始 CUDA 实现。

### 2. **精度容差**
- **无 SiLU**: `1e-3` for dx, `1e-2` for dweight/dbias
- **有 SiLU**: `5e-2` (因为指数运算和累积误差)

### 3. **性能**
这是基础实现，性能优化空间：
- 🔧 调整 block 大小
- 🔧 优化共享内存访问
- 🔧 改进 reduction 算法
- 🔧 使用更高级的向量化

## 🐛 CPU 参考实现说明

CPU 实现用于验证正确性，逻辑如下：

### Backward 公式

对于 Causal Conv1D:
```
y[t] = Σ(w=0 to width-1) weight[w] * x[t - (width-1) + w] + bias
```

如果有 SiLU:
```
y[t] = silu(conv_out[t]) = conv_out[t] / (1 + exp(-conv_out[t]))
```

### 梯度计算

1. **dx (输入梯度)**:
```
dx[t] = Σ(w=0 to width-1) weight[width-1-w] * dout[t+w]
```
如果有 SiLU，需要先计算 `dout' = dout * silu_grad`

2. **dweight (权重梯度)**:
```
dweight[w] = Σ(over all valid t) x[t] * dout[t+w]
```

3. **dbias (偏置梯度)**:
```
dbias = Σ(over all t) dout[t]
```

## 📝 扩展功能

如果需要添加功能，可以修改：

### 添加 seq_idx 支持
```cpp
// 在 kernel 中添加 seq_idx 检查
if (seq_idx[t1] != seq_idx[t2]) {
    // 不同序列，跳过
    continue;
}
```

### 添加 states 支持
```cpp
// 添加 initial_states 和 dinitial_states 处理
if (chunk_l_id == 0 && t < width - 1) {
    // 使用 initial_states
}
```

## 📚 相关文件

```
rocm_backend/hip_backend/bwd/
├── test_channellast_bwd_kernel.cpp    # 本测试文件 ⭐
├── run_channellast_test.sh            # 运行脚本
├── CHANNELLAST_BWD_TEST_README.md     # 本文档
└── causal_conv1d_bwd_hip.cpp          # 完整实现（参考）
```

## 🎯 典型使用场景

### 场景 1: 验证 Kernel 正确性
```bash
./run_channellast_test.sh --accuracy
```

### 场景 2: 性能基线测试
```bash
./run_channellast_test.sh --performance > baseline.txt
```

### 场景 3: 开发调试
修改 kernel → 编译 → 运行 → 验证

## ✅ 测试状态

- ✅ 代码已完成
- ✅ 编译无错误
- ⏳ 需要在 AMD GPU 上运行验证

## 🎉 总结

你现在拥有：
1. ✅ **HIP Kernel 实现** - 从 CUDA 移植
2. ✅ **CPU 参考实现** - 用于精度验证
3. ✅ **完整测试框架** - 精度 + 性能
4. ✅ **自动化脚本** - 一键运行

**快速开始：**
```bash
cd /workspace/causal-conv1d/rocm_backend/hip_backend/bwd
./run_channellast_test.sh
```

🚀 开始测试你的 backward kernel 吧！

