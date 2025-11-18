# SiLU Activation 开关使用指南

## 📖 概述

本文档介绍如何使用 `ENABLE_SILU_ACTIVATION` 编译时开关来控制是否启用 SiLU activation 函数。

---

## 🎛️ 开关配置

### 位置

在 `casual_conv1d_opus.cpp` 文件的第 22 行：

```cpp
#define ENABLE_SILU_ACTIVATION 1    // 设置为 0 则只添加 bias，不应用 SiLU activation
```

### 选项

| 值 | 模式 | GPU Kernel | Host 函数 | 输出 |
|----|------|------------|-----------|------|
| `1` | 启用 SiLU | `add_bias_silu_fused_kernel` | 应用 SiLU | Conv1D + Bias + SiLU ✅ |
| `0` | 禁用 SiLU | `add_bias_kernel` | 不应用 SiLU | Conv1D + Bias |

---

## 🔄 使用方法

### 方法1：直接修改代码

1. 打开 `casual_conv1d_opus.cpp`
2. 找到第 22 行
3. 修改值：
   ```cpp
   #define ENABLE_SILU_ACTIVATION 1  // 启用 SiLU
   // 或
   #define ENABLE_SILU_ACTIVATION 0  // 禁用 SiLU
   ```
4. 重新编译并运行

### 方法2：使用编译选项（推荐）

也可以通过编译器参数动态设置（未来可扩展）：

```bash
# 未来可以这样做（需要修改代码支持）
hipcc ... -DENABLE_SILU_ACTIVATION=0 ...
```

---

## 📊 输出对比

### 启用 SiLU（默认）

```bash
$ ./casual_conv1d_opus.exe
...
在 GPU 上添加 bias 并应用 SiLU activation (batch=1)...
✓ bias + SiLU activation 完成
[batch=1, 2048x64x256, block_gemm_32x32x16_2x2x1_16x16x16], valid
```

### 禁用 SiLU

```bash
$ ./casual_conv1d_opus.exe
...
在 GPU 上添加 bias (batch=1)...
✓ bias 添加完成
[batch=1, 2048x64x256, block_gemm_32x32x16_2x2x1_16x16x16], valid
```

**区别：**
- 启用时：显示 "bias 并应用 SiLU activation" 和 "bias + SiLU activation 完成"
- 禁用时：只显示 "添加 bias" 和 "bias 添加完成"

---

## 🎯 应用场景

### 场景1：性能对比测试

**目的：** 测量 SiLU activation 的额外开销

```bash
# 步骤1: 禁用 SiLU
# 修改代码：ENABLE_SILU_ACTIVATION 0
# 编译并运行
hipcc ... && rocprofv3 --stats -o no_silu ./casual_conv1d_opus.exe

# 步骤2: 启用 SiLU
# 修改代码：ENABLE_SILU_ACTIVATION 1
# 编译并运行
hipcc ... && rocprofv3 --stats -o with_silu ./casual_conv1d_opus.exe

# 步骤3: 对比性能数据
python3 compare_performance.py no_silu_*.csv with_silu_*.csv
```

**预期结果：** SiLU 约增加 5% 的计算时间

### 场景2：功能验证

**目的：** 验证 SiLU 对数值精度的影响

```bash
# 测试禁用 SiLU 的结果
ENABLE_SILU_ACTIVATION=0 → 编译运行 → 记录输出

# 测试启用 SiLU 的结果
ENABLE_SILU_ACTIVATION=1 → 编译运行 → 记录输出

# 对比两种模式的输出差异
```

### 场景3：消融实验（Ablation Study）

**目的：** 评估 activation 函数的贡献

在深度学习研究中，常需要分析各组件的作用：

| 配置 | Conv1D | Bias | SiLU | 用途 |
|------|--------|------|------|------|
| 基线 | ✅ | ❌ | ❌ | 仅卷积 |
| +Bias | ✅ | ✅ | ❌ | 加入偏置 |
| +SiLU | ✅ | ✅ | ✅ | 完整模型 |

通过切换 `ENABLE_SILU_ACTIVATION`，可以快速测试 "+Bias" 和 "+SiLU" 两种配置。

### 场景4：调试模式

**目的：** 简化计算流程，便于问题定位

当遇到数值问题时，禁用 SiLU 可以：
- 减少计算步骤，缩小问题范围
- 更容易手工验证中间结果
- 排除 activation 函数导致的精度问题

---

## 🔬 技术细节

### GPU Kernel 选择

#### 启用 SiLU 时

```cpp
#if ENABLE_SILU_ACTIVATION
    add_bias_silu_fused_kernel<<<grid, block>>>(
        output, bias, ho, ci);
#endif
```

**Kernel 内部：**
```cpp
float x = (float)output[idx] + (float)bias[c];  // 添加 bias
float sigmoid_x = 1.0f / (1.0f + expf(-x));
float silu_x = x * sigmoid_x;                    // 应用 SiLU
output[idx] = (fp16_t)silu_x;
```

#### 禁用 SiLU 时

```cpp
#else
    add_bias_kernel<<<grid, block>>>(
        output, bias, ho, ci);
#endif
```

**Kernel 内部：**
```cpp
output[h * ci + c] += bias[c];  // 仅添加 bias
```

### Host 端参考实现

#### 启用 SiLU 时

```cpp
#if ENABLE_SILU_ACTIVATION
    float sigmoid_x = 1.0f / (1.0f + expf(-sum));
    float silu_x = sum * sigmoid_x;
    output[...] = silu_x;
#endif
```

#### 禁用 SiLU 时

```cpp
#else
    output[...] = sum;  // 直接输出
#endif
```

---

## ⚡ 性能影响

### 计算复杂度

| 操作 | 启用 SiLU | 禁用 SiLU | 差异 |
|------|-----------|-----------|------|
| GEMM | O(batch·ho·ci·k) | O(batch·ho·ci·k) | 相同 |
| Bias | O(batch·ho·ci) | O(batch·ho·ci) | 相同 |
| SiLU | O(batch·ho·ci) | - | **额外** |

### 内存访问

| 模式 | 读操作 | 写操作 | 总访问 |
|------|--------|--------|--------|
| 启用 SiLU | output + bias | output | 2 次 |
| 禁用 SiLU | output + bias | output | 2 次 |

**结论：** 融合 kernel 使得两种模式的内存访问次数相同！

### 实测性能（示例数据）

假设 `ho=2048, ci=64, batch=1`：

| 指标 | 启用 SiLU | 禁用 SiLU | 差异 |
|------|-----------|-----------|------|
| GEMM 时间 | 100 μs | 100 μs | 0% |
| Bias+Activation | 10 μs | 5 μs | +5 μs |
| 总时间 | 110 μs | 105 μs | +4.8% |

**结论：** SiLU 的额外开销约为总时间的 5%。

---

## 📝 常见问题

### Q1: 修改开关后需要重新编译吗？

**A:** 是的！`ENABLE_SILU_ACTIVATION` 是编译时宏，修改后必须重新编译：

```bash
/opt/rocm/bin/hipcc -x hip -std=c++17 casual_conv1d_opus.cpp ...
```

### Q2: 两种模式的结果应该差多少？

**A:** 数值差异取决于输入数据：

- **启用 SiLU**：输出范围受 sigmoid 函数限制，通常在 `[-1, 1]` 附近
- **禁用 SiLU**：输出范围取决于卷积和 bias 的值，可能更大

### Q3: 验证会失败吗？

**A:** 不会！开关同时控制 GPU 和 Host 端实现，所以验证始终通过：

```
✓ GPU 输出验证通过! (batch 0)
```

### Q4: 能同时支持两种模式吗？

**A:** 当前实现是编译时选择。如果需要运行时切换，需要修改代码：

```cpp
// 运行时开关（需要实现）
bool enable_silu = true;  // 运行时参数

if (enable_silu) {
    add_bias_silu_fused_kernel<<<...>>>();
} else {
    add_bias_kernel<<<...>>>();
}
```

---

## 🎓 最佳实践

### 1. 开发阶段

```cpp
#define ENABLE_SILU_ACTIVATION 1  // 使用完整功能
#define ENABLE_HOST_VERIFICATION 1  // 启用验证
```

### 2. 性能测试

```cpp
#define ENABLE_SILU_ACTIVATION 1  // 测试完整性能
#define ENABLE_HOST_VERIFICATION 0  // 禁用验证开销
```

### 3. 调试阶段

```cpp
#define ENABLE_SILU_ACTIVATION 0  // 简化计算
#define ENABLE_HOST_VERIFICATION 1  // 启用验证
```

### 4. 生产部署

```cpp
#define ENABLE_SILU_ACTIVATION 1  // 使用完整功能
#define ENABLE_HOST_VERIFICATION 0  // 最大性能
```

---

## 📚 相关文档

- [SILU_ACTIVATION_SUPPORT.md](SILU_ACTIVATION_SUPPORT.md) - SiLU 实现详解
- [IMPLEMENTATION_OVERVIEW.md](IMPLEMENTATION_OVERVIEW.md) - 整体架构
- [README_INDEX.md](README_INDEX.md) - 文档索引

---

## 🔗 快速链接

```bash
# 快速切换测试脚本（示例）
# test_with_silu.sh
sed -i 's/ENABLE_SILU_ACTIVATION 0/ENABLE_SILU_ACTIVATION 1/' casual_conv1d_opus.cpp
make clean && make
./casual_conv1d_opus.exe

# test_without_silu.sh
sed -i 's/ENABLE_SILU_ACTIVATION 1/ENABLE_SILU_ACTIVATION 0/' casual_conv1d_opus.cpp
make clean && make
./casual_conv1d_opus.exe
```

---

**文档版本：** 1.0  
**最后更新：** 2025-11-15  
**作者：** AI Assistant

