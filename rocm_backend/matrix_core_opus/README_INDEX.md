# 文档索引

本目录包含 Causal Conv1D GPU 加速实现的完整技术文档。

## 📚 核心文档

### 1. [IMPLEMENTATION_OVERVIEW.md](IMPLEMENTATION_OVERVIEW.md)
**整体实现概览** - 新用户必读！

- 项目背景和目标
- 完整的架构图和数据流
- 四大核心 GPU Kernel 详解
- 关键特性实现（批处理、bias、SiLU）
- 性能分析和优化建议
- 完整的使用指南

**适合人群：** 新用户、项目负责人、技术评审

---

### 2. [SILU_ACTIVATION_SUPPORT.md](SILU_ACTIVATION_SUPPORT.md)
**SiLU Activation 实现详解**

- SiLU 函数的数学原理和特性
- 融合 kernel 的设计和实现
- GPU 和 Host 端参考代码
- 性能分析（< 5% 额外开销）
- 其他 activation 函数扩展示例（ReLU、GELU、Mish）

**适合人群：** 想要理解或修改 activation 函数的开发者

---

### 2.1 [SILU_SWITCH_GUIDE.md](SILU_SWITCH_GUIDE.md)
**SiLU 开关使用指南** - 快速参考！

- `ENABLE_SILU_ACTIVATION` 开关的使用方法
- 启用/禁用 SiLU 的输出对比
- 4 个典型应用场景（性能测试、功能验证、消融实验、调试）
- 最佳实践和常见问题

**适合人群：** 需要快速启用/禁用 SiLU 的用户

---

### 2.2 [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) ⭐
**性能基准测试指南** - 测试性能必读！

- Python 和 Bash 两种基准测试脚本
- 100 次迭代统计分析（平均值、标准差、百分位数）
- 测试场景和最佳实践
- 自动生成文本报告和 CSV 数据
- 高级分析方法（时间序列、直方图）

**适合人群：** 需要测试和对比性能的用户

---

## 🎯 功能专题文档

### 3. GPU_PREPROCESS_README.md
**GPU 预处理实现**

- 预处理操作迁移到 GPU 的动机
- padding 和 img2col kernel 实现
- 编译时验证开关 (`ENABLE_HOST_VERIFICATION`)

### 4. BATCH_SUPPORT.md
**批处理支持**

- Batch > 1 的实现方式
- 内存布局调整
- 已知问题（程序退出错误）

### 5. BIAS_SUPPORT.md
**Bias 支持**

- Bias 的初始化和传输
- GPU kernel 实现
- 与 SiLU activation 的融合

---

## 🔧 问题排查

### 6. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) ⭐
**故障排查指南** - 遇到问题先看这里！

- Python 命令问题（`command not found`）
- 性能可视化问题（matplotlib）
- 编译和运行错误
- 开关配置问题
- 内存问题和调试技巧

**适合人群：** 遇到任何问题的用户

### 7. EXIT_ERROR_EXPLANATION.md
**退出错误说明**

- "double free" 和 "invalid next size" 错误分析
- 为什么这些错误不影响计算正确性
- libtorch 内存管理问题的解释

---

## 🚀 快速开始

### 编译和运行

```bash
# 进入项目目录
cd /workspace/causal-conv1d/rocm_backend/matrix_core_opus

# 编译
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

# 运行
./casual_conv1d_opus.exe
```

### 性能分析

```bash
# 使用 rocprofv3 收集性能数据
rocprofv3 --hip-api --stats -o casual_conv1d_perf ./casual_conv1d_opus.exe

# 可视化（需要 Python + matplotlib）
python3 visualize_performance.py \
    casual_conv1d_perf_hip_api_stats.csv \
    casual_conv1d_perf_kernel_stats.csv \
    ./
```

### 性能基准测试 ⭐

**推荐使用 Python 脚本：**

```bash
# 运行 100 次测试并生成统计报告
python benchmark_performance.py

# 快速测试（10 次）
python benchmark_performance.py -n 10

# 精确测试（1000 次）
python benchmark_performance.py -n 1000
```

**输出：**
- 平均时间、标准差、变异系数
- 最小/最大/中位数时间
- P25/P75/P95 百分位数
- 自动生成文本报告和 CSV 数据

详细说明请参考 [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)

### 配置选项

在 `casual_conv1d_opus.cpp` 中可以配置以下编译时开关：

```cpp
// 1. 启用/禁用 Host 端验证
#define ENABLE_HOST_VERIFICATION 1  // 1=启用验证，0=纯GPU模式

// 2. 启用/禁用 SiLU Activation
#define ENABLE_SILU_ACTIVATION 1    // 1=启用 SiLU，0=只添加 bias

// 3. 批处理大小
int batch = 1;  // 支持 batch >= 1
```

**配置组合示例：**

| 场景 | VERIFICATION | SILU | Batch | 说明 |
|------|--------------|------|-------|------|
| 开发验证 | 1 | 1 | 1 | 完整验证，启用 SiLU |
| 性能测试 | 0 | 1 | 1-2 | 纯GPU，启用 SiLU |
| 功能对比 | 1 | 0 | 1 | 验证无 activation 版本 |
| 调试模式 | 1 | 0 | 1 | 简化计算流程 |

---

## 📊 技术栈

- **编程语言：** C++17 + HIP
- **GPU 架构：** AMD gfx942
- **深度学习框架：** libtorch (PyTorch C++ API)
- **矩阵加速：** AMD Matrix Core (WMMA)
- **精度：** FP16 (half precision)

---

## ✅ 功能清单

| 功能 | 状态 | 文档 |
|------|------|------|
| GPU 预处理（padding + img2col） | ✅ 完成 | GPU_PREPROCESS_README.md |
| GPU 权重转换（depthwise → GEMM） | ✅ 完成 | IMPLEMENTATION_OVERVIEW.md |
| Matrix Core GEMM | ✅ 完成 | IMPLEMENTATION_OVERVIEW.md |
| Batch > 1 支持 | ✅ 完成 | BATCH_SUPPORT.md |
| Bias 加法 | ✅ 完成 | BIAS_SUPPORT.md |
| SiLU Activation | ✅ 完成 | SILU_ACTIVATION_SUPPORT.md |
| Kernel 融合（Bias + SiLU） | ✅ 完成 | SILU_ACTIVATION_SUPPORT.md |
| SiLU 开关（可启用/禁用） | ✅ 完成 | SILU_ACTIVATION_SUPPORT.md |
| 编译时验证开关 | ✅ 完成 | GPU_PREPROCESS_README.md |
| 性能基准测试工具 | ✅ 完成 | BENCHMARK_GUIDE.md |
| 性能可视化 | ✅ 完成 | visualize_performance.py |

---

## 🎓 学习路径

### 初学者
1. 阅读 `IMPLEMENTATION_OVERVIEW.md` 了解整体架构
2. 查看 `SILU_ACTIVATION_SUPPORT.md` 学习 kernel 融合
3. 运行程序，观察输出

### 进阶开发者
1. 研究四大核心 kernel 的实现细节
2. 尝试修改参数（batch size、输入尺寸）
3. 使用 rocprofv3 分析性能瓶颈
4. 实验不同的优化策略

### 高级研究者
1. 尝试将 batch 维度融入 kernel
2. 实现内存池化和异步传输
3. 探索将 SiLU 融合到 GEMM kernel
4. 添加其他 activation 函数（GELU、Mish）

---

## 📈 性能特征

### 瓶颈分析

| 组件 | 耗时占比 | 优化优先级 |
|------|---------|----------|
| hipMemcpy (H2D/D2H) | ~70% | 🔴 高 |
| hipMalloc / hipFree | ~20% | 🟡 中 |
| GPU Kernels | ~10% | 🟢 低 |

### 优化建议

1. **内存管理**：实现内存池化，避免频繁分配
2. **异步传输**：使用 hipStream 重叠计算和传输
3. **Batch 融合**：将循环移入 kernel，并行处理所有 batch
4. **Kernel 融合**：将 SiLU 融合到 GEMM 的写回阶段

---

## 🤝 贡献指南

### 添加新 Activation 函数

参考 `SILU_ACTIVATION_SUPPORT.md` 中的示例：

```cpp
// 1. 创建 kernel
__global__ void new_activation_kernel(...) {
    // 实现新的 activation 函数
}

// 2. 在 casual_conv1d_block_run 中调用
new_activation_kernel<<<grid, block>>>(...);

// 3. 更新 Host 端参考实现
// 4. 添加文档
```

### 提交文档

- 新功能必须附带文档
- 使用 Markdown 格式
- 包含代码示例和性能分析
- 添加到本索引文件

---

## 📝 版本历史

- **v1.3** (2025-11-15): 添加 SiLU activation 支持，实现 kernel 融合
- **v1.2**: 添加 Bias 支持
- **v1.1**: 实现 Batch > 1 支持
- **v1.0**: 初始版本，GPU 预处理 + Matrix Core GEMM

---

## 📧 联系方式

如有问题或建议，请参考：
- 项目路径：`/workspace/causal-conv1d/rocm_backend/matrix_core_opus/`
- 主要文件：`casual_conv1d_opus.cpp`

---

**最后更新：** 2025-11-15  
**文档版本：** 1.0  
**维护者：** AI Assistant



