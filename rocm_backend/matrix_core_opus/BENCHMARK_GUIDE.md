# 性能基准测试指南

本文档介绍如何使用性能基准测试脚本对 Causal Conv1D 进行准确的性能评估。

---

## 📋 概述

提供了两个基准测试脚本：

| 脚本 | 语言 | 特点 | 推荐场景 |
|------|------|------|---------|
| `benchmark_performance.py` | Python | 统计分析丰富、易于扩展 | **推荐** - 日常使用 |
| `benchmark_performance.sh` | Bash | 支持 rocprofv3 详细分析 | 深度性能分析 |

---

## 🚀 快速开始

### 使用 Python 脚本（推荐）

```bash
# 1. 确保程序已编译
cd /workspace/causal-conv1d/rocm_backend/matrix_core_opus
ls casual_conv1d_opus.exe  # 检查是否存在

# 2. 运行基准测试（默认 100 次）
python benchmark_performance.py

# 3. 查看结果
cat benchmark_results/benchmark_*.txt
```

**输出示例：**
```
╔════════════════════════════════════════════════════════════════╗
║          Causal Conv1D 性能基准测试 (Python)                  ║
╚════════════════════════════════════════════════════════════════╝

▶ 检测当前配置
================================================================
ENABLE_SILU_ACTIVATION:    0 (禁用)
ENABLE_HOST_VERIFICATION:  1 (启用)
Batch Size:                1

▶ 运行性能测试 (100 次迭代)
================================================================
进度: [██████████████████████████████████████████████████] 100/100 (100.0%)
✓ 完成 100 次迭代

▶ 性能统计
================================================================
样本数量:     100
成功运行:     100/100
成功率:       100.00%

平均时间:     509.44 ms
标准差:       8.28 ms
最小时间:     500.93 ms
最大时间:     527.42 ms
中位数:       506.97 ms
变异系数:     1.63%
P25:          503.46 ms
P75:          514.39 ms
P95:          527.42 ms
```

---

## 📊 Python 脚本详细说明

### 基本用法

```bash
python benchmark_performance.py [选项]
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-n, --iterations N` | 运行次数 | 100 |
| `-e, --executable PATH` | 可执行文件路径 | `./casual_conv1d_opus.exe` |
| `-h, --help` | 显示帮助信息 | - |

### 使用示例

#### 1. 快速测试（10 次）

```bash
python benchmark_performance.py -n 10
```

**用途：** 快速验证程序是否正常工作

#### 2. 标准测试（100 次）

```bash
python benchmark_performance.py
# 或
python benchmark_performance.py -n 100
```

**用途：** 日常性能评估，平衡准确性和时间

#### 3. 精确测试（1000 次）

```bash
python benchmark_performance.py -n 1000
```

**用途：** 获取高精度统计数据，用于发布前验证

#### 4. 指定可执行文件

```bash
python benchmark_performance.py -e ./my_custom_exe -n 50
```

**用途：** 测试不同版本或配置的程序

---

## 🔧 Bash 脚本详细说明

### 基本用法

```bash
./benchmark_performance.sh [选项] [迭代次数]
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `-n, --iterations N` | 运行 N 次迭代 |
| `-d, --detailed` | 使用 rocprofv3 进行详细分析 |
| `-h, --help` | 显示帮助信息 |

### 使用示例

#### 1. 快速模式（快）

```bash
./benchmark_performance.sh 100
# 或
./benchmark_performance.sh -n 100
```

**特点：** 只测量总运行时间，速度快

#### 2. 详细模式（慢）

```bash
./benchmark_performance.sh -d -n 20
```

**特点：** 使用 rocprofv3 收集 kernel 级别的性能数据，耗时较长

**注意：** 需要安装 ROCm 的 rocprofv3 工具

---

## 📈 输出文件说明

### 生成的文件

基准测试会在 `benchmark_results/` 目录下生成以下文件：

```
benchmark_results/
├── benchmark_20251118_032108.txt    # 文本报告
└── benchmark_20251118_032108.csv    # CSV 原始数据
```

### 文本报告格式

```
================================================================
  Causal Conv1D 性能基准测试报告
================================================================

测试时间: 2025-11-18 03:21:14
迭代次数: 100
可执行文件: ./casual_conv1d_opus.exe

性能统计:
  平均时间:     509.44 ms
  标准差:       8.28 ms
  最小时间:     500.93 ms
  最大时间:     527.42 ms
  中位数:       506.97 ms
  变异系数:     1.63%
  成功率:       100.00%

================================================================
```

### CSV 数据格式

```csv
Iteration,Time(ms),Valid
1,505.23,Yes
2,508.41,Yes
3,502.19,Yes
...
```

**字段说明：**
- `Iteration`: 迭代编号
- `Time(ms)`: 运行时间（毫秒）
- `Valid`: 是否验证通过（Yes/No）

---

## 📊 统计指标说明

| 指标 | 说明 |
|------|------|
| **样本数量** | 有效运行次数 |
| **成功率** | 验证通过的百分比 |
| **平均时间** | 所有运行的算术平均值 |
| **标准差** | 数据分散程度（越小越稳定） |
| **最小时间** | 最快的一次运行时间 |
| **最大时间** | 最慢的一次运行时间 |
| **中位数** | 排序后中间位置的值（不受极端值影响） |
| **变异系数 (CV)** | 标准差/平均值×100%（越小越稳定） |
| **P25/P75/P95** | 百分位数（25%、75%、95% 的数据小于此值） |

### 如何解读变异系数

| CV 值 | 稳定性 | 说明 |
|-------|--------|------|
| < 5% | 🟢 优秀 | 性能非常稳定 |
| 5-10% | 🟡 良好 | 性能较稳定 |
| 10-20% | 🟠 一般 | 有一定波动 |
| > 20% | 🔴 较差 | 性能不稳定，需要调查 |

**示例：**
- CV = 1.63% → 性能非常稳定 ✅
- CV = 15.2% → 性能波动较大，可能受系统负载影响 ⚠️

---

## 🎯 测试场景与配置

### 场景1：对比 SiLU 性能影响

```bash
# 步骤1: 禁用 SiLU
# 修改 casual_conv1d_opus.cpp:
#   #define ENABLE_SILU_ACTIVATION 0
hipcc ... && python benchmark_performance.py -n 100
mv benchmark_results/benchmark_*.txt results_no_silu.txt

# 步骤2: 启用 SiLU
# 修改 casual_conv1d_opus.cpp:
#   #define ENABLE_SILU_ACTIVATION 1
hipcc ... && python benchmark_performance.py -n 100
mv benchmark_results/benchmark_*.txt results_with_silu.txt

# 步骤3: 对比结果
diff results_no_silu.txt results_with_silu.txt
```

### 场景2：不同 Batch 大小的性能

```bash
# Batch = 1
# 修改 casual_conv1d_opus.cpp: int batch = 1;
hipcc ... && python benchmark_performance.py -n 100
mv benchmark_results/benchmark_*.csv batch1.csv

# Batch = 2
# 修改 casual_conv1d_opus.cpp: int batch = 2;
hipcc ... && python benchmark_performance.py -n 100
mv benchmark_results/benchmark_*.csv batch2.csv

# 对比吞吐量
# Batch=1: avg_time / 1 = X samples/sec
# Batch=2: avg_time / 2 = Y samples/sec
```

### 场景3：验证开销评估

```bash
# 启用验证
# #define ENABLE_HOST_VERIFICATION 1
hipcc ... && python benchmark_performance.py -n 100
# 记录平均时间: T1

# 禁用验证
# #define ENABLE_HOST_VERIFICATION 0
hipcc ... && python benchmark_performance.py -n 100
# 记录平均时间: T2

# 验证开销 = T1 - T2
```

---

## 🔬 高级分析

### 使用 Python 分析 CSV 数据

创建 `analyze_results.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('benchmark_results/benchmark_*.csv')

# 绘制时间序列
plt.figure(figsize=(12, 6))
plt.plot(df['Iteration'], df['Time(ms)'])
plt.xlabel('Iteration')
plt.ylabel('Time (ms)')
plt.title('Performance Over Iterations')
plt.grid(True)
plt.savefig('performance_timeline.png')

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(df['Time(ms)'], bins=30, edgecolor='black')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency')
plt.title('Performance Distribution')
plt.savefig('performance_histogram.png')

# 统计分析
print(df['Time(ms)'].describe())
```

运行：
```bash
python analyze_results.py
```

### 使用 rocprofv3 进行详细分析

```bash
# 使用 bash 脚本的详细模式
./benchmark_performance.sh -d -n 20

# 分析生成的 rocprof 数据
ls benchmark_results/prof_*/
```

---

## ⚡ 性能优化提示

### 1. 减少系统干扰

```bash
# 关闭不必要的服务
sudo systemctl stop unnecessary_services

# 固定 CPU 频率（需要 root 权限）
sudo cpupower frequency-set -g performance

# 禁用超线程（可选）
echo off | sudo tee /sys/devices/system/cpu/smt/control
```

### 2. 预热运行

```bash
# 先运行几次预热，再开始正式测试
./casual_conv1d_opus.exe  # 预热1
./casual_conv1d_opus.exe  # 预热2
./casual_conv1d_opus.exe  # 预热3

# 开始基准测试
python benchmark_performance.py -n 100
```

### 3. 批量对比测试

创建 `batch_benchmark.sh`:

```bash
#!/bin/bash
# 自动测试多种配置

configs=(
    "SILU=0,VERIFY=1"
    "SILU=1,VERIFY=1"
    "SILU=0,VERIFY=0"
    "SILU=1,VERIFY=0"
)

for config in "${configs[@]}"; do
    echo "Testing: $config"
    # 修改配置、编译、测试
    # ...
done
```

---

## 📝 常见问题

### Q1: 为什么每次运行时间不一样？

**A:** 正常现象，受以下因素影响：
- 系统负载
- CPU 频率调节
- 内存分配
- GPU 调度

**解决方案：** 运行多次取平均值（这就是基准测试的目的）

### Q2: 变异系数很大怎么办？

**A:** CV > 10% 说明性能不稳定，可能原因：
1. 系统负载过高 → 关闭其他程序
2. 测试次数太少 → 增加迭代次数
3. 热节流 → 改善散热
4. 其他进程干扰 → 使用专用测试环境

### Q3: 测试100次需要多久？

**A:** 取决于单次运行时间：
- 单次 ~500ms → 总共 ~50 秒
- 单次 ~1s → 总共 ~100 秒

使用 `-n 10` 可以快速测试

### Q4: 如何只测试 GPU kernel 时间？

**A:** 使用 bash 脚本的详细模式：
```bash
./benchmark_performance.sh -d -n 20
```

这会使用 rocprofv3 提取 kernel 级别的时间

### Q5: CSV 数据能导入 Excel 吗？

**A:** 可以！直接用 Excel 或 Google Sheets 打开 CSV 文件

---

## 📚 相关文档

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - 故障排查
- [SILU_SWITCH_GUIDE.md](SILU_SWITCH_GUIDE.md) - SiLU 开关使用
- [IMPLEMENTATION_OVERVIEW.md](IMPLEMENTATION_OVERVIEW.md) - 整体架构
- [README_INDEX.md](README_INDEX.md) - 文档索引

---

## 🎓 最佳实践

### 性能测试流程

1. **环境准备**
   ```bash
   # 关闭不必要的程序
   # 确保 GPU 空闲
   rocm-smi
   ```

2. **预热运行**
   ```bash
   ./casual_conv1d_opus.exe
   ```

3. **基准测试**
   ```bash
   python benchmark_performance.py -n 100
   ```

4. **记录结果**
   ```bash
   cp benchmark_results/benchmark_*.txt my_results/config_v1.txt
   ```

5. **重复测试**
   - 不同配置
   - 不同时间段
   - 多次运行验证

### 报告性能数据

建议报告以下指标：
- ✅ 平均时间 ± 标准差
- ✅ 中位数
- ✅ 变异系数
- ✅ 测试配置（SILU、VERIFY、Batch）
- ✅ 迭代次数

**示例：**
```
配置: SILU=ON, VERIFY=OFF, Batch=1
性能: 505.2 ± 8.3 ms (CV=1.6%, n=100)
中位数: 506.9 ms
```

---

**文档版本：** 1.0  
**最后更新：** 2025-11-15  
**作者：** AI Assistant

