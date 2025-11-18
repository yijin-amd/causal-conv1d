# 故障排查指南

本文档记录常见问题及其解决方案。

---

## 🐍 Python 命令问题

### 问题描述

```bash
$ python visualize_performance.py ...
bash: python: command not found
```

### 原因

现代 Linux 系统通常使用 `python3` 命令而不是 `python`，以区分 Python 2.x 和 Python 3.x。

### 解决方案

#### 方案1：创建符号链接（推荐）✅

```bash
sudo ln -sf /usr/bin/python3 /usr/bin/python
```

**优点：**
- 一次设置，永久生效
- 所有脚本和文档中的 `python` 命令都能工作
- 兼容性好

**验证：**
```bash
$ python --version
Python 3.10.12
```

#### 方案2：使用 python3 命令

直接使用 `python3` 替代 `python`：

```bash
# 原命令
python visualize_performance.py ...

# 修改为
python3 visualize_performance.py ...
```

#### 方案3：使用别名（临时）

在当前终端会话中创建别名：

```bash
alias python=python3
```

**注意：** 此方法仅对当前终端有效，关闭后失效。

要永久生效，可添加到 `~/.bashrc`：

```bash
echo "alias python=python3" >> ~/.bashrc
source ~/.bashrc
```

---

## 📊 性能可视化问题

### 问题：No module named 'matplotlib'

```bash
$ python visualize_performance.py ...
ModuleNotFoundError: No module named 'matplotlib'
```

**解决方案：**

```bash
# 安装 matplotlib
pip3 install matplotlib pandas numpy

# 或使用系统包管理器
sudo apt-get install python3-matplotlib python3-pandas python3-numpy
```

### 问题：图表不显示

如果是通过 SSH 连接的无图形界面服务器，图表无法直接显示。

**解决方案：**

脚本会自动保存图表到文件：

```bash
$ python visualize_performance.py hip_api_stats.csv kernel_stats.csv ./
# 生成文件：./performance_visualization.png
```

然后通过 SCP 或其他方式下载图片：

```bash
scp user@server:path/to/performance_visualization.png ./
```

---

## 🔧 编译问题

### 问题：hipcc: command not found

**解决方案：**

```bash
# 检查 ROCm 是否安装
ls /opt/rocm/bin/hipcc

# 添加到 PATH
export PATH=/opt/rocm/bin:$PATH

# 永久添加（添加到 ~/.bashrc）
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 问题：找不到头文件

```bash
fatal error: opus/opus.hpp: No such file or directory
```

**解决方案：**

确保包含路径正确：

```bash
hipcc ... -I/workspace/aiter/csrc/include ...
```

检查文件是否存在：

```bash
ls /workspace/aiter/csrc/include/opus/opus.hpp
```

---

## 🏃 运行问题

### 问题：Segmentation fault (batch > 2)

**描述：** 当 `batch >= 3` 时，程序在退出时崩溃。

**原因：** libtorch 内存管理问题（已知问题）。

**解决方案：**

计算结果是正确的，只是程序退出时有问题。可以：

1. 使用 `batch=1` 或 `batch=2`（推荐）
2. 忽略退出错误（结果已正确计算）
3. 参考 `EXIT_ERROR_EXPLANATION.md` 了解详情

**验证结果正确性：**

```bash
$ ./casual_conv1d_opus.exe 2>&1 | grep -E "(valid|✓)"
✓ GPU 输出验证通过! (batch 0)
[batch=1, 2048x64x256, block_gemm_32x32x16_2x2x1_16x16x16], valid
```

只要看到 `valid` 和 `✓`，说明计算是正确的。

---

## 🔀 开关配置问题

### 问题：修改了 ENABLE_SILU_ACTIVATION 但没生效

**原因：** 这是编译时宏，需要重新编译。

**解决方案：**

```bash
# 1. 修改 casual_conv1d_opus.cpp 第 22 行
#define ENABLE_SILU_ACTIVATION 0  # 或 1

# 2. 重新编译
rm casual_conv1d_opus.exe
hipcc -x hip -std=c++17 casual_conv1d_opus.cpp -o casual_conv1d_opus.exe ...

# 3. 运行
./casual_conv1d_opus.exe
```

### 问题：如何确认当前使用的是哪个模式？

**解决方案：** 查看运行输出

**启用 SiLU：**
```
在 GPU 上添加 bias 并应用 SiLU activation (batch=1)...
✓ bias + SiLU activation 完成
```

**禁用 SiLU：**
```
在 GPU 上添加 bias (batch=1)...
✓ bias 添加完成
```

---

## 💾 内存问题

### 问题：hipMalloc failed

```bash
[hiperror](2) fail to call hipMalloc(&dev_in_transposed, ...)
```

**可能原因：**
1. GPU 内存不足
2. 分配的内存过大

**解决方案：**

```bash
# 检查 GPU 内存
rocm-smi

# 减少 batch size 或输入大小
# 在 casual_conv1d_opus.cpp 中修改：
int batch = 1;  # 降低 batch
int hi = 1024;  # 降低输入长度
```

---

## 🔍 调试技巧

### 启用详细输出

程序已包含详细的 printf 输出，直接运行即可看到：

```bash
./casual_conv1d_opus.exe
```

### 使用 rocprof 分析

```bash
# 收集 kernel 统计信息
rocprofv3 --stats -o output ./casual_conv1d_opus.exe

# 收集 HIP API 跟踪
rocprofv3 --hip-api --stats -o output ./casual_conv1d_opus.exe

# 可视化
python visualize_performance.py output_hip_api_stats.csv output_kernel_stats.csv ./
```

### 逐步调试

1. **禁用验证**（减少输出）：
   ```cpp
   #define ENABLE_HOST_VERIFICATION 0
   ```

2. **禁用 SiLU**（简化计算）：
   ```cpp
   #define ENABLE_SILU_ACTIVATION 0
   ```

3. **减少 batch**：
   ```cpp
   int batch = 1;
   ```

---

## 📚 相关文档

- [SILU_SWITCH_GUIDE.md](SILU_SWITCH_GUIDE.md) - SiLU 开关使用
- [EXIT_ERROR_EXPLANATION.md](EXIT_ERROR_EXPLANATION.md) - 退出错误说明
- [IMPLEMENTATION_OVERVIEW.md](IMPLEMENTATION_OVERVIEW.md) - 整体架构
- [README_INDEX.md](README_INDEX.md) - 文档索引

---

## 🆘 获取帮助

如果遇到其他问题：

1. **检查错误信息**：仔细阅读完整的错误输出
2. **查看相关文档**：本目录下的 Markdown 文件
3. **验证环境**：
   ```bash
   # ROCm
   rocm-smi
   /opt/rocm/bin/hipcc --version
   
   # Python
   python --version
   pip3 list | grep -E "(matplotlib|pandas|numpy)"
   
   # libtorch
   ls /root/libtorch/lib/libtorch.so
   ```

---

**最后更新：** 2025-11-15  
**文档版本：** 1.0

