# 快速开始指南 - Causal Conv1D HIP

## 🚀 5分钟快速上手

### 1. 编译扩展

```bash
cd /workspace/causal-conv1d/rocm_backend/causal_conv1d_hip
./compile_hip_extension.sh
```

### 2. 设置环境变量

```bash
export PYTHONPATH=$PWD/build:$PYTHONPATH
```

### 3. 运行示例

```bash
python3 example_usage.py
```

## 📝 最小代码示例

```python
import torch
from causal_conv1d_hip_interface import causal_conv1d_hip_fn

# 创建数据
x = torch.randn(2, 64, 512, device='cuda')
weight = torch.randn(64, 4, device='cuda')
bias = torch.randn(64, device='cuda')

# 运行
out = causal_conv1d_hip_fn(x, weight, bias, activation='silu')
```

## 📋 与 CUDA 版本对比

| 特性 | CUDA 版本 | HIP 版本 |
|-----|----------|---------|
| Python 接口 | `causal_conv1d_fn` | `causal_conv1d_hip_fn` |
| 前向传播 | ✅ 支持 | ✅ 支持 |
| 反向传播 | ✅ 支持 | ❌ 未实现 |
| FP16/BF16 | ✅ 支持 | ❌ 仅 FP32 |
| Channel-Last | ✅ 支持 | ❌ 仅 Channel-First |
| seq_idx | ✅ 支持 | ❌ 未实现 |
| initial/final states | ✅ 支持 | ❌ 未实现 |

## 🔧 API 对照

### CUDA 版本

```python
from causal_conv1d import causal_conv1d_fn

out = causal_conv1d_fn(
    x,                      # (batch, dim, seqlen)
    weight,                 # (dim, width)
    bias,                   # (dim,)
    seq_idx=None,           # (batch, seqlen)
    initial_states=None,    # (batch, dim, width-1)
    return_final_states=False,
    final_states_out=None,
    activation='silu'
)
```

### HIP 版本

```python
from causal_conv1d_hip_interface import causal_conv1d_hip_fn

out = causal_conv1d_hip_fn(
    x,                      # (batch, dim, seqlen)
    weight,                 # (dim, width)
    bias,                   # (dim,)
    activation='silu'       # None, 'silu', or 'swish'
)
```

## ⚙️ 配置选项

### Width 支持

```python
# Width = 2
weight = torch.randn(dim, 2, device='cuda')
out = causal_conv1d_hip_fn(x, weight, bias)

# Width = 3
weight = torch.randn(dim, 3, device='cuda')
out = causal_conv1d_hip_fn(x, weight, bias)

# Width = 4
weight = torch.randn(dim, 4, device='cuda')
out = causal_conv1d_hip_fn(x, weight, bias)
```

### 激活函数

```python
# 无激活函数
out = causal_conv1d_hip_fn(x, weight, bias, activation=None)

# SiLU 激活
out = causal_conv1d_hip_fn(x, weight, bias, activation='silu')

# Swish 激活（等同于 SiLU）
out = causal_conv1d_hip_fn(x, weight, bias, activation='swish')
```

### 可选 Bias

```python
# 带 bias
out = causal_conv1d_hip_fn(x, weight, bias, activation='silu')

# 不带 bias
out = causal_conv1d_hip_fn(x, weight, None, activation='silu')
```

## 🧪 测试和验证

### 运行内置测试

```python
from causal_conv1d_hip_interface import test_causal_conv1d_hip
test_causal_conv1d_hip()
```

### 与参考实现对比

```python
from causal_conv1d_hip_interface import (
    causal_conv1d_hip_fn,
    causal_conv1d_hip_ref
)

# HIP 实现
out_hip = causal_conv1d_hip_fn(x, weight, bias, activation='silu')

# PyTorch 参考实现
out_ref = causal_conv1d_hip_ref(x, weight, bias, activation='silu')

# 验证精度
diff = (out_hip - out_ref).abs()
print(f"Max diff: {diff.max().item():.6f}")
assert diff.max() < 1e-3, "Accuracy check failed"
```

## 🎯 性能优化建议

1. **使用连续内存**: 确保输入张量是连续的
   ```python
   x = x.contiguous()
   ```

2. **预分配输出**: 避免内存分配开销（当前版本自动处理）

3. **批处理**: 增加 batch size 提高 GPU 利用率
   ```python
   # 好：batch_size = 8
   x = torch.randn(8, 128, 2048, device='cuda')
   
   # 不好：batch_size = 1
   x = torch.randn(1, 128, 2048, device='cuda')
   ```

4. **选择合适的维度**: dim 应该是 32 的倍数以获得最佳性能
   ```python
   # 推荐
   dim = 64, 128, 256, 512
   
   # 不推荐
   dim = 63, 127, 255
   ```

## 🐛 常见问题

### Q: 编译失败，找不到 hipcc

**A**: 设置 ROCm 路径
```bash
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm
```

### Q: 运行时找不到扩展模块

**A**: 设置 Python 路径
```bash
export PYTHONPATH=/workspace/causal-conv1d/rocm_backend/hip_backend/fwd/build:$PYTHONPATH
```

### Q: 精度不匹配

**A**: 检查数据类型和设备
```python
assert x.dtype == torch.float32, "Only float32 supported"
assert x.device.type == 'cuda', "Must be on HIP device"
```

### Q: 性能不如预期

**A**: 检查配置
```python
# 1. 增加 batch size
# 2. 使用更大的 dim（128, 256, 512）
# 3. 确保数据是连续的
x = x.contiguous()
```

## 📚 更多资源

- **完整文档**: `HIP_INTEGRATION_README.md`
- **示例代码**: `example_usage.py`
- **内核实现**: `causal_conv1d_kernel.hip`
- **测试脚本**: `causal_conv1d_hip_interface.py`

## 💡 下一步

1. 阅读完整文档：`cat HIP_INTEGRATION_README.md`
2. 运行示例：`python3 example_usage.py`
3. 集成到自己的项目
4. 贡献代码：实现 backward、FP16 支持等

