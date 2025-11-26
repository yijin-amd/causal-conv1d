# Causal Conv1D HIP 实现

仿照 CUDA 版本的 `causal_conv1d_fn` 实现的 HIP 版本接口。

## ✅ 状态

**编译成功 ✓**  
**测试通过 ✓**

## 📁 文件列表

```
/workspace/causal-conv1d/rocm_backend/causal_conv1d_hip/
├── causal_conv1d_kernel.hip              # HIP kernel 实现
├── causal_conv1d_hip_launcher.hip        # Kernel launcher
├── causal_conv1d_hip.cpp                 # C++ PyTorch 绑定
├── causal_conv1d_hip_interface.py        # Python 接口
├── compile_hip_extension.sh              # 编译脚本
├── setup.py                              # 安装脚本
├── example_usage.py                      # 使用示例
├── HIP_INTEGRATION_README.md             # 详细文档
├── QUICKSTART.md                         # 快速开始
├── IMPLEMENTATION_SUMMARY.md             # 实现总结
└── README.md                             # 本文件
```

## 🚀 快速开始

### 1. 编译扩展

```bash
cd /workspace/causal-conv1d/rocm_backend/causal_conv1d_hip
./compile_hip_extension.sh
```

### 2. 设置环境变量

```bash
export PYTHONPATH=/workspace/causal-conv1d/rocm_backend/causal_conv1d_hip/build:$PYTHONPATH
```

### 3. 运行测试

```python
import torch
from causal_conv1d_hip_interface import causal_conv1d_hip_fn

# 创建数据
x = torch.randn(2, 64, 512, device='cuda')
weight = torch.randn(64, 4, device='cuda')
bias = torch.randn(64, device='cuda')

# 运行
out = causal_conv1d_hip_fn(x, weight, bias, activation='silu')
print(f'Output shape: {out.shape}')  # torch.Size([2, 64, 512])
```

## 📊 测试结果

```bash
✅ Successfully imported causal_conv1d_hip_fn
✅ Success! Output shape: torch.Size([2, 64, 512])
```

## 📚 文档

- **快速开始**: [QUICKSTART.md](QUICKSTART.md)
- **详细文档**: [HIP_INTEGRATION_README.md](HIP_INTEGRATION_README.md)
- **实现总结**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **示例代码**: [example_usage.py](example_usage.py)

## 🔄 调用链

```
Python: causal_conv1d_hip_fn()
  ↓
PyTorch Autograd: CausalConv1dHIPFn.forward()
  ↓
C++ Extension: causal_conv1d_fwd_hip()
  ↓
C++ Internal: causal_conv1d_fwd_hip_internal()
  ↓
HIP Launcher: causal_conv1d_fwd_hip_launch_w{2,3,4}()
  ↓
HIP Kernel: causal_conv1d_fwd_kernel<<<>>>()
```

## 📝 支持的功能

- ✅ 前向传播 (Forward Pass)
- ✅ Float32 数据类型
- ✅ Width 2/3/4
- ✅ Bias (可选)
- ✅ SiLU/Swish 激活函数
- ✅ Channel-First 布局
- ❌ 反向传播 (待实现)
- ❌ FP16/BF16 (待实现)
- ❌ Channel-Last 布局 (待实现)

## 💡 使用示例

更多示例请参见 [example_usage.py](example_usage.py)

## 🛠️ 编译选项

可以通过环境变量自定义编译：

```bash
export GPU_ARCH=gfx942  # 设置 GPU 架构
export ROCM_PATH=/opt/rocm  # 设置 ROCm 路径
./compile_hip_extension.sh
```

## 📄 许可证

与原始 Causal Conv1D 项目保持一致。

