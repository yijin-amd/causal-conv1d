# Causal Conv1D HIP Implementation

仿照 CUDA 版本的 `causal_conv1d_fn` 实现的 HIP 版本接口。

## 📁 文件结构

```
fwd/
├── causal_conv1d_kernel.hip          # HIP kernel 实现（已存在）
├── causal_conv1d_hip_launcher.hip    # Kernel launcher（新增）
├── causal_conv1d_hip.cpp             # C++ PyTorch 绑定（新增）
├── causal_conv1d_hip_interface.py    # Python 接口层（新增）
├── compile_hip_extension.sh          # 编译脚本（新增）
├── setup.py                          # 安装脚本（新增）
└── HIP_INTEGRATION_README.md         # 本文档
```

## 🔄 调用链对比

### CUDA 版本调用链

```
Python Layer:
  causal_conv1d_fn()                      (causal_conv1d_interface.py)
    ↓
  CausalConv1dFn.apply()
    ↓
  CausalConv1dFn.forward()
    ↓
  causal_conv1d_fwd_function()            (cpp_functions.py)
    ↓
C++ Layer:
  causal_conv1d_fwd()                     (causal_conv1d.cpp)
    ↓
  causal_conv1d_fwd_cuda()                (causal_conv1d_fwd.cu)
    ↓
  causal_conv1d_fwd_launch()              (causal_conv1d_fwd.cu)
    ↓
CUDA Kernel:
  causal_conv1d_fwd_kernel<<<>>>()        (causal_conv1d_fwd.cu)
```

### HIP 版本调用链（新实现）

```
Python Layer:
  causal_conv1d_hip_fn()                  (causal_conv1d_hip_interface.py)
    ↓
  CausalConv1dHIPFn.apply()
    ↓
  CausalConv1dHIPFn.forward()
    ↓
  causal_conv1d_fwd_hip()                 (causal_conv1d_hip_ext)
    ↓
C++ Layer:
  causal_conv1d_fwd_hip()                 (causal_conv1d_hip.cpp)
    ↓
  causal_conv1d_fwd_hip_internal()        (causal_conv1d_hip.cpp)
    ↓
  causal_conv1d_fwd_hip_launch_w{2,3,4}() (causal_conv1d_hip_launcher.hip)
    ↓
HIP Kernel:
  causal_conv1d_fwd_kernel<<<>>>()        (causal_conv1d_kernel.hip)
```

## 🔧 编译安装

### 方法 1: 使用编译脚本

```bash
cd /workspace/causal-conv1d/rocm_backend/hip_backend/fwd

# 设置 GPU 架构（可选）
export GPU_ARCH=gfx942  # 或 gfx90a, gfx908 等

# 编译
chmod +x compile_hip_extension.sh
./compile_hip_extension.sh

# 添加到 Python 路径
export PYTHONPATH=$PWD/build:$PYTHONPATH
```

### 方法 2: 使用 setup.py

```bash
cd /workspace/causal-conv1d/rocm_backend/hip_backend/fwd

# 安装到当前 Python 环境
python3 setup.py install

# 或开发模式安装（修改代码后无需重新安装）
python3 setup.py develop
```

## 📝 使用示例

### 基本使用

```python
import torch
from causal_conv1d_hip_interface import causal_conv1d_hip_fn

# 创建测试数据
batch, dim, seqlen, width = 2, 64, 512, 4
device = 'cuda'  # 在 ROCm 中，'cuda' 映射到 HIP 设备

x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
weight = torch.randn(dim, width, device=device, dtype=torch.float32)
bias = torch.randn(dim, device=device, dtype=torch.float32)

# 调用 HIP 实现
out = causal_conv1d_hip_fn(x, weight, bias, activation='silu')

print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
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

# 计算差异
diff = (out_hip - out_ref).abs()
print(f"Max difference: {diff.max().item():.6f}")
print(f"Mean difference: {diff.mean().item():.6f}")
```

### 运行测试

```python
from causal_conv1d_hip_interface import test_causal_conv1d_hip

# 运行所有测试
test_causal_conv1d_hip()
```

或在命令行：

```bash
python3 causal_conv1d_hip_interface.py
```

## 🎯 关键设计特点

### 1. 模板分发（Width Dispatch）

```cpp
// C++ 层根据 width 分发到不同的 launcher
DISPATCH_WIDTH(width, "causal_conv1d_fwd_hip", [&] {
    if constexpr (kWidth == 2) {
        causal_conv1d_fwd_hip_launch_w2(...);
    } else if constexpr (kWidth == 3) {
        causal_conv1d_fwd_hip_launch_w3(...);
    } else if constexpr (kWidth == 4) {
        causal_conv1d_fwd_hip_launch_w4(...);
    }
});
```

### 2. Kernel Traits

```cpp
template<int kNThreads_, int kWidth_, int kNElts_>
struct CausalConv1dKernelTraits {
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNElts = kNElts_;
    static constexpr int kChunkSize = kNThreads * kNElts;
    
    using vec_t = float4;
    using BlockLoadT = hipcub::BlockLoad<...>;
    using BlockStoreT = hipcub::BlockStore<...>;
    
    static constexpr int kSmemSize = ...;
};
```

### 3. HIP Kernel Launch

```cpp
hipLaunchKernelGGL(
    causal_conv1d_fwd_kernel<Ktraits>,
    grid, block, smem_size, stream,
    x, weight, bias, out,
    batch, dim, seqlen, width,
    x_batch_stride, x_c_stride,
    weight_c_stride, weight_width_stride,
    out_batch_stride, out_c_stride,
    use_silu
);
```

## 📊 支持的配置

- **数据类型**: 目前仅支持 `float32`
- **Width**: 2, 3, 4
- **激活函数**: None, SiLU/Swish
- **Bias**: 可选
- **Layout**: Channel-First (batch, dim, seqlen)

## ⚠️ 注意事项

1. **向后传播**: 当前版本仅实现了前向传播，backward 尚未实现
2. **数据类型**: 仅支持 float32，不支持 half/bfloat16
3. **内存布局**: 仅支持 Channel-First 布局
4. **Seq_idx/States**: 不支持 seq_idx 和 initial_states/final_states

## 🔮 未来扩展

1. **反向传播**: 实现 `CausalConv1dHIPFn.backward()`
2. **混合精度**: 添加 FP16/BF16 支持
3. **Channel-Last**: 支持 Channel-Last 内存布局
4. **Variable Length**: 支持 seq_idx 功能
5. **States**: 支持 initial_states 和 final_states

## 📚 API 参考

### `causal_conv1d_hip_fn`

```python
def causal_conv1d_hip_fn(
    x: torch.Tensor,          # (batch, dim, seqlen)
    weight: torch.Tensor,     # (dim, width)
    bias: Optional[torch.Tensor] = None,  # (dim,)
    activation: Optional[str] = None,     # None, 'silu', or 'swish'
) -> torch.Tensor:            # (batch, dim, seqlen)
```

### `causal_conv1d_hip_ref`

```python
def causal_conv1d_hip_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
```

参考实现，使用 PyTorch 的 `F.conv1d`，用于验证正确性。

## 🐛 故障排除

### 编译错误

1. **找不到 hipcc**:
   ```bash
   export PATH=/opt/rocm/bin:$PATH
   ```

2. **找不到 PyTorch**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
   ```

3. **GPU 架构不匹配**:
   ```bash
   export GPU_ARCH=gfx942  # 根据你的 GPU 设置
   ```

### 运行时错误

1. **找不到扩展模块**:
   ```bash
   export PYTHONPATH=/workspace/causal-conv1d/rocm_backend/hip_backend/fwd/build:$PYTHONPATH
   ```

2. **HIP 设备不可用**:
   ```bash
   rocm-smi  # 检查 GPU 状态
   ```

## 📄 许可证

与原始 Causal Conv1D 项目保持一致。

