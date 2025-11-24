# Causal Conv1D Backward - HIP Implementation

Pure HIP implementation of Causal Conv1D backward pass for AMD GPUs, with no PyTorch dependencies.

## Quick Start

```bash
# Compile and run all tests
bash run_bwd.sh

# Or manually
hipcc -O3 -std=c++17 causal_conv1d_bwd_hip.cpp -o conv1d_bwd_test
./conv1d_bwd_test --verify
```

## Performance Highlights

- ✅ **100% Test Pass Rate** (20/20 configurations)
- 🚀 **Peak Bandwidth**: 2,292 GB/s (FP32), 1,425 GB/s (FP16)
- ⚡ **Low Latency**: 5.7 μs (small), 88 μs (large)
- 🎯 **High Accuracy**: Max error < 0.0001 (FP32), < 0.01 (FP16)

## Tested On

- **GPU**: AMD Instinct MI308X (192 GB)
- **ROCm**: 7.0.1
- **Compiler**: hipcc (Clang-based)

## Features

- ✅ Full gradient computation (dx, dW, db)
- ✅ Multi-precision support (FP32, FP16)
- ✅ SiLU activation function
- ✅ Flexible convolution width (2, 3, 4)
- ✅ Comprehensive test suite

## Documentation

- 📊 [Performance Report](./PERFORMANCE_REPORT.md) - Detailed benchmarks and analysis (English)
- 📋 [测试报告](./测试报告.md) - 简洁版测试报告（中文）
- 💻 [Source Code](./causal_conv1d_bwd_hip.cpp) - Main implementation

## Test Results Summary

```
╔══════════════════════════════════════════════════════════════╗
║                 Causal Conv1D Backward Tests                 ║
║                   HIP Implementation Summary                 ║
╚══════════════════════════════════════════════════════════════╝

🎯 Total Tests Run:    20
✅ PASSED:            20 (100%)
❌ FAILED:            0 (0%)

📊 Breakdown by Data Type:
   FP32:  10 passed, 0 failed
   FP16:  10 passed, 0 failed

🖥️  Platform: AMD Instinct MI308X (192 GB)
```

## Usage Example

```cpp
ConvParamsBwd params;
params.batch = 8;
params.dim = 2048;
params.seqlen = 1024;
params.width = 4;
params.silu_activation = false;

// Set pointers
params.x_ptr = d_x;
params.weight_ptr = d_weight;
params.dout_ptr = d_dout;
params.dx_ptr = d_dx;
params.dweight_ptr = d_dweight;
params.dbias_ptr = d_dbias;

// Launch
causal_conv1d_bwd_launch<128, 4, float, float>(params, stream);
```

## Performance Tips

1. Use **FP16** for large problems (>1M elements) for 20% speedup
2. **Batch size ≥ 4** for better GPU utilization
3. **Sequence length > 1024** for optimal performance
4. Pre-allocate buffers to avoid allocation overhead

## File Structure

```
hip_backend_bwd/
├── causal_conv1d_bwd_hip.cpp  # Main implementation (1055 lines)
├── run_bwd.sh                 # Build and test script
├── README.md                  # This file
├── PERFORMANCE_REPORT.md      # Detailed performance analysis
└── 测试报告.md                # Chinese summary report
```

## License

Same as parent project (Apache 2.0)

## Citation

If you use this implementation, please cite:
```bibtex
@software{causal_conv1d_hip,
  title = {Causal Conv1D HIP Implementation},
  year = {2024},
  note = {High-performance backward pass for AMD GPUs}
}
```

---

**Status**: ✅ Production Ready  
**Version**: 1.0  
**Last Updated**: November 2024

