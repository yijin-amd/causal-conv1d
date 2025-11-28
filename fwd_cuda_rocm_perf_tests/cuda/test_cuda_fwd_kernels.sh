#!/bin/bash
# 编译并运行 test_cuda_fwd_kernels.cu 的一体化脚本

set -e

PYTORCH_PATH=/usr/local/lib/python3.10/dist-packages/torch
CUDA_PATH=/usr/local/cuda-12.4
PROJ_ROOT=..
BINARY_NAME=test_cuda_fwd_kernels

# 检查是否需要重新编译
NEED_COMPILE=0

if [ ! -f "$BINARY_NAME" ]; then
    echo "未找到可执行文件，需要编译..."
    NEED_COMPILE=1
elif [ "test_cuda_fwd_kernels.cu" -nt "$BINARY_NAME" ] || \
     [ "$PROJ_ROOT/csrc/causal_conv1d_fwd.cu" -nt "$BINARY_NAME" ]; then
    echo "源文件已更新，需要重新编译..."
    NEED_COMPILE=1
else
    echo "可执行文件已是最新，跳过编译"
fi

# 编译（如果需要）
if [ $NEED_COMPILE -eq 1 ]; then
    echo ""
    echo "========================================"
    echo "正在编译 test_cuda_fwd_kernels..."
    echo "========================================"
    
    $CUDA_PATH/bin/nvcc \
      test_cuda_fwd_kernels.cu \
      $PROJ_ROOT/csrc/causal_conv1d_fwd.cu \
      -o $BINARY_NAME \
      -O3 -std=c++17 \
      -I$PROJ_ROOT/csrc \
      -I$CUDA_PATH/include \
      -I$PYTORCH_PATH/include \
      -I$PYTORCH_PATH/include/torch/csrc/api/include \
      -L$PYTORCH_PATH/lib \
      -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda \
      -Xcompiler -fPIC,-D_GLIBCXX_USE_CXX11_ABI=0 \
      --expt-relaxed-constexpr \
      -Wno-deprecated-gpu-targets \
      --generate-code=arch=compute_80,code=[compute_80,sm_80] \
      --generate-code=arch=compute_86,code=[compute_86,sm_86] \
      --generate-code=arch=compute_89,code=[compute_89,sm_89] \
      --generate-code=arch=compute_90,code=[compute_90,sm_90]
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 编译成功!"
    else
        echo ""
        echo "✗ 编译失败!"
        exit 1
    fi
fi

# 运行测试
echo ""
echo "========================================"
echo "运行性能测试..."
echo "========================================"
echo ""

export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:$LD_LIBRARY_PATH
./$BINARY_NAME

echo ""
echo "========================================"
echo "✓ 测试完成!"
echo "========================================"

