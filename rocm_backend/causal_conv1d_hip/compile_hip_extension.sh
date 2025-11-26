#!/bin/bash

# Compilation script for Causal Conv1D HIP extension
# Usage: ./compile_hip_extension.sh

echo "========================================="
echo "Compiling Causal Conv1D HIP Extension"
echo "========================================="

# Set ROCm path
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH=$ROCM_PATH/bin:$PATH

# Check if hipcc is available
if ! command -v hipcc &> /dev/null; then
    echo "Error: hipcc not found. Please install ROCm."
    exit 1
fi

echo "Using ROCm path: $ROCM_PATH"
echo "HIP compiler: $(which hipcc)"

# Get GPU architecture
GPU_ARCH=${GPU_ARCH:-gfx942}
echo "Target GPU architecture: $GPU_ARCH"

# Get Python include path
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
echo "Python include: $PYTHON_INCLUDE"

# Get PyTorch include paths
TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TORCH_INCLUDE="$TORCH_PATH/include"
TORCH_LIB="$TORCH_PATH/lib"
echo "PyTorch path: $TORCH_PATH"
echo "PyTorch include: $TORCH_INCLUDE"
echo "PyTorch lib: $TORCH_LIB"

# Compilation flags
CXX_FLAGS="-O3 -std=c++17 -fPIC -DTORCH_EXTENSION_NAME=causal_conv1d_hip_ext"
HIP_FLAGS="--offload-arch=$GPU_ARCH"
INCLUDE_FLAGS="-I$PYTHON_INCLUDE -I$TORCH_INCLUDE -I$TORCH_INCLUDE/torch/csrc/api/include"
LIB_FLAGS="-L$TORCH_LIB -ltorch -ltorch_python -lc10 -lc10_hip"

# Output directory
OUT_DIR="build"
mkdir -p $OUT_DIR

echo ""
echo "Step 1: Compiling HIP kernel launcher..."
hipcc $HIP_FLAGS $CXX_FLAGS $INCLUDE_FLAGS \
    -c causal_conv1d_hip_launcher.hip \
    -o $OUT_DIR/causal_conv1d_hip_launcher.o

if [ $? -ne 0 ]; then
    echo "Error: Failed to compile HIP kernel launcher"
    exit 1
fi
echo "✓ HIP kernel launcher compiled successfully"

echo ""
echo "Step 2: Compiling C++ interface..."
hipcc $CXX_FLAGS $INCLUDE_FLAGS \
    -c causal_conv1d_hip.cpp \
    -o $OUT_DIR/causal_conv1d_hip.o

if [ $? -ne 0 ]; then
    echo "Error: Failed to compile C++ interface"
    exit 1
fi
echo "✓ C++ interface compiled successfully"

echo ""
echo "Step 3: Linking shared library..."
hipcc $HIP_FLAGS -shared \
    $OUT_DIR/causal_conv1d_hip_launcher.o \
    $OUT_DIR/causal_conv1d_hip.o \
    $LIB_FLAGS \
    -Wl,-rpath,$TORCH_LIB \
    -Wl,-rpath,$ROCM_PATH/lib \
    -o $OUT_DIR/causal_conv1d_hip_ext.so

if [ $? -ne 0 ]; then
    echo "Error: Failed to link shared library"
    exit 1
fi
echo "✓ Shared library linked successfully"

echo ""
echo "========================================="
echo "✅ Compilation completed successfully!"
echo "========================================="
echo "Output: $OUT_DIR/causal_conv1d_hip_ext.so"
echo ""
echo "To use the extension, add the build directory to PYTHONPATH:"
echo "  export PYTHONPATH=$PWD/$OUT_DIR:\$PYTHONPATH"
echo ""
echo "Or install it:"
echo "  python3 setup.py install"

