#!/bin/bash

# Causal Conv1D Backward - Layout Comparison Script
# 对比 Channel-First 和 Channel-Last 两种布局的性能

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================="
echo " Causal Conv1D Backward - Layout Comparison"
echo " Comparing Channel-First vs Channel-Last Performance"
echo "======================================================================="
echo ""

# 检查源文件是否存在
if [ ! -f "causal_conv1d_bwd_kernel.hip" ]; then
    echo -e "${RED}✗ causal_conv1d_bwd_kernel.hip not found!${NC}"
    exit 1
fi

if [ ! -f "causal_conv1d_channellast_bwd_kernel.hip" ]; then
    echo -e "${RED}✗ causal_conv1d_channellast_bwd_kernel.hip not found!${NC}"
    exit 1
fi

if [ ! -f "compare_layouts.cpp" ]; then
    echo -e "${RED}✗ compare_layouts.cpp not found!${NC}"
    exit 1
fi

echo "Compiling kernels and comparison test..."
echo ""

# 编译内核和测试文件
hipcc -O2 -std=c++17 --offload-arch=gfx942 \
    causal_conv1d_bwd_kernel.hip \
    causal_conv1d_channellast_bwd_kernel.hip \
    compare_layouts.cpp \
    -o compare_layouts

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Comparison test compilation failed!${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Comparison test compiled successfully!${NC}"
fi

echo ""
echo "======================================================================="
echo " Running Layout Comparison Tests"
echo "======================================================================="
echo ""

# 运行测试
./compare_layouts

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All comparison tests completed successfully!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed!${NC}"
    exit 1
fi

