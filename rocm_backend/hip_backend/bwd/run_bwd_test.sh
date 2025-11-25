#!/bin/bash

# Causal Conv1D Backward (Channel-First) - Compilation and Test Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Show usage
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [--verify|--perf-only]"
    echo ""
    echo "Options:"
    echo "  --verify      验证正确性 + 性能测试 (默认)"
    echo "  --perf-only   仅性能测试，跳过正确性验证"
    echo "  -h, --help    显示此帮助信息"
    echo ""
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认启用正确性验证
if [ "$1" == "--perf-only" ]; then
    TEST_ARGS=""
    TEST_MODE="PERFORMANCE MODE ONLY"
else
    TEST_ARGS="--verify"
    TEST_MODE="VERIFICATION + PERFORMANCE MODE (Default)"
fi

echo "======================================================================="
echo " Compiling Causal Conv1D Backward (Channel-First Layout)"
echo " FP32 + FP16 Support"
echo "======================================================================="

# Compile
hipcc -O2 -std=c++17 --offload-arch=gfx942 \
    causal_conv1d_bwd_kernel.hip \
    test_causal_conv1d_bwd.cpp \
    -o test_causal_conv1d_bwd

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful!${NC}"
else
    echo -e "${RED}✗ Compilation failed!${NC}"
    exit 1
fi

echo ""
echo "======================================================================="
echo " Running Tests: $TEST_MODE"
echo "======================================================================="
echo ""

# Run tests
./test_causal_conv1d_bwd $TEST_ARGS

# Check test result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests completed successfully!${NC}"
    if [ "$TEST_ARGS" == "" ]; then
        echo -e "${BLUE}ℹ️  提示: 运行 '$0 --verify' 以启用正确性验证${NC}"
    fi
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed!${NC}"
    exit 1
fi

