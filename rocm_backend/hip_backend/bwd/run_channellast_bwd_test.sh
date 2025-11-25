#!/bin/bash

# Channel-Last Backward Kernel - Compilation and Test Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine test mode
TEST_MODE=${1:-0}

echo "======================================================================="
echo " Compiling Channel-Last Backward - Full Feature Support"
echo " (States + seq_idx + Multi-width)"
echo "======================================================================="

# Compile
hipcc -O2 -std=c++17 --offload-arch=gfx942 \
    causal_conv1d_channellast_bwd_kernel.hip \
    test_channellast_bwd.cpp \
    -o test_channellast_bwd

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Compilation successful!${NC}"
else
    echo -e "${RED}✗ Compilation failed!${NC}"
    exit 1
fi

echo ""
echo "======================================================================="
case $TEST_MODE in
    0)
        echo " Running Tests: ALL (Basic + seq_idx + States)"
        ;;
    1)
        echo " Running Tests: BASIC ONLY"
        ;;
    2)
        echo " Running Tests: SEQ_IDX ONLY"
        ;;
    3)
        echo " Running Tests: STATES ONLY"
        ;;
    *)
        echo -e "${RED}Invalid test mode: $TEST_MODE${NC}"
        echo "Usage: $0 [mode]"
        echo "  mode 0 or omit: Run all tests (default)"
        echo "  mode 1: Run basic tests only"
        echo "  mode 2: Run seq_idx tests only"
        echo "  mode 3: Run states tests only"
        exit 1
        ;;
esac
echo "======================================================================="
echo ""

# Run tests
./test_channellast_bwd $TEST_MODE

# Check test result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Some tests failed!${NC}"
    exit 1
fi

