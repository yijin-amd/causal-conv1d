#!/bin/bash

# Compile and run full feature test (States + seq_idx)
# Usage: ./run_full_test.sh [mode]
#   mode: 0 or omit = all tests (default)
#         1 = basic tests only
#         2 = seq_idx tests only
#         3 = states tests only

set -e

MODE=${1:-0}

# Validate mode
if [ $MODE -lt 0 ] || [ $MODE -gt 3 ]; then
    echo "Error: Invalid mode $MODE"
    echo "Usage: $0 [mode]"
    echo "  mode 0 or omit: Run all tests (default)"
    echo "  mode 1: Run basic tests only"
    echo "  mode 2: Run seq_idx tests only"
    echo "  mode 3: Run states tests only"
    exit 1
fi

echo "======================================================================="
echo " Compiling Channel-Last Backward - Full Feature Support"
echo " (States + seq_idx + Multi-width)"
echo "======================================================================="

hipcc -O2 -std=c++17 --offload-arch=gfx942 \
  causal_conv1d_channellast_bwd_hip.cpp \
  -o test_channellast_bwd_full

echo ""
echo "✓ Compilation successful!"
echo ""

if [ $MODE -eq 0 ]; then
    MODE_STR="ALL (Basic + seq_idx + States)"
elif [ $MODE -eq 1 ]; then
    MODE_STR="BASIC ONLY"
elif [ $MODE -eq 2 ]; then
    MODE_STR="SEQ_IDX ONLY"
else
    MODE_STR="STATES ONLY"
fi

echo "======================================================================="
echo " Running Tests: $MODE_STR"
echo "======================================================================="
echo ""

./test_channellast_bwd_full $MODE

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✓ All tests passed!"
else
    echo "✗ Some tests failed!"
fi

exit $exit_code

